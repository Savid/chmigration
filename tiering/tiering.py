#!/usr/bin/env python3
"""Move partitions between disks, per-node (no DDL queue).

Discovers partitions from system.parts on each cluster node and moves those
whose parts aren't yet on the target disk. Partitions already on the target
disk or currently being moved are skipped, making this safe to re-run.

Direction is controlled by --disk: use 's3_cache' to tier down old data to
object storage, or 'default' to restore partitions back to the local disk.

Usage:
    # Dry run — show what would be moved (to s3_cache by default)
    python tiering/tiering.py \
      --host clickhouse-raw.example.com --secure \
      --user admin --password secret \
      --database default \
      --table beacon_api_eth_v1_events_block \
      --dry-run

    # Tier down to s3_cache (default --disk)
    python tiering/tiering.py \
      --host clickhouse-raw.example.com --secure \
      --user admin --password secret \
      --database default \
      --table beacon_api_eth_v1_events_block canonical_beacon_block

    # Restore FROM s3_cache back to 'default' disk (all partitions)
    python tiering/tiering.py \
      --host clickhouse-raw.example.com --secure \
      --user admin --password secret \
      --database default \
      --table beacon_api_eth_v1_events_block \
      --disk default --all-partitions

    # Custom cutoff date
    python tiering/tiering.py \
      --host clickhouse-raw.example.com --secure \
      --user admin --password secret \
      --database default \
      --table beacon_api_eth_v1_events_block \
      --before 2025-01-01

    # Host pattern with shard/replica expansion
    python tiering/tiering.py \
      --host 'clickhouse-raw-%s-%r.example.com' --secure \
      --user admin --password secret \
      --database default \
      --table beacon_api_eth_v1_events_block
"""

import argparse
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

sys.stdout.reconfigure(line_buffering=True)

_shutdown = False

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TaskID


DEFAULT_CLUSTER = "replicated"
DEFAULT_DISK = "s3_cache"
DEFAULT_TIMEOUT = 3600
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_RETRIES = 3
POLL_INTERVAL = 10
SCRIPT_DIR = Path(__file__).resolve().parent
STATE_LOG = SCRIPT_DIR / "state.jsonl"


# ---------------------------------------------------------------------------
# ClickHouse HTTP helpers
# ---------------------------------------------------------------------------

class ClickHouseError(Exception):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


_secure = False


def query(
    host: str,
    port: int,
    sql: str,
    timeout: int = DEFAULT_TIMEOUT,
    user: str | None = None,
    password: str | None = None,
) -> str:
    params = {
        "receive_timeout": timeout,
        "send_timeout": timeout,
    }
    if user:
        params["user"] = user
    if password:
        params["password"] = password
    scheme = "https" if _secure else "http"
    url = f"{scheme}://{host}:{port}/?{urllib.parse.urlencode(params)}"
    data = sql.encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                headers={"User-Agent": "clickhouse-tiering/1.0"})
    ctx = ssl.create_default_context() if _secure else None
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read().decode("utf-8").strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace").strip()
        raise ClickHouseError(body, status=e.code) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ClickHouseError(f"Connection error ({host}:{port}): {e}") from e


def query_json_rows(
    host: str,
    port: int,
    sql: str,
    timeout: int = DEFAULT_TIMEOUT,
    user: str | None = None,
    password: str | None = None,
) -> list[dict]:
    result = query(host, port, f"{sql} FORMAT JSONEachRow", timeout, user, password)
    if not result:
        return []
    rows: list[dict] = []
    for line in result.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Table resolution
# ---------------------------------------------------------------------------

def resolve_table(
    host: str, port: int, database: str, table: str,
    user: str | None = None, password: str | None = None,
) -> tuple[str, str, bool]:
    """Resolve a table name. If Distributed, return the underlying local table."""
    rows = query_json_rows(
        host, port,
        f"SELECT engine, engine_full FROM system.tables "
        f"WHERE database = '{database}' AND name = '{table}'",
        user=user, password=password,
    )
    if not rows:
        return database, table, False

    engine = rows[0].get("engine", "")
    engine_full = rows[0].get("engine_full", "")

    if engine != "Distributed":
        return database, table, False

    m = re.match(r"Distributed\(\s*'[^']+'\s*,\s*'([^']+)'\s*,\s*'([^']+)'", engine_full)
    if m:
        return m.group(1), m.group(2), True

    return database, table, False


# ---------------------------------------------------------------------------
# Partition discovery
# ---------------------------------------------------------------------------

def extract_partition_date(partition: str) -> datetime | None:
    """Try to extract a date from a partition value string."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", partition)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    m = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", partition)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    m = re.search(r"(?<!\d)(\d{4})(\d{2})(?!\d)", partition)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 2000 <= year <= 2100 and 1 <= month <= 12:
            return datetime(year, month, 1)

    return None


def get_node_partitions(
    host: str,
    port: int,
    database: str,
    table: str,
    before: datetime | None = None,
    target_disk: str = DEFAULT_DISK,
    user: str | None = None,
    password: str | None = None,
    partition_filter: set[str] | None = None,
) -> list[dict]:
    """Get partitions on a single node with their disk placement.

    Returns list of dicts with: partition_id, partition, disk_name,
    total_rows, bytes_on_disk, parts_on_target, parts_total.
    Only returns partitions that have at least some parts NOT on target_disk.
    """
    sql = (
        f"SELECT "
        f"  partition_id, "
        f"  partition, "
        f"  disk_name, "
        f"  count() AS part_count, "
        f"  sum(rows) AS total_rows, "
        f"  sum(bytes_on_disk) AS bytes_on_disk "
        f"FROM system.parts "
        f"WHERE database = '{database}' AND table = '{table}' AND active = 1 "
        f"GROUP BY partition_id, partition, disk_name "
        f"ORDER BY partition_id, disk_name"
    )
    rows = query_json_rows(host, port, sql, user=user, password=password)

    # Group by partition_id.
    partitions: dict[str, dict] = {}
    for row in rows:
        pid = row["partition_id"]
        if pid not in partitions:
            partitions[pid] = {
                "partition_id": pid,
                "partition": str(row["partition"]),
                "parts_total": 0,
                "parts_on_target": 0,
                "total_rows": 0,
                "bytes_on_disk": 0,
                "disks": [],
            }
        p = partitions[pid]
        count = int(row["part_count"])
        p["parts_total"] += count
        p["total_rows"] += int(row["total_rows"])
        p["bytes_on_disk"] += int(row["bytes_on_disk"])
        p["disks"].append(row["disk_name"])
        if row["disk_name"] == target_disk:
            p["parts_on_target"] += count

    # Filter by partition ID and date cutoff.
    result: list[dict] = []
    for p in partitions.values():
        if partition_filter is not None and p["partition_id"] not in partition_filter:
            continue
        if before is not None:
            dt = extract_partition_date(p["partition"])
            if dt is None or dt >= before:
                continue
        # Only include if there are parts NOT on the target disk.
        if p["parts_on_target"] < p["parts_total"]:
            result.append(p)

    result.sort(key=lambda x: x["partition_id"])
    return result


def get_node_moving_partitions(
    host: str,
    port: int,
    database: str,
    table: str,
    user: str | None = None,
    password: str | None = None,
) -> set[str]:
    """Get partition IDs that are currently being moved on this node.

    system.moves only has part_name, not partition_id. The partition ID
    is the prefix of the part name before the first '_'.
    """
    sql = (
        f"SELECT DISTINCT substring(part_name, 1, position(part_name, '_') - 1) AS partition_id "
        f"FROM system.moves "
        f"WHERE database = '{database}' AND table = '{table}'"
    )
    rows = query_json_rows(host, port, sql, timeout=30, user=user, password=password)
    return {r["partition_id"] for r in rows}


def get_node_all_partitions(
    host: str,
    port: int,
    database: str,
    table: str,
    before: datetime | None = None,
    target_disk: str = DEFAULT_DISK,
    user: str | None = None,
    password: str | None = None,
    partition_filter: set[str] | None = None,
) -> list[dict]:
    """Get ALL partitions on a single node with disk status (for --status).

    Returns list of dicts: partition_id, partition, total_rows, bytes_on_disk,
    parts_total, parts_on_target, status.
    """
    sql = (
        f"SELECT "
        f"  partition_id, "
        f"  partition, "
        f"  disk_name, "
        f"  count() AS part_count, "
        f"  sum(rows) AS total_rows, "
        f"  sum(bytes_on_disk) AS bytes_on_disk "
        f"FROM system.parts "
        f"WHERE database = '{database}' AND table = '{table}' AND active = 1 "
        f"GROUP BY partition_id, partition, disk_name "
        f"ORDER BY partition_id, disk_name"
    )
    rows = query_json_rows(host, port, sql, user=user, password=password)

    partitions: dict[str, dict] = {}
    for row in rows:
        pid = row["partition_id"]
        if pid not in partitions:
            partitions[pid] = {
                "partition_id": pid,
                "partition": str(row["partition"]),
                "parts_total": 0,
                "parts_on_target": 0,
                "total_rows": 0,
                "bytes_on_disk": 0,
                "disks": set(),
            }
        p = partitions[pid]
        count = int(row["part_count"])
        p["parts_total"] += count
        p["total_rows"] += int(row["total_rows"])
        p["bytes_on_disk"] += int(row["bytes_on_disk"])
        p["disks"].add(row["disk_name"])
        if row["disk_name"] == target_disk:
            p["parts_on_target"] += count

    result: list[dict] = []
    for p in partitions.values():
        if partition_filter is not None and p["partition_id"] not in partition_filter:
            continue
        if before is not None:
            dt = extract_partition_date(p["partition"])
            if dt is None or dt >= before:
                continue
        if p["parts_on_target"] == p["parts_total"]:
            p["status"] = "on_target"
        elif p["parts_on_target"] > 0:
            p["status"] = "partial"
        else:
            p["status"] = "local"
        p["disks"] = sorted(p["disks"])
        result.append(p)

    result.sort(key=lambda x: x["partition_id"])
    return result


# ---------------------------------------------------------------------------
# Cluster node discovery
# ---------------------------------------------------------------------------

def get_cluster_nodes(
    host: str,
    port: int,
    cluster: str,
    user: str | None = None,
    password: str | None = None,
) -> list[tuple[str, int]]:
    """Get (ip, http_port) for every node in the cluster."""
    sql = (
        f"SELECT DISTINCT host_address "
        f"FROM system.clusters "
        f"WHERE cluster = '{cluster}' "
        f"ORDER BY host_address"
    )
    rows = query_json_rows(host, port, sql, timeout=30, user=user, password=password)
    return [(r["host_address"], port) for r in rows]


def get_cluster_hostnames(
    host: str,
    port: int,
    cluster: str,
    user: str | None = None,
    password: str | None = None,
) -> dict[str, str]:
    """Get mapping of host_address -> hostname for cluster nodes."""
    sql = (
        f"SELECT host_address, host_name "
        f"FROM system.clusters "
        f"WHERE cluster = '{cluster}' "
        f"ORDER BY host_address"
    )
    rows = query_json_rows(host, port, sql, timeout=30, user=user, password=password)
    return {r["host_address"]: r["host_name"] for r in rows}


def get_node_hostname(
    host: str,
    port: int,
    user: str | None = None,
    password: str | None = None,
) -> str:
    """Get the hostname of a node by querying it directly."""
    sql = "SELECT hostName() AS h"
    rows = query_json_rows(host, port, sql, timeout=10, user=user, password=password)
    return rows[0]["h"] if rows else host


def expand_host_pattern(
    pattern: str,
    port: int,
    cluster: str,
    user: str | None = None,
    password: str | None = None,
) -> tuple[str, list[tuple[str, int]]]:
    """Expand a host pattern containing %s (shard) and/or %r (replica)."""
    probe_host = pattern.replace("%s", "0").replace("%r", "0")
    sql = (
        f"SELECT shard_num, replica_num "
        f"FROM system.clusters "
        f"WHERE cluster = '{cluster}' "
        f"ORDER BY shard_num, replica_num"
    )
    rows = query_json_rows(probe_host, port, sql, timeout=30, user=user, password=password)
    nodes: list[tuple[str, int]] = []
    for row in rows:
        s = int(row["shard_num"]) - 1
        r = int(row["replica_num"]) - 1
        node_host = pattern.replace("%s", str(s)).replace("%r", str(r))
        nodes.append((node_host, port))
    return probe_host, nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_size(b: int) -> str:
    if b >= 1024 ** 4:
        return f"{b / 1024**4:.1f} TB"
    if b >= 1024 ** 3:
        return f"{b / 1024**3:.1f} GB"
    if b >= 1024 ** 2:
        return f"{b / 1024**2:.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    hours = seconds // 3600
    mins = (seconds % 3600) // 60
    return f"{hours}h{mins:02d}m"


def truncate_error(error: str, max_len: int = 200) -> str:
    idx = error.find("Stack trace")
    if idx > 0:
        error = error[:idx].rstrip()
    if len(error) > max_len:
        return error[:max_len] + "..."
    return error


def _short_host(hostname: str) -> str:
    parts = hostname.rsplit("-", 1)
    return parts[-1] if parts[-1].isdigit() else hostname


# ---------------------------------------------------------------------------
# Per-node partition move
# ---------------------------------------------------------------------------

def move_partition(
    host: str,
    port: int,
    database: str,
    table: str,
    partition_id: str,
    target_disk: str = DEFAULT_DISK,
    user: str | None = None,
    password: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> None:
    """Issue ALTER TABLE MOVE PARTITION to target disk on a single node."""
    sql = (
        f"ALTER TABLE `{database}`.`{table}` "
        f"MOVE PARTITION ID '{partition_id}' "
        f"TO DISK '{target_disk}'"
    )
    query(host, port, sql, timeout=timeout, user=user, password=password)


def poll_move_done(
    host: str,
    port: int,
    database: str,
    table: str,
    partition_id: str,
    target_disk: str = DEFAULT_DISK,
    user: str | None = None,
    password: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    on_progress: Callable[[str], None] | None = None,
) -> bool:
    """Poll until a partition's parts are all on the target disk.

    Returns True if move completed, raises TimeoutError on timeout.
    """
    deadline = time.monotonic() + timeout
    while not _shutdown:
        time.sleep(POLL_INTERVAL)

        # Check if all parts are now on target disk.
        sql = (
            f"SELECT disk_name, count() AS cnt "
            f"FROM system.parts "
            f"WHERE database = '{database}' AND table = '{table}' "
            f"AND partition_id = '{partition_id}' AND active = 1 "
            f"GROUP BY disk_name"
        )
        rows = query_json_rows(host, port, sql, timeout=30, user=user, password=password)

        all_on_target = all(r["disk_name"] == target_disk for r in rows) if rows else False
        if all_on_target:
            return True

        # Check if move is still in progress.
        sql_moves = (
            f"SELECT count() AS cnt FROM system.moves "
            f"WHERE database = '{database}' AND table = '{table}' "
            f"AND substring(part_name, 1, position(part_name, '_') - 1) = '{partition_id}'"
        )
        move_rows = query_json_rows(host, port, sql_moves, timeout=30, user=user, password=password)
        active_moves = int(move_rows[0]["cnt"]) if move_rows else 0

        if on_progress:
            disks = ", ".join(f"{r['disk_name']}={r['cnt']}" for r in rows)
            status = f"moving ({disks})" if active_moves > 0 else f"waiting ({disks})"
            on_progress(status)

        # No active move and not all on target — move may have failed.
        if active_moves == 0 and not all_on_target:
            # Recheck parts to be sure (race condition window).
            time.sleep(2)
            rows2 = query_json_rows(host, port,
                f"SELECT disk_name, count() AS cnt FROM system.parts "
                f"WHERE database = '{database}' AND table = '{table}' "
                f"AND partition_id = '{partition_id}' AND active = 1 "
                f"GROUP BY disk_name",
                timeout=30, user=user, password=password,
            )
            if all(r["disk_name"] == target_disk for r in rows2) if rows2 else False:
                return True
            return False

        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Partition {partition_id} move not complete after {timeout}s"
            )

    return False


# ---------------------------------------------------------------------------
# Per-node table tiering
# ---------------------------------------------------------------------------

def tier_table_on_node(
    node_host: str,
    node_port: int,
    node_name: str,
    database: str,
    table: str,
    target_disk: str = DEFAULT_DISK,
    cutoff: datetime | None = None,
    user: str | None = None,
    password: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
    on_partition_done: Callable[[], None] | None = None,
    partition_filter: set[str] | None = None,
) -> dict:
    """Move all eligible partitions of a table to target disk on one node."""
    # Discover partitions needing move.
    partitions = get_node_partitions(
        node_host, node_port, database, table, cutoff, target_disk, user, password,
        partition_filter,
    )

    # Discover partitions currently being moved.
    moving = get_node_moving_partitions(
        node_host, node_port, database, table, user, password,
    )

    results = {"moved": 0, "skipped": 0, "already_moving": 0, "failed": 0}
    details: list[dict] = []

    if not partitions:
        return {
            "node": node_name,
            "database": database,
            "table": table,
            **results,
            "partition_detail": [],
        }

    for part_info in partitions:
        if _shutdown:
            break

        partition_id = part_info["partition_id"]
        total_rows = part_info["total_rows"]
        bytes_on_disk = part_info["bytes_on_disk"]
        disks = ", ".join(part_info["disks"])

        # Skip if currently being moved.
        if partition_id in moving:
            log(f"  {partition_id}  MOVING  (already in progress)  rows={total_rows:>12,}  size={format_size(bytes_on_disk)}")
            results["already_moving"] += 1
            details.append({
                "partition_id": partition_id,
                "outcome": "already_moving",
                "rows": total_rows,
                "bytes": bytes_on_disk,
            })
            if on_partition_done:
                on_partition_done()
            continue

        if dry_run:
            log(f"  {partition_id}  DRY     disks=[{disks}]  rows={total_rows:>12,}  size={format_size(bytes_on_disk)}")
            details.append({
                "partition_id": partition_id,
                "outcome": "dry_run",
                "disks": disks,
                "rows": total_rows,
                "bytes": bytes_on_disk,
            })
            if on_partition_done:
                on_partition_done()
            continue

        for attempt in range(1, retries + 1):
            try:
                log(f"  {partition_id}  MOVE    disks=[{disks}]  rows={total_rows:>12,}  size={format_size(bytes_on_disk)}")
                move_partition(
                    node_host, node_port, database, table,
                    partition_id, target_disk, user, password, timeout,
                )

                # Verify the move landed.
                def on_progress(status: str):
                    log(f"  {partition_id}  {status}")

                ok = poll_move_done(
                    node_host, node_port, database, table,
                    partition_id, target_disk, user, password, timeout,
                    on_progress=on_progress,
                )

                if ok:
                    log(f"  {partition_id}  DONE")
                    results["moved"] += 1
                    details.append({
                        "partition_id": partition_id,
                        "outcome": "moved",
                        "rows": total_rows,
                        "bytes": bytes_on_disk,
                    })
                else:
                    raise ClickHouseError(f"Move did not complete for {partition_id}")
                break

            except (ClickHouseError, TimeoutError) as e:
                err = truncate_error(str(e))
                if attempt < retries:
                    wait = 10 * attempt
                    log(f"  {partition_id}  RETRY   attempt {attempt}/{retries} ({err}) — waiting {wait}s")
                    time.sleep(wait)
                    continue
                log(f"  {partition_id}  ERROR   {err} (after {retries} attempts)")
                results["failed"] += 1
                details.append({
                    "partition_id": partition_id,
                    "outcome": "failed",
                    "error": err,
                    "rows": total_rows,
                    "bytes": bytes_on_disk,
                })

        if on_partition_done:
            on_partition_done()

    return {
        "node": node_name,
        "database": database,
        "table": table,
        **results,
        "partition_detail": details,
    }


def write_state_log(result: dict, target_disk: str) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **result,
        "target_disk": target_disk,
        "partitions_total": len(result.get("partition_detail", [])),
    }
    with open(STATE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Display modes
# ---------------------------------------------------------------------------

def show_status(
    probe_host: str,
    probe_port: int,
    nodes: list[tuple[str, int]],
    node_names: dict[str, str],
    database: str,
    tables: list[str],
    cutoff: datetime | None,
    target_disk: str,
    user: str | None,
    password: str | None,
    partition_filter: set[str] | None = None,
) -> None:
    """Show partition disk placement per node."""
    for table in tables:
        local_db, local_table, is_dist = resolve_table(
            probe_host, probe_port, database, table, user, password,
        )
        suffix = f" -> {local_table}" if is_dist else ""
        print(f"\n{database}.{table}{suffix}")

        for node_host, node_port in nodes:
            name = node_names.get(node_host, node_host)
            parts = get_node_all_partitions(
                node_host, node_port, local_db, local_table,
                cutoff, target_disk, user, password, partition_filter,
            )
            if not parts:
                print(f"  [{name}] no partitions")
                continue

            on_target = sum(1 for p in parts if p["status"] == "on_target")
            partial = sum(1 for p in parts if p["status"] == "partial")
            local = sum(1 for p in parts if p["status"] == "local")
            total_bytes = sum(p["bytes_on_disk"] for p in parts)
            local_bytes = sum(p["bytes_on_disk"] for p in parts if p["status"] != "on_target")

            print(f"  [{name}] {len(parts)} partitions, {format_size(total_bytes)}")
            print(f"    on {target_disk}: {on_target}  partial: {partial}  local: {local} ({format_size(local_bytes)} to move)")


def show_dry_run(
    probe_host: str,
    probe_port: int,
    nodes: list[tuple[str, int]],
    node_names: dict[str, str],
    database: str,
    tables: list[str],
    cutoff: datetime | None,
    target_disk: str,
    user: str | None,
    password: str | None,
    partition_filter: set[str] | None = None,
) -> None:
    """Preview what would be moved."""
    grand_total = 0
    grand_bytes = 0
    for table in tables:
        local_db, local_table, is_dist = resolve_table(
            probe_host, probe_port, database, table, user, password,
        )
        suffix = f" -> {local_table}" if is_dist else ""
        print(f"\n{database}.{table}{suffix}")

        for node_host, node_port in nodes:
            name = node_names.get(node_host, node_host)
            parts = get_node_partitions(
                node_host, node_port, local_db, local_table,
                cutoff, target_disk, user, password, partition_filter,
            )
            moving = get_node_moving_partitions(
                node_host, node_port, local_db, local_table, user, password,
            )

            if not parts:
                print(f"  [{name}] nothing to move")
                continue

            node_bytes = sum(p["bytes_on_disk"] for p in parts)
            print(f"  [{name}] {len(parts)} partitions to move ({format_size(node_bytes)})")
            for p in parts:
                status = "MOVING" if p["partition_id"] in moving else "MOVE"
                disks = ", ".join(p["disks"])
                print(
                    f"    {p['partition_id']:>30}  {status:<7}  "
                    f"disks=[{disks}]  rows={p['total_rows']:>12,}  "
                    f"size={format_size(p['bytes_on_disk'])}"
                )
            grand_total += len(parts)
            grand_bytes += node_bytes

    print(f"\n[DRY RUN] {grand_total} partitions to move ({format_size(grand_bytes)}) — no changes made")


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_tiering(
    args,
    nodes: list[tuple[str, int]],
    node_names: dict[str, str],
    cutoff: datetime | None,
    partition_filter: set[str] | None = None,
) -> None:
    import threading

    tables = args.table
    target_disk = args.disk
    max_concurrent = args.max_concurrent

    # Build work items: (node, table) pairs.
    print(f"Discovering partitions across {len(nodes)} nodes, {len(tables)} table(s)...")

    work_items: list[dict] = []
    for table in tables:
        local_db, local_table, is_dist = resolve_table(
            nodes[0][0], nodes[0][1], args.database, table, args.user, args.password,
        )
        if is_dist:
            print(f"  {args.database}.{table} -> {local_db}.{local_table}")

        for node_host, node_port in nodes:
            name = node_names.get(node_host, node_host)
            parts = get_node_partitions(
                node_host, node_port, local_db, local_table,
                cutoff, target_disk, args.user, args.password,
                partition_filter,
            )
            moving = get_node_moving_partitions(
                node_host, node_port, local_db, local_table,
                args.user, args.password,
            )
            total = len(parts)
            already_moving = sum(1 for p in parts if p["partition_id"] in moving)
            to_move = total - already_moving
            total_bytes = sum(p["bytes_on_disk"] for p in parts)
            print(f"    [{name}] {total} partitions ({to_move} to move, {already_moving} in progress, {format_size(total_bytes)})")
            work_items.append({
                "node_host": node_host,
                "node_port": node_port,
                "node_name": name,
                "database": local_db,
                "table": local_table,
                "source_table": table,
                "partitions": total,
            })

    total_partitions = sum(w["partitions"] for w in work_items)
    print(f"\nTotal: {len(work_items)} (node, table) pairs, {total_partitions} partitions to process")
    print(f"Target disk: {target_disk}")
    print(f"Max concurrent: {max_concurrent}\n")

    all_results: list[dict] = []
    has_failures = False

    progress = Progress(
        TextColumn("{task.fields[short_name]:<55}"),
        BarColumn(bar_width=30),
        TextColumn("{task.fields[progress_text]:>7}"),
        TextColumn("{task.fields[status_text]}"),
        TimeElapsedColumn(),
    )

    with progress:
        task_ids: dict[str, TaskID] = {}
        for w in work_items:
            key = f"{w['node_name']}/{w['source_table']}"
            tid = progress.add_task(
                key,
                total=w["partitions"] or 1,
                short_name=f"[bold]{key[:55]}",
                progress_text=f"0/{w['partitions']}",
                status_text="queued",
            )
            task_ids[key] = tid

        overall_tid = progress.add_task(
            "overall",
            total=total_partitions or 1,
            short_name="[cyan bold]OVERALL",
            progress_text=f"0/{total_partitions}",
            status_text="",
        )

        done_count = 0
        count_lock = threading.Lock()

        def do_one(w: dict) -> dict:
            nonlocal done_count
            key = f"{w['node_name']}/{w['source_table']}"
            tid = task_ids[key]
            partitions_done = 0

            progress.update(tid, status_text="moving...")

            def on_partition_done():
                nonlocal done_count, partitions_done
                partitions_done += 1
                progress.advance(tid)
                progress.advance(overall_tid)
                progress.update(
                    tid,
                    progress_text=f"{partitions_done}/{w['partitions']}",
                )
                with count_lock:
                    done_count += 1
                progress.update(
                    overall_tid,
                    progress_text=f"{done_count}/{total_partitions}",
                    status_text=f"{done_count} done",
                )

            def log(msg: str):
                progress.update(tid, status_text=msg.strip()[:60])

            result = tier_table_on_node(
                node_host=w["node_host"],
                node_port=w["node_port"],
                node_name=w["node_name"],
                database=w["database"],
                table=w["table"],
                target_disk=target_disk,
                cutoff=cutoff,
                user=args.user,
                password=args.password,
                timeout=args.timeout,
                retries=args.retries,
                dry_run=False,
                log=log,
                on_partition_done=on_partition_done,
                partition_filter=partition_filter,
            )

            m = result["moved"]
            s = result["skipped"]
            am = result["already_moving"]
            f = result["failed"]
            status = f"done: {m} moved, {s} skip, {am} in-prog"
            if f > 0:
                status += f", [red]{f} fail[/red]"
            progress.update(
                tid,
                status_text=status,
                progress_text=f"{w['partitions']}/{w['partitions']}",
                completed=w["partitions"] or 1,
            )
            return result

        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {pool.submit(do_one, w): w for w in work_items}
            for future in as_completed(futures):
                w = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    write_state_log(result, target_disk)
                    if result["failed"] > 0:
                        has_failures = True
                except Exception as e:
                    key = f"{w['node_name']}/{w['source_table']}"
                    progress.update(
                        task_ids[key],
                        status_text=f"[red]CRASHED: {e}[/red]",
                    )
                    has_failures = True

    # Final summary.
    print(f"\n{'=' * 80}")
    print(f"{'Node/Table':<55} {'Moved':>5} {'Skip':>5} {'Prog':>5} {'Fail':>5}")
    print(f"{'-' * 80}")
    totals = {"moved": 0, "skipped": 0, "already_moving": 0, "failed": 0}
    for r in sorted(all_results, key=lambda x: f"{x['node']}/{x['table']}"):
        name = f"{r['node']}/{r['table']}"
        if len(name) > 53:
            name = name[:52] + "…"
        print(f"  {name:<53} {r['moved']:>5} {r['skipped']:>5} {r['already_moving']:>5} {r['failed']:>5}")
        for k in totals:
            totals[k] += r[k]
    print(f"{'-' * 80}")
    print(f"  {'TOTAL':<53} {totals['moved']:>5} {totals['skipped']:>5} {totals['already_moving']:>5} {totals['failed']:>5}")
    print(f"\nState log: {STATE_LOG}")

    if has_failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_cutoff(before: str | None) -> datetime:
    """Return date cutoff. Default: first of current month.

    Partitions with date < cutoff are eligible. So the default moves
    everything before the current month (e.g. in April, moves up to and
    including March).
    """
    if before:
        return datetime.strptime(before, "%Y-%m-%d")

    now = datetime.now()
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move partitions between disks, per-node (no DDL queue). "
                    "Use --disk to choose the target disk (s3_cache tier-down or "
                    "default restore).",
    )
    parser.add_argument("--host", default="localhost",
                        help="ClickHouse HTTP host (supports %%s/%%r pattern for shard/replica)")
    parser.add_argument("--port", type=int, default=None,
                        help="ClickHouse HTTP port (default: 443 if --secure, else 8123)")
    parser.add_argument("--secure", action="store_true", help="Use HTTPS (auto-enabled when port is 443)")
    parser.add_argument("--user", default=None, help="ClickHouse user")
    parser.add_argument("--password", default=None, help="ClickHouse password")
    parser.add_argument("--database", required=True, help="Database name")
    parser.add_argument("--table", required=True, nargs="+", help="Table name(s)")
    parser.add_argument("--disk", default=DEFAULT_DISK,
                        help=f"Target disk name (default: {DEFAULT_DISK})")
    parser.add_argument("--before",
                        help="Move partitions strictly before YYYY-MM-DD (default: 1st of current month)")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER,
                        help=f"Cluster name for node discovery (default: {DEFAULT_CLUSTER})")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be moved")
    parser.add_argument("--status", action="store_true", help="Show partition disk placement")
    parser.add_argument("--partition", nargs="+", default=None,
                        help="Only process these partition IDs (space-separated)")
    parser.add_argument("--all-partitions", action="store_true",
                        help="Include all partitions (ignore date filter)")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
                        help=f"Max concurrent (node, table) pairs (default: {DEFAULT_MAX_CONCURRENT})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Timeout per move in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help=f"Retry failed moves up to N times (default: {DEFAULT_RETRIES})")

    args = parser.parse_args()

    # Resolve secure/port defaults.
    global _secure
    if args.secure or args.port == 443:
        _secure = True
    if args.port is None:
        args.port = 443 if _secure else 8123

    partition_filter: set[str] | None = None
    if args.partition:
        partition_filter = set(args.partition)
        print(f"Partition filter: {', '.join(sorted(partition_filter))}")

    cutoff: datetime | None = None
    if not args.all_partitions:
        cutoff = compute_cutoff(args.before)
        print(f"Partition cutoff: before {cutoff.strftime('%Y-%m-%d')}")

    # Discover cluster nodes.
    if "%s" in args.host or "%r" in args.host:
        probe_host, nodes = expand_host_pattern(
            args.host, args.port, args.cluster, args.user, args.password,
        )
    else:
        probe_host = args.host
        nodes = get_cluster_nodes(
            args.host, args.port, args.cluster, args.user, args.password,
        )

    # Resolve hostnames for display.
    node_names: dict[str, str] = {}
    for node_host, node_port in nodes:
        try:
            name = get_node_hostname(node_host, node_port, args.user, args.password)
            node_names[node_host] = name
        except ClickHouseError:
            node_names[node_host] = node_host

    print(f"Cluster: {args.cluster} ({len(nodes)} nodes)")
    for node_host, _ in nodes:
        print(f"  {node_names.get(node_host, node_host)} ({node_host})")
    print(f"Target disk: {args.disk}\n")

    if args.status:
        show_status(
            probe_host, args.port, nodes, node_names,
            args.database, args.table, cutoff, args.disk,
            args.user, args.password, partition_filter,
        )
        return

    if args.dry_run:
        show_dry_run(
            probe_host, args.port, nodes, node_names,
            args.database, args.table, cutoff, args.disk,
            args.user, args.password, partition_filter,
        )
        return

    run_tiering(args, nodes, node_names, cutoff, partition_filter)


if __name__ == "__main__":
    import signal

    def _handle_sigint(sig, frame):
        global _shutdown
        if _shutdown:
            print("\nForce quit.", file=sys.stderr)
            sys.exit(130)
        _shutdown = True
        print("\nCtrl+C received — finishing in-flight move, then stopping...", file=sys.stderr)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
