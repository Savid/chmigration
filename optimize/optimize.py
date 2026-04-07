#!/usr/bin/env python3
"""Run OPTIMIZE TABLE FINAL on ClickHouse tables, partition by partition.

Discovers active partitions from system.parts and runs OPTIMIZE TABLE
PARTITION ... FINAL on each one. Partitions that are already fully merged
(single active part) are skipped automatically.

Supports multiple tables with concurrent execution and Rich progress display.

Usage:
    # Single table
    python optimize/optimize.py \
      --host localhost --user admin --password secret \
      --database default --table beacon_api_eth_v1_events_block

    # Multiple tables concurrently
    python optimize/optimize.py \
      --host localhost --user admin --password secret \
      --database default \
      --table beacon_api_eth_v1_events_block \
             canonical_beacon_block \
      --max-concurrent 5

    # Dry run — show what would be optimized
    python optimize/optimize.py \
      --host localhost --user admin --password secret \
      --database default --table beacon_api_eth_v1_events_block --dry-run

    # Status — show partition merge state
    python optimize/optimize.py \
      --host localhost --user admin --password secret \
      --database default --table beacon_api_eth_v1_events_block --status
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
DEFAULT_TIMEOUT = 3600
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_RETRIES = 3
POLL_INTERVAL = 10  # seconds between part-count checks
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
                                    headers={"User-Agent": "clickhouse-optimize/1.0"})
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

def resolve_table(host: str, port: int, database: str, table: str, user: str | None = None, password: str | None = None) -> tuple[str, str, bool]:
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

def get_partitions(
    host: str,
    port: int,
    database: str,
    table: str,
    cluster: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> list[dict]:
    """Get partition info from system.parts grouped by partition.

    Returns list of dicts with: partition_id, partition, part_count, total_rows,
    bytes_on_disk.  Sorted by partition_id.
    """
    source = (
        f"clusterAllReplicas('{cluster}', system.parts)"
        if cluster
        else "system.parts"
    )
    sql = (
        f"SELECT "
        f"  partition_id, "
        f"  partition, "
        f"  count() AS part_count, "
        f"  sum(rows) AS total_rows, "
        f"  sum(bytes_on_disk) AS bytes_on_disk "
        f"FROM {source} "
        f"WHERE database = '{database}' AND table = '{table}' AND active = 1 "
        f"GROUP BY partition_id, partition "
        f"ORDER BY partition_id"
    )
    return query_json_rows(host, port, sql, user=user, password=password)


def get_unmerged_partitions(
    host: str,
    port: int,
    database: str,
    table: str,
    cluster: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> list[dict]:
    """Get partitions that have more than one active part (need merging)."""
    all_parts = get_partitions(host, port, database, table, cluster, user, password)
    return [p for p in all_parts if int(p.get("part_count", 0)) > 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_size(b: int) -> str:
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


# ---------------------------------------------------------------------------
# Per-partition optimize
# ---------------------------------------------------------------------------

def submit_optimize(
    host: str,
    port: int,
    database: str,
    table: str,
    partition_id: str,
    cluster: str | None = None,
    user: str | None = None,
    password: str | None = None,
    nodes: list[tuple[str, int]] | None = None,
) -> None:
    """Fire OPTIMIZE TABLE ... PARTITION ... FINAL.

    If *nodes* is provided, sends the OPTIMIZE directly to each node's HTTP
    endpoint (bypasses the distributed DDL queue entirely).  Otherwise falls
    back to ON CLUSTER which goes through the DDL queue.
    """
    if nodes:
        sql = (
            f"OPTIMIZE TABLE `{database}`.`{table}`"
            f" PARTITION ID '{partition_id}'"
            f" FINAL"
            f" SETTINGS optimize_throw_if_noop = 0"
        )
        for node_host, node_port in nodes:
            query(node_host, node_port, sql, timeout=60, user=user, password=password)
    else:
        cluster_clause = f" ON CLUSTER '{cluster}'" if cluster else ""
        settings = ["optimize_throw_if_noop = 0"]
        if cluster:
            settings.append("distributed_ddl_task_timeout = 0")
        settings_clause = ", ".join(settings)
        sql = (
            f"OPTIMIZE TABLE `{database}`.`{table}`"
            f"{cluster_clause}"
            f" PARTITION ID '{partition_id}'"
            f" FINAL"
            f" SETTINGS {settings_clause}"
        )
        query(host, port, sql, timeout=60, user=user, password=password)


def get_partition_part_count(
    host: str,
    port: int,
    database: str,
    table: str,
    partition_id: str,
    cluster: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> int:
    """Check current active part count for a partition (across all replicas if cluster)."""
    source = (
        f"clusterAllReplicas('{cluster}', system.parts)"
        if cluster
        else "system.parts"
    )
    sql = (
        f"SELECT count() AS cnt FROM {source} "
        f"WHERE database = '{database}' AND table = '{table}' "
        f"AND partition_id = '{partition_id}' AND active = 1"
    )
    rows = query_json_rows(host, port, sql, timeout=30, user=user, password=password)
    return int(rows[0].get("cnt", 0)) if rows else 0


def get_merge_progress(
    host: str,
    port: int,
    database: str,
    table: str,
    partition_id: str,
    cluster: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> list[dict] | None:
    """Get per-node merge progress for a partition from system.merges.

    Returns a list of per-node dicts with host, progress, elapsed, num_parts,
    sorted by host.  Returns None if no active merges.
    """
    source = (
        f"clusterAllReplicas('{cluster}', system.merges)"
        if cluster
        else "system.merges"
    )
    sql = (
        f"SELECT "
        f"  hostName() AS host, "
        f"  num_parts, "
        f"  progress, "
        f"  elapsed "
        f"FROM {source} "
        f"WHERE database = '{database}' AND table = '{table}' "
        f"AND partition_id = '{partition_id}' "
        f"ORDER BY host"
    )
    rows = query_json_rows(host, port, sql, timeout=30, user=user, password=password)
    if not rows:
        return None
    return [
        {
            "host": row["host"],
            "num_parts": int(row["num_parts"]),
            "progress": float(row["progress"]),
            "elapsed": float(row["elapsed"]),
        }
        for row in rows
    ]


def get_cluster_hosts(
    host: str,
    port: int,
    cluster: str,
    user: str | None = None,
    password: str | None = None,
) -> list[str]:
    """Get sorted list of hostnames in a cluster."""
    sql = f"SELECT hostName() AS h FROM clusterAllReplicas('{cluster}', system.one) ORDER BY h"
    rows = query_json_rows(host, port, sql, timeout=30, user=user, password=password)
    return [r["h"] for r in rows]


def get_cluster_nodes(
    host: str,
    port: int,
    cluster: str,
    user: str | None = None,
    password: str | None = None,
) -> list[tuple[str, int]]:
    """Get (ip, http_port) for every node in the cluster.

    Uses the same HTTP port as the current connection since all nodes
    share the same config.
    """
    sql = (
        f"SELECT DISTINCT host_address "
        f"FROM system.clusters "
        f"WHERE cluster = '{cluster}' "
        f"ORDER BY host_address"
    )
    rows = query_json_rows(host, port, sql, timeout=30, user=user, password=password)
    return [(r["host_address"], port) for r in rows]


def expand_host_pattern(
    pattern: str,
    port: int,
    cluster: str,
    user: str | None = None,
    password: str | None = None,
) -> tuple[str, list[tuple[str, int]]]:
    """Expand a host pattern containing %s (shard) and/or %r (replica).

    Connects to the 0-0 expansion to discover the cluster topology, then
    generates all (host, port) pairs.

    Returns (probe_host, nodes) where probe_host is the 0-0 expansion
    used for monitoring queries.
    """
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


def _short_host(hostname: str) -> str:
    """Shorten a hostname like 'clickhouse-refined-3' to just the trailing number."""
    parts = hostname.rsplit("-", 1)
    return parts[-1] if parts[-1].isdigit() else hostname


def _format_node_elapsed(nodes: list[dict]) -> str:
    """Format per-node merge progress with elapsed time."""
    parts: list[str] = []
    for n in nodes:
        name = _short_host(n["host"])
        pct = n["progress"] * 100
        elapsed = format_duration(n["elapsed"])
        parts.append(f"{name}:{pct:.0f}%/{elapsed}")
    return " ".join(parts)


def poll_optimize_done(
    host: str,
    port: int,
    database: str,
    table: str,
    partition_id: str,
    initial_parts: int,
    cluster: str | None = None,
    user: str | None = None,
    password: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    on_merge_progress: Callable[[str, list[dict] | None], None] | None = None,
    num_nodes: int = 1,
) -> int:
    """Poll system.parts until the partition's part count drops (merge landed).

    Calls on_merge_progress(partition_id, nodes) each cycle where nodes is the
    per-host merge list or None when no merge is active.

    When *num_nodes* > 1 (cluster with clusterAllReplicas monitoring), the
    minimum possible part count is num_nodes (1 per replica).  If the count
    is already at that minimum and no merge is active, returns immediately.

    Returns the final part count.  Raises TimeoutError if timeout is exceeded.
    """
    deadline = time.monotonic() + timeout
    no_merge_cycles = 0
    while not _shutdown:
        time.sleep(POLL_INTERVAL)
        current = get_partition_part_count(
            host, port, database, table, partition_id, cluster, user, password,
        )
        if current < initial_parts:
            if on_merge_progress:
                on_merge_progress(partition_id, None)
            return current

        merge_nodes = get_merge_progress(
            host, port, database, table, partition_id, cluster, user, password,
        )

        # Already at minimum (1 part per replica) and no merge running — done.
        if current <= num_nodes and not merge_nodes:
            if on_merge_progress:
                on_merge_progress(partition_id, None)
            return current

        # No merge running and count hasn't dropped — OPTIMIZE was a no-op.
        if not merge_nodes:
            no_merge_cycles += 1
            if no_merge_cycles >= 3:
                if on_merge_progress:
                    on_merge_progress(partition_id, None)
                return current
        else:
            no_merge_cycles = 0

        if on_merge_progress:
            on_merge_progress(partition_id, merge_nodes)

        if time.monotonic() > deadline:
            if on_merge_progress:
                on_merge_progress(partition_id, None)
            raise TimeoutError(
                f"Partition {partition_id} still has {current} parts after {timeout}s"
            )
    if on_merge_progress:
        on_merge_progress(partition_id, None)
    return get_partition_part_count(
        host, port, database, table, partition_id, cluster, user, password,
    )


# ---------------------------------------------------------------------------
# Per-table optimize
# ---------------------------------------------------------------------------

def optimize_table(
    host: str,
    port: int,
    database: str,
    table: str,
    cluster: str | None = None,
    user: str | None = None,
    password: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    log: Callable[[str], None] = print,
    on_partition_done: Callable[[], None] | None = None,
    on_merge_progress: Callable[[str, list[dict] | None], None] | None = None,
    nodes: list[tuple[str, int]] | None = None,
    max_partition_concurrent: int = 1,
) -> dict:
    """Optimize all partitions of a single table.

    When *nodes* is set, OPTIMIZEs are sent directly to each node (bypassing
    the DDL queue).  *max_partition_concurrent* controls how many partitions
    are optimized in parallel within this table.

    Returns results dict with per-partition detail.
    """
    import threading

    local_db, local_table, is_distributed = resolve_table(
        host, port, database, table, user, password,
    )
    if is_distributed:
        log(f"  {database}.{table} -> {local_db}.{local_table}")

    all_parts = get_partitions(host, port, local_db, local_table, cluster, user, password)

    results = {"optimized": 0, "skipped": 0, "failed": 0}
    details: list[dict] = []
    results_lock = threading.Lock()

    if not all_parts:
        return {
            "database": local_db,
            "table": local_table,
            **results,
            "partition_detail": [],
        }

    # Handle skipped partitions first.
    to_optimize: list[dict] = []
    for part_info in all_parts:
        partition_id = part_info["partition_id"]
        part_count = int(part_info.get("part_count", 0))
        total_rows = int(part_info.get("total_rows", 0))
        bytes_on_disk = int(part_info.get("bytes_on_disk", 0))

        num_nodes = len(nodes) if nodes else 1
        if part_count <= num_nodes:
            log(f"  {partition_id}  SKIP   parts={part_count} ({num_nodes} nodes)  rows={total_rows:>12,}  size={format_size(bytes_on_disk)}")
            results["skipped"] += 1
            details.append({
                "partition_id": partition_id,
                "outcome": "skipped",
                "part_count": part_count,
                "rows": total_rows,
                "bytes": bytes_on_disk,
            })
            if on_partition_done:
                on_partition_done()
        else:
            to_optimize.append(part_info)

    def _do_partition(part_info: dict) -> None:
        if _shutdown:
            return
        partition_id = part_info["partition_id"]
        part_count = int(part_info.get("part_count", 0))
        total_rows = int(part_info.get("total_rows", 0))
        bytes_on_disk = int(part_info.get("bytes_on_disk", 0))

        for attempt in range(1, retries + 1):
            try:
                log(f"  {partition_id}  OPT    parts={part_count}  rows={total_rows:>12,}  size={format_size(bytes_on_disk)}")
                submit_optimize(
                    host, port, local_db, local_table,
                    partition_id, cluster, user, password,
                    nodes=nodes,
                )
                new_count = poll_optimize_done(
                    host, port, local_db, local_table,
                    partition_id, part_count, cluster, user, password, timeout,
                    on_merge_progress=on_merge_progress,
                    num_nodes=len(nodes) if nodes else 1,
                )
                log(f"  {partition_id}  DONE   parts={part_count} -> {new_count}")
                with results_lock:
                    results["optimized"] += 1
                    details.append({
                        "partition_id": partition_id,
                        "outcome": "optimized",
                        "part_count_before": part_count,
                        "part_count_after": new_count,
                        "rows": total_rows,
                        "bytes": bytes_on_disk,
                    })
                break

            except (ClickHouseError, TimeoutError) as e:
                err = truncate_error(str(e))
                if attempt < retries:
                    wait = 10 * attempt
                    log(f"  {partition_id}  RETRY  attempt {attempt}/{retries} ({err}) — waiting {wait}s")
                    time.sleep(wait)
                    continue
                log(f"  {partition_id}  ERROR  {err} (after {retries} attempts)")
                with results_lock:
                    results["failed"] += 1
                    details.append({
                        "partition_id": partition_id,
                        "outcome": "failed",
                        "error": err,
                        "part_count": part_count,
                        "rows": total_rows,
                        "bytes": bytes_on_disk,
                    })

        if on_partition_done:
            on_partition_done()

    if max_partition_concurrent <= 1 or len(to_optimize) <= 1:
        for part_info in to_optimize:
            if _shutdown:
                break
            _do_partition(part_info)
    else:
        with ThreadPoolExecutor(max_workers=max_partition_concurrent) as pool:
            futures = [pool.submit(_do_partition, p) for p in to_optimize]
            for f in as_completed(futures):
                f.result()

    return {
        "database": local_db,
        "table": local_table,
        **results,
        "partition_detail": details,
    }


def write_state_log(result: dict, host: str) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **result,
        "host": host,
        "partitions_total": len(result.get("partition_detail", [])),
    }
    with open(STATE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Display modes
# ---------------------------------------------------------------------------

def show_status(
    host: str,
    port: int,
    database: str,
    tables: list[str],
    cluster: str | None,
    user: str | None,
    password: str | None,
) -> None:
    """Show partition merge state for each table."""
    for table in tables:
        local_db, local_table, is_dist = resolve_table(host, port, database, table, user, password)
        suffix = f" -> {local_table}" if is_dist else ""
        parts = get_partitions(host, port, local_db, local_table, cluster, user, password)
        if not parts:
            print(f"\n{database}.{table}{suffix}: no partitions found")
            continue

        total_parts = sum(int(p.get("part_count", 0)) for p in parts)
        merged = sum(1 for p in parts if int(p.get("part_count", 0)) <= 1)
        unmerged = len(parts) - merged
        total_bytes = sum(int(p.get("bytes_on_disk", 0)) for p in parts)
        print(f"\n{database}.{table}{suffix}: {len(parts)} partitions, {total_parts} parts, {format_size(total_bytes)}")
        print(f"  Merged: {merged}  Unmerged: {unmerged}")
        print()

        for p in parts:
            part_count = int(p.get("part_count", 0))
            total_rows = int(p.get("total_rows", 0))
            bytes_on_disk = int(p.get("bytes_on_disk", 0))
            status = "OK" if part_count <= 1 else f"NEEDS MERGE ({part_count} parts)"
            print(f"  {p['partition_id']:>30}  {status:<25}  rows={total_rows:>12,}  size={format_size(bytes_on_disk)}")


def show_dry_run(
    host: str,
    port: int,
    database: str,
    tables: list[str],
    cluster: str | None,
    user: str | None,
    password: str | None,
) -> None:
    """Preview what would be optimized."""
    total_to_optimize = 0
    total_skip = 0
    for table in tables:
        local_db, local_table, is_dist = resolve_table(host, port, database, table, user, password)
        suffix = f" -> {local_table}" if is_dist else ""
        parts = get_partitions(host, port, local_db, local_table, cluster, user, password)

        unmerged = [p for p in parts if int(p.get("part_count", 0)) > 1]
        merged = len(parts) - len(unmerged)
        print(f"\n{database}.{table}{suffix}: {len(unmerged)} to optimize, {merged} already merged")

        for p in unmerged:
            cluster_clause = f" ON CLUSTER '{cluster}'" if cluster else ""
            sql = (
                f"OPTIMIZE TABLE `{local_db}`.`{local_table}`"
                f"{cluster_clause}"
                f" PARTITION ID '{p['partition_id']}'"
                f" FINAL"
            )
            part_count = int(p.get("part_count", 0))
            bytes_on_disk = int(p.get("bytes_on_disk", 0))
            print(f"  {p['partition_id']:>30}  parts={part_count}  size={format_size(bytes_on_disk)}  {sql}")

        total_to_optimize += len(unmerged)
        total_skip += merged

    print(f"\n[DRY RUN] {total_to_optimize} partitions to optimize, {total_skip} already merged — no changes made")


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_single_table(args, cluster: str | None, nodes: list[tuple[str, int]] | None = None) -> None:
    table = args.table[0]

    local_db, local_table, is_distributed = resolve_table(
        args.host, args.port, args.database, table, args.user, args.password,
    )
    if is_distributed:
        print(f"Distributed table {args.database}.{table} -> local {local_db}.{local_table}")
    else:
        print(f"Local table: {local_db}.{local_table}")

    all_parts = get_partitions(args.host, args.port, local_db, local_table, cluster, args.user, args.password)
    if not all_parts:
        print("No partitions found")
        return

    unmerged = [p for p in all_parts if int(p.get("part_count", 0)) > 1]
    print(f"Found {len(all_parts)} partition(s), {len(unmerged)} need optimization")
    if nodes:
        print(f"Direct mode: {len(nodes)} nodes")
    print(f"Partitions concurrent: {args.partitions_concurrent}\n")

    def _single_merge_progress(partition_id: str, nodes_data: list[dict] | None):
        if nodes_data:
            print(f"  {partition_id}  MERGE  {_format_node_elapsed(nodes_data)}")

    result = optimize_table(
        host=args.host,
        port=args.port,
        database=args.database,
        table=table,
        cluster=cluster,
        user=args.user,
        password=args.password,
        timeout=args.timeout,
        retries=args.retries,
        on_merge_progress=_single_merge_progress,
        nodes=nodes,
        max_partition_concurrent=args.partitions_concurrent,
    )

    total = sum(result[k] for k in ("optimized", "skipped", "failed"))
    print(f"\n{'=' * 60}")
    print(f"Optimize complete: {local_db}.{local_table}")
    print(f"  Optimized:  {result['optimized']:>4} / {total}")
    print(f"  Skipped:    {result['skipped']:>4} (already merged)")
    print(f"  Failed:     {result['failed']:>4}")

    write_state_log(result, args.host)
    print(f"  State log:  {STATE_LOG}")

    if result["failed"] > 0:
        sys.exit(1)


def run_multi_table(args, cluster: str | None, nodes: list[tuple[str, int]] | None = None) -> None:
    import threading

    tables = args.table
    max_concurrent = args.max_concurrent

    # Discovery phase.
    print(f"Discovering partitions for {len(tables)} table(s)...")
    table_info: list[dict] = []
    for table in tables:
        local_db, local_table, is_dist = resolve_table(
            args.host, args.port, args.database, table, args.user, args.password,
        )
        parts = get_partitions(args.host, args.port, local_db, local_table, cluster, args.user, args.password)
        unmerged = [p for p in parts if int(p.get("part_count", 0)) > 1]
        table_info.append({
            "source_table": table,
            "local_db": local_db,
            "local_table": local_table,
            "total_partitions": len(parts),
            "unmerged": len(unmerged),
        })
        suffix = f" -> {local_table}" if is_dist else ""
        print(f"  {args.database}.{table}{suffix}: {len(parts)} partitions, {len(unmerged)} need optimization")

    # Discover cluster hosts for per-node progress.
    cluster_hosts: list[str] = []
    if cluster:
        try:
            cluster_hosts = get_cluster_hosts(
                args.host, args.port, cluster, args.user, args.password,
            )
        except ClickHouseError:
            pass

    total_partitions = sum(t["total_partitions"] for t in table_info)
    total_unmerged = sum(t["unmerged"] for t in table_info)
    print(f"\nTotal: {len(tables)} tables, {total_partitions} partitions, {total_unmerged} need optimization")
    if cluster_hosts:
        print(f"Cluster nodes: {len(cluster_hosts)} ({', '.join(_short_host(h) for h in cluster_hosts)})")
    if nodes:
        print(f"Direct mode: {len(nodes)} nodes (bypassing DDL queue)")
    print(f"Max concurrent tables: {max_concurrent}, partitions/table: {args.partitions_concurrent}\n")

    all_results: list[dict] = []
    has_failures = False

    progress = Progress(
        TextColumn("{task.fields[short_name]:<45}"),
        BarColumn(bar_width=30),
        TextColumn("{task.fields[progress_text]:>7}"),
        TextColumn("{task.fields[status_text]}"),
        TimeElapsedColumn(),
    )

    with progress:
        task_ids: dict[str, TaskID] = {}
        # node_tasks[table_name][short_host] = TaskID
        node_tasks: dict[str, dict[str, TaskID]] = {}

        for info in table_info:
            table_name = info["source_table"]
            tid = progress.add_task(
                table_name,
                total=info["total_partitions"] or 1,
                short_name=f"[bold]{table_name[:45]}",
                progress_text=f"0/{info['total_partitions']}",
                status_text="queued",
            )
            task_ids[table_name] = tid

            # Pre-create hidden node sub-tasks under each table.
            node_tasks[table_name] = {}
            for ch in cluster_hosts:
                short = _short_host(ch)
                ntid = progress.add_task(
                    f"{table_name}-{short}",
                    total=1000,
                    completed=0,
                    short_name=f"  [dim]node-{short}[/dim]",
                    progress_text="",
                    status_text="",
                    visible=False,
                )
                node_tasks[table_name][short] = ntid

        overall_tid = progress.add_task(
            "overall",
            total=total_partitions or 1,
            short_name="[cyan bold]OVERALL",
            progress_text=f"0/{total_partitions}",
            status_text="",
        )

        optimized_count = 0
        count_lock = threading.Lock()

        def do_one_table(info: dict) -> dict:
            nonlocal optimized_count
            table_name = info["source_table"]
            tid = task_ids[table_name]
            my_nodes = node_tasks.get(table_name, {})
            partitions_done = 0

            progress.update(tid, status_text="optimizing...")

            def on_partition_done():
                nonlocal optimized_count, partitions_done
                partitions_done += 1
                progress.advance(tid)
                progress.advance(overall_tid)
                progress.update(
                    tid,
                    progress_text=f"{partitions_done}/{info['total_partitions']}",
                )
                with count_lock:
                    optimized_count += 1
                progress.update(
                    overall_tid,
                    progress_text=f"{optimized_count}/{total_partitions}",
                    status_text=f"{optimized_count} done",
                )

            def log(msg: str):
                progress.update(tid, status_text=msg.strip())

            def on_merge_progress(partition_id: str, nodes: list[dict] | None):
                if nodes:
                    progress.update(tid, status_text=f"{partition_id}  merging ({len(nodes)} nodes)")
                    hosts_seen = set()
                    for n in nodes:
                        short = _short_host(n["host"])
                        hosts_seen.add(short)
                        ntid = my_nodes.get(short)
                        if ntid is None:
                            continue
                        pct = n["progress"] * 100
                        completed = int(n["progress"] * 1000)
                        elapsed_str = format_duration(n["elapsed"])
                        progress.update(
                            ntid,
                            completed=completed,
                            progress_text=f"{pct:.1f}%",
                            status_text=elapsed_str,
                            visible=True,
                        )
                    # Hide nodes not actively merging.
                    for short, ntid in my_nodes.items():
                        if short not in hosts_seen:
                            progress.update(ntid, visible=False)
                else:
                    # No active merge — hide all node bars.
                    for ntid in my_nodes.values():
                        progress.update(ntid, completed=0, progress_text="",
                                        status_text="", visible=False)

            result = optimize_table(
                host=args.host,
                port=args.port,
                database=args.database,
                table=table_name,
                cluster=cluster,
                user=args.user,
                password=args.password,
                timeout=args.timeout,
                retries=args.retries,
                log=log,
                on_partition_done=on_partition_done,
                on_merge_progress=on_merge_progress,
                nodes=nodes,
                max_partition_concurrent=args.partitions_concurrent,
            )

            # Hide node bars when table finishes.
            for ntid in my_nodes.values():
                progress.update(ntid, visible=False)

            o = result["optimized"]
            s = result["skipped"]
            f = result["failed"]
            status = f"done: {o} opt, {s} skip"
            if f > 0:
                status += f", [red]{f} fail[/red]"
            progress.update(
                tid,
                status_text=status,
                progress_text=f"{info['total_partitions']}/{info['total_partitions']}",
                completed=info["total_partitions"] or 1,
            )
            return result

        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {pool.submit(do_one_table, info): info for info in table_info}
            for future in as_completed(futures):
                info = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    write_state_log(result, args.host)
                    if result["failed"] > 0:
                        has_failures = True
                except Exception as e:
                    progress.update(
                        task_ids[info["source_table"]],
                        status_text=f"[red]CRASHED: {e}[/red]",
                    )
                    has_failures = True

    # Final summary.
    print(f"\n{'=' * 70}")
    print(f"{'Table':<50} {'Opt':>5} {'Skip':>5} {'Fail':>5}")
    print(f"{'-' * 70}")
    totals = {"optimized": 0, "skipped": 0, "failed": 0}
    for r in sorted(all_results, key=lambda x: x["table"]):
        name = f"{r['database']}.{r['table']}"
        if len(name) > 48:
            name = name[:47] + "…"
        print(f"  {name:<48} {r['optimized']:>5} {r['skipped']:>5} {r['failed']:>5}")
        for k in totals:
            totals[k] += r[k]
    print(f"{'-' * 70}")
    print(f"  {'TOTAL':<48} {totals['optimized']:>5} {totals['skipped']:>5} {totals['failed']:>5}")
    print(f"\nState log: {STATE_LOG}")

    if has_failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize ClickHouse tables partition by partition (OPTIMIZE TABLE FINAL)",
    )
    parser.add_argument("--host", default="localhost", help="ClickHouse HTTP host (supports %%s/%%r pattern for shard/replica)")
    parser.add_argument("--port", type=int, default=None, help="ClickHouse HTTP port (default: 443 if --secure, else 8123)")
    parser.add_argument("--secure", action="store_true", help="Use HTTPS (auto-enabled when port is 443)")
    parser.add_argument("--user", default=None, help="ClickHouse user")
    parser.add_argument("--password", default=None, help="ClickHouse password")
    parser.add_argument("--database", required=True, help="Database name")
    parser.add_argument("--table", required=True, nargs="+", help="Table name(s)")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER, help=f"Cluster name (default: {DEFAULT_CLUSTER})")
    parser.add_argument("--no-cluster", action="store_true", help="Don't use ON CLUSTER or clusterAllReplicas")
    parser.add_argument("--direct", action="store_true",
                        help="Send OPTIMIZE directly to each node (bypass DDL queue)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be optimized")
    parser.add_argument("--status", action="store_true", help="Show partition merge state")
    parser.add_argument(
        "--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
        help=f"Max tables to optimize concurrently (default: {DEFAULT_MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--partitions-concurrent", type=int, default=1,
        help="Max partitions to optimize concurrently per table (default: 1)",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Timeout per OPTIMIZE query in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--retries", type=int, default=DEFAULT_RETRIES,
        help=f"Retry failed partitions up to N times (default: {DEFAULT_RETRIES})",
    )
    args = parser.parse_args()

    # Resolve secure/port defaults.
    global _secure
    if args.secure or args.port == 443:
        _secure = True
    if args.port is None:
        args.port = 443 if _secure else 8123

    cluster = None if args.no_cluster else args.cluster

    # Expand host pattern (auto-enables direct mode).
    nodes = None
    if "%s" in args.host or "%r" in args.host:
        if not cluster:
            print("ERROR: host pattern requires a cluster (don't use --no-cluster)", file=sys.stderr)
            sys.exit(1)
        probe_host, nodes = expand_host_pattern(
            args.host, args.port, cluster, args.user, args.password,
        )
        args.host = probe_host
        args.direct = True
        print(f"Host pattern expanded to {len(nodes)} nodes:")
        for node_host, node_port in nodes:
            print(f"  {node_host}:{node_port}")
    elif args.direct:
        if not cluster:
            print("ERROR: --direct requires a cluster (don't use --no-cluster)", file=sys.stderr)
            sys.exit(1)
        nodes = get_cluster_nodes(args.host, args.port, cluster, args.user, args.password)
        print(f"Direct mode: discovered {len(nodes)} nodes")
        for node_host, node_port in nodes:
            print(f"  {node_host}:{node_port}")

    if args.status:
        show_status(args.host, args.port, args.database, args.table, cluster, args.user, args.password)
        return

    if args.dry_run:
        show_dry_run(args.host, args.port, args.database, args.table, cluster, args.user, args.password)
        return

    if len(args.table) == 1:
        run_single_table(args, cluster, nodes)
    else:
        run_multi_table(args, cluster, nodes)


if __name__ == "__main__":
    import signal

    def _handle_sigint(sig, frame):
        global _shutdown
        if _shutdown:
            print("\nForce quit.", file=sys.stderr)
            sys.exit(130)
        _shutdown = True
        print("\nCtrl+C received — finishing in-flight optimize, then stopping...", file=sys.stderr)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
