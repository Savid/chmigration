#!/usr/bin/env python3
"""Restore ClickHouse tables from an S3 backup disk, partition by partition.

Discovers completed backups from the source cluster's system.backup_log, then
restores each partition into the target cluster using native RESTORE TABLE
commands.

The script is idempotent: it checks system.backup_log on the target cluster
before issuing a restore and skips partitions that are already completed or
in progress.

Supports multiple tables with concurrent execution and Rich progress display.

Prerequisites:
  - An S3-backed disk (type: s3_plain) named (default: s3_backup) must be
    configured on the target ClickHouse cluster, pointing at the same bucket
    as the source cluster's backup disk.
  - Target tables must already exist with the same schema as the source.

Usage:
    # Single table
    python restore/restore.py \\
      --source-host old-cluster:8123 \\
      --host localhost --user admin --password secret \\
      --database default --table beacon_api_eth_v1_events_block

    # Multiple tables concurrently
    python restore/restore.py \\
      --source-host old-cluster:8123 \\
      --host localhost --user admin --password secret \\
      --database default \\
      --table beacon_api_eth_v1_events_block \\
             beacon_api_eth_v1_events_head \\
             canonical_beacon_block \\
      --max-concurrent 3

    # Other modes
    python restore/restore.py --source-host old-cluster --host localhost \\
      --database default --table beacon_api_eth_v1_events_block --dry-run
    python restore/restore.py --source-host old-cluster --host localhost \\
      --database default --table beacon_api_eth_v1_events_block --status
"""

import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

# Force line-buffered stdout so output appears immediately when piped/redirected.
sys.stdout.reconfigure(line_buffering=True)

# Graceful shutdown flag — set by SIGINT handler so in-flight restores finish
# but no new partitions are started.
_shutdown = False

from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TaskID


DEFAULT_DISK = "s3_backup"
DEFAULT_CLUSTER = "replicated"
DEFAULT_TIMEOUT = 3600
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_MAX_CONCURRENT_PARTITIONS = 1
SCRIPT_DIR = Path(__file__).resolve().parent
STATE_LOG = SCRIPT_DIR / "state.jsonl"


# ---------------------------------------------------------------------------
# ClickHouse HTTP helpers
# ---------------------------------------------------------------------------

class ClickHouseError(Exception):
    """ClickHouse query error with the server's error message preserved."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


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
    url = f"http://{host}:{port}/?{urllib.parse.urlencode(params)}"
    data = sql.encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8").strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace").strip()
        raise ClickHouseError(body, status=e.code) from e
    except urllib.error.URLError as e:
        print(f"Connection error ({host}:{port}): {e}", file=sys.stderr)
        sys.exit(1)


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
# Table resolution (on source cluster)
# ---------------------------------------------------------------------------

def resolve_table(host: str, port: int, database: str, table: str) -> tuple[str, str, bool]:
    """Resolve a table name. If Distributed, return the underlying local table.

    Returns (database, table, is_distributed).
    """
    rows = query_json_rows(
        host, port,
        f"SELECT engine, engine_full FROM system.tables "
        f"WHERE database = '{database}' AND name = '{table}'",
    )
    if not rows:
        return database, table, False

    engine = rows[0].get("engine", "")
    engine_full = rows[0].get("engine_full", "")

    if engine != "Distributed":
        return database, table, False

    # Distributed('cluster', 'db', 'local_table'[, sharding_key])
    m = re.match(r"Distributed\(\s*'[^']+'\s*,\s*'([^']+)'\s*,\s*'([^']+)'", engine_full)
    if m:
        return m.group(1), m.group(2), True

    return database, table, False


# ---------------------------------------------------------------------------
# Backup discovery from source cluster
# ---------------------------------------------------------------------------

def discover_backups(
    host: str,
    port: int,
    database: str,
    table: str,
    cluster: str | None = None,
) -> list[dict]:
    """Query system.backup_log on the source cluster to find completed backups.

    Returns list of dicts with keys: name, path, partition_id.
    The backup name format is: Disk('s3_backup', '{db}/{table}/{partition_id}')
    """
    source = (
        f"clusterAllReplicas('{cluster}', system.backup_log)"
        if cluster
        else "system.backup_log"
    )
    # Match backup names for this specific table.
    pattern = f"default/{table}/%"
    sql = (
        "SELECT DISTINCT name "
        f"FROM {source} "
        f"WHERE status = 'BACKUP_CREATED' AND name LIKE '%{pattern}'"
        " ORDER BY name"
    )
    rows = query_json_rows(host, port, sql)

    backups: list[dict] = []
    for row in rows:
        name = row["name"]
        # Extract path from Disk('s3_backup', 'default/table/partition_id')
        m = re.search(r"Disk\(\\'s3_backup\\',\s*\\'([^']+)\\'\)", name)
        if not m:
            # Try without escapes (format varies).
            m = re.search(r"Disk\('s3_backup',\s*'([^']+)'\)", name)
        if not m:
            continue
        path = m.group(1)
        parts = path.split("/")
        if len(parts) != 3:
            continue
        partition_id = parts[2]
        backups.append({
            "name": name,
            "path": path,
            "partition_id": partition_id,
        })
    return backups


# ---------------------------------------------------------------------------
# Restore status checking (on target cluster)
# ---------------------------------------------------------------------------

def check_restore_status(
    host: str,
    port: int,
    backup_path: str,
    cluster: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> tuple[str | None, dict | None]:
    """Check for an existing restore of *backup_path* on the target.

    Tries system.backup_log first (persistent), falls back to system.backups
    (in-memory) if backup_log is not available.

    Strategy: first check for terminal statuses (RESTORED / RESTORE_FAILED).
    Only if none are found, check for RESTORING (actually in-flight).
    This avoids confusing stale RESTORING rows (backup_log logs both start
    and end as separate rows) with genuinely in-progress restores.

    Returns (status, row) or (None, None) when no matching restore exists.
    """
    safe_path = backup_path.replace("'", "\\'")

    for table in ("system.backup_log", "system.backups"):
        source = (
            f"clusterAllReplicas('{cluster}', {table})"
            if cluster
            else table
        )

        # 1. Check terminal statuses first (RESTORED wins over RESTORE_FAILED).
        sql = (
            "SELECT id, name, status, error, "
            "  start_time, end_time, num_files, "
            "  total_size "
            f"FROM {source} "
            f"WHERE name LIKE '%{safe_path}%' "
            "  AND status IN ('RESTORED', 'RESTORE_FAILED') "
            "ORDER BY status = 'RESTORED' DESC, start_time DESC "
            "LIMIT 1"
        )
        try:
            rows = query_json_rows(host, port, sql, timeout=30, user=user, password=password)
        except ClickHouseError:
            continue
        if rows:
            return rows[0].get("status"), rows[0]

        # 2. No terminal status — check if genuinely in-flight.
        sql = (
            "SELECT id, name, status, error, "
            "  start_time, end_time, num_files, "
            "  total_size "
            f"FROM {source} "
            f"WHERE name LIKE '%{safe_path}%' "
            "  AND status = 'RESTORING' "
            "ORDER BY start_time DESC LIMIT 1"
        )
        try:
            rows = query_json_rows(host, port, sql, timeout=30, user=user, password=password)
        except ClickHouseError:
            continue
        if rows:
            return "RESTORING", rows[0]

    return None, None


def truncate_error(error: str, max_len: int = 200) -> str:
    """Truncate ClickHouse error messages, stripping stack traces."""
    # Strip everything after "Stack trace" if present.
    idx = error.find("Stack trace")
    if idx > 0:
        error = error[:idx].rstrip()
    if len(error) > max_len:
        return error[:max_len] + "..."
    return error


def format_size(b: int) -> str:
    if b >= 1024 ** 3:
        return f"{b / 1024**3:.1f} GB"
    if b >= 1024 ** 2:
        return f"{b / 1024**2:.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


# ---------------------------------------------------------------------------
# Per-table restore logic
# ---------------------------------------------------------------------------

POLL_INTERVAL = 10  # seconds between async restore status checks


def do_restore_partition(
    host: str,
    port: int,
    database: str,
    table: str,
    partition_id: str,
    disk: str,
    cluster: str | None,
    backup_path: str,
    user: str | None = None,
    password: str | None = None,
) -> tuple[str, dict | None]:
    """Issue an ASYNC RESTORE and poll until completion.

    Returns (status, info_dict) where status is 'RESTORED' or 'RESTORE_FAILED'.
    """
    cluster_clause = f" ON CLUSTER '{cluster}'" if cluster else ""

    sql = (
        f"RESTORE TABLE `{database}`.`{table}`"
        f"{cluster_clause}"
        f" FROM Disk('{disk}', '{backup_path}')"
        f" SETTINGS allow_non_empty_tables=true, allow_different_table_def=true"
        f" ASYNC"
    )
    query(host, port, sql, timeout=60, user=user, password=password)

    # Poll until done.
    while not _shutdown:
        time.sleep(POLL_INTERVAL)
        status, info = check_restore_status(host, port, backup_path, cluster, user, password)
        if status is None:
            continue
        if status != "RESTORING":
            return status, info

    return "INTERRUPTED", None


def restore_table(
    host: str,
    port: int,
    source_host: str,
    source_port: int,
    source_database: str,
    source_table: str,
    disk: str,
    cluster: str | None,
    source_cluster: str | None,
    user: str | None = None,
    password: str | None = None,
    force_partitions: set[str] | None = None,
    max_concurrent_partitions: int = 1,
    log: Callable[[str], None] = print,
    on_partition_done: Callable[[], None] | None = None,
    on_bytes_restored: Callable[[int], None] | None = None,
) -> dict:
    """Restore all backed-up partitions of a single table.

    Partitions run concurrently up to *max_concurrent_partitions*; setting it
    to 1 preserves the original sequential behavior.

    When *force_partitions* is non-empty, the run is restricted to those
    partition IDs and the RESTORED idempotency check is bypassed. Callers
    must ALTER TABLE ... DROP PARTITION on the target first, or the fresh
    restore will mix with existing rows.

    Returns a results dict with keys: database, table, source_table, restored,
    skipped, failed, partition_detail.
    """
    local_db, local_table, is_distributed = resolve_table(
        source_host, source_port, source_database, source_table,
    )
    if is_distributed:
        log(f"  {source_database}.{source_table} -> {local_db}.{local_table}")

    backups = discover_backups(source_host, source_port, local_db, local_table, source_cluster)
    if force_partitions:
        backups = [bk for bk in backups if bk["partition_id"] in force_partitions]
        found = {bk["partition_id"] for bk in backups}
        missing = sorted(force_partitions - found)
        if missing:
            log(f"  WARNING: force-partition not found in source backup_log: {missing}")

    results = {"restored": 0, "skipped": 0, "failed": 0}
    partition_outcomes: dict[str, str] = {}
    results_lock = threading.Lock()

    if not backups:
        return {
            "database": local_db,
            "table": local_table,
            "source_table": f"{source_database}.{source_table}",
            **results,
            "partition_detail": [],
        }

    def handle_partition(bk: dict) -> None:
        if _shutdown:
            return

        partition_id = bk["partition_id"]
        backup_path = bk["path"]

        force = bool(force_partitions) and partition_id in force_partitions
        status, info = check_restore_status(host, port, backup_path, cluster, user, password)

        if status == "RESTORED":
            if force:
                # Bypass the skip; fall through to issue a new RESTORE.
                # Caller is expected to have dropped the target partition.
                log(f"  {partition_id} force re-restore (ignoring previous RESTORED)")
                status, info = None, None
            else:
                log(f"  {partition_id} already restored — skipping")
                with results_lock:
                    results["skipped"] += 1
                    partition_outcomes[partition_id] = "skipped"
                if on_partition_done:
                    on_partition_done()
                return

        if status == "RESTORING":
            # Always wait, even under force — issuing a concurrent RESTORE to
            # the same path would collide.
            log(f"  {partition_id} waiting for in-progress restore...")
            while not _shutdown:
                time.sleep(POLL_INTERVAL)
                status, info = check_restore_status(host, port, backup_path, cluster, user, password)
                if status != "RESTORING":
                    break
            if _shutdown:
                return
            if status == "RESTORED":
                restored_bytes = int(info.get("total_size", 0)) if info else 0
                log(f"  {partition_id} completed -> {format_size(restored_bytes)}")
                with results_lock:
                    results["restored"] += 1
                    partition_outcomes[partition_id] = "restored"
                if on_bytes_restored:
                    on_bytes_restored(restored_bytes)
            else:
                err = truncate_error(info.get("error", "unknown")) if info else "unknown"
                log(f"  {partition_id} FAILED after wait: {err}")
                with results_lock:
                    results["failed"] += 1
                    partition_outcomes[partition_id] = "failed"
            if on_partition_done:
                on_partition_done()
            return

        if status == "RESTORE_FAILED":
            log(f"  {partition_id} retrying after previous failure")

        log(f"  {partition_id} issuing RESTORE...")
        # Run async restore + poll.
        try:
            rs_status, rs_info = do_restore_partition(
                host, port, local_db, local_table,
                partition_id, disk, cluster, backup_path,
                user, password,
            )
            if rs_status == "RESTORED":
                restored_bytes = int(rs_info.get("total_size", 0)) if rs_info else 0
                log(f"  {partition_id} restored -> {format_size(restored_bytes)}")
                with results_lock:
                    results["restored"] += 1
                    partition_outcomes[partition_id] = "restored"
                if on_bytes_restored:
                    on_bytes_restored(restored_bytes)
            elif rs_status == "INTERRUPTED":
                return
            else:
                err = truncate_error(rs_info.get("error", "unknown")) if rs_info else "unknown"
                log(f"  {partition_id} FAILED: {err}")
                with results_lock:
                    results["failed"] += 1
                    partition_outcomes[partition_id] = "failed"
        except ClickHouseError as e:
            log(f"  {partition_id} ERROR: {e}")
            with results_lock:
                results["failed"] += 1
                partition_outcomes[partition_id] = "failed"

        if on_partition_done:
            on_partition_done()

    workers = max(1, max_concurrent_partitions)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(handle_partition, bk) for bk in backups]
        for f in as_completed(futures):
            f.result()

    return {
        "database": local_db,
        "table": local_table,
        "source_table": f"{source_database}.{source_table}",
        **results,
        "partition_detail": [
            {
                "partition_id": bk["partition_id"],
                "backup_path": bk["path"],
                "outcome": partition_outcomes.get(bk["partition_id"], "unknown"),
            }
            for bk in backups
        ],
    }


def write_state_log(result: dict, host: str, cluster: str | None, disk: str) -> None:
    """Append a JSONL entry to the state log."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **result,
        "host": host,
        "cluster": cluster,
        "disk": disk,
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
    source_host: str,
    source_port: int,
    database: str,
    tables: list[str],
    cluster: str | None,
    source_cluster: str | None,
    user: str | None,
    password: str | None,
) -> None:
    """Print restore status for each table's partitions."""
    for table in tables:
        local_db, local_table, _ = resolve_table(source_host, source_port, database, table)
        backups = discover_backups(source_host, source_port, local_db, local_table, source_cluster)
        print(f"\n{database}.{table} ({len(backups)} backed-up partitions)")

        for bk in backups:
            status, info = check_restore_status(host, port, bk["path"], cluster, user, password)

            if status == "RESTORED":
                size_mb = int(info.get("total_size", 0)) / (1024 * 1024)
                print(f"  {bk['partition_id']:>30}  DONE  {info.get('num_files', '?')} files  {size_mb:.1f} MB")
            elif status == "RESTORING":
                print(f"  {bk['partition_id']:>30}  IN PROGRESS  (since {info.get('start_time', '?')})")
            elif status == "RESTORE_FAILED":
                print(f"  {bk['partition_id']:>30}  FAILED  {truncate_error(info.get('error', '?')) if info else '?'}")
            else:
                print(f"  {bk['partition_id']:>30}  NOT RESTORED")


def show_dry_run(
    source_host: str,
    source_port: int,
    database: str,
    tables: list[str],
    disk: str,
    cluster: str | None,
    source_cluster: str | None,
) -> None:
    """Preview what would be restored."""
    total_partitions = 0
    for table in tables:
        local_db, local_table, is_dist = resolve_table(source_host, source_port, database, table)
        backups = discover_backups(source_host, source_port, local_db, local_table, source_cluster)
        suffix = f" -> {local_table}" if is_dist else ""
        print(f"\n{database}.{table}{suffix} ({len(backups)} partitions to restore)")

        for bk in backups:
            cluster_clause = f" ON CLUSTER '{cluster}'" if cluster else ""
            sql = (
                f"RESTORE TABLE `{local_db}`.`{local_table}`"
                f"{cluster_clause}"
                f" FROM Disk('{disk}', '{bk['path']}')"
                f" SETTINGS allow_non_empty_tables=true, allow_different_table_def=true ASYNC"
            )
            print(f"  {bk['partition_id']:>30}  {sql}")

        total_partitions += len(backups)

    print(f"\n[DRY RUN] {total_partitions} partitions across {len(tables)} table(s) — no restores issued")


def show_verify(
    host: str,
    port: int,
    source_host: str,
    source_port: int,
    database: str,
    tables: list[str],
    cluster: str | None,
    source_cluster: str | None,
    user: str | None,
    password: str | None,
) -> bool:
    """Compare row counts on target vs source for each partition.

    Returns True if all partitions match.
    """
    all_ok = True

    for table in tables:
        local_db, local_table, is_dist = resolve_table(source_host, source_port, database, table)
        backups = discover_backups(source_host, source_port, local_db, local_table, source_cluster)
        suffix = f" -> {local_table}" if is_dist else ""
        print(f"\n{database}.{table}{suffix} ({len(backups)} partitions)")

        counts = {"ok": 0, "mismatch": 0, "missing": 0, "failed": 0}

        for bk in backups:
            partition_id = bk["partition_id"]

            # Row count on source.
            source_sql = (
                f"SELECT sum(rows) AS rows FROM "
                f"clusterAllReplicas('{source_cluster}', system.parts) "
                f"WHERE database = '{local_db}' AND table = '{local_table}' "
                f"AND partition_id = '{partition_id}' AND active = 1"
            ) if source_cluster else (
                f"SELECT sum(rows) AS rows FROM system.parts "
                f"WHERE database = '{local_db}' AND table = '{local_table}' "
                f"AND partition_id = '{partition_id}' AND active = 1"
            )
            src_rows_result = query_json_rows(source_host, source_port, source_sql)
            src_rows = int(src_rows_result[0].get("rows", 0)) if src_rows_result else 0

            # Row count on target.
            target_parts_source = (
                f"clusterAllReplicas('{cluster}', system.parts)"
                if cluster
                else "system.parts"
            )
            target_sql = (
                f"SELECT sum(rows) AS rows FROM {target_parts_source} "
                f"WHERE database = '{local_db}' AND table = '{local_table}' "
                f"AND partition_id = '{partition_id}' AND active = 1"
            )
            tgt_rows_result = query_json_rows(host, port, target_sql, user=user, password=password)
            tgt_rows = int(tgt_rows_result[0].get("rows", 0)) if tgt_rows_result else 0

            if tgt_rows == 0:
                print(f"  {partition_id:>15}   MISS   source={src_rows:>12,}   target=0")
                counts["missing"] += 1
            elif src_rows == tgt_rows:
                print(f"  {partition_id:>15}   OK     source={src_rows:>12,}   target={tgt_rows:>12,}")
                counts["ok"] += 1
            else:
                diff_pct = ((tgt_rows - src_rows) / src_rows * 100) if src_rows else 0
                print(f"  {partition_id:>15}   DIFF   source={src_rows:>12,}   target={tgt_rows:>12,}   ({diff_pct:+.1f}%)")
                counts["mismatch"] += 1

        parts = []
        if counts["ok"]:
            parts.append(f"{counts['ok']} ok")
        if counts["mismatch"]:
            parts.append(f"{counts['mismatch']} row mismatch")
        if counts["missing"]:
            parts.append(f"{counts['missing']} missing")
        if counts["failed"]:
            parts.append(f"{counts['failed']} failed")
        print(f"  Summary: {', '.join(parts)}")

        if counts["missing"] or counts["failed"] or counts["mismatch"]:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_single_table(args, cluster: str | None, source_cluster: str | None) -> None:
    """Run restore for a single table with verbose print output."""
    table = args.table[0]

    local_db, local_table, is_distributed = resolve_table(
        args.source_host, args.source_port, args.database, table,
    )
    if is_distributed:
        print(f"Distributed table {args.database}.{table} -> local {local_db}.{local_table}")
    else:
        print(f"Local table: {local_db}.{local_table}")

    backups = discover_backups(args.source_host, args.source_port, local_db, local_table, source_cluster)
    force_partitions: set[str] = set(args.force_partition) if args.force_partition else set()
    if force_partitions:
        backups = [bk for bk in backups if bk["partition_id"] in force_partitions]
        found = {bk["partition_id"] for bk in backups}
        missing = sorted(force_partitions - found)
        if missing:
            print(f"WARNING: force-partition not found in source backup_log: {missing}")
    if not backups:
        print("No backups found for this table")
        return

    print(f"Found {len(backups)} backed-up partition(s)")
    if force_partitions:
        print(f"Force re-restore mode: {sorted(force_partitions)} (RESTORED check bypassed)")

    # Concurrent path: delegate to restore_table (no per-partition polling dots,
    # but partitions run in parallel up to --max-concurrent-partitions).
    if args.max_concurrent_partitions > 1 or force_partitions:
        print(f"Restoring up to {args.max_concurrent_partitions} partitions concurrently\n")
        result = restore_table(
            host=args.host,
            port=args.port,
            source_host=args.source_host,
            source_port=args.source_port,
            source_database=args.database,
            source_table=table,
            disk=args.disk,
            cluster=cluster,
            source_cluster=source_cluster,
            user=args.user,
            password=args.password,
            force_partitions=force_partitions or None,
            max_concurrent_partitions=args.max_concurrent_partitions,
        )
        total = result["restored"] + result["skipped"] + result["failed"]
        print(f"\n{'=' * 60}")
        print(f"Restore complete: {local_db}.{local_table}")
        print(f"  Restored:     {result['restored']:>4} / {total}")
        print(f"  Skipped:      {result['skipped']:>4} (already restored)")
        print(f"  Failed:       {result['failed']:>4}")
        write_state_log(result, args.host, cluster, args.disk)
        print(f"  State log:    {STATE_LOG}")
        if result["failed"] > 0:
            sys.exit(1)
        return

    results = {"restored": 0, "skipped": 0, "failed": 0}
    partition_outcomes: dict[str, str] = {}
    bytes_restored = 0
    t0 = time.monotonic()

    for i, bk in enumerate(backups, 1):
        if _shutdown:
            print("\nInterrupted — stopping after current partition...")
            break

        partition_id = bk["partition_id"]
        backup_path = bk["path"]

        print(f"\n[{i}/{len(backups)}] {partition_id}")

        # Idempotency check.
        status, info = check_restore_status(
            args.host, args.port, backup_path, cluster, args.user, args.password,
        )

        if status == "RESTORED":
            print(f"  Already restored")
            results["skipped"] += 1
            partition_outcomes[partition_id] = "skipped"
            continue

        if status == "RESTORING":
            started = info.get("start_time", "?") if info else "?"
            print(f"  Waiting for in-progress restore (since {started})...", end="", flush=True)
            while not _shutdown:
                time.sleep(POLL_INTERVAL)
                status, info = check_restore_status(
                    args.host, args.port, backup_path, cluster, args.user, args.password,
                )
                if status != "RESTORING":
                    break
                print(".", end="", flush=True)
            print()
            if _shutdown:
                print("\nInterrupted — restore will continue on the cluster")
                break
            if status == "RESTORED":
                restored = int(info.get("total_size", 0)) if info else 0
                bytes_restored += restored
                elapsed = time.monotonic() - t0
                throughput = (bytes_restored / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                print(f"  Completed ({format_size(restored)}) [{throughput:.1f} MB/s avg]")
                results["restored"] += 1
                partition_outcomes[partition_id] = "restored"
            else:
                err = truncate_error(info.get("error", "unknown")) if info else "unknown"
                print(f"  FAILED after wait: {err}")
                results["failed"] += 1
                partition_outcomes[partition_id] = "failed"
            continue

        if status == "RESTORE_FAILED":
            print(f"  Previous restore failed: {truncate_error(info.get('error', '?')) if info else '?'}")
            print(f"  Retrying...")

        try:
            cluster_clause = f" ON CLUSTER '{cluster}'" if cluster else ""
            sql = (
                f"RESTORE TABLE `{local_db}`.`{local_table}`"
                f"{cluster_clause}"
                f" FROM Disk('{args.disk}', '{backup_path}')"
                f" SETTINGS allow_non_empty_tables=true, allow_different_table_def=true"
                f" ASYNC"
            )
            print(f"  SQL: {sql}")
            query(args.host, args.port, sql, timeout=60, user=args.user, password=args.password)

            # Poll until done.
            rs_status, rs_info = None, None
            print(f"  Polling...", end="", flush=True)
            while not _shutdown:
                time.sleep(POLL_INTERVAL)
                rs_status, rs_info = check_restore_status(
                    args.host, args.port, backup_path, cluster, args.user, args.password,
                )
                if rs_status and rs_status != "RESTORING":
                    break
                print(".", end="", flush=True)
            print()

            if _shutdown:
                print("\nInterrupted — restore will continue on the cluster")
                break

            if rs_status == "RESTORED":
                restored = int(rs_info.get("total_size", 0)) if rs_info else 0
                bytes_restored += restored
                elapsed = time.monotonic() - t0
                throughput = (bytes_restored / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                num_files = rs_info.get("num_files", "?") if rs_info else "?"
                print(f"  OK ({num_files} files, {format_size(restored)}) [{throughput:.1f} MB/s avg]")
                results["restored"] += 1
                partition_outcomes[partition_id] = "restored"
            else:
                err = truncate_error(rs_info.get("error", "unknown")) if rs_info else "unknown"
                print(f"  FAILED: {err}")
                results["failed"] += 1
                partition_outcomes[partition_id] = "failed"
        except ClickHouseError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            results["failed"] += 1
            partition_outcomes[partition_id] = "failed"

    total = sum(results.values())
    print(f"\n{'=' * 60}")
    print(f"Restore complete: {local_db}.{local_table}")
    print(f"  Restored:     {results['restored']:>4} / {total}")
    print(f"  Skipped:      {results['skipped']:>4} (already restored)")
    print(f"  Failed:       {results['failed']:>4}")

    result = {
        "database": local_db,
        "table": local_table,
        "source_table": f"{args.database}.{table}",
        **results,
        "partition_detail": [
            {
                "partition_id": bk["partition_id"],
                "backup_path": bk["path"],
                "outcome": partition_outcomes.get(bk["partition_id"], "unknown"),
            }
            for bk in backups
        ],
    }
    write_state_log(result, args.host, cluster, args.disk)
    print(f"  State log:    {STATE_LOG}")

    if results["failed"] > 0:
        sys.exit(1)


def run_multi_table(args, cluster: str | None, source_cluster: str | None) -> None:
    """Run restore for multiple tables concurrently with Rich progress display."""
    import threading

    tables = args.table
    max_concurrent = args.max_concurrent

    # --- Discovery phase ---
    print(f"Discovering backups for {len(tables)} table(s) from {args.source_host}...")
    table_info: list[dict] = []
    for table in tables:
        local_db, local_table, is_dist = resolve_table(
            args.source_host, args.source_port, args.database, table,
        )
        backups = discover_backups(args.source_host, args.source_port, local_db, local_table, source_cluster)
        table_info.append({
            "source_table": table,
            "local_db": local_db,
            "local_table": local_table,
            "partitions": len(backups),
        })
        suffix = f" -> {local_table}" if is_dist else ""
        print(f"  {args.database}.{table}{suffix}: {len(backups)} partitions")

    total_partitions = sum(t["partitions"] for t in table_info)
    print(f"\nTotal: {len(tables)} tables, {total_partitions} partitions")
    print(f"Max concurrent: {max_concurrent}\n")

    # --- Global throughput tracking (thread-safe) ---
    throughput_lock = threading.Lock()
    global_bytes_restored = 0
    t0 = time.monotonic()

    # --- Restore phase with Rich progress ---
    all_results: list[dict] = []
    has_failures = False

    progress = Progress(
        TextColumn("[bold]{task.fields[short_name]:<45}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status_text]}"),
        TimeElapsedColumn(),
    )

    with progress:
        task_ids: dict[str, TaskID] = {}
        for info in table_info:
            tid = progress.add_task(
                info["source_table"],
                total=info["partitions"] or 1,
                short_name=info["source_table"][:45],
                status_text="queued",
            )
            task_ids[info["source_table"]] = tid

        overall_tid = progress.add_task(
            "overall",
            total=total_partitions or 1,
            short_name="[cyan]OVERALL",
            status_text="0 B @ 0.0 MB/s",
        )

        def update_throughput():
            elapsed = time.monotonic() - t0
            mbps = (global_bytes_restored / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            progress.update(
                overall_tid,
                status_text=f"{format_size(global_bytes_restored)} @ {mbps:.1f} MB/s",
            )

        def do_one_table(info: dict) -> dict:
            nonlocal global_bytes_restored
            table_name = info["source_table"]
            tid = task_ids[table_name]
            progress.update(tid, status_text="starting...")

            def on_partition_done():
                progress.advance(tid)
                progress.advance(overall_tid)

            def on_bytes_restored(n: int):
                nonlocal global_bytes_restored
                with throughput_lock:
                    global_bytes_restored += n
                update_throughput()

            def log(msg: str):
                short = msg.strip()[:60]
                progress.update(tid, status_text=short)

            result = restore_table(
                host=args.host,
                port=args.port,
                source_host=args.source_host,
                source_port=args.source_port,
                source_database=args.database,
                source_table=table_name,
                disk=args.disk,
                cluster=cluster,
                source_cluster=source_cluster,
                user=args.user,
                password=args.password,
                force_partitions=set(args.force_partition) if args.force_partition else None,
                max_concurrent_partitions=args.max_concurrent_partitions,
                log=log,
                on_partition_done=on_partition_done,
                on_bytes_restored=on_bytes_restored,
            )

            c, s, f = result["restored"], result["skipped"], result["failed"]
            status = f"done: {c} new, {s} skip"
            if f > 0:
                status += f", [red]{f} fail[/red]"
            progress.update(tid, status_text=status, completed=info["partitions"] or 1)
            return result

        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {pool.submit(do_one_table, info): info for info in table_info}
            for future in as_completed(futures):
                info = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    write_state_log(result, args.host, cluster, args.disk)
                    if result["failed"] > 0:
                        has_failures = True
                except Exception as e:
                    progress.update(
                        task_ids[info["source_table"]],
                        status_text=f"[red]CRASHED: {e}[/red]",
                    )
                    has_failures = True

    # --- Final summary ---
    print(f"\n{'=' * 70}")
    print(f"{'Table':<50} {'New':>5} {'Skip':>5} {'Fail':>5}")
    print(f"{'-' * 70}")
    totals = {"restored": 0, "skipped": 0, "failed": 0}
    for r in sorted(all_results, key=lambda x: x["source_table"]):
        name = r["source_table"]
        if len(name) > 48:
            name = name[:47] + "…"
        fail_str = str(r["failed"])
        if r["failed"] > 0:
            fail_str = f"!{r['failed']}"
        print(f"  {name:<48} {r['restored']:>5} {r['skipped']:>5} {fail_str:>5}")
        totals["restored"] += r["restored"]
        totals["skipped"] += r["skipped"]
        totals["failed"] += r["failed"]
    print(f"{'-' * 70}")
    print(f"  {'TOTAL':<48} {totals['restored']:>5} {totals['skipped']:>5} {totals['failed']:>5}")
    print(f"\nState log: {STATE_LOG}")

    if has_failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_host_port(value: str, default_port: int = 8123) -> tuple[str, int]:
    """Parse host or host:port string."""
    if ":" in value:
        parts = value.rsplit(":", 1)
        return parts[0], int(parts[1])
    return value, default_port


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore ClickHouse tables from S3 backup disk, partition by partition",
    )
    parser.add_argument("--source-host", required=True, help="Source ClickHouse host (for backup discovery)")
    parser.add_argument("--source-port", type=int, default=8123, help="Source ClickHouse HTTP port")
    parser.add_argument("--host", default="localhost", help="Target ClickHouse HTTP host")
    parser.add_argument("--port", type=int, default=8123, help="Target ClickHouse HTTP port")
    parser.add_argument("--user", default=None, help="Target ClickHouse user")
    parser.add_argument("--password", default=None, help="Target ClickHouse password")
    parser.add_argument("--database", required=True, help="Database name")
    parser.add_argument("--table", required=True, nargs="+", help="Table name(s) — can specify multiple")
    parser.add_argument("--disk", default=DEFAULT_DISK, help=f"S3 disk name (default: {DEFAULT_DISK})")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER, help=f"Target cluster name (default: {DEFAULT_CLUSTER})")
    parser.add_argument("--no-cluster", action="store_true", help="Run restore locally (no ON CLUSTER)")
    parser.add_argument("--source-cluster", default=DEFAULT_CLUSTER, help=f"Source cluster name for backup discovery (default: {DEFAULT_CLUSTER})")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be restored")
    parser.add_argument("--status", action="store_true", help="Show restore status for each partition")
    parser.add_argument("--verify", action="store_true", help="Verify restores: compare row counts source vs target")
    parser.add_argument(
        "--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
        help=f"Max tables to restore concurrently (default: {DEFAULT_MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--max-concurrent-partitions", type=int, default=DEFAULT_MAX_CONCURRENT_PARTITIONS,
        help=(
            f"Max partitions to restore concurrently per table "
            f"(default: {DEFAULT_MAX_CONCURRENT_PARTITIONS}). Total in-flight "
            f"restores = --max-concurrent * --max-concurrent-partitions."
        ),
    )
    parser.add_argument(
        "--force-partition",
        nargs="+",
        metavar="PARTITION_ID",
        help=(
            "Force re-restore of specific partition IDs (e.g. 20260201). "
            "Filters to just these partitions and bypasses the RESTORED "
            "idempotency check (including stale in-memory rows in "
            "system.backups). IMPORTANT: ALTER TABLE ... DROP PARTITION ID "
            "on the target first, or the fresh restore will mix with "
            "existing rows."
        ),
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout per restore in seconds (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args()

    cluster = None if args.no_cluster else args.cluster
    source_cluster = args.source_cluster

    if args.status:
        show_status(
            args.host, args.port,
            args.source_host, args.source_port,
            args.database, args.table,
            cluster, source_cluster,
            args.user, args.password,
        )
        return

    if args.verify:
        ok = show_verify(
            args.host, args.port,
            args.source_host, args.source_port,
            args.database, args.table,
            cluster, source_cluster,
            args.user, args.password,
        )
        if not ok:
            sys.exit(1)
        return

    if args.dry_run:
        show_dry_run(
            args.source_host, args.source_port,
            args.database, args.table,
            args.disk, cluster, source_cluster,
        )
        return

    if len(args.table) == 1:
        run_single_table(args, cluster, source_cluster)
    else:
        run_multi_table(args, cluster, source_cluster)


if __name__ == "__main__":
    import signal

    def _handle_sigint(sig, frame):
        global _shutdown
        if _shutdown:
            print("\nForce quit.", file=sys.stderr)
            sys.exit(130)
        _shutdown = True
        print("\nCtrl+C received — finishing in-flight restore, then stopping...", file=sys.stderr)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
