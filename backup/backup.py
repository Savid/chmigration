#!/usr/bin/env python3
"""Backup ClickHouse tables to an S3 disk, partition by partition.

Connects to a ClickHouse cluster via HTTP, resolves Distributed tables to their
underlying local tables, discovers partitions, and backs up each partition to a
pre-configured S3 disk using native BACKUP TABLE commands.

The script is idempotent: it checks system.backups (across all cluster replicas
when --cluster is set) before issuing a backup and skips partitions that are
already completed or in progress.

Supports multiple tables with concurrent execution and Rich progress display.

Prerequisites:
  - An S3-backed disk (type: s3_plain) named (default: s3_backup) must be
    configured in ClickHouse with an allowed_disk entry in the backups section.
  - system.backup_log must NOT be removed (<backup_log remove="1"/> will break
    idempotency checks via system.backups).

Usage:
    # Single table
    python backup/backup.py --database default --table beacon_api_eth_v1_events_block

    # Multiple tables concurrently
    python backup/backup.py --database default \\
      --table beacon_api_eth_v1_events_block \\
             beacon_api_eth_v1_events_head \\
             canonical_beacon_block \\
      --max-concurrent 3

    # Other modes
    python backup/backup.py --database default --table beacon_api_eth_v1_events_block --dry-run
    python backup/backup.py --database default --table beacon_api_eth_v1_events_block --status
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

# Graceful shutdown flag — set by SIGINT handler so in-flight backups finish
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


def query(host: str, port: int, sql: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    params = urllib.parse.urlencode({
        "receive_timeout": timeout,
        "send_timeout": timeout,
    })
    url = f"http://{host}:{port}/?{params}"
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


def query_json_rows(host: str, port: int, sql: str, timeout: int = DEFAULT_TIMEOUT) -> list[dict]:
    result = query(host, port, f"{sql} FORMAT JSONEachRow", timeout)
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
# Partition discovery & date extraction
# ---------------------------------------------------------------------------

def extract_partition_date(partition: str) -> datetime | None:
    """Try to extract a date from a partition value string.

    Handles multiple formats found in ClickHouse partition keys:
      - YYYY-MM-DD  (e.g. '2024-01-15' or ('mainnet', '2024-01-15'))
      - YYYYMMDD    (e.g. 20240115)
      - YYYYMM      (e.g. 202401 or ('mainnet', 202401))
      - Simple integers with no date component (e.g. 1, 42) → returns None
    """
    # YYYY-MM-DD (with or without surrounding tuple)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", partition)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # YYYYMMDD — exactly 8 consecutive digits, not part of a longer number
    m = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", partition)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # YYYYMM — exactly 6 consecutive digits, not part of a longer number
    m = re.search(r"(?<!\d)(\d{4})(\d{2})(?!\d)", partition)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 2000 <= year <= 2100 and 1 <= month <= 12:
            return datetime(year, month, 1)

    return None


def format_partition_expr(partition: str) -> str:
    """Format a partition value from system.parts for the PARTITIONS clause.

    Tuple partitions like ``('mainnet', 202401)`` are used as-is.
    Simple partitions like ``202401`` are quoted as ``'202401'``.
    """
    stripped = partition.strip()
    if stripped.startswith("("):
        return stripped
    return f"'{stripped}'"


def get_partitions(
    host: str,
    port: int,
    database: str,
    table: str,
    before: datetime | None = None,
    cluster: str | None = None,
) -> list[dict]:
    """Return distinct partitions for a table from system.parts.

    Each row has keys: partition, partition_id, part_count, rows, bytes.
    When *cluster* is set, queries across all replicas via clusterAllReplicas()
    to discover partitions on every shard.
    When *before* is set, only partitions whose extracted date is strictly
    before the cutoff are returned. Partitions with no recognisable date
    component are skipped with a warning.
    """
    source = (
        f"clusterAllReplicas('{cluster}', system.parts)"
        if cluster
        else "system.parts"
    )
    sql = (
        "SELECT "
        "  partition, "
        "  partition_id, "
        "  sum(part_count) AS part_count, "
        "  sum(rows) AS rows, "
        "  sum(bytes) AS bytes "
        "FROM ("
        "  SELECT "
        "    partition, "
        "    partition_id, "
        "    count() AS part_count, "
        "    sum(rows) AS rows, "
        "    sum(bytes_on_disk) AS bytes "
        f"  FROM {source} "
        f"  WHERE database = '{database}' AND table = '{table}' AND active = 1 "
        "  GROUP BY partition, partition_id, _shard_num"
        ") "
        "GROUP BY partition, partition_id "
        "ORDER BY partition"
    )
    rows = query_json_rows(host, port, sql)

    if before is None:
        return rows

    filtered: list[dict] = []
    for row in rows:
        partition = str(row["partition"])
        dt = extract_partition_date(partition)
        if dt is None:
            continue
        if dt < before:
            filtered.append(row)

    return filtered


# ---------------------------------------------------------------------------
# Backup path & status
# ---------------------------------------------------------------------------

def make_backup_path(database: str, table: str, partition_id: str) -> str:
    """Deterministic S3 path for a single partition backup."""
    return f"{database}/{table}/{partition_id}"


def check_backup_status(
    host: str,
    port: int,
    backup_path: str,
    cluster: str | None = None,
) -> tuple[str | None, dict | None]:
    """Check system.backups for an existing backup whose name contains *backup_path*.

    When *cluster* is set, queries across all replicas via clusterAllReplicas().

    Priority order (matters when multiple rows exist for the same path):
      1. CREATING_BACKUP — if any replica is actively backing up, we must wait.
         Otherwise a stale BACKUP_FAILED row from an earlier collision could
         look "newer" than an in-flight attempt and trigger another retry,
         which would hit BACKUP_ALREADY_EXISTS.
      2. BACKUP_CREATED (most recent) — done.
      3. BACKUP_FAILED (most recent) — caller may retry.

    Returns (status, row) or (None, None) when no matching backup exists.
    """
    source = (
        f"clusterAllReplicas('{cluster}', system.backups)"
        if cluster
        else "system.backups"
    )
    safe_path = backup_path.replace("'", "\\'")
    common_select = (
        "SELECT id, name, status, error, "
        "  start_time, end_time, num_files, "
        "  total_size, compressed_size "
        f"FROM {source} "
        f"WHERE name LIKE '%{safe_path}%' "
    )

    # 1. Any active backup blocks re-issue.
    sql = common_select + "AND status = 'CREATING_BACKUP' ORDER BY start_time DESC LIMIT 1"
    try:
        rows = query_json_rows(host, port, sql, timeout=30)
    except ClickHouseError:
        return None, None
    if rows:
        return rows[0].get("status"), rows[0]

    # 2. Terminal status — prefer CREATED over FAILED, then most recent.
    sql = (
        common_select
        + "AND status IN ('BACKUP_CREATED', 'BACKUP_FAILED') "
        + "ORDER BY status = 'BACKUP_CREATED' DESC, start_time DESC LIMIT 1"
    )
    try:
        rows = query_json_rows(host, port, sql, timeout=30)
    except ClickHouseError:
        return None, None
    if not rows:
        return None, None
    return rows[0].get("status"), rows[0]


def format_size(b: int) -> str:
    if b >= 1024 ** 3:
        return f"{b / 1024**3:.1f} GB"
    if b >= 1024 ** 2:
        return f"{b / 1024**2:.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


# ---------------------------------------------------------------------------
# Per-table backup logic
# ---------------------------------------------------------------------------

POLL_INTERVAL = 10  # seconds between async backup status checks


def do_backup_partition(
    host: str,
    port: int,
    database: str,
    table: str,
    partition: str,
    disk: str,
    cluster: str | None,
    backup_path: str,
) -> tuple[str, dict | None]:
    """Issue an ASYNC BACKUP and poll until completion.

    Returns (status, info_dict) where status is 'BACKUP_CREATED' or
    'BACKUP_FAILED'.
    """
    cluster_clause = f" ON CLUSTER '{cluster}'" if cluster else ""
    partition_expr = format_partition_expr(partition)

    sql = (
        f"BACKUP TABLE `{database}`.`{table}`"
        f" PARTITIONS {partition_expr}"
        f"{cluster_clause}"
        f" TO Disk('{disk}', '{backup_path}')"
        f" ASYNC"
    )
    query(host, port, sql, timeout=60)

    # Poll until done.
    while not _shutdown:
        time.sleep(POLL_INTERVAL)
        status, info = check_backup_status(host, port, backup_path, cluster)
        if status is None:
            continue
        if status != "CREATING_BACKUP":
            return status, info

    return "INTERRUPTED", None


def backup_table(
    host: str,
    port: int,
    source_database: str,
    source_table: str,
    cutoff: datetime | None,
    disk: str,
    cluster: str | None,
    force_partitions: set[str] | None = None,
    max_concurrent_partitions: int = 1,
    log: Callable[[str], None] = print,
    on_partition_done: Callable[[], None] | None = None,
    on_bytes_created: Callable[[int], None] | None = None,
) -> dict:
    """Back up all eligible partitions of a single table.

    Partitions run concurrently up to *max_concurrent_partitions*; setting it
    to 1 preserves the original sequential behavior.

    When *force_partitions* is non-empty, the run is restricted to those
    partition IDs, the date cutoff is ignored, and the idempotency check is
    bypassed so the backup is re-issued even if a BACKUP_CREATED row exists
    in system.backups. Callers must delete the S3 backup files first or
    ClickHouse will error "backup already exists".

    Returns a results dict with keys: database, table, source_table, created,
    skipped, in_progress, failed, partition_detail.
    """
    local_db, local_table, is_distributed = resolve_table(host, port, source_database, source_table)
    if is_distributed:
        log(f"  {source_database}.{source_table} -> {local_db}.{local_table}")

    effective_cutoff = None if force_partitions else cutoff
    partitions = get_partitions(host, port, local_db, local_table, effective_cutoff, cluster)
    if force_partitions:
        partitions = [p for p in partitions if p["partition_id"] in force_partitions]
        found = {p["partition_id"] for p in partitions}
        missing = sorted(force_partitions - found)
        if missing:
            log(f"  WARNING: force-partition not found in source system.parts: {missing}")

    results = {"created": 0, "skipped": 0, "in_progress": 0, "failed": 0}
    partition_outcomes: dict[str, str] = {}
    results_lock = threading.Lock()

    if not partitions:
        return {
            "database": local_db,
            "table": local_table,
            "source_table": f"{source_database}.{source_table}",
            **results,
            "partition_detail": [],
        }

    def handle_partition(p: dict) -> None:
        if _shutdown:
            return

        partition = p["partition"]
        partition_id = p["partition_id"]
        path = make_backup_path(local_db, local_table, partition_id)

        force = bool(force_partitions) and partition_id in force_partitions
        status, info = check_backup_status(host, port, path, cluster)

        if status == "BACKUP_CREATED":
            if force:
                # Drop through to issue a new backup; existing BACKUP_CREATED
                # row ignored. Caller must have cleared the S3 destination.
                log(f"  {partition} force re-backup (ignoring previous BACKUP_CREATED)")
                status, info = None, None
            else:
                log(f"  {partition} already backed up — skipping")
                with results_lock:
                    results["skipped"] += 1
                    partition_outcomes[partition_id] = "skipped"
                if on_partition_done:
                    on_partition_done()
                return

        if status == "CREATING_BACKUP":
            # Always wait for in-flight backups, even under --force: issuing a
            # second BACKUP to the same destination would hit BACKUP_ALREADY_EXISTS.
            log(f"  {partition} waiting for in-progress backup...")
            while not _shutdown:
                time.sleep(POLL_INTERVAL)
                status, info = check_backup_status(host, port, path, cluster)
                if status != "CREATING_BACKUP":
                    break
            if _shutdown:
                return
            if status == "BACKUP_CREATED":
                backed_bytes = int(info.get("total_size", 0)) if info else 0
                log(f"  {partition} completed -> {format_size(backed_bytes)}")
                with results_lock:
                    results["created"] += 1
                    partition_outcomes[partition_id] = "created"
                if on_bytes_created:
                    on_bytes_created(backed_bytes)
            else:
                err = info.get("error", "unknown") if info else "unknown"
                log(f"  {partition} FAILED after wait: {err}")
                with results_lock:
                    results["failed"] += 1
                    partition_outcomes[partition_id] = "failed"
            if on_partition_done:
                on_partition_done()
            return

        if status == "BACKUP_FAILED":
            log(f"  {partition} retrying after previous failure")

        log(f"  {partition} issuing BACKUP...")
        try:
            bk_status, bk_info = do_backup_partition(
                host, port, local_db, local_table,
                partition, disk, cluster, path,
            )
            if bk_status == "BACKUP_CREATED":
                backed_bytes = int(bk_info.get("total_size", 0)) if bk_info else 0
                log(f"  {partition} backed up -> {format_size(backed_bytes)}")
                with results_lock:
                    results["created"] += 1
                    partition_outcomes[partition_id] = "created"
                if on_bytes_created:
                    on_bytes_created(backed_bytes)
            elif bk_status == "INTERRUPTED":
                return
            else:
                err = bk_info.get("error", "unknown") if bk_info else "unknown"
                log(f"  {partition} FAILED: {err}")
                with results_lock:
                    results["failed"] += 1
                    partition_outcomes[partition_id] = "failed"
        except ClickHouseError as e:
            msg = str(e).lower()
            if "already exists" in msg or "backup_already_exists" in msg:
                log(f"  {partition} destination occupied in S3 — skipping")
                with results_lock:
                    results["skipped"] += 1
                    partition_outcomes[partition_id] = "skipped"
            else:
                log(f"  {partition} ERROR: {e}")
                with results_lock:
                    results["failed"] += 1
                    partition_outcomes[partition_id] = "failed"

        if on_partition_done:
            on_partition_done()

    workers = max(1, max_concurrent_partitions)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(handle_partition, p) for p in partitions]
        for f in as_completed(futures):
            f.result()

    return {
        "database": local_db,
        "table": local_table,
        "source_table": f"{source_database}.{source_table}",
        **results,
        "partition_detail": [
            {
                "partition": p["partition"],
                "partition_id": p["partition_id"],
                "rows": int(p.get("rows", 0)),
                "bytes": int(p.get("bytes", 0)),
                "outcome": partition_outcomes.get(p["partition_id"], "unknown"),
            }
            for p in partitions
        ],
    }


def write_state_log(result: dict, cutoff: datetime | None, host: str, cluster: str | None, disk: str) -> None:
    """Append a JSONL entry to the state log."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **result,
        "cutoff": cutoff.strftime("%Y-%m-%d") if cutoff else None,
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

VERIFY_DIFF_THRESHOLD = 0.01  # 1% — warn if backup size differs by more than this


def verify_tables(
    host: str,
    port: int,
    database: str,
    tables: list[str],
    cutoff: datetime | None,
    cluster: str | None,
    disk: str,
) -> bool:
    """Compare system.backups metadata against system.parts for each partition.

    Returns True if all partitions are verified OK.
    """
    all_ok = True

    for table in tables:
        local_db, local_table, is_dist = resolve_table(host, port, database, table)
        partitions = get_partitions(host, port, local_db, local_table, cutoff, cluster)
        total_rows = sum(int(p.get("rows", 0)) for p in partitions)
        suffix = f" -> {local_table}" if is_dist else ""
        print(f"\n{database}.{table}{suffix} ({len(partitions)} partitions, {total_rows:,} rows)")

        counts = {"ok": 0, "mismatch": 0, "missing": 0, "failed": 0, "in_progress": 0}

        for p in partitions:
            partition = p["partition"]
            partition_id = p["partition_id"]
            cluster_bytes = int(p.get("bytes", 0))
            path = make_backup_path(local_db, local_table, partition_id)
            status, info = check_backup_status(host, port, path, cluster)

            if status == "BACKUP_CREATED":
                backup_bytes = int(info.get("total_size", 0))
                if cluster_bytes == 0:
                    pct = 0.0
                else:
                    pct = (backup_bytes - cluster_bytes) / cluster_bytes
                pct_str = f"{pct:+.1%}"
                if abs(pct) > VERIFY_DIFF_THRESHOLD:
                    print(f"  {partition:>15}   WARN   {format_size(backup_bytes):>10} backup   {format_size(cluster_bytes):>10} cluster   ({pct_str})")
                    counts["mismatch"] += 1
                else:
                    print(f"  {partition:>15}   OK     {format_size(backup_bytes):>10} backup   {format_size(cluster_bytes):>10} cluster   ({pct_str})")
                    counts["ok"] += 1
            elif status == "CREATING_BACKUP":
                started = info.get("start_time", "?") if info else "?"
                print(f"  {partition:>15}   PROG   in progress (since {started})")
                counts["in_progress"] += 1
            elif status == "BACKUP_FAILED":
                err = info.get("error", "?") if info else "?"
                print(f"  {partition:>15}   FAIL   {err}")
                counts["failed"] += 1
            else:
                print(f"  {partition:>15}   MISS   no backup found")
                counts["missing"] += 1

        # Per-table summary.
        parts = []
        if counts["ok"]:
            parts.append(f"{counts['ok']} ok")
        if counts["mismatch"]:
            parts.append(f"{counts['mismatch']} size mismatch")
        if counts["missing"]:
            parts.append(f"{counts['missing']} missing")
        if counts["failed"]:
            parts.append(f"{counts['failed']} failed")
        if counts["in_progress"]:
            parts.append(f"{counts['in_progress']} in progress")
        print(f"  Summary: {', '.join(parts)}")

        if counts["missing"] or counts["failed"] or counts["mismatch"]:
            all_ok = False

    return all_ok


def show_status(host: str, port: int, database: str, tables: list[str], cutoff: datetime | None, cluster: str | None) -> None:
    """Print backup status for each table's partitions."""
    for table in tables:
        local_db, local_table, _ = resolve_table(host, port, database, table)
        partitions = get_partitions(host, port, local_db, local_table, cutoff, cluster)
        total_rows = sum(int(p.get("rows", 0)) for p in partitions)
        print(f"\n{database}.{table} ({len(partitions)} partitions, {total_rows:,} rows)")

        for p in partitions:
            path = make_backup_path(local_db, local_table, p["partition_id"])
            status, info = check_backup_status(host, port, path, cluster)

            if status == "BACKUP_CREATED":
                size_mb = int(info.get("total_size", 0)) / (1024 * 1024)
                print(f"  {p['partition']:>30}  DONE  {info.get('num_files', '?')} files  {size_mb:.1f} MB")
            elif status == "CREATING_BACKUP":
                print(f"  {p['partition']:>30}  IN PROGRESS  (since {info.get('start_time', '?')})")
            elif status == "BACKUP_FAILED":
                print(f"  {p['partition']:>30}  FAILED  {info.get('error', '?')}")
            else:
                print(f"  {p['partition']:>30}  NOT BACKED UP")


def show_dry_run(host: str, port: int, database: str, tables: list[str], cutoff: datetime | None, cluster: str | None) -> None:
    """Preview what would be backed up."""
    for table in tables:
        local_db, local_table, is_dist = resolve_table(host, port, database, table)
        partitions = get_partitions(host, port, local_db, local_table, cutoff, cluster)
        total_rows = sum(int(p.get("rows", 0)) for p in partitions)
        total_bytes = sum(int(p.get("bytes", 0)) for p in partitions)
        suffix = f" -> {local_table}" if is_dist else ""
        print(f"\n{database}.{table}{suffix} ({len(partitions)} partitions, {total_rows:,} rows, {format_size(total_bytes)})")

        for p in partitions:
            rows = int(p.get("rows", 0))
            size = format_size(int(p.get("bytes", 0)))
            parts = p.get("part_count", "?")
            print(f"  {p['partition']:>30}  {rows:>12,} rows  {size:>10}  ({parts} parts)")

    print(f"\n[DRY RUN] No backups created")


def run_single_table(args, cluster: str | None, cutoff: datetime | None) -> None:
    """Run backup for a single table with verbose print output."""
    table = args.table[0]

    local_db, local_table, is_distributed = resolve_table(args.host, args.port, args.database, table)
    if is_distributed:
        print(f"Distributed table {args.database}.{table} -> local {local_db}.{local_table}")
    else:
        print(f"Local table: {local_db}.{local_table}")

    force_partitions: set[str] = set(args.force_partition) if args.force_partition else set()
    effective_cutoff = None if force_partitions else cutoff
    partitions = get_partitions(args.host, args.port, local_db, local_table, effective_cutoff, cluster)
    if force_partitions:
        partitions = [p for p in partitions if p["partition_id"] in force_partitions]
        found = {p["partition_id"] for p in partitions}
        missing = sorted(force_partitions - found)
        if missing:
            print(f"WARNING: force-partition not found in source system.parts: {missing}")
    if not partitions:
        print("No partitions found matching criteria")
        return

    total_rows = sum(int(p.get("rows", 0)) for p in partitions)
    total_bytes = sum(int(p.get("bytes", 0)) for p in partitions)
    print(f"Found {len(partitions)} partition(s): {total_rows:,} rows, {format_size(total_bytes)}")
    if force_partitions:
        print(f"Force re-backup mode: {sorted(force_partitions)} (idempotency check bypassed)")

    # Concurrent path: delegate to backup_table (no per-partition polling dots,
    # but partitions run in parallel up to --max-concurrent-partitions).
    if args.max_concurrent_partitions > 1:
        print(f"Backing up up to {args.max_concurrent_partitions} partitions concurrently\n")
        result = backup_table(
            host=args.host,
            port=args.port,
            source_database=args.database,
            source_table=table,
            cutoff=cutoff,
            disk=args.disk,
            cluster=cluster,
            force_partitions=force_partitions or None,
            max_concurrent_partitions=args.max_concurrent_partitions,
        )
        total = result["created"] + result["skipped"] + result["in_progress"] + result["failed"]
        print(f"\n{'=' * 60}")
        print(f"Backup complete: {local_db}.{local_table}")
        print(f"  Created:      {result['created']:>4} / {total}")
        print(f"  Skipped:      {result['skipped']:>4} (already backed up)")
        print(f"  In progress:  {result['in_progress']:>4}")
        print(f"  Failed:       {result['failed']:>4}")
        write_state_log(result, cutoff, args.host, cluster, args.disk)
        print(f"  State log:    {STATE_LOG}")
        if result["failed"] > 0:
            sys.exit(1)
        return

    results = {"created": 0, "skipped": 0, "in_progress": 0, "failed": 0}
    partition_outcomes: dict[str, str] = {}
    bytes_created = 0
    t0 = time.monotonic()

    for i, p in enumerate(partitions, 1):
        if _shutdown:
            print("\nInterrupted — stopping after current partition...")
            break

        partition = p["partition"]
        partition_id = p["partition_id"]
        path = make_backup_path(local_db, local_table, partition_id)
        rows = int(p.get("rows", 0))
        size = format_size(int(p.get("bytes", 0)))

        print(f"\n[{i}/{len(partitions)}] {partition} ({rows:,} rows, {size})")

        force = partition_id in force_partitions
        status, info = check_backup_status(args.host, args.port, path, cluster)

        if status == "BACKUP_CREATED":
            if force:
                print("  Force re-backup — ignoring previous BACKUP_CREATED row")
                status, info = None, None
            else:
                bk_size = format_size(int(info.get("total_size", 0)))
                print(f"  Already backed up ({info.get('num_files', '?')} files, {bk_size})")
                results["skipped"] += 1
                partition_outcomes[partition_id] = "skipped"
                continue

        if status == "CREATING_BACKUP":
            started = info.get("start_time", "?") if info else "?"
            print(f"  Waiting for in-progress backup (since {started})...", end="", flush=True)
            while not _shutdown:
                time.sleep(POLL_INTERVAL)
                status, info = check_backup_status(args.host, args.port, path, cluster)
                if status != "CREATING_BACKUP":
                    break
                print(".", end="", flush=True)
            print()
            if _shutdown:
                print("\nInterrupted — backup will continue on the cluster")
                break
            if status == "BACKUP_CREATED":
                backed = int(info.get("total_size", 0)) if info else 0
                bytes_created += backed
                elapsed = time.monotonic() - t0
                throughput = (bytes_created / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                num_files = info.get("num_files", "?") if info else "?"
                print(f"  Completed ({num_files} files, {format_size(backed)}) [{throughput:.1f} MB/s avg]")
                results["created"] += 1
                partition_outcomes[partition_id] = "created"
            else:
                err = info.get("error", "unknown") if info else "unknown"
                print(f"  FAILED after wait: {err}")
                results["failed"] += 1
                partition_outcomes[partition_id] = "failed"
            continue

        if status == "BACKUP_FAILED":
            print(f"  Previous backup failed: {info.get('error', '?') if info else '?'}")
            print(f"  Retrying...")

        try:
            cluster_clause = f" ON CLUSTER '{cluster}'" if cluster else ""
            partition_expr = format_partition_expr(partition)
            sql = (
                f"BACKUP TABLE `{local_db}`.`{local_table}`"
                f" PARTITIONS {partition_expr}"
                f"{cluster_clause}"
                f" TO Disk('{args.disk}', '{path}')"
                f" ASYNC"
                    )
            print(f"  SQL: {sql}")
            query(args.host, args.port, sql, timeout=60)

            # Poll until done.
            bk_status, bk_info = None, None
            print(f"  Polling...", end="", flush=True)
            while not _shutdown:
                time.sleep(POLL_INTERVAL)
                bk_status, bk_info = check_backup_status(args.host, args.port, path, cluster)
                if bk_status and bk_status != "CREATING_BACKUP":
                    break
                print(".", end="", flush=True)
            print()

            if _shutdown:
                print("\nInterrupted — backup will continue on the cluster")
                break

            if bk_status == "BACKUP_CREATED":
                backed = int(bk_info.get("total_size", 0)) if bk_info else 0
                bytes_created += backed
                elapsed = time.monotonic() - t0
                throughput = (bytes_created / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                num_files = bk_info.get("num_files", "?") if bk_info else "?"
                print(f"  OK ({num_files} files, {format_size(backed)}) [{throughput:.1f} MB/s avg]")
                results["created"] += 1
                partition_outcomes[partition_id] = "created"
            else:
                err = bk_info.get("error", "unknown") if bk_info else "unknown"
                print(f"  FAILED: {err}")
                results["failed"] += 1
                partition_outcomes[partition_id] = "failed"
        except ClickHouseError as e:
            msg = str(e).lower()
            if "already exists" in msg or "backup_already_exists" in msg:
                print(f"  Already exists in S3 (destination occupied)")
                results["skipped"] += 1
                partition_outcomes[partition_id] = "skipped"
            else:
                print(f"  ERROR: {e}", file=sys.stderr)
                results["failed"] += 1
                partition_outcomes[partition_id] = "failed"

    total = sum(results.values())
    print(f"\n{'=' * 60}")
    print(f"Backup complete: {local_db}.{local_table}")
    print(f"  Created:      {results['created']:>4} / {total}")
    print(f"  Skipped:      {results['skipped']:>4} (already backed up)")
    print(f"  In progress:  {results['in_progress']:>4}")
    print(f"  Failed:       {results['failed']:>4}")

    result = {
        "database": local_db,
        "table": local_table,
        "source_table": f"{args.database}.{table}",
        **results,
        "partition_detail": [
            {
                "partition": p["partition"],
                "partition_id": p["partition_id"],
                "rows": int(p.get("rows", 0)),
                "bytes": int(p.get("bytes", 0)),
                "outcome": partition_outcomes.get(p["partition_id"], "unknown"),
            }
            for p in partitions
        ],
    }
    write_state_log(result, cutoff, args.host, cluster, args.disk)
    print(f"  State log:    {STATE_LOG}")

    if results["failed"] > 0:
        sys.exit(1)


def run_multi_table(args, cluster: str | None, cutoff: datetime | None) -> None:
    """Run backup for multiple tables concurrently with Rich progress display."""
    import threading

    tables = args.table
    max_concurrent = args.max_concurrent

    # --- Discovery phase: resolve all tables and count partitions ---
    print(f"Discovering partitions for {len(tables)} table(s)...")
    table_info: list[dict] = []
    for table in tables:
        local_db, local_table, is_dist = resolve_table(args.host, args.port, args.database, table)
        partitions = get_partitions(args.host, args.port, local_db, local_table, cutoff, cluster)
        total_bytes = sum(int(p.get("bytes", 0)) for p in partitions)
        table_info.append({
            "source_table": table,
            "local_db": local_db,
            "local_table": local_table,
            "partitions": len(partitions),
            "total_bytes": total_bytes,
        })
        suffix = f" -> {local_table}" if is_dist else ""
        print(f"  {args.database}.{table}{suffix}: {len(partitions)} partitions, {format_size(total_bytes)}")

    total_partitions = sum(t["partitions"] for t in table_info)
    total_bytes = sum(t["total_bytes"] for t in table_info)
    print(f"\nTotal: {len(tables)} tables, {total_partitions} partitions, {format_size(total_bytes)}")
    print(f"Max concurrent: {max_concurrent}\n")

    # --- Global throughput tracking (thread-safe) ---
    throughput_lock = threading.Lock()
    global_bytes_created = 0
    t0 = time.monotonic()

    # --- Backup phase with Rich progress ---
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
        # Create a progress task per table + an overall throughput row.
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
            mbps = (global_bytes_created / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            progress.update(
                overall_tid,
                status_text=f"{format_size(global_bytes_created)} @ {mbps:.1f} MB/s",
            )

        def do_one_table(info: dict) -> dict:
            nonlocal global_bytes_created
            table_name = info["source_table"]
            tid = task_ids[table_name]
            progress.update(tid, status_text="starting...")

            def on_partition_done():
                progress.advance(tid)
                progress.advance(overall_tid)

            def on_bytes_created(n: int):
                nonlocal global_bytes_created
                with throughput_lock:
                    global_bytes_created += n
                update_throughput()

            def log(msg: str):
                short = msg.strip()[:60]
                progress.update(tid, status_text=short)

            result = backup_table(
                host=args.host,
                port=args.port,
                source_database=args.database,
                source_table=table_name,
                cutoff=cutoff,
                disk=args.disk,
                cluster=cluster,
                force_partitions=set(args.force_partition) if args.force_partition else None,
                max_concurrent_partitions=args.max_concurrent_partitions,
                log=log,
                on_partition_done=on_partition_done,
                on_bytes_created=on_bytes_created,
            )

            c, s, f = result["created"], result["skipped"], result["failed"]
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
                    write_state_log(result, cutoff, args.host, cluster, args.disk)
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
    totals = {"created": 0, "skipped": 0, "failed": 0}
    for r in sorted(all_results, key=lambda x: x["source_table"]):
        name = r["source_table"]
        if len(name) > 48:
            name = name[:47] + "…"
        fail_str = str(r["failed"])
        if r["failed"] > 0:
            fail_str = f"!{r['failed']}"
        print(f"  {name:<48} {r['created']:>5} {r['skipped']:>5} {fail_str:>5}")
        totals["created"] += r["created"]
        totals["skipped"] += r["skipped"]
        totals["failed"] += r["failed"]
    print(f"{'-' * 70}")
    print(f"  {'TOTAL':<48} {totals['created']:>5} {totals['skipped']:>5} {totals['failed']:>5}")
    print(f"\nState log: {STATE_LOG}")

    if has_failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_cutoff(before: str | None) -> datetime:
    """Return date cutoff. Partitions with date < cutoff are eligible."""
    if before:
        return datetime.strptime(before, "%Y-%m-%d")

    now = datetime.now()
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backup ClickHouse tables to S3 by partition",
    )
    parser.add_argument("--host", default="localhost", help="ClickHouse HTTP host")
    parser.add_argument("--port", type=int, default=8123, help="ClickHouse HTTP port")
    parser.add_argument("--database", required=True, help="Database name")
    parser.add_argument("--table", required=True, nargs="+", help="Table name(s) — can specify multiple")
    parser.add_argument(
        "--before",
        help="Backup partitions strictly before YYYY-MM-DD (default: 1st of current month)",
    )
    parser.add_argument("--disk", default=DEFAULT_DISK, help=f"S3 disk name (default: {DEFAULT_DISK})")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER, help=f"Cluster name (default: {DEFAULT_CLUSTER})")
    parser.add_argument("--no-cluster", action="store_true", help="Run backup locally (no ON CLUSTER)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be backed up")
    parser.add_argument("--status", action="store_true", help="Show backup status for each partition")
    parser.add_argument("--verify", action="store_true", help="Verify backups: compare system.backups sizes vs system.parts")
    parser.add_argument("--all-partitions", action="store_true", help="Include all partitions (ignore date filter)")
    parser.add_argument(
        "--force-partition",
        nargs="+",
        metavar="PARTITION_ID",
        help=(
            "Force re-backup of specific partition IDs (e.g. 20260201). "
            "Filters to just these partitions, ignores --before/--all-partitions, "
            "and bypasses the system.backups idempotency check. "
            "IMPORTANT: delete the existing S3 backup objects first, or "
            "ClickHouse will error 'backup already exists'."
        ),
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
        help=f"Max tables to back up concurrently (default: {DEFAULT_MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--max-concurrent-partitions", type=int, default=DEFAULT_MAX_CONCURRENT_PARTITIONS,
        help=(
            f"Max partitions to back up concurrently per table "
            f"(default: {DEFAULT_MAX_CONCURRENT_PARTITIONS}). Total in-flight "
            f"backups = --max-concurrent * --max-concurrent-partitions."
        ),
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout per backup in seconds (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args()

    cluster = None if args.no_cluster else args.cluster

    cutoff: datetime | None = None
    if not args.all_partitions:
        cutoff = compute_cutoff(args.before)
        print(f"Partition cutoff: before {cutoff.strftime('%Y-%m-%d')}")

    if args.status:
        show_status(args.host, args.port, args.database, args.table, cutoff, cluster)
        return

    if args.verify:
        ok = verify_tables(args.host, args.port, args.database, args.table, cutoff, cluster, args.disk)
        if not ok:
            sys.exit(1)
        return

    if args.dry_run:
        show_dry_run(args.host, args.port, args.database, args.table, cutoff, cluster)
        return

    if len(args.table) == 1:
        run_single_table(args, cluster, cutoff)
    else:
        run_multi_table(args, cluster, cutoff)


if __name__ == "__main__":
    import signal

    def _handle_sigint(sig, frame):
        global _shutdown
        if _shutdown:
            # Second Ctrl+C — force exit.
            print("\nForce quit.", file=sys.stderr)
            sys.exit(130)
        _shutdown = True
        print("\nCtrl+C received — finishing in-flight backup, then stopping...", file=sys.stderr)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
