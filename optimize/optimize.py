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

from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TaskID


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
) -> None:
    """Fire OPTIMIZE TABLE ... PARTITION ... FINAL.

    When ON CLUSTER is used, sets distributed_ddl_task_timeout=0 so the
    query returns immediately instead of blocking until every replica
    finishes (which can hit error 159 on large clusters).
    """
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
) -> int:
    """Poll system.parts until the partition's part count drops (merge landed).

    Returns the final part count.  Raises TimeoutError if timeout is exceeded.
    """
    deadline = time.monotonic() + timeout
    while not _shutdown:
        time.sleep(POLL_INTERVAL)
        current = get_partition_part_count(
            host, port, database, table, partition_id, cluster, user, password,
        )
        if current < initial_parts:
            return current
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Partition {partition_id} still has {current} parts after {timeout}s"
            )
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
) -> dict:
    """Optimize all partitions of a single table.

    Returns results dict with per-partition detail.
    """
    local_db, local_table, is_distributed = resolve_table(
        host, port, database, table, user, password,
    )
    if is_distributed:
        log(f"  {database}.{table} -> {local_db}.{local_table}")

    all_parts = get_partitions(host, port, local_db, local_table, cluster, user, password)

    results = {"optimized": 0, "skipped": 0, "failed": 0}
    details: list[dict] = []

    if not all_parts:
        return {
            "database": local_db,
            "table": local_table,
            **results,
            "partition_detail": [],
        }

    for part_info in all_parts:
        if _shutdown:
            break

        partition_id = part_info["partition_id"]
        part_count = int(part_info.get("part_count", 0))
        total_rows = int(part_info.get("total_rows", 0))
        bytes_on_disk = int(part_info.get("bytes_on_disk", 0))

        if part_count <= 1:
            log(f"  {partition_id}  SKIP   parts=1  rows={total_rows:>12,}  size={format_size(bytes_on_disk)}")
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
            continue

        for attempt in range(1, retries + 1):
            try:
                log(f"  {partition_id}  OPT    parts={part_count}  rows={total_rows:>12,}  size={format_size(bytes_on_disk)}")
                submit_optimize(
                    host, port, local_db, local_table,
                    partition_id, cluster, user, password,
                )
                # Poll until merge lands.
                new_count = poll_optimize_done(
                    host, port, local_db, local_table,
                    partition_id, part_count, cluster, user, password, timeout,
                )
                log(f"  {partition_id}  DONE   parts={part_count} -> {new_count}")
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

def run_single_table(args, cluster: str | None) -> None:
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
    print(f"Found {len(all_parts)} partition(s), {len(unmerged)} need optimization\n")

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


def run_multi_table(args, cluster: str | None) -> None:
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

    total_partitions = sum(t["total_partitions"] for t in table_info)
    total_unmerged = sum(t["unmerged"] for t in table_info)
    print(f"\nTotal: {len(tables)} tables, {total_partitions} partitions, {total_unmerged} need optimization")
    print(f"Max concurrent: {max_concurrent}\n")

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
                total=info["total_partitions"] or 1,
                short_name=info["source_table"][:45],
                status_text="queued",
            )
            task_ids[info["source_table"]] = tid

        overall_tid = progress.add_task(
            "overall",
            total=total_partitions or 1,
            short_name="[cyan]OVERALL",
            status_text="0 optimized",
        )

        optimized_count = 0
        count_lock = threading.Lock()

        def do_one_table(info: dict) -> dict:
            nonlocal optimized_count
            table_name = info["source_table"]
            tid = task_ids[table_name]
            progress.update(tid, status_text="optimizing...")

            def on_partition_done():
                nonlocal optimized_count
                progress.advance(tid)
                progress.advance(overall_tid)
                with count_lock:
                    optimized_count += 1
                progress.update(overall_tid, status_text=f"{optimized_count} done")

            def log(msg: str):
                short = msg.strip()[:60]
                progress.update(tid, status_text=short)

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
            )

            o = result["optimized"]
            s = result["skipped"]
            f = result["failed"]
            status = f"done: {o} opt, {s} skip"
            if f > 0:
                status += f", [red]{f} fail[/red]"
            progress.update(tid, status_text=status, completed=info["total_partitions"] or 1)
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
    parser.add_argument("--host", default="localhost", help="ClickHouse HTTP host")
    parser.add_argument("--port", type=int, default=8123, help="ClickHouse HTTP port")
    parser.add_argument("--user", default=None, help="ClickHouse user")
    parser.add_argument("--password", default=None, help="ClickHouse password")
    parser.add_argument("--database", required=True, help="Database name")
    parser.add_argument("--table", required=True, nargs="+", help="Table name(s)")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER, help=f"Cluster name (default: {DEFAULT_CLUSTER})")
    parser.add_argument("--no-cluster", action="store_true", help="Don't use ON CLUSTER or clusterAllReplicas")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be optimized")
    parser.add_argument("--status", action="store_true", help="Show partition merge state")
    parser.add_argument(
        "--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
        help=f"Max tables to optimize concurrently (default: {DEFAULT_MAX_CONCURRENT})",
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

    cluster = None if args.no_cluster else args.cluster

    if args.status:
        show_status(args.host, args.port, args.database, args.table, cluster, args.user, args.password)
        return

    if args.dry_run:
        show_dry_run(args.host, args.port, args.database, args.table, cluster, args.user, args.password)
        return

    if len(args.table) == 1:
        run_single_table(args, cluster)
    else:
        run_multi_table(args, cluster)


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
