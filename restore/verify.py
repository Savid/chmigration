#!/usr/bin/env python3
"""Verify restored ClickHouse data by comparing row counts and content hashes.

Discovers backed-up partitions from the source cluster's system.backup_log,
then runs count() + groupBitXor(cityHash64(*)) on both source and target
clusters to verify data integrity per partition.

Works correctly even after OPTIMIZE FINAL on the target — the hash is based
on logical data, not physical layout.

Usage:
    # Single table
    python restore/verify.py \\
      --source-host old-cluster \\
      --host localhost --user admin --password secret \\
      --database default --table beacon_api_eth_v1_events_block

    # Multiple tables concurrently
    python restore/verify.py \\
      --source-host old-cluster \\
      --host localhost --user admin --password secret \\
      --database default \\
      --table beacon_api_eth_v1_events_block \\
             canonical_beacon_block \\
      --max-concurrent 5

    # Quick mode (row counts only, no hashing)
    python restore/verify.py \\
      --source-host old-cluster \\
      --host localhost --user admin --password secret \\
      --database default --table canonical_beacon_block --counts-only
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
from pathlib import Path
from typing import Callable

sys.stdout.reconfigure(line_buffering=True)

_shutdown = False

from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TaskID


DEFAULT_CLUSTER = "replicated"
DEFAULT_TIMEOUT = 3600
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_RETRIES = 3
SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_LOG = SCRIPT_DIR / "verify.jsonl"


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

def resolve_table(host: str, port: int, database: str, table: str) -> tuple[str, str, bool]:
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

    m = re.match(r"Distributed\(\s*'[^']+'\s*,\s*'([^']+)'\s*,\s*'([^']+)'", engine_full)
    if m:
        return m.group(1), m.group(2), True

    return database, table, False


def needs_final(host: str, port: int, database: str, table: str) -> bool:
    """Check if a table uses a Replacing/Collapsing engine that needs FINAL."""
    rows = query_json_rows(
        host, port,
        f"SELECT engine FROM system.tables "
        f"WHERE database = '{database}' AND name = '{table}'",
    )
    if not rows:
        return False
    engine = rows[0].get("engine", "")
    # Covers ReplacingMergeTree, ReplicatedReplacingMergeTree,
    # CollapsingMergeTree, VersionedCollapsingMergeTree, etc.
    return "Replacing" in engine or "Collapsing" in engine


# ---------------------------------------------------------------------------
# Partition discovery
# ---------------------------------------------------------------------------

def discover_backups(
    host: str,
    port: int,
    database: str,
    table: str,
    cluster: str | None = None,
) -> list[dict]:
    """Query system.backup_log on the source cluster to find completed backups."""
    source = (
        f"clusterAllReplicas('{cluster}', system.backup_log)"
        if cluster
        else "system.backup_log"
    )
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
        m = re.search(r"Disk\(\\'s3_backup\\',\s*\\'([^']+)\\'\)", name)
        if not m:
            m = re.search(r"Disk\('s3_backup',\s*'([^']+)'\)", name)
        if not m:
            continue
        path = m.group(1)
        parts = path.split("/")
        if len(parts) != 3:
            continue
        backups.append({
            "path": path,
            "partition_id": parts[2],
        })
    return backups


def get_all_partitions(
    host: str,
    port: int,
    database: str,
    table: str,
    cluster: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> list[str]:
    """Get all partition IDs from system.parts on a cluster."""
    source = (
        f"clusterAllReplicas('{cluster}', system.parts)"
        if cluster
        else "system.parts"
    )
    sql = (
        f"SELECT DISTINCT partition_id "
        f"FROM {source} "
        f"WHERE database = '{database}' AND table = '{table}' AND active = 1 "
        f"ORDER BY partition_id"
    )
    rows = query_json_rows(host, port, sql, user=user, password=password)
    return [r["partition_id"] for r in rows]


# ---------------------------------------------------------------------------
# Per-partition verification
# ---------------------------------------------------------------------------

def hash_partition(
    host: str,
    port: int,
    database: str,
    table: str,
    partition_id: str,
    timeout: int = DEFAULT_TIMEOUT,
    user: str | None = None,
    password: str | None = None,
    use_final: bool = False,
) -> dict:
    """Compute row count + content hash for a single partition.

    Queries the table directly (use the Distributed table name to get all
    shards routed correctly — one replica per shard, no double-counting).

    Returns dict with keys: count, hash. Hash is None if partition is empty.
    """
    final = " FINAL" if use_final else ""
    sql = (
        f"SELECT count() AS cnt, groupBitXor(cityHash64(*)) AS hash "
        f"FROM `{database}`.`{table}`{final} "
        f"WHERE _partition_id = '{partition_id}'"
    )
    rows = query_json_rows(host, port, sql, timeout=timeout, user=user, password=password)
    if not rows:
        return {"count": 0, "hash": None}
    # Normalize hash to string — older ClickHouse versions return UInt64 as
    # string in JSON while newer versions return it as a number.
    raw_hash = rows[0].get("hash")
    return {
        "count": int(rows[0].get("cnt", 0)),
        "hash": str(raw_hash) if raw_hash is not None else None,
    }


def count_partition(
    host: str,
    port: int,
    database: str,
    table: str,
    partition_id: str,
    user: str | None = None,
    password: str | None = None,
    use_final: bool = False,
) -> int:
    """Row count for a partition. Uses the Distributed table for cluster-wide view."""
    final = " FINAL" if use_final else ""
    sql = (
        f"SELECT count() AS cnt "
        f"FROM `{database}`.`{table}`{final} "
        f"WHERE _partition_id = '{partition_id}'"
    )
    rows = query_json_rows(host, port, sql, timeout=30, user=user, password=password)
    return int(rows[0].get("cnt", 0)) if rows else 0


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
# Per-table verification
# ---------------------------------------------------------------------------

def verify_table(
    source_host: str,
    source_port: int,
    target_host: str,
    target_port: int,
    source_database: str,
    source_table: str,
    source_cluster: str | None,
    target_cluster: str | None,
    target_user: str | None,
    target_password: str | None,
    counts_only: bool = False,
    hash_timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    log: Callable[[str], None] = print,
    on_partition_done: Callable[[], None] | None = None,
) -> dict:
    """Verify all backed-up partitions of a single table.

    Returns results dict with per-partition detail.
    """
    local_db, local_table, is_distributed = resolve_table(
        source_host, source_port, source_database, source_table,
    )
    if is_distributed:
        log(f"  {source_database}.{source_table} -> {local_db}.{local_table}")

    backups = discover_backups(source_host, source_port, local_db, local_table, source_cluster)

    # Detect if FINAL is needed (ReplacingMergeTree, CollapsingMergeTree, etc.)
    use_final = needs_final(source_host, source_port, local_db, local_table)
    if use_final:
        log(f"  Engine requires FINAL for accurate comparison")

    # Use the Distributed table for data queries (routes to one replica per
    # shard, avoids double-counting with clusterAllReplicas). Fall back to
    # local table name if the input wasn't a Distributed table.
    query_table = source_table if is_distributed else local_table

    results = {"ok": 0, "row_diff": 0, "hash_diff": 0, "missing": 0, "error": 0}
    details: list[dict] = []

    if not backups:
        return {
            "database": local_db,
            "table": local_table,
            "source_table": f"{source_database}.{source_table}",
            **results,
            "partition_detail": [],
        }

    # Check which partitions actually exist on target (use local table + system.parts).
    target_partitions = set(get_all_partitions(
        target_host, target_port, local_db, local_table,
        target_cluster, target_user, target_password,
    ))

    for bk in backups:
        if _shutdown:
            break

        partition_id = bk["partition_id"]

        # Quick check: does the partition exist on target at all?
        if partition_id not in target_partitions:
            log(f"  {partition_id}  MISSING")
            results["missing"] += 1
            details.append({
                "partition_id": partition_id,
                "outcome": "missing",
                "source_count": None,
                "target_count": None,
            })
            if on_partition_done:
                on_partition_done()
            continue

        for attempt in range(1, retries + 1):
            try:
                if counts_only:
                    src_count = count_partition(
                        source_host, source_port, source_database, query_table,
                        partition_id, use_final=use_final,
                    )
                    tgt_count = count_partition(
                        target_host, target_port, source_database, query_table,
                        partition_id, user=target_user, password=target_password,
                        use_final=use_final,
                    )
                    if src_count == tgt_count:
                        log(f"  {partition_id}  OK     rows={src_count:>12,}")
                        results["ok"] += 1
                        details.append({
                            "partition_id": partition_id,
                            "outcome": "ok",
                            "source_count": src_count,
                            "target_count": tgt_count,
                        })
                    else:
                        diff = tgt_count - src_count
                        log(f"  {partition_id}  ROWS   source={src_count:>12,}  target={tgt_count:>12,}  diff={diff:+,}")
                        results["row_diff"] += 1
                        details.append({
                            "partition_id": partition_id,
                            "outcome": "row_diff",
                            "source_count": src_count,
                            "target_count": tgt_count,
                        })
                else:
                    # Full hash comparison — run source and target in parallel.
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        src_future = pool.submit(
                            hash_partition,
                            source_host, source_port, source_database, query_table,
                            partition_id, timeout=hash_timeout,
                            use_final=use_final,
                        )
                        tgt_future = pool.submit(
                            hash_partition,
                            target_host, target_port, source_database, query_table,
                            partition_id, timeout=hash_timeout,
                            user=target_user, password=target_password,
                            use_final=use_final,
                        )
                        src = src_future.result()
                        tgt = tgt_future.result()

                    src_count = src["count"]
                    tgt_count = tgt["count"]
                    src_hash = src["hash"]
                    tgt_hash = tgt["hash"]

                    if src_count == tgt_count and src_hash == tgt_hash:
                        log(f"  {partition_id}  OK     rows={src_count:>12,}  hash={src_hash}")
                        results["ok"] += 1
                        details.append({
                            "partition_id": partition_id,
                            "outcome": "ok",
                            "source_count": src_count,
                            "target_count": tgt_count,
                            "source_hash": src_hash,
                            "target_hash": tgt_hash,
                        })
                    elif src_count != tgt_count:
                        diff = tgt_count - src_count
                        log(f"  {partition_id}  ROWS   source={src_count:>12,}  target={tgt_count:>12,}  diff={diff:+,}")
                        results["row_diff"] += 1
                        details.append({
                            "partition_id": partition_id,
                            "outcome": "row_diff",
                            "source_count": src_count,
                            "target_count": tgt_count,
                            "source_hash": src_hash,
                            "target_hash": tgt_hash,
                        })
                    else:
                        log(f"  {partition_id}  HASH   rows={src_count:>12,}  src_hash={src_hash}  tgt_hash={tgt_hash}")
                        results["hash_diff"] += 1
                        details.append({
                            "partition_id": partition_id,
                            "outcome": "hash_diff",
                            "source_count": src_count,
                            "target_count": tgt_count,
                            "source_hash": src_hash,
                            "target_hash": tgt_hash,
                        })
                break  # success — no retry needed

            except (ClickHouseError, Exception) as e:
                err = truncate_error(str(e))
                if attempt < retries:
                    wait = 5 * attempt
                    log(f"  {partition_id}  RETRY  attempt {attempt}/{retries} ({err}) — waiting {wait}s")
                    time.sleep(wait)
                    continue
                log(f"  {partition_id}  ERROR  {err} (after {retries} attempts)")
                results["error"] += 1
                details.append({
                    "partition_id": partition_id,
                    "outcome": "error",
                    "error": err,
                })

        if on_partition_done:
            on_partition_done()

    return {
        "database": local_db,
        "table": local_table,
        "source_table": f"{source_database}.{source_table}",
        **results,
        "partition_detail": details,
    }


def write_report(result: dict, source_host: str, target_host: str) -> None:
    from datetime import datetime
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **result,
        "source_host": source_host,
        "target_host": target_host,
        "partitions_total": len(result.get("partition_detail", [])),
    }
    with open(REPORT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_single_table(args, target_cluster: str | None, source_cluster: str | None) -> None:
    table = args.table[0]

    local_db, local_table, is_distributed = resolve_table(
        args.source_host, args.source_port, args.database, table,
    )
    if is_distributed:
        print(f"Distributed table {args.database}.{table} -> local {local_db}.{local_table}")
    else:
        print(f"Local table: {local_db}.{local_table}")

    backups = discover_backups(args.source_host, args.source_port, local_db, local_table, source_cluster)
    if not backups:
        print("No backups found for this table")
        return

    mode = "counts only" if args.counts_only else "hash + counts"
    print(f"Verifying {len(backups)} partition(s) ({mode})\n")

    result = verify_table(
        source_host=args.source_host,
        source_port=args.source_port,
        target_host=args.host,
        target_port=args.port,
        source_database=args.database,
        source_table=table,
        source_cluster=source_cluster,
        target_cluster=target_cluster,
        target_user=args.user,
        target_password=args.password,
        counts_only=args.counts_only,
        hash_timeout=args.timeout,
        retries=args.retries,
    )

    total = sum(result[k] for k in ("ok", "row_diff", "hash_diff", "missing", "error"))
    print(f"\n{'=' * 60}")
    print(f"Verify complete: {local_db}.{local_table}")
    print(f"  OK:         {result['ok']:>4} / {total}")
    print(f"  Row diff:   {result['row_diff']:>4}")
    print(f"  Hash diff:  {result['hash_diff']:>4}")
    print(f"  Missing:    {result['missing']:>4}")
    print(f"  Error:      {result['error']:>4}")

    write_report(result, args.source_host, args.host)
    print(f"  Report:     {REPORT_LOG}")

    if result["row_diff"] or result["hash_diff"] or result["missing"] or result["error"]:
        sys.exit(1)


def run_multi_table(args, target_cluster: str | None, source_cluster: str | None) -> None:
    import threading

    tables = args.table
    max_concurrent = args.max_concurrent

    # Discovery phase.
    print(f"Discovering partitions for {len(tables)} table(s)...")
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
    mode = "counts only" if args.counts_only else "hash + counts"
    print(f"\nTotal: {len(tables)} tables, {total_partitions} partitions ({mode})")
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
                total=info["partitions"] or 1,
                short_name=info["source_table"][:45],
                status_text="queued",
            )
            task_ids[info["source_table"]] = tid

        overall_tid = progress.add_task(
            "overall",
            total=total_partitions or 1,
            short_name="[cyan]OVERALL",
            status_text="0 verified",
        )

        verified_count = 0
        count_lock = threading.Lock()

        def do_one_table(info: dict) -> dict:
            nonlocal verified_count
            table_name = info["source_table"]
            tid = task_ids[table_name]
            progress.update(tid, status_text="verifying...")

            def on_partition_done():
                nonlocal verified_count
                progress.advance(tid)
                progress.advance(overall_tid)
                with count_lock:
                    verified_count += 1
                progress.update(overall_tid, status_text=f"{verified_count} verified")

            def log(msg: str):
                short = msg.strip()[:60]
                progress.update(tid, status_text=short)

            result = verify_table(
                source_host=args.source_host,
                source_port=args.source_port,
                target_host=args.host,
                target_port=args.port,
                source_database=args.database,
                source_table=table_name,
                source_cluster=source_cluster,
                target_cluster=target_cluster,
                target_user=args.user,
                target_password=args.password,
                counts_only=args.counts_only,
                hash_timeout=args.timeout,
                retries=args.retries,
                log=log,
                on_partition_done=on_partition_done,
            )

            ok = result["ok"]
            bad = result["row_diff"] + result["hash_diff"] + result["missing"] + result["error"]
            status = f"done: {ok} ok"
            if bad > 0:
                status += f", [red]{bad} issues[/red]"
            progress.update(tid, status_text=status, completed=info["partitions"] or 1)
            return result

        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {pool.submit(do_one_table, info): info for info in table_info}
            for future in as_completed(futures):
                info = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    write_report(result, args.source_host, args.host)
                    bad = result["row_diff"] + result["hash_diff"] + result["missing"] + result["error"]
                    if bad > 0:
                        has_failures = True
                except Exception as e:
                    progress.update(
                        task_ids[info["source_table"]],
                        status_text=f"[red]CRASHED: {e}[/red]",
                    )
                    has_failures = True

    # Final summary.
    print(f"\n{'=' * 80}")
    print(f"{'Table':<45} {'OK':>5} {'Rows':>5} {'Hash':>5} {'Miss':>5} {'Err':>5}")
    print(f"{'-' * 80}")
    totals = {"ok": 0, "row_diff": 0, "hash_diff": 0, "missing": 0, "error": 0}
    for r in sorted(all_results, key=lambda x: x["source_table"]):
        name = r["source_table"]
        if len(name) > 43:
            name = name[:42] + "…"
        print(f"  {name:<43} {r['ok']:>5} {r['row_diff']:>5} {r['hash_diff']:>5} {r['missing']:>5} {r['error']:>5}")
        for k in totals:
            totals[k] += r[k]
    print(f"{'-' * 80}")
    print(f"  {'TOTAL':<43} {totals['ok']:>5} {totals['row_diff']:>5} {totals['hash_diff']:>5} {totals['missing']:>5} {totals['error']:>5}")
    print(f"\nReport: {REPORT_LOG}")

    if has_failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify restored ClickHouse data — compare row counts and content hashes",
    )
    parser.add_argument("--source-host", required=True, help="Source ClickHouse host")
    parser.add_argument("--source-port", type=int, default=8123, help="Source ClickHouse HTTP port")
    parser.add_argument("--host", default="localhost", help="Target ClickHouse HTTP host")
    parser.add_argument("--port", type=int, default=8123, help="Target ClickHouse HTTP port")
    parser.add_argument("--user", default=None, help="Target ClickHouse user")
    parser.add_argument("--password", default=None, help="Target ClickHouse password")
    parser.add_argument("--database", required=True, help="Database name")
    parser.add_argument("--table", required=True, nargs="+", help="Table name(s)")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER, help=f"Target cluster name (default: {DEFAULT_CLUSTER})")
    parser.add_argument("--no-cluster", action="store_true", help="Don't use clusterAllReplicas")
    parser.add_argument("--source-cluster", default=DEFAULT_CLUSTER, help=f"Source cluster name (default: {DEFAULT_CLUSTER})")
    parser.add_argument("--counts-only", action="store_true", help="Only compare row counts (skip hashing — much faster)")
    parser.add_argument(
        "--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
        help=f"Max tables to verify concurrently (default: {DEFAULT_MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Hash query timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--retries", type=int, default=DEFAULT_RETRIES,
        help=f"Retry failed partitions up to N times (default: {DEFAULT_RETRIES})",
    )
    args = parser.parse_args()

    target_cluster = None if args.no_cluster else args.cluster
    source_cluster = args.source_cluster

    if len(args.table) == 1:
        run_single_table(args, target_cluster, source_cluster)
    else:
        run_multi_table(args, target_cluster, source_cluster)


if __name__ == "__main__":
    import signal

    def _handle_sigint(sig, frame):
        global _shutdown
        if _shutdown:
            print("\nForce quit.", file=sys.stderr)
            sys.exit(130)
        _shutdown = True
        print("\nCtrl+C received — stopping...", file=sys.stderr)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
