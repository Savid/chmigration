#!/usr/bin/env python3
"""Migrate data from refined cluster (old schema) to raw cluster (new schema).

Reads per-table definitions from tables.yaml, then for each table + partition
executes INSERT INTO db.table SELECT <columns>
FROM cluster('{refined_cluster}', db, table) WHERE _partition_id = '...'.

Connects to the raw cluster and uses the pre-configured refined_cluster
remote server to read from the refined cluster.

Usage:
    # Dry run
    python migrate/migrate.py \\
      --host raw-endpoint --user admin --password secret \\
      --database default \\
      --table beacon_api_eth_v1_events_block \\
      --dry-run

    # Migrate specific tables
    python migrate/migrate.py \\
      --host raw-endpoint --user admin --password secret \\
      --database default \\
      --table beacon_api_eth_v1_events_block canonical_beacon_block

    # Migrate all tables in a database
    python migrate/migrate.py \\
      --host raw-endpoint --user admin --password secret \\
      --database default

    # Custom refined cluster name
    python migrate/migrate.py \\
      --host raw-endpoint --user admin --password secret \\
      --refined-cluster my_refined_cluster \\
      --database default
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

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc

sys.stdout.reconfigure(line_buffering=True)

_shutdown = False

from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TaskID


DEFAULT_TIMEOUT = 3600
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_RETRIES = 3
SCRIPT_DIR = Path(__file__).resolve().parent
TABLES_YAML = SCRIPT_DIR / "tables.yaml"
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
    params: dict[str, str | int] = {
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
# Partition discovery
# ---------------------------------------------------------------------------

def get_all_partitions(
    host: str,
    port: int,
    database: str,
    table: str,
    refined_cluster: str,
    user: str | None = None,
    password: str | None = None,
) -> list[str]:
    """Get all partition IDs from the refined cluster via the remote server."""
    sql = (
        f"SELECT DISTINCT partition_id "
        f"FROM clusterAllReplicas('{refined_cluster}', system.parts) "
        f"WHERE database = '{database}' AND table = '{table}' AND active = 1 "
        f"ORDER BY partition_id"
    )
    rows = query_json_rows(host, port, sql, user=user, password=password)
    return [r["partition_id"] for r in rows]


def get_partition_row_count(
    host: str,
    port: int,
    database: str,
    table: str,
    partition_id: str,
    refined_cluster: str,
    user: str | None = None,
    password: str | None = None,
) -> int:
    """Quick row count for a partition via system.parts on refined (no full scan)."""
    sql = (
        f"SELECT sum(rows) AS cnt "
        f"FROM clusterAllReplicas('{refined_cluster}', system.parts) "
        f"WHERE database = '{database}' AND table = '{table}' "
        f"AND partition_id = '{partition_id}' AND active = 1"
    )
    rows = query_json_rows(host, port, sql, timeout=30, user=user, password=password)
    return int(rows[0].get("cnt", 0)) if rows else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def truncate_error(error: str, max_len: int = 200) -> str:
    idx = error.find("Stack trace")
    if idx > 0:
        error = error[:idx].rstrip()
    if len(error) > max_len:
        return error[:max_len] + "..."
    return error


# ---------------------------------------------------------------------------
# Load definitions
# ---------------------------------------------------------------------------

def load_table_defs(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("tables", [])


def find_table_def(defs: list[dict], database: str, table: str) -> dict | None:
    """Find a table definition by distributed table name."""
    for d in defs:
        if d["database"] == database and d["distributed_table"] == table:
            return d
    return None


# ---------------------------------------------------------------------------
# Per-table migration
# ---------------------------------------------------------------------------

def migrate_table(
    raw_host: str,
    raw_port: int,
    raw_user: str | None,
    raw_password: str | None,
    refined_cluster: str,
    table_def: dict,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    dry_run: bool = False,
    partition_filter: list[str] | None = None,
    log: Callable[[str], None] = print,
    on_partition_done: Callable[[], None] | None = None,
) -> dict:
    """Migrate all partitions of a single table from refined to raw."""
    database = table_def["database"]
    local_table = table_def["local_table"]
    dist_table = table_def["distributed_table"]
    select_columns = table_def["select_columns"]

    # Discover partitions on refined cluster (queried via raw's remote server).
    if partition_filter:
        partitions = partition_filter
    else:
        partitions = get_all_partitions(
            raw_host, raw_port, database, local_table,
            refined_cluster, raw_user, raw_password,
        )

    results = {"ok": 0, "skipped": 0, "error": 0}
    details: list[dict] = []

    if not partitions:
        log(f"  No partitions found for {database}.{local_table}")
        return {
            "database": database,
            "table": dist_table,
            "local_table": local_table,
            **results,
            "partition_detail": [],
        }

    # Build the SELECT column list.
    col_list = ",\n    ".join(select_columns)

    for partition_id in partitions:
        if _shutdown:
            break

        # INSERT into raw's distributed table (has sharding key),
        # SELECT from refined via the pre-configured refined_cluster remote server.
        insert_sql = (
            f"INSERT INTO `{database}`.`{dist_table}`\n"
            f"SELECT\n    {col_list}\n"
            f"FROM cluster('{refined_cluster}', "
            f"'{database}', '{local_table}')\n"
            f"WHERE _partition_id = '{partition_id}'\n"
            f"SETTINGS force_primary_key = 0"
        )

        if dry_run:
            row_count = get_partition_row_count(
                raw_host, raw_port, database, local_table,
                partition_id, refined_cluster, raw_user, raw_password,
            )
            log(f"  {partition_id}  DRY    rows={row_count:>12,}")
            details.append({
                "partition_id": partition_id,
                "outcome": "dry_run",
                "rows": row_count,
            })
            if on_partition_done:
                on_partition_done()
            continue

        for attempt in range(1, retries + 1):
            try:
                query(
                    raw_host, raw_port, insert_sql,
                    timeout=timeout, user=raw_user, password=raw_password,
                )
                log(f"  {partition_id}  OK")
                results["ok"] += 1
                details.append({
                    "partition_id": partition_id,
                    "outcome": "ok",
                })
                break

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
        "database": database,
        "table": dist_table,
        "local_table": local_table,
        **results,
        "partition_detail": details,
    }


def write_state_log(result: dict, raw_host: str, refined_cluster: str) -> None:
    from datetime import datetime
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **result,
        "raw_host": raw_host,
        "refined_cluster": refined_cluster,
        "partitions_total": len(result.get("partition_detail", [])),
    }
    with open(STATE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_single_table(args, table_def: dict) -> None:
    database = table_def["database"]
    dist_table = table_def["distributed_table"]
    local_table = table_def["local_table"]

    partition_filter = getattr(args, "partition", None)
    if partition_filter:
        partitions = partition_filter
    else:
        partitions = get_all_partitions(
            args.host, args.port, database, local_table,
            args.refined_cluster, args.user, args.password,
        )
    mode = "DRY RUN" if args.dry_run else "migrate"
    print(f"Table: {database}.{dist_table} ({local_table})")
    print(f"Partitions: {len(partitions)} ({mode})\n")

    result = migrate_table(
        raw_host=args.host,
        raw_port=args.port,
        raw_user=args.user,
        raw_password=args.password,
        refined_cluster=args.refined_cluster,
        table_def=table_def,
        timeout=args.timeout,
        retries=args.retries,
        dry_run=args.dry_run,
        partition_filter=partition_filter,
    )

    total = sum(result[k] for k in ("ok", "skipped", "error"))
    print(f"\n{'=' * 60}")
    print(f"Migrate complete: {database}.{dist_table}")
    print(f"  OK:       {result['ok']:>4} / {total}")
    print(f"  Skipped:  {result['skipped']:>4}")
    print(f"  Error:    {result['error']:>4}")

    if not args.dry_run:
        write_state_log(result, args.host, args.refined_cluster)
        print(f"  State:    {STATE_LOG}")

    if result["error"]:
        sys.exit(1)


def run_multi_table(args, table_defs: list[dict]) -> None:
    import threading

    max_concurrent = args.max_concurrent

    # Discovery phase.
    print(f"Discovering partitions for {len(table_defs)} table(s)...")
    table_info: list[dict] = []
    for td in table_defs:
        database = td["database"]
        local_table = td["local_table"]
        partitions = get_all_partitions(
            args.host, args.port, database, local_table,
            args.refined_cluster, args.user, args.password,
        )
        table_info.append({
            "def": td,
            "partitions": len(partitions),
        })
        print(f"  {database}.{td['distributed_table']}: {len(partitions)} partitions")

    total_partitions = sum(t["partitions"] for t in table_info)
    mode = "DRY RUN" if args.dry_run else "migrate"
    print(f"\nTotal: {len(table_defs)} tables, {total_partitions} partitions ({mode})")
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
            td = info["def"]
            name = f"{td['database']}.{td['distributed_table']}"
            tid = progress.add_task(
                name,
                total=info["partitions"] or 1,
                short_name=name[:45],
                status_text="queued",
            )
            task_ids[name] = tid

        overall_tid = progress.add_task(
            "overall",
            total=total_partitions or 1,
            short_name="[cyan]OVERALL",
            status_text="0 done",
        )

        done_count = 0
        count_lock = threading.Lock()

        def do_one_table(info: dict) -> dict:
            nonlocal done_count
            td = info["def"]
            name = f"{td['database']}.{td['distributed_table']}"
            tid = task_ids[name]
            progress.update(tid, status_text="migrating...")

            def on_partition_done():
                nonlocal done_count
                progress.advance(tid)
                progress.advance(overall_tid)
                with count_lock:
                    done_count += 1
                progress.update(overall_tid, status_text=f"{done_count} done")

            def log(msg: str):
                short = msg.strip()[:60]
                progress.update(tid, status_text=short)

            result = migrate_table(
                raw_host=args.host,
                raw_port=args.port,
                raw_user=args.user,
                raw_password=args.password,
                refined_cluster=args.refined_cluster,
                table_def=td,
                timeout=args.timeout,
                retries=args.retries,
                dry_run=args.dry_run,
                log=log,
                on_partition_done=on_partition_done,
            )

            ok = result["ok"]
            bad = result["error"]
            status = f"done: {ok} ok"
            if bad > 0:
                status += f", [red]{bad} errors[/red]"
            progress.update(tid, status_text=status, completed=info["partitions"] or 1)
            return result

        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {pool.submit(do_one_table, info): info for info in table_info}
            for future in as_completed(futures):
                info = futures[future]
                td = info["def"]
                name = f"{td['database']}.{td['distributed_table']}"
                try:
                    result = future.result()
                    all_results.append(result)
                    if not args.dry_run:
                        write_state_log(result, args.host, args.refined_cluster)
                    if result["error"] > 0:
                        has_failures = True
                except Exception as e:
                    progress.update(
                        task_ids[name],
                        status_text=f"[red]CRASHED: {e}[/red]",
                    )
                    has_failures = True

    # Final summary.
    print(f"\n{'=' * 70}")
    print(f"{'Table':<45} {'OK':>5} {'Skip':>5} {'Err':>5}")
    print(f"{'-' * 70}")
    totals = {"ok": 0, "skipped": 0, "error": 0}
    for r in sorted(all_results, key=lambda x: f"{x['database']}.{x['table']}"):
        name = f"{r['database']}.{r['table']}"
        if len(name) > 43:
            name = name[:42] + "…"
        print(f"  {name:<43} {r['ok']:>5} {r['skipped']:>5} {r['error']:>5}")
        for k in totals:
            totals[k] += r[k]
    print(f"{'-' * 70}")
    print(f"  {'TOTAL':<43} {totals['ok']:>5} {totals['skipped']:>5} {totals['error']:>5}")
    if not args.dry_run:
        print(f"\nState: {STATE_LOG}")

    if has_failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate data from refined cluster to raw cluster with schema transformations",
    )
    # Raw cluster (target) — we connect here directly.
    parser.add_argument("--host", default="localhost", help="Raw ClickHouse HTTP host")
    parser.add_argument("--port", type=int, default=8123, help="Raw ClickHouse HTTP port")
    parser.add_argument("--user", default=None, help="Raw ClickHouse user")
    parser.add_argument("--password", default=None, help="Raw ClickHouse password")

    # Refined cluster (source) — uses pre-configured remote server on raw.
    parser.add_argument("--refined-cluster", default="refined_cluster", help="Remote cluster name for refined source (default: refined_cluster)")

    # Table selection.
    parser.add_argument("--database", required=True, help="Database name")
    parser.add_argument("--table", nargs="*", help="Table name(s) — omit to migrate all tables in database")
    parser.add_argument("--partition", nargs="*", help="Partition ID(s) to migrate — only with a single --table")

    # Options.
    parser.add_argument("--defs", default=str(TABLES_YAML), help=f"Path to tables.yaml (default: {TABLES_YAML})")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT, help="Max concurrent tables")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="INSERT query timeout in seconds")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retry failed partitions N times")

    args = parser.parse_args()

    if args.partition and (not args.table or len(args.table) != 1):
        parser.error("--partition requires exactly one --table")


    # Load definitions.
    defs_path = Path(args.defs)
    if not defs_path.exists():
        print(f"Missing definitions file: {defs_path}", file=sys.stderr)
        print(f"Generate with: python migrate/generate_defs.py", file=sys.stderr)
        sys.exit(1)

    all_defs = load_table_defs(defs_path)

    # Filter to requested database + tables.
    if args.table:
        table_defs = []
        for t in args.table:
            td = find_table_def(all_defs, args.database, t)
            if not td:
                print(f"No definition found for {args.database}.{t}", file=sys.stderr)
                sys.exit(1)
            table_defs.append(td)
    else:
        table_defs = [d for d in all_defs if d["database"] == args.database]
        if not table_defs:
            print(f"No tables found for database '{args.database}'", file=sys.stderr)
            sys.exit(1)

    if len(table_defs) == 1:
        run_single_table(args, table_defs[0])
    else:
        run_multi_table(args, table_defs)


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
