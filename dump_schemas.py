#!/usr/bin/env python3
"""Dump current ClickHouse schemas to schemas/{database}/{table}.sql.

Connects to localhost:8123 via HTTP and runs SHOW CREATE TABLE for every table
in the default, observoor, and admin databases.

Usage:
    python dump_schemas.py [--host HOST] [--port PORT]
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DATABASES = ["default", "observoor", "admin"]
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "schemas"


def query(host: str, port: int, sql: str) -> str:
    url = f"http://{host}:{port}/"
    data = sql.encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8").strip()
    except urllib.error.URLError as e:
        print(f"Error connecting to ClickHouse at {host}:{port}: {e}", file=sys.stderr)
        sys.exit(1)


def query_json_rows(host: str, port: int, sql: str) -> list[dict]:
    result = query(host, port, f"{sql} FORMAT JSONEachRow")
    if not result:
        return []
    rows: list[dict] = []
    for line in result.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def list_tables(host: str, port: int, database: str) -> list[str]:
    sql = (
        "SELECT name FROM system.tables "
        f"WHERE database = '{database}' "
        "AND is_temporary = 0 "
        "AND name NOT LIKE '.inner%' "
        "ORDER BY name"
    )
    rows = query_json_rows(host, port, sql)
    return [str(row.get("name", "")) for row in rows if row.get("name")]


def show_create_table(host: str, port: int, database: str, table: str) -> str:
    sql = f"SHOW CREATE TABLE `{database}`.`{table}`"
    return query(host, port, sql)


def get_table_comment(host: str, port: int, database: str, table: str) -> str:
    sql = (
        "SELECT comment FROM system.tables "
        f"WHERE database = '{database}' AND name = '{table}' "
        "LIMIT 1"
    )
    rows = query_json_rows(host, port, sql)
    if not rows:
        return ""
    return str(rows[0].get("comment", ""))


def get_column_comments(host: str, port: int, database: str, table: str) -> dict[str, str]:
    sql = (
        "SELECT name, comment FROM system.columns "
        f"WHERE database = '{database}' AND table = '{table}' "
        "AND comment != '' "
        "ORDER BY position"
    )
    comments: dict[str, str] = {}
    for row in query_json_rows(host, port, sql):
        name = str(row.get("name", ""))
        comment = str(row.get("comment", ""))
        if name and comment:
            comments[name] = comment
    return comments


def escape_comment(comment: str) -> str:
    return comment.replace("\\", "\\\\").replace("'", "\\'")


def normalize_comment_literal(literal: str) -> str:
    text = literal.strip().rstrip(";").strip()
    if text.startswith("\\'") and text.endswith("\\'") and len(text) >= 4:
        text = "'" + text[2:-2] + "'"

    if text.startswith("'") and text.endswith("'") and len(text) >= 2:
        inner = text[1:-1].replace("\\\\", "\\").replace("\\'", "'")
        return f"'{escape_comment(inner)}'"
    return text


def find_table_comment_lines(lines: list[str]) -> list[int]:
    indexes: list[int] = []
    for idx, line in enumerate(lines):
        if re.match(r"^\s*COMMENT\s+.+;?\s*$", line.strip()):
            indexes.append(idx)
    return indexes


def ensure_comments_in_ddl(ddl: str, table_comment: str, column_comments: dict[str, str]) -> str:
    text = ddl.replace("\\n", "\n")
    lines = text.splitlines()

    # Add column comments only when missing.
    for idx, line in enumerate(lines):
        stripped = line.strip()
        m = re.match(r"`([^`]+)`\s+.+", stripped)
        if not m:
            continue
        col_name = m.group(1)
        col_comment = column_comments.get(col_name, "")
        if not col_comment:
            continue
        if " COMMENT " in line:
            continue

        trailing_comma = "," if line.rstrip().endswith(",") else ""
        base = line.rstrip()
        if trailing_comma:
            base = base[:-1]

        esc = escape_comment(col_comment)
        codec_idx = base.find(" CODEC(")
        if codec_idx >= 0:
            before = base[:codec_idx]
            after = base[codec_idx:]
            new_line = f"{before} COMMENT '{esc}'{after}{trailing_comma}"
        else:
            new_line = f"{base} COMMENT '{esc}'{trailing_comma}"
        lines[idx] = new_line

    # Normalize to at most one table COMMENT line.
    existing_comment_literal = None
    comment_indexes = find_table_comment_lines(lines)
    if comment_indexes:
        last_comment_line = lines[comment_indexes[-1]].strip()
        m = re.match(r"^\s*COMMENT\s+(.+?)\s*;?\s*$", last_comment_line)
        if m:
            existing_comment_literal = normalize_comment_literal(m.group(1))

    if comment_indexes:
        lines = [line for idx, line in enumerate(lines) if idx not in set(comment_indexes)]

    target_comment_literal = existing_comment_literal
    if table_comment:
        target_comment_literal = f"'{escape_comment(table_comment)}'"

    if target_comment_literal:
        if lines and lines[-1].rstrip().endswith(";"):
            last = lines[-1].rstrip()
            lines[-1] = last[:-1]
            lines.append(f"COMMENT {target_comment_literal};")
        else:
            lines.append(f"COMMENT {target_comment_literal}")

    updated = "\n".join(lines).rstrip() + "\n"
    return updated.replace("\n", "\\n")


def main():
    parser = argparse.ArgumentParser(description="Dump ClickHouse schemas")
    parser.add_argument("--host", default="localhost", help="ClickHouse host (default: localhost)")
    parser.add_argument("--port", type=int, default=8123, help="ClickHouse HTTP port (default: 8123)")
    args = parser.parse_args()

    total = 0

    for database in DATABASES:
        tables = list_tables(args.host, args.port, database)
        if not tables:
            print(f"  {database}: no tables found (database may not exist)")
            continue

        db_dir = OUTPUT_DIR / database
        db_dir.mkdir(parents=True, exist_ok=True)

        for i, table in enumerate(tables, 1):
            print(f"  {database}: [{i}/{len(tables)}] {table}", flush=True)
            ddl = show_create_table(args.host, args.port, database, table)
            table_comment = get_table_comment(args.host, args.port, database, table)
            column_comments = get_column_comments(args.host, args.port, database, table)
            ddl = ensure_comments_in_ddl(ddl, table_comment, column_comments)
            out_path = db_dir / f"{table}.sql"
            out_path.write_text(ddl + "\n", encoding="utf-8")
            total += 1

        print(f"  {database}: {len(tables)} tables done")

    print(f"\nTotal: {total} schemas written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
