#!/usr/bin/env python3
"""Generate per-table migration definitions from spec.yaml + schemas/ + migration.sql.

Produces migrate/tables.yaml with the exact SELECT column list for each table,
ready for migrate.py to execute INSERT INTO ... SELECT operations.

Usage:
    python migrate/generate_defs.py
    python migrate/generate_defs.py --check  # verify tables.yaml is up-to-date
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc

# Import parsing utilities from the existing generator.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from generate_migration import (
    Column,
    SQLObject,
    decode_schema_sql,
    load_schema_objects,
    parse_columns_block,
)

SPEC_PATH = ROOT / "spec.yaml"
SCHEMAS_DIR = ROOT / "schemas"
MIGRATION_SQL = ROOT / "migration.sql"
OUTPUT_PATH = Path(__file__).resolve().parent / "tables.yaml"


# ---------------------------------------------------------------------------
# Parse new schema from migration.sql
# ---------------------------------------------------------------------------

def load_new_schemas(path: Path) -> dict[str, list[Column]]:
    """Parse CREATE TABLE IF NOT EXISTS ... local tables from migration.sql.

    Returns {qualified_table_name: [Column, ...]}.
    """
    text = path.read_text(encoding="utf-8")
    # Match local table definitions (they have explicit column blocks).
    stmt_re = re.compile(
        r"^CREATE TABLE IF NOT EXISTS\s+(\S+)\s+ON CLUSTER\s+'[^']+'\n\((.*?)\n\)\n",
        flags=re.M | re.S,
    )
    result: dict[str, list[Column]] = {}
    for match in stmt_re.finditer(text):
        table_name = match.group(1)
        if not table_name.endswith("_local"):
            continue
        cols = parse_columns_block(match.group(2))
        result[table_name] = cols
    return result


# ---------------------------------------------------------------------------
# Build column mapping
# ---------------------------------------------------------------------------

def strip_nullable(t: str) -> str:
    m = re.fullmatch(r"Nullable\((.+)\)", t.strip())
    return m.group(1) if m else t.strip()


def types_compatible(old_type: str, new_type: str) -> bool:
    """Check if old_type can be inserted into new_type without explicit CAST."""
    if old_type == new_type:
        return True
    # Nullable wrapping of the same base type is auto-promoted.
    if new_type == f"Nullable({old_type})":
        return True
    # LowCardinality(Nullable(X)) → LowCardinality(X) — ClickHouse handles.
    if old_type == f"LowCardinality(Nullable({strip_nullable(new_type)}))" or \
       new_type == f"LowCardinality(Nullable({strip_nullable(old_type)}))":
        return True
    return False


def build_table_def(
    database: str,
    table_name: str,  # local table name (with _local suffix)
    old_columns: list[Column],
    new_columns: list[Column],
    spec: dict,
) -> dict | None:
    """Build migration definition for a single table.

    Returns dict with select_columns list, or None if table has no old schema
    (skip — it's new and has no data to migrate).
    """
    qualified_old = f"{database}.{table_name}"

    # Build old column map.
    old_map: dict[str, Column] = {c.name: c for c in old_columns}

    # Get transformation rules from spec.
    columns_cfg = spec.get("columns", {})

    # Reverse rename map: new_name → old_name (for this table's local + distributed).
    reverse_renames: dict[str, str] = {}
    for qual_name in [qualified_old, qualified_old.replace("_local", "")]:
        renames = columns_cfg.get("renames", {}).get(qual_name, {})
        for old_name, new_name in renames.items():
            reverse_renames[new_name] = old_name

    # Type overrides: column → new_type.
    type_overrides: dict[str, str] = {}
    for qual_name in [qualified_old, qualified_old.replace("_local", "")]:
        overrides = spec.get("types", {}).get("overrides", {}).get(qual_name, {})
        type_overrides.update(overrides)

    # Columns added by add_or_replace that don't exist in old schema.
    add_or_replace = columns_cfg.get("add_or_replace", {})
    added_columns: set[str] = set()
    for col_name, table_defs in add_or_replace.items():
        if qualified_old in table_defs or qualified_old.replace("_local", "") in table_defs:
            if col_name not in old_map:
                added_columns.add(col_name)

    # Build SELECT column list in new schema column order.
    select_columns: list[str] = []

    for new_col in new_columns:
        col_name = new_col.name
        new_type = new_col.col_type

        # Is this column renamed? Find the old name.
        old_name = reverse_renames.get(col_name, col_name)

        if col_name in added_columns:
            # Column doesn't exist in old schema — need a default.
            if col_name == "updated_date_time":
                select_columns.append("now() AS updated_date_time")
            else:
                # Generic default for unknown added columns.
                select_columns.append(f"defaultValueOfTypeName('{new_type}') AS {col_name}")
            continue

        if old_name not in old_map:
            # Column doesn't exist in old schema and wasn't in add_or_replace.
            # This shouldn't happen for a well-formed migration, but handle it.
            select_columns.append(f"defaultValueOfTypeName('{new_type}') AS {col_name}")
            continue

        old_col = old_map[old_name]
        old_type = old_col.col_type

        # Determine if we need CAST and/or AS.
        needs_cast = not types_compatible(old_type, new_type) or col_name in type_overrides
        needs_rename = old_name != col_name

        if needs_cast and needs_rename:
            select_columns.append(f"CAST(`{old_name}` AS {new_type}) AS `{col_name}`")
        elif needs_cast:
            select_columns.append(f"CAST(`{col_name}` AS {new_type})")
        elif needs_rename:
            select_columns.append(f"`{old_name}` AS `{col_name}`")
        else:
            select_columns.append(f"`{col_name}`")

    # Derive distributed table names.
    dist_name = table_name.removesuffix("_local")

    return {
        "database": database,
        "local_table": table_name,
        "distributed_table": dist_name,
        "select_columns": select_columns,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(spec_path: Path, schemas_dir: Path, migration_sql: Path) -> list[dict]:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    # Load old schemas.
    old_objects = load_schema_objects(schemas_dir)

    # Load new schemas from migration.sql.
    new_schemas = load_new_schemas(migration_sql)

    # Excluded objects.
    exclude = set(spec.get("exclude_objects", []))

    tables: list[dict] = []

    for qualified_name, new_columns in sorted(new_schemas.items()):
        if qualified_name in exclude:
            continue

        database, table_name = qualified_name.split(".", 1)

        # Find corresponding old schema.
        if qualified_name not in old_objects:
            # New table with no old data — skip.
            continue

        old_obj = old_objects[qualified_name]

        table_def = build_table_def(
            database=database,
            table_name=table_name,
            old_columns=old_obj.columns,
            new_columns=new_columns,
            spec=spec,
        )
        if table_def:
            tables.append(table_def)

    return tables


def render_yaml(tables: list[dict]) -> str:
    lines = [
        "# Auto-generated migration definitions.",
        "# Source: spec.yaml + schemas/ + migration.sql",
        "#",
        "# Each table lists the SELECT columns for:",
        "#   INSERT INTO raw_distributed_table SELECT <columns> FROM refined_distributed_table",
        "#",
        "# Regenerate with: python migrate/generate_defs.py",
        "",
        "tables:",
    ]
    for t in tables:
        lines.append(f"  - database: {t['database']}")
        lines.append(f"    local_table: {t['local_table']}")
        lines.append(f"    distributed_table: {t['distributed_table']}")
        lines.append("    select_columns:")
        for col in t["select_columns"]:
            lines.append(f"      - \"{col}\"")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-table migration definitions")
    parser.add_argument("--spec", default=str(SPEC_PATH), help="Path to spec.yaml")
    parser.add_argument("--schemas-dir", default=str(SCHEMAS_DIR), help="Path to schemas/")
    parser.add_argument("--migration-sql", default=str(MIGRATION_SQL), help="Path to migration.sql")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output YAML path")
    parser.add_argument("--check", action="store_true", help="Verify output is up-to-date")
    args = parser.parse_args()

    tables = generate(Path(args.spec), Path(args.schemas_dir), Path(args.migration_sql))
    content = render_yaml(tables)

    if args.check:
        out = Path(args.output)
        if not out.exists():
            print(f"MISSING: {out}", file=sys.stderr)
            sys.exit(1)
        existing = out.read_text(encoding="utf-8")
        if existing != content:
            print(f"OUT OF DATE: {out}", file=sys.stderr)
            sys.exit(1)
        print(f"OK: {out} is up-to-date ({len(tables)} tables)")
        return

    out = Path(args.output)
    out.write_text(content, encoding="utf-8")
    print(f"Generated {out} ({len(tables)} tables)")


if __name__ == "__main__":
    main()
