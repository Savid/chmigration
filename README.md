# Declarative Migration Generator

`migration.sql` and `migration.md` are generated artifacts.

## Source of truth

- Raw input schemas: `schemas/{database}/*.sql`
- Declarative rules: `spec.yaml`
- Generator: `generate_migration.py`
- Optional shard-key reference SQL for distributed tables (configured in `spec.yaml`)
- Parity special-cases are defined directly in `spec.yaml` under `columns.definition_overrides`

## Generate

```bash
python generate_migration.py
```

This rewrites:

- `migration.sql`
- `migration.md`

## Check drift

```bash
python generate_migration.py --check
```

`--check` exits non-zero if committed files differ from generated output.

## Dump schemas

```bash
python dump_schemas.py --host localhost --port 8123
```

Writes source DDL files into `schemas/`.

## Rule precedence

Rules are applied in this order:

1. Exact table overrides (`rules.tables` style maps in `spec.yaml`)
2. Pattern/group rules (database/table family rules)
3. Global defaults

In other words: table-specific settings always win over generic settings.

## Determinism contract

Given identical `schemas/` and `spec.yaml`, the generator must produce byte-identical output files.
