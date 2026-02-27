import tempfile
import unittest
import re
from pathlib import Path

from generate_migration import MANUAL_TABLE_COMMENTS, generate_outputs, load_schema_objects


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = REPO_ROOT / "schemas"
SPEC_PATH = REPO_ROOT / "spec.yaml"
OUT_SQL = REPO_ROOT / "migration.sql"
OUT_MD = REPO_ROOT / "migration.md"


class TestGenerateMigration(unittest.TestCase):
    @staticmethod
    def _extract_table_statement(sql: str, table_name: str) -> str:
        m = re.search(
            rf"CREATE TABLE IF NOT EXISTS {re.escape(table_name)} ON CLUSTER '\{{cluster\}}'\n(.*?);",
            sql,
            flags=re.S,
        )
        if not m:
            raise AssertionError(f"missing table statement: {table_name}")
        return m.group(0)

    @staticmethod
    def _extract_table_columns(sql: str, table_name: str) -> dict[str, str]:
        m = re.search(
            rf"CREATE TABLE IF NOT EXISTS {re.escape(table_name)} ON CLUSTER '\{{cluster\}}'\n"
            r"\((.*?)\n\)\nENGINE =",
            sql,
            flags=re.S,
        )
        if not m:
            raise AssertionError(f"missing local column block: {table_name}")
        columns: dict[str, str] = {}
        for raw in m.group(1).splitlines():
            line = raw.strip().rstrip(",")
            cm = re.match(r"`([^`]+)`\s+(.+)", line)
            if cm:
                columns[cm.group(1)] = cm.group(2)
        return columns

    @staticmethod
    def _normalize_column_definition(definition: str) -> str:
        definition = re.sub(r"COMMENT\s+\\?'(?:\\.|[^'])*'", "", definition)
        return " ".join(definition.split()).strip()

    @staticmethod
    def _extract_table_comment_line(stmt: str) -> str | None:
        comments = [line.strip() for line in stmt.splitlines() if line.strip().startswith("COMMENT ")]
        if not comments:
            return None
        return comments[-1].rstrip(";")

    def test_parses_all_schema_objects(self) -> None:
        objects = load_schema_objects(SCHEMAS_DIR)
        self.assertEqual(len(objects), 292)

    def test_generation_is_deterministic(self) -> None:
        sql1, md1, _, _ = generate_outputs(SCHEMAS_DIR, SPEC_PATH)
        sql2, md2, _, _ = generate_outputs(SCHEMAS_DIR, SPEC_PATH)
        self.assertEqual(sql1, sql2)
        self.assertEqual(md1, md2)

    def test_regenerated_outputs_match_committed_files(self) -> None:
        sql, md, _, _ = generate_outputs(SCHEMAS_DIR, SPEC_PATH)
        self.assertEqual(sql, OUT_SQL.read_text(encoding="utf-8"))
        self.assertEqual(md, OUT_MD.read_text(encoding="utf-8"))

    def test_generate_to_tempdir(self) -> None:
        sql, md, _, _ = generate_outputs(SCHEMAS_DIR, SPEC_PATH)
        with tempfile.TemporaryDirectory() as td:
            out_sql = Path(td) / "migration.sql"
            out_md = Path(td) / "migration.md"
            out_sql.write_text(sql, encoding="utf-8")
            out_md.write_text(md, encoding="utf-8")
            self.assertTrue(out_sql.exists())
            self.assertTrue(out_md.exists())

    def test_local_engine_uses_templated_path(self) -> None:
        sql, _, _, _ = generate_outputs(SCHEMAS_DIR, SPEC_PATH)
        self.assertIn(
            "ReplicatedReplacingMergeTree('/clickhouse/{installation}/{cluster}/tables/{shard}/{database}/{table}', '{replica}', updated_date_time)",
            sql,
        )

    def test_output_omits_index_granularity_settings(self) -> None:
        sql, _, _, _ = generate_outputs(SCHEMAS_DIR, SPEC_PATH)
        self.assertNotIn("SETTINGS index_granularity = 8192", sql)

    def test_uses_og_shard_key_for_known_table(self) -> None:
        sql, _, _, _ = generate_outputs(SCHEMAS_DIR, SPEC_PATH)
        self.assertIn(
            "ENGINE = Distributed('{cluster}', 'default', 'beacon_api_eth_v1_beacon_blob_local', cityHash64(slot_start_date_time, meta_client_name, block_root))",
            sql,
        )

    def test_all_distributed_tables_use_as_local_format(self) -> None:
        sql, _, _, _ = generate_outputs(SCHEMAS_DIR, SPEC_PATH)
        stmt_re = re.compile(
            r"CREATE TABLE IF NOT EXISTS ([^\s]+) ON CLUSTER '\{cluster\}'\n(.*?);",
            flags=re.S,
        )
        for match in stmt_re.finditer(sql):
            table_name = match.group(1)
            if table_name.endswith("_local"):
                continue
            body = match.group(2)
            db, table = table_name.split(".", 1)
            self.assertIn(f"AS {db}.{table}_local", body)
            self.assertNotIn("\n(", body)

    def test_special_case_definition_overrides_are_applied(self) -> None:
        sql, _, _, _ = generate_outputs(SCHEMAS_DIR, SPEC_PATH)
        checks = [
            (
                "default.beacon_api_eth_v1_events_chain_reorg_local",
                "event_date_time",
                "DateTime64(3) CODEC(DoubleDelta, ZSTD(1))",
            ),
            (
                "default.beacon_api_eth_v1_events_contribution_and_proof_local",
                "event_date_time",
                "DateTime64(3) CODEC(DoubleDelta, ZSTD(1))",
            ),
            (
                "default.beacon_api_eth_v1_events_head_local",
                "event_date_time",
                "DateTime64(3) CODEC(DoubleDelta, ZSTD(1))",
            ),
            (
                "default.beacon_api_eth_v1_events_voluntary_exit_local",
                "event_date_time",
                "DateTime64(3) CODEC(DoubleDelta, ZSTD(1))",
            ),
            (
                "default.beacon_api_eth_v1_validator_attestation_data_local",
                "event_date_time",
                "DateTime64(3) CODEC(DoubleDelta, ZSTD(1))",
            ),
            (
                "default.beacon_api_eth_v2_beacon_block_local",
                "event_date_time",
                "DateTime64(3) CODEC(DoubleDelta, ZSTD(1))",
            ),
            (
                "default.beacon_api_eth_v3_validator_block_local",
                "slot_start_date_time",
                "DateTime CODEC(DoubleDelta, ZSTD(1))",
            ),
            (
                "default.consensus_engine_api_new_payload_local",
                "meta_execution_version",
                "LowCardinality(String)",
            ),
        ]
        for table, column, expected in checks:
            gen_cols = self._extract_table_columns(sql, table)
            self.assertIn(column, gen_cols, f"{table}.{column}")
            self.assertEqual(
                self._normalize_column_definition(gen_cols[column]),
                expected,
                f"{table}.{column}",
            )

    def test_execution_version_columns_do_not_have_default_empty_string(self) -> None:
        sql, _, _, _ = generate_outputs(SCHEMAS_DIR, SPEC_PATH)
        table = "default.consensus_engine_api_new_payload_local"
        gen_cols = self._extract_table_columns(sql, table)
        for column in [
            "meta_execution_implementation",
            "meta_execution_version_major",
            "meta_execution_version_minor",
            "meta_execution_version_patch",
        ]:
            definition = self._normalize_column_definition(gen_cols[column])
            self.assertNotIn("DEFAULT ''", definition, f"{table}.{column}")

    def test_distributed_comment_falls_back_to_local_comment(self) -> None:
        sql, _, _, _ = generate_outputs(SCHEMAS_DIR, SPEC_PATH)
        local_table = "default.beacon_api_eth_v1_beacon_committee_local"
        dist_table = "default.beacon_api_eth_v1_beacon_committee"

        local_stmt = self._extract_table_statement(sql, local_table)
        dist_stmt = self._extract_table_statement(sql, dist_table)
        local_comment = self._extract_table_comment_line(local_stmt)
        dist_comment = self._extract_table_comment_line(dist_stmt)

        self.assertIsNotNone(local_comment, local_table)
        self.assertEqual(dist_comment, local_comment)

    def test_manual_comment_fallback_applies_for_known_table(self) -> None:
        sql, _, _, _ = generate_outputs(SCHEMAS_DIR, SPEC_PATH)
        table = "admin.cryo"
        stmt = self._extract_table_statement(sql, table)
        comment = self._extract_table_comment_line(stmt)

        self.assertIn(table, MANUAL_TABLE_COMMENTS)
        self.assertEqual(comment, f"COMMENT {MANUAL_TABLE_COMMENTS[table]}")

    def test_generated_sql_has_no_tables_missing_comment(self) -> None:
        sql, _, _, _ = generate_outputs(SCHEMAS_DIR, SPEC_PATH)
        stmt_re = re.compile(
            r"CREATE TABLE IF NOT EXISTS ([^\s]+) ON CLUSTER '\{cluster\}'\n(.*?);",
            flags=re.S,
        )
        missing: list[str] = []
        for match in stmt_re.finditer(sql):
            table_name = match.group(1)
            body = match.group(0)
            comment_count = sum(1 for line in body.splitlines() if line.strip().startswith("COMMENT "))
            if comment_count == 0:
                missing.append(table_name)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
