import unittest

from dump_schemas import ensure_comments_in_ddl


def decode_dumped_ddl(text: str) -> str:
    return text.replace("\\n", "\n")


def table_comment_count(decoded_ddl: str) -> int:
    return sum(1 for line in decoded_ddl.splitlines() if line.strip().startswith("COMMENT "))


class TestDumpSchemas(unittest.TestCase):
    def test_adds_table_comment_when_missing(self) -> None:
        ddl = (
            "CREATE TABLE default.foo\\n"
            "(\\n"
            "    `x` UInt64\\n"
            ")\\n"
            "ENGINE = MergeTree\\n"
            "ORDER BY x\\n"
        )
        out = decode_dumped_ddl(ensure_comments_in_ddl(ddl, "Hello", {}))
        self.assertIn("COMMENT 'Hello'", out)
        self.assertEqual(table_comment_count(out), 1)

    def test_deduplicates_existing_table_comments(self) -> None:
        ddl = (
            "CREATE TABLE default.foo\\n"
            "(\\n"
            "    `x` UInt64\\n"
            ")\\n"
            "ENGINE = MergeTree\\n"
            "ORDER BY x\\n"
            "COMMENT \\'Hello\\'\\n"
            "COMMENT 'Hello';\\n"
        )
        out = decode_dumped_ddl(ensure_comments_in_ddl(ddl, "Hello", {}))
        self.assertEqual(table_comment_count(out), 1)
        self.assertIn("COMMENT 'Hello'", out)

    def test_keeps_existing_table_comment_when_metadata_empty(self) -> None:
        ddl = (
            "CREATE TABLE default.foo\\n"
            "(\\n"
            "    `x` UInt64\\n"
            ")\\n"
            "ENGINE = MergeTree\\n"
            "ORDER BY x\\n"
            "COMMENT \\'Existing\\';\\n"
        )
        out = decode_dumped_ddl(ensure_comments_in_ddl(ddl, "", {}))
        self.assertEqual(table_comment_count(out), 1)
        self.assertIn("COMMENT 'Existing'", out)

    def test_inserts_column_comment_before_codec(self) -> None:
        ddl = (
            "CREATE TABLE default.foo\\n"
            "(\\n"
            "    `x` UInt64 CODEC(DoubleDelta, ZSTD(1))\\n"
            ")\\n"
            "ENGINE = MergeTree\\n"
            "ORDER BY x\\n"
        )
        out = decode_dumped_ddl(ensure_comments_in_ddl(ddl, "", {"x": "hello"}))
        self.assertIn("`x` UInt64 COMMENT 'hello' CODEC(DoubleDelta, ZSTD(1))", out)

    def test_does_not_duplicate_existing_column_comment(self) -> None:
        ddl = (
            "CREATE TABLE default.foo\\n"
            "(\\n"
            "    `x` UInt64 COMMENT 'hello' CODEC(DoubleDelta, ZSTD(1))\\n"
            ")\\n"
            "ENGINE = MergeTree\\n"
            "ORDER BY x\\n"
        )
        out = decode_dumped_ddl(ensure_comments_in_ddl(ddl, "", {"x": "hello"}))
        self.assertEqual(out.count("`x` UInt64 COMMENT 'hello' CODEC(DoubleDelta, ZSTD(1))"), 1)


if __name__ == "__main__":
    unittest.main()
