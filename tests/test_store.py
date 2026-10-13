import re
from pathlib import Path

from opsrag import store

INIT_SQL = Path(__file__).parent.parent / "db" / "init.sql"


class RecordingConnection:
    """Stands in for psycopg and keeps the SQL, so schema code can be checked without Postgres."""

    def __init__(self):
        self.statements: list[str] = []

    def execute(self, query: str, params=None):
        self.statements.append(query)


def touched_tables(statements: list[str]) -> set[str]:
    tables = set()
    for sql in statements:
        if truncate := re.match(r"TRUNCATE ([\w, ]+?)(?: CASCADE)?$", sql):
            tables.update(name.strip() for name in truncate.group(1).split(","))
        tables.update(re.findall(r"ALTER TABLE (\w+)", sql))
    return tables


def test_reset_only_touches_tables_created_by_init_sql():
    conn = RecordingConnection()
    store.reset(conn, 1024)
    created = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", INIT_SQL.read_text()))
    assert touched_tables(conn.statements) <= created


def test_reset_resizes_the_embedding_column():
    conn = RecordingConnection()
    store.reset(conn, 1024)
    assert "ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1024)" in conn.statements
