"""Wave X.1 — dialect-aware DDL and placeholder helpers in the generated app.

These tests execute the compiled app.py in a namespace, then flip the
_APG_DB_DIALECT flag inside that namespace to assert the helper functions
adapt correctly. Real PostgreSQL integration is out of scope for CI.
"""

from __future__ import annotations

import pytest

from compiler.compiler import compile_apg_string


_DIALECT_APP_SOURCE = """
module dialect_probe version 1.0.0 {}

table Widget {
    name: str;
}
"""


_ENV_KEYS = (
    "APG_API_KEY",
    "APG_AUTH_USERS",
    "APG_AUTO_MIGRATE",
    "APG_DATABASE_URL",
    "APG_JWT_SECRET",
    "APG_PG_URL",
    "APG_PRODUCTION",
    "APG_SESSION_SECRET",
    "APG_SQLITE_PATH",
    "DATABASE_URL",
)


@pytest.fixture()
def generated_ns(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    result = compile_apg_string(_DIALECT_APP_SOURCE)
    assert result.success, result.errors
    ns: dict[str, object] = {"__file__": "generated_dialect_app.py"}
    exec(compile(result.generated_files["app.py"], "generated_dialect_app.py", "exec"), ns)
    ns["_flask_app"].config["TESTING"] = True
    return ns


def _with_dialect(ns, dialect):
    """Temporarily set the module-level _APG_DB_DIALECT flag."""
    prior = ns["_APG_DB_DIALECT"]
    ns["_APG_DB_DIALECT"] = dialect
    return prior


def test_ddl_dialect_helpers_pg(generated_ns):
    prior = _with_dialect(generated_ns, "pg")
    try:
        assert generated_ns["_apg_ddl_pk"]() == "BIGSERIAL PRIMARY KEY"
        assert generated_ns["_apg_ddl_now"]() == "NOW()"
        assert generated_ns["_apg_ddl_text_type"]() == "TEXT"
    finally:
        generated_ns["_APG_DB_DIALECT"] = prior


def test_ddl_dialect_helpers_sqlite(generated_ns):
    prior = _with_dialect(generated_ns, "sqlite")
    try:
        assert generated_ns["_apg_ddl_pk"]() == "INTEGER PRIMARY KEY AUTOINCREMENT"
        assert generated_ns["_apg_ddl_now"]() == "datetime('now')"
        assert generated_ns["_apg_ddl_text_type"]() == "TEXT"
    finally:
        generated_ns["_APG_DB_DIALECT"] = prior


def test_qmark_rewrite_pg(generated_ns):
    prior = _with_dialect(generated_ns, "pg")
    try:
        qmark = generated_ns["_apg_qmark"]
        assert qmark("SELECT * FROM foo WHERE id=?") == "SELECT * FROM foo WHERE id=%s"
        assert qmark("UPDATE t SET a=?, b=? WHERE c=?") == "UPDATE t SET a=%s, b=%s WHERE c=%s"
        # question marks inside string literals must be left alone
        assert qmark("SELECT '?' AS q FROM t WHERE x=?") == "SELECT '?' AS q FROM t WHERE x=%s"
    finally:
        generated_ns["_APG_DB_DIALECT"] = prior


def test_qmark_passthrough_sqlite(generated_ns):
    prior = _with_dialect(generated_ns, "sqlite")
    try:
        qmark = generated_ns["_apg_qmark"]
        sql = "SELECT * FROM foo WHERE id=? AND name=?"
        assert qmark(sql) == sql
    finally:
        generated_ns["_APG_DB_DIALECT"] = prior


def test_touch_updated_at_ddl_only_on_pg(generated_ns):
    ddl_fn = generated_ns["_apg_touch_updated_at_ddl"]
    prior = _with_dialect(generated_ns, "sqlite")
    try:
        assert ddl_fn() == []
    finally:
        generated_ns["_APG_DB_DIALECT"] = prior
    prior = _with_dialect(generated_ns, "pg")
    try:
        stmts = ddl_fn()
        assert len(stmts) == 1
        assert "CREATE OR REPLACE FUNCTION apg_touch_updated_at" in stmts[0]
        assert "LANGUAGE plpgsql" in stmts[0]
    finally:
        generated_ns["_APG_DB_DIALECT"] = prior


def test_insert_returning_id_sqlite_uses_lastrowid(generated_ns):
    """On sqlite, _apg_insert_returning_id executes as-is and returns cursor.lastrowid."""
    prior = _with_dialect(generated_ns, "sqlite")

    class _Cur:
        def __init__(self):
            self.last_sql = None
            self.last_params = None
            self.lastrowid = 42

        def execute(self, sql, params):
            self.last_sql = sql
            self.last_params = params

    try:
        cur = _Cur()
        new_id = generated_ns["_apg_insert_returning_id"](
            cur, "INSERT INTO t (x) VALUES (?)", (1,)
        )
        assert new_id == 42
        assert cur.last_sql == "INSERT INTO t (x) VALUES (?)"
    finally:
        generated_ns["_APG_DB_DIALECT"] = prior


def test_insert_returning_id_pg_appends_returning(generated_ns):
    prior = _with_dialect(generated_ns, "pg")

    class _Cur:
        def __init__(self):
            self.last_sql = None
            self.last_params = None

        def execute(self, sql, params):
            self.last_sql = sql
            self.last_params = params

        def fetchone(self):
            return (99,)

    try:
        cur = _Cur()
        new_id = generated_ns["_apg_insert_returning_id"](
            cur, "INSERT INTO t (x) VALUES (?)", (1,)
        )
        assert new_id == 99
        # placeholder rewritten AND RETURNING id appended
        assert cur.last_sql == "INSERT INTO t (x) VALUES (%s) RETURNING id"
    finally:
        generated_ns["_APG_DB_DIALECT"] = prior


def test_search_returns_empty_on_pg(generated_ns):
    """FTS5 not available on PG - /search should silently return []."""
    prior = _with_dialect(generated_ns, "pg")
    try:
        rows = generated_ns["search_records"]("Widget", {"q": ["anything"]})
        assert rows == []
    finally:
        generated_ns["_APG_DB_DIALECT"] = prior


def test_generated_app_still_runs_on_sqlite(generated_ns):
    """Regression guard: standard CRUD via test client still works under sqlite."""
    client = generated_ns["_flask_app"].test_client()
    resp = client.post("/records/Widget", json={"name": "abc"})
    assert resp.status_code == 201, resp.get_json()
    rid = resp.get_json()["record"]["id"]
    resp = client.get(f"/records/Widget/{rid}")
    assert resp.status_code == 200
    resp = client.delete(f"/records/Widget/{rid}")
    assert resp.status_code in (200, 204)
