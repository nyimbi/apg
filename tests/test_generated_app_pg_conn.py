"""D-12 — generated apps reuse one PostgreSQL connection per thread.

The previous `_pg_connection` opened and closed a connection on every
journal/run/record touch. These tests pin the replacement contract:
cache-hit reuse, release-instead-of-close, dead-cache eviction with a
logged (not silent) connect failure, and caller discipline.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from compiler.compiler import compile_apg_string


_PG_APP_SOURCE = """
module pg_conn_probe version 1.0.0 {}

table Widget {
    name: str;
}
"""

_UNREACHABLE_DSN = "postgres://127.0.0.1:1/apg_probe?connect_timeout=1"

_ENV_KEYS = (
    "APG_DATABASE_URL",
    "APG_PG_URL",
    "DATABASE_URL",
)


@pytest.fixture()
def generated_ns(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    result = compile_apg_string(_PG_APP_SOURCE)
    assert result.success, result.errors
    ns: dict[str, object] = {"__file__": "generated_pg_conn_app.py"}
    exec(compile(result.generated_files["app.py"], "generated_pg_conn_app.py", "exec"), ns)
    ns["_APG_PG_URL"] = _UNREACHABLE_DSN
    ns["_APG_PG_CONNECT_FAILURES"] = set()
    ns["_APG_PG_CONNECTIONS"] = []
    return ns


def test_pg_connection_reuses_thread_local_cache(generated_ns):
    thread_local: Any = generated_ns["_APG_PG_THREAD_LOCAL"]
    cached = sqlite3.connect(":memory:")
    thread_local.connection = cached
    try:
        first = generated_ns["_pg_connection"]()
        second = generated_ns["_pg_connection"]()
        assert first is cached
        assert second is cached
        # still open: a closed connection would raise here
        first.execute("SELECT 1").fetchone()
    finally:
        thread_local.connection = None
        cached.close()


def test_pg_release_keeps_cached_open_and_closes_foreign(generated_ns):
    thread_local: Any = generated_ns["_APG_PG_THREAD_LOCAL"]
    release: Any = generated_ns["_pg_release"]
    cached = sqlite3.connect(":memory:")
    thread_local.connection = cached
    try:
        release(cached)
        assert cached.execute("SELECT 1").fetchone() is not None
    finally:
        thread_local.connection = None
        cached.close()

    foreign = sqlite3.connect(":memory:")
    release(foreign)
    with pytest.raises(sqlite3.ProgrammingError):
        foreign.execute("SELECT 1")


def test_pg_dead_cache_evicted_and_failure_logged_once(generated_ns):
    thread_local: Any = generated_ns["_APG_PG_THREAD_LOCAL"]
    dead = sqlite3.connect(":memory:")
    dead.close()  # reuse attempt must fail and evict rather than crash
    thread_local.connection = dead

    assert generated_ns["_pg_connection"]() is None
    assert thread_local.connection is None
    failures: set = generated_ns["_APG_PG_CONNECT_FAILURES"]
    assert len(failures) >= 1

    before = len(failures)
    assert generated_ns["_pg_connection"]() is None
    assert len(failures) == before  # deduped: one log line, not one per call


def test_pg_callers_release_instead_of_close(generated_ns):
    del generated_ns  # structural check runs against freshly generated source

    result = compile_apg_string(_PG_APP_SOURCE)
    assert result.success, result.errors
    source = result.generated_files["app.py"]

    # all five PG call sites release; the only conn.close() is _pg_discard's
    assert source.count("finally:\n        _pg_release(conn)") == 5
    assert source.count("conn.close()") == 1
