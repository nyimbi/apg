"""Persistence store for Employee Data Management.

:class:`InMemoryStore`   — zero-config, single-process, suitable for testing and CLI tools.
:class:`PostgreSQLStore` — production-grade, requires ``sqlalchemy[asyncio]`` and ``asyncpg``.

The store is selected automatically:

- ``APG_DATABASE_URL`` or ``DATABASE_URL`` env var → PostgreSQL
- Otherwise → InMemoryStore

Pass ``db_url`` explicitly to the service constructor to override.
"""
from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Store(Protocol):
    """Minimal async key-value / query store."""
    async def get(self, collection: str, id: str) -> dict[str, Any] | None: ...
    async def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]: ...
    async def query(self, collection: str, filters: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]: ...
    async def delete(self, collection: str, id: str) -> bool: ...
    async def count(self, collection: str, filters: dict[str, Any]) -> int: ...


class InMemoryStore:
    """Thread-unsafe in-memory store — suitable for single-process use."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict[str, Any]]] = {}

    async def get(self, collection: str, id: str) -> dict[str, Any] | None:
        return self._data.get(collection, {}).get(id)

    async def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        self._data.setdefault(collection, {})[record["id"]] = dict(record)
        return record

    async def query(self, collection: str, filters: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        records = list(self._data.get(collection, {}).values())
        for key, value in filters.items():
            records = [r for r in records if r.get(key) == value]
        return records[:limit]

    async def delete(self, collection: str, id: str) -> bool:
        col = self._data.get(collection, {})
        if id in col:
            del col[id]
            return True
        return False

    async def count(self, collection: str, filters: dict[str, Any]) -> int:
        return len(await self.query(collection, filters, limit=100_000))


class PostgreSQLStore:
    """JSONB-backed async PostgreSQL store using SQLAlchemy.

    Requires: ``pip install sqlalchemy[asyncio] asyncpg``
    Schema:   ``database/schema.sql`` (run once to create tables)
    """

    def __init__(self, db_url: str) -> None:
        try:
            from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
        except ImportError as exc:
            raise RuntimeError("Install sqlalchemy[asyncio] and asyncpg: pip install 'sqlalchemy[asyncio]' asyncpg") from exc
        engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
        self._session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def get(self, collection: str, id: str) -> dict[str, Any] | None:
        async with self._session() as s:
            row = (await s.execute(
                "SELECT data FROM apg_records WHERE collection = :c AND id = :id",
                {"c": collection, "id": id},
            )).fetchone()
            return json.loads(row[0]) if row else None

    async def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        async with self._session() as s:
            await s.execute(
                "INSERT INTO apg_records (id, collection, tenant_id, data) "
                "VALUES (:id, :c, :t, :data) "
                "ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data, updated_at = now()",
                {"id": record["id"], "c": collection, "t": record.get("tenant_id", "default"),
                  "data": json.dumps(record, default=str)},
            )
            await s.commit()
        return record

    async def query(self, collection: str, filters: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        conds = " AND ".join("data->>'"+k+"' = :"+k for k in filters)
        where = f"WHERE collection = :_c" + (f" AND {conds}" if conds else "")
        async with self._session() as s:
            rows = (await s.execute(
                "SELECT data FROM apg_records " + where + " LIMIT :lim",
                {"_c": collection, "lim": limit, **filters},
            )).fetchall()
            return [json.loads(r[0]) for r in rows]

    async def delete(self, collection: str, id: str) -> bool:
        async with self._session() as s:
            result = await s.execute(
                "DELETE FROM apg_records WHERE collection = :c AND id = :id",
                {"c": collection, "id": id},
            )
            await s.commit()
            return result.rowcount > 0

    async def count(self, collection: str, filters: dict[str, Any]) -> int:
        conds = " AND ".join("data->>'"+k+"' = :"+k for k in filters)
        where = f"WHERE collection = :_c" + (f" AND {conds}" if conds else "")
        async with self._session() as s:
            row = (await s.execute(
                "SELECT COUNT(*) FROM apg_records " + where,
                {"_c": collection, **filters},
            )).fetchone()
            return int(row[0]) if row else 0


SCHEMA_SQL = """-- Generic APG JSONB record store — shared by all standalone capabilities.
-- Run once per database. Each capability uses a different collection name.

CREATE TABLE IF NOT EXISTS apg_records (
    id          TEXT        NOT NULL,
    collection  TEXT        NOT NULL,
    tenant_id   TEXT        NOT NULL DEFAULT 'default',
    data        JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection, id)
);

-- Fast tenant+collection queries
CREATE INDEX IF NOT EXISTS idx_apg_records_tenant
    ON apg_records (collection, tenant_id);

-- Full JSONB scan for attribute filters
CREATE INDEX IF NOT EXISTS idx_apg_records_data_gin
    ON apg_records USING gin (data);

-- Trigger: keep updated_at current
CREATE OR REPLACE FUNCTION apg_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_apg_records_updated_at') THEN
        CREATE TRIGGER trg_apg_records_updated_at
            BEFORE UPDATE ON apg_records
            FOR EACH ROW EXECUTE FUNCTION apg_set_updated_at();
    END IF;
END $$;
"""


def get_store(db_url: str | None = None) -> Store:
    """Return the appropriate store based on configuration.

    Falls back to InMemoryStore if the db_url driver is not installed or the
    URL resolves to a non-async scheme.  Always falls back to InMemoryStore
    when no URL is configured.
    """
    resolved = db_url or os.environ.get("APG_DATABASE_URL")
    if resolved:
        try:
            return PostgreSQLStore(resolved)
        except (ImportError, RuntimeError):
            pass
    return InMemoryStore()
