"""Shared async persistence store for APG capabilities.

Two implementations, selected automatically:

:class:`InMemoryStore`   — zero-config, single-process. Used for tests and
                           when no database URL is configured.
:class:`PostgreSQLStore` — JSONB-backed production store. Selected when
                           ``APG_DATABASE_URL`` (or ``DATABASE_URL``) is set.

Usage::

    from capabilities.common.db import get_store

    store = get_store()                    # InMemory unless APG_DATABASE_URL is set
    store = get_store("postgresql+asyncpg://user:pass@host/db")  # explicit

All records must contain an ``"id"`` field (str). Tenant isolation is enforced
by convention: always pass ``tenant_id`` in the record and use it as a filter in
``query()``.

Security note: ``query()`` uses JSONB containment (``@>``) rather than building
SQL from filter keys, eliminating SQL injection risk.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol, runtime_checkable

_log = logging.getLogger(__name__)


@runtime_checkable
class Store(Protocol):
	"""Minimal async record store used by all APG capability services."""

	async def get(self, collection: str, id: str) -> dict[str, Any] | None: ...
	async def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]: ...
	async def query(self, collection: str, filters: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]: ...
	async def delete(self, collection: str, id: str) -> bool: ...
	async def count(self, collection: str, filters: dict[str, Any]) -> int: ...
	async def close(self) -> None: ...


class InMemoryStore:
	"""Thread-unsafe in-memory store — zero dependencies, suitable for tests and CLI tools.

	Data is lost when the process exits. Use PostgreSQLStore for anything that
	needs to survive a restart.
	"""

	def __init__(self) -> None:
		# collection → id → record
		self._data: dict[str, dict[str, dict[str, Any]]] = {}

	async def get(self, collection: str, id: str) -> dict[str, Any] | None:
		return self._data.get(collection, {}).get(id)

	async def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
		if "id" not in record:
			raise ValueError(f"Record put to collection {collection!r} must contain an 'id' field")
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
		return len(await self.query(collection, filters, limit=10_000_000))

	async def close(self) -> None:
		pass


class PostgreSQLStore:
	"""JSONB-backed async PostgreSQL store.

	Requires: ``sqlalchemy[asyncio]`` and ``asyncpg``::

	    pip install "sqlalchemy[asyncio]" asyncpg

	Schema: run ``SCHEMA_SQL`` once against your database before first use, or
	use the Alembic migration in ``migrations/versions/``.

	All queries use parameterised JSONB containment (``data @> :filter``) which
	is both injection-safe and benefits from the GIN index on ``data``.
	"""

	def __init__(self, db_url: str) -> None:
		try:
			from sqlalchemy.ext.asyncio import (
				AsyncEngine,
				AsyncSession,
				async_sessionmaker,
				create_async_engine,
			)
		except ImportError as exc:
			raise RuntimeError(
				"Install sqlalchemy[asyncio] and asyncpg: "
				"pip install 'sqlalchemy[asyncio]' asyncpg"
			) from exc

		if not db_url.startswith("postgresql+asyncpg://"):
			# Auto-fix sync postgres:// URLs
			db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
			db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)

		self._engine = create_async_engine(
			db_url,
			echo=False,
			pool_pre_ping=True,
			pool_size=5,
			max_overflow=10,
		)
		self._session: async_sessionmaker[AsyncSession] = async_sessionmaker(
			self._engine,
			class_=AsyncSession,
			expire_on_commit=False,
		)

	async def get(self, collection: str, id: str) -> dict[str, Any] | None:
		from sqlalchemy import text
		async with self._session() as s:
			row = (await s.execute(
				text("SELECT data FROM apg_records WHERE collection = :c AND id = :id"),
				{"c": collection, "id": id},
			)).fetchone()
			return json.loads(row[0]) if row else None

	async def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
		from sqlalchemy import text
		if "id" not in record:
			raise ValueError(f"Record put to collection {collection!r} must contain an 'id' field")
		async with self._session() as s:
			await s.execute(
				text(
					"INSERT INTO apg_records (id, collection, tenant_id, data) "
					"VALUES (:id, :c, :t, CAST(:data AS jsonb)) "
					"ON CONFLICT (collection, id) DO UPDATE "
					"SET data = EXCLUDED.data, updated_at = now()"
				),
				{
					"id": record["id"],
					"c": collection,
					"t": record.get("tenant_id", "default"),
					"data": json.dumps(record, default=str),
				},
			)
			await s.commit()
		return record

	async def query(self, collection: str, filters: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
		"""Filter using JSONB containment — injection-safe, GIN-indexed."""
		from sqlalchemy import text
		filter_json = json.dumps(filters, default=str) if filters else "{}"
		async with self._session() as s:
			rows = (await s.execute(
				text(
					"SELECT data FROM apg_records "
					"WHERE collection = :c AND data @> CAST(:filter AS jsonb) "
					"ORDER BY created_at DESC "
					"LIMIT :lim"
				),
				{"c": collection, "filter": filter_json, "lim": limit},
			)).fetchall()
			return [json.loads(r[0]) for r in rows]

	async def delete(self, collection: str, id: str) -> bool:
		from sqlalchemy import text
		async with self._session() as s:
			result = await s.execute(
				text("DELETE FROM apg_records WHERE collection = :c AND id = :id"),
				{"c": collection, "id": id},
			)
			await s.commit()
			return result.rowcount > 0

	async def count(self, collection: str, filters: dict[str, Any]) -> int:
		from sqlalchemy import text
		filter_json = json.dumps(filters, default=str) if filters else "{}"
		async with self._session() as s:
			row = (await s.execute(
				text(
					"SELECT COUNT(*) FROM apg_records "
					"WHERE collection = :c AND data @> CAST(:filter AS jsonb)"
				),
				{"c": collection, "filter": filter_json},
			)).fetchone()
			return int(row[0]) if row else 0

	async def close(self) -> None:
		await self._engine.dispose()


SCHEMA_SQL = """\
-- APG shared JSONB record store.
-- Run once per database before starting the application, or use Alembic.

CREATE TABLE IF NOT EXISTS apg_records (
    id          TEXT        NOT NULL,
    collection  TEXT        NOT NULL,
    tenant_id   TEXT        NOT NULL DEFAULT 'default',
    data        JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection, id)
);

-- Tenant + collection index: fast per-tenant list queries
CREATE INDEX IF NOT EXISTS idx_apg_records_tenant
    ON apg_records (collection, tenant_id);

-- GIN index: fast JSONB containment filter queries
CREATE INDEX IF NOT EXISTS idx_apg_records_data_gin
    ON apg_records USING gin (data);

-- Keep updated_at accurate
CREATE OR REPLACE FUNCTION apg_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at := now(); RETURN NEW; END;
$$;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_apg_records_updated_at'
    ) THEN
        CREATE TRIGGER trg_apg_records_updated_at
            BEFORE UPDATE ON apg_records
            FOR EACH ROW EXECUTE FUNCTION apg_set_updated_at();
    END IF;
END $$;
"""


def get_store(db_url: str | None = None) -> Store:
	"""Return a Store appropriate for the current environment.

	Priority:
	1. Explicit ``db_url`` argument
	2. ``APG_DATABASE_URL`` environment variable
	3. ``DATABASE_URL`` environment variable
	4. Falls back to InMemoryStore when none of the above are set or when
	   asyncpg / sqlalchemy[asyncio] are not installed.
	"""
	resolved = db_url or os.environ.get("APG_DATABASE_URL") or os.environ.get("DATABASE_URL")
	if resolved:
		try:
			store = PostgreSQLStore(resolved)
			_log.debug("PostgreSQLStore configured: %s", resolved.split("@")[-1])
			return store
		except (ImportError, RuntimeError) as exc:
			_log.warning("PostgreSQLStore unavailable (%s) — falling back to InMemoryStore", exc)
	return InMemoryStore()
