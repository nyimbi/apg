"""Write-through dict/list shims for gradual service persistence migration.

These classes replace in-memory dicts/lists in services.  When a Store backed
by PostgreSQL is configured they persist every write immediately; when backed by
InMemoryStore the behaviour is identical to a plain dict/list.

Drop-in replacement:

    BEFORE:
        self._coops: dict[str, dict[str, Any]] = {}
        self._ledger: list[dict[str, Any]] = []

    AFTER:
        _store = get_store(db_url)
        self._coops = WriteThruDict("coops", tenant_id, _store)
        self._ledger = WriteThruList("ledger", tenant_id, _store)

No changes to business logic methods required.

Startup restore (call once after __init__ to reload persisted data):

    async def initialize(self) -> None:
        await self._coops.reload()
        await self._ledger.reload()
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterator

from .store import Store

_log = logging.getLogger(__name__)


def _schedule(coro: Any) -> None:
	"""Schedule a coroutine on the running event loop; silently skip if none."""
	try:
		loop = asyncio.get_running_loop()
		loop.create_task(coro)
	except RuntimeError:
		pass  # called outside async context — data is in memory


class WriteThruDict:
	"""Sync dict interface backed by an async Store.

	Reads come from the in-memory cache.  Every write schedules an async
	persist on the running event loop (fire-and-forget).  Call ``await
	self.reload()`` on startup to restore previously persisted data.
	"""

	def __init__(self, collection: str, tenant_id: str, store: Store) -> None:
		self._col = collection
		self._tenant = tenant_id
		self._store = store
		self._cache: dict[str, dict[str, Any]] = {}

	# ── dict interface ────────────────────────────────────────────────────────

	def __getitem__(self, key: str) -> dict[str, Any]:
		return self._cache[key]

	def __setitem__(self, key: str, value: dict[str, Any]) -> None:
		self._cache[key] = value
		_schedule(self._store.put(self._col, value))

	def __delitem__(self, key: str) -> None:
		del self._cache[key]
		_schedule(self._store.delete(self._col, key))

	def __contains__(self, key: object) -> bool:
		return key in self._cache

	def __len__(self) -> int:
		return len(self._cache)

	def __iter__(self) -> Iterator[str]:
		return iter(self._cache)

	def get(self, key: str, default: Any = None) -> dict[str, Any] | None:
		return self._cache.get(key, default)

	def values(self):  # noqa: ANN201
		return self._cache.values()

	def items(self):  # noqa: ANN201
		return self._cache.items()

	def keys(self):  # noqa: ANN201
		return self._cache.keys()

	def pop(self, key: str, *args: Any) -> dict[str, Any] | None:
		value = self._cache.pop(key, *args)
		if value is not None:
			_schedule(self._store.delete(self._col, key))
		return value

	def update(self, other: dict[str, dict[str, Any]]) -> None:
		for k, v in other.items():
			self[k] = v

	# ── async helpers ─────────────────────────────────────────────────────────

	async def reload(self) -> None:
		"""Restore all persisted records from the store into the in-memory cache."""
		try:
			records = await self._store.query(
				self._col, {"tenant_id": self._tenant}, limit=100_000
			)
			for r in records:
				self._cache[r["id"]] = r
			_log.debug("WriteThruDict(%s): loaded %d records", self._col, len(records))
		except Exception as exc:
			_log.warning("WriteThruDict(%s): reload failed: %s", self._col, exc)

	async def flush(self) -> None:
		"""Force-write the entire in-memory cache to the store."""
		for record in self._cache.values():
			await self._store.put(self._col, record)

	async def aquery(self, filters: dict[str, Any], limit: int = 1000) -> list[dict[str, Any]]:
		"""Query from cache (fast, no DB round-trip)."""
		results = list(self._cache.values())
		for k, v in filters.items():
			results = [r for r in results if r.get(k) == v]
		return results[:limit]


class WriteThruList:
	"""Append-only list backed by an async Store.

	Each appended item is assigned to ``collection`` with its ``id`` field as
	the store key.  Call ``await self.reload()`` on startup.
	"""

	def __init__(self, collection: str, tenant_id: str, store: Store) -> None:
		self._col = collection
		self._tenant = tenant_id
		self._store = store
		self._items: list[dict[str, Any]] = []

	# ── list interface ────────────────────────────────────────────────────────

	def append(self, item: dict[str, Any]) -> None:
		self._items.append(item)
		if "id" in item:
			_schedule(self._store.put(self._col, item))

	def extend(self, items: list[dict[str, Any]]) -> None:
		for item in items:
			self.append(item)

	def __getitem__(self, index: int | slice) -> Any:
		return self._items[index]

	def __len__(self) -> int:
		return len(self._items)

	def __iter__(self) -> Iterator[dict[str, Any]]:
		return iter(self._items)

	def __bool__(self) -> bool:
		return bool(self._items)

	# ── async helpers ─────────────────────────────────────────────────────────

	async def reload(self) -> None:
		"""Restore persisted items from the store into the in-memory list."""
		try:
			records = await self._store.query(
				self._col, {"tenant_id": self._tenant}, limit=100_000
			)
			# Sort by created_at so order is deterministic
			records.sort(key=lambda r: r.get("occurred_at") or r.get("created_at") or "")
			self._items = records
			_log.debug("WriteThruList(%s): loaded %d items", self._col, len(records))
		except Exception as exc:
			_log.warning("WriteThruList(%s): reload failed: %s", self._col, exc)
