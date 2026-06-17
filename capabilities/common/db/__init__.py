"""Shared async persistence layer for APG capabilities.

Every capability imports Store / get_store from here:

    from capabilities.common.db import Store, get_store

    class MyService:
        def __init__(self, tenant_id: str, db_url: str | None = None) -> None:
            self.tenant_id = tenant_id
            self._store = get_store(db_url)

        async def create_item(self, record: dict) -> dict:
            return await self._store.put("my_items", record)

        async def get_item(self, item_id: str) -> dict | None:
            return await self._store.get("my_items", item_id)

        async def list_items(self) -> list[dict]:
            return await self._store.query("my_items", {"tenant_id": self.tenant_id})
"""
from .store import Store, InMemoryStore, PostgreSQLStore, get_store, SCHEMA_SQL

__all__ = ["Store", "InMemoryStore", "PostgreSQLStore", "get_store", "SCHEMA_SQL"]
