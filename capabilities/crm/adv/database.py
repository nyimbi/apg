"""CRM Advanced Analytics — database utilities (backward-compat)."""
from dataclasses import dataclass, field
from typing import Any


class DatabaseManager:
    """In-memory database manager for CRM analytics records."""
    def __init__(self) -> None:
        self._tables: dict[str, list[dict[str, Any]]] = {}

    def create_table(self, name: str) -> None:
        self._tables.setdefault(name, [])

    def insert(self, table: str, record: dict[str, Any]) -> None:
        self._tables.setdefault(table, []).append(record)

    def query(self, table: str, **filters) -> list[dict[str, Any]]:
        rows = self._tables.get(table, [])
        return [r for r in rows if all(r.get(k) == v for k, v in filters.items())]

    def delete(self, table: str, **filters) -> int:
        rows = self._tables.get(table, [])
        before = len(rows)
        self._tables[table] = [r for r in rows if not all(r.get(k) == v for k, v in filters.items())]
        return before - len(self._tables[table])
