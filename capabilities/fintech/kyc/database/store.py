"""KYC persistence store — re-exports from the shared APG store.

All capabilities now share the same Store protocol, InMemoryStore, and
PostgreSQLStore from ``capabilities.common.db``.  This module is kept for
backward compatibility with any direct imports of
``capabilities.fintech.kyc.database.store``.
"""
from capabilities.common.db.store import (
	Store,
	InMemoryStore,
	PostgreSQLStore,
	SCHEMA_SQL,
	get_store,
)

__all__ = ["Store", "InMemoryStore", "PostgreSQLStore", "SCHEMA_SQL", "get_store"]
