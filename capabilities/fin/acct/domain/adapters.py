"""
Bank Account Management — Domain adapters.

Provides the adapter protocol for swapping the in-memory stores in
BankAccountService with production-grade repositories (SQLAlchemy,
Redis, NATS, etc.) without touching service logic.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any


class AccountRepository(ABC):
	"""Protocol for account persistence."""

	@abstractmethod
	async def save(self, account: dict[str, Any]) -> None: ...

	@abstractmethod
	async def get(self, account_id: str) -> dict[str, Any] | None: ...

	@abstractmethod
	async def get_by_number(self, account_number: str) -> dict[str, Any] | None: ...

	@abstractmethod
	async def list_by_tenant(
		self,
		tenant_id: str,
		customer_id: str | None = None,
		status: str | None = None,
		account_type: str | None = None,
	) -> list[dict[str, Any]]: ...

	@abstractmethod
	async def delete(self, account_id: str) -> None: ...


class TransactionRepository(ABC):
	"""Protocol for transaction persistence."""

	@abstractmethod
	async def save(self, txn: dict[str, Any]) -> None: ...

	@abstractmethod
	async def get(self, transaction_id: str) -> dict[str, Any] | None: ...

	@abstractmethod
	async def list_by_account(
		self,
		account_id: str,
		tenant_id: str,
		from_date: str | None = None,
		to_date: str | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[dict[str, Any]]: ...


class EventPublisher(ABC):
	"""Protocol for NATS / message bus event publishing."""

	@abstractmethod
	async def publish(self, stream: str, event_type: str, payload: dict[str, Any]) -> None: ...


class GLAdapter(ABC):
	"""Protocol for posting journal entries to the General Ledger capability."""

	@abstractmethod
	async def post_journal(
		self,
		tenant_id: str,
		journal_type: str,
		account_id: str,
		amount: Decimal,
		currency: str,
		reference: str,
		description: str,
		transaction_id: str,
	) -> str:
		"""Returns journal_id."""
		...


# ---------------------------------------------------------------------------
# In-memory adapters (default — used in tests and standalone mode)
# ---------------------------------------------------------------------------

class InMemoryAccountRepository(AccountRepository):
	def __init__(self) -> None:
		self._store: dict[str, dict[str, Any]] = {}
		self._by_number: dict[str, str] = {}

	async def save(self, account: dict[str, Any]) -> None:
		self._store[account["id"]] = account
		self._by_number[account["account_number"]] = account["id"]

	async def get(self, account_id: str) -> dict[str, Any] | None:
		return self._store.get(account_id)

	async def get_by_number(self, account_number: str) -> dict[str, Any] | None:
		acct_id = self._by_number.get(account_number)
		if not acct_id:
			return None
		return self._store.get(acct_id)

	async def list_by_tenant(
		self,
		tenant_id: str,
		customer_id: str | None = None,
		status: str | None = None,
		account_type: str | None = None,
	) -> list[dict[str, Any]]:
		results = []
		for acc in self._store.values():
			if acc["tenant_id"] != tenant_id:
				continue
			if customer_id and acc["customer_id"] != customer_id:
				continue
			if status and acc["status"] != status:
				continue
			if account_type and acc["account_type"] != account_type:
				continue
			results.append(acc)
		return results

	async def delete(self, account_id: str) -> None:
		acc = self._store.pop(account_id, None)
		if acc:
			self._by_number.pop(acc.get("account_number", ""), None)


class InMemoryTransactionRepository(TransactionRepository):
	def __init__(self) -> None:
		self._store: dict[str, dict[str, Any]] = {}

	async def save(self, txn: dict[str, Any]) -> None:
		self._store[txn["id"]] = txn

	async def get(self, transaction_id: str) -> dict[str, Any] | None:
		return self._store.get(transaction_id)

	async def list_by_account(
		self,
		account_id: str,
		tenant_id: str,
		from_date: str | None = None,
		to_date: str | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		results = [
			t for t in self._store.values()
			if t["account_id"] == account_id and t["tenant_id"] == tenant_id
		]
		results.sort(key=lambda t: t["posted_at"], reverse=True)
		return results[offset: offset + limit]


class LoggingEventPublisher(EventPublisher):
	"""Logs events to stdout; swap for NATS publisher in production."""

	import logging
	_log = logging.getLogger("fin_acct.events")

	async def publish(self, stream: str, event_type: str, payload: dict[str, Any]) -> None:
		self._log.info("EVENT stream=%s type=%s payload=%s", stream, event_type, payload)


class NoOpGLAdapter(GLAdapter):
	"""No-op GL adapter; swap for real GLR integration in production."""

	async def post_journal(
		self,
		tenant_id: str,
		journal_type: str,
		account_id: str,
		amount: Decimal,
		currency: str,
		reference: str,
		description: str,
		transaction_id: str,
	) -> str:
		from ..__init__ import uuid7str  # noqa: F401
		try:
			from ..models import uuid7str as _uuid7str
		except ImportError:
			from uuid6 import uuid7
			def _uuid7str() -> str:
				return str(uuid7())
		return _uuid7str()
