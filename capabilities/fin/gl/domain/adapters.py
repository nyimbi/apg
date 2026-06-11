"""GL — Domain adapters (NATS routing pattern).

Protocol-based ports for auth, audit, notify, and event publishing.
Null implementations allow standalone / test operation with no external
dependencies.

NATS subjects consumed by GLService:
  fin.gl.journal.posted
  fin.gl.journal.reversed
  fin.gl.period.opened
  fin.gl.period.closed
  fin.gl.account.created
  fin.gl.fx.revalued

© 2026 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

import json
import logging
from datetime import date
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

_log = logging.getLogger(__name__)


# ── Auth adapter ──────────────────────────────────────────────────────────────

@runtime_checkable
class AuthAdapter(Protocol):
	async def verify_token(self, token: str) -> dict[str, Any]: ...
	async def check_permission(self, user_id: str, permission: str, resource: str | None = None) -> bool: ...
	async def get_current_user(self, token: str) -> dict[str, Any]: ...


class NullAuthAdapter:
	"""Standalone fallback — all tokens accepted, all permissions granted."""

	async def verify_token(self, token: str) -> dict[str, Any]:
		return {"user_id": token or "anonymous", "tenant_id": "default", "roles": ["admin"]}

	async def check_permission(self, user_id: str, permission: str, resource: str | None = None) -> bool:
		return True

	async def get_current_user(self, token: str) -> dict[str, Any]:
		return {"id": token or "anonymous", "name": "Standalone User", "roles": ["admin"]}


# ── Audit adapter ─────────────────────────────────────────────────────────────

@runtime_checkable
class AuditAdapter(Protocol):
	async def log_event(
		self,
		event_type: str,
		actor_id: str,
		tenant_id: str,
		resource_id: str,
		details: dict[str, Any],
	) -> None: ...


class NullAuditAdapter:
	async def log_event(
		self,
		event_type: str,
		actor_id: str,
		tenant_id: str,
		resource_id: str,
		details: dict[str, Any],
	) -> None:
		_log.debug(
			"[audit] %s actor=%s tenant=%s resource=%s",
			event_type, actor_id, tenant_id, resource_id,
		)


# ── Notify adapter ────────────────────────────────────────────────────────────

@runtime_checkable
class NotifyAdapter(Protocol):
	async def send(
		self,
		recipient: str,
		channel: str,
		subject: str,
		body: str,
		metadata: dict[str, Any] | None = None,
	) -> None: ...


class NullNotifyAdapter:
	async def send(
		self,
		recipient: str,
		channel: str,
		subject: str,
		body: str,
		metadata: dict[str, Any] | None = None,
	) -> None:
		_log.debug("[notify] to=%s channel=%s subject=%s", recipient, channel, subject)


# ── Event publisher (NATS) ────────────────────────────────────────────────────

@runtime_checkable
class EventPublisher(Protocol):
	async def publish(self, subject: str, payload: dict[str, Any]) -> None: ...


class NullEventPublisher:
	"""Logs events locally; replace with NATS JetStream publisher in production."""

	async def publish(self, subject: str, payload: dict[str, Any]) -> None:
		_log.info("[event] subject=%s payload=%s", subject, json.dumps(payload, default=str))


# ── Repository protocols ──────────────────────────────────────────────────────

class AccountRepository(Protocol):
	async def get(self, tenant_id: str, code: str) -> dict[str, Any] | None: ...
	async def save(self, account: dict[str, Any]) -> None: ...
	async def list(
		self,
		tenant_id: str,
		account_type: str | None = None,
		search: str | None = None,
	) -> list[dict[str, Any]]: ...
	async def delete(self, tenant_id: str, code: str) -> None: ...


class JournalRepository(Protocol):
	async def get(self, journal_id: str) -> dict[str, Any] | None: ...
	async def save(self, journal: dict[str, Any]) -> None: ...
	async def list(
		self,
		tenant_id: str,
		account_code: str | None = None,
		from_date: date | None = None,
		to_date: date | None = None,
		limit: int = 50,
	) -> list[dict[str, Any]]: ...
	async def list_by_period(self, tenant_id: str, period_id: str) -> list[dict[str, Any]]: ...


class PeriodRepository(Protocol):
	async def get(self, period_id: str) -> dict[str, Any] | None: ...
	async def get_by_year_month(self, tenant_id: str, year: int, month: int) -> dict[str, Any] | None: ...
	async def save(self, period: dict[str, Any]) -> None: ...
	async def list(self, tenant_id: str) -> list[dict[str, Any]]: ...


# ── In-memory repository implementations ─────────────────────────────────────

class InMemoryAccountRepository:
	"""Thread-unsafe in-memory store for tests and standalone use."""

	def __init__(self) -> None:
		self._store: dict[str, dict[str, Any]] = {}  # key: f"{tenant_id}:{code}"

	async def get(self, tenant_id: str, code: str) -> dict[str, Any] | None:
		return self._store.get(f"{tenant_id}:{code}")

	async def save(self, account: dict[str, Any]) -> None:
		key = f"{account['tenant_id']}:{account['code']}"
		self._store[key] = dict(account)

	async def list(
		self,
		tenant_id: str,
		account_type: str | None = None,
		search: str | None = None,
	) -> list[dict[str, Any]]:
		results = [v for v in self._store.values() if v["tenant_id"] == tenant_id]
		if account_type:
			results = [r for r in results if r["account_type"] == account_type]
		if search:
			s = search.lower()
			results = [r for r in results if s in r["code"].lower() or s in r["name"].lower()]
		return sorted(results, key=lambda r: r["code"])

	async def delete(self, tenant_id: str, code: str) -> None:
		self._store.pop(f"{tenant_id}:{code}", None)


class InMemoryJournalRepository:
	"""In-memory journal store keyed by journal ID."""

	def __init__(self) -> None:
		self._store: dict[str, dict[str, Any]] = {}

	async def get(self, journal_id: str) -> dict[str, Any] | None:
		return self._store.get(journal_id)

	async def save(self, journal: dict[str, Any]) -> None:
		self._store[journal["id"]] = dict(journal)

	async def list(
		self,
		tenant_id: str,
		account_code: str | None = None,
		from_date: date | None = None,
		to_date: date | None = None,
		limit: int = 50,
	) -> list[dict[str, Any]]:
		results = [v for v in self._store.values() if v["tenant_id"] == tenant_id]
		if account_code:
			results = [
				r for r in results
				if any(ln["account_code"] == account_code for ln in r.get("lines", []))
			]
		if from_date:
			results = [r for r in results if str(r["posting_date"]) >= str(from_date)]
		if to_date:
			results = [r for r in results if str(r["posting_date"]) <= str(to_date)]
		results.sort(key=lambda r: str(r["posting_date"]), reverse=True)
		return results[:limit]

	async def list_by_period(self, tenant_id: str, period_id: str) -> list[dict[str, Any]]:
		return [
			v for v in self._store.values()
			if v["tenant_id"] == tenant_id and v["period_id"] == period_id
		]


class InMemoryPeriodRepository:
	"""In-memory period store keyed by period ID."""

	def __init__(self) -> None:
		self._store: dict[str, dict[str, Any]] = {}

	async def get(self, period_id: str) -> dict[str, Any] | None:
		return self._store.get(period_id)

	async def get_by_year_month(self, tenant_id: str, year: int, month: int) -> dict[str, Any] | None:
		for v in self._store.values():
			if v["tenant_id"] == tenant_id and v["year"] == year and v["month"] == month:
				return v
		return None

	async def save(self, period: dict[str, Any]) -> None:
		self._store[period["id"]] = dict(period)

	async def list(self, tenant_id: str) -> list[dict[str, Any]]:
		results = [v for v in self._store.values() if v["tenant_id"] == tenant_id]
		return sorted(results, key=lambda r: (r["year"], r["month"]))


# ── Batch idempotency store ───────────────────────────────────────────────────

class InMemoryBatchStore:
	"""Tracks processed batch IDs to enforce exactly-once batch posting."""

	def __init__(self) -> None:
		self._store: dict[str, list[str]] = {}  # batch_id -> list of journal IDs

	async def get(self, batch_id: str) -> list[str] | None:
		return self._store.get(batch_id)

	async def save(self, batch_id: str, journal_ids: list[str]) -> None:
		self._store[batch_id] = list(journal_ids)
