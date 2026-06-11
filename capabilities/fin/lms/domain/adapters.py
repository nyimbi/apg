"""LMS — Domain adapters.

Protocol-based adapters for auth, audit, notify, and GL posting.
Null implementations enable standalone/test operation without any
external APG capability installed.

Usage (standalone)::

	svc = LoanManagementService()  # null adapters

Usage (platform)::

	from apg_common_auth import AuthService
	svc = LoanManagementService(auth=AuthService.from_env())

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable


# ─────────────────────────────────────────────────────────────
# Auth adapter
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Audit adapter
# ─────────────────────────────────────────────────────────────

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
		print(json.dumps({
			"event_type": event_type,
			"actor_id": actor_id,
			"tenant_id": tenant_id,
			"resource_id": resource_id,
			"details": details,
		}, default=str))


# ─────────────────────────────────────────────────────────────
# Notify adapter
# ─────────────────────────────────────────────────────────────

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
		print(f"[NOTIFY] {channel}→{recipient}: {subject}")


# ─────────────────────────────────────────────────────────────
# GL posting adapter
# ─────────────────────────────────────────────────────────────

@runtime_checkable
class GLAdapter(Protocol):
	"""Post double-entry journal lines to the General Ledger capability."""

	async def post_journal(
		self,
		tenant_id: str,
		journal_type: str,
		description: str,
		lines: list[dict[str, Any]],
		ref: str | None = None,
		posting_date: str | None = None,
	) -> str:
		"""Return GL journal entry ID."""
		...


class NullGLAdapter:
	"""Prints GL lines; returns a dummy ID.  Used for standalone/test mode."""

	def __init__(self) -> None:
		self._counter = 0

	async def post_journal(
		self,
		tenant_id: str,
		journal_type: str,
		description: str,
		lines: list[dict[str, Any]],
		ref: str | None = None,
		posting_date: str | None = None,
	) -> str:
		self._counter += 1
		entry_id = f"GL-NULL-{self._counter:06d}"
		print(json.dumps({
			"gl_entry_id": entry_id,
			"tenant_id": tenant_id,
			"journal_type": journal_type,
			"description": description,
			"posting_date": posting_date,
			"lines": lines,
		}, default=str, indent=2))
		return entry_id


# ─────────────────────────────────────────────────────────────
# Loan repository adapter
# ─────────────────────────────────────────────────────────────

@runtime_checkable
class LoanRepository(Protocol):
	"""Persistence protocol for Loan records."""

	async def save(self, loan: dict[str, Any]) -> None: ...
	async def get(self, loan_id: str, tenant_id: str) -> dict[str, Any] | None: ...
	async def list_by_tenant(
		self,
		tenant_id: str,
		customer_id: str | None = None,
		status: str | None = None,
		days_past_due_min: int | None = None,
	) -> list[dict[str, Any]]: ...
	async def delete(self, loan_id: str) -> None: ...


class InMemoryLoanRepository:
	"""Thread-unsafe in-memory store for testing/standalone use."""

	def __init__(self) -> None:
		self._store: dict[str, dict[str, Any]] = {}

	async def save(self, loan: dict[str, Any]) -> None:
		self._store[loan["id"]] = loan

	async def get(self, loan_id: str, tenant_id: str) -> dict[str, Any] | None:
		rec = self._store.get(loan_id)
		if rec and rec.get("tenant_id") == tenant_id:
			return rec
		return None

	async def list_by_tenant(
		self,
		tenant_id: str,
		customer_id: str | None = None,
		status: str | None = None,
		days_past_due_min: int | None = None,
	) -> list[dict[str, Any]]:
		results = [v for v in self._store.values() if v.get("tenant_id") == tenant_id]
		if customer_id:
			results = [r for r in results if r.get("customer_id") == customer_id]
		if status:
			results = [r for r in results if r.get("status") == status]
		if days_past_due_min is not None:
			results = [r for r in results if r.get("days_past_due", 0) >= days_past_due_min]
		return results

	async def delete(self, loan_id: str) -> None:
		self._store.pop(loan_id, None)


# ─────────────────────────────────────────────────────────────
# Schedule / repayment repository adapters
# ─────────────────────────────────────────────────────────────

@runtime_checkable
class ScheduleRepository(Protocol):
	async def save_installments(self, loan_id: str, installments: list[dict[str, Any]]) -> None: ...
	async def get_installments(self, loan_id: str) -> list[dict[str, Any]]: ...
	async def update_installment(self, loan_id: str, installment_no: int, updates: dict[str, Any]) -> None: ...


class InMemoryScheduleRepository:
	def __init__(self) -> None:
		self._store: dict[str, list[dict[str, Any]]] = {}

	async def save_installments(self, loan_id: str, installments: list[dict[str, Any]]) -> None:
		self._store[loan_id] = installments

	async def get_installments(self, loan_id: str) -> list[dict[str, Any]]:
		return list(self._store.get(loan_id, []))

	async def update_installment(self, loan_id: str, installment_no: int, updates: dict[str, Any]) -> None:
		for inst in self._store.get(loan_id, []):
			if inst.get("installment_no") == installment_no:
				inst.update(updates)
				return


@runtime_checkable
class RepaymentRepository(Protocol):
	async def save(self, repayment: dict[str, Any]) -> None: ...
	async def list_by_loan(self, loan_id: str, tenant_id: str) -> list[dict[str, Any]]: ...


class InMemoryRepaymentRepository:
	def __init__(self) -> None:
		self._store: list[dict[str, Any]] = []

	async def save(self, repayment: dict[str, Any]) -> None:
		self._store.append(repayment)

	async def list_by_loan(self, loan_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [r for r in self._store if r.get("loan_id") == loan_id and r.get("tenant_id") == tenant_id]


# ─────────────────────────────────────────────────────────────
# GL entry / provision / restructure / moratorium / write-off
# ─────────────────────────────────────────────────────────────

class InMemoryGLEntryStore:
	def __init__(self) -> None:
		self._store: list[dict[str, Any]] = []

	async def save(self, entry: dict[str, Any]) -> None:
		self._store.append(entry)

	async def list_by_loan(self, loan_id: str) -> list[dict[str, Any]]:
		return [e for e in self._store if e.get("loan_id") == loan_id]

	async def list_by_tenant(self, tenant_id: str) -> list[dict[str, Any]]:
		return [e for e in self._store if e.get("tenant_id") == tenant_id]


class InMemoryEventStore:
	"""Generic store for restructures, moratoriums, write-offs, recoveries, provisions, notices."""
	def __init__(self) -> None:
		self._store: list[dict[str, Any]] = []

	async def save(self, record: dict[str, Any]) -> None:
		self._store.append(record)

	async def list_by_loan(self, loan_id: str) -> list[dict[str, Any]]:
		return [r for r in self._store if r.get("loan_id") == loan_id]

	async def list_by_tenant(self, tenant_id: str) -> list[dict[str, Any]]:
		return [r for r in self._store if r.get("tenant_id") == tenant_id]

	async def get_latest_by_type(self, loan_id: str, record_type: str) -> dict[str, Any] | None:
		matches = [r for r in self._store if r.get("loan_id") == loan_id and r.get("_type") == record_type]
		return matches[-1] if matches else None
