"""Dependency-light API helpers for the General Ledger capability."""

from __future__ import annotations

from typing import Any

try:
	from .service import GeneralLedgerService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import GeneralLedgerService  # type: ignore


_SERVICE = GeneralLedgerService()


def service() -> GeneralLedgerService:
	"""Return the process-local GLR service."""
	return _SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	return {"ok": True, "capability": "glr_general_ledger", "summary": _SERVICE.dashboard_summary(tenant_id)}


def create_account(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_account(
		payload.get("account_id", "account"),
		payload["tenant_id"],
		payload["code"],
		payload["name"],
		payload["account_type"],
		payload.get("parent_account_id"),
		payload.get("allow_posting", True),
		payload.get("currency", "USD"),
	)


def record_dimension(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_dimension(
		payload.get("dimension_id", "dimension"),
		payload["tenant_id"],
		payload["name"],
		payload["value"],
		payload["owner"],
	)


def open_period(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.open_period(
		payload.get("period_id", "period"),
		payload["tenant_id"],
		payload["name"],
		payload["fiscal_year"],
		payload["period_start"],
		payload["period_end"],
	)


def create_journal_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_journal_batch(
		payload.get("batch_id", "batch"),
		payload["tenant_id"],
		payload["period_id"],
		payload.get("source", "manual"),
		payload.get("currency", "USD"),
	)


def create_journal_entry(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_journal_entry(
		payload.get("journal_id", "journal"),
		payload["tenant_id"],
		payload["batch_id"],
		payload["description"],
		payload["lines"],
		payload.get("prepared_by", "system"),
		payload.get("exchange_rate"),
	)


def approve_journal(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.approve_journal(payload["journal_id"], payload["tenant_id"], payload["approved_by"])


def post_journal(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.post_journal(payload["journal_id"], payload["tenant_id"], payload["posted_by"], payload["idempotency_key"])


def reverse_journal(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.reverse_journal(
		payload.get("reversal_id", "reversal"),
		payload["tenant_id"],
		payload["journal_id"],
		payload["reason"],
		payload["approved_by"],
	)


def create_allocation(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_allocation(
		payload.get("allocation_id", "allocation"),
		payload["tenant_id"],
		payload["source_account_id"],
		payload["target_account_ids"],
		payload["basis"],
		payload.get("reviewed_by"),
	)


def register_glr_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_glr_agent(
		payload["tenant_id"],
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("scope", "review ledger activity"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	"""Generic composition helper used by APG package smoke tests."""
	return create_account({
		"tenant_id": payload["tenant_id"],
		"account_id": payload.get("account_id", "api-account"),
		"code": payload.get("code", "1000"),
		"name": payload.get("name", "API Account"),
		"account_type": payload.get("account_type", "asset"),
	})


def list_records(collection: str, tenant_id: str = "default") -> list[dict[str, Any]]:
	return _SERVICE.list_records(collection, tenant_id)
