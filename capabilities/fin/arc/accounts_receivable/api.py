"""Dependency-light API helpers for Accounts Receivable."""

from __future__ import annotations

from typing import Any

try:
	from .service import AccountsReceivableService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import AccountsReceivableService  # type: ignore


_SERVICE = AccountsReceivableService()


def service() -> AccountsReceivableService:
	"""Return the process-local ARC service."""
	return _SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	return {"ok": True, "capability": "arc_accounts_receivable", "summary": _SERVICE.dashboard_summary(tenant_id)}


def create_customer(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_customer(
		payload.get("customer_id", "customer"),
		payload["tenant_id"],
		payload["customer_code"],
		payload["legal_name"],
		payload.get("customer_type", "business"),
		payload.get("currency", "USD"),
	)


def assess_credit(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.assess_credit(
		payload.get("assessment_id", "credit"),
		payload["tenant_id"],
		payload["customer_id"],
		payload["credit_limit"],
		payload["credit_score"],
		payload.get("reviewed_by"),
		payload.get("credit_hold", False),
	)


def create_invoice(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_invoice(
		payload.get("invoice_id", "invoice"),
		payload["tenant_id"],
		payload["customer_id"],
		payload["invoice_number"],
		payload["invoice_date"],
		payload["due_date"],
		payload["lines"],
		payload.get("currency", "USD"),
	)


def issue_invoice(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.issue_invoice(payload["invoice_id"], payload["tenant_id"], payload["approved_by"])


def record_payment(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_payment(
		payload.get("payment_id", "payment"),
		payload["tenant_id"],
		payload["customer_id"],
		payload["payment_reference"],
		payload["payment_date"],
		payload["amount"],
		payload.get("method", "bank_transfer"),
		payload["cash_account_id"],
	)


def apply_cash(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.apply_cash(
		payload.get("application_id", "application"),
		payload["tenant_id"],
		payload["payment_id"],
		payload["invoice_id"],
		payload["allocation_amount"],
		payload.get("reviewed_by"),
	)


def record_collection_activity(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_collection_activity(
		payload.get("activity_id", "activity"),
		payload["tenant_id"],
		payload["invoice_id"],
		payload["contact_method"],
		payload.get("priority", "normal"),
		payload.get("outcome", "queued"),
	)


def open_dispute(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.open_dispute(
		payload.get("dispute_id", "dispute"),
		payload["tenant_id"],
		payload["invoice_id"],
		payload["reason"],
		payload["owner"],
	)


def resolve_dispute(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.resolve_dispute(payload["dispute_id"], payload["tenant_id"], payload["resolution"], payload["reviewed_by"])


def register_arc_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_arc_agent(
		payload["tenant_id"],
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("scope", "review receivables operations"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	"""Generic composition helper used by APG package smoke tests."""
	return create_customer({
		"tenant_id": payload["tenant_id"],
		"customer_id": payload.get("customer_id", "api-customer"),
		"customer_code": payload.get("customer_code", "APICUST"),
		"legal_name": payload.get("legal_name", "API Customer"),
		"customer_type": payload.get("customer_type", "business"),
	})


def list_records(collection: str, tenant_id: str = "default") -> list[dict[str, Any]]:
	return _SERVICE.list_records(collection, tenant_id)
