"""Dependency-light API helpers for APG accounts payable."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import AccountsPayableService
except ImportError:
	from capability_contract import get_capability_contract
	from service import AccountsPayableService


_SERVICE = AccountsPayableService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"ok": True,
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"streaming": contract["streaming"],
		"summary": _SERVICE.dashboard_summary(tenant_id),
	}


def register_vendor(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_vendor(
		payload["vendor_id"],
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["owner"],
		payload["tax_profile"],
		payload["payment_method"],
		payload.get("bank_change", False),
		payload.get("bank_reviewed_by"),
	)


def record_invoice(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_invoice(
		payload["invoice_id"],
		payload.get("tenant_id", "default"),
		payload["vendor_record_id"],
		payload["invoice_number"],
		payload["amount"],
		payload["currency"],
		payload["document_reference"],
		payload.get("duplicate_detected", False),
		payload.get("duplicate_reviewed_by"),
	)


def match_invoice(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.match_invoice(
		payload.get("tenant_id", "default"),
		payload["invoice_record_id"],
		payload.get("po_backed", False),
		payload.get("receipt_reference"),
		payload.get("variance_rate", 0),
		payload.get("variance_reviewed_by"),
	)


def approve_invoice(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.approve_invoice(
		payload.get("tenant_id", "default"),
		payload["invoice_record_id"],
		payload["approved_by"],
		payload["requested_by"],
		payload.get("approval_recorded", True),
	)


def schedule_payment(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.schedule_payment(
		payload["payment_id"],
		payload.get("tenant_id", "default"),
		payload["invoice_record_id"],
		payload["amount"],
		payload["cash_account"],
		payload["scheduled_date"],
	)


def release_payment_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.release_payment_batch(
		payload["batch_id"],
		payload.get("tenant_id", "default"),
		payload["payment_record_ids"],
		payload["reviewed_by"],
	)


def record_expense_report(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_expense_report(
		payload["report_id"],
		payload.get("tenant_id", "default"),
		payload["employee_id"],
		payload["amount"],
		payload["receipt_reference"],
		payload.get("policy_exception", False),
		payload.get("policy_reviewed_by"),
	)


def register_ap_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_ap_agent(
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("instructions", ""),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_record(payload)


def list_records(tenant_id: str = "default") -> list[dict[str, Any]]:
	return _SERVICE.list_records(tenant_id)


def service() -> AccountsPayableService:
	return _SERVICE
