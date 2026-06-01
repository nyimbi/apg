"""Dependency-light API helpers for APG Agency Banking."""

from __future__ import annotations

from typing import Any

try:
	from .service import AgencyBankingService
except ImportError:  # pragma: no cover
	from service import AgencyBankingService  # type: ignore


_SERVICE = AgencyBankingService()


def service() -> AgencyBankingService:
	return _SERVICE


def register_program(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_program(payload["program_id"], payload["tenant_id"], payload["name"], payload["owner_id"], payload["country"], payload["currency"], payload["settlement_model"], list(payload["services"]), payload.get("policy_attached", True))


def onboard_outlet(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.onboard_outlet(payload["outlet_id"], payload["tenant_id"], payload["program_id"], payload["name"], payload["outlet_type"], payload["country"], payload["license_reference"], payload["location_reference"], payload["security_plan_reference"], payload["primary_channel"], payload["initial_float"], payload.get("policy_attached", True))


def accredit_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.accredit_agent(payload["agent_id"], payload["tenant_id"], payload["outlet_id"], payload["name"], payload["identity_reference"], payload["training_reference"], payload["background_check_reference"], payload.get("policy_attached", True))


def open_float_account(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.open_float_account(payload["float_account_id"], payload["tenant_id"], payload["outlet_id"], payload["currency"], payload["opening_balance"], payload["ledger_reference"], payload.get("policy_attached", True))


def onboard_customer(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.onboard_customer(payload["customer_id"], payload["tenant_id"], payload["customer_reference"], payload["tier"], payload["kyc_reference"], payload["consent_reference"], payload["aml_reference"], payload["fraud_reference"], payload.get("policy_attached", True))


def record_transaction(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_transaction(payload["transaction_id"], payload["tenant_id"], payload["outlet_id"], payload["agent_id"], payload["customer_id"], payload["float_account_id"], payload["service"], payload["amount"], payload["currency"], payload["channel"], payload["customer_reference"], payload["risk_reference"], payload.get("human_approval", ""), payload.get("policy_attached", True))


def record_cash_movement(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_cash_movement(payload["movement_id"], payload["tenant_id"], payload["outlet_id"], payload["movement_type"], payload["amount"], payload["currency"], payload["custodian_reference"], payload.get("human_approval", ""))


def settle_commission(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.settle_commission(payload["settlement_id"], payload["tenant_id"], payload["outlet_id"], payload["period"], payload["amount"], payload["currency"], payload["reconciliation_reference"], payload["payment_reference"])


def open_dispute(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.open_dispute(payload["dispute_id"], payload["tenant_id"], payload["transaction_id"], payload["reason"], payload["reviewer_id"], list(payload["evidence_references"]))


def record_supervision_visit(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_supervision_visit(payload["visit_id"], payload["tenant_id"], payload["outlet_id"], payload["supervisor_id"], payload["outcome"], list(payload["evidence_references"]), list(payload.get("findings", [])), payload.get("remediation_plan_reference", ""))


def register_agency_ai_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_agency_ai_agent(payload["agent_id"], payload["tenant_id"], payload["name"], payload["runtime"], payload["role"], payload.get("scope", "agency review"))


def dashboard(tenant_id: str) -> dict[str, Any]:
	return _SERVICE.dashboard_summary(tenant_id)
