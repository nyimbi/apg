"""Dependency-light API helpers for APG Buy Now Pay Later."""

from __future__ import annotations

from typing import Any

try:
	from .service import BNPLService
except ImportError:  # pragma: no cover
	from service import BNPLService  # type: ignore


_SERVICE = BNPLService()


def service() -> BNPLService:
	return _SERVICE


def register_merchant_program(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_merchant_program(payload["program_id"], payload["tenant_id"], payload["name"], payload["owner_id"], payload["country"], payload["currency"], payload["settlement_policy_reference"], payload["fee_disclosure_reference"], payload["max_installments"], payload.get("policy_attached", True))


def onboard_consumer(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.onboard_consumer(payload["consumer_id"], payload["tenant_id"], payload["customer_reference"], payload["kyc_profile_id"], payload["country"], payload["consent_reference"], payload["aml_reference"], payload["fraud_reference"], payload.get("policy_attached", True))


def register_merchant(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_merchant(payload["merchant_id"], payload["tenant_id"], payload["program_id"], payload["legal_entity_reference"], payload["category"], payload["country"], payload["risk_tier"], payload["settlement_account"], payload.get("policy_attached", True))


def create_checkout_session(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_checkout_session(payload["checkout_id"], payload["tenant_id"], payload["merchant_id"], payload["consumer_id"], payload["channel"], payload["category"], payload["amount"], payload["currency"], payload["payment_reference"], payload["fraud_reference"], payload["aml_reference"], payload["consent_reference"], payload.get("human_review", ""), payload.get("policy_attached", True))


def record_affordability_decision(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_affordability_decision(payload["decision_id"], payload["tenant_id"], payload["checkout_id"], payload["score"], payload["decision"], list(payload["evidence_references"]), payload["human_approval"], payload.get("adverse_reason", ""))


def create_bnpl_plan(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_bnpl_plan(payload["plan_id"], payload["tenant_id"], payload["checkout_id"], payload["affordability_id"], payload["plan_type"], payload["principal"], payload["currency"], payload["term_days"], payload.get("down_payment", 0), payload["fee_disclosure_reference"], payload["customer_acceptance_reference"], payload.get("policy_attached", True))


def schedule_installment(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.schedule_installment(payload["schedule_id"], payload["tenant_id"], payload["plan_id"], payload["due_amount"], payload["due_date"], payload.get("status", "scheduled"), payload.get("sequence", 1))


def record_merchant_settlement(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_merchant_settlement(payload["settlement_id"], payload["tenant_id"], payload["merchant_id"], payload["plan_id"], payload["gross_amount"], payload["net_amount"], payload["status"], payload["reconciliation_reference"], payload["payment_rail_reference"], payload.get("human_approval", ""))


def open_bnpl_dispute(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.open_bnpl_dispute(payload["dispute_id"], payload["tenant_id"], payload["plan_id"], payload["reason"], payload["reviewer_id"], list(payload["evidence_references"]))


def register_bnpl_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_bnpl_agent(payload["agent_id"], payload["tenant_id"], payload["name"], payload["runtime"], payload["role"], payload.get("scope", "bnpl review"))


def dashboard(tenant_id: str) -> dict[str, Any]:
	return _SERVICE.dashboard_summary(tenant_id)
