"""Process-local API helpers for APG Digital Cards."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import CardService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import CardService  # type: ignore


SERVICE = CardService()


def service() -> CardService:
	return SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {"capability": contract["capability"], "display_name": contract["display_name"], "tenant_id": tenant_id, "route_count": len(contract["ui"]["routes"]), "rule_count": len(contract["rule_engine"]["rules"]), "card_count": summary["card_count"], "authorization_count": summary["authorization_count"], "streaming": summary["streaming"]}


def register_program(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_program(str(payload["program_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("name") or payload["program_id"]), str(payload.get("owner_id") or ""), str(payload.get("bin_range") or ""), str(payload.get("currency") or ""), str(payload.get("settlement_account") or ""), bool(payload.get("policy_attached", True)))


def onboard_cardholder(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.onboard_cardholder(str(payload["cardholder_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("customer_reference") or ""), str(payload.get("kyc_profile_id") or ""), str(payload.get("country") or ""), bool(payload.get("policy_attached", True)))


def issue_card(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.issue_card(str(payload["card_id"]), str(payload.get("tenant_id") or "default"), str(payload["program_id"]), str(payload["cardholder_id"]), str(payload.get("card_type") or "virtual"), str(payload.get("product") or "debit"), str(payload.get("wallet_reference") or ""), str(payload.get("funding_account") or ""), str(payload.get("consent_reference") or ""), str(payload.get("shipping_reference") or ""), bool(payload.get("policy_attached", True)))


def provision_token(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.provision_token(str(payload["token_id"]), str(payload.get("tenant_id") or "default"), str(payload["card_id"]), str(payload.get("token_type") or "wallet"), str(payload.get("token_reference") or ""), str(payload.get("key_domain_id") or ""), str(payload.get("device_or_merchant_reference") or ""))


def authorize_transaction(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.authorize_transaction(str(payload["authorization_id"]), str(payload.get("tenant_id") or "default"), str(payload["card_id"]), payload.get("amount", 0), str(payload.get("currency") or ""), str(payload.get("merchant_category") or ""), str(payload.get("fraud_reference") or ""), str(payload.get("aml_reference") or ""), str(payload.get("fraud_decision") or "clear"), str(payload.get("aml_result") or "clear"), bool(payload.get("limit_override", False)), str(payload.get("human_approval") or ""))


def file_dispute(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.file_dispute(str(payload["dispute_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("transaction_reference") or ""), str(payload.get("reason") or ""), list(payload.get("evidence_references") or []), str(payload.get("reviewer_id") or ""))


def register_card_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_card_agent(str(payload["agent_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("name") or payload["agent_id"]), str(payload.get("runtime") or "codex"), str(payload.get("role") or "card_ops_reviewer"), str(payload.get("scope") or "review card operations"))


def list_cards(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_cards(tenant_id)
