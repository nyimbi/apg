"""Process-local API helpers for APG Digital Neobanking."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import NeobankingService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import NeobankingService  # type: ignore


SERVICE = NeobankingService()


def service() -> NeobankingService:
	return SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {"capability": contract["capability"], "display_name": contract["display_name"], "tenant_id": tenant_id, "route_count": len(contract["ui"]["routes"]), "rule_count": len(contract["rule_engine"]["rules"]), "account_count": summary["account_count"], "transaction_count": summary["transaction_count"], "streaming": summary["streaming"]}


def register_program(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_program(str(payload["program_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("name") or payload["program_id"]), str(payload.get("owner_id") or ""), str(payload.get("country") or ""), str(payload.get("base_currency") or ""), str(payload.get("settlement_account") or ""), bool(payload.get("policy_attached", True)))


def onboard_customer(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.onboard_customer(str(payload["customer_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("customer_reference") or ""), str(payload.get("kyc_profile_id") or ""), str(payload.get("country") or ""), str(payload.get("consent_reference") or ""), str(payload.get("aml_reference") or ""), str(payload.get("fraud_reference") or ""), bool(payload.get("policy_attached", True)))


def open_account(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.open_account(str(payload["account_id"]), str(payload.get("tenant_id") or "default"), str(payload["program_id"]), str(payload["customer_id"]), str(payload.get("account_type") or "current"), str(payload.get("currency") or ""), payload.get("initial_balance", 0), bool(payload.get("policy_attached", True)))


def link_payment_rail(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.link_payment_rail(str(payload["link_id"]), str(payload.get("tenant_id") or "default"), str(payload["account_id"]), str(payload.get("rail") or "bank_transfer"), str(payload.get("provider_reference") or ""), str(payload.get("wallet_reference") or ""), str(payload.get("card_reference") or ""))


def post_transaction(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.post_transaction(str(payload["transaction_id"]), str(payload.get("tenant_id") or "default"), str(payload["account_id"]), str(payload.get("kind") or "deposit"), payload.get("amount", 0), str(payload.get("currency") or ""), str(payload.get("reference") or ""), str(payload.get("risk_reference") or ""), str(payload.get("human_approval") or ""))


def create_savings_pot(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_savings_pot(str(payload["pot_id"]), str(payload.get("tenant_id") or "default"), str(payload["account_id"]), str(payload.get("name") or ""), payload.get("target_amount", 0))


def issue_statement(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.issue_statement(str(payload["statement_id"]), str(payload.get("tenant_id") or "default"), str(payload["account_id"]), str(payload.get("period_start") or ""), str(payload.get("period_end") or ""))


def open_service_case(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.open_service_case(str(payload["case_id"]), str(payload.get("tenant_id") or "default"), str(payload["customer_id"]), str(payload["account_id"]), str(payload.get("reason") or ""), str(payload.get("reviewer_id") or ""), list(payload.get("evidence_references") or []))


def register_neobanking_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_neobanking_agent(str(payload["agent_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("name") or payload["agent_id"]), str(payload.get("runtime") or "codex"), str(payload.get("role") or "neobank_ops_reviewer"), str(payload.get("scope") or "review neobank operations"))


def list_accounts(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_accounts(tenant_id)
