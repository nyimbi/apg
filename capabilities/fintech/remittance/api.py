"""Process-local API helpers for APG Cross-Border Remittance."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import RemittanceService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import RemittanceService  # type: ignore


SERVICE = RemittanceService()


def service() -> RemittanceService:
	return SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {"capability": contract["capability"], "display_name": contract["display_name"], "tenant_id": tenant_id, "route_count": len(contract["ui"]["routes"]), "rule_count": len(contract["rule_engine"]["rules"]), "transfer_count": summary["transfer_count"], "refund_count": summary["refund_count"], "streaming": summary["streaming"]}


def create_quote(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_quote(str(payload["quote_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("source_country") or ""), str(payload.get("destination_country") or ""), str(payload.get("source_currency") or ""), str(payload.get("destination_currency") or ""), payload.get("send_amount", 0), payload.get("fx_rate", 0), payload.get("fee_amount", 0), str(payload.get("expiry") or ""), bool(payload.get("policy_attached", True)))


def create_transfer(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_transfer(str(payload["transfer_id"]), str(payload.get("tenant_id") or "default"), str(payload["quote_id"]), str(payload.get("sender_reference") or ""), str(payload.get("beneficiary_reference") or ""), str(payload.get("sender_kyc_id") or ""), str(payload.get("beneficiary_kyc_id") or ""), str(payload.get("funding_reference") or ""), str(payload.get("payout_method") or "mobile_money"), str(payload.get("purpose_code") or "family_support"), str(payload.get("source_of_funds") or ""), str(payload.get("aml_screen_id") or ""), str(payload.get("fraud_decision") or "clear"), bool(payload.get("aml_review", False)), bool(payload.get("sanctions_hit", False)), str(payload.get("human_approval") or ""), bool(payload.get("policy_attached", True)))


def release_payout(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.release_payout(str(payload["transfer_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("provider_receipt") or ""), str(payload.get("settlement_reference") or ""))


def file_refund(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.file_refund(str(payload["refund_id"]), str(payload.get("tenant_id") or "default"), str(payload["transfer_id"]), str(payload.get("reason") or ""), str(payload.get("reviewer_id") or ""))


def register_remittance_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_remittance_agent(str(payload["agent_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("name") or payload["agent_id"]), str(payload.get("runtime") or "codex"), str(payload.get("role") or "remittance_ops_reviewer"), str(payload.get("scope") or "review remittance operations"))


def list_transfers(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_transfers(tenant_id)
