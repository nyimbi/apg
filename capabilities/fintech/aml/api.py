"""Process-local API helpers for APG Anti Money Laundering."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import AntiMoneyLaunderingService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import AntiMoneyLaunderingService  # type: ignore


SERVICE = AntiMoneyLaunderingService()


def service() -> AntiMoneyLaunderingService:
	return SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {"capability": contract["capability"], "display_name": contract["display_name"], "tenant_id": tenant_id, "route_count": len(contract["ui"]["routes"]), "rule_count": len(contract["rule_engine"]["rules"]), "alert_count": summary["alert_count"], "case_count": summary["case_count"], "streaming": summary["streaming"]}


def monitor_transaction(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.monitor_transaction(str(payload["transaction_id"]), str(payload.get("tenant_id") or "default"), str(payload["subject_reference"]), str(payload.get("kyc_profile_id") or ""), payload.get("amount", 0), str(payload.get("currency") or ""), str(payload.get("source_capability") or "fintech_payments"), str(payload.get("source_reference") or ""), payload.get("risk_score", 0), bool(payload.get("sanctions_hit", False)), bool(payload.get("velocity_indicator", False)), str(payload.get("review_id") or ""), bool(payload.get("policy_attached", True)))


def create_alert(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_alert(str(payload["alert_id"]), str(payload.get("tenant_id") or "default"), str(payload["alert_type"]), str(payload.get("severity") or "medium"), str(payload["subject_reference"]), list(payload.get("evidence_references") or []))


def create_alert_from_transaction(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_alert_from_transaction(str(payload["alert_id"]), str(payload.get("tenant_id") or "default"), str(payload["transaction_id"]), payload.get("alert_type"))


def triage_alert(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.triage_alert(str(payload["alert_id"]), str(payload.get("tenant_id") or "default"), str(payload["action"]), str(payload.get("disposition") or ""), str(payload.get("reviewer_id") or ""))


def open_case(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.open_case(str(payload["case_id"]), str(payload.get("tenant_id") or "default"), str(payload["alert_id"]), str(payload.get("case_type") or "transaction_monitoring"), str(payload.get("investigator_id") or ""), list(payload.get("evidence_references") or []))


def draft_sar(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.draft_sar(str(payload["sar_id"]), str(payload.get("tenant_id") or "default"), str(payload["case_id"]), str(payload["subject_reference"]), str(payload.get("jurisdiction") or ""), str(payload.get("narrative") or ""), list(payload.get("evidence_references") or []), str(payload.get("approved_by") or ""))


def register_aml_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_aml_agent(str(payload["agent_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("name") or payload["agent_id"]), str(payload.get("runtime") or "codex"), str(payload.get("role") or "aml_ops_reviewer"), str(payload.get("scope") or "triage AML alerts"))


def list_alerts(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_alerts(tenant_id)
