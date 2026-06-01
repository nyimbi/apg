"""Process-local API helpers for APG Fraud Detection."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import FraudDetectionService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import FraudDetectionService  # type: ignore


SERVICE = FraudDetectionService()


def service() -> FraudDetectionService:
	return SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {"capability": contract["capability"], "display_name": contract["display_name"], "tenant_id": tenant_id, "route_count": len(contract["ui"]["routes"]), "rule_count": len(contract["rule_engine"]["rules"]), "signal_count": summary["signal_count"], "case_count": summary["case_count"], "streaming": summary["streaming"]}


def score_signal(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.score_signal(str(payload["signal_id"]), str(payload.get("tenant_id") or "default"), str(payload["subject_reference"]), str(payload.get("kyc_profile_id") or ""), str(payload.get("signal_type") or "payment"), str(payload.get("channel") or "api"), str(payload.get("source_reference") or ""), payload.get("amount", 0), str(payload.get("currency") or ""), payload.get("risk_score", 0), bool(payload.get("velocity_indicator", False)), bool(payload.get("device_anomaly", False)), bool(payload.get("geo_anomaly", False)), bool(payload.get("aml_alert_present", False)), bool(payload.get("chargeback_signal", False)), bool(payload.get("account_takeover_indicator", False)), list(payload.get("evidence_references") or []), str(payload.get("review_id") or ""), bool(payload.get("policy_attached", True)))


def record_decision(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_decision(str(payload["decision_id"]), str(payload.get("tenant_id") or "default"), str(payload["signal_id"]), str(payload.get("decision") or "review"), str(payload.get("reason") or ""), str(payload.get("reviewer_id") or ""), str(payload.get("challenge_reference") or ""), str(payload.get("human_approval") or ""))


def open_case(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.open_case(str(payload["case_id"]), str(payload.get("tenant_id") or "default"), str(payload["signal_id"]), str(payload.get("case_type") or "transaction_fraud"), str(payload.get("investigator_id") or ""), list(payload.get("evidence_references") or []))


def resolve_case(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.resolve_case(str(payload["case_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("disposition") or ""), str(payload.get("reviewer_id") or ""))


def register_fraud_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_fraud_agent(str(payload["agent_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("name") or payload["agent_id"]), str(payload.get("runtime") or "codex"), str(payload.get("role") or "fraud_ops_reviewer"), str(payload.get("scope") or "review fraud signals"))


def list_signals(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_signals(tenant_id)
