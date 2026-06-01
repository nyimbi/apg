"""Process-local API helpers for APG Know Your Customer."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import KnowYourCustomerService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import KnowYourCustomerService  # type: ignore


SERVICE = KnowYourCustomerService()


def service() -> KnowYourCustomerService:
	return SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {"capability": contract["capability"], "display_name": contract["display_name"], "tenant_id": tenant_id, "route_count": len(contract["ui"]["routes"]), "rule_count": len(contract["rule_engine"]["rules"]), "profile_count": summary["profile_count"], "verified_count": summary["verified_count"], "streaming": summary["streaming"]}


def open_profile(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.open_profile(str(payload["profile_id"]), str(payload.get("tenant_id") or "default"), str(payload["subject_reference"]), str(payload["legal_name"]), str(payload.get("customer_type") or "individual"), str(payload.get("country_code") or ""), str(payload.get("consent_reference") or ""), dict(payload.get("metadata") or {}), bool(payload.get("policy_attached", True)))


def register_document(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_document(str(payload["document_id"]), str(payload.get("tenant_id") or "default"), str(payload["profile_id"]), str(payload["document_type"]), str(payload["token_reference"]), str(payload.get("extracted_subject") or ""), payload.get("confidence", 0))


def record_screening(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_screening(str(payload["screening_id"]), str(payload.get("tenant_id") or "default"), str(payload["profile_id"]), bool(payload.get("sanctions_hit", False)), bool(payload.get("pep_hit", False)), bool(payload.get("watchlist_hit", False)), bool(payload.get("adverse_media_hit", False)), str(payload.get("review_id") or ""))


def score_risk(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.score_risk(str(payload["decision_id"]), str(payload.get("tenant_id") or "default"), str(payload["profile_id"]), payload.get("risk_score", 0), str(payload.get("review_id") or ""))


def record_decision(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_decision(str(payload["decision_id"]), str(payload.get("tenant_id") or "default"), str(payload["profile_id"]), str(payload.get("decision") or "approve"), payload.get("risk_score", 0), str(payload.get("review_id") or ""))


def register_kyc_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_kyc_agent(str(payload["agent_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("name") or payload["agent_id"]), str(payload.get("runtime") or "codex"), str(payload.get("role") or "kyc_ops_reviewer"), str(payload.get("scope") or "review onboarding"))


def list_profiles(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_profiles(tenant_id)
