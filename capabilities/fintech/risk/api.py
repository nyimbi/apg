"""Process-local API helpers for APG FinTech Risk Management."""

from __future__ import annotations

try:
	from .service import RiskManagementService
except ImportError:  # pragma: no cover
	from service import RiskManagementService  # type: ignore


_SERVICE = RiskManagementService()


def service() -> RiskManagementService:
	return _SERVICE


def register_appetite(payload: dict):
	return _SERVICE.register_appetite(payload["appetite_id"], payload.get("tenant_id", "default"), payload["risk_domain"], payload["threshold_minor"], payload.get("currency", "USD"), payload["owner_id"], payload["evidence_reference"])


def create_profile(payload: dict):
	return _SERVICE.create_profile(payload["profile_id"], payload.get("tenant_id", "default"), payload["subject_reference"], payload["subject_type"], payload["kyc_reference"], payload["exposure_minor"], payload.get("currency", "USD"), payload["risk_score"], payload["source_reference"])


def record_exposure(payload: dict):
	return _SERVICE.record_exposure(payload["exposure_id"], payload.get("tenant_id", "default"), payload["profile_id"], payload["exposure_type"], payload["amount_minor"], payload.get("currency", "USD"), payload["limit_minor"], payload["source_reference"], payload.get("human_approval", ""))


def evaluate_control(payload: dict):
	return _SERVICE.evaluate_control(payload["control_id"], payload.get("tenant_id", "default"), payload["profile_id"], payload["control_type"], payload["owner_id"], payload["evidence_reference"], payload["effectiveness_score"])


def run_stress_scenario(payload: dict):
	return _SERVICE.run_stress_scenario(payload["scenario_id"], payload.get("tenant_id", "default"), payload["profile_id"], payload["scenario_type"], payload["impact_minor"], payload["probability_bps"], payload["mitigation_reference"])


def record_limit_breach(payload: dict):
	return _SERVICE.record_limit_breach(payload["breach_id"], payload.get("tenant_id", "default"), payload["exposure_id"], payload["severity"], payload["evidence_reference"], payload["remediation_owner"])


def open_risk_event(payload: dict):
	return _SERVICE.open_risk_event(payload["event_id"], payload.get("tenant_id", "default"), payload["profile_id"], payload["event_type"], payload["severity"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_risk_agent(payload: dict):
	return _SERVICE.register_risk_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "risk review"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
