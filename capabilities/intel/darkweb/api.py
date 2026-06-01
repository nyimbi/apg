"""Process-local API helpers for APG Dark Web Monitoring."""

from __future__ import annotations

try:
	from .service import DarkWebMonitoringService
except ImportError:  # pragma: no cover
	from service import DarkWebMonitoringService  # type: ignore


_SERVICE = DarkWebMonitoringService()


def service() -> DarkWebMonitoringService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_program(payload: dict):
	return _SERVICE.record_program(payload["program_id"], payload.get("tenant_id", "default"), payload["program_type"], payload["name"], payload["priority"], payload["authority_id"], payload["evidence_reference"])


def register_source(payload: dict):
	return _SERVICE.register_source(payload["source_id"], payload.get("tenant_id", "default"), payload["source_type"], payload["network_type"], payload["source_reference"], payload["custodian_id"], payload["authority_id"], payload["access_review_reference"], payload["evidence_reference"])


def record_observation(payload: dict):
	return _SERVICE.record_observation(payload["observation_id"], payload.get("tenant_id", "default"), payload["program_id"], payload["source_id"], payload["observation_type"], payload["observation_reference"], payload["content_fingerprint"], payload["observed_at"], payload["confidence_score"], payload["evidence_reference"])


def record_indicator(payload: dict):
	return _SERVICE.record_indicator(payload["indicator_id"], payload.get("tenant_id", "default"), payload["observation_id"], payload["indicator_type"], payload["risk_level"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_marketplace_risk(payload: dict):
	return _SERVICE.record_marketplace_risk(payload["assessment_id"], payload.get("tenant_id", "default"), payload["indicator_id"], payload["assessment_type"], payload["risk_level"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_threat_actor(payload: dict):
	return _SERVICE.record_threat_actor(payload["assessment_id"], payload.get("tenant_id", "default"), payload["indicator_id"], payload["actor_reference"], payload["risk_level"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_referral(payload: dict):
	return _SERVICE.record_referral(payload["referral_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["referral_type"], payload["recipient"], payload["approval_reference"], payload["evidence_reference"])


def record_dissemination(payload: dict):
	return _SERVICE.record_dissemination(payload["dissemination_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["audience"], payload["release_marking"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_darkweb_agent(payload: dict):
	return _SERVICE.register_darkweb_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "darkweb monitoring operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
