"""Process-local API helpers for APG Cyber Intelligence."""

from __future__ import annotations

try:
	from .service import CyberIntelligenceService
except ImportError:  # pragma: no cover
	from service import CyberIntelligenceService  # type: ignore


_SERVICE = CyberIntelligenceService()


def service() -> CyberIntelligenceService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_indicator(payload: dict):
	return _SERVICE.record_indicator(payload["indicator_id"], payload.get("tenant_id", "default"), payload["indicator_type"], payload["indicator_value"], payload["tlp"], payload["confidence_score"], payload["authority_id"], payload["evidence_reference"])


def record_sighting(payload: dict):
	return _SERVICE.record_sighting(payload["sighting_id"], payload.get("tenant_id", "default"), payload["indicator_id"], payload["source_reference"], payload["observed_at"], payload["severity"], payload["evidence_reference"])


def record_enrichment(payload: dict):
	return _SERVICE.record_enrichment(payload["enrichment_id"], payload.get("tenant_id", "default"), payload["indicator_id"], payload["enrichment_type"], payload["provider_reference"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_profile(payload: dict):
	return _SERVICE.record_profile(payload["profile_id"], payload.get("tenant_id", "default"), payload["profile_type"], payload["name"], payload["classification"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_risk(payload: dict):
	return _SERVICE.record_risk(payload["assessment_id"], payload.get("tenant_id", "default"), payload["indicator_id"], payload["profile_id"], payload["risk_level"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_incident_link(payload: dict):
	return _SERVICE.record_incident_link(payload["link_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["incident_reference"], payload["response_priority"], payload["owner_id"], payload["evidence_reference"])


def record_dissemination(payload: dict):
	return _SERVICE.record_dissemination(payload["dissemination_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["audience"], payload["release_marking"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_cybint_agent(payload: dict):
	return _SERVICE.register_cybint_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "cybint operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
