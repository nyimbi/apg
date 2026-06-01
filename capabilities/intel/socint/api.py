"""Process-local API helpers for APG Social Media Intelligence."""

from __future__ import annotations

try:
	from .service import SocialMediaIntelligenceService
except ImportError:  # pragma: no cover
	from service import SocialMediaIntelligenceService  # type: ignore


_SERVICE = SocialMediaIntelligenceService()


def service() -> SocialMediaIntelligenceService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_topic(payload: dict):
	return _SERVICE.record_topic(payload["topic_id"], payload.get("tenant_id", "default"), payload["topic_type"], payload["name"], payload["priority"], payload["authority_id"], payload["evidence_reference"])


def register_source(payload: dict):
	return _SERVICE.register_source(payload["source_id"], payload.get("tenant_id", "default"), payload["source_type"], payload["platform_type"], payload["source_reference"], payload["owner_id"], payload["authority_id"], payload["terms_review_reference"], payload["evidence_reference"])


def record_post(payload: dict):
	return _SERVICE.record_post(payload["post_id"], payload.get("tenant_id", "default"), payload["topic_id"], payload["source_id"], payload["post_type"], payload["post_reference"], payload["content_fingerprint"], payload["observed_at"], payload["confidence_score"], payload["evidence_reference"])


def record_signal(payload: dict):
	return _SERVICE.record_signal(payload["signal_id"], payload.get("tenant_id", "default"), payload["post_id"], payload["signal_type"], payload["risk_level"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_influence(payload: dict):
	return _SERVICE.record_influence(payload["assessment_id"], payload.get("tenant_id", "default"), payload["signal_id"], payload["influence_type"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_network(payload: dict):
	return _SERVICE.record_network(payload["assessment_id"], payload.get("tenant_id", "default"), payload["signal_id"], payload["network_type"], payload["risk_level"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_referral(payload: dict):
	return _SERVICE.record_referral(payload["referral_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["referral_type"], payload["recipient"], payload["approval_reference"], payload["evidence_reference"])


def record_dissemination(payload: dict):
	return _SERVICE.record_dissemination(payload["dissemination_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["audience"], payload["release_marking"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_socint_agent(payload: dict):
	return _SERVICE.register_socint_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "socint operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
