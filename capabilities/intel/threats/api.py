"""Process-local API helpers for APG Threat Intelligence."""

from __future__ import annotations

try:
	from .service import ThreatIntelligenceService
except ImportError:  # pragma: no cover
	from service import ThreatIntelligenceService  # type: ignore


_SERVICE = ThreatIntelligenceService()


def service() -> ThreatIntelligenceService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_workspace(payload: dict):
	return _SERVICE.record_workspace(payload["workspace_id"], payload.get("tenant_id", "default"), payload["workspace_type"], payload["name"], payload["classification"], payload["authority_id"], payload["evidence_reference"])


def register_source(payload: dict):
	return _SERVICE.register_source(payload["source_id"], payload.get("tenant_id", "default"), payload["workspace_id"], payload["source_type"], payload["source_reference"], payload["custodian_id"], payload["lineage_reference"], payload["evidence_reference"])


def record_indicator(payload: dict):
	return _SERVICE.record_indicator(payload["indicator_id"], payload.get("tenant_id", "default"), payload["source_id"], payload["indicator_type"], payload["indicator_reference"], payload["confidence_score"], payload["evidence_reference"])


def record_actor(payload: dict):
	return _SERVICE.record_actor(payload["actor_id"], payload.get("tenant_id", "default"), payload["workspace_id"], payload["actor_type"], payload["actor_reference"], payload["confidence_score"], payload["evidence_reference"])


def record_campaign(payload: dict):
	return _SERVICE.record_campaign(payload["campaign_id"], payload.get("tenant_id", "default"), payload["actor_id"], payload["campaign_type"], payload["campaign_reference"], payload["risk_level"], payload["evidence_reference"])


def record_assessment(payload: dict):
	return _SERVICE.record_assessment(payload["assessment_id"], payload.get("tenant_id", "default"), payload["campaign_id"], payload["assessment_type"], payload["risk_level"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_report(payload: dict):
	return _SERVICE.record_report(payload["report_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["report_type"], payload["report_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_mitigation(payload: dict):
	return _SERVICE.record_mitigation(payload["mitigation_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["mitigation_type"], payload["action_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_threat_agent(payload: dict):
	return _SERVICE.register_threat_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "threat intelligence operations"))


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(payload.get("tenant_id", "default"), payload.get("privileged_scope", False), payload.get("human_approval_recorded", False), payload.get("unsupported_attribution_scope", False), payload.get("fabricated_indicator_scope", False), payload.get("source_tampering_scope", False), payload.get("privacy_bypass_scope", False), payload.get("autonomous_mitigation_scope", False), payload.get("unapproved_publication_scope", False))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))

