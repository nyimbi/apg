"""Process-local API helpers for APG Intelligence Fusion."""

from __future__ import annotations

try:
	from .service import IntelligenceFusionService
except ImportError:  # pragma: no cover
	from service import IntelligenceFusionService  # type: ignore


_SERVICE = IntelligenceFusionService()


def service() -> IntelligenceFusionService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_workspace(payload: dict):
	return _SERVICE.record_workspace(payload["workspace_id"], payload.get("tenant_id", "default"), payload["workspace_type"], payload["name"], payload["classification"], payload["authority_id"], payload["evidence_reference"])


def register_source(payload: dict):
	return _SERVICE.register_source(payload["source_id"], payload.get("tenant_id", "default"), payload["source_type"], payload["source_reference"], payload["custodian_id"], payload["authority_id"], payload["lineage_reference"], payload["evidence_reference"])


def record_artifact(payload: dict):
	return _SERVICE.record_artifact(payload["artifact_id"], payload.get("tenant_id", "default"), payload["workspace_id"], payload["source_id"], payload["artifact_type"], payload["artifact_reference"], payload["content_fingerprint"], payload["confidence_score"], payload["evidence_reference"])


def record_correlation(payload: dict):
	return _SERVICE.record_correlation(payload["correlation_id"], payload.get("tenant_id", "default"), payload["artifact_id"], payload["correlation_type"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_hypothesis(payload: dict):
	return _SERVICE.record_hypothesis(payload["hypothesis_id"], payload.get("tenant_id", "default"), payload["correlation_id"], payload["hypothesis_type"], payload["claim_reference"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_assessment(payload: dict):
	return _SERVICE.record_assessment(payload["assessment_id"], payload.get("tenant_id", "default"), payload["hypothesis_id"], payload["assessment_type"], payload["risk_level"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_referral(payload: dict):
	return _SERVICE.record_referral(payload["referral_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["referral_type"], payload["recipient"], payload["approval_reference"], payload["evidence_reference"])


def record_dissemination(payload: dict):
	return _SERVICE.record_dissemination(payload["dissemination_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["audience"], payload["release_marking"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_fusion_agent(payload: dict):
	return _SERVICE.register_fusion_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "intelligence fusion operations"))


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(payload.get("tenant_id", "default"), payload.get("privileged_scope", False), payload.get("human_approval_recorded", False), payload.get("evidence_fabrication_scope", False), payload.get("source_tampering_scope", False), payload.get("privacy_bypass_scope", False), payload.get("unsupported_identity_resolution_scope", False), payload.get("autonomous_dissemination_scope", False), payload.get("unapproved_attribution_scope", False))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
