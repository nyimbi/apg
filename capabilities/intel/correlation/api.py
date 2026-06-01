"""Process-local API helpers for APG Data Correlation."""

from __future__ import annotations

try:
	from .service import DataCorrelationService
except ImportError:  # pragma: no cover
	from service import DataCorrelationService  # type: ignore


_SERVICE = DataCorrelationService()


def service() -> DataCorrelationService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_workspace(payload: dict):
	return _SERVICE.record_workspace(payload["workspace_id"], payload.get("tenant_id", "default"), payload["workspace_type"], payload["name"], payload["classification"], payload["authority_id"], payload["evidence_reference"])


def register_source(payload: dict):
	return _SERVICE.register_source(payload["source_id"], payload.get("tenant_id", "default"), payload["workspace_id"], payload["source_type"], payload["source_reference"], payload["custodian_id"], payload["lineage_reference"], payload["evidence_reference"])


def record_entity(payload: dict):
	return _SERVICE.record_entity(payload["entity_id"], payload.get("tenant_id", "default"), payload["source_id"], payload["entity_type"], payload["entity_reference"], payload["confidence_score"], payload["evidence_reference"])


def record_observation(payload: dict):
	return _SERVICE.record_observation(payload["observation_id"], payload.get("tenant_id", "default"), payload["entity_id"], payload["observation_type"], payload["observation_reference"], payload["observed_at"], payload["confidence_score"], payload["evidence_reference"])


def record_rule(payload: dict):
	return _SERVICE.record_rule(payload["rule_id"], payload.get("tenant_id", "default"), payload["workspace_id"], payload["rule_type"], payload["rule_reference"], payload["threshold_score"], payload["analyst_id"], payload["evidence_reference"])


def record_run(payload: dict):
	return _SERVICE.record_run(payload["run_id"], payload.get("tenant_id", "default"), payload["rule_id"], payload["run_type"], payload["result_reference"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_cluster(payload: dict):
	return _SERVICE.record_cluster(payload["cluster_id"], payload.get("tenant_id", "default"), payload["run_id"], payload["cluster_type"], payload["cluster_reference"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_decision(payload: dict):
	return _SERVICE.record_decision(payload["decision_id"], payload.get("tenant_id", "default"), payload["cluster_id"], payload["decision_type"], payload["rationale_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_referral(payload: dict):
	return _SERVICE.record_referral(payload["referral_id"], payload.get("tenant_id", "default"), payload["decision_id"], payload["referral_type"], payload["recipient"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_correlation_agent(payload: dict):
	return _SERVICE.register_correlation_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "data correlation operations"))


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(payload.get("tenant_id", "default"), payload.get("privileged_scope", False), payload.get("human_approval_recorded", False), payload.get("unapproved_identity_merge_scope", False), payload.get("source_tampering_scope", False), payload.get("privacy_bypass_scope", False), payload.get("evidence_fabrication_scope", False), payload.get("autonomous_referral_scope", False), payload.get("unreviewed_high_impact_match_scope", False))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
