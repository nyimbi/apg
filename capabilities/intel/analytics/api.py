"""Process-local API helpers for APG Intelligence Analytics."""

from __future__ import annotations

try:
	from .service import IntelligenceAnalyticsService
except ImportError:  # pragma: no cover
	from service import IntelligenceAnalyticsService  # type: ignore


_SERVICE = IntelligenceAnalyticsService()


def service() -> IntelligenceAnalyticsService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_workspace(payload: dict):
	return _SERVICE.record_workspace(payload["workspace_id"], payload.get("tenant_id", "default"), payload["workspace_type"], payload["name"], payload["classification"], payload["authority_id"], payload["evidence_reference"])


def register_dataset(payload: dict):
	return _SERVICE.register_dataset(payload["dataset_id"], payload.get("tenant_id", "default"), payload["workspace_id"], payload["dataset_type"], payload["source_reference"], payload["owner_id"], payload["lineage_reference"], payload["retention_class"], payload["evidence_reference"])


def record_feature_set(payload: dict):
	return _SERVICE.record_feature_set(payload["feature_set_id"], payload.get("tenant_id", "default"), payload["dataset_id"], payload["feature_type"], payload["feature_reference"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_model(payload: dict):
	return _SERVICE.record_model(payload["model_id"], payload.get("tenant_id", "default"), payload["feature_set_id"], payload["model_type"], payload["objective"], payload["validation_reference"], payload["risk_level"], payload["evidence_reference"])


def record_run(payload: dict):
	return _SERVICE.record_run(payload["run_id"], payload.get("tenant_id", "default"), payload["model_id"], payload["run_type"], payload["result_reference"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_insight(payload: dict):
	return _SERVICE.record_insight(payload["insight_id"], payload.get("tenant_id", "default"), payload["run_id"], payload["insight_type"], payload["claim_reference"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_dashboard(payload: dict):
	return _SERVICE.record_dashboard(payload["dashboard_id"], payload.get("tenant_id", "default"), payload["insight_id"], payload["name"], payload["audience"], payload["release_marking"], payload["approval_reference"], payload["evidence_reference"])


def record_narrative(payload: dict):
	return _SERVICE.record_narrative(payload["narrative_id"], payload.get("tenant_id", "default"), payload["insight_id"], payload["narrative_type"], payload["summary_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_recommendation(payload: dict):
	return _SERVICE.record_recommendation(payload["recommendation_id"], payload.get("tenant_id", "default"), payload["insight_id"], payload["recommendation_type"], payload["action_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_analytics_agent(payload: dict):
	return _SERVICE.register_analytics_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "intelligence analytics operations"))


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(payload.get("tenant_id", "default"), payload.get("privileged_scope", False), payload.get("human_approval_recorded", False), payload.get("hallucinated_insight_scope", False), payload.get("training_data_leakage_scope", False), payload.get("privacy_bypass_scope", False), payload.get("unsupported_automated_decision_scope", False), payload.get("unapproved_model_deployment_scope", False), payload.get("autonomous_dissemination_scope", False))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
