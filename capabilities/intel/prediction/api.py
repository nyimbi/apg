"""Process-local API helpers for APG Predictive Intelligence."""

from __future__ import annotations

try:
	from .service import PredictiveIntelligenceService
except ImportError:  # pragma: no cover
	from service import PredictiveIntelligenceService  # type: ignore


_SERVICE = PredictiveIntelligenceService()


def service() -> PredictiveIntelligenceService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_workspace(payload: dict):
	return _SERVICE.record_workspace(payload["workspace_id"], payload.get("tenant_id", "default"), payload["workspace_type"], payload["name"], payload["classification"], payload["authority_id"], payload["evidence_reference"])


def record_scenario(payload: dict):
	return _SERVICE.record_scenario(payload["scenario_id"], payload.get("tenant_id", "default"), payload["workspace_id"], payload["scenario_type"], payload["scenario_reference"], payload["horizon"], payload["owner_id"], payload["evidence_reference"])


def record_indicator(payload: dict):
	return _SERVICE.record_indicator(payload["indicator_id"], payload.get("tenant_id", "default"), payload["scenario_id"], payload["indicator_type"], payload["indicator_reference"], payload["confidence_score"], payload["evidence_reference"])


def record_model(payload: dict):
	return _SERVICE.record_model(payload["model_id"], payload.get("tenant_id", "default"), payload["scenario_id"], payload["model_type"], payload["objective"], payload["validation_reference"], payload["risk_level"], payload["evidence_reference"])


def record_forecast(payload: dict):
	return _SERVICE.record_forecast(payload["forecast_id"], payload.get("tenant_id", "default"), payload["model_id"], payload["forecast_type"], payload["forecast_reference"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_projection(payload: dict):
	return _SERVICE.record_projection(payload["projection_id"], payload.get("tenant_id", "default"), payload["forecast_id"], payload["projection_type"], payload["risk_level"], payload["probability_score"], payload["analyst_id"], payload["evidence_reference"])


def record_warning(payload: dict):
	return _SERVICE.record_warning(payload["warning_id"], payload.get("tenant_id", "default"), payload["projection_id"], payload["warning_type"], payload["severity"], payload["trigger_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_recommendation(payload: dict):
	return _SERVICE.record_recommendation(payload["recommendation_id"], payload.get("tenant_id", "default"), payload["projection_id"], payload["recommendation_type"], payload["action_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_prediction_agent(payload: dict):
	return _SERVICE.register_prediction_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "predictive intelligence operations"))


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(payload.get("tenant_id", "default"), payload.get("privileged_scope", False), payload.get("human_approval_recorded", False), payload.get("unsupported_automated_decision_scope", False), payload.get("hallucinated_forecast_scope", False), payload.get("privacy_bypass_scope", False), payload.get("unapproved_model_deployment_scope", False), payload.get("autonomous_warning_scope", False), payload.get("autonomous_recommendation_scope", False))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
