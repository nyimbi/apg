"""Executable capability contract for APG Predictive Intelligence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_prediction"
CAPABILITY_NAME = "Predictive Intelligence"
CAPABILITY_VERSION = "1.1.0"
PREDICTION_EVENT_STREAM = "apg.intel.prediction.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "legal_mandate", "partner_authority", "consent", "incident_response_authority", "public_interest_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_WORKSPACE_TYPES = ["threat_prediction", "fraud_prediction", "public_safety_prediction", "operational_prediction", "strategic_forecast", "incident_prediction", "risk_forecast"]
SUPPORTED_SCENARIO_TYPES = ["threat_scenario", "fraud_scenario", "incident_scenario", "public_safety_scenario", "operational_scenario", "strategic_scenario", "risk_scenario"]
SUPPORTED_HORIZONS = ["near_term", "mid_term", "long_term", "continuous"]
SUPPORTED_INDICATOR_TYPES = ["leading_indicator", "lagging_indicator", "anomaly_indicator", "behavioral_indicator", "geospatial_indicator", "network_indicator", "text_indicator"]
SUPPORTED_MODEL_TYPES = ["ruleset", "statistical", "machine_learning", "simulation", "graph_forecast", "geospatial_forecast", "nlp_forecast"]
SUPPORTED_FORECAST_TYPES = ["probability", "trend", "scenario_outcome", "event_likelihood", "impact_forecast", "risk_forecast"]
SUPPORTED_PROJECTION_TYPES = ["threat_projection", "fraud_projection", "impact_projection", "resource_projection", "timeline_projection", "confidence_projection"]
SUPPORTED_WARNING_TYPES = ["early_warning", "watchlist_warning", "threshold_warning", "forecast_change", "critical_projection"]
SUPPORTED_RECOMMENDATION_TYPES = ["monitor", "investigate", "mitigate", "escalate", "request_collection", "review_model", "close"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["scenario_planner", "indicator_steward", "model_reviewer", "forecast_analyst", "warning_reviewer", "recommendation_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"workspaces": {"supported_workspace_types": SUPPORTED_WORKSPACE_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "authority_required": True, "evidence_required": True},
	"scenarios": {"supported_scenario_types": SUPPORTED_SCENARIO_TYPES, "supported_horizons": SUPPORTED_HORIZONS, "workspace_required": True, "owner_required": True, "evidence_required": True},
	"indicators": {"supported_indicator_types": SUPPORTED_INDICATOR_TYPES, "scenario_required": True, "confidence_required": True, "evidence_required": True},
	"models": {"supported_model_types": SUPPORTED_MODEL_TYPES, "supported_risk_levels": SUPPORTED_RISK_LEVELS, "scenario_required": True, "validation_required": True, "evidence_required": True},
	"forecasts": {"supported_forecast_types": SUPPORTED_FORECAST_TYPES, "model_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"projections": {"supported_projection_types": SUPPORTED_PROJECTION_TYPES, "forecast_required": True, "probability_required": True, "risk_level_required": True, "analyst_required": True, "evidence_required": True},
	"warnings": {"supported_warning_types": SUPPORTED_WARNING_TYPES, "projection_required": True, "severity_required": True, "trigger_required": True, "approval_required": True, "evidence_required": True},
	"recommendations": {"supported_recommendation_types": SUPPORTED_RECOMMENDATION_TYPES, "projection_required": True, "action_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "name_required": True, "scope_required": True, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "cross_tenant_prediction_denied": True, "unsupported_automated_decision_denied": True, "hallucinated_forecast_denied": True, "privacy_bypass_denied": True, "unapproved_model_deployment_denied": True, "autonomous_warning_denied": True, "autonomous_recommendation_denied": True},
	"observability": {"event_stream": PREDICTION_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "geospatial": "geos", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_workspaces": True, "enable_scenarios": True, "enable_indicators": True, "enable_models": True, "enable_forecasts": True, "enable_projections": True, "enable_warnings": True, "enable_recommendations": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_prediction_control", "allow_tenant_overrides": True},
}

PROVIDES = ["prediction_authority_workflow", "prediction_workspace_workflow", "prediction_scenario_workflow", "prediction_indicator_workflow", "prediction_model_workflow", "prediction_forecast_workflow", "prediction_projection_workflow", "prediction_warning_workflow", "prediction_recommendation_workflow", "prediction_review_workflow", "prediction_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-prediction/dashboard", "component": "PredictionDashboard", "permission": "intel_prediction:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-prediction/authorities", "component": "PredictionAuthorityConsole", "permission": "intel_prediction:authorities", "nav_group": "Governance"},
	{"name": "workspaces", "path": "/intel-prediction/workspaces", "component": "PredictionWorkspaceConsole", "permission": "intel_prediction:workspaces", "nav_group": "Planning"},
	{"name": "scenarios", "path": "/intel-prediction/scenarios", "component": "PredictionScenarioWorkbench", "permission": "intel_prediction:scenarios", "nav_group": "Planning"},
	{"name": "indicators", "path": "/intel-prediction/indicators", "component": "PredictionIndicatorLedger", "permission": "intel_prediction:indicators", "nav_group": "Signals"},
	{"name": "models", "path": "/intel-prediction/models", "component": "PredictionModelWorkbench", "permission": "intel_prediction:models", "nav_group": "Models"},
	{"name": "forecasts", "path": "/intel-prediction/forecasts", "component": "PredictionForecastWorkbench", "permission": "intel_prediction:forecasts", "nav_group": "Forecasts"},
	{"name": "projections", "path": "/intel-prediction/projections", "component": "PredictionProjectionConsole", "permission": "intel_prediction:projections", "nav_group": "Forecasts"},
	{"name": "warnings", "path": "/intel-prediction/warnings", "component": "PredictionWarningConsole", "permission": "intel_prediction:warnings", "nav_group": "Action"},
	{"name": "recommendations", "path": "/intel-prediction/recommendations", "component": "PredictionRecommendationConsole", "permission": "intel_prediction:recommendations", "nav_group": "Action"},
	{"name": "reviews", "path": "/intel-prediction/reviews", "component": "PredictionReviewConsole", "permission": "intel_prediction:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-prediction/agents", "component": "PredictionAgentWorkbench", "permission": "intel_prediction:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-prediction/settings", "component": "PredictionSettings", "permission": "intel_prediction:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_prediction_control",
	"tokens": {"color.primary": "#0E7490", "color.accent": "#7C2D12", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "workspaces": {"icon": "layout-dashboard", "status_indicator": "workspace-chip"}, "scenarios": {"icon": "route", "status_indicator": "scenario-chip"}, "indicators": {"icon": "radar", "status_indicator": "indicator-chip"}, "models": {"icon": "brain-circuit", "status_indicator": "model-chip"}, "forecasts": {"icon": "line-chart", "status_indicator": "forecast-chip"}, "projections": {"icon": "trending-up", "status_indicator": "risk-chip"}, "warnings": {"icon": "shield-alert", "status_indicator": "warning-chip"}, "recommendations": {"icon": "list-checks", "status_indicator": "action-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": PREDICTION_EVENT_STREAM, "key": "tenant_id", "events": ["prediction_authority_recorded", "prediction_workspace_recorded", "prediction_scenario_recorded", "prediction_indicator_recorded", "prediction_model_recorded", "prediction_forecast_recorded", "prediction_projection_recorded", "prediction_warning_recorded", "prediction_recommendation_recorded", "prediction_review_recorded", "prediction_agent_registered"], "guardrails": ["prediction_batch_requires_bytewax", "privileged_prediction_agent_action_requires_human_approval", "unsupported_automated_decision_action_denied", "hallucinated_forecast_action_denied", "privacy_bypass_action_denied", "unapproved_model_deployment_action_denied", "autonomous_warning_action_denied", "autonomous_recommendation_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "prediction_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "prediction_policy_required", "required_action": "attach_prediction_policy"}},
	{"name": "authority_type_supported", "condition": {"operation": "record_authority", "authority_type_supported": False}, "effect": {"decision": "deny", "reason": "authority_type_not_supported", "required_action": "select_supported_authority_type"}},
	{"name": "authority_scope_required", "condition": {"operation": "record_authority", "scope_present": False}, "effect": {"decision": "deny", "reason": "authority_scope_required", "required_action": "attach_scope_reference"}},
	{"name": "authority_classification_supported", "condition": {"operation": "record_authority", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "authority_approver_required", "condition": {"operation": "record_authority", "approver_present": False}, "effect": {"decision": "deny", "reason": "authority_approver_required", "required_action": "record_approver"}},
	{"name": "authority_expiry_required", "condition": {"operation": "record_authority", "expiry_present": False}, "effect": {"decision": "deny", "reason": "authority_expiry_required", "required_action": "set_expiry"}},
	{"name": "authority_evidence_required", "condition": {"operation": "record_authority", "evidence_present": False}, "effect": {"decision": "deny", "reason": "authority_evidence_required", "required_action": "attach_authority_evidence"}},
	{"name": "workspace_type_supported", "condition": {"operation": "record_workspace", "workspace_type_supported": False}, "effect": {"decision": "deny", "reason": "workspace_type_not_supported", "required_action": "select_supported_workspace_type"}},
	{"name": "workspace_name_required", "condition": {"operation": "record_workspace", "workspace_name_present": False}, "effect": {"decision": "deny", "reason": "workspace_name_required", "required_action": "name_workspace"}},
	{"name": "workspace_classification_supported", "condition": {"operation": "record_workspace", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "workspace_authority_required", "condition": {"operation": "record_workspace", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "workspace_evidence_required", "condition": {"operation": "record_workspace", "evidence_present": False}, "effect": {"decision": "deny", "reason": "workspace_evidence_required", "required_action": "attach_workspace_evidence"}},
	{"name": "scenario_workspace_required", "condition": {"operation": "record_scenario", "workspace_present": False}, "effect": {"decision": "deny", "reason": "workspace_required", "required_action": "select_workspace"}},
	{"name": "scenario_type_supported", "condition": {"operation": "record_scenario", "scenario_type_supported": False}, "effect": {"decision": "deny", "reason": "scenario_type_not_supported", "required_action": "select_supported_scenario_type"}},
	{"name": "scenario_reference_required", "condition": {"operation": "record_scenario", "scenario_reference_present": False}, "effect": {"decision": "deny", "reason": "scenario_reference_required", "required_action": "attach_scenario_reference"}},
	{"name": "scenario_horizon_supported", "condition": {"operation": "record_scenario", "horizon_supported": False}, "effect": {"decision": "deny", "reason": "prediction_horizon_not_supported", "required_action": "select_supported_horizon"}},
	{"name": "scenario_owner_required", "condition": {"operation": "record_scenario", "owner_present": False}, "effect": {"decision": "deny", "reason": "scenario_owner_required", "required_action": "assign_scenario_owner"}},
	{"name": "scenario_evidence_required", "condition": {"operation": "record_scenario", "evidence_present": False}, "effect": {"decision": "deny", "reason": "scenario_evidence_required", "required_action": "attach_scenario_evidence"}},
	{"name": "indicator_scenario_required", "condition": {"operation": "record_indicator", "scenario_present": False}, "effect": {"decision": "deny", "reason": "scenario_required", "required_action": "select_scenario"}},
	{"name": "indicator_type_supported", "condition": {"operation": "record_indicator", "indicator_type_supported": False}, "effect": {"decision": "deny", "reason": "indicator_type_not_supported", "required_action": "select_supported_indicator_type"}},
	{"name": "indicator_reference_required", "condition": {"operation": "record_indicator", "indicator_reference_present": False}, "effect": {"decision": "deny", "reason": "indicator_reference_required", "required_action": "attach_indicator_reference"}},
	{"name": "indicator_confidence_valid", "condition": {"operation": "record_indicator", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "indicator_evidence_required", "condition": {"operation": "record_indicator", "evidence_present": False}, "effect": {"decision": "deny", "reason": "indicator_evidence_required", "required_action": "attach_indicator_evidence"}},
	{"name": "model_scenario_required", "condition": {"operation": "record_model", "scenario_present": False}, "effect": {"decision": "deny", "reason": "scenario_required", "required_action": "select_scenario"}},
	{"name": "model_type_supported", "condition": {"operation": "record_model", "model_type_supported": False}, "effect": {"decision": "deny", "reason": "model_type_not_supported", "required_action": "select_supported_model_type"}},
	{"name": "model_objective_required", "condition": {"operation": "record_model", "objective_present": False}, "effect": {"decision": "deny", "reason": "model_objective_required", "required_action": "record_model_objective"}},
	{"name": "model_validation_required", "condition": {"operation": "record_model", "validation_present": False}, "effect": {"decision": "deny", "reason": "model_validation_required", "required_action": "attach_validation_reference"}},
	{"name": "model_risk_supported", "condition": {"operation": "record_model", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "model_evidence_required", "condition": {"operation": "record_model", "evidence_present": False}, "effect": {"decision": "deny", "reason": "model_evidence_required", "required_action": "attach_model_evidence"}},
	{"name": "forecast_model_required", "condition": {"operation": "record_forecast", "model_present": False}, "effect": {"decision": "deny", "reason": "model_required", "required_action": "select_model"}},
	{"name": "forecast_type_supported", "condition": {"operation": "record_forecast", "forecast_type_supported": False}, "effect": {"decision": "deny", "reason": "forecast_type_not_supported", "required_action": "select_supported_forecast_type"}},
	{"name": "forecast_reference_required", "condition": {"operation": "record_forecast", "forecast_reference_present": False}, "effect": {"decision": "deny", "reason": "forecast_reference_required", "required_action": "attach_forecast_reference"}},
	{"name": "forecast_confidence_valid", "condition": {"operation": "record_forecast", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "forecast_analyst_required", "condition": {"operation": "record_forecast", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "forecast_evidence_required", "condition": {"operation": "record_forecast", "evidence_present": False}, "effect": {"decision": "deny", "reason": "forecast_evidence_required", "required_action": "attach_forecast_evidence"}},
	{"name": "projection_forecast_required", "condition": {"operation": "record_projection", "forecast_present": False}, "effect": {"decision": "deny", "reason": "forecast_required", "required_action": "select_forecast"}},
	{"name": "projection_type_supported", "condition": {"operation": "record_projection", "projection_type_supported": False}, "effect": {"decision": "deny", "reason": "projection_type_not_supported", "required_action": "select_supported_projection_type"}},
	{"name": "projection_risk_supported", "condition": {"operation": "record_projection", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "projection_probability_valid", "condition": {"operation": "record_projection", "probability_valid": False}, "effect": {"decision": "deny", "reason": "probability_score_invalid", "required_action": "set_probability_0_to_1"}},
	{"name": "projection_analyst_required", "condition": {"operation": "record_projection", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "projection_evidence_required", "condition": {"operation": "record_projection", "evidence_present": False}, "effect": {"decision": "deny", "reason": "projection_evidence_required", "required_action": "attach_projection_evidence"}},
	{"name": "warning_projection_required", "condition": {"operation": "record_warning", "projection_present": False}, "effect": {"decision": "deny", "reason": "projection_required", "required_action": "select_projection"}},
	{"name": "warning_type_supported", "condition": {"operation": "record_warning", "warning_type_supported": False}, "effect": {"decision": "deny", "reason": "warning_type_not_supported", "required_action": "select_supported_warning_type"}},
	{"name": "warning_severity_supported", "condition": {"operation": "record_warning", "severity_supported": False}, "effect": {"decision": "deny", "reason": "severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "warning_trigger_required", "condition": {"operation": "record_warning", "trigger_present": False}, "effect": {"decision": "deny", "reason": "trigger_reference_required", "required_action": "attach_trigger_reference"}},
	{"name": "warning_approval_required", "condition": {"operation": "record_warning", "approval_present": False}, "effect": {"decision": "deny", "reason": "warning_approval_required", "required_action": "attach_warning_approval"}},
	{"name": "warning_evidence_required", "condition": {"operation": "record_warning", "evidence_present": False}, "effect": {"decision": "deny", "reason": "warning_evidence_required", "required_action": "attach_warning_evidence"}},
	{"name": "recommendation_projection_required", "condition": {"operation": "record_recommendation", "projection_present": False}, "effect": {"decision": "deny", "reason": "projection_required", "required_action": "select_projection"}},
	{"name": "recommendation_type_supported", "condition": {"operation": "record_recommendation", "recommendation_type_supported": False}, "effect": {"decision": "deny", "reason": "recommendation_type_not_supported", "required_action": "select_supported_recommendation_type"}},
	{"name": "recommendation_action_required", "condition": {"operation": "record_recommendation", "action_present": False}, "effect": {"decision": "deny", "reason": "action_reference_required", "required_action": "attach_action_reference"}},
	{"name": "recommendation_approval_required", "condition": {"operation": "record_recommendation", "approval_present": False}, "effect": {"decision": "deny", "reason": "recommendation_approval_required", "required_action": "attach_recommendation_approval"}},
	{"name": "recommendation_evidence_required", "condition": {"operation": "record_recommendation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "recommendation_evidence_required", "required_action": "attach_recommendation_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "prediction_batch_requires_bytewax", "condition": {"operation": "prediction_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_prediction_batch_to_bytewax"}},
	{"name": "prediction_agent_runtime_supported", "condition": {"operation": "register_prediction_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "prediction_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "prediction_agent_role_supported", "condition": {"operation": "register_prediction_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "prediction_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "prediction_agent_name_required", "condition": {"operation": "register_prediction_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "prediction_agent_name_required", "required_action": "name_prediction_agent"}},
	{"name": "prediction_agent_scope_required", "condition": {"operation": "register_prediction_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "prediction_agent_scope_required", "required_action": "bound_prediction_agent_scope"}},
	{"name": "privileged_prediction_agent_action_requires_human_approval", "condition": {"operation": "prediction_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "unsupported_automated_decision_action_denied", "condition": {"operation": "prediction_agent_action", "unsupported_automated_decision_scope": True}, "effect": {"decision": "deny", "reason": "unsupported_automated_decision_scope_denied", "required_action": "remove_automated_decision_scope"}},
	{"name": "hallucinated_forecast_action_denied", "condition": {"operation": "prediction_agent_action", "hallucinated_forecast_scope": True}, "effect": {"decision": "deny", "reason": "hallucinated_forecast_scope_denied", "required_action": "remove_hallucinated_forecast_scope"}},
	{"name": "privacy_bypass_action_denied", "condition": {"operation": "prediction_agent_action", "privacy_bypass_scope": True}, "effect": {"decision": "deny", "reason": "privacy_bypass_scope_denied", "required_action": "remove_privacy_bypass_scope"}},
	{"name": "unapproved_model_deployment_action_denied", "condition": {"operation": "prediction_agent_action", "unapproved_model_deployment_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_model_deployment_scope_denied", "required_action": "remove_model_deployment_scope"}},
	{"name": "autonomous_warning_action_denied", "condition": {"operation": "prediction_agent_action", "autonomous_warning_scope": True}, "effect": {"decision": "deny", "reason": "autonomous_warning_scope_denied", "required_action": "remove_autonomous_warning_scope"}},
	{"name": "autonomous_recommendation_action_denied", "condition": {"operation": "prediction_agent_action", "autonomous_recommendation_scope": True}, "effect": {"decision": "deny", "reason": "autonomous_recommendation_scope_denied", "required_action": "remove_autonomous_recommendation_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-prediction/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions: list[dict[str, Any]] = []
	for rule in RULES:
		if _matches(rule["condition"], context):
			actions.append(rule["effect"] | {"rule": rule["name"]})
	if not actions:
		return {"decision": "allow", "actions": [], "context": dict(context)}
	return {"decision": "deny", "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
