"""Executable capability contract for APG Predictive Analytics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_PRED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_PRED_AGENT_ROLES = [
	"forecast_reviewer",
	"score_reviewer",
	"feature_lineage_reviewer",
	"scenario_reviewer",
	"drift_reviewer",
	"model_release_reviewer",
	"explainability_reviewer",
	"batch_scoring_reviewer",
	"prediction_steward",
]
PRIVILEGED_PRED_AGENT_ROLES = [
	"score_reviewer",
	"drift_reviewer",
	"model_release_reviewer",
	"explainability_reviewer",
	"batch_scoring_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"forecasting": {
		"enabled": True,
		"minimum_history_points": 24,
		"horizon_limit_days": 365,
		"confidence_intervals": True,
		"review_required_for_long_horizon": True,
	},
	"scoring": {
		"real_time_scoring_enabled": True,
		"batch_scoring_enabled": True,
		"feature_lineage_required": True,
		"production_approval_required": True,
		"high_impact_explainability_required": True,
	},
	"feature_sets": {
		"owner_required": True,
		"feature_names_required": True,
		"lineage_required": True,
		"source_system_required": True,
	},
	"models": {
		"approved_model_required": True,
		"owner_required": True,
		"algorithm_required": True,
		"target_required": True,
		"training_history_required": True,
		"explainability_required": True,
		"monitor_drift": True,
	},
	"scenarios": {
		"assumptions_required": True,
		"adjustments_required": True,
		"baseline_required": True,
	},
	"drift": {
		"threshold_required": True,
		"review_required_above_threshold": True,
		"monitoring_window_days": 30,
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_PRED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_PRED_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_PRED_AGENT_ROLES,
		"require_owner": True,
		"require_purpose": True,
		"require_scope": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_prediction_agent_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "pred.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"model_batch",
			"feature_set_batch",
			"forecast_batch",
			"score_batch",
			"scenario_batch",
			"drift_batch",
			"explainability_batch",
			"prediction_agent_batch",
		],
		"topics": [
			"pred.models",
			"pred.features",
			"pred.forecasts",
			"pred.scores",
			"pred.scenarios",
			"pred.drift",
			"pred.explainability",
			"pred.agents",
		],
		"broker_core_dependency_allowed": False,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_predictions": True,
		"auth_required": True,
		"cross_tenant_scoring_allowed": False,
	},
	"observability": {
		"metrics_required": True,
		"trace_required": True,
		"audit_required": True,
		"event_stream": "bytewax",
		"quality_metrics_required": True,
	},
	"adapters": {
		"generated_app_runtime": "service.PredService",
		"helper_runtime": "predictive_runtime.py",
		"production_runtime": "service.PredService",
		"http_api": "api.py",
		"event_stream": "bytewax",
		"ai_core": "aicr",
		"model_lifecycle": "mlcm",
		"data_pipeline": "etlp",
		"configuration": "conf",
		"auth_provider": "auth",
		"audit_sink": "audl",
		"metrics_sink": "moni",
		"cache": "cach",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_forecast_console": True,
		"enable_score_monitor": True,
		"enable_feature_registry": True,
		"enable_scenario_lab": True,
		"enable_model_board": True,
		"enable_drift_monitor": True,
		"enable_batch_scoring": True,
		"enable_explainability": True,
		"enable_prediction_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_governance": True,
		"enable_audit_timeline": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "pred_forecast_console", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"forecasting",
		"scoring",
		"feature_sets",
		"models",
		"scenarios",
		"drift",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"forecasting",
		"scoring",
		"feature_sets",
		"models",
		"scenarios",
		"drift",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All predictive analytics operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "model_requires_owner", "description": "Predictive models require an accountable owner.", "condition": {"operation": "register_model", "owner_present": False}, "effect": {"decision": "deny", "reason": "model_owner_required", "required_action": "assign_model_owner"}},
	{"name": "model_requires_algorithm", "description": "Predictive models require algorithm metadata.", "condition": {"operation": "register_model", "algorithm_present": False}, "effect": {"decision": "deny", "reason": "model_algorithm_required", "required_action": "attach_algorithm_metadata"}},
	{"name": "model_requires_target", "description": "Predictive models require a prediction target.", "condition": {"operation": "register_model", "target_present": False}, "effect": {"decision": "deny", "reason": "model_target_required", "required_action": "attach_prediction_target"}},
	{"name": "model_requires_training_history", "description": "Predictive models require training history evidence.", "condition": {"operation": "register_model", "training_history_points_lt": 24}, "effect": {"decision": "require_review", "reason": "training_history_review_required", "required_action": "record_training_history_review"}},
	{"name": "model_requires_feature_names", "description": "Predictive models require feature metadata.", "condition": {"operation": "register_model", "feature_names_present": False}, "effect": {"decision": "require_review", "reason": "model_feature_metadata_required", "required_action": "attach_feature_metadata"}},
	{"name": "model_approval_requires_explainability", "description": "Model approval requires explainability evidence.", "condition": {"operation": "approve_model", "explainability_attached": False}, "effect": {"decision": "require_review", "reason": "model_explainability_review_required", "required_action": "attach_explainability"}},
	{"name": "feature_set_requires_owner", "description": "Feature sets require an accountable owner.", "condition": {"operation": "register_feature_set", "owner_present": False}, "effect": {"decision": "deny", "reason": "feature_owner_required", "required_action": "assign_feature_owner"}},
	{"name": "feature_set_requires_features", "description": "Feature sets require feature names.", "condition": {"operation": "register_feature_set", "feature_names_present": False}, "effect": {"decision": "deny", "reason": "feature_names_required", "required_action": "attach_feature_names"}},
	{"name": "feature_set_requires_lineage", "description": "Feature sets require lineage references for scoring.", "condition": {"operation": "register_feature_set", "feature_lineage_present": False}, "effect": {"decision": "require_review", "reason": "feature_lineage_review_required", "required_action": "attach_feature_lineage"}},
	{"name": "feature_set_requires_source_system", "description": "Feature sets require a source system.", "condition": {"operation": "register_feature_set", "source_system_present": False}, "effect": {"decision": "deny", "reason": "feature_source_system_required", "required_action": "attach_source_system"}},
	{"name": "forecast_requires_model", "description": "Forecasts require a registered model.", "condition": {"operation": "create_forecast", "model_present": False}, "effect": {"decision": "deny", "reason": "forecast_model_required", "required_action": "select_model"}},
	{"name": "forecast_requires_series", "description": "Forecasts require a series name.", "condition": {"operation": "create_forecast", "series_name_present": False}, "effect": {"decision": "deny", "reason": "forecast_series_required", "required_action": "attach_series_name"}},
	{"name": "forecast_requires_history", "description": "Forecasts require enough historical observations.", "condition": {"operation": "create_forecast", "history_points_lt": 24}, "effect": {"decision": "deny", "reason": "insufficient_history", "required_action": "load_more_history"}},
	{"name": "forecast_requires_positive_horizon", "description": "Forecast horizon must be positive.", "condition": {"operation": "create_forecast", "forecast_horizon_days_lt": 1}, "effect": {"decision": "deny", "reason": "forecast_horizon_required", "required_action": "choose_positive_horizon"}},
	{"name": "long_horizon_requires_review", "description": "Long forecast horizons require review.", "condition": {"operation": "create_forecast", "forecast_horizon_days_gt": 365, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "long_horizon_review_required", "required_action": "record_forecast_review"}},
	{"name": "production_score_requires_approved_model", "description": "Production scoring requires an approved model.", "condition": {"operation": "score", "environment": "production", "model_approved": False}, "effect": {"decision": "deny", "reason": "approved_model_required", "required_action": "approve_model"}},
	{"name": "scoring_requires_feature_lineage", "description": "Predictive scoring requires feature lineage.", "condition": {"operation": "score", "feature_lineage_present": False}, "effect": {"decision": "deny", "reason": "feature_lineage_required", "required_action": "attach_feature_lineage"}},
	{"name": "high_impact_prediction_requires_explainability", "description": "High-impact predictions require explainability artifacts.", "condition": {"operation": "score", "impact": "high", "explainability_attached": False}, "effect": {"decision": "deny", "reason": "explainability_required", "required_action": "attach_explainability"}},
	{"name": "score_requires_entity", "description": "Scores require an entity identifier.", "condition": {"operation": "score", "entity_present": False}, "effect": {"decision": "deny", "reason": "score_entity_required", "required_action": "attach_entity"}},
	{"name": "score_requires_feature_values", "description": "Scores require feature values.", "condition": {"operation": "score", "feature_values_present": False}, "effect": {"decision": "deny", "reason": "score_features_required", "required_action": "attach_feature_values"}},
	{"name": "scenario_requires_model", "description": "Scenarios require a registered model.", "condition": {"operation": "simulate_scenario", "model_present": False}, "effect": {"decision": "deny", "reason": "scenario_model_required", "required_action": "select_model"}},
	{"name": "scenario_requires_assumptions", "description": "Scenarios require explicit assumptions.", "condition": {"operation": "simulate_scenario", "assumptions_present": False}, "effect": {"decision": "deny", "reason": "scenario_assumptions_required", "required_action": "attach_assumptions"}},
	{"name": "scenario_requires_adjustments", "description": "Scenarios require feature adjustments.", "condition": {"operation": "simulate_scenario", "adjustments_present": False}, "effect": {"decision": "deny", "reason": "scenario_adjustments_required", "required_action": "attach_adjustments"}},
	{"name": "scenario_requires_baseline", "description": "Scenarios require a baseline score.", "condition": {"operation": "simulate_scenario", "baseline_present": False}, "effect": {"decision": "deny", "reason": "scenario_baseline_required", "required_action": "attach_baseline_score"}},
	{"name": "drift_requires_metric", "description": "Drift reports require metric metadata.", "condition": {"operation": "record_drift", "metric_name_present": False}, "effect": {"decision": "deny", "reason": "drift_metric_required", "required_action": "attach_drift_metric"}},
	{"name": "drift_requires_threshold", "description": "Drift reports require thresholds.", "condition": {"operation": "record_drift", "threshold_present": False}, "effect": {"decision": "deny", "reason": "drift_threshold_required", "required_action": "attach_drift_threshold"}},
	{"name": "high_drift_requires_review", "description": "Above-threshold drift requires review.", "condition": {"operation": "record_drift", "drift_over_threshold": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "drift_review_required", "required_action": "record_drift_review"}},
	{"name": "batch_scoring_requires_bytewax", "description": "Batch scoring streams must use Bytewax.", "condition": {"operation": "configure_batch_scoring", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "cross_tenant_scoring_denied", "description": "Cross-tenant scoring is denied by default.", "condition": {"cross_tenant_scoring": True}, "effect": {"decision": "deny", "reason": "cross_tenant_scoring_denied", "required_action": "use_tenant_scoped_features"}},
	{"name": "prediction_state_change_requires_audit", "description": "Prediction state changes require audit events.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "audit_event_required", "required_action": "record_audit_event"}},
	{"name": "prediction_agent_runtime_supported", "description": "Prediction agents must use supported runtimes.", "condition": {"operation": "register_prediction_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_prediction_agent_runtime", "required_action": "choose_supported_prediction_agent_runtime"}},
	{"name": "prediction_agent_role_supported", "description": "Prediction agents must use supported analytics-governance roles.", "condition": {"operation": "register_prediction_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_prediction_agent_role", "required_action": "choose_supported_prediction_agent_role"}},
	{"name": "prediction_agent_requires_scope", "description": "Prediction agents require an explicit bounded scope.", "condition": {"operation": "register_prediction_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "prediction_agent_scope_required", "required_action": "declare_prediction_agent_scope"}},
	{"name": "prediction_agent_requires_owner", "description": "Prediction agents require an accountable owner.", "condition": {"operation": "register_prediction_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "prediction_agent_owner_required", "required_action": "assign_prediction_agent_owner"}},
	{"name": "prediction_agent_requires_purpose", "description": "Prediction agents require a documented purpose.", "condition": {"operation": "register_prediction_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "prediction_agent_purpose_required", "required_action": "document_prediction_agent_purpose"}},
	{"name": "prediction_agent_requires_contribution_disclosure", "description": "Prediction agents must disclose machine-authored predictive-analytics contributions.", "condition": {"operation": "register_prediction_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "prediction_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "prediction_agent_privileged_role_requires_human_approval", "description": "Privileged prediction-agent roles require human approval evidence.", "condition": {"operation": "register_prediction_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "prediction_agent_human_approval_required", "required_action": "record_human_prediction_agent_approval"}},
	{"name": "bytewax_pred_stream_required", "description": "PRED lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_pred_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_pred_lifecycle_batch_to_bytewax"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/pred/dashboard", "component": "PREDDashboard", "permission": "pred:view", "nav_group": "Overview"},
	{"name": "forecasts", "path": "/pred/forecasts", "component": "ForecastConsole", "permission": "pred:forecast", "nav_group": "Forecasts"},
	{"name": "scores", "path": "/pred/scores", "component": "ScoreMonitor", "permission": "pred:score", "nav_group": "Scoring"},
	{"name": "features", "path": "/pred/features", "component": "FeatureRegistry", "permission": "pred:manage_models", "nav_group": "Scoring"},
	{"name": "scenarios", "path": "/pred/scenarios", "component": "ScenarioLab", "permission": "pred:simulate", "nav_group": "Simulation"},
	{"name": "models", "path": "/pred/models", "component": "PredictiveModelBoard", "permission": "pred:manage_models", "nav_group": "Models"},
	{"name": "drift", "path": "/pred/drift", "component": "DriftMonitor", "permission": "pred:govern", "nav_group": "Models"},
	{"name": "batch", "path": "/pred/batch", "component": "BatchScoringQueue", "permission": "pred:score", "nav_group": "Scoring"},
	{"name": "explainability", "path": "/pred/explainability", "component": "ExplainabilityWorkbench", "permission": "pred:govern", "nav_group": "Governance"},
	{"name": "agents", "path": "/pred/agents", "component": "PredictionAgentRoster", "permission": "pred:govern", "nav_group": "Governance"},
	{"name": "lifecycle", "path": "/pred/lifecycle", "component": "PREDLifecycleBatchMonitor", "permission": "pred:govern", "nav_group": "Operations"},
	{"name": "governance", "path": "/pred/governance", "component": "PredictionGovernance", "permission": "pred:govern", "nav_group": "Governance"},
	{"name": "audit", "path": "/pred/audit", "component": "PredictionAuditTimeline", "permission": "pred:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/pred/settings", "component": "PREDSettings", "permission": "pred:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "pred_forecast_console",
	"tokens": {
		"color.primary": "#345995",
		"color.accent": "#E07A5F",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"forecast_chart": {"icon": "trending-up", "visual": "confidence-band", "status_indicator": "horizon-chip"},
		"score_card": {"visual": "distribution-bar", "risk_style": "impact-band"},
		"scenario_matrix": {"visual": "comparison-grid", "highlight": "delta-chip"},
		"feature_lineage_panel": {"visual": "lineage-list", "status_style": "evidence-pill"},
		"drift_monitor": {"visual": "threshold-band", "status_style": "review-chip"},
		"model_board": {"visual": "model-card-grid", "status_style": "approval-chip"},
		"batch_queue": {"visual": "queue-table", "status_style": "bytewax-chip"},
		"prediction_agent_roster": {"icon": "bot", "status_indicator": "agent-approval-chip", "risk_style": "impact-scope-band"},
		"bytewax_lifecycle_panel": {"icon": "git-branch", "status_indicator": "stream-chip", "risk_style": "processor-band"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "evidence-chip"},
	},
}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class PRED agent composition manifest."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_PRED_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_PRED_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_PRED_AGENT_ROLES),
		"required_fields": ["tenant_id", "agent_id", "name", "runtime", "role", "scope", "owner", "purpose"],
		"guardrails": [
			"supported_runtime",
			"supported_role",
			"explicit_scope",
			"accountable_owner",
			"declared_purpose",
			"machine_contribution_disclosure",
			"human_approval_for_privileged_roles",
		],
		"adapter_contract": "aicr_provider_neutral_prediction_agent_adapter",
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return the PRED Bytewax lifecycle stream contract."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "pred.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"model_batch",
			"feature_set_batch",
			"forecast_batch",
			"score_batch",
			"scenario_batch",
			"drift_batch",
			"explainability_batch",
			"prediction_agent_batch",
		],
		"topics": [
			"pred.models",
			"pred.features",
			"pred.forecasts",
			"pred.scores",
			"pred.scenarios",
			"pred.drift",
			"pred.explainability",
			"pred.agents",
		],
		"broker_core_dependency_allowed": False,
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable PRED capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "pred",
		"display_name": "Predictive Analytics",
		"provides": ["predictive_analytics", "forecasting", "prediction_agent_composition"],
		"requires": ["aicr", "mlcm", "etlp", "conf"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/pred/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
		"theme": deepcopy(THEME),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default PRED governance rules."""
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if key[:-3] not in context or not context[key[:-3]] < expected:
				return False
		elif key.endswith("_gt"):
			if key[:-3] not in context or not context[key[:-3]] > expected:
				return False
		elif key.endswith("_ne"):
			if key[:-3] not in context or context[key[:-3]] == expected:
				return False
		elif key not in context or context[key] != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
