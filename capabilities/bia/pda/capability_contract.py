"""Executable capability contract for APG Predictive Analytics (bia_pda)."""

from __future__ import annotations
from copy import deepcopy
from typing import Any

CAPABILITY_ID = "bia_pda"
CAPABILITY_NAME = "Predictive Analytics"
CAPABILITY_VERSION = "1.0.0"
PDA_EVENT_STREAM = "apg.bia.pda.lifecycle"

SUPPORTED_MODEL_TYPES = ["linear_regression", "logistic_regression", "random_forest", "gradient_boosting", "neural_network", "arima", "prophet", "lstm", "xgboost", "isolation_forest", "clustering"]
SUPPORTED_FORECAST_HORIZONS = ["1d", "7d", "14d", "30d", "90d", "180d", "365d", "custom"]
SUPPORTED_TREND_TYPES = ["linear", "exponential", "polynomial", "seasonal", "cyclical", "stationary"]
SUPPORTED_SCENARIO_TYPES = ["optimistic", "pessimistic", "base", "stress_test", "custom"]
SUPPORTED_FEATURE_TYPES = ["numerical", "categorical", "datetime", "text", "boolean", "derived"]
SUPPORTED_VALIDATION_METHODS = ["holdout", "cross_validation", "time_series_split", "walk_forward"]
SUPPORTED_MODEL_STATES = ["training", "trained", "deployed", "deprecated", "failed"]
SUPPORTED_OUTPUT_TYPES = ["point_forecast", "interval_forecast", "probability_distribution", "classification", "anomaly_score", "cluster_label"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["feature_engineer", "model_trainer", "forecast_analyst", "scenario_builder", "model_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"models": {"supported_types": SUPPORTED_MODEL_TYPES, "supported_states": SUPPORTED_MODEL_STATES, "require_owner": True, "require_training_data": True, "auto_versioning": True},
	"forecasting": {"supported_horizons": SUPPORTED_FORECAST_HORIZONS, "supported_output_types": SUPPORTED_OUTPUT_TYPES, "confidence_interval_default": 0.95},
	"trends": {"supported_types": SUPPORTED_TREND_TYPES, "decomposition_enabled": True},
	"scenarios": {"supported_types": SUPPORTED_SCENARIO_TYPES, "max_scenarios_per_model": 10},
	"features": {"supported_types": SUPPORTED_FEATURE_TYPES, "max_features": 500, "auto_feature_selection": True},
	"validation": {"supported_methods": SUPPORTED_VALIDATION_METHODS, "min_training_samples": 100},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_model_access_denied": True, "model_explainability_required": False},
	"observability": {"event_stream": PDA_EVENT_STREAM, "stream_processor": "bytewax"},
	"theme": {"default_theme": "bia_pda_predictive", "allow_tenant_overrides": True},
}

PROVIDES = ["ml_model_training", "demand_forecasting", "trend_analysis", "regression_modelling", "scenario_simulation", "anomaly_prediction", "model_versioning", "prediction_serving"]

REQUIRES = ["auth", "audl", "mten", "conf", "schd", "mqeb", "moni", "bia_anl"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/bia/pda/dashboard", "component": "PredictiveDashboard", "permission": "bia_pda:view", "nav_group": "Overview"},
	{"name": "models", "path": "/bia/pda/models", "component": "ModelLibrary", "permission": "bia_pda:models", "nav_group": "Models"},
	{"name": "model_detail", "path": "/bia/pda/models/<id>", "component": "ModelDetail", "permission": "bia_pda:models", "nav_group": "Models"},
	{"name": "model_train", "path": "/bia/pda/models/train", "component": "ModelTrainer", "permission": "bia_pda:train", "nav_group": "Models"},
	{"name": "forecasts", "path": "/bia/pda/forecasts", "component": "ForecastExplorer", "permission": "bia_pda:forecasts", "nav_group": "Forecasting"},
	{"name": "forecast_detail", "path": "/bia/pda/forecasts/<id>", "component": "ForecastDetail", "permission": "bia_pda:forecasts", "nav_group": "Forecasting"},
	{"name": "trends", "path": "/bia/pda/trends", "component": "TrendAnalyser", "permission": "bia_pda:trends", "nav_group": "Analysis"},
	{"name": "scenarios", "path": "/bia/pda/scenarios", "component": "ScenarioBuilder", "permission": "bia_pda:scenarios", "nav_group": "Simulation"},
	{"name": "scenario_detail", "path": "/bia/pda/scenarios/<id>", "component": "ScenarioDetail", "permission": "bia_pda:scenarios", "nav_group": "Simulation"},
	{"name": "features", "path": "/bia/pda/features", "component": "FeatureStore", "permission": "bia_pda:features", "nav_group": "Engineering"},
	{"name": "predictions", "path": "/bia/pda/predictions", "component": "PredictionLog", "permission": "bia_pda:view", "nav_group": "Results"},
	{"name": "audit_log", "path": "/bia/pda/audit", "component": "PredictiveAuditLog", "permission": "bia_pda:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/bia/pda/settings", "component": "PredictiveSettings", "permission": "bia_pda:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "bia_pda_predictive",
	"tokens": {"color.primary": "#7B1FA2", "color.accent": "#00ACC1", "color.success": "#388E3C", "color.warning": "#F57C00", "color.danger": "#D32F2F", "surface.canvas": "#FDF4FF", "surface.panel": "#FFFFFF", "text.primary": "#1A0033", "text.secondary": "#546E7A", "border.radius": "8px", "density": "comfortable"},
	"components": {
		"model": {"icon": "brain", "status_indicator": "model-state-chip"},
		"forecast": {"icon": "trending-up", "status_indicator": "horizon-chip"},
		"scenario": {"icon": "git-branch", "status_indicator": "scenario-type-chip"},
		"feature": {"icon": "layers", "status_indicator": "feature-type-chip"},
		"prediction": {"icon": "target", "status_indicator": "output-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax", "stream": PDA_EVENT_STREAM, "key": "tenant_id",
	"events": ["model_training_started", "model_trained", "model_deployed", "model_deprecated", "forecast_generated", "trend_analysed", "scenario_simulated", "prediction_served", "feature_registered", "anomaly_predicted"],
	"guardrails": ["cross_tenant_model_access_denied", "min_training_sample_enforced", "model_versioning_enforced"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_policy"}},
	{"name": "cross_tenant_model_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_model_access_not_permitted", "required_action": "restrict_to_tenant"}},
	{"name": "model_type_supported", "condition": {"operation": "train_model", "model_type_supported": False}, "effect": {"decision": "deny", "reason": "model_type_not_supported", "required_action": "select_supported_model_type"}},
	{"name": "model_owner_required", "condition": {"operation": "train_model", "owner_present": False}, "effect": {"decision": "deny", "reason": "model_owner_required", "required_action": "attach_model_owner"}},
	{"name": "training_data_required", "condition": {"operation": "train_model", "training_data_present": False}, "effect": {"decision": "deny", "reason": "training_data_required", "required_action": "attach_training_data"}},
	{"name": "min_samples_enforced", "condition": {"operation": "train_model", "sample_count_sufficient": False}, "effect": {"decision": "deny", "reason": "insufficient_training_samples", "required_action": "provide_minimum_100_samples"}},
	{"name": "forecast_horizon_supported", "condition": {"operation": "generate_forecast", "horizon_supported": False}, "effect": {"decision": "deny", "reason": "forecast_horizon_not_supported", "required_action": "select_supported_forecast_horizon"}},
	{"name": "forecast_requires_deployed_model", "condition": {"operation": "generate_forecast", "model_state": "training"}, "effect": {"decision": "deny", "reason": "forecast_requires_deployed_model", "required_action": "wait_for_model_training"}},
	{"name": "scenario_type_supported", "condition": {"operation": "simulate_scenario", "scenario_type_supported": False}, "effect": {"decision": "deny", "reason": "scenario_type_not_supported", "required_action": "select_supported_scenario_type"}},
	{"name": "scenario_limit_enforced", "condition": {"operation": "simulate_scenario", "scenario_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "scenario_limit_exceeded", "required_action": "delete_old_scenario_first"}},
	{"name": "feature_type_supported", "condition": {"operation": "register_feature", "feature_type_supported": False}, "effect": {"decision": "deny", "reason": "feature_type_not_supported", "required_action": "select_supported_feature_type"}},
	{"name": "deprecated_model_cannot_be_deployed", "condition": {"operation": "deploy_model", "model_state": "deprecated"}, "effect": {"decision": "deny", "reason": "deprecated_model_cannot_be_redeployed", "required_action": "train_new_model_version"}},
	{"name": "validation_method_supported", "condition": {"operation": "validate_model", "validation_method_supported": False}, "effect": {"decision": "deny", "reason": "validation_method_not_supported", "required_action": "select_supported_validation_method"}},
	{"name": "trend_type_supported", "condition": {"operation": "analyse_trend", "trend_type_supported": False}, "effect": {"decision": "deny", "reason": "trend_type_not_supported", "required_action": "select_supported_trend_type"}},
	{"name": "output_type_supported", "condition": {"operation": "generate_forecast", "output_type_supported": False}, "effect": {"decision": "deny", "reason": "output_type_not_supported", "required_action": "select_supported_output_type"}},
	{"name": "model_versioning_enforced", "condition": {"operation": "train_model", "versioning_enabled": False}, "effect": {"decision": "deny", "reason": "model_versioning_required", "required_action": "enable_auto_versioning"}},
	{"name": "failed_model_cannot_serve", "condition": {"operation": "serve_prediction", "model_state": "failed"}, "effect": {"decision": "deny", "reason": "failed_model_cannot_serve_predictions", "required_action": "retrain_or_rollback_model"}},
	{"name": "audit_all_predictions", "condition": {"operation": "serve_prediction", "audit_enabled": True}, "effect": {"decision": "allow", "reason": "prediction_serving_audited", "required_action": "emit_prediction_served_event"}},
	{"name": "max_features_enforced", "condition": {"operation": "register_feature", "feature_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "feature_limit_exceeded", "required_action": "remove_unused_features"}},
	{"name": "prediction_requires_input_validation", "condition": {"operation": "serve_prediction", "input_validated": False}, "effect": {"decision": "deny", "reason": "prediction_input_must_be_validated", "required_action": "validate_input_schema"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {"required": ["tenant_id", "ui", "theme"], "properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}}},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["bia/pda/templates"], "routes": UI_ROUTES},
		"theme": THEME, "streaming": STREAMING, "provides": PROVIDES, "requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		if all(context.get(k) == v for k, v in rule["condition"].items()):
			return {"matched_rule": rule["name"], "decision": rule["effect"]["decision"], "reason": rule["effect"]["reason"], "required_action": rule["effect"]["required_action"]}
	return {"matched_rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
