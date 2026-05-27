"""Executable capability contract for APG Predictive Analytics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"forecasting": {"enabled": True, "minimum_history_points": 24, "horizon_limit": 365, "confidence_intervals": True},
	"scoring": {"real_time_scoring_enabled": True, "batch_scoring_enabled": True, "feature_lineage_required": True},
	"models": {"approved_model_required": True, "explainability_required": True, "monitor_drift": True},
	"governance": {"require_tenant_context": True, "audit_predictions": True, "production_approval_required": True},
	"ui": {"enable_forecast_console": True, "enable_scenario_lab": True, "enable_score_monitor": True, "enable_model_board": True},
	"theme": {"default_theme": "pred_forecast_console", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "forecasting", "scoring", "models", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["forecasting", "scoring", "models", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All predictive analytics operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "forecast_requires_history", "description": "Forecasts require enough historical observations.", "condition": {"operation": "create_forecast", "history_points_lt": 24}, "effect": {"decision": "deny", "reason": "insufficient_history", "required_action": "load_more_history"}},
	{"name": "production_score_requires_approved_model", "description": "Production scoring requires an approved model.", "condition": {"environment": "production", "model_approved": False}, "effect": {"decision": "deny", "reason": "approved_model_required", "required_action": "approve_model"}},
	{"name": "scoring_requires_feature_lineage", "description": "Predictive scoring requires feature lineage.", "condition": {"operation": "score", "feature_lineage_present": False}, "effect": {"decision": "deny", "reason": "feature_lineage_required", "required_action": "attach_feature_lineage"}},
	{"name": "high_impact_prediction_requires_explainability", "description": "High-impact predictions require explainability artifacts.", "condition": {"impact": "high", "explainability_attached": False}, "effect": {"decision": "deny", "reason": "explainability_required", "required_action": "attach_explainability"}},
	{"name": "long_horizon_requires_review", "description": "Long forecast horizons require review.", "condition": {"forecast_horizon_days_gt": 365, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "long_horizon_review_required", "required_action": "record_forecast_review"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/pred/dashboard", "component": "PREDDashboard", "permission": "pred:view", "nav_group": "Overview"},
	{"name": "forecasts", "path": "/pred/forecasts", "component": "ForecastConsole", "permission": "pred:forecast", "nav_group": "Forecasts"},
	{"name": "scores", "path": "/pred/scores", "component": "ScoreMonitor", "permission": "pred:score", "nav_group": "Scoring"},
	{"name": "scenarios", "path": "/pred/scenarios", "component": "ScenarioLab", "permission": "pred:simulate", "nav_group": "Simulation"},
	{"name": "models", "path": "/pred/models", "component": "PredictiveModelBoard", "permission": "pred:manage_models", "nav_group": "Models"},
	{"name": "governance", "path": "/pred/governance", "component": "PredictionGovernance", "permission": "pred:govern", "nav_group": "Governance"},
	{"name": "settings", "path": "/pred/settings", "component": "PREDSettings", "permission": "pred:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "pred_forecast_console",
	"tokens": {"color.primary": "#345995", "color.accent": "#E07A5F", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F6F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"forecast_chart": {"icon": "trending-up", "visual": "confidence-band", "status_indicator": "horizon-chip"},
		"score_card": {"visual": "distribution-bar", "risk_style": "impact-band"},
		"scenario_matrix": {"visual": "comparison-grid", "highlight": "delta-chip"},
		"feature_lineage_panel": {"visual": "lineage-list", "status_style": "evidence-pill"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "pred", "display_name": "Predictive Analytics", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "__init__.py", "api_prefix": "/pred/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
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
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
