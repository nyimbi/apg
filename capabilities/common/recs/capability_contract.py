"""Executable capability contract for APG Recommender Systems."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"models": {
		"enabled_algorithms": ["collaborative_filtering", "content_based", "hybrid", "contextual_bandit"],
		"model_owner_required": True,
		"minimum_training_events": 1000,
		"drift_monitoring_required": True
	},
	"ranking": {
		"ranking_policy_required": True,
		"diversity_constraints_enabled": True,
		"sensitive_attribute_filtering": True,
		"minimum_recommendation_confidence": 0.65
	},
	"experiments": {
		"experiment_approval_required": True,
		"holdout_required": True,
		"business_metric_required": True,
		"max_experiment_percent": 25
	},
	"governance": {
		"require_tenant_context": True,
		"audit_recommendations": True,
		"profile_consent_required": True,
		"explainability_required_for_high_impact": True
	},
	"ui": {
		"enable_recommendation_console": True,
		"enable_model_registry": True,
		"enable_experiment_studio": True,
		"enable_ranking_policy": True
	},
	"theme": {
		"default_theme": "recs_recommendation_console",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "models", "ranking", "experiments", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["models", "ranking", "experiments", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All recommendation operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "profile_consent_required", "description": "Personalized recommendations require profile consent.", "condition": {"operation": "recommend", "profile_consent_recorded": False}, "effect": {"decision": "deny", "reason": "profile_consent_required", "required_action": "record_profile_consent"}},
	{"name": "ranking_policy_required", "description": "Recommendations require an attached ranking policy.", "condition": {"operation": "recommend", "ranking_policy_attached": False}, "effect": {"decision": "deny", "reason": "ranking_policy_required", "required_action": "attach_ranking_policy"}},
	{"name": "model_training_requires_events", "description": "Training requires sufficient events.", "condition": {"operation": "train_model", "training_event_count_lt": 1000}, "effect": {"decision": "deny", "reason": "insufficient_training_events", "required_action": "collect_training_events"}},
	{"name": "high_impact_requires_explainability", "description": "High-impact recommendations require explanations.", "condition": {"impact_level": "high", "explanation_attached": False}, "effect": {"decision": "deny", "reason": "explainability_required", "required_action": "attach_explanation"}},
	{"name": "large_experiment_requires_review", "description": "Large recommendation experiments require review.", "condition": {"experiment_percent_gt": 25, "experiment_review_recorded": False}, "effect": {"decision": "require_review", "reason": "experiment_review_required", "required_action": "review_experiment"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/recs/dashboard", "component": "RECSDashboard", "permission": "recs:view", "nav_group": "Overview"},
	{"name": "recommendations", "path": "/recs/recommendations", "component": "RecommendationConsole", "permission": "recs:recommend", "nav_group": "Runtime"},
	{"name": "models", "path": "/recs/models", "component": "RecommendationModels", "permission": "recs:manage_models", "nav_group": "Models"},
	{"name": "catalogs", "path": "/recs/catalogs", "component": "CatalogManager", "permission": "recs:view", "nav_group": "Data"},
	{"name": "profiles", "path": "/recs/profiles", "component": "ProfileFeatures", "permission": "recs:view", "nav_group": "Data"},
	{"name": "experiments", "path": "/recs/experiments", "component": "ExperimentStudio", "permission": "recs:run_experiments", "nav_group": "Optimization"},
	{"name": "policies", "path": "/recs/policies", "component": "RankingPolicies", "permission": "recs:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/recs/settings", "component": "RECSSettings", "permission": "recs:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "recs_recommendation_console",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#DD6B20",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"recommendation_list": {"icon": "sparkles", "status_indicator": "rank-pill", "risk_style": "policy-band"},
		"model_card": {"visual": "model-score-card", "highlight": "drift-chip"},
		"experiment_board": {"visual": "variant-lanes", "status_style": "metric-chip"},
		"ranking_policy": {"visual": "constraint-stack", "status_style": "guardrail-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable RECS capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "recs",
		"display_name": "Recommender Systems",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/recs/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default RECS governance rules."""
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
