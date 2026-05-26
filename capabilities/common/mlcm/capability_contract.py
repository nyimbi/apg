"""Executable capability contract for APG AI Model Lifecycle Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"registry": {"model_registry_enabled": True, "owner_required": True, "versioning_required": True},
	"promotion": {"stage_gates": ["dev", "staging", "production"], "approval_required_for_production": True, "rollback_enabled": True},
	"evaluation": {"baseline_required": True, "drift_monitoring_enabled": True, "minimum_eval_score": 0.8},
	"governance": {"require_tenant_context": True, "audit_model_changes": True, "model_card_required": True, "risk_review_required": True},
	"ui": {"enable_registry": True, "enable_evaluation_console": True, "enable_deployment_board": True, "enable_drift_monitor": True},
	"theme": {"default_theme": "mlcm_model_ops_console", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "registry", "promotion", "evaluation", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["registry", "promotion", "evaluation", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All model lifecycle operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "model_registration_requires_owner", "description": "Model registration requires an owner.", "condition": {"operation": "register_model", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "model_owner_required", "required_action": "assign_model_owner"}},
	{"name": "production_promotion_requires_approval", "description": "Production model promotion requires approval.", "condition": {"target_stage": "production", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "promotion_approval_required", "required_action": "record_promotion_approval"}},
	{"name": "deployment_requires_model_card", "description": "Model deployments require model-card documentation.", "condition": {"operation": "deploy_model", "model_card_present": False}, "effect": {"decision": "deny", "reason": "model_card_required", "required_action": "attach_model_card"}},
	{"name": "low_eval_score_blocks_promotion", "description": "Low evaluation scores block promotion.", "condition": {"eval_score_lt": 0.8, "promotion_requested": True}, "effect": {"decision": "deny", "reason": "evaluation_score_too_low", "required_action": "improve_or_waive_evaluation"}},
	{"name": "drifted_model_requires_review", "description": "Drifted models require review before continued serving.", "condition": {"drift_detected": True, "drift_review_recorded": False}, "effect": {"decision": "require_review", "reason": "drift_review_required", "required_action": "record_drift_review"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/mlcm/dashboard", "component": "MLCMDashboard", "permission": "mlcm:view", "nav_group": "Overview"},
	{"name": "registry", "path": "/mlcm/models", "component": "ModelRegistry", "permission": "mlcm:view_models", "nav_group": "Registry"},
	{"name": "versions", "path": "/mlcm/versions", "component": "ModelVersionManager", "permission": "mlcm:manage_models", "nav_group": "Registry"},
	{"name": "evaluation", "path": "/mlcm/evaluation", "component": "EvaluationConsole", "permission": "mlcm:evaluate", "nav_group": "Quality"},
	{"name": "deployments", "path": "/mlcm/deployments", "component": "DeploymentBoard", "permission": "mlcm:deploy", "nav_group": "Operations"},
	{"name": "drift", "path": "/mlcm/drift", "component": "DriftMonitor", "permission": "mlcm:view_drift", "nav_group": "Operations"},
	{"name": "governance", "path": "/mlcm/governance", "component": "ModelGovernance", "permission": "mlcm:govern", "nav_group": "Governance"},
	{"name": "settings", "path": "/mlcm/settings", "component": "MLCMSettings", "permission": "mlcm:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "mlcm_model_ops_console",
	"tokens": {"color.primary": "#244B5A", "color.accent": "#D97706", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"model_version_row": {"icon": "layers", "status_indicator": "stage-pill", "risk_style": "eval-band"},
		"promotion_gate_panel": {"visual": "gate-stack", "highlight": "approval-chip"},
		"drift_monitor_chart": {"visual": "time-series-grid", "threshold_style": "drift-lines"},
		"model_card_panel": {"visual": "evidence-list", "status_style": "completeness-pill"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "mlcm", "display_name": "AI Model Lifecycle Management", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "__init__.py", "api_prefix": "/mlcm/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
