"""Executable capability contract for APG Deployment Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"releases": {"release_owner_required": True, "manifest_required": True, "approval_required": True, "artifact_signature_required": True},
	"rollouts": {"supported_strategies": ["rolling", "blue_green", "canary"], "health_gate_required": True, "rollback_plan_required": True, "max_canary_percent": 25},
	"evidence": {"log_trace_link_required": True, "health_report_required": True, "deployment_audit_required": True, "change_ticket_required": True},
	"governance": {"require_tenant_context": True, "environment_policy_required": True, "production_approval_required": True, "separation_of_duties_required": True},
	"ui": {"enable_release_console": True, "enable_rollout_monitor": True, "enable_health_gate_view": True, "enable_rollback_center": True},
	"theme": {"default_theme": "depl_release_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "releases", "rollouts", "evidence", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["releases", "rollouts", "evidence", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All deployment operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "release_requires_owner", "description": "Releases require an accountable owner.", "condition": {"operation": "create_release", "release_owner_assigned": False}, "effect": {"decision": "deny", "reason": "release_owner_required", "required_action": "assign_release_owner"}},
	{"name": "deployment_requires_health_gate", "description": "Deployments require a passing health gate.", "condition": {"operation": "deploy", "health_gate_passed": False}, "effect": {"decision": "deny", "reason": "health_gate_required", "required_action": "pass_health_gate"}},
	{"name": "production_requires_approval", "description": "Production deployment requires approval.", "condition": {"target_environment": "production", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "production_approval_required", "required_action": "record_production_approval"}},
	{"name": "rollback_requires_plan", "description": "Deployments require rollback plans.", "condition": {"operation": "deploy", "rollback_plan_attached": False}, "effect": {"decision": "deny", "reason": "rollback_plan_required", "required_action": "attach_rollback_plan"}},
	{"name": "large_canary_requires_review", "description": "Large canary deployments require review.", "condition": {"canary_percent_gt": 25, "canary_review_recorded": False}, "effect": {"decision": "require_review", "reason": "canary_review_required", "required_action": "review_canary_scope"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/depl/dashboard", "component": "DEPLDashboard", "permission": "depl:view", "nav_group": "Overview"},
	{"name": "releases", "path": "/depl/releases", "component": "ReleaseConsole", "permission": "depl:plan", "nav_group": "Releases"},
	{"name": "deployments", "path": "/depl/deployments", "component": "DeploymentMonitor", "permission": "depl:deploy", "nav_group": "Runtime"},
	{"name": "rollouts", "path": "/depl/rollouts", "component": "RolloutStrategies", "permission": "depl:deploy", "nav_group": "Runtime"},
	{"name": "health", "path": "/depl/health", "component": "HealthGates", "permission": "depl:view", "nav_group": "Quality"},
	{"name": "rollback", "path": "/depl/rollback", "component": "RollbackCenter", "permission": "depl:rollback", "nav_group": "Recovery"},
	{"name": "evidence", "path": "/depl/evidence", "component": "DeploymentEvidence", "permission": "depl:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/depl/settings", "component": "DEPLSettings", "permission": "depl:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "depl_release_ops", "tokens": {"color.primary": "#2C5282", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"release_board": {"icon": "rocket", "status_indicator": "release-pill", "risk_style": "approval-band"}, "rollout_monitor": {"visual": "progress-lanes", "highlight": "canary-chip"}, "health_gate": {"visual": "gate-checklist", "status_style": "health-chip"}, "rollback_center": {"visual": "recovery-timeline", "status_style": "rollback-chip"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "depl", "display_name": "Deployment Management", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/depl/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
