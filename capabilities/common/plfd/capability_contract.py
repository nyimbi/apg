"""Executable capability contract for APG Platform Foundation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"foundation": {"service_owner_required": True, "tier_classification_required": True, "dependency_map_required": True, "readiness_score_required": True},
	"baselines": {"configuration_baseline_required": True, "tenant_baseline_required": True, "auth_baseline_required": True, "audit_baseline_required": True},
	"operations": {"health_gate_required": True, "monitoring_required": True, "rollback_plan_required": True, "change_window_required": True},
	"governance": {"require_tenant_context": True, "audit_foundation_changes": True, "broad_change_review_required": True, "security_review_required": True},
	"ui": {"enable_foundation_dashboard": True, "enable_dependency_map": True, "enable_baseline_manager": True, "enable_readiness_gate": True},
	"theme": {"default_theme": "plfd_platform_foundation", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "foundation", "baselines", "operations", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["foundation", "baselines", "operations", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All platform-foundation operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "foundation_service_requires_owner", "description": "Foundation services require owners.", "condition": {"operation": "register_foundation_service", "service_owner_assigned": False}, "effect": {"decision": "deny", "reason": "service_owner_required", "required_action": "assign_service_owner"}},
	{"name": "dependency_health_required", "description": "Foundation changes require healthy dependencies.", "condition": {"operation": "approve_platform_change", "dependencies_healthy": False}, "effect": {"decision": "deny", "reason": "dependency_health_required", "required_action": "restore_dependency_health"}},
	{"name": "configuration_baseline_required", "description": "Foundation services require configuration baselines.", "condition": {"configuration_baseline_present": False}, "effect": {"decision": "deny", "reason": "configuration_baseline_required", "required_action": "attach_configuration_baseline"}},
	{"name": "platform_change_requires_approval", "description": "Platform foundation changes require approval.", "condition": {"operation": "approve_platform_change", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "platform_change_approval_required", "required_action": "record_platform_approval"}},
	{"name": "broad_platform_change_requires_review", "description": "Broad platform changes require review.", "condition": {"affected_capability_count_gt": 10, "broad_review_recorded": False}, "effect": {"decision": "require_review", "reason": "broad_platform_review_required", "required_action": "review_platform_change"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/plfd/dashboard", "component": "PLFDDashboard", "permission": "plfd:view", "nav_group": "Overview"},
	{"name": "services", "path": "/plfd/services", "component": "FoundationServices", "permission": "plfd:manage_services", "nav_group": "Services"},
	{"name": "dependencies", "path": "/plfd/dependencies", "component": "DependencyMap", "permission": "plfd:view", "nav_group": "Readiness"},
	{"name": "baselines", "path": "/plfd/baselines", "component": "BaselineManager", "permission": "plfd:manage_baselines", "nav_group": "Baselines"},
	{"name": "readiness", "path": "/plfd/readiness", "component": "ReadinessGate", "permission": "plfd:view", "nav_group": "Readiness"},
	{"name": "changes", "path": "/plfd/changes", "component": "PlatformChangeQueue", "permission": "plfd:approve_changes", "nav_group": "Governance"},
	{"name": "governance", "path": "/plfd/governance", "component": "FoundationGovernance", "permission": "plfd:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/plfd/settings", "component": "PLFDSettings", "permission": "plfd:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "plfd_platform_foundation",
	"tokens": {"color.primary": "#2A4365", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"foundation_card": {"icon": "layers", "status_indicator": "tier-pill", "risk_style": "readiness-band"}, "dependency_map": {"visual": "service-graph", "highlight": "health-chip"}, "baseline_manager": {"visual": "policy-grid", "status_style": "baseline-chip"}, "change_queue": {"visual": "approval-lanes", "status_style": "risk-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "plfd", "display_name": "Platform Foundation", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/plfd/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
