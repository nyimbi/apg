"""Executable capability contract for APG Shutdown and Lifecycle Control."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"services": {"service_owner_required": True, "dependency_map_required": True, "health_gate_required": True, "drain_timeout_seconds": 300},
	"lifecycle": {"plan_required": True, "production_approval_required": True, "rollback_plan_required": True, "restart_sequence_required": True},
	"recovery": {"backup_snapshot_required": True, "restore_test_required": True, "post_shutdown_health_check_required": True, "incident_link_required": True},
	"governance": {"require_tenant_context": True, "audit_lifecycle_events": True, "force_shutdown_review_required": True, "maintenance_window_required": True},
	"ui": {"enable_service_console": True, "enable_plan_builder": True, "enable_execution_monitor": True, "enable_recovery_center": True},
	"theme": {"default_theme": "shdn_lifecycle_control", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "services", "lifecycle", "recovery", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["services", "lifecycle", "recovery", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All lifecycle operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "service_requires_owner", "description": "Lifecycle-controlled services require an owner.", "condition": {"operation": "register_service", "service_owner_assigned": False}, "effect": {"decision": "deny", "reason": "service_owner_required", "required_action": "assign_service_owner"}},
	{"name": "shutdown_requires_health_gate", "description": "Shutdown plans require current health gate evidence.", "condition": {"operation": "execute_shutdown", "health_gate_passed": False}, "effect": {"decision": "deny", "reason": "health_gate_required", "required_action": "run_health_gate"}},
	{"name": "shutdown_requires_backup_snapshot", "description": "Shutdown requires backup snapshot evidence.", "condition": {"operation": "execute_shutdown", "backup_snapshot_present": False}, "effect": {"decision": "deny", "reason": "backup_snapshot_required", "required_action": "capture_backup_snapshot"}},
	{"name": "production_shutdown_requires_approval", "description": "Production lifecycle changes require approval.", "condition": {"production_service": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "production_approval_required", "required_action": "record_production_approval"}},
	{"name": "force_shutdown_requires_review", "description": "Force shutdown requires review.", "condition": {"force_shutdown": True, "force_review_recorded": False}, "effect": {"decision": "require_review", "reason": "force_shutdown_review_required", "required_action": "review_force_shutdown"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/shdn/dashboard", "component": "SHDNDashboard", "permission": "shdn:view", "nav_group": "Overview"},
	{"name": "services", "path": "/shdn/services", "component": "ServiceLifecycleConsole", "permission": "shdn:view", "nav_group": "Services"},
	{"name": "plans", "path": "/shdn/plans", "component": "ShutdownPlanBuilder", "permission": "shdn:plan", "nav_group": "Planning"},
	{"name": "executions", "path": "/shdn/executions", "component": "LifecycleExecutionMonitor", "permission": "shdn:execute", "nav_group": "Execution"},
	{"name": "approvals", "path": "/shdn/approvals", "component": "LifecycleApprovals", "permission": "shdn:approve", "nav_group": "Governance"},
	{"name": "recovery", "path": "/shdn/recovery", "component": "RecoveryCenter", "permission": "shdn:execute", "nav_group": "Recovery"},
	{"name": "audit", "path": "/shdn/audit", "component": "LifecycleAudit", "permission": "shdn:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/shdn/settings", "component": "SHDNSettings", "permission": "shdn:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "shdn_lifecycle_control",
	"tokens": {"color.primary": "#234E52", "color.accent": "#D69E2E", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"service_card": {"icon": "power", "status_indicator": "state-pill", "risk_style": "lifecycle-band"}, "plan_builder": {"visual": "sequence-list", "highlight": "gate-chip"}, "execution_monitor": {"visual": "operation-timeline", "status_style": "health-chip"}, "recovery_center": {"visual": "backup-checklist", "status_style": "restore-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "shdn", "display_name": "Shutdown and Lifecycle Control", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/shdn/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
