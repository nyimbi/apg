"""Executable capability contract for APG Edge Computing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"nodes": {"node_owner_required": True, "attestation_required": True, "health_check_required": True, "location_policy_required": True},
	"workloads": {"deployment_policy_required": True, "resource_quota_required": True, "offline_mode_supported": True, "artifact_signature_required": True},
	"sync": {"conflict_policy_required": True, "cache_policy_required": True, "max_offline_hours": 72, "event_replay_supported": True},
	"governance": {"require_tenant_context": True, "audit_edge_events": True, "configuration_policy_required": True, "secure_transport_required": True},
	"ui": {"enable_edge_dashboard": True, "enable_node_manager": True, "enable_workload_console": True, "enable_sync_monitor": True},
	"theme": {"default_theme": "edge_operations_console", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "nodes", "workloads", "sync", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["nodes", "workloads", "sync", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All edge operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "node_requires_attestation", "description": "Edge nodes require attestation.", "condition": {"operation": "register_node", "node_attested": False}, "effect": {"decision": "deny", "reason": "node_attestation_required", "required_action": "attest_node"}},
	{"name": "workload_requires_signed_artifact", "description": "Edge workloads require signed artifacts.", "condition": {"operation": "deploy_workload", "artifact_signed": False}, "effect": {"decision": "deny", "reason": "artifact_signature_required", "required_action": "sign_artifact"}},
	{"name": "sync_requires_conflict_policy", "description": "Edge sync requires conflict policy.", "condition": {"operation": "sync_state", "conflict_policy_attached": False}, "effect": {"decision": "deny", "reason": "conflict_policy_required", "required_action": "attach_conflict_policy"}},
	{"name": "edge_transport_requires_security", "description": "Edge traffic requires secure transport.", "condition": {"edge_connection": True, "secure_transport": False}, "effect": {"decision": "deny", "reason": "secure_transport_required", "required_action": "enable_secure_transport"}},
	{"name": "long_offline_window_requires_review", "description": "Long offline windows require review.", "condition": {"offline_hours_gt": 72, "offline_review_recorded": False}, "effect": {"decision": "require_review", "reason": "offline_review_required", "required_action": "review_offline_window"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/edge/dashboard", "component": "EDGEDashboard", "permission": "edge:view", "nav_group": "Overview"},
	{"name": "nodes", "path": "/edge/nodes", "component": "NodeManager", "permission": "edge:manage_nodes", "nav_group": "Nodes"},
	{"name": "fleets", "path": "/edge/fleets", "component": "FleetManager", "permission": "edge:manage_nodes", "nav_group": "Nodes"},
	{"name": "workloads", "path": "/edge/workloads", "component": "WorkloadConsole", "permission": "edge:deploy_workloads", "nav_group": "Workloads"},
	{"name": "deployments", "path": "/edge/deployments", "component": "EdgeDeployments", "permission": "edge:deploy_workloads", "nav_group": "Workloads"},
	{"name": "sync", "path": "/edge/sync", "component": "SyncMonitor", "permission": "edge:sync", "nav_group": "Synchronization"},
	{"name": "analytics", "path": "/edge/analytics", "component": "EdgeAnalytics", "permission": "edge:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/edge/settings", "component": "EDGESettings", "permission": "edge:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "edge_operations_console", "tokens": {"color.primary": "#214E34", "color.accent": "#2B6CB0", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"node_map": {"icon": "router", "status_indicator": "node-pill", "risk_style": "attestation-band"}, "workload_table": {"visual": "edge-workload-list", "highlight": "signature-chip"}, "sync_monitor": {"visual": "sync-timeline", "status_style": "conflict-chip"}, "fleet_panel": {"visual": "fleet-grid", "status_style": "health-chip"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "edge", "display_name": "Edge Computing", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/edge/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
