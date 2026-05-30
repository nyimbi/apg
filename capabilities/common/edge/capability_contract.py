"""Executable capability contract for APG Edge Computing."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_EDGE_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_EDGE_AGENT_ROLES = [
	"fleet_optimizer",
	"node_health_reviewer",
	"workload_placement_reviewer",
	"offline_sync_reviewer",
	"security_reviewer",
]
SUPPORTED_NODE_TYPES = ["gateway", "compute", "sensor_hub", "cache", "inference"]
SUPPORTED_RUNTIME_MODES = ["online", "offline", "hybrid", "store_and_forward"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"nodes": {
		"node_owner_required": True,
		"attestation_required": True,
		"health_check_required": True,
		"location_policy_required": True,
		"secure_transport_required": True,
		"supported_node_types": SUPPORTED_NODE_TYPES,
	},
	"fleets": {
		"fleet_owner_required": True,
		"policy_version_required": True,
		"node_membership_audit_required": True,
	},
	"workloads": {
		"deployment_policy_required": True,
		"resource_quota_required": True,
		"offline_mode_supported": True,
		"artifact_signature_required": True,
		"supported_runtime_modes": SUPPORTED_RUNTIME_MODES,
	},
	"sync": {
		"conflict_policy_required": True,
		"cache_policy_required": True,
		"max_offline_hours": 72,
		"event_replay_supported": True,
		"offline_review_required_after_hours": 72,
	},
	"edge_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_role_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_EDGE_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_EDGE_AGENT_ROLES,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_edge_events": True,
		"configuration_policy_required": True,
		"secure_transport_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"health_metrics_required": True,
		"resource_pressure_required": True,
		"sync_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.EdgeService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"audit_sink": "audl",
		"identity": "auth",
		"configuration": "conf",
		"distribution": "dist",
		"cache": "cach",
		"monitoring": "moni",
		"geospatial": "geos",
	},
	"ui": {
		"enable_edge_dashboard": True,
		"enable_node_manager": True,
		"enable_fleet_manager": True,
		"enable_workload_console": True,
		"enable_deployments": True,
		"enable_sync_monitor": True,
		"enable_agent_panel": True,
		"enable_rules": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {
		"default_theme": "edge_operations_console",
		"allow_tenant_overrides": True,
	},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"nodes",
		"fleets",
		"workloads",
		"sync",
		"edge_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"nodes",
			"fleets",
			"workloads",
			"sync",
			"edge_agents",
			"governance",
			"observability",
			"adapters",
			"ui",
			"theme",
		]
	}
	| {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All edge operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "node_requires_owner", "description": "Edge nodes require accountable ownership.", "condition": {"operation": "register_node", "node_owner_present": False}, "effect": {"decision": "deny", "reason": "node_owner_required", "required_action": "assign_node_owner"}},
	{"name": "node_requires_attestation", "description": "Edge nodes require attestation.", "condition": {"operation": "register_node", "node_attested": False}, "effect": {"decision": "deny", "reason": "node_attestation_required", "required_action": "attest_node"}},
	{"name": "node_requires_location_policy", "description": "Edge nodes require location policy.", "condition": {"operation": "register_node", "location_policy_present": False}, "effect": {"decision": "deny", "reason": "location_policy_required", "required_action": "attach_location_policy"}},
	{"name": "fleet_requires_owner", "description": "Edge fleets require accountable ownership.", "condition": {"operation": "create_fleet", "fleet_owner_present": False}, "effect": {"decision": "deny", "reason": "fleet_owner_required", "required_action": "assign_fleet_owner"}},
	{"name": "fleet_requires_policy_version", "description": "Edge fleets require policy version.", "condition": {"operation": "create_fleet", "policy_version_present": False}, "effect": {"decision": "deny", "reason": "fleet_policy_version_required", "required_action": "set_fleet_policy_version"}},
	{"name": "workload_requires_owner", "description": "Edge workloads require accountable ownership.", "condition": {"operation": "deploy_workload", "workload_owner_present": False}, "effect": {"decision": "deny", "reason": "workload_owner_required", "required_action": "assign_workload_owner"}},
	{"name": "workload_requires_signed_artifact", "description": "Edge workloads require signed artifacts.", "condition": {"operation": "deploy_workload", "artifact_signed": False}, "effect": {"decision": "deny", "reason": "artifact_signature_required", "required_action": "sign_artifact"}},
	{"name": "workload_requires_resource_quota", "description": "Edge workloads require resource quota.", "condition": {"operation": "deploy_workload", "resource_quota_present": False}, "effect": {"decision": "deny", "reason": "resource_quota_required", "required_action": "attach_resource_quota"}},
	{"name": "sync_requires_conflict_policy", "description": "Edge sync requires conflict policy.", "condition": {"operation": "sync_state", "conflict_policy_attached": False}, "effect": {"decision": "deny", "reason": "conflict_policy_required", "required_action": "attach_conflict_policy"}},
	{"name": "sync_requires_cache_policy", "description": "Edge sync requires cache policy.", "condition": {"operation": "sync_state", "cache_policy_attached": False}, "effect": {"decision": "deny", "reason": "cache_policy_required", "required_action": "attach_cache_policy"}},
	{"name": "edge_transport_requires_security", "description": "Edge traffic requires secure transport.", "condition": {"edge_connection": True, "secure_transport": False}, "effect": {"decision": "deny", "reason": "secure_transport_required", "required_action": "enable_secure_transport"}},
	{"name": "long_offline_window_requires_review", "description": "Long offline windows require review.", "condition": {"offline_hours_gt": 72, "offline_review_recorded": False}, "effect": {"decision": "require_review", "reason": "offline_review_required", "required_action": "review_offline_window"}},
	{"name": "edge_agent_requires_registration", "description": "AI edge agents must be registered.", "condition": {"edge_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "edge_agent_registration_required", "required_action": "register_edge_agent"}},
	{"name": "edge_agent_runtime_supported", "description": "AI edge agents must use a supported runtime.", "condition": {"edge_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "edge_agent_runtime_not_supported", "required_action": "choose_supported_edge_agent_runtime"}},
	{"name": "edge_agent_role_supported", "description": "AI edge agents must use a supported role.", "condition": {"edge_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "edge_agent_role_not_supported", "required_action": "choose_supported_edge_agent_role"}},
	{"name": "edge_agent_requires_scope", "description": "AI edge agents require explicit scope.", "condition": {"edge_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "edge_agent_scope_required", "required_action": "set_edge_agent_scope"}},
	{"name": "edge_agent_requires_disclosure", "description": "AI edge-agent contributions require disclosure.", "condition": {"edge_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "edge_agent_disclosure_required", "required_action": "disclose_edge_agent"}},
	{"name": "edge_state_change_requires_audit", "description": "Edge lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "edge_audit_event_required", "required_action": "record_edge_audit_event"}},
	{"name": "batch_edge_mutation_requires_bytewax", "description": "Batch edge mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_edge_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/edge/dashboard", "component": "EdgeDashboard", "permission": "edge:view", "nav_group": "Overview"},
	{"name": "nodes", "path": "/edge/nodes", "component": "NodeManager", "permission": "edge:manage_nodes", "nav_group": "Nodes"},
	{"name": "fleets", "path": "/edge/fleets", "component": "FleetManager", "permission": "edge:manage_nodes", "nav_group": "Nodes"},
	{"name": "workloads", "path": "/edge/workloads", "component": "WorkloadConsole", "permission": "edge:deploy_workloads", "nav_group": "Workloads"},
	{"name": "deployments", "path": "/edge/deployments", "component": "EdgeDeployments", "permission": "edge:deploy_workloads", "nav_group": "Workloads"},
	{"name": "sync", "path": "/edge/sync", "component": "SyncMonitor", "permission": "edge:sync", "nav_group": "Synchronization"},
	{"name": "agents", "path": "/edge/agents", "component": "EdgeAgentPanel", "permission": "edge:govern", "nav_group": "Governance"},
	{"name": "rules", "path": "/edge/rules", "component": "EdgeRules", "permission": "edge:govern", "nav_group": "Governance"},
	{"name": "analytics", "path": "/edge/analytics", "component": "EdgeAnalytics", "permission": "edge:view", "nav_group": "Operations"},
	{"name": "audit", "path": "/edge/audit", "component": "EdgeAudit", "permission": "edge:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/edge/settings", "component": "EdgeSettings", "permission": "edge:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "edge_operations_console",
	"tokens": {
		"color.primary": "#214E34",
		"color.accent": "#2B6CB0",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"node_map": {"icon": "router", "status_indicator": "node-pill", "risk_style": "attestation-band"},
		"fleet_panel": {"visual": "fleet-grid", "status_style": "health-chip"},
		"workload_table": {"visual": "edge-workload-list", "highlight": "signature-chip"},
		"deployment_table": {"visual": "placement-list", "status_style": "runtime-chip"},
		"sync_monitor": {"visual": "sync-timeline", "status_style": "conflict-chip"},
		"edge_agent_panel": {"icon": "bot", "status_indicator": "scope-chip"},
		"stream_health": {"visual": "event-lane", "status_style": "stream-chip"},
		"audit": {"visual": "event-ledger", "status_style": "digest-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.edge.lifecycle",
		"state": ["nodes", "fleets", "workloads", "deployments", "sync_sessions", "edge_agents", "audit_events"],
		"events": [
			"edge_node_registered",
			"edge_fleet_created",
			"edge_workload_registered",
			"edge_workload_deployed",
			"edge_sync_completed",
			"edge_offline_window_reviewed",
			"edge_agent_registered",
		],
		"batch_mutation_guardrail": "batch_edge_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "edge",
		"display_name": "Edge Computing",
		"version": "1.0.0",
		"provides": [
			"edge_nodes",
			"edge_fleets",
			"edge_workloads",
			"edge_deployments",
			"offline_execution",
			"edge_sync",
			"edge_agents",
		],
		"requires": ["auth", "conf", "audl", "dist", "cach", "moni"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/edge/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


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
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
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
