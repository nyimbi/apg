"""Executable capability contract for APG Shutdown and Lifecycle Control."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_SHDN_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_SHDN_AGENT_ROLES = [
	"lifecycle_planner",
	"shutdown_reviewer",
	"dependency_reviewer",
	"recovery_reviewer",
	"approval_reviewer",
	"audit_reviewer",
]
SHDN_EVENT_STREAM = "apg.shdn.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"services": {
		"service_owner_required": True,
		"dependency_map_required": True,
		"health_gate_required": True,
		"drain_timeout_seconds": 300,
		"supported_target_types": ["service", "worker", "database", "queue", "tenant_app", "integration"],
	},
	"lifecycle": {
		"plan_required": True,
		"production_approval_required": True,
		"rollback_plan_required": True,
		"restart_sequence_required": True,
		"batch_mutation_review_required": True,
	},
	"recovery": {
		"backup_snapshot_required": True,
		"restore_test_required": True,
		"post_shutdown_health_check_required": True,
		"incident_link_required": True,
	},
	"shdn_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_SHDN_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_SHDN_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_criticality": "normal",
		"disclose_agent_recommendations": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_lifecycle_events": True,
		"force_shutdown_review_required": True,
		"maintenance_window_required": True,
		"state_change_audit_required": True,
	},
	"observability": {
		"event_stream": SHDN_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_plan_events": True,
		"emit_execution_events": True,
		"emit_recovery_events": True,
	},
	"adapters": {
		"event_stream": "bytewax",
		"health": "adapter",
		"backup": "adapter",
		"deployment": "adapter",
		"scheduler": "adapter",
		"audit": "adapter",
	},
	"ui": {
		"enable_service_console": True,
		"enable_plan_builder": True,
		"enable_execution_monitor": True,
		"enable_recovery_center": True,
		"enable_agent_workbench": True,
		"enable_policy_center": True,
	},
	"theme": {"default_theme": "shdn_lifecycle_control", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"services",
		"lifecycle",
		"recovery",
		"shdn_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"services",
			"lifecycle",
			"recovery",
			"shdn_agents",
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
	{
		"name": "tenant_context_required",
		"description": "All lifecycle operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
	},
	{
		"name": "service_requires_owner",
		"description": "Lifecycle-controlled services require an owner.",
		"condition": {"operation": "register_service", "service_owner_assigned": False},
		"effect": {"decision": "deny", "reason": "service_owner_required", "required_action": "assign_service_owner"},
	},
	{
		"name": "service_requires_dependency_map",
		"description": "Lifecycle targets require dependency context before planned shutdown.",
		"condition": {"operation": "create_shutdown_plan", "dependency_map_present": False},
		"effect": {"decision": "deny", "reason": "dependency_map_required", "required_action": "attach_dependency_map"},
	},
	{
		"name": "shutdown_requires_health_gate",
		"description": "Shutdown plans require current health gate evidence.",
		"condition": {"operation": "execute_shutdown", "health_gate_passed": False},
		"effect": {"decision": "deny", "reason": "health_gate_required", "required_action": "run_health_gate"},
	},
	{
		"name": "shutdown_requires_backup_snapshot",
		"description": "Shutdown requires backup snapshot evidence.",
		"condition": {"operation": "execute_shutdown", "backup_snapshot_present": False},
		"effect": {"decision": "deny", "reason": "backup_snapshot_required", "required_action": "capture_backup_snapshot"},
	},
	{
		"name": "shutdown_requires_actor",
		"description": "Shutdown execution requires an accountable actor.",
		"condition": {"operation": "execute_shutdown", "shutdown_actor_present": False},
		"effect": {"decision": "deny", "reason": "shutdown_actor_required", "required_action": "attach_shutdown_actor"},
	},
	{
		"name": "shutdown_requires_bytewax_stream",
		"description": "Shutdown lifecycle events must be emitted through Bytewax.",
		"condition": {"operation": "execute_shutdown", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_shutdown_lifecycle_to_bytewax"},
	},
	{
		"name": "production_shutdown_requires_approval",
		"description": "Production lifecycle changes require approval.",
		"condition": {"production_service": True, "approval_recorded": False},
		"effect": {"decision": "deny", "reason": "production_approval_required", "required_action": "record_production_approval"},
	},
	{
		"name": "force_shutdown_requires_review",
		"description": "Force shutdown requires review.",
		"condition": {"force_shutdown": True, "force_review_recorded": False},
		"effect": {"decision": "require_review", "reason": "force_shutdown_review_required", "required_action": "review_force_shutdown"},
	},
	{
		"name": "recovery_requires_post_health_check",
		"description": "Recovery records require post-shutdown health evidence.",
		"condition": {"operation": "record_recovery", "post_shutdown_health_check_present": False},
		"effect": {"decision": "deny", "reason": "post_shutdown_health_check_required", "required_action": "attach_post_shutdown_health_check"},
	},
	{
		"name": "recovery_requires_incident_link",
		"description": "Recovery records require an incident, change, or work-order reference.",
		"condition": {"operation": "record_recovery", "incident_link_present": False},
		"effect": {"decision": "deny", "reason": "incident_link_required", "required_action": "attach_incident_link"},
	},
	{
		"name": "shdn_agent_runtime_supported",
		"description": "Lifecycle agents must use an approved runtime.",
		"condition": {"operation": "register_shdn_agent", "agent_runtime_supported": False},
		"effect": {"decision": "deny", "reason": "shdn_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"},
	},
	{
		"name": "shdn_agent_role_supported",
		"description": "Lifecycle agents must use an approved role.",
		"condition": {"operation": "register_shdn_agent", "agent_role_supported": False},
		"effect": {"decision": "deny", "reason": "shdn_agent_role_not_supported", "required_action": "select_supported_agent_role"},
	},
	{
		"name": "critical_agent_shutdown_requires_human_approval",
		"description": "Critical shutdown actions proposed by agents require human approval.",
		"condition": {"operation": "agent_lifecycle_action", "target_criticality": "critical", "human_approval_recorded": False},
		"effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"},
	},
	{
		"name": "batch_lifecycle_mutation_requires_bytewax",
		"description": "Batch lifecycle mutations require Bytewax stream coordination.",
		"condition": {"operation": "batch_lifecycle_mutation", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_batch_lifecycle_mutation_to_bytewax"},
	},
	{
		"name": "tenant_context_required",
		"description": "All shutdown and lifecycle operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
	},
	{
		"name": "write_requires_policy",
		"description": "Shutdown write operations require an explicit authorization policy.",
		"condition": {"operation_type": "write", "write_policy_present": False},
		"effect": {"decision": "deny", "reason": "shdn_write_policy_required", "required_action": "attach_write_policy"},
	},
	{
		"name": "privilege_escalation_denied",
		"description": "Shutdown operators cannot self-grant elevated lifecycle control permissions.",
		"condition": {"operation": "assign_shdn_permission", "target_tier_exceeds_actor_tier": True},
		"effect": {"decision": "deny", "reason": "privilege_escalation_prevented", "required_action": "route_to_higher_authority_approver"},
	},
	{
		"name": "cross_tenant_shutdown_denied",
		"description": "Shutdown operations may not target resources in other tenants.",
		"condition": {"cross_tenant_access": True, "cross_tenant_membership_confirmed": False},
		"effect": {"decision": "deny", "reason": "cross_tenant_shutdown_denied", "required_action": "use_tenant_scoped_resource"},
	},
	{
		"name": "shutdown_audit_event_required",
		"description": "All shutdown lifecycle state changes must produce an immutable audit event.",
		"condition": {"shutdown_state_change_requested": True, "audit_event_recorded": False},
		"effect": {"decision": "deny", "reason": "shutdown_audit_event_required", "required_action": "record_shutdown_audit_event"},
	},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/shdn/dashboard", "component": "SHDNDashboard", "permission": "shdn:view", "nav_group": "Overview"},
	{"name": "services", "path": "/shdn/services", "component": "ServiceLifecycleConsole", "permission": "shdn:view", "nav_group": "Services"},
	{"name": "plans", "path": "/shdn/plans", "component": "ShutdownPlanBuilder", "permission": "shdn:plan", "nav_group": "Planning"},
	{"name": "executions", "path": "/shdn/executions", "component": "LifecycleExecutionMonitor", "permission": "shdn:execute", "nav_group": "Execution"},
	{"name": "approvals", "path": "/shdn/approvals", "component": "LifecycleApprovals", "permission": "shdn:approve", "nav_group": "Governance"},
	{"name": "recovery", "path": "/shdn/recovery", "component": "RecoveryCenter", "permission": "shdn:execute", "nav_group": "Recovery"},
	{"name": "agents", "path": "/shdn/agents", "component": "SHDNAgentWorkbench", "permission": "shdn:admin", "nav_group": "Automation"},
	{"name": "policy", "path": "/shdn/policy", "component": "LifecyclePolicyCenter", "permission": "shdn:admin", "nav_group": "Governance"},
	{"name": "audit", "path": "/shdn/audit", "component": "LifecycleAudit", "permission": "shdn:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/shdn/settings", "component": "SHDNSettings", "permission": "shdn:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "shdn_lifecycle_control",
	"tokens": {
		"color.primary": "#234E52",
		"color.accent": "#D69E2E",
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
		"service_card": {"icon": "power", "status_indicator": "state-pill", "risk_style": "lifecycle-band"},
		"plan_builder": {"visual": "sequence-list", "highlight": "gate-chip"},
		"execution_monitor": {"visual": "operation-timeline", "status_style": "health-chip"},
		"recovery_center": {"visual": "backup-checklist", "status_style": "restore-chip"},
		"agent_workbench": {"visual": "approval-lane", "status_style": "review-chip"},
		"policy_center": {"visual": "guardrail-grid", "status_style": "rule-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": SHDN_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"target_registered",
			"plan_created",
			"drain_started",
			"snapshot_recorded",
			"shutdown_executed",
			"recovery_recorded",
			"shdn_agent_registered",
		],
		"states": ["running", "draining", "quiesced", "snapshot_ready", "stopped", "recovered", "blocked"],
		"guardrails": [
			"shutdown_requires_bytewax_stream",
			"batch_lifecycle_mutation_requires_bytewax",
			"critical_agent_shutdown_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return SHDN_EVENT_STREAM


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "shdn",
		"display_name": "Shutdown and Lifecycle Control",
		"version": "1.0.0",
		"provides": [
			"service_lifecycle",
			"shutdown_orchestration",
			"restart_plans",
			"backup_gates",
			"operational_safety",
			"shdn_agents",
		],
		"requires": ["moni", "hlth", "bkup", "audl", "envm"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/shdn/api/v1",
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
			if not context.get(key[:-4], 0) <= expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gte"):
			if not context.get(key[:-4], 0) >= expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
