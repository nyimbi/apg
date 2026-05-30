"""Executable capability contract for APG Tenants Legacy."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_TENS_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_TENS_AGENT_ROLES = [
	"tenant_mapper",
	"boundary_reviewer",
	"migration_reviewer",
	"deprecation_reviewer",
	"compatibility_reviewer",
	"audit_reviewer",
]
TENS_EVENT_STREAM = "apg.tens.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"legacy_mapping": {
		"legacy_owner_required": True,
		"mapping_validation_required": True,
		"source_system_required": True,
		"compatibility_scope_required": True,
		"batch_mapping_review_required": True,
	},
	"migration": {
		"migration_plan_required": True,
		"approval_required": True,
		"rollback_plan_required": True,
		"post_migration_validation_required": True,
		"migration_stream_required": True,
	},
	"access": {
		"auth_boundary_required": True,
		"tenant_isolation_validation_required": True,
		"legacy_role_mapping_required": True,
		"privileged_access_review_required": True,
	},
	"tens_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_TENS_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_TENS_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "non_privileged",
		"disclose_agent_recommendations": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_legacy_tenant_changes": True,
		"stale_tenant_review_days": 180,
		"deprecation_plan_required": True,
		"state_change_audit_required": True,
	},
	"observability": {
		"event_stream": TENS_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_mapping_events": True,
		"emit_migration_events": True,
		"emit_deprecation_events": True,
	},
	"adapters": {
		"event_stream": "bytewax",
		"identity": "adapter",
		"access_control": "adapter",
		"audit": "adapter",
		"migration": "adapter",
	},
	"ui": {
		"enable_legacy_tenant_registry": True,
		"enable_mapping_workbench": True,
		"enable_migration_queue": True,
		"enable_boundary_review": True,
		"enable_agent_workbench": True,
		"enable_policy_center": True,
	},
	"theme": {"default_theme": "tens_legacy_tenant_migration", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"legacy_mapping",
		"migration",
		"access",
		"tens_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"legacy_mapping",
			"migration",
			"access",
			"tens_agents",
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
		"description": "All legacy tenant operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
	},
	{
		"name": "legacy_tenant_requires_owner",
		"description": "Legacy tenants require an accountable owner.",
		"condition": {"operation": "register_legacy_tenant", "legacy_owner_assigned": False},
		"effect": {"decision": "deny", "reason": "legacy_owner_required", "required_action": "assign_legacy_owner"},
	},
	{
		"name": "legacy_tenant_requires_source_system",
		"description": "Legacy tenants require source-system lineage.",
		"condition": {"operation": "register_legacy_tenant", "source_system_present": False},
		"effect": {"decision": "deny", "reason": "legacy_source_system_required", "required_action": "attach_source_system"},
	},
	{
		"name": "legacy_tenant_requires_compatibility_scope",
		"description": "Legacy tenants require an explicit compatibility scope.",
		"condition": {"operation": "register_legacy_tenant", "compatibility_scope_present": False},
		"effect": {"decision": "deny", "reason": "compatibility_scope_required", "required_action": "attach_compatibility_scope"},
	},
	{
		"name": "mapping_requires_validation",
		"description": "Tenant mappings require validation.",
		"condition": {"operation": "map_tenant", "mapping_validated": False},
		"effect": {"decision": "deny", "reason": "mapping_validation_required", "required_action": "validate_tenant_mapping"},
	},
	{
		"name": "mapping_requires_bytewax_stream",
		"description": "Tenant mapping lifecycle events must be emitted through Bytewax.",
		"condition": {"operation": "map_tenant", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_mapping_lifecycle_to_bytewax"},
	},
	{
		"name": "migration_requires_approval",
		"description": "Legacy tenant migrations require approval.",
		"condition": {"operation": "migrate_tenant", "approval_recorded": False},
		"effect": {"decision": "deny", "reason": "migration_approval_required", "required_action": "record_migration_approval"},
	},
	{
		"name": "migration_requires_rollback_plan",
		"description": "Legacy tenant migrations require rollback planning.",
		"condition": {"operation": "migrate_tenant", "rollback_plan_present": False},
		"effect": {"decision": "deny", "reason": "rollback_plan_required", "required_action": "attach_rollback_plan"},
	},
	{
		"name": "migration_completion_requires_post_validation",
		"description": "Completed migrations require post-migration validation.",
		"condition": {"operation": "complete_migration", "post_migration_validation_present": False},
		"effect": {"decision": "deny", "reason": "post_migration_validation_required", "required_action": "attach_post_migration_validation"},
	},
	{
		"name": "migration_completion_requires_bytewax_stream",
		"description": "Migration completion lifecycle events must be emitted through Bytewax.",
		"condition": {"operation": "complete_migration", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_migration_lifecycle_to_bytewax"},
	},
	{
		"name": "access_boundary_required",
		"description": "Legacy tenant access requires validated auth boundaries.",
		"condition": {"operation": "validate_access_boundary", "auth_boundary_validated": False},
		"effect": {"decision": "deny", "reason": "auth_boundary_required", "required_action": "validate_auth_boundary"},
	},
	{
		"name": "role_mapping_required",
		"description": "Legacy tenant access requires role mapping evidence.",
		"condition": {"operation": "validate_access_boundary", "role_mapping_present": False},
		"effect": {"decision": "deny", "reason": "legacy_role_mapping_required", "required_action": "attach_role_mapping"},
	},
	{
		"name": "isolation_validation_required",
		"description": "Legacy tenant access requires tenant-isolation evidence.",
		"condition": {"operation": "validate_access_boundary", "isolation_validation_present": False},
		"effect": {"decision": "deny", "reason": "tenant_isolation_validation_required", "required_action": "attach_tenant_isolation_validation"},
	},
	{
		"name": "privileged_access_review_required",
		"description": "Legacy privileged access requires review evidence.",
		"condition": {"operation": "validate_access_boundary", "privileged_review_present": False},
		"effect": {"decision": "deny", "reason": "privileged_access_review_required", "required_action": "attach_privileged_access_review"},
	},
	{
		"name": "stale_legacy_tenant_requires_review",
		"description": "Stale legacy tenants require review.",
		"condition": {"days_since_activity_gt": 180, "stale_review_recorded": False},
		"effect": {"decision": "require_review", "reason": "stale_legacy_tenant_review_required", "required_action": "review_legacy_tenant"},
	},
	{
		"name": "tens_agent_runtime_supported",
		"description": "Legacy tenant agents must use an approved runtime.",
		"condition": {"operation": "register_tens_agent", "agent_runtime_supported": False},
		"effect": {"decision": "deny", "reason": "tens_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"},
	},
	{
		"name": "tens_agent_role_supported",
		"description": "Legacy tenant agents must use an approved role.",
		"condition": {"operation": "register_tens_agent", "agent_role_supported": False},
		"effect": {"decision": "deny", "reason": "tens_agent_role_not_supported", "required_action": "select_supported_agent_role"},
	},
	{
		"name": "privileged_agent_mapping_requires_human_approval",
		"description": "Privileged tenant mapping actions proposed by agents require human approval.",
		"condition": {"operation": "agent_tenant_action", "privileged_scope": True, "human_approval_recorded": False},
		"effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"},
	},
	{
		"name": "batch_tenant_mapping_requires_bytewax",
		"description": "Batch legacy tenant mapping requires Bytewax stream coordination.",
		"condition": {"operation": "batch_tenant_mapping", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_batch_tenant_mapping_to_bytewax"},
	},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/tens/dashboard", "component": "TENSDashboard", "permission": "tens:view", "nav_group": "Overview"},
	{"name": "tenants", "path": "/tens/tenants", "component": "LegacyTenantRegistry", "permission": "tens:view", "nav_group": "Tenants"},
	{"name": "mappings", "path": "/tens/mappings", "component": "TenantMappingWorkbench", "permission": "tens:map", "nav_group": "Mapping"},
	{"name": "migrations", "path": "/tens/migrations", "component": "MigrationQueue", "permission": "tens:migrate", "nav_group": "Migration"},
	{"name": "boundaries", "path": "/tens/boundaries", "component": "BoundaryReview", "permission": "tens:approve", "nav_group": "Access"},
	{"name": "deprecation", "path": "/tens/deprecation", "component": "DeprecationPlan", "permission": "tens:approve", "nav_group": "Governance"},
	{"name": "agents", "path": "/tens/agents", "component": "TENSAgentWorkbench", "permission": "tens:admin", "nav_group": "Automation"},
	{"name": "policy", "path": "/tens/policy", "component": "TENSPolicyCenter", "permission": "tens:admin", "nav_group": "Governance"},
	{"name": "audit", "path": "/tens/audit", "component": "LegacyTenantAudit", "permission": "tens:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/tens/settings", "component": "TENSSettings", "permission": "tens:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "tens_legacy_tenant_migration",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#B7791F",
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
		"tenant_card": {"icon": "building-2", "status_indicator": "legacy-pill", "risk_style": "migration-band"},
		"mapping_workbench": {"visual": "mapping-table", "highlight": "validation-chip"},
		"migration_queue": {"visual": "migration-lanes", "status_style": "approval-chip"},
		"boundary_review": {"visual": "access-matrix", "status_style": "isolation-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
		"policy_center": {"visual": "rule-grid", "status_style": "guardrail-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": TENS_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"legacy_tenant_registered",
			"tenant_mapped",
			"boundary_validated",
			"migration_plan_created",
			"migration_completed",
			"deprecation_planned",
			"tens_agent_registered",
		],
		"states": ["active", "stale", "mapped", "migration_ready", "migrated", "deprecated", "blocked"],
		"guardrails": [
			"mapping_requires_bytewax_stream",
			"migration_completion_requires_bytewax_stream",
			"batch_tenant_mapping_requires_bytewax",
			"privileged_agent_mapping_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return TENS_EVENT_STREAM


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "tens",
		"display_name": "Tenants Legacy",
		"version": "1.0.0",
		"provides": [
			"legacy_tenant_registry",
			"tenant_mapping",
			"migration_controls",
			"access_boundaries",
			"deprecation_governance",
			"tens_agents",
		],
		"requires": ["mten", "auth", "audl", "idfd", "usrm"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/tens/api/v1",
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
