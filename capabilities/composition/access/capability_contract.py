"""Executable capability contract for APG composition access control."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_ACCESS_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ACCESS_AGENT_ROLES = [
	"access_architect",
	"policy_reviewer",
	"grant_reviewer",
	"risk_reviewer",
	"session_reviewer",
	"audit_reviewer",
]
ACCESS_EVENT_STREAM = "apg.composition.access.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"identity_providers": {
		"supported_types": ["local", "oidc", "saml", "ldap", "api_key", "jwt"],
		"provider_owner_required": True,
		"metadata_validation_required": True,
		"secret_reference_required": True,
		"test_evidence_required": True,
	},
	"resources": {
		"registry_required": True,
		"owner_required": True,
		"scope_required": True,
		"composition_boundary_required": True,
	},
	"policies": {
		"policy_owner_required": True,
		"effect_required": True,
		"conditions_required_for_sensitive_resources": True,
		"simulation_required_for_high_risk": True,
		"review_required_for_deny_override": True,
	},
	"grants": {
		"approval_required_for_privileged": True,
		"separation_of_duties_required": True,
		"expiry_required_for_elevated_access": True,
		"justification_required": True,
	},
	"sessions": {
		"risk_scoring_enabled": True,
		"adaptive_step_up_enabled": True,
		"max_risk_without_review": 74,
		"continuous_evaluation_enabled": True,
	},
	"access_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_ACCESS_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_ACCESS_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "read_and_recommend",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"cross_capability_guardrails": True,
		"privileged_action_review": True,
	},
	"observability": {
		"event_stream": ACCESS_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_provider_events": True,
		"emit_policy_events": True,
		"emit_grant_events": True,
		"emit_decision_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"identity": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_provider_console": True,
		"enable_policy_studio": True,
		"enable_grant_workbench": True,
		"enable_decision_explorer": True,
		"enable_agent_workbench": True,
		"enable_audit_console": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "composition_access_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"identity_providers",
		"resources",
		"policies",
		"grants",
		"sessions",
		"access_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"identity_providers": {"type": "object"},
		"resources": {"type": "object"},
		"policies": {"type": "object"},
		"grants": {"type": "object"},
		"sessions": {"type": "object"},
		"access_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All composition access operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "provider_requires_owner", "description": "Identity providers require an accountable owner.", "condition": {"operation": "register_provider", "provider_owner_assigned": False}, "effect": {"decision": "deny", "reason": "provider_owner_required", "required_action": "assign_provider_owner"}},
	{"name": "provider_requires_metadata_evidence", "description": "Identity providers require validated metadata and test evidence before activation.", "condition": {"operation": "activate_provider", "provider_metadata_validated": False}, "effect": {"decision": "deny", "reason": "provider_metadata_validation_required", "required_action": "validate_provider_metadata"}},
	{"name": "provider_requires_secret_reference", "description": "External providers require a vault or secret-manager reference.", "condition": {"operation": "activate_provider", "external_provider": True, "secret_reference_present": False}, "effect": {"decision": "deny", "reason": "provider_secret_reference_required", "required_action": "attach_secret_reference"}},
	{"name": "resource_requires_owner", "description": "Protected resources require an owner and registered scope.", "condition": {"operation": "register_resource", "resource_owner_assigned": False}, "effect": {"decision": "deny", "reason": "resource_owner_required", "required_action": "assign_resource_owner"}},
	{"name": "resource_requires_scope", "description": "Protected resources require at least one access scope.", "condition": {"operation": "register_resource", "scope_present": False}, "effect": {"decision": "deny", "reason": "resource_scope_required", "required_action": "define_resource_scope"}},
	{"name": "policy_requires_owner", "description": "Policies require an accountable owner.", "condition": {"operation": "create_policy", "policy_owner_assigned": False}, "effect": {"decision": "deny", "reason": "policy_owner_required", "required_action": "assign_policy_owner"}},
	{"name": "sensitive_policy_requires_conditions", "description": "Sensitive-resource policies require explicit conditions.", "condition": {"operation": "create_policy", "sensitive_resource": True, "policy_conditions_present": False}, "effect": {"decision": "deny", "reason": "policy_conditions_required", "required_action": "define_policy_conditions"}},
	{"name": "high_risk_policy_requires_simulation", "description": "High-risk policy changes require simulation evidence.", "condition": {"operation": "activate_policy", "risk_level": "high", "simulation_evidence_present": False}, "effect": {"decision": "require_review", "reason": "policy_simulation_required", "required_action": "attach_policy_simulation"}},
	{"name": "privileged_grant_requires_approval", "description": "Privileged grants require approval.", "condition": {"operation": "create_grant", "privileged_scope": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privileged_grant_approval_required", "required_action": "record_grant_approval"}},
	{"name": "privileged_grant_requires_expiry", "description": "Privileged grants require an expiry time.", "condition": {"operation": "create_grant", "privileged_scope": True, "expiry_present": False}, "effect": {"decision": "deny", "reason": "privileged_grant_expiry_required", "required_action": "set_grant_expiry"}},
	{"name": "grant_requires_separation_of_duties", "description": "A requester cannot self-approve privileged access.", "condition": {"operation": "create_grant", "separation_of_duties_passed": False}, "effect": {"decision": "deny", "reason": "separation_of_duties_required", "required_action": "select_independent_approver"}},
	{"name": "grant_requires_justification", "description": "Access grants require business justification.", "condition": {"operation": "create_grant", "justification_present": False}, "effect": {"decision": "deny", "reason": "grant_justification_required", "required_action": "attach_grant_justification"}},
	{"name": "high_risk_session_requires_step_up", "description": "High-risk sessions require adaptive step-up authentication.", "condition": {"operation": "evaluate_session", "risk_score_gt": 74, "step_up_completed": False}, "effect": {"decision": "deny", "reason": "adaptive_step_up_required", "required_action": "complete_step_up_authentication"}},
	{"name": "decision_requires_bytewax_stream", "description": "Access decisions must be emitted through Bytewax.", "condition": {"operation": "record_decision", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_access_decision_to_bytewax"}},
	{"name": "batch_grant_requires_bytewax", "description": "Batch grant changes require Bytewax lifecycle coordination.", "condition": {"operation": "batch_grant", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_batch_grants_to_bytewax"}},
	{"name": "access_agent_runtime_supported", "description": "Access-control agents must use an approved runtime.", "condition": {"operation": "register_access_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "access_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "access_agent_role_supported", "description": "Access-control agents must use an approved role.", "condition": {"operation": "register_access_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "access_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_action_requires_human_approval", "description": "Privileged access actions proposed by agents require human approval.", "condition": {"operation": "agent_access_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/composition-access/dashboard", "component": "AccessDashboard", "permission": "composition_access:view", "nav_group": "Overview"},
	{"name": "providers", "path": "/composition-access/providers", "component": "AccessProviderConsole", "permission": "composition_access:admin", "nav_group": "Identity"},
	{"name": "resources", "path": "/composition-access/resources", "component": "AccessResourceRegistry", "permission": "composition_access:govern", "nav_group": "Resources"},
	{"name": "policies", "path": "/composition-access/policies", "component": "AccessPolicyStudio", "permission": "composition_access:govern", "nav_group": "Policy"},
	{"name": "grants", "path": "/composition-access/grants", "component": "AccessGrantWorkbench", "permission": "composition_access:grant", "nav_group": "Access"},
	{"name": "decisions", "path": "/composition-access/decisions", "component": "AccessDecisionExplorer", "permission": "composition_access:view", "nav_group": "Operations"},
	{"name": "sessions", "path": "/composition-access/sessions", "component": "AccessSessionMonitor", "permission": "composition_access:operate", "nav_group": "Operations"},
	{"name": "agents", "path": "/composition-access/agents", "component": "AccessAgentWorkbench", "permission": "composition_access:admin", "nav_group": "Automation"},
	{"name": "audit", "path": "/composition-access/audit", "component": "AccessAuditConsole", "permission": "composition_access:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/composition-access/settings", "component": "AccessSettings", "permission": "composition_access:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "composition_access_control",
	"tokens": {"color.primary": "#28536B", "color.accent": "#C44536", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"provider_console": {"icon": "key-round", "status_indicator": "provider-pill", "risk_style": "trust-band"},
		"policy_studio": {"visual": "rule-grid", "status_style": "policy-chip"},
		"grant_workbench": {"visual": "approval-queue", "status_style": "grant-chip"},
		"decision_explorer": {"visual": "decision-timeline", "status_style": "decision-chip"},
		"session_monitor": {"visual": "risk-lane", "status_style": "session-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "composition_access",
		"display_name": "Access Control Integration Hub",
		"provides": [
			"identity_provider_composition",
			"resource_access_registry",
			"policy_orchestration",
			"grant_lifecycle",
			"session_risk_control",
			"access_decision_audit",
			"access_agents",
		],
		"requires": ["auth", "audl", "ntfy", "conf", "registry"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/composition-access/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": ACCESS_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"provider_registered",
			"provider_activated",
			"resource_registered",
			"policy_created",
			"policy_activated",
			"grant_created",
			"grant_revoked",
			"session_evaluated",
			"access_decision_recorded",
			"access_agent_registered",
		],
		"states": ["draft", "active", "review_required", "approved", "denied", "revoked", "blocked"],
		"guardrails": [
			"decision_requires_bytewax_stream",
			"batch_grant_requires_bytewax",
			"privileged_agent_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return ACCESS_EVENT_STREAM


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
