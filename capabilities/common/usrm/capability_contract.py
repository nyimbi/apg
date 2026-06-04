"""Executable capability contract for APG User Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_USRM_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_USRM_AGENT_ROLES = [
	"identity_reviewer",
	"lifecycle_reviewer",
	"access_reviewer",
	"deprovision_reviewer",
	"privacy_reviewer",
	"entitlement_reviewer",
]
USRM_EVENT_STREAM = "apg.usrm.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"users": {
		"user_owner_required": True,
		"unique_identity_required": True,
		"profile_validation_required": True,
		"status_history_enabled": True,
	},
	"lifecycle": {
		"invite_consent_notice_required": True,
		"deprovision_access_revocation_required": True,
		"deprovision_evidence_required": True,
		"manager_approval_required": True,
		"bulk_action_review_threshold": 25,
		"bulk_action_stream_required": True,
	},
	"access": {
		"privileged_mfa_required": True,
		"privileged_role_approval_required": True,
		"periodic_access_review_required": True,
		"role_assignment_audit_required": True,
		"least_privilege_policy_required": True,
	},
	"usrm_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_USRM_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_USRM_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "non_privileged",
		"disclose_agent_recommendations": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_user_changes": True,
		"privacy_preference_sync_required": True,
		"identity_federation_supported": True,
		"state_change_audit_required": True,
	},
	"observability": {
		"event_stream": USRM_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_user_events": True,
		"emit_access_events": True,
		"emit_lifecycle_events": True,
	},
	"adapters": {
		"identity": "adapter",
		"authorization": "adapter",
		"mfa": "adapter",
		"consent": "adapter",
		"audit": "adapter",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_user_directory": True,
		"enable_lifecycle_queue": True,
		"enable_access_review": True,
		"enable_privacy_preferences": True,
		"enable_agent_workbench": True,
		"enable_policy_center": True,
	},
	"theme": {"default_theme": "usrm_user_lifecycle", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"users",
		"lifecycle",
		"access",
		"usrm_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"users",
			"lifecycle",
			"access",
			"usrm_agents",
			"governance",
			"observability",
			"adapters",
			"ui",
			"theme",
		]
	} | {"tenant_id": {"type": "string", "minLength": 1}},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All user-management operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "user_requires_identity", "description": "Users require a unique identity.", "condition": {"operation": "create_user", "unique_identity_present": False}, "effect": {"decision": "deny", "reason": "unique_identity_required", "required_action": "attach_unique_identity"}},
	{"name": "user_requires_owner", "description": "Users require an accountable owner.", "condition": {"operation": "create_user", "user_owner_assigned": False}, "effect": {"decision": "deny", "reason": "user_owner_required", "required_action": "assign_user_owner"}},
	{"name": "user_requires_profile_validation", "description": "User creation requires profile validation evidence.", "condition": {"operation": "create_user", "profile_validated": False}, "effect": {"decision": "deny", "reason": "profile_validation_required", "required_action": "validate_user_profile"}},
	{"name": "invite_requires_consent_notice", "description": "User invitations require a privacy/consent notice.", "condition": {"operation": "invite_user", "consent_notice_attached": False}, "effect": {"decision": "deny", "reason": "consent_notice_required", "required_action": "attach_consent_notice"}},
	{"name": "invite_requires_bytewax_stream", "description": "User invitation lifecycle events must be emitted through Bytewax.", "condition": {"operation": "invite_user", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_user_invitation_to_bytewax"}},
	{"name": "profile_requires_privacy_sync", "description": "Profile changes require privacy preference sync evidence.", "condition": {"operation": "update_profile", "privacy_sync_recorded": False}, "effect": {"decision": "deny", "reason": "privacy_preference_sync_required", "required_action": "sync_privacy_preferences"}},
	{"name": "privileged_user_requires_mfa", "description": "Privileged users require MFA.", "condition": {"privileged_user": True, "mfa_enabled": False}, "effect": {"decision": "deny", "reason": "mfa_required", "required_action": "enable_mfa"}},
	{"name": "privileged_role_requires_approval", "description": "Privileged role assignments require approval.", "condition": {"operation": "assign_role", "privileged_role": True, "role_approval_recorded": False}, "effect": {"decision": "deny", "reason": "role_assignment_approval_required", "required_action": "record_role_assignment_approval"}},
	{"name": "access_review_requires_reviewer", "description": "Access reviews require reviewer attribution.", "condition": {"operation": "record_access_review", "access_reviewer_present": False}, "effect": {"decision": "deny", "reason": "access_reviewer_required", "required_action": "assign_access_reviewer"}},
	{"name": "deprovision_requires_access_revocation", "description": "Deprovisioning requires access revocation evidence.", "condition": {"operation": "deprovision_user", "access_revoked": False}, "effect": {"decision": "deny", "reason": "access_revocation_required", "required_action": "revoke_user_access"}},
	{"name": "deprovision_requires_evidence", "description": "Deprovisioning requires a durable evidence reference.", "condition": {"operation": "deprovision_user", "deprovision_evidence_present": False}, "effect": {"decision": "deny", "reason": "deprovision_evidence_required", "required_action": "attach_deprovision_evidence"}},
	{"name": "deprovision_requires_bytewax_stream", "description": "Deprovision lifecycle events must be emitted through Bytewax.", "condition": {"operation": "deprovision_user", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_deprovision_lifecycle_to_bytewax"}},
	{"name": "bulk_user_action_requires_review", "description": "Bulk user actions require review.", "condition": {"affected_user_count_gt": 25, "bulk_review_recorded": False}, "effect": {"decision": "require_review", "reason": "bulk_user_review_required", "required_action": "review_bulk_user_action"}},
	{"name": "bulk_user_action_requires_bytewax", "description": "Bulk user lifecycle actions require Bytewax stream coordination.", "condition": {"operation": "bulk_user_action", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_bulk_user_action_to_bytewax"}},
	{"name": "usrm_agent_runtime_supported", "description": "User-management agents must use an approved runtime.", "condition": {"operation": "register_usrm_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "usrm_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "usrm_agent_role_supported", "description": "User-management agents must use an approved role.", "condition": {"operation": "register_usrm_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "usrm_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_user_action_requires_human_approval", "description": "Privileged user actions proposed by agents require human approval.", "condition": {"operation": "agent_user_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "write_requires_policy", "description": "User management write operations require an explicit authorization policy.", "condition": {"operation_type": "write", "write_policy_present": False}, "effect": {"decision": "deny", "reason": "usrm_write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "privilege_escalation_denied", "description": "Users cannot self-grant elevated user-management permissions beyond their current role.", "condition": {"operation": "assign_usrm_permission", "target_tier_exceeds_actor_tier": True}, "effect": {"decision": "deny", "reason": "privilege_escalation_prevented", "required_action": "route_to_higher_authority_approver"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/usrm/dashboard", "component": "USRMDashboard", "permission": "usrm:view", "nav_group": "Overview"},
	{"name": "users", "path": "/usrm/users", "component": "UserDirectory", "permission": "usrm:manage_users", "nav_group": "Users"},
	{"name": "profiles", "path": "/usrm/profiles", "component": "ProfileManager", "permission": "usrm:manage_users", "nav_group": "Users"},
	{"name": "lifecycle", "path": "/usrm/lifecycle", "component": "LifecycleQueue", "permission": "usrm:manage_users", "nav_group": "Lifecycle"},
	{"name": "access", "path": "/usrm/access", "component": "AccessReview", "permission": "usrm:review_access", "nav_group": "Access"},
	{"name": "privacy", "path": "/usrm/privacy", "component": "PrivacyPreferences", "permission": "usrm:view", "nav_group": "Privacy"},
	{"name": "deprovisioning", "path": "/usrm/deprovisioning", "component": "DeprovisioningCenter", "permission": "usrm:deprovision", "nav_group": "Lifecycle"},
	{"name": "agents", "path": "/usrm/agents", "component": "USRMAgentWorkbench", "permission": "usrm:admin", "nav_group": "Automation"},
	{"name": "policy", "path": "/usrm/policy", "component": "USRMPolicyCenter", "permission": "usrm:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/usrm/settings", "component": "USRMSettings", "permission": "usrm:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "usrm_user_lifecycle",
	"tokens": {"color.primary": "#2B4C7E", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"user_card": {"icon": "user-cog", "status_indicator": "status-pill", "risk_style": "access-band"},
		"lifecycle_queue": {"visual": "approval-list", "highlight": "stage-chip"},
		"access_review": {"visual": "entitlement-matrix", "status_style": "mfa-chip"},
		"privacy_panel": {"visual": "preference-list", "status_style": "consent-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
		"policy_center": {"visual": "rule-grid", "status_style": "guardrail-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "usrm",
		"display_name": "User Management",
		"provides": [
			"user_directory",
			"profile_management",
			"consented_invitations",
			"role_assignment_governance",
			"access_review_workflows",
			"deprovisioning_governance",
			"user_audit_events",
			"usrm_agents",
		],
		"requires": ["auth", "mfau", "cons", "audl", "idfd"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/usrm/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": USRM_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"user_created",
			"profile_updated",
			"user_invited",
			"role_assigned",
			"access_review_recorded",
			"user_deprovisioned",
			"bulk_suspend_users",
			"usrm_agent_registered",
		],
		"states": ["active", "invited", "suspended", "review_required", "deprovisioned", "blocked"],
		"guardrails": [
			"invite_requires_bytewax_stream",
			"deprovision_requires_bytewax_stream",
			"bulk_user_action_requires_bytewax",
			"privileged_agent_user_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return USRM_EVENT_STREAM


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
