"""Executable capability contract for APG User Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"users": {"user_owner_required": True, "unique_identity_required": True, "profile_validation_required": True, "status_history_enabled": True},
	"lifecycle": {"invite_consent_notice_required": True, "deprovision_access_revocation_required": True, "manager_approval_required": True, "bulk_action_review_threshold": 25},
	"access": {"privileged_mfa_required": True, "periodic_access_review_required": True, "role_assignment_audit_required": True, "least_privilege_policy_required": True},
	"governance": {"require_tenant_context": True, "audit_user_changes": True, "privacy_preference_sync_required": True, "identity_federation_supported": True},
	"ui": {"enable_user_directory": True, "enable_lifecycle_queue": True, "enable_access_review": True, "enable_privacy_preferences": True},
	"theme": {"default_theme": "usrm_user_lifecycle", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "users", "lifecycle", "access", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["users", "lifecycle", "access", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All user-management operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "user_requires_identity", "description": "Users require a unique identity.", "condition": {"operation": "create_user", "unique_identity_present": False}, "effect": {"decision": "deny", "reason": "unique_identity_required", "required_action": "attach_unique_identity"}},
	{"name": "invite_requires_consent_notice", "description": "User invitations require a privacy/consent notice.", "condition": {"operation": "invite_user", "consent_notice_attached": False}, "effect": {"decision": "deny", "reason": "consent_notice_required", "required_action": "attach_consent_notice"}},
	{"name": "privileged_user_requires_mfa", "description": "Privileged users require MFA.", "condition": {"privileged_user": True, "mfa_enabled": False}, "effect": {"decision": "deny", "reason": "mfa_required", "required_action": "enable_mfa"}},
	{"name": "deprovision_requires_access_revocation", "description": "Deprovisioning requires access revocation evidence.", "condition": {"operation": "deprovision_user", "access_revoked": False}, "effect": {"decision": "deny", "reason": "access_revocation_required", "required_action": "revoke_user_access"}},
	{"name": "bulk_user_action_requires_review", "description": "Bulk user actions require review.", "condition": {"affected_user_count_gt": 25, "bulk_review_recorded": False}, "effect": {"decision": "require_review", "reason": "bulk_user_review_required", "required_action": "review_bulk_user_action"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/usrm/dashboard", "component": "USRMDashboard", "permission": "usrm:view", "nav_group": "Overview"},
	{"name": "users", "path": "/usrm/users", "component": "UserDirectory", "permission": "usrm:manage_users", "nav_group": "Users"},
	{"name": "profiles", "path": "/usrm/profiles", "component": "ProfileManager", "permission": "usrm:manage_users", "nav_group": "Users"},
	{"name": "lifecycle", "path": "/usrm/lifecycle", "component": "LifecycleQueue", "permission": "usrm:manage_users", "nav_group": "Lifecycle"},
	{"name": "access", "path": "/usrm/access", "component": "AccessReview", "permission": "usrm:review_access", "nav_group": "Access"},
	{"name": "privacy", "path": "/usrm/privacy", "component": "PrivacyPreferences", "permission": "usrm:view", "nav_group": "Privacy"},
	{"name": "deprovisioning", "path": "/usrm/deprovisioning", "component": "DeprovisioningCenter", "permission": "usrm:deprovision", "nav_group": "Lifecycle"},
	{"name": "settings", "path": "/usrm/settings", "component": "USRMSettings", "permission": "usrm:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "usrm_user_lifecycle",
	"tokens": {"color.primary": "#2B4C7E", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"user_card": {"icon": "user-cog", "status_indicator": "status-pill", "risk_style": "access-band"}, "lifecycle_queue": {"visual": "approval-list", "highlight": "stage-chip"}, "access_review": {"visual": "entitlement-matrix", "status_style": "mfa-chip"}, "privacy_panel": {"visual": "preference-list", "status_style": "consent-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "usrm", "display_name": "User Management", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/usrm/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
