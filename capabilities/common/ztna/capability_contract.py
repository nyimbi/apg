"""Executable capability contract for APG Zero Trust Network Access."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"identities": {
		"verified_identity_required": True,
		"mfa_required_for_privileged": True,
		"federated_identity_allowed": True,
		"continuous_identity_checks": True
	},
	"devices": {
		"posture_required": True,
		"minimum_device_trust": 0.7,
		"managed_device_preferred": True,
		"attestation_required_for_sensitive_resources": True
	},
	"resources": {
		"resource_policy_required": True,
		"least_privilege_default": True,
		"session_recording_for_privileged": True,
		"microsegmentation_enabled": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_access_decisions": True,
		"risk_threshold": 0.8,
		"deny_by_default": True
	},
	"ui": {
		"enable_access_console": True,
		"enable_device_posture": True,
		"enable_resource_map": True,
		"enable_session_monitor": True
	},
	"theme": {
		"default_theme": "ztna_zero_trust_ops",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "identities", "devices", "resources", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["identities", "devices", "resources", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All zero-trust decisions require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "identity_must_be_verified", "description": "Access requires verified identity.", "condition": {"identity_verified": False}, "effect": {"decision": "deny", "reason": "identity_verification_required", "required_action": "verify_identity"}},
	{"name": "device_posture_required", "description": "Access requires current device posture.", "condition": {"device_posture_present": False}, "effect": {"decision": "deny", "reason": "device_posture_required", "required_action": "collect_device_posture"}},
	{"name": "resource_policy_required", "description": "Resource access requires a matching policy.", "condition": {"resource_policy_attached": False}, "effect": {"decision": "deny", "reason": "resource_policy_required", "required_action": "attach_resource_policy"}},
	{"name": "privileged_access_requires_mfa", "description": "Privileged access requires MFA.", "condition": {"access_level": "privileged", "mfa_completed": False}, "effect": {"decision": "deny", "reason": "privileged_mfa_required", "required_action": "complete_mfa"}},
	{"name": "high_risk_access_requires_review", "description": "High-risk access decisions require review.", "condition": {"access_risk_score_gt": 0.8, "access_review_recorded": False}, "effect": {"decision": "require_review", "reason": "high_risk_access_review_required", "required_action": "review_access_request"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/ztna/dashboard", "component": "ZTNADashboard", "permission": "ztna:view", "nav_group": "Overview"},
	{"name": "policies", "path": "/ztna/policies", "component": "ZeroTrustPolicies", "permission": "ztna:manage_policies", "nav_group": "Policies"},
	{"name": "devices", "path": "/ztna/devices", "component": "DevicePosture", "permission": "ztna:manage_devices", "nav_group": "Devices"},
	{"name": "resources", "path": "/ztna/resources", "component": "ResourceMap", "permission": "ztna:manage_policies", "nav_group": "Resources"},
	{"name": "access", "path": "/ztna/access", "component": "AccessRequests", "permission": "ztna:approve_access", "nav_group": "Access"},
	{"name": "sessions", "path": "/ztna/sessions", "component": "SessionMonitor", "permission": "ztna:view", "nav_group": "Operations"},
	{"name": "risk", "path": "/ztna/risk", "component": "AccessRiskConsole", "permission": "ztna:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/ztna/settings", "component": "ZTNASettings", "permission": "ztna:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "ztna_zero_trust_ops",
	"tokens": {
		"color.primary": "#1A365D",
		"color.accent": "#38A169",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"access_decision": {"icon": "shield", "status_indicator": "decision-pill", "risk_style": "trust-band"},
		"device_posture": {"visual": "posture-checklist", "highlight": "trust-score-chip"},
		"resource_map": {"visual": "segmented-network-map", "status_style": "policy-chip"},
		"session_monitor": {"visual": "active-session-table", "status_style": "reauth-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable ZTNA capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "ztna",
		"display_name": "Zero Trust Network Access",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/ztna/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default ZTNA governance rules."""
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
