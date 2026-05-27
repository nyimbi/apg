"""Executable capability contract for APG Multi-Factor Authentication."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"methods": {
		"enabled": ["totp", "webauthn", "push", "email_otp", "sms_otp", "backup_codes"],
		"phishing_resistant": ["webauthn", "hardware_key"],
		"biometric_methods_allowed": True,
		"max_active_methods_per_user": 8
	},
	"risk": {
		"adaptive_step_up_enabled": True,
		"high_risk_threshold": 0.75,
		"low_trust_device_threshold": 0.4,
		"admin_actions_require_phishing_resistant": True
	},
	"recovery": {
		"backup_codes_enabled": True,
		"verified_channel_required": True,
		"admin_assisted_recovery": True,
		"recovery_audit_required": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_challenges": True,
		"biometric_consent_required": True,
		"step_up_policy_required": True
	},
	"ui": {
		"enable_auth_dashboard": True,
		"enable_enrollment_wizard": True,
		"enable_risk_console": True,
		"enable_recovery_center": True
	},
	"theme": {
		"default_theme": "mfau_adaptive_auth_console",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "methods", "risk", "recovery", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["methods", "risk", "recovery", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All MFA operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "high_risk_requires_step_up", "description": "High-risk authentication requires step-up.", "condition": {"risk_score_gt": 0.75, "step_up_completed": False}, "effect": {"decision": "deny", "reason": "step_up_required", "required_action": "complete_step_up_challenge"}},
	{"name": "biometric_method_requires_consent", "description": "Biometric MFA methods require explicit consent.", "condition": {"method_type": "biometric", "biometric_consent_recorded": False}, "effect": {"decision": "deny", "reason": "biometric_consent_required", "required_action": "record_biometric_consent"}},
	{"name": "recovery_requires_verified_channel", "description": "Account recovery requires a verified channel.", "condition": {"operation": "recover_account", "verified_recovery_channel": False}, "effect": {"decision": "deny", "reason": "verified_recovery_channel_required", "required_action": "verify_recovery_channel"}},
	{"name": "admin_action_requires_phishing_resistant_factor", "description": "Privileged actions require phishing-resistant MFA.", "condition": {"action_risk": "admin", "phishing_resistant_factor_present": False}, "effect": {"decision": "deny", "reason": "phishing_resistant_factor_required", "required_action": "use_webauthn_or_hardware_key"}},
	{"name": "low_trust_device_requires_review", "description": "Low-trust devices require additional review.", "condition": {"device_trust_score_lt": 0.4, "device_review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_trust_device_review_required", "required_action": "review_device_trust"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/mfau/dashboard", "component": "MFAUDashboard", "permission": "mfau:view", "nav_group": "Overview"},
	{"name": "methods", "path": "/mfau/methods", "component": "MFAMethods", "permission": "mfau:manage_methods", "nav_group": "Methods"},
	{"name": "enrollment", "path": "/mfau/enrollment", "component": "MFAEnrollmentWizard", "permission": "mfau:enroll", "nav_group": "Methods"},
	{"name": "risk", "path": "/mfau/risk", "component": "MFARiskConsole", "permission": "mfau:challenge", "nav_group": "Risk"},
	{"name": "recovery", "path": "/mfau/recovery", "component": "MFARecoveryCenter", "permission": "mfau:recover", "nav_group": "Recovery"},
	{"name": "policies", "path": "/mfau/policies", "component": "MFAPolicyStudio", "permission": "mfau:admin", "nav_group": "Governance"},
	{"name": "audit", "path": "/mfau/audit", "component": "MFAAuditTrail", "permission": "mfau:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/mfau/settings", "component": "MFAUSettings", "permission": "mfau:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "mfau_adaptive_auth_console",
	"tokens": {
		"color.primary": "#1F4E5F",
		"color.accent": "#D97706",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F9F9",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"factor_stack": {"icon": "shield-check", "status_indicator": "factor-pill", "risk_style": "step-up-band"},
		"risk_meter": {"visual": "trust-gauge", "highlight": "risk-threshold-chip"},
		"enrollment_wizard": {"visual": "method-stepper", "status_style": "verification-chip"},
		"recovery_timeline": {"visual": "audit-timeline", "status_style": "channel-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable MFAU capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "mfau",
		"display_name": "Multi-Factor Authentication",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/mfau/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default MFAU governance rules."""
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
