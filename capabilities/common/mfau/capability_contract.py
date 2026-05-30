"""Executable capability contract for APG Multi-Factor Authentication."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"profiles": {
		"user_required": True,
		"policy_required": True,
		"default_status": "active",
		"max_failed_attempts": 5,
		"lockout_minutes": 15,
	},
	"methods": {
		"enabled": ["totp", "webauthn", "push", "email_otp", "sms_otp", "backup_codes", "hardware_key", "biometric"],
		"phishing_resistant": ["webauthn", "hardware_key"],
		"biometric_methods_allowed": True,
		"max_active_methods_per_user": 8,
		"verified_channel_required": True,
		"device_binding_required": True,
	},
	"enrollment": {
		"verification_required": True,
		"recent_auth_required": True,
		"biometric_consent_required": True,
		"template_encryption_required": True,
	},
	"risk": {
		"adaptive_step_up_enabled": True,
		"high_risk_threshold": 0.75,
		"critical_risk_threshold": 0.9,
		"low_trust_device_threshold": 0.4,
		"admin_actions_require_phishing_resistant": True,
	},
	"challenge": {
		"ttl_seconds": 300,
		"code_required": True,
		"single_use_required": True,
		"max_attempts": 5,
	},
	"devices": {
		"trust_scoring_enabled": True,
		"binding_required_for_device_methods": True,
		"review_low_trust_devices": True,
		"known_device_ttl_days": 90,
	},
	"recovery": {
		"backup_codes_enabled": True,
		"verified_channel_required": True,
		"admin_assisted_recovery": True,
		"admin_approval_required": True,
		"recovery_audit_required": True,
	},
	"backup_codes": {
		"default_count": 10,
		"single_use_required": True,
		"regeneration_requires_recent_auth": True,
	},
	"policies": {
		"default_policy": "standard-mfa",
		"audit_policy_changes": True,
		"privileged_actions_require_phishing_resistant": True,
	},
	"biometrics": {
		"consent_required": True,
		"template_encryption_required": True,
		"liveness_check_required": True,
	},
	"security": {
		"cross_tenant_access_allowed": False,
		"rbac_required": True,
		"factor_secret_encryption_required": True,
		"token_replay_blocking": True,
		"lockout_enabled": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_profiles": True,
		"audit_enrollments": True,
		"audit_challenges": True,
		"audit_recovery": True,
		"step_up_policy_required": True,
	},
	"observability": {
		"metrics_required": True,
		"trace_required": True,
		"audit_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "mfa_runtime.MfauService",
		"production_runtime": "service.MFAService",
		"helper_runtime": "mfa_runtime.py",
		"http_api": "api.py",
		"event_stream": "bytewax",
		"auth_provider": "auth",
		"security_framework": "secu",
		"encryption": "encr",
		"audit_sink": "audl",
		"notification": "ntfy",
		"biometric": "biop",
		"computer_vision": "cvsn",
		"cache": "cach",
		"metrics_sink": "moni",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_profiles": True,
		"enable_methods": True,
		"enable_enrollment_wizard": True,
		"enable_challenges": True,
		"enable_risk_console": True,
		"enable_devices": True,
		"enable_recovery_center": True,
		"enable_backup_codes": True,
		"enable_policies": True,
		"enable_governance": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "mfau_adaptive_auth_console", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"profiles",
		"methods",
		"enrollment",
		"risk",
		"challenge",
		"devices",
		"recovery",
		"backup_codes",
		"policies",
		"biometrics",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"profiles",
		"methods",
		"enrollment",
		"risk",
		"challenge",
		"devices",
		"recovery",
		"backup_codes",
		"policies",
		"biometrics",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All MFA operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "profile_requires_user", "description": "MFA profiles require users.", "condition": {"operation": "register_profile", "user_present": False}, "effect": {"decision": "deny", "reason": "user_required", "required_action": "select_user"}},
	{"name": "profile_requires_policy", "description": "MFA profiles require a policy.", "condition": {"operation": "register_profile", "policy_present": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "select_policy"}},
	{"name": "profile_status_requires_allowed_value", "description": "MFA profile status must be configured.", "condition": {"operation": "register_profile", "profile_status_allowed": False}, "effect": {"decision": "deny", "reason": "profile_status_invalid", "required_action": "choose_allowed_status"}},
	{"name": "enrollment_requires_profile", "description": "MFA enrollment requires a user profile.", "condition": {"operation": "enroll_method", "profile_present": False}, "effect": {"decision": "deny", "reason": "profile_required", "required_action": "register_profile"}},
	{"name": "enrollment_requires_method_type", "description": "MFA enrollment requires a method type.", "condition": {"operation": "enroll_method", "method_type_present": False}, "effect": {"decision": "deny", "reason": "method_type_required", "required_action": "choose_method_type"}},
	{"name": "method_type_requires_allowed_value", "description": "MFA method type must be enabled.", "condition": {"operation": "enroll_method", "method_type_allowed": False}, "effect": {"decision": "deny", "reason": "method_type_not_enabled", "required_action": "choose_enabled_method"}},
	{"name": "enrollment_requires_verified_channel", "description": "Channel-based methods require a verified channel.", "condition": {"operation": "enroll_method", "channel_method": True, "verified_channel": False}, "effect": {"decision": "deny", "reason": "verified_channel_required", "required_action": "verify_channel"}},
	{"name": "biometric_method_requires_consent", "description": "Biometric MFA methods require explicit consent.", "condition": {"operation": "enroll_method", "method_type": "biometric", "biometric_consent_recorded": False}, "effect": {"decision": "deny", "reason": "biometric_consent_required", "required_action": "record_biometric_consent"}},
	{"name": "biometric_template_requires_encryption", "description": "Biometric templates require encryption evidence.", "condition": {"operation": "enroll_method", "method_type": "biometric", "template_encrypted": False}, "effect": {"decision": "deny", "reason": "biometric_template_encryption_required", "required_action": "encrypt_template"}},
	{"name": "device_binding_required_for_method", "description": "Device-bound methods require device binding.", "condition": {"operation": "enroll_method", "device_bound_method": True, "device_binding_present": False}, "effect": {"decision": "deny", "reason": "device_binding_required", "required_action": "bind_device"}},
	{"name": "active_method_limit_requires_review", "description": "High active method counts require review.", "condition": {"operation": "enroll_method", "active_method_count_gte": 8, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "active_method_limit_review_required", "required_action": "review_active_method_count"}},
	{"name": "factor_secret_requires_encryption", "description": "Factor secrets require encryption evidence.", "condition": {"operation": "enroll_method", "secret_encrypted": False}, "effect": {"decision": "deny", "reason": "factor_secret_encryption_required", "required_action": "encrypt_factor_secret"}},
	{"name": "verification_requires_method", "description": "Challenge verification requires an enrolled method.", "condition": {"operation": "verify_challenge", "method_present": False}, "effect": {"decision": "deny", "reason": "method_required", "required_action": "select_method"}},
	{"name": "verification_requires_code_or_assertion", "description": "Challenge verification requires code or assertion evidence.", "condition": {"operation": "verify_challenge", "verification_evidence_present": False}, "effect": {"decision": "deny", "reason": "verification_evidence_required", "required_action": "attach_verification_evidence"}},
	{"name": "challenge_requires_profile", "description": "Challenges require an MFA profile.", "condition": {"operation": "create_challenge", "profile_present": False}, "effect": {"decision": "deny", "reason": "profile_required", "required_action": "register_profile"}},
	{"name": "challenge_requires_active_method", "description": "Challenges require an active method.", "condition": {"operation": "create_challenge", "active_method_present": False}, "effect": {"decision": "deny", "reason": "active_method_required", "required_action": "enroll_active_method"}},
	{"name": "high_risk_requires_step_up", "description": "High-risk authentication requires step-up.", "condition": {"operation": "create_challenge", "risk_score_gt": 0.75, "step_up_completed": False}, "effect": {"decision": "deny", "reason": "step_up_required", "required_action": "complete_step_up_challenge"}},
	{"name": "critical_risk_requires_block", "description": "Critical risk authentication is blocked.", "condition": {"operation": "create_challenge", "risk_score_gte": 0.9, "risk_override_approved": False}, "effect": {"decision": "deny", "reason": "critical_risk_blocked", "required_action": "approve_risk_override"}},
	{"name": "admin_action_requires_phishing_resistant_factor", "description": "Privileged actions require phishing-resistant MFA.", "condition": {"operation": "create_challenge", "action_risk": "admin", "phishing_resistant_factor_present": False}, "effect": {"decision": "deny", "reason": "phishing_resistant_factor_required", "required_action": "use_webauthn_or_hardware_key"}},
	{"name": "low_trust_device_requires_review", "description": "Low-trust devices require additional review.", "condition": {"operation": "create_challenge", "device_trust_score_lt": 0.4, "device_review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_trust_device_review_required", "required_action": "review_device_trust"}},
	{"name": "locked_profile_blocks_challenge", "description": "Locked MFA profiles block challenges.", "condition": {"operation": "create_challenge", "profile_locked": True}, "effect": {"decision": "deny", "reason": "profile_locked", "required_action": "unlock_profile"}},
	{"name": "expired_challenge_blocks_verification", "description": "Expired challenges block verification.", "condition": {"operation": "verify_challenge", "challenge_expired": True}, "effect": {"decision": "deny", "reason": "challenge_expired", "required_action": "issue_new_challenge"}},
	{"name": "single_use_challenge_blocks_replay", "description": "Used challenges may not be replayed.", "condition": {"operation": "verify_challenge", "challenge_already_used": True}, "effect": {"decision": "deny", "reason": "challenge_replay_blocked", "required_action": "issue_new_challenge"}},
	{"name": "failed_attempt_limit_blocks_profile", "description": "Too many failed attempts lock the profile.", "condition": {"operation": "verify_challenge", "failed_attempt_count_gte": 5}, "effect": {"decision": "deny", "reason": "failed_attempt_limit_reached", "required_action": "lock_profile"}},
	{"name": "recovery_requires_profile", "description": "Account recovery requires an MFA profile.", "condition": {"operation": "recover_account", "profile_present": False}, "effect": {"decision": "deny", "reason": "profile_required", "required_action": "select_profile"}},
	{"name": "recovery_requires_verified_channel", "description": "Account recovery requires a verified channel.", "condition": {"operation": "recover_account", "verified_recovery_channel": False}, "effect": {"decision": "deny", "reason": "verified_recovery_channel_required", "required_action": "verify_recovery_channel"}},
	{"name": "recovery_requires_evidence", "description": "Account recovery requires evidence.", "condition": {"operation": "recover_account", "recovery_evidence_present": False}, "effect": {"decision": "deny", "reason": "recovery_evidence_required", "required_action": "attach_recovery_evidence"}},
	{"name": "admin_recovery_requires_approval", "description": "Admin-assisted recovery requires approval.", "condition": {"operation": "recover_account", "admin_assisted": True, "admin_approval_recorded": False}, "effect": {"decision": "deny", "reason": "admin_recovery_approval_required", "required_action": "record_admin_approval"}},
	{"name": "recovery_state_change_requires_audit", "description": "Recovery state changes require audit evidence.", "condition": {"operation": "recover_account", "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "recovery_audit_required", "required_action": "record_recovery_audit"}},
	{"name": "backup_code_requires_remaining_code", "description": "Backup-code recovery requires an unused code.", "condition": {"operation": "use_backup_code", "backup_codes_remaining_lte": 0}, "effect": {"decision": "deny", "reason": "backup_code_unavailable", "required_action": "regenerate_backup_codes"}},
	{"name": "backup_code_single_use_required", "description": "Backup codes may only be used once.", "condition": {"operation": "use_backup_code", "backup_code_already_used": True}, "effect": {"decision": "deny", "reason": "backup_code_replay_blocked", "required_action": "use_unused_backup_code"}},
	{"name": "method_disable_requires_alternative", "description": "Disabling a method requires another active method.", "condition": {"operation": "disable_method", "alternative_method_present": False}, "effect": {"decision": "deny", "reason": "alternative_method_required", "required_action": "enroll_alternative_method"}},
	{"name": "method_rotation_requires_recent_verification", "description": "Method rotation requires recent verification.", "condition": {"operation": "rotate_method", "recent_verification_present": False}, "effect": {"decision": "deny", "reason": "recent_verification_required", "required_action": "verify_current_method"}},
	{"name": "policy_change_requires_audit", "description": "MFA policy changes require audit events.", "condition": {"operation": "change_policy", "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "policy_audit_required", "required_action": "record_policy_audit"}},
	{"name": "external_risk_signal_requires_review", "description": "External risk signals require review.", "condition": {"external_risk_signal": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "external_risk_signal_review_required", "required_action": "review_external_signal"}},
	{"name": "batch_mfa_mutation_requires_bytewax", "description": "Batch MFA mutations must use Bytewax event streams.", "condition": {"operation": "batch_mfa_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "cross_tenant_mfa_access_denied", "description": "MFA operations may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_mfa_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "mfa_state_change_requires_audit", "description": "MFA state changes require audit events.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "audit_event_required", "required_action": "record_audit_event"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/mfau/dashboard", "component": "MFAUDashboard", "permission": "mfau:view", "nav_group": "Overview"},
	{"name": "profiles", "path": "/mfau/profiles", "component": "MFAProfileRegistry", "permission": "mfau:view", "nav_group": "Users"},
	{"name": "methods", "path": "/mfau/methods", "component": "MFAMethods", "permission": "mfau:manage_methods", "nav_group": "Methods"},
	{"name": "enrollment", "path": "/mfau/enrollment", "component": "MFAEnrollmentWizard", "permission": "mfau:enroll", "nav_group": "Methods"},
	{"name": "challenges", "path": "/mfau/challenges", "component": "MFAChallengeConsole", "permission": "mfau:challenge", "nav_group": "Challenges"},
	{"name": "risk", "path": "/mfau/risk", "component": "MFARiskConsole", "permission": "mfau:challenge", "nav_group": "Risk"},
	{"name": "devices", "path": "/mfau/devices", "component": "MFADeviceTrust", "permission": "mfau:challenge", "nav_group": "Risk"},
	{"name": "recovery", "path": "/mfau/recovery", "component": "MFARecoveryCenter", "permission": "mfau:recover", "nav_group": "Recovery"},
	{"name": "backup_codes", "path": "/mfau/backup-codes", "component": "BackupCodeManager", "permission": "mfau:recover", "nav_group": "Recovery"},
	{"name": "policies", "path": "/mfau/policies", "component": "MFAPolicyStudio", "permission": "mfau:admin", "nav_group": "Governance"},
	{"name": "biometrics", "path": "/mfau/biometrics", "component": "MFABiometricConsent", "permission": "mfau:manage_methods", "nav_group": "Methods"},
	{"name": "governance", "path": "/mfau/governance", "component": "MFAUGovernance", "permission": "mfau:admin", "nav_group": "Governance"},
	{"name": "audit", "path": "/mfau/audit", "component": "MFAAuditTrail", "permission": "mfau:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/mfau/settings", "component": "MFAUSettings", "permission": "mfau:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "mfau_adaptive_auth_console",
	"tokens": {
		"color.primary": "#1F4E5F",
		"color.accent": "#F2A541",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F9F9",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"factor_stack": {"icon": "shield-check", "status_indicator": "factor-pill", "risk_style": "step-up-band"},
		"profile_card": {"visual": "user-security-summary", "status_style": "lock-chip"},
		"method_card": {"visual": "factor-list", "status_style": "verification-chip"},
		"challenge_panel": {"visual": "challenge-timeline", "status_style": "attempt-chip"},
		"risk_meter": {"visual": "trust-gauge", "highlight": "risk-threshold-chip"},
		"device_trust": {"visual": "device-list", "status_style": "trust-chip"},
		"enrollment_wizard": {"visual": "method-stepper", "status_style": "verification-chip"},
		"recovery_timeline": {"visual": "audit-timeline", "status_style": "channel-chip"},
		"backup_code_panel": {"visual": "single-use-code-list", "status_style": "remaining-chip"},
		"policy_editor": {"visual": "policy-rule-builder", "status_style": "audit-chip"},
		"biometric_consent": {"visual": "consent-evidence-panel", "status_style": "consent-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "security-chip"},
	},
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
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
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
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
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
