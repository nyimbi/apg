"""Executable capability contract for APG Multi-Factor Authentication."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_MFA_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_MFA_AGENT_ROLES = [
	"enrollment_reviewer",
	"risk_reviewer",
	"challenge_reviewer",
	"device_trust_reviewer",
	"recovery_reviewer",
	"policy_reviewer",
	"biometric_reviewer",
	"backup_code_reviewer",
	"lifecycle_batch_reviewer",
	"mfa_security_steward",
]
PRIVILEGED_MFA_AGENT_ROLES = [
	"risk_reviewer",
	"challenge_reviewer",
	"recovery_reviewer",
	"policy_reviewer",
	"biometric_reviewer",
	"lifecycle_batch_reviewer",
	"mfa_security_steward",
]


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
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_MFA_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_MFA_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_MFA_AGENT_ROLES,
		"require_owner": True,
		"require_purpose": True,
		"require_scope": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_mfa_agent_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "mfau.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"profile_batch",
			"method_batch",
			"device_batch",
			"risk_batch",
			"challenge_batch",
			"recovery_batch",
			"backup_code_batch",
			"policy_batch",
			"biometric_batch",
			"mfa_agent_batch",
		],
		"topics": [
			"mfau.profiles",
			"mfau.methods",
			"mfau.devices",
			"mfau.risk",
			"mfau.challenges",
			"mfau.recovery",
			"mfau.backup_codes",
			"mfau.policies",
			"mfau.biometrics",
			"mfau.agents",
		],
		"broker_core_dependency_allowed": False,
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
		"enable_mfa_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
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
		"agents",
		"streaming",
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
		"agents",
		"streaming",
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
	{"name": "mfa_agent_runtime_supported", "description": "MFA security agents must use supported provider-neutral runtimes.", "condition": {"operation": "register_mfa_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_mfa_agent_runtime", "required_action": "choose_supported_mfa_agent_runtime"}},
	{"name": "mfa_agent_role_supported", "description": "MFA security agents must use supported security governance roles.", "condition": {"operation": "register_mfa_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_mfa_agent_role", "required_action": "choose_supported_mfa_agent_role"}},
	{"name": "mfa_agent_requires_scope", "description": "MFA security agents require an explicit bounded enrollment, risk, challenge, device, recovery, policy, biometric, backup-code, or lifecycle scope.", "condition": {"operation": "register_mfa_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "mfa_agent_scope_required", "required_action": "declare_mfa_agent_scope"}},
	{"name": "mfa_agent_requires_owner", "description": "MFA security agents require an accountable owner.", "condition": {"operation": "register_mfa_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "mfa_agent_owner_required", "required_action": "assign_mfa_agent_owner"}},
	{"name": "mfa_agent_requires_purpose", "description": "MFA security agents require a documented purpose.", "condition": {"operation": "register_mfa_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "mfa_agent_purpose_required", "required_action": "document_mfa_agent_purpose"}},
	{"name": "mfa_agent_requires_contribution_disclosure", "description": "MFA security agents must disclose machine-authored authentication, risk, recovery, and policy-review contributions.", "condition": {"operation": "register_mfa_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "mfa_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "mfa_agent_privileged_role_requires_human_approval", "description": "Privileged MFA security-agent roles require human approval evidence.", "condition": {"operation": "register_mfa_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "mfa_agent_human_approval_required", "required_action": "record_human_mfa_agent_approval"}},
	{"name": "mfa_lifecycle_batch_requires_mutations", "description": "MFAU lifecycle batches must include at least one mutation.", "condition": {"operation": "validate_mfa_lifecycle_batch", "mutation_count_lte": 0}, "effect": {"decision": "deny", "reason": "mfa_lifecycle_batch_empty", "required_action": "include_mfa_lifecycle_mutations"}},
	{"name": "bytewax_mfa_stream_required", "description": "MFAU lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_mfa_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_mfa_lifecycle_batch_to_bytewax"}},
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
	{"name": "agents", "path": "/mfau/agents", "component": "MFASecurityAgentRoster", "permission": "mfau:admin", "nav_group": "Governance"},
	{"name": "lifecycle", "path": "/mfau/lifecycle", "component": "MFAULifecycleBatchMonitor", "permission": "mfau:admin", "nav_group": "Operations"},
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
		"mfa_agent_roster": {"icon": "bot", "status_indicator": "agent-approval-chip", "risk_style": "assurance-scope-band"},
		"bytewax_lifecycle_panel": {"icon": "git-branch", "status_indicator": "stream-chip", "risk_style": "processor-band"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "security-chip"},
	},
}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class MFAU agent composition manifest."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_MFA_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_MFA_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_MFA_AGENT_ROLES),
		"required_fields": ["tenant_id", "agent_id", "name", "runtime", "role", "scope", "owner", "purpose"],
		"guardrails": [
			"supported_runtime",
			"supported_role",
			"explicit_scope",
			"accountable_owner",
			"declared_purpose",
			"machine_contribution_disclosure",
			"human_approval_for_privileged_roles",
		],
		"adapter_contract": "aicr_provider_neutral_mfa_agent_adapter",
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return the MFAU Bytewax lifecycle stream contract."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "mfau.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"profile_batch",
			"method_batch",
			"device_batch",
			"risk_batch",
			"challenge_batch",
			"recovery_batch",
			"backup_code_batch",
			"policy_batch",
			"biometric_batch",
			"mfa_agent_batch",
		],
		"topics": [
			"mfau.profiles",
			"mfau.methods",
			"mfau.devices",
			"mfau.risk",
			"mfau.challenges",
			"mfau.recovery",
			"mfau.backup_codes",
			"mfau.policies",
			"mfau.biometrics",
			"mfau.agents",
		],
		"broker_core_dependency_allowed": False,
	}


STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"stream": "apg.mfau.lifecycle",
	"key": "tenant_id",
	"events": [
		"factor_enrolled",
		"factor_retired",
		"challenge_issued",
		"challenge_passed",
		"challenge_failed",
		"authentication_completed",
		"authentication_denied",
		"adaptive_policy_triggered",
		"risk_score_evaluated",
		"session_elevated",
		"agent_registered",
	],
	"guardrails": [
		"mfau_batch_requires_bytewax",
		"mfau_privileged_action_requires_human_approval",
	],
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
		"provides": ["multi_factor_authentication", "adaptive_authentication", "mfa_agent_composition"],
		"requires": ["auth", "secu", "encr", "aicr", "conf", "audl"],
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
		"agents": agent_manifest(),
		"streaming": deepcopy(STREAMING),
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
