"""Executable capability contract for APG Mobile App Platform."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "mob_map"
CAPABILITY_NAME = "Mobile App Platform"
CAPABILITY_VERSION = "1.0.0"
MAP_EVENT_STREAM = "apg.mob.map.lifecycle"

SUPPORTED_PLATFORMS = ["ios", "android", "web_pwa", "windows", "macos", "linux"]
SUPPORTED_APP_STATES = ["draft", "review", "approved", "published", "suspended", "retired"]
SUPPORTED_SYNC_STRATEGIES = ["full", "incremental", "delta", "conflict_last_write_wins", "conflict_manual_resolution"]
SUPPORTED_SYNC_STATES = ["pending", "in_progress", "completed", "failed", "conflict", "cancelled"]
SUPPORTED_NOTIFICATION_CHANNELS = ["push_apns", "push_fcm", "push_web", "in_app", "sms", "email"]
SUPPORTED_AUTH_METHODS = ["biometric_fingerprint", "biometric_face", "pin", "password", "oauth2", "saml", "certificate", "passkey"]
SUPPORTED_BIOMETRIC_STATES = ["enrolled", "not_enrolled", "locked", "disabled", "suspended"]
SUPPORTED_VERSION_CHANNELS = ["alpha", "beta", "canary", "stable", "lts"]
SUPPORTED_UPDATE_POLICIES = ["mandatory", "recommended", "optional", "silent", "deferred"]
SUPPORTED_OFFLINE_MODES = ["read_only", "read_write", "full_offline", "disabled"]
SUPPORTED_ENCRYPTION_STANDARDS = ["aes_256_gcm", "chacha20_poly1305", "aes_128_gcm"]
SUPPORTED_COMPRESSION_ALGORITHMS = ["gzip", "brotli", "zstd", "lz4", "none"]
SUPPORTED_CONFLICT_POLICIES = ["server_wins", "client_wins", "last_write_wins", "manual", "merge"]
SUPPORTED_DEPLOYMENT_ENVIRONMENTS = ["development", "staging", "production", "hotfix"]
SUPPORTED_PERMISSION_SCOPES = ["camera", "location", "contacts", "notifications", "storage", "microphone", "biometric", "bluetooth", "nfc"]
SUPPORTED_APP_CATEGORIES = ["enterprise", "consumer", "field_ops", "kiosk", "embedded", "iot_companion"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"apps": {
		"supported_platforms": SUPPORTED_PLATFORMS,
		"supported_states": SUPPORTED_APP_STATES,
		"supported_categories": SUPPORTED_APP_CATEGORIES,
		"deployment_approval_required": True,
		"evidence_required": True,
	},
	"sync": {
		"supported_strategies": SUPPORTED_SYNC_STRATEGIES,
		"supported_states": SUPPORTED_SYNC_STATES,
		"supported_offline_modes": SUPPORTED_OFFLINE_MODES,
		"supported_conflict_policies": SUPPORTED_CONFLICT_POLICIES,
		"encryption_required": True,
		"compression_enabled": True,
	},
	"notifications": {
		"supported_channels": SUPPORTED_NOTIFICATION_CHANNELS,
		"approval_required": True,
		"rate_limit_per_device_per_hour": 50,
	},
	"auth": {
		"supported_methods": SUPPORTED_AUTH_METHODS,
		"supported_biometric_states": SUPPORTED_BIOMETRIC_STATES,
		"mfa_required_for_sensitive": True,
		"biometric_enrollment_audited": True,
	},
	"versions": {
		"supported_channels": SUPPORTED_VERSION_CHANNELS,
		"supported_update_policies": SUPPORTED_UPDATE_POLICIES,
		"supported_environments": SUPPORTED_DEPLOYMENT_ENVIRONMENTS,
		"rollback_supported": True,
		"phased_rollout_supported": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"cross_tenant_access_denied": True,
		"unapproved_deployment_denied": True,
		"unencrypted_sync_denied": True,
		"platform_scope_enforced": True,
	},
	"observability": {
		"event_stream": MAP_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"monitoring": "moni",
		"event_stream": "bytewax",
		"mdm": "mob_mdm",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_apps": True,
		"enable_versions": True,
		"enable_sync": True,
		"enable_notifications": True,
		"enable_auth": True,
		"enable_permissions": True,
		"enable_analytics": True,
	},
	"theme": {"default_theme": "mob_map_platform", "allow_tenant_overrides": True},
}

PROVIDES = [
	"mobile_app_registry",
	"cross_platform_build_workflow",
	"offline_sync_workflow",
	"push_notification_dispatch",
	"biometric_auth_enrollment",
	"app_version_management",
	"phased_rollout_workflow",
	"permission_scope_governance",
	"app_analytics_pipeline",
	"sync_conflict_resolution",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "moni", "mqeb", "mob_mdm"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mob-map/dashboard", "component": "MapDashboard", "permission": "mob_map:view", "nav_group": "Overview"},
	{"name": "apps", "path": "/mob-map/apps", "component": "AppRegistry", "permission": "mob_map:apps:list", "nav_group": "Applications"},
	{"name": "app_detail", "path": "/mob-map/apps/<app_id>", "component": "AppDetail", "permission": "mob_map:apps:view", "nav_group": "Applications"},
	{"name": "versions", "path": "/mob-map/versions", "component": "VersionManager", "permission": "mob_map:versions:list", "nav_group": "Releases"},
	{"name": "version_deploy", "path": "/mob-map/versions/<version_id>/deploy", "component": "VersionDeploy", "permission": "mob_map:versions:deploy", "nav_group": "Releases"},
	{"name": "sync_sessions", "path": "/mob-map/sync", "component": "SyncMonitor", "permission": "mob_map:sync:list", "nav_group": "Sync"},
	{"name": "sync_conflicts", "path": "/mob-map/sync/conflicts", "component": "ConflictResolver", "permission": "mob_map:sync:resolve", "nav_group": "Sync"},
	{"name": "push_notifications", "path": "/mob-map/notifications", "component": "PushNotificationConsole", "permission": "mob_map:notifications:list", "nav_group": "Notifications"},
	{"name": "send_notification", "path": "/mob-map/notifications/send", "component": "NotificationComposer", "permission": "mob_map:notifications:send", "nav_group": "Notifications"},
	{"name": "biometric_auth", "path": "/mob-map/auth/biometric", "component": "BiometricEnrollmentConsole", "permission": "mob_map:auth:manage", "nav_group": "Security"},
	{"name": "permission_scopes", "path": "/mob-map/permissions", "component": "PermissionScopeManager", "permission": "mob_map:permissions:manage", "nav_group": "Security"},
	{"name": "analytics", "path": "/mob-map/analytics", "component": "AppAnalytics", "permission": "mob_map:analytics:view", "nav_group": "Insights"},
	{"name": "settings", "path": "/mob-map/settings", "component": "MapSettings", "permission": "mob_map:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mob_map_platform",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#0EA5E9",
		"color.success": "#16A34A",
		"color.warning": "#D97706",
		"color.danger": "#DC2626",
		"surface.canvas": "#F0F9FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0F172A",
		"text.secondary": "#475569",
		"border.radius": "12px",
		"density": "comfortable",
	},
	"components": {
		"apps": {"icon": "smartphone", "status_indicator": "app-state-chip"},
		"versions": {"icon": "git-branch", "status_indicator": "version-channel-chip"},
		"sync": {"icon": "refresh-cw", "status_indicator": "sync-state-chip"},
		"notifications": {"icon": "bell", "status_indicator": "channel-chip"},
		"biometric": {"icon": "fingerprint", "status_indicator": "biometric-state-chip"},
		"permissions": {"icon": "shield", "status_indicator": "scope-chip"},
		"analytics": {"icon": "bar-chart-2", "status_indicator": "metric-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": MAP_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"app_registered",
		"app_state_changed",
		"app_version_published",
		"app_version_deployed",
		"sync_session_started",
		"sync_session_completed",
		"sync_conflict_detected",
		"sync_conflict_resolved",
		"push_notification_sent",
		"biometric_enrolled",
		"biometric_revoked",
		"permission_scope_granted",
		"permission_scope_revoked",
		"app_analytics_event",
	],
	"guardrails": [
		"unencrypted_sync_denied",
		"unapproved_deployment_denied",
		"cross_tenant_access_denied",
		"platform_scope_enforced",
		"notification_rate_limit_enforced",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_policy"}},
	{"name": "platform_must_be_supported", "condition": {"operation": "register_app", "platform_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_platform", "required_action": "select_supported_platform"}},
	{"name": "app_category_must_be_supported", "condition": {"operation": "register_app", "category_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_app_category", "required_action": "select_supported_category"}},
	{"name": "deployment_requires_approval", "condition": {"operation": "deploy_version", "approval_present": False}, "effect": {"decision": "deny", "reason": "deployment_approval_required", "required_action": "obtain_deployment_approval"}},
	{"name": "sync_encryption_mandatory", "condition": {"operation": "start_sync", "encryption_enabled": False}, "effect": {"decision": "deny", "reason": "sync_must_be_encrypted", "required_action": "enable_sync_encryption"}},
	{"name": "sync_strategy_must_be_supported", "condition": {"operation": "start_sync", "sync_strategy_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_sync_strategy", "required_action": "select_supported_sync_strategy"}},
	{"name": "offline_mode_must_be_supported", "condition": {"operation": "configure_offline", "offline_mode_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_offline_mode", "required_action": "select_supported_offline_mode"}},
	{"name": "notification_channel_must_be_supported", "condition": {"operation": "send_notification", "channel_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_notification_channel", "required_action": "select_supported_channel"}},
	{"name": "notification_requires_approval", "condition": {"operation": "send_notification", "approval_present": False}, "effect": {"decision": "deny", "reason": "notification_approval_required", "required_action": "obtain_notification_approval"}},
	{"name": "notification_rate_limit", "condition": {"operation": "send_notification", "rate_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "notification_rate_limit_exceeded", "required_action": "wait_rate_limit_window"}},
	{"name": "auth_method_must_be_supported", "condition": {"operation": "enroll_auth", "auth_method_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_auth_method", "required_action": "select_supported_auth_method"}},
	{"name": "biometric_requires_device_enrollment", "condition": {"operation": "enroll_biometric", "device_enrolled": False}, "effect": {"decision": "deny", "reason": "device_must_be_enrolled_before_biometric", "required_action": "enroll_device_first"}},
	{"name": "sensitive_operation_requires_mfa", "condition": {"operation_is_sensitive": True, "mfa_present": False}, "effect": {"decision": "deny", "reason": "mfa_required_for_sensitive_operations", "required_action": "complete_mfa"}},
	{"name": "permission_scope_must_be_supported", "condition": {"operation": "grant_permission", "scope_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_permission_scope", "required_action": "select_supported_scope"}},
	{"name": "version_channel_must_be_supported", "condition": {"operation": "publish_version", "channel_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_version_channel", "required_action": "select_supported_channel"}},
	{"name": "update_policy_must_be_supported", "condition": {"operation": "publish_version", "update_policy_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_update_policy", "required_action": "select_supported_update_policy"}},
	{"name": "cross_tenant_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_not_permitted", "required_action": "use_tenant_scoped_context"}},
	{"name": "app_suspension_requires_reason", "condition": {"operation": "suspend_app", "reason_present": False}, "effect": {"decision": "deny", "reason": "suspension_reason_required", "required_action": "provide_suspension_reason"}},
	{"name": "conflict_resolution_requires_policy", "condition": {"operation": "resolve_conflict", "conflict_policy_supported": False}, "effect": {"decision": "deny", "reason": "supported_conflict_policy_required", "required_action": "select_conflict_policy"}},
	{"name": "rollback_requires_previous_version", "condition": {"operation": "rollback_version", "previous_version_exists": False}, "effect": {"decision": "deny", "reason": "no_previous_version_to_rollback_to", "required_action": "check_version_history"}},
	{"name": "retired_app_blocks_deployment", "condition": {"operation": "deploy_version", "app_state": "retired"}, "effect": {"decision": "deny", "reason": "retired_apps_cannot_be_deployed", "required_action": "reinstate_app_before_deploying"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	"""Return full capability contract scoped to tenant."""
	cfg = deepcopy(DEFAULT_CONFIGURATION)
	cfg["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": cfg,
		"configuration_schema": {
			"required": ["tenant_id", "ui", "theme"],
			"properties": {
				"tenant_id": {"type": "string"},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
				"apps": {"type": "object"},
				"sync": {"type": "object"},
				"notifications": {"type": "object"},
				"auth": {"type": "object"},
				"versions": {"type": "object"},
			},
		},
		"rule_engine": {
			"type": "deterministic",
			"default_decision": "allow",
			"rules": RULES,
		},
		"ui": {
			"shell": "apg_python",
			"requires_theme": True,
			"template_roots": ["mob/map/templates"],
			"routes": UI_ROUTES,
		},
		"theme": THEME,
		"provides": PROVIDES,
		"requires": REQUIRES,
		"streaming": STREAMING,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic rules against a context dict. Returns first matching deny, else allow."""
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			return {"decision": rule["effect"]["decision"], "rule": rule["name"], "reason": rule["effect"]["reason"], "required_action": rule["effect"]["required_action"]}
	return {"decision": "allow", "rule": None, "reason": "no_matching_deny_rule", "required_action": None}
