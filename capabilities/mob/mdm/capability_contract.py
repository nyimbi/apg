"""Executable capability contract for APG Mobile Device Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "mob_mdm"
CAPABILITY_NAME = "Mobile Device Management"
CAPABILITY_VERSION = "1.0.0"
MDM_EVENT_STREAM = "apg.mob.mdm.lifecycle"

SUPPORTED_DEVICE_TYPES = ["smartphone", "tablet", "laptop", "desktop", "iot_gateway", "rugged_device", "kiosk", "wearable"]
SUPPORTED_OS_PLATFORMS = ["ios", "android", "windows", "macos", "linux", "chromeos", "tizen"]
SUPPORTED_ENROLMENT_METHODS = ["dep", "zero_touch", "qr_code", "email_invite", "manual", "nfc_tap", "bulk_csv"]
SUPPORTED_ENROLMENT_STATES = ["pending", "enrolled", "unenrolled", "suspended", "wiped", "blocked"]
SUPPORTED_OWNERSHIP_TYPES = ["corporate_owned", "byod", "copo", "company_shared"]
SUPPORTED_COMPLIANCE_STATES = ["compliant", "non_compliant", "pending_evaluation", "grace_period", "exempted"]
SUPPORTED_POLICY_TYPES = ["security", "network", "application", "passcode", "encryption", "vpn", "wifi", "certificate", "restrictions", "kiosk"]
SUPPORTED_POLICY_STATES = ["draft", "active", "suspended", "retired"]
SUPPORTED_APP_DISTRIBUTION_TYPES = ["required", "available", "blocked", "removal_required"]
SUPPORTED_WIPE_TYPES = ["full_wipe", "selective_wipe", "corporate_wipe", "factory_reset"]
SUPPORTED_WIPE_STATES = ["pending", "in_progress", "completed", "failed", "cancelled"]
SUPPORTED_PROFILE_TYPES = ["configuration", "certificate", "vpn", "wifi", "email", "ldap", "web_clip", "font", "custom"]
SUPPORTED_ALERT_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_AGENT_ROLES = ["enrollment_agent", "policy_enforcer", "compliance_monitor", "wipe_executor", "profile_deployer", "app_distributor"]
SUPPORTED_LOCK_ACTIONS = ["lock", "unlock", "lost_mode", "activation_lock"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"devices": {
		"supported_types": SUPPORTED_DEVICE_TYPES,
		"supported_os_platforms": SUPPORTED_OS_PLATFORMS,
		"supported_enrolment_methods": SUPPORTED_ENROLMENT_METHODS,
		"supported_enrolment_states": SUPPORTED_ENROLMENT_STATES,
		"supported_ownership_types": SUPPORTED_OWNERSHIP_TYPES,
		"approval_required_for_enrolment": True,
		"evidence_required": True,
	},
	"policies": {
		"supported_policy_types": SUPPORTED_POLICY_TYPES,
		"supported_states": SUPPORTED_POLICY_STATES,
		"approval_required": True,
		"version_history_kept": True,
	},
	"compliance": {
		"supported_states": SUPPORTED_COMPLIANCE_STATES,
		"evaluation_interval_minutes": 60,
		"grace_period_hours": 24,
		"auto_remediation_enabled": True,
	},
	"app_distribution": {
		"supported_types": SUPPORTED_APP_DISTRIBUTION_TYPES,
		"silent_install_supported": True,
		"approval_required": True,
	},
	"remote_actions": {
		"wipe_types": SUPPORTED_WIPE_TYPES,
		"wipe_approval_required": True,
		"lock_actions": SUPPORTED_LOCK_ACTIONS,
		"lock_approval_required": False,
	},
	"profiles": {
		"supported_types": SUPPORTED_PROFILE_TYPES,
		"signing_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"wipe_requires_dual_approval": True,
		"cross_tenant_access_denied": True,
		"unenrolled_device_blocks_app_install": True,
		"non_compliant_device_blocks_access": True,
	},
	"observability": {
		"event_stream": MDM_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"compliance": "comp",
		"monitoring": "moni",
		"workflow": "wflo",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_devices": True,
		"enable_policies": True,
		"enable_compliance": True,
		"enable_apps": True,
		"enable_profiles": True,
		"enable_remote_actions": True,
		"enable_alerts": True,
	},
	"theme": {"default_theme": "mob_mdm_console", "allow_tenant_overrides": True},
}

PROVIDES = [
	"device_enrolment_workflow",
	"mdm_policy_enforcement",
	"compliance_monitoring",
	"remote_wipe_workflow",
	"app_distribution_workflow",
	"mdm_profile_deployment",
	"device_lock_workflow",
	"enrolment_state_machine",
	"corporate_wipe_workflow",
	"device_inventory_registry",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "comp", "moni", "wflo", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mob-mdm/dashboard", "component": "MdmDashboard", "permission": "mob_mdm:view", "nav_group": "Overview"},
	{"name": "devices", "path": "/mob-mdm/devices", "component": "DeviceInventory", "permission": "mob_mdm:devices:list", "nav_group": "Devices"},
	{"name": "device_detail", "path": "/mob-mdm/devices/<device_id>", "component": "DeviceDetail", "permission": "mob_mdm:devices:view", "nav_group": "Devices"},
	{"name": "enrolment", "path": "/mob-mdm/enrolment", "component": "EnrolmentConsole", "permission": "mob_mdm:enrolment:manage", "nav_group": "Devices"},
	{"name": "policies", "path": "/mob-mdm/policies", "component": "PolicyWorkbench", "permission": "mob_mdm:policies:list", "nav_group": "Policies"},
	{"name": "policy_detail", "path": "/mob-mdm/policies/<policy_id>", "component": "PolicyDetail", "permission": "mob_mdm:policies:view", "nav_group": "Policies"},
	{"name": "compliance", "path": "/mob-mdm/compliance", "component": "ComplianceDashboard", "permission": "mob_mdm:compliance:view", "nav_group": "Compliance"},
	{"name": "compliance_detail", "path": "/mob-mdm/compliance/<device_id>", "component": "DeviceComplianceDetail", "permission": "mob_mdm:compliance:view", "nav_group": "Compliance"},
	{"name": "apps", "path": "/mob-mdm/apps", "component": "AppDistributionConsole", "permission": "mob_mdm:apps:list", "nav_group": "Applications"},
	{"name": "profiles", "path": "/mob-mdm/profiles", "component": "ProfileManager", "permission": "mob_mdm:profiles:list", "nav_group": "Profiles"},
	{"name": "remote_actions", "path": "/mob-mdm/remote-actions", "component": "RemoteActionConsole", "permission": "mob_mdm:remote:execute", "nav_group": "Actions"},
	{"name": "wipe_requests", "path": "/mob-mdm/remote-actions/wipes", "component": "WipeRequestQueue", "permission": "mob_mdm:remote:wipe", "nav_group": "Actions"},
	{"name": "alerts", "path": "/mob-mdm/alerts", "component": "MdmAlertQueue", "permission": "mob_mdm:alerts:view", "nav_group": "Monitoring"},
	{"name": "settings", "path": "/mob-mdm/settings", "component": "MdmSettings", "permission": "mob_mdm:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mob_mdm_console",
	"tokens": {
		"color.primary": "#1E3A5F",
		"color.accent": "#2563EB",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F1F5F9",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0F172A",
		"text.secondary": "#334155",
		"border.radius": "6px",
		"density": "compact",
	},
	"components": {
		"devices": {"icon": "monitor-smartphone", "status_indicator": "enrolment-state-chip"},
		"policies": {"icon": "file-shield", "status_indicator": "policy-state-chip"},
		"compliance": {"icon": "shield-check", "status_indicator": "compliance-state-chip"},
		"apps": {"icon": "package", "status_indicator": "distribution-type-chip"},
		"profiles": {"icon": "layers", "status_indicator": "profile-type-chip"},
		"remote_actions": {"icon": "zap", "status_indicator": "action-state-chip"},
		"wipes": {"icon": "trash-2", "status_indicator": "wipe-state-chip"},
		"alerts": {"icon": "bell-ring", "status_indicator": "severity-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": MDM_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"device_enrolled",
		"device_unenrolled",
		"device_suspended",
		"device_wiped",
		"policy_created",
		"policy_activated",
		"policy_assigned",
		"compliance_evaluated",
		"compliance_state_changed",
		"app_distributed",
		"app_removed",
		"profile_deployed",
		"profile_removed",
		"wipe_requested",
		"wipe_completed",
		"device_locked",
		"device_unlocked",
		"mdm_alert_raised",
	],
	"guardrails": [
		"wipe_requires_dual_approval",
		"cross_tenant_access_denied",
		"unenrolled_device_blocks_app_install",
		"non_compliant_device_blocks_access",
		"policy_activation_requires_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_policy"}},
	{"name": "device_type_must_be_supported", "condition": {"operation": "enrol_device", "device_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_device_type", "required_action": "select_supported_device_type"}},
	{"name": "os_platform_must_be_supported", "condition": {"operation": "enrol_device", "os_platform_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_os_platform", "required_action": "select_supported_os_platform"}},
	{"name": "enrolment_method_must_be_supported", "condition": {"operation": "enrol_device", "enrolment_method_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_enrolment_method", "required_action": "select_supported_enrolment_method"}},
	{"name": "enrolment_requires_approval", "condition": {"operation": "enrol_device", "approval_present": False}, "effect": {"decision": "deny", "reason": "enrolment_approval_required", "required_action": "obtain_enrolment_approval"}},
	{"name": "policy_type_must_be_supported", "condition": {"operation": "create_policy", "policy_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_policy_type", "required_action": "select_supported_policy_type"}},
	{"name": "policy_activation_requires_approval", "condition": {"operation": "activate_policy", "approval_present": False}, "effect": {"decision": "deny", "reason": "policy_activation_approval_required", "required_action": "obtain_policy_approval"}},
	{"name": "wipe_requires_approval", "condition": {"operation": "request_wipe", "approval_present": False}, "effect": {"decision": "deny", "reason": "wipe_approval_required", "required_action": "obtain_wipe_approval"}},
	{"name": "wipe_requires_dual_approval", "condition": {"operation": "request_wipe", "dual_approval_present": False}, "effect": {"decision": "deny", "reason": "wipe_requires_dual_approval", "required_action": "obtain_second_approver"}},
	{"name": "wipe_type_must_be_supported", "condition": {"operation": "request_wipe", "wipe_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_wipe_type", "required_action": "select_supported_wipe_type"}},
	{"name": "unenrolled_device_blocks_app_install", "condition": {"operation": "distribute_app", "device_enrolled": False}, "effect": {"decision": "deny", "reason": "device_must_be_enrolled_for_app_distribution", "required_action": "enrol_device_first"}},
	{"name": "app_distribution_type_must_be_supported", "condition": {"operation": "distribute_app", "distribution_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_app_distribution_type", "required_action": "select_supported_distribution_type"}},
	{"name": "profile_type_must_be_supported", "condition": {"operation": "deploy_profile", "profile_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_profile_type", "required_action": "select_supported_profile_type"}},
	{"name": "profile_requires_enrolled_device", "condition": {"operation": "deploy_profile", "device_enrolled": False}, "effect": {"decision": "deny", "reason": "device_must_be_enrolled_for_profile_deployment", "required_action": "enrol_device_first"}},
	{"name": "non_compliant_device_blocks_access", "condition": {"operation": "grant_access", "device_compliance_state": "non_compliant"}, "effect": {"decision": "deny", "reason": "non_compliant_device_blocked", "required_action": "remediate_compliance_issues"}},
	{"name": "suspended_device_blocks_all_actions", "condition": {"device_state": "suspended"}, "effect": {"decision": "deny", "reason": "suspended_device_blocked", "required_action": "reinstate_device"}},
	{"name": "wiped_device_blocks_all_actions", "condition": {"device_state": "wiped"}, "effect": {"decision": "deny", "reason": "wiped_device_blocked", "required_action": "re_enrol_device"}},
	{"name": "cross_tenant_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_not_permitted", "required_action": "use_tenant_scoped_context"}},
	{"name": "lock_action_must_be_supported", "condition": {"operation": "lock_device", "lock_action_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_lock_action", "required_action": "select_supported_lock_action"}},
	{"name": "compliance_evaluation_requires_enrolled_device", "condition": {"operation": "evaluate_compliance", "device_enrolled": False}, "effect": {"decision": "deny", "reason": "only_enrolled_devices_can_be_evaluated", "required_action": "enrol_device_first"}},
	{"name": "ownership_type_must_be_supported", "condition": {"operation": "enrol_device", "ownership_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_ownership_type", "required_action": "select_supported_ownership_type"}},
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
				"devices": {"type": "object"},
				"policies": {"type": "object"},
				"compliance": {"type": "object"},
				"app_distribution": {"type": "object"},
				"remote_actions": {"type": "object"},
				"profiles": {"type": "object"},
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
			"template_roots": ["mob/mdm/templates"],
			"routes": UI_ROUTES,
		},
		"theme": THEME,
		"provides": PROVIDES,
		"requires": REQUIRES,
		"streaming": STREAMING,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic rules against context. Returns first matching deny, else allow."""
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			return {"decision": rule["effect"]["decision"], "rule": rule["name"], "reason": rule["effect"]["reason"], "required_action": rule["effect"]["required_action"]}
	return {"decision": "allow", "rule": None, "reason": "no_matching_deny_rule", "required_action": None}
