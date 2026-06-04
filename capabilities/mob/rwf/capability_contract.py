"""Executable capability contract for APG Remote Workforce."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "mob_rwf"
CAPABILITY_NAME = "Remote Workforce"
CAPABILITY_VERSION = "1.0.0"
RWF_EVENT_STREAM = "apg.mob.rwf.lifecycle"

SUPPORTED_WORK_POLICY_TYPES = ["fully_remote", "hybrid", "flexible", "on_site", "field_ops", "travel_policy", "shift_work"]
SUPPORTED_WORK_POLICY_STATES = ["draft", "active", "under_review", "suspended", "retired"]
SUPPORTED_VPN_PROTOCOLS = ["wireguard", "openvpn", "ipsec", "ssl_tls", "zerotrust"]
SUPPORTED_VPN_STATES = ["pending", "active", "revoked", "expired", "suspended"]
SUPPORTED_PRODUCTIVITY_METRICS = ["active_hours", "task_completion", "response_time", "focus_score", "collaboration_score", "availability_score"]
SUPPORTED_EQUIPMENT_TYPES = ["laptop", "monitor", "keyboard", "mouse", "headset", "webcam", "docking_station", "desk_phone", "mobile_hotspot", "ergonomic_chair", "standing_desk"]
SUPPORTED_EQUIPMENT_STATES = ["requested", "approved", "shipped", "delivered", "in_use", "returned", "lost", "damaged"]
SUPPORTED_ONBOARDING_STATES = ["not_started", "in_progress", "pending_equipment", "pending_access", "completed", "paused", "cancelled"]
SUPPORTED_ONBOARDING_STEP_TYPES = ["identity_verification", "policy_acknowledgment", "equipment_setup", "vpn_provisioning", "tool_access", "security_training", "team_introduction", "manager_checkin"]
SUPPORTED_COMPLIANCE_CHECK_TYPES = ["policy_acknowledgment", "security_training", "equipment_audit", "vpn_usage", "data_handling", "working_hours_reporting"]
SUPPORTED_INCIDENT_TYPES = ["data_breach_attempt", "vpn_anomaly", "policy_violation", "equipment_loss", "unauthorized_access", "productivity_concern"]
SUPPORTED_LEAVE_TYPES = ["annual", "sick", "parental", "bereavement", "study", "unpaid", "compassionate"]
SUPPORTED_TIMEZONE_REGIONS = ["africa", "europe", "americas", "asia_pacific", "middle_east"]
SUPPORTED_COLLABORATION_TOOLS = ["slack", "teams", "zoom", "google_workspace", "notion", "jira", "github", "confluence"]
SUPPORTED_AGENT_ROLES = ["policy_advisor", "onboarding_coordinator", "vpn_provisioner", "productivity_analyst", "equipment_tracker", "compliance_monitor"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"work_policies": {
		"supported_types": SUPPORTED_WORK_POLICY_TYPES,
		"supported_states": SUPPORTED_WORK_POLICY_STATES,
		"approval_required": True,
		"acknowledgment_required": True,
		"review_cycle_days": 90,
	},
	"vpn": {
		"supported_protocols": SUPPORTED_VPN_PROTOCOLS,
		"supported_states": SUPPORTED_VPN_STATES,
		"approval_required": True,
		"mfa_required": True,
		"max_session_hours": 12,
		"split_tunneling_allowed": False,
	},
	"productivity": {
		"supported_metrics": SUPPORTED_PRODUCTIVITY_METRICS,
		"tracking_consent_required": True,
		"data_retention_days": 90,
		"aggregation_only": True,
	},
	"equipment": {
		"supported_types": SUPPORTED_EQUIPMENT_TYPES,
		"supported_states": SUPPORTED_EQUIPMENT_STATES,
		"approval_required": True,
		"asset_tracking_required": True,
		"max_items_per_employee": 5,
	},
	"onboarding": {
		"supported_states": SUPPORTED_ONBOARDING_STATES,
		"supported_step_types": SUPPORTED_ONBOARDING_STEP_TYPES,
		"manager_approval_required": True,
		"it_provisioning_required": True,
	},
	"compliance": {
		"supported_check_types": SUPPORTED_COMPLIANCE_CHECK_TYPES,
		"check_interval_days": 30,
		"grace_period_days": 7,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"cross_tenant_access_denied": True,
		"productivity_tracking_requires_consent": True,
		"vpn_split_tunneling_denied_by_default": True,
		"equipment_requisition_requires_approval": True,
		"onboarding_requires_manager_approval": True,
	},
	"observability": {
		"event_stream": RWF_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"nlp": "nlpc",
		"monitoring": "moni",
		"workflow": "wflo",
		"scheduling": "schd",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_policies": True,
		"enable_vpn": True,
		"enable_productivity": True,
		"enable_equipment": True,
		"enable_onboarding": True,
		"enable_compliance": True,
		"enable_incidents": True,
	},
	"theme": {"default_theme": "mob_rwf_workspace", "allow_tenant_overrides": True},
}

PROVIDES = [
	"remote_work_policy_management",
	"vpn_access_governance",
	"productivity_tracking_workflow",
	"equipment_requisition_workflow",
	"digital_onboarding_workflow",
	"remote_compliance_monitoring",
	"remote_incident_management",
	"onboarding_step_orchestration",
	"policy_acknowledgment_workflow",
	"remote_workforce_analytics",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "nlpc", "moni", "wflo", "schd", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mob-rwf/dashboard", "component": "RwfDashboard", "permission": "mob_rwf:view", "nav_group": "Overview"},
	{"name": "work_policies", "path": "/mob-rwf/policies", "component": "WorkPolicyManager", "permission": "mob_rwf:policies:list", "nav_group": "Policies"},
	{"name": "policy_detail", "path": "/mob-rwf/policies/<policy_id>", "component": "WorkPolicyDetail", "permission": "mob_rwf:policies:view", "nav_group": "Policies"},
	{"name": "policy_acknowledge", "path": "/mob-rwf/policies/<policy_id>/acknowledge", "component": "PolicyAcknowledgment", "permission": "mob_rwf:policies:acknowledge", "nav_group": "Policies"},
	{"name": "vpn_access", "path": "/mob-rwf/vpn", "component": "VpnAccessConsole", "permission": "mob_rwf:vpn:list", "nav_group": "VPN"},
	{"name": "vpn_provision", "path": "/mob-rwf/vpn/provision", "component": "VpnProvisioner", "permission": "mob_rwf:vpn:provision", "nav_group": "VPN"},
	{"name": "productivity", "path": "/mob-rwf/productivity", "component": "ProductivityDashboard", "permission": "mob_rwf:productivity:view", "nav_group": "Productivity"},
	{"name": "productivity_employee", "path": "/mob-rwf/productivity/<employee_id>", "component": "EmployeeProductivity", "permission": "mob_rwf:productivity:view", "nav_group": "Productivity"},
	{"name": "equipment", "path": "/mob-rwf/equipment", "component": "EquipmentInventory", "permission": "mob_rwf:equipment:list", "nav_group": "Equipment"},
	{"name": "equipment_request", "path": "/mob-rwf/equipment/request", "component": "EquipmentRequisition", "permission": "mob_rwf:equipment:request", "nav_group": "Equipment"},
	{"name": "onboarding", "path": "/mob-rwf/onboarding", "component": "OnboardingConsole", "permission": "mob_rwf:onboarding:list", "nav_group": "Onboarding"},
	{"name": "onboarding_detail", "path": "/mob-rwf/onboarding/<record_id>", "component": "OnboardingDetail", "permission": "mob_rwf:onboarding:view", "nav_group": "Onboarding"},
	{"name": "compliance", "path": "/mob-rwf/compliance", "component": "RemoteComplianceDashboard", "permission": "mob_rwf:compliance:view", "nav_group": "Compliance"},
	{"name": "incidents", "path": "/mob-rwf/incidents", "component": "RemoteIncidentQueue", "permission": "mob_rwf:incidents:view", "nav_group": "Incidents"},
	{"name": "settings", "path": "/mob-rwf/settings", "component": "RwfSettings", "permission": "mob_rwf:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mob_rwf_workspace",
	"tokens": {
		"color.primary": "#0F4C75",
		"color.accent": "#1B98E0",
		"color.success": "#1A936F",
		"color.warning": "#C9A227",
		"color.danger": "#C1121F",
		"surface.canvas": "#F8F9FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#212529",
		"text.secondary": "#6C757D",
		"border.radius": "10px",
		"density": "comfortable",
	},
	"components": {
		"work_policies": {"icon": "file-text", "status_indicator": "policy-state-chip"},
		"vpn": {"icon": "shield-lock", "status_indicator": "vpn-state-chip"},
		"productivity": {"icon": "activity", "status_indicator": "score-chip"},
		"equipment": {"icon": "package", "status_indicator": "equipment-state-chip"},
		"onboarding": {"icon": "user-plus", "status_indicator": "onboarding-state-chip"},
		"compliance": {"icon": "check-square", "status_indicator": "compliance-chip"},
		"incidents": {"icon": "alert-triangle", "status_indicator": "incident-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": RWF_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"work_policy_created",
		"work_policy_activated",
		"work_policy_acknowledged",
		"vpn_access_provisioned",
		"vpn_access_revoked",
		"vpn_session_started",
		"vpn_session_ended",
		"productivity_metric_recorded",
		"equipment_requested",
		"equipment_approved",
		"equipment_delivered",
		"equipment_returned",
		"onboarding_started",
		"onboarding_step_completed",
		"onboarding_completed",
		"compliance_check_completed",
		"remote_incident_raised",
		"remote_incident_resolved",
	],
	"guardrails": [
		"productivity_tracking_requires_consent",
		"vpn_split_tunneling_denied_by_default",
		"equipment_requisition_requires_approval",
		"onboarding_requires_manager_approval",
		"cross_tenant_access_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_policy"}},
	{"name": "work_policy_type_must_be_supported", "condition": {"operation": "create_work_policy", "policy_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_work_policy_type", "required_action": "select_supported_policy_type"}},
	{"name": "work_policy_activation_requires_approval", "condition": {"operation": "activate_work_policy", "approval_present": False}, "effect": {"decision": "deny", "reason": "work_policy_activation_requires_approval", "required_action": "obtain_policy_approval"}},
	{"name": "policy_acknowledgment_requires_active_policy", "condition": {"operation": "acknowledge_policy", "policy_state": "draft"}, "effect": {"decision": "deny", "reason": "only_active_policies_can_be_acknowledged", "required_action": "activate_policy_first"}},
	{"name": "vpn_protocol_must_be_supported", "condition": {"operation": "provision_vpn", "vpn_protocol_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_vpn_protocol", "required_action": "select_supported_vpn_protocol"}},
	{"name": "vpn_requires_approval", "condition": {"operation": "provision_vpn", "approval_present": False}, "effect": {"decision": "deny", "reason": "vpn_provisioning_requires_approval", "required_action": "obtain_vpn_approval"}},
	{"name": "vpn_requires_mfa", "condition": {"operation": "provision_vpn", "mfa_verified": False}, "effect": {"decision": "deny", "reason": "vpn_requires_mfa_verification", "required_action": "complete_mfa"}},
	{"name": "vpn_split_tunneling_denied", "condition": {"operation": "provision_vpn", "split_tunneling_requested": True}, "effect": {"decision": "deny", "reason": "vpn_split_tunneling_not_permitted", "required_action": "disable_split_tunneling"}},
	{"name": "productivity_tracking_requires_consent", "condition": {"operation": "record_productivity", "consent_given": False}, "effect": {"decision": "deny", "reason": "productivity_tracking_requires_employee_consent", "required_action": "obtain_employee_consent"}},
	{"name": "productivity_metric_must_be_supported", "condition": {"operation": "record_productivity", "metric_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_productivity_metric", "required_action": "select_supported_metric"}},
	{"name": "equipment_type_must_be_supported", "condition": {"operation": "request_equipment", "equipment_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_equipment_type", "required_action": "select_supported_equipment_type"}},
	{"name": "equipment_requires_approval", "condition": {"operation": "approve_equipment", "approval_present": False}, "effect": {"decision": "deny", "reason": "equipment_requisition_requires_approval", "required_action": "obtain_equipment_approval"}},
	{"name": "equipment_limit_per_employee", "condition": {"operation": "request_equipment", "equipment_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "equipment_limit_per_employee_exceeded", "required_action": "return_unused_equipment_first"}},
	{"name": "onboarding_requires_manager_approval", "condition": {"operation": "start_onboarding", "manager_approval_present": False}, "effect": {"decision": "deny", "reason": "onboarding_requires_manager_approval", "required_action": "obtain_manager_approval"}},
	{"name": "onboarding_step_type_must_be_supported", "condition": {"operation": "complete_onboarding_step", "step_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_onboarding_step_type", "required_action": "select_supported_step_type"}},
	{"name": "compliance_check_type_must_be_supported", "condition": {"operation": "record_compliance_check", "check_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_compliance_check_type", "required_action": "select_supported_check_type"}},
	{"name": "incident_type_must_be_supported", "condition": {"operation": "raise_incident", "incident_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_incident_type", "required_action": "select_supported_incident_type"}},
	{"name": "cross_tenant_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_not_permitted", "required_action": "use_tenant_scoped_context"}},
	{"name": "revoked_vpn_blocks_session", "condition": {"vpn_state": "revoked"}, "effect": {"decision": "deny", "reason": "revoked_vpn_cannot_start_session", "required_action": "provision_new_vpn_access"}},
	{"name": "suspended_vpn_blocks_session", "condition": {"vpn_state": "suspended"}, "effect": {"decision": "deny", "reason": "suspended_vpn_cannot_start_session", "required_action": "contact_it_to_reinstate_vpn"}},
	{"name": "retired_policy_cannot_be_acknowledged", "condition": {"operation": "acknowledge_policy", "policy_state": "retired"}, "effect": {"decision": "deny", "reason": "retired_policies_cannot_be_acknowledged", "required_action": "use_active_policy"}},
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
				"work_policies": {"type": "object"},
				"vpn": {"type": "object"},
				"productivity": {"type": "object"},
				"equipment": {"type": "object"},
				"onboarding": {"type": "object"},
				"compliance": {"type": "object"},
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
			"template_roots": ["mob/rwf/templates"],
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
