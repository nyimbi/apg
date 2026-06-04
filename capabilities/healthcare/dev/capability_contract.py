"""Executable capability contract for APG Medical Device Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "healthcare_dev"
CAPABILITY_NAME = "Medical Device Management"
CAPABILITY_VERSION = "1.0.0"
DEV_EVENT_STREAM = "apg.healthcare.dev.lifecycle"

SUPPORTED_DEVICE_CLASSES = ["class_i", "class_ii", "class_iii"]
SUPPORTED_DEVICE_TYPES = [
	"infusion_pump", "ventilator", "defibrillator", "patient_monitor",
	"imaging_system", "laboratory_analyzer", "surgical_robot", "implantable",
	"diagnostic_kit", "ppm_device", "dialysis_machine", "anesthesia_machine",
]
SUPPORTED_MAINTENANCE_TYPES = ["preventive", "corrective", "calibration", "inspection", "qualification"]
SUPPORTED_DEVICE_STATUSES = ["active", "in_maintenance", "out_of_service", "recalled", "retired", "on_loan"]
SUPPORTED_CALIBRATION_STATUSES = ["current", "overdue", "in_progress", "failed", "not_required"]
SUPPORTED_ADVERSE_EVENT_TYPES = [
	"malfunction", "patient_injury", "near_miss", "unexpected_shutdown",
	"alarm_failure", "data_error", "software_error", "battery_failure",
]
SUPPORTED_ADVERSE_EVENT_SEVERITIES = ["minor", "moderate", "serious", "life_threatening", "death"]
SUPPORTED_UDI_FORMATS = ["gs1", "hibcc", "iccbba"]
SUPPORTED_WORK_ORDER_STATUSES = ["open", "in_progress", "pending_parts", "completed", "cancelled"]
SUPPORTED_AGENT_ROLES = ["device_steward", "maintenance_reviewer", "adverse_event_reviewer", "calibration_reviewer"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"devices": {"supported_classes": SUPPORTED_DEVICE_CLASSES, "supported_types": SUPPORTED_DEVICE_TYPES, "udi_required_for_class_ii_iii": True},
	"maintenance": {"supported_types": SUPPORTED_MAINTENANCE_TYPES, "supported_work_order_statuses": SUPPORTED_WORK_ORDER_STATUSES, "preventive_schedule_required": True},
	"calibration": {"supported_statuses": SUPPORTED_CALIBRATION_STATUSES, "certificate_required": True, "overdue_alert_days": 7},
	"adverse_events": {"supported_types": SUPPORTED_ADVERSE_EVENT_TYPES, "supported_severities": SUPPORTED_ADVERSE_EVENT_SEVERITIES, "fda_mdr_reporting_threshold": "serious"},
	"governance": {
		"require_tenant_context": True, "policy_attached_for_writes": True,
		"audit_events": True, "udi_required_for_implantable": True,
		"cross_tenant_device_access_denied": True,
		"recalled_device_use_denied": True,
		"calibration_overdue_blocks_use": True,
		"adverse_event_reporting_required_for_serious": True,
	},
	"observability": {"event_stream": DEV_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "scheduler": "schd", "monitoring": "moni", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_inventory": True, "enable_maintenance": True, "enable_calibration": True, "enable_adverse_events": True, "enable_udi": True},
	"theme": {"default_theme": "healthcare_dev_clinical", "allow_tenant_overrides": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
}

PROVIDES = [
	"device_inventory_management", "maintenance_schedule_management",
	"calibration_record_tracking", "fda_udi_tracking",
	"adverse_event_reporting", "work_order_management",
	"device_lifecycle_management", "regulatory_submission_support",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "schd", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/healthcare-dev/dashboard", "component": "DevDashboard", "permission": "healthcare_dev:view", "nav_group": "Overview"},
	{"name": "inventory", "path": "/healthcare-dev/inventory", "component": "DevInventoryList", "permission": "healthcare_dev:inventory", "nav_group": "Devices"},
	{"name": "device_register", "path": "/healthcare-dev/inventory/register", "component": "DevDeviceRegister", "permission": "healthcare_dev:inventory_write", "nav_group": "Devices"},
	{"name": "device_detail", "path": "/healthcare-dev/inventory/<id>", "component": "DevDeviceDetail", "permission": "healthcare_dev:inventory", "nav_group": "Devices"},
	{"name": "maintenance", "path": "/healthcare-dev/maintenance", "component": "DevMaintenanceQueue", "permission": "healthcare_dev:maintenance", "nav_group": "Maintenance"},
	{"name": "work_orders", "path": "/healthcare-dev/work-orders", "component": "DevWorkOrderList", "permission": "healthcare_dev:maintenance", "nav_group": "Maintenance"},
	{"name": "calibration", "path": "/healthcare-dev/calibration", "component": "DevCalibrationLog", "permission": "healthcare_dev:calibration", "nav_group": "Calibration"},
	{"name": "adverse_events", "path": "/healthcare-dev/adverse-events", "component": "DevAdverseEventList", "permission": "healthcare_dev:adverse_events", "nav_group": "Safety"},
	{"name": "adverse_event_new", "path": "/healthcare-dev/adverse-events/new", "component": "DevAdverseEventForm", "permission": "healthcare_dev:adverse_events_write", "nav_group": "Safety"},
	{"name": "udi_lookup", "path": "/healthcare-dev/udi", "component": "DevUdiLookup", "permission": "healthcare_dev:udi", "nav_group": "Compliance"},
	{"name": "recalls", "path": "/healthcare-dev/recalls", "component": "DevRecallList", "permission": "healthcare_dev:recalls", "nav_group": "Compliance"},
	{"name": "agents", "path": "/healthcare-dev/agents", "component": "DevAgentWorkbench", "permission": "healthcare_dev:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/healthcare-dev/settings", "component": "DevSettings", "permission": "healthcare_dev:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "healthcare_dev_clinical",
	"tokens": {
		"color.primary": "#374151", "color.accent": "#2563EB", "color.success": "#166534",
		"color.warning": "#A16207", "color.danger": "#B91C1C",
		"surface.canvas": "#F9FAFB", "surface.panel": "#FFFFFF",
		"text.primary": "#111827", "text.secondary": "#4B5563",
		"border.radius": "6px", "density": "compact",
	},
	"components": {
		"devices": {"icon": "monitor", "status_indicator": "device-status-chip"},
		"maintenance": {"icon": "tool", "status_indicator": "maintenance-type-chip"},
		"calibration": {"icon": "sliders", "status_indicator": "calibration-status-chip"},
		"adverse_events": {"icon": "alert-triangle", "status_indicator": "adverse-severity-chip"},
		"udi": {"icon": "tag", "status_indicator": "udi-format-chip"},
		"recalls": {"icon": "x-octagon", "status_indicator": "recall-status-chip"},
		"work_orders": {"icon": "clipboard-list", "status_indicator": "work-order-status-chip"},
		"agents": {"icon": "cpu", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax", "stream": DEV_EVENT_STREAM, "key": "tenant_id",
	"events": [
		"device_registered", "device_status_changed", "maintenance_scheduled",
		"work_order_completed", "calibration_recorded", "calibration_overdue",
		"adverse_event_reported", "device_recalled",
	],
	"guardrails": [
		"recalled_device_use_denied", "calibration_overdue_blocks_use",
		"udi_required_for_class_ii_iii", "adverse_event_reporting_required_for_serious",
		"privileged_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "cross_tenant_device_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_device_access_prohibited", "required_action": "use_tenant_scoped_query"}},
	{"name": "recalled_device_use_denied", "condition": {"operation": "assign_device", "device_status": "recalled"}, "effect": {"decision": "deny", "reason": "recalled_device_cannot_be_assigned", "required_action": "quarantine_recalled_device"}},
	{"name": "calibration_overdue_blocks_use", "condition": {"operation": "assign_device", "calibration_status": "overdue"}, "effect": {"decision": "deny", "reason": "calibration_overdue_device_cannot_be_used", "required_action": "complete_calibration_first"}},
	{"name": "udi_required_for_class_ii_iii", "condition": {"operation": "register_device", "device_class_requires_udi": True, "udi_present": False}, "effect": {"decision": "deny", "reason": "udi_required_for_class_ii_and_iii_devices", "required_action": "provide_fda_udi"}},
	{"name": "device_type_supported", "condition": {"operation": "register_device", "device_type_supported": False}, "effect": {"decision": "deny", "reason": "device_type_not_supported", "required_action": "select_supported_device_type"}},
	{"name": "device_class_supported", "condition": {"operation": "register_device", "device_class_supported": False}, "effect": {"decision": "deny", "reason": "device_class_not_supported", "required_action": "select_supported_device_class"}},
	{"name": "maintenance_type_supported", "condition": {"operation": "schedule_maintenance", "maintenance_type_supported": False}, "effect": {"decision": "deny", "reason": "maintenance_type_not_supported", "required_action": "select_supported_maintenance_type"}},
	{"name": "work_order_status_supported", "condition": {"operation": "update_work_order", "work_order_status_supported": False}, "effect": {"decision": "deny", "reason": "work_order_status_not_supported", "required_action": "select_supported_work_order_status"}},
	{"name": "adverse_event_type_supported", "condition": {"operation": "report_adverse_event", "adverse_event_type_supported": False}, "effect": {"decision": "deny", "reason": "adverse_event_type_not_supported", "required_action": "select_supported_adverse_event_type"}},
	{"name": "adverse_event_severity_supported", "condition": {"operation": "report_adverse_event", "adverse_event_severity_supported": False}, "effect": {"decision": "deny", "reason": "adverse_event_severity_not_supported", "required_action": "select_supported_adverse_event_severity"}},
	{"name": "serious_adverse_event_requires_fda_report", "condition": {"operation": "report_adverse_event", "severity": "serious", "fda_mdr_initiated": False}, "effect": {"decision": "warn", "reason": "serious_adverse_event_may_require_fda_mdr_report", "required_action": "initiate_fda_mdr_report"}},
	{"name": "udi_format_supported", "condition": {"operation": "register_udi", "udi_format_supported": False}, "effect": {"decision": "deny", "reason": "udi_format_not_supported", "required_action": "select_supported_udi_format"}},
	{"name": "out_of_service_device_not_assignable", "condition": {"operation": "assign_device", "device_status": "out_of_service"}, "effect": {"decision": "deny", "reason": "out_of_service_device_cannot_be_assigned", "required_action": "restore_device_to_service"}},
	{"name": "retired_device_not_modifiable", "condition": {"operation": "update_device", "device_status": "retired"}, "effect": {"decision": "deny", "reason": "retired_device_record_is_locked", "required_action": "use_amendment_workflow"}},
	{"name": "calibration_certificate_required", "condition": {"operation": "record_calibration", "certificate_present": False}, "effect": {"decision": "deny", "reason": "calibration_certificate_required", "required_action": "attach_calibration_certificate"}},
	{"name": "device_status_supported", "condition": {"operation": "update_device_status", "device_status_supported": False}, "effect": {"decision": "deny", "reason": "device_status_not_supported", "required_action": "select_supported_device_status"}},
	{"name": "agent_privileged_action_requires_approval", "condition": {"agent_action": True, "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "privileged_agent_action_requires_human_approval", "required_action": "record_human_approval"}},
	{"name": "calibration_overdue_warning", "condition": {"operation": "check_device", "calibration_status": "overdue"}, "effect": {"decision": "warn", "reason": "device_calibration_overdue", "required_action": "schedule_calibration"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {"required": ["tenant_id", "ui", "theme"], "properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}}},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["healthcare/dev/templates"], "routes": UI_ROUTES},
		"theme": THEME, "streaming": STREAMING, "provides": PROVIDES, "requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			effect = rule["effect"]
			return {"rule": rule["name"], "decision": effect["decision"], "reason": effect["reason"], "required_action": effect.get("required_action")}
	return {"rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
