"""Executable capability contract for APG Smart Metering & AMI."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "energy_met"
CAPABILITY_NAME = "Smart Metering & AMI"
CAPABILITY_VERSION = "1.0.0"
MET_EVENT_STREAM = "apg.energy.met.lifecycle"

SUPPORTED_METER_TYPES = ["smart_meter_electricity", "smart_meter_gas", "smart_meter_water", "smart_meter_heat", "prepayment_meter", "interval_meter", "net_meter", "revenue_grade_meter", "submetering_unit"]
SUPPORTED_COMMUNICATION_TECHNOLOGIES = ["plc_g3", "plc_prime", "rf_mesh_900mhz", "rf_mesh_2_4ghz", "nbiot", "lte_cat_m1", "lorawan", "zigbee", "wifi", "gprs"]
SUPPORTED_METER_STATUSES = ["active", "inactive", "tampered", "disconnected", "faulty", "replaced", "decommissioned", "awaiting_installation"]
SUPPORTED_READING_TYPES = ["active_energy_import", "active_energy_export", "reactive_energy_import", "reactive_energy_export", "apparent_energy", "max_demand", "average_power_factor", "voltage", "current", "frequency"]
SUPPORTED_INTERVAL_LENGTHS = ["1min", "5min", "15min", "30min", "60min"]
SUPPORTED_TAMPER_TYPES = ["magnetic_tamper", "cover_open", "meter_tilt", "terminal_cover", "strong_magnetic_field", "neutral_missing", "reverse_current", "bypass_detected"]
SUPPORTED_COMMAND_TYPES = ["remote_connect", "remote_disconnect", "set_load_limit", "clear_load_limit", "on_demand_read", "time_sync", "firmware_update", "demand_response_activate", "demand_response_deactivate"]
SUPPORTED_COMMAND_STATUSES = ["pending", "sent", "acknowledged", "executed", "failed", "timed_out", "rejected"]
SUPPORTED_DR_EVENT_TYPES = ["direct_load_control", "price_signal", "emergency_curtailment", "capacity_commitment", "voluntary_reduction"]
SUPPORTED_DR_STATUSES = ["active", "completed", "cancelled", "failed", "partial"]
SUPPORTED_DATA_QUALITY_FLAGS = ["valid", "estimated", "substituted", "missing", "suspect", "calibration", "test"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["tamper_detector", "dr_coordinator", "mdm_analyst", "data_quality_manager", "ami_monitor"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ui": {"enable_dashboard": True, "enable_meters": True, "enable_readings": True, "enable_tamper": True, "enable_commands": True, "enable_demand_response": True, "enable_data_quality": True},
	"theme": {"default_theme": "energy_met_ami", "allow_tenant_overrides": True},
	"meters": {"supported_types": SUPPORTED_METER_TYPES, "supported_statuses": SUPPORTED_METER_STATUSES, "supported_comm_tech": SUPPORTED_COMMUNICATION_TECHNOLOGIES, "serial_required": True},
	"readings": {"supported_reading_types": SUPPORTED_READING_TYPES, "supported_intervals": SUPPORTED_INTERVAL_LENGTHS, "supported_quality_flags": SUPPORTED_DATA_QUALITY_FLAGS, "retention_days": 730},
	"tamper": {"supported_tamper_types": SUPPORTED_TAMPER_TYPES, "auto_alert": True, "evidence_required": True},
	"commands": {"supported_types": SUPPORTED_COMMAND_TYPES, "supported_statuses": SUPPORTED_COMMAND_STATUSES, "approval_required_for_disconnect": True, "retry_limit": 3},
	"demand_response": {"supported_event_types": SUPPORTED_DR_EVENT_TYPES, "supported_statuses": SUPPORTED_DR_STATUSES, "opt_out_allowed": True, "notification_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_disconnect": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_denied": True, "unapproved_disconnect_denied": True},
	"observability": {"event_stream": MET_EVENT_STREAM, "stream_processor": "bytewax"},
}

PROVIDES = [
	"meter_registry",
	"ami_head_end_management",
	"interval_data_collection",
	"tamper_detection",
	"remote_connect_disconnect",
	"demand_response_coordination",
	"data_quality_management",
	"meter_data_export",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/energy-met/dashboard", "component": "MetDashboard", "permission": "energy_met:view", "nav_group": "Overview"},
	{"name": "meters", "path": "/energy-met/meters", "component": "MeterRegistry", "permission": "energy_met:meters", "nav_group": "Assets"},
	{"name": "meter_detail", "path": "/energy-met/meters/<id>", "component": "MeterDetail", "permission": "energy_met:meters", "nav_group": "Assets"},
	{"name": "readings", "path": "/energy-met/readings", "component": "IntervalDataConsole", "permission": "energy_met:readings", "nav_group": "Data"},
	{"name": "tamper", "path": "/energy-met/tamper", "component": "TamperAlertConsole", "permission": "energy_met:tamper", "nav_group": "Security"},
	{"name": "commands", "path": "/energy-met/commands", "component": "RemoteCommandCenter", "permission": "energy_met:commands", "nav_group": "Operations"},
	{"name": "demand_response", "path": "/energy-met/demand-response", "component": "DemandResponseConsole", "permission": "energy_met:demand_response", "nav_group": "Programs"},
	{"name": "data_quality", "path": "/energy-met/data-quality", "component": "DataQualityConsole", "permission": "energy_met:data_quality", "nav_group": "Quality"},
	{"name": "head_end", "path": "/energy-met/head-end", "component": "AmiHeadEndStatus", "permission": "energy_met:admin", "nav_group": "Infrastructure"},
	{"name": "reports", "path": "/energy-met/reports", "component": "MeteringReports", "permission": "energy_met:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/energy-met/agents", "component": "MetAgentWorkbench", "permission": "energy_met:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/energy-met/settings", "component": "MetSettings", "permission": "energy_met:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "energy_met_ami",
	"tokens": {
		"color.primary": "#7C3AED",
		"color.accent": "#0891B2",
		"color.success": "#15803D",
		"color.warning": "#D97706",
		"color.danger": "#DC2626",
		"surface.canvas": "#F5F3FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E1B4B",
		"text.secondary": "#4C1D95",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"meters": {"icon": "cpu", "status_indicator": "meter-status-chip"},
		"readings": {"icon": "trending-up", "status_indicator": "data-quality-chip"},
		"tamper": {"icon": "shield-alert", "status_indicator": "tamper-type-chip"},
		"commands": {"icon": "terminal", "status_indicator": "command-status-chip"},
		"demand_response": {"icon": "battery-charging", "status_indicator": "dr-status-chip"},
		"data_quality": {"icon": "check-circle", "status_indicator": "quality-flag-chip"},
		"head_end": {"icon": "radio", "status_indicator": "comm-tech-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": MET_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"meter_registered", "meter_status_changed", "interval_reading_received",
		"tamper_event_detected", "remote_command_sent", "remote_command_executed",
		"demand_response_event_created", "demand_response_event_completed",
		"data_quality_flag_set", "ami_head_end_heartbeat",
	],
	"guardrails": [
		"unapproved_disconnect_denied",
		"cross_tenant_meter_data_denied",
		"privileged_met_agent_requires_human_approval",
		"tamper_evidence_required",
		"disconnect_approval_required",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "meter_type_supported", "condition": {"operation": "register_meter", "meter_type_supported": False}, "effect": {"decision": "deny", "reason": "meter_type_not_supported", "required_action": "select_supported_meter_type"}},
	{"name": "meter_serial_required", "condition": {"operation": "register_meter", "serial_present": False}, "effect": {"decision": "deny", "reason": "meter_serial_required", "required_action": "provide_meter_serial_number"}},
	{"name": "meter_comm_tech_supported", "condition": {"operation": "register_meter", "comm_tech_supported": False}, "effect": {"decision": "deny", "reason": "comm_technology_not_supported", "required_action": "select_supported_comm_technology"}},
	{"name": "meter_location_required", "condition": {"operation": "register_meter", "location_present": False}, "effect": {"decision": "deny", "reason": "meter_installation_location_required", "required_action": "provide_meter_location"}},
	{"name": "reading_type_supported", "condition": {"operation": "submit_reading", "reading_type_supported": False}, "effect": {"decision": "deny", "reason": "reading_type_not_supported", "required_action": "select_supported_reading_type"}},
	{"name": "reading_interval_supported", "condition": {"operation": "submit_reading", "interval_supported": False}, "effect": {"decision": "deny", "reason": "interval_length_not_supported", "required_action": "select_supported_interval"}},
	{"name": "reading_meter_active", "condition": {"operation": "submit_reading", "meter_active": False}, "effect": {"decision": "deny", "reason": "meter_not_active", "required_action": "activate_meter_first"}},
	{"name": "reading_quality_flag_supported", "condition": {"operation": "submit_reading", "quality_flag_supported": False}, "effect": {"decision": "deny", "reason": "quality_flag_not_supported", "required_action": "select_supported_quality_flag"}},
	{"name": "tamper_type_supported", "condition": {"operation": "report_tamper", "tamper_type_supported": False}, "effect": {"decision": "deny", "reason": "tamper_type_not_supported", "required_action": "select_supported_tamper_type"}},
	{"name": "tamper_evidence_required", "condition": {"operation": "report_tamper", "evidence_present": False}, "effect": {"decision": "deny", "reason": "tamper_evidence_required", "required_action": "attach_tamper_evidence"}},
	{"name": "command_type_supported", "condition": {"operation": "issue_command", "command_type_supported": False}, "effect": {"decision": "deny", "reason": "command_type_not_supported", "required_action": "select_supported_command_type"}},
	{"name": "disconnect_approval_required", "condition": {"operation": "issue_command", "command_is_disconnect": True, "approval_present": False}, "effect": {"decision": "deny", "reason": "disconnect_approval_required", "required_action": "obtain_disconnect_approval"}},
	{"name": "command_meter_active", "condition": {"operation": "issue_command", "meter_active": False}, "effect": {"decision": "deny", "reason": "meter_not_active_for_command", "required_action": "check_meter_status"}},
	{"name": "dr_event_type_supported", "condition": {"operation": "create_dr_event", "dr_event_type_supported": False}, "effect": {"decision": "deny", "reason": "dr_event_type_not_supported", "required_action": "select_supported_dr_event_type"}},
	{"name": "dr_notification_required", "condition": {"operation": "create_dr_event", "notification_sent": False}, "effect": {"decision": "deny", "reason": "dr_customer_notification_required", "required_action": "send_dr_notification_first"}},
	{"name": "dr_opt_out_respected", "condition": {"operation": "activate_dr_event", "customer_opted_out": True}, "effect": {"decision": "deny", "reason": "customer_opted_out_of_dr", "required_action": "exclude_opted_out_customer"}},
	{"name": "cross_tenant_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "data_quality_flag_supported", "condition": {"operation": "set_quality_flag", "quality_flag_supported": False}, "effect": {"decision": "deny", "reason": "quality_flag_not_supported", "required_action": "select_supported_quality_flag"}},
	{"name": "met_agent_runtime_supported", "condition": {"operation": "register_met_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "met_agent_role_supported", "condition": {"operation": "register_met_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_met_agent_requires_human_approval", "condition": {"operation": "met_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required_for_disconnect_command", "required_action": "record_human_approval"}},
	{"name": "firmware_update_approval_required", "condition": {"operation": "issue_command", "command_is_firmware": True, "approval_present": False}, "effect": {"decision": "deny", "reason": "firmware_update_requires_approval", "required_action": "obtain_firmware_update_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": ["tenant_id", "ui", "theme"],
			"properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/energy-met/api/v1",
			"requires_theme": True,
			"template_roots": ["templates/", "static/"],
			"routes": deepcopy(UI_ROUTES),
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions: list[dict[str, Any]] = []
	for rule in RULES:
		if _matches(rule["condition"], context):
			actions.append(rule["effect"] | {"rule": rule["name"]})
	if not actions:
		return {"decision": "allow", "actions": [], "context": dict(context)}
	return {"decision": "deny", "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
