"""Executable capability contract for APG Dispatch Operations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "transport_dis"
CAPABILITY_NAME = "Dispatch Operations"
CAPABILITY_VERSION = "1.0.0"
DISPATCH_EVENT_STREAM = "apg.transport.dispatch.lifecycle"

SUPPORTED_LOAD_TYPES = ["full_truckload", "less_than_truckload", "partial_load", "express_load", "intermodal", "bulk_load", "temperature_controlled", "oversized_load"]
SUPPORTED_DISPATCH_STATUSES = ["planned", "assigned", "dispatched", "in_transit", "at_stop", "completed", "cancelled", "exception"]
SUPPORTED_EXCEPTION_TYPES = ["vehicle_breakdown", "driver_unavailable", "traffic_delay", "customs_hold", "weather_delay", "cargo_damage", "route_deviation", "time_window_missed"]
SUPPORTED_DRIVER_ASSIGNMENT_TYPES = ["primary", "co_driver", "relay", "standby", "temp_assignment"]
SUPPORTED_OPTIMISATION_MODES = ["cost", "time", "distance", "co2", "balanced", "priority_first"]
SUPPORTED_TRACKING_UPDATE_TYPES = ["departure", "arrival", "waypoint", "stop_completed", "exception", "eta_update", "checkpoint"]
SUPPORTED_COMMUNICATION_CHANNELS = ["driver_app", "radio", "sms", "in_cab_terminal", "telematics_platform", "phone"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["load_planner", "driver_assigner", "dispatch_optimiser", "exception_handler", "tracking_monitor"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"loads": {"supported_types": SUPPORTED_LOAD_TYPES, "max_load_weight_kg": 44000, "vehicle_capacity_check_required": True, "driver_hours_check_required": True},
	"dispatch": {"supported_statuses": SUPPORTED_DISPATCH_STATUSES, "auto_dispatch_enabled": True, "optimisation_modes": SUPPORTED_OPTIMISATION_MODES, "real_time_eta_enabled": True},
	"driver_assignment": {"assignment_types": SUPPORTED_DRIVER_ASSIGNMENT_TYPES, "hours_of_service_check": True, "licence_check_required": True, "qualification_check_required": True},
	"exceptions": {"supported_types": SUPPORTED_EXCEPTION_TYPES, "auto_escalate_enabled": True, "notification_required": True, "resolution_required": True},
	"tracking": {"update_types": SUPPORTED_TRACKING_UPDATE_TYPES, "gps_interval_seconds": 30, "geofence_alerts_enabled": True, "eta_recalculation_on_deviation": True},
	"communication": {"channels": SUPPORTED_COMMUNICATION_CHANNELS, "broadcast_enabled": True, "priority_messaging_enabled": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_dispatch_denied": True, "overload_dispatch_denied": True},
	"observability": {"event_stream": DISPATCH_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_loads": True, "enable_dispatch": True, "enable_driver_assignment": True, "enable_exceptions": True, "enable_tracking": True},
	"theme": {"default_theme": "transport_dispatch_control", "allow_tenant_overrides": True},
}

PROVIDES = ["load_planning_workflow", "driver_assignment_workflow", "dispatch_optimisation_workflow", "real_time_tracking_workflow", "exception_management_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "schd", "mqeb", "nlpc"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/transport-dispatch/dashboard", "component": "DispatchDashboard", "permission": "transport_dis:view", "nav_group": "Overview"},
	{"name": "loads", "path": "/transport-dispatch/loads", "component": "LoadPlanningConsole", "permission": "transport_dis:loads", "nav_group": "Planning"},
	{"name": "load_create", "path": "/transport-dispatch/loads/create", "component": "LoadPlanningForm", "permission": "transport_dis:loads_write", "nav_group": "Planning"},
	{"name": "dispatch_board", "path": "/transport-dispatch/board", "component": "DispatchBoard", "permission": "transport_dis:dispatch", "nav_group": "Operations"},
	{"name": "driver_assignment", "path": "/transport-dispatch/drivers", "component": "DriverAssignmentConsole", "permission": "transport_dis:drivers", "nav_group": "Operations"},
	{"name": "tracking", "path": "/transport-dispatch/tracking", "component": "DispatchTrackingMap", "permission": "transport_dis:tracking", "nav_group": "Operations"},
	{"name": "exceptions", "path": "/transport-dispatch/exceptions", "component": "ExceptionConsole", "permission": "transport_dis:exceptions", "nav_group": "Exceptions"},
	{"name": "optimisation", "path": "/transport-dispatch/optimisation", "component": "DispatchOptimisationConsole", "permission": "transport_dis:optimisation", "nav_group": "Planning"},
	{"name": "communication", "path": "/transport-dispatch/communication", "component": "DispatchCommunicationConsole", "permission": "transport_dis:communication", "nav_group": "Operations"},
	{"name": "reports", "path": "/transport-dispatch/reports", "component": "DispatchReportConsole", "permission": "transport_dis:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/transport-dispatch/agents", "component": "DispatchAgentWorkbench", "permission": "transport_dis:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/transport-dispatch/settings", "component": "DispatchSettings", "permission": "transport_dis:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "transport_dispatch_control",
	"tokens": {"color.primary": "#1D4ED8", "color.accent": "#7C3AED", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#991B1B", "surface.canvas": "#EFF6FF", "surface.panel": "#FFFFFF", "text.primary": "#0F172A", "text.secondary": "#475569", "border.radius": "6px", "density": "compact"},
	"components": {
		"loads": {"icon": "layers", "status_indicator": "load-type-chip"},
		"dispatch": {"icon": "send", "status_indicator": "dispatch-status-chip"},
		"driver_assignment": {"icon": "user-check", "status_indicator": "assignment-type-chip"},
		"tracking": {"icon": "map", "status_indicator": "tracking-update-chip"},
		"exceptions": {"icon": "alert-octagon", "status_indicator": "exception-type-chip"},
		"optimisation": {"icon": "zap", "status_indicator": "optimisation-mode-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": DISPATCH_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["load_planned", "driver_assigned", "dispatch_created", "dispatch_started", "tracking_updated", "exception_raised", "exception_resolved", "dispatch_completed", "dispatch_agent_registered"],
	"guardrails": ["dispatch_batch_requires_bytewax", "overload_dispatch_denied", "cross_tenant_dispatch_denied", "privileged_dispatch_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "dispatch_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "dispatch_policy_required", "required_action": "attach_dispatch_policy"}},
	{"name": "load_type_supported", "condition": {"operation": "plan_load", "load_type_supported": False}, "effect": {"decision": "deny", "reason": "load_type_not_supported", "required_action": "select_supported_load_type"}},
	{"name": "load_vehicle_capacity_required", "condition": {"operation": "plan_load", "vehicle_assigned": True, "capacity_check_passed": False}, "effect": {"decision": "deny", "reason": "vehicle_capacity_exceeded", "required_action": "reduce_load_or_change_vehicle"}},
	{"name": "overload_dispatch_denied", "condition": {"operation": "plan_load", "load_exceeds_legal_limit": True}, "effect": {"decision": "deny", "reason": "overload_dispatch_denied", "required_action": "reduce_load_weight"}},
	{"name": "driver_assignment_type_supported", "condition": {"operation": "assign_driver", "assignment_type_supported": False}, "effect": {"decision": "deny", "reason": "assignment_type_not_supported", "required_action": "select_supported_assignment_type"}},
	{"name": "driver_hours_of_service_check", "condition": {"operation": "assign_driver", "hours_of_service_compliant": False}, "effect": {"decision": "deny", "reason": "driver_hours_exceeded", "required_action": "assign_compliant_driver"}},
	{"name": "driver_licence_required", "condition": {"operation": "assign_driver", "licence_valid": False}, "effect": {"decision": "deny", "reason": "valid_licence_required", "required_action": "verify_driver_licence"}},
	{"name": "dispatch_status_supported", "condition": {"operation": "update_dispatch_status", "status_supported": False}, "effect": {"decision": "deny", "reason": "dispatch_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "dispatch_vehicle_required", "condition": {"operation": "create_dispatch", "vehicle_present": False}, "effect": {"decision": "deny", "reason": "vehicle_required", "required_action": "assign_vehicle"}},
	{"name": "dispatch_driver_required", "condition": {"operation": "create_dispatch", "driver_present": False}, "effect": {"decision": "deny", "reason": "driver_required", "required_action": "assign_driver"}},
	{"name": "dispatch_route_required", "condition": {"operation": "create_dispatch", "route_present": False}, "effect": {"decision": "deny", "reason": "route_required", "required_action": "plan_route"}},
	{"name": "optimisation_mode_supported", "condition": {"operation": "optimise_dispatch", "optimisation_mode_supported": False}, "effect": {"decision": "deny", "reason": "optimisation_mode_not_supported", "required_action": "select_supported_optimisation_mode"}},
	{"name": "tracking_update_type_supported", "condition": {"operation": "update_tracking", "update_type_supported": False}, "effect": {"decision": "deny", "reason": "tracking_update_type_not_supported", "required_action": "select_supported_update_type"}},
	{"name": "exception_type_supported", "condition": {"operation": "raise_exception", "exception_type_supported": False}, "effect": {"decision": "deny", "reason": "exception_type_not_supported", "required_action": "select_supported_exception_type"}},
	{"name": "exception_dispatch_required", "condition": {"operation": "raise_exception", "dispatch_present": False}, "effect": {"decision": "deny", "reason": "dispatch_reference_required", "required_action": "select_dispatch"}},
	{"name": "communication_channel_supported", "condition": {"operation": "send_communication", "channel_supported": False}, "effect": {"decision": "deny", "reason": "channel_not_supported", "required_action": "select_supported_channel"}},
	{"name": "cross_tenant_dispatch_denied", "condition": {"operation_type": "write", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_dispatch_denied", "required_action": "use_tenant_scoped_context"}},
	{"name": "dispatch_batch_requires_bytewax", "condition": {"operation": "dispatch_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_dispatch_batch_to_bytewax"}},
	{"name": "dispatch_agent_runtime_supported", "condition": {"operation": "register_dispatch_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "dispatch_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "dispatch_agent_role_supported", "condition": {"operation": "register_dispatch_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "dispatch_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_dispatch_agent_action_requires_human_approval", "condition": {"operation": "dispatch_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"name": CAPABILITY_NAME,
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
		"ui": {"shell": "apg_python", "api_prefix": "/transport-dispatch/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)},
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
