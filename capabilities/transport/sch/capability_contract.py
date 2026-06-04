"""Executable capability contract for APG Transport Scheduling."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "transport_sch"
CAPABILITY_NAME = "Transport Scheduling"
CAPABILITY_VERSION = "1.0.0"
SCHEDULING_EVENT_STREAM = "apg.transport.scheduling.lifecycle"

SUPPORTED_SCHEDULE_TYPES = ["load_schedule", "driver_shift", "vehicle_assignment", "charter", "recurring_run", "express_schedule", "ad_hoc", "contract_schedule"]
SUPPORTED_SCHEDULE_STATUSES = ["draft", "published", "in_progress", "completed", "cancelled", "rescheduled", "on_hold"]
SUPPORTED_SHIFT_TYPES = ["day_shift", "night_shift", "split_shift", "rest_day", "on_call", "overtime", "bank_holiday"]
SUPPORTED_CHARTER_TYPES = ["school_charter", "corporate_charter", "event_charter", "tourist_charter", "airport_transfer", "medical_transport", "funeral_transport"]
SUPPORTED_OPTIMISATION_MODES = ["cost", "driver_satisfaction", "vehicle_utilisation", "customer_sla", "co2_minimise", "balanced"]
SUPPORTED_CONFLICT_TYPES = ["double_booking", "driver_hours_breach", "vehicle_unavailable", "route_conflict", "resource_shortage", "skill_mismatch"]
SUPPORTED_NOTIFICATION_TYPES = ["schedule_published", "schedule_changed", "shift_reminder", "overtime_alert", "conflict_alert", "charter_confirmation"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["schedule_planner", "conflict_resolver", "charter_manager", "shift_optimiser", "resource_allocator"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"schedules": {"supported_types": SUPPORTED_SCHEDULE_TYPES, "supported_statuses": SUPPORTED_SCHEDULE_STATUSES, "advance_planning_days": 14, "auto_publish_enabled": False},
	"shifts": {"types": SUPPORTED_SHIFT_TYPES, "max_daily_hours": 10, "max_weekly_hours": 56, "break_rules_enforced": True, "tacho_compliance_enabled": True},
	"charters": {"types": SUPPORTED_CHARTER_TYPES, "customer_confirmation_required": True, "vehicle_inspection_required": True, "driver_briefing_required": True},
	"optimisation": {"modes": SUPPORTED_OPTIMISATION_MODES, "default_mode": "balanced", "auto_optimise_on_publish": True, "constraint_relaxation_allowed": False},
	"conflicts": {"types": SUPPORTED_CONFLICT_TYPES, "auto_detection_enabled": True, "block_publish_on_conflict": True, "alert_on_detection": True},
	"notifications": {"types": SUPPORTED_NOTIFICATION_TYPES, "advance_notice_hours": 24, "change_notification_enabled": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_schedule_denied": True, "driver_hours_breach_denied": True, "double_booking_denied": True},
	"observability": {"event_stream": SCHEDULING_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_schedules": True, "enable_shifts": True, "enable_charters": True, "enable_optimisation": True, "enable_conflicts": True},
	"theme": {"default_theme": "transport_scheduling_control", "allow_tenant_overrides": True},
}

PROVIDES = ["load_scheduling_workflow", "driver_shift_planning_workflow", "vehicle_assignment_workflow", "charter_management_workflow", "schedule_optimisation_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "schd", "mqeb", "comp"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/transport-scheduling/dashboard", "component": "SchedulingDashboard", "permission": "transport_sch:view", "nav_group": "Overview"},
	{"name": "schedules", "path": "/transport-scheduling/schedules", "component": "ScheduleConsole", "permission": "transport_sch:schedules", "nav_group": "Schedules"},
	{"name": "schedule_create", "path": "/transport-scheduling/schedules/create", "component": "ScheduleForm", "permission": "transport_sch:schedules_write", "nav_group": "Schedules"},
	{"name": "calendar", "path": "/transport-scheduling/calendar", "component": "SchedulingCalendar", "permission": "transport_sch:view", "nav_group": "Overview"},
	{"name": "shifts", "path": "/transport-scheduling/shifts", "component": "ShiftConsole", "permission": "transport_sch:shifts", "nav_group": "Drivers"},
	{"name": "vehicle_assignment", "path": "/transport-scheduling/vehicles", "component": "VehicleAssignmentConsole", "permission": "transport_sch:vehicles", "nav_group": "Vehicles"},
	{"name": "charters", "path": "/transport-scheduling/charters", "component": "CharterConsole", "permission": "transport_sch:charters", "nav_group": "Charters"},
	{"name": "optimisation", "path": "/transport-scheduling/optimisation", "component": "ScheduleOptimisationConsole", "permission": "transport_sch:optimisation", "nav_group": "Optimisation"},
	{"name": "conflicts", "path": "/transport-scheduling/conflicts", "component": "ConflictConsole", "permission": "transport_sch:conflicts", "nav_group": "Exceptions"},
	{"name": "reports", "path": "/transport-scheduling/reports", "component": "SchedulingReportConsole", "permission": "transport_sch:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/transport-scheduling/agents", "component": "SchedulingAgentWorkbench", "permission": "transport_sch:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/transport-scheduling/settings", "component": "SchedulingSettings", "permission": "transport_sch:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "transport_scheduling_control",
	"tokens": {"color.primary": "#0369A1", "color.accent": "#7C3AED", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#991B1B", "surface.canvas": "#F0F9FF", "surface.panel": "#FFFFFF", "text.primary": "#0F172A", "text.secondary": "#475569", "border.radius": "8px", "density": "comfortable"},
	"components": {
		"schedules": {"icon": "calendar", "status_indicator": "schedule-status-chip"},
		"shifts": {"icon": "clock", "status_indicator": "shift-type-chip"},
		"charters": {"icon": "star", "status_indicator": "charter-type-chip"},
		"vehicle_assignment": {"icon": "truck", "status_indicator": "assignment-chip"},
		"conflicts": {"icon": "alert-triangle", "status_indicator": "conflict-type-chip"},
		"optimisation": {"icon": "sliders", "status_indicator": "optimisation-mode-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": SCHEDULING_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["schedule_created", "schedule_published", "shift_assigned", "vehicle_assigned", "charter_confirmed", "conflict_detected", "conflict_resolved", "schedule_optimised", "scheduling_agent_registered"],
	"guardrails": ["scheduling_batch_requires_bytewax", "driver_hours_breach_denied", "double_booking_denied", "cross_tenant_schedule_denied", "privileged_scheduling_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "scheduling_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "scheduling_policy_required", "required_action": "attach_scheduling_policy"}},
	{"name": "schedule_type_supported", "condition": {"operation": "create_schedule", "schedule_type_supported": False}, "effect": {"decision": "deny", "reason": "schedule_type_not_supported", "required_action": "select_supported_schedule_type"}},
	{"name": "schedule_status_supported", "condition": {"operation": "update_schedule_status", "status_supported": False}, "effect": {"decision": "deny", "reason": "schedule_status_not_supported", "required_action": "select_supported_schedule_status"}},
	{"name": "shift_type_supported", "condition": {"operation": "create_shift", "shift_type_supported": False}, "effect": {"decision": "deny", "reason": "shift_type_not_supported", "required_action": "select_supported_shift_type"}},
	{"name": "driver_hours_breach_denied", "condition": {"operation": "create_shift", "driver_hours_compliant": False}, "effect": {"decision": "deny", "reason": "driver_hours_breach_denied", "required_action": "adjust_shift_to_comply_with_hours"}},
	{"name": "double_booking_denied", "condition": {"operation": "assign_resource", "double_booking_detected": True}, "effect": {"decision": "deny", "reason": "double_booking_denied", "required_action": "resolve_booking_conflict"}},
	{"name": "charter_type_supported", "condition": {"operation": "create_charter", "charter_type_supported": False}, "effect": {"decision": "deny", "reason": "charter_type_not_supported", "required_action": "select_supported_charter_type"}},
	{"name": "charter_customer_confirmation_required", "condition": {"operation": "create_charter", "customer_confirmed": False}, "effect": {"decision": "deny", "reason": "customer_confirmation_required", "required_action": "obtain_customer_confirmation"}},
	{"name": "optimisation_mode_supported", "condition": {"operation": "optimise_schedule", "optimisation_mode_supported": False}, "effect": {"decision": "deny", "reason": "optimisation_mode_not_supported", "required_action": "select_supported_optimisation_mode"}},
	{"name": "conflict_type_supported", "condition": {"operation": "record_conflict", "conflict_type_supported": False}, "effect": {"decision": "deny", "reason": "conflict_type_not_supported", "required_action": "select_supported_conflict_type"}},
	{"name": "publish_blocked_on_conflict", "condition": {"operation": "publish_schedule", "unresolved_conflicts_present": True}, "effect": {"decision": "deny", "reason": "unresolved_conflicts_block_publish", "required_action": "resolve_all_conflicts_before_publish"}},
	{"name": "vehicle_assignment_vehicle_required", "condition": {"operation": "assign_vehicle", "vehicle_present": False}, "effect": {"decision": "deny", "reason": "vehicle_required", "required_action": "select_vehicle"}},
	{"name": "vehicle_assignment_schedule_required", "condition": {"operation": "assign_vehicle", "schedule_present": False}, "effect": {"decision": "deny", "reason": "schedule_required", "required_action": "select_schedule"}},
	{"name": "cross_tenant_schedule_denied", "condition": {"operation_type": "write", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_schedule_denied", "required_action": "use_tenant_scoped_context"}},
	{"name": "scheduling_batch_requires_bytewax", "condition": {"operation": "scheduling_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_scheduling_batch_to_bytewax"}},
	{"name": "scheduling_agent_runtime_supported", "condition": {"operation": "register_scheduling_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "scheduling_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "scheduling_agent_role_supported", "condition": {"operation": "register_scheduling_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "scheduling_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_scheduling_agent_action_requires_human_approval", "condition": {"operation": "scheduling_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "charter_vehicle_inspection_required", "condition": {"operation": "dispatch_charter", "vehicle_inspected": False}, "effect": {"decision": "deny", "reason": "vehicle_inspection_required_for_charter", "required_action": "complete_vehicle_inspection"}},
	{"name": "tacho_compliance_required", "condition": {"operation": "create_shift", "tacho_compliant": False}, "effect": {"decision": "deny", "reason": "tachograph_compliance_required", "required_action": "ensure_tacho_compliance"}},
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
		"ui": {"shell": "apg_python", "api_prefix": "/transport-scheduling/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)},
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
