"""Executable APG capability contract for HCM Time and Attendance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "tat_time_attendance"
CAPABILITY_NAME = "Time and Attendance"
CAPABILITY_VERSION = "2.1.0"
ATTENDANCE_EVENT_STREAM = "apg.hcm.tat.time_attendance.lifecycle"

SUPPORTED_ENTRY_METHODS = ["web", "mobile", "kiosk", "biometric", "api", "import"]
SUPPORTED_ENTRY_TYPES = ["regular", "overtime", "leave", "holiday", "training", "on_call"]
SUPPORTED_SCHEDULE_TYPES = ["fixed", "flexible", "rotating", "compressed", "remote"]
SUPPORTED_LEAVE_TYPES = ["vacation", "sick", "parental", "unpaid", "bereavement", "public_holiday"]
SUPPORTED_EXCEPTION_TYPES = [
	"missing_clock_out",
	"late_arrival",
	"early_departure",
	"overtime",
	"geofence",
	"biometric",
	"duplicate_entry",
]
SUPPORTED_ATTENDANCE_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ATTENDANCE_AGENT_ROLES = [
	"attendance_reviewer",
	"compliance_reviewer",
	"schedule_reviewer",
	"fraud_reviewer",
	"payroll_export_reviewer",
	"employee_query_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"policies": {
		"name_required": True,
		"timezone_required": True,
		"workweek_required": True,
		"overtime_threshold_required": True,
		"default_overtime_threshold_hours": 40.0,
	},
	"schedules": {
		"employee_required": True,
		"policy_required": True,
		"supported_types": SUPPORTED_SCHEDULE_TYPES,
		"date_range_required": True,
	},
	"shifts": {
		"schedule_required": True,
		"shift_date_required": True,
		"start_time_required": True,
		"end_time_required": True,
	},
	"time_entries": {
		"employee_required": True,
		"shift_required": True,
		"supported_methods": SUPPORTED_ENTRY_METHODS,
		"supported_types": SUPPORTED_ENTRY_TYPES,
		"device_required_for_methods": ["mobile", "kiosk", "biometric"],
		"geofence_review_required": True,
		"biometric_review_threshold": 0.85,
	},
	"breaks": {
		"time_entry_required": True,
		"start_time_required": True,
		"end_time_required": True,
	},
	"timesheets": {
		"employee_required": True,
		"period_required": True,
		"entries_required": True,
		"submitter_required": True,
		"approval_required_before_export": True,
	},
	"leave_requests": {
		"employee_required": True,
		"supported_types": SUPPORTED_LEAVE_TYPES,
		"date_range_required": True,
		"reason_required": True,
		"extended_leave_days": 10,
		"review_required_for_unpaid_or_extended": True,
	},
	"exceptions": {
		"employee_required": True,
		"supported_types": SUPPORTED_EXCEPTION_TYPES,
		"owner_required_for_high_severity": True,
	},
	"payroll_exports": {
		"period_required": True,
		"approved_timesheets_required": True,
		"approval_required": True,
		"event_stream_required": ATTENDANCE_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"attendance_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_ATTENDANCE_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_ATTENDANCE_AGENT_ROLES,
		"max_autonomous_scope": "inspect_prepare_and_recommend",
		"human_approval_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_state_changes": True,
		"segregation_of_duties": True,
		"approval_before_payroll_export": True,
	},
	"observability": {
		"event_stream": ATTENDANCE_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_policy_events": True,
		"emit_schedule_events": True,
		"emit_time_entry_events": True,
		"emit_timesheet_events": True,
		"emit_exception_events": True,
		"emit_export_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"workflow": "adapter",
		"employee_data": "adapter",
		"payroll": "adapter",
		"device_registry": "adapter",
		"location_policy": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_policies": True,
		"enable_schedules": True,
		"enable_shifts": True,
		"enable_time_entries": True,
		"enable_timesheets": True,
		"enable_leave": True,
		"enable_exceptions": True,
		"enable_exports": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "time_attendance_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"time_policy_lifecycle",
	"work_schedule_lifecycle",
	"shift_lifecycle",
	"time_entry_lifecycle",
	"break_lifecycle",
	"timesheet_lifecycle",
	"leave_request_lifecycle",
	"attendance_approval_workflow",
	"attendance_exception_workflow",
	"attendance_payroll_export",
	"attendance_dashboard_service",
	"attendance_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"composition_events",
	"composition_config",
	"workflow",
	"employee_profile_lifecycle",
	"payroll_period_lifecycle",
	"device_registry",
	"location_policy",
	"privacy_policy",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/hcm/time-attendance/dashboard", "component": "AttendanceDashboard", "permission": "tat_time_attendance:view", "nav_group": "Overview"},
	{"name": "policies", "path": "/hcm/time-attendance/policies", "component": "AttendancePolicyWorkbench", "permission": "tat_time_attendance:manage_policies", "nav_group": "Setup"},
	{"name": "schedules", "path": "/hcm/time-attendance/schedules", "component": "ScheduleWorkbench", "permission": "tat_time_attendance:manage_schedules", "nav_group": "Planning"},
	{"name": "shifts", "path": "/hcm/time-attendance/shifts", "component": "ShiftBoard", "permission": "tat_time_attendance:manage_schedules", "nav_group": "Planning"},
	{"name": "time_entries", "path": "/hcm/time-attendance/time-entries", "component": "TimeEntryWorkbench", "permission": "tat_time_attendance:record_time", "nav_group": "Operations"},
	{"name": "timesheets", "path": "/hcm/time-attendance/timesheets", "component": "TimesheetApprovalDesk", "permission": "tat_time_attendance:approve", "nav_group": "Operations"},
	{"name": "leave", "path": "/hcm/time-attendance/leave", "component": "LeaveRequestDesk", "permission": "tat_time_attendance:manage_leave", "nav_group": "Operations"},
	{"name": "exceptions", "path": "/hcm/time-attendance/exceptions", "component": "AttendanceExceptionCenter", "permission": "tat_time_attendance:govern", "nav_group": "Governance"},
	{"name": "exports", "path": "/hcm/time-attendance/payroll-exports", "component": "AttendancePayrollExportDesk", "permission": "tat_time_attendance:export", "nav_group": "Integration"},
	{"name": "agents", "path": "/hcm/time-attendance/agents", "component": "AttendanceAgentWorkbench", "permission": "tat_time_attendance:agent_manage", "nav_group": "Automation"},
	{"name": "rules", "path": "/hcm/time-attendance/rules", "component": "AttendanceRules", "permission": "tat_time_attendance:govern", "nav_group": "Governance"},
	{"name": "settings", "path": "/hcm/time-attendance/settings", "component": "AttendanceSettings", "permission": "tat_time_attendance:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "time_attendance_control",
	"tokens": {
		"border.radius": "8px",
		"color.primary": "#255E56",
		"color.accent": "#9A6A1B",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"density": "compact",
	},
	"components": {
		"dashboard": {"icon": "layout-dashboard", "status_indicator": "health-pill", "visual": "coverage-grid"},
		"policies": {"icon": "shield-check", "status_style": "policy-band", "visual": "policy-list"},
		"schedules": {"icon": "calendar-days", "status_style": "coverage-chip", "visual": "schedule-board"},
		"shifts": {"icon": "clock-3", "status_style": "shift-chip", "visual": "shift-board"},
		"time_entries": {"icon": "timer", "status_style": "entry-chip", "visual": "entry-table"},
		"timesheets": {"icon": "clipboard-check", "status_style": "approval-chip", "visual": "approval-queue"},
		"leave": {"icon": "calendar-minus", "status_style": "request-chip", "visual": "request-queue"},
		"exceptions": {"icon": "triangle-alert", "status_style": "risk-chip", "visual": "exception-list"},
		"exports": {"icon": "send", "status_style": "export-chip", "visual": "export-ledger"},
		"agents": {"icon": "bot", "status_style": "agent-chip", "visual": "agent-roster"},
		"rules": {"icon": "list-checks", "status_style": "decision-chip", "visual": "rule-list"},
		"settings": {"icon": "settings", "density": "compact", "visual": "settings-panel"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"event_stream": ATTENDANCE_EVENT_STREAM,
	"events": [
		"attendance_policy_created",
		"attendance_schedule_created",
		"attendance_shift_created",
		"time_entry_recorded",
		"break_recorded",
		"timesheet_submitted",
		"timesheet_approved",
		"leave_requested",
		"attendance_exception_recorded",
		"attendance_payroll_export_created",
		"attendance_agent_registered",
	],
	"delivery": "at_least_once",
	"ordering_key": "tenant_id",
}


RULES = [
	{"name": "tenant_context_required", "description": "Attendance operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "operation_policy_required", "description": "Attendance write operations require policy enforcement.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "policy_name_required", "description": "Attendance policies require a name.", "condition": {"operation": "create_time_policy", "name_present": False}, "effect": {"decision": "deny", "reason": "time_policy_name_required", "required_action": "provide_policy_name"}},
	{"name": "policy_timezone_required", "description": "Attendance policies require a timezone.", "condition": {"operation": "create_time_policy", "timezone_present": False}, "effect": {"decision": "deny", "reason": "time_policy_timezone_required", "required_action": "provide_timezone"}},
	{"name": "policy_workweek_required", "description": "Attendance policies require a workweek.", "condition": {"operation": "create_time_policy", "workweek_present": False}, "effect": {"decision": "deny", "reason": "time_policy_workweek_required", "required_action": "provide_workweek"}},
	{"name": "policy_overtime_threshold_required", "description": "Attendance policies require an overtime threshold.", "condition": {"operation": "create_time_policy", "overtime_threshold_present": False}, "effect": {"decision": "deny", "reason": "overtime_threshold_required", "required_action": "provide_overtime_threshold"}},
	{"name": "policy_overtime_threshold_positive", "description": "Attendance overtime thresholds must be positive.", "condition": {"operation": "create_time_policy", "overtime_threshold_positive": False}, "effect": {"decision": "deny", "reason": "overtime_threshold_must_be_positive", "required_action": "set_positive_overtime_threshold"}},
	{"name": "schedule_employee_required", "description": "Schedules require an employee.", "condition": {"operation": "create_schedule", "employee_present": False}, "effect": {"decision": "deny", "reason": "schedule_employee_required", "required_action": "select_employee"}},
	{"name": "schedule_policy_required", "description": "Schedules require an active policy.", "condition": {"operation": "create_schedule", "policy_present": False}, "effect": {"decision": "deny", "reason": "schedule_policy_required", "required_action": "select_policy"}},
	{"name": "schedule_type_supported", "description": "Schedules must use a supported schedule type.", "condition": {"operation": "create_schedule", "schedule_type_supported": False}, "effect": {"decision": "deny", "reason": "schedule_type_not_supported", "required_action": "choose_supported_schedule_type"}},
	{"name": "schedule_start_required", "description": "Schedules require a start date.", "condition": {"operation": "create_schedule", "start_date_present": False}, "effect": {"decision": "deny", "reason": "schedule_start_date_required", "required_action": "provide_start_date"}},
	{"name": "schedule_end_required", "description": "Schedules require an end date.", "condition": {"operation": "create_schedule", "end_date_present": False}, "effect": {"decision": "deny", "reason": "schedule_end_date_required", "required_action": "provide_end_date"}},
	{"name": "shift_schedule_required", "description": "Shifts require a schedule.", "condition": {"operation": "create_shift", "schedule_present": False}, "effect": {"decision": "deny", "reason": "shift_schedule_required", "required_action": "select_schedule"}},
	{"name": "shift_date_required", "description": "Shifts require a shift date.", "condition": {"operation": "create_shift", "shift_date_present": False}, "effect": {"decision": "deny", "reason": "shift_date_required", "required_action": "provide_shift_date"}},
	{"name": "shift_start_required", "description": "Shifts require a start time.", "condition": {"operation": "create_shift", "start_time_present": False}, "effect": {"decision": "deny", "reason": "shift_start_time_required", "required_action": "provide_start_time"}},
	{"name": "shift_end_required", "description": "Shifts require an end time.", "condition": {"operation": "create_shift", "end_time_present": False}, "effect": {"decision": "deny", "reason": "shift_end_time_required", "required_action": "provide_end_time"}},
	{"name": "time_entry_employee_required", "description": "Time entries require an employee.", "condition": {"operation": "record_time_entry", "employee_present": False}, "effect": {"decision": "deny", "reason": "time_entry_employee_required", "required_action": "select_employee"}},
	{"name": "time_entry_shift_required", "description": "Time entries require a shift.", "condition": {"operation": "record_time_entry", "shift_present": False}, "effect": {"decision": "deny", "reason": "time_entry_shift_required", "required_action": "select_shift"}},
	{"name": "time_entry_type_supported", "description": "Time entries must use a supported entry type.", "condition": {"operation": "record_time_entry", "entry_type_supported": False}, "effect": {"decision": "deny", "reason": "time_entry_type_not_supported", "required_action": "choose_supported_entry_type"}},
	{"name": "time_entry_method_supported", "description": "Time entries must use a supported entry method.", "condition": {"operation": "record_time_entry", "entry_method_supported": False}, "effect": {"decision": "deny", "reason": "time_entry_method_not_supported", "required_action": "choose_supported_entry_method"}},
	{"name": "time_entry_clock_in_required", "description": "Time entries require a clock-in time.", "condition": {"operation": "record_time_entry", "clock_in_present": False}, "effect": {"decision": "deny", "reason": "clock_in_required", "required_action": "provide_clock_in"}},
	{"name": "device_required_for_tracked_method", "description": "Mobile, kiosk, and biometric entries require a registered device.", "condition": {"operation": "record_time_entry", "tracked_method": True, "device_present": False}, "effect": {"decision": "deny", "reason": "attendance_device_required", "required_action": "attach_registered_device"}},
	{"name": "geofence_requires_review", "description": "Entries outside a geofence require review.", "condition": {"operation": "record_time_entry", "geofence_verified": False, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "geofence_review_required", "required_action": "record_geofence_review"}},
	{"name": "biometric_confidence_requires_review", "description": "Low biometric confidence requires review.", "condition": {"operation": "record_time_entry", "biometric_low_confidence": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "biometric_review_required", "required_action": "record_biometric_review"}},
	{"name": "break_time_entry_required", "description": "Breaks require a time entry.", "condition": {"operation": "record_break", "time_entry_present": False}, "effect": {"decision": "deny", "reason": "break_time_entry_required", "required_action": "select_time_entry"}},
	{"name": "break_start_required", "description": "Breaks require a start time.", "condition": {"operation": "record_break", "start_time_present": False}, "effect": {"decision": "deny", "reason": "break_start_time_required", "required_action": "provide_break_start"}},
	{"name": "break_end_required", "description": "Breaks require an end time.", "condition": {"operation": "record_break", "end_time_present": False}, "effect": {"decision": "deny", "reason": "break_end_time_required", "required_action": "provide_break_end"}},
	{"name": "timesheet_employee_required", "description": "Timesheets require an employee.", "condition": {"operation": "submit_timesheet", "employee_present": False}, "effect": {"decision": "deny", "reason": "timesheet_employee_required", "required_action": "select_employee"}},
	{"name": "timesheet_period_required", "description": "Timesheets require a period.", "condition": {"operation": "submit_timesheet", "period_present": False}, "effect": {"decision": "deny", "reason": "timesheet_period_required", "required_action": "select_period"}},
	{"name": "timesheet_entries_required", "description": "Timesheets require at least one time entry.", "condition": {"operation": "submit_timesheet", "entries_present": False}, "effect": {"decision": "deny", "reason": "timesheet_entries_required", "required_action": "attach_entries"}},
	{"name": "timesheet_submitter_required", "description": "Timesheets require a submitter.", "condition": {"operation": "submit_timesheet", "submitter_present": False}, "effect": {"decision": "deny", "reason": "timesheet_submitter_required", "required_action": "record_submitter"}},
	{"name": "timesheet_hours_nonnegative", "description": "Timesheet hours cannot be negative.", "condition": {"operation": "submit_timesheet", "total_hours_negative": True}, "effect": {"decision": "deny", "reason": "timesheet_hours_must_be_nonnegative", "required_action": "correct_entries"}},
	{"name": "timesheet_approver_required", "description": "Timesheet approval requires an approver.", "condition": {"operation": "approve_timesheet", "approver_present": False}, "effect": {"decision": "deny", "reason": "timesheet_approver_required", "required_action": "record_approver"}},
	{"name": "leave_employee_required", "description": "Leave requests require an employee.", "condition": {"operation": "request_leave", "employee_present": False}, "effect": {"decision": "deny", "reason": "leave_employee_required", "required_action": "select_employee"}},
	{"name": "leave_type_supported", "description": "Leave requests must use a supported leave type.", "condition": {"operation": "request_leave", "leave_type_supported": False}, "effect": {"decision": "deny", "reason": "leave_type_not_supported", "required_action": "choose_supported_leave_type"}},
	{"name": "leave_start_required", "description": "Leave requests require a start date.", "condition": {"operation": "request_leave", "start_date_present": False}, "effect": {"decision": "deny", "reason": "leave_start_date_required", "required_action": "provide_start_date"}},
	{"name": "leave_end_required", "description": "Leave requests require an end date.", "condition": {"operation": "request_leave", "end_date_present": False}, "effect": {"decision": "deny", "reason": "leave_end_date_required", "required_action": "provide_end_date"}},
	{"name": "leave_reason_required", "description": "Leave requests require a reason.", "condition": {"operation": "request_leave", "reason_present": False}, "effect": {"decision": "deny", "reason": "leave_reason_required", "required_action": "provide_reason"}},
	{"name": "leave_review_required", "description": "Unpaid or extended leave requires review.", "condition": {"operation": "request_leave", "review_required": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "leave_review_required", "required_action": "record_leave_approval"}},
	{"name": "exception_employee_required", "description": "Attendance exceptions require an employee.", "condition": {"operation": "record_exception", "employee_present": False}, "effect": {"decision": "deny", "reason": "attendance_exception_employee_required", "required_action": "select_employee"}},
	{"name": "exception_type_supported", "description": "Attendance exceptions must use a supported exception type.", "condition": {"operation": "record_exception", "exception_type_supported": False}, "effect": {"decision": "deny", "reason": "attendance_exception_type_not_supported", "required_action": "choose_supported_exception_type"}},
	{"name": "exception_owner_required", "description": "High severity exceptions require an owner.", "condition": {"operation": "record_exception", "high_severity": True, "owner_present": False}, "effect": {"decision": "deny", "reason": "attendance_exception_owner_required", "required_action": "assign_exception_owner"}},
	{"name": "export_period_required", "description": "Payroll exports require a period.", "condition": {"operation": "create_payroll_export", "period_present": False}, "effect": {"decision": "deny", "reason": "attendance_export_period_required", "required_action": "select_period"}},
	{"name": "export_timesheets_required", "description": "Payroll exports require timesheets.", "condition": {"operation": "create_payroll_export", "timesheets_present": False}, "effect": {"decision": "deny", "reason": "attendance_export_timesheets_required", "required_action": "attach_timesheets"}},
	{"name": "export_timesheets_approved", "description": "Payroll exports can only include approved timesheets.", "condition": {"operation": "create_payroll_export", "all_timesheets_approved": False}, "effect": {"decision": "deny", "reason": "timesheets_must_be_approved", "required_action": "approve_timesheets"}},
	{"name": "export_approver_required", "description": "Payroll exports require approval.", "condition": {"operation": "create_payroll_export", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "attendance_export_approval_required", "required_action": "record_export_approval"}},
	{"name": "bytewax_event_stream_required", "description": "Attendance batches and exports must use Bytewax event stream metadata.", "condition": {"operation": "attendance_batch", "event_stream": "queue"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_to_bytewax_stream"}},
	{"name": "agent_runtime_supported", "description": "Attendance agents must use a supported runtime.", "condition": {"operation": "register_attendance_agent", "runtime_supported": False}, "effect": {"decision": "deny", "reason": "attendance_agent_runtime_not_supported", "required_action": "choose_supported_runtime"}},
	{"name": "agent_role_supported", "description": "Attendance agents must use a supported role.", "condition": {"operation": "register_attendance_agent", "role_supported": False}, "effect": {"decision": "deny", "reason": "attendance_agent_role_not_supported", "required_action": "choose_supported_role"}},
	{"name": "agent_scope_limited", "description": "Attendance agents cannot autonomously post privileged state changes.", "condition": {"operation": "agent_action", "privileged_action": True, "human_approved": False}, "effect": {"decision": "require_review", "reason": "attendance_agent_human_approval_required", "required_action": "record_human_approval"}},
	{"name": "audit_required_for_state_change", "description": "Attendance state changes must be auditable.", "condition": {"operation_type": "write", "audit_enabled": False}, "effect": {"decision": "deny", "reason": "attendance_audit_required", "required_action": "enable_audit"}},
]


CONFIGURATION_SCHEMA = {
	"type": "object",
	"required": ["tenant_id", "ui", "theme"],
	"properties": {
		"tenant_id": {"type": "string"},
		"policies": {"type": "object"},
		"schedules": {"type": "object"},
		"shifts": {"type": "object"},
		"time_entries": {"type": "object"},
		"breaks": {"type": "object"},
		"timesheets": {"type": "object"},
		"leave_requests": {"type": "object"},
		"exceptions": {"type": "object"},
		"payroll_exports": {"type": "object"},
		"attendance_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}


def _merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
	merged = deepcopy(base)
	for key, value in overrides.items():
		if isinstance(value, dict) and isinstance(merged.get(key), dict):
			merged[key] = _merge_dict(merged[key], value)
		else:
			merged[key] = deepcopy(value)
	return merged


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the executable Time and Attendance capability contract."""
	configuration = _merge_dict(DEFAULT_CONFIGURATION, overrides or {})
	configuration["tenant_id"] = tenant_id or configuration.get("tenant_id", "default")
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": deepcopy(PROVIDES),
		"requires": deepcopy(REQUIRES),
		"configuration": configuration,
		"configuration_schema": deepcopy(CONFIGURATION_SCHEMA),
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/hcm/time-attendance/api/v1",
			"requires_theme": True,
			"template_roots": ["templates/", "static/"],
			"view_module": "views.py",
			"routes": deepcopy(UI_ROUTES),
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


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


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic Time and Attendance guardrails."""
	matched_rules: list[str] = []
	effects: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched_rules.append(rule["name"])
			effect = deepcopy(rule["effect"])
			effects.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {
		"decision": decision,
		"matched_rules": matched_rules,
		"effects": effects,
		"context": deepcopy(context),
	}
