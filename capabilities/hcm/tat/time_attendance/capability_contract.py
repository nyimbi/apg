"""Executable APG capability contract for HCM Time and Attendance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "tat_time_attendance"
CAPABILITY_NAME = "Time and Attendance Tracking"
CAPABILITY_VERSION = "2.2.0"
ATTENDANCE_EVENT_STREAM = "apg.hcm.tat.time_attendance.lifecycle"

SUPPORTED_ENTRY_METHODS = ["web", "mobile", "kiosk", "biometric", "api", "import", "supervisor_entry"]
SUPPORTED_ENTRY_TYPES = ["regular", "overtime", "double_time", "leave", "holiday", "training", "on_call", "standby", "comp_time"]
SUPPORTED_SCHEDULE_TYPES = ["fixed", "flexible", "rotating", "compressed", "remote", "shift_based", "annualised_hours"]
SUPPORTED_LEAVE_TYPES = ["vacation", "sick", "parental", "unpaid", "bereavement", "public_holiday", "study", "compassionate", "jury_duty"]
SUPPORTED_EXCEPTION_TYPES = [
	"missing_clock_out",
	"late_arrival",
	"early_departure",
	"overtime",
	"double_time",
	"geofence",
	"biometric",
	"duplicate_entry",
	"schedule_deviation",
	"unplanned_absence",
	"comp_time_accrual",
]
SUPPORTED_OVERTIME_CALCULATION_METHODS = ["daily", "weekly", "pay_period", "rolling_7_day"]
SUPPORTED_COMP_TIME_POLICIES = ["accrual_based", "hour_for_hour", "time_and_half", "none"]
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
		"overtime_calculation_method_required": True,
		"supported_overtime_methods": SUPPORTED_OVERTIME_CALCULATION_METHODS,
		"double_time_threshold_hours": 60.0,
		"comp_time_policy_required": True,
		"supported_comp_time_policies": SUPPORTED_COMP_TIME_POLICIES,
		"max_comp_time_accrual_hours": 80.0,
		"minimum_rest_between_shifts_hours": 11.0,
		"max_consecutive_working_days": 6,
	},
	"schedules": {
		"employee_required": True,
		"policy_required": True,
		"supported_types": SUPPORTED_SCHEDULE_TYPES,
		"date_range_required": True,
		"prevent_overlapping_schedules": True,
	},
	"shifts": {
		"schedule_required": True,
		"shift_date_required": True,
		"start_time_required": True,
		"end_time_required": True,
		"minimum_shift_duration_minutes": 30,
		"maximum_shift_duration_hours": 16,
	},
	"time_entries": {
		"employee_required": True,
		"shift_required": True,
		"supported_methods": SUPPORTED_ENTRY_METHODS,
		"supported_types": SUPPORTED_ENTRY_TYPES,
		"device_required_for_methods": ["mobile", "kiosk", "biometric"],
		"geofence_review_required": True,
		"biometric_review_threshold": 0.85,
		"duplicate_entry_window_minutes": 5,
	},
	"breaks": {
		"time_entry_required": True,
		"start_time_required": True,
		"end_time_required": True,
		"minimum_break_minutes_per_shift_hours": {"4": 0, "6": 15, "8": 30, "12": 45},
	},
	"timesheets": {
		"employee_required": True,
		"period_required": True,
		"entries_required": True,
		"submitter_required": True,
		"approval_required_before_export": True,
		"late_submission_allowed_with_reason": True,
		"resubmission_requires_change_reason": True,
	},
	"overtime_rules": {
		"daily_overtime_threshold_hours": 8.0,
		"weekly_overtime_threshold_hours": 40.0,
		"overtime_rate_multiplier": 1.5,
		"double_time_rate_multiplier": 2.0,
		"double_time_daily_threshold_hours": 12.0,
		"double_time_weekly_threshold_hours": 60.0,
		"overtime_requires_preauthorization": True,
		"retroactive_overtime_approval_window_hours": 24,
		"public_holiday_work_multiplier": 2.0,
		"rest_day_work_multiplier": 1.5,
		"comp_time_accrual_rate": 1.5,
	},
	"leave_requests": {
		"employee_required": True,
		"supported_types": SUPPORTED_LEAVE_TYPES,
		"date_range_required": True,
		"reason_required": True,
		"extended_leave_days": 10,
		"review_required_for_unpaid_or_extended": True,
		"sick_leave_medical_cert_threshold_days": 3,
	},
	"exceptions": {
		"employee_required": True,
		"supported_types": SUPPORTED_EXCEPTION_TYPES,
		"owner_required_for_high_severity": True,
		"auto_escalation_hours": 48,
	},
	"payroll_exports": {
		"period_required": True,
		"approved_timesheets_required": True,
		"approval_required": True,
		"event_stream_required": ATTENDANCE_EVENT_STREAM,
		"stream_processor": "bytewax",
		"overtime_hours_exported_separately": True,
		"comp_time_balance_exported": True,
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
		"cross_tenant_access_denied": True,
		"privilege_escalation_denied": True,
		"employee_cannot_approve_own_timesheet": True,
		"overtime_preauthorization_enforced": True,
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
		"emit_overtime_events": True,
		"emit_comp_time_events": True,
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
		"enable_overtime": True,
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
	"overtime_calculation_service",
	"comp_time_accrual_service",
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
	"mten",
	"conf",
	"ntfy",
	"wflo",
	"schd",
	"mqeb",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/hcm/time-attendance/dashboard", "component": "AttendanceDashboard", "permission": "tat_time_attendance:view", "nav_group": "Overview"},
	{"name": "policies", "path": "/hcm/time-attendance/policies", "component": "AttendancePolicyWorkbench", "permission": "tat_time_attendance:manage_policies", "nav_group": "Setup"},
	{"name": "overtime_rules", "path": "/hcm/time-attendance/overtime-rules", "component": "OvertimeRuleWorkbench", "permission": "tat_time_attendance:manage_policies", "nav_group": "Setup"},
	{"name": "schedules", "path": "/hcm/time-attendance/schedules", "component": "ScheduleWorkbench", "permission": "tat_time_attendance:manage_schedules", "nav_group": "Planning"},
	{"name": "shifts", "path": "/hcm/time-attendance/shifts", "component": "ShiftBoard", "permission": "tat_time_attendance:manage_schedules", "nav_group": "Planning"},
	{"name": "time_entries", "path": "/hcm/time-attendance/time-entries", "component": "TimeEntryWorkbench", "permission": "tat_time_attendance:record_time", "nav_group": "Operations"},
	{"name": "timesheets", "path": "/hcm/time-attendance/timesheets", "component": "TimesheetApprovalDesk", "permission": "tat_time_attendance:approve", "nav_group": "Operations"},
	{"name": "overtime", "path": "/hcm/time-attendance/overtime", "component": "OvertimeApprovalDesk", "permission": "tat_time_attendance:approve_overtime", "nav_group": "Operations"},
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
		"color.primary": "#255E56",
		"color.accent": "#9A6A1B",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"dashboard": {"icon": "layout-dashboard", "status_indicator": "health-pill", "visual": "coverage-grid"},
		"policies": {"icon": "shield-check", "status_style": "policy-band", "visual": "policy-list"},
		"overtime_rules": {"icon": "clock-plus", "status_style": "rule-chip", "visual": "overtime-matrix"},
		"schedules": {"icon": "calendar-days", "status_style": "coverage-chip", "visual": "schedule-board"},
		"shifts": {"icon": "clock-3", "status_style": "shift-chip", "visual": "shift-board"},
		"time_entries": {"icon": "timer", "status_style": "entry-chip", "visual": "entry-table"},
		"timesheets": {"icon": "clipboard-check", "status_style": "approval-chip", "visual": "approval-queue"},
		"overtime": {"icon": "trending-up", "status_style": "ot-chip", "visual": "ot-queue"},
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
	"stream": ATTENDANCE_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"attendance_policy_created",
		"attendance_policy_updated",
		"overtime_rule_created",
		"overtime_rule_updated",
		"attendance_schedule_created",
		"schedule_overlap_blocked",
		"attendance_shift_created",
		"time_entry_recorded",
		"overtime_time_entry_recorded",
		"double_time_entry_recorded",
		"comp_time_accrued",
		"break_recorded",
		"timesheet_submitted",
		"timesheet_approved",
		"timesheet_rejected",
		"overtime_approved",
		"overtime_denied",
		"leave_requested",
		"leave_approved",
		"leave_rejected",
		"attendance_exception_recorded",
		"attendance_exception_escalated",
		"attendance_payroll_export_created",
		"attendance_agent_registered",
		"cross_tenant_access_blocked",
	],
	"states": ["draft", "submitted", "approved", "rejected", "exported", "locked", "queued", "blocked"],
	"guardrails": [
		"attendance_batch_requires_bytewax",
		"attendance_event_requires_bytewax",
		"privileged_agent_action_requires_human_approval",
		"cross_tenant_access_denied",
		"employee_cannot_approve_own_timesheet",
	],
}


RULES: list[dict[str, Any]] = [
	# --- Tenant context and write policy (mandatory gates) ---
	{"name": "tenant_context_required", "description": "Attendance operations require tenant context; deny if missing.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "operation_policy_required", "description": "Attendance write operations require an attached policy.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},

	# --- Cross-tenant access prevention ---
	{"name": "cross_tenant_employee_access_denied", "description": "Attendance operations referencing an employee from a different tenant are denied.", "condition": {"operation_type": "write", "employee_tenant_mismatch": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_same_tenant_employee"}},
	{"name": "cross_tenant_schedule_access_denied", "description": "Attendance operations referencing a schedule from a different tenant are denied.", "condition": {"operation_type": "write", "schedule_tenant_mismatch": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_same_tenant_schedule"}},
	{"name": "cross_tenant_policy_access_denied", "description": "Attendance policies cannot be applied across tenants.", "condition": {"operation_type": "write", "policy_tenant_mismatch": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_same_tenant_policy"}},

	# --- Privilege escalation prevention ---
	{"name": "employee_cannot_approve_own_timesheet", "description": "An employee cannot approve their own timesheet (SoD).", "condition": {"operation": "approve_timesheet", "submitter_equals_approver": True}, "effect": {"decision": "deny", "reason": "self_approval_segregation_of_duties_violation", "required_action": "assign_independent_approver"}},
	{"name": "employee_cannot_approve_own_overtime", "description": "An employee cannot approve their own overtime (SoD).", "condition": {"operation": "approve_overtime", "requester_equals_approver": True}, "effect": {"decision": "deny", "reason": "self_approval_segregation_of_duties_violation", "required_action": "assign_independent_overtime_approver"}},
	{"name": "supervisor_entry_requires_employee_acknowledgement", "description": "Supervisor-entered time records require employee acknowledgement within 24 hours.", "condition": {"operation": "record_time_entry", "entry_method": "supervisor_entry", "employee_acknowledged": False}, "effect": {"decision": "require_review", "reason": "supervisor_entry_requires_employee_acknowledgement", "required_action": "request_employee_acknowledgement"}},

	# --- Attendance policy ---
	{"name": "policy_name_required", "description": "Attendance policies require a name.", "condition": {"operation": "create_time_policy", "name_present": False}, "effect": {"decision": "deny", "reason": "time_policy_name_required", "required_action": "provide_policy_name"}},
	{"name": "policy_timezone_required", "description": "Attendance policies require a timezone.", "condition": {"operation": "create_time_policy", "timezone_present": False}, "effect": {"decision": "deny", "reason": "time_policy_timezone_required", "required_action": "provide_timezone"}},
	{"name": "policy_workweek_required", "description": "Attendance policies require a workweek definition.", "condition": {"operation": "create_time_policy", "workweek_present": False}, "effect": {"decision": "deny", "reason": "time_policy_workweek_required", "required_action": "provide_workweek"}},
	{"name": "policy_overtime_threshold_required", "description": "Attendance policies require an overtime threshold.", "condition": {"operation": "create_time_policy", "overtime_threshold_present": False}, "effect": {"decision": "deny", "reason": "overtime_threshold_required", "required_action": "provide_overtime_threshold"}},
	{"name": "policy_overtime_threshold_positive", "description": "Overtime thresholds must be positive.", "condition": {"operation": "create_time_policy", "overtime_threshold_positive": False}, "effect": {"decision": "deny", "reason": "overtime_threshold_must_be_positive", "required_action": "set_positive_overtime_threshold"}},
	{"name": "policy_overtime_method_supported", "description": "Overtime calculation method must be from the supported set.", "condition": {"operation": "create_time_policy", "overtime_method_supported": False}, "effect": {"decision": "deny", "reason": "overtime_method_not_supported", "required_action": "choose_supported_overtime_method"}},
	{"name": "policy_comp_time_policy_supported", "description": "Comp-time policy must be from the supported set.", "condition": {"operation": "create_time_policy", "comp_time_policy_supported": False}, "effect": {"decision": "deny", "reason": "comp_time_policy_not_supported", "required_action": "choose_supported_comp_time_policy"}},
	{"name": "policy_minimum_rest_between_shifts_enforced", "description": "Attendance policy must enforce minimum 11-hour rest between shifts.", "condition": {"operation": "create_time_policy", "min_rest_hours_lt": 11}, "effect": {"decision": "deny", "reason": "minimum_rest_between_shifts_required", "required_action": "set_minimum_rest_to_11_hours"}},

	# --- Overtime calculation rules ---
	{"name": "overtime_rule_requires_daily_threshold", "description": "Overtime rules must define a daily overtime threshold.", "condition": {"operation": "create_overtime_rule", "daily_threshold_present": False}, "effect": {"decision": "deny", "reason": "daily_overtime_threshold_required", "required_action": "set_daily_overtime_threshold"}},
	{"name": "overtime_rule_requires_weekly_threshold", "description": "Overtime rules must define a weekly overtime threshold.", "condition": {"operation": "create_overtime_rule", "weekly_threshold_present": False}, "effect": {"decision": "deny", "reason": "weekly_overtime_threshold_required", "required_action": "set_weekly_overtime_threshold"}},
	{"name": "overtime_rate_multiplier_positive", "description": "Overtime rate multiplier must be greater than 1.0.", "condition": {"operation": "create_overtime_rule", "overtime_rate_multiplier_lte": 1.0}, "effect": {"decision": "deny", "reason": "overtime_rate_multiplier_must_exceed_1", "required_action": "set_valid_overtime_rate_multiplier"}},
	{"name": "double_time_threshold_exceeds_overtime_threshold", "description": "Double-time daily threshold must exceed the overtime daily threshold.", "condition": {"operation": "create_overtime_rule", "double_time_below_overtime": True}, "effect": {"decision": "deny", "reason": "double_time_threshold_must_exceed_overtime_threshold", "required_action": "correct_double_time_threshold"}},
	{"name": "overtime_preauthorization_required", "description": "Overtime work requires manager preauthorization before the fact.", "condition": {"operation": "record_time_entry", "entry_type": "overtime", "preauthorized": False}, "effect": {"decision": "require_review", "reason": "overtime_preauthorization_required", "required_action": "obtain_overtime_preauthorization"}},
	{"name": "comp_time_accrual_cap_enforced", "description": "Comp-time accrual cannot exceed the tenant-configured maximum balance.", "condition": {"operation": "record_time_entry", "entry_type": "comp_time", "comp_time_cap_exceeded": True}, "effect": {"decision": "deny", "reason": "comp_time_accrual_cap_exceeded", "required_action": "use_comp_time_before_accruing_more"}},
	{"name": "public_holiday_work_requires_approval", "description": "Working on a public holiday requires prior approval.", "condition": {"operation": "record_time_entry", "public_holiday": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "public_holiday_work_requires_approval", "required_action": "obtain_public_holiday_work_approval"}},
	{"name": "rest_day_work_requires_approval", "description": "Working on a scheduled rest day requires prior approval.", "condition": {"operation": "record_time_entry", "scheduled_rest_day": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "rest_day_work_requires_approval", "required_action": "obtain_rest_day_work_approval"}},
	{"name": "maximum_consecutive_days_exceeded_denied", "description": "Scheduling an employee beyond the maximum consecutive working days is denied.", "condition": {"operation": "create_shift", "consecutive_days_exceeded": True}, "effect": {"decision": "deny", "reason": "maximum_consecutive_working_days_exceeded", "required_action": "insert_rest_day"}},
	{"name": "maximum_shift_duration_exceeded_denied", "description": "Shifts exceeding the maximum allowed duration are denied.", "condition": {"operation": "create_shift", "shift_duration_hours_gt": 16}, "effect": {"decision": "deny", "reason": "maximum_shift_duration_exceeded", "required_action": "shorten_shift_or_split"}},

	# --- Schedule rules ---
	{"name": "schedule_employee_required", "description": "Work schedules require an employee.", "condition": {"operation": "create_schedule", "employee_present": False}, "effect": {"decision": "deny", "reason": "schedule_employee_required", "required_action": "select_employee"}},
	{"name": "schedule_policy_required", "description": "Work schedules require an active attendance policy.", "condition": {"operation": "create_schedule", "policy_present": False}, "effect": {"decision": "deny", "reason": "schedule_policy_required", "required_action": "select_policy"}},
	{"name": "schedule_type_supported", "description": "Work schedules must use a supported schedule type.", "condition": {"operation": "create_schedule", "schedule_type_supported": False}, "effect": {"decision": "deny", "reason": "schedule_type_not_supported", "required_action": "choose_supported_schedule_type"}},
	{"name": "schedule_start_required", "description": "Work schedules require a start date.", "condition": {"operation": "create_schedule", "start_date_present": False}, "effect": {"decision": "deny", "reason": "schedule_start_date_required", "required_action": "provide_start_date"}},
	{"name": "schedule_end_required", "description": "Work schedules require an end date.", "condition": {"operation": "create_schedule", "end_date_present": False}, "effect": {"decision": "deny", "reason": "schedule_end_date_required", "required_action": "provide_end_date"}},
	{"name": "overlapping_schedule_denied", "description": "An employee cannot have two overlapping active work schedules.", "condition": {"operation": "create_schedule", "schedule_overlap_exists": True}, "effect": {"decision": "deny", "reason": "overlapping_schedules_not_allowed", "required_action": "end_or_delete_existing_schedule_first"}},

	# --- Shift rules ---
	{"name": "shift_schedule_required", "description": "Shifts require a schedule.", "condition": {"operation": "create_shift", "schedule_present": False}, "effect": {"decision": "deny", "reason": "shift_schedule_required", "required_action": "select_schedule"}},
	{"name": "shift_date_required", "description": "Shifts require a shift date.", "condition": {"operation": "create_shift", "shift_date_present": False}, "effect": {"decision": "deny", "reason": "shift_date_required", "required_action": "provide_shift_date"}},
	{"name": "shift_start_required", "description": "Shifts require a start time.", "condition": {"operation": "create_shift", "start_time_present": False}, "effect": {"decision": "deny", "reason": "shift_start_time_required", "required_action": "provide_start_time"}},
	{"name": "shift_end_required", "description": "Shifts require an end time.", "condition": {"operation": "create_shift", "end_time_present": False}, "effect": {"decision": "deny", "reason": "shift_end_time_required", "required_action": "provide_end_time"}},
	{"name": "shift_minimum_duration_enforced", "description": "Shifts must be at least 30 minutes long.", "condition": {"operation": "create_shift", "shift_duration_minutes_lt": 30}, "effect": {"decision": "deny", "reason": "shift_too_short", "required_action": "extend_shift_to_minimum_duration"}},
	{"name": "shift_insufficient_rest_denied", "description": "Shifts must be separated by at least 11 hours of rest from the previous shift.", "condition": {"operation": "create_shift", "rest_before_shift_hours_lt": 11}, "effect": {"decision": "deny", "reason": "insufficient_rest_between_shifts", "required_action": "reschedule_to_ensure_minimum_rest"}},

	# --- Time entry rules ---
	{"name": "time_entry_employee_required", "description": "Time entries require an employee.", "condition": {"operation": "record_time_entry", "employee_present": False}, "effect": {"decision": "deny", "reason": "time_entry_employee_required", "required_action": "select_employee"}},
	{"name": "time_entry_shift_required", "description": "Time entries require a shift.", "condition": {"operation": "record_time_entry", "shift_present": False}, "effect": {"decision": "deny", "reason": "time_entry_shift_required", "required_action": "select_shift"}},
	{"name": "time_entry_type_supported", "description": "Time entries must use a supported entry type.", "condition": {"operation": "record_time_entry", "entry_type_supported": False}, "effect": {"decision": "deny", "reason": "time_entry_type_not_supported", "required_action": "choose_supported_entry_type"}},
	{"name": "time_entry_method_supported", "description": "Time entries must use a supported entry method.", "condition": {"operation": "record_time_entry", "entry_method_supported": False}, "effect": {"decision": "deny", "reason": "time_entry_method_not_supported", "required_action": "choose_supported_entry_method"}},
	{"name": "time_entry_clock_in_required", "description": "Time entries require a clock-in time.", "condition": {"operation": "record_time_entry", "clock_in_present": False}, "effect": {"decision": "deny", "reason": "clock_in_required", "required_action": "provide_clock_in"}},
	{"name": "device_required_for_tracked_method", "description": "Mobile, kiosk, and biometric entries require a registered device.", "condition": {"operation": "record_time_entry", "tracked_method": True, "device_present": False}, "effect": {"decision": "deny", "reason": "attendance_device_required", "required_action": "attach_registered_device"}},
	{"name": "geofence_requires_review", "description": "Entries recorded outside the geofence require review.", "condition": {"operation": "record_time_entry", "geofence_verified": False, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "geofence_review_required", "required_action": "record_geofence_review"}},
	{"name": "biometric_confidence_requires_review", "description": "Time entries with low biometric confidence require review.", "condition": {"operation": "record_time_entry", "biometric_low_confidence": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "biometric_review_required", "required_action": "record_biometric_review"}},
	{"name": "duplicate_time_entry_denied", "description": "Duplicate clock-in entries within the duplicate window are denied.", "condition": {"operation": "record_time_entry", "duplicate_within_window": True}, "effect": {"decision": "deny", "reason": "duplicate_time_entry_detected", "required_action": "verify_entry_is_not_duplicate"}},

	# --- Break rules ---
	{"name": "break_time_entry_required", "description": "Breaks require a parent time entry.", "condition": {"operation": "record_break", "time_entry_present": False}, "effect": {"decision": "deny", "reason": "break_time_entry_required", "required_action": "select_time_entry"}},
	{"name": "break_start_required", "description": "Breaks require a start time.", "condition": {"operation": "record_break", "start_time_present": False}, "effect": {"decision": "deny", "reason": "break_start_time_required", "required_action": "provide_break_start"}},
	{"name": "break_end_required", "description": "Breaks require an end time.", "condition": {"operation": "record_break", "end_time_present": False}, "effect": {"decision": "deny", "reason": "break_end_time_required", "required_action": "provide_break_end"}},
	{"name": "break_cannot_exceed_shift_duration", "description": "Total break time cannot exceed shift duration.", "condition": {"operation": "record_break", "break_exceeds_shift": True}, "effect": {"decision": "deny", "reason": "break_duration_exceeds_shift", "required_action": "correct_break_times"}},

	# --- Timesheet rules ---
	{"name": "timesheet_employee_required", "description": "Timesheets require an employee.", "condition": {"operation": "submit_timesheet", "employee_present": False}, "effect": {"decision": "deny", "reason": "timesheet_employee_required", "required_action": "select_employee"}},
	{"name": "timesheet_period_required", "description": "Timesheets require a period.", "condition": {"operation": "submit_timesheet", "period_present": False}, "effect": {"decision": "deny", "reason": "timesheet_period_required", "required_action": "select_period"}},
	{"name": "timesheet_entries_required", "description": "Timesheets require at least one time entry.", "condition": {"operation": "submit_timesheet", "entries_present": False}, "effect": {"decision": "deny", "reason": "timesheet_entries_required", "required_action": "attach_entries"}},
	{"name": "timesheet_submitter_required", "description": "Timesheets require a submitter identity.", "condition": {"operation": "submit_timesheet", "submitter_present": False}, "effect": {"decision": "deny", "reason": "timesheet_submitter_required", "required_action": "record_submitter"}},
	{"name": "timesheet_hours_nonnegative", "description": "Timesheet total hours cannot be negative.", "condition": {"operation": "submit_timesheet", "total_hours_negative": True}, "effect": {"decision": "deny", "reason": "timesheet_hours_must_be_nonnegative", "required_action": "correct_entries"}},
	{"name": "timesheet_approver_required", "description": "Timesheet approval requires an approver identity.", "condition": {"operation": "approve_timesheet", "approver_present": False}, "effect": {"decision": "deny", "reason": "timesheet_approver_required", "required_action": "record_approver"}},
	{"name": "timesheet_late_submission_requires_reason", "description": "Timesheets submitted after the period deadline require a reason.", "condition": {"operation": "submit_timesheet", "late_submission": True, "reason_present": False}, "effect": {"decision": "deny", "reason": "late_submission_reason_required", "required_action": "provide_late_submission_reason"}},
	{"name": "timesheet_resubmission_requires_change_reason", "description": "Resubmitting a rejected timesheet requires a documented change reason.", "condition": {"operation": "resubmit_timesheet", "change_reason_present": False}, "effect": {"decision": "deny", "reason": "resubmission_change_reason_required", "required_action": "document_change_reason"}},

	# --- Leave rules ---
	{"name": "leave_employee_required", "description": "Leave requests require an employee.", "condition": {"operation": "request_leave", "employee_present": False}, "effect": {"decision": "deny", "reason": "leave_employee_required", "required_action": "select_employee"}},
	{"name": "leave_type_supported", "description": "Leave requests must use a supported leave type.", "condition": {"operation": "request_leave", "leave_type_supported": False}, "effect": {"decision": "deny", "reason": "leave_type_not_supported", "required_action": "choose_supported_leave_type"}},
	{"name": "leave_start_required", "description": "Leave requests require a start date.", "condition": {"operation": "request_leave", "start_date_present": False}, "effect": {"decision": "deny", "reason": "leave_start_date_required", "required_action": "provide_start_date"}},
	{"name": "leave_end_required", "description": "Leave requests require an end date.", "condition": {"operation": "request_leave", "end_date_present": False}, "effect": {"decision": "deny", "reason": "leave_end_date_required", "required_action": "provide_end_date"}},
	{"name": "leave_reason_required", "description": "Leave requests require a reason.", "condition": {"operation": "request_leave", "reason_present": False}, "effect": {"decision": "deny", "reason": "leave_reason_required", "required_action": "provide_reason"}},
	{"name": "extended_leave_requires_review", "description": "Leave requests exceeding 10 days or of unpaid type require manager review.", "condition": {"operation": "request_leave", "review_required": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "leave_review_required", "required_action": "record_leave_approval"}},
	{"name": "sick_leave_over_threshold_requires_medical_cert", "description": "Sick leave exceeding 3 days requires a medical certificate.", "condition": {"operation": "request_leave", "leave_type": "sick", "duration_days_gt": 3, "medical_cert_present": False}, "effect": {"decision": "require_review", "reason": "medical_certificate_required_for_extended_sick_leave", "required_action": "submit_medical_certificate"}},

	# --- Exception rules ---
	{"name": "exception_employee_required", "description": "Attendance exceptions require an employee.", "condition": {"operation": "record_exception", "employee_present": False}, "effect": {"decision": "deny", "reason": "attendance_exception_employee_required", "required_action": "select_employee"}},
	{"name": "exception_type_supported", "description": "Attendance exceptions must use a supported exception type.", "condition": {"operation": "record_exception", "exception_type_supported": False}, "effect": {"decision": "deny", "reason": "attendance_exception_type_not_supported", "required_action": "choose_supported_exception_type"}},
	{"name": "exception_owner_required_high_severity", "description": "High-severity attendance exceptions require an assigned owner.", "condition": {"operation": "record_exception", "high_severity": True, "owner_present": False}, "effect": {"decision": "deny", "reason": "attendance_exception_owner_required", "required_action": "assign_exception_owner"}},
	{"name": "exception_auto_escalation_after_48_hours", "description": "Unresolved high-severity exceptions must be escalated after 48 hours.", "condition": {"operation": "record_exception", "high_severity": True, "hours_open_gt": 48, "escalated": False}, "effect": {"decision": "require_review", "reason": "exception_auto_escalation_required", "required_action": "escalate_exception_to_hr"}},

	# --- Payroll export rules ---
	{"name": "export_period_required", "description": "Payroll exports require a period.", "condition": {"operation": "create_payroll_export", "period_present": False}, "effect": {"decision": "deny", "reason": "attendance_export_period_required", "required_action": "select_period"}},
	{"name": "export_timesheets_required", "description": "Payroll exports require timesheets.", "condition": {"operation": "create_payroll_export", "timesheets_present": False}, "effect": {"decision": "deny", "reason": "attendance_export_timesheets_required", "required_action": "attach_timesheets"}},
	{"name": "export_timesheets_approved", "description": "Payroll exports can only include fully approved timesheets.", "condition": {"operation": "create_payroll_export", "all_timesheets_approved": False}, "effect": {"decision": "deny", "reason": "timesheets_must_be_approved", "required_action": "approve_timesheets"}},
	{"name": "export_approver_required", "description": "Payroll exports require documented approval.", "condition": {"operation": "create_payroll_export", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "attendance_export_approval_required", "required_action": "record_export_approval"}},
	{"name": "export_open_overtime_blocked", "description": "Payroll exports are blocked if unapproved overtime entries exist for the period.", "condition": {"operation": "create_payroll_export", "unapproved_overtime_present": True}, "effect": {"decision": "deny", "reason": "unapproved_overtime_blocks_export", "required_action": "resolve_all_overtime_approvals_before_export"}},
	{"name": "export_open_exceptions_blocked", "description": "Payroll exports are blocked if unresolved high-severity exceptions exist for the period.", "condition": {"operation": "create_payroll_export", "open_high_severity_exceptions": True}, "effect": {"decision": "deny", "reason": "open_high_severity_exceptions_block_export", "required_action": "resolve_exceptions_before_export"}},

	# --- Audit and streaming ---
	{"name": "audit_required_for_state_change", "description": "Attendance state changes must be written to the audit trail.", "condition": {"operation_type": "write", "audit_enabled": False}, "effect": {"decision": "deny", "reason": "attendance_audit_required", "required_action": "enable_audit"}},
	{"name": "attendance_batch_requires_bytewax", "description": "Attendance batch operations must use the Bytewax event stream.", "condition": {"operation": "attendance_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_to_bytewax_stream"}},
	{"name": "agent_runtime_supported", "description": "Attendance agents must use a supported runtime.", "condition": {"operation": "register_attendance_agent", "runtime_supported": False}, "effect": {"decision": "deny", "reason": "attendance_agent_runtime_not_supported", "required_action": "choose_supported_runtime"}},
	{"name": "agent_role_supported", "description": "Attendance agents must use a supported role.", "condition": {"operation": "register_attendance_agent", "role_supported": False}, "effect": {"decision": "deny", "reason": "attendance_agent_role_not_supported", "required_action": "choose_supported_role"}},
	{"name": "agent_scope_limited", "description": "Attendance agents cannot autonomously post privileged state changes without human approval.", "condition": {"operation": "agent_action", "privileged_action": True, "human_approved": False}, "effect": {"decision": "require_review", "reason": "attendance_agent_human_approval_required", "required_action": "record_human_approval"}},
]


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "ui", "theme"],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"policies": {"type": "object"},
		"schedules": {"type": "object"},
		"shifts": {"type": "object"},
		"time_entries": {"type": "object"},
		"breaks": {"type": "object"},
		"timesheets": {"type": "object"},
		"overtime_rules": {"type": "object"},
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
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
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
		if key.endswith("_lte"):
			if context.get(key[:-4]) is None or context[key[:-4]] > expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gte"):
			if context.get(key[:-4]) is None or context[key[:-4]] < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
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
