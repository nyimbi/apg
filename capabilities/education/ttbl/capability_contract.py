"""Executable capability contract for APG Timetabling & Scheduling."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "education_ttbl"
CAPABILITY_NAME = "Timetabling & Scheduling"
CAPABILITY_VERSION = "1.0.0"
TTBL_EVENT_STREAM = "apg.education.ttbl.lifecycle"

# --- supported value sets ---------------------------------------------------

SUPPORTED_TIMETABLE_TYPES = [
	"master", "class", "teacher", "room", "exam", "substitution", "draft",
]
SUPPORTED_CONSTRAINT_TYPES = [
	"teacher_availability", "room_capacity", "subject_per_day_limit",
	"consecutive_periods_limit", "lunch_break_required", "double_period",
	"room_type_required", "teacher_qualification_required",
	"student_group_conflict", "special_needs_accommodation",
]
SUPPORTED_SLOT_DURATIONS = [30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120]
SUPPORTED_DAYS_OF_WEEK = [
	"monday", "tuesday", "wednesday", "thursday", "friday",
	"saturday", "sunday",
]
SUPPORTED_ROOM_TYPES = [
	"classroom", "lab", "computer_lab", "library", "gymnasium",
	"auditorium", "sports_field", "workshop", "art_studio", "music_room",
]
SUPPORTED_TIMETABLE_STATUSES = [
	"draft", "generating", "conflict_review", "approved", "published",
	"superseded", "archived",
]
SUPPORTED_CONFLICT_TYPES = [
	"teacher_double_booked", "room_double_booked", "student_group_overlap",
	"teacher_unavailable", "room_unavailable", "capacity_exceeded",
	"constraint_violated", "qualification_mismatch",
]
SUPPORTED_CONFLICT_RESOLUTIONS = [
	"swap_periods", "reassign_room", "reassign_teacher", "split_group",
	"request_substitution", "manual_override", "defer_to_next_slot",
]
SUPPORTED_SUBSTITUTION_STATUSES = [
	"pending", "assigned", "confirmed", "declined", "completed", "cancelled",
]
SUPPORTED_GENERATION_ALGORITHMS = [
	"constraint_propagation", "genetic_algorithm", "simulated_annealing",
	"backtracking", "greedy_heuristic",
]
SUPPORTED_EXPORT_FORMATS = [
	"ical", "csv", "pdf", "json", "html", "excel",
]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = [
	"constraint_analyst", "conflict_resolver", "substitution_coordinator",
	"schedule_optimizer", "compliance_auditor",
]

# --- wiring -----------------------------------------------------------------

PROVIDES = [
	"timetable_generation_workflow",
	"constraint_management_workflow",
	"room_allocation_workflow",
	"teacher_assignment_workflow",
	"conflict_detection_workflow",
	"conflict_resolution_workflow",
	"substitution_management_workflow",
	"timetable_publication_workflow",
	"exam_scheduling_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "mqeb", "schd", "comp"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/ttbl/dashboard", "component": "TtblDashboard", "permission": "education_ttbl:view", "nav_group": "Overview"},
	{"name": "timetables", "path": "/ttbl/timetables", "component": "TimetableList", "permission": "education_ttbl:view", "nav_group": "Timetables"},
	{"name": "timetable_builder", "path": "/ttbl/timetables/<timetable_id>/build", "component": "TimetableBuilder", "permission": "education_ttbl:manage_timetables", "nav_group": "Timetables"},
	{"name": "timetable_viewer", "path": "/ttbl/timetables/<timetable_id>/view", "component": "TimetableViewer", "permission": "education_ttbl:view", "nav_group": "Timetables"},
	{"name": "constraints", "path": "/ttbl/constraints", "component": "ConstraintEditor", "permission": "education_ttbl:manage_constraints", "nav_group": "Configuration"},
	{"name": "rooms", "path": "/ttbl/rooms", "component": "RoomInventory", "permission": "education_ttbl:manage_rooms", "nav_group": "Resources"},
	{"name": "conflicts", "path": "/ttbl/conflicts", "component": "ConflictResolutionWorkbench", "permission": "education_ttbl:resolve_conflicts", "nav_group": "Operations"},
	{"name": "substitutions", "path": "/ttbl/substitutions", "component": "SubstitutionConsole", "permission": "education_ttbl:manage_substitutions", "nav_group": "Operations"},
	{"name": "exam_schedules", "path": "/ttbl/exams", "component": "ExamScheduleConsole", "permission": "education_ttbl:manage_exams", "nav_group": "Examinations"},
	{"name": "teacher_timetable", "path": "/ttbl/teachers/<teacher_id>", "component": "TeacherTimetableView", "permission": "education_ttbl:view", "nav_group": "Resources"},
	{"name": "room_timetable", "path": "/ttbl/rooms/<room_id>", "component": "RoomTimetableView", "permission": "education_ttbl:view", "nav_group": "Resources"},
	{"name": "exports", "path": "/ttbl/exports", "component": "TimetableExportConsole", "permission": "education_ttbl:export", "nav_group": "Export"},
	{"name": "agents", "path": "/ttbl/agents", "component": "TtblAgentWorkbench", "permission": "education_ttbl:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/ttbl/settings", "component": "TtblSettings", "permission": "education_ttbl:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "ttbl_scheduler",
	"tokens": {
		"color.primary": "#6D28D9",
		"color.accent": "#0891B2",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#FAF5FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E1B4B",
		"text.secondary": "#6366F1",
		"border.radius": "6px",
		"density": "compact",
	},
	"components": {
		"timetables": {"icon": "grid", "status_indicator": "timetable-status-chip"},
		"constraints": {"icon": "sliders", "status_indicator": "constraint-type-chip"},
		"rooms": {"icon": "home", "status_indicator": "room-type-chip"},
		"conflicts": {"icon": "alert-triangle", "status_indicator": "conflict-type-chip"},
		"substitutions": {"icon": "repeat", "status_indicator": "substitution-status-chip"},
		"exams": {"icon": "pen-tool", "status_indicator": "exam-status-chip"},
		"teacher_view": {"icon": "user", "status_indicator": "availability-chip"},
		"exports": {"icon": "download", "status_indicator": "export-format-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": TTBL_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"timetable_created", "timetable_generation_started", "timetable_generation_completed",
		"conflict_detected", "conflict_resolved", "constraint_added",
		"room_allocated", "teacher_assigned", "substitution_requested",
		"substitution_assigned", "timetable_published", "exam_schedule_published",
	],
	"guardrails": [
		"ttbl_batch_requires_bytewax",
		"timetable_publish_requires_zero_conflicts",
		"constraint_removal_requires_approval",
		"exam_schedule_requires_room_confirmation",
		"substitution_assignment_requires_teacher_consent",
		"privileged_agent_action_requires_human_approval",
		"cross_tenant_room_booking_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "ttbl_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "ttbl_policy_required", "required_action": "attach_ttbl_policy"}},
	{"name": "timetable_type_supported", "condition": {"operation": "create_timetable", "timetable_type_supported": False}, "effect": {"decision": "deny", "reason": "timetable_type_not_supported", "required_action": "select_supported_timetable_type"}},
	{"name": "timetable_publish_requires_zero_conflicts", "condition": {"operation": "publish_timetable", "unresolved_conflicts_present": True}, "effect": {"decision": "deny", "reason": "timetable_cannot_be_published_with_unresolved_conflicts", "required_action": "resolve_all_conflicts"}},
	{"name": "timetable_publish_requires_approval", "condition": {"operation": "publish_timetable", "approval_reference_present": False}, "effect": {"decision": "deny", "reason": "timetable_publication_requires_approver_sign_off", "required_action": "obtain_publication_approval"}},
	{"name": "constraint_type_supported", "condition": {"operation": "add_constraint", "constraint_type_supported": False}, "effect": {"decision": "deny", "reason": "constraint_type_not_supported", "required_action": "select_supported_constraint_type"}},
	{"name": "constraint_removal_requires_approval", "condition": {"operation": "remove_constraint", "approval_reference_present": False}, "effect": {"decision": "deny", "reason": "constraint_removal_requires_approval", "required_action": "obtain_constraint_removal_approval"}},
	{"name": "slot_duration_supported", "condition": {"operation": "create_time_slot", "slot_duration_supported": False}, "effect": {"decision": "deny", "reason": "slot_duration_not_supported", "required_action": "select_supported_slot_duration"}},
	{"name": "room_type_supported", "condition": {"operation": "create_room", "room_type_supported": False}, "effect": {"decision": "deny", "reason": "room_type_not_supported", "required_action": "select_supported_room_type"}},
	{"name": "room_capacity_check_required", "condition": {"operation": "allocate_room", "capacity_check_performed": False}, "effect": {"decision": "deny", "reason": "room_capacity_check_required_before_allocation", "required_action": "perform_capacity_check"}},
	{"name": "cross_tenant_room_booking_denied", "condition": {"operation": "allocate_room", "room_tenant_matches_requestor_tenant": False}, "effect": {"decision": "deny", "reason": "cross_tenant_room_booking_denied", "required_action": "book_within_tenant"}},
	{"name": "conflict_type_supported", "condition": {"operation": "log_conflict", "conflict_type_supported": False}, "effect": {"decision": "deny", "reason": "conflict_type_not_supported", "required_action": "select_supported_conflict_type"}},
	{"name": "conflict_resolution_supported", "condition": {"operation": "resolve_conflict", "resolution_type_supported": False}, "effect": {"decision": "deny", "reason": "conflict_resolution_type_not_supported", "required_action": "select_supported_resolution_type"}},
	{"name": "substitution_requires_teacher_consent", "condition": {"operation": "assign_substitution", "teacher_consent_recorded": False}, "effect": {"decision": "deny", "reason": "substitution_requires_teacher_consent", "required_action": "record_teacher_consent"}},
	{"name": "substitution_status_supported", "condition": {"operation": "update_substitution_status", "substitution_status_supported": False}, "effect": {"decision": "deny", "reason": "substitution_status_not_supported", "required_action": "select_supported_substitution_status"}},
	{"name": "generation_algorithm_supported", "condition": {"operation": "generate_timetable", "algorithm_supported": False}, "effect": {"decision": "deny", "reason": "generation_algorithm_not_supported", "required_action": "select_supported_algorithm"}},
	{"name": "exam_schedule_requires_room_confirmation", "condition": {"operation": "publish_exam_schedule", "rooms_confirmed": False}, "effect": {"decision": "deny", "reason": "exam_schedule_requires_confirmed_rooms", "required_action": "confirm_exam_rooms"}},
	{"name": "export_format_supported", "condition": {"operation": "export_timetable", "export_format_supported": False}, "effect": {"decision": "deny", "reason": "export_format_not_supported", "required_action": "select_supported_export_format"}},
	{"name": "timetable_status_transition_valid", "condition": {"operation": "update_timetable_status", "status_transition_valid": False}, "effect": {"decision": "deny", "reason": "invalid_timetable_status_transition", "required_action": "follow_valid_status_transition"}},
	{"name": "privileged_agent_action_requires_human_approval", "condition": {"operation": "agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required_for_privileged_agent_action", "required_action": "record_human_approval"}},
	{"name": "batch_import_requires_bytewax", "condition": {"operation": "batch_import", "event_stream": "bytewax", "item_count_valid": False}, "effect": {"decision": "deny", "reason": "batch_import_requires_bytewax_stream", "required_action": "configure_bytewax_stream"}},
]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ui": {"enable_dashboard": True, "enable_timetables": True, "enable_constraints": True, "enable_rooms": True, "enable_conflicts": True, "enable_substitutions": True, "enable_exams": True, "enable_exports": True, "enable_agents": True},
	"theme": {"default_theme": "ttbl_scheduler", "allow_tenant_overrides": True},
	"timetables": {"supported_types": SUPPORTED_TIMETABLE_TYPES, "supported_statuses": SUPPORTED_TIMETABLE_STATUSES, "supported_algorithms": SUPPORTED_GENERATION_ALGORITHMS, "publish_requires_zero_conflicts": True, "publish_requires_approval": True},
	"constraints": {"supported_types": SUPPORTED_CONSTRAINT_TYPES, "removal_requires_approval": True},
	"time_slots": {"supported_durations_minutes": SUPPORTED_SLOT_DURATIONS, "supported_days": SUPPORTED_DAYS_OF_WEEK},
	"rooms": {"supported_types": SUPPORTED_ROOM_TYPES, "capacity_check_required": True, "cross_tenant_booking_denied": True},
	"conflicts": {"supported_types": SUPPORTED_CONFLICT_TYPES, "supported_resolutions": SUPPORTED_CONFLICT_RESOLUTIONS},
	"substitutions": {"supported_statuses": SUPPORTED_SUBSTITUTION_STATUSES, "consent_required": True},
	"exports": {"supported_formats": SUPPORTED_EXPORT_FORMATS},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_room_booking_denied": True, "publish_requires_zero_conflicts": True},
	"observability": {"event_stream": TTBL_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "monitoring": "moni", "compliance": "comp", "event_stream": "bytewax", "scheduler": "schd"},
}


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	"""Return the full capability contract for the given tenant."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	config["ui"]["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {
			"required": ["tenant_id", "ui", "theme"],
			"properties": {
				"tenant_id": {"type": "string"},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
				"timetables": {"type": "object"},
				"constraints": {"type": "object"},
				"rooms": {"type": "object"},
				"conflicts": {"type": "object"},
				"governance": {"type": "object"},
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
			"template_roots": ["education/ttbl/templates"],
			"routes": UI_ROUTES,
		},
		"theme": THEME,
		"provides": PROVIDES,
		"requires": REQUIRES,
		"streaming": STREAMING,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic rules against the provided context dict."""
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			return {
				"matched_rule": rule["name"],
				**rule["effect"],
			}
	return {"matched_rule": None, "decision": "allow", "reason": "no_rule_matched"}
