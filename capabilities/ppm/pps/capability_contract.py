"""Executable capability contract for APG Project Planning & Scheduling (pps)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "ppm_pps"
CAPABILITY_NAME = "Project Planning & Scheduling"
CAPABILITY_VERSION = "1.0.0"
PPS_EVENT_STREAM = "apg.ppm.pps.lifecycle"

# ── Supported enum values ────────────────────────────────────────────────────
SUPPORTED_PROJECT_STATUSES = ["draft", "planned", "in_progress", "on_hold", "completed", "cancelled", "archived"]
SUPPORTED_TASK_STATUSES = ["not_started", "in_progress", "completed", "blocked", "deferred", "cancelled"]
SUPPORTED_TASK_TYPES = ["work_package", "milestone", "summary", "deliverable", "gate", "buffer", "hammock"]
SUPPORTED_DEPENDENCY_TYPES = ["finish_to_start", "start_to_start", "finish_to_finish", "start_to_finish"]
SUPPORTED_CONSTRAINT_TYPES = ["as_soon_as_possible", "as_late_as_possible", "must_start_on", "must_finish_on", "start_no_earlier_than", "finish_no_later_than"]
SUPPORTED_WBS_LEVELS = ["project", "phase", "deliverable", "work_package", "activity", "task"]
SUPPORTED_LEVELLING_ALGORITHMS = ["priority_based", "earliest_deadline", "minimum_slack", "resource_critical_chain", "genetic"]
SUPPORTED_CRITICAL_PATH_METHODS = ["cpm", "pert", "ccpm", "monte_carlo"]
SUPPORTED_SCHEDULING_MODES = ["fixed_duration", "fixed_work", "fixed_units", "effort_driven"]
SUPPORTED_CALENDAR_TYPES = ["standard_5x8", "standard_7x24", "custom", "project_specific", "resource_specific"]
SUPPORTED_PROGRESS_METHODS = ["percent_complete", "physical_percent_complete", "milestones_achieved", "earned_value"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["schedule_builder", "critical_path_analyst", "resource_leveller", "dependency_mapper", "timeline_reviewer"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_METHODOLOGIES = ["waterfall", "agile_scrum", "agile_kanban", "hybrid", "prince2", "pmbok", "critical_chain"]

PROVIDES = [
	"wbs_creation_and_management",
	"critical_path_analysis",
	"resource_levelling",
	"dependency_management",
	"timeline_management",
	"schedule_optimisation",
	"project_calendar_management",
	"milestone_tracking",
	"schedule_risk_analysis",
	"gantt_chart_generation",
	"schedule_baseline_export",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "schd", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/ppm-pps/dashboard", "component": "PpsDashboard", "permission": "ppm_pps:view", "nav_group": "Overview"},
	{"name": "projects", "path": "/ppm-pps/projects", "component": "ProjectList", "permission": "ppm_pps:projects", "nav_group": "Projects"},
	{"name": "project_detail", "path": "/ppm-pps/projects/<id>", "component": "ProjectDetail", "permission": "ppm_pps:projects", "nav_group": "Projects"},
	{"name": "wbs", "path": "/ppm-pps/projects/<id>/wbs", "component": "WbsEditor", "permission": "ppm_pps:wbs", "nav_group": "Planning"},
	{"name": "gantt", "path": "/ppm-pps/projects/<id>/gantt", "component": "GanttChartView", "permission": "ppm_pps:gantt", "nav_group": "Planning"},
	{"name": "critical_path", "path": "/ppm-pps/projects/<id>/critical-path", "component": "CriticalPathAnalyser", "permission": "ppm_pps:critical_path", "nav_group": "Analysis"},
	{"name": "dependencies", "path": "/ppm-pps/projects/<id>/dependencies", "component": "DependencyGraph", "permission": "ppm_pps:dependencies", "nav_group": "Planning"},
	{"name": "resource_levelling", "path": "/ppm-pps/projects/<id>/levelling", "component": "ResourceLevellingConsole", "permission": "ppm_pps:levelling", "nav_group": "Resources"},
	{"name": "milestones", "path": "/ppm-pps/milestones", "component": "MilestoneTracker", "permission": "ppm_pps:milestones", "nav_group": "Tracking"},
	{"name": "calendars", "path": "/ppm-pps/calendars", "component": "CalendarManager", "permission": "ppm_pps:calendars", "nav_group": "Configuration"},
	{"name": "schedule_risk", "path": "/ppm-pps/projects/<id>/risk", "component": "ScheduleRiskAnalysis", "permission": "ppm_pps:risk", "nav_group": "Analysis"},
	{"name": "reports", "path": "/ppm-pps/reports", "component": "ScheduleReportBuilder", "permission": "ppm_pps:reports", "nav_group": "Reports"},
	{"name": "agents", "path": "/ppm-pps/agents", "component": "PpsAgentWorkbench", "permission": "ppm_pps:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/ppm-pps/settings", "component": "PpsSettings", "permission": "ppm_pps:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "ppm_pps_control",
	"tokens": {
		"color.primary": "#1E40AF",
		"color.accent": "#0D9488",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "6px",
		"density": "compact",
	},
	"components": {
		"project": {"icon": "folder", "status_indicator": "project-status-chip"},
		"task": {"icon": "check-square", "status_indicator": "task-status-chip"},
		"wbs": {"icon": "git-branch", "status_indicator": "wbs-level-chip"},
		"dependency": {"icon": "arrow-right", "status_indicator": "dependency-type-chip"},
		"milestone": {"icon": "flag", "status_indicator": "milestone-chip"},
		"critical_path": {"icon": "zap", "status_indicator": "critical-chip"},
		"gantt": {"icon": "bar-chart-horizontal", "status_indicator": "progress-chip"},
		"calendar": {"icon": "calendar", "status_indicator": "calendar-type-chip"},
		"agent": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": PPS_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"project_created",
		"project_updated",
		"wbs_element_added",
		"task_status_changed",
		"dependency_linked",
		"critical_path_recalculated",
		"resource_levelling_completed",
		"milestone_status_changed",
		"schedule_risk_assessed",
		"baseline_exported",
		"agent_registered",
	],
	"guardrails": [
		"schedule_batch_requires_bytewax",
		"wbs_circular_dependency_denied",
		"retroactive_schedule_edit_requires_change_request",
		"cross_tenant_schedule_access_denied",
		"critical_path_manipulation_denied",
		"privileged_agent_action_requires_human_approval",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"projects": {
		"supported_statuses": SUPPORTED_PROJECT_STATUSES,
		"supported_methodologies": SUPPORTED_METHODOLOGIES,
		"owner_required": True,
		"start_date_required": True,
		"evidence_required": True,
	},
	"tasks": {
		"supported_statuses": SUPPORTED_TASK_STATUSES,
		"supported_types": SUPPORTED_TASK_TYPES,
		"supported_scheduling_modes": SUPPORTED_SCHEDULING_MODES,
		"supported_constraint_types": SUPPORTED_CONSTRAINT_TYPES,
		"supported_progress_methods": SUPPORTED_PROGRESS_METHODS,
		"wbs_element_required": True,
		"duration_positive_required": True,
	},
	"wbs": {
		"supported_levels": SUPPORTED_WBS_LEVELS,
		"project_required": True,
		"code_required": True,
		"max_depth": 10,
	},
	"dependencies": {
		"supported_types": SUPPORTED_DEPENDENCY_TYPES,
		"predecessor_required": True,
		"successor_required": True,
		"circular_dependency_check": True,
	},
	"scheduling": {
		"supported_critical_path_methods": SUPPORTED_CRITICAL_PATH_METHODS,
		"supported_levelling_algorithms": SUPPORTED_LEVELLING_ALGORITHMS,
		"supported_calendar_types": SUPPORTED_CALENDAR_TYPES,
	},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"name_required": True,
		"scope_required": True,
		"human_approval_required_for_privileged_actions": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"wbs_circular_dependency_denied": True,
		"retroactive_schedule_edit_requires_change_request": True,
		"cross_tenant_schedule_access_denied": True,
		"critical_path_manipulation_denied": True,
	},
	"observability": {"event_stream": PPS_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_projects": True, "enable_wbs": True, "enable_gantt": True, "enable_critical_path": True, "enable_dependencies": True, "enable_levelling": True, "enable_milestones": True, "enable_calendars": True, "enable_agents": True},
	"theme": {"default_theme": "ppm_pps_control", "allow_tenant_overrides": True},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "scheduling_policy_required", "required_action": "attach_scheduling_policy"}},
	{"name": "project_status_supported", "condition": {"operation": "create_project", "status_supported": False}, "effect": {"decision": "deny", "reason": "project_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "project_owner_required", "condition": {"operation": "create_project", "owner_present": False}, "effect": {"decision": "deny", "reason": "project_owner_required", "required_action": "assign_project_owner"}},
	{"name": "project_start_date_required", "condition": {"operation": "create_project", "start_date_present": False}, "effect": {"decision": "deny", "reason": "project_start_date_required", "required_action": "set_project_start_date"}},
	{"name": "project_methodology_supported", "condition": {"operation": "create_project", "methodology_supported": False}, "effect": {"decision": "deny", "reason": "project_methodology_not_supported", "required_action": "select_supported_methodology"}},
	{"name": "project_evidence_required", "condition": {"operation": "create_project", "evidence_present": False}, "effect": {"decision": "deny", "reason": "project_evidence_required", "required_action": "attach_project_evidence"}},
	{"name": "task_type_supported", "condition": {"operation": "add_task", "task_type_supported": False}, "effect": {"decision": "deny", "reason": "task_type_not_supported", "required_action": "select_supported_task_type"}},
	{"name": "task_wbs_required", "condition": {"operation": "add_task", "wbs_element_present": False}, "effect": {"decision": "deny", "reason": "wbs_element_required", "required_action": "select_wbs_element"}},
	{"name": "task_duration_positive", "condition": {"operation": "add_task", "duration_positive": False}, "effect": {"decision": "deny", "reason": "task_duration_must_be_positive", "required_action": "set_positive_duration"}},
	{"name": "task_scheduling_mode_supported", "condition": {"operation": "add_task", "scheduling_mode_supported": False}, "effect": {"decision": "deny", "reason": "scheduling_mode_not_supported", "required_action": "select_supported_scheduling_mode"}},
	{"name": "task_constraint_type_supported", "condition": {"operation": "add_task", "constraint_type_supported": False}, "effect": {"decision": "deny", "reason": "constraint_type_not_supported", "required_action": "select_supported_constraint_type"}},
	{"name": "wbs_level_supported", "condition": {"operation": "add_wbs_element", "wbs_level_supported": False}, "effect": {"decision": "deny", "reason": "wbs_level_not_supported", "required_action": "select_supported_wbs_level"}},
	{"name": "wbs_project_required", "condition": {"operation": "add_wbs_element", "project_present": False}, "effect": {"decision": "deny", "reason": "project_required", "required_action": "select_project"}},
	{"name": "wbs_code_required", "condition": {"operation": "add_wbs_element", "code_present": False}, "effect": {"decision": "deny", "reason": "wbs_code_required", "required_action": "assign_wbs_code"}},
	{"name": "dependency_type_supported", "condition": {"operation": "link_dependency", "dependency_type_supported": False}, "effect": {"decision": "deny", "reason": "dependency_type_not_supported", "required_action": "select_supported_dependency_type"}},
	{"name": "dependency_predecessor_required", "condition": {"operation": "link_dependency", "predecessor_present": False}, "effect": {"decision": "deny", "reason": "predecessor_required", "required_action": "select_predecessor_task"}},
	{"name": "dependency_successor_required", "condition": {"operation": "link_dependency", "successor_present": False}, "effect": {"decision": "deny", "reason": "successor_required", "required_action": "select_successor_task"}},
	{"name": "wbs_circular_dependency_denied", "condition": {"circular_dependency": True}, "effect": {"decision": "deny", "reason": "wbs_circular_dependency_denied", "required_action": "resolve_circular_dependency"}},
	{"name": "critical_path_method_supported", "condition": {"operation": "calculate_critical_path", "cpm_method_supported": False}, "effect": {"decision": "deny", "reason": "critical_path_method_not_supported", "required_action": "select_supported_cpm_method"}},
	{"name": "critical_path_manipulation_denied", "condition": {"critical_path_manipulation": True}, "effect": {"decision": "deny", "reason": "critical_path_manipulation_denied", "required_action": "use_actual_schedule_data"}},
	{"name": "levelling_algorithm_supported", "condition": {"operation": "level_resources", "levelling_algorithm_supported": False}, "effect": {"decision": "deny", "reason": "levelling_algorithm_not_supported", "required_action": "select_supported_levelling_algorithm"}},
	{"name": "retroactive_edit_requires_change_request", "condition": {"operation": "edit_task", "retroactive": True}, "effect": {"decision": "deny", "reason": "retroactive_schedule_edit_requires_change_request", "required_action": "submit_change_request"}},
	{"name": "cross_tenant_schedule_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_schedule_access_denied", "required_action": "use_own_tenant_context"}},
	{"name": "schedule_batch_requires_bytewax", "condition": {"operation": "schedule_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_schedule_batch_to_bytewax"}},
	{"name": "agent_runtime_supported", "condition": {"operation": "register_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "agent_role_supported", "condition": {"operation": "register_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "agent_name_required", "condition": {"operation": "register_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "agent_name_required", "required_action": "name_agent"}},
	{"name": "agent_scope_required", "condition": {"operation": "register_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "agent_scope_required", "required_action": "bound_agent_scope"}},
	{"name": "privileged_agent_action_requires_human_approval", "condition": {"operation": "agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/ppm-pps/api/v1",
			"requires_theme": True,
			"view_module": "views.py",
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
