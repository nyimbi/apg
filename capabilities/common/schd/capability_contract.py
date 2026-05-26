"""Executable capability contract for APG Scheduling and Job Orchestration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"schedules": {
		"schedule_owner_required": True,
		"timezone_required": True,
		"calendar_policy_required": True,
		"max_active_schedules": 10000
	},
	"jobs": {
		"retry_policy_required": True,
		"critical_job_monitoring_required": True,
		"dead_letter_queue_enabled": True,
		"max_runtime_minutes": 720
	},
	"workers": {
		"worker_pool_required": True,
		"health_check_required": True,
		"capacity_limits_required": True,
		"autoscaling_supported": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_job_runs": True,
		"external_job_approval_required": True,
		"manual_run_reason_required": True
	},
	"ui": {
		"enable_schedule_console": True,
		"enable_job_monitor": True,
		"enable_worker_dashboard": True,
		"enable_calendar_manager": True
	},
	"theme": {
		"default_theme": "schd_scheduler_ops",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "schedules", "jobs", "workers", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["schedules", "jobs", "workers", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All scheduling operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "schedule_requires_owner", "description": "Schedules require an accountable owner.", "condition": {"operation": "create_schedule", "schedule_owner_assigned": False}, "effect": {"decision": "deny", "reason": "schedule_owner_required", "required_action": "assign_schedule_owner"}},
	{"name": "timezone_required", "description": "Schedules require an explicit timezone.", "condition": {"operation": "create_schedule", "timezone_present": False}, "effect": {"decision": "deny", "reason": "timezone_required", "required_action": "set_timezone"}},
	{"name": "critical_job_requires_monitoring", "description": "Critical jobs require monitoring.", "condition": {"job_criticality": "critical", "monitoring_attached": False}, "effect": {"decision": "deny", "reason": "critical_job_monitoring_required", "required_action": "attach_monitoring"}},
	{"name": "external_job_requires_approval", "description": "External jobs require approval.", "condition": {"external_job": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "external_job_approval_required", "required_action": "record_external_job_approval"}},
	{"name": "long_running_job_requires_review", "description": "Long-running jobs require review.", "condition": {"expected_runtime_minutes_gt": 720, "runtime_review_recorded": False}, "effect": {"decision": "require_review", "reason": "long_running_job_review_required", "required_action": "review_job_runtime"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/schd/dashboard", "component": "SCHDDashboard", "permission": "schd:view", "nav_group": "Overview"},
	{"name": "schedules", "path": "/schd/schedules", "component": "ScheduleConsole", "permission": "schd:schedule", "nav_group": "Schedules"},
	{"name": "jobs", "path": "/schd/jobs", "component": "JobLibrary", "permission": "schd:run_jobs", "nav_group": "Jobs"},
	{"name": "runs", "path": "/schd/runs", "component": "RunMonitor", "permission": "schd:view", "nav_group": "Runtime"},
	{"name": "workers", "path": "/schd/workers", "component": "WorkerDashboard", "permission": "schd:manage_workers", "nav_group": "Workers"},
	{"name": "calendars", "path": "/schd/calendars", "component": "CalendarManager", "permission": "schd:schedule", "nav_group": "Schedules"},
	{"name": "analytics", "path": "/schd/analytics", "component": "SchedulerAnalytics", "permission": "schd:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/schd/settings", "component": "SCHDSettings", "permission": "schd:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "schd_scheduler_ops",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#D69E2E",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"schedule_calendar": {"icon": "calendar-clock", "status_indicator": "schedule-pill", "risk_style": "calendar-band"},
		"job_run_table": {"visual": "run-list", "highlight": "runtime-chip"},
		"worker_pool": {"visual": "capacity-grid", "status_style": "health-chip"},
		"retry_panel": {"visual": "retry-ladder", "status_style": "backoff-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable SCHD capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "schd",
		"display_name": "Scheduling and Job Orchestration",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "flask_appbuilder",
			"view_module": "views.py",
			"api_prefix": "/schd/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default SCHD governance rules."""
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


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


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
