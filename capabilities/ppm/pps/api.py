"""Process-local API helpers for APG Project Planning & Scheduling (pps)."""

from __future__ import annotations

try:
	from .service import ProjectPlanningService
except ImportError:  # pragma: no cover
	import sys as _sys, pathlib as _pl
	_here = str(_pl.Path(__file__).parent)
	if _here not in _sys.path:
		_sys.path.insert(0, _here)
	from service import ProjectPlanningService  # type: ignore

_SERVICE = ProjectPlanningService()


def service() -> ProjectPlanningService:
	return _SERVICE


def create_project(payload: dict):
	return _SERVICE.create_project(
		payload["project_id"], payload.get("tenant_id", "default"),
		payload["name"], payload.get("status", "planned"),
		payload.get("methodology", "waterfall"), payload["owner_id"],
		payload["start_date"], payload.get("end_date", ""),
		payload["evidence_reference"], payload.get("policy_attached", True),
	)


def add_wbs_element(payload: dict):
	return _SERVICE.add_wbs_element(
		payload["wbs_id"], payload.get("tenant_id", "default"),
		payload["project_id"], payload.get("parent_id"),
		payload["level"], payload["code"], payload["name"],
		payload.get("description", ""),
	)


def add_task(payload: dict):
	return _SERVICE.add_task(
		payload["task_id"], payload.get("tenant_id", "default"),
		payload["project_id"], payload["wbs_element_id"],
		payload.get("task_type", "work_package"), payload.get("status", "not_started"),
		payload["name"], float(payload["duration_days"]),
		payload.get("scheduling_mode", "fixed_duration"),
		payload.get("constraint_type", "as_soon_as_possible"),
		payload.get("progress_method", "percent_complete"),
		float(payload.get("progress_pct", 0.0)),
		payload.get("start_date", ""), payload.get("end_date", ""),
	)


def update_task_status(payload: dict):
	return _SERVICE.update_task_status(
		payload["task_id"], payload.get("tenant_id", "default"),
		payload["status"], float(payload.get("progress_pct", 0.0)),
	)


def link_dependency(payload: dict):
	return _SERVICE.link_dependency(
		payload["dep_id"], payload.get("tenant_id", "default"),
		payload["predecessor_id"], payload["successor_id"],
		payload.get("dependency_type", "finish_to_start"),
		float(payload.get("lag_days", 0.0)),
	)


def calculate_critical_path(payload: dict):
	return _SERVICE.calculate_critical_path(
		payload["result_id"], payload.get("tenant_id", "default"),
		payload["project_id"], payload.get("method", "cpm"),
		payload.get("critical_task_ids", "[]"),
		float(payload.get("total_float_days", 0.0)),
		float(payload.get("project_duration_days", 0.0)),
		payload.get("calculated_at", ""),
	)


def level_resources(payload: dict):
	return _SERVICE.level_resources(
		payload["result_id"], payload.get("tenant_id", "default"),
		payload["project_id"], payload.get("algorithm", "priority_based"),
		int(payload.get("over_allocations_resolved", 0)),
		float(payload.get("schedule_extension_days", 0.0)),
		payload.get("levelled_at", ""),
	)


def create_calendar(payload: dict):
	return _SERVICE.create_calendar(
		payload["calendar_id"], payload.get("tenant_id", "default"),
		payload["name"], payload.get("calendar_type", "standard_5x8"),
		float(payload.get("working_hours_per_day", 8.0)),
		payload.get("working_days", '["monday","tuesday","wednesday","thursday","friday"]'),
	)


def register_agent(payload: dict):
	return _SERVICE.register_agent(
		payload["agent_id"], payload.get("tenant_id", "default"),
		payload["name"], payload["runtime"], payload["role"],
		payload.get("scope", "project scheduling operations"),
	)


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(
		payload.get("tenant_id", "default"),
		payload.get("privileged_scope", False),
		payload.get("human_approval_recorded", False),
	)


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(
		payload.get("tenant_id", "default"),
		payload["item_count"],
		payload.get("event_stream", "bytewax"),
	)


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
