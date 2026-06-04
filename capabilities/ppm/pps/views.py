"""View models for generated Project Planning & Scheduling screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import ProjectPlanningService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import ProjectPlanningService  # type: ignore


def dashboard_model(service: ProjectPlanningService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the scheduling dashboard."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Project Planning & Scheduling",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def project_list_model(service: ProjectPlanningService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the project list."""
	return {
		"tenant_id": tenant_id,
		"projects": _tenant_items(service.projects, tenant_id),
	}


def wbs_editor_model(service: ProjectPlanningService, tenant_id: str = "default", project_id: str | None = None) -> dict[str, Any]:
	"""View model for the WBS editor."""
	elements = [
		v.to_dict() for v in sorted(service.wbs_elements.values(), key=lambda x: x.code)
		if v.tenant_id == tenant_id and (project_id is None or v.project_id == project_id)
	]
	return {
		"tenant_id": tenant_id,
		"project_id": project_id,
		"wbs_elements": elements,
		"tasks": [
			v.to_dict() for v in sorted(service.tasks.values(), key=lambda x: x.id)
			if v.tenant_id == tenant_id and (project_id is None or v.project_id == project_id)
		],
	}


def gantt_chart_model(service: ProjectPlanningService, tenant_id: str = "default", project_id: str | None = None) -> dict[str, Any]:
	"""View model for Gantt chart rendering."""
	tasks = [
		v.to_dict() for v in sorted(service.tasks.values(), key=lambda x: x.start_date)
		if v.tenant_id == tenant_id and (project_id is None or v.project_id == project_id)
	]
	dependencies = [
		v.to_dict() for v in service.dependencies.values()
		if v.tenant_id == tenant_id
	]
	return {
		"tenant_id": tenant_id,
		"project_id": project_id,
		"tasks": tasks,
		"dependencies": dependencies,
	}


def critical_path_model(service: ProjectPlanningService, tenant_id: str = "default", project_id: str | None = None) -> dict[str, Any]:
	"""View model for critical path analysis."""
	results = [
		v.to_dict() for v in sorted(service.critical_path_results.values(), key=lambda x: x.id)
		if v.tenant_id == tenant_id and (project_id is None or v.project_id == project_id)
	]
	return {
		"tenant_id": tenant_id,
		"project_id": project_id,
		"critical_path_results": results,
		"latest_result": results[-1] if results else None,
	}


def resource_levelling_model(service: ProjectPlanningService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for resource levelling results."""
	return {
		"tenant_id": tenant_id,
		"levelling_results": _tenant_items(service.levelling_results, tenant_id),
	}


def milestone_tracker_model(service: ProjectPlanningService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for milestone tracking — tasks of type 'milestone'."""
	milestones = [
		v.to_dict() for v in sorted(service.tasks.values(), key=lambda x: x.start_date)
		if v.tenant_id == tenant_id and v.task_type == "milestone"
	]
	return {"tenant_id": tenant_id, "milestones": milestones}


def agent_workbench_model(service: ProjectPlanningService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the scheduling agent workbench."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [v.to_dict() for v in service.agents.values() if v.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [v.to_dict() for v in sorted(items.values(), key=lambda x: x.id) if v.tenant_id == tenant_id]
