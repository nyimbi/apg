"""View models for generated Project Baseline Management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import ProjectBaselineService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import ProjectBaselineService  # type: ignore


def dashboard_model(service: ProjectBaselineService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the baseline management dashboard."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Project Baseline Management",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def baseline_list_model(service: ProjectBaselineService, tenant_id: str = "default", project_id: str | None = None) -> dict[str, Any]:
	"""View model for the baseline list screen."""
	return {
		"tenant_id": tenant_id,
		"project_id": project_id,
		"baselines": [
			v.to_dict() for v in sorted(service.baselines.values(), key=lambda x: x.id)
			if v.tenant_id == tenant_id and (project_id is None or v.project_id == project_id)
		],
	}


def change_request_queue_model(service: ProjectBaselineService, tenant_id: str = "default", baseline_id: str | None = None) -> dict[str, Any]:
	"""View model for the change request queue."""
	return {
		"tenant_id": tenant_id,
		"baseline_id": baseline_id,
		"change_requests": [
			v.to_dict() for v in sorted(service.change_requests.values(), key=lambda x: x.id)
			if v.tenant_id == tenant_id and (baseline_id is None or v.baseline_id == baseline_id)
		],
		"impact_assessments": _tenant_items(service.impact_assessments, tenant_id),
	}


def earned_value_dashboard_model(service: ProjectBaselineService, tenant_id: str = "default", baseline_id: str | None = None) -> dict[str, Any]:
	"""View model for earned value analysis."""
	snapshots = [
		v.to_dict() for v in sorted(service.ev_snapshots.values(), key=lambda x: x.id)
		if v.tenant_id == tenant_id and (baseline_id is None or v.baseline_id == baseline_id)
	]
	# Compute latest EV indicators
	latest_spi = latest_cpi = None
	if snapshots:
		last = snapshots[-1]
		latest_spi = round(last["ev"] / last["pv"], 3) if last["pv"] else None
		latest_cpi = round(last["ev"] / last["ac"], 3) if last["ac"] else None
	return {
		"tenant_id": tenant_id,
		"baseline_id": baseline_id,
		"ev_snapshots": snapshots,
		"latest_spi": latest_spi,
		"latest_cpi": latest_cpi,
	}


def variance_report_model(service: ProjectBaselineService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for variance reporting."""
	return {
		"tenant_id": tenant_id,
		"variance_reports": _tenant_items(service.variance_reports, tenant_id),
	}


def approval_console_model(service: ProjectBaselineService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for baseline approval console."""
	return {
		"tenant_id": tenant_id,
		"approvals": _tenant_items(service.approvals, tenant_id),
	}


def agent_workbench_model(service: ProjectBaselineService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the baseline agent workbench."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [v.to_dict() for v in service.agents.values() if v.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [v.to_dict() for v in sorted(items.values(), key=lambda x: x.id) if v.tenant_id == tenant_id]
