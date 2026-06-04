"""View models for generated Resource Management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import ResourceManagementService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import ResourceManagementService  # type: ignore


def dashboard_model(service: ResourceManagementService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the resource management dashboard."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Resource Management",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def resource_pool_model(service: ResourceManagementService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the resource pool list."""
	return {
		"tenant_id": tenant_id,
		"resources": _tenant_items(service.resources, tenant_id),
	}


def skill_catalog_model(service: ResourceManagementService, tenant_id: str = "default", resource_id: str | None = None) -> dict[str, Any]:
	"""View model for skill catalog and proficiency view."""
	return {
		"tenant_id": tenant_id,
		"resource_id": resource_id,
		"skills": [
			v.to_dict() for v in sorted(service.skills.values(), key=lambda x: x.skill_name)
			if v.tenant_id == tenant_id and (resource_id is None or v.resource_id == resource_id)
		],
	}


def allocation_console_model(service: ResourceManagementService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the resource allocation console."""
	return {
		"tenant_id": tenant_id,
		"allocations": _tenant_items(service.allocations, tenant_id),
	}


def utilisation_tracker_model(service: ResourceManagementService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for utilisation tracking with band summary."""
	snapshots = [v.to_dict() for v in service.utilisation_snapshots.values() if v.tenant_id == tenant_id]
	band_counts: dict[str, int] = {}
	for s in snapshots:
		band_counts[s["utilisation_band"]] = band_counts.get(s["utilisation_band"], 0) + 1
	return {
		"tenant_id": tenant_id,
		"utilisation_snapshots": sorted(snapshots, key=lambda x: x["id"]),
		"band_summary": band_counts,
	}


def capacity_planning_model(service: ResourceManagementService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for capacity plans and demand forecasts."""
	return {
		"tenant_id": tenant_id,
		"capacity_plans": _tenant_items(service.capacity_plans, tenant_id),
		"demand_forecasts": _tenant_items(service.demand_forecasts, tenant_id),
	}


def availability_calendar_model(service: ResourceManagementService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the resource availability calendar."""
	return {
		"tenant_id": tenant_id,
		"leave_records": _tenant_items(service.leave_records, tenant_id),
		"allocations": _tenant_items(service.allocations, tenant_id),
	}


def cost_rate_model(service: ResourceManagementService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for cost rate table."""
	return {
		"tenant_id": tenant_id,
		"cost_rates": _tenant_items(service.cost_rates, tenant_id),
	}


def agent_workbench_model(service: ResourceManagementService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the resource agent workbench."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [v.to_dict() for v in service.agents.values() if v.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [v.to_dict() for v in sorted(items.values(), key=lambda x: x.id) if v.tenant_id == tenant_id]
