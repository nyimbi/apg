"""View models for generated Vehicle Maintenance screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import VehicleMaintenanceService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import VehicleMaintenanceService  # type: ignore


def dashboard_model(service: VehicleMaintenanceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Vehicle Maintenance", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def job_console_model(service: VehicleMaintenanceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "jobs": _tenant_items(service.jobs, tenant_id), "maintenance_types": contract["configuration"]["maintenance"]["supported_types"], "job_statuses": contract["configuration"]["jobs"]["supported_statuses"]}


def workshop_console_model(service: VehicleMaintenanceService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "allocations": _tenant_items(service.workshop_allocations, tenant_id)}


def parts_console_model(service: VehicleMaintenanceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "parts_orders": _tenant_items(service.parts_orders, tenant_id), "categories": contract["configuration"]["parts"]["categories"]}


def inspection_console_model(service: VehicleMaintenanceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "inspections": _tenant_items(service.inspections, tenant_id), "inspection_types": contract["configuration"]["inspections"]["types"]}


def schedule_console_model(service: VehicleMaintenanceService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "schedules": _tenant_items(service.schedules, tenant_id)}


def agent_workbench_model(service: VehicleMaintenanceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _tenant_items(service.agents, tenant_id)}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
