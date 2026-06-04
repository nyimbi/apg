"""View models for generated Cargo Management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import CargoManagementService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import CargoManagementService  # type: ignore


def dashboard_model(service: CargoManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Cargo Management", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def booking_console_model(service: CargoManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "bookings": _tenant_items(service.bookings, tenant_id), "supported_cargo_types": get_capability_contract(tenant_id)["configuration"]["cargo_types"]["supported_types"]}


def manifest_console_model(service: CargoManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "manifests": _tenant_items(service.manifests, tenant_id)}


def dg_console_model(service: CargoManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "declarations": _tenant_items(service.dg_declarations, tenant_id), "dg_classes": contract["configuration"]["cargo_types"]["dg_classes"]}


def tracking_board_model(service: CargoManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "tracking_events": _tenant_items(service.tracking_events, tenant_id)}


def revenue_console_model(service: CargoManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "revenue_records": _tenant_items(service.revenue_records, tenant_id)}


def agent_workbench_model(service: CargoManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _tenant_items(service.agents, tenant_id)}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
