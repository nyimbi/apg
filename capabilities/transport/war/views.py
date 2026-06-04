"""View models for generated Warehouse Operations screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import WarehouseOperationsService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import WarehouseOperationsService  # type: ignore


def dashboard_model(service: WarehouseOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Warehouse Operations", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def receiving_console_model(service: WarehouseOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "receipts": _tenant_items(service.receipts, tenant_id), "receipt_methods": contract["configuration"]["receiving"]["methods"]}


def putaway_console_model(service: WarehouseOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "tasks": _tenant_items(service.putaway_tasks, tenant_id), "strategies": contract["configuration"]["putaway"]["strategies"]}


def picking_console_model(service: WarehouseOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "pick_tasks": _tenant_items(service.pick_tasks, tenant_id), "open_tasks": service.list_open_pick_tasks(tenant_id), "pick_methods": contract["configuration"]["picking"]["methods"]}


def packing_console_model(service: WarehouseOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "pack_tasks": _tenant_items(service.pack_tasks, tenant_id), "pack_types": contract["configuration"]["packing"]["pack_types"]}


def cycle_count_console_model(service: WarehouseOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "cycle_counts": _tenant_items(service.cycle_counts, tenant_id), "count_types": contract["configuration"]["cycle_counting"]["types"]}


def dock_door_console_model(service: WarehouseOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "dock_doors": _tenant_items(service.dock_doors, tenant_id), "statuses": contract["configuration"]["dock_doors"]["statuses"]}


def agent_workbench_model(service: WarehouseOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _tenant_items(service.agents, tenant_id)}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
