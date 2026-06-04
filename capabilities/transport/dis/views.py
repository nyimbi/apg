"""View models for generated Dispatch Operations screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import DispatchOperationsService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import DispatchOperationsService  # type: ignore


def dashboard_model(service: DispatchOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Dispatch Operations", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def dispatch_board_model(service: DispatchOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "dispatches": _tenant_items(service.dispatches, tenant_id), "load_plans": _tenant_items(service.load_plans, tenant_id)}


def driver_assignment_model(service: DispatchOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "assignments": _tenant_items(service.driver_assignments, tenant_id), "assignment_types": contract["configuration"]["driver_assignment"]["assignment_types"]}


def exception_console_model(service: DispatchOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "exceptions": _tenant_items(service.exceptions, tenant_id), "exception_types": contract["configuration"]["exceptions"]["supported_types"]}


def tracking_map_model(service: DispatchOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "tracking_updates": _tenant_items(service.tracking_updates, tenant_id)}


def agent_workbench_model(service: DispatchOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _tenant_items(service.agents, tenant_id)}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
