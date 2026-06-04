"""View models for generated Transport Scheduling screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import TransportSchedulingService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import TransportSchedulingService  # type: ignore


def dashboard_model(service: TransportSchedulingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Transport Scheduling", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def schedule_console_model(service: TransportSchedulingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "schedules": _tenant_items(service.schedules, tenant_id), "schedule_types": contract["configuration"]["schedules"]["supported_types"]}


def shift_console_model(service: TransportSchedulingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "shifts": _tenant_items(service.shifts, tenant_id), "shift_types": contract["configuration"]["shifts"]["types"]}


def charter_console_model(service: TransportSchedulingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "charters": _tenant_items(service.charters, tenant_id), "charter_types": contract["configuration"]["charters"]["types"]}


def conflict_console_model(service: TransportSchedulingService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "conflicts": _tenant_items(service.conflicts, tenant_id), "open_conflicts": service.list_open_conflicts(tenant_id)}


def vehicle_assignment_model(service: TransportSchedulingService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "assignments": _tenant_items(service.vehicle_assignments, tenant_id)}


def agent_workbench_model(service: TransportSchedulingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _tenant_items(service.agents, tenant_id)}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
