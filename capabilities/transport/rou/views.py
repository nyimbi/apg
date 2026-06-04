"""View models for generated Route Optimisation screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import RouteOptimisationService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import RouteOptimisationService  # type: ignore


def dashboard_model(service: RouteOptimisationService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Route Optimisation", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def route_console_model(service: RouteOptimisationService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "routes": _tenant_items(service.routes, tenant_id), "route_types": contract["configuration"]["routes"]["supported_types"]}


def optimisation_console_model(service: RouteOptimisationService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "objectives": contract["configuration"]["optimisation"]["objectives"], "default_objective": contract["configuration"]["optimisation"]["default_objective"]}


def traffic_console_model(service: RouteOptimisationService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "traffic_events": _tenant_items(service.traffic_events, tenant_id)}


def rerouting_console_model(service: RouteOptimisationService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "reroute_events": _tenant_items(service.reroute_events, tenant_id), "triggers": contract["configuration"]["rerouting"]["triggers"]}


def multimodal_console_model(service: RouteOptimisationService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "segments": _tenant_items(service.multimodal_segments, tenant_id)}


def agent_workbench_model(service: RouteOptimisationService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _tenant_items(service.agents, tenant_id)}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
