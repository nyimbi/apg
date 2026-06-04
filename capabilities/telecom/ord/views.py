"""View models for APG Order Management screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import TelecomOrdService


def dashboard_model(service: TelecomOrdService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Order Management", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def order_console_model(service: TelecomOrdService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "orders": _items(service.orders, tenant_id)}


def fallout_console_model(service: TelecomOrdService, tenant_id: str = "default") -> dict[str, Any]:
	open_fallouts = [f.to_dict() for f in service.fallouts.values() if f.tenant_id == tenant_id and f.status == "open"]
	return {"tenant_id": tenant_id, "open_fallouts": open_fallouts, "all_fallouts": _items(service.fallouts, tenant_id)}


def task_queue_model(service: TelecomOrdService, tenant_id: str = "default") -> dict[str, Any]:
	queued = [t.to_dict() for t in service.tasks.values() if t.tenant_id == tenant_id and t.status == "queued"]
	return {"tenant_id": tenant_id, "queued_tasks": queued, "all_tasks": _items(service.tasks, tenant_id)}


def portability_console_model(service: TelecomOrdService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "portability_requests": _items(service.portability_requests, tenant_id)}


def bulk_order_console_model(service: TelecomOrdService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "bulk_orders": _items(service.bulk_orders, tenant_id)}


def agent_workbench_model(service: TelecomOrdService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _items(service.agents, tenant_id)}


def _items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(store.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
