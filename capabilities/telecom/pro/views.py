"""View models for APG Service Provisioning screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import TelecomProService


def dashboard_model(service: TelecomProService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Service Provisioning", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def workflow_console_model(service: TelecomProService, tenant_id: str = "default") -> dict[str, Any]:
	failed = [w.to_dict() for w in service.workflows.values() if w.tenant_id == tenant_id and w.status == "failed"]
	return {"tenant_id": tenant_id, "failed_workflows": failed, "all_workflows": _items(service.workflows, tenant_id)}


def resource_console_model(service: TelecomProService, tenant_id: str = "default") -> dict[str, Any]:
	active = [r.to_dict() for r in service.resource_reservations.values() if r.tenant_id == tenant_id and not r.released]
	return {"tenant_id": tenant_id, "active_reservations": active, "all_reservations": _items(service.resource_reservations, tenant_id)}


def config_push_console_model(service: TelecomProService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "config_pushes": _items(service.config_pushes, tenant_id)}


def activation_console_model(service: TelecomProService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "activations": _items(service.activations, tenant_id)}


def rollback_console_model(service: TelecomProService, tenant_id: str = "default") -> dict[str, Any]:
	in_progress = [r.to_dict() for r in service.rollbacks.values() if r.tenant_id == tenant_id and r.status == "in_progress"]
	return {"tenant_id": tenant_id, "in_progress_rollbacks": in_progress, "all_rollbacks": _items(service.rollbacks, tenant_id)}


def agent_workbench_model(service: TelecomProService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _items(service.agents, tenant_id)}


def _items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(store.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
