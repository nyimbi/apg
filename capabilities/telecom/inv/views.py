"""View models for APG Network Inventory screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import TelecomInvService


def dashboard_model(service: TelecomInvService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Network Inventory", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def asset_console_model(service: TelecomInvService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "assets": _items(service.assets, tenant_id), "sites": _items(service.sites, tenant_id)}


def circuit_console_model(service: TelecomInvService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "circuits": _items(service.circuits, tenant_id)}


def ipam_console_model(service: TelecomInvService, tenant_id: str = "default") -> dict[str, Any]:
	allocated = [b.to_dict() for b in service.ip_blocks.values() if b.tenant_id == tenant_id and b.allocated_to is not None]
	free = [b.to_dict() for b in service.ip_blocks.values() if b.tenant_id == tenant_id and b.allocated_to is None]
	return {"tenant_id": tenant_id, "allocated_blocks": allocated, "free_blocks": free, "all_blocks": _items(service.ip_blocks, tenant_id)}


def topology_viewer_model(service: TelecomInvService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "topologies": _items(service.topologies, tenant_id)}


def reconciliation_console_model(service: TelecomInvService, tenant_id: str = "default") -> dict[str, Any]:
	open_discrepancies = [r.to_dict() for r in service.reconciliations.values() if r.tenant_id == tenant_id and r.status == "open"]
	return {"tenant_id": tenant_id, "open_discrepancies": open_discrepancies, "all_reconciliations": _items(service.reconciliations, tenant_id)}


def agent_workbench_model(service: TelecomInvService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _items(service.agents, tenant_id)}


def _items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(store.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
