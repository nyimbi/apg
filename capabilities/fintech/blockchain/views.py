"""View models for generated Blockchain Services screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import BlockchainServicesService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import BlockchainServicesService  # type: ignore


def dashboard_model(service: BlockchainServicesService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Blockchain Services", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def blockchain_console_model(service: BlockchainServicesService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "networks": _tenant_items(service.networks, tenant_id), "wallets": _tenant_items(service.wallets, tenant_id), "contracts": _tenant_items(service.contracts, tenant_id), "transactions": _tenant_items(service.transactions, tenant_id), "anchors": _tenant_items(service.anchors, tenant_id), "oracles": _tenant_items(service.oracles, tenant_id), "nodes": _tenant_items(service.nodes, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: BlockchainServicesService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id]}


def _tenant_items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]
