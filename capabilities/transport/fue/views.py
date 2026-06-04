"""View models for generated Fuel Management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import FuelManagementService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import FuelManagementService  # type: ignore


def dashboard_model(service: FuelManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Fuel Management", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def procurement_console_model(service: FuelManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "procurements": _tenant_items(service.procurements, tenant_id)}


def transaction_console_model(service: FuelManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "transactions": _tenant_items(service.transactions, tenant_id), "transaction_types": contract["configuration"]["transactions"]["types"]}


def fuel_card_console_model(service: FuelManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "fuel_cards": _tenant_items(service.fuel_cards, tenant_id), "providers": contract["configuration"]["fuel_cards"]["providers"]}


def carbon_console_model(service: FuelManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "carbon_records": _tenant_items(service.carbon_records, tenant_id), "standards": contract["configuration"]["carbon"]["standards"]}


def storage_console_model(service: FuelManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "storage_tanks": _tenant_items(service.storage_tanks, tenant_id)}


def agent_workbench_model(service: FuelManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _tenant_items(service.agents, tenant_id)}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
