"""View models for generated Cryptocurrency Services screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import CryptocurrencyServicesService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import CryptocurrencyServicesService  # type: ignore


def dashboard_model(service: CryptocurrencyServicesService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Cryptocurrency Services", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def crypto_console_model(service: CryptocurrencyServicesService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "assets": _tenant_items(service.assets, tenant_id), "custody_accounts": _tenant_items(service.accounts, tenant_id), "balances": _tenant_items(service.balances, tenant_id), "orders": _tenant_items(service.orders, tenant_id), "trades": _tenant_items(service.trades, tenant_id), "transfers": _tenant_items(service.transfers, tenant_id), "screenings": _tenant_items(service.screenings, tenant_id), "prices": _tenant_items(service.prices, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: CryptocurrencyServicesService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id]}


def _tenant_items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]
