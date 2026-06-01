"""View models for APG Wealth Management."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import WealthManagementService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import WealthManagementService  # type: ignore


def dashboard_model(service: WealthManagementService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = service.dashboard_summary(tenant_id)
	return {
		"title": "Wealth Management",
		"tenant_id": tenant_id,
		"summary": summary,
		"cards": [
			{"label": "Clients", "value": summary["client_count"], "icon": "users"},
			{"label": "Portfolios", "value": summary["portfolio_count"], "icon": "pie-chart"},
			{"label": "Mandates", "value": summary["mandate_count"], "icon": "file-signature"},
			{"label": "Orders", "value": summary["order_count"], "icon": "list-ordered"},
			{"label": "Performance", "value": summary["performance_count"], "icon": "line-chart"},
			{"label": "Agents", "value": len([item for item in service.evidence.values() if item.tenant_id == tenant_id and item.kind == "agent"]), "icon": "bot"},
		],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def wealth_console_model(service: WealthManagementService, tenant_id: str) -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"clients": _items(service.clients, tenant_id),
		"suitability": _items(service.suitability, tenant_id),
		"portfolios": _items(service.portfolios, tenant_id),
		"mandates": _items(service.mandates, tenant_id),
		"rebalances": _items(service.rebalances, tenant_id),
		"orders": _items(service.orders, tenant_id),
		"performance": _items(service.performance, tenant_id),
		"fees": _items(service.fees, tenant_id),
		"agents": [item.to_dict() for item in sorted(service.evidence.values(), key=lambda item: item.id) if item.tenant_id == tenant_id and item.kind == "agent"],
	}


def route_models(service: WealthManagementService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	console = wealth_console_model(service, tenant_id)
	return {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"], "data": console if route["name"] != "dashboard" else dashboard_model(service, tenant_id)} for route in contract["ui"]["routes"]}


def _items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda item: item.id) if item.tenant_id == tenant_id]
