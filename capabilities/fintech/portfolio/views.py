"""View models for APG Portfolio Management."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import PortfolioManagementService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import PortfolioManagementService  # type: ignore


async def dashboard_model(service: PortfolioManagementService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = await service.dashboard_summary()
	return {
		"title": "Portfolio Management",
		"tenant_id": tenant_id,
		"summary": summary,
		"cards": [
			{"label": "Portfolios", "value": summary["portfolio_count"], "icon": "briefcase-business"},
			{"label": "Holdings", "value": summary["holding_count"], "icon": "layers"},
			{"label": "Allocations", "value": summary["allocation_count"], "icon": "pie-chart"},
			{"label": "Valuations", "value": summary["valuation_count"], "icon": "line-chart"},
			{"label": "Risk", "value": summary["risk_count"], "icon": "shield-alert"},
			{"label": "Agents", "value": len([item for item in service.evidence.values() if item.tenant_id == tenant_id and item.kind == "agent"]), "icon": "bot"},
		],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def portfolio_console_model(service: PortfolioManagementService, tenant_id: str) -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"portfolios": _items(service.portfolios, tenant_id),
		"holdings": _items(service.holdings, tenant_id),
		"allocations": _items(service.allocations, tenant_id),
		"valuations": _items(service.valuations, tenant_id),
		"benchmarks": _items(service.benchmarks, tenant_id),
		"risk": _items(service.risk, tenant_id),
		"attribution": _items(service.attribution, tenant_id),
		"cash": _items(service.cash, tenant_id),
		"corporate_actions": _items(service.corporate_actions, tenant_id),
		"compliance": _items(service.compliance, tenant_id),
		"reviews": _items(service.reviews, tenant_id),
		"agents": [item.to_dict() for item in sorted(service.evidence.values(), key=lambda item: item.id) if item.tenant_id == tenant_id and item.kind == "agent"],
	}


def route_models(service: PortfolioManagementService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	console = portfolio_console_model(service, tenant_id)
	return {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"], "data": console if route["name"] != "dashboard" else dashboard_model(service, tenant_id)} for route in contract["ui"]["routes"]}


def _items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda item: item.id) if item.tenant_id == tenant_id]
