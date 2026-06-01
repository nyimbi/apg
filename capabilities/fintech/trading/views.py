"""View models for APG Algorithmic Trading."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import AlgorithmicTradingService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import AlgorithmicTradingService  # type: ignore


def dashboard_model(service: AlgorithmicTradingService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = service.dashboard_summary(tenant_id)
	return {
		"title": "Algorithmic Trading",
		"tenant_id": tenant_id,
		"summary": summary,
		"cards": [
			{"label": "Strategies", "value": summary["strategy_count"], "icon": "workflow"},
			{"label": "Signals", "value": summary["signal_count"], "icon": "radio-tower"},
			{"label": "Backtests", "value": summary["backtest_count"], "icon": "history"},
			{"label": "Orders", "value": summary["order_count"], "icon": "send-horizontal"},
			{"label": "Executions", "value": summary["execution_count"], "icon": "receipt-text"},
			{"label": "Agents", "value": len([item for item in service.evidence.values() if item.tenant_id == tenant_id and item.kind == "agent"]), "icon": "bot"},
		],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def trading_console_model(service: AlgorithmicTradingService, tenant_id: str) -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"strategies": _items(service.strategies, tenant_id),
		"signals": _items(service.signals, tenant_id),
		"backtests": _items(service.backtests, tenant_id),
		"risk_limits": _items(service.risk_limits, tenant_id),
		"orders": _items(service.orders, tenant_id),
		"executions": _items(service.executions, tenant_id),
		"positions": _items(service.positions, tenant_id),
		"surveillance": _items(service.surveillance, tenant_id),
		"reviews": _items(service.reviews, tenant_id),
		"agents": [item.to_dict() for item in sorted(service.evidence.values(), key=lambda item: item.id) if item.tenant_id == tenant_id and item.kind == "agent"],
	}


def route_models(service: AlgorithmicTradingService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	console = trading_console_model(service, tenant_id)
	return {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"], "data": console if route["name"] != "dashboard" else dashboard_model(service, tenant_id)} for route in contract["ui"]["routes"]}


def _items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda item: item.id) if item.tenant_id == tenant_id]
