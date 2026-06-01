"""View models for APG Buy Now Pay Later."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import BNPLService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import BNPLService  # type: ignore


def dashboard_model(service: BNPLService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = service.dashboard_summary(tenant_id)
	return {
		"title": "Buy Now Pay Later",
		"tenant_id": tenant_id,
		"summary": summary,
		"cards": [
			{"label": "Programs", "value": summary["program_count"], "icon": "badge-percent"},
			{"label": "Consumers", "value": summary["consumer_count"], "icon": "user-check"},
			{"label": "Merchants", "value": summary["merchant_count"], "icon": "store"},
			{"label": "Checkouts", "value": summary["checkout_count"], "icon": "shopping-cart"},
			{"label": "Plans", "value": summary["plan_count"], "icon": "calendar-clock"},
			{"label": "Settlements", "value": summary["settlement_count"], "icon": "landmark"},
		],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def bnpl_console_model(service: BNPLService, tenant_id: str) -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"programs": [item.to_dict() for item in sorted(service.programs.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"consumers": [item.to_dict() for item in sorted(service.consumers.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"merchants": [item.to_dict() for item in sorted(service.merchants.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"checkouts": service.list_checkouts(tenant_id),
		"affordability": [item.to_dict() for item in sorted(service.affordability.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"plans": service.list_plans(tenant_id),
		"installments": [item.to_dict() for item in sorted(service.installments.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"settlements": service.list_settlements(tenant_id),
		"disputes": [item.to_dict() for item in sorted(service.disputes.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"agents": [item.to_dict() for item in sorted(service.evidence.values(), key=lambda item: item.id) if item.tenant_id == tenant_id and item.kind == "agent"],
	}


def route_models(service: BNPLService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	console = bnpl_console_model(service, tenant_id)
	return {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"], "data": console if route["name"] != "dashboard" else dashboard_model(service, tenant_id)} for route in contract["ui"]["routes"]}
