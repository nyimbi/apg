"""View models for APG Embedded Finance."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import EmbeddedFinanceService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import EmbeddedFinanceService  # type: ignore


def dashboard_model(service: EmbeddedFinanceService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = service.dashboard_summary(tenant_id)
	return {
		"title": "Embedded Finance",
		"tenant_id": tenant_id,
		"summary": summary,
		"cards": [
			{"label": "Programs", "value": summary["program_count"], "icon": "handshake"},
			{"label": "Applications", "value": summary["application_count"], "icon": "app-window"},
			{"label": "Placements", "value": summary["placement_count"], "icon": "layout-template"},
			{"label": "Payments", "value": summary["payment_count"], "icon": "credit-card"},
			{"label": "Settlements", "value": summary["settlement_count"], "icon": "receipt"},
			{"label": "Agents", "value": len([item for item in service.evidence.values() if item.tenant_id == tenant_id and item.kind == "agent"]), "icon": "bot"},
		],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def embedded_console_model(service: EmbeddedFinanceService, tenant_id: str) -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"programs": _items(service.programs, tenant_id),
		"applications": _items(service.applications, tenant_id),
		"placements": _items(service.placements, tenant_id),
		"consents": _items(service.consents, tenant_id),
		"accounts": _items(service.accounts, tenant_id),
		"payments": _items(service.payments, tenant_id),
		"cards": _items(service.cards, tenant_id),
		"lending": _items(service.lending, tenant_id),
		"settlements": _items(service.settlements, tenant_id),
		"revenue_share": _items(service.revenue_shares, tenant_id),
		"agents": [item.to_dict() for item in sorted(service.evidence.values(), key=lambda item: item.id) if item.tenant_id == tenant_id and item.kind == "agent"],
	}


def route_models(service: EmbeddedFinanceService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	console = embedded_console_model(service, tenant_id)
	return {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"], "data": console if route["name"] != "dashboard" else dashboard_model(service, tenant_id)} for route in contract["ui"]["routes"]}


def _items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda item: item.id) if item.tenant_id == tenant_id]
