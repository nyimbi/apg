"""View models for APG Banking APIs."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import BankingAPIService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import BankingAPIService  # type: ignore


def dashboard_model(service: BankingAPIService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = service.dashboard_summary(tenant_id)
	return {
		"title": "Banking APIs",
		"tenant_id": tenant_id,
		"summary": summary,
		"cards": [
			{"label": "Products", "value": summary["product_count"], "icon": "box"},
			{"label": "Developers", "value": summary["developer_count"], "icon": "building-2"},
			{"label": "Applications", "value": summary["application_count"], "icon": "app-window"},
			{"label": "Clients", "value": summary["client_count"], "icon": "key-round"},
			{"label": "API Calls", "value": summary["call_count"], "icon": "activity"},
			{"label": "Incidents", "value": summary["incident_count"], "icon": "sirens"},
		],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def apis_console_model(service: BankingAPIService, tenant_id: str) -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"products": [item.to_dict() for item in sorted(service.products.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"developers": [item.to_dict() for item in sorted(service.developers.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"applications": [item.to_dict() for item in sorted(service.applications.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"consents": [item.to_dict() for item in sorted(service.consents.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"clients": [item.to_dict() for item in sorted(service.clients.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"endpoints": [item.to_dict() for item in sorted(service.endpoints.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"webhooks": [item.to_dict() for item in sorted(service.webhooks.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"calls": service.list_calls(tenant_id),
		"rate_limits": [item.to_dict() for item in sorted(service.rate_limits.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"incidents": [item.to_dict() for item in sorted(service.incidents.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"agents": [item.to_dict() for item in sorted(service.evidence.values(), key=lambda item: item.id) if item.tenant_id == tenant_id and item.kind == "agent"],
	}


def route_models(service: BankingAPIService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	console = apis_console_model(service, tenant_id)
	return {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"], "data": console if route["name"] != "dashboard" else dashboard_model(service, tenant_id)} for route in contract["ui"]["routes"]}
