"""View models for APG Mobile Banking."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import MobileBankingService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import MobileBankingService  # type: ignore


def dashboard_model(service: MobileBankingService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = service.dashboard_summary(tenant_id)
	return {
		"title": "Mobile Banking",
		"tenant_id": tenant_id,
		"summary": summary,
		"cards": [
			{"label": "Customers", "value": summary["customer_count"], "icon": "user-check"},
			{"label": "Devices", "value": summary["device_count"], "icon": "shield-check"},
			{"label": "Auth Factors", "value": summary["auth_factor_count"], "icon": "key-round"},
			{"label": "Payments", "value": summary["payment_count"], "icon": "send"},
			{"label": "Service", "value": summary["service_request_count"], "icon": "life-buoy"},
			{"label": "Fraud", "value": summary["fraud_event_count"], "icon": "shield-alert"},
		],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def mobile_console_model(service: MobileBankingService, tenant_id: str) -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"programs": [item.to_dict() for item in sorted(service.programs.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"customers": [item.to_dict() for item in sorted(service.customers.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"devices": service.list_devices(tenant_id),
		"auth_factors": [item.to_dict() for item in sorted(service.auth_factors.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"account_links": [item.to_dict() for item in sorted(service.account_links.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"payments": service.list_payments(tenant_id),
		"bills": [item.to_dict() for item in sorted(service.bills.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"airtime": [item.to_dict() for item in sorted(service.airtime.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"service_requests": [item.to_dict() for item in sorted(service.service_requests.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"notifications": [item.to_dict() for item in sorted(service.notifications.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"fraud_events": [item.to_dict() for item in sorted(service.fraud_events.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"agents": [item.to_dict() for item in sorted(service.evidence.values(), key=lambda item: item.id) if item.tenant_id == tenant_id and item.kind == "agent"],
	}


def route_models(service: MobileBankingService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	console = mobile_console_model(service, tenant_id)
	return {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"], "data": console if route["name"] != "dashboard" else dashboard_model(service, tenant_id)} for route in contract["ui"]["routes"]}
