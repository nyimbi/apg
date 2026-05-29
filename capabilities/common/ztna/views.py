"""UI metadata helpers for the Zero Trust Network Access capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import ZtnaService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: ZtnaService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ZtnaService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"recent_audit_events": service.list_audit_events(tenant_id)[-10:],
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def policy_console_model(service: ZtnaService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or ZtnaService()
	resources = service.list_resources(tenant_id)
	return {
		"route": "/ztna/policies",
		"tenant_id": tenant_id,
		"resources": resources,
		"policy_required": [resource for resource in resources if not resource["policy_attached"]],
		"actions": ["attach_resource_policy", "review_high_risk_access"],
	}


def device_posture_model(service: ZtnaService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or ZtnaService()
	devices = service.list_devices(tenant_id)
	return {
		"route": "/ztna/devices",
		"tenant_id": tenant_id,
		"devices": devices,
		"quarantined": [device for device in devices if device["status"] == "quarantined"],
		"actions": ["register_device", "update_device_posture", "quarantine_device"],
	}


def resource_map_model(service: ZtnaService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or ZtnaService()
	resources = service.list_resources(tenant_id)
	segments = sorted({resource["network_segment"] for resource in resources})
	return {
		"route": "/ztna/resources",
		"tenant_id": tenant_id,
		"resources": resources,
		"segments": segments,
		"actions": ["register_resource", "attach_resource_policy"],
	}


def access_requests_model(service: ZtnaService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or ZtnaService()
	requests = service.list_access_requests(tenant_id)
	return {
		"route": "/ztna/access",
		"tenant_id": tenant_id,
		"access_requests": requests,
		"review_required": [request for request in requests if request["status"] == "review_required"],
		"actions": ["request_access", "approve_access_request", "start_session"],
	}


def session_monitor_model(service: ZtnaService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or ZtnaService()
	sessions = service.list_sessions(tenant_id)
	return {
		"route": "/ztna/sessions",
		"tenant_id": tenant_id,
		"sessions": sessions,
		"reauth_required": [session for session in sessions if session["reauth_required"]],
		"actions": ["reevaluate_session", "close_session"],
	}


def risk_console_model(service: ZtnaService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or ZtnaService()
	requests = service.list_access_requests(tenant_id)
	sessions = service.list_sessions(tenant_id)
	return {
		"route": "/ztna/risk",
		"tenant_id": tenant_id,
		"high_risk_requests": [request for request in requests if request["risk_score"] > 0.8],
		"high_risk_sessions": [session for session in sessions if session["risk_score"] > 0.8],
		"signals": {
			"review_rate": _ratio(sum(1 for request in requests if request["status"] == "review_required"), len(requests)),
			"revocation_rate": _ratio(sum(1 for session in sessions if session["status"] == "revoked"), len(sessions)),
		},
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/ztna/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
		"permissions": [route["permission"] for route in contract["ui"]["routes"]],
	}


def _ratio(numerator: int, denominator: int) -> float:
	return round(numerator / denominator, 4) if denominator else 0.0
