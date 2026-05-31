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
		"zero_trust_agents": service.list_zero_trust_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"agents": contract["agents"],
		"streaming": contract["streaming"],
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
		"theme_component": "resource_map",
	}


def identity_console_model(service: ZtnaService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or ZtnaService()
	identities = service.list_identities(tenant_id)
	return {
		"route": "/ztna/identities",
		"tenant_id": tenant_id,
		"identities": identities,
		"pending": [identity for identity in identities if not identity["verified"]],
		"privileged": [identity for identity in identities if identity["privileged"]],
		"theme_component": "identity_console",
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
		"theme_component": "device_posture",
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
		"theme_component": "resource_map",
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
		"theme_component": "access_decision",
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
		"theme_component": "session_monitor",
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
		"theme_component": "risk_console",
	}


def review_queue_model(service: ZtnaService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or ZtnaService()
	contract = service.describe(tenant_id)
	requests = service.list_access_requests(tenant_id)
	return {
		"route": "/ztna/reviews",
		"tenant_id": tenant_id,
		"review_required": [request for request in requests if request["status"] == "review_required"],
		"review_rules": [rule for rule in contract["rule_engine"]["rules"] if rule["effect"]["decision"] == "require_review"],
		"theme_component": "review_queue",
	}


def zero_trust_agent_roster_model(service: ZtnaService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or ZtnaService()
	contract = service.describe(tenant_id)
	return {
		"route": "/ztna/agents",
		"tenant_id": tenant_id,
		"agents": service.list_zero_trust_agents(tenant_id),
		"agent_manifest": contract["agents"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"theme_component": "zero_trust_agent_roster",
	}


def lifecycle_batch_model(service: ZtnaService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or ZtnaService()
	contract = service.describe(tenant_id)
	return {
		"route": "/ztna/lifecycle",
		"tenant_id": tenant_id,
		"streaming": contract["streaming"],
		"batches": service.list_lifecycle_batches(tenant_id),
		"required_operations": contract["streaming"]["required_operations"],
		"theme_component": "bytewax_lifecycle_panel",
	}


def audit_model(service: ZtnaService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or ZtnaService()
	return {
		"route": "/ztna/audit",
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"theme_component": "audit_timeline",
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
