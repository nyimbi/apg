"""UI metadata helpers for the Security Framework capability."""

from __future__ import annotations

from . import api
from .capability_contract import get_capability_contract
from .service import SecuService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: SecuService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"assessments": service.list_assessments(tenant_id),
		"devices": service.list_devices(tenant_id),
		"threats": service.list_threats(tenant_id),
		"controls": service.list_controls(tenant_id),
		"policy_exceptions": service.list_policy_exceptions(tenant_id),
		"incidents": service.list_incidents(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def risk_console_model(
	service: SecuService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/secu/risk",
		"tenant_id": tenant_id,
		"assessments": service.list_assessments(tenant_id),
		"devices": service.list_devices(tenant_id),
		"decision_filters": ["allow", "challenge", "quarantine", "deny"],
	}


def threat_console_model(
	service: SecuService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/secu/threats",
		"tenant_id": tenant_id,
		"threats": service.list_threats(tenant_id),
		"severity_filters": ["info", "low", "medium", "high", "critical"],
	}


def policy_workbench_model(
	service: SecuService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/secu/policies",
		"tenant_id": tenant_id,
		"policies": service.list_policies(tenant_id),
		"security_levels": ["public", "internal", "confidential", "restricted", "critical"],
	}


def compliance_console_model(
	service: SecuService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/secu/compliance",
		"tenant_id": tenant_id,
		"controls": service.list_controls(tenant_id),
		"statuses": ["implemented", "evidence_required", "non_compliant", "waived"],
	}


def exception_queue_model(
	service: SecuService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/secu/exceptions",
		"tenant_id": tenant_id,
		"exceptions": service.list_policy_exceptions(tenant_id),
		"pending": [
			item for item in service.list_policy_exceptions(tenant_id)
			if item["status"] == "pending"
		],
	}


def incident_response_model(
	service: SecuService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/secu/incidents",
		"tenant_id": tenant_id,
		"incidents": service.list_incidents(tenant_id),
		"open_incidents": [
			item for item in service.list_incidents(tenant_id)
			if item["status"] != "resolved"
		],
	}


def quarantine_model(
	service: SecuService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/secu/quarantine",
		"tenant_id": tenant_id,
		"devices": [
			item for item in service.list_devices(tenant_id)
			if item["quarantined"]
		],
	}


def quarantine_console_model(
	service: SecuService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	return quarantine_model(service, tenant_id)


def audit_model(
	service: SecuService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/secu/audit",
		"tenant_id": tenant_id,
		"events": service.list_audit_events(tenant_id),
	}


def audit_timeline_model(
	service: SecuService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	return audit_model(service, tenant_id)


def rule_workbench_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/secu/rules",
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"decision_order": ["deny", "quarantine", "challenge", "allow"],
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/secu/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
	}
