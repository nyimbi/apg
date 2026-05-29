"""UI metadata helpers for the APG Identity Federation capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import IdfdService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: IdfdService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or IdfdService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def provider_console_model(service: IdfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/idfd/providers",
		"providers": service.list_providers(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"actions": ["register_provider", "refresh_metadata", "disable_provider"],
	}


def protocol_workbench_model(service: IdfdService, tenant_id: str = "default") -> dict[str, object]:
	providers = service.list_providers(tenant_id)
	return {
		"route": "/idfd/protocols",
		"protocols": {
			"saml": [provider for provider in providers if provider["protocol"] == "saml"],
			"oidc": [provider for provider in providers if provider["protocol"] == "oidc"],
			"ldap": [provider for provider in providers if provider["protocol"] == "ldap"],
			"scim": [provider for provider in providers if provider["protocol"] == "scim"],
		},
		"guardrails": [rule["name"] for rule in service.describe(tenant_id)["rule_engine"]["rules"]],
	}


def mapping_table_model(service: IdfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/idfd/mappings",
		"mappings": service.list_claim_mappings(tenant_id),
		"columns": ["source_claim", "target_claim", "transform", "reviewed"],
	}


def session_monitor_model(service: IdfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/idfd/sessions",
		"sessions": service.list_sessions(tenant_id),
		"actions": ["revoke_session", "inspect_risk"],
	}


def certificate_center_model(service: IdfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/idfd/certificates",
		"certificates": service.list_certificates(tenant_id),
		"actions": ["register_certificate", "rotate_certificate"],
	}


def audit_model(service: IdfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/idfd/audit",
		"events": service.list_audit_events(tenant_id),
		"health_reports": service.list_health_reports(tenant_id),
	}
