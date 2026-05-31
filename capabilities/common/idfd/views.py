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
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"federation_agents": service.list_federation_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
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


def scim_directory_model(service: IdfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/idfd/scim",
		"providers": [provider for provider in service.list_providers(tenant_id) if provider["protocol"] == "scim"],
		"guardrails": ["scim_requires_external_id", "scim_deprovisioning_required"],
		"theme_component": "scim_directory",
	}


def risk_console_model(service: IdfdService, tenant_id: str = "default") -> dict[str, object]:
	sessions = service.list_sessions(tenant_id)
	return {
		"route": "/idfd/risk",
		"high_risk_sessions": [session for session in sessions if session["risk_score"] > 0.7],
		"session_count": len(sessions),
		"theme_component": "risk_console",
	}


def review_queue_model(service: IdfdService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	return {
		"route": "/idfd/reviews",
		"review_rules": [rule for rule in contract["rule_engine"]["rules"] if rule["effect"]["decision"] == "require_review"],
		"health_reports": service.list_health_reports(tenant_id),
		"theme_component": "review_queue",
	}


def federation_agent_roster_model(service: IdfdService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	agents = service.list_federation_agents(tenant_id)
	return {
		"route": "/idfd/agents",
		"component": "FederationAgentRoster",
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"theme_component": "federation_agent_roster",
	}


def lifecycle_batch_model(service: IdfdService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"route": "/idfd/lifecycle",
		"component": "IDFDLifecycleBatchMonitor",
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"topics": contract["streaming"]["topics"],
		"theme_component": "bytewax_lifecycle_panel",
	}


def audit_model(service: IdfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/idfd/audit",
		"events": service.list_audit_events(tenant_id),
		"health_reports": service.list_health_reports(tenant_id),
		"agents": service.describe(tenant_id)["agents"],
		"streaming": service.describe(tenant_id)["streaming"],
		"theme_component": "audit_timeline",
	}


def settings_model(service: IdfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/idfd/settings",
		"configuration": service.describe(tenant_id)["configuration"],
		"theme": service.describe(tenant_id)["theme"],
		"agents": service.describe(tenant_id)["agents"],
		"streaming": service.describe(tenant_id)["streaming"],
	}
