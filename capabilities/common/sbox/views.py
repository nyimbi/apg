"""UI metadata helpers for the APG Sandbox/Testing Environment capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import SboxService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: SboxService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or SboxService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"sandboxes": service.list_sandboxes(tenant_id),
		"runs": service.list_runs(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


def sandbox_console_model(service: SboxService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or SboxService()
	return {
		"tenant_id": tenant_id,
		"route": "/sbox/sandboxes",
		"title": "Sandbox Console",
		"sandboxes": service.list_sandboxes(tenant_id),
		"templates": service.list_templates(tenant_id),
		"isolation_profiles": service.list_isolation_profiles(tenant_id),
		"actions": ["create_sandbox", "expire_sandbox", "start_run"],
	}


def template_library_model(service: SboxService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or SboxService()
	return {
		"tenant_id": tenant_id,
		"route": "/sbox/templates",
		"title": "Template Library",
		"templates": service.list_templates(tenant_id),
		"required_fields": ["name", "runtime", "owner", "default_ttl_hours"],
	}


def dataset_manager_model(service: SboxService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or SboxService()
	return {
		"tenant_id": tenant_id,
		"route": "/sbox/datasets",
		"title": "Dataset Manager",
		"datasets": service.list_datasets(tenant_id),
		"guardrails": ["dataset_lineage_required", "retention_policy_required", "production_data_review_required", "dataset_masking_required"],
	}


def run_monitor_model(service: SboxService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or SboxService()
	return {
		"tenant_id": tenant_id,
		"route": "/sbox/runs",
		"title": "Run Monitor",
		"runs": service.list_runs(tenant_id),
		"statuses": ["queued", "running", "passed", "failed", "blocked", "cancelled"],
	}


def policy_center_model(service: SboxService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or SboxService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/sbox/policies",
		"title": "Policy Center",
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def logs_model(service: SboxService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or SboxService()
	return {
		"tenant_id": tenant_id,
		"route": "/sbox/logs",
		"title": "Sandbox Logs",
		"audit_events": service.audit_events(tenant_id),
	}


def settings_model(service: SboxService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or SboxService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/sbox/settings",
		"title": "SBOX Settings",
		"configuration_schema": contract["configuration_schema"],
		"configuration": contract["configuration"],
	}


def sbox_agent_model(service: SboxService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or SboxService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/sbox/agents",
		"title": "SBOX Agent Panel",
		"sbox_agents": service.list_sbox_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["sbox_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["sbox_agents"]["allowed_roles"],
		"permissions": ["sbox:view", "sbox:admin"],
	}


def audit_trail_model(service: SboxService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or SboxService()
	return {
		"tenant_id": tenant_id,
		"route": "/sbox/audit",
		"title": "Sandbox Audit Trail",
		"audit_events": service.list_audit_events(tenant_id),
		"permissions": ["sbox:admin"],
	}


def sandbox_policy_model(service: SboxService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or SboxService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/sbox/policies",
		"title": "Sandbox Policy",
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"isolation_profiles": service.list_isolation_profiles(tenant_id),
	}
