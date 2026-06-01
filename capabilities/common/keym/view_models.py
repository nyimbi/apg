"""Dependency-light UI view models for the KEYM capability package."""

from __future__ import annotations

from . import api
from .capability_contract import get_capability_contract
from .service import KeymService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: KeymService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"keys": service.list_keys(tenant_id),
		"operations": service.list_operations(tenant_id),
		"export_approvals": service.list_export_approvals(tenant_id),
		"rotation_exceptions": service.list_rotation_exceptions(tenant_id),
		"rotations": service.list_rotations(tenant_id),
		"key_agents": service.list_key_agents(tenant_id),
		"key_lifecycle_batches": service.list_key_lifecycle_batches(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"review_evidence": contract["review_evidence"],
		"theme": contract["theme"],
	}


def inventory_model(service: KeymService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/keym/keys",
		"tenant_id": tenant_id,
		"keys": service.list_keys(tenant_id),
		"key_classes": ["data", "root", "tenant", "signing", "wrapping"],
		"statuses": ["active", "disabled", "compromised", "destroyed"],
	}


def lifecycle_model(service: KeymService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/keym/lifecycle",
		"tenant_id": tenant_id,
		"rotations": service.list_rotations(tenant_id),
		"rotation_exceptions": service.list_rotation_exceptions(tenant_id),
	}


def export_approval_queue_model(service: KeymService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	approvals = service.list_export_approvals(tenant_id)
	return {
		"route": "/keym/export-approvals",
		"tenant_id": tenant_id,
		"export_approvals": approvals,
		"pending": [item for item in approvals if item["status"] == "pending"],
	}


def rotation_exception_queue_model(service: KeymService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	exceptions = service.list_rotation_exceptions(tenant_id)
	return {
		"route": "/keym/rotation-exceptions",
		"tenant_id": tenant_id,
		"rotation_exceptions": exceptions,
		"pending": [item for item in exceptions if item["status"] == "pending"],
	}


def hsm_console_model(service: KeymService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/keym/hsm",
		"tenant_id": tenant_id,
		"root_keys": [item for item in service.list_keys(tenant_id) if item["key_class"] == "root"],
		"attestation_required": True,
	}


def compromise_console_model(service: KeymService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/keym/compromise",
		"tenant_id": tenant_id,
		"compromised_keys": [item for item in service.list_keys(tenant_id) if item["status"] == "compromised"],
	}


def audit_timeline_model(service: KeymService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/keym/audit",
		"tenant_id": tenant_id,
		"events": service.list_audit_events(tenant_id),
	}


def analytics_model(service: KeymService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/keym/analytics",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
	}


def key_agents_model(service: KeymService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	contract = service.describe(tenant_id)
	agents = contract["agents"]
	return {
		"route": "/keym/agents",
		"tenant_id": tenant_id,
		"key_agents": service.list_key_agents(tenant_id),
		"pending_reviews": [
			agent for agent in service.list_key_agents(tenant_id)
			if agent["status"] == "pending_review"
		],
		"supported_runtimes": agents["supported_runtimes"],
		"supported_roles": agents["supported_roles"],
		"privileged_roles": agents["privileged_roles"],
		"guardrails": agents["guardrails"],
		"required_fields": [
			"id",
			"name",
			"runtime",
			"role",
			"scope",
			"owner",
			"purpose",
			"contribution_disclosed",
		],
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/keym/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"review_evidence": contract["review_evidence"],
		"theme": contract["theme"],
	}
