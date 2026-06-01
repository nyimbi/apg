"""UI metadata helpers for the Encryption Services capability."""

from __future__ import annotations

from . import api
from .capability_contract import get_capability_contract
from .service import EncrService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: EncrService | None = None,
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
		"key_domains": service.list_key_domains(tenant_id),
		"operations": service.list_operations(tenant_id),
		"exception_reviews": service.list_exception_reviews(tenant_id),
		"rotations": service.list_rotations(tenant_id),
		"crypto_agents": service.list_crypto_agents(tenant_id),
		"crypto_lifecycle_batches": service.list_crypto_lifecycle_batches(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"review_evidence": contract["review_evidence"],
		"theme": contract["theme"],
	}


def operations_console_model(
	service: EncrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	operations = service.list_operations(tenant_id)
	return {
		"route": "/encr/operations",
		"tenant_id": tenant_id,
		"operations": operations,
		"denied": [item for item in operations if item["status"] == "denied"],
		"review_required": [item for item in operations if item["status"] == "review_required"],
		"decision_filters": ["allowed", "review_required", "denied"],
	}


def key_domain_model(
	service: EncrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/encr/keys",
		"tenant_id": tenant_id,
		"key_domains": service.list_key_domains(tenant_id),
		"classifications": ["public", "internal", "confidential", "restricted", "critical"],
		"rotation_states": ["current", "scheduled", "rotated"],
	}


def policy_designer_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/encr/policies",
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"decision_order": ["deny", "require_review", "allow"],
	}


def entropy_console_model(
	service: EncrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/encr/entropy",
		"tenant_id": tenant_id,
		"key_domains": service.list_key_domains(tenant_id),
		"minimum_entropy_quality": get_capability_contract(tenant_id)["configuration"]["cryptography"]["minimum_entropy_quality"],
	}


def exception_queue_model(
	service: EncrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	reviews = service.list_exception_reviews(tenant_id)
	return {
		"route": "/encr/exceptions",
		"tenant_id": tenant_id,
		"exception_reviews": reviews,
		"pending": [item for item in reviews if item["status"] == "pending"],
	}


def rotation_console_model(
	service: EncrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	rotations = service.list_rotations(tenant_id)
	return {
		"route": "/encr/rotations",
		"tenant_id": tenant_id,
		"rotations": rotations,
		"scheduled": [item for item in rotations if item["status"] == "scheduled"],
	}


def homomorphic_workspace_model(tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/encr/homomorphic",
		"tenant_id": tenant_id,
		"supported_operations": ["add", "sum", "aggregate", "count", "concat", "digest"],
		"result_mode": "sealed-output",
	}


def analytics_model(
	service: EncrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/encr/analytics",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
	}


def audit_timeline_model(
	service: EncrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/encr/audit",
		"tenant_id": tenant_id,
		"events": service.list_audit_events(tenant_id),
	}


def crypto_agents_model(
	service: EncrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	contract = service.describe(tenant_id)
	agents = contract["agents"]
	return {
		"route": "/encr/agents",
		"tenant_id": tenant_id,
		"crypto_agents": service.list_crypto_agents(tenant_id),
		"pending_reviews": [
			agent for agent in service.list_crypto_agents(tenant_id)
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
		"route": "/encr/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"review_evidence": contract["review_evidence"],
		"theme": contract["theme"],
	}
