"""UI metadata helpers for the Federated Learning capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import FedlService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: FedlService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or FedlService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"federations": service.list_federations(tenant_id),
		"participants": service.list_participants(tenant_id),
		"rounds": service.list_rounds(tenant_id),
		"updates": service.list_updates(tenant_id),
		"aggregations": service.list_aggregations(tenant_id),
		"models": service.list_models(tenant_id),
		"releases": service.list_releases(tenant_id),
		"federation_agents": service.list_federation_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def federation_console_model(service: FedlService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or FedlService()
	return {
		"tenant_id": tenant_id,
		"federations": service.list_federations(tenant_id),
		"participants": service.list_participants(tenant_id),
		"states": ["draft", "active", "paused", "retired"],
	}


def round_monitor_model(service: FedlService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or FedlService()
	return {
		"tenant_id": tenant_id,
		"rounds": service.list_rounds(tenant_id),
		"updates": service.list_updates(tenant_id),
		"aggregations": service.list_aggregations(tenant_id),
		"states": ["running", "aggregated", "blocked"],
	}


def attestation_center_model(service: FedlService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or FedlService()
	participants = service.list_participants(tenant_id)
	return {
		"tenant_id": tenant_id,
		"participants": participants,
		"missing_attestation": [item for item in participants if not item["attested"]],
		"route": "/fedl/attestation",
	}


def update_queue_model(service: FedlService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or FedlService()
	updates = service.list_updates(tenant_id)
	return {
		"tenant_id": tenant_id,
		"updates": updates,
		"quarantined": [item for item in updates if item["status"] == "quarantined"],
		"route": "/fedl/updates",
	}


def aggregation_console_model(service: FedlService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or FedlService()
	return {
		"tenant_id": tenant_id,
		"aggregations": service.list_aggregations(tenant_id),
		"models": service.list_models(tenant_id),
		"route": "/fedl/aggregation",
	}


def privacy_budget_model(service: FedlService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or FedlService()
	return {
		"tenant_id": tenant_id,
		"budget": service.privacy_budget_summary(tenant_id),
		"rounds": service.list_rounds(tenant_id),
		"required_controls": ["privacy_epsilon", "privacy_review_recorded", "secure_aggregation"],
	}


def model_registry_model(service: FedlService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or FedlService()
	return {
		"tenant_id": tenant_id,
		"models": service.list_models(tenant_id),
		"aggregations": service.list_aggregations(tenant_id),
	}


def security_console_model(service: FedlService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or FedlService()
	updates = service.list_updates(tenant_id)
	return {
		"tenant_id": tenant_id,
		"poisoning_signals": [item for item in updates if item["poisoning_signal"]],
		"federation_agents": service.list_federation_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"rules": service.describe(tenant_id)["rule_engine"]["rules"],
		"route": "/fedl/security",
	}


def release_console_model(service: FedlService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or FedlService()
	return {
		"tenant_id": tenant_id,
		"models": service.list_models(tenant_id),
		"releases": service.list_releases(tenant_id),
		"route": "/fedl/release",
	}


def audit_timeline_model(service: FedlService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or FedlService()
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"route": "/fedl/audit",
	}


def federation_agent_roster_model(service: FedlService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or FedlService()
	contract = service.describe(tenant_id)
	agents = service.list_federation_agents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"route": "/fedl/agents",
	}


def lifecycle_batch_model(service: FedlService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or FedlService()
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"tenant_id": tenant_id,
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"topics": contract["streaming"]["topics"],
		"route": "/fedl/lifecycle",
	}
