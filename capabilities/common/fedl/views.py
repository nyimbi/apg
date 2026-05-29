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
