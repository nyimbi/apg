"""UI metadata helpers for the Configuration Management capability."""

from __future__ import annotations

from . import api
from .capability_contract import get_capability_contract
from .service import ConfService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: ConfService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"records": service.list_records(tenant_id),
		"changes": service.list_changes(tenant_id),
		"deployments": service.list_deployments(tenant_id),
		"drift_remediations": service.list_drift_remediations(tenant_id),
		"agents": service.list_agents(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"summary": service.governance_summary(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def change_queue_model(
	service: ConfService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"tenant_id": tenant_id,
		"pending_changes": [
			item for item in service.list_changes(tenant_id)
			if item["status"] == "pending"
		],
		"approved_changes": [
			item for item in service.list_changes(tenant_id)
			if item["status"] == "approved"
		],
	}


def drift_remediation_model(
	service: ConfService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"tenant_id": tenant_id,
		"remediations": service.list_drift_remediations(tenant_id),
		"drifted_records": [
			item for item in service.list_records(tenant_id)
			if item["status"] == "drifted"
		],
	}


def audit_model(
	service: ConfService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"tenant_id": tenant_id,
		"events": service.list_audit_events(tenant_id),
	}


def agent_model(
	service: ConfService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or api.SERVICE
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_agents(tenant_id),
		"policy": contract["configuration"]["conf_agents"],
		"route": "/config/agents",
	}
