"""UI view-model helpers for the APG Data Loss Prevention capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import DlpdService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: DlpdService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or DlpdService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"policies": service.list_policies(tenant_id),
		"classifiers": service.list_classifiers(tenant_id),
		"recent_inspections": service.list_inspections(tenant_id)[-10:],
		"open_incidents": [incident for incident in service.list_incidents(tenant_id) if incident["status"] == "open"],
		"theme": contract["theme"],
	}


def policy_console_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"policies": service.list_policies(tenant_id),
		"classifiers": service.list_classifiers(tenant_id),
		"routes": capability_routes(tenant_id),
	}


def incident_queue_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	incidents = service.list_incidents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"open": [incident for incident in incidents if incident["status"] == "open"],
		"resolved": [incident for incident in incidents if incident["status"] == "resolved"],
		"quarantine": service.list_quarantine(tenant_id),
	}
