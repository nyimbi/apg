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
		"dlp_agents": service.list_dlp_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def policy_console_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"route": "/dlpd/policies",
		"policies": service.list_policies(tenant_id),
		"classifiers": service.list_classifiers(tenant_id),
		"routes": capability_routes(tenant_id),
		"theme_component": "policy_matrix",
	}


def classifier_workbench_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"route": "/dlpd/classifiers",
		"classifiers": service.list_classifiers(tenant_id),
		"theme_component": "classifier_grid",
	}


def channel_monitor_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"route": "/dlpd/channels",
		"policies": service.list_policies(tenant_id),
		"inspections": service.list_inspections(tenant_id),
		"theme_component": "channel_flow",
	}


def inspection_workbench_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"route": "/dlpd/inspections",
		"inspections": service.list_inspections(tenant_id),
		"review_required": [item for item in service.list_inspections(tenant_id) if item["review_required"]],
		"theme_component": "inspection_table",
	}


def incident_queue_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	incidents = service.list_incidents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/dlpd/incidents",
		"open": [incident for incident in incidents if incident["status"] == "open"],
		"resolved": [incident for incident in incidents if incident["status"] == "resolved"],
		"quarantine": service.list_quarantine(tenant_id),
		"theme_component": "incident_queue",
	}


def quarantine_vault_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"route": "/dlpd/quarantine",
		"quarantine": service.list_quarantine(tenant_id),
		"theme_component": "quarantine_vault",
	}


def review_queue_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/dlpd/reviews",
		"review_rules": [rule for rule in contract["rule_engine"]["rules"] if rule["effect"]["decision"] == "require_review"],
		"review_required": [item for item in service.list_inspections(tenant_id) if item["review_required"]],
		"theme_component": "review_queue",
	}


def legal_hold_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"route": "/dlpd/legal-hold",
		"legal_hold_items": [item for item in service.list_quarantine(tenant_id) if item["legal_hold"]],
		"theme_component": "legal_hold",
	}


def analytics_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"route": "/dlpd/analytics",
		"summary": service.dashboard_summary(tenant_id),
		"theme_component": "inspection_table",
	}


def dlp_agent_roster_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/dlpd/agents",
		"agents": service.list_dlp_agents(tenant_id),
		"agent_manifest": contract["agents"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"theme_component": "dlp_agent_roster",
	}


def lifecycle_batch_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/dlpd/lifecycle",
		"streaming": contract["streaming"],
		"batches": service.list_lifecycle_batches(tenant_id),
		"required_operations": contract["streaming"]["required_operations"],
		"theme_component": "bytewax_lifecycle_panel",
	}


def audit_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"route": "/dlpd/audit",
		"audit_events": service.list_audit_events(tenant_id),
		"theme_component": "audit_timeline",
	}


def settings_model(service: DlpdService, tenant_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"route": "/dlpd/settings",
		"configuration": service.describe(tenant_id)["configuration"],
		"theme": service.describe(tenant_id)["theme"],
	}
