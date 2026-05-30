"""UI metadata helpers for the Security Operations capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import SeopService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: SeopService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SeopService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


def detection_console_model(
	service: SeopService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SeopService()
	return {
		"route": "/seop/detections",
		"tenant_id": tenant_id,
		"detections": service.list_detections(tenant_id),
		"status_filters": ["new", "review_required", "triaged", "linked"],
		"severity_filters": ["low", "medium", "high", "critical"],
	}


def incident_queue_model(
	service: SeopService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SeopService()
	return {
		"route": "/seop/incidents",
		"tenant_id": tenant_id,
		"incidents": service.list_incidents(tenant_id),
		"state_filters": ["open", "escalated", "responding", "contained", "closed"],
	}


def playbook_manager_model(
	service: SeopService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SeopService()
	return {
		"route": "/seop/playbooks",
		"tenant_id": tenant_id,
		"playbooks": service.list_playbooks(tenant_id),
		"approval_required": True,
	}


def response_actions_model(
	service: SeopService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SeopService()
	return {
		"route": "/seop/responses",
		"tenant_id": tenant_id,
		"responses": service.list_responses(tenant_id),
		"statuses": ["planned", "executed", "blocked"],
	}


def posture_model(
	service: SeopService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SeopService()
	return {
		"route": "/seop/posture",
		"tenant_id": tenant_id,
		"controls": service.list_posture_controls(tenant_id),
		"coverage_bands": ["gap", "partial", "covered"],
	}


def agent_workbench_model(
	service: SeopService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SeopService()
	contract = service.describe(tenant_id)
	return {
		"route": "/seop/agents",
		"tenant_id": tenant_id,
		"agents": service.list_seop_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["seop_agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["seop_agents"]["supported_roles"],
		"approval_required": contract["configuration"]["seop_agents"]["human_approval_required"],
	}


def audit_trail_model(
	service: SeopService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SeopService()
	return {
		"route": "/seop/audit",
		"tenant_id": tenant_id,
		"events": service.list_audit_events(tenant_id),
		"event_types": [
			"detection_created",
			"incident_opened",
			"playbook_approved",
			"response_executed",
			"incident_closed",
			"seop_agent_registered",
		],
	}


def triage_model(
	service: SeopService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SeopService()
	return {
		"route": "/seop/triage",
		"tenant_id": tenant_id,
		"review_required": [
			detection
			for detection in service.list_detections(tenant_id)
			if detection["status"] == "review_required"
		],
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/seop/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}
