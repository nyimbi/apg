"""UI metadata helpers for the Video Conferencing capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import VidcService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: VidcService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or VidcService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def meeting_console_model(
	service: VidcService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or VidcService()
	return {
		"route": "/vidc/meetings",
		"tenant_id": tenant_id,
		"meetings": service.list_meetings(tenant_id),
		"statuses": ["scheduled", "active", "review_required", "ended", "blocked"],
	}


def room_manager_model(
	service: VidcService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or VidcService()
	return {
		"route": "/vidc/rooms",
		"tenant_id": tenant_id,
		"rooms": service.list_rooms(tenant_id),
		"waiting_room_supported": True,
	}


def participant_panel_model(
	service: VidcService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or VidcService()
	return {
		"route": "/vidc/participants",
		"tenant_id": tenant_id,
		"participants": service.list_participants(tenant_id),
		"roles": ["host", "cohost", "participant", "guest", "observer"],
	}


def recording_library_model(
	service: VidcService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or VidcService()
	return {
		"route": "/vidc/recordings",
		"tenant_id": tenant_id,
		"recordings": service.list_recordings(tenant_id),
		"encryption_required": True,
		"retention_policy_required": True,
	}


def caption_workbench_model(
	service: VidcService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or VidcService()
	contract = service.describe(tenant_id)
	return {
		"route": "/vidc/captions",
		"tenant_id": tenant_id,
		"captions": service.list_captions(tenant_id),
		"languages_supported": contract["configuration"]["media"]["supported_caption_languages"],
	}


def meeting_agent_model(
	service: VidcService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or VidcService()
	contract = service.describe(tenant_id)
	return {
		"route": "/vidc/agents",
		"tenant_id": tenant_id,
		"meeting_agents": service.list_meeting_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["meeting_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["meeting_agents"]["allowed_roles"],
		"theme": contract["theme"]["components"]["agent_panel"],
	}


def analytics_model(
	service: VidcService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or VidcService()
	return {
		"route": "/vidc/analytics",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"review_required_meetings": [
			meeting
			for meeting in service.list_meetings(tenant_id)
			if meeting["status"] == "review_required"
		],
	}


def audit_model(
	service: VidcService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or VidcService()
	contract = service.describe(tenant_id)
	return {
		"route": "/vidc/audit",
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"event_stream": contract["configuration"]["observability"]["event_stream"],
		"theme": contract["theme"]["components"]["audit_timeline"],
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/vidc/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}
