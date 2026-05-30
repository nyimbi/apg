"""UI metadata helpers for APG Chat and Messaging."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import ChatService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: ChatService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ChatService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.conversation_summary(tenant_id),
		"rooms": service.list_rooms(tenant_id),
		"messages": service.list_messages(tenant_id),
		"presence": service.list_presence(tenant_id),
		"moderation_queue": [
			item for item in service.list_moderation_items(tenant_id)
			if item["status"] == "pending"
		],
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def room_manager_model(
	service: ChatService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ChatService()
	rooms = service.list_rooms(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rooms": rooms,
		"active": [item for item in rooms if item["status"] == "active"],
		"pending_review": [item for item in rooms if item["status"] == "pending_review"],
	}


def message_console_model(
	service: ChatService | None = None,
	tenant_id: str = "default",
	room_id: str | None = None,
) -> dict[str, object]:
	service = service or ChatService()
	return {
		"tenant_id": tenant_id,
		"room_id": room_id,
		"messages": service.list_messages(tenant_id, room_id),
		"presence": service.list_presence(tenant_id, room_id),
	}


def moderation_queue_model(
	service: ChatService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ChatService()
	items = service.list_moderation_items(tenant_id)
	return {
		"tenant_id": tenant_id,
		"pending": [item for item in items if item["status"] == "pending"],
		"reviewed": [item for item in items if item["status"] != "pending"],
	}


def agent_participant_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"enabled": contract["configuration"]["ai_agents"]["agent_participants_enabled"],
		"supported_runtimes": contract["configuration"]["ai_agents"]["supported_runtimes"],
		"required_controls": [
			"agent_registration_required",
			"agent_scope_required",
			"agent_response_disclosure_required",
		],
		"theme": contract["theme"]["components"]["agent_panel"],
	}


def audit_model(
	service: ChatService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ChatService()
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"moderation_events": [
			item for item in service.list_audit_events(tenant_id)
			if item["event_type"] == "moderation_reviewed"
		],
	}


def analytics_model(
	service: ChatService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ChatService()
	summary = service.conversation_summary(tenant_id)
	message_count = summary["message_count"]
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"attachment_rate": summary["attachment_count"] / message_count if message_count else 0.0,
		"active_room_rate": summary["active_room_count"] / summary["room_count"] if summary["room_count"] else 0.0,
		"moderation_queue_count": summary["moderation_queue_count"],
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}
