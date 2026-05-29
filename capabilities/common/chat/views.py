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
