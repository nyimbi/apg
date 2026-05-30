"""API helpers for APG Chat and Messaging."""

from __future__ import annotations

from typing import Any

from .service import ChatService


SERVICE = ChatService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**SERVICE.conversation_summary(tenant_id),
	}


def create_room(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_room(
		room_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		members=[str(item) for item in payload.get("members", [])],
		retention_policy=str(payload.get("retention_policy") or ""),
		visibility=str(payload.get("visibility") or "private"),
		external_guests=[str(item) for item in payload.get("external_guests", [])],
		guest_policy_attached=bool(payload.get("guest_policy_attached", True)),
		guest_access_expiry_present=bool(payload.get("guest_access_expiry_present", True)),
		access_review_recorded=bool(payload.get("access_review_recorded", True)),
	)


def approve_room(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_room(
		room_id=str(payload["id"]),
		reviewer=str(payload.get("reviewer") or "reviewer"),
		tenant_id=str(payload["tenant_id"]) if payload.get("tenant_id") else None,
	)


def send_message(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.send_message(
		message_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		room_id=str(payload["room_id"]),
		sender=str(payload.get("sender") or ""),
		body=str(payload.get("body") or ""),
		attachments=[str(item) for item in payload.get("attachments", [])],
		delivery_receipts=[str(item) for item in payload.get("delivery_receipts", [])],
		restricted_content_detected=bool(payload.get("restricted_content_detected", False)),
		moderation_completed=bool(payload.get("moderation_completed", True)),
		attachment_scan_completed=bool(payload.get("attachment_scan_completed", True)),
		dlp_check_completed=bool(payload.get("dlp_check_completed", True)),
		ai_agent_participant=bool(payload.get("ai_agent_participant", False)),
		agent_registered=bool(payload.get("agent_registered", True)),
		agent_scope_present=bool(payload.get("agent_scope_present", True)),
		ai_response_disclosed=bool(payload.get("ai_response_disclosed", True)),
	)


def update_presence(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.update_presence(
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_id=str(payload["user_id"]),
		status=str(payload.get("status") or "online"),
		room_id=str(payload["room_id"]) if payload.get("room_id") else None,
		typing=bool(payload.get("typing", False)),
		metadata=dict(payload.get("metadata") or {}),
	)


def review_moderation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.review_moderation(
		item_id=str(payload["id"]),
		reviewer=str(payload.get("reviewer") or "moderator"),
		decision=str(payload.get("decision") or "approved"),
		tenant_id=str(payload["tenant_id"]) if payload.get("tenant_id") else None,
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_rooms(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_rooms(tenant_id)


def list_messages(tenant_id: str | None = None, room_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_messages(tenant_id, room_id)


def list_presence(tenant_id: str | None = None, room_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_presence(tenant_id, room_id)


def list_moderation_items(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_moderation_items(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)
