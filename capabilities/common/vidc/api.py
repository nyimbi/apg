"""API helpers for the Video Conferencing capability."""

from __future__ import annotations

from typing import Any

from .service import VidcService


SERVICE = VidcService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"room_count": summary["room_count"],
		"meeting_count": summary["meeting_count"],
		"active_meeting_count": summary["active_meeting_count"],
		"recording_count": summary["recording_count"],
		"caption_count": summary["caption_count"],
	}


def create_room(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_room(
		tenant_id=str(payload.get("tenant_id") or "default"),
		room_name=str(payload["room_name"]),
		owner=str(payload.get("owner") or ""),
		guest_policy_ref=str(payload.get("guest_policy_ref") or ""),
		moderation_policy_ref=str(payload.get("moderation_policy_ref") or ""),
		waiting_room_enabled=bool(payload.get("waiting_room_enabled", True)),
	)


def start_meeting(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.start_meeting(
		tenant_id=str(payload.get("tenant_id") or "default"),
		room_id=str(payload["room_id"]),
		title=str(payload["title"]),
		host_id=str(payload.get("host_id") or ""),
		participant_count=int(payload.get("participant_count", 1)),
		external_guest_count=int(payload.get("external_guest_count", 0)),
		guest_policy_ref=payload.get("guest_policy_ref"),
		recording_requested=bool(payload.get("recording_requested", False)),
		recording_consent_ref=payload.get("recording_consent_ref"),
		recording_encrypted=bool(payload.get("recording_encrypted", False)),
		capacity_review_recorded=bool(payload.get("capacity_review_recorded", False)),
	)


def add_participant(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.add_participant(
		tenant_id=str(payload.get("tenant_id") or "default"),
		meeting_id=str(payload["meeting_id"]),
		user_ref=str(payload["user_ref"]),
		display_name=str(payload.get("display_name") or payload["user_ref"]),
		role=str(payload.get("role") or "participant"),
		external_guest=bool(payload.get("external_guest", False)),
	)


def create_recording(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_recording(
		tenant_id=str(payload.get("tenant_id") or "default"),
		meeting_id=str(payload["meeting_id"]),
		recording_ref=str(payload["recording_ref"]),
		consent_ref=str(payload.get("consent_ref") or ""),
		retention_policy_ref=str(payload.get("retention_policy_ref") or ""),
		encrypted=bool(payload.get("encrypted", False)),
		created_by=str(payload.get("created_by") or ""),
	)


def generate_captions(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.generate_captions(
		tenant_id=str(payload.get("tenant_id") or "default"),
		meeting_id=str(payload["meeting_id"]),
		language=str(payload.get("language") or "en"),
		transcript_ref=str(payload.get("transcript_ref") or ""),
		caption_count=int(payload.get("caption_count", 0)),
		generated_by=str(payload.get("generated_by") or ""),
	)


def end_meeting(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.end_meeting(
		tenant_id=str(payload.get("tenant_id") or "default"),
		meeting_id=str(payload["meeting_id"]),
		actor=str(payload.get("actor") or ""),
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


def list_video_conferencing(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"rooms": SERVICE.list_rooms(tenant_id),
		"meetings": SERVICE.list_meetings(tenant_id),
		"participants": SERVICE.list_participants(tenant_id),
		"recordings": SERVICE.list_recordings(tenant_id),
		"captions": SERVICE.list_captions(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
