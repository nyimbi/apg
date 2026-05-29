"""Service layer for the Video Conferencing capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .video_runtime import (
	CaptionRecord,
	MeetingAuditEventRecord,
	MeetingRecord,
	MeetingRoomRecord,
	ParticipantRecord,
	RecordingRecord,
	meeting_required_actions,
	normalize_participant_role,
	stable_id,
	utc_now,
)


class VidcService:
	"""Deterministic video meeting service for APG composition."""

	def __init__(self) -> None:
		self.rooms: dict[str, MeetingRoomRecord] = {}
		self.meetings: dict[str, MeetingRecord] = {}
		self.participants: dict[str, ParticipantRecord] = {}
		self.recordings: dict[str, RecordingRecord] = {}
		self.captions: dict[str, CaptionRecord] = {}
		self.audit_events: dict[str, MeetingAuditEventRecord] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_room(
		self,
		tenant_id: str,
		room_name: str,
		owner: str,
		guest_policy_ref: str,
		moderation_policy_ref: str,
		waiting_room_enabled: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(room_name or "").strip():
			raise ValueError("room_name_required")
		if not str(owner or "").strip():
			raise ValueError("room_owner_required")
		if not str(moderation_policy_ref or "").strip():
			raise PermissionError("moderation_policy_required")
		record = MeetingRoomRecord(
			id=stable_id("vidc_room", tenant_id, room_name),
			tenant_id=tenant_id,
			room_name=room_name,
			owner=owner,
			guest_policy_ref=guest_policy_ref,
			moderation_policy_ref=moderation_policy_ref,
			waiting_room_enabled=bool(waiting_room_enabled),
		)
		self.rooms[record.id] = record
		self._record_event(tenant_id, "room_created", record.id, f"Meeting room created: {room_name}", owner)
		return record.to_dict()

	def start_meeting(
		self,
		tenant_id: str,
		room_id: str,
		title: str,
		host_id: str,
		participant_count: int,
		external_guest_count: int = 0,
		guest_policy_ref: str | None = None,
		recording_requested: bool = False,
		recording_consent_ref: str | None = None,
		recording_encrypted: bool = False,
		capacity_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		room = self._get_room(tenant_id, room_id)
		if not str(title or "").strip():
			raise ValueError("meeting_title_required")
		if participant_count <= 0:
			raise ValueError("participant_count_must_be_positive")
		context = {
			"tenant_context_present": True,
			"operation": "start_meeting",
			"host_present": bool(str(host_id or "").strip()),
			"external_guest_present": external_guest_count > 0,
			"guest_policy_attached": bool(str(guest_policy_ref or room.guest_policy_ref or "").strip()),
			"recording_requested": bool(recording_requested),
			"recording_consent_recorded": bool(str(recording_consent_ref or "").strip()),
			"recording_encrypted": bool(recording_encrypted),
			"participant_count": int(participant_count),
			"capacity_review_recorded": bool(capacity_review_recorded),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		status = "review_required" if result["decision"] == "require_review" else "active"
		record = MeetingRecord(
			id=stable_id("vidc_meeting", tenant_id, room.id, title, len(self.meetings)),
			tenant_id=tenant_id,
			room_id=room.id,
			title=title,
			host_id=host_id,
			participant_count=int(participant_count),
			external_guest_count=int(external_guest_count),
			status=status,
			recording_requested=bool(recording_requested),
			capacity_review_recorded=bool(capacity_review_recorded),
			required_actions=meeting_required_actions(result),
			matched_rules=list(result["matched_rules"]),
		)
		self.meetings[record.id] = record
		self._record_event(tenant_id, "meeting_started", record.id, f"Meeting {status}: {title}", host_id or room.owner)
		return record.to_dict()

	def add_participant(
		self,
		tenant_id: str,
		meeting_id: str,
		user_ref: str,
		display_name: str,
		role: str = "participant",
		external_guest: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		meeting = self._get_meeting(tenant_id, meeting_id)
		if not str(user_ref or "").strip():
			raise ValueError("participant_user_ref_required")
		if external_guest and meeting.external_guest_count <= 0:
			meeting.external_guest_count = 1
		record = ParticipantRecord(
			id=stable_id("vidc_participant", tenant_id, meeting.id, user_ref),
			tenant_id=tenant_id,
			meeting_id=meeting.id,
			user_ref=user_ref,
			display_name=str(display_name or user_ref),
			role=normalize_participant_role(role),
			external_guest=bool(external_guest),
		)
		self.participants[record.id] = record
		meeting.participant_count = max(meeting.participant_count, self._participant_count(tenant_id, meeting.id))
		self._record_event(tenant_id, "participant_joined", record.id, f"Participant joined: {record.display_name}", user_ref)
		return record.to_dict()

	def create_recording(
		self,
		tenant_id: str,
		meeting_id: str,
		recording_ref: str,
		consent_ref: str,
		retention_policy_ref: str,
		encrypted: bool,
		created_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		meeting = self._get_meeting(tenant_id, meeting_id)
		context = {
			"tenant_context_present": True,
			"recording_requested": True,
			"recording_consent_recorded": bool(str(consent_ref or "").strip()),
			"recording_encrypted": bool(encrypted),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if not str(recording_ref or "").strip():
			raise ValueError("recording_ref_required")
		if not str(retention_policy_ref or "").strip():
			raise PermissionError("retention_policy_required")
		record = RecordingRecord(
			id=stable_id("vidc_recording", tenant_id, meeting.id, recording_ref),
			tenant_id=tenant_id,
			meeting_id=meeting.id,
			recording_ref=recording_ref,
			consent_ref=consent_ref,
			retention_policy_ref=retention_policy_ref,
			encrypted=bool(encrypted),
			created_by=created_by,
		)
		self.recordings[record.id] = record
		meeting.recording_requested = True
		self._record_event(tenant_id, "recording_created", record.id, f"Recording created: {meeting.title}", created_by)
		return record.to_dict()

	def generate_captions(
		self,
		tenant_id: str,
		meeting_id: str,
		language: str,
		transcript_ref: str,
		caption_count: int,
		generated_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		meeting = self._get_meeting(tenant_id, meeting_id)
		if not str(language or "").strip():
			raise ValueError("caption_language_required")
		if not str(transcript_ref or "").strip():
			raise ValueError("transcript_ref_required")
		record = CaptionRecord(
			id=stable_id("vidc_caption", tenant_id, meeting.id, language),
			tenant_id=tenant_id,
			meeting_id=meeting.id,
			language=language,
			transcript_ref=transcript_ref,
			caption_count=int(caption_count),
			generated_by=generated_by,
		)
		self.captions[record.id] = record
		self._record_event(tenant_id, "captions_generated", record.id, f"Captions generated: {meeting.title}", generated_by)
		return record.to_dict()

	def end_meeting(self, tenant_id: str, meeting_id: str, actor: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		meeting = self._get_meeting(tenant_id, meeting_id)
		meeting.status = "ended"
		meeting.ended_at = utc_now()
		self._record_event(tenant_id, "meeting_ended", meeting.id, f"Meeting ended: {meeting.title}", actor)
		return meeting.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		record = self.create_room(
			tenant_id=tenant_id,
			room_name=record_id,
			owner=str(metadata.get("owner") or "compatibility-owner"),
			guest_policy_ref=str(metadata.get("guest_policy_ref") or "guest-policy://compatibility"),
			moderation_policy_ref=str(metadata.get("moderation_policy_ref") or "moderation://compatibility"),
			waiting_room_enabled=bool(metadata.get("waiting_room_enabled", True)),
		)
		if status != "active":
			room = self._get_room(tenant_id, record["id"])
			room.status = status
			room.updated_at = utc_now()
			record = room.to_dict()
		return record

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_rooms(tenant_id)

	def list_rooms(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.rooms, tenant_id)

	def list_meetings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.meetings, tenant_id)

	def list_participants(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.participants, tenant_id)

	def list_recordings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.recordings, tenant_id)

	def list_captions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.captions, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		meetings = self.list_meetings(tenant_id)
		return {
			"tenant_id": tenant_id,
			"room_count": len(self.list_rooms(tenant_id)),
			"meeting_count": len(meetings),
			"active_meeting_count": sum(1 for item in meetings if item["status"] == "active"),
			"review_required_meeting_count": sum(1 for item in meetings if item["status"] == "review_required"),
			"participant_count": len(self.list_participants(tenant_id)),
			"external_guest_count": sum(item["external_guest_count"] for item in meetings),
			"recording_count": len(self.list_recordings(tenant_id)),
			"caption_count": len(self.list_captions(tenant_id)),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
		}

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			self._raise_policy(self.evaluate({"tenant_context_present": False}))

	def _raise_policy(self, result: dict[str, Any]) -> None:
		reasons = ", ".join(action.get("reason", "video_policy_blocked") for action in result["actions"])
		raise PermissionError(reasons or "video_policy_blocked")

	def _get_room(self, tenant_id: str, room_id: str) -> MeetingRoomRecord:
		room = self.rooms.get(room_id)
		if room is None:
			room = next((item for item in self.rooms.values() if item.tenant_id == tenant_id and item.room_name == room_id), None)
		if room is None or room.tenant_id != tenant_id:
			raise KeyError(f"room_not_found:{room_id}")
		return room

	def _get_meeting(self, tenant_id: str, meeting_id: str) -> MeetingRecord:
		meeting = self.meetings.get(meeting_id)
		if meeting is None:
			meeting = next((item for item in self.meetings.values() if item.tenant_id == tenant_id and item.title == meeting_id), None)
		if meeting is None or meeting.tenant_id != tenant_id:
			raise KeyError(f"meeting_not_found:{meeting_id}")
		return meeting

	def _participant_count(self, tenant_id: str, meeting_id: str) -> int:
		return len([
			participant
			for participant in self.participants.values()
			if participant.tenant_id == tenant_id and participant.meeting_id == meeting_id
		])

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "low",
	) -> dict[str, Any]:
		record = MeetingAuditEventRecord(
			id=stable_id("vidc_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])
