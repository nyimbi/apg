"""Service layer for the Video Conferencing capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	PRIVILEGED_VIDC_AGENT_ROLES,
	SUPPORTED_VIDC_AGENT_ROLES,
	SUPPORTED_VIDC_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .video_runtime import (
	CaptionRecord,
	MeetingAgentRecord,
	MeetingAuditEventRecord,
	MeetingRecord,
	MeetingRoomRecord,
	ParticipantRecord,
	RecordingRecord,
	VideoAgentRecord,
	VidcLifecycleBatchRecord,
	meeting_required_actions,
	normalize_meeting_agent_role,
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
		self.meeting_agents: dict[str, MeetingAgentRecord] = {}
		self.video_agents: dict[str, VideoAgentRecord] = {}
		self.lifecycle_batches: dict[str, VidcLifecycleBatchRecord] = {}
		self.audit_events: dict[str, MeetingAuditEventRecord] = {}
		self._agent_runtimes = {_normalize_token(item) for item in SUPPORTED_VIDC_AGENT_RUNTIMES}
		self._agent_roles = {_normalize_token(item) for item in SUPPORTED_VIDC_AGENT_ROLES}
		self._privileged_agent_roles = {_normalize_token(item) for item in PRIVILEGED_VIDC_AGENT_ROLES}
		self._lifecycle_operations = {
			_normalize_token(item)
			for item in get_capability_contract()["streaming"]["required_operations"]
		}

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
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "create_room",
			"room_name_present": bool(str(room_name or "").strip()),
			"room_owner_present": bool(str(owner or "").strip()),
			"moderation_policy_attached": bool(str(moderation_policy_ref or "").strip()),
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
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
		recording_retention_policy_attached: bool = True,
		recording_access_audit_enabled: bool = True,
		secure_transport: bool = True,
		screen_share_requested: bool = False,
		screen_share_policy_attached: bool = True,
		computer_vision_assist_requested: bool = False,
		computer_vision_policy_attached: bool = True,
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
			"room_present": True,
			"host_present": bool(str(host_id or "").strip()),
			"secure_transport": bool(secure_transport),
			"screen_share_requested": bool(screen_share_requested),
			"screen_share_policy_attached": bool(screen_share_policy_attached),
			"external_guest_present": external_guest_count > 0,
			"guest_policy_attached": bool(str(guest_policy_ref or room.guest_policy_ref or "").strip()),
			"waiting_room_enabled": bool(room.waiting_room_enabled),
			"recording_requested": bool(recording_requested),
			"recording_consent_recorded": bool(str(recording_consent_ref or "").strip()),
			"recording_encrypted": bool(recording_encrypted),
			"recording_retention_policy_attached": bool(recording_retention_policy_attached),
			"recording_access_audit_enabled": bool(recording_access_audit_enabled),
			"participant_count": int(participant_count),
			"computer_vision_assist_requested": bool(computer_vision_assist_requested),
			"computer_vision_policy_attached": bool(computer_vision_policy_attached),
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
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "add_participant",
			"meeting_present": True,
			"user_ref_present": bool(str(user_ref or "").strip()),
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
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
			"recording_retention_policy_attached": bool(str(retention_policy_ref or "").strip()),
			"recording_access_audit_enabled": True,
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
		supported_languages = set(self.describe(tenant_id)["configuration"]["media"]["supported_caption_languages"])
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "generate_captions",
			"transcript_ref_present": bool(str(transcript_ref or "").strip()),
			"caption_language_supported": language in supported_languages,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
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

	def register_meeting_agent(
		self,
		tenant_id: str,
		meeting_id: str,
		agent_ref: str,
		runtime: str,
		role: str,
		scope_ref: str,
		disclosure_ref: str,
		registered_by: str,
		agent_registered: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		meeting = self._get_meeting(tenant_id, meeting_id)
		supported_runtimes = set(self.describe(tenant_id)["configuration"]["meeting_agents"]["supported_runtimes"])
		context = {
			"tenant_context_present": True,
			"meeting_agent_present": True,
			"agent_registered": bool(agent_registered and str(agent_ref or "").strip()),
			"agent_runtime_supported": runtime in supported_runtimes,
			"agent_scope_present": bool(str(scope_ref or "").strip()),
			"agent_contribution_disclosed": bool(str(disclosure_ref or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = MeetingAgentRecord(
			id=stable_id("vidc_agent", tenant_id, meeting.id, agent_ref, role),
			tenant_id=tenant_id,
			meeting_id=meeting.id,
			agent_ref=agent_ref,
			runtime=runtime,
			role=normalize_meeting_agent_role(role),
			scope_ref=scope_ref,
			disclosure_ref=disclosure_ref,
			registered_by=registered_by,
		)
		self.meeting_agents[record.id] = record
		self._record_event(tenant_id, "meeting_agent_registered", record.id, f"Meeting agent registered: {agent_ref}", registered_by)
		return record.to_dict()

	def register_video_agent(
		self,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		record_key = self._tenant_key(tenant_id, agent_id)
		if record_key in self.video_agents:
			raise ValueError(f"video_agent_already_exists:{agent_id}")
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "register_video_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		if not str(name or "").strip():
			raise ValueError("video_agent_name_required")
		record = VideoAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=str(name).strip(),
			runtime=runtime_value,
			role=role_value,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status="pending_review" if result["decision"] == "require_review" else "active",
		)
		self.video_agents[record_key] = record
		self._record_event(tenant_id, "video_agent_registered", agent_id, f"Video agent registered: {record.name}", record.owner)
		return record.to_dict()

	def validate_vidc_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "video_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		mutation_count = int(mutation_count)
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		operation_supported = operation_value in self._lifecycle_operations
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "validate_vidc_lifecycle_batch",
			"event_stream": stream_value,
			"mutation_count": mutation_count,
			"lifecycle_operation_supported": operation_supported,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		accepted = result["decision"] == "allow"
		record_id = batch_id or f"vidc-batch-{len(self.lifecycle_batches) + 1:06d}"
		record = VidcLifecycleBatchRecord(
			id=record_id,
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			status="accepted" if accepted else "denied",
		)
		self.lifecycle_batches[self._tenant_key(tenant_id, record_id)] = record
		self._record_event(tenant_id, f"vidc_lifecycle_batch_{record.status}", record_id, f"VIDC lifecycle batch {record.status}: {operation_value}", "bytewax")
		if result["decision"] == "deny":
			self._raise_policy(result)
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

	def list_meeting_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.meeting_agents, tenant_id)

	def list_video_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.video_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.lifecycle_batches, tenant_id)

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
			"meeting_agent_count": len(self.list_meeting_agents(tenant_id)),
			"video_agent_count": len(self.list_video_agents(tenant_id)),
			"pending_video_agent_review_count": sum(1 for item in self.list_video_agents(tenant_id) if item["status"] == "pending_review"),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": sum(1 for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"),
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

	def _tenant_key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
