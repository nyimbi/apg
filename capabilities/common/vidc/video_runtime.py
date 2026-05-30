"""Dependency-light video meeting runtime primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


ROOM_STATUSES = {"ready", "disabled"}
MEETING_STATUSES = {"scheduled", "active", "review_required", "ended", "blocked"}
PARTICIPANT_ROLES = {"host", "cohost", "participant", "guest", "observer"}
RECORDING_STATUSES = {"available", "blocked", "retained", "expired"}
MEETING_AGENT_ROLES = {"captioner", "summarizer", "moderator", "action_tracker"}


def utc_now() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part).strip().lower() for part in parts if str(part).strip())
	digest = sha256(seed.encode("utf-8")).hexdigest()[:16]
	return f"{prefix}_{digest}"


def normalize_participant_role(role: str) -> str:
	value = str(role or "participant").strip().lower()
	if value not in PARTICIPANT_ROLES:
		raise ValueError(f"unsupported_participant_role:{role}")
	return value


def normalize_meeting_agent_role(role: str) -> str:
	value = str(role or "summarizer").strip().lower()
	if value not in MEETING_AGENT_ROLES:
		raise ValueError(f"unsupported_meeting_agent_role:{role}")
	return value


def meeting_required_actions(rule_result: dict[str, Any]) -> list[str]:
	return [
		str(action["required_action"])
		for action in rule_result.get("actions", [])
		if action.get("required_action")
	]


def serialize(record: object) -> dict[str, Any]:
	return asdict(record)


@dataclass(slots=True)
class MeetingRoomRecord:
	id: str
	tenant_id: str
	room_name: str
	owner: str
	guest_policy_ref: str
	moderation_policy_ref: str
	waiting_room_enabled: bool = True
	status: str = "ready"
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class MeetingRecord:
	id: str
	tenant_id: str
	room_id: str
	title: str
	host_id: str
	participant_count: int
	external_guest_count: int
	status: str
	recording_requested: bool
	capacity_review_recorded: bool
	required_actions: list[str] = field(default_factory=list)
	matched_rules: list[str] = field(default_factory=list)
	started_at: str = field(default_factory=utc_now)
	ended_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class ParticipantRecord:
	id: str
	tenant_id: str
	meeting_id: str
	user_ref: str
	display_name: str
	role: str
	external_guest: bool = False
	joined_at: str = field(default_factory=utc_now)
	left_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class RecordingRecord:
	id: str
	tenant_id: str
	meeting_id: str
	recording_ref: str
	consent_ref: str
	retention_policy_ref: str
	encrypted: bool
	created_by: str
	status: str = "available"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class CaptionRecord:
	id: str
	tenant_id: str
	meeting_id: str
	language: str
	transcript_ref: str
	caption_count: int
	generated_by: str
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class MeetingAgentRecord:
	id: str
	tenant_id: str
	meeting_id: str
	agent_ref: str
	runtime: str
	role: str
	scope_ref: str
	disclosure_ref: str
	registered_by: str
	status: str = "active"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class MeetingAuditEventRecord:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "low"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


__all__ = [
	"MEETING_STATUSES",
	"MEETING_AGENT_ROLES",
	"PARTICIPANT_ROLES",
	"RECORDING_STATUSES",
	"ROOM_STATUSES",
	"CaptionRecord",
	"MeetingAuditEventRecord",
	"MeetingRecord",
	"MeetingAgentRecord",
	"MeetingRoomRecord",
	"ParticipantRecord",
	"RecordingRecord",
	"meeting_required_actions",
	"normalize_meeting_agent_role",
	"normalize_participant_role",
	"stable_id",
	"utc_now",
]
