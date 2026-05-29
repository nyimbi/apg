"""Domain models for APG Chat and Messaging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChatRoom:
	"""Tenant-scoped chat room with ownership, membership, and retention policy."""

	id: str
	tenant_id: str
	name: str
	owner: str
	members: tuple[str, ...]
	retention_policy: str
	visibility: str = "private"
	external_guests: tuple[str, ...] = ()
	status: str = "active"
	review_status: str = "approved"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"members": list(self.members),
			"member_count": len(self.members) + len(self.external_guests),
			"retention_policy": self.retention_policy,
			"visibility": self.visibility,
			"external_guests": list(self.external_guests),
			"status": self.status,
			"review_status": self.review_status,
		}


@dataclass(frozen=True)
class ChatMessage:
	"""Message payload with delivery, receipt, and moderation evidence."""

	id: str
	tenant_id: str
	room_id: str
	sender: str
	body: str
	fingerprint: str
	thread_key: str
	attachments: tuple[str, ...] = ()
	status: str = "delivered"
	delivery_receipts: tuple[str, ...] = ()
	moderation_status: str = "clear"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"room_id": self.room_id,
			"sender": self.sender,
			"body": self.body,
			"fingerprint": self.fingerprint,
			"thread_key": self.thread_key,
			"attachments": list(self.attachments),
			"status": self.status,
			"delivery_receipts": list(self.delivery_receipts),
			"moderation_status": self.moderation_status,
		}


@dataclass(frozen=True)
class ChatPresence:
	"""User availability state for room-level collaboration surfaces."""

	id: str
	tenant_id: str
	user_id: str
	status: str
	room_id: str | None = None
	typing: bool = False
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"user_id": self.user_id,
			"status": self.status,
			"room_id": self.room_id,
			"typing": self.typing,
			"metadata": dict(self.metadata),
		}


@dataclass(frozen=True)
class ModerationItem:
	"""Moderation queue item for restricted content and room access review."""

	id: str
	tenant_id: str
	subject_id: str
	subject_type: str
	status: str
	reason: str
	reviewer: str | None = None
	terms: tuple[str, ...] = ()

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"subject_type": self.subject_type,
			"status": self.status,
			"reason": self.reason,
			"reviewer": self.reviewer,
			"terms": list(self.terms),
		}


@dataclass(frozen=True)
class ChatAuditEvent:
	"""Governance event emitted by chat room, message, and moderation actions."""

	id: str
	tenant_id: str
	subject_id: str
	event_type: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = ()
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"event_type": self.event_type,
			"actor": self.actor,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"metadata": dict(self.metadata),
		}
