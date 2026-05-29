"""Service layer for APG Chat and Messaging."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .chat_engine import ChatEngine
from .models import ChatAuditEvent, ChatMessage, ChatPresence, ChatRoom, ModerationItem


class ChatService:
	"""Room registry, message stream, presence store, and moderation queue."""

	def __init__(self) -> None:
		self._rooms: dict[str, ChatRoom] = {}
		self._messages: dict[str, ChatMessage] = {}
		self._presence: dict[str, ChatPresence] = {}
		self._moderation: dict[str, ModerationItem] = {}
		self._audit_events: dict[str, ChatAuditEvent] = {}
		self._engine = ChatEngine()
		self._restricted_terms = ("secret", "credential", "restricted")

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_room(
		self,
		room_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		members: list[str] | tuple[str, ...],
		retention_policy: str,
		visibility: str = "private",
		external_guests: list[str] | tuple[str, ...] | None = None,
		guest_policy_attached: bool = True,
		access_review_recorded: bool = True,
	) -> dict[str, Any]:
		external_guest_list = tuple(str(item) for item in (external_guests or ()))
		member_list = tuple(dict.fromkeys(str(item) for item in members if str(item)))
		member_count = len(member_list) + len(external_guest_list)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_room",
			"room_owner_assigned": bool(owner),
			"retention_policy_attached": bool(retention_policy),
			"external_guest_present": bool(external_guest_list),
			"guest_policy_attached": bool(guest_policy_attached),
			"member_count": member_count,
			"access_review_recorded": bool(access_review_recorded),
		})
		self._raise_if_denied(result)
		max_members = int(self.describe(tenant_id)["configuration"]["rooms"]["max_members_per_room"])
		if member_count > max_members and result["decision"] != "require_review":
			raise PermissionError("max_room_members_exceeded")
		review_status = "required" if result["decision"] == "require_review" else "approved"
		status = "pending_review" if review_status == "required" else "active"
		room = ChatRoom(
			id=room_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			members=member_list,
			retention_policy=retention_policy,
			visibility=visibility,
			external_guests=external_guest_list,
			status=status,
			review_status=review_status,
		)
		self._rooms[room_id] = room
		if review_status == "required":
			self._record_moderation(
				tenant_id=tenant_id,
				subject_id=room_id,
				subject_type="room",
				status="pending",
				reason="large_room_review_required",
			)
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=room_id,
			event_type="room_created",
			actor=owner,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"member_count": member_count, "visibility": visibility},
		)
		return room.to_dict()

	def approve_room(self, room_id: str, reviewer: str) -> dict[str, Any]:
		room = self._require_room(room_id)
		if room.status != "pending_review":
			return room.to_dict()
		approved = ChatRoom(
			id=room.id,
			tenant_id=room.tenant_id,
			name=room.name,
			owner=room.owner,
			members=room.members,
			retention_policy=room.retention_policy,
			visibility=room.visibility,
			external_guests=room.external_guests,
			status="active",
			review_status="approved",
		)
		self._rooms[room_id] = approved
		self._record_audit(
			tenant_id=approved.tenant_id,
			subject_id=room_id,
			event_type="room_review_approved",
			actor=reviewer,
			decision="allow",
		)
		return approved.to_dict()

	def list_rooms(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		rooms = list(self._rooms.values())
		if tenant_id is not None:
			rooms = [item for item in rooms if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(rooms, key=lambda item: item.id)]

	def send_message(
		self,
		message_id: str,
		tenant_id: str,
		room_id: str,
		sender: str,
		body: str,
		attachments: list[str] | tuple[str, ...] | None = None,
		delivery_receipts: list[str] | tuple[str, ...] | None = None,
		restricted_content_detected: bool = False,
		moderation_completed: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		room = self._require_room(room_id, tenant_id)
		if room.status != "active":
			raise PermissionError("room_not_active")
		participants = set(room.members) | set(room.external_guests) | {room.owner}
		if sender not in participants:
			raise PermissionError("sender_not_room_member")
		max_length = int(self.describe(tenant_id)["configuration"]["messaging"]["max_message_length"])
		if len(body) > max_length:
			raise PermissionError("message_length_exceeded")
		terms = self._engine.restricted_terms(body, self._restricted_terms)
		restricted = bool(restricted_content_detected or terms)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "send_message",
			"restricted_content_detected": restricted,
			"moderation_completed": bool(moderation_completed),
		})
		if restricted and result["decision"] == "deny":
			self._record_moderation(
				tenant_id=tenant_id,
				subject_id=message_id,
				subject_type="message",
				status="pending",
				reason="moderation_required",
				terms=terms,
			)
		self._raise_if_denied(result)
		payload = {
			"id": message_id,
			"tenant_id": tenant_id,
			"room_id": room_id,
			"sender": sender,
			"body": body,
			"attachments": list(attachments or ()),
		}
		message = ChatMessage(
			id=message_id,
			tenant_id=tenant_id,
			room_id=room_id,
			sender=sender,
			body=body,
			fingerprint=self._engine.message_fingerprint(payload),
			thread_key=self._engine.thread_key(room_id, sender, body),
			attachments=tuple(str(item) for item in (attachments or ())),
			delivery_receipts=tuple(str(item) for item in (delivery_receipts or ())),
			moderation_status="approved" if restricted else "clear",
		)
		self._messages[message_id] = message
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=message_id,
			event_type="message_sent",
			actor=sender,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"room_id": room_id, "attachment_count": len(message.attachments)},
		)
		return message.to_dict()

	def list_messages(self, tenant_id: str | None = None, room_id: str | None = None) -> list[dict[str, Any]]:
		messages = list(self._messages.values())
		if tenant_id is not None:
			messages = [item for item in messages if item.tenant_id == tenant_id]
		if room_id is not None:
			messages = [item for item in messages if item.room_id == room_id]
		return [item.to_dict() for item in sorted(messages, key=lambda item: item.id)]

	def update_presence(
		self,
		tenant_id: str,
		user_id: str,
		status: str,
		room_id: str | None = None,
		typing: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if room_id:
			room = self._require_room(room_id, tenant_id)
			participants = set(room.members) | set(room.external_guests) | {room.owner}
			if user_id not in participants:
				raise PermissionError("presence_user_not_room_member")
		presence = ChatPresence(
			id=f"{tenant_id}:{user_id}:{room_id or 'global'}",
			tenant_id=tenant_id,
			user_id=user_id,
			status=status,
			room_id=room_id,
			typing=typing,
			metadata=dict(metadata or {}),
		)
		self._presence[presence.id] = presence
		return presence.to_dict()

	def list_presence(self, tenant_id: str | None = None, room_id: str | None = None) -> list[dict[str, Any]]:
		presence = list(self._presence.values())
		if tenant_id is not None:
			presence = [item for item in presence if item.tenant_id == tenant_id]
		if room_id is not None:
			presence = [item for item in presence if item.room_id == room_id]
		return [item.to_dict() for item in sorted(presence, key=lambda item: item.id)]

	def review_moderation(self, item_id: str, reviewer: str, decision: str) -> dict[str, Any]:
		item = self._moderation.get(item_id)
		if item is None:
			raise KeyError(f"unknown moderation item: {item_id}")
		reviewed = ModerationItem(
			id=item.id,
			tenant_id=item.tenant_id,
			subject_id=item.subject_id,
			subject_type=item.subject_type,
			status=decision,
			reason=item.reason,
			reviewer=reviewer,
			terms=item.terms,
		)
		self._moderation[item_id] = reviewed
		self._record_audit(
			tenant_id=item.tenant_id,
			subject_id=item.subject_id,
			event_type="moderation_reviewed",
			actor=reviewer,
			decision=decision,
			metadata={"moderation_item_id": item_id},
		)
		return reviewed.to_dict()

	def list_moderation_items(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._moderation.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = list(self._audit_events.values())
		if tenant_id is not None:
			events = [item for item in events if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(events, key=lambda item: item.id)]

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Compatibility surface exposing messages as CHAT records."""
		return self.list_messages(tenant_id)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper that records an auditable chat event."""
		self._require_tenant(tenant_id)
		event = self._record_audit(
			tenant_id=tenant_id,
			subject_id=record_id,
			event_type=str((metadata or {}).get("event_type") or "chat_note"),
			actor=str((metadata or {}).get("actor") or "system"),
			decision=status,
			metadata=dict(metadata or {}),
		)
		return event.to_dict()

	def conversation_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		rooms = self.list_rooms(tenant_id)
		messages = self.list_messages(tenant_id)
		moderation = self.list_moderation_items(tenant_id)
		return {
			"room_count": len(rooms),
			"active_room_count": len([item for item in rooms if item["status"] == "active"]),
			"pending_room_review_count": len([item for item in rooms if item["status"] == "pending_review"]),
			"message_count": len(messages),
			"attachment_count": sum(len(item["attachments"]) for item in messages),
			"presence_count": len(self.list_presence(tenant_id)),
			"moderation_queue_count": len([item for item in moderation if item["status"] == "pending"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_room(self, room_id: str, tenant_id: str | None = None) -> ChatRoom:
		room = self._rooms.get(room_id)
		if room is None or (tenant_id is not None and room.tenant_id != tenant_id):
			raise KeyError(f"unknown chat room: {room_id}")
		return room

	def _record_moderation(
		self,
		tenant_id: str,
		subject_id: str,
		subject_type: str,
		status: str,
		reason: str,
		terms: tuple[str, ...] = (),
	) -> ModerationItem:
		item_id = f"mod:{len(self._moderation) + 1:06d}"
		item = ModerationItem(
			id=item_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			subject_type=subject_type,
			status=status,
			reason=reason,
			terms=terms,
		)
		self._moderation[item_id] = item
		return item

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> ChatAuditEvent:
		event_id = f"audit:{len(self._audit_events) + 1:06d}"
		event = ChatAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
		)
		self._audit_events[event_id] = event
		return event

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "chat_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "chat_policy_blocked")
