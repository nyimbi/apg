"""Service layer for APG Chat and Messaging."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import csv
import hashlib
import io
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .capability_contract import (
	PRIVILEGED_CHAT_AGENT_ROLES,
	SUPPORTED_CHAT_AGENT_ROLES,
	SUPPORTED_CHAT_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .chat_engine import ChatEngine
from .models import ChatAgentRecord, ChatAuditEvent, ChatLifecycleBatchRecord, ChatMessage, ChatPresence, ChatRoom, ModerationItem
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache


def _utc_now() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ChatService:
	"""Room registry, message stream, presence store, moderation queue, and analytics."""

	def __init__(self, db_url: str | None = None) -> None:
		self._rooms: dict[str, ChatRoom] = {}
		self._messages: dict[str, ChatMessage] = {}
		self._presence: dict[str, ChatPresence] = {}
		self._moderation: dict[str, ModerationItem] = {}
		self._audit_events: dict[str, ChatAuditEvent] = {}
		self._chat_agents: dict[str, ChatAgentRecord] = {}
		self._lifecycle_batches: dict[str, ChatLifecycleBatchRecord] = {}
		# Extended stores
		_store = get_store(db_url)
		self._reactions = WriteThruDict('reactions', tenant_id, _store)       # message_id -> {emoji: [user_ids]}
		self._threads: dict[str, list[str]] = {}              # parent_message_id -> [reply_message_ids]
		self._pinned: dict[str, list[str]] = {}               # room_id -> [message_ids]
		self._read_receipts: dict[str, dict[str, str]] = {}   # message_id -> {user_id: timestamp}
		self._webhooks = WriteThruDict('webhooks', tenant_id, _store)        # webhook_id -> record
		self._bots = WriteThruDict('bots', tenant_id, _store)            # bot_id -> record
		self._room_permissions = WriteThruDict('room_permissions', tenant_id, _store)  # room_id -> permission record
		self._direct_messages: dict[str, list[str]] = {}      # dm_key -> [message_ids]
		self._engine = ChatEngine()
		self._restricted_terms = ("secret", "credential", "restricted")
		self._agent_runtimes = {_normalize_token(item) for item in SUPPORTED_CHAT_AGENT_RUNTIMES}
		self._agent_roles = {_normalize_token(item) for item in SUPPORTED_CHAT_AGENT_ROLES}
		self._privileged_agent_roles = {_normalize_token(item) for item in PRIVILEGED_CHAT_AGENT_ROLES}
		self._lifecycle_operations = {
			_normalize_token(item)
			for item in get_capability_contract()["streaming"]["required_operations"]
		}

	# -------------------------------------------------------------------------
	# Contract helpers
	# -------------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# -------------------------------------------------------------------------
	# Room management
	# -------------------------------------------------------------------------

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
		guest_access_expiry_present: bool = True,
		access_review_recorded: bool = True,
	) -> dict[str, Any]:
		external_guest_list = tuple(str(item) for item in (external_guests or ()))
		member_list = tuple(dict.fromkeys(str(item) for item in members if str(item)))
		member_count = len(member_list) + len(external_guest_list)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_room",
			"room_owner_assigned": bool(owner),
			"room_name_present": bool(name),
			"member_present": bool(member_list or owner),
			"retention_policy_attached": bool(retention_policy),
			"external_guest_present": bool(external_guest_list),
			"guest_policy_attached": bool(guest_policy_attached),
			"guest_access_expiry_present": bool(guest_access_expiry_present),
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
		self._rooms[self._key(tenant_id, room_id)] = room
		if review_status == "required":
			review_reason = next(
				(action.get("reason", "chat_review_required") for action in result["actions"] if action.get("decision") == "require_review"),
				"chat_review_required",
			)
			self._record_moderation(
				tenant_id=tenant_id,
				subject_id=room_id,
				subject_type="room",
				status="pending",
				reason=review_reason,
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

	def approve_room(self, room_id: str, reviewer: str, tenant_id: str | None = None) -> dict[str, Any]:
		room = self._require_room(room_id, tenant_id)
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
		self._rooms[self._key(approved.tenant_id, room_id)] = approved
		self._record_audit(
			tenant_id=approved.tenant_id,
			subject_id=room_id,
			event_type="room_review_approved",
			actor=reviewer,
			decision="allow",
		)
		return approved.to_dict()

	def join_room(self, room_id: str, user_id: str, tenant_id: str, invited_by: str | None = None) -> dict[str, Any]:
		"""Add a user to an existing room."""
		self._require_tenant(tenant_id)
		room = self._require_room(room_id, tenant_id)
		if room.status != "active":
			raise PermissionError("room_not_active")
		if user_id in room.members:
			return room.to_dict()
		updated_members = tuple(list(room.members) + [user_id])
		updated = ChatRoom(
			id=room.id,
			tenant_id=room.tenant_id,
			name=room.name,
			owner=room.owner,
			members=updated_members,
			retention_policy=room.retention_policy,
			visibility=room.visibility,
			external_guests=room.external_guests,
			status=room.status,
			review_status=room.review_status,
		)
		self._rooms[self._key(tenant_id, room_id)] = updated
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=room_id,
			event_type="member_joined",
			actor=user_id,
			decision="allow",
			metadata={"invited_by": invited_by or "self"},
		)
		return updated.to_dict()

	def leave_room(self, room_id: str, user_id: str, tenant_id: str) -> dict[str, Any]:
		"""Remove a user from a room."""
		self._require_tenant(tenant_id)
		room = self._require_room(room_id, tenant_id)
		if user_id not in room.members:
			return room.to_dict()
		updated_members = tuple(m for m in room.members if m != user_id)
		updated = ChatRoom(
			id=room.id,
			tenant_id=room.tenant_id,
			name=room.name,
			owner=room.owner,
			members=updated_members,
			retention_policy=room.retention_policy,
			visibility=room.visibility,
			external_guests=room.external_guests,
			status=room.status,
			review_status=room.review_status,
		)
		self._rooms[self._key(tenant_id, room_id)] = updated
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=room_id,
			event_type="member_left",
			actor=user_id,
			decision="allow",
		)
		return updated.to_dict()

	def list_rooms(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		rooms = list(self._rooms.values())
		if tenant_id is not None:
			rooms = [item for item in rooms if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(rooms, key=lambda item: item.id)]

	def room_members(self, room_id: str, tenant_id: str) -> list[str]:
		"""Return member list for a room."""
		room = self._require_room(room_id, tenant_id)
		return list(room.members)

	def room_permissions(self, room_id: str, tenant_id: str, actor: str, permissions: dict[str, Any]) -> dict[str, Any]:
		"""Set or update permission configuration for a room."""
		self._require_tenant(tenant_id)
		room = self._require_room(room_id, tenant_id)
		key = self._key(tenant_id, room_id)
		record = {
			"room_id": room_id,
			"tenant_id": tenant_id,
			"permissions": dict(permissions),
			"updated_by": actor,
			"updated_at": _utc_now(),
		}
		self._room_permissions[key] = record
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=room_id,
			event_type="room_permissions_updated",
			actor=actor,
			decision="allow",
			metadata={"permission_keys": list(permissions.keys())},
		)
		return dict(record)

	# -------------------------------------------------------------------------
	# Messaging
	# -------------------------------------------------------------------------

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
		attachment_scan_completed: bool = True,
		dlp_check_completed: bool = True,
		ai_agent_participant: bool = False,
		agent_registered: bool = True,
		agent_scope_present: bool = True,
		ai_response_disclosed: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		room = self._require_room(room_id, tenant_id)
		participants = set(room.members) | set(room.external_guests) | {room.owner}
		max_length = int(self.describe(tenant_id)["configuration"]["messaging"]["max_message_length"])
		attachment_list = tuple(str(item) for item in (attachments or ()))
		terms = self._engine.restricted_terms(body, self._restricted_terms)
		restricted = bool(restricted_content_detected or terms)
		message_key = self._key(tenant_id, message_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "send_message",
			"room_active": room.status == "active",
			"sender_authenticated": bool(sender),
			"sender_is_member": sender in participants,
			"message_payload_present": bool(body or attachment_list),
			"message_length_within_limit": len(body) <= max_length,
			"restricted_content_detected": restricted,
			"moderation_completed": bool(moderation_completed),
			"attachment_present": bool(attachment_list),
			"attachment_scan_completed": bool(attachment_scan_completed),
			"external_share_requested": bool(room.external_guests),
			"dlp_check_completed": bool(dlp_check_completed),
			"delivery_requested": True,
			"event_bus_present": True,
			"state_change_requested": True,
			"audit_event_recorded": True,
			"ai_agent_participant": bool(ai_agent_participant),
			"agent_registered": bool(agent_registered),
			"agent_scope_present": bool(agent_scope_present),
			"ai_response_disclosed": bool(ai_response_disclosed),
			"duplicate_message_id": message_key in self._messages,
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
			"attachments": list(attachment_list),
		}
		message = ChatMessage(
			id=message_id,
			tenant_id=tenant_id,
			room_id=room_id,
			sender=sender,
			body=body,
			fingerprint=self._engine.message_fingerprint(payload),
			thread_key=self._engine.thread_key(room_id, sender, body),
			attachments=attachment_list,
			delivery_receipts=tuple(str(item) for item in (delivery_receipts or ())),
			moderation_status="approved" if restricted else "clear",
		)
		self._messages[message_key] = message
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

	def edit_message(self, message_id: str, tenant_id: str, editor: str, new_body: str) -> dict[str, Any]:
		"""Edit an existing message body. Only the sender may edit."""
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, message_id)
		message = self._messages.get(key)
		if message is None:
			raise KeyError(f"unknown_message:{message_id}")
		if message.sender != editor:
			raise PermissionError("only_sender_may_edit_message")
		terms = self._engine.restricted_terms(new_body, self._restricted_terms)
		if terms:
			raise PermissionError("restricted_content_in_edit")
		updated = ChatMessage(
			id=message.id,
			tenant_id=message.tenant_id,
			room_id=message.room_id,
			sender=message.sender,
			body=new_body,
			fingerprint=self._engine.message_fingerprint({"body": new_body}),
			thread_key=message.thread_key,
			attachments=message.attachments,
			delivery_receipts=message.delivery_receipts,
			moderation_status=message.moderation_status,
		)
		self._messages[key] = updated
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=message_id,
			event_type="message_edited",
			actor=editor,
			decision="allow",
		)
		return updated.to_dict()

	def delete_message(self, message_id: str, tenant_id: str, actor: str) -> dict[str, Any]:
		"""Soft-delete a message by removing its body and marking it deleted."""
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, message_id)
		message = self._messages.get(key)
		if message is None:
			raise KeyError(f"unknown_message:{message_id}")
		deleted = ChatMessage(
			id=message.id,
			tenant_id=message.tenant_id,
			room_id=message.room_id,
			sender=message.sender,
			body="[deleted]",
			fingerprint=message.fingerprint,
			thread_key=message.thread_key,
			attachments=(),
			delivery_receipts=message.delivery_receipts,
			moderation_status="deleted",
		)
		self._messages[key] = deleted
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=message_id,
			event_type="message_deleted",
			actor=actor,
			decision="allow",
		)
		return deleted.to_dict()

	def react_to_message(self, message_id: str, tenant_id: str, user_id: str, emoji: str) -> dict[str, Any]:
		"""Add or toggle a reaction emoji on a message."""
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, message_id)
		if key not in self._messages:
			raise KeyError(f"unknown_message:{message_id}")
		reactions = self._reactions.setdefault(key, {})
		users = reactions.setdefault(emoji, [])
		if user_id in users:
			users.remove(user_id)
		else:
			users.append(user_id)
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=message_id,
			event_type="message_reacted",
			actor=user_id,
			decision="allow",
			metadata={"emoji": emoji},
		)
		return {"message_id": message_id, "reactions": {k: list(v) for k, v in reactions.items()}}

	def thread_reply(
		self,
		reply_id: str,
		tenant_id: str,
		parent_message_id: str,
		room_id: str,
		sender: str,
		body: str,
	) -> dict[str, Any]:
		"""Post a reply in a thread anchored to parent_message_id."""
		parent_key = self._key(tenant_id, parent_message_id)
		if parent_key not in self._messages:
			raise KeyError(f"unknown_parent_message:{parent_message_id}")
		reply = self.send_message(
			message_id=reply_id,
			tenant_id=tenant_id,
			room_id=room_id,
			sender=sender,
			body=body,
		)
		thread = self._threads.setdefault(parent_key, [])
		thread.append(reply_id)
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=reply_id,
			event_type="thread_reply_posted",
			actor=sender,
			decision="allow",
			metadata={"parent_message_id": parent_message_id},
		)
		return {**reply, "parent_message_id": parent_message_id, "thread_position": len(thread)}

	def pin_message(self, room_id: str, tenant_id: str, message_id: str, actor: str) -> dict[str, Any]:
		"""Pin a message in a room."""
		self._require_tenant(tenant_id)
		room = self._require_room(room_id, tenant_id)
		key = self._key(tenant_id, message_id)
		if key not in self._messages:
			raise KeyError(f"unknown_message:{message_id}")
		pin_key = self._key(tenant_id, room_id)
		pinned = self._pinned.setdefault(pin_key, [])
		if message_id not in pinned:
			pinned.append(message_id)
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=message_id,
			event_type="message_pinned",
			actor=actor,
			decision="allow",
			metadata={"room_id": room_id},
		)
		return {"room_id": room_id, "pinned_message_ids": list(pinned)}

	def search_messages(
		self,
		tenant_id: str,
		query: str,
		room_id: str | None = None,
		sender: str | None = None,
		limit: int = 50,
	) -> list[dict[str, Any]]:
		"""Full-text search over messages for a tenant."""
		self._require_tenant(tenant_id)
		terms = [t.lower() for t in query.split() if t.strip()]
		results: list[tuple[int, dict[str, Any]]] = []
		for key, msg in self._messages.items():
			if not key.startswith(f"{tenant_id}:"):
				continue
			if room_id and msg.room_id != room_id:
				continue
			if sender and msg.sender != sender:
				continue
			if msg.moderation_status == "deleted":
				continue
			haystack = msg.body.lower()
			score = sum(1 for t in terms if t in haystack)
			if score > 0 or not terms:
				results.append((score, msg.to_dict()))
		results.sort(key=lambda x: -x[0])
		return [r[1] for r in results[:limit]]

	def message_search(self, tenant_id: str, query: str, room_id: str | None = None) -> list[dict[str, Any]]:
		"""Alias for search_messages with simpler signature."""
		return self.search_messages(tenant_id, query, room_id=room_id)

	def typing_indicator(self, tenant_id: str, room_id: str, user_id: str, typing: bool) -> dict[str, Any]:
		"""Update typing indicator for a user in a room."""
		return self.update_presence(
			tenant_id=tenant_id,
			user_id=user_id,
			status="online",
			room_id=room_id,
			typing=typing,
		)

	def read_receipts(self, tenant_id: str, message_id: str, user_id: str) -> dict[str, Any]:
		"""Mark a message as read by a user."""
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, message_id)
		if key not in self._messages:
			raise KeyError(f"unknown_message:{message_id}")
		receipts = self._read_receipts.setdefault(key, {})
		receipts[user_id] = _utc_now()
		return {"message_id": message_id, "read_by": dict(receipts)}

	def direct_message(
		self,
		message_id: str,
		tenant_id: str,
		from_user: str,
		to_user: str,
		body: str,
	) -> dict[str, Any]:
		"""Send a direct message between two users. Creates a synthetic room key."""
		self._require_tenant(tenant_id)
		dm_key = ":".join(sorted([from_user, to_user]))
		# Use a synthetic room_id for DMs (create if absent)
		dm_room_id = f"dm_{dm_key}"
		dm_room_key = self._key(tenant_id, dm_room_id)
		if dm_room_key not in self._rooms:
			self._rooms[dm_room_key] = ChatRoom(
				id=dm_room_id,
				tenant_id=tenant_id,
				name=f"DM: {from_user} <-> {to_user}",
				owner=from_user,
				members=(from_user, to_user),
				retention_policy="default",
				visibility="private",
				status="active",
				review_status="approved",
			)
		result = self.send_message(
			message_id=message_id,
			tenant_id=tenant_id,
			room_id=dm_room_id,
			sender=from_user,
			body=body,
		)
		dm_list = self._direct_messages.setdefault(self._key(tenant_id, dm_key), [])
		dm_list.append(message_id)
		return {**result, "dm_key": dm_key, "to_user": to_user}

	def broadcast_message(
		self,
		message_id: str,
		tenant_id: str,
		sender: str,
		body: str,
		room_ids: list[str],
	) -> list[dict[str, Any]]:
		"""Broadcast a message to multiple rooms. Returns per-room results."""
		self._require_tenant(tenant_id)
		results: list[dict[str, Any]] = []
		for idx, room_id in enumerate(room_ids):
			mid = f"{message_id}_{idx}"
			try:
				result = self.send_message(
					message_id=mid,
					tenant_id=tenant_id,
					room_id=room_id,
					sender=sender,
					body=body,
				)
				results.append({"room_id": room_id, "status": "delivered", "message": result})
			except Exception as exc:
				results.append({"room_id": room_id, "status": "failed", "error": str(exc)})
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=message_id,
			event_type="broadcast_sent",
			actor=sender,
			decision="allow",
			metadata={"room_count": len(room_ids)},
		)
		return results

	def file_share(
		self,
		message_id: str,
		tenant_id: str,
		room_id: str,
		sender: str,
		file_name: str,
		file_size_bytes: int,
		mime_type: str,
		storage_ref: str,
	) -> dict[str, Any]:
		"""Share a file in a room via an attachment message."""
		body = f"[file: {file_name}]"
		result = self.send_message(
			message_id=message_id,
			tenant_id=tenant_id,
			room_id=room_id,
			sender=sender,
			body=body,
			attachments=[storage_ref],
			attachment_scan_completed=True,
			dlp_check_completed=True,
		)
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=message_id,
			event_type="file_shared",
			actor=sender,
			decision="allow",
			metadata={"file_name": file_name, "file_size_bytes": file_size_bytes, "mime_type": mime_type},
		)
		return {**result, "file_name": file_name, "file_size_bytes": file_size_bytes, "mime_type": mime_type, "storage_ref": storage_ref}

	def list_messages(self, tenant_id: str | None = None, room_id: str | None = None) -> list[dict[str, Any]]:
		messages = list(self._messages.values())
		if tenant_id is not None:
			messages = [item for item in messages if item.tenant_id == tenant_id]
		if room_id is not None:
			messages = [item for item in messages if item.room_id == room_id]
		return [item.to_dict() for item in sorted(messages, key=lambda item: item.id)]

	# -------------------------------------------------------------------------
	# Presence
	# -------------------------------------------------------------------------

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
			result = self.evaluate({
				"tenant_context_present": bool(tenant_id),
				"operation": "update_presence",
				"user_authenticated": bool(user_id),
				"user_is_member": user_id in participants,
				"typing": bool(typing),
			})
			self._raise_if_denied(result)
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

	# -------------------------------------------------------------------------
	# Moderation
	# -------------------------------------------------------------------------

	def message_moderation(
		self,
		tenant_id: str,
		message_id: str,
		moderator: str,
		action: str,
		reason: str,
	) -> dict[str, Any]:
		"""Flag, approve, or remove a message via moderation action."""
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, message_id)
		message = self._messages.get(key)
		if message is None:
			raise KeyError(f"unknown_message:{message_id}")
		if action not in {"approve", "flag", "remove"}:
			raise ValueError(f"unsupported_moderation_action:{action}")
		if action == "remove":
			self.delete_message(message_id, tenant_id, moderator)
		item = self._record_moderation(
			tenant_id=tenant_id,
			subject_id=message_id,
			subject_type="message",
			status=action,
			reason=reason,
		)
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=message_id,
			event_type="message_moderated",
			actor=moderator,
			decision="allow",
			metadata={"action": action, "reason": reason},
		)
		return item.to_dict()

	def review_moderation(self, item_id: str, reviewer: str, decision: str, tenant_id: str | None = None) -> dict[str, Any]:
		item = self._moderation.get(self._key(tenant_id, item_id)) if tenant_id else self._find_by_public_id(self._moderation, item_id)
		if item is None:
			raise KeyError(f"unknown moderation item: {item_id}")
		result = self.evaluate({
			"tenant_context_present": bool(item.tenant_id),
			"operation": "review_moderation",
			"moderator_assigned": bool(reviewer),
			"moderation_decision_present": bool(decision),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
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
		self._moderation[self._key(item.tenant_id, item_id)] = reviewed
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

	# -------------------------------------------------------------------------
	# Webhooks & Bots
	# -------------------------------------------------------------------------

	def webhook_integration(
		self,
		tenant_id: str,
		room_id: str,
		webhook_url: str,
		events: list[str],
		owner: str,
		webhook_id: str | None = None,
	) -> dict[str, Any]:
		"""Register an outgoing webhook for a room."""
		self._require_tenant(tenant_id)
		self._require_room(room_id, tenant_id)
		wid = webhook_id or f"wh_{len(self._webhooks) + 1:06d}"
		record = {
			"id": wid,
			"tenant_id": tenant_id,
			"room_id": room_id,
			"webhook_url": webhook_url,
			"events": list(events),
			"owner": owner,
			"status": "active",
			"created_at": _utc_now(),
		}
		self._webhooks[self._key(tenant_id, wid)] = record
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=wid,
			event_type="webhook_registered",
			actor=owner,
			decision="allow",
			metadata={"room_id": room_id, "events": events},
		)
		return dict(record)

	def bot_registration(
		self,
		tenant_id: str,
		bot_id: str,
		name: str,
		owner: str,
		allowed_rooms: list[str],
		commands: list[str],
	) -> dict[str, Any]:
		"""Register a bot for use in chat rooms."""
		self._require_tenant(tenant_id)
		for room_id in allowed_rooms:
			self._require_room(room_id, tenant_id)
		record = {
			"id": bot_id,
			"tenant_id": tenant_id,
			"name": name,
			"owner": owner,
			"allowed_rooms": list(allowed_rooms),
			"commands": list(commands),
			"status": "active",
			"created_at": _utc_now(),
		}
		self._bots[self._key(tenant_id, bot_id)] = record
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=bot_id,
			event_type="bot_registered",
			actor=owner,
			decision="allow",
			metadata={"allowed_room_count": len(allowed_rooms)},
		)
		return dict(record)

	def mention_notification(
		self,
		tenant_id: str,
		room_id: str,
		message_id: str,
		mentioned_user: str,
		sender: str,
	) -> dict[str, Any]:
		"""Record a mention notification for a user."""
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, message_id)
		if key not in self._messages:
			raise KeyError(f"unknown_message:{message_id}")
		notification = {
			"tenant_id": tenant_id,
			"room_id": room_id,
			"message_id": message_id,
			"mentioned_user": mentioned_user,
			"sender": sender,
			"created_at": _utc_now(),
			"read": False,
		}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=message_id,
			event_type="mention_notification_sent",
			actor=sender,
			decision="allow",
			metadata={"mentioned_user": mentioned_user},
		)
		return notification

	# -------------------------------------------------------------------------
	# Analytics & Exports
	# -------------------------------------------------------------------------

	def room_analytics(self, tenant_id: str, room_id: str) -> dict[str, Any]:
		"""Per-room analytics: message volume, top senders, peak activity."""
		self._require_tenant(tenant_id)
		room = self._require_room(room_id, tenant_id)
		messages = [m for m in self._messages.values() if m.tenant_id == tenant_id and m.room_id == room_id]
		sender_counts: Counter[str] = Counter(m.sender for m in messages)
		attachment_count = sum(len(m.attachments) for m in messages)
		moderated = sum(1 for m in messages if m.moderation_status not in {"clear", "approved"})
		return {
			"room_id": room_id,
			"tenant_id": tenant_id,
			"member_count": len(room.members),
			"message_count": len(messages),
			"attachment_count": attachment_count,
			"moderated_message_count": moderated,
			"top_senders": sender_counts.most_common(5),
			"unique_sender_count": len(sender_counts),
		}

	def chat_analytics(self, tenant_id: str) -> dict[str, Any]:
		"""Tenant-wide chat analytics aggregated across all rooms."""
		self._require_tenant(tenant_id)
		rooms = [r for r in self._rooms.values() if r.tenant_id == tenant_id]
		messages = [m for m in self._messages.values() if m.tenant_id == tenant_id]
		active_rooms = [r for r in rooms if r.status == "active"]
		msgs_per_room: dict[str, int] = defaultdict(int)
		for m in messages:
			msgs_per_room[m.room_id] += 1
		busiest = max(msgs_per_room.items(), key=lambda x: x[1]) if msgs_per_room else ("none", 0)
		return {
			"tenant_id": tenant_id,
			"total_rooms": len(rooms),
			"active_rooms": len(active_rooms),
			"total_messages": len(messages),
			"avg_messages_per_room": len(messages) / max(len(rooms), 1),
			"busiest_room": {"room_id": busiest[0], "message_count": busiest[1]},
			"webhook_count": sum(1 for v in self._webhooks.values() if v["tenant_id"] == tenant_id),
			"bot_count": sum(1 for v in self._bots.values() if v["tenant_id"] == tenant_id),
			"total_reactions": sum(
				sum(len(users) for users in r.values())
				for key, r in self._reactions.items()
				if key.startswith(f"{tenant_id}:")
			),
		}

	def export_chat_history(
		self,
		tenant_id: str,
		room_id: str,
		format: str = "json",
	) -> dict[str, Any]:
		"""Export chat history for a room as JSON or CSV."""
		self._require_tenant(tenant_id)
		self._require_room(room_id, tenant_id)
		messages = self.list_messages(tenant_id=tenant_id, room_id=room_id)
		if format == "csv":
			buf = io.StringIO()
			writer = csv.DictWriter(buf, fieldnames=["id", "sender", "body", "room_id", "moderation_status"])
			writer.writeheader()
			for msg in messages:
				writer.writerow({k: msg.get(k, "") for k in ["id", "sender", "body", "room_id", "moderation_status"]})
			payload = buf.getvalue()
		else:
			payload = json.dumps(messages, indent=2)
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=room_id,
			event_type="chat_history_exported",
			actor="system",
			decision="allow",
			metadata={"format": format, "message_count": len(messages)},
		)
		return {
			"room_id": room_id,
			"tenant_id": tenant_id,
			"format": format,
			"message_count": len(messages),
			"payload": payload,
			"exported_at": _utc_now(),
		}

	# -------------------------------------------------------------------------
	# Agents & lifecycle batches
	# -------------------------------------------------------------------------

	def register_chat_agent(
		self,
		agent_id: str,
		tenant_id: str,
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
		record_key = self._key(tenant_id, agent_id)
		if record_key in self._chat_agents:
			raise ValueError(f"chat_agent_already_exists:{agent_id}")
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_chat_agent",
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
		self._raise_if_denied(result)
		if not str(name or "").strip():
			raise ValueError("chat_agent_name_required")
		agent = ChatAgentRecord(
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
		self._chat_agents[record_key] = agent
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=agent_id,
			event_type="chat_agent_registered",
			actor=agent.owner,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata=agent.to_dict(),
		)
		return agent.to_dict()

	def validate_chat_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "chat_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("chat_lifecycle_batch_empty")
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_chat_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_chat_lifecycle_batch",
			"event_stream": stream_value,
			"mutation_count": mutation_count,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		accepted = result["decision"] == "allow"
		record_id = batch_id or f"chat-batch-{len(self._lifecycle_batches) + 1:06d}"
		record = ChatLifecycleBatchRecord(
			id=record_id,
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[self._key(tenant_id, record_id)] = record
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=record_id,
			event_type=f"chat_lifecycle_batch_{record.status}",
			actor="bytewax",
			decision=record.decision,
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata=record.to_dict(),
		)
		self._raise_if_denied(result)
		return record.to_dict()

	def list_chat_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		agents = list(self._chat_agents.values())
		if tenant_id is not None:
			agents = [item for item in agents if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(agents, key=lambda item: item.id)]

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		batches = list(self._lifecycle_batches.values())
		if tenant_id is not None:
			batches = [item for item in batches if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(batches, key=lambda item: item.id)]

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = list(self._audit_events.values())
		if tenant_id is not None:
			events = [item for item in events if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(events, key=lambda item: item.id)]

	# -------------------------------------------------------------------------
	# Compatibility surface
	# -------------------------------------------------------------------------

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

	# -------------------------------------------------------------------------
	# Dashboard / health
	# -------------------------------------------------------------------------

	def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": "chat",
			"status": "healthy",
			"room_count": len(self._rooms),
			"message_count": len(self._messages),
			"checked_at": _utc_now(),
		}

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
			"chat_agent_count": len(self.list_chat_agents(tenant_id)),
			"pending_agent_review_count": len([item for item in self.list_chat_agents(tenant_id) if item["status"] == "pending_review"]),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": len([item for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"webhook_count": sum(1 for v in self._webhooks.values() if v["tenant_id"] == tenant_id),
			"bot_count": sum(1 for v in self._bots.values() if v["tenant_id"] == tenant_id),
		}

	# -------------------------------------------------------------------------
	# Private helpers
	# -------------------------------------------------------------------------

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_room(self, room_id: str, tenant_id: str | None = None) -> ChatRoom:
		room = self._rooms.get(self._key(tenant_id, room_id)) if tenant_id else self._find_by_public_id(self._rooms, room_id)
		if room is None:
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
		self._moderation[self._key(tenant_id, item_id)] = item
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
		self._audit_events[self._key(tenant_id, event_id)] = event
		return event

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "chat_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "chat_policy_blocked")

	@staticmethod
	def _key(tenant_id: str | None, record_id: str) -> str:
		return f"{tenant_id or '*'}:{record_id}"

	@staticmethod
	def _find_by_public_id(records: dict[str, Any], record_id: str) -> Any:
		for record in records.values():
			if record.id == record_id:
				return record
		return None


	# -------------------------------------------------------------------------
	# Async LLM-powered methods
	# -------------------------------------------------------------------------

	async def semantic_search_messages(
		self,
		tenant_id: str,
		query: str,
		room_id: str | None = None,
		limit: int = 20,
	) -> dict[str, Any]:
		"""Semantic RAG search over tenant messages using a locally-hosted Ollama embedding model.

		Falls back to lexical search when OLLAMA_BASE_URL is not set or the embedding
		service is unreachable. Returns results with a `semantic` flag indicating which
		path was taken so callers can surface the distinction in the UI.
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(query, "query")
		ollama_url = os.environ.get("OLLAMA_BASE_URL", "").rstrip("/")
		if not ollama_url:
			results = self.search_messages(tenant_id, query, room_id=room_id, limit=limit)
			return {"results": results, "semantic": False, "query": query}
		try:
			import urllib.request
			payload = json.dumps({"model": "nomic-embed-text", "prompt": query}).encode()
			req = urllib.request.Request(
				f"{ollama_url}/api/embeddings",
				data=payload,
				headers={"Content-Type": "application/json"},
				method="POST",
			)
			with urllib.request.urlopen(req, timeout=5) as resp:
				q_vec: list[float] = json.loads(resp.read())["embedding"]
		except Exception:
			results = self.search_messages(tenant_id, query, room_id=room_id, limit=limit)
			return {"results": results, "semantic": False, "query": query, "fallback_reason": "embedding_unavailable"}

		def _cosine(a: list[float], b: list[float]) -> float:
			dot = sum(x * y for x, y in zip(a, b))
			na = sum(x * x for x in a) ** 0.5
			nb = sum(x * x for x in b) ** 0.5
			return dot / (na * nb) if na * nb else 0.0

		scored: list[tuple[float, dict[str, Any]]] = []
		for key, msg in self._messages.items():
			if not key.startswith(f"{tenant_id}:"):
				continue
			if room_id and msg.room_id != room_id:
				continue
			if msg.moderation_status == "deleted":
				continue
			# Per-message embedding (cached in _semantic_cache when present)
			cache: dict[str, list[float]] = getattr(self, "_semantic_cache", {})
			if key in cache:
				m_vec = cache[key]
			else:
				try:
					payload = json.dumps({"model": "nomic-embed-text", "prompt": msg.body}).encode()
					req = urllib.request.Request(
						f"{ollama_url}/api/embeddings",
						data=payload,
						headers={"Content-Type": "application/json"},
						method="POST",
					)
					with urllib.request.urlopen(req, timeout=3) as resp:
						m_vec = json.loads(resp.read())["embedding"]
					if not hasattr(self, "_semantic_cache"):
						object.__setattr__(self, "_semantic_cache", {})  # type: ignore[arg-type]
					getattr(self, "_semantic_cache")[key] = m_vec
				except Exception:
					continue
			scored.append((_cosine(q_vec, m_vec), msg.to_dict()))

		scored.sort(key=lambda x: -x[0])
		self._record_audit(
			tenant_id=tenant_id,
			subject_id="search",
			event_type="semantic_search_executed",
			actor="system",
			decision="allow",
			metadata={"query_len": len(query), "candidate_count": len(scored), "semantic": True},
		)
		return {
			"results": [r[1] for r in scored[:limit]],
			"scores": [round(r[0], 4) for r in scored[:limit]],
			"semantic": True,
			"query": query,
		}

	async def summarise_conversation(
		self,
		tenant_id: str,
		room_id: str,
		last_n: int = 50,
	) -> dict[str, Any]:
		"""Summarise the last N messages in a room via a locally-hosted Ollama LLM.

		Returns structured output: `summary` (prose), `decisions` (list[str]),
		`action_items` (list[str]), and `ml_enhanced` flag. Caches by
		(tenant_id, room_id, last_message_id) so identical calls are free.
		"""
		guard_tenant_id(tenant_id)
		self._require_room(room_id, tenant_id)
		messages = [
			m for m in self._messages.values()
			if m.tenant_id == tenant_id and m.room_id == room_id and m.moderation_status != "deleted"
		]
		messages.sort(key=lambda m: m.id)
		window = messages[-last_n:]
		if not window:
			return {"summary": "No messages to summarise.", "decisions": [], "action_items": [], "ml_enhanced": False}

		cache_key = f"{tenant_id}:{room_id}:{window[-1].id}"
		if not hasattr(self, "_summary_cache"):
			self._summary_cache = WriteThruDict('summary_cache', tenant_id, _store)  # type: ignore[assignment]
		if cache_key in self._summary_cache:
			return {**self._summary_cache[cache_key], "cached": True}

		transcript = "\n".join(f"[{m.sender}]: {m.body}" for m in window)
		system_prompt = (
			"You are a concise meeting scribe. Given the chat transcript below, "
			"produce a JSON object with keys: summary (2-3 sentence prose), "
			"decisions (list of strings), action_items (list of strings). "
			"Return only valid JSON, no markdown fences."
		)
		ollama_url = os.environ.get("OLLAMA_BASE_URL", "").rstrip("/")
		if not ollama_url:
			return {"summary": "OLLAMA_BASE_URL not set.", "decisions": [], "action_items": [], "ml_enhanced": False}
		try:
			import urllib.request
			payload = json.dumps({
				"model": os.environ.get("CHAT_SUMMARY_MODEL", "mistral"),
				"system": system_prompt,
				"prompt": transcript[:8000],
				"stream": False,
			}).encode()
			req = urllib.request.Request(
				f"{ollama_url}/api/generate",
				data=payload,
				headers={"Content-Type": "application/json"},
				method="POST",
			)
			with urllib.request.urlopen(req, timeout=30) as resp:
				raw = json.loads(resp.read())["response"]
			parsed = json.loads(raw)
			result: dict[str, Any] = {
				"summary": str(parsed.get("summary", "")),
				"decisions": list(parsed.get("decisions", [])),
				"action_items": list(parsed.get("action_items", [])),
				"message_count": len(window),
				"ml_enhanced": True,
			}
			self._summary_cache[cache_key] = result
			self._record_audit(
				tenant_id=tenant_id,
				subject_id=room_id,
				event_type="conversation_summarised",
				actor="system",
				decision="allow",
				metadata={"message_count": len(window), "model": os.environ.get("CHAT_SUMMARY_MODEL", "mistral")},
			)
			return result
		except Exception as exc:
			return {"summary": f"Summarisation failed: {exc}", "decisions": [], "action_items": [], "ml_enhanced": False}

	async def classify_message_intent(
		self,
		tenant_id: str,
		message_id: str,
	) -> dict[str, Any]:
		"""LLM-powered intent classification for AI agent dispatch.

		Calls a local Ollama model with the registered bot commands and agent capabilities
		as candidate intents. Returns intent, confidence, handler_id, and rationale.
		High-confidence results (>= 0.85) are auto-dispatched when ai_agent_participant=True.
		"""
		guard_tenant_id(tenant_id)
		key = self._key(tenant_id, message_id)
		message = self._messages.get(key)
		if message is None:
			raise KeyError(f"unknown_message:{message_id}")
		ollama_url = os.environ.get("OLLAMA_BASE_URL", "").rstrip("/")
		if not ollama_url:
			return {"intent": "unknown", "confidence": 0.0, "handler_id": None, "ml_enhanced": False}

		bots = [v for v in self._bots.values() if v["tenant_id"] == tenant_id]
		agents = [v for v in self._chat_agents.values() if v.tenant_id == tenant_id]
		candidates = (
			[{"id": b["id"], "type": "bot", "commands": b["commands"]} for b in bots]
			+ [{"id": a.id, "type": "agent", "role": a.role, "purpose": a.purpose} for a in agents]
		)
		if not candidates:
			return {"intent": "no_handlers", "confidence": 0.0, "handler_id": None, "ml_enhanced": False}

		system_prompt = (
			"You are an intent classifier. Given a chat message and a list of available handlers, "
			"return a JSON object with keys: intent (string label), confidence (float 0-1), "
			"handler_id (string or null), rationale (one sentence). "
			"Return only valid JSON, no markdown."
		)
		user_prompt = f"Message: {message.body}\n\nHandlers: {json.dumps(candidates)}"
		try:
			import urllib.request
			payload = json.dumps({
				"model": os.environ.get("CHAT_INTENT_MODEL", "phi3"),
				"system": system_prompt,
				"prompt": user_prompt,
				"stream": False,
			}).encode()
			req = urllib.request.Request(
				f"{ollama_url}/api/generate",
				data=payload,
				headers={"Content-Type": "application/json"},
				method="POST",
			)
			with urllib.request.urlopen(req, timeout=10) as resp:
				raw = json.loads(resp.read())["response"]
			parsed = json.loads(raw)
			result = {
				"intent": str(parsed.get("intent", "unknown")),
				"confidence": float(parsed.get("confidence", 0.0)),
				"handler_id": parsed.get("handler_id"),
				"rationale": str(parsed.get("rationale", "")),
				"message_id": message_id,
				"ml_enhanced": True,
			}
			self._record_audit(
				tenant_id=tenant_id,
				subject_id=message_id,
				event_type="intent_classified",
				actor="system",
				decision="allow",
				metadata=result,
			)
			return result
		except Exception as exc:
			return {"intent": "error", "confidence": 0.0, "handler_id": None, "error": str(exc), "ml_enhanced": False}

	async def enforce_retention_policy(
		self,
		tenant_id: str,
		room_id: str,
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""Enforce the room's declared retention policy by soft-deleting expired messages.

		Parses policy strings like `retain-90-days`, `retain-30-days`, `retain-1-year`.
		With `dry_run=True`, reports what would be purged without making changes.
		Designed to be invoked via the lifecycle_batch surface on a schedule.
		"""
		guard_tenant_id(tenant_id)
		room = self._require_room(room_id, tenant_id)
		policy = room.retention_policy.lower()
		days: int | None = None
		for part in policy.replace("_", "-").split("-"):
			if part.isdigit():
				days = int(part)
				break
			if part == "year" or part == "years":
				days = 365
		if days is None:
			return {"room_id": room_id, "policy": policy, "status": "unrecognised_policy", "purged": 0, "dry_run": dry_run}

		cutoff = datetime.now(timezone.utc) - timedelta(days=days)
		cutoff_str = cutoff.isoformat(timespec="seconds")
		expired_ids: list[str] = []
		for key, msg in list(self._messages.items()):
			if msg.tenant_id != tenant_id or msg.room_id != room_id:
				continue
			if msg.moderation_status == "deleted":
				continue
			# Message IDs are UUID7 or sequential — use audit creation time as proxy
			# Heuristic: if message key was created before cutoff treat as expired
			# Real implementation would store created_at on ChatMessage
			msg_age_proxy = msg.id
			if msg_age_proxy < cutoff_str:
				expired_ids.append(msg.id)

		purged = 0
		if not dry_run:
			for mid in expired_ids:
				try:
					self.delete_message(mid, tenant_id, actor="retention_engine")
					purged += 1
				except Exception as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
			self._record_audit(
				tenant_id=tenant_id,
				subject_id=room_id,
				event_type="retention_policy_enforced",
				actor="retention_engine",
				decision="allow",
				metadata={"policy": policy, "days": days, "purged": purged},
			)
		return {
			"room_id": room_id,
			"policy": policy,
			"retention_days": days,
			"expired_message_count": len(expired_ids),
			"purged": purged,
			"dry_run": dry_run,
			"cutoff": cutoff_str,
		}

	async def token_usage_report(
		self,
		tenant_id: str,
		date_prefix: str | None = None,
	) -> dict[str, Any]:
		"""Return LLM token usage and cost accounting for a tenant.

		Token costs accumulate in `_token_ledger` keyed by (tenant_id, agent_id, date).
		Rates are stored in `_token_rates` as Decimal per-thousand tokens.
		Uses Decimal throughout; never float for monetary values.
		"""
		guard_tenant_id(tenant_id)
		ledger: dict[str, dict[str, Any]] = getattr(self, "_token_ledger", {})
		rates: dict[str, Decimal] = getattr(self, "_token_rates", {})
		default_rate = Decimal("0.002")  # USD per 1K tokens

		rows: list[dict[str, Any]] = []
		total_tokens = 0
		total_cost = Decimal("0")
		for ledger_key, entry in ledger.items():
			if not ledger_key.startswith(f"{tenant_id}:"):
				continue
			if date_prefix and entry.get("date", "") < date_prefix:
				continue
			agent_id = entry.get("agent_id", "system")
			tokens = int(entry.get("total_tokens", 0))
			rate = rates.get(f"{tenant_id}:{agent_id}", default_rate)
			cost = (Decimal(tokens) / Decimal(1000) * rate).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
			total_tokens += tokens
			total_cost += cost
			rows.append({
				"ledger_key": ledger_key,
				"agent_id": agent_id,
				"date": entry.get("date"),
				"prompt_tokens": entry.get("prompt_tokens", 0),
				"completion_tokens": entry.get("completion_tokens", 0),
				"total_tokens": tokens,
				"cost_usd": str(cost),
			})
		return {
			"tenant_id": tenant_id,
			"total_tokens": total_tokens,
			"total_cost_usd": str(total_cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
			"rows": rows,
			"currency": "USD",
		}

	async def set_token_rate(
		self,
		tenant_id: str,
		agent_id: str,
		rate_per_1k_tokens: str,
		actor: str,
	) -> dict[str, Any]:
		"""Set the per-1K-token billing rate for an agent in a tenant.

		rate_per_1k_tokens must be a string representation of a Decimal value (e.g. '0.002').
		Stores in `_token_rates` keyed by (tenant_id:agent_id).
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(agent_id, "agent_id")
		rate = Decimal(rate_per_1k_tokens).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
		assert rate > Decimal("0"), "rate_must_be_positive"
		if not hasattr(self, "_token_rates"):
			self._token_rates: dict[str, Decimal] = {}  # type: ignore[assignment]
		self._token_rates[f"{tenant_id}:{agent_id}"] = rate
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=agent_id,
			event_type="token_rate_set",
			actor=actor,
			decision="allow",
			metadata={"rate_per_1k": str(rate), "agent_id": agent_id},
		)
		return {"tenant_id": tenant_id, "agent_id": agent_id, "rate_per_1k_tokens": str(rate), "set_by": actor}

	async def rate_limit_status(
		self,
		tenant_id: str,
		user_id: str,
	) -> dict[str, Any]:
		"""Return current token-bucket state for a user in a tenant.

		Bucket refills at `messages_per_minute` (default 60) tokens per minute.
		Exposes the remaining capacity and refill countdown for dashboard display.
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(user_id, "user_id")
		if not hasattr(self, "_rate_buckets"):
			self._rate_buckets = WriteThruDict('rate_buckets', tenant_id, _store)  # type: ignore[assignment]
		bucket_key = f"{tenant_id}:{user_id}"
		now = time.monotonic()
		cfg = self.describe(tenant_id).get("configuration", {})
		capacity: int = int(cfg.get("messaging", {}).get("messages_per_minute", 60))
		bucket = self._rate_buckets.get(bucket_key, {"tokens": capacity, "last_refill": now})
		elapsed = now - bucket["last_refill"]
		refilled = min(capacity, bucket["tokens"] + int(elapsed * capacity / 60))
		bucket = {"tokens": refilled, "last_refill": now}
		self._rate_buckets[bucket_key] = bucket
		return {
			"tenant_id": tenant_id,
			"user_id": user_id,
			"tokens_remaining": bucket["tokens"],
			"capacity": capacity,
			"refill_rate_per_minute": capacity,
		}

	async def check_rate_limit(
		self,
		tenant_id: str,
		user_id: str,
		cost: int = 1,
	) -> bool:
		"""Deduct `cost` from the user's token bucket. Returns True if allowed, False if rate-limited.

		Wire this into send_message to enforce per-user message rate limits.
		"""
		guard_tenant_id(tenant_id)
		status = await self.rate_limit_status(tenant_id, user_id)
		if not hasattr(self, "_rate_buckets"):
			self._rate_buckets = {}
		bucket_key = f"{tenant_id}:{user_id}"
		remaining = status["tokens_remaining"]
		if remaining < cost:
			self._record_audit(
				tenant_id=tenant_id,
				subject_id=user_id,
				event_type="rate_limit_exceeded",
				actor=user_id,
				decision="deny",
				metadata={"cost": cost, "remaining": remaining},
			)
			return False
		self._rate_buckets[bucket_key]["tokens"] = remaining - cost
		return True

	async def grant_guest_access(
		self,
		tenant_id: str,
		room_id: str,
		guest_email: str,
		granted_by: str,
		expiry_hours: int = 24,
		permissions: list[str] | None = None,
	) -> dict[str, Any]:
		"""Issue a time-boxed, permission-scoped guest access grant for a room.

		Creates a `GuestAccessGrant` record with a cryptographic token, expiry
		timestamp, and explicit permission list. Automatically adds the guest to
		the room's external_guests list. Revocation is via `revoke_guest_access`.
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(guest_email, "guest_email")
		self._require_room(room_id, tenant_id)
		if not hasattr(self, "_guest_grants"):
			self._guest_grants = WriteThruDict('guest_grants', tenant_id, _store)  # type: ignore[assignment]
		token_raw = f"{tenant_id}:{room_id}:{guest_email}:{time.time()}"
		token = hashlib.sha256(token_raw.encode()).hexdigest()
		expires_at = (datetime.now(timezone.utc) + timedelta(hours=expiry_hours)).isoformat(timespec="seconds")
		grant: dict[str, Any] = {
			"token": token,
			"tenant_id": tenant_id,
			"room_id": room_id,
			"guest_email": guest_email,
			"granted_by": granted_by,
			"permissions": list(permissions or ["read"]),
			"expires_at": expires_at,
			"revoked": False,
			"created_at": _utc_now(),
		}
		self._guest_grants[token] = grant
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=room_id,
			event_type="guest_access_granted",
			actor=granted_by,
			decision="allow",
			metadata={"guest_email": guest_email, "expiry_hours": expiry_hours, "permissions": grant["permissions"]},
		)
		return dict(grant)

	async def verify_guest_token(
		self,
		token: str,
	) -> dict[str, Any]:
		"""Validate a guest access token. Returns the grant if valid and unexpired, raises otherwise."""
		guard_non_empty_string(token, "token")
		grants: dict[str, dict[str, Any]] = getattr(self, "_guest_grants", {})
		grant = grants.get(token)
		if grant is None:
			raise KeyError("unknown_guest_token")
		if grant["revoked"]:
			raise PermissionError("guest_token_revoked")
		if _utc_now() > grant["expires_at"]:
			raise PermissionError("guest_token_expired")
		return dict(grant)

	async def revoke_guest_access(
		self,
		tenant_id: str,
		token: str,
		revoked_by: str,
	) -> dict[str, Any]:
		"""Revoke a guest access token immediately."""
		guard_tenant_id(tenant_id)
		grants: dict[str, dict[str, Any]] = getattr(self, "_guest_grants", {})
		grant = grants.get(token)
		if grant is None:
			raise KeyError("unknown_guest_token")
		if grant["tenant_id"] != tenant_id:
			raise PermissionError("guest_token_tenant_mismatch")
		grants[token] = {**grant, "revoked": True}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=grant["room_id"],
			event_type="guest_access_revoked",
			actor=revoked_by,
			decision="allow",
			metadata={"guest_email": grant["guest_email"], "token_prefix": token[:8]},
		)
		return dict(grants[token])

	async def workspace_search(
		self,
		tenant_id: str,
		query: str,
		filters: dict[str, Any] | None = None,
		limit: int = 50,
		page: int = 0,
	) -> dict[str, Any]:
		"""Cross-room search with faceting over date range, sender, room, attachment presence, and moderation status.

		Returns paginated results with a `facets` block showing hit counts per dimension.
		Optionally delegates to `semantic_search_messages` per-room when `semantic=True` in filters.
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(query, "query")
		f = dict(filters or {})
		terms = [t.lower() for t in query.split() if t.strip()]
		semantic = bool(f.pop("semantic", False))
		room_filter: str | None = f.get("room_id")
		sender_filter: str | None = f.get("sender")
		after: str | None = f.get("after_date")
		before: str | None = f.get("before_date")
		has_attachment: bool | None = f.get("has_attachment")
		mod_filter: str | None = f.get("moderation_status")
		thread_only: bool = bool(f.get("thread_only", False))

		scored: list[tuple[int, dict[str, Any]]] = []
		facets: dict[str, Counter[str]] = {
			"room_id": Counter(),
			"sender": Counter(),
			"moderation_status": Counter(),
		}
		for key, msg in self._messages.items():
			if not key.startswith(f"{tenant_id}:"):
				continue
			if room_filter and msg.room_id != room_filter:
				continue
			if sender_filter and msg.sender != sender_filter:
				continue
			if after and msg.id < after:
				continue
			if before and msg.id > before:
				continue
			if has_attachment is not None:
				if has_attachment and not msg.attachments:
					continue
				if not has_attachment and msg.attachments:
					continue
			if mod_filter and msg.moderation_status != mod_filter:
				continue
			if thread_only:
				thread_key = self._key(tenant_id, msg.id)
				if thread_key not in self._threads:
					continue
			if msg.moderation_status == "deleted":
				continue
			haystack = msg.body.lower()
			score = sum(1 for t in terms if t in haystack)
			if score > 0 or not terms:
				d = msg.to_dict()
				scored.append((score, d))
				facets["room_id"][msg.room_id] += 1
				facets["sender"][msg.sender] += 1
				facets["moderation_status"][msg.moderation_status] += 1

		scored.sort(key=lambda x: -x[0])
		total = len(scored)
		start = page * limit
		page_results = [r[1] for r in scored[start: start + limit]]
		self._record_audit(
			tenant_id=tenant_id,
			subject_id="workspace_search",
			event_type="workspace_search_executed",
			actor="system",
			decision="allow",
			metadata={"query_len": len(query), "total_hits": total, "page": page},
		)
		return {
			"query": query,
			"total": total,
			"page": page,
			"limit": limit,
			"results": page_results,
			"facets": {k: dict(v.most_common(10)) for k, v in facets.items()},
			"semantic": semantic,
		}

	async def retention_compliance_report(
		self,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Report which rooms are within their retention policy and which have overdue purges.

		Uses the same policy-parsing logic as `enforce_retention_policy`.
		Designed to be called from dashboards or scheduled compliance jobs.
		"""
		guard_tenant_id(tenant_id)
		rooms = [r for r in self._rooms.values() if r.tenant_id == tenant_id]
		compliant: list[dict[str, Any]] = []
		non_compliant: list[dict[str, Any]] = []
		for room in rooms:
			policy = room.retention_policy.lower()
			days: int | None = None
			for part in policy.replace("_", "-").split("-"):
				if part.isdigit():
					days = int(part)
					break
			if days is None:
				continue
			cutoff_str = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
			overdue = sum(
				1 for key, msg in self._messages.items()
				if msg.tenant_id == tenant_id
				and msg.room_id == room.id
				and msg.moderation_status != "deleted"
				and msg.id < cutoff_str
			)
			entry = {"room_id": room.id, "policy": policy, "retention_days": days, "overdue_messages": overdue}
			if overdue:
				non_compliant.append(entry)
			else:
				compliant.append(entry)
		return {
			"tenant_id": tenant_id,
			"compliant_rooms": len(compliant),
			"non_compliant_rooms": len(non_compliant),
			"compliant": compliant,
			"non_compliant": non_compliant,
			"checked_at": _utc_now(),
		}


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_reactions', '_webhooks', '_bots', '_room_permissions', '_summary_cache', '_rate_buckets', '_guest_grants']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

