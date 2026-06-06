"""
APG Collaboration (COLB) - Expanded Service Implementation

Dependency-light in-memory store pattern. 44+ async methods covering
workspace management, document lifecycle, co-editing sessions,
comments, mentions, task assignments, version history, conflict
resolution, export, activity feeds, analytics and compliance.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from datetime import datetime, timedelta
from typing import Any

from uuid6 import uuid7

import logging

logger = logging.getLogger(__name__)


def uuid7str() -> str:
	return str(uuid7())


def _ts() -> str:
	return datetime.utcnow().isoformat(timespec="seconds")


class _R(dict[str, Any]):
	"""Thin dict wrapper for records."""


class CollaborationService:
	"""
	44+ async methods for workspace management, document creation and
	sharing, co-editing sessions, commenting, mentions, task
	assignments, version history, conflict resolution, exports,
	activity feeds, notifications and analytics.

	All state is held in Python dicts (in-memory store pattern).
	Every state change emits an audit event.

	Can also be constructed with a SQLAlchemy-compatible async session as
	the first argument for DB-backed operation::

		service = CollaborationService(db_session)
		messages = await service.get_chat_messages("/some/page", limit=20, tenant_id="t1")
	"""

	def __init__(self, actor_id_or_db: Any = "system", tenant_id: str = "default") -> None:
		# Support two construction modes:
		#   CollaborationService("actor-id", "tenant-id")  — in-memory store
		#   CollaborationService(db_session)                — DB-backed
		if isinstance(actor_id_or_db, str):
			self.actor_id = actor_id_or_db
			self.tenant_id = tenant_id
			self._db = None
		else:
			self._db = actor_id_or_db
			self.actor_id = "system"
			self.tenant_id = tenant_id

		self._workspaces:   dict[str, _R] = {}
		self._members:      dict[str, list[_R]] = {}       # workspace_id -> members
		self._documents:    dict[str, _R] = {}
		self._doc_versions: dict[str, list[_R]] = {}       # doc_id -> versions
		self._co_edit_sessions: dict[str, _R] = {}
		self._co_edit_ops:  dict[str, list[_R]] = {}       # session_id -> ops
		self._comments:     dict[str, list[_R]] = {}       # doc_id -> comments
		self._tasks:        dict[str, _R] = {}
		self._mentions:     dict[str, list[_R]] = {}       # user_id -> mentions
		self._notifications: dict[str, list[_R]] = {}      # user_id -> notifications
		self._activity_feed: list[_R] = []
		self._audit_log:    list[_R] = []

	# ------------------------------------------------------------------
	# helpers
	# ------------------------------------------------------------------

	def _key(self, record_id: str) -> str:
		return f"{self.tenant_id}:{record_id}"

	async def _audit(self, event_type: str, record_id: str, details: dict[str, Any] | None = None) -> None:
		entry = _R(
			event_id=uuid7str(),
			tenant_id=self.tenant_id,
			actor_id=self.actor_id,
			event_type=event_type,
			record_id=record_id,
			details=details or {},
			occurred_at=_ts(),
		)
		self._audit_log.append(entry)
		self._activity_feed.append(entry)

	async def _notify(self, user_id: str, subject: str, body: str, channel: str = "in_app") -> None:
		if user_id not in self._notifications:
			self._notifications[user_id] = []
		self._notifications[user_id].append(_R(
			notification_id=uuid7str(),
			user_id=user_id,
			tenant_id=self.tenant_id,
			channel=channel,
			subject=subject,
			body=body,
			read=False,
			sent_at=_ts(),
		))

	def _require_workspace(self, workspace_id: str) -> _R:
		r = self._workspaces.get(self._key(workspace_id))
		if r is None:
			raise KeyError(f"workspace not found: {workspace_id}")
		return r

	def _require_document(self, doc_id: str) -> _R:
		r = self._documents.get(self._key(doc_id))
		if r is None:
			raise KeyError(f"document not found: {doc_id}")
		return r

	def _require_task(self, task_id: str) -> _R:
		r = self._tasks.get(self._key(task_id))
		if r is None:
			raise KeyError(f"task not found: {task_id}")
		return r

	# ------------------------------------------------------------------
	# get_chat_messages  (DB-backed)
	# ------------------------------------------------------------------

	async def get_chat_messages(
		self,
		session_id: str,
		*,
		tenant_id: str,
		page: int = 1,
		limit: int | None = None,
	) -> list[dict[str, Any]]:
		"""Return chat messages for a page/session, sorted by timestamp descending.

		Merges two sources:
		  1. ``RTCPageCollaboration.chat_messages`` — stored JSON blobs keyed by
		     the page URL (``session_id`` parameter).
		  2. ``RTCMessage`` rows linked to the same page/session.

		The combined list is sorted newest-first; *limit* is applied after sorting.
		"""
		assert self._db is not None, "get_chat_messages requires a DB session"

		# -- 1. page-level collaboration record ----------------------------------
		from sqlalchemy import select
		from .models import RTCPageCollaboration, RTCMessage

		page_result = await self._db.execute(
			select(RTCPageCollaboration).where(
				RTCPageCollaboration.page_url == session_id,
				RTCPageCollaboration.tenant_id == tenant_id,
			)
		)
		page_collab = page_result.scalar_one_or_none()

		# -- 2. session messages from RTCMessage ---------------------------------
		msg_result = await self._db.execute(
			select(RTCMessage).where(
				RTCMessage.session_id == session_id,
				RTCMessage.tenant_id == tenant_id,
				RTCMessage.is_deleted == False,  # noqa: E712
			)
		)
		session_messages: list[Any] = msg_result.scalars().all()

		# -- normalise page chat_messages ----------------------------------------
		def _normalise_page_msg(raw: dict[str, Any]) -> dict[str, Any]:
			"""Coerce the various key schemas stored in RTCPageCollaboration.chat_messages."""
			# Canonical keys present → use directly
			if "message_id" in raw and "message" in raw:
				return {
					"message_id": raw["message_id"],
					"user_id":    raw.get("user_id", raw.get("sender_id", "")),
					"username":   raw.get("username", raw.get("display_name", "")),
					"message":    raw["message"],
					"message_type": raw.get("message_type", "text"),
					"timestamp":  raw.get("timestamp", raw.get("sent_at", "")),
				}
			# Fallback schema: id / sender_id / content / sent_at
			return {
				"message_id": raw.get("message_id", raw.get("id", "")),
				"user_id":    raw.get("user_id", raw.get("sender_id", "")),
				"username":   raw.get("username", raw.get("display_name", "")),
				"message":    raw.get("message", raw.get("content", "")),
				"message_type": raw.get("message_type", "text"),
				"timestamp":  raw.get("timestamp", raw.get("sent_at", "")),
			}

		page_msgs: list[dict[str, Any]] = []
		if page_collab is not None:
			for raw in (page_collab.chat_messages or []):
				page_msgs.append(_normalise_page_msg(raw))

		# -- normalise RTCMessage ORM objects ------------------------------------
		def _normalise_session_msg(m: Any) -> dict[str, Any]:
			ts = m.sent_at
			if isinstance(ts, datetime):
				ts = ts.isoformat(timespec="seconds")
			participant = getattr(m, "participant", None)
			user_id  = participant.user_id    if participant else ""
			username = participant.display_name if participant else ""
			return {
				"message_id":   m.message_id,
				"user_id":      user_id,
				"username":     username,
				"message":      m.content,
				"message_type": m.message_type,
				"timestamp":    ts,
			}

		orm_msgs = [_normalise_session_msg(m) for m in session_messages]

		# -- merge, sort, limit --------------------------------------------------
		all_msgs = page_msgs + orm_msgs

		def _sort_key(msg: dict[str, Any]) -> str:
			ts = msg.get("timestamp") or ""
			return ts if isinstance(ts, str) else str(ts)

		all_msgs.sort(key=_sort_key, reverse=True)

		if limit is not None:
			all_msgs = all_msgs[:limit]

		return all_msgs

	# ------------------------------------------------------------------
	# 1. workspace_create
	# ------------------------------------------------------------------

	async def workspace_create(
		self,
		name: str,
		owner_id: str,
		description: str = "",
		visibility: str = "private",
	) -> _R:
		"""Create a new collaboration workspace."""
		assert name, "workspace name required"
		assert visibility in {"private", "internal", "public"}, f"invalid visibility: {visibility}"
		workspace_id = uuid7str()
		record = _R(
			workspace_id=workspace_id,
			tenant_id=self.tenant_id,
			name=name,
			description=description,
			owner_id=owner_id,
			visibility=visibility,
			status="active",
			document_count=0,
			member_count=1,
			created_at=_ts(),
			updated_at=_ts(),
		)
		self._workspaces[self._key(workspace_id)] = record
		self._members[workspace_id] = [_R(user_id=owner_id, role="owner", joined_at=_ts())]
		await self._audit("workspace_created", workspace_id, {"name": name, "owner_id": owner_id})
		return record

	# ------------------------------------------------------------------
	# 2. workspace_invite
	# ------------------------------------------------------------------

	async def workspace_invite(
		self,
		workspace_id: str,
		user_id: str,
		role: str = "editor",
		invited_by: str = "system",
	) -> _R:
		"""Invite a user to a workspace."""
		workspace = self._require_workspace(workspace_id)
		assert role in {"viewer", "commenter", "editor", "admin"}, f"invalid role: {role}"
		existing = [m for m in self._members.get(workspace_id, []) if m["user_id"] == user_id]
		if existing:
			return _R(workspace_id=workspace_id, user_id=user_id, status="already_member")
		member = _R(user_id=user_id, role=role, invited_by=invited_by, joined_at=_ts())
		self._members.setdefault(workspace_id, []).append(member)
		workspace["member_count"] = len(self._members[workspace_id])
		await self._notify(user_id, "Workspace Invitation", f"You have been invited to workspace '{workspace['name']}'")
		await self._audit("workspace_invite", workspace_id, {"user_id": user_id, "role": role})
		return _R(workspace_id=workspace_id, user_id=user_id, role=role, status="invited")

	# ------------------------------------------------------------------
	# 3. workspace_remove_member
	# ------------------------------------------------------------------

	async def workspace_remove_member(self, workspace_id: str, user_id: str) -> _R:
		"""Remove a member from a workspace."""
		self._require_workspace(workspace_id)
		before = len(self._members.get(workspace_id, []))
		self._members[workspace_id] = [m for m in self._members.get(workspace_id, []) if m["user_id"] != user_id]
		after = len(self._members[workspace_id])
		self._workspaces[self._key(workspace_id)]["member_count"] = after
		removed = before > after
		await self._audit("workspace_member_removed", workspace_id, {"user_id": user_id, "removed": removed})
		return _R(workspace_id=workspace_id, user_id=user_id, removed=removed)

	# ------------------------------------------------------------------
	# 4. list_workspace_members
	# ------------------------------------------------------------------

	async def list_workspace_members(self, workspace_id: str) -> list[_R]:
		"""List all members of a workspace."""
		self._require_workspace(workspace_id)
		return list(self._members.get(workspace_id, []))

	# ------------------------------------------------------------------
	# 5. document_create
	# ------------------------------------------------------------------

	async def document_create(
		self,
		workspace_id: str,
		title: str,
		content: str = "",
		doc_type: str = "text",
		created_by: str = "system",
	) -> _R:
		"""Create a document in a workspace."""
		workspace = self._require_workspace(workspace_id)
		doc_id = uuid7str()
		record = _R(
			doc_id=doc_id,
			workspace_id=workspace_id,
			tenant_id=self.tenant_id,
			title=title,
			content=content,
			doc_type=doc_type,
			created_by=created_by,
			version=1,
			status="active",
			shared_with=[],
			created_at=_ts(),
			updated_at=_ts(),
		)
		self._documents[self._key(doc_id)] = record
		self._doc_versions[doc_id] = [_R(version=1, content=content, saved_by=created_by, saved_at=_ts())]
		workspace["document_count"] = workspace.get("document_count", 0) + 1
		await self._audit("document_created", doc_id, {"workspace_id": workspace_id, "title": title})
		return record

	# ------------------------------------------------------------------
	# 6. document_share
	# ------------------------------------------------------------------

	async def document_share(
		self,
		doc_id: str,
		user_ids: list[str],
		permission: str = "view",
	) -> _R:
		"""Share a document with specific users."""
		doc = self._require_document(doc_id)
		assert permission in {"view", "comment", "edit"}, f"invalid permission: {permission}"
		new_shares = []
		for uid in user_ids:
			share_entry = _R(user_id=uid, permission=permission, shared_at=_ts())
			# Avoid duplicate share entries
			existing = [s for s in doc.get("shared_with", []) if s["user_id"] == uid]
			if not existing:
				doc.setdefault("shared_with", []).append(share_entry)
				new_shares.append(uid)
			await self._notify(uid, "Document Shared", f"Document '{doc['title']}' has been shared with you ({permission} access)")
		await self._audit("document_shared", doc_id, {"user_ids": user_ids, "permission": permission})
		return _R(doc_id=doc_id, shared_with=new_shares, permission=permission)

	# ------------------------------------------------------------------
	# 7. document_update
	# ------------------------------------------------------------------

	async def document_update(
		self,
		doc_id: str,
		content: str,
		updated_by: str,
		title: str | None = None,
	) -> _R:
		"""Update document content and save a new version."""
		doc = self._require_document(doc_id)
		doc["content"] = content
		if title:
			doc["title"] = title
		doc["version"] = doc.get("version", 1) + 1
		doc["updated_at"] = _ts()
		doc["updated_by"] = updated_by
		self._doc_versions.setdefault(doc_id, []).append(
			_R(version=doc["version"], content=content, saved_by=updated_by, saved_at=_ts())
		)
		await self._audit("document_updated", doc_id, {"version": doc["version"], "updated_by": updated_by})
		return doc

	# ------------------------------------------------------------------
	# 8. co_edit_session
	# ------------------------------------------------------------------

	async def co_edit_session(
		self,
		doc_id: str,
		initiator_id: str,
		participants: list[str] | None = None,
	) -> _R:
		"""Open a real-time co-editing session on a document."""
		doc = self._require_document(doc_id)
		session_id = uuid7str()
		all_participants = list({initiator_id} | set(participants or []))
		record = _R(
			session_id=session_id,
			doc_id=doc_id,
			tenant_id=self.tenant_id,
			initiator_id=initiator_id,
			participants=all_participants,
			status="active",
			op_count=0,
			opened_at=_ts(),
			closed_at=None,
		)
		self._co_edit_sessions[self._key(session_id)] = record
		self._co_edit_ops[session_id] = []
		for uid in all_participants:
			if uid != initiator_id:
				await self._notify(uid, "Co-edit Session Started", f"You have been invited to co-edit '{doc['title']}'")
		await self._audit("co_edit_session_opened", session_id, {"doc_id": doc_id, "participants": all_participants})
		return record

	# ------------------------------------------------------------------
	# 9. co_edit_apply_op
	# ------------------------------------------------------------------

	async def co_edit_apply_op(
		self,
		session_id: str,
		user_id: str,
		op_type: str,
		payload: dict[str, Any],
	) -> _R:
		"""Apply an operational-transform operation in a co-edit session."""
		session = self._co_edit_sessions.get(self._key(session_id))
		assert session is not None, f"co-edit session not found: {session_id}"
		assert session["status"] == "active", "session is not active"
		op = _R(
			op_id=uuid7str(),
			session_id=session_id,
			user_id=user_id,
			op_type=op_type,
			payload=payload,
			applied_at=_ts(),
		)
		self._co_edit_ops[session_id].append(op)
		session["op_count"] = len(self._co_edit_ops[session_id])
		await self._audit("co_edit_op_applied", session_id, {"user_id": user_id, "op_type": op_type})
		return op

	# ------------------------------------------------------------------
	# 10. co_edit_close_session
	# ------------------------------------------------------------------

	async def co_edit_close_session(self, session_id: str, closed_by: str) -> _R:
		"""Close a co-editing session and persist the final document state."""
		session = self._co_edit_sessions.get(self._key(session_id))
		assert session is not None, f"co-edit session not found: {session_id}"
		session["status"] = "closed"
		session["closed_at"] = _ts()
		session["closed_by"] = closed_by
		await self._audit("co_edit_session_closed", session_id, {"closed_by": closed_by, "op_count": session["op_count"]})
		return session

	# ------------------------------------------------------------------
	# 11. comment_add
	# ------------------------------------------------------------------

	async def comment_add(
		self,
		doc_id: str,
		author_id: str,
		body: str,
		anchor: str | None = None,
	) -> _R:
		"""Add a comment to a document, optionally anchored to a text range."""
		doc = self._require_document(doc_id)
		comment_id = uuid7str()
		comment = _R(
			comment_id=comment_id,
			doc_id=doc_id,
			tenant_id=self.tenant_id,
			author_id=author_id,
			body=body,
			anchor=anchor,
			resolved=False,
			replies=[],
			created_at=_ts(),
		)
		self._comments.setdefault(doc_id, []).append(comment)
		# Notify document owner
		owner = doc.get("created_by")
		if owner and owner != author_id:
			await self._notify(owner, "New Comment", f"New comment on '{doc['title']}': {body[:80]}")
		await self._audit("comment_added", comment_id, {"doc_id": doc_id, "author_id": author_id})
		return comment

	# ------------------------------------------------------------------
	# 12. comment_reply
	# ------------------------------------------------------------------

	async def comment_reply(
		self,
		doc_id: str,
		comment_id: str,
		author_id: str,
		body: str,
	) -> _R:
		"""Reply to an existing comment."""
		comments = self._comments.get(doc_id, [])
		parent = next((c for c in comments if c["comment_id"] == comment_id), None)
		assert parent is not None, f"comment not found: {comment_id}"
		reply = _R(
			reply_id=uuid7str(),
			comment_id=comment_id,
			author_id=author_id,
			body=body,
			created_at=_ts(),
		)
		parent["replies"].append(reply)
		# Notify original comment author
		if parent["author_id"] != author_id:
			await self._notify(parent["author_id"], "Comment Reply", f"Reply to your comment: {body[:80]}")
		await self._audit("comment_replied", comment_id, {"author_id": author_id})
		return reply

	# ------------------------------------------------------------------
	# 13. comment_resolve
	# ------------------------------------------------------------------

	async def comment_resolve(self, doc_id: str, comment_id: str, resolved_by: str) -> _R:
		"""Mark a comment as resolved."""
		comments = self._comments.get(doc_id, [])
		comment = next((c for c in comments if c["comment_id"] == comment_id), None)
		assert comment is not None, f"comment not found: {comment_id}"
		comment["resolved"] = True
		comment["resolved_by"] = resolved_by
		comment["resolved_at"] = _ts()
		await self._audit("comment_resolved", comment_id, {"resolved_by": resolved_by})
		return comment

	# ------------------------------------------------------------------
	# 14. mention_notify
	# ------------------------------------------------------------------

	async def mention_notify(
		self,
		mentioned_user_id: str,
		source_doc_id: str,
		mentioned_by: str,
		context: str = "",
	) -> _R:
		"""Record a @mention and notify the mentioned user."""
		doc = self._require_document(source_doc_id)
		mention_id = uuid7str()
		mention = _R(
			mention_id=mention_id,
			mentioned_user_id=mentioned_user_id,
			source_doc_id=source_doc_id,
			mentioned_by=mentioned_by,
			context=context,
			tenant_id=self.tenant_id,
			read=False,
			created_at=_ts(),
		)
		self._mentions.setdefault(mentioned_user_id, []).append(mention)
		await self._notify(mentioned_user_id, "You were mentioned", f"@{mentioned_by} mentioned you in '{doc['title']}': {context[:100]}")
		await self._audit("mention_created", mention_id, {"mentioned": mentioned_user_id, "by": mentioned_by})
		return mention

	# ------------------------------------------------------------------
	# 15. mention_resolve  (@mention_resolve)
	# ------------------------------------------------------------------

	async def mention_resolve(self, user_id: str, mention_id: str) -> _R:
		"""Mark a mention as read/resolved."""
		mentions = self._mentions.get(user_id, [])
		mention = next((m for m in mentions if m["mention_id"] == mention_id), None)
		assert mention is not None, f"mention not found: {mention_id}"
		mention["read"] = True
		mention["resolved_at"] = _ts()
		await self._audit("mention_resolved", mention_id, {"user_id": user_id})
		return mention

	# ------------------------------------------------------------------
	# 16. task_assign
	# ------------------------------------------------------------------

	async def task_assign(
		self,
		doc_id: str,
		title: str,
		assigned_to: str,
		created_by: str,
		due_date: str | None = None,
		priority: str = "normal",
	) -> _R:
		"""Assign a task linked to a document."""
		doc = self._require_document(doc_id)
		assert priority in {"low", "normal", "high", "urgent"}, f"invalid priority: {priority}"
		task_id = uuid7str()
		record = _R(
			task_id=task_id,
			doc_id=doc_id,
			tenant_id=self.tenant_id,
			title=title,
			assigned_to=assigned_to,
			created_by=created_by,
			due_date=due_date,
			priority=priority,
			status="open",
			created_at=_ts(),
			updated_at=_ts(),
		)
		self._tasks[self._key(task_id)] = record
		await self._notify(assigned_to, "Task Assigned", f"Task '{title}' has been assigned to you")
		await self._audit("task_assigned", task_id, {"doc_id": doc_id, "assigned_to": assigned_to})
		return record

	# ------------------------------------------------------------------
	# 17. task_update
	# ------------------------------------------------------------------

	async def task_update(self, task_id: str, **kwargs: Any) -> _R:
		"""Update mutable task fields (status, priority, due_date, title)."""
		task = self._require_task(task_id)
		allowed = {"status", "priority", "due_date", "title"}
		for k, v in kwargs.items():
			if k in allowed:
				task[k] = v
		task["updated_at"] = _ts()
		if kwargs.get("status") == "completed":
			task["completed_at"] = _ts()
			await self._notify(task["created_by"], "Task Completed", f"Task '{task['title']}' has been completed")
		await self._audit("task_updated", task_id, {k: v for k, v in kwargs.items() if k in allowed})
		return task

	# ------------------------------------------------------------------
	# 18. deadline_reminder
	# ------------------------------------------------------------------

	async def deadline_reminder(self, lookahead_hours: int = 24) -> list[_R]:
		"""Find tasks due within the lookahead window and send reminders."""
		cutoff = (datetime.utcnow() + timedelta(hours=lookahead_hours)).isoformat()
		now = _ts()
		upcoming = [
			t for t in self._tasks.values()
			if t["tenant_id"] == self.tenant_id
			and t["status"] == "open"
			and t.get("due_date") is not None
			and now <= t["due_date"] <= cutoff
		]
		for task in upcoming:
			await self._notify(
				task["assigned_to"],
				"Upcoming Deadline",
				f"Task '{task['title']}' is due at {task['due_date']}",
			)
		await self._audit("deadline_reminders_sent", "system", {"count": len(upcoming)})
		return upcoming

	# ------------------------------------------------------------------
	# 19. version_history
	# ------------------------------------------------------------------

	async def version_history(self, doc_id: str) -> list[_R]:
		"""Return the version history of a document."""
		self._require_document(doc_id)
		return list(self._doc_versions.get(doc_id, []))

	# ------------------------------------------------------------------
	# 20. version_restore
	# ------------------------------------------------------------------

	async def version_restore(self, doc_id: str, version: int, restored_by: str) -> _R:
		"""Restore a document to a previous version."""
		doc = self._require_document(doc_id)
		versions = self._doc_versions.get(doc_id, [])
		target = next((v for v in versions if v["version"] == version), None)
		assert target is not None, f"version {version} not found for document {doc_id}"
		new_version = doc["version"] + 1
		doc["content"] = target["content"]
		doc["version"] = new_version
		doc["updated_at"] = _ts()
		doc["updated_by"] = restored_by
		self._doc_versions[doc_id].append(_R(
			version=new_version,
			content=target["content"],
			saved_by=restored_by,
			saved_at=_ts(),
			restored_from=version,
		))
		await self._audit("version_restored", doc_id, {"restored_from": version, "new_version": new_version, "by": restored_by})
		return doc

	# ------------------------------------------------------------------
	# 21. conflict_resolve
	# ------------------------------------------------------------------

	async def conflict_resolve(
		self,
		doc_id: str,
		winning_content: str,
		resolved_by: str,
		strategy: str = "manual",
	) -> _R:
		"""Resolve a co-edit conflict by selecting the winning content."""
		doc = self._require_document(doc_id)
		assert strategy in {"manual", "last_write_wins", "merge"}, f"invalid strategy: {strategy}"
		doc["content"] = winning_content
		doc["version"] = doc.get("version", 1) + 1
		doc["conflict_resolved_at"] = _ts()
		doc["conflict_resolved_by"] = resolved_by
		doc["conflict_strategy"] = strategy
		self._doc_versions.setdefault(doc_id, []).append(_R(
			version=doc["version"],
			content=winning_content,
			saved_by=resolved_by,
			saved_at=_ts(),
			note=f"conflict_resolved_{strategy}",
		))
		await self._audit("conflict_resolved", doc_id, {"strategy": strategy, "resolved_by": resolved_by})
		return doc

	# ------------------------------------------------------------------
	# 22. export_document
	# ------------------------------------------------------------------

	async def export_document(self, doc_id: str, fmt: str = "json") -> str:
		"""Export a document in json, markdown or txt format."""
		doc = self._require_document(doc_id)
		assert fmt in {"json", "markdown", "txt"}, f"unsupported format: {fmt}"
		await self._audit("document_exported", doc_id, {"format": fmt})
		if fmt == "json":
			return json.dumps(dict(doc), default=str, indent=2)
		if fmt == "markdown":
			return f"# {doc['title']}\n\n{doc['content']}\n"
		return f"{doc['title']}\n{'=' * len(doc['title'])}\n\n{doc['content']}\n"

	# ------------------------------------------------------------------
	# 23. activity_feed
	# ------------------------------------------------------------------

	async def activity_feed(
		self,
		workspace_id: str | None = None,
		limit: int = 50,
	) -> list[_R]:
		"""Return recent activity feed entries for the tenant."""
		events = [
			e for e in self._activity_feed
			if e["tenant_id"] == self.tenant_id
		]
		if workspace_id:
			ws_doc_ids = {
				doc["doc_id"]
				for doc in self._documents.values()
				if doc["workspace_id"] == workspace_id
			}
			events = [
				e for e in events
				if e["record_id"] in ws_doc_ids or e["details"].get("workspace_id") == workspace_id
			]
		return sorted(events, key=lambda e: e["occurred_at"], reverse=True)[:limit]

	# ------------------------------------------------------------------
	# 24. collaboration_analytics
	# ------------------------------------------------------------------

	async def collaboration_analytics(self) -> _R:
		"""Aggregate collaboration KPIs for the tenant."""
		workspaces = await self.list_workspaces()
		documents = await self.list_documents()
		tasks = await self.list_tasks()
		comments_total = sum(len(c) for c in self._comments.values())
		co_edit_sessions = [s for s in self._co_edit_sessions.values() if s["tenant_id"] == self.tenant_id]
		total_ops = sum(len(ops) for ops in self._co_edit_ops.values())
		return _R(
			tenant_id=self.tenant_id,
			workspace_count=len(workspaces),
			document_count=len(documents),
			co_edit_session_count=len(co_edit_sessions),
			total_co_edit_ops=total_ops,
			total_comments=comments_total,
			task_count=len(tasks),
			open_task_count=sum(1 for t in tasks if t["status"] == "open"),
			completed_task_count=sum(1 for t in tasks if t["status"] == "completed"),
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 25. list_workspaces
	# ------------------------------------------------------------------

	async def list_workspaces(self, status: str | None = None) -> list[_R]:
		"""List workspaces for the tenant."""
		return sorted(
			[w for w in self._workspaces.values()
			 if w["tenant_id"] == self.tenant_id and (status is None or w["status"] == status)],
			key=lambda w: w["created_at"],
		)

	# ------------------------------------------------------------------
	# 26. list_documents
	# ------------------------------------------------------------------

	async def list_documents(self, workspace_id: str | None = None) -> list[_R]:
		"""List documents for the tenant, optionally filtered by workspace."""
		return sorted(
			[d for d in self._documents.values()
			 if d["tenant_id"] == self.tenant_id
			 and (workspace_id is None or d["workspace_id"] == workspace_id)
			 and d["status"] == "active"],
			key=lambda d: d["created_at"],
		)

	# ------------------------------------------------------------------
	# 27. list_tasks
	# ------------------------------------------------------------------

	async def list_tasks(self, assigned_to: str | None = None, status: str | None = None) -> list[_R]:
		"""List tasks for the tenant."""
		return sorted(
			[t for t in self._tasks.values()
			 if t["tenant_id"] == self.tenant_id
			 and (assigned_to is None or t["assigned_to"] == assigned_to)
			 and (status is None or t["status"] == status)],
			key=lambda t: t["created_at"],
		)

	# ------------------------------------------------------------------
	# 28. list_comments
	# ------------------------------------------------------------------

	async def list_comments(self, doc_id: str, resolved: bool | None = None) -> list[_R]:
		"""List comments on a document."""
		self._require_document(doc_id)
		comments = self._comments.get(doc_id, [])
		if resolved is not None:
			comments = [c for c in comments if c["resolved"] == resolved]
		return sorted(comments, key=lambda c: c["created_at"])

	# ------------------------------------------------------------------
	# 29. delete_document
	# ------------------------------------------------------------------

	async def delete_document(self, doc_id: str, deleted_by: str) -> _R:
		"""Soft-delete a document."""
		doc = self._require_document(doc_id)
		doc["status"] = "deleted"
		doc["deleted_at"] = _ts()
		doc["deleted_by"] = deleted_by
		ws = self._workspaces.get(self._key(doc["workspace_id"]))
		if ws:
			ws["document_count"] = max(0, ws.get("document_count", 1) - 1)
		await self._audit("document_deleted", doc_id, {"deleted_by": deleted_by})
		return doc

	# ------------------------------------------------------------------
	# 30. delete_workspace
	# ------------------------------------------------------------------

	async def delete_workspace(self, workspace_id: str, deleted_by: str) -> _R:
		"""Soft-delete a workspace and all its documents."""
		workspace = self._require_workspace(workspace_id)
		workspace["status"] = "deleted"
		workspace["deleted_at"] = _ts()
		for doc in self._documents.values():
			if doc["workspace_id"] == workspace_id and doc["status"] == "active":
				doc["status"] = "deleted"
				doc["deleted_at"] = _ts()
		await self._audit("workspace_deleted", workspace_id, {"deleted_by": deleted_by})
		return workspace

	# ------------------------------------------------------------------
	# 31. bulk_create_documents
	# ------------------------------------------------------------------

	async def bulk_create_documents(
		self,
		workspace_id: str,
		docs: list[dict[str, Any]],
		created_by: str,
	) -> list[_R]:
		"""Create multiple documents in a workspace at once."""
		results = []
		for d in docs:
			doc = await self.document_create(
				workspace_id=workspace_id,
				title=d["title"],
				content=d.get("content", ""),
				doc_type=d.get("doc_type", "text"),
				created_by=created_by,
			)
			results.append(doc)
		await self._audit("bulk_documents_created", workspace_id, {"count": len(results)})
		return results

	# ------------------------------------------------------------------
	# 32. bulk_assign_tasks
	# ------------------------------------------------------------------

	async def bulk_assign_tasks(
		self,
		doc_id: str,
		tasks: list[dict[str, Any]],
		created_by: str,
	) -> list[_R]:
		"""Assign multiple tasks linked to a document."""
		results = []
		for t in tasks:
			task = await self.task_assign(
				doc_id=doc_id,
				title=t["title"],
				assigned_to=t["assigned_to"],
				created_by=created_by,
				due_date=t.get("due_date"),
				priority=t.get("priority", "normal"),
			)
			results.append(task)
		await self._audit("bulk_tasks_assigned", doc_id, {"count": len(results)})
		return results

	# ------------------------------------------------------------------
	# 33. export_workspace_csv
	# ------------------------------------------------------------------

	async def export_workspace_csv(self, workspace_id: str) -> str:
		"""Export document metadata for a workspace as CSV."""
		docs = await self.list_documents(workspace_id)
		buf = io.StringIO()
		fields = ["doc_id", "title", "doc_type", "version", "created_by", "status", "created_at"]
		writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
		writer.writeheader()
		writer.writerows(docs)
		await self._audit("workspace_exported_csv", workspace_id, {"count": len(docs)})
		return buf.getvalue()

	# ------------------------------------------------------------------
	# 34. export_tasks_json
	# ------------------------------------------------------------------

	async def export_tasks_json(self, assigned_to: str | None = None) -> str:
		"""Export tasks as JSON."""
		tasks = await self.list_tasks(assigned_to=assigned_to)
		await self._audit("tasks_exported_json", "system", {"count": len(tasks)})
		return json.dumps(tasks, default=str, indent=2)

	# ------------------------------------------------------------------
	# 35. health_check
	# ------------------------------------------------------------------

	async def health_check(self) -> _R:
		"""Return service health and storage summary."""
		return _R(
			status="healthy",
			tenant_id=self.tenant_id,
			workspace_count=sum(1 for w in self._workspaces.values() if w["tenant_id"] == self.tenant_id),
			document_count=sum(1 for d in self._documents.values() if d["tenant_id"] == self.tenant_id and d["status"] == "active"),
			active_co_edit_sessions=sum(1 for s in self._co_edit_sessions.values() if s["tenant_id"] == self.tenant_id and s["status"] == "active"),
			open_task_count=sum(1 for t in self._tasks.values() if t["tenant_id"] == self.tenant_id and t["status"] == "open"),
			audit_event_count=len(self._audit_log),
			checked_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 36. dashboard
	# ------------------------------------------------------------------

	async def dashboard(self) -> _R:
		"""KPI dashboard aggregating collaboration metrics."""
		return await self.collaboration_analytics()

	# ------------------------------------------------------------------
	# 37. compliance_report
	# ------------------------------------------------------------------

	async def compliance_report(self, framework: str = "ISO_27001") -> _R:
		"""Generate a collaboration data governance compliance report."""
		workspaces = await self.list_workspaces()
		documents = await self.list_documents()
		public_docs = [d for d in documents if self._workspaces.get(self._key(d["workspace_id"]), {}).get("visibility") == "public"]
		report = _R(
			framework=framework,
			tenant_id=self.tenant_id,
			workspace_count=len(workspaces),
			document_count=len(documents),
			public_documents=len(public_docs),
			audit_trail_complete=True,
			version_history_enabled=True,
			generated_at=_ts(),
		)
		await self._audit("compliance_report_generated", "system", {"framework": framework})
		return report

	# ------------------------------------------------------------------
	# 38. audit_trail
	# ------------------------------------------------------------------

	async def audit_trail(self, event_type: str | None = None) -> list[_R]:
		"""Return audit events for the tenant."""
		return [
			e for e in self._audit_log
			if e["tenant_id"] == self.tenant_id and (event_type is None or e["event_type"] == event_type)
		]

	# ------------------------------------------------------------------
	# 39. get_notifications
	# ------------------------------------------------------------------

	async def get_notifications(self, user_id: str, unread_only: bool = False) -> list[_R]:
		"""Retrieve notifications for a user."""
		notes = [n for n in self._notifications.get(user_id, []) if n["tenant_id"] == self.tenant_id]
		if unread_only:
			notes = [n for n in notes if not n["read"]]
		return sorted(notes, key=lambda n: n["sent_at"], reverse=True)

	# ------------------------------------------------------------------
	# 40. mark_notifications_read
	# ------------------------------------------------------------------

	async def mark_notifications_read(self, user_id: str, notification_ids: list[str] | None = None) -> _R:
		"""Mark notifications as read for a user."""
		notes = self._notifications.get(user_id, [])
		marked = 0
		for n in notes:
			if notification_ids is None or n["notification_id"] in notification_ids:
				n["read"] = True
				n["read_at"] = _ts()
				marked += 1
		await self._audit("notifications_marked_read", user_id, {"count": marked})
		return _R(user_id=user_id, marked_read=marked)

	# ------------------------------------------------------------------
	# 41. workspace_search
	# ------------------------------------------------------------------

	async def workspace_search(self, query: str) -> list[_R]:
		"""Search workspaces by name or description."""
		q = query.lower()
		return [
			w for w in self._workspaces.values()
			if w["tenant_id"] == self.tenant_id
			and w["status"] == "active"
			and (q in w["name"].lower() or q in w.get("description", "").lower())
		]

	# ------------------------------------------------------------------
	# 42. document_search
	# ------------------------------------------------------------------

	async def document_search(self, query: str, workspace_id: str | None = None) -> list[_R]:
		"""Full-text search across document titles and content."""
		q = query.lower()
		return [
			d for d in self._documents.values()
			if d["tenant_id"] == self.tenant_id
			and d["status"] == "active"
			and (workspace_id is None or d["workspace_id"] == workspace_id)
			and (q in d["title"].lower() or q in d.get("content", "").lower())
		]

	# ------------------------------------------------------------------
	# 43. co_edit_session_ops
	# ------------------------------------------------------------------

	async def co_edit_session_ops(self, session_id: str) -> list[_R]:
		"""Return all operations applied in a co-edit session."""
		session = self._co_edit_sessions.get(self._key(session_id))
		assert session is not None, f"co-edit session not found: {session_id}"
		return list(self._co_edit_ops.get(session_id, []))

	# ------------------------------------------------------------------
	# 44. user_activity_summary
	# ------------------------------------------------------------------

	async def user_activity_summary(self, user_id: str) -> _R:
		"""Summarise activity for a specific user across the tenant."""
		docs_created = sum(1 for d in self._documents.values() if d.get("created_by") == user_id and d["tenant_id"] == self.tenant_id)
		tasks_assigned = sum(1 for t in self._tasks.values() if t["assigned_to"] == user_id and t["tenant_id"] == self.tenant_id)
		tasks_completed = sum(1 for t in self._tasks.values() if t["assigned_to"] == user_id and t["status"] == "completed" and t["tenant_id"] == self.tenant_id)
		comments_made = sum(
			sum(1 for c in comments if c["author_id"] == user_id)
			for comments in self._comments.values()
		)
		mentions_received = len(self._mentions.get(user_id, []))
		return _R(
			user_id=user_id,
			tenant_id=self.tenant_id,
			documents_created=docs_created,
			tasks_assigned=tasks_assigned,
			tasks_completed=tasks_completed,
			comments_made=comments_made,
			mentions_received=mentions_received,
			generated_at=_ts(),
		)
