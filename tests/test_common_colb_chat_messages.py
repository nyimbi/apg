"""COLB chat retrieval regressions for executable collaboration state."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from capabilities.common.colb.service import CollaborationService


class _ScalarResult:
	def __init__(self, values):
		self._values = values

	def all(self):
		return self._values


class _ExecuteResult:
	def __init__(self, scalar=None, values=None):
		self._scalar = scalar
		self._values = values or []

	def scalar_one_or_none(self):
		return self._scalar

	def scalars(self):
		return _ScalarResult(self._values)


class _FakeDB:
	def __init__(self, results):
		self._results = list(results)

	async def execute(self, _query):
		return self._results.pop(0)


@pytest.mark.asyncio
async def test_chat_messages_are_loaded_from_page_and_session_state() -> None:
	page_collaboration = SimpleNamespace(
		chat_messages=[
			{
				"message_id": "page-old",
				"user_id": "user-page",
				"username": "Page User",
				"message": "Stored page message",
				"message_type": "text",
				"timestamp": "2026-05-29T01:00:00",
			}
		]
	)
	session_message = SimpleNamespace(
		message_id="session-new",
		content="Stored session message",
		message_type="text",
		sent_at=datetime(2026, 5, 29, 1, 5, 0),
		participant=SimpleNamespace(user_id="user-session", display_name="Session User"),
	)
	service = CollaborationService(_FakeDB([
		_ExecuteResult(scalar=page_collaboration),
		_ExecuteResult(values=[session_message]),
	]))

	messages = await service.get_chat_messages("/customers/1", limit=10, tenant_id="tenant-a")

	assert messages == [
		{
			"message_id": "session-new",
			"user_id": "user-session",
			"username": "Session User",
			"message": "Stored session message",
			"message_type": "text",
			"timestamp": "2026-05-29T01:05:00",
		},
		{
			"message_id": "page-old",
			"user_id": "user-page",
			"username": "Page User",
			"message": "Stored page message",
			"message_type": "text",
			"timestamp": "2026-05-29T01:00:00",
		},
	]


@pytest.mark.asyncio
async def test_chat_message_limit_is_applied_after_timestamp_sorting() -> None:
	page_collaboration = SimpleNamespace(
		chat_messages=[
			{"id": "old", "sender_id": "u1", "content": "old", "sent_at": "2026-05-29T00:59:00"},
			{"id": "new", "sender_id": "u2", "content": "new", "sent_at": "2026-05-29T01:10:00"},
		]
	)
	service = CollaborationService(_FakeDB([
		_ExecuteResult(scalar=page_collaboration),
		_ExecuteResult(values=[]),
	]))

	messages = await service.get_chat_messages("/customers/1", limit=1, tenant_id="tenant-a")

	assert len(messages) == 1
	assert messages[0]["message_id"] == "new"
	assert messages[0]["message"] == "new"
