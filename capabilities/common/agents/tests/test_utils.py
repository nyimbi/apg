"""Reusable in-memory services for common agent tests."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from uuid_extensions import uuid7str


class InMemoryAuthService:
	async def check_permission(self, user_id: str, permission: str) -> bool:
		return bool(user_id and permission)


class InMemoryAuditService:
	def __init__(self) -> None:
		self._logs: list[dict[str, Any]] = []

	async def record_event(self, **event: Any) -> str:
		event_id = uuid7str()
		self._logs.append({
			"id": event_id,
			"timestamp": datetime.utcnow().isoformat(),
			**event,
		})
		return event_id

	async def get_audit_logs(self, **filters: Any) -> list[dict[str, Any]]:
		logs = self._logs
		for key, value in filters.items():
			if value is not None:
				logs = [log for log in logs if log.get(key) == value]
		return list(logs)


class InMemoryAIOrchestration:
	def __init__(self) -> None:
		self._tasks: dict[str, dict[str, Any]] = {}

	async def submit_task(self, task_definition: dict[str, Any]) -> str:
		task_id = uuid7str()
		self._tasks[task_id] = {
			"id": task_id,
			"definition": dict(task_definition),
			"status": "completed",
			"created_at": datetime.utcnow().isoformat(),
		}
		return task_id

	async def get_task_status(self, task_id: str) -> dict[str, Any]:
		return dict(self._tasks[task_id])


class InMemoryFederatedLearning:
	def __init__(self) -> None:
		self.rounds: list[dict[str, Any]] = []

	async def submit_update(self, update: dict[str, Any]) -> str:
		update_id = uuid7str()
		self.rounds.append({"id": update_id, **update})
		return update_id


class InMemoryCollaborationService:
	def __init__(self) -> None:
		self._messages: dict[str, list[dict[str, Any]]] = {}

	async def create_collaboration_session(self, agent_ids: list[str], purpose: str) -> str:
		session_id = uuid7str()
		self._messages[session_id] = [{
			"type": "system",
			"sender_id": "system",
			"content": purpose,
			"agent_ids": list(agent_ids),
		}]
		return session_id

	async def send_message(self, session_id: str, sender_id: str, content: str, message_type: str) -> str:
		message_id = uuid7str()
		self._messages.setdefault(session_id, []).append({
			"id": message_id,
			"sender_id": sender_id,
			"content": content,
			"type": message_type,
		})
		return message_id

	async def get_messages(self, session_id: str) -> list[dict[str, Any]]:
		return list(self._messages.get(session_id, []))


def create_test_services() -> dict[str, Any]:
	"""Create in-memory service doubles for local agent tests."""

	return {
		"auth_service": InMemoryAuthService(),
		"audit_service": InMemoryAuditService(),
		"ai_orchestration": InMemoryAIOrchestration(),
		"federated_learning": InMemoryFederatedLearning(),
		"collaboration_service": InMemoryCollaborationService(),
	}
