"""In-memory communication hub for APG agents."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from uuid_extensions import uuid7str


class AgentCommunicationHub:
	"""Store lightweight agent-to-agent messages by channel."""

	def __init__(self) -> None:
		self._messages: dict[str, list[dict[str, Any]]] = {}

	async def send(self, channel_id: str, sender_id: str, content: str, message_type: str = "text") -> str:
		message_id = uuid7str()
		self._messages.setdefault(channel_id, []).append({
			"id": message_id,
			"sender_id": sender_id,
			"content": content,
			"type": message_type,
			"timestamp": datetime.utcnow().isoformat(),
		})
		return message_id

	async def messages(self, channel_id: str) -> list[dict[str, Any]]:
		return list(self._messages.get(channel_id, []))


_COMMUNICATION_HUB = AgentCommunicationHub()


def get_communication_hub() -> AgentCommunicationHub:
	"""Return the process-wide communication hub."""

	return _COMMUNICATION_HUB
