"""Learning engine for common APG intelligent agents."""

from __future__ import annotations

from datetime import datetime
from typing import Any


class AgentLearningEngine:
	"""Track lightweight learning state for one agent."""

	def __init__(self, agent_id: str, learning_enabled: bool = True):
		self.agent_id = agent_id
		self.learning_enabled = learning_enabled
		self.observations: list[dict[str, Any]] = []
		self.created_at = datetime.utcnow()

	def record_observation(self, observation: dict[str, Any]) -> None:
		self.observations.append(dict(observation))

	def get_learning_status(self) -> dict[str, Any]:
		return {
			"agent_id": self.agent_id,
			"learning_enabled": self.learning_enabled,
			"observations": len(self.observations),
			"created_at": self.created_at.isoformat(),
		}
