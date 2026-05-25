"""Agent-network orchestration primitives."""

from __future__ import annotations

from .models import AgentNetwork, IntelligentAgent


class NetworkOrchestrator:
	"""Coordinate a set of agents inside one network."""

	def __init__(self, network: AgentNetwork, agents: list[IntelligentAgent]):
		self.network = network
		self.agents = list(agents)

	async def route_task(self, task: dict) -> dict:
		"""Route a task to the first active agent with matching capabilities."""

		required = set(task.get("capabilities", []))
		for agent in self.agents:
			if not required or required.issubset(set(agent.capabilities)):
				return {"agent_id": agent.id, "status": "routed", "task": task}
		return {"agent_id": None, "status": "unassigned", "task": task}
