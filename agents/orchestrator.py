"""APG agent orchestration runtime."""

from __future__ import annotations

from typing import Any

from uuid_extensions import uuid7str

from .base_agent import AgentTask, BaseAgent


class AgentOrchestrator:
	"""Coordinate APG agents and project tasks in memory."""

	def __init__(self):
		self.agents: dict[str, BaseAgent] = {}
		self.active_projects: dict[str, dict[str, Any]] = {}
		self.task_queue: list[AgentTask] = []
		self.completed_tasks: list[AgentTask] = []

	async def register_agent(self, agent: BaseAgent) -> None:
		"""Register an agent with the orchestrator."""

		self.agents[agent.agent_id] = agent

	async def start_project(self, project_spec: dict[str, Any], name: str) -> str:
		"""Start a project and enqueue initial work for registered agents."""

		project_id = uuid7str()
		project = {
			"id": project_id,
			"name": name,
			"spec": project_spec,
			"status": "started",
		}
		self.active_projects[project_id] = project

		task = AgentTask(
			name=f"Plan {name}",
			description=f"Initial APG project plan for {name}",
			requirements={
				"type": "project_generation",
				"project_id": project_id,
				"project_spec": project_spec,
				"collaboration": True,
			},
		)
		self.task_queue.append(task)

		for agent in self.agents.values():
			await agent.receive_task(task)

		return project_id

	def get_system_status(self) -> dict[str, Any]:
		"""Return project and agent status."""

		return {
			"agents": {
				"total": len(self.agents),
				"roles": [agent.role.value for agent in self.agents.values()],
			},
			"projects": {
				"active": len(self.active_projects),
				"ids": list(self.active_projects),
			},
			"tasks": {
				"queued": len(self.task_queue),
				"completed": len(self.completed_tasks),
			},
		}
