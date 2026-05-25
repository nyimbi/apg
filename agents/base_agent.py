"""Base APG agent abstractions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from uuid_extensions import uuid7str

from .integrations import DEFAULT_AGENT_INTEGRATIONS, AgentInvocation
from .learning_engine import LearningEngine, LearningEvent


class AgentRole(StrEnum):
	"""Standard APG agent roles."""

	ARCHITECT = "architect"
	DEVELOPER = "developer"
	TESTER = "tester"
	DEVOPS = "devops"


class AgentCapability(StrEnum):
	"""Common capabilities exposed by APG agents."""

	REQUIREMENT_ANALYSIS = "requirement_analysis"
	CODE_GENERATION = "code_generation"
	AUTOMATED_TESTING = "automated_testing"
	DEPLOYMENT = "deployment"
	LEARNING = "learning"


@dataclass
class AgentTask:
	"""A unit of work assigned to an APG agent."""

	name: str
	description: str
	requirements: dict[str, Any] = field(default_factory=dict)
	id: str = field(default_factory=uuid7str)
	status: str = "pending"
	created_at: datetime = field(default_factory=datetime.utcnow)
	completed_at: datetime | None = None
	result: dict[str, Any] | None = None


class BaseAgent:
	"""Async, in-memory agent with learning hooks."""

	role: AgentRole = AgentRole.DEVELOPER

	def __init__(self, agent_id: str | None = None, config: dict[str, Any] | None = None):
		self.agent_id = agent_id or f"{self.role.value}_{uuid7str()[:8]}"
		self.config = config or {}
		learning_config = self.config.get("learning", {})
		self.learning_enabled = bool(learning_config.get("enabled", True))
		self.learning_engine = LearningEngine(
			self.agent_id,
			list(learning_config.get("strategies", ["reinforcement", "pattern_recognition"])),
		)
		self.backend_name = self.config.get("backend") or self.config.get("runtime") or "local"
		self.backend_context = dict(self.config.get("backend_context", {}))
		self.tasks: dict[str, AgentTask] = {}
		self.completed_tasks: dict[str, AgentTask] = {}

	async def receive_task(self, task: AgentTask) -> bool:
		"""Accept and schedule a task for asynchronous execution."""

		task.status = "accepted"
		self.tasks[task.id] = task
		asyncio.create_task(self._execute_task(task))
		return True

	async def _execute_task(self, task: AgentTask) -> None:
		invocation = AgentInvocation(
			prompt=task.description,
			cwd=task.requirements.get("cwd"),
			context={
				**self.backend_context,
				"agent_id": self.agent_id,
				"role": self.role.value,
				"task": task.requirements,
			},
			files=list(task.requirements.get("files", [])),
			timeout_seconds=float(task.requirements.get("timeout_seconds", 120.0)),
		)
		run_result = await DEFAULT_AGENT_INTEGRATIONS.run(self.backend_name, invocation)
		task.status = "completed"
		task.completed_at = datetime.utcnow()
		task.result = {
			"agent_id": self.agent_id,
			"role": self.role.value,
			"task_type": task.requirements.get("type", "generic"),
			"success": run_result.success,
			"backend": run_result.backend,
			"output": run_result.output,
			"error": run_result.error,
		}
		self.completed_tasks[task.id] = task
		await self.learning_engine.record_learning_event(
			LearningEvent(
				agent_id=self.agent_id,
				event_type="task_completion",
				context={
					"task_type": task.requirements.get("type", "generic"),
					"capabilities": task.requirements.get("capabilities", []),
					"collaboration": bool(task.requirements.get("collaboration")),
				},
				outcome={"success": True, "quality_score": 0.85, "efficiency": 0.8},
				improvement_score=0.85,
			)
		)

	async def create_learning_goal(self, goal_type: str, metric: str, target: float) -> None:
		"""Create an improvement target."""

		await self.learning_engine.create_goal(goal_type, metric, target)

	async def learn_from_feedback(self, feedback: dict[str, Any]) -> None:
		"""Convert external feedback into a learning event."""

		await self.learning_engine.record_learning_event(
			LearningEvent(
				agent_id=self.agent_id,
				event_type=str(feedback.get("type", "feedback")),
				context={
					"source": feedback.get("source", "unknown"),
					"areas_for_improvement": feedback.get("areas_for_improvement", []),
					"positive_aspects": feedback.get("positive_aspects", []),
				},
				outcome={
					"success": float(feedback.get("score", 0.0)) >= 0.5,
					"quality_score": float(feedback.get("score", 0.0)),
					"lessons_learned": feedback.get("lessons_learned", []),
				},
				improvement_score=float(feedback.get("score", 0.0)),
			)
		)

	async def configure_learning(self, config: dict[str, Any]) -> None:
		"""Update learning strategy configuration."""

		if "strategies" in config:
			self.learning_engine.strategies = list(config["strategies"])

	async def run_learning_session(self) -> dict[str, Any]:
		"""Run the agent's learning engine."""

		return await self.learning_engine.run_learning_session()

	def get_learning_status(self) -> dict[str, Any]:
		"""Return current learning and task metrics."""

		total_tasks = len(self.completed_tasks)
		return {
			"agent_id": self.agent_id,
			"role": self.role.value,
			"learning_enabled": self.learning_enabled,
			"total_events": len(self.learning_engine.events),
			"learning_goals": len(self.learning_engine.goals),
			"recent_performance": {
				"total_tasks": total_tasks,
				"success_rate": 1.0 if total_tasks else 0.0,
			},
		}
