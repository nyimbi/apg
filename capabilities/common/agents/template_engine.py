"""Agent template utilities."""

from __future__ import annotations

from typing import Any

from .models import AgentRole, AgentType


class AgentTemplateEngine:
	"""Provide reusable starting templates for APG agents."""

	def __init__(self) -> None:
		self._templates: dict[str, dict[str, Any]] = {
			"task_manager": {
				"type": AgentType.WORKER,
				"role": AgentRole.TASK_MANAGER,
				"capabilities": ["reasoning", "communication"],
			},
			"orchestrator": {
				"type": AgentType.COORDINATOR,
				"role": AgentRole.ORCHESTRATOR,
				"capabilities": ["orchestration", "communication"],
			},
		}

	def list_available_templates(self) -> dict[str, dict[str, Any]]:
		return {name: dict(template) for name, template in self._templates.items()}

	def get_template(self, name: str) -> dict[str, Any]:
		return dict(self._templates[name])
