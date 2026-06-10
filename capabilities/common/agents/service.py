"""In-memory service for APG common intelligent agents."""

from __future__ import annotations

from typing import Any

from .models import AgentRole, AgentType, IntelligentAgent
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache


class AgentManagerService:
	"""Manage intelligent agents with optional auth and audit integrations."""

	def __init__(self) -> None:
		self._agents: dict[str, IntelligentAgent] = {}
		self._auth_service_available = False
		self._audit_service_available = False
		self.auth_service: Any = None
		self.audit_service: Any = None
		self.ai_orchestration: Any = None
		self.federated_learning: Any = None
		self.collaboration_service: Any = None

	async def create_agent(self, user_id: str, agent_config: dict[str, Any]) -> IntelligentAgent:
		"""Create and store an intelligent agent for a user."""

		if self._auth_service_available and self.auth_service:
			allowed = await self.auth_service.check_permission(user_id, "agent:create")
			if not allowed:
				raise PermissionError(f"User {user_id} cannot create agents")

		agent = IntelligentAgent(
			name=str(agent_config.get("name", "Unnamed Agent")),
			description=str(agent_config.get("description", "")),
			type=self._coerce_agent_type(agent_config.get("type", AgentType.WORKER)),
			role=self._coerce_agent_role(agent_config.get("role", AgentRole.TASK_MANAGER)),
			capabilities=list(agent_config.get("capabilities", [])),
			configuration=dict(agent_config.get("configuration", {})),
			created_by=user_id,
			tenant_id=str(agent_config.get("tenant_id", "default")),
		)
		self._agents[agent.id] = agent

		if self._audit_service_available and self.audit_service:
			await self.audit_service.record_event(
				user_id=user_id,
				action="agent:create",
				resource_id=agent.id,
				details={"name": agent.name, "role": agent.role.value},
			)

		return agent

	async def get_agent(self, user_id: str, agent_id: str) -> IntelligentAgent:
		"""Fetch an agent by id."""

		if agent_id not in self._agents:
			raise KeyError(f"Agent not found: {agent_id}")

		if self._auth_service_available and self.auth_service:
			allowed = await self.auth_service.check_permission(user_id, "agent:read")
			if not allowed:
				raise PermissionError(f"User {user_id} cannot read agents")

		return self._agents[agent_id]

	async def list_agents(self, user_id: str, tenant_id: str | None = None) -> list[IntelligentAgent]:
		"""List agents visible to the user."""

		if self._auth_service_available and self.auth_service:
			allowed = await self.auth_service.check_permission(user_id, "agent:list")
			if not allowed:
				raise PermissionError(f"User {user_id} cannot list agents")

		agents = list(self._agents.values())
		if tenant_id is not None:
			agents = [agent for agent in agents if agent.tenant_id == tenant_id]
		return agents

	def _coerce_agent_type(self, value: AgentType | str) -> AgentType:
		if isinstance(value, AgentType):
			return value
		return AgentType(str(value))

	def _coerce_agent_role(self, value: AgentRole | str) -> AgentRole:
		if isinstance(value, AgentRole):
			return value
		return AgentRole(str(value))
