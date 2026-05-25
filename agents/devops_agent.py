"""DevOps APG agent."""

from .base_agent import AgentRole, BaseAgent


class DevOpsAgent(BaseAgent):
	role = AgentRole.DEVOPS
