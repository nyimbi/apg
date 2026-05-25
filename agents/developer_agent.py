"""Developer APG agent."""

from .base_agent import AgentRole, BaseAgent


class DeveloperAgent(BaseAgent):
	role = AgentRole.DEVELOPER
