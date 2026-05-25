"""Architect APG agent."""

from .base_agent import AgentRole, BaseAgent


class ArchitectAgent(BaseAgent):
	role = AgentRole.ARCHITECT
