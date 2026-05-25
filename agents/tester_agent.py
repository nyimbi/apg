"""Tester APG agent."""

from .base_agent import AgentRole, BaseAgent


class TesterAgent(BaseAgent):
	__test__ = False
	role = AgentRole.TESTER
