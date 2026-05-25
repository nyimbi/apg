"""Capability registry for common APG intelligent agents."""

from __future__ import annotations


class AgentCapabilityRegistry:
	"""Registry of capabilities that agents may advertise."""

	def __init__(self) -> None:
		self._capabilities: dict[str, str] = {
			"reasoning": "Structured task reasoning",
			"learning": "Feedback-driven improvement",
			"communication": "Agent collaboration and messaging",
			"orchestration": "Network and task coordination",
		}

	def register_capability(self, name: str, description: str = "") -> None:
		self._capabilities[name] = description

	def list_available_capabilities(self) -> list[str]:
		return sorted(self._capabilities)


global_capability_registry = AgentCapabilityRegistry()
