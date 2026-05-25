"""Decision engine for common intelligent agents."""

from __future__ import annotations

from typing import Any


class AgentDecisionEngine:
	"""Small deterministic decision engine used by local APG tests."""

	def decide(self, options: list[Any], criteria: dict[str, Any] | None = None) -> Any:
		if not options:
			return None
		if not criteria:
			return options[0]
		preferred = criteria.get("preferred")
		return preferred if preferred in options else options[0]


_DECISION_ENGINE = AgentDecisionEngine()


def get_decision_engine() -> AgentDecisionEngine:
	"""Return the process-wide decision engine."""

	return _DECISION_ENGINE
