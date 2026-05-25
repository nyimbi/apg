"""Learning primitives for APG software agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from uuid_extensions import uuid7str


@dataclass
class LearningEvent:
	"""A single observation used to improve agent behavior."""

	agent_id: str
	event_type: str
	context: dict[str, Any] = field(default_factory=dict)
	outcome: dict[str, Any] = field(default_factory=dict)
	improvement_score: float = 0.0
	id: str = field(default_factory=uuid7str)
	timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LearningGoal:
	"""A tracked improvement target for an agent."""

	goal_type: str
	metric: str
	target: float
	id: str = field(default_factory=uuid7str)
	current_value: float = 0.0
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


class LearningEngine:
	"""In-memory learning engine with deterministic strategy summaries."""

	def __init__(self, agent_id: str, strategies: list[str] | None = None):
		self.agent_id = agent_id
		self.strategies = strategies or ["reinforcement", "pattern_recognition"]
		self.events: list[LearningEvent] = []
		self.goals: list[LearningGoal] = []
		self.q_table: dict[str, float] = {}

	async def record_learning_event(self, event: LearningEvent) -> None:
		"""Record a learning event for later analysis."""

		self.events.append(event)

	def _get_recent_events(self, days: int = 1) -> list[LearningEvent]:
		"""Return events recorded within the requested day window."""

		cutoff = datetime.utcnow() - timedelta(days=days)
		return [event for event in self.events if event.timestamp >= cutoff]

	async def create_goal(self, goal_type: str, metric: str, target: float) -> LearningGoal:
		"""Create and track a learning goal."""

		goal = LearningGoal(goal_type=goal_type, metric=metric, target=target)
		self.goals.append(goal)
		return goal

	async def run_learning_session(self) -> dict[str, Any]:
		"""Analyze recorded events and return strategy-level progress."""

		strategy_results: dict[str, Any] = {}
		events_processed = len(self.events)

		if "reinforcement" in self.strategies:
			for event in self.events:
				key = str(event.context.get("approach") or event.context.get("task_type") or event.event_type)
				score = float(event.outcome.get("quality_score", event.improvement_score or 0.0))
				self.q_table[key] = max(self.q_table.get(key, 0.0), score)
			strategy_results["reinforcement"] = {
				"learning_events_processed": events_processed,
				"q_table_size": len(self.q_table),
				"improvements": dict(self.q_table),
			}

		if "pattern_recognition" in self.strategies:
			pattern_counts: dict[str, int] = {}
			for event in self.events:
				task_type = str(event.context.get("task_type") or event.event_type)
				pattern_counts[task_type] = pattern_counts.get(task_type, 0) + 1
			strategy_results["pattern_recognition"] = {
				"total_patterns": len(pattern_counts),
				"new_patterns": sum(1 for count in pattern_counts.values() if count >= 2),
				"patterns": pattern_counts,
			}

		if "meta_learning" in self.strategies:
			effectiveness: dict[str, list[float]] = {}
			for event in self.events:
				strategy = event.context.get("learning_strategy")
				if strategy:
					effectiveness.setdefault(str(strategy), []).append(
						float(event.outcome.get("improvement_score", event.improvement_score or 0.0))
					)
			averages = {
				strategy: sum(scores) / len(scores)
				for strategy, scores in effectiveness.items()
				if scores
			}
			best_strategy = max(averages, key=averages.get) if averages else None
			strategy_results["meta_learning"] = {
				"best_strategy": best_strategy,
				"adaptations": averages,
			}

		for goal in self.goals:
			goal.current_value = min(goal.target, max(goal.current_value, 0.1 * events_processed))
			goal.updated_at = datetime.utcnow()

		return {
			"status": "completed",
			"events_processed": events_processed,
			"goals_updated": len(self.goals),
			"strategy_results": strategy_results,
		}
