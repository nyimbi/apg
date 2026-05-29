"""Planning helpers for APG AI agent teams."""

from __future__ import annotations

from typing import Any

from .models import AgentDefinition, AgentRuntime, AgentTeam, ExecutionPlan


class AgentCompositionPlanner:
	"""Build deterministic execution plans from registered agents, runtimes, and handoffs."""

	def build_plan(
		self,
		team: AgentTeam,
		agents: dict[str, AgentDefinition],
		runtimes: dict[str, AgentRuntime],
		objective: str,
	) -> ExecutionPlan:
		missing_agents = [agent_id for agent_id in team.agent_ids if agent_id not in agents]
		if missing_agents:
			raise KeyError(f"unknown team agent(s): {', '.join(missing_agents)}")

		steps: list[dict[str, Any]] = []
		runtime_assignments: dict[str, str] = {}
		approvals: list[dict[str, Any]] = []
		for index, agent_id in enumerate(team.agent_ids, start=1):
			agent = agents[agent_id]
			runtime = runtimes.get(agent.runtime)
			if runtime is None:
				raise KeyError(f"unknown runtime for agent {agent.id}: {agent.runtime}")
			runtime_assignments[agent.id] = runtime.name
			if runtime.external_runtime and not runtime.approved:
				approvals.append({
					"agent": agent.id,
					"runtime": runtime.name,
					"reason": "external_runtime_approval_required",
				})
			steps.append({
				"order": index,
				"agent": agent.id,
				"name": agent.name,
				"runtime": runtime.name,
				"model": agent.model,
				"objective": objective,
				"tools": list(agent.tool_allowlist),
				"handoff_targets": [
					edge.target for edge in team.handoffs
					if edge.source == agent.id
				],
			})

		return ExecutionPlan(
			id=f"{team.id}:plan",
			tenant_id=team.tenant_id,
			team_id=team.id,
			steps=tuple(steps),
			runtime_assignments=runtime_assignments,
			approvals_required=tuple(approvals),
			estimated_cost_limit=_minimum_cost_limit(runtimes.values()),
		)


def _minimum_cost_limit(runtimes: object) -> float | None:
	limits = [
		runtime.cost_limit for runtime in runtimes
		if isinstance(runtime, AgentRuntime) and runtime.cost_limit is not None
	]
	return min(limits) if limits else None
