"""Service layer for first-class APG AI agent composition."""

from __future__ import annotations

from typing import Any

from .agent_composition import AgentCompositionPlanner
from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import AgentDefinition, AgentRuntime, AgentTeam, HandoffEdge


class AgntService:
	"""In-memory agent registry, runtime registry, team validator, and plan builder."""

	def __init__(self) -> None:
		self._agents: dict[str, AgentDefinition] = {}
		self._runtimes: dict[str, AgentRuntime] = {}
		self._teams: dict[str, AgentTeam] = {}
		self._planner = AgentCompositionPlanner()
		for runtime in _default_runtimes():
			self._runtimes[runtime.name] = runtime

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_runtime(
		self,
		name: str,
		kind: str = "local",
		approved: bool = True,
		workspace_runtime: bool = False,
		external_runtime: bool = False,
		sandbox_policy: str | None = "workspace-read",
		capabilities: list[str] | tuple[str, ...] | None = None,
		cost_limit: float | None = None,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		self._enforce_runtime_policy(
			tenant_id=tenant_id,
			registered=True,
			workspace_runtime=workspace_runtime,
			sandbox_policy_attached=bool(sandbox_policy),
			external_runtime=external_runtime,
			approval_recorded=approved,
		)
		runtime = AgentRuntime(
			name=name,
			kind=kind,
			approved=approved,
			workspace_runtime=workspace_runtime,
			external_runtime=external_runtime,
			sandbox_policy=sandbox_policy,
			capabilities=tuple(capabilities or ()),
			cost_limit=cost_limit,
		)
		self._runtimes[name] = runtime
		return runtime.to_dict()

	def list_runtimes(self) -> list[dict[str, Any]]:
		return [runtime.to_dict() for runtime in sorted(self._runtimes.values(), key=lambda item: item.name)]

	def register_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		model: str,
		runtime: str,
		system_prompt: str,
		tool_allowlist: list[str] | tuple[str, ...] | None = None,
		input_contract: dict[str, Any] | None = None,
		output_contract: dict[str, Any] | None = None,
		memory_policy: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		runtime_record = self._runtimes.get(runtime)
		self._enforce_agent_policy(
			tenant_id=tenant_id,
			model_present=bool(model),
			runtime_registered=runtime_record is not None and runtime_record.registered,
			workspace_runtime=bool(runtime_record and runtime_record.workspace_runtime),
			sandbox_policy_attached=bool(runtime_record and runtime_record.sandbox_policy),
		)
		if not system_prompt:
			raise ValueError("agent system_prompt is required")
		agent = AgentDefinition(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			model=model,
			runtime=runtime,
			system_prompt=system_prompt,
			tool_allowlist=tuple(tool_allowlist or ()),
			input_contract=dict(input_contract or {}),
			output_contract=dict(output_contract or {}),
			memory_policy=dict(memory_policy or {}),
			status=status,
		)
		self._agents[agent_id] = agent
		return agent.to_dict()

	def list_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		agents = list(self._agents.values())
		if tenant_id is not None:
			agents = [agent for agent in agents if agent.tenant_id == tenant_id]
		return [agent.to_dict() for agent in sorted(agents, key=lambda item: item.id)]

	def register_team(
		self,
		team_id: str,
		tenant_id: str,
		name: str,
		agent_ids: list[str] | tuple[str, ...],
		handoffs: list[dict[str, str]] | tuple[dict[str, str], ...] | None = None,
		execution_mode: str = "sequential",
		parallel_execution_enabled: bool = False,
	) -> dict[str, Any]:
		edges = tuple(
			HandoffEdge(
				source=str(edge["source"]),
				target=str(edge["target"]),
				trigger=str(edge.get("trigger") or "complete"),
				condition=str(edge.get("condition") or "always"),
			)
			for edge in (handoffs or ())
		)
		self._enforce_team_policy(tenant_id, len(agent_ids), agent_ids, edges)
		team = AgentTeam(
			id=team_id,
			tenant_id=tenant_id,
			name=name,
			agent_ids=tuple(agent_ids),
			handoffs=edges,
			execution_mode=execution_mode,
			parallel_execution_enabled=parallel_execution_enabled,
		)
		self._teams[team_id] = team
		return team.to_dict()

	def list_teams(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		teams = list(self._teams.values())
		if tenant_id is not None:
			teams = [team for team in teams if team.tenant_id == tenant_id]
		return [team.to_dict() for team in sorted(teams, key=lambda item: item.id)]

	def plan_execution(self, team_id: str, objective: str, tenant_id: str | None = None) -> dict[str, Any]:
		team = self._teams.get(team_id)
		if team is None:
			raise KeyError(f"unknown agent team: {team_id}")
		if tenant_id is not None and team.tenant_id != tenant_id:
			raise KeyError(f"unknown agent team for tenant: {team_id}")
		return self._planner.build_plan(
			team=team,
			agents=self._agents,
			runtimes=self._runtimes,
			objective=objective,
		).to_dict()

	def composition_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		contract = self.describe(tenant_id)
		return {
			"capability": contract["capability"],
			"display_name": contract["display_name"],
			"tenant_id": tenant_id,
			"agent_count": len(self.list_agents(tenant_id)),
			"team_count": len(self.list_teams(tenant_id)),
			"runtime_count": len(self._runtimes),
			"routes": contract["ui"]["routes"],
			"theme": contract["theme"],
		}

	def _enforce_agent_policy(
		self,
		tenant_id: str,
		model_present: bool,
		runtime_registered: bool,
		workspace_runtime: bool,
		sandbox_policy_attached: bool,
	) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_agent",
			"model_present": model_present,
			"runtime_registered": runtime_registered,
			"workspace_runtime": workspace_runtime,
			"sandbox_policy_attached": sandbox_policy_attached,
		})
		_raise_if_blocked(result)

	def _enforce_runtime_policy(
		self,
		tenant_id: str,
		registered: bool,
		workspace_runtime: bool,
		sandbox_policy_attached: bool,
		external_runtime: bool,
		approval_recorded: bool,
	) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"runtime_registered": registered,
			"workspace_runtime": workspace_runtime,
			"sandbox_policy_attached": sandbox_policy_attached,
			"external_runtime": external_runtime,
			"approval_recorded": approval_recorded,
		})
		_raise_if_blocked(result)

	def _enforce_team_policy(
		self,
		tenant_id: str,
		agent_count: int,
		agent_ids: list[str] | tuple[str, ...],
		handoffs: tuple[HandoffEdge, ...],
	) -> None:
		unknown_agents = [agent_id for agent_id in agent_ids if agent_id not in self._agents]
		endpoints = set(agent_ids)
		unresolved_handoff = any(edge.source not in endpoints or edge.target not in endpoints for edge in handoffs)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_team",
			"agent_count": agent_count,
			"runtime_registered": True,
			"handoff_endpoint_resolved": not unknown_agents and not unresolved_handoff,
		})
		_raise_if_blocked(result)
		if unknown_agents:
			raise KeyError(f"unknown team agent(s): {', '.join(unknown_agents)}")


def _default_runtimes() -> tuple[AgentRuntime, ...]:
	return (
		AgentRuntime("local", kind="local", capabilities=("offline", "deterministic")),
		AgentRuntime("codex", kind="external", approved=True, external_runtime=True, workspace_runtime=True, capabilities=("code", "tests", "docs"), cost_limit=25.0),
		AgentRuntime("claude_code", kind="external", approved=True, external_runtime=True, workspace_runtime=True, capabilities=("code", "analysis"), cost_limit=25.0),
		AgentRuntime("opencode", kind="external", approved=True, external_runtime=True, workspace_runtime=True, capabilities=("code", "shell"), cost_limit=15.0),
		AgentRuntime("pi", kind="external", approved=True, external_runtime=True, capabilities=("conversation", "assistant"), cost_limit=10.0),
	)


def _raise_if_blocked(result: dict[str, Any]) -> None:
	if result["decision"] == "allow":
		return
	reasons = ", ".join(action.get("reason", "agent_composition_policy_blocked") for action in result["actions"])
	if result["decision"] == "require_review":
		raise PermissionError(reasons or "agent_composition_review_required")
	raise PermissionError(reasons or "agent_composition_policy_blocked")
