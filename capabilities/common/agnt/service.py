"""Service layer for first-class APG AI agent composition."""

from __future__ import annotations

from typing import Any

from .agent_composition import AgentCompositionPlanner
from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import AgentDefinition, AgentRuntime, AgentTeam, HandoffEdge
from .models import AgentAuditEvent, RuntimeApprovalRequest


class AgntService:
	"""In-memory agent registry, runtime registry, team validator, and plan builder."""

	def __init__(self) -> None:
		self._agents: dict[str, AgentDefinition] = {}
		self._runtimes: dict[str, AgentRuntime] = {}
		self._teams: dict[str, AgentTeam] = {}
		self._runtime_approvals: dict[str, RuntimeApprovalRequest] = {}
		self._events: list[AgentAuditEvent] = []
		self._planner = AgentCompositionPlanner()
		for runtime in _default_runtimes():
			self._runtimes[self._key(runtime.tenant_id, runtime.name)] = runtime

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
			cost_limit_present=cost_limit is not None,
		)
		runtime = AgentRuntime(
			name=name,
			tenant_id=tenant_id,
			kind=kind,
			approved=approved,
			workspace_runtime=workspace_runtime,
			external_runtime=external_runtime,
			sandbox_policy=sandbox_policy,
			capabilities=tuple(capabilities or ()),
			cost_limit=cost_limit,
		)
		key = self._key(tenant_id, name)
		if key in self._runtimes:
			raise ValueError(f"duplicate agent runtime for tenant: {name}")
		self._runtimes[key] = runtime
		self._record_event(
			tenant_id=tenant_id,
			event_type="runtime_registered",
			subject_id=name,
			message=f"Registered agent runtime {name}.",
			evidence={"kind": kind, "external_runtime": external_runtime, "workspace_runtime": workspace_runtime},
		)
		return runtime.to_dict()

	def list_runtimes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		runtimes = list(self._runtimes.values())
		if tenant_id is not None:
			runtimes = [runtime for runtime in runtimes if runtime.tenant_id in {tenant_id, "default"}]
		return [runtime.to_dict() for runtime in sorted(runtimes, key=lambda item: (item.tenant_id, item.name))]

	def request_runtime_approval(
		self,
		request_id: str,
		tenant_id: str,
		runtime_name: str,
		requested_by: str,
		kind: str = "external",
		workspace_runtime: bool = False,
		sandbox_policy: str | None = "workspace-read",
		capabilities: list[str] | tuple[str, ...] | None = None,
		cost_limit: float | None = None,
	) -> dict[str, Any]:
		self._enforce_runtime_policy(
			tenant_id=tenant_id,
			registered=True,
			workspace_runtime=workspace_runtime,
			sandbox_policy_attached=bool(sandbox_policy),
			external_runtime=False,
			approval_recorded=True,
			cost_limit_present=cost_limit is not None,
			operation="request_runtime_approval",
			runtime_requester_present=bool(requested_by),
		)
		if not runtime_name:
			raise ValueError("runtime_name is required")
		if not requested_by:
			raise ValueError("requested_by is required")
		request = RuntimeApprovalRequest(
			id=request_id,
			tenant_id=tenant_id,
			runtime_name=runtime_name,
			kind=kind,
			requested_by=requested_by,
			workspace_runtime=workspace_runtime,
			sandbox_policy=sandbox_policy,
			capabilities=tuple(capabilities or ()),
			cost_limit=cost_limit,
		)
		key = self._key(tenant_id, request_id)
		if key in self._runtime_approvals:
			raise ValueError(f"duplicate runtime approval request for tenant: {request_id}")
		self._runtime_approvals[key] = request
		self._record_event(
			tenant_id=tenant_id,
			event_type="runtime_approval_requested",
			subject_id=request_id,
			message=f"Requested approval for external runtime {runtime_name}.",
			evidence={"runtime_name": runtime_name, "requested_by": requested_by},
		)
		return request.to_dict()

	def decide_runtime_approval(
		self,
		request_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		self._enforce_approval_decision_policy(tenant_id, reviewer, notes)
		key = self._key(tenant_id, request_id)
		request = self._runtime_approvals.get(key)
		if request is None or request.tenant_id != tenant_id:
			raise KeyError(f"unknown runtime approval request for tenant: {request_id}")
		if decision not in {"approved", "rejected"}:
			raise ValueError("runtime approval decision must be approved or rejected")
		decided = RuntimeApprovalRequest(
			id=request.id,
			tenant_id=request.tenant_id,
			runtime_name=request.runtime_name,
			kind=request.kind,
			requested_by=request.requested_by,
			workspace_runtime=request.workspace_runtime,
			sandbox_policy=request.sandbox_policy,
			capabilities=request.capabilities,
			cost_limit=request.cost_limit,
			decision=decision,
			reviewer=reviewer,
			notes=notes,
		)
		self._runtime_approvals[key] = decided
		if decision == "approved":
			self.register_runtime(
				name=request.runtime_name,
				kind=request.kind,
				approved=True,
				workspace_runtime=request.workspace_runtime,
				external_runtime=True,
				sandbox_policy=request.sandbox_policy,
				capabilities=request.capabilities,
				cost_limit=request.cost_limit,
				tenant_id=tenant_id,
			)
		self._record_event(
			tenant_id=tenant_id,
			event_type="runtime_approval_decided",
			subject_id=request_id,
			message=f"Runtime approval {request_id} was {decision}.",
			evidence={"runtime_name": request.runtime_name, "reviewer": reviewer},
		)
		return decided.to_dict()

	def list_runtime_approvals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		requests = list(self._runtime_approvals.values())
		if tenant_id is not None:
			requests = [request for request in requests if request.tenant_id == tenant_id]
		return [request.to_dict() for request in sorted(requests, key=lambda item: item.id)]

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = list(self._events)
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.to_dict() for event in events]

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
		runtime_record = self._get_runtime(tenant_id, runtime)
		self._enforce_agent_policy(
			tenant_id=tenant_id,
			model_present=bool(model),
			system_prompt_present=bool(system_prompt),
			tool_allowlist_present=bool(tool_allowlist),
			io_contract_present=bool(input_contract) and bool(output_contract),
			memory_policy_present=bool(memory_policy),
			runtime_registered=runtime_record is not None and runtime_record.registered,
			workspace_runtime=bool(runtime_record and runtime_record.workspace_runtime),
			sandbox_policy_attached=bool(runtime_record and runtime_record.sandbox_policy),
		)
		if runtime_record and runtime_record.tenant_id not in {tenant_id, "default"}:
			raise KeyError(f"unknown runtime for tenant: {runtime}")
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
		key = self._key(tenant_id, agent_id)
		if key in self._agents:
			raise ValueError(f"duplicate agent for tenant: {agent_id}")
		self._agents[key] = agent
		self._record_event(
			tenant_id=tenant_id,
			event_type="agent_registered",
			subject_id=agent_id,
			message=f"Registered agent {name}.",
			evidence={"runtime": runtime, "model": model},
		)
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
		key = self._key(tenant_id, team_id)
		if key in self._teams:
			raise ValueError(f"duplicate agent team for tenant: {team_id}")
		self._teams[key] = team
		self._record_event(
			tenant_id=tenant_id,
			event_type="team_registered",
			subject_id=team_id,
			message=f"Registered agent team {name}.",
			evidence={"agent_count": len(agent_ids), "handoff_count": len(edges)},
		)
		return team.to_dict()

	def list_teams(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		teams = list(self._teams.values())
		if tenant_id is not None:
			teams = [team for team in teams if team.tenant_id == tenant_id]
		return [team.to_dict() for team in sorted(teams, key=lambda item: item.id)]

	def plan_execution(self, team_id: str, objective: str, tenant_id: str | None = None) -> dict[str, Any]:
		if not objective:
			result = self.evaluate({
				"tenant_context_present": bool(tenant_id),
				"operation": "plan_execution",
				"objective_present": False,
			})
			_raise_if_blocked(result)
		team = self._get_team(team_id, tenant_id)
		if team is None:
			if tenant_id is not None:
				raise KeyError(f"unknown agent team for tenant: {team_id}")
			raise KeyError(f"unknown agent team: {team_id}")
		if tenant_id is not None and team.tenant_id != tenant_id:
			raise KeyError(f"unknown agent team for tenant: {team_id}")
		plan = self._planner.build_plan(
			team=team,
			agents=self._tenant_agent_map(team.tenant_id),
			runtimes=self._tenant_runtime_map(team.tenant_id),
			objective=objective,
		).to_dict()
		self._record_event(
			tenant_id=team.tenant_id,
			event_type="execution_plan_built",
			subject_id=plan["id"],
			message=f"Built execution plan for team {team.id}.",
			evidence={"step_count": len(plan["steps"]), "objective": objective},
		)
		return plan

	def composition_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		contract = self.describe(tenant_id)
		return {
			"capability": contract["capability"],
			"display_name": contract["display_name"],
			"tenant_id": tenant_id,
			"agent_count": len(self.list_agents(tenant_id)),
			"team_count": len(self.list_teams(tenant_id)),
			"runtime_count": len(self.list_runtimes(tenant_id)),
			"runtime_approval_count": len(self.list_runtime_approvals(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"routes": contract["ui"]["routes"],
			"theme": contract["theme"],
			"streaming": contract["streaming"],
		}

	def _enforce_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		_raise_if_blocked(result)

	def _enforce_agent_policy(
		self,
		tenant_id: str,
		model_present: bool,
		system_prompt_present: bool,
		tool_allowlist_present: bool,
		io_contract_present: bool,
		memory_policy_present: bool,
		runtime_registered: bool,
		workspace_runtime: bool,
		sandbox_policy_attached: bool,
	) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_agent",
			"model_present": model_present,
			"system_prompt_present": system_prompt_present,
			"tool_allowlist_present": tool_allowlist_present,
			"io_contract_present": io_contract_present,
			"memory_policy_present": memory_policy_present,
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
		cost_limit_present: bool,
		operation: str = "register_runtime",
		runtime_requester_present: bool = True,
	) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": operation,
			"runtime_registered": registered,
			"workspace_runtime": workspace_runtime,
			"sandbox_policy_attached": sandbox_policy_attached,
			"external_runtime": external_runtime,
			"approval_recorded": approval_recorded,
			"cost_limit_present": cost_limit_present,
			"runtime_requester_present": runtime_requester_present,
		})
		_raise_if_blocked(result)

	def _enforce_approval_decision_policy(self, tenant_id: str, reviewer: str, notes: str) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "decide_runtime_approval",
			"reviewer_present": bool(reviewer),
			"decision_notes_present": bool(notes),
		})
		_raise_if_blocked(result)

	def _enforce_team_policy(
		self,
		tenant_id: str,
		agent_count: int,
		agent_ids: list[str] | tuple[str, ...],
		handoffs: tuple[HandoffEdge, ...],
	) -> None:
		tenant_agents = self._tenant_agent_map(tenant_id)
		unknown_agents = [agent_id for agent_id in agent_ids if agent_id not in tenant_agents]
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

	def validate_batch_agent_mutation(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "batch_agent_mutation",
			"event_stream": event_stream,
			"mutation_count": mutation_count,
		})
		_raise_if_blocked(result)
		return {
			"tenant_id": tenant_id,
			"event_stream": event_stream,
			"mutation_count": mutation_count,
			"accepted": True,
			"rule_result": result,
		}

	def _key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"

	def _get_runtime(self, tenant_id: str, runtime_name: str) -> AgentRuntime | None:
		return self._runtimes.get(self._key(tenant_id, runtime_name)) or self._runtimes.get(self._key("default", runtime_name))

	def _get_team(self, team_id: str, tenant_id: str | None = None) -> AgentTeam | None:
		if tenant_id is not None:
			return self._teams.get(self._key(tenant_id, team_id))
		matches = [team for team in self._teams.values() if team.id == team_id]
		return matches[0] if len(matches) == 1 else None

	def _tenant_agent_map(self, tenant_id: str) -> dict[str, AgentDefinition]:
		return {agent.id: agent for agent in self._agents.values() if agent.tenant_id == tenant_id}

	def _tenant_runtime_map(self, tenant_id: str) -> dict[str, AgentRuntime]:
		runtime_map = {runtime.name: runtime for runtime in self._runtimes.values() if runtime.tenant_id == "default"}
		runtime_map.update({runtime.name: runtime for runtime in self._runtimes.values() if runtime.tenant_id == tenant_id})
		return runtime_map

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		evidence: dict[str, Any] | None = None,
	) -> None:
		self._events.append(
			AgentAuditEvent(
				id=f"agnt-event-{len(self._events) + 1}",
				tenant_id=tenant_id,
				event_type=event_type,
				subject_id=subject_id,
				message=message,
				evidence=dict(evidence or {}),
			)
		)


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
