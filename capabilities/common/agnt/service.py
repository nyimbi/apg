"""Service layer for first-class APG AI agent composition."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import time
from typing import Any, AsyncGenerator

from .agent_composition import AgentCompositionPlanner
from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import AgentDefinition, AgentExecutionRun, AgentRuntime, AgentTeam, HandoffEdge
from .models import AgentAuditEvent, RuntimeApprovalRequest
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache


def _now() -> str:
	from datetime import datetime, timezone
	return datetime.now(timezone.utc).isoformat()


class AgntService:
	"""In-memory agent registry, runtime registry, team validator, plan builder, and execution engine."""

	def __init__(self, db_url: str | None = None) -> None:
		self._agents: dict[str, AgentDefinition] = {}
		self._runtimes: dict[str, AgentRuntime] = {}
		self._teams: dict[str, AgentTeam] = {}
		self._execution_runs: dict[str, AgentExecutionRun] = {}
		self._runtime_approvals: dict[str, RuntimeApprovalRequest] = {}
		self._events: list[AgentAuditEvent] = []
		self._planner = AgentCompositionPlanner()

		# Extended stores
		_store = get_store(db_url)
		self._tools = WriteThruDict('tools', tenant_id, _store)           # tool_name -> spec
		self._agent_tools: dict[str, set[str]] = {}           # agent_key -> {tool_names}
		self._memory_store: dict[str, list[dict[str, Any]]] = {}  # agent_key -> [chunks]
		self._working_memory: dict[str, list[dict[str, Any]]] = {}  # session_id -> [turns]
		self._session_contexts = WriteThruDict('session_contexts', tenant_id, _store)     # session_id -> context
		self._cost_ledger: dict[str, list[dict[str, Any]]] = {}    # agent_key -> [{cost, ts}]
		self._guardrail_hits: dict[str, list[dict[str, Any]]] = {} # agent_key -> [hits]
		self._ab_results = WriteThruDict('ab_results', tenant_id, _store)

		for runtime in _default_runtimes():
			self._runtimes[self._key(runtime.tenant_id, runtime.name)] = runtime

	# ── original capability contract helpers ─────────────────────────────────

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ── REGISTRY ──────────────────────────────────────────────────────────────

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
			tenant_id=tenant_id, registered=True, workspace_runtime=workspace_runtime,
			sandbox_policy_attached=bool(sandbox_policy), external_runtime=external_runtime,
			approval_recorded=approved, cost_limit_present=cost_limit is not None,
		)
		runtime = AgentRuntime(
			name=name, tenant_id=tenant_id, kind=kind, approved=approved,
			workspace_runtime=workspace_runtime, external_runtime=external_runtime,
			sandbox_policy=sandbox_policy, capabilities=tuple(capabilities or ()),
			cost_limit=cost_limit,
		)
		key = self._key(tenant_id, name)
		if key in self._runtimes:
			raise ValueError(f"duplicate agent runtime for tenant: {name}")
		self._runtimes[key] = runtime
		self._record_event(tenant_id=tenant_id, event_type="runtime_registered", subject_id=name,
						   message=f"Registered agent runtime {name}.",
						   evidence={"kind": kind, "external_runtime": external_runtime})
		return runtime.to_dict()

	def list_runtimes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		runtimes = list(self._runtimes.values())
		if tenant_id is not None:
			runtimes = [r for r in runtimes if r.tenant_id in {tenant_id, "default"}]
		return [r.to_dict() for r in sorted(runtimes, key=lambda r: (r.tenant_id, r.name))]

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
			tenant_id=tenant_id, registered=True, workspace_runtime=workspace_runtime,
			sandbox_policy_attached=bool(sandbox_policy), external_runtime=False,
			approval_recorded=True, cost_limit_present=cost_limit is not None,
			operation="request_runtime_approval", runtime_requester_present=bool(requested_by),
		)
		if not runtime_name:
			raise ValueError("runtime_name is required")
		if not requested_by:
			raise ValueError("requested_by is required")
		review_result = self.evaluate({
			"tenant_context_present": bool(tenant_id), "operation": "register_runtime",
			"runtime_registered": True, "workspace_runtime": workspace_runtime,
			"sandbox_policy_attached": bool(sandbox_policy), "external_runtime": True,
			"approval_recorded": False, "cost_limit_present": True,
		})
		request = RuntimeApprovalRequest(
			id=request_id, tenant_id=tenant_id, runtime_name=runtime_name, kind=kind,
			requested_by=requested_by, workspace_runtime=workspace_runtime,
			sandbox_policy=sandbox_policy, capabilities=tuple(capabilities or ()),
			cost_limit=cost_limit, policy_decision=review_result["decision"],
			matched_rules=list(review_result["matched_rules"]),
			review_reasons=self._review_reasons(review_result),
			audit_evidence=self._audit_evidence(review_result),
		)
		key = self._key(tenant_id, request_id)
		if key in self._runtime_approvals:
			raise ValueError(f"duplicate runtime approval request for tenant: {request_id}")
		self._runtime_approvals[key] = request
		self._record_event(tenant_id=tenant_id, event_type="runtime_approval_requested",
						   subject_id=request_id,
						   message=f"Requested approval for external runtime {runtime_name}.",
						   evidence={"runtime_name": runtime_name, "requested_by": requested_by},
						   policy_result=review_result)
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
			id=request.id, tenant_id=request.tenant_id, runtime_name=request.runtime_name,
			kind=request.kind, requested_by=request.requested_by,
			workspace_runtime=request.workspace_runtime, sandbox_policy=request.sandbox_policy,
			capabilities=request.capabilities, cost_limit=request.cost_limit,
			decision=decision, policy_decision=request.policy_decision,
			matched_rules=list(request.matched_rules), review_reasons=list(request.review_reasons),
			audit_evidence=dict(request.audit_evidence), reviewer=reviewer, notes=notes,
		)
		self._runtime_approvals[key] = decided
		if decision == "approved":
			self.register_runtime(
				name=request.runtime_name, kind=request.kind, approved=True,
				workspace_runtime=request.workspace_runtime, external_runtime=True,
				sandbox_policy=request.sandbox_policy, capabilities=request.capabilities,
				cost_limit=request.cost_limit, tenant_id=tenant_id,
			)
		self._record_event(tenant_id=tenant_id, event_type="runtime_approval_decided",
						   subject_id=request_id,
						   message=f"Runtime approval {request_id} was {decision}.",
						   evidence={"runtime_name": request.runtime_name, "reviewer": reviewer})
		return decided.to_dict()

	def list_runtime_approvals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		requests = list(self._runtime_approvals.values())
		if tenant_id is not None:
			requests = [r for r in requests if r.tenant_id == tenant_id]
		return [r.to_dict() for r in sorted(requests, key=lambda r: r.id)]

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = list(self._events)
		if tenant_id is not None:
			events = [e for e in events if e.tenant_id == tenant_id]
		return [e.to_dict() for e in events]

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
			tenant_id=tenant_id, model_present=bool(model),
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
			id=agent_id, tenant_id=tenant_id, name=name, model=model, runtime=runtime,
			system_prompt=system_prompt, tool_allowlist=tuple(tool_allowlist or ()),
			input_contract=dict(input_contract or {}), output_contract=dict(output_contract or {}),
			memory_policy=dict(memory_policy or {}), status=status,
		)
		key = self._key(tenant_id, agent_id)
		if key in self._agents:
			raise ValueError(f"duplicate agent for tenant: {agent_id}")
		self._agents[key] = agent
		self._agent_tools[key] = set(tool_allowlist or [])
		self._memory_store[key] = []
		self._cost_ledger[key] = []
		self._guardrail_hits[key] = []
		self._record_event(tenant_id=tenant_id, event_type="agent_registered", subject_id=agent_id,
						   message=f"Registered agent {name}.", evidence={"runtime": runtime, "model": model})
		return agent.to_dict()

	def list_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		agents = list(self._agents.values())
		if tenant_id is not None:
			agents = [a for a in agents if a.tenant_id == tenant_id]
		return [a.to_dict() for a in sorted(agents, key=lambda a: a.id)]

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
			HandoffEdge(source=str(e["source"]), target=str(e["target"]),
						trigger=str(e.get("trigger") or "complete"), condition=str(e.get("condition") or "always"))
			for e in (handoffs or ())
		)
		self._enforce_team_policy(tenant_id, len(agent_ids), agent_ids, edges)
		team = AgentTeam(id=team_id, tenant_id=tenant_id, name=name, agent_ids=tuple(agent_ids),
						 handoffs=edges, execution_mode=execution_mode,
						 parallel_execution_enabled=parallel_execution_enabled)
		key = self._key(tenant_id, team_id)
		if key in self._teams:
			raise ValueError(f"duplicate agent team for tenant: {team_id}")
		self._teams[key] = team
		self._record_event(tenant_id=tenant_id, event_type="team_registered", subject_id=team_id,
						   message=f"Registered agent team {name}.",
						   evidence={"agent_count": len(agent_ids), "handoff_count": len(edges)})
		return team.to_dict()

	def list_teams(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		teams = list(self._teams.values())
		if tenant_id is not None:
			teams = [t for t in teams if t.tenant_id == tenant_id]
		return [t.to_dict() for t in sorted(teams, key=lambda t: t.id)]

	def plan_execution(self, team_id: str, objective: str, tenant_id: str | None = None) -> dict[str, Any]:
		if not objective:
			result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "plan_execution",
									"objective_present": False})
			_raise_if_blocked(result)
		team = self._get_team(team_id, tenant_id)
		if team is None:
			raise KeyError(f"unknown agent team for tenant: {team_id}" if tenant_id else f"unknown agent team: {team_id}")
		if tenant_id is not None and team.tenant_id != tenant_id:
			raise KeyError(f"unknown agent team for tenant: {team_id}")
		plan = self._planner.build_plan(
			team=team, agents=self._tenant_agent_map(team.tenant_id),
			runtimes=self._tenant_runtime_map(team.tenant_id), objective=objective,
		).to_dict()
		self._record_event(tenant_id=team.tenant_id, event_type="execution_plan_built",
						   subject_id=plan["id"], message=f"Built execution plan for team {team.id}.",
						   evidence={"step_count": len(plan["steps"]), "objective": objective})
		return plan

	def record_execution_run(
		self,
		run_id: str,
		tenant_id: str,
		team_id: str,
		objective: str,
		requested_by: str,
		trace_sink: str,
		side_effects_requested: bool = False,
		human_approval_recorded: bool = False,
		status: str = "planned",
	) -> dict[str, Any]:
		result = self._enforce_execution_run_policy(
			tenant_id=tenant_id, requester_present=bool(requested_by),
			trace_sink_present=bool(trace_sink), side_effects_requested=side_effects_requested,
			human_approval_recorded=human_approval_recorded,
		)
		if status not in {"planned", "running", "completed", "failed", "cancelled"}:
			raise ValueError("execution run status must be planned, running, completed, failed, or cancelled")
		key = self._key(tenant_id, run_id)
		if key in self._execution_runs:
			raise ValueError(f"duplicate execution run for tenant: {run_id}")
		plan = self.plan_execution(team_id, objective, tenant_id=tenant_id)
		run = AgentExecutionRun(
			id=run_id, tenant_id=tenant_id, team_id=team_id, plan_id=plan["id"], objective=objective,
			requested_by=requested_by, trace_sink=trace_sink,
			status="pending_review" if result["decision"] == "require_review" else status,
			side_effects_requested=side_effects_requested, human_approval_recorded=human_approval_recorded,
			decision=result["decision"], matched_rules=list(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
			audit_evidence=self._audit_evidence(result, human_approval_recorded), plan_snapshot=plan,
		)
		self._execution_runs[key] = run
		self._record_event(tenant_id=tenant_id, event_type="execution_run_recorded", subject_id=run_id,
						   message=f"Recorded agent execution run {run_id}.",
						   evidence={"team_id": team_id, "plan_id": plan["id"],
									 "requested_by": requested_by, "trace_sink": trace_sink},
						   policy_result=result)
		return run.to_dict()

	def list_execution_runs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		runs = list(self._execution_runs.values())
		if tenant_id is not None:
			runs = [r for r in runs if r.tenant_id == tenant_id]
		return [r.to_dict() for r in sorted(runs, key=lambda r: r.id)]

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [item for item in (self.list_runtime_approvals(tenant_id) + self.list_execution_runs(tenant_id))
				if item.get("decision") == "pending" or item.get("status") == "pending_review"]

	def composition_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		contract = self.describe(tenant_id)
		pending_reviews = self.list_pending_reviews(tenant_id)
		return {
			"capability": contract["capability"], "display_name": contract["display_name"],
			"tenant_id": tenant_id, "agent_count": len(self.list_agents(tenant_id)),
			"team_count": len(self.list_teams(tenant_id)),
			"execution_run_count": len(self.list_execution_runs(tenant_id)),
			"runtime_count": len(self.list_runtimes(tenant_id)),
			"runtime_approval_count": len(self.list_runtime_approvals(tenant_id)),
			"pending_review_count": len(pending_reviews),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"routes": contract["ui"]["routes"], "theme": contract["theme"],
			"streaming": contract["streaming"],
		}

	def validate_batch_agent_mutation(self, tenant_id: str, event_stream: str, mutation_count: int) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "batch_agent_mutation",
								"event_stream": event_stream, "mutation_count": mutation_count})
		_raise_if_blocked(result)
		return {"tenant_id": tenant_id, "event_stream": event_stream, "mutation_count": mutation_count,
				"accepted": True, "rule_result": result}

	# ── REGISTRY EXTENSIONS ───────────────────────────────────────────────────

	async def update_agent_config(self, agent_id: str, config: dict[str, Any], tenant_id: str = "default") -> dict[str, Any]:
		"""Patch mutable agent configuration fields."""
		key = self._key(tenant_id, agent_id)
		agent = self._agents.get(key)
		if agent is None:
			raise KeyError(f"unknown agent: {agent_id}")
		updatable = {"system_prompt", "model", "status", "memory_policy"}
		for field, value in config.items():
			if field in updatable:
				object.__setattr__(agent, field, value)
		self._record_event(tenant_id=tenant_id, event_type="agent_config_updated",
						   subject_id=agent_id, message=f"Updated agent config for {agent_id}.",
						   evidence={"fields": list(config.keys())})
		return agent.to_dict()

	async def deactivate_agent(self, agent_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Mark agent as inactive without deleting."""
		key = self._key(tenant_id, agent_id)
		agent = self._agents.get(key)
		if agent is None:
			raise KeyError(f"unknown agent: {agent_id}")
		object.__setattr__(agent, "status", "inactive")
		self._record_event(tenant_id=tenant_id, event_type="agent_deactivated",
						   subject_id=agent_id, message=f"Deactivated agent {agent_id}.", evidence={})
		return {"agent_id": agent_id, "status": "inactive", "ts": _now()}

	async def get_agent_manifest(self, agent_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Return complete agent manifest including tools, memory policy, contracts."""
		key = self._key(tenant_id, agent_id)
		agent = self._agents.get(key)
		if agent is None:
			raise KeyError(f"unknown agent: {agent_id}")
		return {
			**agent.to_dict(),
			"assigned_tools": list(self._agent_tools.get(key, set())),
			"memory_chunk_count": len(self._memory_store.get(key, [])),
			"cost_events": len(self._cost_ledger.get(key, [])),
			"guardrail_hits": len(self._guardrail_hits.get(key, [])),
		}

	async def validate_agent_config(self, config: dict[str, Any]) -> dict[str, Any]:
		"""Validate an agent config dict without persisting it."""
		required = {"model", "system_prompt", "runtime"}
		missing = required - set(config.keys())
		if missing:
			return {"valid": False, "missing_fields": list(missing)}
		issues: list[str] = []
		if not config.get("model"):
			issues.append("model_empty")
		if not config.get("system_prompt"):
			issues.append("system_prompt_empty")
		runtime_name = config.get("runtime", "")
		runtime_known = any(r.name == runtime_name for r in self._runtimes.values())
		if not runtime_known:
			issues.append(f"unknown_runtime:{runtime_name}")
		return {"valid": len(issues) == 0, "issues": issues}

	async def agent_version_history(self, agent_id: str, tenant_id: str = "default") -> list[dict[str, Any]]:
		"""Return audit events related to this agent as version history."""
		return [e.to_dict() for e in self._events
				if e.subject_id == agent_id and e.tenant_id == tenant_id]

	# ── EXECUTION ─────────────────────────────────────────────────────────────

	async def execute_pipeline(
		self,
		agent_ids: list[str],
		task: dict[str, Any],
		strategy: str = "sequential",
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Execute a pipeline of agents on a task."""
		assert agent_ids, "agent_ids must not be empty"
		results: list[dict[str, Any]] = []
		context = dict(task)
		t0 = time.monotonic()
		if strategy == "sequential":
			for aid in agent_ids:
				key = self._key(tenant_id, aid)
				agent = self._agents.get(key)
				if agent is None:
					results.append({"agent_id": aid, "status": "skipped", "error": "agent_not_found"})
					continue
				step_result = await self._simulate_agent_step(agent, context)
				context.update(step_result.get("output", {}))
				cost = step_result.get("cost_usd", 0.0)
				self._cost_ledger.setdefault(key, []).append({"cost_usd": cost, "ts": _now(), "task": str(task)[:80]})
				results.append({"agent_id": aid, **step_result})
		elif strategy == "parallel":
			async def _run(aid: str) -> dict[str, Any]:
				key = self._key(tenant_id, aid)
				agent = self._agents.get(key)
				if agent is None:
					return {"agent_id": aid, "status": "skipped", "error": "agent_not_found"}
				step_result = await self._simulate_agent_step(agent, context)
				cost = step_result.get("cost_usd", 0.0)
				self._cost_ledger.setdefault(key, []).append({"cost_usd": cost, "ts": _now()})
				return {"agent_id": aid, **step_result}
			results = list(await asyncio.gather(*[_run(aid) for aid in agent_ids]), return_exceptions=True)

		else:
			raise ValueError(f"unknown strategy: {strategy}")
		elapsed = time.monotonic() - t0
		return {"pipeline_status": "completed", "strategy": strategy, "elapsed_s": elapsed,
				"step_results": results, "ts": _now()}

	async def execute_parallel_agents(
		self,
		agent_task_pairs: list[tuple[str, dict[str, Any]]],
		tenant_id: str = "default",
	) -> list[dict[str, Any]]:
		"""Execute heterogeneous agent/task pairs concurrently."""
		async def _run(aid: str, task: dict[str, Any]) -> dict[str, Any]:
			key = self._key(tenant_id, aid)
			agent = self._agents.get(key)
			if agent is None:
				return {"agent_id": aid, "status": "error", "error": "not_found"}
			return {"agent_id": aid, **(await self._simulate_agent_step(agent, task))}
		return list(await asyncio.gather(*[_run(aid, t) for aid, t in agent_task_pairs]), return_exceptions=True)


	async def stream_execution(
		self,
		agent_id: str,
		task: dict[str, Any],
		tenant_id: str = "default",
	) -> AsyncGenerator[dict[str, Any], None]:
		"""Yield incremental output tokens from a streaming agent execution."""
		key = self._key(tenant_id, agent_id)
		agent = self._agents.get(key)
		if agent is None:
			yield {"event": "error", "error": "agent_not_found"}
			return
		yield {"event": "start", "agent_id": agent_id, "ts": _now()}
		tokens = [f"step_{i}" for i in range(5)]
		for token in tokens:
			await asyncio.sleep(0)   # yield control
			yield {"event": "token", "token": token, "ts": _now()}
		result = await self._simulate_agent_step(agent, task)
		self._cost_ledger.setdefault(key, []).append({"cost_usd": result.get("cost_usd", 0.0), "ts": _now()})
		yield {"event": "complete", "result": result, "ts": _now()}

	async def resume_execution(self, session_id: str, context: dict[str, Any]) -> dict[str, Any]:
		"""Resume a paused or interrupted execution session."""
		existing = self._session_contexts.get(session_id)
		if existing is None:
			return {"status": "error", "error": "session_not_found", "session_id": session_id}
		existing.update(context)
		existing["status"] = "resumed"
		existing["resumed_at"] = _now()
		return {"status": "resumed", "session_id": session_id, "context": existing}

	async def cancel_execution(self, session_id: str) -> dict[str, Any]:
		"""Cancel an in-flight execution session."""
		ctx = self._session_contexts.get(session_id)
		if ctx is None:
			return {"cancelled": False, "error": "session_not_found"}
		ctx["status"] = "cancelled"
		ctx["cancelled_at"] = _now()
		return {"cancelled": True, "session_id": session_id, "ts": _now()}

	async def execution_history(self, agent_id: str, limit: int = 20, tenant_id: str = "default") -> list[dict[str, Any]]:
		"""Return recent execution events for an agent."""
		return [e.to_dict() for e in self._events
				if e.subject_id == agent_id and e.tenant_id == tenant_id][-limit:]

	async def execution_cost_estimate(self, agent_id: str, task: dict[str, Any], tenant_id: str = "default") -> dict[str, Any]:
		"""Estimate execution cost for a task based on model pricing."""
		key = self._key(tenant_id, agent_id)
		agent = self._agents.get(key)
		if agent is None:
			return {"error": "agent_not_found"}
		task_len = len(str(task))
		# Rough token approximation: 4 chars per token, $0.01 per 1k tokens
		estimated_tokens = task_len / 4 + 500
		model_rates: dict[str, float] = {
			"gpt-4": 0.03, "gpt-3.5-turbo": 0.002, "claude-3-opus": 0.015,
			"claude-3-sonnet": 0.003, "llama3": 0.001,
		}
		rate = model_rates.get(agent.model, 0.005)
		estimated_cost = (estimated_tokens / 1000) * rate
		return {"agent_id": agent_id, "model": agent.model, "estimated_tokens": int(estimated_tokens),
				"estimated_cost_usd": round(estimated_cost, 6), "rate_per_1k": rate}

	async def batch_execute(
		self,
		agent_id: str,
		tasks_list: list[dict[str, Any]],
		tenant_id: str = "default",
	) -> list[dict[str, Any]]:
		"""Execute the same agent over a batch of tasks concurrently."""
		key = self._key(tenant_id, agent_id)
		agent = self._agents.get(key)
		if agent is None:
			return [{"error": "agent_not_found"} for _ in tasks_list]
		results = await asyncio.gather(*[self._simulate_agent_step(agent, t) for t in tasks_list], return_exceptions=True)

		total_cost = sum(r.get("cost_usd", 0.0) for r in results)
		self._cost_ledger.setdefault(key, []).append(
			{"cost_usd": total_cost, "ts": _now(), "batch_size": len(tasks_list)})
		return [{"agent_id": agent_id, **r} for r in results]

	# ── TOOLS ─────────────────────────────────────────────────────────────────

	async def assign_tool(self, agent_id: str, tool_name: str, access_level: str = "read",
						  tenant_id: str = "default") -> dict[str, Any]:
		"""Grant an agent access to a named tool."""
		key = self._key(tenant_id, agent_id)
		if key not in self._agents:
			raise KeyError(f"unknown agent: {agent_id}")
		self._agent_tools.setdefault(key, set()).add(tool_name)
		self._record_event(tenant_id=tenant_id, event_type="tool_assigned", subject_id=agent_id,
						   message=f"Assigned tool {tool_name} to {agent_id}.",
						   evidence={"tool": tool_name, "access_level": access_level})
		return {"agent_id": agent_id, "tool_name": tool_name, "access_level": access_level, "ts": _now()}

	async def revoke_tool(self, agent_id: str, tool_name: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Revoke an agent's access to a named tool."""
		key = self._key(tenant_id, agent_id)
		tools = self._agent_tools.get(key, set())
		was_present = tool_name in tools
		tools.discard(tool_name)
		return {"agent_id": agent_id, "tool_name": tool_name, "revoked": was_present, "ts": _now()}

	async def list_available_tools(self, agent_id: str, tenant_id: str = "default") -> list[dict[str, Any]]:
		"""List tools the agent is allowed to invoke."""
		key = self._key(tenant_id, agent_id)
		tools = self._agent_tools.get(key, set())
		return [{"tool_name": t, "spec": self._tools.get(t, {})} for t in sorted(tools)]

	async def tool_usage_report(self, tool_name: str, period: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Count how many agents have this tool assigned and audit invocations."""
		agents_with_tool = [
			k.split(":", 1)[1] for k, tools in self._agent_tools.items()
			if tool_name in tools and k.startswith(tenant_id + ":")
		]
		invocations = [e for e in self._events
					   if e.evidence.get("tool") == tool_name and e.tenant_id == tenant_id]
		return {"tool_name": tool_name, "period": period, "assigned_agent_count": len(agents_with_tool),
				"invocation_count": len(invocations), "agents": agents_with_tool}

	async def validate_tool_invocation(
		self,
		agent_id: str,
		tool_name: str,
		parameters: dict[str, Any],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Validate that an agent is allowed to invoke a tool with given parameters."""
		key = self._key(tenant_id, agent_id)
		if key not in self._agents:
			return {"allowed": False, "reason": "agent_not_found"}
		tools = self._agent_tools.get(key, set())
		if tool_name not in tools:
			return {"allowed": False, "reason": "tool_not_assigned"}
		spec = self._tools.get(tool_name, {})
		required_params = set(spec.get("required_params", []))
		missing = required_params - set(parameters.keys())
		if missing:
			return {"allowed": False, "reason": "missing_params", "missing": list(missing)}
		return {"allowed": True, "agent_id": agent_id, "tool_name": tool_name}

	# ── MEMORY ────────────────────────────────────────────────────────────────

	async def store_memory(self, agent_id: str, content: dict[str, Any], tenant_id: str = "default") -> dict[str, Any]:
		"""Append a memory chunk for an agent."""
		key = self._key(tenant_id, agent_id)
		chunk = {"id": f"mem_{len(self._memory_store.get(key, []))}", "content": content, "ts": _now()}
		self._memory_store.setdefault(key, []).append(chunk)
		return {"stored": True, "chunk_id": chunk["id"]}

	async def retrieve_memory(self, agent_id: str, query: str, top_k: int = 5,
							  tenant_id: str = "default") -> list[dict[str, Any]]:
		"""Retrieve top_k relevant memory chunks via keyword overlap."""
		key = self._key(tenant_id, agent_id)
		chunks = self._memory_store.get(key, [])
		terms = set(query.lower().split())

		def _score(chunk: dict[str, Any]) -> int:
			text = str(chunk.get("content", "")).lower()
			return sum(1 for t in terms if t in text)

		ranked = sorted(chunks, key=_score, reverse=True)
		return ranked[:top_k]

	async def compress_memory(self, agent_id: str, strategy: str = "truncate",
							  tenant_id: str = "default") -> dict[str, Any]:
		"""Compress agent memory according to strategy (truncate | deduplicate | summarise)."""
		key = self._key(tenant_id, agent_id)
		chunks = self._memory_store.get(key, [])
		before = len(chunks)
		if strategy == "truncate":
			self._memory_store[key] = chunks[-100:]
		elif strategy == "deduplicate":
			seen: set[str] = set()
			deduped = []
			for c in chunks:
				sig = str(c.get("content", ""))[:128]
				if sig not in seen:
					seen.add(sig)
					deduped.append(c)
			self._memory_store[key] = deduped
		elif strategy == "summarise":
			# Stub: in production this would call an LLM summarisation step
			self._memory_store[key] = [{"id": "summary_0",
										"content": {"summary": f"Compressed {before} chunks"},
										"ts": _now(), "compressed": True}]
		after = len(self._memory_store[key])
		return {"agent_id": agent_id, "strategy": strategy, "before": before, "after": after}

	async def export_memory(self, agent_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Export all memory chunks for an agent."""
		key = self._key(tenant_id, agent_id)
		chunks = self._memory_store.get(key, [])
		return {"agent_id": agent_id, "chunk_count": len(chunks), "chunks": chunks,
				"exported_at": _now()}

	async def memory_usage_report(self, agent_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Return memory utilisation statistics for an agent."""
		key = self._key(tenant_id, agent_id)
		chunks = self._memory_store.get(key, [])
		total_chars = sum(len(str(c.get("content", ""))) for c in chunks)
		return {"agent_id": agent_id, "chunk_count": len(chunks),
				"approx_tokens": total_chars // 4, "total_chars": total_chars}

	async def clear_working_memory(self, session_id: str) -> dict[str, Any]:
		"""Clear transient working memory for a session."""
		cleared = len(self._working_memory.pop(session_id, []))
		self._session_contexts.pop(session_id, None)
		return {"session_id": session_id, "cleared_turns": cleared, "ts": _now()}

	# ── SAFETY & PERFORMANCE ──────────────────────────────────────────────────

	async def guardrail_check(self, agent_id: str, input_text: str,
							  tenant_id: str = "default") -> dict[str, Any]:
		"""Check input against guardrail rules (PII, toxicity, prompt injection)."""
		key = self._key(tenant_id, agent_id)
		violations: list[str] = []
		lower = input_text.lower()
		if any(kw in lower for kw in ("ignore previous instructions", "jailbreak", "dan mode")):
			violations.append("prompt_injection")
		pii_patterns = ("password", "ssn", "social security", "credit card", "cvv")
		if any(p in lower for p in pii_patterns):
			violations.append("pii_detected")
		if len(input_text) > 50_000:
			violations.append("input_too_long")
		if violations:
			self._guardrail_hits.setdefault(key, []).append(
				{"violations": violations, "input_len": len(input_text), "ts": _now()})
		return {"passed": len(violations) == 0, "violations": violations,
				"agent_id": agent_id, "input_length": len(input_text)}

	async def cost_report(self, agent_id: str, period: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Return cost summary for an agent over period."""
		key = self._key(tenant_id, agent_id)
		ledger = self._cost_ledger.get(key, [])
		total = sum(e.get("cost_usd", 0.0) for e in ledger)
		return {"agent_id": agent_id, "period": period, "total_cost_usd": round(total, 6),
				"event_count": len(ledger), "ledger": ledger}

	async def performance_report(self, agent_id: str, period: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Return execution performance metrics for an agent."""
		events = [e for e in self._events if e.subject_id == agent_id and e.tenant_id == tenant_id]
		execution_events = [e for e in events if e.event_type in ("agent_step_completed", "execution_run_recorded")]
		return {"agent_id": agent_id, "period": period, "total_events": len(events),
				"execution_events": len(execution_events),
				"guardrail_hit_count": len(self._guardrail_hits.get(self._key(tenant_id, agent_id), []))}

	async def safety_audit(self, agent_id: str, period: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Return safety audit summary: guardrail hits, policy violations."""
		key = self._key(tenant_id, agent_id)
		hits = self._guardrail_hits.get(key, [])
		violation_counts: dict[str, int] = {}
		for hit in hits:
			for v in hit.get("violations", []):
				violation_counts[v] = violation_counts.get(v, 0) + 1
		return {"agent_id": agent_id, "period": period, "total_guardrail_hits": len(hits),
				"violation_breakdown": violation_counts, "ts": _now()}

	async def a_b_test(
		self,
		agent_id_a: str,
		agent_id_b: str,
		task_sample: list[dict[str, Any]],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Run both agents on a task sample and compare results."""
		results_a = await self.batch_execute(agent_id_a, task_sample, tenant_id)
		results_b = await self.batch_execute(agent_id_b, task_sample, tenant_id)
		successes_a = sum(1 for r in results_a if r.get("status") == "completed")
		successes_b = sum(1 for r in results_b if r.get("status") == "completed")
		cost_a = sum(r.get("cost_usd", 0.0) for r in results_a)
		cost_b = sum(r.get("cost_usd", 0.0) for r in results_b)
		winner = agent_id_a if successes_a >= successes_b else agent_id_b
		result = {"agent_a": agent_id_a, "agent_b": agent_id_b, "sample_size": len(task_sample),
				  "successes_a": successes_a, "successes_b": successes_b,
				  "cost_a_usd": cost_a, "cost_b_usd": cost_b,
				  "recommended": winner, "ts": _now()}
		self._ab_results[f"{agent_id_a}_vs_{agent_id_b}"] = result
		return result

	# ── ORCHESTRATION ─────────────────────────────────────────────────────────

	async def orchestration_plan(
		self,
		task: dict[str, Any],
		available_agent_ids: list[str],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Suggest an orchestration plan: which agents to use in which order."""
		task_text = str(task).lower()
		agents = []
		for aid in available_agent_ids:
			key = self._key(tenant_id, aid)
			agent = self._agents.get(key)
			if agent is None:
				continue
			score = 0
			for cap in agent.tool_allowlist:
				if str(cap).lower() in task_text:
					score += 1
			agents.append((score, aid, agent))
		agents.sort(key=lambda x: -x[0])
		steps = [{"step": i + 1, "agent_id": aid, "agent_name": agent.name, "relevance_score": score}
				 for i, (score, aid, agent) in enumerate(agents[:5])]
		return {"task_summary": str(task)[:200], "recommended_steps": steps,
				"strategy": "sequential", "tenant_id": tenant_id, "ts": _now()}

	async def handoff_evaluate(self, session_id: str, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate whether a handoff should occur and to which agent."""
		ctx = self._session_contexts.get(session_id, {})
		ctx.update(context)
		self._session_contexts[session_id] = ctx
		current_agent = ctx.get("current_agent")
		tenant_id = ctx.get("tenant_id", "default")
		# Heuristic: handoff if context signals completion or error
		should_handoff = ctx.get("status") in ("completed", "error", "blocked")
		target = ctx.get("next_agent")
		if should_handoff and target:
			return {"handoff": True, "from_agent": current_agent, "to_agent": target,
					"reason": ctx.get("status"), "session_id": session_id}
		return {"handoff": False, "current_agent": current_agent, "session_id": session_id}

	# ── private helpers ───────────────────────────────────────────────────────

	async def _simulate_agent_step(self, agent: AgentDefinition, task: dict[str, Any]) -> dict[str, Any]:
		"""Simulate executing one agent step (wire to real LLM runtime in production)."""
		await asyncio.sleep(0)  # yield control
		token_estimate = (len(str(task)) // 4) + 200
		model_rates: dict[str, float] = {"gpt-4": 0.03, "llama3": 0.001}
		rate = model_rates.get(agent.model, 0.005)
		cost = (token_estimate / 1000) * rate
		return {"status": "completed", "agent_id": agent.id, "model": agent.model,
				"output": {"result": f"processed_by_{agent.name}"},
				"tokens_used": token_estimate, "cost_usd": round(cost, 6), "ts": _now()}

	def _enforce_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		_raise_if_blocked(result)

	def _enforce_agent_policy(self, tenant_id: str, model_present: bool, system_prompt_present: bool,
							  tool_allowlist_present: bool, io_contract_present: bool,
							  memory_policy_present: bool, runtime_registered: bool,
							  workspace_runtime: bool, sandbox_policy_attached: bool) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id), "operation": "register_agent",
			"model_present": model_present, "system_prompt_present": system_prompt_present,
			"tool_allowlist_present": tool_allowlist_present, "io_contract_present": io_contract_present,
			"memory_policy_present": memory_policy_present, "runtime_registered": runtime_registered,
			"workspace_runtime": workspace_runtime, "sandbox_policy_attached": sandbox_policy_attached,
		})
		_raise_if_blocked(result)

	def _enforce_runtime_policy(self, tenant_id: str, registered: bool, workspace_runtime: bool,
								sandbox_policy_attached: bool, external_runtime: bool, approval_recorded: bool,
								cost_limit_present: bool, operation: str = "register_runtime",
								runtime_requester_present: bool = True) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id), "operation": operation,
			"runtime_registered": registered, "workspace_runtime": workspace_runtime,
			"sandbox_policy_attached": sandbox_policy_attached, "external_runtime": external_runtime,
			"approval_recorded": approval_recorded, "cost_limit_present": cost_limit_present,
			"runtime_requester_present": runtime_requester_present,
		})
		_raise_if_blocked(result)

	def _enforce_approval_decision_policy(self, tenant_id: str, reviewer: str, notes: str) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id), "operation": "decide_runtime_approval",
			"reviewer_present": bool(reviewer), "decision_notes_present": bool(notes),
		})
		_raise_if_blocked(result)

	def _enforce_execution_run_policy(self, tenant_id: str, requester_present: bool,
									  trace_sink_present: bool, side_effects_requested: bool,
									  human_approval_recorded: bool) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id), "operation": "record_execution_run",
			"requester_present": requester_present, "trace_sink_present": trace_sink_present,
			"side_effects_requested": side_effects_requested,
			"human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		return result

	def _enforce_team_policy(self, tenant_id: str, agent_count: int,
							 agent_ids: list[str] | tuple[str, ...],
							 handoffs: tuple[HandoffEdge, ...]) -> None:
		tenant_agents = self._tenant_agent_map(tenant_id)
		unknown_agents = [aid for aid in agent_ids if aid not in tenant_agents]
		endpoints = set(agent_ids)
		unresolved = any(e.source not in endpoints or e.target not in endpoints for e in handoffs)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id), "operation": "register_team",
			"agent_count": agent_count, "runtime_registered": True,
			"handoff_endpoint_resolved": not unknown_agents and not unresolved,
		})
		_raise_if_blocked(result)
		if unknown_agents:
			raise KeyError(f"unknown team agent(s): {', '.join(unknown_agents)}")

	def _key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"

	def _get_runtime(self, tenant_id: str, runtime_name: str) -> AgentRuntime | None:
		return (self._runtimes.get(self._key(tenant_id, runtime_name))
				or self._runtimes.get(self._key("default", runtime_name)))

	def _get_team(self, team_id: str, tenant_id: str | None = None) -> AgentTeam | None:
		if tenant_id is not None:
			return self._teams.get(self._key(tenant_id, team_id))
		matches = [t for t in self._teams.values() if t.id == team_id]
		return matches[0] if len(matches) == 1 else None

	def _tenant_agent_map(self, tenant_id: str) -> dict[str, AgentDefinition]:
		return {a.id: a for a in self._agents.values() if a.tenant_id == tenant_id}

	def _tenant_runtime_map(self, tenant_id: str) -> dict[str, AgentRuntime]:
		runtime_map = {r.name: r for r in self._runtimes.values() if r.tenant_id == "default"}
		runtime_map.update({r.name: r for r in self._runtimes.values() if r.tenant_id == tenant_id})
		return runtime_map

	def _record_event(self, tenant_id: str, event_type: str, subject_id: str, message: str,
					  evidence: dict[str, Any] | None = None,
					  policy_result: dict[str, Any] | None = None) -> None:
		policy_result = policy_result or {"decision": "allow", "matched_rules": [], "actions": []}
		self._events.append(AgentAuditEvent(
			id=f"agnt-event-{len(self._events) + 1}", tenant_id=tenant_id, event_type=event_type,
			subject_id=subject_id, message=message, evidence=dict(evidence or {}),
			policy_decision=policy_result["decision"],
			matched_rules=list(policy_result["matched_rules"]),
			review_reasons=self._review_reasons(policy_result),
			audit_evidence=self._audit_evidence(policy_result),
		))

	def _review_reasons(self, result: dict[str, Any]) -> list[str]:
		if result["decision"] != "require_review":
			return []
		return [a.get("reason", "agent_composition_review_required") for a in result.get("actions", [])]

	def _audit_evidence(self, result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
		return {
			"required_actions": [a["required_action"] for a in result.get("actions", []) if a.get("required_action")],
			"reasons": [a.get("reason", "agent_composition_policy_blocked") for a in result.get("actions", [])],
			"review_recorded": bool(review_recorded),
		}


def _default_runtimes() -> tuple[AgentRuntime, ...]:
	return (
		AgentRuntime("local", kind="local", capabilities=("offline", "deterministic")),
		AgentRuntime("codex", kind="external", approved=True, external_runtime=True,
					 workspace_runtime=True, capabilities=("code", "tests", "docs"), cost_limit=25.0),
		AgentRuntime("claude_code", kind="external", approved=True, external_runtime=True,
					 workspace_runtime=True, capabilities=("code", "analysis"), cost_limit=25.0),
		AgentRuntime("opencode", kind="external", approved=True, external_runtime=True,
					 workspace_runtime=True, capabilities=("code", "shell"), cost_limit=15.0),
		AgentRuntime("pi", kind="external", approved=True, external_runtime=True,
					 capabilities=("conversation", "assistant"), cost_limit=10.0),
	)


def _raise_if_blocked(result: dict[str, Any]) -> None:
	if result["decision"] == "allow":
		return
	reasons = ", ".join(a.get("reason", "agent_composition_policy_blocked") for a in result["actions"])
	if result["decision"] == "require_review":
		raise PermissionError(reasons or "agent_composition_review_required")
	raise PermissionError(reasons or "agent_composition_policy_blocked")

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_tools', '_session_contexts', '_ab_results']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

