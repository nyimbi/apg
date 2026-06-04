"""Domain models for APG AI agent composition."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentRuntime:
	"""Provider-neutral runtime backend for first-class APG agents."""

	name: str
	tenant_id: str = "default"
	kind: str = "local"
	registered: bool = True
	approved: bool = True
	workspace_runtime: bool = False
	external_runtime: bool = False
	sandbox_policy: str | None = "workspace-read"
	capabilities: tuple[str, ...] = ()
	cost_limit: float | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"name": self.name,
			"tenant_id": self.tenant_id,
			"kind": self.kind,
			"registered": self.registered,
			"approved": self.approved,
			"workspace_runtime": self.workspace_runtime,
			"external_runtime": self.external_runtime,
			"sandbox_policy": self.sandbox_policy,
			"capabilities": list(self.capabilities),
			"cost_limit": self.cost_limit,
		}


@dataclass(frozen=True)
class AgentDefinition:
	"""First-class AI agent declaration with model, runtime, tools, memory, and IO contracts."""

	id: str
	tenant_id: str
	name: str
	model: str
	runtime: str
	system_prompt: str
	tool_allowlist: tuple[str, ...] = ()
	input_contract: dict[str, Any] = field(default_factory=dict)
	output_contract: dict[str, Any] = field(default_factory=dict)
	memory_policy: dict[str, Any] = field(default_factory=dict)
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"model": self.model,
			"runtime": self.runtime,
			"system_prompt": self.system_prompt,
			"tool_allowlist": list(self.tool_allowlist),
			"input_contract": dict(self.input_contract),
			"output_contract": dict(self.output_contract),
			"memory_policy": dict(self.memory_policy),
			"status": self.status,
		}


@dataclass(frozen=True)
class HandoffEdge:
	"""Directed handoff from one declared agent to another."""

	source: str
	target: str
	trigger: str = "complete"
	condition: str = "always"

	def to_dict(self) -> dict[str, str]:
		return {
			"source": self.source,
			"target": self.target,
			"trigger": self.trigger,
			"condition": self.condition,
		}


@dataclass(frozen=True)
class AgentTeam:
	"""Composable team of first-class APG agents plus validated handoff edges."""

	id: str
	tenant_id: str
	name: str
	agent_ids: tuple[str, ...]
	handoffs: tuple[HandoffEdge, ...] = ()
	execution_mode: str = "sequential"
	cycle_review_required: bool = True
	parallel_execution_enabled: bool = False

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"agent_ids": list(self.agent_ids),
			"handoffs": [handoff.to_dict() for handoff in self.handoffs],
			"execution_mode": self.execution_mode,
			"cycle_review_required": self.cycle_review_required,
			"parallel_execution_enabled": self.parallel_execution_enabled,
		}


@dataclass(frozen=True)
class ExecutionPlan:
	"""Deterministic, reviewable execution plan for one agent team."""

	id: str
	tenant_id: str
	team_id: str
	steps: tuple[dict[str, Any], ...]
	runtime_assignments: dict[str, str]
	approvals_required: tuple[dict[str, Any], ...] = ()
	estimated_cost_limit: float | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"team_id": self.team_id,
			"steps": [dict(step) for step in self.steps],
			"runtime_assignments": dict(self.runtime_assignments),
			"approvals_required": [dict(item) for item in self.approvals_required],
			"estimated_cost_limit": self.estimated_cost_limit,
		}


@dataclass(frozen=True)
class AgentExecutionRun:
	"""Provider-neutral execution run record for a planned agent team."""

	id: str
	tenant_id: str
	team_id: str
	plan_id: str
	objective: str
	requested_by: str
	trace_sink: str
	status: str = "planned"
	side_effects_requested: bool = False
	human_approval_recorded: bool = False
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	plan_snapshot: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"team_id": self.team_id,
			"plan_id": self.plan_id,
			"objective": self.objective,
			"requested_by": self.requested_by,
			"trace_sink": self.trace_sink,
			"status": self.status,
			"side_effects_requested": self.side_effects_requested,
			"human_approval_recorded": self.human_approval_recorded,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"plan_snapshot": dict(self.plan_snapshot),
		}


@dataclass(frozen=True)
class RuntimeApprovalRequest:
	"""Governed request to enable an external agent runtime."""

	id: str
	tenant_id: str
	runtime_name: str
	kind: str
	requested_by: str
	workspace_runtime: bool = False
	sandbox_policy: str | None = None
	capabilities: tuple[str, ...] = ()
	cost_limit: float | None = None
	decision: str = "pending"
	policy_decision: str = "require_review"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	reviewer: str | None = None
	notes: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"runtime_name": self.runtime_name,
			"kind": self.kind,
			"requested_by": self.requested_by,
			"workspace_runtime": self.workspace_runtime,
			"sandbox_policy": self.sandbox_policy,
			"capabilities": list(self.capabilities),
			"cost_limit": self.cost_limit,
			"decision": self.decision,
			"policy_decision": self.policy_decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"reviewer": self.reviewer,
			"notes": self.notes,
		}


@dataclass(frozen=True)
class AgentAuditEvent:
	"""Tenant-scoped evidence event for AGNT lifecycle changes."""

	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	evidence: dict[str, Any] = field(default_factory=dict)
	policy_decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	audit_evidence: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"message": self.message,
			"evidence": dict(self.evidence),
			"policy_decision": self.policy_decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
		}


# ---------------------------------------------------------------------------
# New models for expanded AGNT capability (registry, execution, tools, memory,
# handoffs, performance, safety)
# ---------------------------------------------------------------------------


@dataclass
class ToolDefinition:
	"""Registered tool that agents may invoke."""

	tool_name: str
	description: str
	schema: dict[str, Any]
	handler_endpoint: str
	tenant_id: str = "default"
	created_at: str = field(default_factory=lambda: _now())
	invocation_count: int = 0

	def to_dict(self) -> dict[str, Any]:
		return {
			"tool_name": self.tool_name,
			"description": self.description,
			"schema": dict(self.schema),
			"handler_endpoint": self.handler_endpoint,
			"tenant_id": self.tenant_id,
			"created_at": self.created_at,
			"invocation_count": self.invocation_count,
		}


@dataclass
class ToolAssignment:
	"""Access grant giving an agent permission to use a tool."""

	agent_id: str
	tool_name: str
	access_level: str = "read"  # read | write | admin
	tenant_id: str = "default"
	granted_at: str = field(default_factory=lambda: _now())
	revoked: bool = False
	revoked_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"agent_id": self.agent_id,
			"tool_name": self.tool_name,
			"access_level": self.access_level,
			"tenant_id": self.tenant_id,
			"granted_at": self.granted_at,
			"revoked": self.revoked,
			"revoked_at": self.revoked_at,
		}


@dataclass
class ToolInvocationRecord:
	"""Single tool invocation recorded for analytics."""

	id: str
	tool_name: str
	agent_id: str
	tenant_id: str
	parameters: dict[str, Any]
	outcome: str = "success"  # success | error | blocked
	latency_ms: float = 0.0
	invoked_at: str = field(default_factory=lambda: _now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tool_name": self.tool_name,
			"agent_id": self.agent_id,
			"tenant_id": self.tenant_id,
			"parameters": dict(self.parameters),
			"outcome": self.outcome,
			"latency_ms": self.latency_ms,
			"invoked_at": self.invoked_at,
		}


@dataclass
class MemoryRecord:
	"""Single memory entry stored for an agent session."""

	id: str
	agent_id: str
	session_id: str
	tenant_id: str
	memory_type: str  # episodic | semantic | working
	content: str
	embedding_hint: str = ""  # pseudo-embedding key for search
	created_at: str = field(default_factory=lambda: _now())
	compressed: bool = False

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"agent_id": self.agent_id,
			"session_id": self.session_id,
			"tenant_id": self.tenant_id,
			"memory_type": self.memory_type,
			"content": self.content,
			"embedding_hint": self.embedding_hint,
			"created_at": self.created_at,
			"compressed": self.compressed,
		}


@dataclass
class ExecutionSession:
	"""Live or completed single-agent execution session."""

	session_id: str
	agent_id: str
	tenant_id: str
	task: str
	context: dict[str, Any]
	status: str = "pending"  # pending | running | paused | completed | cancelled | failed
	output_so_far: str = ""
	progress_pct: float = 0.0
	started_at: str = field(default_factory=lambda: _now())
	finished_at: str | None = None
	cancel_reason: str | None = None
	latency_ms: float = 0.0
	token_usage: int = 0
	cost_usd: float = 0.0
	streaming: bool = False

	def to_dict(self) -> dict[str, Any]:
		return {
			"session_id": self.session_id,
			"agent_id": self.agent_id,
			"tenant_id": self.tenant_id,
			"task": self.task,
			"context": dict(self.context),
			"status": self.status,
			"output_so_far": self.output_so_far,
			"progress_pct": self.progress_pct,
			"started_at": self.started_at,
			"finished_at": self.finished_at,
			"cancel_reason": self.cancel_reason,
			"latency_ms": self.latency_ms,
			"token_usage": self.token_usage,
			"cost_usd": self.cost_usd,
			"streaming": self.streaming,
		}


@dataclass
class HandoffRule:
	"""Condition-based rule that routes context from one agent to another."""

	rule_id: str
	from_agent: str
	to_agent: str
	condition: str
	priority: int = 0
	tenant_id: str = "default"
	created_at: str = field(default_factory=lambda: _now())
	active: bool = True

	def to_dict(self) -> dict[str, Any]:
		return {
			"rule_id": self.rule_id,
			"from_agent": self.from_agent,
			"to_agent": self.to_agent,
			"condition": self.condition,
			"priority": self.priority,
			"tenant_id": self.tenant_id,
			"created_at": self.created_at,
			"active": self.active,
		}


@dataclass
class HandoffEvent:
	"""Recorded transfer of context between agents."""

	id: str
	session_id: str
	from_agent: str
	to_agent: str
	context_snapshot: dict[str, Any]
	rule_id: str | None
	tenant_id: str = "default"
	occurred_at: str = field(default_factory=lambda: _now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"session_id": self.session_id,
			"from_agent": self.from_agent,
			"to_agent": self.to_agent,
			"context_snapshot": dict(self.context_snapshot),
			"rule_id": self.rule_id,
			"tenant_id": self.tenant_id,
			"occurred_at": self.occurred_at,
		}


@dataclass
class PerformanceRecord:
	"""Single execution outcome used to compute agent performance statistics."""

	id: str
	agent_id: str
	tenant_id: str
	success: bool
	latency_ms: float
	token_usage: int
	cost_usd: float
	recorded_at: str = field(default_factory=lambda: _now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"agent_id": self.agent_id,
			"tenant_id": self.tenant_id,
			"success": self.success,
			"latency_ms": self.latency_ms,
			"token_usage": self.token_usage,
			"cost_usd": self.cost_usd,
			"recorded_at": self.recorded_at,
		}


@dataclass
class GuardrailViolation:
	"""Recorded content safety or policy violation."""

	id: str
	agent_id: str
	tenant_id: str
	violation_type: str  # pii | content_safety | policy | anomaly
	input_snippet: str
	severity: str = "medium"  # low | medium | high | critical
	action_taken: str = "blocked"  # blocked | redacted | escalated | logged
	recorded_at: str = field(default_factory=lambda: _now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"agent_id": self.agent_id,
			"tenant_id": self.tenant_id,
			"violation_type": self.violation_type,
			"input_snippet": self.input_snippet,
			"severity": self.severity,
			"action_taken": self.action_taken,
			"recorded_at": self.recorded_at,
		}


def _now() -> str:
	return datetime.datetime.now(datetime.timezone.utc).isoformat()
