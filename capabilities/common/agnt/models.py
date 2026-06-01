"""Domain models for APG AI agent composition."""

from __future__ import annotations

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
