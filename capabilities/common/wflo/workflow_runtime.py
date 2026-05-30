"""Dependency-light workflow orchestration runtime primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


DEFINITION_STATUSES = {"draft", "review_required", "published", "retired"}
EXECUTION_STATUSES = {"running", "waiting_approval", "completed", "failed", "cancelled"}
TASK_STATUSES = {"open", "claimed", "completed", "escalated"}
APPROVAL_STATUSES = {"pending", "approved", "rejected", "delegated"}
STEP_TYPES = {"human", "automation", "approval", "ai", "event"}
AGENT_STATUSES = {"active", "suspended"}


def utc_now() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part).strip().lower() for part in parts if str(part).strip())
	digest = sha256(seed.encode("utf-8")).hexdigest()[:16]
	return f"{prefix}_{digest}"


def normalize_step_type(step_type: str) -> str:
	value = str(step_type or "human").strip().lower()
	if value not in STEP_TYPES:
		raise ValueError(f"unsupported_workflow_step_type:{step_type}")
	return value


def workflow_required_actions(rule_result: dict[str, Any]) -> list[str]:
	return [
		str(action["required_action"])
		for action in rule_result.get("actions", [])
		if action.get("required_action")
	]


def serialize(record: object) -> dict[str, Any]:
	return asdict(record)


@dataclass(slots=True)
class WorkflowStepRecord:
	id: str
	name: str
	step_type: str
	assignee_ref: str = ""
	sla_minutes: int = 1440
	requires_approval: bool = False
	ai_policy_ref: str = ""
	automation_policy_ref: str = ""
	event_policy_ref: str = ""
	compensation_ref: str = ""

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class WorkflowDefinitionRecord:
	id: str
	tenant_id: str
	name: str
	owner_ref: str
	version: int
	steps: list[dict[str, Any]]
	trigger_type: str
	trigger_policy_ref: str
	retry_policy_ref: str
	compensation_ref: str
	expected_runtime_minutes: int
	status: str = "draft"
	required_actions: list[str] = field(default_factory=list)
	matched_rules: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now)
	published_at: str | None = None
	published_by: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class WorkflowExecutionRecord:
	id: str
	tenant_id: str
	definition_id: str
	correlation_id: str
	started_by: str
	status: str = "running"
	current_step: str | None = None
	payload: dict[str, Any] = field(default_factory=dict)
	event_stream: str = "bytewax"
	cancel_reason: str = ""
	failure_reason: str = ""
	compensation_status: str = "not_required"
	started_at: str = field(default_factory=utc_now)
	completed_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class WorkflowTaskRecord:
	id: str
	tenant_id: str
	execution_id: str
	step_id: str
	title: str
	assignee_ref: str
	status: str = "open"
	due_at: str | None = None
	claimed_by: str | None = None
	claimed_at: str | None = None
	escalated_at: str | None = None
	escalation_reason: str = ""
	created_at: str = field(default_factory=utc_now)
	completed_at: str | None = None
	completed_by: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class WorkflowApprovalRecord:
	id: str
	tenant_id: str
	execution_id: str
	subject_ref: str
	approver_ref: str
	reason: str
	status: str = "pending"
	requested_at: str = field(default_factory=utc_now)
	decided_at: str | None = None
	decision_by: str | None = None
	decision_evidence_ref: str = ""
	delegated_to: str = ""

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class WorkflowEventRecord:
	id: str
	tenant_id: str
	execution_id: str
	event_type: str
	payload: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class WorkflowAuditEventRecord:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "low"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class WorkflowAgentRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope_ref: str
	registered_by: str
	contribution_disclosed: bool
	status: str = "active"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


__all__ = [
	"AGENT_STATUSES",
	"APPROVAL_STATUSES",
	"DEFINITION_STATUSES",
	"EXECUTION_STATUSES",
	"STEP_TYPES",
	"TASK_STATUSES",
	"WorkflowApprovalRecord",
	"WorkflowAuditEventRecord",
	"WorkflowAgentRecord",
	"WorkflowDefinitionRecord",
	"WorkflowEventRecord",
	"WorkflowExecutionRecord",
	"WorkflowStepRecord",
	"WorkflowTaskRecord",
	"normalize_step_type",
	"stable_id",
	"utc_now",
	"workflow_required_actions",
]
