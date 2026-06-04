"""Pydantic v2 domain models for the Workflow Orchestration capability.

Entities: WorkflowDefinition, WorkflowInstance, Task, UserTask, ServiceTask,
Timer, Gateway, BoundaryEvent, Escalation, Compensation, WorkflowVariable,
WorkflowHistory.

All models use:
- uuid7str for IDs
- tenant_id isolation on every record
- Pydantic v2 ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
- Modern Python 3.12+ typing
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from uuid6 import uuid7


def uuid7str() -> str:
	"""Generate a UUID7 string (time-ordered)."""
	return str(uuid7())


def utc_now() -> datetime:
	return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class DefinitionStatus(str, Enum):
	DRAFT = "draft"
	REVIEW_REQUIRED = "review_required"
	PUBLISHED = "published"
	DEPRECATED = "deprecated"
	RETIRED = "retired"


class InstanceStatus(str, Enum):
	PENDING = "pending"
	RUNNING = "running"
	SUSPENDED = "suspended"
	WAITING_TIMER = "waiting_timer"
	WAITING_APPROVAL = "waiting_approval"
	WAITING_SIGNAL = "waiting_signal"
	COMPENSATING = "compensating"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	MIGRATED = "migrated"


class TaskStatus(str, Enum):
	CREATED = "created"
	READY = "ready"
	CLAIMED = "claimed"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	ESCALATED = "escalated"
	CANCELLED = "cancelled"
	TIMED_OUT = "timed_out"


class TaskType(str, Enum):
	USER = "user"
	SERVICE = "service"
	SCRIPT = "script"
	MANUAL = "manual"
	RECEIVE = "receive"
	SEND = "send"
	BUSINESS_RULE = "business_rule"
	CALL_ACTIVITY = "call_activity"


class GatewayType(str, Enum):
	EXCLUSIVE = "exclusive"    # XOR — exactly one outgoing path
	PARALLEL = "parallel"      # AND — all outgoing paths
	INCLUSIVE = "inclusive"    # OR — one or more paths
	EVENT_BASED = "event_based"
	COMPLEX = "complex"


class TimerType(str, Enum):
	DATE = "date"         # fire at absolute datetime
	DURATION = "duration" # fire after ISO 8601 duration
	CYCLE = "cycle"       # repeating ISO 8601 cycle


class BoundaryEventType(str, Enum):
	TIMER = "timer"
	ERROR = "error"
	ESCALATION = "escalation"
	COMPENSATION = "compensation"
	SIGNAL = "signal"
	MESSAGE = "message"
	CONDITIONAL = "conditional"


class CompensationStatus(str, Enum):
	NOT_REQUIRED = "not_required"
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"


class EscalationStatus(str, Enum):
	ACTIVE = "active"
	RESOLVED = "resolved"
	EXPIRED = "expired"


class VariableScope(str, Enum):
	GLOBAL = "global"     # visible to all nodes in the instance
	LOCAL = "local"       # visible only to creating node
	PROCESS = "process"   # visible to the process and sub-processes


class TriggerType(str, Enum):
	MANUAL = "manual"
	SCHEDULED = "scheduled"
	API = "api"
	EVENT = "event"
	WEBHOOK = "webhook"
	MESSAGE = "message"


# ─────────────────────────────────────────────────────────────────────────────
# Base model
# ─────────────────────────────────────────────────────────────────────────────

class WfloBase(BaseModel):
	"""Common audit fields shared by every wflo entity."""
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
		use_enum_values=True,
	)

	id: str = Field(default_factory=uuid7str, description="UUID7 primary key")
	tenant_id: str = Field(..., min_length=1, description="Tenant isolation key")
	created_at: datetime = Field(default_factory=utc_now)
	updated_at: datetime = Field(default_factory=utc_now)
	created_by: str = Field(default="system", description="Actor who created this record")
	is_deleted: bool = Field(default=False)

	@field_validator("tenant_id")
	@classmethod
	def tenant_id_not_empty(cls, v: str) -> str:
		if not v.strip():
			raise ValueError("tenant_id must not be blank")
		return v.strip()


# ─────────────────────────────────────────────────────────────────────────────
# WorkflowDefinition
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowDefinition(WfloBase):
	"""Immutable blueprint describing a business process.

	A definition captures the full BPMN-compatible process graph, trigger
	conditions, SLA expectations, retry/compensation policies, and version
	lineage.  Only PUBLISHED definitions may spawn instances.
	"""
	name: str = Field(..., min_length=1, max_length=255)
	description: str = Field(default="")
	version: int = Field(default=1, ge=1)
	status: DefinitionStatus = Field(default=DefinitionStatus.DRAFT)
	trigger_type: TriggerType = Field(default=TriggerType.MANUAL)
	trigger_config: dict[str, Any] = Field(default_factory=dict)
	owner_ref: str = Field(..., min_length=1, description="User/role that owns this definition")
	category: str = Field(default="general")
	tags: list[str] = Field(default_factory=list)
	# Process graph stored as serialised JSON (nodes + edges)
	process_graph: dict[str, Any] = Field(default_factory=dict)
	# Flat ordered step list for simple linear workflows
	steps: list[dict[str, Any]] = Field(default_factory=list)
	# Policy refs for cross-cutting concerns
	retry_policy_ref: str = Field(default="")
	compensation_ref: str = Field(default="")
	sla_minutes: int = Field(default=1440, ge=1, description="Expected total runtime in minutes")
	max_runtime_minutes: int = Field(default=1440, ge=1)
	review_required: bool = Field(default=False)
	publish_approval_ref: str = Field(default="")
	published_at: datetime | None = Field(default=None)
	published_by: str | None = Field(default=None)
	parent_definition_id: str | None = Field(default=None, description="Previous version for lineage")
	# Metadata
	metadata: dict[str, Any] = Field(default_factory=dict)

	@field_validator("name")
	@classmethod
	def name_stripped(cls, v: str) -> str:
		return v.strip()


class WorkflowDefinitionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name: str = Field(..., min_length=1, max_length=255)
	tenant_id: str = Field(..., min_length=1)
	description: str = Field(default="")
	owner_ref: str = Field(..., min_length=1)
	trigger_type: TriggerType = Field(default=TriggerType.MANUAL)
	trigger_config: dict[str, Any] = Field(default_factory=dict)
	category: str = Field(default="general")
	tags: list[str] = Field(default_factory=list)
	process_graph: dict[str, Any] = Field(default_factory=dict)
	steps: list[dict[str, Any]] = Field(default_factory=list)
	retry_policy_ref: str = Field(default="")
	compensation_ref: str = Field(default="")
	sla_minutes: int = Field(default=1440, ge=1)
	max_runtime_minutes: int = Field(default=1440, ge=1)
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_by: str = Field(default="system")


class WorkflowDefinitionUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name: str | None = None
	description: str | None = None
	trigger_type: TriggerType | None = None
	trigger_config: dict[str, Any] | None = None
	category: str | None = None
	tags: list[str] | None = None
	process_graph: dict[str, Any] | None = None
	steps: list[dict[str, Any]] | None = None
	retry_policy_ref: str | None = None
	compensation_ref: str | None = None
	sla_minutes: int | None = None
	max_runtime_minutes: int | None = None
	metadata: dict[str, Any] | None = None


class WorkflowDefinitionResponse(WfloBase):
	"""Full definition including computed fields."""
	name: str
	description: str
	version: int
	status: DefinitionStatus
	trigger_type: TriggerType
	trigger_config: dict[str, Any]
	owner_ref: str
	category: str
	tags: list[str]
	process_graph: dict[str, Any]
	steps: list[dict[str, Any]]
	retry_policy_ref: str
	compensation_ref: str
	sla_minutes: int
	max_runtime_minutes: int
	review_required: bool
	publish_approval_ref: str
	published_at: datetime | None
	published_by: str | None
	parent_definition_id: str | None
	metadata: dict[str, Any]
	# Computed
	instance_count: int = Field(default=0)
	active_instance_count: int = Field(default=0)


# ─────────────────────────────────────────────────────────────────────────────
# WorkflowInstance
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowInstance(WfloBase):
	"""A running (or completed) execution of a WorkflowDefinition.

	Tracks current position in the process graph, variable bindings,
	SLA breaches, compensation state, and full lifecycle timestamps.
	"""
	definition_id: str = Field(..., description="FK → WorkflowDefinition.id")
	definition_version: int = Field(default=1)
	correlation_id: str = Field(default="", description="External correlation key")
	status: InstanceStatus = Field(default=InstanceStatus.PENDING)
	current_node_id: str | None = Field(default=None, description="Active BPMN node")
	# Variables snapshot at start
	input_variables: dict[str, Any] = Field(default_factory=dict)
	# Mutable runtime variables
	runtime_variables: dict[str, Any] = Field(default_factory=dict)
	# Compensation
	compensation_status: CompensationStatus = Field(default=CompensationStatus.NOT_REQUIRED)
	compensation_log: list[dict[str, Any]] = Field(default_factory=list)
	# Lifecycle timestamps
	started_at: datetime | None = Field(default=None)
	suspended_at: datetime | None = Field(default=None)
	resumed_at: datetime | None = Field(default=None)
	completed_at: datetime | None = Field(default=None)
	failed_at: datetime | None = Field(default=None)
	cancelled_at: datetime | None = Field(default=None)
	# SLA
	due_at: datetime | None = Field(default=None, description="Derived from definition.sla_minutes")
	sla_breached: bool = Field(default=False)
	# Error / cancel context
	error_code: str = Field(default="")
	error_message: str = Field(default="")
	cancel_reason: str = Field(default="")
	# Migration
	migrated_from_version: int | None = Field(default=None)
	# Parent instance for sub-processes
	parent_instance_id: str | None = Field(default=None)
	metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowInstanceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	definition_id: str = Field(..., min_length=1)
	tenant_id: str = Field(..., min_length=1)
	correlation_id: str = Field(default="")
	input_variables: dict[str, Any] = Field(default_factory=dict)
	created_by: str = Field(default="system")
	metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowInstanceResponse(WfloBase):
	definition_id: str
	definition_version: int
	correlation_id: str
	status: InstanceStatus
	current_node_id: str | None
	input_variables: dict[str, Any]
	runtime_variables: dict[str, Any]
	compensation_status: CompensationStatus
	started_at: datetime | None
	completed_at: datetime | None
	failed_at: datetime | None
	cancelled_at: datetime | None
	due_at: datetime | None
	sla_breached: bool
	error_code: str
	error_message: str
	cancel_reason: str
	migrated_from_version: int | None
	parent_instance_id: str | None
	metadata: dict[str, Any]
	# Computed
	open_task_count: int = Field(default=0)
	pending_approval_count: int = Field(default=0)
	elapsed_minutes: float = Field(default=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Task (base + specialisations)
# ─────────────────────────────────────────────────────────────────────────────

class Task(WfloBase):
	"""Generic workflow task node."""
	instance_id: str = Field(..., description="FK → WorkflowInstance.id")
	definition_id: str = Field(..., description="Denormalised for fast queries")
	node_id: str = Field(..., description="BPMN node reference in process graph")
	task_type: TaskType = Field(default=TaskType.USER)
	status: TaskStatus = Field(default=TaskStatus.CREATED)
	name: str = Field(..., min_length=1, max_length=255)
	description: str = Field(default="")
	# Assignment
	assignee_ref: str = Field(default="", description="User ID or role")
	candidate_refs: list[str] = Field(default_factory=list, description="Candidate users/groups")
	# Timing
	created_at: datetime = Field(default_factory=utc_now)
	ready_at: datetime | None = Field(default=None)
	claimed_at: datetime | None = Field(default=None)
	started_at: datetime | None = Field(default=None)
	due_at: datetime | None = Field(default=None)
	completed_at: datetime | None = Field(default=None)
	# Outcomes
	outcome: str = Field(default="", description="Completion outcome / decision")
	output_variables: dict[str, Any] = Field(default_factory=dict)
	# Claim
	claimed_by: str | None = Field(default=None)
	completed_by: str | None = Field(default=None)
	# Escalation
	escalated: bool = Field(default=False)
	escalation_reason: str = Field(default="")
	escalated_at: datetime | None = Field(default=None)
	escalated_to: str = Field(default="")
	# Priority
	priority: int = Field(default=50, ge=0, le=100, description="0=lowest, 100=highest")
	metadata: dict[str, Any] = Field(default_factory=dict)


class UserTask(WfloBase):
	"""Human-facing task with form schema and assignment rules."""
	task_id: str = Field(..., description="FK → Task.id")
	instance_id: str
	node_id: str
	form_schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for task form")
	form_data: dict[str, Any] = Field(default_factory=dict)
	# Assignment strategy
	assignment_strategy: str = Field(default="direct", description="direct | role | group | load_balance")
	assignee_ref: str = Field(default="")
	candidate_groups: list[str] = Field(default_factory=list)
	# SLA
	sla_minutes: int = Field(default=480, ge=1)
	reminder_minutes: int = Field(default=60, ge=1)
	# Completion
	status: TaskStatus = Field(default=TaskStatus.CREATED)
	completed_by: str | None = Field(default=None)
	completed_at: datetime | None = Field(default=None)
	outcome: str = Field(default="")
	metadata: dict[str, Any] = Field(default_factory=dict)


class ServiceTask(WfloBase):
	"""Automated service invocation task."""
	task_id: str = Field(..., description="FK → Task.id")
	instance_id: str
	node_id: str
	# Service endpoint config
	service_ref: str = Field(..., description="Service identifier or URL")
	operation: str = Field(default="")
	input_mapping: dict[str, Any] = Field(default_factory=dict, description="Variable → request mapping")
	output_mapping: dict[str, Any] = Field(default_factory=dict, description="Response → variable mapping")
	# Retry
	retry_count: int = Field(default=0, ge=0)
	max_retries: int = Field(default=3, ge=0)
	retry_backoff_seconds: int = Field(default=5, ge=1)
	# Timeout
	timeout_seconds: int = Field(default=30, ge=1)
	# Results
	status: TaskStatus = Field(default=TaskStatus.CREATED)
	last_error: str = Field(default="")
	last_response: dict[str, Any] = Field(default_factory=dict)
	started_at: datetime | None = Field(default=None)
	completed_at: datetime | None = Field(default=None)
	metadata: dict[str, Any] = Field(default_factory=dict)


class TaskCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	instance_id: str = Field(..., min_length=1)
	definition_id: str = Field(..., min_length=1)
	node_id: str = Field(..., min_length=1)
	tenant_id: str = Field(..., min_length=1)
	task_type: TaskType = Field(default=TaskType.USER)
	name: str = Field(..., min_length=1)
	description: str = Field(default="")
	assignee_ref: str = Field(default="")
	candidate_refs: list[str] = Field(default_factory=list)
	due_at: datetime | None = Field(default=None)
	priority: int = Field(default=50, ge=0, le=100)
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_by: str = Field(default="system")


class TaskUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	assignee_ref: str | None = None
	candidate_refs: list[str] | None = None
	due_at: datetime | None = None
	priority: int | None = None
	description: str | None = None
	metadata: dict[str, Any] | None = None


class TaskResponse(WfloBase):
	instance_id: str
	definition_id: str
	node_id: str
	task_type: TaskType
	status: TaskStatus
	name: str
	description: str
	assignee_ref: str
	candidate_refs: list[str]
	ready_at: datetime | None
	claimed_at: datetime | None
	due_at: datetime | None
	completed_at: datetime | None
	outcome: str
	output_variables: dict[str, Any]
	claimed_by: str | None
	completed_by: str | None
	escalated: bool
	escalation_reason: str
	escalated_at: datetime | None
	escalated_to: str
	priority: int
	metadata: dict[str, Any]
	# Computed
	overdue: bool = Field(default=False)
	minutes_until_due: float | None = Field(default=None)


# ─────────────────────────────────────────────────────────────────────────────
# Timer
# ─────────────────────────────────────────────────────────────────────────────

class Timer(WfloBase):
	"""Workflow timer — fires at a date, after a duration, or on a cycle."""
	instance_id: str = Field(..., description="FK → WorkflowInstance.id")
	node_id: str
	timer_type: TimerType = Field(default=TimerType.DURATION)
	# For DATE timers: absolute ISO 8601 datetime
	fire_at: datetime | None = Field(default=None)
	# For DURATION timers: ISO 8601 duration string e.g. "PT1H30M"
	duration_iso: str = Field(default="")
	# For CYCLE timers: ISO 8601 repeating interval e.g. "R3/PT1H"
	cycle_expression: str = Field(default="")
	fired: bool = Field(default=False)
	fired_at: datetime | None = Field(default=None)
	cancelled: bool = Field(default=False)
	cancelled_at: datetime | None = Field(default=None)
	fire_count: int = Field(default=0, ge=0)
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Gateway
# ─────────────────────────────────────────────────────────────────────────────

class Gateway(WfloBase):
	"""Process flow gateway — controls branching and merging."""
	instance_id: str
	node_id: str
	gateway_type: GatewayType = Field(default=GatewayType.EXCLUSIVE)
	# Condition expressions for each outgoing sequence flow
	# key = target_node_id, value = expression (e.g. "${amount > 1000}")
	conditions: dict[str, str] = Field(default_factory=dict)
	# For parallel/inclusive gateways: track which branches joined
	incoming_branches: list[str] = Field(default_factory=list)
	completed_branches: list[str] = Field(default_factory=list)
	# Evaluated path(s)
	selected_paths: list[str] = Field(default_factory=list)
	evaluated_at: datetime | None = Field(default=None)
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# BoundaryEvent
# ─────────────────────────────────────────────────────────────────────────────

class BoundaryEvent(WfloBase):
	"""Event attached to a task boundary (timer, error, escalation, etc.)."""
	instance_id: str
	attached_to_task_id: str = Field(..., description="Task this event is attached to")
	node_id: str
	event_type: BoundaryEventType = Field(default=BoundaryEventType.TIMER)
	interrupting: bool = Field(default=True, description="True=cancel host task; False=non-interrupting")
	# Trigger config varies by event_type
	trigger_config: dict[str, Any] = Field(default_factory=dict)
	triggered: bool = Field(default=False)
	triggered_at: datetime | None = Field(default=None)
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Escalation
# ─────────────────────────────────────────────────────────────────────────────

class Escalation(WfloBase):
	"""Tracks a task or approval escalation to a higher authority."""
	instance_id: str
	task_id: str | None = Field(default=None)
	escalated_from: str = Field(..., description="Original assignee or role")
	escalated_to: str = Field(..., description="Escalation target user/role")
	reason: str = Field(..., min_length=1)
	status: EscalationStatus = Field(default=EscalationStatus.ACTIVE)
	level: int = Field(default=1, ge=1, description="Escalation level (1=first, 2=second, ...)")
	escalated_at: datetime = Field(default_factory=utc_now)
	resolved_at: datetime | None = Field(default=None)
	resolved_by: str | None = Field(default=None)
	resolution_note: str = Field(default="")
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Compensation
# ─────────────────────────────────────────────────────────────────────────────

class Compensation(WfloBase):
	"""Compensation activity for undoing completed work on failure."""
	instance_id: str
	# Node in the process graph that performs compensation
	compensation_node_id: str
	# Task that is being compensated
	compensates_task_id: str
	status: CompensationStatus = Field(default=CompensationStatus.PENDING)
	triggered_at: datetime = Field(default_factory=utc_now)
	completed_at: datetime | None = Field(default=None)
	failed_at: datetime | None = Field(default=None)
	error_message: str = Field(default="")
	compensation_data: dict[str, Any] = Field(default_factory=dict)
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# WorkflowVariable
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowVariable(WfloBase):
	"""A named, typed variable scoped to a workflow instance (or globally)."""
	instance_id: str
	scope: VariableScope = Field(default=VariableScope.GLOBAL)
	# node_id is set for LOCAL scope
	node_id: str | None = Field(default=None)
	name: str = Field(..., min_length=1, max_length=255)
	value_type: str = Field(default="string", description="string | number | boolean | object | array")
	# Stored as JSON-compatible Any
	value: Any = Field(default=None)
	# Version for optimistic locking
	version: int = Field(default=1, ge=1)
	# Who last mutated this variable
	mutated_by: str = Field(default="system")
	mutated_at: datetime = Field(default_factory=utc_now)
	metadata: dict[str, Any] = Field(default_factory=dict)

	@model_validator(mode="after")
	def local_scope_requires_node(self) -> "WorkflowVariable":
		if self.scope == VariableScope.LOCAL and not self.node_id:
			raise ValueError("LOCAL scope variables require node_id")
		return self


class WorkflowVariableCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	instance_id: str = Field(..., min_length=1)
	tenant_id: str = Field(..., min_length=1)
	name: str = Field(..., min_length=1, max_length=255)
	scope: VariableScope = Field(default=VariableScope.GLOBAL)
	node_id: str | None = None
	value_type: str = Field(default="string")
	value: Any = None
	created_by: str = Field(default="system")


# ─────────────────────────────────────────────────────────────────────────────
# WorkflowHistory
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowHistory(WfloBase):
	"""Immutable audit log of every state transition in a workflow instance.

	Written once, never updated. Provides the full timeline for forensics,
	SLA analysis, and compliance reporting.
	"""
	instance_id: str
	definition_id: str
	event_type: str = Field(..., description="e.g. instance.started, task.completed, gateway.evaluated")
	node_id: str | None = Field(default=None)
	task_id: str | None = Field(default=None)
	actor_id: str = Field(default="system")
	from_status: str | None = Field(default=None)
	to_status: str | None = Field(default=None)
	# Snapshot of relevant variables at the time of the event
	variable_snapshot: dict[str, Any] = Field(default_factory=dict)
	# Human-readable summary
	summary: str = Field(default="")
	# Machine-readable details for analytics
	details: dict[str, Any] = Field(default_factory=dict)
	sequence_number: int = Field(default=0, ge=0, description="Monotonic per-instance counter")
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Report / Aggregation models
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowAnalytics(BaseModel):
	"""Aggregated metrics for a workflow definition over a time window."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	definition_id: str
	definition_name: str
	period_start: datetime
	period_end: datetime
	# Instance counts
	total_instances: int = 0
	completed_instances: int = 0
	failed_instances: int = 0
	cancelled_instances: int = 0
	active_instances: int = 0
	# Timing
	avg_duration_minutes: float = 0.0
	p50_duration_minutes: float = 0.0
	p95_duration_minutes: float = 0.0
	p99_duration_minutes: float = 0.0
	# SLA
	sla_breach_count: int = 0
	sla_breach_rate: float = 0.0
	# Task metrics
	total_tasks: int = 0
	avg_task_claim_minutes: float = 0.0
	avg_task_completion_minutes: float = 0.0
	escalation_count: int = 0
	escalation_rate: float = 0.0
	# Bottleneck node
	bottleneck_node_id: str | None = None
	bottleneck_avg_minutes: float = 0.0


class SLAReport(BaseModel):
	"""SLA monitoring snapshot for active instances."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	generated_at: datetime = Field(default_factory=utc_now)
	at_risk_count: int = 0        # within 20% of SLA
	breached_count: int = 0       # past due
	healthy_count: int = 0
	total_active: int = 0
	at_risk_instances: list[str] = Field(default_factory=list)
	breached_instances: list[str] = Field(default_factory=list)


class DashboardKPI(BaseModel):
	"""Real-time dashboard key performance indicators."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	generated_at: datetime = Field(default_factory=utc_now)
	# Definitions
	total_definitions: int = 0
	published_definitions: int = 0
	draft_definitions: int = 0
	# Instances
	active_instances: int = 0
	completed_today: int = 0
	failed_today: int = 0
	# Tasks
	open_tasks: int = 0
	overdue_tasks: int = 0
	my_tasks: int = 0
	# Approvals
	pending_approvals: int = 0
	# SLA
	sla_breached_active: int = 0
	# Throughput
	instances_per_hour: float = 0.0


__all__ = [
	# Enums
	"DefinitionStatus",
	"InstanceStatus",
	"TaskStatus",
	"TaskType",
	"GatewayType",
	"TimerType",
	"BoundaryEventType",
	"CompensationStatus",
	"EscalationStatus",
	"VariableScope",
	"TriggerType",
	# Base
	"WfloBase",
	"uuid7str",
	"utc_now",
	# WorkflowDefinition
	"WorkflowDefinition",
	"WorkflowDefinitionCreate",
	"WorkflowDefinitionUpdate",
	"WorkflowDefinitionResponse",
	# WorkflowInstance
	"WorkflowInstance",
	"WorkflowInstanceCreate",
	"WorkflowInstanceResponse",
	# Task
	"Task",
	"UserTask",
	"ServiceTask",
	"TaskCreate",
	"TaskUpdate",
	"TaskResponse",
	# Other entities
	"Timer",
	"Gateway",
	"BoundaryEvent",
	"Escalation",
	"Compensation",
	"WorkflowVariable",
	"WorkflowVariableCreate",
	"WorkflowHistory",
	# Reports
	"WorkflowAnalytics",
	"SLAReport",
	"DashboardKPI",
]
