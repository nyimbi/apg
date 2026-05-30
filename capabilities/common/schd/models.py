"""Domain models for the APG Scheduling and Job Orchestration capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CalendarPolicy:
	"""Tenant calendar, timezone, blackout, and business-day policy."""

	id: str
	tenant_id: str
	name: str
	timezone: str
	owner: str
	business_days: list[str] = field(default_factory=list)
	blackout_windows: list[str] = field(default_factory=list)
	holiday_calendar: str | None = None
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"timezone": self.timezone,
			"owner": self.owner,
			"business_days": list(self.business_days),
			"blackout_windows": list(self.blackout_windows),
			"holiday_calendar": self.holiday_calendar,
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


@dataclass
class WorkerPool:
	"""Execution capacity lane for scheduled jobs."""

	id: str
	tenant_id: str
	name: str
	queue: str
	max_concurrency: int
	state: str = "ready"
	health_check_required: bool = True
	capacity_limits_required: bool = True
	autoscaling_enabled: bool = False
	heartbeat_ref: str = ""
	state_reason: str = ""
	updated_at: datetime | None = None
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"queue": self.queue,
			"max_concurrency": self.max_concurrency,
			"state": self.state,
			"health_check_required": self.health_check_required,
			"capacity_limits_required": self.capacity_limits_required,
			"autoscaling_enabled": self.autoscaling_enabled,
			"heartbeat_ref": self.heartbeat_ref,
			"state_reason": self.state_reason,
			"updated_at": self.updated_at.isoformat() if self.updated_at else None,
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


@dataclass
class JobDefinition:
	"""Reusable job definition with runtime governance metadata."""

	id: str
	tenant_id: str
	name: str
	command: str
	owner: str
	criticality: str = "normal"
	expected_runtime_minutes: int = 30
	external_job: bool = False
	monitoring_attached: bool = False
	approval_recorded: bool = False
	retry_strategy: str = "fixed"
	retry_policy_ref: str = ""
	max_attempts: int = 3
	dead_letter_enabled: bool = True
	enabled: bool = True
	tags: list[str] = field(default_factory=list)
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"command": self.command,
			"owner": self.owner,
			"criticality": self.criticality,
			"expected_runtime_minutes": self.expected_runtime_minutes,
			"external_job": self.external_job,
			"monitoring_attached": self.monitoring_attached,
			"approval_recorded": self.approval_recorded,
			"retry_strategy": self.retry_strategy,
			"retry_policy_ref": self.retry_policy_ref,
			"max_attempts": self.max_attempts,
			"dead_letter_enabled": self.dead_letter_enabled,
			"enabled": self.enabled,
			"tags": list(self.tags),
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


@dataclass
class ScheduleDefinition:
	"""Tenant-owned schedule binding a job, calendar, worker pool, and trigger."""

	id: str
	tenant_id: str
	name: str
	job_id: str
	calendar_policy_id: str
	worker_pool_id: str
	trigger_type: str
	timezone: str
	owner: str
	enabled: bool = True
	interval_minutes: int | None = None
	cron: str | None = None
	event_policy_ref: str = ""
	manual_reason: str | None = None
	next_run_hint: str = ""
	state: str = "active"
	state_reason: str = ""
	created_at: datetime | None = None
	updated_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"job_id": self.job_id,
			"calendar_policy_id": self.calendar_policy_id,
			"worker_pool_id": self.worker_pool_id,
			"trigger_type": self.trigger_type,
			"timezone": self.timezone,
			"owner": self.owner,
			"enabled": self.enabled,
			"interval_minutes": self.interval_minutes,
			"cron": self.cron,
			"event_policy_ref": self.event_policy_ref,
			"manual_reason": self.manual_reason,
			"next_run_hint": self.next_run_hint,
			"state": self.state,
			"state_reason": self.state_reason,
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"updated_at": self.updated_at.isoformat() if self.updated_at else None,
		}


@dataclass
class JobRun:
	"""Execution attempt for a schedule."""

	id: str
	tenant_id: str
	schedule_id: str
	job_id: str
	worker_pool_id: str
	requested_by: str
	event_stream: str = "bytewax"
	status: str = "queued"
	attempt: int = 1
	records_processed: int = 0
	error_count: int = 0
	exit_code: int | None = None
	logs: list[str] = field(default_factory=list)
	next_retry_seconds: int = 0
	cancel_reason: str = ""
	dead_letter_reason: str = ""
	parent_run_id: str = ""
	completion_evidence_ref: str = ""
	started_at: datetime | None = None
	completed_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"schedule_id": self.schedule_id,
			"job_id": self.job_id,
			"worker_pool_id": self.worker_pool_id,
			"requested_by": self.requested_by,
			"event_stream": self.event_stream,
			"status": self.status,
			"attempt": self.attempt,
			"records_processed": self.records_processed,
			"error_count": self.error_count,
			"exit_code": self.exit_code,
			"logs": list(self.logs),
			"next_retry_seconds": self.next_retry_seconds,
			"cancel_reason": self.cancel_reason,
			"dead_letter_reason": self.dead_letter_reason,
			"parent_run_id": self.parent_run_id,
			"completion_evidence_ref": self.completion_evidence_ref,
			"started_at": self.started_at.isoformat() if self.started_at else None,
			"completed_at": self.completed_at.isoformat() if self.completed_at else None,
		}


@dataclass
class SchdAuditEvent:
	"""Audit trail entry for scheduler operations."""

	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "info"
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"message": self.message,
			"actor": self.actor,
			"severity": self.severity,
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


@dataclass
class SchedulerAgent:
	"""Scoped AI assistant for scheduler design, recovery, and operations."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope_ref: str
	registered_by: str
	contribution_disclosed: bool
	status: str = "active"
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope_ref": self.scope_ref,
			"registered_by": self.registered_by,
			"contribution_disclosed": self.contribution_disclosed,
			"status": self.status,
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


SchdRecord = ScheduleDefinition
