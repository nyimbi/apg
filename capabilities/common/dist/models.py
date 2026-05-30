"""Distributed-computing domain models for the APG DIST capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
	return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
	return value.astimezone(timezone.utc).isoformat()


@dataclass
class WorkerPool:
	id: str
	tenant_id: str
	name: str
	owner: str
	capacity_quota: int
	health_check: str
	queue_name: str
	autoscaling: bool = True
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"capacity_quota": self.capacity_quota,
			"health_check": self.health_check,
			"queue_name": self.queue_name,
			"autoscaling": self.autoscaling,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class WorkerNode:
	id: str
	tenant_id: str
	pool_id: str
	hostname: str
	cpu_slots: int
	memory_gb: float
	labels: dict[str, str]
	healthy: bool = True
	active_partitions: tuple[str, ...] = ()
	last_heartbeat_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"pool_id": self.pool_id,
			"hostname": self.hostname,
			"cpu_slots": self.cpu_slots,
			"memory_gb": self.memory_gb,
			"labels": dict(self.labels),
			"healthy": self.healthy,
			"active_partitions": list(self.active_partitions),
			"last_heartbeat_at": isoformat(self.last_heartbeat_at),
		}


@dataclass
class DistributedJob:
	id: str
	tenant_id: str
	name: str
	owner: str
	worker_pool_id: str
	idempotency_key: str
	retry_policy: str
	partition_count: int
	quota_policy: str
	event_bus_topic: str
	aggregation_strategy: str
	status: str = "queued"
	review_status: str = "approved"
	submitted_at: datetime = field(default_factory=utc_now)
	started_at: datetime | None = None
	completed_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"worker_pool_id": self.worker_pool_id,
			"idempotency_key": self.idempotency_key,
			"retry_policy": self.retry_policy,
			"partition_count": self.partition_count,
			"quota_policy": self.quota_policy,
			"event_bus_topic": self.event_bus_topic,
			"aggregation_strategy": self.aggregation_strategy,
			"status": self.status,
			"review_status": self.review_status,
			"submitted_at": isoformat(self.submitted_at),
			"started_at": isoformat(self.started_at) if self.started_at else None,
			"completed_at": isoformat(self.completed_at) if self.completed_at else None,
		}


@dataclass
class JobPartition:
	id: str
	tenant_id: str
	job_id: str
	ordinal: int
	shard_key: str
	status: str = "queued"
	assigned_worker_id: str | None = None
	result_hash: str | None = None
	attempt_count: int = 0
	created_at: datetime = field(default_factory=utc_now)
	completed_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"job_id": self.job_id,
			"ordinal": self.ordinal,
			"shard_key": self.shard_key,
			"status": self.status,
			"assigned_worker_id": self.assigned_worker_id,
			"result_hash": self.result_hash,
			"attempt_count": self.attempt_count,
			"created_at": isoformat(self.created_at),
			"completed_at": isoformat(self.completed_at) if self.completed_at else None,
		}


@dataclass
class ResultAggregation:
	id: str
	tenant_id: str
	job_id: str
	strategy: str
	partition_count: int
	completed_count: int
	failed_count: int
	result_hash: str
	status: str
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"job_id": self.job_id,
			"strategy": self.strategy,
			"partition_count": self.partition_count,
			"completed_count": self.completed_count,
			"failed_count": self.failed_count,
			"result_hash": self.result_hash,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class ScalingDecision:
	id: str
	tenant_id: str
	pool_id: str
	decision: str
	reason: str
	desired_capacity: int
	current_capacity: int
	recorded_by: str
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"pool_id": self.pool_id,
			"decision": self.decision,
			"reason": self.reason,
			"desired_capacity": self.desired_capacity,
			"current_capacity": self.current_capacity,
			"recorded_by": self.recorded_by,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class ComputeAgent:
	"""AI compute-agent registration with runtime, role, scope, and disclosure."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
	policy_ref: str | None = None
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"registered": self.registered,
			"contribution_disclosed": self.contribution_disclosed,
			"policy_ref": self.policy_ref,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class DistAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = ()
	payload_hash: str = ""
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"actor": self.actor,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"payload_hash": self.payload_hash,
			"created_at": isoformat(self.created_at),
		}
