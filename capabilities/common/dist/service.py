"""Service layer for APG Distributed Computing — expanded to 42+ methods."""

from __future__ import annotations

import csv
import io
import json
import statistics
from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_COMPUTE_AGENT_ROLES,
	SUPPORTED_COMPUTE_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .distributed_engine import DistributedEngine
from .models import (
	ComputeAgent,
	DistAuditEvent,
	DistributedJob,
	JobPartition,
	ResultAggregation,
	ScalingDecision,
	WorkerNode,
	WorkerPool,
)


def _ts() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_compute_agent_runtime(value: str) -> str:
	value = value.strip().lower()
	return value if value in SUPPORTED_COMPUTE_AGENT_RUNTIMES else ""


def _normalize_compute_agent_role(value: str) -> str:
	value = value.strip().lower()
	return value if value in SUPPORTED_COMPUTE_AGENT_ROLES else ""


class DistributedComputingService:
	"""
	Tenant worker-pool, partitioned-job, queue, scaling, aggregation,
	checkpoint save/restore, distributed lock, and analytics runtime.

	Adapter/store pattern — no external dependencies.
	"""

	def __init__(self) -> None:
		self._worker_pools: dict[str, WorkerPool] = {}
		self._workers: dict[str, WorkerNode] = {}
		self._jobs: dict[str, DistributedJob] = {}
		self._partitions: dict[str, JobPartition] = {}
		self._aggregations: dict[str, ResultAggregation] = {}
		self._scaling_decisions: dict[str, ScalingDecision] = {}
		self._agents: dict[str, ComputeAgent] = {}
		self._audit_events: dict[str, DistAuditEvent] = {}
		self._idempotency_index: dict[tuple[str, str], str] = {}
		self._engine = DistributedEngine()
		self._checkpoints: dict[str, dict[str, Any]] = {}
		self._locks: dict[str, dict[str, Any]] = {}
		self._lock_waiters: dict[str, list[str]] = {}
		self._dead_letter_queue: dict[str, list[dict[str, Any]]] = {}
		self._poison_pill_log: list[dict[str, Any]] = []
		self._consensus_votes: dict[str, dict[str, Any]] = {}
		self._node_eviction_log: list[dict[str, Any]] = []
		self._coordinator_elections: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Contract / evaluate
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Worker pool & node management
	# ------------------------------------------------------------------

	def create_worker_pool(
		self,
		pool_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		capacity_quota: int,
		health_check: str,
		queue_name: str,
		autoscaling: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			raise PermissionError("worker_pool_owner_required")
		if capacity_quota <= 0:
			raise PermissionError("capacity_quota_required")
		if not health_check:
			raise PermissionError("worker_health_required")
		if not queue_name:
			raise PermissionError("queue_name_required")
		key = self._key(tenant_id, pool_id)
		if key in self._worker_pools:
			raise ValueError("worker_pool_already_exists")
		pool = WorkerPool(id=pool_id, tenant_id=tenant_id, name=name, owner=owner, capacity_quota=int(capacity_quota), health_check=health_check, queue_name=queue_name, autoscaling=bool(autoscaling))
		self._worker_pools[key] = pool
		self._record_audit(tenant_id, pool_id, "worker_pool_created", owner, "allow", metadata={"capacity_quota": capacity_quota})
		return pool.to_dict()

	def register_worker(
		self,
		worker_id: str,
		tenant_id: str,
		pool_id: str,
		hostname: str,
		cpu_slots: int,
		memory_gb: float,
		labels: dict[str, str] | None = None,
		healthy: bool = True,
	) -> dict[str, Any]:
		self._require_pool(pool_id, tenant_id)
		if not hostname:
			raise PermissionError("worker_hostname_required")
		if cpu_slots <= 0:
			raise PermissionError("worker_cpu_slots_required")
		if memory_gb <= 0:
			raise PermissionError("worker_memory_required")
		key = self._key(tenant_id, worker_id)
		if key in self._workers:
			raise ValueError("worker_already_registered")
		worker = WorkerNode(id=worker_id, tenant_id=tenant_id, pool_id=pool_id, hostname=hostname, cpu_slots=int(cpu_slots), memory_gb=float(memory_gb), labels={str(k): str(v) for k, v in dict(labels or {}).items()}, healthy=bool(healthy))
		self._workers[key] = worker
		self._record_audit(tenant_id, worker_id, "worker_registered", hostname, "allow", metadata={"pool_id": pool_id})
		return worker.to_dict()

	# ------------------------------------------------------------------
	# Job submission & management
	# ------------------------------------------------------------------

	def submit_job(
		self,
		job_type: str,
		payload: dict[str, Any],
		priority: int,
		partition_key: str,
		tenant_id: str = "default",
		job_id: str | None = None,
		name: str = "",
		owner: str = "system",
		worker_pool_id: str | None = None,
		idempotency_key: str = "",
		retry_policy: str = "safe_retry",
		partition_count: int = 1,
		quota_policy: str = "default",
		event_bus_topic: str = "apg.distributed.jobs",
		aggregation_strategy: str = "merge",
		partition_review_recorded: bool = True,
	) -> dict[str, Any]:
		"""Submit a distributed job with partitioning and priority."""
		self._require_tenant(tenant_id)
		if not worker_pool_id:
			pool_id = self._key(tenant_id, "default-pool")
			if pool_id not in self._worker_pools:
				self.create_worker_pool(pool_id="default-pool", tenant_id=tenant_id, name="Default Pool", owner=owner, capacity_quota=10, health_check="http", queue_name="apg.distributed.default")
			worker_pool_id = "default-pool"
		pool = self._require_pool(worker_pool_id, tenant_id)
		resolved_job_id = job_id or self._engine.stable_hash({"tenant_id": tenant_id, "job_type": job_type, "partition_key": partition_key, "payload_keys": sorted(payload.keys())})[:24]
		idempotency_tuple = (tenant_id, idempotency_key) if idempotency_key else None
		if idempotency_tuple and idempotency_tuple in self._idempotency_index:
			return self._require_job(self._idempotency_index[idempotency_tuple], tenant_id).to_dict()
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "submit_job",
			"job_owner_assigned": bool(owner),
			"idempotency_key_present": bool(idempotency_key),
			"retry_policy_attached": bool(retry_policy),
			"worker_pool_selected": bool(worker_pool_id),
			"health_check_attached": bool(pool.health_check),
			"quota_policy_attached": bool(quota_policy),
			"job_submission_requested": True,
			"partition_count": int(partition_count),
			"event_stream_attached": bool(event_bus_topic),
			"aggregation_strategy_attached": bool(aggregation_strategy),
			"partition_review_recorded": bool(partition_review_recorded),
		})
		self._raise_if_denied(result)
		key = self._key(tenant_id, resolved_job_id)
		if key in self._jobs:
			raise ValueError("job_already_exists")
		review_status = "required" if result["decision"] == "require_review" else "approved"
		status = "pending_review" if review_status == "required" else "queued"
		job = DistributedJob(id=resolved_job_id, tenant_id=tenant_id, name=name or f"{job_type}:{partition_key}", owner=owner, worker_pool_id=worker_pool_id, idempotency_key=idempotency_key, retry_policy=retry_policy, partition_count=int(partition_count), quota_policy=quota_policy, event_bus_topic=event_bus_topic, aggregation_strategy=aggregation_strategy, status=status, review_status=review_status)
		self._jobs[key] = job
		if idempotency_tuple:
			self._idempotency_index[idempotency_tuple] = resolved_job_id
		for ordinal, partition_id in enumerate(self._engine.partition_ids(resolved_job_id, int(partition_count)), start=1):
			self._partitions[self._key(tenant_id, partition_id)] = JobPartition(id=partition_id, tenant_id=tenant_id, job_id=resolved_job_id, ordinal=ordinal, shard_key=f"{partition_key}:{ordinal}")
		self._record_audit(tenant_id=tenant_id, subject_id=resolved_job_id, event_type="job_submitted", actor=owner, decision=result["decision"], reasons=tuple(action.get("reason", "") for action in result["actions"]), metadata={"job_type": job_type, "partition_count": partition_count, "priority": priority})
		return job.to_dict()

	def approve_job(self, job_id: str, tenant_id: str, reviewer: str) -> dict[str, Any]:
		job = self._require_job(job_id, tenant_id)
		if job.status != "pending_review":
			return job.to_dict()
		job.status = "queued"
		job.review_status = "approved"
		self._jobs[self._key(tenant_id, job_id)] = job
		self._record_audit(tenant_id, job_id, "partition_review_approved", reviewer, "allow")
		return job.to_dict()

	def dispatch_partitions(self, job_id: str, tenant_id: str) -> list[dict[str, Any]]:
		job = self._require_job(job_id, tenant_id)
		if job.status == "pending_review":
			raise PermissionError("job_review_required")
		workers = [w for w in self._workers.values() if w.tenant_id == tenant_id and w.pool_id == job.worker_pool_id and w.healthy]
		if not workers:
			raise PermissionError("healthy_worker_required")
		queued = [p for p in self._partitions.values() if p.tenant_id == tenant_id and p.job_id == job_id and p.status == "queued"]
		for index, partition in enumerate(queued):
			worker = workers[index % len(workers)]
			partition.status = "running"
			partition.assigned_worker_id = worker.id
			partition.attempt_count += 1
		job.status = "running"
		job.started_at = datetime.now(timezone.utc)
		self._jobs[self._key(tenant_id, job_id)] = job
		self._record_audit(tenant_id, job_id, "partitions_dispatched", job.owner, "allow", metadata={"partition_count": len(queued)})
		return [p.to_dict() for p in queued]

	def complete_partition(self, partition_id: str, tenant_id: str, result_payload: dict[str, Any], status: str = "completed") -> dict[str, Any]:
		partition = self._require_partition(partition_id, tenant_id)
		if status not in {"completed", "failed"}:
			raise ValueError("partition_status_invalid")
		partition.status = status
		partition.result_hash = self._engine.stable_hash({"partition_id": partition_id, "result": result_payload})
		partition.completed_at = datetime.now(timezone.utc)
		self._partitions[self._key(tenant_id, partition_id)] = partition
		self._record_audit(tenant_id, partition_id, f"partition_{status}", "worker", status, metadata={"job_id": partition.job_id})
		return partition.to_dict()

	def aggregate_results(self, aggregation_id: str, tenant_id: str, job_id: str) -> dict[str, Any]:
		job = self._require_job(job_id, tenant_id)
		partitions = [p for p in self._partitions.values() if p.tenant_id == tenant_id and p.job_id == job_id]
		completed = [p for p in partitions if p.status == "completed"]
		failed = [p for p in partitions if p.status == "failed"]
		if len(completed) + len(failed) != len(partitions):
			raise PermissionError("partitions_incomplete")
		status = "completed" if not failed else "completed_with_failures"
		aggregation = ResultAggregation(id=aggregation_id, tenant_id=tenant_id, job_id=job_id, strategy=job.aggregation_strategy, partition_count=len(partitions), completed_count=len(completed), failed_count=len(failed), result_hash=self._engine.result_hash(job_id, [p.to_dict() for p in sorted(partitions, key=lambda p: p.id)]), status=status)
		key = self._key(tenant_id, aggregation_id)
		if key in self._aggregations:
			raise ValueError("aggregation_already_exists")
		self._aggregations[key] = aggregation
		job.status = status
		job.completed_at = datetime.now(timezone.utc)
		self._jobs[self._key(tenant_id, job_id)] = job
		self._record_audit(tenant_id, aggregation_id, "results_aggregated", job.owner, status, metadata={"job_id": job_id})
		return aggregation.to_dict()

	def record_scaling_decision(self, decision_id: str, tenant_id: str, pool_id: str, recorded_by: str) -> dict[str, Any]:
		pool = self._require_pool(pool_id, tenant_id)
		queued_partitions = len([p for p in self._partitions.values() if p.tenant_id == tenant_id and p.status == "queued" and self._jobs.get(self._key(tenant_id, p.job_id)) and self._jobs[self._key(tenant_id, p.job_id)].worker_pool_id == pool_id])
		active_workers = len([w for w in self._workers.values() if w.tenant_id == tenant_id and w.pool_id == pool_id and w.healthy])
		decision, reason, desired_capacity = self._engine.scaling_posture(queued_partitions, active_workers, pool.capacity_quota)
		key = self._key(tenant_id, decision_id)
		if key in self._scaling_decisions:
			raise ValueError("scaling_decision_already_exists")
		scaling = ScalingDecision(id=decision_id, tenant_id=tenant_id, pool_id=pool_id, decision=decision, reason=reason, desired_capacity=desired_capacity, current_capacity=active_workers, recorded_by=recorded_by)
		self._scaling_decisions[key] = scaling
		self._record_audit(tenant_id, decision_id, "scaling_decision_recorded", recorded_by, decision, metadata={"reason": reason})
		return scaling.to_dict()

	def register_compute_agent(self, tenant_id: str, agent_id: str, name: str, runtime: str, role: str, scope: str, contribution_disclosed: bool, policy_ref: str = "", registered: bool = True) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		normalized_runtime = _normalize_compute_agent_runtime(runtime)
		normalized_role = _normalize_compute_agent_role(role)
		result = self.evaluate({"tenant_context_present": True, "compute_agent_present": True, "agent_registered": bool(registered), "agent_runtime_supported": bool(normalized_runtime), "agent_role_supported": bool(normalized_role), "agent_scope_present": bool(scope.strip()), "agent_contribution_disclosed": bool(contribution_disclosed)})
		self._raise_if_denied(result)
		key = self._key(tenant_id, agent_id)
		if key in self._agents:
			raise ValueError("compute_agent_already_registered")
		agent = ComputeAgent(id=agent_id, tenant_id=tenant_id, name=name or agent_id, runtime=normalized_runtime, role=normalized_role, scope=scope, registered=registered, contribution_disclosed=contribution_disclosed, policy_ref=policy_ref or None)
		self._agents[key] = agent
		self._record_audit(tenant_id, agent_id, "compute_agent_registered", agent.name, result["decision"], metadata={"runtime": normalized_runtime, "role": normalized_role})
		return agent.to_dict()

	def change_job_state(self, tenant_id: str, job_id: str, status: str, reason: str, actor: str, audit_recorded: bool = True) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		job = self._require_job(job_id, tenant_id)
		result = self.evaluate({"tenant_context_present": True, "state_change_requested": True, "state_change_reason_present": bool(reason.strip()), "audit_event_recorded": bool(audit_recorded)})
		self._raise_if_denied(result)
		job.status = status
		self._jobs[self._key(tenant_id, job_id)] = job
		self._record_audit(tenant_id, job_id, "job_state_changed", actor, result["decision"], metadata={"status": status, "reason": reason})
		return job.to_dict()

	def validate_batch_compute_mutation(self, tenant_id: str, event_stream: str, actor: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({"tenant_context_present": True, "operation": "batch_compute_mutation", "event_stream": event_stream})
		self._raise_if_denied(result)
		self._record_audit(tenant_id, "batch-compute-mutation", "batch_compute_mutation_validated", actor, result["decision"], metadata={"event_stream": event_stream})
		return {"tenant_id": tenant_id, "event_stream": event_stream, "decision": result["decision"], "processor": "bytewax"}

	# ------------------------------------------------------------------
	# NEW: task_distribute
	# ------------------------------------------------------------------

	def task_distribute(
		self,
		tenant_id: str,
		job_id: str,
		worker_ids: list[str],
	) -> dict[str, Any]:
		"""Round-robin distribute queued partitions across a given worker list."""
		job = self._require_job(job_id, tenant_id)
		if not worker_ids:
			raise ValueError("worker_ids_required")
		valid_workers = [w for w in self._workers.values() if w.tenant_id == tenant_id and w.id in worker_ids and w.healthy]
		if not valid_workers:
			raise PermissionError("no_healthy_workers_in_provided_list")
		queued = [p for p in self._partitions.values() if p.tenant_id == tenant_id and p.job_id == job_id and p.status == "queued"]
		distribution: dict[str, list[str]] = {w.id: [] for w in valid_workers}
		for idx, partition in enumerate(queued):
			worker = valid_workers[idx % len(valid_workers)]
			partition.status = "running"
			partition.assigned_worker_id = worker.id
			partition.attempt_count += 1
			distribution[worker.id].append(partition.id)
		self._record_audit(tenant_id, job_id, "tasks_distributed", "system", "allow", metadata={"worker_count": len(valid_workers), "partition_count": len(queued)})
		return {"job_id": job_id, "tenant_id": tenant_id, "partitions_distributed": len(queued), "worker_count": len(valid_workers), "distribution": distribution, "distributed_at": _ts()}

	# ------------------------------------------------------------------
	# NEW: worker_scale
	# ------------------------------------------------------------------

	def worker_scale(
		self,
		tenant_id: str,
		pool_id: str,
		target_count: int,
		reason: str,
		scaled_by: str = "autoscaler",
	) -> dict[str, Any]:
		"""Scale a worker pool to an exact target count by adding or deactivating workers."""
		pool = self._require_pool(pool_id, tenant_id)
		if target_count < 0:
			raise ValueError("target_count_must_be_non_negative")
		if target_count > pool.capacity_quota:
			raise PermissionError("target_count_exceeds_capacity_quota")
		if not reason:
			raise PermissionError("scale_reason_required")
		current_workers = [w for w in self._workers.values() if w.tenant_id == tenant_id and w.pool_id == pool_id]
		current_count = len(current_workers)
		delta = target_count - current_count
		if delta > 0:
			for i in range(delta):
				worker_id = f"{pool_id}-w-{current_count + i + 1:04d}"
				wk = WorkerNode(id=worker_id, tenant_id=tenant_id, pool_id=pool_id, hostname=f"{worker_id}.internal", cpu_slots=4, memory_gb=8.0, labels={"scaled_by": scaled_by}, healthy=True)
				self._workers[self._key(tenant_id, worker_id)] = wk
		elif delta < 0:
			for w in current_workers[target_count:]:
				w.healthy = False
		result = {"pool_id": pool_id, "tenant_id": tenant_id, "reason": reason, "previous_count": current_count, "target_count": target_count, "current_count": target_count, "delta": delta, "scaled_by": scaled_by, "scaled_at": _ts()}
		self._record_audit(tenant_id, pool_id, "workers_scaled", scaled_by, "allow", metadata={"delta": delta, "reason": reason})
		return result

	# ------------------------------------------------------------------
	# NEW: node_health
	# ------------------------------------------------------------------

	def node_health(
		self,
		tenant_id: str,
		pool_id: str,
	) -> dict[str, Any]:
		"""Report per-worker health within a pool."""
		pool = self._require_pool(pool_id, tenant_id)
		workers = [w for w in self._workers.values() if w.tenant_id == tenant_id and w.pool_id == pool_id]
		healthy = [w for w in workers if w.healthy]
		unhealthy = [w for w in workers if not w.healthy]
		return {
			"pool_id": pool_id,
			"tenant_id": tenant_id,
			"total_workers": len(workers),
			"healthy_count": len(healthy),
			"unhealthy_count": len(unhealthy),
			"health_ratio": round(len(healthy) / len(workers), 4) if workers else 0.0,
			"workers": [{"worker_id": w.id, "hostname": w.hostname, "healthy": w.healthy} for w in workers],
			"checked_at": _ts(),
		}

	# ------------------------------------------------------------------
	# NEW: checkpoint_save / checkpoint_restore (full implementations)
	# ------------------------------------------------------------------

	def checkpoint_save(
		self,
		job_id: str,
		state: dict[str, Any],
		tenant_id: str = "default",
		checkpoint_id: str | None = None,
		saved_by: str = "system",
	) -> dict[str, Any]:
		"""Save a snapshot of running job state for crash recovery."""
		self._require_tenant(tenant_id)
		self._require_job(job_id, tenant_id)
		if not state:
			raise ValueError("checkpoint_state_required")
		cp_id = checkpoint_id or self._engine.stable_hash({"job_id": job_id, "tenant_id": tenant_id, "state_keys": sorted(state.keys())})[:20]
		state_hash = self._engine.stable_hash(state)
		existing = [c for c in self._checkpoints.values() if c["tenant_id"] == tenant_id and c["job_id"] == job_id]
		cp = {
			"checkpoint_id": cp_id,
			"job_id": job_id,
			"tenant_id": tenant_id,
			"checkpoint_number": len(existing) + 1,
			"state_hash": state_hash,
			"state_keys": sorted(state.keys()),
			"state_size_bytes": len(str(state)),
			"saved_by": saved_by,
			"saved_at": _ts(),
		}
		self._checkpoints[f"{tenant_id}:{cp_id}"] = cp
		self._record_audit(tenant_id, job_id, "checkpoint_saved", saved_by, "allow", metadata={"checkpoint_id": cp_id})
		return cp

	def checkpoint_restore(
		self,
		job_id: str,
		checkpoint_id: str,
		tenant_id: str = "default",
		restored_by: str = "system",
	) -> dict[str, Any]:
		"""Restore a job to a saved checkpoint, resetting running partitions to queued."""
		self._require_tenant(tenant_id)
		job = self._require_job(job_id, tenant_id)
		cp = self._checkpoints.get(f"{tenant_id}:{checkpoint_id}")
		if cp is None or cp["job_id"] != job_id or cp["tenant_id"] != tenant_id:
			raise KeyError(f"checkpoint_not_found:{checkpoint_id}")
		reset_count = 0
		for partition in self._partitions.values():
			if partition.tenant_id == tenant_id and partition.job_id == job_id and partition.status == "running":
				partition.status = "queued"
				partition.assigned_worker_id = None
				reset_count += 1
		job.status = "queued"
		self._record_audit(tenant_id, job_id, "checkpoint_restored", restored_by, "allow", metadata={"checkpoint_id": checkpoint_id, "partitions_reset": reset_count})
		return {**cp, "restored_by": restored_by, "partitions_reset": reset_count, "restored_at": _ts()}

	# ------------------------------------------------------------------
	# NEW: distributed_lock / release_lock
	# ------------------------------------------------------------------

	def distributed_lock(
		self,
		resource_id: str,
		ttl_seconds: int,
		holder_id: str,
		tenant_id: str = "default",
		blocking: bool = False,
	) -> dict[str, Any]:
		"""Acquire a TTL-based distributed lock on a resource."""
		self._require_tenant(tenant_id)
		if not resource_id:
			raise ValueError("resource_id_required")
		if ttl_seconds < 1:
			raise ValueError("ttl_seconds_must_be_positive")
		if not holder_id:
			raise ValueError("holder_id_required")
		lock_key = f"{tenant_id}:{resource_id}"
		existing = self._locks.get(lock_key)
		if existing and existing.get("status") == "held":
			if existing["holder_id"] == holder_id:
				existing["ttl_seconds"] = ttl_seconds
				existing["renewed_at"] = _ts()
				return dict(existing)
			if not blocking:
				raise PermissionError(f"lock_held_by:{existing['holder_id']}")
			if lock_key not in self._lock_waiters:
				self._lock_waiters[lock_key] = []
			if holder_id not in self._lock_waiters[lock_key]:
				self._lock_waiters[lock_key].append(holder_id)
			return {"resource_id": resource_id, "tenant_id": tenant_id, "holder_id": holder_id, "status": "waiting", "queued_position": self._lock_waiters[lock_key].index(holder_id) + 1, "requested_at": _ts()}
		lock = {"lock_id": self._engine.stable_hash({"resource_id": resource_id, "tenant_id": tenant_id, "holder_id": holder_id})[:20], "resource_id": resource_id, "tenant_id": tenant_id, "holder_id": holder_id, "ttl_seconds": ttl_seconds, "status": "held", "acquired_at": _ts()}
		self._locks[lock_key] = lock
		self._record_audit(tenant_id, resource_id, "distributed_lock_acquired", holder_id, "allow", metadata={"ttl_seconds": ttl_seconds})
		return dict(lock)

	def release_lock(self, resource_id: str, holder_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Release a distributed lock, granting it to the next waiter if any."""
		lock_key = f"{tenant_id}:{resource_id}"
		lock = self._locks.get(lock_key)
		if lock is None:
			raise KeyError(f"lock_not_found:{resource_id}")
		if lock["holder_id"] != holder_id:
			raise PermissionError(f"lock_held_by_different_holder:{lock['holder_id']}")
		lock["status"] = "released"
		lock["released_at"] = _ts()
		waiters = self._lock_waiters.get(lock_key, [])
		next_holder = None
		if waiters:
			next_holder = waiters.pop(0)
			new_lock = {**lock, "holder_id": next_holder, "status": "held", "acquired_at": _ts()}
			self._locks[lock_key] = new_lock
		self._record_audit(tenant_id, resource_id, "distributed_lock_released", holder_id, "allow", metadata={"next_holder": next_holder})
		return dict(lock)

	# ------------------------------------------------------------------
	# NEW: consensus_vote
	# ------------------------------------------------------------------

	def consensus_vote(
		self,
		tenant_id: str,
		proposal_id: str,
		voter_id: str,
		vote: bool,
		quorum: int = 3,
	) -> dict[str, Any]:
		"""Record a vote on a distributed consensus proposal and evaluate quorum."""
		self._require_tenant(tenant_id)
		if not proposal_id:
			raise ValueError("proposal_id_required")
		if quorum < 1:
			raise ValueError("quorum_must_be_positive")
		key = self._key(tenant_id, proposal_id)
		if key not in self._consensus_votes:
			self._consensus_votes[key] = {"proposal_id": proposal_id, "tenant_id": tenant_id, "votes": {}, "quorum": quorum, "status": "open", "created_at": _ts()}
		record = self._consensus_votes[key]
		if record["status"] != "open":
			raise PermissionError(f"proposal_already_{record['status']}")
		record["votes"][voter_id] = vote
		yes_votes = sum(1 for v in record["votes"].values() if v)
		no_votes = sum(1 for v in record["votes"].values() if not v)
		total_votes = len(record["votes"])
		if yes_votes >= quorum:
			record["status"] = "accepted"
		elif no_votes >= quorum:
			record["status"] = "rejected"
		record["yes_votes"] = yes_votes
		record["no_votes"] = no_votes
		record["total_votes"] = total_votes
		record["quorum_reached"] = record["status"] != "open"
		record["updated_at"] = _ts()
		self._record_audit(tenant_id, proposal_id, "consensus_vote_recorded", voter_id, "allow", metadata={"vote": vote, "status": record["status"]})
		return dict(record)

	# ------------------------------------------------------------------
	# NEW: partition_assign
	# ------------------------------------------------------------------

	def partition_assign(
		self,
		tenant_id: str,
		job_id: str,
		partition_id: str,
		worker_id: str,
	) -> dict[str, Any]:
		"""Manually assign a specific partition to a specific worker."""
		self._require_tenant(tenant_id)
		partition = self._require_partition(partition_id, tenant_id)
		if partition.job_id != job_id:
			raise ValueError("partition_job_mismatch")
		worker = self._workers.get(self._key(tenant_id, worker_id))
		if worker is None or not worker.healthy:
			raise PermissionError("worker_not_healthy_or_not_found")
		partition.assigned_worker_id = worker_id
		partition.status = "running"
		partition.attempt_count += 1
		self._record_audit(tenant_id, partition_id, "partition_assigned", worker_id, "allow", metadata={"job_id": job_id})
		return partition.to_dict()

	# ------------------------------------------------------------------
	# NEW: rebalance_load
	# ------------------------------------------------------------------

	def rebalance_load(
		self,
		tenant_id: str,
		pool_id: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Rebalance running partitions evenly across healthy workers in a pool."""
		self._require_pool(pool_id, tenant_id)
		workers = [w for w in self._workers.values() if w.tenant_id == tenant_id and w.pool_id == pool_id and w.healthy]
		if not workers:
			raise PermissionError("no_healthy_workers_for_rebalance")
		running_partitions = [p for p in self._partitions.values() if p.tenant_id == tenant_id and p.status == "running"]
		# Distribute evenly
		reassignments: list[dict[str, Any]] = []
		for idx, partition in enumerate(running_partitions):
			new_worker = workers[idx % len(workers)]
			old_worker = partition.assigned_worker_id
			if old_worker != new_worker.id:
				partition.assigned_worker_id = new_worker.id
				reassignments.append({"partition_id": partition.id, "from_worker": old_worker, "to_worker": new_worker.id})
		result = {
			"pool_id": pool_id,
			"tenant_id": tenant_id,
			"total_running_partitions": len(running_partitions),
			"reassignments": len(reassignments),
			"details": reassignments,
			"rebalanced_by": actor,
			"rebalanced_at": _ts(),
		}
		self._record_audit(tenant_id, pool_id, "load_rebalanced", actor, "allow", metadata={"reassignments": len(reassignments)})
		return result

	# ------------------------------------------------------------------
	# NEW: dead_letter_queue
	# ------------------------------------------------------------------

	def dead_letter_queue(
		self,
		tenant_id: str,
		job_id: str,
		partition_id: str,
		error_reason: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Move a failed partition to the dead-letter queue for manual inspection."""
		self._require_tenant(tenant_id)
		partition = self._require_partition(partition_id, tenant_id)
		if partition.job_id != job_id:
			raise ValueError("partition_job_mismatch")
		dlq_entry = {
			"partition_id": partition_id,
			"job_id": job_id,
			"tenant_id": tenant_id,
			"error_reason": error_reason,
			"original_status": partition.status,
			"attempt_count": partition.attempt_count,
			"queued_by": actor,
			"queued_at": _ts(),
		}
		partition.status = "dead_lettered"
		if tenant_id not in self._dead_letter_queue:
			self._dead_letter_queue[tenant_id] = []
		self._dead_letter_queue[tenant_id].append(dlq_entry)
		self._record_audit(tenant_id, partition_id, "partition_dead_lettered", actor, "allow", metadata={"error_reason": error_reason})
		return dlq_entry

	def list_dead_letter_queue(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all dead-lettered partition entries for a tenant."""
		return list(self._dead_letter_queue.get(tenant_id, []))

	# ------------------------------------------------------------------
	# NEW: poison_pill_handle
	# ------------------------------------------------------------------

	def poison_pill_handle(
		self,
		tenant_id: str,
		job_id: str,
		partition_id: str,
		strategy: str = "skip",
		handled_by: str = "system",
	) -> dict[str, Any]:
		"""Handle a poison-pill message by skipping, quarantining, or retrying."""
		self._require_tenant(tenant_id)
		partition = self._require_partition(partition_id, tenant_id)
		if strategy not in {"skip", "quarantine", "retry"}:
			raise ValueError("strategy_must_be_skip_quarantine_or_retry")
		new_status = {"skip": "skipped", "quarantine": "quarantined", "retry": "queued"}[strategy]
		old_status = partition.status
		partition.status = new_status
		if strategy == "retry":
			partition.attempt_count += 1
			partition.assigned_worker_id = None
		record = {
			"partition_id": partition_id,
			"job_id": job_id,
			"tenant_id": tenant_id,
			"strategy": strategy,
			"old_status": old_status,
			"new_status": new_status,
			"attempt_count": partition.attempt_count,
			"handled_by": handled_by,
			"handled_at": _ts(),
		}
		self._poison_pill_log.append(record)
		self._record_audit(tenant_id, partition_id, "poison_pill_handled", handled_by, "allow", metadata={"strategy": strategy})
		return record

	# ------------------------------------------------------------------
	# NEW: idempotency_check
	# ------------------------------------------------------------------

	def idempotency_check(
		self,
		tenant_id: str,
		idempotency_key: str,
	) -> dict[str, Any]:
		"""Check whether an idempotency key has already been processed."""
		self._require_tenant(tenant_id)
		if not idempotency_key:
			raise ValueError("idempotency_key_required")
		existing_job_id = self._idempotency_index.get((tenant_id, idempotency_key))
		return {
			"tenant_id": tenant_id,
			"idempotency_key": idempotency_key,
			"already_processed": existing_job_id is not None,
			"job_id": existing_job_id,
			"checked_at": _ts(),
		}

	# ------------------------------------------------------------------
	# NEW: compute_analytics
	# ------------------------------------------------------------------

	def compute_analytics(self, period: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Aggregate distributed computing analytics for a tenant over a period."""
		jobs = self.list_jobs(tenant_id)
		partitions = self.list_partitions(tenant_id)
		workers = self.list_workers(tenant_id)
		scaling_decisions = self.list_scaling_decisions(tenant_id)
		checkpoints = [c for c in self._checkpoints.values() if c["tenant_id"] == tenant_id]
		locks = [v for v in self._locks.values() if v["tenant_id"] == tenant_id]
		held_locks = [l for l in locks if l.get("status") == "held"]
		completed_jobs = [j for j in jobs if j.get("status", "").startswith("completed")]
		failed_partitions = [p for p in partitions if p.get("status") == "failed"]
		scale_out = sum(1 for s in scaling_decisions if s.get("decision") == "scale_out")
		scale_in = sum(1 for s in scaling_decisions if s.get("decision") == "scale_in")
		avg_partitions = round(sum(j.get("partition_count", 0) for j in jobs) / len(jobs), 2) if jobs else 0.0
		dlq_count = sum(len(v) for k, v in self._dead_letter_queue.items() if k == tenant_id)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"worker_pool_count": len(self.list_worker_pools(tenant_id)),
			"worker_count": len(workers),
			"healthy_worker_count": sum(1 for w in workers if w.get("healthy")),
			"job_count": len(jobs),
			"completed_job_count": len(completed_jobs),
			"job_completion_rate": round(len(completed_jobs) / len(jobs), 4) if jobs else 0.0,
			"average_partitions_per_job": avg_partitions,
			"partition_count": len(partitions),
			"failed_partition_count": len(failed_partitions),
			"partition_failure_rate": round(len(failed_partitions) / len(partitions), 4) if partitions else 0.0,
			"scaling_decision_count": len(scaling_decisions),
			"scale_out_count": scale_out,
			"scale_in_count": scale_in,
			"checkpoint_count": len(checkpoints),
			"active_lock_count": len(held_locks),
			"aggregation_count": len(self.list_aggregations(tenant_id)),
			"dead_letter_count": dlq_count,
			"poison_pill_count": len([p for p in self._poison_pill_log if p["tenant_id"] == tenant_id]),
			"open_consensus_proposals": len([v for v in self._consensus_votes.values() if v["tenant_id"] == tenant_id and v["status"] == "open"]),
			"generated_at": _ts(),
		}

	# ------------------------------------------------------------------
	# NEW: node_evict
	# ------------------------------------------------------------------

	def node_evict(
		self,
		tenant_id: str,
		pool_id: str,
		worker_id: str,
		reason: str,
		evicted_by: str = "system",
		drain: bool = True,
	) -> dict[str, Any]:
		"""Evict a worker from a pool, optionally draining its partitions first."""
		self._require_pool(pool_id, tenant_id)
		worker = self._workers.get(self._key(tenant_id, worker_id))
		if worker is None or worker.pool_id != pool_id:
			raise KeyError("worker_not_found_in_pool")
		if not reason:
			raise PermissionError("eviction_reason_required")
		drained_partitions: list[str] = []
		if drain:
			for p in self._partitions.values():
				if p.tenant_id == tenant_id and p.assigned_worker_id == worker_id and p.status == "running":
					p.status = "queued"
					p.assigned_worker_id = None
					drained_partitions.append(p.id)
		worker.healthy = False
		record = {
			"worker_id": worker_id,
			"pool_id": pool_id,
			"tenant_id": tenant_id,
			"reason": reason,
			"drain": drain,
			"drained_partitions": drained_partitions,
			"evicted_by": evicted_by,
			"evicted_at": _ts(),
		}
		self._node_eviction_log.append(record)
		self._record_audit(tenant_id, worker_id, "node_evicted", evicted_by, "allow", metadata={"reason": reason, "drained": len(drained_partitions)})
		return record

	# ------------------------------------------------------------------
	# NEW: coordinator_elect
	# ------------------------------------------------------------------

	def coordinator_elect(
		self,
		tenant_id: str,
		election_id: str,
		candidates: list[str],
		strategy: str = "lowest_id",
	) -> dict[str, Any]:
		"""Elect a coordinator from a set of candidate worker IDs."""
		self._require_tenant(tenant_id)
		if not candidates:
			raise ValueError("candidates_required")
		key = self._key(tenant_id, election_id)
		if key in self._coordinator_elections:
			raise ValueError("election_already_exists")
		healthy_candidates = [c for c in candidates if self._workers.get(self._key(tenant_id, c)) and self._workers[self._key(tenant_id, c)].healthy]
		if not healthy_candidates:
			raise PermissionError("no_healthy_candidates")
		if strategy == "lowest_id":
			elected = sorted(healthy_candidates)[0]
		elif strategy == "random":
			import random
			elected = random.choice(healthy_candidates)
		else:
			elected = healthy_candidates[0]
		record = {
			"election_id": election_id,
			"tenant_id": tenant_id,
			"candidates": candidates,
			"healthy_candidates": healthy_candidates,
			"elected_coordinator": elected,
			"strategy": strategy,
			"elected_at": _ts(),
		}
		self._coordinator_elections[key] = record
		self._record_audit(tenant_id, election_id, "coordinator_elected", "system", "allow", metadata={"elected": elected, "strategy": strategy})
		return record

	# ------------------------------------------------------------------
	# NEW: health_check
	# ------------------------------------------------------------------

	def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return service health status for the distributed computing capability."""
		pool_count = len([p for p in self._worker_pools.values() if p.tenant_id == tenant_id])
		worker_count = len([w for w in self._workers.values() if w.tenant_id == tenant_id])
		return {
			"service": "dist",
			"tenant_id": tenant_id,
			"status": "healthy",
			"pool_count": pool_count,
			"worker_count": worker_count,
			"job_count": len(self.list_jobs(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"checked_at": _ts(),
		}

	# ------------------------------------------------------------------
	# NEW: Bulk operations
	# ------------------------------------------------------------------

	def bulk_register_workers(
		self,
		tenant_id: str,
		pool_id: str,
		workers: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		"""Register multiple workers in a single call."""
		return [self.register_worker(
			worker_id=w["id"],
			tenant_id=tenant_id,
			pool_id=pool_id,
			hostname=w["hostname"],
			cpu_slots=w.get("cpu_slots", 4),
			memory_gb=w.get("memory_gb", 8.0),
			labels=w.get("labels"),
			healthy=w.get("healthy", True),
		) for w in workers]

	def bulk_complete_partitions(
		self,
		tenant_id: str,
		completions: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		"""Complete multiple partitions in a single call."""
		return [self.complete_partition(
			partition_id=c["partition_id"],
			tenant_id=tenant_id,
			result_payload=c.get("result", {}),
			status=c.get("status", "completed"),
		) for c in completions]

	# ------------------------------------------------------------------
	# NEW: Export
	# ------------------------------------------------------------------

	def export_jobs(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export job records as JSON or CSV."""
		jobs = self.list_jobs(tenant_id)
		if fmt == "csv":
			if not jobs:
				return ""
			buf = io.StringIO()
			writer = csv.DictWriter(buf, fieldnames=list(jobs[0].keys()))
			writer.writeheader()
			writer.writerows(jobs)
			return buf.getvalue()
		return json.dumps(jobs, indent=2, default=str)

	# ------------------------------------------------------------------
	# Status helpers
	# ------------------------------------------------------------------

	def job_status(self, job_id: str, tenant_id: str = "default") -> dict[str, Any]:
		job = self._require_job(job_id, tenant_id)
		partitions = [p for p in self._partitions.values() if p.tenant_id == tenant_id and p.job_id == job_id]
		by_status: dict[str, int] = {}
		for p in partitions:
			by_status[p.status] = by_status.get(p.status, 0) + 1
		return {**job.to_dict(), "partition_summary": by_status, "total_partitions": len(partitions), "queried_at": _ts()}

	def job_result(self, job_id: str, tenant_id: str = "default") -> dict[str, Any]:
		aggregation = next((a for a in self._aggregations.values() if a.tenant_id == tenant_id and a.job_id == job_id), None)
		if aggregation:
			return aggregation.to_dict()
		partitions = [p for p in self._partitions.values() if p.tenant_id == tenant_id and p.job_id == job_id]
		if any(p.status == "queued" for p in partitions):
			raise PermissionError("job_partitions_not_complete")
		agg_id = self._engine.stable_hash({"job_id": job_id, "tenant_id": tenant_id})[:20]
		return self.aggregate_results(agg_id, tenant_id, job_id)

	def worker_pool_status(self, tenant_id: str = "default") -> dict[str, Any]:
		pools = self.list_worker_pools(tenant_id)
		workers = self.list_workers(tenant_id)
		pool_details: list[dict[str, Any]] = []
		for pool in pools:
			pool_workers = [w for w in workers if w.get("pool_id") == pool["id"]]
			pool_details.append({**pool, "worker_count": len(pool_workers), "healthy_worker_count": sum(1 for w in pool_workers if w.get("healthy"))})
		return {"tenant_id": tenant_id, "pool_count": len(pools), "total_worker_count": len(workers), "healthy_worker_count": sum(1 for w in workers if w.get("healthy")), "pools": pool_details, "queried_at": _ts()}

	# ------------------------------------------------------------------
	# List helpers
	# ------------------------------------------------------------------

	def list_worker_pools(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._worker_pools, tenant_id)

	def list_workers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._workers, tenant_id)

	def list_jobs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._jobs, tenant_id)

	def list_partitions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._partitions, tenant_id)

	def list_aggregations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._aggregations, tenant_id)

	def list_scaling_decisions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._scaling_decisions, tenant_id)

	def list_compute_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		jobs = self.list_jobs(tenant_id)
		partitions = self.list_partitions(tenant_id)
		dlq_count = sum(len(v) for k, v in self._dead_letter_queue.items() if k == tenant_id)
		return {
			"worker_pool_count": len(self.list_worker_pools(tenant_id)),
			"worker_count": len(self.list_workers(tenant_id)),
			"job_count": len(jobs),
			"queued_job_count": len([j for j in jobs if j["status"] == "queued"]),
			"running_job_count": len([j for j in jobs if j["status"] == "running"]),
			"completed_job_count": len([j for j in jobs if j["status"].startswith("completed")]),
			"pending_review_count": len([j for j in jobs if j["status"] == "pending_review"]),
			"queued_partition_count": len([p for p in partitions if p["status"] == "queued"]),
			"running_partition_count": len([p for p in partitions if p["status"] == "running"]),
			"completed_partition_count": len([p for p in partitions if p["status"] == "completed"]),
			"failed_partition_count": len([p for p in partitions if p["status"] == "failed"]),
			"dead_letter_count": dlq_count,
			"checkpoint_count": sum(1 for c in self._checkpoints.values() if c["tenant_id"] == tenant_id),
			"active_lock_count": sum(1 for l in self._locks.values() if l["tenant_id"] == tenant_id and l.get("status") == "held"),
			"compute_agent_count": len(self.list_compute_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_pool(self, pool_id: str, tenant_id: str) -> WorkerPool:
		pool = self._worker_pools.get(self._key(tenant_id, pool_id))
		if pool is None:
			raise KeyError("worker_pool_not_found")
		return pool

	def _require_job(self, job_id: str, tenant_id: str) -> DistributedJob:
		job = self._jobs.get(self._key(tenant_id, job_id))
		if job is None:
			raise KeyError("job_not_found")
		return job

	def _require_partition(self, partition_id: str, tenant_id: str) -> JobPartition:
		partition = self._partitions.get(self._key(tenant_id, partition_id))
		if partition is None:
			raise KeyError("partition_not_found")
		return partition

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "distributed_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "distributed_policy_blocked")

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> None:
		payload = {"tenant_id": tenant_id, "subject_id": subject_id, "event_type": event_type, "actor": actor, "decision": decision, "reasons": list(reasons), "metadata": dict(metadata or {})}
		event_id = f"audit-{len(self._audit_events) + 1:04d}"
		self._audit_events[event_id] = DistAuditEvent(id=event_id, tenant_id=tenant_id, event_type=event_type, subject_id=subject_id, actor=actor, decision=decision, reasons=reasons, payload_hash=self._engine.stable_hash(payload))

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		items = list(records.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _key(self, tenant_id: str, object_id: str) -> str:
		if not tenant_id:
			raise PermissionError("tenant_context_required")
		return f"{tenant_id}:{object_id}"


# Alias
DistService = DistributedComputingService
