"""Service layer for APG Distributed Computing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .distributed_engine import DistributedEngine
from .models import (
	DistAuditEvent,
	DistributedJob,
	JobPartition,
	ResultAggregation,
	ScalingDecision,
	WorkerNode,
	WorkerPool,
)


class DistService:
	"""Tenant worker-pool, partitioned-job, queue, scaling, and aggregation runtime."""

	def __init__(self) -> None:
		self._worker_pools: dict[str, WorkerPool] = {}
		self._workers: dict[str, WorkerNode] = {}
		self._jobs: dict[str, DistributedJob] = {}
		self._partitions: dict[str, JobPartition] = {}
		self._aggregations: dict[str, ResultAggregation] = {}
		self._scaling_decisions: dict[str, ScalingDecision] = {}
		self._audit_events: dict[str, DistAuditEvent] = {}
		self._idempotency_index: dict[tuple[str, str], str] = {}
		self._engine = DistributedEngine()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

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
		pool = WorkerPool(
			id=pool_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			capacity_quota=int(capacity_quota),
			health_check=health_check,
			queue_name=queue_name,
			autoscaling=bool(autoscaling),
		)
		self._worker_pools[pool_id] = pool
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
		worker = WorkerNode(
			id=worker_id,
			tenant_id=tenant_id,
			pool_id=pool_id,
			hostname=hostname,
			cpu_slots=int(cpu_slots),
			memory_gb=float(memory_gb),
			labels={str(key): str(value) for key, value in dict(labels or {}).items()},
			healthy=bool(healthy),
		)
		self._workers[worker_id] = worker
		self._record_audit(tenant_id, worker_id, "worker_registered", hostname, "allow", metadata={"pool_id": pool_id})
		return worker.to_dict()

	def submit_job(
		self,
		job_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		worker_pool_id: str,
		idempotency_key: str,
		retry_policy: str,
		partition_count: int,
		quota_policy: str,
		event_bus_topic: str,
		aggregation_strategy: str,
		partition_review_recorded: bool = True,
	) -> dict[str, Any]:
		pool = self._require_pool(worker_pool_id, tenant_id)
		idempotency_tuple = (tenant_id, idempotency_key)
		if idempotency_key and idempotency_tuple in self._idempotency_index:
			return self._require_job(self._idempotency_index[idempotency_tuple], tenant_id).to_dict()
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "submit_job",
			"job_owner_assigned": bool(owner),
			"idempotency_key_present": bool(idempotency_key),
			"worker_pool_selected": bool(worker_pool_id),
			"health_check_attached": bool(pool.health_check),
			"quota_policy_attached": bool(quota_policy),
			"job_submission_requested": True,
			"partition_count": int(partition_count),
			"partition_review_recorded": bool(partition_review_recorded),
		})
		self._raise_if_denied(result)
		if not retry_policy:
			raise PermissionError("retry_policy_required")
		if partition_count <= 0:
			raise PermissionError("partition_count_required")
		if not event_bus_topic:
			raise PermissionError("event_bus_required")
		if not aggregation_strategy:
			raise PermissionError("result_aggregation_required")
		review_status = "required" if result["decision"] == "require_review" else "approved"
		status = "pending_review" if review_status == "required" else "queued"
		job = DistributedJob(
			id=job_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			worker_pool_id=worker_pool_id,
			idempotency_key=idempotency_key,
			retry_policy=retry_policy,
			partition_count=int(partition_count),
			quota_policy=quota_policy,
			event_bus_topic=event_bus_topic,
			aggregation_strategy=aggregation_strategy,
			status=status,
			review_status=review_status,
		)
		self._jobs[job_id] = job
		self._idempotency_index[idempotency_tuple] = job_id
		for ordinal, partition_id in enumerate(self._engine.partition_ids(job_id, int(partition_count)), start=1):
			self._partitions[partition_id] = JobPartition(
				id=partition_id,
				tenant_id=tenant_id,
				job_id=job_id,
				ordinal=ordinal,
				shard_key=f"{job_id}:{ordinal}",
			)
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=job_id,
			event_type="job_submitted",
			actor=owner,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"partition_count": partition_count, "worker_pool_id": worker_pool_id},
		)
		return job.to_dict()

	def approve_job(self, job_id: str, tenant_id: str, reviewer: str) -> dict[str, Any]:
		job = self._require_job(job_id, tenant_id)
		if job.status != "pending_review":
			return job.to_dict()
		job.status = "queued"
		job.review_status = "approved"
		self._record_audit(tenant_id, job_id, "partition_review_approved", reviewer, "allow")
		return job.to_dict()

	def dispatch_partitions(self, job_id: str, tenant_id: str) -> list[dict[str, Any]]:
		job = self._require_job(job_id, tenant_id)
		if job.status == "pending_review":
			raise PermissionError("job_review_required")
		workers = [worker for worker in self._workers.values() if worker.tenant_id == tenant_id and worker.pool_id == job.worker_pool_id and worker.healthy]
		if not workers:
			raise PermissionError("healthy_worker_required")
		queued = [partition for partition in self._partitions.values() if partition.tenant_id == tenant_id and partition.job_id == job_id and partition.status == "queued"]
		for index, partition in enumerate(queued):
			worker = workers[index % len(workers)]
			partition.status = "running"
			partition.assigned_worker_id = worker.id
			partition.attempt_count += 1
		job.status = "running"
		job.started_at = datetime.now(timezone.utc)
		self._record_audit(tenant_id, job_id, "partitions_dispatched", job.owner, "allow", metadata={"partition_count": len(queued)})
		return [partition.to_dict() for partition in queued]

	def complete_partition(
		self,
		partition_id: str,
		tenant_id: str,
		result_payload: dict[str, Any],
		status: str = "completed",
	) -> dict[str, Any]:
		partition = self._require_partition(partition_id, tenant_id)
		if status not in {"completed", "failed"}:
			raise ValueError("partition_status_invalid")
		partition.status = status
		partition.result_hash = self._engine.stable_hash({"partition_id": partition_id, "result": result_payload})
		partition.completed_at = datetime.now(timezone.utc)
		self._record_audit(tenant_id, partition_id, f"partition_{status}", "worker", status, metadata={"job_id": partition.job_id})
		return partition.to_dict()

	def aggregate_results(self, aggregation_id: str, tenant_id: str, job_id: str) -> dict[str, Any]:
		job = self._require_job(job_id, tenant_id)
		partitions = [item for item in self._partitions.values() if item.tenant_id == tenant_id and item.job_id == job_id]
		completed = [item for item in partitions if item.status == "completed"]
		failed = [item for item in partitions if item.status == "failed"]
		if len(completed) + len(failed) != len(partitions):
			raise PermissionError("partitions_incomplete")
		status = "completed" if not failed else "completed_with_failures"
		aggregation = ResultAggregation(
			id=aggregation_id,
			tenant_id=tenant_id,
			job_id=job_id,
			strategy=job.aggregation_strategy,
			partition_count=len(partitions),
			completed_count=len(completed),
			failed_count=len(failed),
			result_hash=self._engine.result_hash(job_id, [item.to_dict() for item in sorted(partitions, key=lambda item: item.id)]),
			status=status,
		)
		self._aggregations[aggregation_id] = aggregation
		job.status = status
		job.completed_at = datetime.now(timezone.utc)
		self._record_audit(tenant_id, aggregation_id, "results_aggregated", job.owner, status, metadata={"job_id": job_id})
		return aggregation.to_dict()

	def record_scaling_decision(self, decision_id: str, tenant_id: str, pool_id: str, recorded_by: str) -> dict[str, Any]:
		pool = self._require_pool(pool_id, tenant_id)
		queued_partitions = len([
			item for item in self._partitions.values()
			if item.tenant_id == tenant_id and item.status == "queued" and self._jobs[item.job_id].worker_pool_id == pool_id
		])
		active_workers = len([item for item in self._workers.values() if item.tenant_id == tenant_id and item.pool_id == pool_id and item.healthy])
		decision, reason, desired_capacity = self._engine.scaling_posture(queued_partitions, active_workers, pool.capacity_quota)
		scaling = ScalingDecision(
			id=decision_id,
			tenant_id=tenant_id,
			pool_id=pool_id,
			decision=decision,
			reason=reason,
			desired_capacity=desired_capacity,
			current_capacity=active_workers,
			recorded_by=recorded_by,
		)
		self._scaling_decisions[decision_id] = scaling
		self._record_audit(tenant_id, decision_id, "scaling_decision_recorded", recorded_by, decision, metadata={"reason": reason})
		return scaling.to_dict()

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		jobs = self.list_jobs(tenant_id)
		partitions = self.list_partitions(tenant_id)
		return {
			"worker_pool_count": len(self.list_worker_pools(tenant_id)),
			"worker_count": len(self.list_workers(tenant_id)),
			"job_count": len(jobs),
			"queued_job_count": len([item for item in jobs if item["status"] == "queued"]),
			"running_job_count": len([item for item in jobs if item["status"] == "running"]),
			"completed_job_count": len([item for item in jobs if item["status"].startswith("completed")]),
			"pending_review_count": len([item for item in jobs if item["status"] == "pending_review"]),
			"queued_partition_count": len([item for item in partitions if item["status"] == "queued"]),
			"running_partition_count": len([item for item in partitions if item["status"] == "running"]),
			"completed_partition_count": len([item for item in partitions if item["status"] == "completed"]),
			"failed_partition_count": len([item for item in partitions if item["status"] == "failed"]),
		}

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

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_pool(self, pool_id: str, tenant_id: str) -> WorkerPool:
		pool = self._worker_pools.get(pool_id)
		if pool is None or pool.tenant_id != tenant_id:
			raise KeyError("worker_pool_not_found")
		return pool

	def _require_job(self, job_id: str, tenant_id: str) -> DistributedJob:
		job = self._jobs.get(job_id)
		if job is None or job.tenant_id != tenant_id:
			raise KeyError("job_not_found")
		return job

	def _require_partition(self, partition_id: str, tenant_id: str) -> JobPartition:
		partition = self._partitions.get(partition_id)
		if partition is None or partition.tenant_id != tenant_id:
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
		payload = {
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"event_type": event_type,
			"actor": actor,
			"decision": decision,
			"reasons": list(reasons),
			"metadata": dict(metadata or {}),
		}
		event_id = f"audit-{len(self._audit_events) + 1:04d}"
		self._audit_events[event_id] = DistAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			actor=actor,
			decision=decision,
			reasons=reasons,
			payload_hash=self._engine.stable_hash(payload),
		)

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		items = list(records.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]
