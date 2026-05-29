"""API helpers for the Distributed Computing capability."""

from __future__ import annotations

from typing import Any

from .service import DistService


SERVICE = DistService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"worker_pool_count": summary["worker_pool_count"],
		"worker_count": summary["worker_count"],
		"job_count": summary["job_count"],
		"queued_partition_count": summary["queued_partition_count"],
	}


def create_worker_pool(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_worker_pool(
		pool_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload.get("owner") or ""),
		capacity_quota=int(payload.get("capacity_quota") or 0),
		health_check=str(payload.get("health_check") or ""),
		queue_name=str(payload.get("queue_name") or ""),
		autoscaling=bool(payload.get("autoscaling", True)),
	)


def register_worker(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_worker(
		worker_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		pool_id=str(payload["pool_id"]),
		hostname=str(payload["hostname"]),
		cpu_slots=int(payload.get("cpu_slots") or 0),
		memory_gb=float(payload.get("memory_gb") or 0),
		labels={str(key): str(value) for key, value in dict(payload.get("labels") or {}).items()},
		healthy=bool(payload.get("healthy", True)),
	)


def submit_job(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.submit_job(
		job_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload.get("owner") or ""),
		worker_pool_id=str(payload["worker_pool_id"]),
		idempotency_key=str(payload.get("idempotency_key") or ""),
		retry_policy=str(payload.get("retry_policy") or ""),
		partition_count=int(payload.get("partition_count") or 0),
		quota_policy=str(payload.get("quota_policy") or ""),
		event_bus_topic=str(payload.get("event_bus_topic") or ""),
		aggregation_strategy=str(payload.get("aggregation_strategy") or ""),
		partition_review_recorded=bool(payload.get("partition_review_recorded", True)),
	)


def approve_job(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_job(
		job_id=str(payload["job_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
	)


def dispatch_partitions(payload: dict[str, Any]) -> list[dict[str, Any]]:
	return SERVICE.dispatch_partitions(
		job_id=str(payload["job_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
	)


def complete_partition(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.complete_partition(
		partition_id=str(payload["partition_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		result_payload=dict(payload.get("result") or {}),
		status=str(payload.get("status") or "completed"),
	)


def aggregate_results(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.aggregate_results(
		aggregation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		job_id=str(payload["job_id"]),
	)


def record_scaling_decision(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_scaling_decision(
		decision_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		pool_id=str(payload["pool_id"]),
		recorded_by=str(payload["recorded_by"]),
	)


def distributed_state(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"summary": SERVICE.dashboard_summary(tenant_id),
		"worker_pools": SERVICE.list_worker_pools(tenant_id),
		"workers": SERVICE.list_workers(tenant_id),
		"jobs": SERVICE.list_jobs(tenant_id),
		"partitions": SERVICE.list_partitions(tenant_id),
		"aggregations": SERVICE.list_aggregations(tenant_id),
		"scaling_decisions": SERVICE.list_scaling_decisions(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
	}
