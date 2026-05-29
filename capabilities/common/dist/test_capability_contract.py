"""Regression coverage for the DIST executable capability contract."""

import pytest

from capabilities.common.dist import register_capability
from capabilities.common.dist.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.dist.service import DistService


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-dist", {"jobs": {"max_partitions": 100}})

	assert contract["capability"] == "dist"
	assert contract["configuration"]["tenant_id"] == "tenant-dist"
	assert contract["configuration"]["jobs"]["max_partitions"] == 100
	assert contract["configuration_schema"]["required"] == ["tenant_id", "jobs", "workers", "coordination", "governance", "ui", "theme"]
	assert contract["theme"]["name"] == "dist_compute_grid"


def test_rule_engine_enforces_distributed_compute_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "submit_job", "job_owner_assigned": False, "idempotency_key_present": False, "worker_pool_selected": True, "health_check_attached": False, "quota_policy_attached": False, "job_submission_requested": True, "partition_count": 2000, "partition_review_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "job_requires_owner", "idempotency_key_required", "worker_pool_requires_health", "quota_policy_required", "large_partition_job_requires_review"}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "dist"
	assert "mqeb" in registration["dependencies"]
	assert registration["ui_components"]["workers"] == "/dist/workers"
	assert "dist:submit_jobs" in registration["permissions"]


def test_service_runs_partitioned_job_lifecycle_with_aggregation_and_scaling():
	service = DistService()

	pool = service.create_worker_pool(
		"pool-main",
		"tenant-dist",
		"Main compute pool",
		"compute-owner",
		capacity_quota=4,
		health_check="worker-heartbeat",
		queue_name="dist.jobs",
	)
	worker_a = service.register_worker(
		"worker-a",
		"tenant-dist",
		"pool-main",
		"worker-a.internal",
		cpu_slots=4,
		memory_gb=16,
		labels={"zone": "a"},
	)
	service.register_worker(
		"worker-b",
		"tenant-dist",
		"pool-main",
		"worker-b.internal",
		cpu_slots=4,
		memory_gb=16,
		labels={"zone": "b"},
	)
	job = service.submit_job(
		"job-001",
		"tenant-dist",
		"Reprice portfolio",
		"risk-owner",
		"pool-main",
		"idempotency-001",
		"retry-3-exponential",
		partition_count=4,
		quota_policy="tenant-quota-standard",
		event_bus_topic="dist.jobs",
		aggregation_strategy="merge_hashes",
	)
	dispatched = service.dispatch_partitions("job-001", "tenant-dist")
	for partition in dispatched:
		service.complete_partition(partition["id"], "tenant-dist", {"ok": True, "partition": partition["ordinal"]})
	aggregation = service.aggregate_results("agg-001", "tenant-dist", "job-001")
	scaling = service.record_scaling_decision("scale-001", "tenant-dist", "pool-main", "autoscaler")
	summary = service.dashboard_summary("tenant-dist")

	assert pool["capacity_quota"] == 4
	assert worker_a["healthy"] is True
	assert job["status"] == "queued"
	assert len(dispatched) == 4
	assert {item["assigned_worker_id"] for item in dispatched} == {"worker-a", "worker-b"}
	assert aggregation["status"] == "completed"
	assert aggregation["completed_count"] == 4
	assert aggregation["result_hash"]
	assert scaling["decision"] in {"hold", "scale_down"}
	assert summary["completed_job_count"] == 1
	assert summary["completed_partition_count"] == 4
	assert len(service.list_audit_events("tenant-dist")) >= 8


def test_service_enforces_distributed_compute_guardrails():
	service = DistService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_worker_pool("pool-no-tenant", "", "Pool", "owner", 4, "heartbeat", "queue")

	with pytest.raises(PermissionError, match="capacity_quota_required"):
		service.create_worker_pool("pool-no-quota", "tenant-dist", "Pool", "owner", 0, "heartbeat", "queue")

	with pytest.raises(PermissionError, match="worker_health_required"):
		service.create_worker_pool("pool-no-health", "tenant-dist", "Pool", "owner", 4, "", "queue")

	service.create_worker_pool("pool-main", "tenant-dist", "Pool", "owner", 4, "heartbeat", "queue")

	with pytest.raises(PermissionError, match="worker_cpu_slots_required"):
		service.register_worker("worker-bad", "tenant-dist", "pool-main", "worker.internal", 0, 16)

	with pytest.raises(PermissionError, match="job_owner_required"):
		service.submit_job(
			"job-no-owner",
			"tenant-dist",
			"No owner",
			"",
			"pool-main",
			"idempotency-001",
			"retry",
			partition_count=1,
			quota_policy="quota",
			event_bus_topic="topic",
			aggregation_strategy="merge",
		)

	with pytest.raises(PermissionError, match="idempotency_key_required"):
		service.submit_job(
			"job-no-idempotency",
			"tenant-dist",
			"No key",
			"owner",
			"pool-main",
			"",
			"retry",
			partition_count=1,
			quota_policy="quota",
			event_bus_topic="topic",
			aggregation_strategy="merge",
		)

	with pytest.raises(PermissionError, match="quota_policy_required"):
		service.submit_job(
			"job-no-quota",
			"tenant-dist",
			"No quota",
			"owner",
			"pool-main",
			"idempotency-002",
			"retry",
			partition_count=1,
			quota_policy="",
			event_bus_topic="topic",
			aggregation_strategy="merge",
		)

	service.submit_job(
		"job-no-worker",
		"tenant-dist",
		"No worker",
		"owner",
		"pool-main",
		"idempotency-003",
		"retry",
		partition_count=1,
		quota_policy="quota",
		event_bus_topic="topic",
		aggregation_strategy="merge",
	)
	with pytest.raises(PermissionError, match="healthy_worker_required"):
		service.dispatch_partitions("job-no-worker", "tenant-dist")


def test_large_partition_job_requires_review_and_idempotency_is_stable():
	service = DistService()
	service.create_worker_pool("pool-main", "tenant-dist", "Pool", "owner", 8, "heartbeat", "queue")
	service.register_worker("worker-a", "tenant-dist", "pool-main", "worker.internal", 4, 16)

	job = service.submit_job(
		"job-large",
		"tenant-dist",
		"Large fanout",
		"owner",
		"pool-main",
		"idempotency-large",
		"retry",
		partition_count=1001,
		quota_policy="quota",
		event_bus_topic="topic",
		aggregation_strategy="merge",
		partition_review_recorded=False,
	)
	duplicate = service.submit_job(
		"job-large-duplicate",
		"tenant-dist",
		"Large fanout duplicate",
		"owner",
		"pool-main",
		"idempotency-large",
		"retry",
		partition_count=1001,
		quota_policy="quota",
		event_bus_topic="topic",
		aggregation_strategy="merge",
	)

	assert job["status"] == "pending_review"
	assert job["review_status"] == "required"
	assert duplicate["id"] == "job-large"
	with pytest.raises(PermissionError, match="job_review_required"):
		service.dispatch_partitions("job-large", "tenant-dist")
	approved = service.approve_job("job-large", "tenant-dist", "capacity-reviewer")

	assert approved["status"] == "queued"
	assert service.dashboard_summary("tenant-dist")["pending_review_count"] == 0
