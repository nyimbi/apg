"""Regression coverage for the DIST executable capability contract."""

import pytest

from capabilities.common.dist import register_capability
from capabilities.common.dist.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.dist.service import DistService
from capabilities.common.dist.views import analytics_model, audit_trail_model, compute_agents_model, dashboard_model, settings_model


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-dist", {"jobs": {"max_partitions": 100}})

	assert contract["capability"] == "dist"
	assert contract["configuration"]["tenant_id"] == "tenant-dist"
	assert contract["configuration"]["jobs"]["max_partitions"] == 100
	assert contract["configuration_schema"]["required"] == ["tenant_id", "jobs", "workers", "coordination", "compute_agents", "governance", "observability", "adapters", "ui", "theme"]
	assert contract["configuration"]["compute_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["topic"] == "apg.dist.lifecycle"
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
	assert registration["ui_components"]["agents"] == "/dist/agents"
	assert registration["streaming"]["processor"] == "bytewax"
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
	agent = service.register_compute_agent(
		"tenant-dist",
		"codex-partition-reviewer",
		"Codex Partition Reviewer",
		"codex",
		"result_reviewer",
		"Review partition completion evidence and aggregation readiness.",
		True,
		"policy:dist:agents:v1",
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
	views = dashboard_model(service, "tenant-dist")
	agents = compute_agents_model(service, "tenant-dist")
	analytics = analytics_model(service, "tenant-dist")
	audit = audit_trail_model(service, "tenant-dist")
	settings = settings_model("tenant-dist")

	assert pool["capacity_quota"] == 4
	assert worker_a["healthy"] is True
	assert agent["runtime"] == "codex"
	assert agent["role"] == "result_reviewer"
	assert job["status"] == "queued"
	assert len(dispatched) == 4
	assert {item["assigned_worker_id"] for item in dispatched} == {"worker-a", "worker-b"}
	assert aggregation["status"] == "completed"
	assert aggregation["completed_count"] == 4
	assert aggregation["result_hash"]
	assert scaling["decision"] in {"hold", "scale_down"}
	assert summary["completed_job_count"] == 1
	assert summary["completed_partition_count"] == 4
	assert summary["compute_agent_count"] == 1
	assert len(service.list_audit_events("tenant-dist")) >= 8
	assert views["compute_agents"][0]["id"] == "codex-partition-reviewer"
	assert agents["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert analytics["signals"]["completion_rate"] == 1.0
	assert audit["guardrails"]
	assert settings["streaming"]["processor"] == "bytewax"


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

	with pytest.raises(PermissionError, match="retry_policy_required"):
		service.submit_job(
			"job-no-retry",
			"tenant-dist",
			"No retry",
			"owner",
			"pool-main",
			"idempotency-004",
			"",
			partition_count=1,
			quota_policy="quota",
			event_bus_topic="topic",
			aggregation_strategy="merge",
		)

	with pytest.raises(PermissionError, match="partition_count_required"):
		service.submit_job(
			"job-no-partitions",
			"tenant-dist",
			"No partitions",
			"owner",
			"pool-main",
			"idempotency-005",
			"retry",
			partition_count=0,
			quota_policy="quota",
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


def test_compute_agents_state_changes_bytewax_and_tenant_scope():
	service = DistService()
	for tenant_id in ("tenant-a", "tenant-b"):
		service.create_worker_pool("shared-pool", tenant_id, "Shared Pool", "owner", 4, "heartbeat", "queue")
		service.register_worker("shared-worker", tenant_id, "shared-pool", f"{tenant_id}.worker", 2, 8)
		service.submit_job(
			"shared-job",
			tenant_id,
			"Shared fanout",
			"owner",
			"shared-pool",
			"shared-idempotency",
			"retry",
			partition_count=2,
			quota_policy="quota",
			event_bus_topic="bytewax",
			aggregation_strategy="merge",
		)
		service.register_compute_agent(
			tenant_id,
			"shared-agent",
			"Shared Agent",
			"codex",
			"partition_operator",
			f"Operate partitions for {tenant_id}.",
			True,
		)

	assert len(service.list_compute_agents("tenant-a")) == 1
	assert len(service.list_compute_agents("tenant-b")) == 1
	assert service.list_jobs("tenant-a")[0]["tenant_id"] == "tenant-a"

	paused = service.change_job_state("tenant-a", "shared-job", "paused", "Pause for capacity window.", "owner")
	assert paused["status"] == "paused"
	assert service.validate_batch_compute_mutation("tenant-a", "bytewax", "owner")["processor"] == "bytewax"

	with pytest.raises(PermissionError, match="dist_state_change_reason_required"):
		service.change_job_state("tenant-a", "shared-job", "paused", "", "owner")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch_compute_mutation("tenant-a", "custom-stream", "owner")
	with pytest.raises(PermissionError, match="compute_agent_runtime_not_supported"):
		service.register_compute_agent("tenant-a", "bad-runtime", "Bad Runtime", "custom", "partition_operator", "Operate.", True)
	with pytest.raises(PermissionError, match="compute_agent_role_not_supported"):
		service.register_compute_agent("tenant-a", "bad-role", "Bad Role", "codex", "owner", "Operate.", True)
	with pytest.raises(PermissionError, match="compute_agent_disclosure_required"):
		service.register_compute_agent("tenant-a", "undisclosed", "Undisclosed", "codex", "partition_operator", "Operate.", False)
