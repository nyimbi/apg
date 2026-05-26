"""Regression coverage for the DIST executable capability contract."""

from capabilities.common.dist import register_capability
from capabilities.common.dist.capability_contract import evaluate_capability_rules, get_capability_contract


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
