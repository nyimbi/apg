"""Regression coverage for the CICD executable capability contract."""

from capabilities.common.cicd import register_capability
from capabilities.common.cicd.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-ci", {"pipelines": {"max_parallel_jobs": 25}})

	assert contract["capability"] == "cicd"
	assert contract["configuration"]["tenant_id"] == "tenant-ci"
	assert contract["configuration"]["pipelines"]["max_parallel_jobs"] == 25
	assert contract["configuration_schema"]["required"] == ["tenant_id", "pipelines", "builds", "gates", "governance", "ui", "theme"]
	assert contract["theme"]["name"] == "cicd_pipeline_ops"


def test_rule_engine_enforces_cicd_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_pipeline", "pipeline_owner_assigned": False, "artifact_promotion_requested": True, "artifact_signed": False, "parallel_job_count": 200, "capacity_review_recorded": False})
	build_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "run_build", "secret_scope_attached": False})
	promote_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "promote_artifact", "quality_gate_passed": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "pipeline_requires_owner", "artifact_requires_signature", "high_parallelism_requires_review"}
	assert build_result["matched_rules"] == ["build_requires_secret_scope"]
	assert promote_result["matched_rules"] == ["promotion_requires_quality_gate"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "cicd"
	assert "depl" in registration["dependencies"]
	assert registration["ui_components"]["pipelines"] == "/cicd/pipelines"
	assert "cicd:run_builds" in registration["permissions"]
