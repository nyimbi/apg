"""Regression coverage for the CICD executable capability contract."""

import pytest

from capabilities.common.cicd import register_capability
from capabilities.common.cicd.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.cicd.service import CicdService
from capabilities.common.cicd.views import dashboard_model


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


def test_service_runs_pipeline_build_artifact_gate_and_promotion():
	service = CicdService()
	pipeline = service.create_pipeline(
		pipeline_id="orders-api",
		tenant_id="tenant-ci",
		name="Orders API",
		owner="delivery-owner",
		source_ref="git://orders-api",
		worker_pool="python-workers",
		stages=["build", "test", "scan", "package"],
		secret_scope="orders-ci",
		cache_policy="python-cache",
		quality_gate="default-release",
	)
	build = service.run_build(
		build_id="build-1",
		tenant_id="tenant-ci",
		pipeline_id="orders-api",
		commit_ref="abc123",
		triggered_by="developer",
	)
	artifact = service.publish_artifact(
		artifact_id="artifact-1",
		tenant_id="tenant-ci",
		build_id="build-1",
		name="orders-api",
		version="1.0.0",
		signed=True,
	)
	gate = service.record_quality_gate(
		gate_id="gate-1",
		tenant_id="tenant-ci",
		artifact_id="artifact-1",
		tests_passed=True,
		security_scan_passed=True,
		approval_recorded=True,
	)
	promotion = service.promote_artifact(
		promotion_id="promotion-1",
		tenant_id="tenant-ci",
		artifact_id="artifact-1",
		quality_gate_id="gate-1",
		source_environment="staging",
		target_environment="production",
		requested_by="release-manager",
		approval_recorded=True,
	)
	model = dashboard_model(service, "tenant-ci")

	assert pipeline["status"] == "active"
	assert build["status"] == "passed"
	assert build["trace_id"].startswith("trace-")
	assert len(artifact["digest"]) == 64
	assert artifact["signed"] is True
	assert gate["status"] == "passed"
	assert gate["findings"] == []
	assert promotion["status"] == "promoted"
	assert model["summary"]["pipeline_count"] == 1
	assert model["summary"]["build_count"] == 1
	assert model["summary"]["promotion_count"] == 1


def test_service_enforces_pipeline_build_and_promotion_guardrails():
	service = CicdService()

	with pytest.raises(PermissionError, match="pipeline_owner_required"):
		service.create_pipeline(
			pipeline_id="missing-owner",
			tenant_id="tenant-ci",
			name="Missing Owner",
			owner="",
			source_ref="git://repo",
			worker_pool="workers",
			stages=["build"],
			secret_scope="scope",
			cache_policy="cache",
			quality_gate="gate",
		)

	service.create_pipeline(
		pipeline_id="guarded",
		tenant_id="tenant-ci",
		name="Guarded",
		owner="delivery-owner",
		source_ref="git://repo",
		worker_pool="workers",
		stages=["build"],
		secret_scope="scope",
		cache_policy="cache",
		quality_gate="gate",
	)

	with pytest.raises(PermissionError, match="secret_scope_required"):
		service.run_build(
			build_id="bad-build",
			tenant_id="tenant-ci",
			pipeline_id="guarded",
			commit_ref="abc123",
			triggered_by="developer",
			secret_scope_attached=False,
		)

	service.run_build(
		build_id="build-1",
		tenant_id="tenant-ci",
		pipeline_id="guarded",
		commit_ref="abc123",
		triggered_by="developer",
	)
	unsigned = service.publish_artifact(
		artifact_id="unsigned",
		tenant_id="tenant-ci",
		build_id="build-1",
		name="guarded",
		version="1.0.0",
		signed=False,
	)
	failed_gate = service.record_quality_gate(
		gate_id="failed-gate",
		tenant_id="tenant-ci",
		artifact_id="unsigned",
		tests_passed=True,
		security_scan_passed=True,
		approval_recorded=True,
	)

	with pytest.raises(PermissionError, match="artifact_signature_required"):
		service.promote_artifact(
			promotion_id="bad-promotion",
			tenant_id="tenant-ci",
			artifact_id="unsigned",
			quality_gate_id="failed-gate",
			source_environment="staging",
			target_environment="production",
			requested_by="release-manager",
			approval_recorded=True,
		)

	assert unsigned["signed"] is False
	assert failed_gate["status"] == "failed"
	assert "artifact signature missing" in failed_gate["findings"]


def test_service_routes_high_parallelism_pipeline_to_review():
	service = CicdService()
	pipeline = service.create_pipeline(
		pipeline_id="wide-pipeline",
		tenant_id="tenant-ci",
		name="Wide Pipeline",
		owner="delivery-owner",
		source_ref="git://wide",
		worker_pool="large-workers",
		stages=["build"],
		secret_scope="scope",
		cache_policy="cache",
		quality_gate="gate",
		parallel_job_count=250,
		capacity_review_recorded=False,
	)
	approved = service.approve_pipeline("wide-pipeline", reviewer="capacity-reviewer")

	assert pipeline["status"] == "pending_review"
	assert pipeline["review_status"] == "required"
	assert approved["status"] == "active"
	assert service.pipeline_summary("tenant-ci")["active_pipeline_count"] == 1
