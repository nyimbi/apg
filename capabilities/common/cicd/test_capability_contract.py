"""Regression coverage for the CICD executable capability contract."""

import pytest

from capabilities.common.cicd import register_capability
from capabilities.common.cicd.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.cicd.service import CicdService
from capabilities.common.cicd.views import analytics_model, audit_trail_model, dashboard_model, delivery_agents_model, settings_model


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-ci", {"pipelines": {"max_parallel_jobs": 25}})

	assert contract["capability"] == "cicd"
	assert contract["configuration"]["tenant_id"] == "tenant-ci"
	assert contract["configuration"]["pipelines"]["max_parallel_jobs"] == 25
	assert contract["configuration_schema"]["required"] == ["tenant_id", "pipelines", "builds", "gates", "delivery_agents", "governance", "observability", "adapters", "ui", "theme"]
	assert contract["configuration"]["delivery_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["theme"]["name"] == "cicd_pipeline_ops"
	assert contract["streaming"]["processor"] == "bytewax"
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"agents", "audit", "analytics"}


def test_rule_engine_enforces_cicd_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_pipeline", "pipeline_owner_assigned": False, "artifact_promotion_requested": True, "artifact_signed": False, "parallel_job_count": 200, "capacity_review_recorded": False})
	build_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "run_build", "secret_scope_attached": False})
	promote_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "promote_artifact", "quality_gate_passed": False})
	agent_result = evaluate_capability_rules({"tenant_context_present": True, "delivery_agent_present": True, "agent_registered": True, "agent_runtime_supported": False, "agent_role_supported": True, "agent_scope_present": True, "agent_contribution_disclosed": True})
	stream_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_pipeline_mutation", "event_stream": "custom"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "pipeline_requires_owner", "artifact_requires_signature", "high_parallelism_requires_review"}
	assert build_result["matched_rules"] == ["build_requires_secret_scope"]
	assert promote_result["matched_rules"] == ["promotion_requires_quality_gate"]
	assert agent_result["matched_rules"] == ["delivery_agent_runtime_supported"]
	assert stream_result["matched_rules"] == ["batch_pipeline_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "cicd"
	assert "depl" in registration["dependencies"]
	assert "bytewax" in registration["optional_dependencies"]
	assert registration["ui_components"]["pipelines"] == "/cicd/pipelines"
	assert registration["ui_components"]["agents"] == "/cicd/agents"
	assert "cicd:run_builds" in registration["permissions"]
	assert "cicd:audit" in registration["permissions"]
	assert registration["streaming"]["processor"] == "bytewax"


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
	agent = service.register_delivery_agent(
		tenant_id="tenant-ci",
		agent_id="codex-release",
		name="Codex Release Reviewer",
		runtime="codex",
		role="security_reviewer",
		scope="Review gates and promotion evidence.",
		contribution_disclosed=True,
		policy_ref="policy:cicd:agents",
	)
	paused = service.change_pipeline_state("tenant-ci", "orders-api", "paused", "Pause before release hardening.")
	resumed = service.change_pipeline_state("tenant-ci", "orders-api", "active", "Resume after release hardening.")
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
	assert agent["runtime"] == "codex"
	assert paused["status"] == "paused"
	assert resumed["status"] == "active"
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
	assert model["summary"]["delivery_agent_count"] == 1
	assert delivery_agents_model(service, "tenant-ci")["agents"][0]["id"] == "codex-release"
	assert audit_trail_model(service, "tenant-ci")["events"]
	assert analytics_model(service, "tenant-ci")["streaming"]["processor"] == "bytewax"
	assert settings_model(service, "tenant-ci")["streaming"]["processor"] == "bytewax"


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

	with pytest.raises(PermissionError, match="delivery_agent_runtime_not_supported"):
		service.register_delivery_agent("tenant-ci", "bad-runtime", "Bad Runtime", "custom", "security_reviewer", "Review release.", True)

	with pytest.raises(PermissionError, match="delivery_agent_role_not_supported"):
		service.register_delivery_agent("tenant-ci", "bad-role", "Bad Role", "codex", "owner", "Review release.", True)

	with pytest.raises(PermissionError, match="delivery_agent_disclosure_required"):
		service.register_delivery_agent("tenant-ci", "undisclosed", "Undisclosed", "codex", "security_reviewer", "Review release.", False)

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

	with pytest.raises(PermissionError, match="cicd_state_change_reason_required"):
		service.change_pipeline_state("tenant-ci", "guarded", "paused", "")

	with pytest.raises(PermissionError, match="cicd_audit_event_required"):
		service.change_pipeline_state("tenant-ci", "guarded", "paused", "Pause without audit.", audit_recorded=False)

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


def test_service_allows_duplicate_pipeline_and_agent_ids_across_tenants():
	service = CicdService()

	for tenant_id in ("tenant-a", "tenant-b"):
		service.create_pipeline(
			pipeline_id="shared",
			tenant_id=tenant_id,
			name="Shared",
			owner="delivery-owner",
			source_ref="git://shared",
			worker_pool="workers",
			stages=["build"],
			secret_scope="scope",
			cache_policy="cache",
			quality_gate="gate",
		)
		service.register_delivery_agent(tenant_id, "shared-agent", "Shared Agent", "codex", "pipeline_designer", f"Design {tenant_id}.", True)

	assert service.list_pipelines("tenant-a")[0]["id"] == "shared"
	assert service.list_pipelines("tenant-b")[0]["id"] == "shared"
	assert service.list_delivery_agents("tenant-a")[0]["id"] == "shared-agent"
	assert service.list_delivery_agents("tenant-b")[0]["id"] == "shared-agent"
