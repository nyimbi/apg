"""Regression coverage for the ENVM executable capability contract and service."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from capabilities.common.envm import register_capability
from capabilities.common.envm.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.envm.service import EnvmService
from capabilities.common.envm.views import dashboard_model, envm_agent_model


PACKAGE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-env", {"drift": {"drift_threshold_percent": 10}})

	assert contract["capability"] == "envm"
	assert contract["configuration"]["tenant_id"] == "tenant-env"
	assert contract["configuration"]["drift"]["drift_threshold_percent"] == 10
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"environments",
		"promotion",
		"drift",
		"secrets",
		"envm_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["theme"]["name"] == "envm_environment_ops"
	assert contract["provides"] == [
		"environment_inventory",
		"environment_promotion",
		"configuration_drift",
		"secret_scopes",
		"environment_policy",
		"envm_agents",
	]
	assert contract["requires"] == ["auth", "conf", "audl", "depl", "keym", "moni"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["batch_mutation_guardrail"] == "batch_environment_mutation_requires_bytewax"
	assert contract["configuration"]["envm_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"agents", "rules", "audit"}


def test_rule_engine_enforces_environment_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_environment", "environment_owner_assigned": False, "environment": "production", "approval_recorded": False, "secret_scope_present": True, "secret_policy_attached": False, "drift_percent": 15, "drift_review_recorded": False})
	promote_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "promote", "promotion_path_attached": False})
	agent_result = evaluate_capability_rules({"envm_agent_present": True, "agent_runtime_supported": False})
	batch_result = evaluate_capability_rules({"requested_operation": "batch_environment_mutation", "event_stream": "other-stream"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "environment_requires_owner", "production_change_requires_approval", "secret_scope_requires_policy", "high_drift_requires_review"}
	assert promote_result["matched_rules"] == ["promotion_requires_path"]
	assert agent_result["decision"] == "deny"
	assert agent_result["matched_rules"] == ["envm_agent_runtime_supported"]
	assert batch_result["decision"] == "deny"
	assert batch_result["matched_rules"] == ["batch_environment_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "envm"
	assert "depl" in registration["dependencies"]
	assert registration["ui_components"]["promotion"] == "/envm/promotion"
	assert registration["ui_components"]["agents"] == "/envm/agents"
	assert "envm:promote" in registration["permissions"]
	assert registration["streaming"]["processor"] == "bytewax"


def test_environment_promotion_drift_and_secret_scope_lifecycle():
	service = EnvmService()
	dev = service.register_environment(
		environment_id="env-dev",
		tenant_id="tenant-env",
		name="Development",
		stage="development",
		region="ke-nairobi",
		owner="platform",
		configuration_source="git://config/env-dev",
		rbac_policy="rbac-dev",
		secret_scope_policy="secret-dev",
	)
	prod = service.register_environment(
		environment_id="env-prod",
		tenant_id="tenant-env",
		name="Production",
		stage="production",
		region="ke-nairobi",
		owner="operations",
		configuration_source="git://config/env-prod",
		rbac_policy="rbac-prod",
		secret_scope_policy="secret-prod",
		approval_recorded=True,
	)
	path = service.create_promotion_path(
		path_id="path-dev-prod",
		tenant_id="tenant-env",
		source_environment_id="env-dev",
		target_environment_id="env-prod",
		deployment_link="depl:release-42",
		rollback_environment_id="env-dev",
		approval_recorded=True,
	)
	run = service.run_promotion(
		run_id="promotion-1",
		tenant_id="tenant-env",
		promotion_path_id="path-dev-prod",
		requested_by="release-manager",
		artifact_ref="artifact:42",
		approval_recorded=True,
	)
	drift = service.record_drift(
		report_id="drift-1",
		tenant_id="tenant-env",
		environment_id="env-prod",
		declared_version="git:abc",
		observed_version="live:def",
		changed_items=2,
		total_items=100,
	)
	scope = service.register_secret_scope(
		scope_id="scope-prod",
		tenant_id="tenant-env",
		environment_id="env-prod",
		name="prod-db",
		policy_ref="keym-policy-prod",
		secret_refs=("keym://prod/db",),
		access_roles=("envm-admin",),
	)
	agent = service.register_envm_agent(
		tenant_id="tenant-env",
		name="Drift reviewer",
		runtime="codex",
		role="drift_reviewer",
		scope="review drift reports and remediation evidence",
	)
	model = dashboard_model(service, "tenant-env")

	assert dev["fingerprint"] != prod["fingerprint"]
	assert prod["production_locked"] is True
	assert path["status"] == "approved"
	assert run["status"] == "promoted"
	assert drift["status"] == "minor_drift"
	assert scope["policy_ref"] == "keym-policy-prod"
	assert agent["runtime"] == "codex"
	assert agent["role"] == "drift_reviewer"
	assert model["summary"]["environment_count"] == 2
	assert model["summary"]["promotion_run_count"] == 1
	assert model["summary"]["secret_scope_count"] == 1
	assert model["summary"]["envm_agent_count"] == 1
	assert service.validate_batch_environment_mutation("bytewax")["decision"] == "allow"
	assert service.validate_batch_environment_mutation("other-stream")["decision"] == "deny"
	assert len(model["audit_events"]) == 7


def test_environment_guardrails_block_missing_policy_inputs():
	service = EnvmService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_environment(
			environment_id="env-1",
			tenant_id="",
			name="No Tenant",
			stage="development",
			region="ke-nairobi",
			owner="platform",
			configuration_source="git://config",
			rbac_policy="rbac",
			secret_scope_policy="secrets",
		)

	with pytest.raises(PermissionError, match="environment_owner_required"):
		service.register_environment(
			environment_id="env-2",
			tenant_id="tenant-env",
			name="No Owner",
			stage="development",
			region="ke-nairobi",
			owner="",
			configuration_source="git://config",
			rbac_policy="rbac",
			secret_scope_policy="secrets",
		)

	with pytest.raises(PermissionError, match="production_approval_required"):
		service.register_environment(
			environment_id="env-3",
			tenant_id="tenant-env",
			name="Production",
			stage="production",
			region="ke-nairobi",
			owner="operations",
			configuration_source="git://config",
			rbac_policy="rbac",
			secret_scope_policy="secrets",
			approval_recorded=False,
		)

	with pytest.raises(PermissionError, match="stage_policy_required"):
		service.register_environment(
			environment_id="env-4",
			tenant_id="tenant-env",
			name="Unknown",
			stage="sandbox",
			region="ke-nairobi",
			owner="platform",
			configuration_source="git://config",
			rbac_policy="rbac",
			secret_scope_policy="secrets",
		)

	with pytest.raises(PermissionError, match="envm_agent_runtime_not_supported"):
		service.register_envm_agent(
			tenant_id="tenant-env",
			name="Unsupported reviewer",
			runtime="unsupported",
			role="drift_reviewer",
			scope="review drift",
		)


def test_promotion_secret_and_drift_guardrails():
	service = EnvmService()
	service.register_environment("env-dev", "tenant-env", "Development", "development", "ke-nairobi", "platform", "git://dev", "rbac-dev", "secret-dev")
	service.register_environment("env-prod", "tenant-env", "Production", "production", "ke-nairobi", "operations", "git://prod", "rbac-prod", "secret-prod")

	with pytest.raises(PermissionError, match="promotion_path_required"):
		service.create_promotion_path(
			path_id="bad-path",
			tenant_id="tenant-env",
			source_environment_id="env-dev",
			target_environment_id="env-prod",
			deployment_link="depl:1",
			rollback_environment_id="env-dev",
			approval_recorded=True,
			promotion_path_attached=False,
		)

	with pytest.raises(PermissionError, match="production_approval_required"):
		service.create_promotion_path(
			path_id="unapproved-prod",
			tenant_id="tenant-env",
			source_environment_id="env-dev",
			target_environment_id="env-prod",
			deployment_link="depl:1",
			rollback_environment_id="env-dev",
			approval_recorded=False,
		)

	with pytest.raises(PermissionError, match="secret_policy_required"):
		service.register_secret_scope(
			scope_id="bad-scope",
			tenant_id="tenant-env",
			environment_id="env-prod",
			name="bad",
			policy_ref="",
			secret_refs=("keym://prod/db",),
			access_roles=("envm-admin",),
			secret_policy_attached=False,
		)

	drift = service.record_drift(
		report_id="drift-review",
		tenant_id="tenant-env",
		environment_id="env-prod",
		declared_version="git:abc",
		observed_version="live:def",
		changed_items=12,
		total_items=100,
	)

	assert drift["status"] == "review_required"
	assert service.dashboard_summary("tenant-env")["review_required_drift_count"] == 1
	service.register_envm_agent("tenant-env", "Policy reviewer", "codex", "policy_reviewer", "review policy changes")
	assert envm_agent_model(service, "tenant-env")["envm_agents"][0]["role"] == "policy_reviewer"


def test_generated_evidence_and_docs_are_current():
	app = _load_module("envm_app_under_test", PACKAGE_DIR / "app.py")
	model = app.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text(encoding="utf-8"))

	assert app.self_test()["passed"] is True
	assert model == committed_model
	assert model["capabilities"]["envm"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["envm"]["screens"]["agents"]["route"] == "/envm/agents"
	for name in ("README.md", "SPECIFICATION.md", "PLAN.md"):
		assert (PACKAGE_DIR / name).exists()
