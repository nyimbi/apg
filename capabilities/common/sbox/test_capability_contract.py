"""Regression coverage for the SBOX executable capability contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from capabilities.common.sbox import register_capability
from capabilities.common.sbox.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.sbox.service import SboxService
from capabilities.common.sbox import views


PACKAGE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_exposes_configuration_rules_ui_theme_and_streaming():
	contract = get_capability_contract("tenant-sbox", {"sandboxes": {"ttl_hours": 12}})

	assert contract["capability"] == "sbox"
	assert contract["configuration"]["tenant_id"] == "tenant-sbox"
	assert contract["configuration"]["sandboxes"]["ttl_hours"] == 12
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"sandboxes",
		"isolation",
		"datasets",
		"sbox_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["provides"] == [
		"sandbox_registry",
		"isolation_profiles",
		"test_runs",
		"synthetic_datasets",
		"safety_policy",
		"sbox_agents",
	]
	assert contract["requires"] == ["plgn", "secu", "envm", "audl"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["configuration"]["sbox_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "sandboxes", "templates", "datasets", "runs", "agents", "policies", "audit", "logs", "settings"}
	assert contract["theme"]["name"] == "sbox_safe_testing"


def test_rule_engine_enforces_sbox_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_sandbox",
		"sandbox_owner_assigned": False,
		"template_present": False,
		"isolation_profile_attached": False,
		"secret_access_requested": True,
		"secret_redaction_enabled": False,
		"ttl_hours": 72,
		"lifecycle_review_recorded": False,
	})
	dataset_result = evaluate_capability_rules({
		"operation": "register_dataset",
		"dataset_owner_assigned": False,
		"dataset_lineage_present": False,
		"retention_days": 0,
		"production_dataset": True,
		"production_review_recorded": False,
		"sensitive_dataset": True,
		"dataset_masked": False,
	})
	run_result = evaluate_capability_rules({
		"operation": "start_run",
		"run_requester_present": False,
		"tests_requested": 0,
		"plugin_run": True,
		"plugin_test_policy_present": False,
		"event_stream": "other-stream",
	})
	agent_result = evaluate_capability_rules({"sbox_agent_present": True, "agent_runtime_supported": False})
	batch_result = evaluate_capability_rules({"requested_operation": "batch_sandbox_mutation", "event_stream": "other-stream"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"sandbox_requires_owner",
		"sandbox_requires_template",
		"sandbox_requires_isolation_profile",
		"secrets_require_redaction",
		"long_lived_sandbox_requires_review",
	}
	assert set(dataset_result["matched_rules"]) == {"dataset_requires_owner", "dataset_requires_lineage", "dataset_requires_retention", "production_dataset_requires_review", "sensitive_dataset_requires_masking"}
	assert set(run_result["matched_rules"]) == {"run_requires_requester", "run_requires_test_count", "plugin_run_requires_policy", "run_requires_bytewax_stream"}
	assert agent_result["matched_rules"] == ["sbox_agent_runtime_supported"]
	assert batch_result["matched_rules"] == ["batch_sandbox_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "sbox"
	assert "plgn" in registration["dependencies"]
	assert registration["ui_components"]["runs"] == "/sbox/runs"
	assert registration["ui_components"]["agents"] == "/sbox/agents"
	assert registration["streaming"]["processor"] == "bytewax"
	assert "sbox:run_tests" in registration["permissions"]


def test_sandbox_lifecycle_runs_and_agents():
	service = SboxService()

	isolation = service.create_isolation_profile(
		tenant_id="tenant-sbox",
		name="strict-network",
		level="strict",
		approved_by="security-reviewer",
	)
	template = service.create_template(
		tenant_id="tenant-sbox",
		name="python-plugin-tests",
		runtime="python",
		owner="platform-owner",
		default_ttl_hours=24,
	)
	dataset = service.register_dataset(
		tenant_id="tenant-sbox",
		name="safe-fixture",
		dataset_type="fixture",
		owner="qa-owner",
		lineage="fixture://safe-fixture",
		retention_days=30,
	)
	sandbox = service.create_sandbox(
		tenant_id="tenant-sbox",
		name="plugin-check",
		template_id=template["id"],
		isolation_profile_id=isolation["id"],
		owner="qa-owner",
		dataset_ids=[dataset["id"]],
	)
	run = service.start_run(
		tenant_id="tenant-sbox",
		sandbox_id=sandbox["id"],
		run_type="plugin",
		requested_by="qa-owner",
		tests_requested=12,
		event_stream="bytewax://sandbox-runs",
	)
	completed = service.complete_run("tenant-sbox", run["id"], tests_passed=12)
	agent = service.register_sbox_agent(
		tenant_id="tenant-sbox",
		name="Isolation reviewer",
		runtime="codex",
		role="isolation_reviewer",
		scope="review network, data, secret, TTL, and stream guardrails",
	)
	summary = service.dashboard_summary("tenant-sbox")

	assert isolation["level"] == "strict"
	assert template["runtime"] == "python"
	assert dataset["dataset_type"] == "fixture"
	assert sandbox["state"] == "ready"
	assert run["status"] == "running"
	assert completed["status"] == "passed"
	assert agent["role"] == "isolation_reviewer"
	assert summary["sandbox_count"] == 1
	assert summary["passed_run_count"] == 1
	assert summary["sbox_agent_count"] == 1
	assert summary["streaming"]["processor"] == "bytewax"
	assert service.validate_batch_sandbox_mutation("bytewax")["decision"] == "allow"
	assert service.validate_batch_sandbox_mutation("other-stream")["decision"] == "deny"
	assert views.dashboard_model(service, "tenant-sbox")["streaming"]["processor"] == "bytewax"
	assert views.sandbox_console_model(service, "tenant-sbox")["sandboxes"][0]["id"] == sandbox["id"]
	assert views.run_monitor_model(service, "tenant-sbox")["runs"][0]["status"] == "passed"
	assert views.sbox_agent_model(service, "tenant-sbox")["sbox_agents"][0]["runtime"] == "codex"
	assert views.audit_trail_model(service, "tenant-sbox")["audit_events"]
	assert views.sandbox_policy_model(service, "tenant-sbox")["streaming"]["processor"] == "bytewax"


def test_sandbox_guardrails_block_unsafe_operations():
	service = SboxService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_isolation_profile("", "no-tenant")

	with pytest.raises(PermissionError, match="outbound_network_approval_required"):
		service.create_isolation_profile("tenant-sbox", "open", outbound_network_allowed=True)

	with pytest.raises(PermissionError, match="secret_redaction_required"):
		service.create_isolation_profile("tenant-sbox", "unsafe", secret_redaction_enabled=False)

	isolation = service.create_isolation_profile("tenant-sbox", "strict", approved_by="reviewer")

	with pytest.raises(PermissionError, match="sandbox_owner_required"):
		service.create_template("tenant-sbox", "template", "python", "")

	template = service.create_template("tenant-sbox", "template", "python", "owner", plugin_test_policy_required=False)

	with pytest.raises(PermissionError, match="dataset_owner_required"):
		service.register_dataset("tenant-sbox", "data", "fixture", "", "lineage", 30)

	with pytest.raises(PermissionError, match="dataset_lineage_required"):
		service.register_dataset("tenant-sbox", "data", "fixture", "owner", "", 30)

	with pytest.raises(PermissionError, match="retention_policy_required"):
		service.register_dataset("tenant-sbox", "data", "fixture", "owner", "lineage", 0)

	with pytest.raises(PermissionError, match="production_data_review_required"):
		service.register_dataset("tenant-sbox", "prod", "production_sample", "owner", "lineage", 30)

	with pytest.raises(PermissionError, match="dataset_masking_required"):
		service.register_dataset("tenant-sbox", "prod", "production_sample", "owner", "lineage", 30, production_review_recorded=True, masked=False)

	sandbox = service.create_sandbox("tenant-sbox", "sandbox", template["id"], isolation["id"], "owner")

	with pytest.raises(PermissionError, match="plugin_test_policy_required"):
		service.start_run("tenant-sbox", sandbox["id"], "plugin", "owner", 1)

	with pytest.raises(PermissionError, match="run_requester_required"):
		service.start_run("tenant-sbox", sandbox["id"], "integration", "", 1)

	with pytest.raises(PermissionError, match="tests_requested_required"):
		service.start_run("tenant-sbox", sandbox["id"], "integration", "owner", 0)

	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.start_run("tenant-sbox", sandbox["id"], "integration", "owner", 1, event_stream="other-stream")

	with pytest.raises(PermissionError, match="sbox_agent_runtime_not_supported"):
		service.register_sbox_agent("tenant-sbox", "Unsupported", "unsupported", "run_reviewer", "review")

	with pytest.raises(PermissionError, match="sbox_agent_scope_required"):
		service.register_sbox_agent("tenant-sbox", "No scope", "codex", "run_reviewer", "")


def test_lifecycle_ids_are_tenant_scoped():
	service = SboxService()

	for tenant_id, owner in (("tenant-a", "owner-a"), ("tenant-b", "owner-b")):
		isolation = service.create_isolation_profile(tenant_id, "shared-isolation", approved_by=owner)
		template = service.create_template(tenant_id, "shared-template", "python", owner)
		service.create_sandbox(tenant_id, "shared-sandbox", template["id"], isolation["id"], owner)
		service.register_sbox_agent(tenant_id, "Reviewer", "codex", "run_reviewer", "review tenant sandbox runs", agent_id="shared-agent")

	assert service.list_templates("tenant-a")[0]["owner"] == "owner-a"
	assert service.list_templates("tenant-b")[0]["owner"] == "owner-b"
	assert service.list_sandboxes("tenant-a")[0]["owner"] == "owner-a"
	assert service.list_sandboxes("tenant-b")[0]["owner"] == "owner-b"
	assert service.list_sbox_agents("tenant-a")[0]["id"] == "shared-agent"
	assert service.list_sbox_agents("tenant-b")[0]["id"] == "shared-agent"


def test_generated_evidence_and_docs_are_current():
	app = _load_module("sbox_app_under_test", PACKAGE_DIR / "app.py")
	model = app.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text(encoding="utf-8"))

	assert app.self_test()["passed"] is True
	assert model == committed_model
	assert model["capabilities"]["sbox"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["sbox"]["screens"]["agents"]["route"] == "/sbox/agents"
	for name in ("README.md", "SPECIFICATION.md", "PLAN.md"):
		assert (PACKAGE_DIR / name).exists()
