"""SCRP package runtime and publish contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.scrp.service import ScrpService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_scrp_contract_shape_is_valid():
	module = _load_module("scrp_contract", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-harvest")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "scrp"
	assert contract["configuration"]["sources"]["terms_evidence_required"] is True
	assert contract["configuration"]["extraction"]["schema_validation_required"] is True
	assert contract["configuration"]["compliance"]["pii_handling_policy_required"] is True
	assert contract["configuration"]["harvest_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["ui"]["routes"]
	assert contract["theme"]["name"] == "scrp_harvest_ops"


def test_scrp_app_entrypoint_is_publishable():
	module = _load_module("scrp_app", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "scrp" in model["capabilities"]
	assert "harvest_requires_schedule_policy" in model["rules"]
	assert model["capabilities"]["scrp"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["scrp"]["screens"]["agents"]["route"] == "/scrp/agents"
	assert model["capabilities"]["scrp"]["theme"]["name"] == "scrp_harvest_ops"


def test_scrp_lifecycle_executes_with_guardrails():
	service = ScrpService()
	tenant_id = "tenant-scrp"

	source = service.register_source(
		tenant_id,
		"orders-api",
		"api",
		"https://example.invalid/orders",
		"data-owner",
		"contract:orders:v1",
		"vault://orders-api",
		60,
		pii_expected=True,
		pii_policy_attached=True,
		tags=["Orders", "API"],
	)
	extractor = service.create_extractor_profile(tenant_id, "orders-json", "json", "data-owner", {"order_id": "str"})
	agent = service.register_harvest_agent(tenant_id, "codex-orders", "Codex Orders Reviewer", "codex", "source_reviewer", "Review orders API terms and extraction evidence.", True, "policy:scrp:agents")
	job = service.create_harvest_job(tenant_id, "orders-hourly", source["id"], extractor["id"], "data-owner", pipeline_target="etlp:orders")
	disabled = service.change_harvest_job_state(tenant_id, job["id"], False, "Pause while validating new source terms.")
	enabled = service.change_harvest_job_state(tenant_id, job["id"], True, "Resume after source terms validation.")
	run = service.run_harvest(tenant_id, job["id"], "scheduler")
	completed = service.complete_harvest_run(tenant_id, run["id"], 25, dlp_scanned=True, storage_ref="memory://orders")
	summary = service.dashboard_summary(tenant_id)

	assert source["tags"] == ["api", "orders"]
	assert agent["runtime"] == "codex"
	assert disabled["enabled"] is False
	assert enabled["enabled"] is True
	assert run["dlp_status"] == "pending"
	assert completed["status"] == "succeeded"
	assert completed["result"]["record_count"] == 25
	assert service.list_handoffs(tenant_id)[0]["pipeline_target"] == "etlp:orders"
	assert service.list_harvest_agents(tenant_id)[0]["id"] == "codex-orders"
	assert summary["source_count"] == 1
	assert summary["succeeded_run_count"] == 1
	assert summary["agent_count"] == 1
	assert summary["audit_event_count"] >= 7
	assert service.audit_events(tenant_id)


def test_scrp_allows_same_agent_id_across_tenants():
	service = ScrpService()

	first = service.register_harvest_agent("tenant-a", "shared-agent", "Shared Agent", "codex", "source_reviewer", "Review tenant A.", True)
	second = service.register_harvest_agent("tenant-b", "shared-agent", "Shared Agent", "codex", "source_reviewer", "Review tenant B.", True)

	assert first["id"] == second["id"] == "shared-agent"
	assert len(service.list_harvest_agents("tenant-a")) == 1
	assert len(service.list_harvest_agents("tenant-b")) == 1


def test_scrp_policy_failures_are_enforced():
	service = ScrpService()
	tenant_id = "tenant-guardrails"

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_source("", "missing-tenant", "api", "https://example.invalid", "owner", "terms", "vault://x", 1)

	with pytest.raises(PermissionError, match="source_owner_required"):
		service.register_source(tenant_id, "missing-owner", "api", "https://example.invalid", "", "terms", "vault://x", 1)

	with pytest.raises(PermissionError, match="source_terms_required"):
		service.register_source(tenant_id, "missing-terms", "api", "https://example.invalid", "owner", "", "vault://x", 1)

	with pytest.raises(PermissionError, match="pii_policy_required"):
		service.register_source(tenant_id, "pii", "api", "https://example.invalid", "owner", "terms", "vault://x", 1, pii_expected=True)

	with pytest.raises(PermissionError, match="sensitive_source_review_required"):
		service.register_source(tenant_id, "sensitive", "api", "https://example.invalid", "owner", "terms", "vault://x", 1, sensitive_source=True)

	source = service.register_source(tenant_id, "safe", "api", "https://example.invalid", "owner", "terms", "vault://x", 1)
	extractor = service.create_extractor_profile(tenant_id, "json", "json", "owner", {"id": "str"})
	job = service.create_harvest_job(tenant_id, "unsafe-schedule", source["id"], extractor["id"], "owner", schedule_policy_attached=False, pipeline_target="etlp:safe")
	with pytest.raises(PermissionError, match="schedule_policy_required"):
		service.run_harvest(tenant_id, job["id"], "owner")

	with pytest.raises(PermissionError, match="harvest_agent_runtime_not_supported"):
		service.register_harvest_agent(tenant_id, "bad-runtime", "Bad Runtime", "custom", "source_reviewer", "Review source.", True)

	with pytest.raises(PermissionError, match="harvest_agent_role_not_supported"):
		service.register_harvest_agent(tenant_id, "bad-role", "Bad Role", "codex", "owner", "Review source.", True)

	with pytest.raises(PermissionError, match="harvest_agent_disclosure_required"):
		service.register_harvest_agent(tenant_id, "undisclosed", "Undisclosed", "codex", "source_reviewer", "Review source.", False)

	with pytest.raises(PermissionError, match="scrp_state_change_reason_required"):
		service.change_harvest_job_state(tenant_id, job["id"], False, "")

	with pytest.raises(PermissionError, match="scrp_audit_event_required"):
		service.change_harvest_job_state(tenant_id, job["id"], False, "Pause without audit.", audit_recorded=False)


def test_scrp_view_models_expose_composable_surfaces():
	from capabilities.common.scrp.views import (
		compliance_review_model,
		audit_trail_model,
		analytics_model,
		dashboard_model,
		extractor_workbench_model,
		harvest_agents_model,
		job_monitor_model,
		pipeline_handoff_model,
		results_model,
		settings_model,
		source_registry_model,
	)

	service = ScrpService()
	tenant_id = "tenant-view"
	source = service.register_source(tenant_id, "feed", "feed", "https://example.invalid/feed", "owner", "terms", "vault://feed", 10)
	extractor = service.create_extractor_profile(tenant_id, "feed-json", "json", "owner", {"id": "str"})
	service.register_harvest_agent(tenant_id, "codex-feed", "Codex Feed", "codex", "source_reviewer", "Review feed terms.", True)
	job = service.create_harvest_job(tenant_id, "feed-job", source["id"], extractor["id"], "owner", pipeline_target="etlp:feed")
	run = service.run_harvest(tenant_id, job["id"], "owner")
	service.complete_harvest_run(tenant_id, run["id"], 2)

	assert dashboard_model(service, tenant_id)["summary"]["source_count"] == 1
	assert source_registry_model(service, tenant_id)["actions"] == ["register_source"]
	assert job_monitor_model(service, tenant_id)["runs"][0]["status"] == "succeeded"
	assert extractor_workbench_model(service, tenant_id)["extractors"][0]["name"] == "feed-json"
	assert pipeline_handoff_model(service, tenant_id)["handoffs"][0]["pipeline_target"] == "etlp:feed"
	assert compliance_review_model(service, tenant_id)["audit_events"]
	assert results_model(service, tenant_id)["results"][0]["record_count"] == 2
	assert harvest_agents_model(service, tenant_id)["agents"][0]["runtime"] == "codex"
	assert audit_trail_model(service, tenant_id)["events"]
	assert analytics_model(service, tenant_id)["streaming"]["processor"] == "bytewax"
	assert settings_model(service, tenant_id)["theme"]["name"] == "scrp_harvest_ops"
	assert settings_model(service, tenant_id)["streaming"]["processor"] == "bytewax"
