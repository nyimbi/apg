"""Executable ESG capability package tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def _build_lifecycle(service):
	profile = service.create_esg_profile("profile-1", "tenant-test", "Acme ESG", "manufacturing", "KE", 2026, "owner-1")
	framework = service.add_framework("framework-1", "tenant-test", profile["id"], "gri", "2026", True, "owner-1")
	metric = service.define_metric("metric-1", "tenant-test", profile["id"], "environmental", "emissions", "tco2e", "Scope 1 emissions", "owner-1")
	measurement = service.record_measurement("measure-1", "tenant-test", metric["id"], "2026-Q1", 125.4, "manual", "doc-1")
	target = service.set_target("target-1", "tenant-test", metric["id"], "reduction", 150, 100, "2030-12-31", "owner-1")
	supplier = service.record_supplier_assessment("supplier-1", "tenant-test", "supplier-1", "2026-Q1", 82, "medium", "doc-2")
	initiative = service.record_initiative("initiative-1", "tenant-test", profile["id"], "Solar rollout", "environmental", 100000, "owner-2", "reduce emissions")
	risk = service.record_risk("risk-1", "tenant-test", profile["id"], "medium", "climate", "transition risk")
	report = service.create_report("report-1", "tenant-test", profile["id"], "quarterly", "2026-Q1", [framework["id"]], [measurement["id"]], "approver-1")
	stakeholder = service.register_stakeholder("stakeholder-1", "tenant-test", profile["id"], "investor", "Investor Group", "email", True)
	engagement = service.record_engagement("engagement-1", "tenant-test", stakeholder["id"], "quarterly ESG update", "email")
	agent = service.register_esg_agent("tenant-test", "Carbon Reviewer", "codex", "carbon_reviewer", "review emissions data")
	return {"profile": profile, "framework": framework, "metric": metric, "measurement": measurement, "target": target, "supplier": supplier, "initiative": initiative, "risk": risk, "report": report, "stakeholder": stakeholder, "engagement": engagement, "agent": agent}


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("contract_esg", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "ecd_esg"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "esg_agents" in contract["provides"]
	assert "/ecd/esg/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_rule_engine_blocks_missing_context_non_bytewax_and_review_gaps():
	module = _load_module("rules_esg", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "esg_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "record_measurement", "review_required": True, "review_recorded": False})["matched_rules"] == ["measurement_review_required"]
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "agent_action", "privileged_action": True, "human_approved": False})["decision"] == "require_review"


def test_service_executes_esg_lifecycle():
	service_module = _load_module("service_esg", PACKAGE_DIR / "service.py")
	service = service_module.ESGManagementLifecycleService()
	records = _build_lifecycle(service)
	summary = service.dashboard_summary("tenant-test")

	assert records["profile"]["industry"] == "manufacturing"
	assert records["framework"]["code"] == "gri"
	assert records["metric"]["unit"] == "tco2e"
	assert records["measurement"]["value"] == 125.4
	assert records["target"]["target_type"] == "reduction"
	assert records["supplier"]["score"] == 82
	assert records["initiative"]["pillar"] == "environmental"
	assert records["report"]["status"] == "approved"
	assert records["agent"]["role"] == "carbon_reviewer"
	assert summary["profile_count"] == 1
	assert summary["audit_event_count"] == 12
	assert summary["streaming"]["processor"] == "bytewax"


def test_service_guardrails_reject_invalid_actions():
	service_module = _load_module("guardrail_service_esg", PACKAGE_DIR / "service.py")
	service = service_module.ESGManagementLifecycleService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_esg_profile("profile", "", "Acme", "manufacturing", "KE", 2026, "owner")
	with pytest.raises(PermissionError, match="esg_profile_name_required"):
		service.create_esg_profile("profile", "tenant-test", "", "manufacturing", "KE", 2026, "owner")
	profile = service.create_esg_profile("profile", "tenant-test", "Acme", "manufacturing", "KE", 2026, "owner")
	with pytest.raises(PermissionError, match="esg_framework_not_supported"):
		service.add_framework("framework", "tenant-test", profile["id"], "unknown", "2026", True, "owner")
	framework = service.add_framework("framework", "tenant-test", profile["id"], "gri", "2026", True, "owner")
	metric = service.define_metric("metric", "tenant-test", profile["id"], "environmental", "emissions", "tco2e", "Emissions", "owner")
	with pytest.raises(PermissionError, match="esg_measurement_review_required"):
		service.record_measurement("measurement", "tenant-test", metric["id"], "2026-Q1", 10, "supplier", "doc")
	measurement = service.record_measurement("measurement", "tenant-test", metric["id"], "2026-Q1", 10, "supplier", "doc", "reviewer")
	with pytest.raises(PermissionError, match="esg_supplier_owner_required"):
		service.record_supplier_assessment("supplier", "tenant-test", "supplier", "2026-Q1", 55, "high", "doc")
	with pytest.raises(PermissionError, match="esg_risk_owner_required"):
		service.record_risk("risk", "tenant-test", profile["id"], "critical", "climate", "transition")
	with pytest.raises(PermissionError, match="esg_report_approval_required"):
		service.create_report("report", "tenant-test", profile["id"], "quarterly", "2026-Q1", [framework["id"]], [measurement["id"]], "")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, "queue")
	with pytest.raises(PermissionError, match="esg_agent_runtime_not_supported"):
		service.register_esg_agent("tenant-test", "Agent", "unsupported", "carbon_reviewer", "review")


def test_agents_batch_api_views_and_app_are_executable():
	api_module = _load_module("api_esg", PACKAGE_DIR / "api.py")
	views = _load_module("views_esg", PACKAGE_DIR / "views.py")
	app = _load_module("app_esg", PACKAGE_DIR / "app.py")

	profile = api_module.create_esg_profile({"tenant_id": "tenant-api", "id": "profile-api", "name": "API ESG", "industry": "finance", "country": "US", "reporting_year": 2026, "owner_id": "owner"})
	agent = api_module.register_esg_agent({"tenant_id": "tenant-api", "name": "Compliance Reviewer", "runtime": "claude_code", "role": "compliance_reviewer"})
	batch = api_module.service().validate_batch("tenant-api", 2)
	model = views.profile_model(api_module.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert profile["id"] == "profile-api"
	assert agent["role"] == "compliance_reviewer"
	assert batch["processor"] == "bytewax"
	assert model["records"][0]["name"] == "API ESG"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["ecd_esg"]["screens"]["agents"]["route"] == "/ecd/esg/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_esg", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["ecd_esg"]["streaming"]["processor"] == "bytewax"
