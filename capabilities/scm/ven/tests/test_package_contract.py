"""Executable SCM Vendor Management capability package tests."""

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
	vendor = service.create_vendor("vendor-1", "tenant-test", "ACME", "Acme Supplies", "distributor", "industrial", "KE", "owner-1")
	qualification = service.qualify_vendor("qual-1", "tenant-test", vendor["id"], ["tax", "capacity"], "reviewer-1", 82)
	onboarding = service.onboard_vendor("onboard-1", "tenant-test", vendor["id"], ["profile", "banking"], "owner-1")
	performance = service.record_performance("perf-1", "tenant-test", vendor["id"], "2026-Q2", {"quality": 88, "delivery": 91})
	risk = service.record_risk("risk-1", "tenant-test", vendor["id"], "operational", "medium", "capacity concentration")
	compliance = service.record_compliance("comp-1", "tenant-test", vendor["id"], "tax", "compliant", "doc-1")
	contract = service.create_contract("contract-1", "tenant-test", vendor["id"], 250000, "USD", "2026-01-01", "2026-12-31", "legal-1")
	communication = service.record_communication("comm-1", "tenant-test", vendor["id"], "email", "quarterly review")
	portal = service.create_portal_user("portal-1", "tenant-test", vendor["id"], "vendor@example.com", "account_manager", "owner-1")
	scorecard = service.create_scorecard("score-1", "tenant-test", vendor["id"], "2026-Q2", performance["id"], risk["id"], [compliance["id"]], "analyst-1")
	agent = service.register_vendor_agent("tenant-test", "Vendor Reviewer", "codex", "risk_reviewer", "review vendor risks")
	return {"vendor": vendor, "qualification": qualification, "onboarding": onboarding, "performance": performance, "risk": risk, "compliance": compliance, "contract": contract, "communication": communication, "portal": portal, "scorecard": scorecard, "agent": agent}


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("contract_vendor", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "scm_ven"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "vendor_agents" in contract["provides"]
	assert "/scm/vendors/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_rule_engine_blocks_missing_context_non_bytewax_and_review_gaps():
	module = _load_module("rules_vendor", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "vendor_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "record_performance", "low_score": True, "review_recorded": False})["matched_rules"] == ["performance_low_score_review"]
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "agent_action", "privileged_action": True, "human_approved": False})["decision"] == "require_review"


def test_service_executes_vendor_lifecycle():
	service_module = _load_module("service_vendor", PACKAGE_DIR / "service.py")
	service = service_module.VendorManagementLifecycleService()
	records = _build_lifecycle(service)
	summary = service.dashboard_summary("tenant-test")

	assert records["vendor"]["code"] == "ACME"
	assert records["qualification"]["score"] == 82
	assert records["onboarding"]["status"] == "complete"
	assert records["performance"]["average_score"] == 89.5
	assert records["risk"]["tier"] == "medium"
	assert records["compliance"]["evidence_id"] == "doc-1"
	assert records["contract"]["approved_by"] == "legal-1"
	assert records["portal"]["email"] == "vendor@example.com"
	assert records["scorecard"]["overall_score"] == 89.5
	assert records["agent"]["role"] == "risk_reviewer"
	assert summary["vendor_count"] == 1
	assert summary["audit_event_count"] == 11
	assert summary["streaming"]["processor"] == "bytewax"


def test_service_guardrails_reject_invalid_actions():
	service_module = _load_module("guardrail_service_vendor", PACKAGE_DIR / "service.py")
	service = service_module.VendorManagementLifecycleService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_vendor("vendor", "", "ACME", "Acme", "distributor", "industrial", "KE", "owner")
	with pytest.raises(PermissionError, match="vendor_type_not_supported"):
		service.create_vendor("vendor", "tenant-test", "ACME", "Acme", "supplier", "industrial", "KE", "owner")
	vendor = service.create_vendor("vendor", "tenant-test", "ACME", "Acme", "distributor", "industrial", "KE", "owner")
	with pytest.raises(PermissionError, match="qualification_review_required"):
		service.qualify_vendor("qual", "tenant-test", vendor["id"], ["tax"], "reviewer", 55)
	with pytest.raises(PermissionError, match="performance_review_required"):
		service.record_performance("perf", "tenant-test", vendor["id"], "2026-Q2", {"quality": 50})
	with pytest.raises(PermissionError, match="risk_owner_required"):
		service.record_risk("risk", "tenant-test", vendor["id"], "financial", "critical", "credit concern")
	with pytest.raises(PermissionError, match="compliance_review_required"):
		service.record_compliance("comp", "tenant-test", vendor["id"], "tax", "expired", "doc")
	with pytest.raises(PermissionError, match="contract_approval_required"):
		service.create_contract("contract", "tenant-test", vendor["id"], 100, "USD", "2026-01-01", "2026-12-31", "")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, "queue")
	with pytest.raises(PermissionError, match="vendor_agent_runtime_not_supported"):
		service.register_vendor_agent("tenant-test", "Agent", "unsupported", "risk_reviewer", "review")


def test_agents_batch_api_views_and_app_are_executable():
	api_module = _load_module("api_vendor", PACKAGE_DIR / "api.py")
	views = _load_module("views_vendor", PACKAGE_DIR / "views.py")
	app = _load_module("app_vendor", PACKAGE_DIR / "app.py")

	vendor = api_module.create_vendor({"tenant_id": "tenant-api", "id": "vendor-api", "code": "API", "name": "API Vendor", "vendor_type": "technology", "category": "software", "country": "US", "owner_id": "owner"})
	agent = api_module.register_vendor_agent({"tenant_id": "tenant-api", "name": "Compliance Reviewer", "runtime": "claude_code", "role": "compliance_reviewer"})
	batch = api_module.service().validate_batch("tenant-api", 2)
	model = views.vendor_model(api_module.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert vendor["id"] == "vendor-api"
	assert agent["role"] == "compliance_reviewer"
	assert batch["processor"] == "bytewax"
	assert model["records"][0]["name"] == "API Vendor"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["scm_ven"]["screens"]["agents"]["route"] == "/scm/vendors/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_vendor", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["scm_ven"]["streaming"]["processor"] == "bytewax"
