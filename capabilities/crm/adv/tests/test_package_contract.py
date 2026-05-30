"""Advanced CRM analytics package contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "tested_crm_adv"


def _load_module(name: str):
	if PACKAGE_NAME not in sys.modules:
		package = types.ModuleType(PACKAGE_NAME)
		package.__path__ = [str(PACKAGE_DIR)]
		sys.modules[PACKAGE_NAME] = package
	spec = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.{name}", PACKAGE_DIR / f"{name}.py")
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("capability_contract")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "crm_adv"
	assert "crm_agents" in contract["provides"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert any(route["path"] == "/crm-adv/pipeline" for route in contract["ui"]["routes"])
	assert any(route["path"] == "/crm-adv/agents" for route in contract["ui"]["routes"])


def test_rule_engine_blocks_missing_context_and_non_bytewax_import():
	module = _load_module("capability_contract")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	bad_stream = module.evaluate_capability_rules({"tenant_context_present": True, "operation": "crm_batch_import", "event_stream": "other"})

	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]
	assert bad_stream["decision"] == "deny"
	assert "crm_batch_import_requires_bytewax" in bad_stream["matched_rules"]


def test_account_contact_lead_pipeline_activity_campaign_and_forecast_lifecycle():
	service_module = _load_module("service")
	service = service_module.AdvancedCRMService()

	account = service.create_account("acme", "tenant-test", "Acme", "owner-1", "enterprise", "north")
	contact = service.create_contact("jane", "tenant-test", "acme", "Jane Buyer", "jane@example.com", True, True)
	lead = service.create_lead("lead-1", "tenant-test", "Acme Expansion", "web", 82)
	assigned = service.assign_lead("tenant-test", lead["id"], "seller-1", "round_robin")
	opportunity = service.create_opportunity("opp-1", "tenant-test", "acme", "Expansion", "qualification", 25000, "2026-12-31")
	activity = service.record_activity("act-1", "tenant-test", opportunity["id"], "seller-1", "Discovery call", "Send proposal")
	campaign = service.launch_campaign("camp-1", "tenant-test", "Expansion Campaign", ["jane"], "consent-list-1", 5000)
	forecast = service.record_forecast("fc-1", "tenant-test", "2026-Q4", 25000, 0.82, "pipeline-evidence")

	assert account["status"] == "active"
	assert contact["consent_recorded"] is True
	assert assigned["status"] == "assigned"
	assert opportunity["status"] == "open"
	assert activity["next_step"] == "Send proposal"
	assert campaign["status"] == "active"
	assert forecast["confidence"] == 0.82
	assert service.dashboard_summary("tenant-test")["audit_event_count"] >= 8


def test_service_enforces_crm_guardrails():
	service_module = _load_module("service")
	service = service_module.AdvancedCRMService()

	try:
		service.create_account("bad", "", "Bad", "owner", "enterprise")
	except PermissionError as exc:
		assert "tenant_context_required" in str(exc)
	else:
		raise AssertionError("expected tenant guardrail")

	try:
		service.create_account("bad", "tenant-test", "Bad", "", "enterprise")
	except PermissionError as exc:
		assert "account_requires_owner" in str(exc)
	else:
		raise AssertionError("expected owner guardrail")

	service.create_account("acme", "tenant-test", "Acme", "owner", "enterprise")
	try:
		service.create_contact("jane", "tenant-test", "acme", "Jane", "jane@example.com", True, False)
	except PermissionError as exc:
		assert "contact_outreach_requires_consent" in str(exc)
	else:
		raise AssertionError("expected consent guardrail")

	lead = service.create_lead("lead", "tenant-test", "Lead", "web")
	try:
		service.assign_lead("tenant-test", lead["id"], "seller", "round_robin")
	except PermissionError as exc:
		assert "lead_assignment_requires_score" in str(exc)
	else:
		raise AssertionError("expected score guardrail")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("service")
	api_module = _load_module("api")
	views_module = _load_module("views")
	app_module = _load_module("app")

	service = service_module.AdvancedCRMService()
	agent = service.register_crm_agent("tenant-test", "Pipeline Review", "codex", "pipeline_analyst", "Review pipeline quality.")
	agent_result = service.validate_agent_crm_action("tenant-test", agent["id"], "review_pipeline", True, True)
	batch = service.validate_batch_import("tenant-test", 4)
	dashboard = views_module.dashboard_model(service, "tenant-test")
	agent_view = views_module.agent_workbench_model(service, "tenant-test")
	api_record = api_module.create_record({"id": "api-account", "tenant_id": "tenant-api"})
	status = api_module.capability_status("tenant-api")
	self_test = app_module.self_test()
	model = app_module.semantic_model()

	assert agent_result["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["crm_agent_count"] == 1
	assert agent_view["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert api_record["id"].startswith("crm_account_")
	assert status["streaming"]["processor"] == "bytewax"
	assert self_test["passed"] is True
	assert model["capabilities"]["crm_adv"]["streaming"]["processor"] == "bytewax"
