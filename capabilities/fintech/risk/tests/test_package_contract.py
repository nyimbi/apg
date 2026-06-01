"""Executable FinTech Risk Management capability package tests."""

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


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("contract_fintech_risk", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_risk"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "risk_agent_workflow" in contract["provides"]
	assert "/fintech-risk/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_fintech_risk", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "risk_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "risk_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_risk_lifecycle():
	service_module = _load_module("service_fintech_risk", PACKAGE_DIR / "service.py")
	service = service_module.RiskManagementService()

	appetite = service.register_appetite("appetite-1", "tenant-test", "credit", 500000, "KES", "owner-1", "evidence-1")
	profile = service.create_profile("profile-1", "tenant-test", "customer-1", "customer", "kyc-1", 250000, "KES", 54, "risk-engine-1")
	exposure = service.record_exposure("exposure-1", "tenant-test", profile["id"], "credit_limit", 300000, "KES", appetite["threshold_minor"], "loan-ledger-1")
	control = service.evaluate_control("control-1", "tenant-test", profile["id"], "preventive", "control-owner", "control-evidence", 82)
	scenario = service.run_stress_scenario("scenario-1", "tenant-test", profile["id"], "macro_shock", 125000, 2500, "mitigation-1")
	breach = service.record_limit_breach("breach-1", "tenant-test", exposure["id"], "medium", "breach-evidence", "risk-owner")
	event = service.open_risk_event("event-1", "tenant-test", profile["id"], "limit_breach", "medium", "event-evidence")
	review = service.record_review("review-1", "tenant-test", event["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_risk_agent("agent-1", "tenant-test", "Risk Agent", "codex", "exposure_monitor", "monitor limits")
	batch = service.validate_batch("tenant-test", 4)
	summary = service.dashboard_summary("tenant-test")

	assert profile["risk_band"] == "medium"
	assert exposure["status"] == "within_limit"
	assert control["effectiveness_score"] == 82
	assert scenario["probability_bps"] == 2500
	assert breach["status"] == "open"
	assert review["status"] == "approved"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["profile_count"] == 1
	assert summary["audit_event_count"] == 9


def test_service_guardrails_reject_invalid_risk_actions():
	service_module = _load_module("guardrail_service_fintech_risk", PACKAGE_DIR / "service.py")
	service = service_module.RiskManagementService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_appetite("appetite", "", "credit", 1, "KES", "owner", "evidence")
	with pytest.raises(PermissionError, match="risk_domain_not_supported"):
		service.register_appetite("appetite", "tenant-test", "unknown", 1, "KES", "owner", "evidence")
	appetite = service.register_appetite("appetite-ok", "tenant-test", "credit", 100, "KES", "owner", "evidence")
	with pytest.raises(PermissionError, match="risk_kyc_required"):
		service.create_profile("profile", "tenant-test", "subject", "customer", "", 10, "KES", 20, "source")
	with pytest.raises(PermissionError, match="risk_score_out_of_range"):
		service.create_profile("profile", "tenant-test", "subject", "customer", "kyc", 10, "KES", 120, "source")
	profile = service.create_profile("profile-ok", "tenant-test", "subject", "customer", "kyc", 10, "KES", 20, "source")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.record_exposure("exposure", "tenant-test", profile["id"], "credit_limit", 200, "KES", appetite["threshold_minor"], "source")
	with pytest.raises(PermissionError, match="risk_currency_not_supported"):
		service.record_exposure("exposure-currency", "tenant-test", profile["id"], "credit_limit", 20, "XYZ", 100, "source")
	exposure = service.record_exposure("exposure-ok", "tenant-test", profile["id"], "credit_limit", 20, "KES", 100, "source")
	with pytest.raises(PermissionError, match="control_effectiveness_out_of_range"):
		service.evaluate_control("control", "tenant-test", profile["id"], "preventive", "owner", "evidence", 101)
	with pytest.raises(PermissionError, match="scenario_probability_out_of_range"):
		service.run_stress_scenario("scenario", "tenant-test", profile["id"], "macro_shock", 10, 10001, "mitigation")
	with pytest.raises(PermissionError, match="remediation_owner_required"):
		service.record_limit_breach("breach", "tenant-test", exposure["id"], "medium", "evidence", "")
	with pytest.raises(PermissionError, match="risk_event_type_not_supported"):
		service.open_risk_event("event", "tenant-test", profile["id"], "unknown", "medium", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", profile["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="risk_agent_runtime_not_supported"):
		service.register_risk_agent("agent", "tenant-test", "Bad Agent", "unsupported", "exposure_monitor", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_risk", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_risk", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_risk", PACKAGE_DIR / "app.py")

	appetite = api.register_appetite({"tenant_id": "tenant-api", "appetite_id": "api-appetite", "risk_domain": "credit", "threshold_minor": 100000, "currency": "KES", "owner_id": "owner", "evidence_reference": "evidence"})
	profile = api.create_profile({"tenant_id": "tenant-api", "profile_id": "api-profile", "subject_reference": "customer", "subject_type": "customer", "kyc_reference": "kyc", "exposure_minor": 10, "currency": "KES", "risk_score": 42, "source_reference": "source"})
	api.record_exposure({"tenant_id": "tenant-api", "exposure_id": "api-exposure", "profile_id": profile["id"], "exposure_type": "credit_limit", "amount_minor": 10, "currency": "KES", "limit_minor": appetite["threshold_minor"], "source_reference": "source"})
	agent = api.register_risk_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Risk Agent", "runtime": "claude_code", "role": "risk_event_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.risk_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "risk_event_reviewer"
	assert dashboard["summary"]["profile_count"] == 1
	assert console["exposures"][0]["id"] == "api-exposure"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_risk"]["screens"]["agents"]["route"] == "/fintech-risk/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_risk", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_risk"]["streaming"]["processor"] == "bytewax"
