"""Executable Robo Advisory capability package tests."""

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
	module = _load_module("contract_fintech_robo", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_robo"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "robo_recommendation_workflow" in contract["provides"]
	assert "/fintech-robo/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_fintech_robo", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "robo_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "robo_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_robo_lifecycle():
	service_module = _load_module("service_fintech_robo", PACKAGE_DIR / "service.py")
	service = service_module.RoboAdvisoryService()

	profile = service.create_investor_profile("profile-1", "tenant-test", "client-1", "kyc-1", "suitability-1", "balanced")
	goal = service.define_goal_plan("goal-1", "tenant-test", profile["id"], "retirement", 50000000, "usd", "2036-12-31")
	model = service.publish_model_portfolio("model-1", "tenant-test", "Balanced Core", "balanced", {"equity": 60, "fixed_income": 35, "cash": 5}, "policy-1")
	recommendation = service.generate_recommendation("rec-1", "tenant-test", profile["id"], goal["id"], model["id"], "analysis-1")
	approved = service.approve_recommendation(recommendation["id"], "tenant-test", "reviewer-1")
	automation = service.configure_automation_plan("plan-1", "tenant-test", approved["id"], "wallet-1", "monthly")
	drift = service.record_drift("drift-1", "tenant-test", profile["id"], 650, "drift-analysis-1")
	tax = service.record_tax_loss_candidate("tax-1", "tenant-test", profile["id"], "ETF-1", 25000, "taxlot-1")
	review = service.record_review("review-1", "tenant-test", recommendation["id"], "reviewer-1", "approved", "evidence-1")
	agent = service.register_robo_agent("agent-1", "tenant-test", "Robo Agent", "codex", "recommendation_reviewer", "review recommendations")
	batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert profile["risk_profile"] == "balanced"
	assert goal["currency"] == "USD"
	assert model["target_allocation"]["cash"] == 5
	assert approved["status"] == "approved"
	assert automation["cadence"] == "monthly"
	assert drift["drift_bps"] == 650
	assert tax["loss_minor"] == 25000
	assert review["status"] == "approved"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["recommendation_count"] == 1
	assert summary["audit_event_count"] == 10


def test_service_guardrails_reject_invalid_robo_actions():
	service_module = _load_module("guardrail_service_fintech_robo", PACKAGE_DIR / "service.py")
	service = service_module.RoboAdvisoryService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_investor_profile("profile", "", "client", "kyc", "suitability", "balanced")
	with pytest.raises(PermissionError, match="investor_kyc_required"):
		service.create_investor_profile("profile", "tenant-test", "client", "", "suitability", "balanced")
	profile = service.create_investor_profile("profile-ok", "tenant-test", "client", "kyc", "suitability", "balanced")
	with pytest.raises(PermissionError, match="goal_type_not_supported"):
		service.define_goal_plan("goal", "tenant-test", profile["id"], "unsupported", 100, "USD", "2036-12-31")
	goal = service.define_goal_plan("goal-ok", "tenant-test", profile["id"], "retirement", 100, "USD", "2036-12-31")
	with pytest.raises(PermissionError, match="model_allocation_total_must_equal_100"):
		service.publish_model_portfolio("model", "tenant-test", "Model", "balanced", {"equity": 50}, "policy")
	model = service.publish_model_portfolio("model-ok", "tenant-test", "Model", "balanced", {"equity": 60, "fixed_income": 40}, "policy")
	with pytest.raises(PermissionError, match="recommendation_analysis_required"):
		service.generate_recommendation("rec", "tenant-test", profile["id"], goal["id"], model["id"], "")
	rec = service.generate_recommendation("rec-ok", "tenant-test", profile["id"], goal["id"], model["id"], "analysis")
	with pytest.raises(PermissionError, match="approved_recommendation_required"):
		service.configure_automation_plan("plan", "tenant-test", rec["id"], "wallet", "monthly")
	service.approve_recommendation(rec["id"], "tenant-test", "reviewer")
	with pytest.raises(PermissionError, match="automation_cadence_not_supported"):
		service.configure_automation_plan("plan-bad", "tenant-test", rec["id"], "wallet", "daily")
	with pytest.raises(PermissionError, match="drift_analysis_required"):
		service.record_drift("drift", "tenant-test", profile["id"], 100, "")
	with pytest.raises(PermissionError, match="tax_lot_required"):
		service.record_tax_loss_candidate("tax", "tenant-test", profile["id"], "ETF", 100, "")
	with pytest.raises(PermissionError, match="review_status_not_supported"):
		service.record_review("review", "tenant-test", rec["id"], "reviewer", "maybe", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="robo_agent_runtime_not_supported"):
		service.register_robo_agent("agent", "tenant-test", "Bad Agent", "unsupported", "recommendation_reviewer", "scope")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_robo", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_robo", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_robo", PACKAGE_DIR / "app.py")

	profile = api.create_investor_profile({"tenant_id": "tenant-api", "profile_id": "api-profile", "client_id": "client", "kyc_reference": "kyc", "suitability_reference": "suitability", "risk_profile": "balanced"})
	goal = api.define_goal_plan({"tenant_id": "tenant-api", "goal_id": "api-goal", "profile_id": profile["id"], "goal_type": "retirement", "target_amount_minor": 100, "currency": "USD", "horizon_date": "2036-12-31"})
	model = api.publish_model_portfolio({"tenant_id": "tenant-api", "model_id": "api-model", "name": "Model", "risk_profile": "balanced", "target_allocation": {"equity": 60, "fixed_income": 40}, "policy_reference": "policy"})
	rec = api.generate_recommendation({"tenant_id": "tenant-api", "recommendation_id": "api-rec", "profile_id": profile["id"], "goal_id": goal["id"], "model_id": model["id"], "analysis_reference": "analysis"})
	api.approve_recommendation({"tenant_id": "tenant-api", "recommendation_id": rec["id"], "reviewer_id": "reviewer"})
	agent = api.register_robo_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Agent", "runtime": "claude_code", "role": "robo_compliance_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.robo_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "robo_compliance_reviewer"
	assert dashboard["summary"]["recommendation_count"] == 1
	assert console["recommendations"][0]["id"] == "api-rec"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_robo"]["screens"]["agents"]["route"] == "/fintech-robo/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_robo", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_robo"]["streaming"]["processor"] == "bytewax"
