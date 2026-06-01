"""Executable Wealth Management capability package tests."""

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
	module = _load_module("contract_fintech_wealth", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_wealth"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "portfolio_management_workflow" in contract["provides"]
	assert "/fintech-wealth/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_large_order_review():
	module = _load_module("rules_fintech_wealth", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "wealth_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "stage_order", "large_order": True, "human_approval_recorded": False})["decision"] == "require_review"


def test_service_executes_wealth_lifecycle():
	service_module = _load_module("service_fintech_wealth", PACKAGE_DIR / "service.py")
	service = service_module.WealthManagementService()

	client = service.register_client_profile("client-1", "tenant-test", "Amina Client", "kyc-1", "tax-1", "risk-1")
	suitability = service.capture_suitability_profile("suitability-1", "tenant-test", client["id"], "balanced", "medium", "five_years", ["capital_growth", "income"])
	portfolio = service.create_portfolio("portfolio-1", "tenant-test", client["id"], "Core Portfolio", "usd", "advisor-1", "ips-1")
	mandate = service.create_advisory_mandate("mandate-1", "tenant-test", portfolio["id"], suitability["id"], "discretionary", "policy-1")
	rebalance = service.propose_rebalance("rebalance-1", "tenant-test", portfolio["id"], mandate["id"], {"equity": 60, "fixed_income": 35, "cash": 5}, "analysis-1")
	order = service.stage_order("order-1", "tenant-test", portfolio["id"], "ETF-1", "buy", 10, 500000, "risk-order-1")
	performance = service.record_performance("performance-1", "tenant-test", portfolio["id"], "2026-Q1", "valuation-1", "benchmark-1", 4.2)
	fee = service.record_fee_schedule("fee-1", "tenant-test", portfolio["id"], 1.0, 10.0, 0.25, "fee-contract-1")
	agent = service.register_wealth_agent("agent-1", "tenant-test", "Wealth Agent", "codex", "portfolio_reviewer", "review portfolios")
	batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert suitability["risk_profile"] == "balanced"
	assert portfolio["base_currency"] == "USD"
	assert mandate["mandate_type"] == "discretionary"
	assert rebalance["target_allocation"]["cash"] == 5
	assert order["side"] == "buy"
	assert performance["return_percent"] == 4.2
	assert fee["platform_percent"] == 0.25
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["portfolio_count"] == 1
	assert summary["audit_event_count"] == 9


def test_service_guardrails_reject_invalid_wealth_actions():
	service_module = _load_module("guardrail_service_fintech_wealth", PACKAGE_DIR / "service.py")
	service = service_module.WealthManagementService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_client_profile("client", "", "Client", "kyc", "tax", "risk")
	with pytest.raises(PermissionError, match="client_kyc_required"):
		service.register_client_profile("client", "tenant-test", "Client", "", "tax", "risk")
	client = service.register_client_profile("client-ok", "tenant-test", "Client", "kyc", "tax", "risk")
	with pytest.raises(PermissionError, match="risk_profile_not_supported"):
		service.capture_suitability_profile("suitability", "tenant-test", client["id"], "unsupported", "medium", "five_years", ["growth"])
	suitability = service.capture_suitability_profile("suitability-ok", "tenant-test", client["id"], "balanced", "medium", "five_years", ["growth"])
	with pytest.raises(PermissionError, match="portfolio_currency_not_supported"):
		service.create_portfolio("portfolio", "tenant-test", client["id"], "Portfolio", "XYZ", "advisor", "ips")
	portfolio = service.create_portfolio("portfolio-ok", "tenant-test", client["id"], "Portfolio", "USD", "advisor", "ips")
	with pytest.raises(PermissionError, match="mandate_type_not_supported"):
		service.create_advisory_mandate("mandate", "tenant-test", portfolio["id"], suitability["id"], "unsupported", "policy")
	mandate = service.create_advisory_mandate("mandate-ok", "tenant-test", portfolio["id"], suitability["id"], "advisory", "policy")
	with pytest.raises(PermissionError, match="allocation_total_must_equal_100"):
		service.propose_rebalance("rebalance", "tenant-test", portfolio["id"], mandate["id"], {"equity": 50}, "analysis")
	with pytest.raises(PermissionError, match="large_order_approval_required"):
		service.stage_order("order-large", "tenant-test", portfolio["id"], "ETF", "buy", 1, 10000000, "risk")
	with pytest.raises(PermissionError, match="performance_benchmark_required"):
		service.record_performance("performance", "tenant-test", portfolio["id"], "2026-Q1", "valuation", "", 1.0)
	with pytest.raises(PermissionError, match="fee_percent_out_of_bounds"):
		service.record_fee_schedule("fee", "tenant-test", portfolio["id"], 101, 1, 1, "contract")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="wealth_agent_runtime_not_supported"):
		service.register_wealth_agent("agent", "tenant-test", "Bad Agent", "unsupported", "portfolio_reviewer", "scope")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_wealth", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_wealth", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_wealth", PACKAGE_DIR / "app.py")

	client = api.register_client_profile({"tenant_id": "tenant-api", "client_id": "api-client", "name": "Client", "kyc_reference": "kyc", "tax_reference": "tax", "risk_reference": "risk"})
	suitability = api.capture_suitability_profile({"tenant_id": "tenant-api", "suitability_id": "api-suitability", "client_id": client["id"], "risk_profile": "balanced", "risk_tolerance": "medium", "horizon": "five_years", "goals": ["growth"]})
	portfolio = api.create_portfolio({"tenant_id": "tenant-api", "portfolio_id": "api-portfolio", "client_id": client["id"], "name": "Portfolio", "base_currency": "USD", "advisor_id": "advisor", "policy_reference": "ips"})
	api.create_advisory_mandate({"tenant_id": "tenant-api", "mandate_id": "api-mandate", "portfolio_id": portfolio["id"], "suitability_id": suitability["id"], "mandate_type": "advisory", "policy_reference": "policy"})
	api.stage_order({"tenant_id": "tenant-api", "order_id": "api-order", "portfolio_id": portfolio["id"], "instrument_id": "ETF", "side": "buy", "quantity": 1, "notional_minor": 1000, "risk_reference": "risk-order"})
	agent = api.register_wealth_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Agent", "runtime": "claude_code", "role": "wealth_compliance_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.wealth_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "wealth_compliance_reviewer"
	assert dashboard["summary"]["order_count"] == 1
	assert console["orders"][0]["id"] == "api-order"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_wealth"]["screens"]["agents"]["route"] == "/fintech-wealth/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_wealth", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_wealth"]["streaming"]["processor"] == "bytewax"
