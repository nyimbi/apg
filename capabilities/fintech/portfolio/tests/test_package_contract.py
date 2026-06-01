"""Executable Portfolio Management capability package tests."""

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
	module = _load_module("contract_fintech_portfolio", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_portfolio"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "portfolio_valuation_workflow" in contract["provides"]
	assert "/fintech-portfolio/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_fintech_portfolio", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "portfolio_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "portfolio_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_portfolio_lifecycle():
	service_module = _load_module("service_fintech_portfolio", PACKAGE_DIR / "service.py")
	service = service_module.PortfolioManagementService()

	portfolio = service.create_portfolio_book("portfolio-1", "tenant-test", "owner-1", "Core Portfolio", "discretionary", "usd", "ips-1")
	holding = service.record_holding("holding-1", "tenant-test", portfolio["id"], "ETF-1", 12.5, 1000000, "usd")
	allocation = service.activate_allocation_policy("allocation-1", "tenant-test", portfolio["id"], {"equity": 60, "fixed_income": 35, "cash": 5}, "policy-1")
	valuation = service.record_valuation("valuation-1", "tenant-test", portfolio["id"], 1500000, "usd", "2026-06-01", "pricing-1")
	benchmark = service.assign_benchmark("benchmark-1", "tenant-test", portfolio["id"], "MSCI-WORLD", "benchmark-policy-1")
	risk = service.record_risk_exposure("risk-1", "tenant-test", portfolio["id"], "var_95", 2.7, "2026-06-01", "risk-engine-1", "limit-1")
	attribution = service.record_attribution("attribution-1", "tenant-test", portfolio["id"], "2026-05", benchmark["id"], "attribution-engine-1", {"allocation": 0.6, "selection": 0.2})
	cash = service.record_cash_movement("cash-1", "tenant-test", portfolio["id"], 50000, "usd", "wallet-1")
	action = service.record_corporate_action("action-1", "tenant-test", holding["instrument_id"], "dividend", "2026-06-15", "notice-1")
	breach = service.record_compliance_breach("breach-1", "tenant-test", portfolio["id"], "medium", "breach-evidence-1")
	review = service.record_review("review-1", "tenant-test", breach["id"], "reviewer-1", "approved", "review-evidence-1")
	agent = service.register_portfolio_agent("agent-1", "tenant-test", "Portfolio Agent", "codex", "portfolio_compliance_reviewer", "review breaches")
	batch = service.validate_batch("tenant-test", 4)
	summary = service.dashboard_summary("tenant-test")

	assert portfolio["base_currency"] == "USD"
	assert holding["quantity"] == 12.5
	assert allocation["target_allocation"]["cash"] == 5
	assert valuation["market_value_minor"] == 1500000
	assert benchmark["index_id"] == "MSCI-WORLD"
	assert risk["metric"] == "var_95"
	assert attribution["period"] == "2026-05"
	assert cash["currency"] == "USD"
	assert action["action_type"] == "dividend"
	assert breach["severity"] == "medium"
	assert review["status"] == "approved"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["portfolio_count"] == 1
	assert summary["audit_event_count"] == 12


def test_service_guardrails_reject_invalid_portfolio_actions():
	service_module = _load_module("guardrail_service_fintech_portfolio", PACKAGE_DIR / "service.py")
	service = service_module.PortfolioManagementService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_portfolio_book("portfolio", "", "owner", "Core", "discretionary", "USD", "policy")
	with pytest.raises(PermissionError, match="portfolio_owner_required"):
		service.create_portfolio_book("portfolio", "tenant-test", "", "Core", "discretionary", "USD", "policy")
	with pytest.raises(PermissionError, match="portfolio_type_not_supported"):
		service.create_portfolio_book("portfolio", "tenant-test", "owner", "Core", "unsupported", "USD", "policy")
	portfolio = service.create_portfolio_book("portfolio-ok", "tenant-test", "owner", "Core", "discretionary", "USD", "policy")
	with pytest.raises(PermissionError, match="positive_holding_quantity_required"):
		service.record_holding("holding", "tenant-test", portfolio["id"], "ETF", 0, 100, "USD")
	with pytest.raises(PermissionError, match="allocation_total_must_equal_100"):
		service.activate_allocation_policy("allocation", "tenant-test", portfolio["id"], {"equity": 50}, "policy")
	with pytest.raises(PermissionError, match="valuation_source_required"):
		service.record_valuation("valuation", "tenant-test", portfolio["id"], 100, "USD", "2026-06-01", "")
	with pytest.raises(PermissionError, match="benchmark_index_required"):
		service.assign_benchmark("benchmark", "tenant-test", portfolio["id"], "", "policy")
	with pytest.raises(PermissionError, match="risk_source_required"):
		service.record_risk_exposure("risk", "tenant-test", portfolio["id"], "var", 1.0, "2026-06-01", "", "limit")
	with pytest.raises(PermissionError, match="attribution_period_required"):
		service.record_attribution("attribution", "tenant-test", portfolio["id"], "", "benchmark", "source", {})
	with pytest.raises(PermissionError, match="cash_currency_not_supported"):
		service.record_cash_movement("cash", "tenant-test", portfolio["id"], 100, "XXX", "reference")
	with pytest.raises(PermissionError, match="corporate_action_type_not_supported"):
		service.record_corporate_action("action", "tenant-test", "ETF", "unsupported", "2026-06-01", "evidence")
	with pytest.raises(PermissionError, match="compliance_severity_not_supported"):
		service.record_compliance_breach("breach", "tenant-test", portfolio["id"], "unknown", "evidence")
	with pytest.raises(PermissionError, match="review_status_not_supported"):
		service.record_review("review", "tenant-test", portfolio["id"], "reviewer", "maybe", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="portfolio_agent_runtime_not_supported"):
		service.register_portfolio_agent("agent", "tenant-test", "Bad Agent", "unsupported", "portfolio_compliance_reviewer", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_portfolio", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_portfolio", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_portfolio", PACKAGE_DIR / "app.py")

	portfolio = api.create_portfolio_book({"tenant_id": "tenant-api", "portfolio_id": "api-portfolio", "owner_id": "owner", "name": "API Portfolio", "portfolio_type": "advisory", "base_currency": "USD", "policy_reference": "policy"})
	api.record_holding({"tenant_id": "tenant-api", "holding_id": "api-holding", "portfolio_id": portfolio["id"], "instrument_id": "ETF", "quantity": 2, "cost_minor": 100, "currency": "USD"})
	api.activate_allocation_policy({"tenant_id": "tenant-api", "allocation_id": "api-allocation", "portfolio_id": portfolio["id"], "target_allocation": {"equity": 70, "cash": 30}, "policy_reference": "policy"})
	api.record_valuation({"tenant_id": "tenant-api", "valuation_id": "api-valuation", "portfolio_id": portfolio["id"], "market_value_minor": 1000, "currency": "USD", "valuation_date": "2026-06-01", "source_reference": "source"})
	agent = api.register_portfolio_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Agent", "runtime": "claude_code", "role": "portfolio_compliance_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.portfolio_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "portfolio_compliance_reviewer"
	assert dashboard["summary"]["portfolio_count"] == 1
	assert console["valuations"][0]["id"] == "api-valuation"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_portfolio"]["screens"]["agents"]["route"] == "/fintech-portfolio/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_portfolio", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_portfolio"]["streaming"]["processor"] == "bytewax"
