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


async def test_service_executes_portfolio_lifecycle():
	service_module = _load_module("service_fintech_portfolio", PACKAGE_DIR / "service.py")
	service = service_module.PortfolioManagementService(tenant_id="tenant-test")

	portfolio = await service.create_portfolio(
		name="Core Portfolio", client_id="owner-1", strategy="growth",
		benchmark="MSCI-WORLD", portfolio_type="discretionary",
		base_currency="usd", policy_reference="ips-1",
		portfolio_id="portfolio-1",
	)
	holding = await service.add_holding(
		portfolio_id=portfolio["id"], asset_id="ETF-1",
		quantity=12.5, cost_basis=10000.0, currency="usd",
	)
	allocation = await service.activate_allocation_policy(
		allocation_id="allocation-1", portfolio_id=portfolio["id"],
		target_allocation={"equity": 60, "fixed_income": 35, "cash": 5},
		policy_reference="policy-1",
	)
	valuation = await service.portfolio_valuation(
		portfolio_id=portfolio["id"], as_of_date="2026-06-01",
		source_reference="pricing-1",
	)
	benchmark = await service.assign_benchmark(
		benchmark_id="benchmark-1", portfolio_id=portfolio["id"],
		index_id="MSCI-WORLD", policy_reference="benchmark-policy-1",
	)
	risk = await service.record_risk_exposure(
		exposure_id="risk-1", portfolio_id=portfolio["id"],
		metric="var_95", value=2.7, as_of_date="2026-06-01",
		source_reference="risk-engine-1", limit_reference="limit-1",
	)
	attribution = await service.performance_attribution(
		portfolio_id=portfolio["id"], period="2026-05",
		benchmark_id=benchmark["id"],
	)
	cash = await service.record_cash_movement(
		movement_id="cash-1", portfolio_id=portfolio["id"],
		amount_minor=50000, currency="usd", reference="wallet-1",
	)
	action = await service.record_corporate_action(
		action_id="action-1", instrument_id=holding["instrument_id"],
		action_type="dividend", effective_date="2026-06-15",
		evidence_reference="notice-1",
	)
	breach = await service.record_compliance_breach(
		breach_id="breach-1", portfolio_id=portfolio["id"],
		severity="medium", evidence_reference="breach-evidence-1",
	)
	review = await service.record_review(
		review_id="review-1", reference_id=breach["id"],
		reviewer_id="reviewer-1", status="approved",
		evidence_reference="review-evidence-1",
	)
	agent = await service.register_portfolio_agent(
		agent_id="agent-1", name="Portfolio Agent",
		runtime="codex", role="portfolio_compliance_reviewer",
		scope="review breaches",
	)
	batch = await service.validate_batch(4)
	summary = await service.dashboard_summary()

	assert portfolio["base_currency"] == "USD"
	assert holding["quantity"] == 12.5
	assert allocation["target_allocation"]["cash"] == 5
	assert valuation["market_value_minor"] > 0
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
	assert summary["audit_event_count"] >= 10


async def test_service_guardrails_reject_invalid_portfolio_actions():
	service_module = _load_module("guardrail_service_fintech_portfolio", PACKAGE_DIR / "service.py")
	service = service_module.PortfolioManagementService(tenant_id="tenant-test")

	# portfolio_type_not_supported → normalize_code("unsupported") not in SUPPORTED_PORTFOLIO_TYPES
	with pytest.raises(PermissionError):
		await service.create_portfolio("Core", "owner", "", "", portfolio_type="unsupported", base_currency="USD", policy_reference="policy")

	portfolio = await service.create_portfolio(
		"Core", "owner", "", "", portfolio_type="discretionary",
		base_currency="USD", policy_reference="policy",
	)
	pid = portfolio["id"]

	# positive_quantity guard: quantity=0 triggers assert
	with pytest.raises((PermissionError, AssertionError)):
		await service.add_holding(pid, "ETF", 0, 100, "USD")

	# allocation must total 100 (percentage scale)
	with pytest.raises(PermissionError):
		await service.activate_allocation_policy("alloc", pid, {"equity": 50}, "policy")

	# valuation source required
	with pytest.raises(PermissionError):
		await service.record_risk_exposure("r", pid, "var", 1.0, "2026-06-01", "", "limit")

	# unsupported currency
	with pytest.raises(PermissionError):
		await service.record_cash_movement("cash", pid, 100, "XXX", "reference")

	# unsupported corporate action type
	with pytest.raises(PermissionError):
		await service.record_corporate_action("act", "ETF", "unsupported", "2026-06-01", "evidence")

	# unsupported compliance severity
	with pytest.raises(PermissionError):
		await service.record_compliance_breach("breach", pid, "unknown", "evidence")

	# unsupported review status
	with pytest.raises(PermissionError):
		await service.record_review("rev", pid, "reviewer", "maybe", "evidence")

	# non-bytewax event stream
	with pytest.raises(PermissionError):
		await service.validate_batch(1, event_stream="queue")

	# unsupported agent runtime
	with pytest.raises(PermissionError):
		await service.register_portfolio_agent("agent", "Bad Agent", "unsupported", "portfolio_compliance_reviewer", "scope")

	# privileged agent action without approval
	with pytest.raises(PermissionError):
		await service.validate_agent_action(privileged_scope=True, human_approval_recorded=False)


async def test_api_views_and_app_are_executable():
	import asyncio as _asyncio
	app = _load_module("app_fintech_portfolio", PACKAGE_DIR / "app.py")
	service_module = _load_module("svc_api_test_fintech_portfolio", PACKAGE_DIR / "service.py")
	views_module = _load_module("views_api_test_fintech_portfolio", PACKAGE_DIR / "views.py")

	# Use the async service directly — api.py wrappers tested separately via their sync shell
	svc = service_module.PortfolioManagementService(tenant_id="tenant-api")
	portfolio = await svc.create_portfolio(
		name="API Portfolio", client_id="owner", strategy="", benchmark="",
		portfolio_type="advisory", base_currency="USD", policy_reference="policy",
		portfolio_id="api-portfolio",
	)
	await svc.add_holding(portfolio["id"], "ETF", 2, 1.0, "USD", holding_id="api-holding")
	await svc.activate_allocation_policy(
		"api-allocation", portfolio["id"], {"equity": 70, "cash": 30}, "policy",
	)
	val = await svc.portfolio_valuation(portfolio["id"], "2026-06-01", "source")
	agent = await svc.register_portfolio_agent(
		"api-agent", "Agent", "claude_code", "portfolio_compliance_reviewer", "scope",
	)

	# views model — dashboard_model is async; portfolio_console_model is sync
	dashboard = await views_module.dashboard_model(svc, "tenant-api")
	console = views_module.portfolio_console_model(svc, "tenant-api")

	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "portfolio_compliance_reviewer"
	assert dashboard["summary"]["portfolio_count"] == 1
	assert len(console["valuations"]) == 1
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
