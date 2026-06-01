"""Executable Algorithmic Trading capability package tests."""

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
	module = _load_module("contract_fintech_trading", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_trading"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "trading_order_intent_workflow" in contract["provides"]
	assert "/fintech-trading/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_fintech_trading", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "trading_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "trading_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_trading_lifecycle():
	service_module = _load_module("service_fintech_trading", PACKAGE_DIR / "service.py")
	service = service_module.AlgorithmicTradingService()

	strategy = service.register_strategy("strategy-1", "tenant-test", "owner-1", "Momentum Core", "momentum", "equity", "policy-1")
	signal = service.attach_signal_source("signal-1", "tenant-test", strategy["id"], "market-feed-1", "PT5S", "lineage-1")
	backtest = service.record_backtest("backtest-1", "tenant-test", strategy["id"], "2021-2025", 1200, "dataset-1", {"sharpe": 1.7, "max_drawdown": 0.12})
	limit = service.set_risk_limit("limit-1", "tenant-test", strategy["id"], "gross_exposure", 2500000, "risk-approval-1")
	order = service.stage_order_intent("order-1", "tenant-test", strategy["id"], limit["id"], "ETF-1", "limit", 100.0, "order-approval-1")
	execution = service.record_execution("execution-1", "tenant-test", order["id"], "exchange", 100.0, "execution-feed-1")
	position = service.record_position_snapshot("position-1", "tenant-test", strategy["id"], "2026-06-01", 1500000, 900000, "position-source-1")
	alert = service.record_surveillance_alert("alert-1", "tenant-test", strategy["id"], "medium", "alert-evidence-1")
	review = service.record_review("review-1", "tenant-test", alert["id"], "reviewer-1", "approved", "review-evidence-1")
	agent = service.register_trading_agent("agent-1", "tenant-test", "Trading Agent", "codex", "trading_compliance_reviewer", "review trading controls")
	batch = service.validate_batch("tenant-test", 5)
	summary = service.dashboard_summary("tenant-test")

	assert strategy["strategy_type"] == "momentum"
	assert signal["freshness_sla"] == "PT5S"
	assert backtest["metrics"]["sharpe"] == 1.7
	assert limit["metric"] == "gross_exposure"
	assert order["order_type"] == "limit"
	assert execution["venue"] == "exchange"
	assert position["net_exposure_minor"] == 900000
	assert alert["severity"] == "medium"
	assert review["status"] == "approved"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["strategy_count"] == 1
	assert summary["audit_event_count"] == 10


def test_service_guardrails_reject_invalid_trading_actions():
	service_module = _load_module("guardrail_service_fintech_trading", PACKAGE_DIR / "service.py")
	service = service_module.AlgorithmicTradingService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_strategy("strategy", "", "owner", "Core", "momentum", "equity", "policy")
	with pytest.raises(PermissionError, match="strategy_owner_required"):
		service.register_strategy("strategy", "tenant-test", "", "Core", "momentum", "equity", "policy")
	with pytest.raises(PermissionError, match="strategy_type_not_supported"):
		service.register_strategy("strategy", "tenant-test", "owner", "Core", "unsupported", "equity", "policy")
	strategy = service.register_strategy("strategy-ok", "tenant-test", "owner", "Core", "momentum", "equity", "policy")
	with pytest.raises(PermissionError, match="signal_source_required"):
		service.attach_signal_source("signal", "tenant-test", strategy["id"], "", "PT5S", "lineage")
	with pytest.raises(PermissionError, match="positive_trade_count_required"):
		service.record_backtest("backtest", "tenant-test", strategy["id"], "2025", 0, "dataset", {})
	with pytest.raises(PermissionError, match="risk_approval_required"):
		service.set_risk_limit("limit", "tenant-test", strategy["id"], "gross", 10, "")
	limit = service.set_risk_limit("limit-ok", "tenant-test", strategy["id"], "gross", 10, "approval")
	with pytest.raises(PermissionError, match="order_type_not_supported"):
		service.stage_order_intent("order", "tenant-test", strategy["id"], limit["id"], "ETF", "unsupported", 10, "approval")
	order = service.stage_order_intent("order-ok", "tenant-test", strategy["id"], limit["id"], "ETF", "limit", 10, "approval")
	with pytest.raises(PermissionError, match="execution_venue_not_supported"):
		service.record_execution("execution", "tenant-test", order["id"], "unsupported", 10, "source")
	with pytest.raises(PermissionError, match="position_as_of_date_required"):
		service.record_position_snapshot("position", "tenant-test", strategy["id"], "", 100, 80, "source")
	with pytest.raises(PermissionError, match="surveillance_severity_not_supported"):
		service.record_surveillance_alert("alert", "tenant-test", strategy["id"], "unknown", "evidence")
	with pytest.raises(PermissionError, match="review_status_not_supported"):
		service.record_review("review", "tenant-test", strategy["id"], "reviewer", "maybe", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="trading_agent_runtime_not_supported"):
		service.register_trading_agent("agent", "tenant-test", "Bad Agent", "unsupported", "trading_compliance_reviewer", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_trading", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_trading", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_trading", PACKAGE_DIR / "app.py")

	strategy = api.register_strategy({"tenant_id": "tenant-api", "strategy_id": "api-strategy", "owner_id": "owner", "name": "API Strategy", "strategy_type": "momentum", "asset_class": "equity", "policy_reference": "policy"})
	api.attach_signal_source({"tenant_id": "tenant-api", "signal_id": "api-signal", "strategy_id": strategy["id"], "source_reference": "source", "freshness_sla": "PT5S", "lineage_reference": "lineage"})
	api.record_backtest({"tenant_id": "tenant-api", "backtest_id": "api-backtest", "strategy_id": strategy["id"], "period": "2025", "trade_count": 10, "data_source_reference": "dataset", "metrics": {"sharpe": 1.1}})
	limit = api.set_risk_limit({"tenant_id": "tenant-api", "limit_id": "api-limit", "strategy_id": strategy["id"], "metric": "gross", "limit_value": 100, "approval_reference": "approval"})
	api.stage_order_intent({"tenant_id": "tenant-api", "order_id": "api-order", "strategy_id": strategy["id"], "risk_limit_id": limit["id"], "instrument_id": "ETF", "order_type": "limit", "quantity": 5, "approval_reference": "approval"})
	agent = api.register_trading_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Agent", "runtime": "claude_code", "role": "trading_compliance_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.trading_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "trading_compliance_reviewer"
	assert dashboard["summary"]["strategy_count"] == 1
	assert console["orders"][0]["id"] == "api-order"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_trading"]["screens"]["agents"]["route"] == "/fintech-trading/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_trading", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_trading"]["streaming"]["processor"] == "bytewax"
