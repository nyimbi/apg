"""Executable Digital Payments capability package tests."""

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
	module = _load_module("contract_fintech_payments", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_payments"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "payment_agents" in contract["provides"]
	assert "/fintech-payments/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_overrefund():
	module = _load_module("rules_fintech_payments", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "payment_batch",
		"event_stream": "queue",
	})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "refund_payment",
		"overrefund": True,
	})["matched_rules"] == ["refund_blocks_overrefund"]


def test_service_executes_payment_lifecycle():
	service_module = _load_module("service_fintech_payments", PACKAGE_DIR / "service.py")
	service = service_module.DigitalPaymentsService()

	account = service.open_payment_account("acct-1", "tenant-test", "customer-1", "KES")
	instrument = service.register_instrument("inst-1", "tenant-test", account["id"], "mobile_money", "vault://mpesa/customer-1")
	order = service.create_payment_order("pay-1", "tenant-test", account["id"], instrument["id"], 1500, "KES", "merchant-1", "invoice")
	risk = service.screen_payment_risk("risk-1", "tenant-test", order["id"], "medium", "0.35")
	authorization = service.authorize_payment("auth-1", "tenant-test", order["id"], "provider://mpesa")
	capture = service.capture_payment("cap-1", "tenant-test", order["id"], 1500)
	refund = service.refund_payment("refund-1", "tenant-test", order["id"], 250, "customer_request")
	payout = service.schedule_payout("payout-1", "tenant-test", account["id"], 100, "KES", "bank://settlement")
	settlement = service.record_settlement("settle-1", "tenant-test", order["id"], "SETTLE-1", 1250)
	dispute = service.open_dispute("dispute-1", "tenant-test", order["id"], "ops-1", "authorization")
	agent = service.register_payment_agent("agent-1", "tenant-test", "Risk Agent", "codex", "risk_reviewer", "review high risk payments")
	batch = service.validate_batch("tenant-test", 2)
	summary = service.dashboard_summary("tenant-test")

	assert risk["status"] == "screened"
	assert authorization["status"] == "authorized"
	assert capture["status"] == "captured"
	assert refund["status"] == "refunded"
	assert payout["status"] == "scheduled"
	assert settlement["status"] == "settled"
	assert dispute["status"] == "opened"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["captured_volume"] == "1500"
	assert summary["open_disputes"] == 1
	assert summary["audit_event_count"] == 11


def test_service_guardrails_reject_invalid_actions():
	service_module = _load_module("guardrail_service_fintech_payments", PACKAGE_DIR / "service.py")
	service = service_module.DigitalPaymentsService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.open_payment_account("acct", "", "owner", "KES")
	with pytest.raises(PermissionError, match="currency_not_supported"):
		service.open_payment_account("acct", "tenant-test", "owner", "BTC")

	account = service.open_payment_account("acct", "tenant-test", "owner", "KES")
	with pytest.raises(PermissionError, match="instrument_type_not_supported"):
		service.register_instrument("inst", "tenant-test", account["id"], "crypto", "vault://x")
	instrument = service.register_instrument("inst", "tenant-test", account["id"], "mobile_money", "vault://x")
	order = service.create_payment_order("pay", "tenant-test", account["id"], instrument["id"], 100, "KES", "merchant")
	with pytest.raises(PermissionError, match="payment_risk_review_required"):
		service.screen_payment_risk("risk", "tenant-test", order["id"], "high", "0.8")
	service.screen_payment_risk("risk", "tenant-test", order["id"], "blocked", "0.99", "reviewer")
	with pytest.raises(PermissionError, match="payment_risk_blocked"):
		service.authorize_payment("auth", "tenant-test", order["id"], "provider://mpesa")
	service.screen_payment_risk("risk-2", "tenant-test", order["id"], "low", "0.1")
	service.authorize_payment("auth", "tenant-test", order["id"], "provider://mpesa")
	with pytest.raises(PermissionError, match="overcapture_blocked"):
		service.capture_payment("cap", "tenant-test", order["id"], 101)


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_payments", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_payments", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_payments", PACKAGE_DIR / "app.py")

	account = api.open_payment_account({"tenant_id": "tenant-api", "account_id": "api-acct", "owner_reference": "customer-api", "currency": "KES"})
	instrument = api.register_instrument({"tenant_id": "tenant-api", "instrument_id": "api-inst", "account_id": account["id"], "instrument_type": "mobile_money", "token_reference": "vault://api"})
	order = api.create_payment_order({"tenant_id": "tenant-api", "order_id": "api-pay", "account_id": account["id"], "instrument_id": instrument["id"], "amount": 42, "currency": "KES", "counterparty_reference": "merchant-api"})
	agent = api.register_payment_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Settlement Agent", "runtime": "claude_code", "role": "settlement_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.order_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert order["amount"] == "42"
	assert agent["metadata"]["role"] == "settlement_reviewer"
	assert dashboard["summary"]["order_count"] == 1
	assert console["orders"][0]["id"] == "api-pay"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_payments"]["screens"]["agents"]["route"] == "/fintech-payments/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_payments", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_payments"]["streaming"]["processor"] == "bytewax"
