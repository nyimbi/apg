"""Executable Fintech Gateway capability package tests."""

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
	module = _load_module("contract_fintech_gateway", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_gateway"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "gateway_agents" in contract["provides"]
	assert "/fintech-gateway/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_rule_engine_blocks_missing_context_non_bytewax_and_overcapture():
	module = _load_module("rules_fintech_gateway", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "gateway_batch",
		"event_stream": "queue",
	})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "capture_payment",
		"overcapture": True,
	})["matched_rules"] == ["capture_blocks_overcapture"]


def test_service_executes_gateway_lifecycle():
	service_module = _load_module("service_fintech_gateway", PACKAGE_DIR / "service.py")
	service = service_module.FintechGatewayService()

	merchant = service.onboard_merchant("merchant-1", "tenant-test", "MERCH-001", "Merchant One", "KE")
	provider = service.connect_provider("provider-1", "tenant-test", "mpesa", "mobile_money", "vault://mpesa")
	method = service.tokenize_payment_method("method-1", "tenant-test", merchant["id"], "customer-1", "mobile_money", "tok-1")
	intent = service.create_payment_intent("intent-1", "tenant-test", merchant["id"], method["id"], 1000, "KES", "Order 1")
	risk = service.assess_payment_risk("risk-1", "tenant-test", intent["id"], "medium", 0.35)
	authorization = service.authorize_payment("auth-1", "tenant-test", intent["id"], provider["id"])
	capture = service.capture_payment("capture-1", "tenant-test", authorization["id"], 1000)
	webhook = service.ingest_webhook("webhook-1", "tenant-test", provider["id"], "evt-1", "sig", "idem-1", "payment.captured")
	settlement = service.record_settlement("settlement-1", "tenant-test", provider["id"], "SETTLE-1", 1000)
	dispute = service.open_dispute("dispute-1", "tenant-test", intent["id"], "authorization", "owner-1")
	resolved = service.resolve_dispute(dispute["id"], "tenant-test", "merchant evidence accepted", "reviewer-1")
	agent = service.register_gateway_agent("tenant-test", "Risk Agent", "codex", "fraud_reviewer", "review payment risk")

	summary = service.dashboard_summary("tenant-test")
	assert risk["status"] == "assessed"
	assert capture["status"] == "captured"
	assert webhook["status"] == "ingested"
	assert settlement["status"] == "settled"
	assert resolved["status"] == "resolved"
	assert agent["runtime"] == "codex"
	assert summary["captured_volume"] == "1000"
	assert summary["audit_event_count"] == 12
	assert summary["streaming"]["processor"] == "bytewax"


def test_service_guardrails_reject_invalid_actions():
	service_module = _load_module("guardrail_service_fintech_gateway", PACKAGE_DIR / "service.py")
	service = service_module.FintechGatewayService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.onboard_merchant("merchant", "", "M", "Merchant", "KE")
	with pytest.raises(PermissionError, match="merchant_review_required"):
		service.onboard_merchant("merchant", "tenant-test", "M", "Merchant", "KE", "high")
	with pytest.raises(PermissionError, match="provider_not_supported"):
		service.connect_provider("provider", "tenant-test", "unsupported", "card", "vault://provider")

	merchant = service.onboard_merchant("merchant", "tenant-test", "M", "Merchant", "KE")
	provider = service.connect_provider("provider", "tenant-test", "stripe", "card", "vault://stripe")
	method = service.tokenize_payment_method("method", "tenant-test", merchant["id"], "customer", "card", "tok")
	intent = service.create_payment_intent("intent", "tenant-test", merchant["id"], method["id"], 100, "USD")
	service.assess_payment_risk("risk", "tenant-test", intent["id"], "blocked", 0.99, "reviewer")
	with pytest.raises(PermissionError, match="payment_risk_blocked"):
		service.authorize_payment("auth", "tenant-test", intent["id"], provider["id"])
	service.assess_payment_risk("risk-2", "tenant-test", intent["id"], "low", 0.1)
	auth = service.authorize_payment("auth", "tenant-test", intent["id"], provider["id"])
	with pytest.raises(PermissionError, match="overcapture_blocked"):
		service.capture_payment("capture", "tenant-test", auth["id"], 101)
	service.capture_payment("capture", "tenant-test", auth["id"], 100)
	with pytest.raises(PermissionError, match="overrefund_blocked"):
		service.refund_payment("refund", "tenant-test", intent["id"], 101, "customer_request")
	with pytest.raises(PermissionError, match="webhook_signature_required"):
		service.ingest_webhook("webhook", "tenant-test", provider["id"], "evt", "", "idem", "payment.updated")


def test_agents_batch_api_views_and_app_are_executable():
	api = _load_module("api_fintech_gateway", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_gateway", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_gateway", PACKAGE_DIR / "app.py")

	merchant = api.create_record({"tenant_id": "tenant-api", "merchant_id": "api-merchant"})
	agent = api.register_gateway_agent({
		"tenant_id": "tenant-api",
		"name": "Routing Agent",
		"runtime": "claude_code",
		"role": "routing_reviewer",
	})
	batch = api.service().validate_batch("tenant-api", 2)
	model = views.merchant_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert merchant["id"] == "api-merchant"
	assert agent["role"] == "routing_reviewer"
	assert batch["processor"] == "bytewax"
	assert model["records"][0]["merchant_code"] == "APIMERCH"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_gateway"]["screens"]["agents"]["route"] == "/fintech-gateway/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_gateway", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_gateway"]["streaming"]["processor"] == "bytewax"
