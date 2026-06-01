"""Executable Mobile Banking capability package tests."""

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
	module = _load_module("contract_fintech_mobile", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_mobile"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "mobile_payment_workflow" in contract["provides"]
	assert "/fintech-mobile/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_high_value_payment():
	module = _load_module("rules_fintech_mobile", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "mobile_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "initiate_payment", "high_value": True, "human_approval_recorded": False})["decision"] == "require_review"


def test_service_executes_mobile_lifecycle():
	service_module = _load_module("service_fintech_mobile", PACKAGE_DIR / "service.py")
	service = service_module.MobileBankingService()

	program = service.register_program("program-1", "tenant-test", "Everyday Mobile", "mobile-ops", "KE", "KES", ["ios", "android", "ussd"])
	customer = service.enroll_customer("customer-1", "tenant-test", "crm-1", "KE", "kyc-1", "consent-1", "aml-1", "fraud-1")
	device = service.bind_device("device-1", "tenant-test", customer["id"], "ios", "fingerprint-1", "attestation-1", "low")
	factor = service.register_auth_factor("factor-1", "tenant-test", customer["id"], device["id"], "biometric", "strength-1")
	link = service.link_account("link-1", "tenant-test", customer["id"], "deposit", "account-1", "KES", "neobank-link-1")
	payment = service.initiate_payment("payment-1", "tenant-test", customer["id"], device["id"], link["id"], "peer_transfer", 2500, "KES", "recipient-1", "risk-1")
	bill_payment = service.initiate_payment("payment-bill", "tenant-test", customer["id"], device["id"], link["id"], "bill_payment", 800, "KES", "biller-1", "risk-bill")
	bill = service.record_bill_payment("bill-1", "tenant-test", bill_payment["id"], "biller-1", "bill-account-1")
	airtime_payment = service.initiate_payment("payment-airtime", "tenant-test", customer["id"], device["id"], link["id"], "airtime", 100, "KES", "phone-1", "risk-2")
	airtime = service.purchase_airtime("airtime-1", "tenant-test", airtime_payment["id"], "operator-1", "phone-1")
	request = service.open_service_request("request-1", "tenant-test", customer["id"], "device_change", "reviewer-1", ["device-1"])
	notification = service.set_notification_preference("notification-1", "tenant-test", customer["id"], "push", "notification-consent-1")
	fraud = service.record_fraud_event("fraud-1", "tenant-test", customer["id"], "medium", ["signal-1"])
	agent = service.register_mobile_agent("agent-1", "tenant-test", "Mobile Agent", "codex", "mobile_fraud_reviewer", "review fraud events")
	batch = service.validate_batch("tenant-test", 6)
	summary = service.dashboard_summary("tenant-test")

	assert program["currency"] == "KES"
	assert len(device["fingerprint"]) == 16
	assert factor["factor_type"] == "biometric"
	assert payment["direction"] == "debit"
	assert bill["biller_reference"] == "biller-1"
	assert airtime["operator_reference"] == "operator-1"
	assert request["status"] == "open"
	assert notification["enabled"] is True
	assert fraud["severity"] == "medium"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["payment_count"] == 3
	assert summary["audit_event_count"] == 14


def test_service_guardrails_reject_invalid_mobile_actions():
	service_module = _load_module("guardrail_service_fintech_mobile", PACKAGE_DIR / "service.py")
	service = service_module.MobileBankingService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_program("program", "", "Mobile", "owner", "KE", "KES", ["ios"])
	with pytest.raises(PermissionError, match="program_owner_required"):
		service.register_program("program", "tenant-test", "Mobile", "", "KE", "KES", ["ios"])
	service.register_program("program-ok", "tenant-test", "Mobile", "owner", "KE", "KES", ["ios"])
	with pytest.raises(PermissionError, match="customer_kyc_required"):
		service.enroll_customer("customer", "tenant-test", "crm", "KE", "", "consent", "aml", "fraud")
	customer = service.enroll_customer("customer-ok", "tenant-test", "crm", "KE", "kyc", "consent", "aml", "fraud")
	with pytest.raises(PermissionError, match="device_attestation_required"):
		service.bind_device("device", "tenant-test", customer["id"], "ios", "fingerprint", "", "low")
	device = service.bind_device("device-ok", "tenant-test", customer["id"], "ios", "fingerprint", "attestation", "low")
	with pytest.raises(PermissionError, match="auth_factor_type_not_supported"):
		service.register_auth_factor("factor", "tenant-test", customer["id"], device["id"], "unsupported", "strength")
	with pytest.raises(PermissionError, match="provider_reference_required"):
		service.link_account("link", "tenant-test", customer["id"], "deposit", "account", "KES", "")
	link = service.link_account("link-ok", "tenant-test", customer["id"], "deposit", "account", "KES", "provider")
	with pytest.raises(PermissionError, match="payment_link_currency_mismatch"):
		service.initiate_payment("payment-currency", "tenant-test", customer["id"], device["id"], link["id"], "peer_transfer", 10, "USD", "recipient", "risk")
	with pytest.raises(PermissionError, match="payment_approval_required"):
		service.initiate_payment("payment-high", "tenant-test", customer["id"], device["id"], link["id"], "peer_transfer", 150000, "KES", "recipient", "risk")
	payment = service.initiate_payment("payment-ok", "tenant-test", customer["id"], device["id"], link["id"], "peer_transfer", 10, "KES", "recipient", "risk")
	with pytest.raises(PermissionError, match="bill_payment_type_required"):
		service.record_bill_payment("bill-wrong-type", "tenant-test", payment["id"], "biller", "bill-account")
	with pytest.raises(PermissionError, match="biller_reference_required"):
		bill_payment = service.initiate_payment("payment-bill", "tenant-test", customer["id"], device["id"], link["id"], "bill_payment", 10, "KES", "biller", "risk")
		service.record_bill_payment("bill", "tenant-test", bill_payment["id"], "", "bill-account")
	with pytest.raises(PermissionError, match="airtime_payment_type_required"):
		service.purchase_airtime("airtime-wrong-type", "tenant-test", payment["id"], "operator", "phone")
	with pytest.raises(PermissionError, match="airtime_operator_required"):
		airtime_payment = service.initiate_payment("payment-airtime", "tenant-test", customer["id"], device["id"], link["id"], "airtime", 10, "KES", "phone", "risk")
		service.purchase_airtime("airtime", "tenant-test", airtime_payment["id"], "", "phone")
	with pytest.raises(PermissionError, match="service_evidence_required"):
		service.open_service_request("request", "tenant-test", customer["id"], "device_change", "reviewer", [])
	with pytest.raises(PermissionError, match="notification_consent_required"):
		service.set_notification_preference("notification", "tenant-test", customer["id"], "push", "")
	with pytest.raises(PermissionError, match="fraud_approval_required"):
		service.record_fraud_event("fraud", "tenant-test", customer["id"], "critical", ["signal"], "")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="mobile_agent_runtime_not_supported"):
		service.register_mobile_agent("agent", "tenant-test", "Bad Agent", "unsupported", "mobile_fraud_reviewer", "scope")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_mobile", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_mobile", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_mobile", PACKAGE_DIR / "app.py")

	api.register_program({"tenant_id": "tenant-api", "program_id": "api-program", "name": "Mobile", "owner_id": "owner", "country": "KE", "currency": "KES", "platforms": ["ios"]})
	customer = api.enroll_customer({"tenant_id": "tenant-api", "customer_id": "api-customer", "customer_reference": "crm", "country": "KE", "kyc_reference": "kyc", "consent_reference": "consent", "aml_reference": "aml", "fraud_reference": "fraud"})
	device = api.bind_device({"tenant_id": "tenant-api", "device_id": "api-device", "customer_id": customer["id"], "platform": "ios", "fingerprint": "fingerprint", "attestation_reference": "attestation", "risk_tier": "low"})
	link = api.link_account({"tenant_id": "tenant-api", "link_id": "api-link", "customer_id": customer["id"], "link_type": "deposit", "account_reference": "account", "currency": "KES", "provider_reference": "provider"})
	api.initiate_payment({"tenant_id": "tenant-api", "payment_id": "api-payment", "customer_id": customer["id"], "device_id": device["id"], "account_link_id": link["id"], "payment_type": "peer_transfer", "amount": 100, "currency": "KES", "recipient_reference": "recipient", "risk_reference": "risk"})
	agent = api.register_mobile_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Mobile Agent", "runtime": "claude_code", "role": "mobile_payments_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.mobile_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "mobile_payments_reviewer"
	assert dashboard["summary"]["payment_count"] == 1
	assert console["payments"][0]["id"] == "api-payment"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_mobile"]["screens"]["agents"]["route"] == "/fintech-mobile/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_mobile", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_mobile"]["streaming"]["processor"] == "bytewax"
