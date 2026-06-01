"""Executable Embedded Finance capability package tests."""

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
	module = _load_module("contract_fintech_embedded", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_embedded"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "embedded_payment_workflow" in contract["provides"]
	assert "/fintech-embedded/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_fintech_embedded", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "embedded_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "embedded_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_embedded_finance_lifecycle():
	service_module = _load_module("service_fintech_embedded", PACKAGE_DIR / "service.py")
	service = service_module.EmbeddedFinanceService()

	program = service.register_partner_program("program-1", "tenant-test", "Merchant Partner", "kyb-1", "contract-1", "risk-1")
	application = service.register_host_application("app-1", "tenant-test", program["id"], "Merchant Checkout", "production", "https://merchant.example", "terms-1")
	placement = service.publish_product_placement("placement-1", "tenant-test", application["id"], "payments", "checkout", ["payments.write"], "risk-policy-1")
	consent = service.capture_customer_consent("consent-1", "tenant-test", application["id"], "customer-1", ["payments.write"], "2026-12-31")
	account = service.open_embedded_account("account-1", "tenant-test", application["id"], "customer-1", "wallet-1", "kyc-1")
	payment = service.initiate_embedded_payment("payment-1", "tenant-test", application["id"], placement["id"], consent["id"], "wallet-1", "merchant-1", 1250, "usd", "risk-pay-1")
	card = service.offer_embedded_card("card-1", "tenant-test", application["id"], "customer-1", 50000, "risk-card-1")
	offer = service.create_lending_offer("offer-1", "tenant-test", application["id"], "customer-1", 250000, "affordability-1", "underwriting-1")
	settlement = service.close_settlement_batch("settlement-1", "tenant-test", program["id"], 1250, "USD", "recon-1")
	share = service.record_revenue_share("share-1", "tenant-test", program["id"], 12.5, "contract-1")
	agent = service.register_embedded_agent("agent-1", "tenant-test", "Embedded Agent", "codex", "settlement_reviewer", "review settlements")
	batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert application["domain"] == "merchant.example"
	assert placement["product_type"] == "payments"
	assert account["public_account_reference"].startswith("acct_")
	assert payment["currency"] == "USD"
	assert card["limit_minor"] == 50000
	assert offer["underwriting_reference"] == "underwriting-1"
	assert settlement["status"] == "closed"
	assert share["percent"] == 12.5
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["payment_count"] == 1
	assert summary["audit_event_count"] == 11


def test_service_guardrails_reject_invalid_embedded_actions():
	service_module = _load_module("guardrail_service_fintech_embedded", PACKAGE_DIR / "service.py")
	service = service_module.EmbeddedFinanceService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_partner_program("program", "", "Partner", "kyb", "contract", "risk")
	with pytest.raises(PermissionError, match="partner_kyb_required"):
		service.register_partner_program("program", "tenant-test", "Partner", "", "contract", "risk")
	program = service.register_partner_program("program-ok", "tenant-test", "Partner", "kyb", "contract", "risk")
	with pytest.raises(PermissionError, match="host_domain_required"):
		service.register_host_application("app", "tenant-test", program["id"], "App", "production", "", "terms")
	application = service.register_host_application("app-ok", "tenant-test", program["id"], "App", "production", "merchant.example", "terms")
	with pytest.raises(PermissionError, match="embedded_product_not_supported"):
		service.publish_product_placement("placement", "tenant-test", application["id"], "unsupported", "checkout", ["payments.write"], "risk")
	placement = service.publish_product_placement("placement-ok", "tenant-test", application["id"], "payments", "checkout", ["payments.write"], "risk")
	with pytest.raises(PermissionError, match="consent_expiry_required"):
		service.capture_customer_consent("consent", "tenant-test", application["id"], "customer", ["payments.write"], "")
	consent = service.capture_customer_consent("consent-ok", "tenant-test", application["id"], "customer", ["wallet.read"], "2026-12-31")
	with pytest.raises(PermissionError, match="payment_scope_not_consented"):
		service.initiate_embedded_payment("payment-unconsented", "tenant-test", application["id"], placement["id"], consent["id"], "wallet", "merchant", 100, "USD", "risk")
	payment_consent = service.capture_customer_consent("consent-pay", "tenant-test", application["id"], "customer", ["payments.write"], "2026-12-31")
	other_program = service.register_partner_program("program-other", "tenant-test", "Other", "kyb", "contract", "risk")
	other_app = service.register_host_application("app-other", "tenant-test", other_program["id"], "Other", "production", "other.example", "terms")
	other_placement = service.publish_product_placement("placement-other", "tenant-test", other_app["id"], "payments", "checkout", ["payments.write"], "risk")
	with pytest.raises(PermissionError, match="payment_placement_application_mismatch"):
		service.initiate_embedded_payment("payment-mismatch", "tenant-test", application["id"], other_placement["id"], payment_consent["id"], "wallet", "merchant", 100, "USD", "risk")
	with pytest.raises(PermissionError, match="positive_payment_amount_required"):
		service.initiate_embedded_payment("payment", "tenant-test", application["id"], placement["id"], payment_consent["id"], "wallet", "merchant", 0, "USD", "risk")
	with pytest.raises(PermissionError, match="settlement_reconciliation_required"):
		service.close_settlement_batch("settlement", "tenant-test", program["id"], 100, "USD", "")
	with pytest.raises(PermissionError, match="revenue_share_percent_out_of_bounds"):
		service.record_revenue_share("share", "tenant-test", program["id"], 150, "contract")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="embedded_agent_runtime_not_supported"):
		service.register_embedded_agent("agent", "tenant-test", "Bad Agent", "unsupported", "settlement_reviewer", "scope")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_embedded", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_embedded", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_embedded", PACKAGE_DIR / "app.py")

	program = api.register_partner_program({"tenant_id": "tenant-api", "program_id": "api-program", "name": "Partner", "kyb_reference": "kyb", "contract_reference": "contract", "risk_reference": "risk"})
	application = api.register_host_application({"tenant_id": "tenant-api", "application_id": "api-app", "program_id": program["id"], "name": "App", "environment": "production", "domain": "app.example", "terms_reference": "terms"})
	placement = api.publish_product_placement({"tenant_id": "tenant-api", "placement_id": "api-placement", "application_id": application["id"], "product_type": "payments", "channel": "checkout", "scopes": ["payments.write"], "risk_policy_reference": "risk-policy"})
	consent = api.capture_customer_consent({"tenant_id": "tenant-api", "consent_id": "api-consent", "application_id": application["id"], "customer_reference": "customer", "scopes": ["payments.write"], "expiry_date": "2026-12-31"})
	api.initiate_embedded_payment({"tenant_id": "tenant-api", "payment_id": "api-payment", "application_id": application["id"], "placement_id": placement["id"], "consent_id": consent["id"], "source_reference": "wallet", "destination_reference": "merchant", "amount_minor": 100, "currency": "USD", "risk_reference": "risk"})
	agent = api.register_embedded_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Agent", "runtime": "claude_code", "role": "embedded_compliance_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.embedded_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "embedded_compliance_reviewer"
	assert dashboard["summary"]["payment_count"] == 1
	assert console["payments"][0]["id"] == "api-payment"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_embedded"]["screens"]["agents"]["route"] == "/fintech-embedded/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_embedded", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_embedded"]["streaming"]["processor"] == "bytewax"
