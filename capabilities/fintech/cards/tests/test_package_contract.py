"""Executable Digital Cards capability package tests."""

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
	module = _load_module("contract_fintech_cards", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_cards"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "card_authorization_control" in contract["provides"]
	assert "/fintech-cards/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_high_impact_authorizations():
	module = _load_module("rules_fintech_cards", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "card_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "authorize_transaction", "high_impact": True, "human_approval_recorded": False})["decision"] == "require_review"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "authorize_transaction", "fraud_blocked": True})["decision"] == "deny"


def test_service_executes_card_lifecycle():
	service_module = _load_module("service_fintech_cards", PACKAGE_DIR / "service.py")
	service = service_module.CardService()

	program = service.register_program("program-1", "tenant-test", "Everyday Debit", "issuer-ops", "411111", "KES", "settlement-1")
	holder = service.onboard_cardholder("holder-1", "tenant-test", "customer-1", "kyc-1", "KE")
	card = service.issue_card("card-1", "tenant-test", program["id"], holder["id"], "virtual", "debit", "wallet-1", "funding-1", "consent-1")
	token = service.provision_token("token-1", "tenant-test", card["id"], "wallet", "tok-1", "key-domain-1", "device-1")
	auth = service.authorize_transaction("auth-1", "tenant-test", card["id"], 500, "KES", "grocery", "fraud-clear", "aml-clear")
	dispute = service.file_dispute("dispute-1", "tenant-test", auth["id"], "fraud", ["auth-1"], "reviewer-1")
	agent = service.register_card_agent("agent-1", "tenant-test", "Card Agent", "codex", "card_ops_reviewer", "review cards")
	batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert card["masked_pan"].startswith("411111")
	assert token["status"] == "active"
	assert auth["decision"] == "approve"
	assert dispute["status"] == "filed"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["program_count"] == 1
	assert summary["card_count"] == 1
	assert summary["authorization_count"] == 1
	assert summary["audit_event_count"] == 7


def test_service_guardrails_reject_invalid_card_actions():
	service_module = _load_module("guardrail_service_fintech_cards", PACKAGE_DIR / "service.py")
	service = service_module.CardService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_program("program", "", "Cards", "owner", "411111", "KES", "settlement")
	with pytest.raises(PermissionError, match="bin_range_required"):
		service.register_program("program", "tenant-test", "Cards", "owner", "", "KES", "settlement")
	program = service.register_program("program-ok", "tenant-test", "Cards", "owner", "411111", "KES", "settlement")
	with pytest.raises(PermissionError, match="cardholder_kyc_required"):
		service.onboard_cardholder("holder", "tenant-test", "customer", "", "KE")
	holder = service.onboard_cardholder("holder-ok", "tenant-test", "customer", "kyc", "KE")
	with pytest.raises(PermissionError, match="shipping_address_required"):
		service.issue_card("card", "tenant-test", program["id"], holder["id"], "physical", "debit", "wallet", "funding", "consent")
	card = service.issue_card("card-ok", "tenant-test", program["id"], holder["id"], "virtual", "debit", "wallet", "funding", "consent")
	with pytest.raises(PermissionError, match="key_domain_required"):
		service.provision_token("token", "tenant-test", card["id"], "wallet", "tok", "", "device")
	with pytest.raises(PermissionError, match="fraud_blocked_authorization"):
		service.authorize_transaction("auth", "tenant-test", card["id"], 10, "KES", "grocery", "fraud", "aml", fraud_decision="block")
	with pytest.raises(PermissionError, match="authorization_approval_required"):
		service.authorize_transaction("auth", "tenant-test", card["id"], 100001, "KES", "grocery", "fraud", "aml")
	with pytest.raises(PermissionError, match="dispute_evidence_required"):
		service.file_dispute("dispute", "tenant-test", "auth", "fraud", [], "reviewer")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="card_agent_runtime_not_supported"):
		service.register_card_agent("agent", "tenant-test", "Bad Agent", "unsupported", "card_ops_reviewer", "scope")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_cards", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_cards", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_cards", PACKAGE_DIR / "app.py")

	program = api.register_program({"tenant_id": "tenant-api", "program_id": "api-program", "name": "Cards", "owner_id": "owner", "bin_range": "411111", "currency": "KES", "settlement_account": "settlement"})
	holder = api.onboard_cardholder({"tenant_id": "tenant-api", "cardholder_id": "api-holder", "customer_reference": "customer-api", "kyc_profile_id": "kyc-api", "country": "KE"})
	card = api.issue_card({"tenant_id": "tenant-api", "card_id": "api-card", "program_id": program["id"], "cardholder_id": holder["id"], "card_type": "virtual", "product": "debit", "wallet_reference": "wallet-api", "funding_account": "funding-api", "consent_reference": "consent-api"})
	api.authorize_transaction({"tenant_id": "tenant-api", "authorization_id": "api-auth", "card_id": card["id"], "amount": 50, "currency": "KES", "merchant_category": "grocery", "fraud_reference": "fraud-api", "aml_reference": "aml-api"})
	agent = api.register_card_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Card Agent", "runtime": "claude_code", "role": "authorization_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.card_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "authorization_reviewer"
	assert dashboard["summary"]["card_count"] == 1
	assert console["cards"][0]["id"] == "api-card"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_cards"]["screens"]["agents"]["route"] == "/fintech-cards/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_cards", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_cards"]["streaming"]["processor"] == "bytewax"
