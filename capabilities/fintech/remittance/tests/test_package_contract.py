"""Executable Cross-Border Remittance capability package tests."""

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
	module = _load_module("contract_fintech_remittance", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_remittance"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "cross_border_transfer_workflow" in contract["provides"]
	assert "/fintech-remittance/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_high_impact_transfers():
	module = _load_module("rules_fintech_remittance", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "remittance_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "create_transfer", "high_value": True, "human_approval_recorded": False})["decision"] == "require_review"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "create_transfer", "sanctions_hit": True})["decision"] == "deny"


def test_service_executes_remittance_lifecycle():
	service_module = _load_module("service_fintech_remittance", PACKAGE_DIR / "service.py")
	service = service_module.RemittanceService()

	quote = service.create_quote("quote-1", "tenant-test", "KE", "UG", "KES", "UGX", 1000, 28.5, 20, "2026-06-02T00:00:00Z")
	transfer = service.create_transfer("transfer-1", "tenant-test", quote["id"], "sender-a", "beneficiary-a", "sender-kyc", "beneficiary-kyc", "wallet-hold", "mobile_money", "family_support", "salary", "aml-clear", "clear")
	payout = service.release_payout(transfer["id"], "tenant-test", "provider-receipt", "settlement-1")
	refund = service.file_refund("refund-1", "tenant-test", transfer["id"], "customer request", "reviewer-1")
	agent = service.register_remittance_agent("agent-1", "tenant-test", "Remittance Agent", "codex", "remittance_ops_reviewer", "review transfers")
	batch = service.validate_batch("tenant-test", 5)
	summary = service.dashboard_summary("tenant-test")

	assert quote["corridor"] == "KE-UG:KES-UGX"
	assert transfer["status"] == "ready_for_payout"
	assert payout["status"] == "paid"
	assert refund["status"] == "filed"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["quote_count"] == 1
	assert summary["transfer_count"] == 1
	assert summary["refund_count"] == 1
	assert summary["audit_event_count"] == 5


def test_service_guardrails_reject_invalid_remittance_actions():
	service_module = _load_module("guardrail_service_fintech_remittance", PACKAGE_DIR / "service.py")
	service = service_module.RemittanceService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_quote("quote", "", "KE", "UG", "KES", "UGX", 100, 28, 1, "expiry")
	with pytest.raises(PermissionError, match="cross_border_corridor_required"):
		service.create_quote("quote", "tenant-test", "KE", "KE", "KES", "KES", 100, 1, 1, "expiry")
	with pytest.raises(PermissionError, match="positive_fx_rate_required"):
		service.create_quote("quote", "tenant-test", "KE", "UG", "KES", "UGX", 100, 0, 1, "expiry")
	quote = service.create_quote("quote-ok", "tenant-test", "KE", "UG", "KES", "UGX", 100000, 28, 0, "expiry")
	with pytest.raises(PermissionError, match="sender_kyc_required"):
		service.create_transfer("transfer", "tenant-test", quote["id"], "sender", "beneficiary", "", "beneficiary-kyc", "funding", "mobile_money", "family_support", "salary", "aml", "clear")
	with pytest.raises(PermissionError, match="sanctions_hit_blocked"):
		service.create_transfer("transfer", "tenant-test", quote["id"], "sender", "beneficiary", "sender-kyc", "beneficiary-kyc", "funding", "mobile_money", "family_support", "salary", "aml", "clear", sanctions_hit=True)
	with pytest.raises(PermissionError, match="fraud_review_approval_required"):
		service.create_transfer("transfer", "tenant-test", quote["id"], "sender", "beneficiary", "sender-kyc", "beneficiary-kyc", "funding", "mobile_money", "family_support", "salary", "aml", "review")
	transfer = service.create_transfer("transfer-ok", "tenant-test", quote["id"], "sender", "beneficiary", "sender-kyc", "beneficiary-kyc", "funding", "mobile_money", "family_support", "salary", "aml", "clear", human_approval="approval-1")
	with pytest.raises(PermissionError, match="provider_receipt_required"):
		service.release_payout(transfer["id"], "tenant-test", "", "settlement")
	with pytest.raises(PermissionError, match="refund_reviewer_required"):
		service.file_refund("refund", "tenant-test", transfer["id"], "failed payout", "")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="remittance_agent_runtime_not_supported"):
		service.register_remittance_agent("agent", "tenant-test", "Bad Agent", "unsupported", "remittance_ops_reviewer", "scope")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_remittance", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_remittance", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_remittance", PACKAGE_DIR / "app.py")

	quote = api.create_quote({"tenant_id": "tenant-api", "quote_id": "api-quote", "source_country": "KE", "destination_country": "UG", "source_currency": "KES", "destination_currency": "UGX", "send_amount": 500, "fx_rate": 28, "fee_amount": 2, "expiry": "expiry"})
	api.create_transfer({"tenant_id": "tenant-api", "transfer_id": "api-transfer", "quote_id": quote["id"], "sender_reference": "sender-api", "beneficiary_reference": "beneficiary-api", "sender_kyc_id": "sender-kyc", "beneficiary_kyc_id": "beneficiary-kyc", "funding_reference": "funding-api", "payout_method": "mobile_money", "purpose_code": "family_support", "source_of_funds": "salary", "aml_screen_id": "aml-api", "fraud_decision": "clear"})
	agent = api.register_remittance_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Remittance Agent", "runtime": "claude_code", "role": "compliance_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.transfer_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "compliance_reviewer"
	assert dashboard["summary"]["transfer_count"] == 1
	assert console["transfers"][0]["id"] == "api-transfer"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_remittance"]["screens"]["agents"]["route"] == "/fintech-remittance/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_remittance", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_remittance"]["streaming"]["processor"] == "bytewax"
