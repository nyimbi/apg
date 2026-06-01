"""Executable Buy Now Pay Later capability package tests."""

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
	module = _load_module("contract_fintech_bnpl", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_bnpl"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "affordability_decisioning" in contract["provides"]
	assert "/fintech-bnpl/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_high_value_checkout():
	module = _load_module("rules_fintech_bnpl", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "bnpl_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "create_checkout_session", "high_value": True, "human_review_recorded": False})["decision"] == "require_review"


def test_service_executes_bnpl_lifecycle():
	service_module = _load_module("service_fintech_bnpl", PACKAGE_DIR / "service.py")
	service = service_module.BNPLService()

	program = service.register_merchant_program("program-1", "tenant-test", "Everyday BNPL", "bnpl-ops", "KE", "KES", "settlement-policy-1", "fee-disclosure-1", 4)
	consumer = service.onboard_consumer("consumer-1", "tenant-test", "crm-1", "kyc-1", "KE", "consent-1", "aml-1", "fraud-1")
	merchant = service.register_merchant("merchant-1", "tenant-test", program["id"], "legal-1", "retail", "KE", "standard", "settlement-account-1")
	checkout = service.create_checkout_session("checkout-1", "tenant-test", merchant["id"], consumer["id"], "mobile", "retail", 12000, "KES", "payment-1", "fraud-1", "aml-1", "consent-1")
	decision = service.record_affordability_decision("decision-1", "tenant-test", checkout["id"], 760, "approve", ["income-1", "bureau-1"], "reviewer-1")
	plan = service.create_bnpl_plan("plan-1", "tenant-test", checkout["id"], decision["id"], "pay_in_4", 12000, "KES", 45, 0, "fee-disclosure-1", "acceptance-1")
	installment = service.schedule_installment("installment-1", "tenant-test", plan["id"], 3000, "2026-07-01", "scheduled", 1)
	settlement = service.record_merchant_settlement("settlement-1", "tenant-test", merchant["id"], plan["id"], 12000, 11640, "released", "recon-1", "rail-1")
	dispute = service.open_bnpl_dispute("dispute-1", "tenant-test", plan["id"], "quality_issue", "reviewer-2", ["photo-1", "chat-1"])
	agent = service.register_bnpl_agent("agent-1", "tenant-test", "BNPL Agent", "codex", "affordability_reviewer", "review affordability decisions")
	batch = service.validate_batch("tenant-test", 5)
	estimate = service.estimate_plan_installment(plan["id"], "tenant-test")
	summary = service.dashboard_summary("tenant-test")

	assert program["currency"] == "KES"
	assert checkout["channel"] == "mobile"
	assert decision["decision"] == "approve"
	assert plan["plan_type"] == "pay_in_4"
	assert installment["due_amount"] == 3000
	assert settlement["net_amount"] == 11640
	assert dispute["status"] == "open"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert estimate["installment"] == 3000
	assert summary["plan_count"] == 1
	assert summary["audit_event_count"] == 10


def test_service_guardrails_reject_invalid_bnpl_actions():
	service_module = _load_module("guardrail_service_fintech_bnpl", PACKAGE_DIR / "service.py")
	service = service_module.BNPLService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_merchant_program("program", "", "BNPL", "owner", "KE", "KES", "settlement", "fees", 4)
	with pytest.raises(PermissionError, match="program_owner_required"):
		service.register_merchant_program("program", "tenant-test", "BNPL", "", "KE", "KES", "settlement", "fees", 4)
	program = service.register_merchant_program("program-ok", "tenant-test", "BNPL", "owner", "KE", "KES", "settlement", "fees", 4)
	with pytest.raises(PermissionError, match="consumer_kyc_required"):
		service.onboard_consumer("consumer", "tenant-test", "crm", "", "KE", "consent", "aml", "fraud")
	consumer = service.onboard_consumer("consumer-ok", "tenant-test", "crm", "kyc", "KE", "consent", "aml", "fraud")
	with pytest.raises(PermissionError, match="merchant_category_not_supported"):
		service.register_merchant("merchant", "tenant-test", program["id"], "legal", "unsupported", "KE", "standard", "settlement")
	merchant = service.register_merchant("merchant-ok", "tenant-test", program["id"], "legal", "retail", "KE", "standard", "settlement")
	with pytest.raises(PermissionError, match="payment_reference_required"):
		service.create_checkout_session("checkout", "tenant-test", merchant["id"], consumer["id"], "mobile", "retail", 100, "KES", "", "fraud", "aml", "consent")
	with pytest.raises(PermissionError, match="checkout_review_required"):
		service.create_checkout_session("checkout-high", "tenant-test", merchant["id"], consumer["id"], "mobile", "retail", 150000, "KES", "payment", "fraud", "aml", "consent")
	checkout = service.create_checkout_session("checkout-ok", "tenant-test", merchant["id"], consumer["id"], "mobile", "retail", 100, "KES", "payment", "fraud", "aml", "consent")
	with pytest.raises(PermissionError, match="affordability_approval_required"):
		service.record_affordability_decision("decision", "tenant-test", checkout["id"], 700, "approve", ["income"], "")
	decision = service.record_affordability_decision("decision-ok", "tenant-test", checkout["id"], 700, "approve", ["income"], "approval")
	with pytest.raises(PermissionError, match="bnpl_plan_type_not_supported"):
		service.create_bnpl_plan("plan", "tenant-test", checkout["id"], decision["id"], "unsupported", 100, "KES", 30, 0, "fees", "acceptance")
	with pytest.raises(PermissionError, match="down_payment_invalid"):
		service.create_bnpl_plan("plan-bad-down", "tenant-test", checkout["id"], decision["id"], "pay_in_3", 100, "KES", 30, 150, "fees", "acceptance")
	plan = service.create_bnpl_plan("plan-ok", "tenant-test", checkout["id"], decision["id"], "pay_in_3", 100, "KES", 30, 0, "fees", "acceptance")
	with pytest.raises(PermissionError, match="installment_due_date_required"):
		service.schedule_installment("installment", "tenant-test", plan["id"], 33.33, "")
	with pytest.raises(PermissionError, match="settlement_reconciliation_required"):
		service.record_merchant_settlement("settlement", "tenant-test", merchant["id"], plan["id"], 100, 95, "released", "", "rail")
	with pytest.raises(PermissionError, match="dispute_evidence_required"):
		service.open_bnpl_dispute("dispute", "tenant-test", plan["id"], "quality_issue", "reviewer", [])
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="bnpl_agent_runtime_not_supported"):
		service.register_bnpl_agent("agent", "tenant-test", "Bad Agent", "unsupported", "affordability_reviewer", "scope")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_bnpl", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_bnpl", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_bnpl", PACKAGE_DIR / "app.py")

	program = api.register_merchant_program({"tenant_id": "tenant-api", "program_id": "api-program", "name": "BNPL", "owner_id": "owner", "country": "KE", "currency": "KES", "settlement_policy_reference": "settlement", "fee_disclosure_reference": "fees", "max_installments": 4})
	consumer = api.onboard_consumer({"tenant_id": "tenant-api", "consumer_id": "api-consumer", "customer_reference": "crm-api", "kyc_profile_id": "kyc-api", "country": "KE", "consent_reference": "consent-api", "aml_reference": "aml-api", "fraud_reference": "fraud-api"})
	merchant = api.register_merchant({"tenant_id": "tenant-api", "merchant_id": "api-merchant", "program_id": program["id"], "legal_entity_reference": "legal-api", "category": "retail", "country": "KE", "risk_tier": "standard", "settlement_account": "settlement-api"})
	checkout = api.create_checkout_session({"tenant_id": "tenant-api", "checkout_id": "api-checkout", "merchant_id": merchant["id"], "consumer_id": consumer["id"], "channel": "mobile", "category": "retail", "amount": 1000, "currency": "KES", "payment_reference": "payment-api", "fraud_reference": "fraud-api", "aml_reference": "aml-api", "consent_reference": "consent-api"})
	decision = api.record_affordability_decision({"tenant_id": "tenant-api", "decision_id": "api-decision", "checkout_id": checkout["id"], "score": 720, "decision": "approve", "evidence_references": ["income-api"], "human_approval": "approval-api"})
	api.create_bnpl_plan({"tenant_id": "tenant-api", "plan_id": "api-plan", "checkout_id": checkout["id"], "affordability_id": decision["id"], "plan_type": "pay_in_4", "principal": 1000, "currency": "KES", "term_days": 45, "fee_disclosure_reference": "fees-api", "customer_acceptance_reference": "acceptance-api"})
	agent = api.register_bnpl_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "BNPL Agent", "runtime": "claude_code", "role": "settlement_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.bnpl_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "settlement_reviewer"
	assert dashboard["summary"]["plan_count"] == 1
	assert console["plans"][0]["id"] == "api-plan"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_bnpl"]["screens"]["agents"]["route"] == "/fintech-bnpl/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_bnpl", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_bnpl"]["streaming"]["processor"] == "bytewax"
