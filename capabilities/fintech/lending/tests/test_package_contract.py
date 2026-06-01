"""Executable Digital Lending capability package tests."""

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
	module = _load_module("contract_fintech_lending", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_lending"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "underwriting_decisioning" in contract["provides"]
	assert "/fintech-lending/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_credit_reviews():
	module = _load_module("rules_fintech_lending", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "lending_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "submit_application", "high_amount": True, "human_review_recorded": False})["decision"] == "require_review"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "record_underwriting", "final_decision": True, "human_approval_recorded": False})["decision"] == "require_review"


def test_service_executes_lending_lifecycle():
	service_module = _load_module("service_fintech_lending", PACKAGE_DIR / "service.py")
	service = service_module.LendingService()

	product = service.register_product("product-1", "tenant-test", "SME Working Capital", "credit-ops", "term_loan", "KES", 1000, 200000, 30, 365, 0.24, "monthly")
	borrower = service.onboard_borrower("borrower-1", "tenant-test", "customer-1", "kyc-1", "KE", "income-1", "consent-1")
	application = service.submit_application("application-1", "tenant-test", borrower["id"], product["id"], 50000, "working_capital", "affordability-1", "statement-1", "aml-1", "fraud-1", "card-activity-1")
	decision = service.record_underwriting("uw-1", "tenant-test", application["id"], 720, "approve", ["scorecard-1", "bank-analysis-1"], "credit-manager-1")
	offer = service.issue_offer("offer-1", "tenant-test", application["id"], decision["id"], 50000, 0.24, 180, "2026-07-01", "accepted", "acceptance-1")
	disbursement = service.record_disbursement("disb-1", "tenant-test", offer["id"], 50000, "wallet", "funding-1", "wallet-1", "treasury-approval-1")
	repayment = service.schedule_repayment("schedule-1", "tenant-test", offer["id"], 9800, "2026-08-01", "monthly", 6)
	case = service.open_collection_case("case-1", "tenant-test", repayment["id"], "missed_payment", "collections-1", "contact-policy-1")
	agent = service.register_lending_agent("agent-1", "tenant-test", "Lending Agent", "codex", "underwriting_reviewer", "review lending")
	batch = service.validate_batch("tenant-test", 5)
	summary = service.dashboard_summary("tenant-test")
	installment = service.estimate_offer_installment(offer["id"], "tenant-test")

	assert application["currency"] == "KES"
	assert decision["decision"] == "approve"
	assert offer["status"] == "accepted"
	assert disbursement["rail"] == "wallet"
	assert repayment["status"] == "scheduled"
	assert case["status"] == "open"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert installment["installment"] > 0
	assert summary["product_count"] == 1
	assert summary["application_count"] == 1
	assert summary["offer_count"] == 1
	assert summary["audit_event_count"] == 9


def test_service_guardrails_reject_invalid_lending_actions():
	service_module = _load_module("guardrail_service_fintech_lending", PACKAGE_DIR / "service.py")
	service = service_module.LendingService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_product("product", "", "Loan", "owner", "term_loan", "KES", 1000, 10000, 30, 365, 0.2, "monthly")
	with pytest.raises(PermissionError, match="loan_product_owner_required"):
		service.register_product("product", "tenant-test", "Loan", "", "term_loan", "KES", 1000, 10000, 30, 365, 0.2, "monthly")
	product = service.register_product("product-ok", "tenant-test", "Loan", "owner", "term_loan", "KES", 1000, 10000, 30, 365, 0.2, "monthly")
	with pytest.raises(PermissionError, match="borrower_kyc_required"):
		service.onboard_borrower("borrower", "tenant-test", "customer", "", "KE", "income", "consent")
	borrower = service.onboard_borrower("borrower-ok", "tenant-test", "customer", "kyc", "KE", "income", "consent")
	with pytest.raises(PermissionError, match="affordability_evidence_required"):
		service.submit_application("application", "tenant-test", borrower["id"], product["id"], 5000, "working_capital", "", "statement", "aml", "fraud", "card")
	with pytest.raises(PermissionError, match="high_amount_application_review_required"):
		service.submit_application("application-high", "tenant-test", borrower["id"], product["id"], 100000, "working_capital", "afford", "statement", "aml", "fraud", "card")
	application = service.submit_application("application-ok", "tenant-test", borrower["id"], product["id"], 5000, "working_capital", "afford", "statement", "aml", "fraud", "card")
	with pytest.raises(PermissionError, match="underwriting_approval_required"):
		service.record_underwriting("uw", "tenant-test", application["id"], 700, "approve", ["scorecard"], "")
	with pytest.raises(PermissionError, match="adverse_reason_required"):
		service.record_underwriting("uw-decline", "tenant-test", application["id"], 420, "decline", ["scorecard"], "manager")
	decision = service.record_underwriting("uw-ok", "tenant-test", application["id"], 700, "approve", ["scorecard"], "manager")
	with pytest.raises(PermissionError, match="borrower_acceptance_required"):
		service.issue_offer("offer", "tenant-test", application["id"], decision["id"], 5000, 0.2, 180, "2026-07-01", "accepted")
	offer = service.issue_offer("offer-ok", "tenant-test", application["id"], decision["id"], 5000, 0.2, 180, "2026-07-01", "accepted", "acceptance")
	with pytest.raises(PermissionError, match="disbursement_approval_required"):
		service.record_disbursement("disb", "tenant-test", offer["id"], 5000, "wallet", "funding", "wallet", "")
	with pytest.raises(PermissionError, match="due_date_required"):
		service.schedule_repayment("schedule", "tenant-test", offer["id"], 1000, "", "monthly", 6)
	with pytest.raises(PermissionError, match="contact_policy_required"):
		service.open_collection_case("case", "tenant-test", "account", "missed_payment", "reviewer", "")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="lending_agent_runtime_not_supported"):
		service.register_lending_agent("agent", "tenant-test", "Bad Agent", "unsupported", "underwriting_reviewer", "scope")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_lending", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_lending", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_lending", PACKAGE_DIR / "app.py")

	product = api.register_product({"tenant_id": "tenant-api", "product_id": "api-product", "name": "Loan", "owner_id": "owner", "product_type": "term_loan", "currency": "KES", "min_amount": 1000, "max_amount": 10000, "min_term_days": 30, "max_term_days": 365, "annual_rate": 0.2, "repayment_frequency": "monthly"})
	borrower = api.onboard_borrower({"tenant_id": "tenant-api", "borrower_id": "api-borrower", "customer_reference": "customer-api", "kyc_profile_id": "kyc-api", "country": "KE", "income_evidence_id": "income-api", "consent_reference": "consent-api"})
	application = api.submit_application({"tenant_id": "tenant-api", "application_id": "api-application", "borrower_id": borrower["id"], "product_id": product["id"], "requested_amount": 5000, "purpose": "working_capital", "affordability_reference": "afford-api", "bank_statement_reference": "statement-api", "aml_reference": "aml-api", "fraud_reference": "fraud-api", "behavior_evidence_reference": "card-api"})
	decision = api.record_underwriting({"tenant_id": "tenant-api", "underwriting_id": "api-uw", "application_id": application["id"], "score": 710, "decision": "approve", "evidence_references": ["scorecard-api"], "human_approval": "manager-api"})
	offer = api.issue_offer({"tenant_id": "tenant-api", "offer_id": "api-offer", "application_id": application["id"], "underwriting_id": decision["id"], "amount": 5000, "apr": 0.2, "term_days": 180, "expiry_date": "2026-07-01", "status": "accepted", "borrower_acceptance_reference": "accept-api"})
	api.schedule_repayment({"tenant_id": "tenant-api", "schedule_id": "api-schedule", "offer_id": offer["id"], "due_amount": 950, "due_date": "2026-08-01", "frequency": "monthly", "installment_count": 6})
	agent = api.register_lending_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Lending Agent", "runtime": "claude_code", "role": "credit_risk_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.lending_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "credit_risk_reviewer"
	assert dashboard["summary"]["application_count"] == 1
	assert console["offers"][0]["id"] == "api-offer"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_lending"]["screens"]["agents"]["route"] == "/fintech-lending/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_lending", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_lending"]["streaming"]["processor"] == "bytewax"
