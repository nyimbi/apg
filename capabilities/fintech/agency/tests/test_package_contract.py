"""Executable Agency Banking capability package tests."""

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
	module = _load_module("contract_fintech_agency", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_agency"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "agency_transaction_workflow" in contract["provides"]
	assert "/fintech-agency/ai-agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_high_value_transaction():
	module = _load_module("rules_fintech_agency", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "agency_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "record_transaction", "high_value": True, "human_approval_recorded": False})["decision"] == "require_review"


def test_service_executes_agency_lifecycle():
	service_module = _load_module("service_fintech_agency", PACKAGE_DIR / "service.py")
	service = service_module.AgencyBankingService()

	program = service.register_program("program-1", "tenant-test", "Rural Network", "agency-ops", "KE", "KES", "real_time", ["cash_in", "cash_out", "bill_payment"])
	outlet = service.onboard_outlet("outlet-1", "tenant-test", program["id"], "Village Shop", "retail_shop", "KE", "license-1", "location-1", "security-1", "pos_terminal", 25000)
	agent = service.accredit_agent("agent-1", "tenant-test", outlet["id"], "Jane Teller", "identity-1", "training-1", "background-1")
	float_account = service.open_float_account("float-1", "tenant-test", outlet["id"], "KES", 50000, "ledger-1")
	customer = service.onboard_customer("customer-1", "tenant-test", "crm-1", "tier_2", "kyc-1", "consent-1", "aml-1", "fraud-1")
	transaction = service.record_transaction("txn-1", "tenant-test", outlet["id"], agent["id"], customer["id"], float_account["id"], "cash_out", 2500, "KES", "pos_terminal", "customer-ref-1", "risk-1")
	movement = service.record_cash_movement("movement-1", "tenant-test", outlet["id"], "float_topup", 10000, "KES", "custodian-1")
	commission = service.settle_commission("commission-1", "tenant-test", outlet["id"], "2026-06", 25, "KES", "recon-1", "payment-1")
	dispute = service.open_dispute("dispute-1", "tenant-test", transaction["id"], "agent_error", "reviewer-1", ["receipt-1"])
	visit = service.record_supervision_visit("visit-1", "tenant-test", outlet["id"], "supervisor-1", "remediation_required", ["checklist-1"], ["cash log gap"], "remediation-1")
	ai_agent = service.register_agency_ai_agent("ai-agent-1", "tenant-test", "Agency Reviewer", "codex", "liquidity_reviewer", "review float liquidity")
	batch = service.validate_batch("tenant-test", 6)
	estimate = service.estimate_transaction_commission(transaction["id"], "tenant-test")
	summary = service.dashboard_summary("tenant-test")

	assert program["settlement_model"] == "real_time"
	assert outlet["primary_channel"] == "pos_terminal"
	assert agent["status"] == "active"
	assert service.float_accounts[float_account["id"]].available_balance == 47500
	assert transaction["service"] == "cash_out"
	assert movement["movement_type"] == "float_topup"
	assert commission["amount"] == 25
	assert dispute["status"] == "open"
	assert visit["outcome"] == "remediation_required"
	assert ai_agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert estimate["commission"] == 25
	assert summary["transaction_count"] == 1
	assert summary["audit_event_count"] == 11


def test_service_guardrails_reject_invalid_agency_actions():
	service_module = _load_module("guardrail_service_fintech_agency", PACKAGE_DIR / "service.py")
	service = service_module.AgencyBankingService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_program("program", "", "Agency", "owner", "KE", "KES", "real_time", ["cash_in"])
	with pytest.raises(PermissionError, match="program_owner_required"):
		service.register_program("program", "tenant-test", "Agency", "", "KE", "KES", "real_time", ["cash_in"])
	program = service.register_program("program-ok", "tenant-test", "Agency", "owner", "KE", "KES", "real_time", ["cash_in", "cash_out"])
	with pytest.raises(PermissionError, match="outlet_license_required"):
		service.onboard_outlet("outlet", "tenant-test", program["id"], "Shop", "retail_shop", "KE", "", "location", "security", "pos_terminal", 1000)
	outlet = service.onboard_outlet("outlet-ok", "tenant-test", program["id"], "Shop", "retail_shop", "KE", "license", "location", "security", "pos_terminal", 1000)
	with pytest.raises(PermissionError, match="agent_training_required"):
		service.accredit_agent("agent", "tenant-test", outlet["id"], "Agent", "identity", "", "background")
	agent = service.accredit_agent("agent-ok", "tenant-test", outlet["id"], "Agent", "identity", "training", "background")
	with pytest.raises(PermissionError, match="float_ledger_reference_required"):
		service.open_float_account("float", "tenant-test", outlet["id"], "KES", 1000, "")
	float_account = service.open_float_account("float-ok", "tenant-test", outlet["id"], "KES", 1000, "ledger")
	with pytest.raises(PermissionError, match="customer_kyc_required"):
		service.onboard_customer("customer", "tenant-test", "crm", "tier_2", "", "consent", "aml", "fraud")
	customer = service.onboard_customer("customer-ok", "tenant-test", "crm", "tier_2", "kyc", "consent", "aml", "fraud")
	with pytest.raises(PermissionError, match="transaction_float_currency_mismatch"):
		service.record_transaction("txn-currency", "tenant-test", outlet["id"], agent["id"], customer["id"], float_account["id"], "cash_in", 100, "USD", "pos_terminal", "customer-ref", "risk")
	with pytest.raises(PermissionError, match="insufficient_agent_float"):
		service.record_transaction("txn", "tenant-test", outlet["id"], agent["id"], customer["id"], float_account["id"], "cash_out", 2000, "KES", "pos_terminal", "customer-ref", "risk")
	with pytest.raises(PermissionError, match="transaction_approval_required"):
		service.record_transaction("txn-high", "tenant-test", outlet["id"], agent["id"], customer["id"], float_account["id"], "cash_in", 150000, "KES", "pos_terminal", "customer-ref", "risk")
	transaction = service.record_transaction("txn-ok", "tenant-test", outlet["id"], agent["id"], customer["id"], float_account["id"], "cash_in", 500, "KES", "pos_terminal", "customer-ref", "risk")
	with pytest.raises(PermissionError, match="cash_movement_custodian_required"):
		service.record_cash_movement("movement", "tenant-test", outlet["id"], "float_topup", 100, "KES", "")
	with pytest.raises(PermissionError, match="commission_reconciliation_required"):
		service.settle_commission("commission", "tenant-test", outlet["id"], "2026-06", 10, "KES", "", "payment")
	with pytest.raises(PermissionError, match="dispute_evidence_required"):
		service.open_dispute("dispute", "tenant-test", transaction["id"], "agent_error", "reviewer", [])
	with pytest.raises(PermissionError, match="remediation_plan_required"):
		service.record_supervision_visit("visit", "tenant-test", outlet["id"], "supervisor", "remediation_required", ["checklist"], ["finding"], "")
	with pytest.raises(PermissionError, match="supervision_supervisor_required"):
		service.record_supervision_visit("visit-supervisor", "tenant-test", outlet["id"], "", "passed", ["checklist"], [], "")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="agency_agent_runtime_not_supported"):
		service.register_agency_ai_agent("ai-agent", "tenant-test", "Bad Agent", "unsupported", "liquidity_reviewer", "scope")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_agency", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_agency", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_agency", PACKAGE_DIR / "app.py")

	program = api.register_program({"tenant_id": "tenant-api", "program_id": "api-program", "name": "Agency", "owner_id": "owner", "country": "KE", "currency": "KES", "settlement_model": "real_time", "services": ["cash_in", "cash_out"]})
	outlet = api.onboard_outlet({"tenant_id": "tenant-api", "outlet_id": "api-outlet", "program_id": program["id"], "name": "Shop", "outlet_type": "retail_shop", "country": "KE", "license_reference": "license", "location_reference": "location", "security_plan_reference": "security", "primary_channel": "pos_terminal", "initial_float": 1000})
	agent = api.accredit_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "outlet_id": outlet["id"], "name": "Agent", "identity_reference": "identity", "training_reference": "training", "background_check_reference": "background"})
	float_account = api.open_float_account({"tenant_id": "tenant-api", "float_account_id": "api-float", "outlet_id": outlet["id"], "currency": "KES", "opening_balance": 1000, "ledger_reference": "ledger"})
	customer = api.onboard_customer({"tenant_id": "tenant-api", "customer_id": "api-customer", "customer_reference": "crm", "tier": "tier_2", "kyc_reference": "kyc", "consent_reference": "consent", "aml_reference": "aml", "fraud_reference": "fraud"})
	api.record_transaction({"tenant_id": "tenant-api", "transaction_id": "api-txn", "outlet_id": outlet["id"], "agent_id": agent["id"], "customer_id": customer["id"], "float_account_id": float_account["id"], "service": "cash_in", "amount": 100, "currency": "KES", "channel": "pos_terminal", "customer_reference": "customer-ref", "risk_reference": "risk"})
	ai_agent = api.register_agency_ai_agent({"tenant_id": "tenant-api", "agent_id": "api-ai-agent", "name": "Agency Agent", "runtime": "claude_code", "role": "field_supervisor"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.agency_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert ai_agent["metadata"]["role"] == "field_supervisor"
	assert dashboard["summary"]["transaction_count"] == 1
	assert console["transactions"][0]["id"] == "api-txn"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_agency"]["screens"]["ai_agents"]["route"] == "/fintech-agency/ai-agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_agency", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_agency"]["streaming"]["processor"] == "bytewax"
