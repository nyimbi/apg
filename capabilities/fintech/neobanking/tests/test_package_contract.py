"""Executable Digital Neobanking capability package tests."""

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
	module = _load_module("contract_fintech_neobanking", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_neobanking"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "deposit_account_lifecycle" in contract["provides"]
	assert "/fintech-neobanking/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_high_impact_transactions():
	module = _load_module("rules_fintech_neobanking", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "neobanking_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "post_transaction", "high_impact": True, "human_approval_recorded": False})["decision"] == "require_review"


def test_service_executes_neobanking_lifecycle():
	service_module = _load_module("service_fintech_neobanking", PACKAGE_DIR / "service.py")
	service = service_module.NeobankingService()

	program = service.register_program("program-1", "tenant-test", "Everyday Bank", "bank-ops", "KE", "KES", "settlement-1")
	customer = service.onboard_customer("customer-1", "tenant-test", "crm-1", "kyc-1", "KE", "consent-1", "aml-1", "fraud-1")
	account = service.open_account("account-1", "tenant-test", program["id"], customer["id"], "current", "KES", 1000)
	rail = service.link_payment_rail("rail-1", "tenant-test", account["id"], "wallet", "wallet-provider-1", wallet_reference="wallet-1")
	transaction = service.post_transaction("txn-1", "tenant-test", account["id"], "deposit", 2500, "KES", "deposit-1", "risk-clear-1")
	pot = service.create_savings_pot("pot-1", "tenant-test", account["id"], "School Fees", 15000)
	statement = service.issue_statement("statement-1", "tenant-test", account["id"], "2026-06-01", "2026-06-30")
	case = service.open_service_case("case-1", "tenant-test", customer["id"], account["id"], "statement_query", "reviewer-1", ["statement-1"])
	agent = service.register_neobanking_agent("agent-1", "tenant-test", "Neobank Agent", "codex", "account_risk_reviewer", "review neobank accounts")
	batch = service.validate_batch("tenant-test", 4)
	summary = service.dashboard_summary("tenant-test")

	assert account["account_number"].startswith("KE")
	assert rail["rail"] == "wallet"
	assert transaction["direction"] == "credit"
	assert pot["target_amount"] == 15000
	assert statement["transaction_count"] == 1
	assert case["status"] == "open"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["program_count"] == 1
	assert summary["account_count"] == 1
	assert summary["audit_event_count"] == 9


def test_service_guardrails_reject_invalid_neobanking_actions():
	service_module = _load_module("guardrail_service_fintech_neobanking", PACKAGE_DIR / "service.py")
	service = service_module.NeobankingService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_program("program", "", "Bank", "owner", "KE", "KES", "settlement")
	with pytest.raises(PermissionError, match="program_owner_required"):
		service.register_program("program", "tenant-test", "Bank", "", "KE", "KES", "settlement")
	program = service.register_program("program-ok", "tenant-test", "Bank", "owner", "KE", "KES", "settlement")
	with pytest.raises(PermissionError, match="customer_kyc_required"):
		service.onboard_customer("customer", "tenant-test", "crm", "", "KE", "consent", "aml", "fraud")
	customer = service.onboard_customer("customer-ok", "tenant-test", "crm", "kyc", "KE", "consent", "aml", "fraud")
	with pytest.raises(PermissionError, match="account_type_not_supported"):
		service.open_account("account", "tenant-test", program["id"], customer["id"], "unsupported", "KES")
	account = service.open_account("account-ok", "tenant-test", program["id"], customer["id"], "current", "KES")
	with pytest.raises(PermissionError, match="provider_reference_required"):
		service.link_payment_rail("rail", "tenant-test", account["id"], "wallet", "")
	with pytest.raises(PermissionError, match="risk_reference_required"):
		service.post_transaction("txn", "tenant-test", account["id"], "deposit", 10, "KES", "ref", "")
	with pytest.raises(PermissionError, match="transaction_approval_required"):
		service.post_transaction("txn-high", "tenant-test", account["id"], "transfer_out", 50000, "KES", "ref", "risk")
	with pytest.raises(PermissionError, match="positive_target_required"):
		service.create_savings_pot("pot", "tenant-test", account["id"], "Goal", 0)
	with pytest.raises(PermissionError, match="statement_period_required"):
		service.issue_statement("statement", "tenant-test", account["id"], "", "2026-06-30")
	with pytest.raises(PermissionError, match="case_evidence_required"):
		service.open_service_case("case", "tenant-test", customer["id"], account["id"], "statement_query", "reviewer", [])
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="neobanking_agent_runtime_not_supported"):
		service.register_neobanking_agent("agent", "tenant-test", "Bad Agent", "unsupported", "account_risk_reviewer", "scope")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_neobanking", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_neobanking", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_neobanking", PACKAGE_DIR / "app.py")

	program = api.register_program({"tenant_id": "tenant-api", "program_id": "api-program", "name": "Bank", "owner_id": "owner", "country": "KE", "base_currency": "KES", "settlement_account": "settlement"})
	customer = api.onboard_customer({"tenant_id": "tenant-api", "customer_id": "api-customer", "customer_reference": "crm-api", "kyc_profile_id": "kyc-api", "country": "KE", "consent_reference": "consent-api", "aml_reference": "aml-api", "fraud_reference": "fraud-api"})
	account = api.open_account({"tenant_id": "tenant-api", "account_id": "api-account", "program_id": program["id"], "customer_id": customer["id"], "account_type": "current", "currency": "KES", "initial_balance": 100})
	api.post_transaction({"tenant_id": "tenant-api", "transaction_id": "api-txn", "account_id": account["id"], "kind": "deposit", "amount": 50, "currency": "KES", "reference": "deposit-api", "risk_reference": "risk-api"})
	agent = api.register_neobanking_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Neobank Agent", "runtime": "claude_code", "role": "payments_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.neobanking_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "payments_reviewer"
	assert dashboard["summary"]["account_count"] == 1
	assert console["accounts"][0]["id"] == "api-account"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_neobanking"]["screens"]["agents"]["route"] == "/fintech-neobanking/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_neobanking", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_neobanking"]["streaming"]["processor"] == "bytewax"
