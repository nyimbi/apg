"""Executable Accounts Receivable capability package tests."""

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
	module = _load_module("contract_arc_accounts_receivable", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "arc_accounts_receivable"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "arc_agents" in contract["provides"]
	assert "/arc-accounts-receivable/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_rule_engine_blocks_missing_context_non_bytewax_and_overapplication():
	module = _load_module("rules_arc_accounts_receivable", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "arc_batch",
		"event_stream": "queue",
	})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "apply_cash",
		"overapplication": True,
	})["matched_rules"] == ["cash_application_blocks_overapplication"]


def test_service_executes_receivables_lifecycle():
	service_module = _load_module("service_arc_accounts_receivable", PACKAGE_DIR / "service.py")
	service = service_module.AccountsReceivableService()

	customer = service.create_customer("cust-1", "tenant-test", "CUST-001", "Customer One", "business", "USD")
	credit = service.assess_credit("credit-1", "tenant-test", customer["id"], 10000, 0.82)
	invoice = service.create_invoice(
		"inv-1",
		"tenant-test",
		customer["id"],
		"INV-001",
		"2026-05-31",
		"2026-06-30",
		[{"description": "Subscription", "quantity": 2, "unit_price": 250, "revenue_account": "4000"}],
	)
	issued = service.issue_invoice(invoice["id"], "tenant-test", "approver-1")
	collection = service.record_collection_activity("collect-1", "tenant-test", issued["id"], "email", "high", "promise requested")
	dispute = service.open_dispute("dispute-1", "tenant-test", issued["id"], "pricing", "owner-1")
	resolved = service.resolve_dispute(dispute["id"], "tenant-test", "price confirmed", "reviewer-1")
	payment = service.record_payment("pay-1", "tenant-test", customer["id"], "PAY-001", "2026-06-01", 500, "bank_transfer", "cash-1")
	application = service.apply_cash("apply-1", "tenant-test", payment["id"], invoice["id"], 500)
	agent = service.register_arc_agent("tenant-test", "Invoice Review Agent", "codex", "invoice_reviewer", "review invoices")

	summary = service.dashboard_summary("tenant-test")
	assert credit["status"] == "assessed"
	assert collection["status"] == "recorded"
	assert resolved["status"] == "resolved"
	assert application["status"] == "applied"
	assert service.invoices[invoice["id"]]["status"] == "paid"
	assert agent["runtime"] == "codex"
	assert summary["audit_event_count"] == 10
	assert summary["streaming"]["processor"] == "bytewax"


def test_service_guardrails_reject_invalid_actions():
	service_module = _load_module("guardrail_service_arc_accounts_receivable", PACKAGE_DIR / "service.py")
	service = service_module.AccountsReceivableService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_customer("cust", "", "CUST", "Customer", "business")
	with pytest.raises(PermissionError, match="customer_type_not_supported"):
		service.create_customer("cust", "tenant-test", "CUST", "Customer", "unsupported")

	customer = service.create_customer("cust", "tenant-test", "CUST", "Customer", "business")
	with pytest.raises(PermissionError, match="credit_review_required"):
		service.assess_credit("credit", "tenant-test", customer["id"], 1000, 0.2)
	invoice = service.create_invoice(
		"inv",
		"tenant-test",
		customer["id"],
		"INV",
		"2026-05-31",
		"2026-06-30",
		[{"description": "Services", "quantity": 1, "unit_price": 100, "revenue_account": "4000"}],
	)
	with pytest.raises(PermissionError, match="invoice_approval_required"):
		service.issue_invoice(invoice["id"], "tenant-test", "")
	service.issue_invoice(invoice["id"], "tenant-test", "approver")
	payment = service.record_payment("pay", "tenant-test", customer["id"], "PAY", "2026-06-01", 100, "bank_transfer", "cash")
	with pytest.raises(PermissionError, match="cash_overapplication_blocked"):
		service.apply_cash("apply", "tenant-test", payment["id"], invoice["id"], 101)
	with pytest.raises(PermissionError, match="dispute_reason_not_supported"):
		service.open_dispute("dispute", "tenant-test", invoice["id"], "unsupported", "owner")


def test_agents_batch_api_views_and_app_are_executable():
	api = _load_module("api_arc_accounts_receivable", PACKAGE_DIR / "api.py")
	views = _load_module("views_arc_accounts_receivable", PACKAGE_DIR / "views.py")
	app = _load_module("app_arc_accounts_receivable", PACKAGE_DIR / "app.py")

	customer = api.create_record({"tenant_id": "tenant-api", "customer_id": "api-cust"})
	agent = api.register_arc_agent({
		"tenant_id": "tenant-api",
		"name": "Credit Review Agent",
		"runtime": "claude_code",
		"role": "credit_reviewer",
	})
	batch = api.service().validate_batch("tenant-api", 2)
	model = views.customer_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert customer["id"] == "api-cust"
	assert agent["role"] == "credit_reviewer"
	assert batch["processor"] == "bytewax"
	assert model["records"][0]["customer_code"] == "APICUST"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["arc_accounts_receivable"]["screens"]["agents"]["route"] == "/arc-accounts-receivable/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_arc_accounts_receivable", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["arc_accounts_receivable"]["streaming"]["processor"] == "bytewax"
