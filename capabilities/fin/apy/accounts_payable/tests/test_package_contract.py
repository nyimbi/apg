"""Accounts payable capability package tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	if str(PACKAGE_DIR) not in sys.path:
		sys.path.insert(0, str(PACKAGE_DIR))
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("package_contract_apy_accounts_payable", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "apy_accounts_payable"
	assert "vendor_payables_lifecycle" in contract["provides"]
	assert "ap_agents" in contract["provides"]
	assert contract["requires"] == ["auth", "audl", "ntfy", "composition_events", "composition_config", "general_ledger", "cash_management", "document_management"]
	assert contract["configuration"]["tenant_id"] == "tenant-test"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["stream"] == "apg.fin.apy.lifecycle"
	assert "/apy-accounts-payable/payments" in {route["path"] for route in contract["ui"]["routes"]}
	assert "/apy-accounts-payable/agents" in {route["path"] for route in contract["ui"]["routes"]}
	assert "codex" in contract["configuration"]["ap_agents"]["supported_runtimes"]


def test_rule_engine_blocks_missing_context_and_non_bytewax_batches():
	module = _load_module("rule_contract_apy_accounts_payable", PACKAGE_DIR / "capability_contract.py")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]

	wrong_stream = module.evaluate_capability_rules({"tenant_context_present": True, "operation": "ap_batch", "event_stream": "other"})
	assert wrong_stream["decision"] == "deny"
	assert "ap_batch_requires_bytewax" in wrong_stream["matched_rules"]

	close_blocked = module.evaluate_capability_rules({"tenant_context_present": True, "operation": "close_period", "open_exception_count": 1})
	assert close_blocked["decision"] == "deny"
	assert "period_close_blocks_open_exceptions" in close_blocked["matched_rules"]


def test_service_vendor_invoice_matching_payment_expense_close_lifecycle():
	service_module = _load_module("service_apy_accounts_payable", PACKAGE_DIR / "service.py")
	service = service_module.AccountsPayableService()

	vendor = service.register_vendor("vendor", "tenant-test", "Vendor", "owner", "tax", "ach")
	invoice = service.record_invoice("invoice", "tenant-test", vendor["id"], "INV-1", 5000, "USD", "doc")
	matched = service.match_invoice("tenant-test", invoice["id"], po_backed=True, receipt_reference="receipt")
	approved = service.approve_invoice("tenant-test", matched["id"], approved_by="approver", requested_by="requester")
	payment = service.schedule_payment("payment", "tenant-test", approved["id"], 5000, "cash", "2026-06-05")
	batch = service.release_payment_batch("batch", "tenant-test", [payment["id"]], "reviewer")
	expense = service.record_expense_report("expense", "tenant-test", "employee", 150, "receipt")
	close = service.close_period("close", "tenant-test", "2026-05", 0, 0, "controller")

	summary = service.dashboard_summary("tenant-test")
	aging = service.aging_summary("tenant-test")
	events = service.audit_events("tenant-test")
	assert vendor["status"] == "active"
	assert approved["status"] == "approved"
	assert batch["status"] == "released"
	assert expense["status"] == "recorded"
	assert close["status"] == "closed"
	assert summary["vendor_count"] == 1
	assert summary["payment_batch_count"] == 1
	assert aging["open_invoice_count"] == 0
	assert events[-1]["processor"] == "bytewax"


def test_service_enforces_ap_guardrails():
	service_module = _load_module("guardrail_service_apy_accounts_payable", PACKAGE_DIR / "service.py")
	service = service_module.AccountsPayableService()

	try:
		service.register_vendor("vendor", "", "Vendor", "owner", "tax", "ach")
	except PermissionError as error:
		assert "tenant_context_required" in str(error)
	else:
		raise AssertionError("missing tenant context should fail")

	try:
		service.register_vendor("vendor", "tenant-test", "Vendor", "", "tax", "ach")
	except PermissionError as error:
		assert "vendor_requires_owner" in str(error)
	else:
		raise AssertionError("missing owner should fail")

	try:
		service.register_vendor("vendor", "tenant-test", "Vendor", "owner", "tax", "ach", bank_change=True)
	except PermissionError as error:
		assert "vendor_bank_change_requires_review" in str(error)
	else:
		raise AssertionError("bank change without review should fail")

	vendor = service.register_vendor("vendor", "tenant-test", "Vendor", "owner", "tax", "ach")
	try:
		service.record_invoice("invoice", "tenant-test", vendor["id"], "INV-1", 5000, "USD", "doc", duplicate_detected=True)
	except PermissionError as error:
		assert "duplicate_invoice_requires_review" in str(error)
	else:
		raise AssertionError("duplicate without review should fail")

	invoice = service.record_invoice("invoice", "tenant-test", vendor["id"], "INV-1", 5000, "USD", "doc")
	try:
		service.match_invoice("tenant-test", invoice["id"], po_backed=True)
	except PermissionError as error:
		assert "po_invoice_requires_receipt" in str(error)
	else:
		raise AssertionError("PO invoice without receipt should fail")

	service.match_invoice("tenant-test", invoice["id"], po_backed=True, receipt_reference="receipt")
	try:
		service.approve_invoice("tenant-test", invoice["id"], approved_by="requester", requested_by="requester")
	except PermissionError as error:
		assert "approval_requires_separation" in str(error)
	else:
		raise AssertionError("self approval should fail")

	try:
		service.schedule_payment("payment", "tenant-test", invoice["id"], 5000, "cash", "2026-06-05")
	except PermissionError as error:
		assert "payment_requires_approved_invoice" in str(error)
	else:
		raise AssertionError("payment against unapproved invoice should fail")

	try:
		service.record_expense_report("expense", "tenant-test", "employee", 0, "receipt")
	except PermissionError as error:
		assert "expense_amount_positive" in str(error)
	else:
		raise AssertionError("expense without positive amount should fail")

	try:
		service.close_period("close", "tenant-test", "2026-05", 1, 0, "controller")
	except PermissionError as error:
		assert "period_close_blocks_open_exceptions" in str(error)
	else:
		raise AssertionError("close with open exceptions should fail")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("agent_service_apy_accounts_payable", PACKAGE_DIR / "service.py")
	api_module = _load_module("api_apy_accounts_payable", PACKAGE_DIR / "api.py")
	views_module = _load_module("views_apy_accounts_payable", PACKAGE_DIR / "views.py")
	app_module = _load_module("app_apy_accounts_payable", PACKAGE_DIR / "app.py")
	service = service_module.AccountsPayableService()

	service.register_vendor("vendor", "tenant-test", "Vendor", "owner", "tax", "ach")
	agent = service.register_ap_agent("tenant-test", "Proof agent", "codex", "payment_run_reviewer", "review payments")
	action = service.validate_agent_ap_action("tenant-test", agent["id"], "release_payment_batch", True, True)
	batch = service.validate_batch("tenant-test", 3)

	assert action["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert views_module.dashboard_model(service, "tenant-test")["summary"]["vendor_count"] == 1
	assert views_module.vendor_model(service, "tenant-test")["records"]
	assert views_module.agent_workbench_model(service, "tenant-test")["records"]
	assert api_module.create_record({"tenant_id": "tenant-api", "vendor_id": "api-vendor", "name": "API Vendor"})["status"] == "active"
	assert api_module.capability_status("tenant-api")["ok"] is True

	self_test = app_module.self_test()
	model = app_module.semantic_model()
	assert self_test["passed"] is True
	assert model["capabilities"]["apy_accounts_payable"]["streaming"]["processor"] == "bytewax"


def test_app_entrypoint_is_publishable():
	module = _load_module("package_app_apy_accounts_payable", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "apy_accounts_payable" in model["capabilities"]
