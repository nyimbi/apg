"""Cash management capability package tests."""

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
	module = _load_module("package_contract_cbm_cash_management", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "cbm_cash_management"
	assert "cash_position_service" in contract["provides"]
	assert "cash_forecasting_workflow" in contract["provides"]
	assert "cbm_agents" in contract["provides"]
	assert contract["configuration"]["tenant_id"] == "tenant-test"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["stream"] == "apg.fin.cbm.lifecycle"
	assert "/cbm-cash-management/forecasts" in {route["path"] for route in contract["ui"]["routes"]}
	assert "/cbm-cash-management/agents" in {route["path"] for route in contract["ui"]["routes"]}
	assert "codex" in contract["configuration"]["cbm_agents"]["supported_runtimes"]


def test_rule_engine_blocks_missing_context_bad_flows_and_non_bytewax_batches():
	module = _load_module("rule_contract_cbm_cash_management", PACKAGE_DIR / "capability_contract.py")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]

	wrong_stream = module.evaluate_capability_rules({"tenant_context_present": True, "operation": "cbm_batch", "event_stream": "other"})
	assert wrong_stream["decision"] == "deny"
	assert "cbm_batch_requires_bytewax" in wrong_stream["matched_rules"]

	negative_flow = module.evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "record_cash_flow",
		"account_present": True,
		"flow_type_supported": True,
		"amount": 0,
		"category_present": True,
	})
	assert negative_flow["decision"] == "deny"
	assert "cash_flow_amount_positive" in negative_flow["matched_rules"]


def test_service_cash_lifecycle_forecast_reconciliation_and_payment_run():
	service_module = _load_module("service_cbm_cash_management", PACKAGE_DIR / "service.py")
	service = service_module.CashManagementService()

	bank = service.create_bank("bank", "tenant-test", "BANK", "Primary Bank")
	account = service.create_cash_account("account", "tenant-test", bank["id"], "001", "Operating", "operating", "USD", 100)
	position = service.record_cash_position("position", "tenant-test", account["id"], "2026-05-31", 1000, 1000)
	flow = service.record_cash_flow("flow", "tenant-test", account["id"], "inflow", 250, "customer_receipt", "2026-06-01")
	forecast = service.create_cash_forecast("forecast", "tenant-test", 30, "base", 0.9)
	reconciliation = service.record_bank_reconciliation("recon", "tenant-test", account["id"], 1000, 1000)
	investment = service.create_treasury_investment("investment", "tenant-test", "deposit", "Bank", 500, "2026-09-30", 0.05, "approver")
	payment_run = service.validate_payment_run("payrun", "tenant-test", account["id"], 200)

	summary = service.dashboard_summary("tenant-test")
	assert position["status"] == "recorded"
	assert flow["status"] == "recorded"
	assert forecast["source_flow_count"] == 1
	assert reconciliation["status"] == "matched"
	assert investment["status"] == "approved"
	assert payment_run["status"] == "funded"
	assert summary["cash_account_count"] == 1
	assert summary["payment_run_count"] == 1
	assert service.audit_events("tenant-test")[-1]["processor"] == "bytewax"


def test_service_enforces_cash_management_guardrails():
	service_module = _load_module("guardrail_service_cbm_cash_management", PACKAGE_DIR / "service.py")
	service = service_module.CashManagementService()

	try:
		service.create_bank("bank", "", "BANK", "Primary Bank")
	except PermissionError as error:
		assert "tenant_context_required" in str(error)
	else:
		raise AssertionError("missing tenant context should fail")

	bank = service.create_bank("bank", "tenant-test", "BANK", "Primary Bank")
	try:
		service.create_cash_account("account", "tenant-test", bank["id"], "001", "Operating", "unsupported")
	except PermissionError as error:
		assert "account_type_not_supported" in str(error)
	else:
		raise AssertionError("unsupported account type should fail")

	account = service.create_cash_account("account", "tenant-test", bank["id"], "001", "Operating", "operating", "USD", 1000)
	try:
		service.record_cash_position("position", "tenant-test", account["id"], "2026-05-31", 100)
	except PermissionError as error:
		assert "liquidity_review_required" in str(error)
	else:
		raise AssertionError("low liquidity without review should fail")

	try:
		service.record_cash_flow("flow", "tenant-test", account["id"], "outflow", 0, "supplier", "2026-06-01")
	except PermissionError as error:
		assert "cash_flow_amount_positive_required" in str(error)
	else:
		raise AssertionError("zero flow should fail")

	service.record_cash_position("position", "tenant-test", account["id"], "2026-05-31", 100, 100, "reviewer")
	try:
		service.validate_payment_run("payrun", "tenant-test", account["id"], 200)
	except PermissionError as error:
		assert "cash_deficit_approval_required" in str(error)
	else:
		raise AssertionError("deficit payment run should fail without approval")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("agent_service_cbm_cash_management", PACKAGE_DIR / "service.py")
	api_module = _load_module("api_cbm_cash_management", PACKAGE_DIR / "api.py")
	views_module = _load_module("views_cbm_cash_management", PACKAGE_DIR / "views.py")
	app_module = _load_module("app_cbm_cash_management", PACKAGE_DIR / "app.py")
	service = service_module.CashManagementService()

	bank = service.create_bank("bank", "tenant-test", "BANK", "Primary Bank")
	service.create_cash_account("account", "tenant-test", bank["id"], "001", "Operating", "operating")
	agent = service.register_cbm_agent("tenant-test", "Proof agent", "codex", "cash_position_reviewer", "review positions")
	action = service.validate_agent_cbm_action("tenant-test", agent["id"], "validate_payment_run", True, True)
	batch = service.validate_batch("tenant-test", 3)

	assert action["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert views_module.dashboard_model(service, "tenant-test")["summary"]["cash_account_count"] == 1
	assert views_module.bank_model(service, "tenant-test")["records"]
	assert views_module.agent_workbench_model(service, "tenant-test")["records"]
	assert api_module.create_record({"tenant_id": "tenant-api", "bank_id": "api-bank", "name": "API Bank"})["status"] == "active"
	assert api_module.capability_status("tenant-api")["ok"] is True

	self_test = app_module.self_test()
	model = app_module.semantic_model()
	assert self_test["passed"] is True
	assert model["capabilities"]["cbm_cash_management"]["streaming"]["processor"] == "bytewax"


def test_app_entrypoint_is_publishable():
	module = _load_module("package_app_cbm_cash_management", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "cbm_cash_management" in model["capabilities"]
