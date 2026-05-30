"""Budgeting and forecasting capability package tests."""

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
	module = _load_module("package_contract_bfc_budgeting_forecasting", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "bfc_budgeting_forecasting"
	assert "budget_planning_lifecycle" in contract["provides"]
	assert "bfc_agents" in contract["provides"]
	assert contract["configuration"]["tenant_id"] == "tenant-test"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["stream"] == "apg.fin.bfc.lifecycle"
	assert "/bfc-budgeting-forecasting/forecasts" in {route["path"] for route in contract["ui"]["routes"]}
	assert "/bfc-budgeting-forecasting/agents" in {route["path"] for route in contract["ui"]["routes"]}
	assert "codex" in contract["configuration"]["bfc_agents"]["supported_runtimes"]


def test_rule_engine_blocks_missing_context_and_non_bytewax_batches():
	module = _load_module("rule_contract_bfc_budgeting_forecasting", PACKAGE_DIR / "capability_contract.py")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]

	wrong_stream = module.evaluate_capability_rules({"tenant_context_present": True, "operation": "bfc_batch", "event_stream": "other"})
	assert wrong_stream["decision"] == "deny"
	assert "bfc_batch_requires_bytewax" in wrong_stream["matched_rules"]

	invalid_forecast = module.evaluate_capability_rules({"tenant_context_present": True, "operation": "create_forecast", "horizon_months": 61})
	assert invalid_forecast["decision"] == "deny"
	assert "forecast_horizon_within_limit" in invalid_forecast["matched_rules"]


def test_service_budget_forecast_scenario_variance_lifecycle():
	service_module = _load_module("service_bfc_budgeting_forecasting", PACKAGE_DIR / "service.py")
	service = service_module.BudgetingForecastingService()

	budget = service.create_budget("budget", "tenant-test", "Budget", "owner", 2026, "USD", "2026-01-01", "2026-12-31")
	line = service.add_budget_line("line", "tenant-test", budget["id"], "4000", "revenue", 5000, "2026")
	submitted = service.submit_budget("tenant-test", budget["id"], "submitter")
	approved = service.approve_budget("tenant-test", submitted["id"], "approver")
	forecast = service.create_forecast("forecast", "tenant-test", "Forecast", "trend", 12, 80, approved["id"])
	point = service.record_forecast_point("point", "tenant-test", forecast["id"], "2026-01", 420)
	scenario = service.create_scenario("scenario", "tenant-test", "Base", 70, [{"driver": "growth", "value": 5}])
	variance = service.record_variance("variance", "tenant-test", budget["id"], "4000", 5000, 5400)
	session = service.start_collaboration_session("session", "tenant-test", budget["id"], ["owner", "approver"])

	summary = service.dashboard_summary("tenant-test")
	assert line["status"] == "active"
	assert approved["status"] == "approved"
	assert point["period"] == "2026-01"
	assert scenario["probability"] == 70
	assert variance["variance_percent"] == 8
	assert session["status"] == "active"
	assert summary["budget_count"] == 1
	assert summary["forecast_count"] == 1
	assert service.audit_events("tenant-test")[-1]["processor"] == "bytewax"


def test_service_enforces_bfc_guardrails():
	service_module = _load_module("guardrail_service_bfc_budgeting_forecasting", PACKAGE_DIR / "service.py")
	service = service_module.BudgetingForecastingService()

	try:
		service.create_budget("budget", "", "Budget", "owner", 2026, "USD", "2026-01-01", "2026-12-31")
	except PermissionError as error:
		assert "tenant_context_required" in str(error)
	else:
		raise AssertionError("missing tenant context should fail")

	try:
		service.create_budget("budget", "tenant-test", "Budget", "", 2026, "USD", "2026-01-01", "2026-12-31")
	except PermissionError as error:
		assert "budget_requires_owner" in str(error)
	else:
		raise AssertionError("missing owner should fail")

	budget = service.create_budget("budget", "tenant-test", "Budget", "owner", 2026, "USD", "2026-01-01", "2026-12-31")
	try:
		service.submit_budget("tenant-test", budget["id"], "submitter")
	except PermissionError as error:
		assert "budget_submission_requires_lines" in str(error)
	else:
		raise AssertionError("submit without lines should fail")

	try:
		service.add_budget_line("line", "tenant-test", budget["id"], "4000", "unsupported", 100, "2026")
	except PermissionError as error:
		assert "budget_line_type_supported" in str(error)
	else:
		raise AssertionError("unsupported line type should fail")

	service.add_budget_line("line", "tenant-test", budget["id"], "4000", "expense", 100, "2026")
	service.submit_budget("tenant-test", budget["id"], "submitter")
	try:
		service.approve_budget("tenant-test", budget["id"], "submitter")
	except PermissionError as error:
		assert "budget_approval_requires_separation" in str(error)
	else:
		raise AssertionError("self approval should fail")

	try:
		service.create_forecast("forecast", "tenant-test", "Forecast", "unsupported", 12)
	except PermissionError as error:
		assert "forecast_method_supported" in str(error)
	else:
		raise AssertionError("unsupported forecast method should fail")

	try:
		service.create_scenario("scenario", "tenant-test", "Base", 70, [])
	except PermissionError as error:
		assert "scenario_requires_driver" in str(error)
	else:
		raise AssertionError("scenario without drivers should fail")

	try:
		service.record_variance("variance", "tenant-test", budget["id"], "4000", 100, 150)
	except PermissionError as error:
		assert "variance_above_threshold_requires_review" in str(error)
	else:
		raise AssertionError("material variance without review should fail")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("agent_service_bfc_budgeting_forecasting", PACKAGE_DIR / "service.py")
	api_module = _load_module("api_bfc_budgeting_forecasting", PACKAGE_DIR / "api.py")
	views_module = _load_module("views_bfc_budgeting_forecasting", PACKAGE_DIR / "views.py")
	app_module = _load_module("app_bfc_budgeting_forecasting", PACKAGE_DIR / "app.py")
	service = service_module.BudgetingForecastingService()

	budget = service.create_budget("budget", "tenant-test", "Budget", "owner", 2026, "USD", "2026-01-01", "2026-12-31")
	service.add_budget_line("line", "tenant-test", budget["id"], "4000", "revenue", 5000, "2026")
	agent = service.register_bfc_agent("tenant-test", "Proof agent", "codex", "forecast_reviewer", "review forecasts")
	action = service.validate_agent_bfc_action("tenant-test", agent["id"], "approve_budget", True, True)
	batch = service.validate_batch("tenant-test", 3)

	assert action["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert views_module.dashboard_model(service, "tenant-test")["summary"]["budget_count"] == 1
	assert views_module.budget_model(service, "tenant-test")["records"]
	assert views_module.agent_workbench_model(service, "tenant-test")["records"]
	assert api_module.create_record({"tenant_id": "tenant-api", "budget_id": "api-budget", "name": "API Budget"})["status"] == "draft"
	assert api_module.capability_status("tenant-api")["ok"] is True

	self_test = app_module.self_test()
	model = app_module.semantic_model()
	assert self_test["passed"] is True
	assert model["capabilities"]["bfc_budgeting_forecasting"]["streaming"]["processor"] == "bytewax"


def test_app_entrypoint_is_publishable():
	module = _load_module("package_app_bfc_budgeting_forecasting", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "bfc_budgeting_forecasting" in model["capabilities"]
