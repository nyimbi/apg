"""Enterprise asset management package contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "tested_eam_ast"


def _load_module(name: str):
	if PACKAGE_NAME not in sys.modules:
		package = types.ModuleType(PACKAGE_NAME)
		package.__path__ = [str(PACKAGE_DIR)]
		sys.modules[PACKAGE_NAME] = package
	spec = importlib.util.spec_from_file_location(
		f"{PACKAGE_NAME}.{name}",
		PACKAGE_DIR / f"{name}.py",
	)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("capability_contract")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "eam_ast"
	assert "asset_registry_lifecycle" in contract["provides"]
	assert "eam_agents" in contract["provides"]
	assert contract["requires"] == ["auth", "audl", "ntfy", "composition_events", "composition_config", "fixed_asset_management"]
	assert contract["configuration"]["tenant_id"] == "tenant-test"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["stream"] == "apg.eam.ast.lifecycle"
	assert any(route["path"] == "/eam-ast/maintenance-plans" for route in contract["ui"]["routes"])
	assert any(route["path"] == "/eam-ast/agents" for route in contract["ui"]["routes"])


def test_rule_engine_blocks_missing_context_and_non_bytewax_imports():
	module = _load_module("capability_contract")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	bad_stream = module.evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "eam_batch_import",
		"event_stream": "other",
	})
	bad_interval = module.evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_maintenance_plan",
		"interval_days": 0,
	})

	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]
	assert bad_stream["decision"] == "deny"
	assert "eam_batch_import_requires_bytewax" in bad_stream["matched_rules"]
	assert bad_interval["decision"] == "deny"
	assert "maintenance_plan_interval_positive" in bad_interval["matched_rules"]


def test_service_location_asset_work_order_inspection_condition_inventory_lifecycle():
	service_module = _load_module("service")
	service = service_module.EnterpriseAssetManagementService()

	location = service.register_location("plant-1", "tenant-test", "Plant 1", "site")
	asset = service.register_asset(
		"pump-1",
		"tenant-test",
		"Main pump",
		"operations",
		"rotating_equipment",
		location["location_id"],
		"critical",
		92,
		capitalized=True,
		fixed_asset_ref="fa-001",
	)
	plan = service.create_maintenance_plan("plan-1", "tenant-test", asset["id"], "predictive", 30, "vibration_sensor")
	work_order = service.open_work_order(
		"wo-1",
		"tenant-test",
		asset["id"],
		"Inspect pump vibration",
		"high",
		"lockout_tagout",
		approved_by="safety-1",
	)
	reservation = service.reserve_inventory("res-1", "tenant-test", "seal-kit", 2, work_order["id"])
	inspection = service.record_inspection("insp-1", "tenant-test", asset["id"], "pass", "inspector-1", 88)
	condition = service.record_condition_reading("read-1", "tenant-test", asset["id"], "vibration", 4.2, "mm/s", review_recorded=True, alert_threshold=3.5)
	completed = service.complete_work_order("tenant-test", work_order["id"], "bearing replaced", "tech-1")
	summary = service.dashboard_summary("tenant-test")
	reliability = service.reliability_summary("tenant-test")

	assert plan["status"] == "active"
	assert reservation["status"] == "reserved"
	assert inspection["condition_score"] == 88
	assert condition["status"] == "degraded"
	assert completed["status"] == "work_complete"
	assert summary["asset_count"] == 1
	assert summary["inventory_reservation_count"] == 1
	assert reliability["degraded_asset_count"] == 1
	assert service.audit_events("tenant-test")[-1]["processor"] == "bytewax"


def test_service_enforces_eam_guardrails():
	service_module = _load_module("service")
	service = service_module.EnterpriseAssetManagementService()

	try:
		service.register_location("bad", "", "Bad", "site")
	except PermissionError as exc:
		assert "tenant_context_required" in str(exc)
	else:
		raise AssertionError("expected tenant guardrail")

	location = service.register_location("plant", "tenant-test", "Plant", "site")
	try:
		service.register_asset("bad-asset", "tenant-test", "Bad asset", "", "equipment", location["location_id"], "medium")
	except PermissionError as exc:
		assert "asset_requires_owner" in str(exc)
	else:
		raise AssertionError("expected owner guardrail")

	try:
		service.register_asset("capital", "tenant-test", "Capital", "owner", "equipment", location["location_id"], "medium", capitalized=True)
	except PermissionError as exc:
		assert "capital_asset_requires_fixed_asset_reference" in str(exc)
	else:
		raise AssertionError("expected fixed-asset guardrail")

	asset = service.register_asset("pump", "tenant-test", "Pump", "owner", "equipment", location["location_id"], "critical", fixed_asset_ref=None)
	try:
		service.create_maintenance_plan("bad-plan", "tenant-test", asset["id"], "predictive", 30)
	except PermissionError as exc:
		assert "predictive_plan_requires_condition_source" in str(exc)
	else:
		raise AssertionError("expected predictive source guardrail")

	try:
		service.open_work_order("bad-wo", "tenant-test", asset["id"], "Critical repair", "high", "lockout_tagout")
	except PermissionError as exc:
		assert "critical_work_order_requires_approval" in str(exc)
	else:
		raise AssertionError("expected critical work approval guardrail")

	try:
		service.reserve_inventory("bad-res", "tenant-test", "seal-kit", 0)
	except PermissionError as exc:
		assert "inventory_quantity_positive" in str(exc)
	else:
		raise AssertionError("expected inventory quantity guardrail")

	try:
		service.record_condition_reading("bad-read", "tenant-test", asset["id"], "vibration", 5, "mm/s", alert_threshold=3)
	except PermissionError as exc:
		assert "condition_alert_requires_review" in str(exc)
	else:
		raise AssertionError("expected condition review guardrail")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("service")
	api_module = _load_module("api")
	views_module = _load_module("views")
	app_module = _load_module("app")

	service = service_module.EnterpriseAssetManagementService()
	location = service.register_location("plant", "tenant-test", "Plant", "site")
	asset = service.register_asset("pump", "tenant-test", "Pump", "owner", "equipment", location["location_id"], "medium")
	agent = service.register_eam_agent(
		"tenant-test",
		"Maintenance Review",
		"codex",
		"maintenance_planner",
		"Review planned maintenance.",
	)
	agent_result = service.validate_agent_eam_action(
		"tenant-test",
		agent["id"],
		"recommend_work_order",
		privileged_scope=True,
		human_approval_recorded=True,
	)
	batch = service.validate_batch_import("tenant-test", 3)
	dashboard = views_module.dashboard_model(service, "tenant-test")
	asset_view = views_module.asset_model(service, "tenant-test")
	inventory_view = views_module.inventory_model(service, "tenant-test")
	analytics_view = views_module.analytics_model(service, "tenant-test")
	agent_view = views_module.agent_workbench_model(service, "tenant-test")
	api_location = api_module.register_location({"location_id": "api-plant", "tenant_id": "tenant-api", "name": "API Plant", "location_type": "site"})
	api_record = api_module.create_record({"asset_id": "api-asset", "tenant_id": "tenant-api", "location_id": api_location["location_id"]})
	status = api_module.capability_status("tenant-api")
	self_test = app_module.self_test()
	model = app_module.semantic_model()

	assert asset["status"] == "in_service"
	assert agent_result["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["eam_agent_count"] == 1
	assert asset_view["records"][0]["asset_id"] == "pump"
	assert inventory_view["screen"] == "inventory"
	assert analytics_view["summary"]["asset_count"] == 1
	assert agent_view["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert api_record["id"].startswith("eam_asset_")
	assert status["streaming"]["processor"] == "bytewax"
	assert self_test["passed"] is True
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["eam_ast"]["streaming"]["processor"] == "bytewax"
