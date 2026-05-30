"""Financial reporting capability package tests."""

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
	module = _load_module("package_contract_fin_rpt", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fin_rpt"
	assert "financial_statement_generation" in contract["provides"]
	assert "rpt_agents" in contract["provides"]
	assert contract["configuration"]["tenant_id"] == "tenant-test"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["stream"] == "apg.fin.rpt.lifecycle"
	assert "/fin-rpt/generation" in {route["path"] for route in contract["ui"]["routes"]}
	assert "/fin-rpt/agents" in {route["path"] for route in contract["ui"]["routes"]}
	assert "codex" in contract["configuration"]["rpt_agents"]["supported_runtimes"]


def test_rule_engine_blocks_missing_context_and_non_bytewax_batches():
	module = _load_module("rule_contract_fin_rpt", PACKAGE_DIR / "capability_contract.py")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]

	wrong_stream = module.evaluate_capability_rules({"tenant_context_present": True, "operation": "rpt_batch", "event_stream": "other"})
	assert wrong_stream["decision"] == "deny"
	assert "rpt_batch_requires_bytewax" in wrong_stream["matched_rules"]

	low_quality = module.evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "generate_report",
		"template_present": True,
		"period_present": True,
		"template_line_count": 1,
		"output_format_supported": True,
		"data_quality_score": 0.9,
		"quality_review_recorded": False,
	})
	assert low_quality["decision"] == "require_review"
	assert "generation_quality_requires_review" in low_quality["matched_rules"]


def test_service_reporting_publication_distribution_lifecycle():
	service_module = _load_module("service_fin_rpt", PACKAGE_DIR / "service.py")
	service = service_module.FinancialReportingService()

	template = service.create_template("template", "tenant-test", "Income Statement", "income_statement", "owner")
	line = service.add_report_line("line", "tenant-test", template["id"], "Revenue", "4*", 10)
	period = service.open_period("period", "tenant-test", "FY2026 Q1", "2026-01-01", "2026-03-31")
	generation = service.generate_report("generation", "tenant-test", template["id"], period["id"], "pdf")
	statement = service.publish_statement("statement", "tenant-test", generation["id"], "FY2026 Q1 Income", True, "approver", "reviewer")
	consolidation = service.create_consolidation("consolidation", "tenant-test", "parent", "subsidiary", "full", 75, "reviewer")
	disclosure = service.record_disclosure("disclosure", "tenant-test", statement["id"], "Revenue policy", "owner", "reviewer")
	distribution = service.distribute_statement("distribution", "tenant-test", statement["id"], ["cfo@example.com"], "pdf")

	summary = service.dashboard_summary("tenant-test")
	assert line["status"] == "active"
	assert statement["status"] == "published"
	assert consolidation["status"] == "reviewed"
	assert disclosure["status"] == "reviewed"
	assert distribution["status"] == "distributed"
	assert summary["template_count"] == 1
	assert summary["distribution_count"] == 1
	assert service.audit_events("tenant-test")[-1]["processor"] == "bytewax"


def test_service_enforces_rpt_guardrails():
	service_module = _load_module("guardrail_service_fin_rpt", PACKAGE_DIR / "service.py")
	service = service_module.FinancialReportingService()

	try:
		service.create_template("template", "", "Income Statement", "income_statement", "owner")
	except PermissionError as error:
		assert "tenant_context_required" in str(error)
	else:
		raise AssertionError("missing tenant context should fail")

	try:
		service.create_template("template", "tenant-test", "", "income_statement", "owner")
	except PermissionError as error:
		assert "template_requires_name" in str(error)
	else:
		raise AssertionError("missing template name should fail")

	template = service.create_template("template", "tenant-test", "Income Statement", "income_statement", "owner")
	period = service.open_period("period", "tenant-test", "FY2026 Q1", "2026-01-01", "2026-03-31")
	try:
		service.generate_report("generation", "tenant-test", template["id"], period["id"], "pdf")
	except PermissionError as error:
		assert "generation_requires_template_lines" in str(error)
	else:
		raise AssertionError("generation without lines should fail")

	service.add_report_line("line", "tenant-test", template["id"], "Revenue", "4*", 10)
	try:
		service.generate_report("generation", "tenant-test", template["id"], period["id"], "xml")
	except PermissionError as error:
		assert "generation_output_format_supported" in str(error)
	else:
		raise AssertionError("unsupported output format should fail")

	try:
		service.generate_report("generation", "tenant-test", template["id"], period["id"], "pdf", data_quality_score=0.5)
	except PermissionError as error:
		assert "generation_quality_requires_review" in str(error)
	else:
		raise AssertionError("low quality without review should fail")

	generation = service.generate_report("generation", "tenant-test", template["id"], period["id"], "pdf", data_quality_score=0.5, quality_reviewed_by="reviewer")
	try:
		service.publish_statement("statement", "tenant-test", generation["id"], "Statement", False, "approver", "reviewer")
	except PermissionError as error:
		assert "statement_requires_balance_check" in str(error)
	else:
		raise AssertionError("publish without balance check should fail")

	try:
		service.create_consolidation("consolidation", "tenant-test", "parent", "subsidiary", "full", 150, "reviewer")
	except PermissionError as error:
		assert "consolidation_ownership_within_bounds" in str(error)
	else:
		raise AssertionError("invalid ownership should fail")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("agent_service_fin_rpt", PACKAGE_DIR / "service.py")
	api_module = _load_module("api_fin_rpt", PACKAGE_DIR / "api.py")
	views_module = _load_module("views_fin_rpt", PACKAGE_DIR / "views.py")
	app_module = _load_module("app_fin_rpt", PACKAGE_DIR / "app.py")
	service = service_module.FinancialReportingService()

	template = service.create_template("template", "tenant-test", "Income Statement", "income_statement", "owner")
	service.add_report_line("line", "tenant-test", template["id"], "Revenue", "4*", 10)
	agent = service.register_rpt_agent("tenant-test", "Proof agent", "codex", "statement_reviewer", "review statements")
	action = service.validate_agent_rpt_action("tenant-test", agent["id"], "publish_statement", True, True)
	batch = service.validate_batch("tenant-test", 3)

	assert action["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert views_module.dashboard_model(service, "tenant-test")["summary"]["template_count"] == 1
	assert views_module.template_model(service, "tenant-test")["records"]
	assert views_module.agent_workbench_model(service, "tenant-test")["records"]
	assert api_module.create_record({"tenant_id": "tenant-api", "template_id": "api-template", "name": "API Template"})["status"] == "draft"
	assert api_module.capability_status("tenant-api")["ok"] is True

	self_test = app_module.self_test()
	model = app_module.semantic_model()
	assert self_test["passed"] is True
	assert model["capabilities"]["fin_rpt"]["streaming"]["processor"] == "bytewax"


def test_app_entrypoint_is_publishable():
	module = _load_module("package_app_fin_rpt", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "fin_rpt" in model["capabilities"]
