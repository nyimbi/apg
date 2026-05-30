"""General ledger capability package tests."""

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
	module = _load_module("package_contract_glr_general_ledger", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "glr_general_ledger"
	assert "chart_of_accounts_lifecycle" in contract["provides"]
	assert "journal_posting_workflow" in contract["provides"]
	assert "glr_agents" in contract["provides"]
	assert contract["configuration"]["tenant_id"] == "tenant-test"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["stream"] == "apg.fin.glr.lifecycle"
	assert "/glr-general-ledger/journals" in {route["path"] for route in contract["ui"]["routes"]}
	assert "/glr-general-ledger/agents" in {route["path"] for route in contract["ui"]["routes"]}
	assert "codex" in contract["configuration"]["glr_agents"]["supported_runtimes"]


def test_rule_engine_blocks_missing_context_unbalanced_journals_and_non_bytewax_batches():
	module = _load_module("rule_contract_glr_general_ledger", PACKAGE_DIR / "capability_contract.py")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]

	wrong_stream = module.evaluate_capability_rules({"tenant_context_present": True, "operation": "glr_batch", "event_stream": "other"})
	assert wrong_stream["decision"] == "deny"
	assert "glr_batch_requires_bytewax" in wrong_stream["matched_rules"]

	unbalanced = module.evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_journal_entry",
		"batch_present": True,
		"journal_description_present": True,
		"journal_line_count": 2,
		"posting_accounts_valid": True,
		"balanced": False,
		"foreign_currency": False,
	})
	assert unbalanced["decision"] == "deny"
	assert "journal_must_balance" in unbalanced["matched_rules"]


def test_service_account_period_journal_posting_trial_balance_lifecycle():
	service_module = _load_module("service_glr_general_ledger", PACKAGE_DIR / "service.py")
	service = service_module.GeneralLedgerService()

	cash = service.create_account("cash", "tenant-test", "1000", "Cash", "asset")
	revenue = service.create_account("revenue", "tenant-test", "4000", "Revenue", "revenue")
	dimension = service.record_dimension("department", "tenant-test", "department", "finance", "controller")
	period = service.open_period("period", "tenant-test", "FY2026 M01", 2026, "2026-01-01", "2026-01-31")
	batch = service.create_journal_batch("batch", "tenant-test", period["id"], "manual", "USD")
	journal = service.create_journal_entry(
		"journal",
		"tenant-test",
		batch["id"],
		"Record service revenue",
		[
			{"account_id": cash["id"], "debit": 1000, "credit": 0, "description": "cash"},
			{"account_id": revenue["id"], "debit": 0, "credit": 1000, "description": "revenue"},
		],
		"preparer",
	)
	approved = service.approve_journal(journal["id"], "tenant-test", "approver")
	posting = service.post_journal(journal["id"], "tenant-test", "poster", "idem-1")
	trial_balance = service.generate_trial_balance("tenant-test")
	allocation = service.create_allocation("allocation", "tenant-test", cash["id"], [revenue["id"]], "headcount", "reviewer")

	summary = service.dashboard_summary("tenant-test")
	assert dimension["status"] == "active"
	assert approved["status"] == "approved"
	assert posting["status"] == "posted"
	assert trial_balance["status"] == "balanced"
	assert allocation["status"] == "reviewed"
	assert summary["account_count"] == 2
	assert summary["posted_journal_count"] == 1
	assert service.audit_events("tenant-test")[-1]["processor"] == "bytewax"


def test_service_enforces_glr_guardrails():
	service_module = _load_module("guardrail_service_glr_general_ledger", PACKAGE_DIR / "service.py")
	service = service_module.GeneralLedgerService()

	try:
		service.create_account("cash", "", "1000", "Cash", "asset")
	except PermissionError as error:
		assert "tenant_context_required" in str(error)
	else:
		raise AssertionError("missing tenant context should fail")

	try:
		service.create_account("bad", "tenant-test", "9000", "Bad", "unsupported")
	except PermissionError as error:
		assert "account_type_not_supported" in str(error)
	else:
		raise AssertionError("unsupported account type should fail")

	cash = service.create_account("cash", "tenant-test", "1000", "Cash", "asset")
	revenue = service.create_account("revenue", "tenant-test", "4000", "Revenue", "revenue")
	period = service.open_period("period", "tenant-test", "FY2026 M01", 2026, "2026-01-01", "2026-01-31")
	batch = service.create_journal_batch("batch", "tenant-test", period["id"], "manual", "USD")

	try:
		service.create_journal_entry("journal", "tenant-test", batch["id"], "Bad entry", [{"account_id": cash["id"], "debit": 1, "credit": 0}])
	except PermissionError as error:
		assert "journal_lines_required" in str(error)
	else:
		raise AssertionError("single-line journal should fail")

	journal = service.create_journal_entry(
		"journal",
		"tenant-test",
		batch["id"],
		"Balanced entry",
		[
			{"account_id": cash["id"], "debit": 100, "credit": 0},
			{"account_id": revenue["id"], "debit": 0, "credit": 100},
		],
		"same-user",
	)
	try:
		service.post_journal(journal["id"], "tenant-test", "same-user", "idem-guard")
	except PermissionError as error:
		assert "journal_approval_required" in str(error)
	else:
		raise AssertionError("posting without approval should fail")

	service.approve_journal(journal["id"], "tenant-test", "approver")
	try:
		service.post_journal(journal["id"], "tenant-test", "same-user", "idem-guard")
	except PermissionError as error:
		assert "segregation_of_duties_required" in str(error)
	else:
		raise AssertionError("same preparer and poster should fail")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("agent_service_glr_general_ledger", PACKAGE_DIR / "service.py")
	api_module = _load_module("api_glr_general_ledger", PACKAGE_DIR / "api.py")
	views_module = _load_module("views_glr_general_ledger", PACKAGE_DIR / "views.py")
	app_module = _load_module("app_glr_general_ledger", PACKAGE_DIR / "app.py")
	service = service_module.GeneralLedgerService()

	service.create_account("cash", "tenant-test", "1000", "Cash", "asset")
	agent = service.register_glr_agent("tenant-test", "Proof agent", "codex", "journal_reviewer", "review journals")
	action = service.validate_agent_glr_action("tenant-test", agent["id"], "post_journal", True, True)
	batch = service.validate_batch("tenant-test", 3)

	assert action["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert views_module.dashboard_model(service, "tenant-test")["summary"]["account_count"] == 1
	assert views_module.account_model(service, "tenant-test")["records"]
	assert views_module.agent_workbench_model(service, "tenant-test")["records"]
	assert api_module.create_record({"tenant_id": "tenant-api", "account_id": "api-account", "name": "API Account"})["status"] == "active"
	assert api_module.capability_status("tenant-api")["ok"] is True

	self_test = app_module.self_test()
	model = app_module.semantic_model()
	assert self_test["passed"] is True
	assert model["capabilities"]["glr_general_ledger"]["streaming"]["processor"] == "bytewax"


def test_app_entrypoint_is_publishable():
	module = _load_module("package_app_glr_general_ledger", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "glr_general_ledger" in model["capabilities"]
