"""Workflow orchestration capability package contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "tested_composition_orchestration"


def _load_module(name: str):
	if PACKAGE_NAME not in sys.modules:
		package = types.ModuleType(PACKAGE_NAME)
		package.__path__ = [str(PACKAGE_DIR)]
		sys.modules[PACKAGE_NAME] = package
	spec = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.{name}", PACKAGE_DIR / f"{name}.py")
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def _workflow_tasks() -> list[dict[str, object]]:
	return [
		{"id": "intake", "name": "Intake", "type": "automated", "handler": "orders.intake", "depends_on": []},
		{"id": "review", "name": "Review", "type": "human", "assignee": "ops-review", "depends_on": ["intake"], "sla": "PT4H", "escalation": {"after": "PT4H", "to": "ops-lead"}},
		{"id": "approve", "name": "Approve", "type": "approval", "approval_policy": "orders.approval", "depends_on": ["review"]},
		{"id": "post", "name": "Post", "type": "integration", "handler": "orders.post", "capability": "fin_walt", "capability_contract": "fin_walt.payment", "depends_on": ["approve"]},
	]


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("capability_contract")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "composition_orchestration"
	assert "workflow_agents" in contract["provides"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert any(route["path"] == "/composition-orchestration/designer" for route in contract["ui"]["routes"])
	assert any(route["path"] == "/composition-orchestration/agents" for route in contract["ui"]["routes"])


def test_rule_engine_blocks_missing_context_and_non_bytewax_execution():
	module = _load_module("capability_contract")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	bad_stream = module.evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "start_execution",
		"event_stream": "other",
		"idempotency_key_present": True,
		"risk_level": "normal",
		"review_recorded": True,
	})

	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]
	assert bad_stream["decision"] == "deny"
	assert "execution_requires_bytewax_stream" in bad_stream["matched_rules"]


def test_workflow_definition_release_execution_and_task_lifecycle():
	service_module = _load_module("service")
	service = service_module.WorkflowOrchestrationService()

	definition = service.define_workflow(
		"order-fulfilment",
		"tenant-test",
		"Order Fulfilment",
		"ops-owner",
		"1.0.0",
		_workflow_tasks(),
		"order.created",
		"completed",
	)
	release = service.release_workflow("order-fulfilment-1", "tenant-test", definition["id"], "graph-valid", "rollback-to-previous", dry_run_passed=True, approved_by="release-manager")
	execution = service.start_execution("run-1", "tenant-test", definition["id"], "idem-1", {"order_id": "O-100"})
	advanced = service.complete_task("tenant-test", execution["id"], "intake", {"accepted": True})
	assignment = service.assign_human_task("tenant-test", execution["id"], "review", "ops-review")

	assert definition["status"] == "validated"
	assert release["status"] == "released"
	assert execution["event_stream"] == "bytewax"
	assert advanced["current_tasks"] == ["review"]
	assert assignment["status"] == "assigned"
	assert service.dashboard_summary("tenant-test")["audit_event_count"] >= 5


def test_service_enforces_orchestration_guardrails():
	service_module = _load_module("service")
	service = service_module.WorkflowOrchestrationService()

	try:
		service.define_workflow("bad", "", "Bad", "owner", "1.0.0", _workflow_tasks(), "start", "done")
	except PermissionError as exc:
		assert "tenant_context_required" in str(exc)
	else:
		raise AssertionError("expected tenant guardrail")

	try:
		service.define_workflow("bad", "tenant-test", "Bad", "", "1.0.0", _workflow_tasks(), "start", "done")
	except PermissionError as exc:
		assert "workflow_requires_owner" in str(exc)
	else:
		raise AssertionError("expected owner guardrail")

	try:
		service.define_workflow("bad", "tenant-test", "Bad", "owner", "1.0.0", [{"id": "human", "type": "human"}], "start", "done")
	except PermissionError as exc:
		assert "human_task_requires_assignee" in str(exc)
	else:
		raise AssertionError("expected human assignment guardrail")

	definition = service.define_workflow("good", "tenant-test", "Good", "owner", "1.0.0", _workflow_tasks(), "start", "done")
	try:
		service.start_execution("run-high", "tenant-test", definition["id"], "idem-high", risk_level="high")
	except PermissionError as exc:
		assert "high_risk_execution_requires_review" in str(exc)
	else:
		raise AssertionError("expected high-risk review guardrail")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("service")
	api_module = _load_module("api")
	views_module = _load_module("views")
	app_module = _load_module("app")

	service = service_module.WorkflowOrchestrationService()
	agent = service.register_workflow_agent("tenant-test", "Release Review", "codex", "release_reviewer", "Review workflow releases.")
	agent_result = service.validate_agent_workflow_action("tenant-test", agent["id"], "prepare_release", True, True)
	batch = service.validate_batch_schedule("tenant-test", 3)
	dashboard = views_module.dashboard_model(service, "tenant-test")
	designer = views_module.designer_model("tenant-test")
	agent_view = views_module.agent_workbench_model(service, "tenant-test")
	api_record = api_module.create_record({"id": "api-workflow", "tenant_id": "tenant-api"})
	status = api_module.capability_status("tenant-api")
	self_test = app_module.self_test()
	model = app_module.semantic_model()

	assert agent_result["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["workflow_agent_count"] == 1
	assert "parallel" in designer["node_types"]
	assert agent_view["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert api_record["id"].startswith("workflow_definition_")
	assert status["streaming"]["processor"] == "bytewax"
	assert self_test["passed"] is True
	assert model["capabilities"]["composition_orchestration"]["streaming"]["processor"] == "bytewax"
