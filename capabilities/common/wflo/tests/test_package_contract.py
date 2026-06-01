"""WFLO package contract and deterministic workflow runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.wflo import api, views
from capabilities.common.wflo.service import WfloService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_is_valid():
	module = _load_module("package_contract_wflo", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "wflo"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_app_entrypoint_is_publishable():
	module = _load_module("package_app_wflo", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "wflo" in model["capabilities"]
	assert model["capabilities"]["wflo"]["review_evidence"]["durable_statuses"] == ["review_required", "pending_review", "denied"]


def test_definition_publish_execution_task_approval_and_completion_lifecycle_executes():
	service = WfloService()

	definition = service.create_workflow_definition(
		tenant_id="tenant-a",
		name="Purchase Approval",
		owner_ref="process-owner",
		steps=[
			{"name": "review_request", "step_type": "human", "assignee_ref": "manager"},
			{"name": "approve_request", "step_type": "approval", "requires_approval": True},
		],
		trigger_type="external",
		trigger_policy_ref="trigger-policy://purchase",
		retry_policy_ref="retry://default",
		compensation_ref="compensation://purchase",
		expected_runtime_minutes=120,
		actor="designer-1",
	)
	published = service.publish_workflow("tenant-a", definition["id"], "approval://publish/1", "workflow-admin")
	execution = service.start_execution("tenant-a", published["id"], "purchase-123", "requester-1", {"amount": 500})
	task = service.create_task("tenant-a", execution["id"], published["steps"][0]["id"], "Review purchase", "manager")
	claimed_task = service.claim_task("tenant-a", task["id"], "manager")
	completed_task = service.complete_task("tenant-a", task["id"], "manager")
	approval = service.request_approval("tenant-a", execution["id"], "purchase-123", "approver-1", "High value purchase")
	approved = service.record_approval("tenant-a", approval["id"], "approved", "approver-1", "evidence://approval/1")
	agent = service.register_workflow_agent(
		"agent-1",
		"tenant-a",
		"Runtime observer",
		"codex",
		"runtime_observer",
		execution["id"],
		"workflow-admin",
		True,
		owner_ref="workflow-admin",
		purpose="Observe execution state and recommend safe workflow recovery actions.",
	)
	batch = service.validate_lifecycle_batch("tenant-a", "bytewax", 1, "workflow_agent_batch")
	completed = service.complete_execution("tenant-a", execution["id"], "workflow-admin")
	summary = service.dashboard_summary("tenant-a")

	assert definition["status"] == "draft"
	assert published["status"] == "published"
	assert execution["status"] == "running"
	assert task["status"] == "open"
	assert claimed_task["status"] == "claimed"
	assert completed_task["status"] == "completed"
	assert approval["status"] == "pending"
	assert approved["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert agent["purpose"]
	assert batch["processor"] == "bytewax"
	assert batch["status"] == "accepted"
	assert completed["status"] == "completed"
	assert summary["definition_count"] == 1
	assert summary["published_definition_count"] == 1
	assert summary["completed_execution_count"] == 1
	assert summary["pending_approval_count"] == 0
	assert summary["agent_count"] == 1
	assert summary["lifecycle_batch_count"] == 1
	assert summary["event_count"] >= 5


def test_workflow_guardrails_require_tenant_owner_policies_approval_and_open_work_completion():
	service = WfloService()

	try:
		service.create_workflow_definition("", "No Tenant", "owner", [{"name": "x"}], retry_policy_ref="retry://x")
	except PermissionError as exc:
		assert str(exc) == "tenant_context_required"
	else:
		raise AssertionError("missing tenant was accepted")

	try:
		service.create_workflow_definition("tenant-a", "No Owner", "", [{"name": "x"}], retry_policy_ref="retry://x")
	except PermissionError as exc:
		assert str(exc) == "workflow_owner_required"
	else:
		raise AssertionError("missing owner was accepted")

	try:
		service.create_workflow_definition(
			"tenant-a",
			"External No Policy",
			"owner",
			[{"name": "receive"}],
			trigger_type="external",
			retry_policy_ref="retry://x",
		)
	except PermissionError as exc:
		assert str(exc) == "external_trigger_policy_required"
	else:
		raise AssertionError("external trigger without policy was accepted")

	try:
		service.create_workflow_definition(
			"tenant-a",
			"AI No Policy",
			"owner",
			[{"name": "summarize", "step_type": "ai"}],
			retry_policy_ref="retry://x",
		)
	except PermissionError as exc:
		assert str(exc) == "ai_step_policy_required"
	else:
		raise AssertionError("AI step without policy was accepted")

	try:
		service.create_workflow_definition(
			"tenant-a",
			"Automation No Policy",
			"owner",
			[{"name": "sync", "step_type": "automation"}],
			retry_policy_ref="retry://x",
		)
	except PermissionError as exc:
		assert str(exc) == "automation_policy_required"
	else:
		raise AssertionError("automation step without policy was accepted")

	long_running = service.create_workflow_definition(
		"tenant-a",
		"Long Runtime",
		"owner",
		[{"name": "long_task"}],
		retry_policy_ref="retry://x",
		expected_runtime_minutes=2000,
		runtime_review_recorded=False,
	)
	assert long_running["status"] == "review_required"
	assert long_running["required_actions"] == ["review_runtime"]
	assert long_running["decision"] == "require_review"
	assert long_running["review_reasons"] == ["long_running_execution_review_required"]
	assert long_running["audit_evidence"]["required_actions"] == ["review_runtime"]
	assert service.list_pending_reviews("tenant-a")[0]["id"] == long_running["id"]

	definition = service.create_workflow_definition(
		"tenant-a",
		"Publish Guard",
		"owner",
		[{"name": "review"}],
		retry_policy_ref="retry://x",
	)
	try:
		service.publish_workflow("tenant-a", definition["id"], "", "publisher")
	except PermissionError as exc:
		assert str(exc) == "workflow_publish_approval_required"
	else:
		raise AssertionError("publish without approval was accepted")

	published = service.publish_workflow("tenant-a", definition["id"], "approval://publish", "publisher")
	execution = service.start_execution("tenant-a", published["id"], "corr-1", "starter")
	service.create_task("tenant-a", execution["id"], published["steps"][0]["id"], "Review", "owner")
	try:
		service.complete_execution("tenant-a", execution["id"], "publisher")
	except PermissionError as exc:
		assert str(exc) == "open_tasks_block_completion"
	else:
		raise AssertionError("execution completed with open tasks")


def test_task_approval_compensation_agent_and_stream_guardrails():
	service = WfloService()
	definition = service.create_workflow_definition(
		"tenant-c",
		"Compensated Workflow",
		"owner",
		[{"name": "review", "step_type": "human", "assignee_ref": "owner"}],
		retry_policy_ref="retry://x",
		compensation_ref="compensation://x",
	)
	published = service.publish_workflow("tenant-c", definition["id"], "approval://publish", "publisher")
	execution = service.start_execution("tenant-c", published["id"], "corr-c", "starter")
	task = service.create_task("tenant-c", execution["id"], published["steps"][0]["id"], "Review", "owner")

	try:
		service.complete_task("tenant-c", task["id"], "owner")
	except PermissionError as exc:
		assert str(exc) == "task_claim_required"
	else:
		raise AssertionError("task completed without claim")

	try:
		service.escalate_task("tenant-c", task["id"], "owner", "")
	except PermissionError as exc:
		assert str(exc) == "task_escalation_reason_required"
	else:
		raise AssertionError("task escalated without reason")

	try:
		service.request_approval("tenant-c", execution["id"], "subject", "approver", "")
	except PermissionError as exc:
		assert str(exc) == "approval_reason_required"
	else:
		raise AssertionError("approval requested without reason")

	approval = service.request_approval("tenant-c", execution["id"], "subject", "approver", "Needs review")
	try:
		service.record_approval("tenant-c", approval["id"], "approved", "approver")
	except PermissionError as exc:
		assert str(exc) == "approval_decision_evidence_required"
	else:
		raise AssertionError("approval decision accepted without evidence")

	try:
		service.record_approval("tenant-c", approval["id"], "delegated", "approver", "evidence://approval")
	except PermissionError as exc:
		assert str(exc) == "approval_delegate_required"
	else:
		raise AssertionError("delegation accepted without delegate")

	delegated = service.record_approval("tenant-c", approval["id"], "delegated", "approver", "evidence://approval", "delegate-1")
	assert delegated["delegated_to"] == "delegate-1"

	try:
		service.cancel_execution("tenant-c", execution["id"], "starter", "")
	except PermissionError as exc:
		assert str(exc) == "execution_state_change_reason_required"
	else:
		raise AssertionError("execution cancelled without reason")

	failed = service.fail_execution("tenant-c", execution["id"], "starter", "Downstream service failed", compensation_requested=True)
	compensated = service.run_compensation("tenant-c", execution["id"], "operator")
	assert failed["status"] == "failed"
	assert compensated["compensation_status"] == "completed"

	try:
		service.register_workflow_agent(
			"agent-bad",
			"tenant-c",
			"Agent",
			"unknown",
			"runtime_observer",
			execution["id"],
			"owner",
			True,
			owner_ref="owner",
			purpose="Observe runtime health.",
		)
	except PermissionError as exc:
		assert str(exc) == "workflow_agent_runtime_not_supported"
	else:
		raise AssertionError("unsupported workflow agent runtime accepted")

	try:
		service.register_workflow_agent(
			"agent-no-purpose",
			"tenant-c",
			"Agent",
			"codex",
			"runtime_observer",
			execution["id"],
			"owner",
			True,
			owner_ref="owner",
			purpose="",
		)
	except PermissionError as exc:
		assert str(exc) == "workflow_agent_purpose_required"
	else:
		raise AssertionError("workflow agent without purpose accepted")

	review_agent = service.register_workflow_agent(
		"agent-review",
		"tenant-c",
		"Process steward",
		"codex",
		"process_steward",
		execution["id"],
		"owner",
		True,
		owner_ref="owner",
		purpose="Steward process changes that affect published workflow behavior.",
		human_approval_required=False,
	)
	assert review_agent["status"] == "pending_review"
	assert review_agent["decision"] == "require_review"
	assert review_agent["review_reasons"] == ["workflow_agent_human_approval_required"]
	assert review_agent["audit_evidence"]["required_actions"] == ["record_workflow_agent_human_approval"]
	assert service.list_pending_reviews("tenant-c")[0]["id"] == review_agent["id"]

	try:
		service.validate_batch_mutation("legacy_queue")
	except PermissionError as exc:
		assert str(exc) == "bytewax_event_stream_required"
	else:
		raise AssertionError("non-Bytewax batch mutation accepted")
	assert service.validate_batch_mutation("bytewax")["decision"] == "allow"

	try:
		service.validate_lifecycle_batch("tenant-c", "legacy_replay", 1, "workflow_agent_batch")
	except PermissionError as exc:
		assert str(exc) == "bytewax_lifecycle_stream_required"
	else:
		raise AssertionError("non-Bytewax lifecycle batch accepted")
	denied_batch = service.list_lifecycle_batches("tenant-c")[0]
	assert denied_batch["status"] == "denied"
	assert denied_batch["decision"] == "deny"
	assert denied_batch["review_reasons"] == ["bytewax_lifecycle_stream_required"]
	assert denied_batch["audit_evidence"]["required_actions"] == ["route_wflo_lifecycle_batch_to_bytewax"]

	try:
		service.validate_lifecycle_batch("tenant-c", "bytewax", 0, "workflow_agent_batch")
	except PermissionError as exc:
		assert str(exc) == "wflo_lifecycle_batch_empty"
	else:
		raise AssertionError("empty lifecycle batch accepted")

	assert service.validate_lifecycle_batch("tenant-c", "bytewax", 2, "approval_batch")["status"] == "accepted"
	assert service.list_audit_events("tenant-c")[-1]["policy_decision"] == "allow"


def test_api_and_view_models_expose_workflow_surfaces():
	local_service = WfloService()
	api.SERVICE = local_service

	definition = api.create_workflow_definition({
		"tenant_id": "tenant-b",
		"name": "Customer Onboarding",
		"owner_ref": "process-owner",
		"steps": [
			{"name": "collect_documents", "step_type": "human", "assignee_ref": "agent"},
			{"name": "risk_summary", "step_type": "ai", "ai_policy_ref": "ai-policy://onboarding"},
		],
		"trigger_type": "external",
		"trigger_policy_ref": "trigger-policy://onboarding",
		"retry_policy_ref": "retry://onboarding",
	})
	published = api.publish_workflow({
		"tenant_id": "tenant-b",
		"definition_id": definition["id"],
		"approval_ref": "approval://publish/onboarding",
		"published_by": "workflow-admin",
	})
	execution = api.start_execution({
		"tenant_id": "tenant-b",
		"definition_id": published["id"],
		"correlation_id": "customer-1",
		"started_by": "agent",
		"payload": {"customer_id": "customer-1"},
	})
	task = api.create_task({
		"tenant_id": "tenant-b",
		"execution_id": execution["id"],
		"step_id": published["steps"][0]["id"],
		"title": "Collect documents",
		"assignee_ref": "agent",
	})
	api.claim_task({
		"tenant_id": "tenant-b",
		"task_id": task["id"],
		"claimed_by": "agent",
	})
	api.complete_task({
		"tenant_id": "tenant-b",
		"task_id": task["id"],
		"completed_by": "agent",
	})
	approval = api.request_approval({
		"tenant_id": "tenant-b",
		"execution_id": execution["id"],
		"subject_ref": "customer-1",
		"approver_ref": "supervisor",
		"reason": "Customer onboarding risk review",
	})
	api.record_approval({
		"tenant_id": "tenant-b",
		"approval_id": approval["id"],
		"decision": "approved",
		"decision_by": "supervisor",
		"decision_evidence_ref": "evidence://approval/onboarding",
	})
	agent = api.register_workflow_agent({
		"id": "agent-onboarding",
		"tenant_id": "tenant-b",
		"name": "Onboarding observer",
		"runtime": "codex",
		"role": "runtime_observer",
		"scope_ref": execution["id"],
		"registered_by": "workflow-admin",
		"contribution_disclosed": True,
		"owner_ref": "workflow-admin",
		"purpose": "Observe onboarding workflow runtime and flag blocked transitions.",
		"human_approval_required": True,
	})
	batch = api.validate_lifecycle_batch({
		"tenant_id": "tenant-b",
		"event_stream": "bytewax",
		"mutation_count": 1,
		"operation": "workflow_agent_batch",
	})
	api.complete_execution({
		"tenant_id": "tenant-b",
		"execution_id": execution["id"],
		"actor": "workflow-admin",
	})

	status = api.capability_status("tenant-b")
	system = api.list_workflow_orchestration("tenant-b")
	dashboard = views.dashboard_model(local_service, "tenant-b")
	designer = views.designer_model(local_service, "tenant-b")
	library = views.definition_library_model(local_service, "tenant-b")
	monitor = views.execution_monitor_model(local_service, "tenant-b")
	tasks = views.task_inbox_model(local_service, "tenant-b")
	approvals = views.approval_center_model(local_service, "tenant-b")
	analytics = views.analytics_model(local_service, "tenant-b")
	agents = views.agent_panel_model(local_service, "tenant-b")
	lifecycle = views.lifecycle_batch_model(local_service, "tenant-b")
	audit = views.audit_trail_model(local_service, "tenant-b")
	settings = views.settings_model("tenant-b")

	assert status["definition_count"] == 1
	assert status["pending_review_count"] == 0
	assert system["summary"]["completed_execution_count"] == 1
	assert system["pending_reviews"] == []
	assert api.list_pending_reviews("tenant-b") == []
	assert dashboard["summary"]["event_count"] >= 5
	assert dashboard["pending_reviews"] == []
	assert designer["step_types"]
	assert library["definitions"][0]["status"] == "published"
	assert library["pending_reviews"] == []
	assert monitor["executions"][0]["status"] == "completed"
	assert tasks["tasks"][0]["status"] == "completed"
	assert approvals["approvals"][0]["status"] == "approved"
	assert agent["role"] == "runtime_observer"
	assert agents["agents"][0]["runtime"] == "codex"
	assert agents["pending_reviews"] == []
	assert batch["processor"] == "bytewax"
	assert lifecycle["batches"][0]["status"] == "accepted"
	assert lifecycle["denied"] == []
	assert audit["audit_events"]
	assert analytics["review_required_definitions"] == []
	assert analytics["pending_reviews"] == []
	assert settings["configuration"]["tenant_id"] == "tenant-b"
