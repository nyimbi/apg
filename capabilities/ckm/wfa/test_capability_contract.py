"""Focused contract and lifecycle tests for the CKM WFA capability."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_wfa_contract_declares_lifecycle_surfaces():
	module = _load_module("ckm_wfa_contract_under_test", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "ckm_wfa"
	assert contract["display_name"] == "Workflow Automation"
	assert contract["configuration"]["tenant_id"] == "tenant-test"
	assert contract["configuration"]["governance"]["batch_event_stream"] == "bytewax"
	assert contract["configuration"]["wfa_agents"]["supported_runtimes"] == [
		"codex",
		"claude_code",
		"opencode",
		"pi",
	]
	assert contract["provides"] == [
		"workflow_definitions",
		"workflow_instances",
		"task_orchestration",
		"approval_governance",
		"exception_management",
		"workflow_analytics",
		"wfa_agents",
	]
	assert contract["requires"] == ["auth", "conf", "audl", "ckm_not", "ckm_rtc"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["batch_mutation_guardrail"] == "batch_workflow_mutation_requires_bytewax"
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"designer",
		"definitions",
		"instances",
		"tasks",
		"approvals",
		"exceptions",
		"agents",
		"analytics",
		"audit",
	}


def test_wfa_contract_rules_cover_processes_tasks_agents_and_bytewax():
	module = _load_module("ckm_wfa_contract_rules_under_test", PACKAGE_DIR / "capability_contract.py")

	no_tenant = module.evaluate_capability_rules({"tenant_context_present": False})
	assert no_tenant["decision"] == "deny"
	assert "tenant_context_required" in no_tenant["matched_rules"]

	activation_blocked = module.evaluate_capability_rules({
		"operation": "activate_definition",
		"approval_recorded": False,
	})
	assert activation_blocked["decision"] == "deny"
	assert "activation_requires_approval" in activation_blocked["matched_rules"]

	task_blocked = module.evaluate_capability_rules({
		"operation": "create_task",
		"task_type": "human",
		"assignee_present": False,
	})
	assert task_blocked["decision"] == "deny"
	assert "human_task_requires_assignee" in task_blocked["matched_rules"]

	sla_blocked = module.evaluate_capability_rules({
		"operation": "create_task",
		"sla_tracked": True,
		"due_at_present": False,
	})
	assert sla_blocked["decision"] == "deny"
	assert "sla_task_requires_due_at" in sla_blocked["matched_rules"]

	approval_reason = module.evaluate_capability_rules({
		"operation": "record_approval",
		"decision_reason_present": False,
	})
	assert approval_reason["decision"] == "deny"
	assert "approval_requires_decision_reason" in approval_reason["matched_rules"]

	agent_runtime = module.evaluate_capability_rules({
		"wfa_agent_present": True,
		"agent_runtime_supported": False,
	})
	assert agent_runtime["decision"] == "deny"
	assert "wfa_agent_runtime_supported" in agent_runtime["matched_rules"]

	batch = module.evaluate_capability_rules({
		"requested_operation": "batch_workflow_mutation",
		"event_stream": "other_stream",
	})
	assert batch["decision"] == "deny"
	assert "batch_workflow_mutation_requires_bytewax" in batch["matched_rules"]


def test_wfa_lifecycle_service_enforces_guardrails():
	package = importlib.import_module("capabilities.ckm.wfa")
	service = package.WfaLifecycleService("tenant-test")

	agent = service.register_wfa_agent(
		name="Approval reviewer",
		runtime="codex",
		role="approval_reviewer",
		scope="review approval independence and evidence",
	)
	assert agent["runtime"] == "codex"
	assert agent["role"] == "approval_reviewer"

	process = service.create_process(
		process_id="close-approval",
		name="Close approval",
		owner_id="user-controller",
		version="1.0.0",
		variable_schema={"amount": {"type": "number"}},
	)
	assert process["status"] == "draft"

	with pytest.raises(PermissionError, match="workflow_activation_approval_required"):
		service.activate_process("close-approval", approval_recorded=False, reviewer_id="user-cfo")

	activated = service.activate_process("close-approval", approval_recorded=True, reviewer_id="user-cfo")
	assert activated["status"] == "active"

	instance = service.start_instance(
		instance_id="inst-close",
		process_id="close-approval",
		initiated_by="user-controller",
		context={"period": "2026-05"},
		correlation_key="close/2026-05",
	)
	assert instance["status"] == "running"

	with pytest.raises(PermissionError, match="task_assignee_required"):
		service.create_task("task-unassigned", "inst-close", "Unassigned review")

	with pytest.raises(PermissionError, match="task_due_at_required"):
		service.create_task(
			task_id="task-sla",
			instance_id="inst-close",
			name="SLA tracked review",
			assignee_id="user-cfo",
			sla_tracked=True,
		)

	task = service.create_task(
		task_id="task-review",
		instance_id="inst-close",
		name="Review accrual batch",
		assignee_id="user-cfo",
		due_at="2026-06-01T12:00:00+00:00",
		sla_tracked=True,
	)
	assert task["status"] == "open"

	with pytest.raises(PermissionError, match="task_completion_evidence_required"):
		service.complete_task("task-review", "user-cfo", completion_evidence=None)

	completed = service.complete_task(
		"task-review",
		"user-cfo",
		completion_evidence={"journal_batch": "JB-2026-05-A"},
	)
	assert completed["status"] == "complete"

	with pytest.raises(PermissionError, match="independent_reviewer_required"):
		service.record_approval(
			task_id="task-review",
			reviewer_id="user-controller",
			requester_id="user-controller",
			decision="approved",
			reason="ready",
		)

	with pytest.raises(PermissionError, match="approval_decision_reason_required"):
		service.record_approval(
			task_id="task-review",
			reviewer_id="user-cfo",
			requester_id="user-controller",
			decision="approved",
			reason="",
		)

	approval = service.record_approval(
		task_id="task-review",
		reviewer_id="user-cfo",
		requester_id="user-controller",
		decision="approved",
		reason="evidence complete",
	)
	assert approval["reason"] == "evidence complete"

	exception = service.record_exception(
		instance_id="inst-close",
		code="connector_delay",
		severity="medium",
		details={"connector": "erp-close"},
		owner_id="user-controller",
	)
	assert exception["owner_id"] == "user-controller"

	assert service.validate_batch_wfa_mutation("bytewax")["decision"] == "allow"
	assert service.validate_batch_wfa_mutation("other-stream")["decision"] == "deny"
	summary = service.dashboard_summary()
	assert summary["wfa_agent_count"] == 1
	assert summary["active_process_count"] == 1
	assert summary["instance_count"] == 1
	assert summary["approval_count"] == 1
	assert summary["exception_count"] == 1


def test_wfa_generated_evidence_and_docs_are_current():
	app = _load_module("ckm_wfa_app_under_test", PACKAGE_DIR / "app.py")
	model = app.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text(encoding="utf-8"))

	assert app.self_test()["passed"] is True
	assert model == committed_model
	assert model["capabilities"]["ckm_wfa"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["ckm_wfa"]["screens"]["agents"]["route"] == "/ckm-wfa/agents"
	for name in ("README.md", "SPECIFICATION.md", "PLAN.md"):
		assert (PACKAGE_DIR / name).exists()
