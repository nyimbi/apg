"""Regression coverage for the WFLO executable capability contract."""

from __future__ import annotations

from capabilities.common.wflo import register_capability
from capabilities.common.wflo.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_theme_and_streaming():
	contract = get_capability_contract("tenant-flow", {"definitions": {"max_steps_per_workflow": 50}})

	assert contract["capability"] == "wflo"
	assert contract["configuration"]["tenant_id"] == "tenant-flow"
	assert contract["configuration"]["definitions"]["max_steps_per_workflow"] == 50
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"definitions",
		"steps",
		"execution",
		"tasks",
		"approvals",
		"workflow_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 35
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "designer", "definitions", "executions", "tasks", "approvals", "agents", "audit", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/wflo/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "workflow_canvas" in contract["theme"]["components"]
	assert contract["streaming"]["processor"] == "bytewax"


def test_rule_engine_enforces_workflow_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_workflow",
		"workflow_owner_assigned": False,
		"workflow_name_present": False,
		"step_count": 0,
		"retry_policy_attached": False,
		"approval_recorded": False,
		"external_trigger": True,
		"trigger_policy_attached": False,
		"ai_step_present": True,
		"ai_policy_attached": False,
		"automation_step_present": True,
		"automation_policy_attached": False,
		"event_step_present": True,
		"event_policy_attached": False,
		"expected_runtime_minutes": 2000,
		"runtime_review_recorded": False,
	})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_workflow", "approval_recorded": False})
	task_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "complete_task", "task_claimed": False})
	approval_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "record_approval", "decision_evidence_present": False, "approval_delegated": True, "delegate_present": False})
	stream_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_workflow_mutation", "event_stream": "legacy_queue"})

	assert result["decision"] == "deny"
	assert {
		"tenant_context_required",
		"workflow_requires_owner",
		"workflow_requires_name",
		"workflow_requires_steps",
		"workflow_requires_retry_policy",
		"external_trigger_requires_policy",
		"ai_step_requires_policy",
		"automation_step_requires_policy",
		"event_step_requires_policy",
		"long_running_execution_requires_review",
	} <= set(result["matched_rules"])
	assert publish_result["matched_rules"] == ["publish_requires_approval"]
	assert task_result["matched_rules"] == ["task_completion_requires_claim"]
	assert approval_result["matched_rules"] == ["approval_decision_requires_evidence", "approval_delegation_requires_delegate"]
	assert stream_result["matched_rules"] == ["batch_workflow_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "wflo"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "wflo_workflow_studio"
	assert registration["ui_components"]["designer"] == "/wflo/designer"
	assert registration["ui_components"]["agents"] == "/wflo/agents"
	assert "mqeb" in registration["dependencies"]
	assert "them" in registration["optional_dependencies"]
	assert "wflo:audit" in registration["permissions"]
