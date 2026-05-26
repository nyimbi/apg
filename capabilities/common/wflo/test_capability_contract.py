"""Regression coverage for the WFLO executable capability contract."""

from capabilities.common.wflo import register_capability
from capabilities.common.wflo.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-flow", {"definitions": {"max_steps_per_workflow": 50}})

	assert contract["capability"] == "wflo"
	assert contract["configuration"]["tenant_id"] == "tenant-flow"
	assert contract["configuration"]["definitions"]["max_steps_per_workflow"] == 50
	assert contract["configuration_schema"]["required"] == ["tenant_id", "definitions", "execution", "approvals", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "designer", "definitions", "executions", "tasks", "approvals", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/wflo/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "workflow_canvas" in contract["theme"]["components"]


def test_rule_engine_enforces_workflow_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_workflow",
		"workflow_owner_assigned": False,
		"approval_recorded": False,
		"external_trigger": True,
		"trigger_policy_attached": False,
		"ai_step_present": True,
		"ai_policy_attached": False,
		"expected_runtime_minutes": 2000,
		"runtime_review_recorded": False
	})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_workflow", "approval_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "workflow_requires_owner", "external_trigger_requires_policy", "ai_step_requires_policy", "long_running_execution_requires_review"}
	assert publish_result["matched_rules"] == ["publish_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "wflo"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "wflo_workflow_studio"
	assert registration["ui_components"]["designer"] == "/wflo/designer"
	assert "mqeb" in registration["dependencies"]
	assert "wflo:execute" in registration["permissions"]
