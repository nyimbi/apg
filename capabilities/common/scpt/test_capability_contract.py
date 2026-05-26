"""Regression coverage for the SCPT executable capability contract."""

from capabilities.common.scpt import register_capability
from capabilities.common.scpt.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-script", {"sandbox": {"max_memory_mb": 256}})

	assert contract["capability"] == "scpt"
	assert contract["configuration"]["tenant_id"] == "tenant-script"
	assert contract["configuration"]["sandbox"]["max_memory_mb"] == 256
	assert contract["configuration_schema"]["required"] == ["tenant_id", "scripts", "sandbox", "packages", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "workbench", "scripts", "executions", "sandboxes", "packages", "approvals", "settings"}
	assert contract["ui"]["api_prefix"] == "/scpt/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "script_editor" in contract["theme"]["components"]


def test_rule_engine_enforces_scripting_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_script",
		"script_owner_assigned": False,
		"sandbox_attached": False,
		"dangerous_permission_requested": True,
		"approval_recorded": False,
		"network_access_requested": True,
		"network_policy_attached": False,
		"requested_memory_mb": 1024,
		"resource_review_recorded": False
	})
	execute_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "execute_script", "sandbox_attached": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "script_requires_owner", "dangerous_permission_requires_approval", "external_network_requires_policy", "high_resource_script_requires_review"}
	assert execute_result["matched_rules"] == ["sandbox_required"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "scpt"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "scpt_script_workbench"
	assert registration["ui_components"]["workbench"] == "/scpt/workbench"
	assert "wflo" in registration["dependencies"]
	assert "scpt:execute" in registration["permissions"]
