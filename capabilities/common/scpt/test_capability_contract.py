"""Regression coverage for the SCPT executable capability contract."""

from capabilities.common.scpt import register_capability
from capabilities.common.scpt.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-script", {"sandbox": {"max_memory_mb": 256}})

	assert contract["capability"] == "scpt"
	assert contract["configuration"]["tenant_id"] == "tenant-script"
	assert contract["configuration"]["sandbox"]["max_memory_mb"] == 256
	assert contract["configuration_schema"]["required"] == ["tenant_id", "scripts", "sandbox", "packages", "executions", "scripting_agents", "agents", "governance", "observability", "streaming", "adapters", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 42
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "workbench", "scripts", "executions", "sandboxes", "packages", "approvals", "agents", "lifecycle", "audit", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/scpt/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "script_editor" in contract["theme"]["components"]
	assert contract["theme"]["components"]["bytewax_lifecycle_panel"]["visual"] == "stream-batch-monitor"
	assert contract["agents"]["first_class"] is True
	assert "codex" in contract["agents"]["supported_runtimes"]
	assert "script_steward" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["streaming"]["broker_core_dependency_allowed"] is False
	assert "scripting_agent_composition" in contract["provides"]


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
	execute_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "execute_script", "sandbox_attached": False, "event_stream": "bytewax"})
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_scripting_agent",
		"agent_id_present": False,
		"agent_name_present": False,
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"agent_scope_present": False,
		"agent_owner_present": False,
		"agent_purpose_present": False,
		"agent_contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})
	lifecycle_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_scpt_lifecycle_batch",
		"mutation_count": 0,
		"lifecycle_operation_supported": False,
		"event_stream": "legacy_queue",
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {"tenant_context_required", "script_requires_owner", "script_requires_sandbox_policy", "dangerous_permission_requires_approval", "external_network_requires_policy", "high_resource_script_requires_review"}
	assert execute_result["matched_rules"] == ["sandbox_required"]
	assert agent_result["decision"] == "deny"
	assert {
		"scripting_agent_requires_id",
		"scripting_agent_requires_name",
		"scripting_agent_runtime_supported",
		"scripting_agent_role_supported",
		"scripting_agent_requires_scope",
		"scripting_agent_requires_owner",
		"scripting_agent_requires_purpose",
		"scripting_agent_requires_disclosure",
		"scripting_agent_privileged_role_requires_human_approval",
	} <= set(agent_result["matched_rules"])
	assert lifecycle_result["matched_rules"] == [
		"scpt_lifecycle_batch_requires_mutations",
		"scpt_lifecycle_operation_supported",
		"bytewax_scpt_lifecycle_stream_required",
	]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "scpt"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "scpt_script_workbench"
	assert registration["ui_components"]["workbench"] == "/scpt/workbench"
	assert registration["ui_components"]["agents"] == "/scpt/agents"
	assert registration["ui_components"]["lifecycle"] == "/scpt/lifecycle"
	assert "wflo" in registration["dependencies"]
	assert "audl" in registration["dependencies"]
	assert "aicr" in registration["dependencies"]
	assert "scpt:execute" in registration["permissions"]
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["processor"] == "bytewax"
