"""Regression coverage for the AGNT executable capability contract."""

from agents import DEFAULT_AGENT_INTEGRATIONS
from capabilities.common.agnt import register_capability
from capabilities.common.agnt.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_agent_runtimes_rules_ui_and_theme():
	contract = get_capability_contract("tenant-agnt", {"runtimes": {"default_runtime": "codex"}})

	assert contract["capability"] == "agnt"
	assert contract["configuration"]["tenant_id"] == "tenant-agnt"
	assert contract["configuration"]["runtimes"]["default_runtime"] == "codex"
	assert set(contract["configuration"]["runtimes"]["registered"]) >= {"local", "codex", "claude_code", "opencode", "pi"}
	assert set(DEFAULT_AGENT_INTEGRATIONS.names()) >= {"local", "codex", "claude_code", "opencode", "pi"}
	assert contract["configuration_schema"]["required"] == ["tenant_id", "agents", "teams", "runtimes", "memory", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "agents", "teams", "handoffs", "runtimes", "executions", "memory", "settings"}
	assert contract["theme"]["name"] == "agnt_agent_ops"


def test_rule_engine_enforces_agent_composition_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_agent", "model_present": False, "runtime_registered": False, "handoff_endpoint_resolved": False, "workspace_runtime": True, "sandbox_policy_attached": False})
	team_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_team", "agent_count": 0, "runtime_registered": True, "handoff_endpoint_resolved": True})
	review_result = evaluate_capability_rules({"tenant_context_present": True, "runtime_registered": True, "handoff_endpoint_resolved": True, "external_runtime": True, "approval_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "agent_requires_model", "agent_runtime_must_be_registered", "handoff_endpoint_must_resolve", "workspace_runtime_requires_sandbox"}
	assert team_result["matched_rules"] == ["team_requires_agent"]
	assert review_result["decision"] == "require_review"


def test_registration_includes_full_agent_capability_contract():
	registration = register_capability()

	assert registration["name"] == "agnt"
	assert "aicr" in registration["dependencies"]
	assert registration["ui_components"]["runtimes"] == "/agnt/runtimes"
	assert "agnt:run" in registration["permissions"]
	assert "runtime_adapters" in registration["capabilities"]
