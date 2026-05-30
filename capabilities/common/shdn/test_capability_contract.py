"""Regression coverage for the SHDN executable capability contract."""

from capabilities.common.shdn import register_capability
from capabilities.common.shdn.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-shdn", {"services": {"drain_timeout_seconds": 120}})

	assert contract["capability"] == "shdn"
	assert contract["configuration"]["tenant_id"] == "tenant-shdn"
	assert contract["configuration"]["services"]["drain_timeout_seconds"] == 120
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"services",
		"lifecycle",
		"recovery",
		"shdn_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["requires"] == ["moni", "hlth", "bkup", "audl", "envm"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert "codex" in contract["configuration"]["shdn_agents"]["supported_runtimes"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "services", "plans", "executions", "approvals", "recovery", "agents", "policy", "audit", "settings"}
	assert contract["theme"]["name"] == "shdn_lifecycle_control"


def test_rule_engine_enforces_shdn_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_service", "service_owner_assigned": False, "production_service": True, "approval_recorded": False, "force_shutdown": True, "force_review_recorded": False})
	shutdown_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "execute_shutdown", "health_gate_passed": False, "backup_snapshot_present": False, "shutdown_actor_present": True, "event_stream": "bytewax"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "service_requires_owner", "production_shutdown_requires_approval", "force_shutdown_requires_review"}
	assert shutdown_result["matched_rules"] == ["shutdown_requires_health_gate", "shutdown_requires_backup_snapshot"]


def test_rules_enforce_bytewax_and_agent_guardrails():
	stream_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "execute_shutdown", "health_gate_passed": True, "backup_snapshot_present": True, "shutdown_actor_present": True, "event_stream": "memory"})
	agent_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_shdn_agent", "agent_runtime_supported": False, "agent_role_supported": False})
	critical_action = evaluate_capability_rules({"tenant_context_present": True, "operation": "agent_lifecycle_action", "target_criticality": "critical", "human_approval_recorded": False})
	batch_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_lifecycle_mutation", "event_stream": "memory"})

	assert stream_result["decision"] == "deny"
	assert stream_result["matched_rules"] == ["shutdown_requires_bytewax_stream"]
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) == {"shdn_agent_runtime_supported", "shdn_agent_role_supported"}
	assert critical_action["matched_rules"] == ["critical_agent_shutdown_requires_human_approval"]
	assert batch_result["matched_rules"] == ["batch_lifecycle_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "shdn"
	assert "bkup" in registration["dependencies"]
	assert "audl" in registration["dependencies"]
	assert registration["streaming"]["processor"] == "bytewax"
	assert registration["ui_components"]["executions"] == "/shdn/executions"
	assert registration["ui_components"]["agents"] == "/shdn/agents"
	assert "shdn:execute" in registration["permissions"]
