"""Regression coverage for the TENS executable capability contract."""

from capabilities.common.tens import register_capability
from capabilities.common.tens.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-tens", {"governance": {"stale_tenant_review_days": 90}})

	assert contract["capability"] == "tens"
	assert contract["configuration"]["tenant_id"] == "tenant-tens"
	assert contract["configuration"]["governance"]["stale_tenant_review_days"] == 90
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"legacy_mapping",
		"migration",
		"access",
		"tens_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["requires"] == ["mten", "auth", "audl", "idfd", "usrm"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert "codex" in contract["configuration"]["tens_agents"]["supported_runtimes"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "tenants", "mappings", "migrations", "boundaries", "deprecation", "agents", "policy", "audit", "settings"}
	assert contract["theme"]["name"] == "tens_legacy_tenant_migration"


def test_rule_engine_enforces_tens_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_legacy_tenant", "legacy_owner_assigned": False, "source_system_present": False, "compatibility_scope_present": False, "days_since_activity": 240, "stale_review_recorded": False})
	mapping_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "map_tenant", "mapping_validated": False, "event_stream": "bytewax"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "legacy_tenant_requires_owner", "legacy_tenant_requires_source_system", "legacy_tenant_requires_compatibility_scope", "stale_legacy_tenant_requires_review"}
	assert mapping_result["matched_rules"] == ["mapping_requires_validation"]


def test_rules_enforce_bytewax_and_agent_guardrails():
	stream_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "map_tenant", "mapping_validated": True, "event_stream": "memory"})
	agent_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_tens_agent", "agent_runtime_supported": False, "agent_role_supported": False})
	privileged_action = evaluate_capability_rules({"tenant_context_present": True, "operation": "agent_tenant_action", "privileged_scope": True, "human_approval_recorded": False})
	batch_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_tenant_mapping", "event_stream": "memory"})

	assert stream_result["decision"] == "deny"
	assert stream_result["matched_rules"] == ["mapping_requires_bytewax_stream"]
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) == {"tens_agent_runtime_supported", "tens_agent_role_supported"}
	assert privileged_action["matched_rules"] == ["privileged_agent_mapping_requires_human_approval"]
	assert batch_result["matched_rules"] == ["batch_tenant_mapping_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "tens"
	assert "mten" in registration["dependencies"]
	assert "audl" in registration["dependencies"]
	assert registration["streaming"]["processor"] == "bytewax"
	assert registration["ui_components"]["migrations"] == "/tens/migrations"
	assert registration["ui_components"]["agents"] == "/tens/agents"
	assert "tens:migrate" in registration["permissions"]
