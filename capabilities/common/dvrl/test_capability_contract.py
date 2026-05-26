"""Regression coverage for the DVRL executable capability contract."""

from capabilities.common.dvrl import register_capability
from capabilities.common.dvrl.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-federation", {"queries": {"default_timeout_seconds": 120}})

	assert contract["capability"] == "dvrl"
	assert contract["configuration"]["tenant_id"] == "tenant-federation"
	assert contract["configuration"]["queries"]["default_timeout_seconds"] == 120
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"sources",
		"queries",
		"cache",
		"governance",
		"optimization",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"query",
		"sources",
		"schemas",
		"federation",
		"policies",
		"metrics",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/dvrl/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "federation_map" in contract["theme"]["components"]


def test_rule_engine_enforces_virtualization_guardrails():
	query_result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "execute_query",
		"data_classification": "restricted",
		"rbac_authorized": False,
		"result_contains_sensitive_data": True,
		"cache_requested": True,
		"lineage_capture_enabled": False,
		"estimated_query_cost": 2500.0,
		"cost_review_recorded": False
	})
	source_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_source",
		"credentials_vaulted": False
	})

	assert query_result["decision"] == "deny"
	assert set(query_result["matched_rules"]) == {
		"tenant_context_required",
		"restricted_query_requires_rbac",
		"sensitive_results_block_cache",
		"query_requires_lineage_capture",
		"high_cost_query_requires_review"
	}
	assert source_result["decision"] == "deny"
	assert source_result["matched_rules"] == ["source_registration_requires_credentials"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "dvrl_federation_console"
	assert registration["ui_components"]["query"] == "/dvrl/query"
	assert "etlp" in registration["dependencies"]
