"""Regression coverage for the CACH executable capability contract."""

from capabilities.common.cach import register_capability
from capabilities.common.cach.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-cache",
		{"policy": {"default_ttl_seconds": 120}}
	)

	assert contract["capability"] == "cach"
	assert contract["configuration"]["tenant_id"] == "tenant-cache"
	assert contract["configuration"]["policy"]["default_ttl_seconds"] == 120
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"hierarchy",
		"policy",
		"warming",
		"security",
		"optimization",
		"telemetry",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"entries",
		"policies",
		"warming",
		"hierarchy",
		"analytics",
		"security",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/cach/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "tier_hierarchy_map" in contract["theme"]["components"]


def test_rule_engine_enforces_cache_governance_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "write",
		"namespace_present": False,
		"data_classification": "sensitive",
		"entry_encrypted": False,
		"cross_tenant_access": True,
		"data_criticality": "critical",
		"entry_stale": True,
		"memory_utilization_percent": 95,
		"eviction_plan_ready": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"write_requires_namespace",
		"sensitive_entry_requires_encryption",
		"cross_tenant_cache_access_denied",
		"high_memory_pressure_requires_review"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "cach_memory_fabric"
	assert registration["ui_components"]["warming"] == "/cach/warming"
	assert "auth" in registration["dependencies"]
