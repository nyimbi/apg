"""Regression coverage for the REGY executable capability contract."""

from capabilities.common.regy import register_capability
from capabilities.common.regy.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-registry", {"discovery": {"cache_ttl_seconds": 15}})

	assert contract["capability"] == "regy"
	assert contract["configuration"]["tenant_id"] == "tenant-registry"
	assert contract["configuration"]["discovery"]["cache_ttl_seconds"] == 15
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"registration",
		"discovery",
		"health",
		"governance",
		"routing",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"services",
		"register",
		"discovery",
		"health",
		"versions",
		"gateway_sync",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/regy/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "service_catalog_row" in contract["theme"]["components"]


def test_rule_engine_enforces_registry_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "register_service",
		"owner_assigned": False,
		"health_endpoint_present": False,
		"duplicate_service_name": True,
		"breaking_change_detected": True,
		"compatibility_review_recorded": False,
		"cross_tenant_discovery": True
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"service_registration_requires_owner",
		"service_registration_requires_health_endpoint",
		"duplicate_service_name_blocked",
		"breaking_change_requires_review",
		"cross_tenant_discovery_denied"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "regy_service_catalog"
	assert registration["ui_components"]["discovery"] == "/regy/discovery"
	assert "apig" in registration["dependencies"]
