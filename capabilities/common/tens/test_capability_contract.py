"""Regression coverage for the TENS executable capability contract."""

from capabilities.common.tens import register_capability
from capabilities.common.tens.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-tens", {"governance": {"stale_tenant_review_days": 90}})

	assert contract["capability"] == "tens"
	assert contract["configuration"]["tenant_id"] == "tenant-tens"
	assert contract["configuration"]["governance"]["stale_tenant_review_days"] == 90
	assert contract["configuration_schema"]["required"] == ["tenant_id", "legacy_mapping", "migration", "access", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "tenants", "mappings", "migrations", "boundaries", "deprecation", "audit", "settings"}
	assert contract["theme"]["name"] == "tens_legacy_tenant_migration"


def test_rule_engine_enforces_tens_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_legacy_tenant", "legacy_owner_assigned": False, "auth_boundary_validated": False, "days_since_activity": 240, "stale_review_recorded": False})
	mapping_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "map_tenant", "mapping_validated": False, "auth_boundary_validated": True})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "legacy_tenant_requires_owner", "access_boundary_required", "stale_legacy_tenant_requires_review"}
	assert mapping_result["matched_rules"] == ["mapping_requires_validation"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "tens"
	assert "mten" in registration["dependencies"]
	assert registration["ui_components"]["migrations"] == "/tens/migrations"
	assert "tens:migrate" in registration["permissions"]
