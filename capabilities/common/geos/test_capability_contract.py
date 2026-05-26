"""Regression coverage for the GEOS executable capability contract."""

from capabilities.common.geos import register_capability
from capabilities.common.geos.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-geo", {"events": {"event_retention_days": 30}})

	assert contract["capability"] == "geos"
	assert contract["configuration"]["tenant_id"] == "tenant-geo"
	assert contract["configuration"]["events"]["event_retention_days"] == 30
	assert contract["configuration_schema"]["required"] == ["tenant_id", "geofencing", "events", "analytics", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "maps", "geofences", "events", "territories", "analytics", "privacy", "settings"}
	assert contract["ui"]["api_prefix"] == "/geos/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "map_console" in contract["theme"]["components"]


def test_rule_engine_enforces_geo_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "process_location_event",
		"location_consent_recorded": False,
		"geofence_owner_assigned": False,
		"event_source_registered": False,
		"location_event_received": True,
		"sensitive_location": True,
		"privacy_review_recorded": False,
		"polygon_vertices": 9000,
		"spatial_review_recorded": False
	})
	create_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "create_geofence", "geofence_owner_assigned": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "location_consent_required", "event_source_must_be_registered", "sensitive_location_requires_review", "large_polygon_requires_review"}
	assert create_result["matched_rules"] == ["geofence_requires_owner"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "geos"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "geos_location_intelligence"
	assert registration["ui_components"]["geofences"] == "/geos/geofences"
	assert "pred" in registration["dependencies"]
	assert "geos:analyze" in registration["permissions"]
