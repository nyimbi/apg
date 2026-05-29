"""Regression coverage for the GEOS executable capability contract."""

import pytest

from capabilities.common.geos import register_capability
from capabilities.common.geos.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.geos.service import GeosService


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


def test_geo_spatial_lifecycle_is_executable():
	service = GeosService()

	source = service.register_event_source(
		source_id="src-mobile",
		tenant_id="tenant-geo",
		name="Mobile GPS",
		source_type="mobile",
		consent_model="explicit",
		data_residency_policy="ke-residency",
	)
	geofence = service.create_geofence(
		geofence_id="geo-yard",
		tenant_id="tenant-geo",
		name="Nairobi Yard",
		owner="fleet-ops",
		boundary={
			"type": "circle",
			"center": {"latitude": -1.286389, "longitude": 36.817223},
			"radius_meters": 500,
		},
		trigger_events=["enter", "exit"],
	)
	event = service.process_location_event(
		event_id="evt-001",
		tenant_id="tenant-geo",
		source_id="src-mobile",
		entity_id="vehicle-001",
		entity_type="vehicle",
		latitude=-1.2865,
		longitude=36.8171,
		location_consent_recorded=True,
		accuracy_meters=8,
	)
	territory = service.create_territory(
		territory_id="terr-nbo",
		tenant_id="tenant-geo",
		name="Nairobi Service Territory",
		owner="dispatch",
		territory_type="service",
		boundary={
			"type": "polygon",
			"coordinates": [
				{"latitude": -1.30, "longitude": 36.80},
				{"latitude": -1.30, "longitude": 36.84},
				{"latitude": -1.26, "longitude": 36.84},
				{"latitude": -1.26, "longitude": 36.80},
			],
		},
	)
	analysis = service.run_spatial_analysis(
		analysis_id="ana-001",
		tenant_id="tenant-geo",
		spatial_index_available=True,
		aggregation_privacy_applied=True,
	)
	summary = service.dashboard_summary("tenant-geo")

	assert source["status"] == "registered"
	assert geofence["status"] == "active"
	assert event["matched_geofences"] == ["geo-yard"]
	assert territory["status"] == "active"
	assert analysis["hotspot_count"] == 1
	assert service.list_geofences("tenant-geo")[0]["event_count"] == 1
	assert summary == {
		"event_source_count": 1,
		"geofence_count": 1,
		"location_event_count": 1,
		"territory_count": 1,
		"analytics_count": 1,
		"audit_event_count": 5,
	}


def test_geos_service_enforces_policy_guardrails():
	service = GeosService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_event_source("src", "", "Source", "mobile", "explicit", "ke")
	with pytest.raises(PermissionError, match="sensitive_location_review_required"):
		service.register_event_source("src", "tenant-geo", "Source", "mobile", "explicit", "ke", sensitive_location=True, privacy_review_recorded=False)
	with pytest.raises(PermissionError, match="data_residency_policy_required"):
		service.register_event_source("src", "tenant-geo", "Source", "mobile", "explicit", "")
	with pytest.raises(PermissionError, match="geofence_owner_required"):
		service.create_geofence(
			"geo",
			"tenant-geo",
			"Fence",
			"",
			{"type": "circle", "center": {"latitude": 0, "longitude": 0}, "radius_meters": 100},
		)
	with pytest.raises(PermissionError, match="active_geofence_rule_required"):
		service.create_geofence(
			"geo",
			"tenant-geo",
			"Fence",
			"owner",
			{"type": "circle", "center": {"latitude": 0, "longitude": 0}, "radius_meters": 100},
			active_rule=False,
		)
	with pytest.raises(PermissionError, match="large_polygon_review_required"):
		service.create_geofence(
			"geo-large",
			"tenant-geo",
			"Large",
			"owner",
			{"type": "polygon", "coordinates": [{"latitude": 0, "longitude": 0}, {"latitude": 0, "longitude": 1}, {"latitude": 1, "longitude": 1}] * 1700},
			spatial_review_recorded=False,
		)

	service.register_event_source("src", "tenant-geo", "Source", "mobile", "explicit", "ke")
	with pytest.raises(PermissionError, match="event_source_registration_required"):
		service.process_location_event("evt", "tenant-geo", "unknown", "asset", "asset", 0, 0, True)
	with pytest.raises(PermissionError, match="location_consent_required"):
		service.process_location_event("evt", "tenant-geo", "src", "asset", "asset", 0, 0, False)
	with pytest.raises(PermissionError, match="minimum_accuracy_required"):
		service.process_location_event("evt", "tenant-geo", "src", "asset", "asset", 0, 0, True, accuracy_meters=75)
	with pytest.raises(PermissionError, match="spatial_index_required"):
		service.run_spatial_analysis("ana", "tenant-geo", spatial_index_available=False, aggregation_privacy_applied=True)
	with pytest.raises(PermissionError, match="aggregation_privacy_required"):
		service.run_spatial_analysis("ana", "tenant-geo", spatial_index_available=True, aggregation_privacy_applied=False)
