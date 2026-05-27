"""Executable capability contract for APG Geo-Spatial Services."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"geofencing": {
		"geofence_owner_required": True,
		"max_vertices_per_polygon": 5000,
		"minimum_accuracy_meters": 50,
		"active_rule_required": True
	},
	"events": {
		"event_source_registered_required": True,
		"location_consent_required": True,
		"event_retention_days": 90,
		"edge_event_ingestion_supported": True
	},
	"analytics": {
		"spatial_index_required": True,
		"predictive_location_enabled": True,
		"territory_overlap_review_required": True,
		"aggregation_privacy_required": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_location_events": True,
		"data_residency_policy_required": True,
		"sensitive_location_review_required": True
	},
	"ui": {
		"enable_map_console": True,
		"enable_geofence_editor": True,
		"enable_event_monitor": True,
		"enable_spatial_analytics": True
	},
	"theme": {
		"default_theme": "geos_location_intelligence",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "geofencing", "events", "analytics", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["geofencing", "events", "analytics", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All geo-spatial operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "location_consent_required", "description": "Location processing requires consent.", "condition": {"operation": "process_location_event", "location_consent_recorded": False}, "effect": {"decision": "deny", "reason": "location_consent_required", "required_action": "record_location_consent"}},
	{"name": "geofence_requires_owner", "description": "Geofences require an accountable owner.", "condition": {"operation": "create_geofence", "geofence_owner_assigned": False}, "effect": {"decision": "deny", "reason": "geofence_owner_required", "required_action": "assign_geofence_owner"}},
	{"name": "event_source_must_be_registered", "description": "Location events require registered sources.", "condition": {"event_source_registered": False, "location_event_received": True}, "effect": {"decision": "deny", "reason": "event_source_registration_required", "required_action": "register_event_source"}},
	{"name": "sensitive_location_requires_review", "description": "Sensitive location processing requires review.", "condition": {"sensitive_location": True, "privacy_review_recorded": False}, "effect": {"decision": "deny", "reason": "sensitive_location_review_required", "required_action": "record_privacy_review"}},
	{"name": "large_polygon_requires_review", "description": "Large geofence polygons require spatial review.", "condition": {"polygon_vertices_gt": 5000, "spatial_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_polygon_review_required", "required_action": "review_geofence_geometry"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/geos/dashboard", "component": "GEOSDashboard", "permission": "geos:view", "nav_group": "Overview"},
	{"name": "maps", "path": "/geos/maps", "component": "MapConsole", "permission": "geos:view", "nav_group": "Maps"},
	{"name": "geofences", "path": "/geos/geofences", "component": "GeofenceEditor", "permission": "geos:manage_geofences", "nav_group": "Geofencing"},
	{"name": "events", "path": "/geos/events", "component": "LocationEventMonitor", "permission": "geos:process_events", "nav_group": "Events"},
	{"name": "territories", "path": "/geos/territories", "component": "TerritoryManager", "permission": "geos:manage_geofences", "nav_group": "Planning"},
	{"name": "analytics", "path": "/geos/analytics", "component": "SpatialAnalytics", "permission": "geos:analyze", "nav_group": "Analysis"},
	{"name": "privacy", "path": "/geos/privacy", "component": "LocationPrivacy", "permission": "geos:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/geos/settings", "component": "GEOSSettings", "permission": "geos:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "geos_location_intelligence",
	"tokens": {
		"color.primary": "#214E34",
		"color.accent": "#2B6CB0",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"map_console": {"icon": "map", "status_indicator": "location-pill", "risk_style": "privacy-band"},
		"geofence_editor": {"visual": "geometry-canvas", "highlight": "vertex-chip"},
		"event_monitor": {"visual": "event-stream", "status_style": "source-chip"},
		"territory_manager": {"visual": "region-grid", "status_style": "overlap-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable GEOS capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "geos",
		"display_name": "Geo-Spatial Services",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/geos/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default GEOS governance rules."""
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
