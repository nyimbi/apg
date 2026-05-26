"""APG Geo-Spatial Services (GEOS) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "geos"
__capability_name__ = "Geo-Spatial Services"
__apg_dependencies__ = ["pred", "aicr", "mdm"]

capability_metadata: dict[str, Any] = {
	"name": "geos",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware geofencing, spatial analytics, territory management, location events, and predictive location intelligence",
	"category": "specialized_ai_analytics",
	"subcategory": "geo_spatial_services",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["geofencing", "location_events", "spatial_analytics", "territory_management", "location_prediction"],
	"permissions": ["geos:view", "geos:manage_geofences", "geos:process_events", "geos:analyze", "geos:admin"]
}

CAPABILITY_METADATA = capability_metadata


def register_capability() -> dict[str, Any]:
	"""Register GEOS with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "geos",
		"aliases": ["geo_spatial", "geofencing", "location_intelligence"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["ntfy", "edge", "audl", "wflo"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"geofencing": "Create and evaluate tenant-scoped geofences and spatial rules",
			"location_events": "Process enter, exit, dwell, and movement events with policy controls",
			"spatial_analytics": "Analyze density, proximity, routing, coverage, and territory patterns",
			"territory_management": "Govern regions, service areas, routing zones, and ownership",
			"capability_rules": "Evaluate deterministic geo-spatial governance rules",
			"visual_theming": "Apply location-intelligence theme tokens and components"
		},
		"endpoints": {
			"geofences": "/geos/api/v1/geofences",
			"events": "/geos/api/v1/events",
			"territories": "/geos/api/v1/territories",
			"analytics": "/geos/api/v1/analytics",
			"maps": "/geos/api/v1/maps"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get GEOS capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["CAPABILITY_METADATA", "capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
