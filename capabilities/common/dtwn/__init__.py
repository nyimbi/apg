"""APG Digital Twin Framework capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .service import DtwnService

__version__ = "1.0.0"
__capability_id__ = "dtwn"
__capability_name__ = "Digital Twin Framework"
__apg_dependencies__ = ["pred", "iotd", "geos", "cvsn"]

capability_metadata: dict[str, Any] = {
	"name": "dtwn",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant digital twins, simulation models, telemetry fusion, prediction workflows, and governed virtual asset operations",
	"category": "advanced_infrastructure",
	"subcategory": "digital_twins",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["twin_registry", "simulation_models", "telemetry_fusion", "prediction_workflows", "asset_topology"],
	"permissions": ["dtwn:view", "dtwn:model", "dtwn:simulate", "dtwn:manage_twins", "dtwn:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register DTWN with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "dtwn",
		"aliases": ["digital-twin", "simulation-twin", "asset-twin"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["aicr", "anom", "edge", "mchn"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"twin_registry": "Register virtual assets, owners, topology, and lifecycle state",
			"simulation_models": "Manage executable model versions, calibration evidence, and approvals",
			"telemetry_fusion": "Fuse IoT, geospatial, vision, and prediction signals into twin state",
			"prediction_workflows": "Run forecasting and anomaly workflows against twin state",
			"capability_rules": "Evaluate deterministic digital-twin governance rules",
			"visual_theming": "Apply digital-twin operations theme tokens and components"
		},
		"endpoints": {"twins": "/dtwn/api/v1/twins", "models": "/dtwn/api/v1/models", "telemetry": "/dtwn/api/v1/telemetry", "simulations": "/dtwn/api/v1/simulations", "topology": "/dtwn/api/v1/topology"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get DTWN capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "DtwnService", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
