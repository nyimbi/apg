"""APG ESG/Carbon Tracking capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "esgc"
__capability_name__ = "ESG/Carbon Tracking"
__apg_dependencies__ = ["pred", "geos", "comp"]

capability_metadata: dict[str, Any] = {
	"name": "esgc",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant ESG boundaries, emissions data, factor libraries, targets, sustainability reporting, and compliance evidence",
	"category": "sustainability",
	"subcategory": "esg_carbon",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["emissions_inventory", "factor_library", "sustainability_reporting", "target_tracking", "esg_evidence"],
	"permissions": ["esgc:view", "esgc:manage_data", "esgc:report", "esgc:approve", "esgc:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register ESGC with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "esgc",
		"aliases": ["esg", "carbon", "sustainability"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["dtwn", "iotd", "mchn", "audl"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"emissions_inventory": "Track Scope 1, Scope 2, Scope 3, and activity-based emissions",
			"factor_library": "Manage approved emission factors, sources, units, and validity periods",
			"sustainability_reporting": "Produce governed ESG and carbon reports with compliance evidence",
			"target_tracking": "Track reduction targets, baselines, forecasts, and progress",
			"capability_rules": "Evaluate deterministic ESG/carbon-governance rules",
			"visual_theming": "Apply sustainability reporting theme tokens and components"
		},
		"endpoints": {"emissions": "/esgc/api/v1/emissions", "factors": "/esgc/api/v1/factors", "reports": "/esgc/api/v1/reports", "targets": "/esgc/api/v1/targets", "evidence": "/esgc/api/v1/evidence"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get ESGC capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
