"""APG ESG and Carbon Tracking capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_ESGC_AGENT_ROLES,
	SUPPORTED_ESGC_AGENT_RUNTIMES,
	SUPPORTED_SCOPES,
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)
from .models import EsgcAgent
from .service import EsgcService

__version__ = "1.0.0"
__capability_id__ = "esgc"
__capability_name__ = "ESG and Carbon Tracking"
__apg_dependencies__ = ["auth", "conf", "audl", "geos", "pred", "comp"]

capability_metadata: dict[str, Any] = {
	"name": "esgc",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant ESG boundaries, emissions data, factor libraries, targets, sustainability reporting, compliance evidence, and AI-agent review",
	"category": "sustainability",
	"subcategory": "esg_carbon",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": get_capability_contract()["provides"],
	"permissions": ["esgc:view", "esgc:manage_data", "esgc:report", "esgc:approve", "esgc:govern", "esgc:admin"],
	"streaming": streaming_manifest(),
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
		"dependencies": contract["requires"],
		"optional_dependencies": ["dtwn", "iotd", "mchn"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"emissions_inventory": "Track Scope 1, Scope 2, Scope 3, and activity-based emissions",
			"factor_library": "Manage approved emission factors, sources, units, and validity periods",
			"sustainability_reporting": "Produce governed ESG and carbon reports with compliance evidence",
			"target_tracking": "Track reduction targets, baselines, forecasts, and progress",
			"esgc_agents": "Register AI agents for inventory, factor, activity, report, and target review",
			"capability_rules": "Evaluate deterministic ESG and carbon governance rules",
			"visual_theming": "Apply sustainability reporting theme tokens and components",
		},
		"endpoints": {
			"emissions": "/esgc/api/v1/emissions",
			"factors": "/esgc/api/v1/factors",
			"reports": "/esgc/api/v1/reports",
			"targets": "/esgc/api/v1/targets",
			"evidence": "/esgc/api/v1/evidence",
			"agents": "/esgc/api/v1/agents",
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"],
	}


def get_capability_info() -> dict[str, Any]:
	"""Get ESGC capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = [
	"EsgcAgent",
	"EsgcService",
	"SUPPORTED_ESGC_AGENT_ROLES",
	"SUPPORTED_ESGC_AGENT_RUNTIMES",
	"SUPPORTED_SCOPES",
	"capability_metadata",
	"evaluate_capability_rules",
	"get_capability_contract",
	"get_capability_info",
	"register_capability",
	"streaming_manifest",
	"__apg_dependencies__",
	"__capability_id__",
	"__capability_name__",
	"__version__",
]
