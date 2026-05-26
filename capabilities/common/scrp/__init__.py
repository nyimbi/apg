"""APG Scraper/Data Harvesting capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "scrp"
__capability_name__ = "Scraper/Data Harvesting"
__apg_dependencies__ = ["conn", "etlp", "auth"]

capability_metadata: dict[str, Any] = {
	"name": "scrp",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant data-source harvesting, extraction jobs, compliance controls, scheduling, and pipeline handoff",
	"category": "data_platform",
	"subcategory": "data_harvesting",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["source_registry", "harvest_jobs", "extractor_profiles", "compliance_controls", "pipeline_handoff"],
	"permissions": ["scrp:view", "scrp:configure_sources", "scrp:run_jobs", "scrp:approve_harvests", "scrp:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register SCRP with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "scrp",
		"aliases": ["scraper", "harvesting", "data-harvest"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["i18n", "nlpc", "schd", "dlpd"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"source_registry": "Register data sources, ownership, credentials, limits, and terms evidence",
			"harvest_jobs": "Schedule and execute governed data-harvesting jobs",
			"extractor_profiles": "Configure extraction rules, parsers, schemas, and output mappings",
			"compliance_controls": "Enforce consent, terms, PII handling, rate limits, and audit policy",
			"capability_rules": "Evaluate deterministic harvesting-governance rules",
			"visual_theming": "Apply data-harvesting theme tokens and components"
		},
		"endpoints": {"sources": "/scrp/api/v1/sources", "jobs": "/scrp/api/v1/jobs", "extractors": "/scrp/api/v1/extractors", "results": "/scrp/api/v1/results", "compliance": "/scrp/api/v1/compliance"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get SCRP capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
