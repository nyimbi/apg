"""APG Tenants Legacy capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "tens"
__capability_name__ = "Tenants Legacy"
__apg_dependencies__ = ["mten", "auth"]

capability_metadata: dict[str, Any] = {
	"name": "tens",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Legacy tenant compatibility, tenant mapping, migration controls, access boundary validation, and deprecation governance",
	"category": "platform",
	"subcategory": "legacy_tenants",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["legacy_tenant_registry", "tenant_mapping", "migration_controls", "access_boundaries", "deprecation_governance"],
	"permissions": ["tens:view", "tens:map", "tens:migrate", "tens:approve", "tens:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register TENS with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "tens",
		"aliases": ["legacy-tenants", "tenant-compatibility", "tenant-migration"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["usrm", "cons", "audl", "idfd"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"legacy_tenant_registry": "Track legacy tenant records, owners, source systems, and lifecycle state",
			"tenant_mapping": "Map legacy tenant identifiers to APG multi-tenant boundaries",
			"migration_controls": "Govern migration plans, approvals, validation, and rollback",
			"access_boundaries": "Validate tenant auth boundaries and compatibility scopes",
			"capability_rules": "Evaluate deterministic legacy-tenant governance rules",
			"visual_theming": "Apply legacy-tenant migration theme tokens and components"
		},
		"endpoints": {"tenants": "/tens/api/v1/tenants", "mappings": "/tens/api/v1/mappings", "migrations": "/tens/api/v1/migrations", "boundaries": "/tens/api/v1/boundaries", "deprecation": "/tens/api/v1/deprecation"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get TENS capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
