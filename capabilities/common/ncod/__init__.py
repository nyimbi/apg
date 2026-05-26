"""APG No-Code/Low-Code Builder (NCOD) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "ncod"
__capability_name__ = "No-Code/Low-Code Builder"
__apg_dependencies__ = ["wflo", "scpt", "auth"]

capability_metadata: dict[str, Any] = {
	"name": "ncod",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware app builder, form/page composition, workflow binding, scripting, publishing, and governance",
	"category": "workflow_automation",
	"subcategory": "no_code_low_code",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["app_builder", "page_composer", "workflow_binding", "script_extensions", "app_publishing"],
	"permissions": ["ncod:view", "ncod:build", "ncod:publish", "ncod:manage_apps", "ncod:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register NCOD with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "ncod",
		"aliases": ["no_code", "low_code", "app_builder"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["audl", "conn", "them", "accs"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"app_builder": "Compose tenant-scoped applications from pages, forms, actions, data, and workflows",
			"page_composer": "Build UI pages and forms from governed components and themes",
			"workflow_binding": "Attach low-code apps to workflows, tasks, schedules, and events",
			"script_extensions": "Use approved scripts for custom validation and automation",
			"capability_rules": "Evaluate deterministic no-code governance rules",
			"visual_theming": "Apply app-builder theme tokens and components"
		},
		"endpoints": {
			"apps": "/ncod/api/v1/apps",
			"pages": "/ncod/api/v1/pages",
			"components": "/ncod/api/v1/components",
			"publishing": "/ncod/api/v1/publishing",
			"connectors": "/ncod/api/v1/connectors"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get NCOD capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
