"""APG Environment Management (ENVM) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "envm"
__capability_name__ = "Environment Management"
__apg_dependencies__ = ["depl", "conf", "auth"]

capability_metadata: dict[str, Any] = {
	"name": "envm",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware environment inventory, promotion, drift, secrets, policy, and multi-environment governance",
	"category": "infrastructure_operations",
	"subcategory": "environment_management",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["environment_inventory", "environment_promotion", "configuration_drift", "secret_scopes", "environment_policy"],
	"permissions": ["envm:view", "envm:manage_environments", "envm:promote", "envm:manage_secrets", "envm:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register ENVM with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "envm",
		"aliases": ["environment_management", "environments", "environment_governance"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["cicd", "audl", "secu", "moni"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"environment_inventory": "Register tenant environments, regions, stages, ownership, and status",
			"environment_promotion": "Promote config and releases across governed environment paths",
			"configuration_drift": "Detect and remediate drift between declared and observed state",
			"secret_scopes": "Manage environment-scoped secret references and access policies",
			"capability_rules": "Evaluate deterministic environment-governance rules",
			"visual_theming": "Apply environment-management theme tokens and components"
		},
		"endpoints": {"environments": "/envm/api/v1/environments", "promotion": "/envm/api/v1/promotion", "drift": "/envm/api/v1/drift", "secrets": "/envm/api/v1/secrets", "policies": "/envm/api/v1/policies"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get ENVM capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
