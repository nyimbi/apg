"""APG User Management capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "usrm"
__capability_name__ = "User Management"
__apg_dependencies__ = ["auth", "mfau", "cons"]

capability_metadata: dict[str, Any] = {
	"name": "usrm",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant user lifecycle, profile administration, access reviews, MFA posture, privacy consent, and deprovisioning workflows",
	"category": "identity",
	"subcategory": "user_lifecycle",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["user_directory", "lifecycle_workflows", "access_reviews", "mfa_posture", "privacy_preferences"],
	"permissions": ["usrm:view", "usrm:manage_users", "usrm:review_access", "usrm:deprovision", "usrm:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register USRM with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "usrm",
		"aliases": ["users", "user-management", "user-lifecycle"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["i18n", "audl", "idfd", "mten"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"user_directory": "Manage tenant user records, profiles, status, and directory metadata",
			"lifecycle_workflows": "Invite, onboard, suspend, reactivate, transfer, and deprovision users",
			"access_reviews": "Coordinate role, entitlement, and privileged access reviews",
			"mfa_posture": "Track MFA requirements and user authentication posture",
			"capability_rules": "Evaluate deterministic user-lifecycle governance rules",
			"visual_theming": "Apply user-management theme tokens and components"
		},
		"endpoints": {"users": "/usrm/api/v1/users", "profiles": "/usrm/api/v1/profiles", "access": "/usrm/api/v1/access", "lifecycle": "/usrm/api/v1/lifecycle", "privacy": "/usrm/api/v1/privacy"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get USRM capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
