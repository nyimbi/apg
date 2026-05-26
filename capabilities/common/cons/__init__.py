"""APG Consent and Privacy Management capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "cons"
__capability_name__ = "Consent and Privacy Management"
__apg_dependencies__ = ["comp", "auth", "dlpd"]

capability_metadata: dict[str, Any] = {
	"name": "cons",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant privacy purposes, consent capture, preference centers, privacy requests, and auditable processing controls",
	"category": "governance",
	"subcategory": "privacy",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["purpose_registry", "consent_capture", "privacy_requests", "preference_center", "privacy_audit"],
	"permissions": ["cons:view", "cons:manage_purposes", "cons:capture", "cons:process_requests", "cons:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register CONS with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "cons",
		"aliases": ["consent", "privacy", "preferences"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["i18n", "audl", "mchn", "wsbl"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"purpose_registry": "Manage lawful purposes, legal basis, retention, and notice links",
			"consent_capture": "Capture consent events with notice, provenance, source, and audit metadata",
			"privacy_requests": "Process data access, correction, deletion, export, and objection requests",
			"preference_center": "Expose tenant-branded privacy preferences and consent withdrawal",
			"capability_rules": "Evaluate deterministic privacy-governance rules",
			"visual_theming": "Apply privacy-center theme tokens and components"
		},
		"endpoints": {"purposes": "/cons/api/v1/purposes", "notices": "/cons/api/v1/notices", "consents": "/cons/api/v1/consents", "requests": "/cons/api/v1/requests", "preferences": "/cons/api/v1/preferences"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get CONS capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
