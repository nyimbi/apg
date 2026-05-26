"""APG Facial Recognition (FREC) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "frec"
__capability_name__ = "Facial Recognition"
__apg_dependencies__ = ["biop", "cvsn", "aicr"]

capability_metadata: dict[str, Any] = {
	"name": "frec",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Face enrollment, verification, identification, liveness, watchlist, and privacy-governed recognition workflows",
	"category": "security_compliance",
	"subcategory": "facial_recognition",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["face_enrollment", "face_verification", "face_identification", "face_liveness", "watchlist_matching"],
	"permissions": ["frec:view", "frec:enroll", "frec:verify", "frec:identify", "frec:manage_watchlists", "frec:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register FREC with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "frec",
		"aliases": ["facial_recognition", "face_recognition", "face_identity"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["auth", "audl", "encr", "mfau"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"face_enrollment": "Enroll tenant-scoped facial templates with consent and quality gates",
			"face_verification": "Verify one-to-one face matches with governed thresholds",
			"face_identification": "Run one-to-many identification with watchlist policy controls",
			"face_liveness": "Validate liveness and anti-spoofing evidence for authentication",
			"capability_rules": "Evaluate deterministic facial-recognition rules",
			"visual_theming": "Apply identity-vision theme tokens and components"
		},
		"endpoints": {
			"enrollment": "/frec/api/v1/enrollment",
			"verification": "/frec/api/v1/verification",
			"identification": "/frec/api/v1/identification",
			"liveness": "/frec/api/v1/liveness",
			"watchlists": "/frec/api/v1/watchlists"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get FREC capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
