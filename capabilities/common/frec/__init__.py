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
	"provides": [
		"face_consent",
		"face_enrollment",
		"face_verification",
		"face_identification",
		"face_liveness",
		"watchlist_matching",
		"emotion_governance",
		"face_reviews",
		"face_audit",
	],
	"permissions": ["frec:view", "frec:enroll", "frec:verify", "frec:identify", "frec:manage_watchlists", "frec:review", "frec:audit", "frec:admin"]
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
		"optional_dependencies": ["auth", "audl", "encr", "mfau", "moni", "cach"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"face_consent": "Record, revoke, and audit facial-recognition consent",
			"face_enrollment": "Enroll tenant-scoped facial templates with consent and quality gates",
			"face_verification": "Verify one-to-one face matches with governed thresholds",
			"face_identification": "Run one-to-many identification with watchlist policy controls",
			"face_liveness": "Validate liveness, anti-spoofing, and deepfake evidence for authentication",
			"emotion_governance": "Gate emotion analysis behind explicit approved purpose",
			"face_reviews": "Route low-confidence matches and watchlist hits to review",
			"capability_rules": "Evaluate deterministic facial-recognition rules",
			"visual_theming": "Apply identity-vision theme tokens and components"
		},
		"endpoints": {
			"status": "/frec/api/v1/status",
			"consents": "/frec/api/v1/consents",
			"enrollment": "/frec/api/v1/enrollment",
			"templates": "/frec/api/v1/templates",
			"verification": "/frec/api/v1/verification",
			"identification": "/frec/api/v1/identification",
			"liveness": "/frec/api/v1/liveness",
			"watchlists": "/frec/api/v1/watchlists",
			"reviews": "/frec/api/v1/reviews",
			"emotion": "/frec/api/v1/emotion",
			"audit": "/frec/api/v1/audit"
		},
		"adapters": contract["configuration"]["adapters"],
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
