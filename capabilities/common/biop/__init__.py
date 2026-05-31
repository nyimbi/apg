"""APG Biometric Processing (BIOP) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "biop"
__capability_name__ = "Biometric Processing"
__apg_dependencies__ = ["mfau", "cvsn", "aicr", "encr", "audl", "conf"]

capability_metadata: dict[str, Any] = {
	"name": "biop",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Governed biometric enrollment, template protection, liveness, matching, and verification workflows",
	"category": "security_compliance",
	"subcategory": "biometric_processing",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": [
		"biometric_consent",
		"biometric_enrollment",
		"template_governance",
		"liveness_detection",
		"biometric_verification",
		"match_review",
		"privacy_review",
		"biometric_agent_composition",
		"bytewax_lifecycle_batches",
		"biometric_audit",
		"visual_theming",
	],
	"permissions": ["biop:view", "biop:enroll", "biop:verify", "biop:manage_consent", "biop:manage_templates", "biop:review", "biop:review_privacy", "biop:admin"]
}

CAPABILITY_NAME = "biometric_processing"
CAPABILITY_VERSION = __version__
CAPABILITY_DESCRIPTION = capability_metadata["description"]
COMPOSITION_KEYWORDS = ["biometric_processing", "identity_verification", "liveness_detection", "template_governance", "consent_management"]


def register_capability() -> dict[str, Any]:
	"""Register BIOP with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "biop",
		"aliases": ["biometric_processing", "biometric_authentication", "identity_verification"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["auth", "frec", "moni", "cach"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"biometric_consent": "Record, scope, revoke, and audit biometric consent evidence",
			"biometric_enrollment": "Enroll tenant-scoped biometric templates with explicit consent",
			"consent_management": "Record, scope, and revoke biometric consent evidence",
			"template_governance": "Protect, rotate, retire, and audit biometric templates",
			"liveness_detection": "Require presentation-attack checks for authentication flows",
			"biometric_verification": "Run match and verification decisions with confidence thresholds",
			"match_review": "Route low-confidence biometric matches for independent review",
			"privacy_review": "Route cross-border biometric processing for independent privacy review",
			"biometric_agent_composition": "Compose Codex, Claude Code, opencode, and Pi style biometric governance agents behind provider-neutral guardrails",
			"bytewax_lifecycle_batches": "Validate biometric lifecycle batches through Bytewax-first processor contracts",
			"capability_rules": "Evaluate deterministic biometric-processing rules",
			"visual_theming": "Apply biometric-control theme tokens and components"
		},
		"endpoints": {
			"status": "/biop/api/v1/status",
			"consents": "/biop/api/v1/consents",
			"enrollments": "/biop/api/v1/enrollments",
			"verification": "/biop/api/v1/verification",
			"templates": "/biop/api/v1/templates",
			"match_reviews": "/biop/api/v1/reviews/matches",
			"privacy_reviews": "/biop/api/v1/reviews/privacy",
			"agents": "/biop/api/v1/agents",
			"lifecycle": "/biop/api/v1/lifecycle",
			"liveness": "/biop/api/v1/liveness",
			"compliance": "/biop/api/v1/compliance",
			"analytics": "/biop/api/v1/analytics",
			"audit": "/biop/api/v1/audit"
		},
		"adapters": contract["configuration"]["adapters"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get BIOP capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["composition_keywords"] = COMPOSITION_KEYWORDS
	info["contract"] = get_capability_contract()
	return info


def register_with_apg() -> dict[str, Any]:
	"""Compatibility alias for older biometric registration callers."""
	return {"registration_status": "ready", "capability_info": get_capability_info(), "registration": register_capability()}


__all__ = ["CAPABILITY_NAME", "CAPABILITY_VERSION", "CAPABILITY_DESCRIPTION", "COMPOSITION_KEYWORDS", "capability_metadata", "register_capability", "register_with_apg", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
