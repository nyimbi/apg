"""APG Multi-Factor Authentication (MFAU) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "mfau"
__capability_name__ = "Multi-Factor Authentication"
__apg_dependencies__ = ["auth", "secu", "encr"]

capability_metadata: dict[str, Any] = {
	"name": "mfau",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Adaptive multi-factor authentication, enrollment, recovery, and risk-based step-up controls",
	"category": "security_compliance",
	"subcategory": "advanced_authentication",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["adaptive_mfa", "factor_enrollment", "risk_step_up", "account_recovery", "phishing_resistant_auth"],
	"permissions": ["mfau:view", "mfau:enroll", "mfau:challenge", "mfau:manage_methods", "mfau:recover", "mfau:admin"]
}

APG_CAPABILITY_INFO = capability_metadata


def register_capability() -> dict[str, Any]:
	"""Register MFAU with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "mfau",
		"aliases": ["mfa", "multi_factor_authentication", "adaptive_authentication"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["audl", "ntfy", "cvsn", "biop"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"adaptive_mfa": "Select factors and step-up requirements from tenant risk context",
			"factor_enrollment": "Enroll, verify, disable, and rotate user authentication methods",
			"risk_step_up": "Require stronger factors for elevated action or device risk",
			"account_recovery": "Govern recovery channels, backup methods, and escalation flows",
			"capability_rules": "Evaluate deterministic MFA governance rules",
			"visual_theming": "Apply adaptive-auth console theme tokens and components"
		},
		"endpoints": {
			"challenge": "/mfau/api/v1/challenge",
			"methods": "/mfau/api/v1/methods",
			"enrollment": "/mfau/api/v1/enrollment",
			"risk": "/mfau/api/v1/risk",
			"recovery": "/mfau/api/v1/recovery"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get MFAU capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


def register_mfa_capability() -> dict[str, Any]:
	"""Compatibility alias for older MFA registration callers."""
	return register_capability()


__all__ = ["APG_CAPABILITY_INFO", "capability_metadata", "register_capability", "register_mfa_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
