"""APG Multi-Factor Authentication (MFAU) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "mfau"
__capability_name__ = "Multi-Factor Authentication"
__apg_dependencies__ = ["auth", "secu", "encr", "aicr", "conf", "audl"]

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
	"provides": [
		"adaptive_mfa",
		"user_profile_governance",
		"method_enrollment",
		"adaptive_challenges",
		"risk_assessment",
		"device_trust",
		"recovery_governance",
		"backup_codes",
		"policy_management",
		"phishing_resistant_auth",
		"mfa_agent_composition",
		"bytewax_lifecycle_batches",
	],
	"permissions": [
		"mfau:view",
		"mfau:enroll",
		"mfau:challenge",
		"mfau:manage_methods",
		"mfau:recover",
		"mfau:manage_policies",
		"mfau:govern",
		"mfau:audit",
		"mfau:admin",
	]
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
		"optional_dependencies": ["ntfy", "cvsn", "biop", "cach", "moni"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"adaptive_mfa": "Select factors and step-up requirements from tenant risk context",
			"user_profile_governance": "Govern tenant-scoped MFA profiles and policy assignment",
			"method_enrollment": "Enroll, verify, disable, and rotate user authentication methods",
			"adaptive_challenges": "Issue risk-aware challenges with replay and expiry guardrails",
			"risk_assessment": "Assess user, device, and action risk for authentication events",
			"device_trust": "Bind devices and require review for low-trust posture",
			"recovery_governance": "Govern recovery channels, backup methods, and escalation flows",
			"backup_codes": "Generate and consume single-use backup code sets",
			"policy_management": "Manage MFA policy changes with audit evidence",
			"mfa_agent_composition": "Compose Codex, Claude Code, opencode, and Pi style MFA security agents behind provider-neutral guardrails",
			"bytewax_lifecycle_batches": "Validate MFA lifecycle batches through Bytewax-first processor contracts",
			"capability_rules": "Evaluate deterministic MFA governance rules",
			"visual_theming": "Apply adaptive-auth console theme tokens and components"
		},
		"endpoints": {
			"status": "/mfau/api/v1/status",
			"profiles": "/mfau/api/v1/profiles",
			"methods": "/mfau/api/v1/methods",
			"enrollment": "/mfau/api/v1/enrollment",
			"challenges": "/mfau/api/v1/challenges",
			"risk": "/mfau/api/v1/risk",
			"devices": "/mfau/api/v1/devices",
			"recovery": "/mfau/api/v1/recovery",
			"backup_codes": "/mfau/api/v1/backup-codes",
			"policies": "/mfau/api/v1/policies",
			"biometrics": "/mfau/api/v1/biometrics",
			"agents": "/mfau/api/v1/agents",
			"lifecycle": "/mfau/api/v1/lifecycle",
			"audit": "/mfau/api/v1/audit"
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
	"""Get MFAU capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


def register_mfa_capability() -> dict[str, Any]:
	"""Compatibility alias for older MFA registration callers."""
	return register_capability()


__all__ = ["APG_CAPABILITY_INFO", "capability_metadata", "register_capability", "register_mfa_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
