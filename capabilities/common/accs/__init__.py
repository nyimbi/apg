"""APG Accessibility Services capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .service import AccsService

__version__ = "1.0.0"
__capability_id__ = "accs"
__capability_name__ = "Accessibility Services"
__apg_dependencies__ = ["them", "i18n", "nlpc"]

capability_metadata: dict[str, Any] = {
	"name": "accs",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Accessibility standards, audits, remediation workflows, assistive metadata, and inclusive UI governance",
	"category": "experience",
	"subcategory": "accessibility",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["accessibility_audits", "remediation_workflows", "assistive_metadata", "media_accessibility", "standards_governance", "accessibility_agents"],
	"permissions": ["accs:view", "accs:audit", "accs:remediate", "accs:manage_standards", "accs:review", "accs:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register ACCS with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "accs",
		"aliases": ["accessibility", "a11y", "inclusive-ui"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["audl", "comp", "help", "wsbl", "bytewax", "aicr"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"accessibility_audits": "Run standards-based audits across UI, content, and media surfaces",
			"remediation_workflows": "Track findings, owners, due dates, review, and closure",
			"review_governance": "Record formal review decisions before critical findings can close",
			"assistive_metadata": "Manage labels, descriptions, landmarks, and assistive hints",
			"media_accessibility": "Govern captions, transcripts, alt text, and media alternatives",
			"accessibility_agents": "Register governed AI accessibility agents with runtime, role, scope, disclosure, and audit",
			"capability_rules": "Evaluate deterministic accessibility-governance rules",
			"visual_theming": "Apply accessibility-operations theme tokens and components"
		},
		"endpoints": {"audits": "/accs/api/v1/audits", "findings": "/accs/api/v1/findings", "remediation": "/accs/api/v1/remediation", "reviews": "/accs/api/v1/reviews", "assistive": "/accs/api/v1/assistive", "standards": "/accs/api/v1/standards", "agents": "/accs/api/v1/agents", "audit": "/accs/api/v1/audit"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get ACCS capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["AccsService", "capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
