"""APG Compliance Management (COMP) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "comp"
__capability_name__ = "Compliance Management"
__apg_dependencies__ = ["audl", "dlpd", "encr", "auth"]

capability_metadata: dict[str, Any] = {
	"name": "comp",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Regulatory framework, control, evidence, finding, reporting, and compliance automation",
	"category": "security_compliance",
	"subcategory": "compliance_management",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["framework_management", "control_assurance", "evidence_collection", "finding_remediation", "regulatory_reporting"],
	"permissions": ["comp:view", "comp:manage_controls", "comp:collect_evidence", "comp:remediate", "comp:approve_reports", "comp:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register COMP with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "comp",
		"aliases": ["compliance", "compliance_management", "regulatory_compliance"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["secu", "mten", "idfd", "ztna"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"framework_management": "Map tenant obligations to frameworks, controls, owners, and evidence",
			"control_assurance": "Assess control design and operating effectiveness",
			"evidence_collection": "Collect, verify, retain, and audit compliance evidence",
			"finding_remediation": "Track findings, exceptions, and corrective-action plans",
			"capability_rules": "Evaluate deterministic compliance-management rules",
			"visual_theming": "Apply compliance-command-center theme tokens and components"
		},
		"endpoints": {
			"frameworks": "/comp/api/v1/frameworks",
			"controls": "/comp/api/v1/controls",
			"evidence": "/comp/api/v1/evidence",
			"findings": "/comp/api/v1/findings",
			"reports": "/comp/api/v1/reports"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get COMP capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
