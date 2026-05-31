"""APG Compliance Management (COMP) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .service import CompService

__version__ = "1.0.0"
__capability_id__ = "comp"
__capability_name__ = "Compliance Management"
__apg_dependencies__ = ["audl", "dlpd", "encr", "auth", "secu", "mten", "idfd", "ztna", "mqeb", "cach"]

capability_metadata: dict[str, Any] = {
	"name": "comp",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Regulatory framework, control, evidence, finding, reporting, compliance automation, and governed AI-agent composition",
	"category": "security_compliance",
	"subcategory": "compliance_management",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": [
		"framework_management",
		"obligation_mapping",
		"control_assurance",
		"evidence_collection",
		"assessment_management",
		"finding_remediation",
		"exception_management",
		"regulatory_reporting",
		"attestation_management",
		"compliance_audit_events",
		"compliance_agent_composition",
		"bytewax_lifecycle_governance",
	],
	"permissions": ["comp:view", "comp:manage_controls", "comp:collect_evidence", "comp:remediate", "comp:approve_reports", "comp:audit", "comp:admin"]
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
		"optional_dependencies": ["docm", "wflo", "ntfy"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"framework_management": "Map tenant obligations to frameworks, controls, owners, and evidence",
			"obligation_mapping": "Track framework obligations, policy versions, and control coverage",
			"control_assurance": "Assess control design and operating effectiveness",
			"evidence_collection": "Collect, verify, retain, and audit compliance evidence",
			"assessment_management": "Route stale, failed, or owner-tested controls into review workflows",
			"finding_remediation": "Track findings, exceptions, and corrective-action plans",
			"exception_management": "Govern compliance exceptions with ownership and expiry evidence",
			"regulatory_reporting": "Prepare, approve, attest, publish, and export compliance reports",
			"attestation_management": "Record accountable attestation statements before report publication",
			"compliance_audit_events": "Hash and expose immutable audit-event metadata for compliance state changes",
			"compliance_agent_composition": "Register provider-neutral compliance agents with scoped roles, accountable owners, contribution disclosure, and human approval guardrails",
			"bytewax_lifecycle_governance": "Validate compliance lifecycle batches against Bytewax stream and mutation guardrails",
			"capability_rules": "Evaluate deterministic compliance-management rules",
			"visual_theming": "Apply compliance-command-center theme tokens and components"
		},
		"endpoints": {
			"status": "/comp/api/v1/status",
			"frameworks": "/comp/api/v1/frameworks",
			"controls": "/comp/api/v1/controls",
			"evidence": "/comp/api/v1/evidence",
			"assessments": "/comp/api/v1/assessments",
			"findings": "/comp/api/v1/findings",
			"exceptions": "/comp/api/v1/exceptions",
			"reports": "/comp/api/v1/reports",
			"attestations": "/comp/api/v1/attestations",
			"exports": "/comp/api/v1/exports",
			"audit": "/comp/api/v1/audit",
			"agents": "/comp/api/v1/agents",
			"lifecycle": "/comp/api/v1/lifecycle"
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
	"""Get COMP capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "CompService", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
