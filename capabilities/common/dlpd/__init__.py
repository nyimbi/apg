"""APG Data Loss Prevention (DLPD) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .service import DlpdService

__version__ = "1.0.0"
__capability_id__ = "dlpd"
__capability_name__ = "Data Loss Prevention"
__apg_dependencies__ = ["secu", "encr", "nlpc", "anom"]

capability_metadata: dict[str, Any] = {
	"name": "dlpd",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware sensitive-data discovery, channel inspection, exfiltration prevention, and incident response",
	"category": "security_compliance",
	"subcategory": "data_loss_prevention",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["sensitive_data_discovery", "channel_inspection", "exfiltration_detection", "incident_response", "policy_enforcement"],
	"permissions": ["dlpd:view", "dlpd:inspect", "dlpd:manage_policies", "dlpd:respond", "dlpd:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register DLPD with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "dlpd",
		"aliases": ["data_loss_prevention", "dlp", "data_exfiltration_prevention"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["audl", "mqeb", "srch", "comp"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"sensitive_data_discovery": "Classify governed data across documents, streams, messages, and exports",
			"channel_inspection": "Inspect egress channels with tenant policy and anomaly context",
			"exfiltration_detection": "Detect unusual movement and block or quarantine risky transfers",
			"incident_response": "Create, assign, escalate, and audit DLP incidents",
			"capability_rules": "Evaluate deterministic data-loss-prevention rules",
			"visual_theming": "Apply DLP operations theme tokens and components"
		},
		"endpoints": {
			"policies": "/dlpd/api/v1/policies",
			"inspection": "/dlpd/api/v1/inspection",
			"incidents": "/dlpd/api/v1/incidents",
			"channels": "/dlpd/api/v1/channels",
			"classifiers": "/dlpd/api/v1/classifiers"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get DLPD capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "DlpdService", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
