"""APG Digital Forms and eSign (ESGN) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "esgn"
__capability_name__ = "Digital Forms and eSign"
__apg_dependencies__ = ["auth", "encr", "audl", "comp"]

capability_metadata: dict[str, Any] = {
	"name": "esgn",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Digital form templates, submissions, e-signature envelopes, signing ceremonies, evidence, and compliance controls",
	"category": "collaboration_communication",
	"subcategory": "digital_forms_esign",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["digital_forms", "signature_envelopes", "signing_ceremonies", "evidence_packages", "form_workflows"],
	"permissions": ["esgn:view", "esgn:create_forms", "esgn:send_envelopes", "esgn:sign", "esgn:manage_templates", "esgn:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register ESGN with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "esgn",
		"aliases": ["esign", "digital_forms", "electronic_signature"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["wflo", "ntfy", "idfd", "dlpd"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"digital_forms": "Create, validate, publish, and submit governed digital forms",
			"signature_envelopes": "Prepare signature envelopes, recipients, routing, and reminders",
			"signing_ceremonies": "Run identity-verified electronic signature ceremonies",
			"evidence_packages": "Assemble encrypted audit evidence for completed signatures",
			"capability_rules": "Evaluate deterministic digital-form and e-signature rules",
			"visual_theming": "Apply forms and signing theme tokens and components"
		},
		"endpoints": {
			"forms": "/esgn/api/v1/forms",
			"submissions": "/esgn/api/v1/submissions",
			"envelopes": "/esgn/api/v1/envelopes",
			"signing": "/esgn/api/v1/signing",
			"evidence": "/esgn/api/v1/evidence"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get ESGN capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
