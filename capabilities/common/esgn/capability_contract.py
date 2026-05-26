"""Executable capability contract for APG Digital Forms and eSign."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"forms": {
		"template_owner_required": True,
		"schema_validation_required": True,
		"publication_approval_required": True,
		"regulated_field_dlp_required": True
	},
	"signatures": {
		"identity_verification_required": True,
		"signature_intent_required": True,
		"tamper_seal_required": True,
		"multi_party_routing_enabled": True
	},
	"evidence": {
		"audit_trail_required": True,
		"encrypted_evidence_required": True,
		"certificate_of_completion": True,
		"retention_policy_required": True
	},
	"governance": {
		"require_tenant_context": True,
		"compliance_framework_link_required": True,
		"recipient_consent_required": True,
		"delegated_signing_policy_required": True
	},
	"ui": {
		"enable_form_builder": True,
		"enable_envelope_console": True,
		"enable_signing_room": True,
		"enable_evidence_vault": True
	},
	"theme": {
		"default_theme": "esgn_forms_signing",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "forms", "signatures", "evidence", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["forms", "signatures", "evidence", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All e-sign operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "form_template_requires_owner", "description": "Form templates require an accountable owner.", "condition": {"operation": "create_form_template", "template_owner_assigned": False}, "effect": {"decision": "deny", "reason": "template_owner_required", "required_action": "assign_template_owner"}},
	{"name": "form_publication_requires_approval", "description": "Forms require approval before publication.", "condition": {"operation": "publish_form", "publication_approved": False}, "effect": {"decision": "deny", "reason": "publication_approval_required", "required_action": "approve_form_publication"}},
	{"name": "signing_requires_identity_verification", "description": "Signing requires verified signer identity.", "condition": {"operation": "sign_envelope", "identity_verified": False}, "effect": {"decision": "deny", "reason": "identity_verification_required", "required_action": "verify_signer_identity"}},
	{"name": "evidence_requires_encryption", "description": "Signature evidence packages must be encrypted.", "condition": {"evidence_package_created": True, "evidence_encrypted": False}, "effect": {"decision": "deny", "reason": "evidence_encryption_required", "required_action": "encrypt_evidence_package"}},
	{"name": "regulated_form_requires_compliance_review", "description": "Regulated forms require compliance review.", "condition": {"regulated_form": True, "compliance_review_recorded": False}, "effect": {"decision": "require_review", "reason": "compliance_review_required", "required_action": "review_regulated_form"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/esgn/dashboard", "component": "ESGNDashboard", "permission": "esgn:view", "nav_group": "Overview"},
	{"name": "forms", "path": "/esgn/forms", "component": "FormLibrary", "permission": "esgn:create_forms", "nav_group": "Forms"},
	{"name": "builder", "path": "/esgn/builder", "component": "FormBuilder", "permission": "esgn:create_forms", "nav_group": "Forms"},
	{"name": "submissions", "path": "/esgn/submissions", "component": "SubmissionQueue", "permission": "esgn:view", "nav_group": "Forms"},
	{"name": "envelopes", "path": "/esgn/envelopes", "component": "EnvelopeConsole", "permission": "esgn:send_envelopes", "nav_group": "Signatures"},
	{"name": "signing", "path": "/esgn/signing", "component": "SigningRoom", "permission": "esgn:sign", "nav_group": "Signatures"},
	{"name": "evidence", "path": "/esgn/evidence", "component": "EvidenceVault", "permission": "esgn:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/esgn/settings", "component": "ESGNSettings", "permission": "esgn:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "esgn_forms_signing",
	"tokens": {
		"color.primary": "#2C5282",
		"color.accent": "#B7791F",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"form_builder": {"icon": "file-pen", "status_indicator": "template-pill", "risk_style": "schema-band"},
		"envelope_console": {"visual": "recipient-routing", "highlight": "signature-chip"},
		"signing_room": {"visual": "signature-ceremony", "status_style": "identity-chip"},
		"evidence_vault": {"visual": "sealed-record-list", "status_style": "audit-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable ESGN capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "esgn",
		"display_name": "Digital Forms and eSign",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "flask_appbuilder",
			"view_module": "views.py",
			"api_prefix": "/esgn/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default ESGN governance rules."""
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
