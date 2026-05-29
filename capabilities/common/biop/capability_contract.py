"""Executable capability contract for APG Biometric Processing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"modalities": {
		"enabled": ["face", "fingerprint", "voice", "iris", "behavioral"],
		"multi_modal_required_for_high_risk": True,
		"minimum_match_confidence": 0.86,
		"quality_threshold": 0.72
	},
	"templates": {
		"encrypted_storage_required": True,
		"template_rotation_days": 365,
		"raw_sample_retention": "disabled",
		"revocation_supported": True
	},
	"liveness": {
		"required_for_authentication": True,
		"minimum_liveness_score": 0.82,
		"presentation_attack_detection": True,
		"passive_liveness_allowed": True
	},
	"governance": {
		"require_tenant_context": True,
		"explicit_consent_required": True,
		"audit_template_access": True,
		"cross_border_processing_review": True
	},
	"ui": {
		"enable_biometric_dashboard": True,
		"enable_consent_center": True,
		"enable_enrollment_console": True,
		"enable_template_vault": True,
		"enable_verification_workbench": True,
		"enable_review_queues": True,
		"enable_compliance_view": True
	},
	"theme": {
		"default_theme": "biop_biometric_control",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "modalities", "templates", "liveness", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["modalities", "templates", "liveness", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All biometric operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "biometric_processing_requires_consent", "description": "Biometric processing requires explicit consent.", "condition": {"operation": "process_biometric", "consent_recorded": False}, "effect": {"decision": "deny", "reason": "biometric_consent_required", "required_action": "record_consent"}},
	{"name": "template_storage_requires_encryption", "description": "Stored biometric templates must be encrypted.", "condition": {"operation": "store_template", "template_encrypted": False}, "effect": {"decision": "deny", "reason": "template_encryption_required", "required_action": "encrypt_template"}},
	{"name": "authentication_requires_liveness", "description": "Authentication using biometrics requires liveness evidence.", "condition": {"operation": "authenticate", "liveness_passed": False}, "effect": {"decision": "deny", "reason": "liveness_required", "required_action": "complete_liveness_check"}},
	{"name": "cross_border_use_requires_review", "description": "Cross-border biometric use requires governance review.", "condition": {"cross_border_processing": True, "privacy_review_recorded": False}, "effect": {"decision": "deny", "reason": "privacy_review_required", "required_action": "record_privacy_review"}},
	{"name": "low_match_confidence_requires_review", "description": "Low-confidence biometric matches require human review.", "condition": {"match_confidence_lt": 0.86, "human_review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_match_confidence", "required_action": "review_match"}},
	{"name": "biometric_operation_requires_active_consent", "description": "Biometric operations require active scoped consent evidence.", "condition": {"active_consent_present": False}, "effect": {"decision": "deny", "reason": "active_biometric_consent_required", "required_action": "record_or_restore_consent"}},
	{"name": "verification_requires_active_template", "description": "Biometric verification requires an active encrypted template.", "condition": {"active_template_present": False}, "effect": {"decision": "deny", "reason": "active_biometric_template_required", "required_action": "enroll_active_template"}},
	{"name": "match_review_requires_independent_reviewer", "description": "Low-confidence biometric match reviews require an independent reviewer.", "condition": {"operation": "approve_match_review", "match_reviewer_same_as_requester": True}, "effect": {"decision": "deny", "reason": "independent_match_reviewer_required", "required_action": "route_to_independent_match_reviewer"}},
	{"name": "privacy_review_requires_independent_reviewer", "description": "Cross-border biometric privacy reviews require an independent reviewer.", "condition": {"operation": "approve_privacy_review", "privacy_reviewer_same_as_requester": True}, "effect": {"decision": "deny", "reason": "independent_privacy_reviewer_required", "required_action": "route_to_independent_privacy_reviewer"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/biop/dashboard", "component": "BIOPDashboard", "permission": "biop:view", "nav_group": "Overview"},
	{"name": "users", "path": "/biop/users", "component": "BiometricUsers", "permission": "biop:view", "nav_group": "Identity"},
	{"name": "consents", "path": "/biop/consents", "component": "BiometricConsentCenter", "permission": "biop:manage_consent", "nav_group": "Identity"},
	{"name": "enrollments", "path": "/biop/enrollments", "component": "BiometricEnrollments", "permission": "biop:enroll", "nav_group": "Identity"},
	{"name": "templates", "path": "/biop/templates", "component": "BiometricTemplateVault", "permission": "biop:manage_templates", "nav_group": "Identity"},
	{"name": "verification", "path": "/biop/verification", "component": "BiometricVerification", "permission": "biop:verify", "nav_group": "Verification"},
	{"name": "liveness", "path": "/biop/liveness", "component": "LivenessWorkbench", "permission": "biop:verify", "nav_group": "Verification"},
	{"name": "match_reviews", "path": "/biop/reviews/matches", "component": "BiometricMatchReviewQueue", "permission": "biop:review", "nav_group": "Governance"},
	{"name": "privacy_reviews", "path": "/biop/reviews/privacy", "component": "BiometricPrivacyReviewQueue", "permission": "biop:review_privacy", "nav_group": "Governance"},
	{"name": "compliance", "path": "/biop/compliance", "component": "BiometricCompliance", "permission": "biop:review", "nav_group": "Governance"},
	{"name": "analytics", "path": "/biop/analytics", "component": "BiometricAnalytics", "permission": "biop:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/biop/settings", "component": "BIOPSettings", "permission": "biop:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "biop_biometric_control",
	"tokens": {
		"color.primary": "#214E34",
		"color.accent": "#2B6CB0",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F8F7",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"modality_matrix": {"icon": "fingerprint", "status_indicator": "modality-pill", "risk_style": "consent-band"},
		"consent_center": {"visual": "scope-ledger", "status_style": "consent-chip"},
		"template_vault": {"visual": "encrypted-record-list", "highlight": "rotation-chip"},
		"liveness_panel": {"visual": "challenge-meter", "status_style": "pad-chip"},
		"match_result": {"visual": "confidence-meter", "status_style": "review-chip"},
		"match_review_queue": {"visual": "confidence-review-lane", "status_style": "match-review-chip"},
		"privacy_review_queue": {"visual": "jurisdiction-review-lane", "status_style": "privacy-chip"},
		"privacy_posture": {"visual": "jurisdiction-matrix", "status_style": "cross-border-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable BIOP capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "biop",
		"display_name": "Biometric Processing",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/biop/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default BIOP governance rules."""
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
