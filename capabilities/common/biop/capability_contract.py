"""Executable capability contract for APG Biometric Processing."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"consent": {
		"explicit_consent_required": True,
		"purpose_required": True,
		"jurisdiction_scope_required": True,
		"revocation_supported": True,
	},
	"modalities": {
		"enabled": ["face", "fingerprint", "voice", "iris", "behavioral"],
		"multi_modal_required_for_high_risk": True,
		"minimum_match_confidence": 0.86,
		"quality_threshold": 0.72
	},
	"enrollment": {
		"active_consent_required": True,
		"template_hash_required": True,
		"quality_gate_required": True,
		"retention_policy_required": True,
	},
	"templates": {
		"encrypted_storage_required": True,
		"template_rotation_days": 365,
		"raw_sample_retention": "disabled",
		"revocation_supported": True
	},
	"verification": {
		"active_template_required": True,
		"subject_template_match_required": True,
		"modality_template_match_required": True,
		"review_low_confidence": True,
	},
	"liveness": {
		"required_for_authentication": True,
		"minimum_liveness_score": 0.82,
		"presentation_attack_detection": True,
		"passive_liveness_allowed": True
	},
	"reviews": {
		"independent_reviewer_required": True,
		"review_notes_required": True,
		"duplicate_pending_review_blocked": True,
		"stale_review_mutation_blocked": True,
	},
	"privacy": {
		"cross_border_processing_review": True,
		"jurisdiction_mismatch_blocks_processing": True,
		"privacy_review_notes_required": True,
	},
	"retention": {
		"raw_sample_retention": "disabled",
		"retire_templates_on_consent_revocation": True,
		"retirement_reason_required": True,
	},
	"security": {
		"tenant_isolation_required": True,
		"template_hash_required": True,
		"raw_sample_storage_allowed": False,
		"audit_state_changes": True,
	},
	"governance": {
		"require_tenant_context": True,
		"explicit_consent_required": True,
		"audit_template_access": True,
		"cross_border_processing_review": True
	},
	"observability": {
		"audit_required": True,
		"metrics_required": True,
		"trace_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "biometric_runtime.BiopService",
		"helper_runtime": "biometric_runtime.py",
		"api_helpers": "api_helpers.py",
		"view_models": "view_models.py",
		"production_runtime": "service.BiometricAuthenticationService",
		"production_api": "api.py",
		"production_views": "views.py",
		"event_stream": "bytewax",
		"mfa_provider": "mfau",
		"computer_vision": "cvsn",
		"ai_core": "aicr",
		"encryption": "encr",
		"audit_sink": "audl",
		"identity_federation": "idfd",
		"facial_recognition": "frec",
		"cache": "cach",
		"metrics_sink": "moni",
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
	"required": [
		"tenant_id",
		"consent",
		"modalities",
		"enrollment",
		"templates",
		"verification",
		"liveness",
		"reviews",
		"privacy",
		"retention",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"consent",
		"modalities",
		"enrollment",
		"templates",
		"verification",
		"liveness",
		"reviews",
		"privacy",
		"retention",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All biometric operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "consent_requires_subject", "description": "Biometric consent requires a subject.", "condition": {"operation": "record_consent", "subject_present": False}, "effect": {"decision": "deny", "reason": "biometric_subject_required", "required_action": "select_subject"}},
	{"name": "consent_requires_purpose", "description": "Biometric consent requires a processing purpose.", "condition": {"operation": "record_consent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "biometric_consent_purpose_required", "required_action": "record_consent_purpose"}},
	{"name": "consent_requires_modalities", "description": "Biometric consent requires scoped modalities.", "condition": {"operation": "record_consent", "modalities_present": False}, "effect": {"decision": "deny", "reason": "biometric_consent_modalities_required", "required_action": "record_modality_scope"}},
	{"name": "consent_requires_jurisdictions", "description": "Biometric consent requires jurisdiction scope.", "condition": {"operation": "record_consent", "jurisdictions_present": False}, "effect": {"decision": "deny", "reason": "biometric_consent_jurisdictions_required", "required_action": "record_jurisdiction_scope"}},
	{"name": "consent_requires_evidence", "description": "Biometric consent requires evidence.", "condition": {"operation": "record_consent", "evidence_present": False}, "effect": {"decision": "deny", "reason": "biometric_consent_evidence_required", "required_action": "attach_consent_evidence"}},
	{"name": "biometric_processing_requires_consent", "description": "Biometric processing requires explicit consent.", "condition": {"operation": "process_biometric", "consent_recorded": False}, "effect": {"decision": "deny", "reason": "biometric_consent_required", "required_action": "record_consent"}},
	{"name": "enrollment_requires_active_consent", "description": "Biometric enrollment requires active consent.", "condition": {"operation": "enroll_template", "active_consent_present": False}, "effect": {"decision": "deny", "reason": "active_biometric_consent_required", "required_action": "record_or_restore_consent"}},
	{"name": "enrollment_modality_requires_consent_scope", "description": "Enrollment modality must be within consent scope.", "condition": {"operation": "enroll_template", "modality_in_consent_scope": False}, "effect": {"decision": "deny", "reason": "biometric_consent_modality_mismatch", "required_action": "update_consent_scope"}},
	{"name": "template_requires_hash", "description": "Biometric template enrollment requires a template hash.", "condition": {"operation": "enroll_template", "template_hash_present": False}, "effect": {"decision": "deny", "reason": "biometric_template_hash_required", "required_action": "attach_template_hash"}},
	{"name": "template_storage_requires_encryption", "description": "Stored biometric templates must be encrypted.", "condition": {"operation": "store_template", "template_encrypted": False}, "effect": {"decision": "deny", "reason": "template_encryption_required", "required_action": "encrypt_template"}},
	{"name": "template_quality_requires_threshold", "description": "Template quality must meet tenant threshold.", "condition": {"operation": "enroll_template", "quality_score_lt": 0.72}, "effect": {"decision": "deny", "reason": "biometric_template_quality_too_low", "required_action": "capture_higher_quality_sample"}},
	{"name": "template_requires_retention_policy", "description": "Template enrollment requires a retention policy.", "condition": {"operation": "enroll_template", "retention_policy_present": False}, "effect": {"decision": "deny", "reason": "biometric_template_retention_policy_required", "required_action": "select_retention_policy"}},
	{"name": "raw_sample_retention_denied", "description": "Raw biometric samples may not be retained in the package runtime.", "condition": {"raw_sample_retention_requested": True}, "effect": {"decision": "deny", "reason": "raw_biometric_sample_retention_denied", "required_action": "store_encrypted_template_metadata_only"}},
	{"name": "authentication_requires_liveness", "description": "Authentication using biometrics requires liveness evidence.", "condition": {"operation": "authenticate", "liveness_passed": False}, "effect": {"decision": "deny", "reason": "liveness_required", "required_action": "complete_liveness_check"}},
	{"name": "liveness_score_requires_threshold", "description": "Liveness score must meet tenant threshold.", "condition": {"operation": "authenticate", "liveness_score_lt": 0.82}, "effect": {"decision": "deny", "reason": "liveness_required", "required_action": "complete_liveness_check"}},
	{"name": "presentation_attack_blocks_authentication", "description": "Presentation-attack signals block biometric authentication.", "condition": {"operation": "authenticate", "presentation_attack_detected": True}, "effect": {"decision": "deny", "reason": "presentation_attack_detected", "required_action": "escalate_security_review"}},
	{"name": "cross_border_use_requires_review", "description": "Cross-border biometric use requires governance review.", "condition": {"cross_border_processing": True, "privacy_review_recorded": False}, "effect": {"decision": "deny", "reason": "privacy_review_required", "required_action": "record_privacy_review"}},
	{"name": "jurisdiction_scope_blocks_processing", "description": "Processing outside consent jurisdiction scope is denied.", "condition": {"operation": "process_biometric", "jurisdiction_in_consent_scope": False}, "effect": {"decision": "deny", "reason": "biometric_consent_jurisdiction_mismatch", "required_action": "update_consent_or_change_processing_jurisdiction"}},
	{"name": "low_match_confidence_requires_review", "description": "Low-confidence biometric matches require human review.", "condition": {"match_confidence_lt": 0.86, "human_review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_match_confidence", "required_action": "review_match"}},
	{"name": "biometric_operation_requires_active_consent", "description": "Biometric operations require active scoped consent evidence.", "condition": {"active_consent_present": False}, "effect": {"decision": "deny", "reason": "active_biometric_consent_required", "required_action": "record_or_restore_consent"}},
	{"name": "verification_requires_active_template", "description": "Biometric verification requires an active encrypted template.", "condition": {"active_template_present": False}, "effect": {"decision": "deny", "reason": "active_biometric_template_required", "required_action": "enroll_active_template"}},
	{"name": "verification_subject_requires_template_subject", "description": "Verification subject must match template subject.", "condition": {"operation": "verify_template", "subject_matches_template": False}, "effect": {"decision": "deny", "reason": "biometric_template_subject_mismatch", "required_action": "select_subject_template"}},
	{"name": "verification_modality_requires_template_modality", "description": "Verification modality must match template modality.", "condition": {"operation": "verify_template", "modality_matches_template": False}, "effect": {"decision": "deny", "reason": "biometric_template_modality_mismatch", "required_action": "select_matching_template"}},
	{"name": "high_risk_verification_requires_multi_modal", "description": "High-risk biometric verification requires multiple modalities.", "condition": {"operation": "verify_template", "risk_level": "high", "multi_modal_evidence_present": False}, "effect": {"decision": "deny", "reason": "multi_modal_biometric_evidence_required", "required_action": "collect_additional_modality"}},
	{"name": "privacy_review_requires_justification", "description": "Privacy review requests require justification.", "condition": {"operation": "request_privacy_review", "justification_present": False}, "effect": {"decision": "deny", "reason": "privacy_review_justification_required", "required_action": "record_privacy_justification"}},
	{"name": "match_review_requires_justification", "description": "Match review requests require justification.", "condition": {"operation": "request_match_review", "justification_present": False}, "effect": {"decision": "deny", "reason": "match_review_justification_required", "required_action": "record_match_justification"}},
	{"name": "duplicate_pending_review_blocked", "description": "Duplicate pending biometric reviews are blocked.", "condition": {"operation": "request_review", "pending_review_exists": True}, "effect": {"decision": "deny", "reason": "biometric_review_already_pending", "required_action": "complete_existing_review"}},
	{"name": "match_review_requires_independent_reviewer", "description": "Low-confidence biometric match reviews require an independent reviewer.", "condition": {"operation": "approve_match_review", "match_reviewer_same_as_requester": True}, "effect": {"decision": "deny", "reason": "independent_match_reviewer_required", "required_action": "route_to_independent_match_reviewer"}},
	{"name": "privacy_review_requires_independent_reviewer", "description": "Cross-border biometric privacy reviews require an independent reviewer.", "condition": {"operation": "approve_privacy_review", "privacy_reviewer_same_as_requester": True}, "effect": {"decision": "deny", "reason": "independent_privacy_reviewer_required", "required_action": "route_to_independent_privacy_reviewer"}},
	{"name": "review_decision_requires_notes", "description": "Biometric review decisions require notes.", "condition": {"operation": "decide_review", "notes_present": False}, "effect": {"decision": "deny", "reason": "biometric_review_notes_required", "required_action": "record_review_notes"}},
	{"name": "stale_review_decision_blocked", "description": "Already-decided biometric reviews cannot be changed.", "condition": {"operation": "decide_review", "review_already_decided": True}, "effect": {"decision": "deny", "reason": "biometric_review_already_decided", "required_action": "open_new_review_if_needed"}},
	{"name": "rejected_privacy_review_blocks_verification", "description": "Rejected privacy reviews deny cross-border verification.", "condition": {"operation": "apply_privacy_review", "privacy_review_decision": "rejected"}, "effect": {"decision": "deny", "reason": "privacy_review_rejected", "required_action": "stop_cross_border_processing"}},
	{"name": "rejected_match_review_blocks_verification", "description": "Rejected match reviews deny biometric verification.", "condition": {"operation": "apply_match_review", "match_review_decision": "rejected"}, "effect": {"decision": "deny", "reason": "match_review_rejected", "required_action": "reject_verification"}},
	{"name": "consent_revocation_retires_templates", "description": "Consent revocation requires active templates to be retired.", "condition": {"operation": "revoke_consent", "active_templates_retired": False}, "effect": {"decision": "deny", "reason": "consent_revocation_requires_template_retirement", "required_action": "retire_templates_for_consent"}},
	{"name": "template_retirement_requires_reason", "description": "Template retirement requires a reason.", "condition": {"operation": "retire_template", "retirement_reason_present": False}, "effect": {"decision": "deny", "reason": "biometric_template_retirement_reason_required", "required_action": "record_retirement_reason"}},
	{"name": "batch_biometric_mutation_requires_bytewax", "description": "Batch biometric lifecycle mutations must use Bytewax event streams.", "condition": {"operation": "batch_biometric_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "cross_tenant_biometric_access_denied", "description": "Biometric records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_biometric_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "biometric_state_change_requires_audit", "description": "Biometric state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "biometric_audit_event_required", "required_action": "record_biometric_audit_event"}}
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
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
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
