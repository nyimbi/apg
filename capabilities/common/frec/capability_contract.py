"""Executable capability contract for APG Facial Recognition."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"recognition": {
		"enabled_modes": ["verification", "identification", "watchlist_matching"],
		"minimum_face_quality": 0.72,
		"verification_threshold": 0.88,
		"identification_threshold": 0.92
	},
	"liveness": {
		"required_for_authentication": True,
		"minimum_liveness_score": 0.84,
		"anti_spoofing_enabled": True,
		"deepfake_detection_enabled": True
	},
	"emotion": {
		"emotion_analysis_enabled": False,
		"explicit_purpose_required": True,
		"aggregate_only_by_default": True
	},
	"privacy": {
		"explicit_consent_required": True,
		"watchlist_policy_required": True,
		"audit_identification": True,
		"template_encryption_required": True
	},
	"ui": {
		"enable_identity_dashboard": True,
		"enable_enrollment_console": True,
		"enable_watchlist_manager": True,
		"enable_privacy_center": True
	},
	"theme": {
		"default_theme": "frec_identity_vision",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "recognition", "liveness", "emotion", "privacy", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["recognition", "liveness", "emotion", "privacy", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All facial recognition operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "face_enrollment_requires_consent", "description": "Face enrollment requires explicit consent.", "condition": {"operation": "enroll_face", "consent_recorded": False}, "effect": {"decision": "deny", "reason": "face_consent_required", "required_action": "record_face_consent"}},
	{"name": "identification_requires_watchlist_policy", "description": "Identification requires an active watchlist policy.", "condition": {"operation": "identify_face", "watchlist_policy_attached": False}, "effect": {"decision": "deny", "reason": "watchlist_policy_required", "required_action": "attach_watchlist_policy"}},
	{"name": "authentication_requires_liveness", "description": "Face authentication requires liveness evidence.", "condition": {"operation": "authenticate_face", "liveness_passed": False}, "effect": {"decision": "deny", "reason": "liveness_required", "required_action": "complete_liveness_check"}},
	{"name": "emotion_analysis_requires_explicit_purpose", "description": "Emotion analysis requires an explicit approved purpose.", "condition": {"emotion_analysis_requested": True, "approved_purpose_recorded": False}, "effect": {"decision": "deny", "reason": "emotion_purpose_required", "required_action": "record_approved_purpose"}},
	{"name": "low_face_quality_requires_recapture", "description": "Low-quality face captures require recapture or review.", "condition": {"face_quality_lt": 0.72, "recapture_completed": False}, "effect": {"decision": "require_review", "reason": "low_face_quality", "required_action": "recapture_or_review"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/frec/dashboard", "component": "FRECDashboard", "permission": "frec:view", "nav_group": "Overview"},
	{"name": "enrollment", "path": "/frec/enrollment", "component": "FaceEnrollment", "permission": "frec:enroll", "nav_group": "Identity"},
	{"name": "verification", "path": "/frec/verification", "component": "FaceVerification", "permission": "frec:verify", "nav_group": "Identity"},
	{"name": "identification", "path": "/frec/identification", "component": "FaceIdentification", "permission": "frec:identify", "nav_group": "Identity"},
	{"name": "liveness", "path": "/frec/liveness", "component": "FaceLiveness", "permission": "frec:verify", "nav_group": "Security"},
	{"name": "emotion", "path": "/frec/emotion", "component": "EmotionGovernance", "permission": "frec:admin", "nav_group": "Governance"},
	{"name": "watchlists", "path": "/frec/watchlists", "component": "WatchlistManager", "permission": "frec:manage_watchlists", "nav_group": "Governance"},
	{"name": "settings", "path": "/frec/settings", "component": "FRECSettings", "permission": "frec:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "frec_identity_vision",
	"tokens": {
		"color.primary": "#234E70",
		"color.accent": "#C05621",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"face_quality_panel": {"icon": "scan-face", "status_indicator": "quality-pill", "risk_style": "capture-band"},
		"match_gallery": {"visual": "ranked-face-grid", "highlight": "confidence-chip"},
		"liveness_trace": {"visual": "challenge-timeline", "status_style": "spoof-chip"},
		"watchlist_table": {"visual": "identity-list", "status_style": "policy-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable FREC capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "frec",
		"display_name": "Facial Recognition",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/frec/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default FREC governance rules."""
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
