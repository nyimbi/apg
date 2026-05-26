"""Executable capability contract for APG Pose Estimation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"models": {
		"enabled_models": ["movenet", "rtmpose", "vitpose", "swin_pose"],
		"model_policy_required": True,
		"minimum_keypoint_confidence": 0.72,
		"edge_model_supported": True
	},
	"tracking": {
		"session_owner_required": True,
		"max_persons_per_frame": 50,
		"temporal_consistency_enabled": True,
		"secure_stream_required": True
	},
	"analysis": {
		"biomechanical_analysis_enabled": True,
		"medical_grade_review_required": True,
		"three_d_reconstruction_enabled": True,
		"minimum_quality_score": 0.7
	},
	"governance": {
		"require_tenant_context": True,
		"audit_pose_sessions": True,
		"subject_consent_required": True,
		"sensitive_use_approval_required": True
	},
	"ui": {
		"enable_pose_dashboard": True,
		"enable_tracking_console": True,
		"enable_analysis_workbench": True,
		"enable_model_registry": True
	},
	"theme": {
		"default_theme": "pose_motion_intelligence",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "models", "tracking", "analysis", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["models", "tracking", "analysis", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All pose operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "subject_consent_required", "description": "Pose analysis requires subject consent.", "condition": {"operation": "analyze_pose", "subject_consent_recorded": False}, "effect": {"decision": "deny", "reason": "subject_consent_required", "required_action": "record_subject_consent"}},
	{"name": "tracking_session_requires_owner", "description": "Tracking sessions require an owner.", "condition": {"operation": "start_tracking", "session_owner_assigned": False}, "effect": {"decision": "deny", "reason": "session_owner_required", "required_action": "assign_session_owner"}},
	{"name": "secure_stream_required", "description": "Realtime pose tracking requires secure streams.", "condition": {"realtime_stream": True, "secure_stream": False}, "effect": {"decision": "deny", "reason": "secure_stream_required", "required_action": "enable_secure_stream"}},
	{"name": "sensitive_use_requires_approval", "description": "Sensitive pose use cases require approval.", "condition": {"sensitive_use": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "sensitive_use_approval_required", "required_action": "record_sensitive_use_approval"}},
	{"name": "low_pose_quality_requires_review", "description": "Low-quality pose results require recapture or review.", "condition": {"pose_quality_score_lt": 0.7, "quality_review_recorded": False}, "effect": {"decision": "require_review", "reason": "pose_quality_review_required", "required_action": "review_pose_quality"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/pose/dashboard", "component": "POSEDashboard", "permission": "pose:view", "nav_group": "Overview"},
	{"name": "estimate", "path": "/pose/estimate", "component": "PoseEstimator", "permission": "pose:estimate", "nav_group": "Runtime"},
	{"name": "tracking", "path": "/pose/tracking", "component": "TrackingConsole", "permission": "pose:track", "nav_group": "Runtime"},
	{"name": "analysis", "path": "/pose/analysis", "component": "BiomechanicalAnalysis", "permission": "pose:analyze", "nav_group": "Analysis"},
	{"name": "sessions", "path": "/pose/sessions", "component": "PoseSessions", "permission": "pose:view", "nav_group": "Analysis"},
	{"name": "models", "path": "/pose/models", "component": "PoseModelRegistry", "permission": "pose:manage_models", "nav_group": "Models"},
	{"name": "quality", "path": "/pose/quality", "component": "PoseQuality", "permission": "pose:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/pose/settings", "component": "POSESettings", "permission": "pose:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "pose_motion_intelligence",
	"tokens": {
		"color.primary": "#234E70",
		"color.accent": "#38A169",
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
		"pose_viewer": {"icon": "activity", "status_indicator": "quality-pill", "risk_style": "consent-band"},
		"tracking_timeline": {"visual": "motion-timeline", "highlight": "latency-chip"},
		"biomechanics_panel": {"visual": "angle-metrics", "status_style": "review-chip"},
		"model_registry": {"visual": "model-grid", "status_style": "confidence-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable POSE capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "pose",
		"display_name": "Pose Estimation",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "flask_appbuilder",
			"view_module": "views.py",
			"api_prefix": "/pose/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default POSE governance rules."""
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
