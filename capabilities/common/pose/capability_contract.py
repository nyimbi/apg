"""Executable capability contract for APG Pose Estimation."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["model_reviewer", "movement_analyst", "quality_reviewer", "session_observer", "edge_optimizer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"models": {
		"enabled_models": ["movenet", "rtmpose", "vitpose", "swin_pose"],
		"model_policy_required": True,
		"model_owner_required": True,
		"minimum_keypoint_confidence": 0.72,
		"edge_model_supported": True,
	},
	"sessions": {
		"session_owner_required": True,
		"subject_consent_required": True,
		"source_reference_required": True,
		"secure_stream_required": True,
		"sensitive_use_approval_required": True,
		"state_change_reason_required": True,
	},
	"tracking": {
		"max_persons_per_frame": 50,
		"temporal_consistency_enabled": True,
		"frame_timestamp_required": True,
		"keypoint_schema": "coco17",
	},
	"analysis": {
		"biomechanical_analysis_enabled": True,
		"medical_grade_review_required": True,
		"three_d_reconstruction_enabled": True,
		"minimum_quality_score": 0.7,
		"three_d_calibration_required": True,
	},
	"pose_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_AGENT_ROLES,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_pose_sessions": True,
		"subject_consent_required": True,
		"sensitive_use_approval_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"quality_metrics_required": True,
		"latency_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.PoseService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"computer_vision": "cvsn",
		"ai_core": "aicr",
		"model_lifecycle": "mlcm",
		"audit_sink": "audl",
		"monitoring": "moni",
		"edge_runtime": "edge",
	},
	"ui": {
		"enable_pose_dashboard": True,
		"enable_tracking_console": True,
		"enable_analysis_workbench": True,
		"enable_model_registry": True,
		"enable_quality_review": True,
		"enable_3d_reconstruction": True,
		"enable_agent_panel": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {
		"default_theme": "pose_motion_intelligence",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"models",
		"sessions",
		"tracking",
		"analysis",
		"pose_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"models",
		"sessions",
		"tracking",
		"analysis",
		"pose_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {
		"tenant_id": {"type": "string", "minLength": 1},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All pose operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "pose_model_requires_owner", "description": "Pose models require an accountable owner.", "condition": {"operation": "register_model", "model_owner_present": False}, "effect": {"decision": "deny", "reason": "model_owner_required", "required_action": "assign_model_owner"}},
	{"name": "pose_model_requires_policy", "description": "Pose models require model-use policy.", "condition": {"operation": "register_model", "model_policy_attached": False}, "effect": {"decision": "deny", "reason": "model_policy_required", "required_action": "attach_model_policy"}},
	{"name": "tracking_session_requires_owner", "description": "Tracking sessions require an owner.", "condition": {"operation": "start_tracking", "session_owner_assigned": False}, "effect": {"decision": "deny", "reason": "session_owner_required", "required_action": "assign_session_owner"}},
	{"name": "subject_consent_required", "description": "Pose sessions and analysis require subject consent.", "condition": {"subject_consent_recorded": False}, "effect": {"decision": "deny", "reason": "subject_consent_required", "required_action": "record_subject_consent"}},
	{"name": "tracking_source_required", "description": "Tracking sessions require a source reference.", "condition": {"operation": "start_tracking", "source_reference_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "secure_stream_required", "description": "Realtime pose tracking requires secure streams.", "condition": {"realtime_stream": True, "secure_stream": False}, "effect": {"decision": "deny", "reason": "secure_stream_required", "required_action": "enable_secure_stream"}},
	{"name": "sensitive_use_requires_approval", "description": "Sensitive pose use cases require approval.", "condition": {"sensitive_use": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "sensitive_use_approval_required", "required_action": "record_sensitive_use_approval"}},
	{"name": "frame_requires_timestamp", "description": "Pose frames require timestamp evidence.", "condition": {"operation": "record_frame", "frame_timestamp_present": False}, "effect": {"decision": "deny", "reason": "frame_timestamp_required", "required_action": "attach_frame_timestamp"}},
	{"name": "pose_estimate_requires_keypoints", "description": "Pose estimates require keypoints.", "condition": {"operation": "estimate_pose", "keypoint_count_lte": 0}, "effect": {"decision": "deny", "reason": "pose_keypoints_required", "required_action": "provide_pose_keypoints"}},
	{"name": "pose_estimate_person_limit", "description": "Pose estimates must respect max person limit.", "condition": {"operation": "estimate_pose", "person_count_gt": 50}, "effect": {"decision": "deny", "reason": "max_persons_exceeded", "required_action": "reduce_person_count_or_adjust_policy"}},
	{"name": "low_pose_quality_requires_review", "description": "Low-quality pose results require recapture or review.", "condition": {"pose_quality_score_lt": 0.7, "quality_review_recorded": False}, "effect": {"decision": "require_review", "reason": "pose_quality_review_required", "required_action": "review_pose_quality"}},
	{"name": "medical_analysis_requires_review", "description": "Medical-grade biomechanical analysis requires review evidence.", "condition": {"operation": "analyze_pose", "medical_grade": True, "medical_review_recorded": False}, "effect": {"decision": "deny", "reason": "medical_review_required", "required_action": "record_medical_review"}},
	{"name": "three_d_requires_calibration", "description": "3D reconstruction requires camera calibration.", "condition": {"operation": "reconstruct_3d", "camera_calibration_present": False}, "effect": {"decision": "deny", "reason": "camera_calibration_required", "required_action": "attach_camera_calibration"}},
	{"name": "pose_agent_requires_registration", "description": "AI pose agents must be registered.", "condition": {"pose_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "pose_agent_registration_required", "required_action": "register_pose_agent"}},
	{"name": "pose_agent_runtime_supported", "description": "AI pose agents must use a supported runtime.", "condition": {"pose_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "pose_agent_runtime_not_supported", "required_action": "choose_supported_pose_agent_runtime"}},
	{"name": "pose_agent_requires_scope", "description": "AI pose agents require explicit scope.", "condition": {"pose_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "pose_agent_scope_required", "required_action": "set_pose_agent_scope"}},
	{"name": "pose_agent_requires_disclosure", "description": "AI pose-agent contributions require disclosure.", "condition": {"pose_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "pose_agent_disclosure_required", "required_action": "disclose_pose_agent"}},
	{"name": "state_change_requires_reason", "description": "Pose session state changes require a reason.", "condition": {"state_change_requested": True, "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "state_change_reason_required", "required_action": "record_state_change_reason"}},
	{"name": "state_change_requires_audit", "description": "Pose session state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "pose_audit_event_required", "required_action": "record_pose_audit"}},
	{"name": "cross_tenant_pose_access_denied", "description": "Pose records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_pose_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_pose_mutation_requires_bytewax", "description": "Batch pose mutations must use Bytewax event streams.", "condition": {"operation": "batch_pose_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/pose/dashboard", "component": "POSEDashboard", "permission": "pose:view", "nav_group": "Overview"},
	{"name": "estimate", "path": "/pose/estimate", "component": "PoseEstimator", "permission": "pose:estimate", "nav_group": "Runtime"},
	{"name": "tracking", "path": "/pose/tracking", "component": "TrackingConsole", "permission": "pose:track", "nav_group": "Runtime"},
	{"name": "analysis", "path": "/pose/analysis", "component": "BiomechanicalAnalysis", "permission": "pose:analyze", "nav_group": "Analysis"},
	{"name": "reconstruction", "path": "/pose/reconstruction", "component": "Pose3DReconstruction", "permission": "pose:analyze", "nav_group": "Analysis"},
	{"name": "sessions", "path": "/pose/sessions", "component": "PoseSessions", "permission": "pose:view", "nav_group": "Analysis"},
	{"name": "models", "path": "/pose/models", "component": "PoseModelRegistry", "permission": "pose:manage_models", "nav_group": "Models"},
	{"name": "quality", "path": "/pose/quality", "component": "PoseQuality", "permission": "pose:view", "nav_group": "Governance"},
	{"name": "agents", "path": "/pose/agents", "component": "PoseAgentPanel", "permission": "pose:manage_models", "nav_group": "Agents"},
	{"name": "audit", "path": "/pose/audit", "component": "POSEAuditTrail", "permission": "pose:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/pose/analytics", "component": "POSEAnalytics", "permission": "pose:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/pose/settings", "component": "POSESettings", "permission": "pose:admin", "nav_group": "Administration"},
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
		"density": "compact",
	},
	"components": {
		"pose_viewer": {"icon": "activity", "status_indicator": "quality-pill", "risk_style": "consent-band"},
		"tracking_timeline": {"visual": "motion-timeline", "highlight": "latency-chip"},
		"biomechanics_panel": {"visual": "angle-metrics", "status_style": "review-chip"},
		"reconstruction_panel": {"visual": "skeleton-3d", "status_style": "calibration-chip"},
		"model_registry": {"visual": "model-grid", "status_style": "confidence-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
	}
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.pose.lifecycle",
	"state": ["models", "sessions", "frames", "estimates", "analyses", "reconstructions", "pose_agents"],
	"events": [
		"pose_model_registered",
		"pose_session_started",
		"pose_frame_recorded",
		"pose_estimated",
		"pose_analysis_completed",
		"pose_3d_reconstructed",
		"pose_agent_registered",
		"pose_session_state_changed",
	],
	"batch_mutation_guardrail": "batch_pose_mutation_requires_bytewax",
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
		"provides": ["pose_estimation", "multi_person_tracking", "biomechanical_analysis", "pose_3d_reconstruction", "edge_pose_inference", "pose_agents", "pose_quality_governance"],
		"requires": ["cvsn", "aicr", "mlcm"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/pose/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
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
