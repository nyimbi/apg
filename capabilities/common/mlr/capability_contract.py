"""Executable capability contract for APG MLOps Pipeline."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = "common_mlr"
CAPABILITY_NAME = "MLOps Pipeline"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "common"
CAPABILITY_DESCRIPTION = (
    "MLOps pipeline: experiment tracking, feature store with point-in-time correctness, "
    "model registry with A/B promotion workflow, data drift detection. "
    "Completes the MLOps loop that common/mlcm (governance) and common/mlx (inference) begin."
)

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ml_engineer", "data_scientist", "mlops_admin", "model_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"experiment_tracking": {
		"max_experiments_per_project": 1000,
		"max_runs_per_experiment": 10000,
		"artifact_store": "local",
		"metric_history_retention_days": 365,
	},
	"feature_store": {
		"point_in_time_enabled": True,
		"online_store": "redis",
		"offline_store": "postgresql",
		"max_feature_staleness_minutes": 60,
	},
	"model_registry": {
		"stages": ["staging", "production", "archived"],
		"a_b_testing_enabled": True,
		"shadow_mode_enabled": True,
		"approval_required_for_production": True,
	},
	"drift_detection": {
		"enabled": True,
		"check_interval_hours": 24,
		"psi_threshold": 0.2,
		"js_distance_threshold": 0.1,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_events": True,
		"human_approval_required_for_production_promotion": True,
	},
}

PROVIDES = [
	"experiment_tracking", "run_comparison", "artifact_versioning",
	"feature_store", "feature_serving", "point_in_time_features",
	"model_registry", "model_promotion", "ab_testing", "shadow_deployment",
	"drift_detection", "data_quality_monitoring", "retraining_triggers",
]
REQUIRES = ["auth", "audl", "ntfy", "common_mlcm", "common_mlx"]
PUBLISHES = [
	"experiment.run_completed", "model.promoted_to_production",
	"drift.detected", "retraining.triggered", "a_b_test.winner_declared",
]
SUBSCRIBES = [
	{"source_capability": "common_mlx", "event_type": "inference.completed", "handler": "on_inference_for_drift_monitoring"},
]

UI_ROUTES = [
	{"name": "experiments", "path": "/mlops/experiments", "component": "MlrExperiments", "permission": "common_mlr:view", "nav_group": "Experiments"},
	{"name": "feature_store", "path": "/mlops/features", "component": "MlrFeatureStore", "permission": "common_mlr:view", "nav_group": "Features"},
	{"name": "model_registry", "path": "/mlops/registry", "component": "MlrModelRegistry", "permission": "common_mlr:view", "nav_group": "Models"},
	{"name": "drift", "path": "/mlops/drift", "component": "MlrDriftMonitor", "permission": "common_mlr:view", "nav_group": "Monitoring"},
	{"name": "settings", "path": "/mlops/settings", "component": "MlrSettings", "permission": "common_mlr:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "common_mlr_theme",
	"tokens": {
		"color.primary": "#5B21B6",
		"color.accent": "#7C3AED",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "8px",
		"density": "compact",
	},
}


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"id": CAPABILITY_ID, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION,
		"provides": PROVIDES, "requires": REQUIRES, "publishes": PUBLISHES,
		"subscribes": SUBSCRIBES, "ui_routes": UI_ROUTES, "theme": THEME,
		"configuration": DEFAULT_CONFIGURATION,
	
		"rule_engine": {
			"type": "deterministic",
			"default_decision": "deny",
			"rules": [
				{"name": "tenant_required", "condition": {"tenant_context_present": True}, "effect": {"decision": "allow"}},
				{"name": "write_policy", "condition": {"write_requires_policy": True}, "effect": {"decision": "allow"}},
				{"name": "cross_tenant_denied", "condition": {"cross_tenant_access": "cross_tenant"}, "effect": {"decision": "deny"}},
				{"name": "audit_required", "condition": {"audit_enabled": True}, "effect": {"decision": "allow"}},
				{"name": "rate_limit_enforced", "condition": {"rate_limit_exceeded": False}, "effect": {"decision": "allow"}},
				{"name": "auth_required", "condition": {"authenticated": True}, "effect": {"decision": "allow"}},
				{"name": "permission_check", "condition": {"has_permission": True}, "effect": {"decision": "allow"}},
				{"name": "data_validation", "condition": {"data_valid": True}, "effect": {"decision": "allow"}},
				{"name": "resource_exists", "condition": {"resource_present": True}, "effect": {"decision": "allow"}},
				{"name": "scope_enforced", "condition": {"scope_valid": True}, "effect": {"decision": "allow"}},
			],
		},
		"ui": {
			"shell": "apg_python",
			"requires_theme": True,
			"template_roots": ["templates"],
			"routes": [{'name': 'experiments', 'path': '/mlops/experiments', 'component': 'MlrExperiments', 'permission': 'common_mlr:view', 'nav_group': 'Experiments'}, {'name': 'feature_store', 'path': '/mlops/features', 'component': 'MlrFeatureStore', 'permission': 'common_mlr:view', 'nav_group': 'Features'}, {'name': 'model_registry', 'path': '/mlops/registry', 'component': 'MlrModelRegistry', 'permission': 'common_mlr:view', 'nav_group': 'Models'}, {'name': 'drift', 'path': '/mlops/drift', 'component': 'MlrDriftMonitor', 'permission': 'common_mlr:view', 'nav_group': 'Monitoring'}, {'name': 'settings', 'path': '/mlops/settings', 'component': 'MlrSettings', 'permission': 'common_mlr:admin', 'nav_group': 'Administration'}],
		},
		"configuration_schema": {
			"type": "object",
			"required": ['tenant_id'],
			"properties": {
				"tenant_id": {"type": "string"},
				"experiment_tracking": {"type": "object"},
				"feature_store": {"type": "object"},
				"model_registry": {"type": "object"},
				"drift_detection": {"type": "object"},
				"governance": {"type": "object"},
			},
		},
}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	if not context.get("tenant_context_present"):
		return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": [{"type": "deny", "reason": "missing_tenant_context"}]}
	if context.get("operation") == "promote_to_production" and not context.get("human_approval_present"):
		return {"decision": "require_review", "matched_rules": ["production_approval_required"], "actions": [{"type": "require_approval", "reason": "production_promotion_requires_human_sign_off"}]}
	return {"decision": "allow", "matched_rules": [], "actions": []}
