"""Executable capability contract for APG AI Model Lifecycle Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"registry": {
		"model_registry_enabled": True,
		"owner_required": True,
		"risk_level_required": True,
		"max_models_per_tenant": 10000,
	},
	"versions": {
		"versioning_required": True,
		"artifact_uri_required": True,
		"model_card_required": True,
		"training_data_required": True,
		"baseline_required": True,
	},
	"evaluation": {
		"baseline_required": True,
		"evidence_required": True,
		"minimum_eval_score": 0.8,
		"fairness_review_required_for_high_risk": True,
		"explainability_required_for_high_risk": True,
	},
	"promotion": {
		"stage_gates": ["dev", "staging", "production"],
		"approval_required_for_production": True,
		"evaluation_required": True,
		"rollback_enabled": True,
	},
	"deployment": {
		"active_target_required": True,
		"canary_limit_percent": 50,
		"approval_required_for_production": True,
		"health_check_required": True,
		"rollback_enabled": True,
	},
	"monitoring": {
		"drift_monitoring_enabled": True,
		"quality_metrics_required": True,
		"default_drift_threshold": 0.2,
		"unresolved_drift_blocks_deployment": True,
		"monitoring_event_stream": "bytewax",
	},
	"governance": {
		"require_tenant_context": True,
		"auth_required": True,
		"audit_model_changes": True,
		"model_card_required": True,
		"risk_review_required": True,
		"retirement_impact_review_required": True,
		"cross_tenant_access_allowed": False,
	},
	"observability": {
		"metrics_required": True,
		"trace_required": True,
		"audit_required": True,
		"lineage_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.MlcmService",
		"production_runtime": "service.MlcmService",
		"http_api": "api.py",
		"event_stream": "bytewax",
		"ai_core": "aicr",
		"auth_provider": "auth",
		"audit_sink": "audl",
		"metrics_sink": "moni",
		"artifact_store": "filestore",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_registry": True,
		"enable_versions": True,
		"enable_model_cards": True,
		"enable_evaluation_console": True,
		"enable_promotion_board": True,
		"enable_deployment_board": True,
		"enable_drift_monitor": True,
		"enable_rollback_console": True,
		"enable_governance": True,
		"enable_audit_timeline": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "mlcm_model_ops_console", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"registry",
		"versions",
		"evaluation",
		"promotion",
		"deployment",
		"monitoring",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"registry",
		"versions",
		"evaluation",
		"promotion",
		"deployment",
		"monitoring",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All model lifecycle operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "model_registration_requires_owner", "description": "Model registration requires an owner.", "condition": {"operation": "register_model", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "model_owner_required", "required_action": "assign_model_owner"}},
	{"name": "model_registration_requires_name", "description": "Model registration requires a model name.", "condition": {"operation": "register_model", "name_present": False}, "effect": {"decision": "deny", "reason": "model_name_required", "required_action": "name_model"}},
	{"name": "model_registration_requires_problem_type", "description": "Model registration requires problem-type metadata.", "condition": {"operation": "register_model", "problem_type_present": False}, "effect": {"decision": "deny", "reason": "problem_type_required", "required_action": "classify_model_problem_type"}},
	{"name": "model_registration_requires_risk_level", "description": "Model registration requires risk-level metadata.", "condition": {"operation": "register_model", "risk_level_present": False}, "effect": {"decision": "deny", "reason": "model_risk_level_required", "required_action": "assign_model_risk_level"}},
	{"name": "version_creation_requires_registered_model", "description": "Version creation requires a registered model.", "condition": {"operation": "create_version", "model_registered": False}, "effect": {"decision": "deny", "reason": "registered_model_required", "required_action": "register_model"}},
	{"name": "version_creation_requires_artifact_uri", "description": "Version creation requires an artifact URI.", "condition": {"operation": "create_version", "artifact_uri_present": False}, "effect": {"decision": "deny", "reason": "artifact_uri_required", "required_action": "attach_artifact_uri"}},
	{"name": "version_creation_requires_training_data", "description": "Version creation requires training data lineage.", "condition": {"operation": "create_version", "training_data_ref_present": False}, "effect": {"decision": "require_review", "reason": "training_data_lineage_required", "required_action": "attach_training_data_ref"}},
	{"name": "version_creation_requires_baseline", "description": "Version creation requires baseline lineage.", "condition": {"operation": "create_version", "baseline_ref_present": False}, "effect": {"decision": "require_review", "reason": "baseline_ref_required", "required_action": "attach_baseline_ref"}},
	{"name": "version_creation_requires_model_card_for_non_dev", "description": "Non-development versions require model-card evidence.", "condition": {"operation": "create_version", "non_dev_stage": True, "model_card_present": False}, "effect": {"decision": "deny", "reason": "model_card_required", "required_action": "attach_model_card"}},
	{"name": "evaluation_requires_baseline", "description": "Evaluation runs require a baseline reference.", "condition": {"operation": "record_evaluation", "baseline_ref_present": False}, "effect": {"decision": "deny", "reason": "evaluation_baseline_required", "required_action": "attach_evaluation_baseline"}},
	{"name": "evaluation_requires_evidence", "description": "Evaluation runs require evidence references.", "condition": {"operation": "record_evaluation", "evidence_refs_present": False}, "effect": {"decision": "require_review", "reason": "evaluation_evidence_required", "required_action": "attach_evaluation_evidence"}},
	{"name": "high_risk_evaluation_requires_fairness_review", "description": "High-risk model evaluation requires fairness review evidence.", "condition": {"operation": "record_evaluation", "risk_level": "high", "fairness_review_recorded": False}, "effect": {"decision": "require_review", "reason": "fairness_review_required", "required_action": "record_fairness_review"}},
	{"name": "high_risk_evaluation_requires_explainability", "description": "High-risk model evaluation requires explainability evidence.", "condition": {"operation": "record_evaluation", "risk_level": "high", "explainability_recorded": False}, "effect": {"decision": "require_review", "reason": "explainability_required", "required_action": "record_explainability_evidence"}},
	{"name": "promotion_requires_evaluation", "description": "Promotion requires evaluation evidence.", "condition": {"operation": "promote_model", "evaluation_recorded": False}, "effect": {"decision": "deny", "reason": "model_evaluation_required", "required_action": "record_model_evaluation"}},
	{"name": "production_promotion_requires_approval", "description": "Production model promotion requires approval.", "condition": {"target_stage": "production", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "promotion_approval_required", "required_action": "record_promotion_approval"}},
	{"name": "deployment_requires_model_card", "description": "Model deployments require model-card documentation.", "condition": {"operation": "deploy_model", "model_card_present": False}, "effect": {"decision": "deny", "reason": "model_card_required", "required_action": "attach_model_card"}},
	{"name": "low_eval_score_blocks_promotion", "description": "Low evaluation scores block promotion.", "condition": {"eval_score_lt": 0.8, "promotion_requested": True}, "effect": {"decision": "deny", "reason": "evaluation_score_too_low", "required_action": "improve_or_waive_evaluation"}},
	{"name": "deployment_requires_active_target", "description": "Deployments require active serving targets.", "condition": {"operation": "deploy_model", "target_active": False}, "effect": {"decision": "deny", "reason": "deployment_target_inactive", "required_action": "activate_or_change_target"}},
	{"name": "production_deployment_requires_approval", "description": "Production deployments require approval evidence.", "condition": {"operation": "deploy_model", "target_stage": "production", "deployment_approval_recorded": False}, "effect": {"decision": "deny", "reason": "deployment_approval_required", "required_action": "record_deployment_approval"}},
	{"name": "canary_limit_requires_review", "description": "Large canary percentages require rollout review.", "condition": {"operation": "deploy_model", "canary_percent_gt": 50, "rollout_review_recorded": False}, "effect": {"decision": "require_review", "reason": "rollout_review_required", "required_action": "record_rollout_review"}},
	{"name": "deployment_requires_health_check", "description": "Deployments require health-check evidence.", "condition": {"operation": "deploy_model", "health_check_recorded": False}, "effect": {"decision": "require_review", "reason": "deployment_health_check_required", "required_action": "record_health_check"}},
	{"name": "drifted_model_requires_review", "description": "Drifted models require review before continued serving.", "condition": {"drift_detected": True, "drift_review_recorded": False}, "effect": {"decision": "require_review", "reason": "drift_review_required", "required_action": "record_drift_review"}},
	{"name": "unresolved_drift_blocks_production_deployment", "description": "Production deployments are blocked when drift is unresolved.", "condition": {"operation": "deploy_model", "target_stage": "production", "unresolved_drift_present": True}, "effect": {"decision": "deny", "reason": "unresolved_drift_blocks_deployment", "required_action": "resolve_drift_review"}},
	{"name": "drift_signal_requires_threshold", "description": "Drift signals require a threshold.", "condition": {"operation": "record_drift", "threshold_present": False}, "effect": {"decision": "deny", "reason": "drift_threshold_required", "required_action": "attach_drift_threshold"}},
	{"name": "quality_metric_requires_owner", "description": "Quality metrics require an accountable owner.", "condition": {"operation": "record_metric", "owner_assigned": False}, "effect": {"decision": "require_review", "reason": "metric_owner_required", "required_action": "assign_metric_owner"}},
	{"name": "rollback_requires_reason", "description": "Rollbacks require a reason.", "condition": {"operation": "rollback_deployment", "reason_present": False}, "effect": {"decision": "deny", "reason": "rollback_reason_required", "required_action": "record_rollback_reason"}},
	{"name": "rollback_requires_same_model", "description": "Rollbacks must target a version of the same model.", "condition": {"operation": "rollback_deployment", "same_model": False}, "effect": {"decision": "deny", "reason": "rollback_version_model_mismatch", "required_action": "select_same_model_version"}},
	{"name": "retirement_requires_impact_review", "description": "Model retirement requires impact review.", "condition": {"operation": "retire_model", "impact_review_recorded": False}, "effect": {"decision": "deny", "reason": "model_retirement_review_required", "required_action": "record_retirement_impact"}},
	{"name": "retirement_requires_no_serving_deployments", "description": "Model retirement requires serving deployments to be drained.", "condition": {"operation": "retire_model", "serving_deployments_present": True}, "effect": {"decision": "deny", "reason": "serving_deployments_present", "required_action": "drain_serving_deployments"}},
	{"name": "cross_tenant_model_access_denied", "description": "Cross-tenant model access is denied by default.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_model"}},
	{"name": "audit_event_required_for_state_change", "description": "Lifecycle state changes require audit events.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "audit_event_required", "required_action": "record_audit_event"}},
	{"name": "bytewax_stream_required_for_monitoring", "description": "Monitoring event flows must use Bytewax.", "condition": {"operation": "configure_monitoring", "event_stream": "kafka"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "lineage_required_for_release", "description": "Release candidates require dataset, baseline, and artifact lineage.", "condition": {"operation": "release_candidate", "lineage_complete": False}, "effect": {"decision": "deny", "reason": "release_lineage_required", "required_action": "complete_release_lineage"}},
	{"name": "human_review_required_for_critical_risk", "description": "Critical-risk model operations require human review.", "condition": {"risk_level": "critical", "human_review_recorded": False}, "effect": {"decision": "require_review", "reason": "critical_risk_human_review_required", "required_action": "record_human_review"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/mlcm/dashboard", "component": "MLCMDashboard", "permission": "mlcm:view", "nav_group": "Overview"},
	{"name": "registry", "path": "/mlcm/models", "component": "ModelRegistry", "permission": "mlcm:view_models", "nav_group": "Registry"},
	{"name": "versions", "path": "/mlcm/versions", "component": "ModelVersionManager", "permission": "mlcm:manage_models", "nav_group": "Registry"},
	{"name": "model_cards", "path": "/mlcm/model-cards", "component": "ModelCardLibrary", "permission": "mlcm:view_models", "nav_group": "Registry"},
	{"name": "evaluation", "path": "/mlcm/evaluation", "component": "EvaluationConsole", "permission": "mlcm:evaluate", "nav_group": "Quality"},
	{"name": "baselines", "path": "/mlcm/baselines", "component": "BaselineEvidence", "permission": "mlcm:evaluate", "nav_group": "Quality"},
	{"name": "promotion", "path": "/mlcm/promotion", "component": "PromotionBoard", "permission": "mlcm:promote", "nav_group": "Release"},
	{"name": "deployments", "path": "/mlcm/deployments", "component": "DeploymentBoard", "permission": "mlcm:deploy", "nav_group": "Operations"},
	{"name": "drift", "path": "/mlcm/drift", "component": "DriftMonitor", "permission": "mlcm:view_drift", "nav_group": "Operations"},
	{"name": "rollback", "path": "/mlcm/rollback", "component": "RollbackConsole", "permission": "mlcm:deploy", "nav_group": "Operations"},
	{"name": "governance", "path": "/mlcm/governance", "component": "ModelGovernance", "permission": "mlcm:govern", "nav_group": "Governance"},
	{"name": "audit", "path": "/mlcm/audit", "component": "MLCMAuditTimeline", "permission": "mlcm:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/mlcm/settings", "component": "MLCMSettings", "permission": "mlcm:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "mlcm_model_ops_console",
	"tokens": {
		"color.primary": "#244B5A",
		"color.accent": "#D97706",
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
		"model_version_row": {"icon": "layers", "status_indicator": "stage-pill", "risk_style": "eval-band"},
		"model_card_library": {"icon": "file-check", "status_indicator": "completeness-pill", "risk_style": "lineage-band"},
		"promotion_gate_panel": {"visual": "gate-stack", "highlight": "approval-chip"},
		"baseline_evidence_panel": {"visual": "dataset-lineage", "highlight": "baseline-chip"},
		"deployment_rollout_panel": {"visual": "rollout-meter", "highlight": "canary-chip"},
		"drift_monitor_chart": {"visual": "time-series-grid", "threshold_style": "drift-lines"},
		"rollback_console": {"visual": "version-timeline", "status_style": "rollback-chip"},
		"model_card_panel": {"visual": "evidence-list", "status_style": "completeness-pill"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "evidence-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable MLCM capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "mlcm",
		"display_name": "AI Model Lifecycle Management",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "__init__.py",
			"api_prefix": "/mlcm/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
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
