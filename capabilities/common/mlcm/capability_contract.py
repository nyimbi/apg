"""Executable capability contract for APG AI Model Lifecycle Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_MLCM_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_MLCM_AGENT_ROLES = [
	"model_card_reviewer",
	"evaluation_reviewer",
	"fairness_reviewer",
	"explainability_reviewer",
	"promotion_reviewer",
	"deployment_reviewer",
	"drift_reviewer",
	"rollback_reviewer",
	"retirement_reviewer",
	"model_steward",
]
PRIVILEGED_MLCM_AGENT_ROLES = [
	"evaluation_reviewer",
	"fairness_reviewer",
	"explainability_reviewer",
	"promotion_reviewer",
	"deployment_reviewer",
	"drift_reviewer",
	"rollback_reviewer",
	"retirement_reviewer",
]


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
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_MLCM_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_MLCM_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_MLCM_AGENT_ROLES,
		"require_owner": True,
		"require_purpose": True,
		"require_scope": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_agent_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "mlcm.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"model_batch",
			"version_batch",
			"evaluation_batch",
			"promotion_batch",
			"deployment_batch",
			"drift_batch",
			"rollback_batch",
			"retirement_batch",
			"model_lifecycle_agent_batch",
		],
		"topics": [
			"mlcm.models",
			"mlcm.versions",
			"mlcm.evaluations",
			"mlcm.promotions",
			"mlcm.deployments",
			"mlcm.drift",
			"mlcm.rollbacks",
			"mlcm.retirements",
			"mlcm.agents",
		],
		"broker_core_dependency_allowed": False,
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
		"enable_model_lifecycle_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
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
		"agents",
		"streaming",
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
		"agents",
		"streaming",
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
	{"name": "bytewax_stream_required_for_monitoring", "description": "Monitoring event flows must use Bytewax.", "condition": {"operation": "configure_monitoring", "event_stream": "legacy_queue"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "model_lifecycle_agent_runtime_supported", "description": "Model lifecycle agents must use supported runtimes.", "condition": {"operation": "register_model_lifecycle_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_model_lifecycle_agent_runtime", "required_action": "choose_supported_model_lifecycle_agent_runtime"}},
	{"name": "model_lifecycle_agent_role_supported", "description": "Model lifecycle agents must use supported lifecycle roles.", "condition": {"operation": "register_model_lifecycle_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_model_lifecycle_agent_role", "required_action": "choose_supported_model_lifecycle_agent_role"}},
	{"name": "model_lifecycle_agent_requires_scope", "description": "Model lifecycle agents require an explicit bounded scope.", "condition": {"operation": "register_model_lifecycle_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "model_lifecycle_agent_scope_required", "required_action": "declare_model_lifecycle_agent_scope"}},
	{"name": "model_lifecycle_agent_requires_owner", "description": "Model lifecycle agents require an accountable owner.", "condition": {"operation": "register_model_lifecycle_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "model_lifecycle_agent_owner_required", "required_action": "assign_model_lifecycle_agent_owner"}},
	{"name": "model_lifecycle_agent_requires_purpose", "description": "Model lifecycle agents require a documented purpose.", "condition": {"operation": "register_model_lifecycle_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "model_lifecycle_agent_purpose_required", "required_action": "document_model_lifecycle_agent_purpose"}},
	{"name": "model_lifecycle_agent_requires_contribution_disclosure", "description": "Model lifecycle agents must disclose machine-authored lifecycle contributions.", "condition": {"operation": "register_model_lifecycle_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "model_lifecycle_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "model_lifecycle_agent_privileged_role_requires_human_approval", "description": "Privileged model lifecycle-agent roles require human approval evidence.", "condition": {"operation": "register_model_lifecycle_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "model_lifecycle_agent_human_approval_required", "required_action": "record_human_model_lifecycle_agent_approval"}},
	{"name": "bytewax_mlcm_stream_required", "description": "MLCM lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_mlcm_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_mlcm_lifecycle_batch_to_bytewax"}},
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
	{"name": "agents", "path": "/mlcm/agents", "component": "ModelLifecycleAgentRoster", "permission": "mlcm:govern", "nav_group": "Governance"},
	{"name": "lifecycle", "path": "/mlcm/lifecycle", "component": "MLCMLifecycleBatchMonitor", "permission": "mlcm:govern", "nav_group": "Operations"},
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
		"model_lifecycle_agent_roster": {"icon": "bot-message-square", "status_indicator": "agent-approval-chip", "risk_style": "scope-band"},
		"bytewax_lifecycle_panel": {"icon": "git-branch", "status_indicator": "stream-chip", "risk_style": "processor-band"},
		"model_card_panel": {"visual": "evidence-list", "status_style": "completeness-pill"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "evidence-chip"},
	},
}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class MLCM agent composition manifest."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_MLCM_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_MLCM_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_MLCM_AGENT_ROLES),
		"required_fields": ["tenant_id", "agent_id", "name", "runtime", "role", "scope", "owner", "purpose"],
		"guardrails": [
			"supported_runtime",
			"supported_role",
			"explicit_scope",
			"accountable_owner",
			"declared_purpose",
			"machine_contribution_disclosure",
			"human_approval_for_privileged_roles",
		],
		"adapter_contract": "aicr_provider_neutral_agent_adapter",
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return the MLCM Bytewax lifecycle stream contract."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "mlcm.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"model_batch",
			"version_batch",
			"evaluation_batch",
			"promotion_batch",
			"deployment_batch",
			"drift_batch",
			"rollback_batch",
			"retirement_batch",
			"model_lifecycle_agent_batch",
		],
		"topics": [
			"mlcm.models",
			"mlcm.versions",
			"mlcm.evaluations",
			"mlcm.promotions",
			"mlcm.deployments",
			"mlcm.drift",
			"mlcm.rollbacks",
			"mlcm.retirements",
			"mlcm.agents",
		],
		"broker_core_dependency_allowed": False,
	}


STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"stream": "apg.mlcm.lifecycle",
	"key": "tenant_id",
	"events": [
		"model_registered",
		"model_versioned",
		"model_promoted",
		"model_deprecated",
		"model_retired",
		"experiment_started",
		"experiment_completed",
		"evaluation_recorded",
		"drift_detected",
		"retraining_triggered",
		"deployment_approved",
		"agent_registered",
	],
	"guardrails": [
		"mlcm_batch_requires_bytewax",
		"mlcm_privileged_action_requires_human_approval",
	],
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
		"provides": ["model_lifecycle", "model_governance", "model_lifecycle_agent_composition"],
		"requires": ["aicr", "moni", "audl"],
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
		"agents": agent_manifest(),
		"streaming": deepcopy(STREAMING),
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
