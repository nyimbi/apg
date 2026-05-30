"""Executable capability contract for APG Recommender Systems."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ranking_designer", "feature_engineer", "experiment_designer", "drift_observer", "policy_reviewer"]
SUPPORTED_DEPLOYMENT_TARGETS = ["python", "apg_runtime", "batch_ranker", "edge_ranker"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"datasets": {
		"dataset_owner_required": True,
		"source_policy_required": True,
		"schema_required": True,
		"event_timestamp_required": True,
		"minimum_training_events": 1000,
	},
	"models": {
		"enabled_algorithms": ["collaborative_filtering", "content_based", "hybrid", "contextual_bandit"],
		"model_owner_required": True,
		"minimum_training_events": 1000,
		"drift_monitoring_required": True,
		"model_approval_required": True,
		"deployment_approval_required": True,
	},
	"ranking": {
		"ranking_policy_required": True,
		"ranking_policy_owner_required": True,
		"diversity_constraints_enabled": True,
		"sensitive_attribute_filtering": True,
		"minimum_recommendation_confidence": 0.65,
		"empty_result_review_required": True,
	},
	"experiments": {
		"experiment_approval_required": True,
		"holdout_required": True,
		"business_metric_required": True,
		"max_experiment_percent": 25,
	},
	"feedback": {
		"feedback_capture_required": True,
		"feedback_actor_required": True,
		"feedback_event_required": True,
		"feedback_audit_required": True,
	},
	"recommender_agents": {
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
		"audit_recommendations": True,
		"profile_consent_required": True,
		"explainability_required_for_high_impact": True,
		"tenant_isolation_required": True,
		"state_change_reason_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"ranking_metrics_required": True,
		"drift_metrics_required": True,
		"feedback_metrics_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.RecsService",
		"runtime_helpers": "recommendation_runtime.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"prediction": "pred",
		"ai_core": "aicr",
		"nlp": "nlpc",
		"master_data": "mdm",
		"etl": "etlp",
		"audit_sink": "audl",
		"monitoring": "moni",
	},
	"deployments": {
		"deployment_target_required": True,
		"deployment_approval_required": True,
		"rollback_plan_required": True,
		"supported_targets": SUPPORTED_DEPLOYMENT_TARGETS,
	},
	"ui": {
		"enable_recommendation_console": True,
		"enable_dataset_manager": True,
		"enable_model_registry": True,
		"enable_experiment_studio": True,
		"enable_ranking_policy": True,
		"enable_feedback_console": True,
		"enable_agent_panel": True,
		"enable_deployment_center": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {
		"default_theme": "recs_recommendation_console",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"datasets",
		"models",
		"ranking",
		"experiments",
		"feedback",
		"recommender_agents",
		"governance",
		"observability",
		"adapters",
		"deployments",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"datasets",
		"models",
		"ranking",
		"experiments",
		"feedback",
		"recommender_agents",
		"governance",
		"observability",
		"adapters",
		"deployments",
		"ui",
		"theme",
	]} | {
		"tenant_id": {"type": "string", "minLength": 1},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All recommendation operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "dataset_requires_owner", "description": "Recommendation datasets require an accountable owner.", "condition": {"operation": "register_dataset", "dataset_owner_present": False}, "effect": {"decision": "deny", "reason": "dataset_owner_required", "required_action": "assign_dataset_owner"}},
	{"name": "dataset_requires_source", "description": "Recommendation datasets require a source reference.", "condition": {"operation": "register_dataset", "dataset_source_present": False}, "effect": {"decision": "deny", "reason": "dataset_source_required", "required_action": "attach_dataset_source"}},
	{"name": "dataset_requires_schema", "description": "Recommendation datasets require schema fields.", "condition": {"operation": "register_dataset", "dataset_schema_present": False}, "effect": {"decision": "deny", "reason": "dataset_schema_required", "required_action": "define_dataset_schema"}},
	{"name": "dataset_requires_policy", "description": "Recommendation datasets require a source governance policy.", "condition": {"operation": "register_dataset", "dataset_policy_present": False}, "effect": {"decision": "deny", "reason": "dataset_policy_required", "required_action": "attach_dataset_policy"}},
	{"name": "interaction_event_requires_actor", "description": "Interaction events require a user or profile id.", "condition": {"operation": "record_interaction", "interaction_actor_present": False}, "effect": {"decision": "deny", "reason": "interaction_actor_required", "required_action": "set_interaction_actor"}},
	{"name": "interaction_event_requires_item", "description": "Interaction events require an item id.", "condition": {"operation": "record_interaction", "interaction_item_present": False}, "effect": {"decision": "deny", "reason": "interaction_item_required", "required_action": "set_interaction_item"}},
	{"name": "interaction_event_requires_timestamp", "description": "Interaction events require timestamp evidence.", "condition": {"operation": "record_interaction", "interaction_timestamp_present": False}, "effect": {"decision": "deny", "reason": "interaction_timestamp_required", "required_action": "set_interaction_timestamp"}},
	{"name": "profile_consent_required", "description": "Personalized recommendations require profile consent.", "condition": {"operation": "recommend", "profile_consent_recorded": False}, "effect": {"decision": "deny", "reason": "profile_consent_required", "required_action": "record_profile_consent"}},
	{"name": "ranking_policy_required", "description": "Recommendations require an attached ranking policy.", "condition": {"operation": "recommend", "ranking_policy_attached": False}, "effect": {"decision": "deny", "reason": "ranking_policy_required", "required_action": "attach_ranking_policy"}},
	{"name": "recommendation_candidates_required", "description": "Recommendation requests require candidate items.", "condition": {"operation": "recommend", "candidate_count_lte": 0}, "effect": {"decision": "deny", "reason": "recommendation_candidates_required", "required_action": "provide_candidate_items"}},
	{"name": "recommendation_output_requires_results", "description": "Recommendation output should contain ranked results.", "condition": {"operation": "recommend", "recommendation_count_lte": 0}, "effect": {"decision": "require_review", "reason": "empty_recommendation_review_required", "required_action": "review_empty_recommendations"}},
	{"name": "ranking_policy_requires_owner", "description": "Ranking policies require an owner.", "condition": {"operation": "attach_ranking_policy", "ranking_policy_owner_present": False}, "effect": {"decision": "deny", "reason": "ranking_policy_owner_required", "required_action": "assign_ranking_policy_owner"}},
	{"name": "model_training_requires_events", "description": "Training requires sufficient events.", "condition": {"operation": "train_model", "training_event_count_lt": 1000}, "effect": {"decision": "deny", "reason": "insufficient_training_events", "required_action": "collect_training_events"}},
	{"name": "model_requires_owner", "description": "Models require an accountable owner.", "condition": {"operation": "train_model", "model_owner_present": False}, "effect": {"decision": "deny", "reason": "model_owner_required", "required_action": "assign_model_owner"}},
	{"name": "model_requires_drift_monitoring", "description": "Models require drift monitoring before activation.", "condition": {"operation": "train_model", "drift_monitoring_enabled": False}, "effect": {"decision": "deny", "reason": "drift_monitoring_required", "required_action": "enable_drift_monitoring"}},
	{"name": "model_approval_required", "description": "Models require approval before deployment.", "condition": {"operation": "deploy_model", "model_approved": False}, "effect": {"decision": "deny", "reason": "model_approval_required", "required_action": "approve_model"}},
	{"name": "model_deployment_requires_target", "description": "Model deployments require a supported target.", "condition": {"operation": "deploy_model", "deployment_target_supported": False}, "effect": {"decision": "deny", "reason": "deployment_target_required", "required_action": "choose_supported_deployment_target"}},
	{"name": "model_deployment_requires_approval", "description": "Model deployments require approval evidence.", "condition": {"operation": "deploy_model", "deployment_approval_recorded": False}, "effect": {"decision": "deny", "reason": "deployment_approval_required", "required_action": "record_deployment_approval"}},
	{"name": "model_deployment_requires_rollback", "description": "Model deployments require rollback evidence.", "condition": {"operation": "deploy_model", "rollback_plan_present": False}, "effect": {"decision": "deny", "reason": "rollback_plan_required", "required_action": "attach_rollback_plan"}},
	{"name": "high_impact_requires_explainability", "description": "High-impact recommendations require explanations.", "condition": {"impact_level": "high", "explanation_attached": False}, "effect": {"decision": "deny", "reason": "explainability_required", "required_action": "attach_explanation"}},
	{"name": "large_experiment_requires_review", "description": "Large recommendation experiments require review.", "condition": {"experiment_percent_gt": 25, "experiment_review_recorded": False}, "effect": {"decision": "require_review", "reason": "experiment_review_required", "required_action": "review_experiment"}},
	{"name": "feedback_requires_actor", "description": "Feedback requires the acting user or profile.", "condition": {"operation": "record_feedback", "feedback_actor_present": False}, "effect": {"decision": "deny", "reason": "feedback_actor_required", "required_action": "record_feedback_actor"}},
	{"name": "feedback_requires_event", "description": "Feedback requires an event type.", "condition": {"operation": "record_feedback", "feedback_event_present": False}, "effect": {"decision": "deny", "reason": "feedback_event_required", "required_action": "record_feedback_event"}},
	{"name": "recommender_agent_requires_registration", "description": "AI recommender agents must be registered.", "condition": {"recommender_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "recommender_agent_registration_required", "required_action": "register_recommender_agent"}},
	{"name": "recommender_agent_runtime_supported", "description": "AI recommender agents must use a supported runtime.", "condition": {"recommender_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "recommender_agent_runtime_not_supported", "required_action": "choose_supported_recommender_agent_runtime"}},
	{"name": "recommender_agent_requires_scope", "description": "AI recommender agents require explicit scope.", "condition": {"recommender_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "recommender_agent_scope_required", "required_action": "set_recommender_agent_scope"}},
	{"name": "recommender_agent_requires_disclosure", "description": "AI recommender-agent contributions require disclosure.", "condition": {"recommender_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "recommender_agent_disclosure_required", "required_action": "disclose_recommender_agent"}},
	{"name": "state_change_requires_reason", "description": "Model and policy state changes require a reason.", "condition": {"state_change_requested": True, "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "state_change_reason_required", "required_action": "record_state_change_reason"}},
	{"name": "state_change_requires_audit", "description": "Model and policy state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "recommendation_audit_event_required", "required_action": "record_recommendation_audit"}},
	{"name": "cross_tenant_recommendation_access_denied", "description": "Recommendation records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_recommendation_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_recommendation_mutation_requires_bytewax", "description": "Batch recommendation mutations must use Bytewax event streams.", "condition": {"operation": "batch_recommendation_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/recs/dashboard", "component": "RECSDashboard", "permission": "recs:view", "nav_group": "Overview"},
	{"name": "recommendations", "path": "/recs/recommendations", "component": "RecommendationConsole", "permission": "recs:recommend", "nav_group": "Runtime"},
	{"name": "datasets", "path": "/recs/datasets", "component": "RecommendationDatasets", "permission": "recs:manage_data", "nav_group": "Data"},
	{"name": "models", "path": "/recs/models", "component": "RecommendationModels", "permission": "recs:manage_models", "nav_group": "Models"},
	{"name": "deployments", "path": "/recs/deployments", "component": "ModelDeployments", "permission": "recs:deploy", "nav_group": "Models"},
	{"name": "catalogs", "path": "/recs/catalogs", "component": "CatalogManager", "permission": "recs:view", "nav_group": "Data"},
	{"name": "profiles", "path": "/recs/profiles", "component": "ProfileFeatures", "permission": "recs:view", "nav_group": "Data"},
	{"name": "feedback", "path": "/recs/feedback", "component": "FeedbackConsole", "permission": "recs:recommend", "nav_group": "Runtime"},
	{"name": "experiments", "path": "/recs/experiments", "component": "ExperimentStudio", "permission": "recs:run_experiments", "nav_group": "Optimization"},
	{"name": "policies", "path": "/recs/policies", "component": "RankingPolicies", "permission": "recs:admin", "nav_group": "Governance"},
	{"name": "agents", "path": "/recs/agents", "component": "RecommenderAgentPanel", "permission": "recs:manage_models", "nav_group": "Agents"},
	{"name": "audit", "path": "/recs/audit", "component": "RECSAuditTrail", "permission": "recs:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/recs/analytics", "component": "RECSAnalytics", "permission": "recs:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/recs/settings", "component": "RECSSettings", "permission": "recs:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "recs_recommendation_console",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#DD6B20",
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
		"recommendation_list": {"icon": "sparkles", "status_indicator": "rank-pill", "risk_style": "policy-band"},
		"dataset_card": {"visual": "event-stream-card", "status_style": "source-chip"},
		"model_card": {"visual": "model-score-card", "highlight": "drift-chip"},
		"deployment_center": {"visual": "deployment-lane", "status_style": "target-chip"},
		"feedback_console": {"visual": "feedback-timeline", "status_style": "signal-chip"},
		"experiment_board": {"visual": "variant-lanes", "status_style": "metric-chip"},
		"ranking_policy": {"visual": "constraint-stack", "status_style": "guardrail-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
	},
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.recs.lifecycle",
	"state": ["datasets", "interaction_events", "catalog_items", "profiles", "policies", "models", "deployments", "recommendations", "experiments", "feedback", "recommender_agents"],
	"events": [
		"dataset_registered",
		"interaction_recorded",
		"catalog_item_registered",
		"profile_recorded",
		"ranking_policy_attached",
		"model_trained",
		"model_approved",
		"model_deployed",
		"recommendations_generated",
		"feedback_recorded",
		"experiment_created",
		"model_drift_recorded",
		"recommender_agent_registered",
		"model_state_changed",
	],
	"batch_mutation_guardrail": "batch_recommendation_mutation_requires_bytewax",
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable RECS capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "recs",
		"display_name": "Recommender Systems",
		"provides": ["personalized_recommendations", "ranking_policies", "catalog_matching", "interaction_datasets", "model_training", "model_deployments", "feedback_loops", "experiment_optimization", "recommender_agents"],
		"requires": ["pred", "aicr", "nlpc"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/recs/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default RECS governance rules."""
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
