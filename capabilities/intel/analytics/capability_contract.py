"""Executable capability contract for APG Intelligence Analytics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_analytics"
CAPABILITY_NAME = "Intelligence Analytics"
CAPABILITY_VERSION = "1.1.0"
ANALYTICS_EVENT_STREAM = "apg.intel.analytics.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "legal_mandate", "partner_authority", "consent", "incident_response_authority", "public_interest_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_WORKSPACE_TYPES = ["threat_analytics", "fraud_analytics", "public_safety_analytics", "operational_analytics", "strategic_analytics", "incident_analytics", "risk_analytics"]
SUPPORTED_DATASET_TYPES = ["fusion_extract", "event_stream", "entity_table", "geospatial_layer", "graph_projection", "document_corpus", "transaction_set", "partner_dataset", "metric_series"]
SUPPORTED_RETENTION_CLASSES = ["short", "standard", "extended", "legal_hold"]
SUPPORTED_FEATURE_TYPES = ["indicator_features", "entity_features", "temporal_features", "geospatial_features", "network_features", "text_features", "behavioral_features"]
SUPPORTED_MODEL_TYPES = ["ruleset", "statistical", "machine_learning", "graph_analytics", "nlp_analytics", "geospatial_analytics", "simulation"]
SUPPORTED_RUN_TYPES = ["batch", "streaming", "backtest", "scenario", "what_if", "validation"]
SUPPORTED_INSIGHT_TYPES = ["trend", "anomaly", "risk_signal", "forecast", "cluster", "relationship", "explanation", "quality_issue"]
SUPPORTED_NARRATIVE_TYPES = ["briefing", "analytic_note", "situation_report", "executive_summary", "watchlist_update"]
SUPPORTED_RECOMMENDATION_TYPES = ["monitor", "investigate", "escalate", "mitigate", "review_policy", "request_collection", "close"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["data_steward", "feature_engineer", "model_reviewer", "insight_analyst", "dashboard_curator", "recommendation_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"workspaces": {"supported_workspace_types": SUPPORTED_WORKSPACE_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "authority_required": True, "evidence_required": True},
	"datasets": {"supported_dataset_types": SUPPORTED_DATASET_TYPES, "supported_retention_classes": SUPPORTED_RETENTION_CLASSES, "owner_required": True, "lineage_required": True, "evidence_required": True},
	"feature_sets": {"supported_feature_types": SUPPORTED_FEATURE_TYPES, "dataset_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"models": {"supported_model_types": SUPPORTED_MODEL_TYPES, "supported_risk_levels": SUPPORTED_RISK_LEVELS, "feature_set_required": True, "validation_required": True, "evidence_required": True},
	"runs": {"supported_run_types": SUPPORTED_RUN_TYPES, "model_required": True, "result_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"insights": {"supported_insight_types": SUPPORTED_INSIGHT_TYPES, "run_required": True, "claim_reference_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"dashboards": {"insight_required": True, "audience_required": True, "release_marking_required": True, "approval_required": True, "evidence_required": True},
	"narratives": {"supported_narrative_types": SUPPORTED_NARRATIVE_TYPES, "insight_required": True, "summary_required": True, "approval_required": True, "evidence_required": True},
	"recommendations": {"supported_recommendation_types": SUPPORTED_RECOMMENDATION_TYPES, "insight_required": True, "action_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True, "fabrication_denied": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "cross_tenant_analytics_denied": True, "hallucinated_insights_denied": True, "training_data_leakage_denied": True, "privacy_bypass_denied": True, "unsupported_automated_decision_denied": True, "unapproved_model_deployment_denied": True, "autonomous_dissemination_denied": True},
	"observability": {"event_stream": ANALYTICS_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "geospatial": "geos", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_workspaces": True, "enable_datasets": True, "enable_feature_sets": True, "enable_models": True, "enable_runs": True, "enable_insights": True, "enable_analytic_dashboards": True, "enable_narratives": True, "enable_recommendations": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_analytics_control", "allow_tenant_overrides": True},
}

PROVIDES = ["analytics_authority_workflow", "analytics_workspace_workflow", "analytics_dataset_workflow", "analytics_feature_workflow", "analytics_model_workflow", "analytics_run_workflow", "analytics_insight_workflow", "analytics_dashboard_workflow", "analytics_narrative_workflow", "analytics_recommendation_workflow", "analytics_review_workflow", "analytics_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-analytics/dashboard", "component": "AnalyticsDashboard", "permission": "intel_analytics:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-analytics/authorities", "component": "AnalyticsAuthorityConsole", "permission": "intel_analytics:authorities", "nav_group": "Governance"},
	{"name": "workspaces", "path": "/intel-analytics/workspaces", "component": "AnalyticsWorkspaceConsole", "permission": "intel_analytics:workspaces", "nav_group": "Planning"},
	{"name": "datasets", "path": "/intel-analytics/datasets", "component": "AnalyticsDatasetRegistry", "permission": "intel_analytics:datasets", "nav_group": "Data"},
	{"name": "features", "path": "/intel-analytics/features", "component": "AnalyticsFeatureWorkbench", "permission": "intel_analytics:features", "nav_group": "Data"},
	{"name": "models", "path": "/intel-analytics/models", "component": "AnalyticsModelWorkbench", "permission": "intel_analytics:models", "nav_group": "Analysis"},
	{"name": "runs", "path": "/intel-analytics/runs", "component": "AnalyticsRunConsole", "permission": "intel_analytics:runs", "nav_group": "Analysis"},
	{"name": "insights", "path": "/intel-analytics/insights", "component": "AnalyticsInsightWorkbench", "permission": "intel_analytics:insights", "nav_group": "Analysis"},
	{"name": "analytic_dashboards", "path": "/intel-analytics/analytic-dashboards", "component": "AnalyticsDashboardConsole", "permission": "intel_analytics:dashboards", "nav_group": "Publication"},
	{"name": "narratives", "path": "/intel-analytics/narratives", "component": "AnalyticsNarrativeConsole", "permission": "intel_analytics:narratives", "nav_group": "Publication"},
	{"name": "recommendations", "path": "/intel-analytics/recommendations", "component": "AnalyticsRecommendationConsole", "permission": "intel_analytics:recommendations", "nav_group": "Action"},
	{"name": "reviews", "path": "/intel-analytics/reviews", "component": "AnalyticsReviewConsole", "permission": "intel_analytics:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-analytics/agents", "component": "AnalyticsAgentWorkbench", "permission": "intel_analytics:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-analytics/settings", "component": "AnalyticsSettings", "permission": "intel_analytics:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_analytics_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#7C3AED", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "workspaces": {"icon": "layout-dashboard", "status_indicator": "workspace-chip"}, "datasets": {"icon": "database", "status_indicator": "dataset-chip"}, "features": {"icon": "sliders-horizontal", "status_indicator": "feature-chip"}, "models": {"icon": "brain-circuit", "status_indicator": "model-chip"}, "runs": {"icon": "activity", "status_indicator": "run-chip"}, "insights": {"icon": "sparkles", "status_indicator": "confidence-chip"}, "dashboards": {"icon": "bar-chart-3", "status_indicator": "release-chip"}, "narratives": {"icon": "file-text", "status_indicator": "narrative-chip"}, "recommendations": {"icon": "list-checks", "status_indicator": "action-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": ANALYTICS_EVENT_STREAM, "key": "tenant_id", "events": ["analytics_authority_recorded", "analytics_workspace_recorded", "analytics_dataset_registered", "analytics_feature_set_recorded", "analytics_model_recorded", "analytics_run_recorded", "analytics_insight_recorded", "analytics_dashboard_recorded", "analytics_narrative_recorded", "analytics_recommendation_recorded", "analytics_review_recorded", "analytics_agent_registered"], "guardrails": ["analytics_batch_requires_bytewax", "privileged_analytics_agent_action_requires_human_approval", "hallucinated_insight_action_denied", "training_data_leakage_action_denied", "privacy_bypass_action_denied", "unsupported_automated_decision_action_denied", "unapproved_model_deployment_action_denied", "autonomous_dissemination_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "analytics_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "analytics_policy_required", "required_action": "attach_analytics_policy"}},
	{"name": "authority_type_supported", "condition": {"operation": "record_authority", "authority_type_supported": False}, "effect": {"decision": "deny", "reason": "authority_type_not_supported", "required_action": "select_supported_authority_type"}},
	{"name": "authority_scope_required", "condition": {"operation": "record_authority", "scope_present": False}, "effect": {"decision": "deny", "reason": "authority_scope_required", "required_action": "attach_scope_reference"}},
	{"name": "authority_classification_supported", "condition": {"operation": "record_authority", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "authority_approver_required", "condition": {"operation": "record_authority", "approver_present": False}, "effect": {"decision": "deny", "reason": "authority_approver_required", "required_action": "record_approver"}},
	{"name": "authority_expiry_required", "condition": {"operation": "record_authority", "expiry_present": False}, "effect": {"decision": "deny", "reason": "authority_expiry_required", "required_action": "set_expiry"}},
	{"name": "authority_evidence_required", "condition": {"operation": "record_authority", "evidence_present": False}, "effect": {"decision": "deny", "reason": "authority_evidence_required", "required_action": "attach_authority_evidence"}},
	{"name": "workspace_type_supported", "condition": {"operation": "record_workspace", "workspace_type_supported": False}, "effect": {"decision": "deny", "reason": "workspace_type_not_supported", "required_action": "select_supported_workspace_type"}},
	{"name": "workspace_name_required", "condition": {"operation": "record_workspace", "workspace_name_present": False}, "effect": {"decision": "deny", "reason": "workspace_name_required", "required_action": "name_workspace"}},
	{"name": "workspace_classification_supported", "condition": {"operation": "record_workspace", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "workspace_authority_required", "condition": {"operation": "record_workspace", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "workspace_evidence_required", "condition": {"operation": "record_workspace", "evidence_present": False}, "effect": {"decision": "deny", "reason": "workspace_evidence_required", "required_action": "attach_workspace_evidence"}},
	{"name": "dataset_workspace_required", "condition": {"operation": "register_dataset", "workspace_present": False}, "effect": {"decision": "deny", "reason": "workspace_required", "required_action": "select_workspace"}},
	{"name": "dataset_type_supported", "condition": {"operation": "register_dataset", "dataset_type_supported": False}, "effect": {"decision": "deny", "reason": "dataset_type_not_supported", "required_action": "select_supported_dataset_type"}},
	{"name": "dataset_source_required", "condition": {"operation": "register_dataset", "source_reference_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "dataset_owner_required", "condition": {"operation": "register_dataset", "owner_present": False}, "effect": {"decision": "deny", "reason": "dataset_owner_required", "required_action": "assign_dataset_owner"}},
	{"name": "dataset_lineage_required", "condition": {"operation": "register_dataset", "lineage_present": False}, "effect": {"decision": "deny", "reason": "dataset_lineage_required", "required_action": "record_lineage"}},
	{"name": "dataset_retention_supported", "condition": {"operation": "register_dataset", "retention_supported": False}, "effect": {"decision": "deny", "reason": "retention_class_not_supported", "required_action": "select_supported_retention"}},
	{"name": "dataset_evidence_required", "condition": {"operation": "register_dataset", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dataset_evidence_required", "required_action": "attach_dataset_evidence"}},
	{"name": "feature_dataset_required", "condition": {"operation": "record_feature_set", "dataset_present": False}, "effect": {"decision": "deny", "reason": "dataset_required", "required_action": "select_dataset"}},
	{"name": "feature_type_supported", "condition": {"operation": "record_feature_set", "feature_type_supported": False}, "effect": {"decision": "deny", "reason": "feature_type_not_supported", "required_action": "select_supported_feature_type"}},
	{"name": "feature_reference_required", "condition": {"operation": "record_feature_set", "feature_reference_present": False}, "effect": {"decision": "deny", "reason": "feature_reference_required", "required_action": "attach_feature_reference"}},
	{"name": "feature_confidence_valid", "condition": {"operation": "record_feature_set", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "feature_analyst_required", "condition": {"operation": "record_feature_set", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "feature_evidence_required", "condition": {"operation": "record_feature_set", "evidence_present": False}, "effect": {"decision": "deny", "reason": "feature_evidence_required", "required_action": "attach_feature_evidence"}},
	{"name": "model_feature_required", "condition": {"operation": "record_model", "feature_set_present": False}, "effect": {"decision": "deny", "reason": "feature_set_required", "required_action": "select_feature_set"}},
	{"name": "model_type_supported", "condition": {"operation": "record_model", "model_type_supported": False}, "effect": {"decision": "deny", "reason": "model_type_not_supported", "required_action": "select_supported_model_type"}},
	{"name": "model_objective_required", "condition": {"operation": "record_model", "objective_present": False}, "effect": {"decision": "deny", "reason": "model_objective_required", "required_action": "record_model_objective"}},
	{"name": "model_validation_required", "condition": {"operation": "record_model", "validation_present": False}, "effect": {"decision": "deny", "reason": "model_validation_required", "required_action": "attach_validation_reference"}},
	{"name": "model_risk_supported", "condition": {"operation": "record_model", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "model_evidence_required", "condition": {"operation": "record_model", "evidence_present": False}, "effect": {"decision": "deny", "reason": "model_evidence_required", "required_action": "attach_model_evidence"}},
	{"name": "run_model_required", "condition": {"operation": "record_run", "model_present": False}, "effect": {"decision": "deny", "reason": "model_required", "required_action": "select_model"}},
	{"name": "run_type_supported", "condition": {"operation": "record_run", "run_type_supported": False}, "effect": {"decision": "deny", "reason": "run_type_not_supported", "required_action": "select_supported_run_type"}},
	{"name": "run_result_required", "condition": {"operation": "record_run", "result_reference_present": False}, "effect": {"decision": "deny", "reason": "run_result_required", "required_action": "attach_result_reference"}},
	{"name": "run_confidence_valid", "condition": {"operation": "record_run", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "run_analyst_required", "condition": {"operation": "record_run", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "run_evidence_required", "condition": {"operation": "record_run", "evidence_present": False}, "effect": {"decision": "deny", "reason": "run_evidence_required", "required_action": "attach_run_evidence"}},
	{"name": "insight_run_required", "condition": {"operation": "record_insight", "run_present": False}, "effect": {"decision": "deny", "reason": "run_required", "required_action": "select_run"}},
	{"name": "insight_type_supported", "condition": {"operation": "record_insight", "insight_type_supported": False}, "effect": {"decision": "deny", "reason": "insight_type_not_supported", "required_action": "select_supported_insight_type"}},
	{"name": "insight_claim_required", "condition": {"operation": "record_insight", "claim_present": False}, "effect": {"decision": "deny", "reason": "claim_reference_required", "required_action": "attach_claim_reference"}},
	{"name": "insight_confidence_valid", "condition": {"operation": "record_insight", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "insight_analyst_required", "condition": {"operation": "record_insight", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "insight_evidence_required", "condition": {"operation": "record_insight", "evidence_present": False}, "effect": {"decision": "deny", "reason": "insight_evidence_required", "required_action": "attach_insight_evidence"}},
	{"name": "dashboard_insight_required", "condition": {"operation": "record_dashboard", "insight_present": False}, "effect": {"decision": "deny", "reason": "insight_required", "required_action": "select_insight"}},
	{"name": "dashboard_name_required", "condition": {"operation": "record_dashboard", "dashboard_name_present": False}, "effect": {"decision": "deny", "reason": "dashboard_name_required", "required_action": "name_dashboard"}},
	{"name": "dashboard_audience_required", "condition": {"operation": "record_dashboard", "audience_present": False}, "effect": {"decision": "deny", "reason": "audience_required", "required_action": "select_audience"}},
	{"name": "dashboard_release_required", "condition": {"operation": "record_dashboard", "release_marking_present": False}, "effect": {"decision": "deny", "reason": "release_marking_required", "required_action": "set_release_marking"}},
	{"name": "dashboard_approval_required", "condition": {"operation": "record_dashboard", "approval_present": False}, "effect": {"decision": "deny", "reason": "dashboard_approval_required", "required_action": "attach_dashboard_approval"}},
	{"name": "dashboard_evidence_required", "condition": {"operation": "record_dashboard", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dashboard_evidence_required", "required_action": "attach_dashboard_evidence"}},
	{"name": "narrative_insight_required", "condition": {"operation": "record_narrative", "insight_present": False}, "effect": {"decision": "deny", "reason": "insight_required", "required_action": "select_insight"}},
	{"name": "narrative_type_supported", "condition": {"operation": "record_narrative", "narrative_type_supported": False}, "effect": {"decision": "deny", "reason": "narrative_type_not_supported", "required_action": "select_supported_narrative_type"}},
	{"name": "narrative_summary_required", "condition": {"operation": "record_narrative", "summary_present": False}, "effect": {"decision": "deny", "reason": "summary_reference_required", "required_action": "attach_summary_reference"}},
	{"name": "narrative_approval_required", "condition": {"operation": "record_narrative", "approval_present": False}, "effect": {"decision": "deny", "reason": "narrative_approval_required", "required_action": "attach_narrative_approval"}},
	{"name": "narrative_evidence_required", "condition": {"operation": "record_narrative", "evidence_present": False}, "effect": {"decision": "deny", "reason": "narrative_evidence_required", "required_action": "attach_narrative_evidence"}},
	{"name": "recommendation_insight_required", "condition": {"operation": "record_recommendation", "insight_present": False}, "effect": {"decision": "deny", "reason": "insight_required", "required_action": "select_insight"}},
	{"name": "recommendation_type_supported", "condition": {"operation": "record_recommendation", "recommendation_type_supported": False}, "effect": {"decision": "deny", "reason": "recommendation_type_not_supported", "required_action": "select_supported_recommendation_type"}},
	{"name": "recommendation_action_required", "condition": {"operation": "record_recommendation", "action_present": False}, "effect": {"decision": "deny", "reason": "action_reference_required", "required_action": "attach_action_reference"}},
	{"name": "recommendation_approval_required", "condition": {"operation": "record_recommendation", "approval_present": False}, "effect": {"decision": "deny", "reason": "recommendation_approval_required", "required_action": "attach_recommendation_approval"}},
	{"name": "recommendation_evidence_required", "condition": {"operation": "record_recommendation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "recommendation_evidence_required", "required_action": "attach_recommendation_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "analytics_batch_requires_bytewax", "condition": {"operation": "analytics_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_analytics_batch_to_bytewax"}},
	{"name": "analytics_agent_runtime_supported", "condition": {"operation": "register_analytics_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "analytics_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "analytics_agent_role_supported", "condition": {"operation": "register_analytics_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "analytics_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_analytics_agent_action_requires_human_approval", "condition": {"operation": "analytics_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "hallucinated_insight_action_denied", "condition": {"operation": "analytics_agent_action", "hallucinated_insight_scope": True}, "effect": {"decision": "deny", "reason": "hallucinated_insight_scope_denied", "required_action": "remove_hallucinated_insight_scope"}},
	{"name": "training_data_leakage_action_denied", "condition": {"operation": "analytics_agent_action", "training_data_leakage_scope": True}, "effect": {"decision": "deny", "reason": "training_data_leakage_scope_denied", "required_action": "remove_training_data_leakage_scope"}},
	{"name": "privacy_bypass_action_denied", "condition": {"operation": "analytics_agent_action", "privacy_bypass_scope": True}, "effect": {"decision": "deny", "reason": "privacy_bypass_scope_denied", "required_action": "remove_privacy_bypass_scope"}},
	{"name": "unsupported_automated_decision_action_denied", "condition": {"operation": "analytics_agent_action", "unsupported_automated_decision_scope": True}, "effect": {"decision": "deny", "reason": "unsupported_automated_decision_scope_denied", "required_action": "remove_automated_decision_scope"}},
	{"name": "unapproved_model_deployment_action_denied", "condition": {"operation": "analytics_agent_action", "unapproved_model_deployment_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_model_deployment_scope_denied", "required_action": "remove_model_deployment_scope"}},
	{"name": "autonomous_dissemination_action_denied", "condition": {"operation": "analytics_agent_action", "autonomous_dissemination_scope": True}, "effect": {"decision": "deny", "reason": "autonomous_dissemination_scope_denied", "required_action": "remove_autonomous_dissemination_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-analytics/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions: list[dict[str, Any]] = []
	for rule in RULES:
		if _matches(rule["condition"], context):
			actions.append(rule["effect"] | {"rule": rule["name"]})
	if not actions:
		return {"decision": "allow", "actions": [], "context": dict(context)}
	return {"decision": "deny", "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
