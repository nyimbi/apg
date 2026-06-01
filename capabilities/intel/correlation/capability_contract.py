"""Executable capability contract for APG Data Correlation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_correlation"
CAPABILITY_NAME = "Data Correlation"
CAPABILITY_VERSION = "1.1.0"
CORRELATION_EVENT_STREAM = "apg.intel.correlation.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "legal_mandate", "partner_authority", "consent", "incident_response_authority", "public_interest_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_WORKSPACE_TYPES = ["entity_resolution", "link_analysis", "fraud_correlation", "threat_correlation", "public_safety_correlation", "operational_correlation", "incident_correlation"]
SUPPORTED_SOURCE_TYPES = ["fusion_extract", "graph_projection", "entity_table", "event_stream", "geospatial_layer", "transaction_set", "document_corpus", "partner_dataset"]
SUPPORTED_ENTITY_TYPES = ["person", "organization", "device", "account", "location", "vehicle", "event", "asset", "indicator"]
SUPPORTED_OBSERVATION_TYPES = ["attribute", "relationship", "location", "time_event", "transaction", "communication", "behavior", "document_mention"]
SUPPORTED_RULE_TYPES = ["exact_match", "fuzzy_match", "temporal_overlap", "geospatial_overlap", "network_link", "behavioral_similarity", "cross_source_confirmation", "contradiction"]
SUPPORTED_RUN_TYPES = ["batch", "streaming", "backtest", "review_sample", "what_if"]
SUPPORTED_CLUSTER_TYPES = ["entity_cluster", "relationship_cluster", "event_cluster", "location_cluster", "transaction_cluster", "risk_cluster"]
SUPPORTED_DECISION_TYPES = ["confirmed_match", "possible_match", "not_match", "needs_review", "merged_identity", "split_identity", "suppressed_match"]
SUPPORTED_REFERRAL_TYPES = ["analyst_review", "case_escalation", "fraud_review", "threat_review", "compliance_review", "public_safety_notice"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["source_steward", "entity_reviewer", "rule_reviewer", "cluster_analyst", "resolution_reviewer", "referral_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"workspaces": {"supported_workspace_types": SUPPORTED_WORKSPACE_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "authority_required": True, "evidence_required": True},
	"sources": {"supported_source_types": SUPPORTED_SOURCE_TYPES, "workspace_required": True, "custodian_required": True, "lineage_required": True, "evidence_required": True},
	"entities": {"supported_entity_types": SUPPORTED_ENTITY_TYPES, "source_required": True, "confidence_required": True, "evidence_required": True},
	"observations": {"supported_observation_types": SUPPORTED_OBSERVATION_TYPES, "entity_required": True, "observed_at_required": True, "confidence_required": True, "evidence_required": True},
	"rules": {"supported_rule_types": SUPPORTED_RULE_TYPES, "workspace_required": True, "threshold_required": True, "analyst_required": True, "evidence_required": True},
	"runs": {"supported_run_types": SUPPORTED_RUN_TYPES, "rule_required": True, "result_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"clusters": {"supported_cluster_types": SUPPORTED_CLUSTER_TYPES, "run_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"decisions": {"supported_decision_types": SUPPORTED_DECISION_TYPES, "cluster_required": True, "rationale_required": True, "approval_required": True, "evidence_required": True},
	"referrals": {"supported_referral_types": SUPPORTED_REFERRAL_TYPES, "decision_required": True, "recipient_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "cross_tenant_correlation_denied": True, "unapproved_identity_merge_denied": True, "source_tampering_denied": True, "privacy_bypass_denied": True, "evidence_fabrication_denied": True, "autonomous_referral_denied": True, "unreviewed_high_impact_match_denied": True},
	"observability": {"event_stream": CORRELATION_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "geospatial": "geos", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_workspaces": True, "enable_sources": True, "enable_entities": True, "enable_observations": True, "enable_rules": True, "enable_runs": True, "enable_clusters": True, "enable_decisions": True, "enable_referrals": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_correlation_control", "allow_tenant_overrides": True},
}

PROVIDES = ["correlation_authority_workflow", "correlation_workspace_workflow", "correlation_source_workflow", "correlation_entity_workflow", "correlation_observation_workflow", "correlation_rule_workflow", "correlation_run_workflow", "correlation_cluster_workflow", "correlation_decision_workflow", "correlation_referral_workflow", "correlation_review_workflow", "correlation_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-correlation/dashboard", "component": "CorrelationDashboard", "permission": "intel_correlation:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-correlation/authorities", "component": "CorrelationAuthorityConsole", "permission": "intel_correlation:authorities", "nav_group": "Governance"},
	{"name": "workspaces", "path": "/intel-correlation/workspaces", "component": "CorrelationWorkspaceConsole", "permission": "intel_correlation:workspaces", "nav_group": "Planning"},
	{"name": "sources", "path": "/intel-correlation/sources", "component": "CorrelationSourceRegistry", "permission": "intel_correlation:sources", "nav_group": "Data"},
	{"name": "entities", "path": "/intel-correlation/entities", "component": "CorrelationEntityLedger", "permission": "intel_correlation:entities", "nav_group": "Data"},
	{"name": "observations", "path": "/intel-correlation/observations", "component": "CorrelationObservationLedger", "permission": "intel_correlation:observations", "nav_group": "Data"},
	{"name": "rules", "path": "/intel-correlation/rules", "component": "CorrelationRuleWorkbench", "permission": "intel_correlation:rules", "nav_group": "Analysis"},
	{"name": "runs", "path": "/intel-correlation/runs", "component": "CorrelationRunConsole", "permission": "intel_correlation:runs", "nav_group": "Analysis"},
	{"name": "clusters", "path": "/intel-correlation/clusters", "component": "CorrelationClusterWorkbench", "permission": "intel_correlation:clusters", "nav_group": "Analysis"},
	{"name": "decisions", "path": "/intel-correlation/decisions", "component": "CorrelationDecisionConsole", "permission": "intel_correlation:decisions", "nav_group": "Resolution"},
	{"name": "referrals", "path": "/intel-correlation/referrals", "component": "CorrelationReferralConsole", "permission": "intel_correlation:referrals", "nav_group": "Resolution"},
	{"name": "reviews", "path": "/intel-correlation/reviews", "component": "CorrelationReviewConsole", "permission": "intel_correlation:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-correlation/agents", "component": "CorrelationAgentWorkbench", "permission": "intel_correlation:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-correlation/settings", "component": "CorrelationSettings", "permission": "intel_correlation:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_correlation_control",
	"tokens": {"color.primary": "#1D4ED8", "color.accent": "#9333EA", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "workspaces": {"icon": "layout-dashboard", "status_indicator": "workspace-chip"}, "sources": {"icon": "database", "status_indicator": "source-chip"}, "entities": {"icon": "user-round-search", "status_indicator": "entity-chip"}, "observations": {"icon": "scan-search", "status_indicator": "observation-chip"}, "rules": {"icon": "git-compare", "status_indicator": "rule-chip"}, "runs": {"icon": "activity", "status_indicator": "run-chip"}, "clusters": {"icon": "network", "status_indicator": "confidence-chip"}, "decisions": {"icon": "git-merge", "status_indicator": "decision-chip"}, "referrals": {"icon": "file-output", "status_indicator": "referral-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": CORRELATION_EVENT_STREAM, "key": "tenant_id", "events": ["correlation_authority_recorded", "correlation_workspace_recorded", "correlation_source_registered", "correlation_entity_recorded", "correlation_observation_recorded", "correlation_rule_recorded", "correlation_run_recorded", "correlation_cluster_recorded", "correlation_decision_recorded", "correlation_referral_recorded", "correlation_review_recorded", "correlation_agent_registered"], "guardrails": ["correlation_batch_requires_bytewax", "privileged_correlation_agent_action_requires_human_approval", "unapproved_identity_merge_action_denied", "source_tampering_action_denied", "privacy_bypass_action_denied", "evidence_fabrication_action_denied", "autonomous_referral_action_denied", "unreviewed_high_impact_match_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "correlation_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "correlation_policy_required", "required_action": "attach_correlation_policy"}},
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
	{"name": "source_workspace_required", "condition": {"operation": "register_source", "workspace_present": False}, "effect": {"decision": "deny", "reason": "workspace_required", "required_action": "select_workspace"}},
	{"name": "source_type_supported", "condition": {"operation": "register_source", "source_type_supported": False}, "effect": {"decision": "deny", "reason": "source_type_not_supported", "required_action": "select_supported_source_type"}},
	{"name": "source_reference_required", "condition": {"operation": "register_source", "source_reference_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "source_custodian_required", "condition": {"operation": "register_source", "custodian_present": False}, "effect": {"decision": "deny", "reason": "source_custodian_required", "required_action": "assign_source_custodian"}},
	{"name": "source_lineage_required", "condition": {"operation": "register_source", "lineage_present": False}, "effect": {"decision": "deny", "reason": "source_lineage_required", "required_action": "record_lineage"}},
	{"name": "source_evidence_required", "condition": {"operation": "register_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "entity_source_required", "condition": {"operation": "record_entity", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "entity_type_supported", "condition": {"operation": "record_entity", "entity_type_supported": False}, "effect": {"decision": "deny", "reason": "entity_type_not_supported", "required_action": "select_supported_entity_type"}},
	{"name": "entity_reference_required", "condition": {"operation": "record_entity", "entity_reference_present": False}, "effect": {"decision": "deny", "reason": "entity_reference_required", "required_action": "attach_entity_reference"}},
	{"name": "entity_confidence_valid", "condition": {"operation": "record_entity", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "entity_evidence_required", "condition": {"operation": "record_entity", "evidence_present": False}, "effect": {"decision": "deny", "reason": "entity_evidence_required", "required_action": "attach_entity_evidence"}},
	{"name": "observation_entity_required", "condition": {"operation": "record_observation", "entity_present": False}, "effect": {"decision": "deny", "reason": "entity_required", "required_action": "select_entity"}},
	{"name": "observation_type_supported", "condition": {"operation": "record_observation", "observation_type_supported": False}, "effect": {"decision": "deny", "reason": "observation_type_not_supported", "required_action": "select_supported_observation_type"}},
	{"name": "observation_reference_required", "condition": {"operation": "record_observation", "observation_reference_present": False}, "effect": {"decision": "deny", "reason": "observation_reference_required", "required_action": "attach_observation_reference"}},
	{"name": "observation_time_required", "condition": {"operation": "record_observation", "observed_at_present": False}, "effect": {"decision": "deny", "reason": "observed_at_required", "required_action": "record_observation_time"}},
	{"name": "observation_confidence_valid", "condition": {"operation": "record_observation", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "observation_evidence_required", "condition": {"operation": "record_observation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "observation_evidence_required", "required_action": "attach_observation_evidence"}},
	{"name": "rule_workspace_required", "condition": {"operation": "record_rule", "workspace_present": False}, "effect": {"decision": "deny", "reason": "workspace_required", "required_action": "select_workspace"}},
	{"name": "rule_type_supported", "condition": {"operation": "record_rule", "rule_type_supported": False}, "effect": {"decision": "deny", "reason": "rule_type_not_supported", "required_action": "select_supported_rule_type"}},
	{"name": "rule_reference_required", "condition": {"operation": "record_rule", "rule_reference_present": False}, "effect": {"decision": "deny", "reason": "rule_reference_required", "required_action": "attach_rule_reference"}},
	{"name": "rule_threshold_valid", "condition": {"operation": "record_rule", "threshold_valid": False}, "effect": {"decision": "deny", "reason": "threshold_score_invalid", "required_action": "set_threshold_0_to_1"}},
	{"name": "rule_analyst_required", "condition": {"operation": "record_rule", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "rule_evidence_required", "condition": {"operation": "record_rule", "evidence_present": False}, "effect": {"decision": "deny", "reason": "rule_evidence_required", "required_action": "attach_rule_evidence"}},
	{"name": "run_rule_required", "condition": {"operation": "record_run", "rule_present": False}, "effect": {"decision": "deny", "reason": "rule_required", "required_action": "select_rule"}},
	{"name": "run_type_supported", "condition": {"operation": "record_run", "run_type_supported": False}, "effect": {"decision": "deny", "reason": "run_type_not_supported", "required_action": "select_supported_run_type"}},
	{"name": "run_result_required", "condition": {"operation": "record_run", "result_reference_present": False}, "effect": {"decision": "deny", "reason": "run_result_required", "required_action": "attach_result_reference"}},
	{"name": "run_confidence_valid", "condition": {"operation": "record_run", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "run_analyst_required", "condition": {"operation": "record_run", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "run_evidence_required", "condition": {"operation": "record_run", "evidence_present": False}, "effect": {"decision": "deny", "reason": "run_evidence_required", "required_action": "attach_run_evidence"}},
	{"name": "cluster_run_required", "condition": {"operation": "record_cluster", "run_present": False}, "effect": {"decision": "deny", "reason": "run_required", "required_action": "select_run"}},
	{"name": "cluster_type_supported", "condition": {"operation": "record_cluster", "cluster_type_supported": False}, "effect": {"decision": "deny", "reason": "cluster_type_not_supported", "required_action": "select_supported_cluster_type"}},
	{"name": "cluster_reference_required", "condition": {"operation": "record_cluster", "cluster_reference_present": False}, "effect": {"decision": "deny", "reason": "cluster_reference_required", "required_action": "attach_cluster_reference"}},
	{"name": "cluster_confidence_valid", "condition": {"operation": "record_cluster", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "cluster_analyst_required", "condition": {"operation": "record_cluster", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "cluster_evidence_required", "condition": {"operation": "record_cluster", "evidence_present": False}, "effect": {"decision": "deny", "reason": "cluster_evidence_required", "required_action": "attach_cluster_evidence"}},
	{"name": "decision_cluster_required", "condition": {"operation": "record_decision", "cluster_present": False}, "effect": {"decision": "deny", "reason": "cluster_required", "required_action": "select_cluster"}},
	{"name": "decision_type_supported", "condition": {"operation": "record_decision", "decision_type_supported": False}, "effect": {"decision": "deny", "reason": "decision_type_not_supported", "required_action": "select_supported_decision_type"}},
	{"name": "decision_rationale_required", "condition": {"operation": "record_decision", "rationale_present": False}, "effect": {"decision": "deny", "reason": "decision_rationale_required", "required_action": "attach_rationale"}},
	{"name": "decision_approval_required", "condition": {"operation": "record_decision", "approval_present": False}, "effect": {"decision": "deny", "reason": "decision_approval_required", "required_action": "attach_decision_approval"}},
	{"name": "decision_evidence_required", "condition": {"operation": "record_decision", "evidence_present": False}, "effect": {"decision": "deny", "reason": "decision_evidence_required", "required_action": "attach_decision_evidence"}},
	{"name": "referral_decision_required", "condition": {"operation": "record_referral", "decision_present": False}, "effect": {"decision": "deny", "reason": "decision_required", "required_action": "select_decision"}},
	{"name": "referral_type_supported", "condition": {"operation": "record_referral", "referral_type_supported": False}, "effect": {"decision": "deny", "reason": "referral_type_not_supported", "required_action": "select_supported_referral_type"}},
	{"name": "referral_recipient_required", "condition": {"operation": "record_referral", "recipient_present": False}, "effect": {"decision": "deny", "reason": "recipient_required", "required_action": "select_recipient"}},
	{"name": "referral_approval_required", "condition": {"operation": "record_referral", "approval_present": False}, "effect": {"decision": "deny", "reason": "referral_approval_required", "required_action": "attach_referral_approval"}},
	{"name": "referral_evidence_required", "condition": {"operation": "record_referral", "evidence_present": False}, "effect": {"decision": "deny", "reason": "referral_evidence_required", "required_action": "attach_referral_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "correlation_batch_requires_bytewax", "condition": {"operation": "correlation_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_correlation_batch_to_bytewax"}},
	{"name": "correlation_agent_runtime_supported", "condition": {"operation": "register_correlation_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "correlation_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "correlation_agent_role_supported", "condition": {"operation": "register_correlation_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "correlation_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_correlation_agent_action_requires_human_approval", "condition": {"operation": "correlation_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "unapproved_identity_merge_action_denied", "condition": {"operation": "correlation_agent_action", "unapproved_identity_merge_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_identity_merge_scope_denied", "required_action": "remove_identity_merge_scope"}},
	{"name": "source_tampering_action_denied", "condition": {"operation": "correlation_agent_action", "source_tampering_scope": True}, "effect": {"decision": "deny", "reason": "source_tampering_scope_denied", "required_action": "remove_source_tampering_scope"}},
	{"name": "privacy_bypass_action_denied", "condition": {"operation": "correlation_agent_action", "privacy_bypass_scope": True}, "effect": {"decision": "deny", "reason": "privacy_bypass_scope_denied", "required_action": "remove_privacy_bypass_scope"}},
	{"name": "evidence_fabrication_action_denied", "condition": {"operation": "correlation_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_scope_denied", "required_action": "remove_evidence_fabrication_scope"}},
	{"name": "autonomous_referral_action_denied", "condition": {"operation": "correlation_agent_action", "autonomous_referral_scope": True}, "effect": {"decision": "deny", "reason": "autonomous_referral_scope_denied", "required_action": "remove_autonomous_referral_scope"}},
	{"name": "unreviewed_high_impact_match_action_denied", "condition": {"operation": "correlation_agent_action", "unreviewed_high_impact_match_scope": True}, "effect": {"decision": "deny", "reason": "unreviewed_high_impact_match_scope_denied", "required_action": "remove_high_impact_match_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-correlation/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
