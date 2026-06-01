"""Executable capability contract for APG Intelligence Fusion."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_fusion"
CAPABILITY_NAME = "Intelligence Fusion"
CAPABILITY_VERSION = "1.1.0"
FUSION_EVENT_STREAM = "apg.intel.fusion.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "legal_mandate", "partner_authority", "consent", "incident_response_authority", "public_interest_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_WORKSPACE_TYPES = ["case_fusion", "threat_fusion", "fraud_fusion", "public_safety", "strategic_assessment", "operational_picture", "incident_fusion"]
SUPPORTED_SOURCE_TYPES = ["osint", "sigint", "humint", "geoint", "cybint", "finint", "socint", "darkweb", "radio", "monitoring", "partner_report"]
SUPPORTED_ARTIFACT_TYPES = ["report", "observation", "indicator", "entity", "event", "geospatial_feature", "transaction", "signal", "document"]
SUPPORTED_CORRELATION_TYPES = ["entity_match", "time_sequence", "location_overlap", "network_link", "pattern_match", "cross_source_confirmation", "contradiction"]
SUPPORTED_HYPOTHESIS_TYPES = ["attribution", "intent", "capability", "risk", "relationship", "timeline", "course_of_action"]
SUPPORTED_ASSESSMENT_TYPES = ["threat", "fraud", "public_safety", "operational", "strategic", "confidence", "impact"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_REFERRAL_TYPES = ["case_escalation", "incident_response", "public_safety_notice", "partner_notice", "policy_review", "compliance_review", "fraud_review"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["authority_reviewer", "workspace_planner", "source_steward", "correlation_analyst", "hypothesis_analyst", "dissemination_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"workspaces": {"supported_workspace_types": SUPPORTED_WORKSPACE_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "authority_required": True, "evidence_required": True},
	"sources": {"supported_source_types": SUPPORTED_SOURCE_TYPES, "custodian_required": True, "authority_required": True, "lineage_required": True, "evidence_required": True},
	"artifacts": {"supported_artifact_types": SUPPORTED_ARTIFACT_TYPES, "workspace_required": True, "source_required": True, "fingerprint_required": True, "confidence_required": True, "evidence_required": True},
	"correlations": {"supported_types": SUPPORTED_CORRELATION_TYPES, "artifact_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"hypotheses": {"supported_types": SUPPORTED_HYPOTHESIS_TYPES, "correlation_required": True, "claim_reference_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"assessments": {"supported_types": SUPPORTED_ASSESSMENT_TYPES, "supported_risk_levels": SUPPORTED_RISK_LEVELS, "hypothesis_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"referrals": {"supported_types": SUPPORTED_REFERRAL_TYPES, "assessment_required": True, "recipient_required": True, "approval_required": True, "evidence_required": True},
	"dissemination": {"assessment_required": True, "audience_required": True, "release_marking_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True, "fabrication_denied": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "cross_tenant_fusion_denied": True, "evidence_fabrication_denied": True, "source_tampering_denied": True, "privacy_bypass_denied": True, "unsupported_identity_resolution_denied": True, "autonomous_dissemination_denied": True, "unapproved_attribution_denied": True},
	"observability": {"event_stream": FUSION_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "geospatial": "geos", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_workspaces": True, "enable_sources": True, "enable_artifacts": True, "enable_correlations": True, "enable_hypotheses": True, "enable_assessments": True, "enable_referrals": True, "enable_dissemination": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_fusion_control", "allow_tenant_overrides": True},
}

PROVIDES = ["fusion_authority_workflow", "fusion_workspace_workflow", "fusion_source_workflow", "fusion_artifact_workflow", "fusion_correlation_workflow", "fusion_hypothesis_workflow", "fusion_assessment_workflow", "fusion_referral_workflow", "fusion_dissemination_workflow", "fusion_review_workflow", "fusion_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-fusion/dashboard", "component": "FusionDashboard", "permission": "intel_fusion:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-fusion/authorities", "component": "FusionAuthorityConsole", "permission": "intel_fusion:authorities", "nav_group": "Governance"},
	{"name": "workspaces", "path": "/intel-fusion/workspaces", "component": "FusionWorkspaceConsole", "permission": "intel_fusion:workspaces", "nav_group": "Planning"},
	{"name": "sources", "path": "/intel-fusion/sources", "component": "FusionSourceRegistry", "permission": "intel_fusion:sources", "nav_group": "Sources"},
	{"name": "artifacts", "path": "/intel-fusion/artifacts", "component": "FusionArtifactLedger", "permission": "intel_fusion:artifacts", "nav_group": "Evidence"},
	{"name": "correlations", "path": "/intel-fusion/correlations", "component": "FusionCorrelationWorkbench", "permission": "intel_fusion:correlations", "nav_group": "Analysis"},
	{"name": "hypotheses", "path": "/intel-fusion/hypotheses", "component": "FusionHypothesisWorkbench", "permission": "intel_fusion:hypotheses", "nav_group": "Analysis"},
	{"name": "assessments", "path": "/intel-fusion/assessments", "component": "FusionAssessmentWorkbench", "permission": "intel_fusion:assessments", "nav_group": "Analysis"},
	{"name": "referrals", "path": "/intel-fusion/referrals", "component": "FusionReferralConsole", "permission": "intel_fusion:referrals", "nav_group": "Release"},
	{"name": "dissemination", "path": "/intel-fusion/dissemination", "component": "FusionDisseminationConsole", "permission": "intel_fusion:dissemination", "nav_group": "Release"},
	{"name": "reviews", "path": "/intel-fusion/reviews", "component": "FusionReviewConsole", "permission": "intel_fusion:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-fusion/agents", "component": "FusionAgentWorkbench", "permission": "intel_fusion:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-fusion/settings", "component": "FusionSettings", "permission": "intel_fusion:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_fusion_control",
	"tokens": {"color.primary": "#2563EB", "color.accent": "#0F766E", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "workspaces": {"icon": "layout-dashboard", "status_indicator": "workspace-chip"}, "sources": {"icon": "database", "status_indicator": "source-chip"}, "artifacts": {"icon": "file-search", "status_indicator": "evidence-chip"}, "correlations": {"icon": "git-merge", "status_indicator": "confidence-chip"}, "hypotheses": {"icon": "brain", "status_indicator": "hypothesis-chip"}, "assessments": {"icon": "shield-alert", "status_indicator": "risk-chip"}, "referrals": {"icon": "file-output", "status_indicator": "referral-chip"}, "dissemination": {"icon": "send", "status_indicator": "release-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": FUSION_EVENT_STREAM, "key": "tenant_id", "events": ["fusion_authority_recorded", "fusion_workspace_recorded", "fusion_source_registered", "fusion_artifact_recorded", "fusion_correlation_recorded", "fusion_hypothesis_recorded", "fusion_assessment_recorded", "fusion_referral_recorded", "fusion_dissemination_recorded", "fusion_review_recorded", "fusion_agent_registered"], "guardrails": ["fusion_batch_requires_bytewax", "privileged_fusion_agent_action_requires_human_approval", "evidence_fabrication_action_denied", "source_tampering_action_denied", "privacy_bypass_action_denied", "unsupported_identity_resolution_action_denied", "autonomous_dissemination_action_denied", "unapproved_attribution_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "fusion_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "fusion_policy_required", "required_action": "attach_fusion_policy"}},
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
	{"name": "source_type_supported", "condition": {"operation": "register_source", "source_type_supported": False}, "effect": {"decision": "deny", "reason": "source_type_not_supported", "required_action": "select_supported_source_type"}},
	{"name": "source_reference_required", "condition": {"operation": "register_source", "source_reference_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "source_custodian_required", "condition": {"operation": "register_source", "custodian_present": False}, "effect": {"decision": "deny", "reason": "source_custodian_required", "required_action": "assign_source_custodian"}},
	{"name": "source_authority_required", "condition": {"operation": "register_source", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "source_lineage_required", "condition": {"operation": "register_source", "lineage_present": False}, "effect": {"decision": "deny", "reason": "source_lineage_required", "required_action": "record_lineage"}},
	{"name": "source_evidence_required", "condition": {"operation": "register_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "artifact_workspace_required", "condition": {"operation": "record_artifact", "workspace_present": False}, "effect": {"decision": "deny", "reason": "workspace_required", "required_action": "select_workspace"}},
	{"name": "artifact_source_required", "condition": {"operation": "record_artifact", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "artifact_workspace_source_authority_match", "condition": {"operation": "record_artifact", "workspace_source_authority_match": False}, "effect": {"decision": "deny", "reason": "authority_mismatch", "required_action": "align_workspace_source_authority"}},
	{"name": "artifact_type_supported", "condition": {"operation": "record_artifact", "artifact_type_supported": False}, "effect": {"decision": "deny", "reason": "artifact_type_not_supported", "required_action": "select_supported_artifact_type"}},
	{"name": "artifact_reference_required", "condition": {"operation": "record_artifact", "artifact_reference_present": False}, "effect": {"decision": "deny", "reason": "artifact_reference_required", "required_action": "attach_artifact_reference"}},
	{"name": "artifact_fingerprint_required", "condition": {"operation": "record_artifact", "fingerprint_present": False}, "effect": {"decision": "deny", "reason": "content_fingerprint_required", "required_action": "record_content_fingerprint"}},
	{"name": "artifact_confidence_valid", "condition": {"operation": "record_artifact", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "artifact_evidence_required", "condition": {"operation": "record_artifact", "evidence_present": False}, "effect": {"decision": "deny", "reason": "artifact_evidence_required", "required_action": "attach_artifact_evidence"}},
	{"name": "correlation_artifact_required", "condition": {"operation": "record_correlation", "artifact_present": False}, "effect": {"decision": "deny", "reason": "artifact_required", "required_action": "select_artifact"}},
	{"name": "correlation_type_supported", "condition": {"operation": "record_correlation", "correlation_type_supported": False}, "effect": {"decision": "deny", "reason": "correlation_type_not_supported", "required_action": "select_supported_correlation_type"}},
	{"name": "correlation_confidence_valid", "condition": {"operation": "record_correlation", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "correlation_analyst_required", "condition": {"operation": "record_correlation", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "correlation_evidence_required", "condition": {"operation": "record_correlation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "correlation_evidence_required", "required_action": "attach_correlation_evidence"}},
	{"name": "hypothesis_correlation_required", "condition": {"operation": "record_hypothesis", "correlation_present": False}, "effect": {"decision": "deny", "reason": "correlation_required", "required_action": "select_correlation"}},
	{"name": "hypothesis_type_supported", "condition": {"operation": "record_hypothesis", "hypothesis_type_supported": False}, "effect": {"decision": "deny", "reason": "hypothesis_type_not_supported", "required_action": "select_supported_hypothesis_type"}},
	{"name": "hypothesis_claim_required", "condition": {"operation": "record_hypothesis", "claim_present": False}, "effect": {"decision": "deny", "reason": "claim_reference_required", "required_action": "attach_claim_reference"}},
	{"name": "hypothesis_confidence_valid", "condition": {"operation": "record_hypothesis", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "hypothesis_analyst_required", "condition": {"operation": "record_hypothesis", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "hypothesis_evidence_required", "condition": {"operation": "record_hypothesis", "evidence_present": False}, "effect": {"decision": "deny", "reason": "hypothesis_evidence_required", "required_action": "attach_hypothesis_evidence"}},
	{"name": "assessment_hypothesis_required", "condition": {"operation": "record_assessment", "hypothesis_present": False}, "effect": {"decision": "deny", "reason": "hypothesis_required", "required_action": "select_hypothesis"}},
	{"name": "assessment_type_supported", "condition": {"operation": "record_assessment", "assessment_type_supported": False}, "effect": {"decision": "deny", "reason": "assessment_type_not_supported", "required_action": "select_supported_assessment_type"}},
	{"name": "assessment_risk_supported", "condition": {"operation": "record_assessment", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "assessment_confidence_valid", "condition": {"operation": "record_assessment", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "assessment_analyst_required", "condition": {"operation": "record_assessment", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "assessment_evidence_required", "condition": {"operation": "record_assessment", "evidence_present": False}, "effect": {"decision": "deny", "reason": "assessment_evidence_required", "required_action": "attach_assessment_evidence"}},
	{"name": "referral_assessment_required", "condition": {"operation": "record_referral", "assessment_present": False}, "effect": {"decision": "deny", "reason": "assessment_required", "required_action": "select_assessment"}},
	{"name": "referral_type_supported", "condition": {"operation": "record_referral", "referral_type_supported": False}, "effect": {"decision": "deny", "reason": "referral_type_not_supported", "required_action": "select_supported_referral_type"}},
	{"name": "referral_recipient_required", "condition": {"operation": "record_referral", "recipient_present": False}, "effect": {"decision": "deny", "reason": "recipient_required", "required_action": "select_recipient"}},
	{"name": "referral_approval_required", "condition": {"operation": "record_referral", "approval_present": False}, "effect": {"decision": "deny", "reason": "referral_approval_required", "required_action": "attach_referral_approval"}},
	{"name": "referral_evidence_required", "condition": {"operation": "record_referral", "evidence_present": False}, "effect": {"decision": "deny", "reason": "referral_evidence_required", "required_action": "attach_referral_evidence"}},
	{"name": "dissemination_assessment_required", "condition": {"operation": "record_dissemination", "assessment_present": False}, "effect": {"decision": "deny", "reason": "assessment_required", "required_action": "select_assessment"}},
	{"name": "dissemination_audience_required", "condition": {"operation": "record_dissemination", "audience_present": False}, "effect": {"decision": "deny", "reason": "audience_required", "required_action": "select_audience"}},
	{"name": "dissemination_release_required", "condition": {"operation": "record_dissemination", "release_marking_present": False}, "effect": {"decision": "deny", "reason": "release_marking_required", "required_action": "set_release_marking"}},
	{"name": "dissemination_approval_required", "condition": {"operation": "record_dissemination", "approval_present": False}, "effect": {"decision": "deny", "reason": "dissemination_approval_required", "required_action": "attach_release_approval"}},
	{"name": "dissemination_evidence_required", "condition": {"operation": "record_dissemination", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dissemination_evidence_required", "required_action": "attach_dissemination_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "fusion_batch_requires_bytewax", "condition": {"operation": "fusion_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_fusion_batch_to_bytewax"}},
	{"name": "fusion_agent_runtime_supported", "condition": {"operation": "register_fusion_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "fusion_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "fusion_agent_role_supported", "condition": {"operation": "register_fusion_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "fusion_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_fusion_agent_action_requires_human_approval", "condition": {"operation": "fusion_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "evidence_fabrication_action_denied", "condition": {"operation": "fusion_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_scope_denied", "required_action": "remove_evidence_fabrication_scope"}},
	{"name": "source_tampering_action_denied", "condition": {"operation": "fusion_agent_action", "source_tampering_scope": True}, "effect": {"decision": "deny", "reason": "source_tampering_scope_denied", "required_action": "remove_source_tampering_scope"}},
	{"name": "privacy_bypass_action_denied", "condition": {"operation": "fusion_agent_action", "privacy_bypass_scope": True}, "effect": {"decision": "deny", "reason": "privacy_bypass_scope_denied", "required_action": "remove_privacy_bypass_scope"}},
	{"name": "unsupported_identity_resolution_action_denied", "condition": {"operation": "fusion_agent_action", "unsupported_identity_resolution_scope": True}, "effect": {"decision": "deny", "reason": "unsupported_identity_resolution_scope_denied", "required_action": "remove_identity_resolution_scope"}},
	{"name": "autonomous_dissemination_action_denied", "condition": {"operation": "fusion_agent_action", "autonomous_dissemination_scope": True}, "effect": {"decision": "deny", "reason": "autonomous_dissemination_scope_denied", "required_action": "remove_autonomous_dissemination_scope"}},
	{"name": "unapproved_attribution_action_denied", "condition": {"operation": "fusion_agent_action", "unapproved_attribution_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_attribution_scope_denied", "required_action": "remove_unapproved_attribution_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-fusion/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
