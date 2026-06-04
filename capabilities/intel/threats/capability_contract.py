"""Executable capability contract for APG Threat Intelligence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_threats"
CAPABILITY_NAME = "Threat Intelligence"
CAPABILITY_VERSION = "1.1.0"
THREAT_EVENT_STREAM = "apg.intel.threats.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "legal_mandate", "partner_authority", "consent", "incident_response_authority", "public_interest_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_WORKSPACE_TYPES = ["strategic_threat", "cyber_threat", "physical_security", "fraud_threat", "geopolitical_threat", "insider_threat", "supply_chain_threat"]
SUPPORTED_SOURCE_TYPES = ["osint", "sigint", "humint", "geoint", "cybint", "finint", "partner_report", "sensor_feed"]
SUPPORTED_INDICATOR_TYPES = ["ioc", "tactic", "technique", "procedure", "behavior", "vulnerability", "infrastructure", "narrative", "financial_signal"]
SUPPORTED_ACTOR_TYPES = ["state_actor", "criminal_group", "insider", "hacktivist", "terrorist_network", "competitor", "unknown"]
SUPPORTED_CAMPAIGN_TYPES = ["intrusion_campaign", "fraud_campaign", "disinformation_campaign", "physical_threat_campaign", "insider_campaign", "supply_chain_campaign"]
SUPPORTED_ASSESSMENT_TYPES = ["threat_profile", "risk_assessment", "priority_assessment", "attribution_assessment", "intent_assessment", "capability_assessment"]
SUPPORTED_REPORT_TYPES = ["brief", "advisory", "bulletin", "estimate", "watchlist", "situation_report"]
SUPPORTED_MITIGATION_TYPES = ["monitor", "block", "patch", "investigate", "harden", "disrupt", "escalate", "notify"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["source_triage", "indicator_curator", "actor_analyst", "assessment_reviewer", "mitigation_reviewer", "report_writer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"workspaces": {"supported_workspace_types": SUPPORTED_WORKSPACE_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "authority_required": True, "evidence_required": True},
	"sources": {"supported_source_types": SUPPORTED_SOURCE_TYPES, "workspace_required": True, "custodian_required": True, "lineage_required": True, "evidence_required": True},
	"indicators": {"supported_indicator_types": SUPPORTED_INDICATOR_TYPES, "source_required": True, "confidence_required": True, "evidence_required": True},
	"actors": {"supported_actor_types": SUPPORTED_ACTOR_TYPES, "workspace_required": True, "confidence_required": True, "evidence_required": True},
	"campaigns": {"supported_campaign_types": SUPPORTED_CAMPAIGN_TYPES, "actor_required": True, "risk_level_required": True, "evidence_required": True},
	"assessments": {"supported_assessment_types": SUPPORTED_ASSESSMENT_TYPES, "campaign_required": True, "risk_level_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"reports": {"supported_report_types": SUPPORTED_REPORT_TYPES, "assessment_required": True, "approval_required": True, "evidence_required": True},
	"mitigations": {"supported_mitigation_types": SUPPORTED_MITIGATION_TYPES, "assessment_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "name_required": True, "scope_required": True, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "cross_tenant_threat_denied": True, "unsupported_attribution_denied": True, "fabricated_indicator_denied": True, "source_tampering_denied": True, "privacy_bypass_denied": True, "autonomous_mitigation_denied": True, "unapproved_publication_denied": True},
	"observability": {"event_stream": THREAT_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "geospatial": "geos", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_workspaces": True, "enable_sources": True, "enable_indicators": True, "enable_actors": True, "enable_campaigns": True, "enable_assessments": True, "enable_reports": True, "enable_mitigations": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_threats_control", "allow_tenant_overrides": True},
}

PROVIDES = ["threat_authority_workflow", "threat_workspace_workflow", "threat_source_workflow", "threat_indicator_workflow", "threat_actor_workflow", "threat_campaign_workflow", "threat_assessment_workflow", "threat_report_workflow", "threat_mitigation_workflow", "threat_review_workflow", "threat_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-threats/dashboard", "component": "ThreatDashboard", "permission": "intel_threats:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-threats/authorities", "component": "ThreatAuthorityConsole", "permission": "intel_threats:authorities", "nav_group": "Governance"},
	{"name": "workspaces", "path": "/intel-threats/workspaces", "component": "ThreatWorkspaceConsole", "permission": "intel_threats:workspaces", "nav_group": "Planning"},
	{"name": "sources", "path": "/intel-threats/sources", "component": "ThreatSourceLedger", "permission": "intel_threats:sources", "nav_group": "Evidence"},
	{"name": "indicators", "path": "/intel-threats/indicators", "component": "ThreatIndicatorLedger", "permission": "intel_threats:indicators", "nav_group": "Evidence"},
	{"name": "actors", "path": "/intel-threats/actors", "component": "ThreatActorWorkbench", "permission": "intel_threats:actors", "nav_group": "Analysis"},
	{"name": "campaigns", "path": "/intel-threats/campaigns", "component": "ThreatCampaignWorkbench", "permission": "intel_threats:campaigns", "nav_group": "Analysis"},
	{"name": "assessments", "path": "/intel-threats/assessments", "component": "ThreatAssessmentConsole", "permission": "intel_threats:assessments", "nav_group": "Analysis"},
	{"name": "reports", "path": "/intel-threats/reports", "component": "ThreatReportConsole", "permission": "intel_threats:reports", "nav_group": "Products"},
	{"name": "mitigations", "path": "/intel-threats/mitigations", "component": "ThreatMitigationConsole", "permission": "intel_threats:mitigations", "nav_group": "Action"},
	{"name": "reviews", "path": "/intel-threats/reviews", "component": "ThreatReviewConsole", "permission": "intel_threats:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-threats/agents", "component": "ThreatAgentWorkbench", "permission": "intel_threats:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-threats/settings", "component": "ThreatSettings", "permission": "intel_threats:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_threats_control",
	"tokens": {"color.primary": "#991B1B", "color.accent": "#0F766E", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "workspaces": {"icon": "layout-dashboard", "status_indicator": "workspace-chip"}, "sources": {"icon": "database", "status_indicator": "source-chip"}, "indicators": {"icon": "radar", "status_indicator": "indicator-chip"}, "actors": {"icon": "user-round-search", "status_indicator": "actor-chip"}, "campaigns": {"icon": "route", "status_indicator": "campaign-chip"}, "assessments": {"icon": "clipboard-list", "status_indicator": "risk-chip"}, "reports": {"icon": "file-text", "status_indicator": "report-chip"}, "mitigations": {"icon": "list-checks", "status_indicator": "action-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": THREAT_EVENT_STREAM, "key": "tenant_id", "events": ["threat_authority_recorded", "threat_workspace_recorded", "threat_source_registered", "threat_indicator_recorded", "threat_actor_recorded", "threat_campaign_recorded", "threat_assessment_recorded", "threat_report_recorded", "threat_mitigation_recorded", "threat_review_recorded", "threat_agent_registered"], "guardrails": ["threat_batch_requires_bytewax", "privileged_threat_agent_action_requires_human_approval", "unsupported_attribution_action_denied", "fabricated_indicator_action_denied", "source_tampering_action_denied", "privacy_bypass_action_denied", "autonomous_mitigation_action_denied", "unapproved_publication_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "threat_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "threat_policy_required", "required_action": "attach_threat_policy"}},
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
	{"name": "source_lineage_required", "condition": {"operation": "register_source", "lineage_present": False}, "effect": {"decision": "deny", "reason": "source_lineage_required", "required_action": "attach_source_lineage"}},
	{"name": "source_evidence_required", "condition": {"operation": "register_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "indicator_source_required", "condition": {"operation": "record_indicator", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "indicator_type_supported", "condition": {"operation": "record_indicator", "indicator_type_supported": False}, "effect": {"decision": "deny", "reason": "indicator_type_not_supported", "required_action": "select_supported_indicator_type"}},
	{"name": "indicator_reference_required", "condition": {"operation": "record_indicator", "indicator_reference_present": False}, "effect": {"decision": "deny", "reason": "indicator_reference_required", "required_action": "attach_indicator_reference"}},
	{"name": "indicator_confidence_valid", "condition": {"operation": "record_indicator", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "indicator_evidence_required", "condition": {"operation": "record_indicator", "evidence_present": False}, "effect": {"decision": "deny", "reason": "indicator_evidence_required", "required_action": "attach_indicator_evidence"}},
	{"name": "actor_workspace_required", "condition": {"operation": "record_actor", "workspace_present": False}, "effect": {"decision": "deny", "reason": "workspace_required", "required_action": "select_workspace"}},
	{"name": "actor_type_supported", "condition": {"operation": "record_actor", "actor_type_supported": False}, "effect": {"decision": "deny", "reason": "actor_type_not_supported", "required_action": "select_supported_actor_type"}},
	{"name": "actor_reference_required", "condition": {"operation": "record_actor", "actor_reference_present": False}, "effect": {"decision": "deny", "reason": "actor_reference_required", "required_action": "attach_actor_reference"}},
	{"name": "actor_confidence_valid", "condition": {"operation": "record_actor", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "actor_evidence_required", "condition": {"operation": "record_actor", "evidence_present": False}, "effect": {"decision": "deny", "reason": "actor_evidence_required", "required_action": "attach_actor_evidence"}},
	{"name": "campaign_actor_required", "condition": {"operation": "record_campaign", "actor_present": False}, "effect": {"decision": "deny", "reason": "actor_required", "required_action": "select_actor"}},
	{"name": "campaign_type_supported", "condition": {"operation": "record_campaign", "campaign_type_supported": False}, "effect": {"decision": "deny", "reason": "campaign_type_not_supported", "required_action": "select_supported_campaign_type"}},
	{"name": "campaign_reference_required", "condition": {"operation": "record_campaign", "campaign_reference_present": False}, "effect": {"decision": "deny", "reason": "campaign_reference_required", "required_action": "attach_campaign_reference"}},
	{"name": "campaign_risk_supported", "condition": {"operation": "record_campaign", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "campaign_evidence_required", "condition": {"operation": "record_campaign", "evidence_present": False}, "effect": {"decision": "deny", "reason": "campaign_evidence_required", "required_action": "attach_campaign_evidence"}},
	{"name": "assessment_campaign_required", "condition": {"operation": "record_assessment", "campaign_present": False}, "effect": {"decision": "deny", "reason": "campaign_required", "required_action": "select_campaign"}},
	{"name": "assessment_type_supported", "condition": {"operation": "record_assessment", "assessment_type_supported": False}, "effect": {"decision": "deny", "reason": "assessment_type_not_supported", "required_action": "select_supported_assessment_type"}},
	{"name": "assessment_risk_supported", "condition": {"operation": "record_assessment", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "assessment_confidence_valid", "condition": {"operation": "record_assessment", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "assessment_analyst_required", "condition": {"operation": "record_assessment", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "assessment_evidence_required", "condition": {"operation": "record_assessment", "evidence_present": False}, "effect": {"decision": "deny", "reason": "assessment_evidence_required", "required_action": "attach_assessment_evidence"}},
	{"name": "report_assessment_required", "condition": {"operation": "record_report", "assessment_present": False}, "effect": {"decision": "deny", "reason": "assessment_required", "required_action": "select_assessment"}},
	{"name": "report_type_supported", "condition": {"operation": "record_report", "report_type_supported": False}, "effect": {"decision": "deny", "reason": "report_type_not_supported", "required_action": "select_supported_report_type"}},
	{"name": "report_reference_required", "condition": {"operation": "record_report", "report_reference_present": False}, "effect": {"decision": "deny", "reason": "report_reference_required", "required_action": "attach_report_reference"}},
	{"name": "report_approval_required", "condition": {"operation": "record_report", "approval_present": False}, "effect": {"decision": "deny", "reason": "report_approval_required", "required_action": "attach_report_approval"}},
	{"name": "report_evidence_required", "condition": {"operation": "record_report", "evidence_present": False}, "effect": {"decision": "deny", "reason": "report_evidence_required", "required_action": "attach_report_evidence"}},
	{"name": "mitigation_assessment_required", "condition": {"operation": "record_mitigation", "assessment_present": False}, "effect": {"decision": "deny", "reason": "assessment_required", "required_action": "select_assessment"}},
	{"name": "mitigation_type_supported", "condition": {"operation": "record_mitigation", "mitigation_type_supported": False}, "effect": {"decision": "deny", "reason": "mitigation_type_not_supported", "required_action": "select_supported_mitigation_type"}},
	{"name": "mitigation_action_required", "condition": {"operation": "record_mitigation", "action_present": False}, "effect": {"decision": "deny", "reason": "mitigation_action_required", "required_action": "attach_action_reference"}},
	{"name": "mitigation_approval_required", "condition": {"operation": "record_mitigation", "approval_present": False}, "effect": {"decision": "deny", "reason": "mitigation_approval_required", "required_action": "attach_mitigation_approval"}},
	{"name": "mitigation_evidence_required", "condition": {"operation": "record_mitigation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "mitigation_evidence_required", "required_action": "attach_mitigation_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "threat_batch_requires_bytewax", "condition": {"operation": "threat_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_threat_batch_to_bytewax"}},
	{"name": "threat_agent_runtime_supported", "condition": {"operation": "register_threat_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "threat_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "threat_agent_role_supported", "condition": {"operation": "register_threat_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "threat_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "threat_agent_name_required", "condition": {"operation": "register_threat_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "threat_agent_name_required", "required_action": "name_threat_agent"}},
	{"name": "threat_agent_scope_required", "condition": {"operation": "register_threat_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "threat_agent_scope_required", "required_action": "bound_threat_agent_scope"}},
	{"name": "privileged_threat_agent_action_requires_human_approval", "condition": {"operation": "threat_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "unsupported_attribution_action_denied", "condition": {"operation": "threat_agent_action", "unsupported_attribution_scope": True}, "effect": {"decision": "deny", "reason": "unsupported_attribution_scope_denied", "required_action": "remove_unsupported_attribution_scope"}},
	{"name": "fabricated_indicator_action_denied", "condition": {"operation": "threat_agent_action", "fabricated_indicator_scope": True}, "effect": {"decision": "deny", "reason": "fabricated_indicator_scope_denied", "required_action": "remove_fabricated_indicator_scope"}},
	{"name": "source_tampering_action_denied", "condition": {"operation": "threat_agent_action", "source_tampering_scope": True}, "effect": {"decision": "deny", "reason": "source_tampering_scope_denied", "required_action": "remove_source_tampering_scope"}},
	{"name": "privacy_bypass_action_denied", "condition": {"operation": "threat_agent_action", "privacy_bypass_scope": True}, "effect": {"decision": "deny", "reason": "privacy_bypass_scope_denied", "required_action": "remove_privacy_bypass_scope"}},
	{"name": "autonomous_mitigation_action_denied", "condition": {"operation": "threat_agent_action", "autonomous_mitigation_scope": True}, "effect": {"decision": "deny", "reason": "autonomous_mitigation_scope_denied", "required_action": "remove_autonomous_mitigation_scope"}},
	{"name": "unapproved_publication_action_denied", "condition": {"operation": "threat_agent_action", "unapproved_publication_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_publication_scope_denied", "required_action": "remove_unapproved_publication_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-threats/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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

