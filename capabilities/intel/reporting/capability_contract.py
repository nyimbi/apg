"""Executable capability contract for APG Intelligence Reporting."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_reporting"
CAPABILITY_NAME = "Intelligence Reporting"
CAPABILITY_VERSION = "1.1.0"
REPORTING_EVENT_STREAM = "apg.intel.reporting.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "legal_mandate", "partner_authority", "consent", "incident_response_authority", "public_interest_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_WORKSPACE_TYPES = ["strategic_reporting", "tactical_reporting", "threat_reporting", "incident_reporting", "investigative_reporting", "executive_reporting", "partner_reporting"]
SUPPORTED_TEMPLATE_TYPES = ["brief", "advisory", "bulletin", "estimate", "situation_report", "watchlist", "case_summary", "executive_summary"]
SUPPORTED_PRODUCT_TYPES = ["intelligence_brief", "threat_advisory", "situation_report", "investigative_report", "watchlist", "executive_digest", "partner_notice"]
SUPPORTED_SECTION_TYPES = ["summary", "key_judgement", "background", "evidence", "assessment", "recommendation", "annex", "dissemination_note"]
SUPPORTED_CITATION_TYPES = ["source_extract", "case_reference", "graph_reference", "rag_reference", "geospatial_reference", "model_output", "analyst_note"]
SUPPORTED_APPROVAL_TYPES = ["editorial", "classification", "legal", "operational", "partner_release", "executive_release"]
SUPPORTED_DISTRIBUTION_TYPES = ["internal", "partner", "executive", "field_team", "watch_center", "case_team"]
SUPPORTED_PUBLICATION_TYPES = ["portal", "email_digest", "notification", "case_file", "secure_export", "briefing_pack"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["draft_writer", "source_citation_reviewer", "classification_reviewer", "editorial_reviewer", "distribution_reviewer", "briefing_preparer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"workspaces": {"supported_workspace_types": SUPPORTED_WORKSPACE_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "authority_required": True, "evidence_required": True},
	"templates": {"supported_template_types": SUPPORTED_TEMPLATE_TYPES, "workspace_required": True, "classification_required": True, "evidence_required": True},
	"products": {"supported_product_types": SUPPORTED_PRODUCT_TYPES, "template_required": True, "title_required": True, "author_required": True, "classification_required": True, "evidence_required": True},
	"sections": {"supported_section_types": SUPPORTED_SECTION_TYPES, "product_required": True, "confidence_required": True, "evidence_required": True},
	"citations": {"supported_citation_types": SUPPORTED_CITATION_TYPES, "section_required": True, "source_required": True, "evidence_required": True},
	"approvals": {"supported_approval_types": SUPPORTED_APPROVAL_TYPES, "supported_statuses": SUPPORTED_REVIEW_STATUSES, "product_required": True, "approver_required": True, "evidence_required": True},
	"distributions": {"supported_distribution_types": SUPPORTED_DISTRIBUTION_TYPES, "product_required": True, "recipient_required": True, "approval_required": True, "evidence_required": True},
	"publications": {"supported_publication_types": SUPPORTED_PUBLICATION_TYPES, "distribution_required": True, "publication_reference_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "name_required": True, "scope_required": True, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "cross_tenant_reporting_denied": True, "uncited_claim_denied": True, "classification_downgrade_denied": True, "source_fabrication_denied": True, "privacy_bypass_denied": True, "autonomous_publication_denied": True, "unapproved_distribution_denied": True},
	"observability": {"event_stream": REPORTING_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "geospatial": "geos", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_workspaces": True, "enable_templates": True, "enable_products": True, "enable_sections": True, "enable_citations": True, "enable_approvals": True, "enable_distributions": True, "enable_publications": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_reporting_control", "allow_tenant_overrides": True},
}

PROVIDES = ["reporting_authority_workflow", "reporting_workspace_workflow", "reporting_template_workflow", "reporting_product_workflow", "reporting_section_workflow", "reporting_citation_workflow", "reporting_approval_workflow", "reporting_distribution_workflow", "reporting_publication_workflow", "reporting_review_workflow", "reporting_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-reporting/dashboard", "component": "ReportingDashboard", "permission": "intel_reporting:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-reporting/authorities", "component": "ReportingAuthorityConsole", "permission": "intel_reporting:authorities", "nav_group": "Governance"},
	{"name": "workspaces", "path": "/intel-reporting/workspaces", "component": "ReportingWorkspaceConsole", "permission": "intel_reporting:workspaces", "nav_group": "Planning"},
	{"name": "templates", "path": "/intel-reporting/templates", "component": "ReportingTemplateLibrary", "permission": "intel_reporting:templates", "nav_group": "Products"},
	{"name": "products", "path": "/intel-reporting/products", "component": "ReportingProductWorkbench", "permission": "intel_reporting:products", "nav_group": "Products"},
	{"name": "sections", "path": "/intel-reporting/sections", "component": "ReportingSectionEditor", "permission": "intel_reporting:sections", "nav_group": "Products"},
	{"name": "citations", "path": "/intel-reporting/citations", "component": "ReportingCitationLedger", "permission": "intel_reporting:citations", "nav_group": "Evidence"},
	{"name": "approvals", "path": "/intel-reporting/approvals", "component": "ReportingApprovalConsole", "permission": "intel_reporting:approvals", "nav_group": "Governance"},
	{"name": "distributions", "path": "/intel-reporting/distributions", "component": "ReportingDistributionConsole", "permission": "intel_reporting:distributions", "nav_group": "Dissemination"},
	{"name": "publications", "path": "/intel-reporting/publications", "component": "ReportingPublicationConsole", "permission": "intel_reporting:publications", "nav_group": "Dissemination"},
	{"name": "reviews", "path": "/intel-reporting/reviews", "component": "ReportingReviewConsole", "permission": "intel_reporting:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-reporting/agents", "component": "ReportingAgentWorkbench", "permission": "intel_reporting:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-reporting/settings", "component": "ReportingSettings", "permission": "intel_reporting:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_reporting_control",
	"tokens": {"color.primary": "#1D4ED8", "color.accent": "#0F766E", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "workspaces": {"icon": "layout-dashboard", "status_indicator": "workspace-chip"}, "templates": {"icon": "file-cog", "status_indicator": "template-chip"}, "products": {"icon": "file-text", "status_indicator": "product-chip"}, "sections": {"icon": "pilcrow", "status_indicator": "section-chip"}, "citations": {"icon": "quote", "status_indicator": "citation-chip"}, "approvals": {"icon": "badge-check", "status_indicator": "approval-chip"}, "distributions": {"icon": "send", "status_indicator": "distribution-chip"}, "publications": {"icon": "upload-cloud", "status_indicator": "publication-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": REPORTING_EVENT_STREAM, "key": "tenant_id", "events": ["reporting_authority_recorded", "reporting_workspace_recorded", "reporting_template_recorded", "reporting_product_recorded", "reporting_section_recorded", "reporting_citation_recorded", "reporting_approval_recorded", "reporting_distribution_recorded", "reporting_publication_recorded", "reporting_review_recorded", "reporting_agent_registered"], "guardrails": ["reporting_batch_requires_bytewax", "privileged_reporting_agent_action_requires_human_approval", "uncited_claim_action_denied", "classification_downgrade_action_denied", "source_fabrication_action_denied", "privacy_bypass_action_denied", "autonomous_publication_action_denied", "unapproved_distribution_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "reporting_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "reporting_policy_required", "required_action": "attach_reporting_policy"}},
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
	{"name": "template_workspace_required", "condition": {"operation": "record_template", "workspace_present": False}, "effect": {"decision": "deny", "reason": "workspace_required", "required_action": "select_workspace"}},
	{"name": "template_type_supported", "condition": {"operation": "record_template", "template_type_supported": False}, "effect": {"decision": "deny", "reason": "template_type_not_supported", "required_action": "select_supported_template_type"}},
	{"name": "template_reference_required", "condition": {"operation": "record_template", "template_reference_present": False}, "effect": {"decision": "deny", "reason": "template_reference_required", "required_action": "attach_template_reference"}},
	{"name": "template_classification_supported", "condition": {"operation": "record_template", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "template_evidence_required", "condition": {"operation": "record_template", "evidence_present": False}, "effect": {"decision": "deny", "reason": "template_evidence_required", "required_action": "attach_template_evidence"}},
	{"name": "product_template_required", "condition": {"operation": "record_product", "template_present": False}, "effect": {"decision": "deny", "reason": "template_required", "required_action": "select_template"}},
	{"name": "product_type_supported", "condition": {"operation": "record_product", "product_type_supported": False}, "effect": {"decision": "deny", "reason": "product_type_not_supported", "required_action": "select_supported_product_type"}},
	{"name": "product_title_required", "condition": {"operation": "record_product", "title_present": False}, "effect": {"decision": "deny", "reason": "product_title_required", "required_action": "title_product"}},
	{"name": "product_author_required", "condition": {"operation": "record_product", "author_present": False}, "effect": {"decision": "deny", "reason": "product_author_required", "required_action": "assign_author"}},
	{"name": "product_classification_supported", "condition": {"operation": "record_product", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "product_evidence_required", "condition": {"operation": "record_product", "evidence_present": False}, "effect": {"decision": "deny", "reason": "product_evidence_required", "required_action": "attach_product_evidence"}},
	{"name": "section_product_required", "condition": {"operation": "record_section", "product_present": False}, "effect": {"decision": "deny", "reason": "product_required", "required_action": "select_product"}},
	{"name": "section_type_supported", "condition": {"operation": "record_section", "section_type_supported": False}, "effect": {"decision": "deny", "reason": "section_type_not_supported", "required_action": "select_supported_section_type"}},
	{"name": "section_reference_required", "condition": {"operation": "record_section", "section_reference_present": False}, "effect": {"decision": "deny", "reason": "section_reference_required", "required_action": "attach_section_reference"}},
	{"name": "section_confidence_valid", "condition": {"operation": "record_section", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "section_evidence_required", "condition": {"operation": "record_section", "evidence_present": False}, "effect": {"decision": "deny", "reason": "section_evidence_required", "required_action": "attach_section_evidence"}},
	{"name": "citation_section_required", "condition": {"operation": "record_citation", "section_present": False}, "effect": {"decision": "deny", "reason": "section_required", "required_action": "select_section"}},
	{"name": "citation_type_supported", "condition": {"operation": "record_citation", "citation_type_supported": False}, "effect": {"decision": "deny", "reason": "citation_type_not_supported", "required_action": "select_supported_citation_type"}},
	{"name": "citation_source_required", "condition": {"operation": "record_citation", "source_present": False}, "effect": {"decision": "deny", "reason": "citation_source_required", "required_action": "attach_source_reference"}},
	{"name": "citation_evidence_required", "condition": {"operation": "record_citation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "citation_evidence_required", "required_action": "attach_citation_evidence"}},
	{"name": "approval_product_required", "condition": {"operation": "record_approval", "product_present": False}, "effect": {"decision": "deny", "reason": "product_required", "required_action": "select_product"}},
	{"name": "approval_type_supported", "condition": {"operation": "record_approval", "approval_type_supported": False}, "effect": {"decision": "deny", "reason": "approval_type_not_supported", "required_action": "select_supported_approval_type"}},
	{"name": "approval_approver_required", "condition": {"operation": "record_approval", "approver_present": False}, "effect": {"decision": "deny", "reason": "approval_approver_required", "required_action": "assign_approver"}},
	{"name": "approval_status_supported", "condition": {"operation": "record_approval", "status_supported": False}, "effect": {"decision": "deny", "reason": "approval_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "approval_evidence_required", "condition": {"operation": "record_approval", "evidence_present": False}, "effect": {"decision": "deny", "reason": "approval_evidence_required", "required_action": "attach_approval_evidence"}},
	{"name": "distribution_product_required", "condition": {"operation": "record_distribution", "product_present": False}, "effect": {"decision": "deny", "reason": "product_required", "required_action": "select_product"}},
	{"name": "distribution_type_supported", "condition": {"operation": "record_distribution", "distribution_type_supported": False}, "effect": {"decision": "deny", "reason": "distribution_type_not_supported", "required_action": "select_supported_distribution_type"}},
	{"name": "distribution_recipient_required", "condition": {"operation": "record_distribution", "recipient_present": False}, "effect": {"decision": "deny", "reason": "recipient_reference_required", "required_action": "attach_recipient_reference"}},
	{"name": "distribution_approval_required", "condition": {"operation": "record_distribution", "approval_present": False}, "effect": {"decision": "deny", "reason": "distribution_approval_required", "required_action": "attach_distribution_approval"}},
	{"name": "distribution_evidence_required", "condition": {"operation": "record_distribution", "evidence_present": False}, "effect": {"decision": "deny", "reason": "distribution_evidence_required", "required_action": "attach_distribution_evidence"}},
	{"name": "publication_distribution_required", "condition": {"operation": "record_publication", "distribution_present": False}, "effect": {"decision": "deny", "reason": "distribution_required", "required_action": "select_distribution"}},
	{"name": "publication_type_supported", "condition": {"operation": "record_publication", "publication_type_supported": False}, "effect": {"decision": "deny", "reason": "publication_type_not_supported", "required_action": "select_supported_publication_type"}},
	{"name": "publication_reference_required", "condition": {"operation": "record_publication", "publication_reference_present": False}, "effect": {"decision": "deny", "reason": "publication_reference_required", "required_action": "attach_publication_reference"}},
	{"name": "publication_approval_required", "condition": {"operation": "record_publication", "approval_present": False}, "effect": {"decision": "deny", "reason": "publication_approval_required", "required_action": "attach_publication_approval"}},
	{"name": "publication_evidence_required", "condition": {"operation": "record_publication", "evidence_present": False}, "effect": {"decision": "deny", "reason": "publication_evidence_required", "required_action": "attach_publication_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "reporting_batch_requires_bytewax", "condition": {"operation": "reporting_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_reporting_batch_to_bytewax"}},
	{"name": "reporting_agent_runtime_supported", "condition": {"operation": "register_reporting_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "reporting_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "reporting_agent_role_supported", "condition": {"operation": "register_reporting_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "reporting_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "reporting_agent_name_required", "condition": {"operation": "register_reporting_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "reporting_agent_name_required", "required_action": "name_reporting_agent"}},
	{"name": "reporting_agent_scope_required", "condition": {"operation": "register_reporting_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "reporting_agent_scope_required", "required_action": "bound_reporting_agent_scope"}},
	{"name": "privileged_reporting_agent_action_requires_human_approval", "condition": {"operation": "reporting_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "uncited_claim_action_denied", "condition": {"operation": "reporting_agent_action", "uncited_claim_scope": True}, "effect": {"decision": "deny", "reason": "uncited_claim_scope_denied", "required_action": "remove_uncited_claim_scope"}},
	{"name": "classification_downgrade_action_denied", "condition": {"operation": "reporting_agent_action", "classification_downgrade_scope": True}, "effect": {"decision": "deny", "reason": "classification_downgrade_scope_denied", "required_action": "remove_classification_downgrade_scope"}},
	{"name": "source_fabrication_action_denied", "condition": {"operation": "reporting_agent_action", "source_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "source_fabrication_scope_denied", "required_action": "remove_source_fabrication_scope"}},
	{"name": "privacy_bypass_action_denied", "condition": {"operation": "reporting_agent_action", "privacy_bypass_scope": True}, "effect": {"decision": "deny", "reason": "privacy_bypass_scope_denied", "required_action": "remove_privacy_bypass_scope"}},
	{"name": "autonomous_publication_action_denied", "condition": {"operation": "reporting_agent_action", "autonomous_publication_scope": True}, "effect": {"decision": "deny", "reason": "autonomous_publication_scope_denied", "required_action": "remove_autonomous_publication_scope"}},
	{"name": "unapproved_distribution_action_denied", "condition": {"operation": "reporting_agent_action", "unapproved_distribution_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_distribution_scope_denied", "required_action": "remove_unapproved_distribution_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-reporting/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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

