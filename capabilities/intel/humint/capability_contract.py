"""Executable capability contract for APG Human Intelligence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_humint"
CAPABILITY_NAME = "Human Intelligence"
CAPABILITY_VERSION = "1.1.0"
HUMINT_EVENT_STREAM = "apg.intel.humint.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "consent", "partner_authority", "legal_mandate", "oversight_authorization"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_SOURCE_TYPES = ["voluntary_source", "confidential_human_source", "partner_liaison", "field_interview", "community_contact", "internal_report", "diplomatic_contact"]
SUPPORTED_HANDLING_STATUSES = ["prospective", "active", "paused", "closed", "protected"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_CONTACT_METHODS = ["in_person", "secure_call", "secure_message", "partner_channel", "interview", "debrief"]
SUPPORTED_RELIABILITY_GRADES = ["a", "b", "c", "d", "e", "f"]
SUPPORTED_LEAD_TYPES = ["identity", "location", "network", "event", "threat", "financial", "protective"]
SUPPORTED_PRIORITIES = ["low", "medium", "high", "urgent"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["authority_reviewer", "source_manager", "contact_planner", "debriefing_analyst", "welfare_reviewer", "dissemination_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"sources": {"supported_source_types": SUPPORTED_SOURCE_TYPES, "supported_statuses": SUPPORTED_HANDLING_STATUSES, "supported_risk_levels": SUPPORTED_RISK_LEVELS, "owner_required": True, "authority_required": True, "protection_plan_required": True, "evidence_required": True},
	"contact_plans": {"supported_methods": SUPPORTED_CONTACT_METHODS, "authority_required": True, "source_required": True, "objective_required": True, "safety_plan_required": True, "approval_required": True, "evidence_required": True},
	"contact_reports": {"plan_required": True, "handler_required": True, "source_welfare_required": True, "evidence_required": True},
	"debriefings": {"supported_classifications": SUPPORTED_CLASSIFICATIONS, "topic_required": True, "credibility_required": True, "analyst_required": True, "evidence_required": True},
	"reliability": {"supported_grades": SUPPORTED_RELIABILITY_GRADES, "source_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"leads": {"supported_types": SUPPORTED_LEAD_TYPES, "supported_priorities": SUPPORTED_PRIORITIES, "debriefing_required": True, "analyst_required": True, "evidence_required": True},
	"dissemination": {"lead_required": True, "audience_required": True, "release_marking_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "source_welfare_required": True, "coercion_prohibited": True},
	"observability": {"event_stream": HUMINT_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "case_management": "case", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_sources": True, "enable_contact_plans": True, "enable_contact_reports": True, "enable_debriefings": True, "enable_reliability": True, "enable_leads": True, "enable_dissemination": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_humint_control", "allow_tenant_overrides": True},
}

PROVIDES = ["humint_authority_workflow", "humint_source_workflow", "humint_contact_plan_workflow", "humint_contact_report_workflow", "humint_debriefing_workflow", "humint_reliability_workflow", "humint_lead_workflow", "humint_dissemination_workflow", "humint_review_workflow", "humint_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-humint/dashboard", "component": "HUMINTDashboard", "permission": "intel_humint:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-humint/authorities", "component": "SourceAuthorityConsole", "permission": "intel_humint:authorities", "nav_group": "Governance"},
	{"name": "sources", "path": "/intel-humint/sources", "component": "HumanSourceRegistry", "permission": "intel_humint:sources", "nav_group": "Source Management"},
	{"name": "contact_plans", "path": "/intel-humint/contact-plans", "component": "ContactPlanWorkbench", "permission": "intel_humint:contacts", "nav_group": "Operations"},
	{"name": "contact_reports", "path": "/intel-humint/contact-reports", "component": "ContactReportLedger", "permission": "intel_humint:reports", "nav_group": "Operations"},
	{"name": "debriefings", "path": "/intel-humint/debriefings", "component": "DebriefingWorkbench", "permission": "intel_humint:analysis", "nav_group": "Analysis"},
	{"name": "reliability", "path": "/intel-humint/reliability", "component": "ReliabilityAssessmentWorkbench", "permission": "intel_humint:analysis", "nav_group": "Analysis"},
	{"name": "leads", "path": "/intel-humint/leads", "component": "HUMINTLeadWorkbench", "permission": "intel_humint:leads", "nav_group": "Analysis"},
	{"name": "dissemination", "path": "/intel-humint/dissemination", "component": "HUMINTDisseminationConsole", "permission": "intel_humint:dissemination", "nav_group": "Release"},
	{"name": "reviews", "path": "/intel-humint/reviews", "component": "HUMINTReviewConsole", "permission": "intel_humint:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-humint/agents", "component": "HUMINTAgentWorkbench", "permission": "intel_humint:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-humint/settings", "component": "HUMINTSettings", "permission": "intel_humint:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_humint_control",
	"tokens": {"color.primary": "#244E3B", "color.accent": "#7C2D12", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "sources": {"icon": "user-round-check", "status_indicator": "source-status-chip"}, "contact_plans": {"icon": "calendar-check", "status_indicator": "safety-chip"}, "contact_reports": {"icon": "clipboard-list", "status_indicator": "welfare-chip"}, "debriefings": {"icon": "messages-square", "status_indicator": "credibility-chip"}, "reliability": {"icon": "badge-check", "status_indicator": "grade-chip"}, "leads": {"icon": "network", "status_indicator": "priority-chip"}, "dissemination": {"icon": "send", "status_indicator": "release-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": HUMINT_EVENT_STREAM, "key": "tenant_id", "events": ["humint_authority_recorded", "humint_source_registered", "humint_contact_plan_recorded", "humint_contact_report_recorded", "humint_debriefing_recorded", "humint_reliability_recorded", "humint_lead_recorded", "humint_dissemination_recorded", "humint_review_recorded", "humint_agent_registered"], "guardrails": ["humint_batch_requires_bytewax", "privileged_humint_agent_action_requires_human_approval", "coercive_humint_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "humint_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "humint_policy_required", "required_action": "attach_humint_policy"}},
	{"name": "authority_type_supported", "condition": {"operation": "record_authority", "authority_type_supported": False}, "effect": {"decision": "deny", "reason": "authority_type_not_supported", "required_action": "select_supported_authority_type"}},
	{"name": "authority_scope_required", "condition": {"operation": "record_authority", "scope_present": False}, "effect": {"decision": "deny", "reason": "authority_scope_required", "required_action": "attach_scope_reference"}},
	{"name": "authority_classification_supported", "condition": {"operation": "record_authority", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "authority_approver_required", "condition": {"operation": "record_authority", "approver_present": False}, "effect": {"decision": "deny", "reason": "authority_approver_required", "required_action": "record_approver"}},
	{"name": "authority_expiry_required", "condition": {"operation": "record_authority", "expiry_present": False}, "effect": {"decision": "deny", "reason": "authority_expiry_required", "required_action": "set_expiry"}},
	{"name": "authority_evidence_required", "condition": {"operation": "record_authority", "evidence_present": False}, "effect": {"decision": "deny", "reason": "authority_evidence_required", "required_action": "attach_authority_evidence"}},
	{"name": "source_type_supported", "condition": {"operation": "register_source", "source_type_supported": False}, "effect": {"decision": "deny", "reason": "source_type_not_supported", "required_action": "select_supported_source_type"}},
	{"name": "source_status_supported", "condition": {"operation": "register_source", "handling_status_supported": False}, "effect": {"decision": "deny", "reason": "handling_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "source_risk_supported", "condition": {"operation": "register_source", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk"}},
	{"name": "source_owner_required", "condition": {"operation": "register_source", "owner_present": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_authority_required", "condition": {"operation": "register_source", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "source_protection_required", "condition": {"operation": "register_source", "protection_present": False}, "effect": {"decision": "deny", "reason": "source_protection_required", "required_action": "attach_protection_reference"}},
	{"name": "source_evidence_required", "condition": {"operation": "register_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "contact_authority_required", "condition": {"operation": "record_contact_plan", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "contact_source_required", "condition": {"operation": "record_contact_plan", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "contact_source_authority_match", "condition": {"operation": "record_contact_plan", "source_authority_match": False}, "effect": {"decision": "deny", "reason": "source_authority_mismatch", "required_action": "select_source_for_authority"}},
	{"name": "contact_method_supported", "condition": {"operation": "record_contact_plan", "contact_method_supported": False}, "effect": {"decision": "deny", "reason": "contact_method_not_supported", "required_action": "select_supported_contact_method"}},
	{"name": "contact_objective_required", "condition": {"operation": "record_contact_plan", "objective_present": False}, "effect": {"decision": "deny", "reason": "contact_objective_required", "required_action": "attach_objective_reference"}},
	{"name": "contact_safety_plan_required", "condition": {"operation": "record_contact_plan", "safety_plan_present": False}, "effect": {"decision": "deny", "reason": "safety_plan_required", "required_action": "attach_safety_plan"}},
	{"name": "contact_approval_required", "condition": {"operation": "record_contact_plan", "approval_present": False}, "effect": {"decision": "deny", "reason": "contact_approval_required", "required_action": "attach_contact_approval"}},
	{"name": "contact_evidence_required", "condition": {"operation": "record_contact_plan", "evidence_present": False}, "effect": {"decision": "deny", "reason": "contact_plan_evidence_required", "required_action": "attach_contact_plan_evidence"}},
	{"name": "coercive_humint_action_denied", "condition": {"operation": "humint_agent_action", "coercive_scope": True}, "effect": {"decision": "deny", "reason": "coercive_humint_action_denied", "required_action": "remove_coercive_scope"}},
	{"name": "report_plan_required", "condition": {"operation": "record_contact_report", "plan_present": False}, "effect": {"decision": "deny", "reason": "contact_plan_required", "required_action": "select_contact_plan"}},
	{"name": "report_reference_required", "condition": {"operation": "record_contact_report", "report_reference_present": False}, "effect": {"decision": "deny", "reason": "contact_report_reference_required", "required_action": "attach_report_reference"}},
	{"name": "report_handler_required", "condition": {"operation": "record_contact_report", "handler_present": False}, "effect": {"decision": "deny", "reason": "handler_required", "required_action": "assign_handler"}},
	{"name": "report_welfare_valid", "condition": {"operation": "record_contact_report", "source_welfare_valid": False}, "effect": {"decision": "deny", "reason": "source_welfare_score_invalid", "required_action": "set_welfare_0_to_1"}},
	{"name": "report_evidence_required", "condition": {"operation": "record_contact_report", "evidence_present": False}, "effect": {"decision": "deny", "reason": "contact_report_evidence_required", "required_action": "attach_report_evidence"}},
	{"name": "debriefing_report_required", "condition": {"operation": "record_debriefing", "report_present": False}, "effect": {"decision": "deny", "reason": "contact_report_required", "required_action": "select_contact_report"}},
	{"name": "debriefing_topic_required", "condition": {"operation": "record_debriefing", "topic_present": False}, "effect": {"decision": "deny", "reason": "debriefing_topic_required", "required_action": "record_topic"}},
	{"name": "debriefing_classification_supported", "condition": {"operation": "record_debriefing", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "debriefing_credibility_valid", "condition": {"operation": "record_debriefing", "credibility_valid": False}, "effect": {"decision": "deny", "reason": "credibility_score_invalid", "required_action": "set_credibility_0_to_1"}},
	{"name": "debriefing_analyst_required", "condition": {"operation": "record_debriefing", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "debriefing_evidence_required", "condition": {"operation": "record_debriefing", "evidence_present": False}, "effect": {"decision": "deny", "reason": "debriefing_evidence_required", "required_action": "attach_debriefing_evidence"}},
	{"name": "reliability_source_required", "condition": {"operation": "record_reliability", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "reliability_grade_supported", "condition": {"operation": "record_reliability", "reliability_grade_supported": False}, "effect": {"decision": "deny", "reason": "reliability_grade_not_supported", "required_action": "select_supported_grade"}},
	{"name": "reliability_confidence_valid", "condition": {"operation": "record_reliability", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "reliability_analyst_required", "condition": {"operation": "record_reliability", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "reliability_evidence_required", "condition": {"operation": "record_reliability", "evidence_present": False}, "effect": {"decision": "deny", "reason": "reliability_evidence_required", "required_action": "attach_reliability_evidence"}},
	{"name": "lead_debriefing_required", "condition": {"operation": "record_lead", "debriefing_present": False}, "effect": {"decision": "deny", "reason": "debriefing_required", "required_action": "select_debriefing"}},
	{"name": "lead_type_supported", "condition": {"operation": "record_lead", "lead_type_supported": False}, "effect": {"decision": "deny", "reason": "lead_type_not_supported", "required_action": "select_supported_lead_type"}},
	{"name": "lead_priority_supported", "condition": {"operation": "record_lead", "priority_supported": False}, "effect": {"decision": "deny", "reason": "priority_not_supported", "required_action": "select_supported_priority"}},
	{"name": "lead_analyst_required", "condition": {"operation": "record_lead", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "lead_evidence_required", "condition": {"operation": "record_lead", "evidence_present": False}, "effect": {"decision": "deny", "reason": "lead_evidence_required", "required_action": "attach_lead_evidence"}},
	{"name": "dissemination_lead_required", "condition": {"operation": "record_dissemination", "lead_present": False}, "effect": {"decision": "deny", "reason": "lead_required", "required_action": "select_lead"}},
	{"name": "dissemination_audience_required", "condition": {"operation": "record_dissemination", "audience_present": False}, "effect": {"decision": "deny", "reason": "audience_required", "required_action": "select_audience"}},
	{"name": "dissemination_release_required", "condition": {"operation": "record_dissemination", "release_marking_present": False}, "effect": {"decision": "deny", "reason": "release_marking_required", "required_action": "set_release_marking"}},
	{"name": "dissemination_approval_required", "condition": {"operation": "record_dissemination", "approval_present": False}, "effect": {"decision": "deny", "reason": "dissemination_approval_required", "required_action": "attach_release_approval"}},
	{"name": "dissemination_evidence_required", "condition": {"operation": "record_dissemination", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dissemination_evidence_required", "required_action": "attach_dissemination_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "humint_batch_requires_bytewax", "condition": {"operation": "humint_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_humint_batch_to_bytewax"}},
	{"name": "humint_agent_runtime_supported", "condition": {"operation": "register_humint_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "humint_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "humint_agent_role_supported", "condition": {"operation": "register_humint_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "humint_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_humint_agent_action_requires_human_approval", "condition": {"operation": "humint_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-humint/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
