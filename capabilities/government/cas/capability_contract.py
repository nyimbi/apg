"""Executable capability contract for APG Case Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "government_cas"
CAPABILITY_NAME = "Case Management"
CAPABILITY_VERSION = "1.0.0"
CAS_EVENT_STREAM = "apg.government.cas.lifecycle"

SUPPORTED_CASE_TYPES = ["complaint", "enquiry", "application", "investigation", "appeal", "feedback", "service_request", "regulatory_referral"]
SUPPORTED_INTAKE_CHANNELS = ["online_portal", "walk_in", "telephone", "email", "postal", "mobile_app", "third_party_referral"]
SUPPORTED_PRIORITY_LEVELS = ["low", "medium", "high", "urgent", "critical"]
SUPPORTED_STATUSES = ["open", "assigned", "in_progress", "pending_info", "escalated", "resolved", "closed", "withdrawn", "reopened"]
SUPPORTED_ASSIGNMENT_TYPES = ["officer", "team", "supervisor", "specialist", "external_agency", "inter_department"]
SUPPORTED_ESCALATION_REASONS = ["sla_breach", "complexity", "political_sensitivity", "legal_risk", "citizen_request", "supervisor_directive"]
SUPPORTED_OUTCOME_TYPES = ["resolved_satisfied", "resolved_unsatisfied", "referred", "withdrawn", "no_further_action", "prosecution_initiated"]
SUPPORTED_SLA_CATEGORIES = ["standard", "urgent", "statutory", "ministerial", "court_ordered"]
SUPPORTED_NOTIFICATION_TYPES = ["sms", "email", "in_app", "postal", "webhook"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["case_router", "sla_monitor", "outcome_recorder", "escalation_reviewer", "notification_sender"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"cases": {
		"supported_case_types": SUPPORTED_CASE_TYPES,
		"supported_intake_channels": SUPPORTED_INTAKE_CHANNELS,
		"supported_priority_levels": SUPPORTED_PRIORITY_LEVELS,
		"citizen_id_required": True,
		"intake_channel_required": True,
		"evidence_required": True,
	},
	"assignments": {
		"supported_assignment_types": SUPPORTED_ASSIGNMENT_TYPES,
		"case_required": True,
		"assignee_required": True,
		"evidence_required": True,
	},
	"escalations": {
		"supported_escalation_reasons": SUPPORTED_ESCALATION_REASONS,
		"case_required": True,
		"reason_required": True,
		"supervisor_required": True,
		"evidence_required": True,
	},
	"outcomes": {
		"supported_outcome_types": SUPPORTED_OUTCOME_TYPES,
		"case_required": True,
		"approval_required": True,
		"evidence_required": True,
	},
	"sla": {
		"supported_sla_categories": SUPPORTED_SLA_CATEGORIES,
		"breach_alerts_enabled": True,
		"auto_escalate_on_breach": True,
	},
	"notifications": {
		"supported_notification_types": SUPPORTED_NOTIFICATION_TYPES,
		"case_required": True,
		"recipient_required": True,
	},
	"reviews": {
		"supported_statuses": SUPPORTED_REVIEW_STATUSES,
		"reviewer_required": True,
		"evidence_required": True,
	},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"name_required": True,
		"scope_required": True,
		"human_approval_required_for_privileged_actions": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"unassigned_case_auto_escalation": True,
		"sla_breach_triggers_escalation": True,
		"outcome_without_approval_denied": True,
		"cross_tenant_case_denied": True,
		"citizen_data_privacy_enforced": True,
		"evidence_fabrication_denied": True,
	},
	"observability": {"event_stream": CAS_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"workflow": "wflo",
		"nlp": "nlpc",
		"search": "srch",
		"monitoring": "moni",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_cases": True,
		"enable_assignments": True,
		"enable_escalations": True,
		"enable_outcomes": True,
		"enable_sla": True,
		"enable_notifications": True,
		"enable_reviews": True,
		"enable_agents": True,
	},
	"theme": {"default_theme": "government_cas_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"case_intake_workflow",
	"case_assignment_workflow",
	"case_routing_workflow",
	"sla_tracking_workflow",
	"case_escalation_workflow",
	"case_outcome_workflow",
	"case_notification_workflow",
	"case_review_workflow",
	"case_agent_workflow",
	"citizen_case_portal_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "nlpc", "srch", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/government-cas/dashboard", "component": "CaseDashboard", "permission": "government_cas:view", "nav_group": "Overview"},
	{"name": "intake", "path": "/government-cas/intake", "component": "CaseIntakeConsole", "permission": "government_cas:create", "nav_group": "Intake"},
	{"name": "cases", "path": "/government-cas/cases", "component": "CaseQueue", "permission": "government_cas:cases", "nav_group": "Cases"},
	{"name": "assignments", "path": "/government-cas/assignments", "component": "CaseAssignmentConsole", "permission": "government_cas:assign", "nav_group": "Operations"},
	{"name": "escalations", "path": "/government-cas/escalations", "component": "CaseEscalationConsole", "permission": "government_cas:escalate", "nav_group": "Operations"},
	{"name": "sla", "path": "/government-cas/sla", "component": "SlaTrackingDashboard", "permission": "government_cas:sla", "nav_group": "Monitoring"},
	{"name": "outcomes", "path": "/government-cas/outcomes", "component": "CaseOutcomeConsole", "permission": "government_cas:outcomes", "nav_group": "Resolution"},
	{"name": "notifications", "path": "/government-cas/notifications", "component": "CaseNotificationConsole", "permission": "government_cas:notify", "nav_group": "Communications"},
	{"name": "reviews", "path": "/government-cas/reviews", "component": "CaseReviewConsole", "permission": "government_cas:review", "nav_group": "Governance"},
	{"name": "search", "path": "/government-cas/search", "component": "CaseSearchConsole", "permission": "government_cas:view", "nav_group": "Search"},
	{"name": "agents", "path": "/government-cas/agents", "component": "CaseAgentWorkbench", "permission": "government_cas:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/government-cas/settings", "component": "CaseSettings", "permission": "government_cas:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "government_cas_control",
	"tokens": {
		"color.primary": "#0369A1",
		"color.accent": "#7C3AED",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#991B1B",
		"surface.canvas": "#F0F9FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0C1A2E",
		"text.secondary": "#475569",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"cases": {"icon": "folder-open", "status_indicator": "case-status-chip"},
		"assignments": {"icon": "user-check", "status_indicator": "assignment-chip"},
		"escalations": {"icon": "arrow-up-right", "status_indicator": "escalation-chip"},
		"sla": {"icon": "clock", "status_indicator": "sla-health-chip"},
		"outcomes": {"icon": "check-square", "status_indicator": "outcome-type-chip"},
		"notifications": {"icon": "send", "status_indicator": "notification-chip"},
		"reviews": {"icon": "clipboard-check", "status_indicator": "review-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": CAS_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"case_opened",
		"case_assigned",
		"case_escalated",
		"case_sla_breached",
		"case_outcome_recorded",
		"case_closed",
		"case_notification_sent",
		"case_review_recorded",
		"case_agent_registered",
		"case_reopened",
	],
	"guardrails": [
		"case_batch_requires_bytewax",
		"outcome_without_approval_denied",
		"cross_tenant_case_denied",
		"citizen_data_privacy_enforced",
		"evidence_fabrication_denied",
		"privileged_case_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "case_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "case_policy_required", "required_action": "attach_case_policy"}},
	{"name": "case_type_supported", "condition": {"operation": "open_case", "case_type_supported": False}, "effect": {"decision": "deny", "reason": "case_type_not_supported", "required_action": "select_supported_case_type"}},
	{"name": "case_intake_channel_supported", "condition": {"operation": "open_case", "intake_channel_supported": False}, "effect": {"decision": "deny", "reason": "intake_channel_not_supported", "required_action": "select_supported_intake_channel"}},
	{"name": "case_citizen_id_required", "condition": {"operation": "open_case", "citizen_id_present": False}, "effect": {"decision": "deny", "reason": "citizen_id_required", "required_action": "provide_citizen_id"}},
	{"name": "case_priority_supported", "condition": {"operation": "open_case", "priority_supported": False}, "effect": {"decision": "deny", "reason": "priority_not_supported", "required_action": "select_supported_priority"}},
	{"name": "case_evidence_required", "condition": {"operation": "open_case", "evidence_present": False}, "effect": {"decision": "deny", "reason": "case_evidence_required", "required_action": "attach_case_evidence"}},
	{"name": "assignment_case_required", "condition": {"operation": "assign_case", "case_present": False}, "effect": {"decision": "deny", "reason": "case_required", "required_action": "select_case"}},
	{"name": "assignment_type_supported", "condition": {"operation": "assign_case", "assignment_type_supported": False}, "effect": {"decision": "deny", "reason": "assignment_type_not_supported", "required_action": "select_supported_assignment_type"}},
	{"name": "assignment_assignee_required", "condition": {"operation": "assign_case", "assignee_present": False}, "effect": {"decision": "deny", "reason": "assignee_required", "required_action": "select_assignee"}},
	{"name": "assignment_evidence_required", "condition": {"operation": "assign_case", "evidence_present": False}, "effect": {"decision": "deny", "reason": "assignment_evidence_required", "required_action": "attach_assignment_evidence"}},
	{"name": "escalation_case_required", "condition": {"operation": "escalate_case", "case_present": False}, "effect": {"decision": "deny", "reason": "case_required", "required_action": "select_case"}},
	{"name": "escalation_reason_supported", "condition": {"operation": "escalate_case", "escalation_reason_supported": False}, "effect": {"decision": "deny", "reason": "escalation_reason_not_supported", "required_action": "select_supported_escalation_reason"}},
	{"name": "escalation_supervisor_required", "condition": {"operation": "escalate_case", "supervisor_present": False}, "effect": {"decision": "deny", "reason": "supervisor_required", "required_action": "assign_supervisor"}},
	{"name": "escalation_evidence_required", "condition": {"operation": "escalate_case", "evidence_present": False}, "effect": {"decision": "deny", "reason": "escalation_evidence_required", "required_action": "attach_escalation_evidence"}},
	{"name": "outcome_case_required", "condition": {"operation": "record_outcome", "case_present": False}, "effect": {"decision": "deny", "reason": "case_required", "required_action": "select_case"}},
	{"name": "outcome_type_supported", "condition": {"operation": "record_outcome", "outcome_type_supported": False}, "effect": {"decision": "deny", "reason": "outcome_type_not_supported", "required_action": "select_supported_outcome_type"}},
	{"name": "outcome_approval_required", "condition": {"operation": "record_outcome", "approval_present": False}, "effect": {"decision": "deny", "reason": "outcome_approval_required", "required_action": "attach_outcome_approval"}},
	{"name": "outcome_evidence_required", "condition": {"operation": "record_outcome", "evidence_present": False}, "effect": {"decision": "deny", "reason": "outcome_evidence_required", "required_action": "attach_outcome_evidence"}},
	{"name": "cross_tenant_case_denied", "condition": {"operation": "open_case", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_case_denied", "required_action": "use_tenant_scoped_case"}},
	{"name": "sla_category_supported", "condition": {"operation": "set_sla", "sla_category_supported": False}, "effect": {"decision": "deny", "reason": "sla_category_not_supported", "required_action": "select_supported_sla_category"}},
	{"name": "notification_type_supported", "condition": {"operation": "send_notification", "notification_type_supported": False}, "effect": {"decision": "deny", "reason": "notification_type_not_supported", "required_action": "select_supported_notification_type"}},
	{"name": "notification_recipient_required", "condition": {"operation": "send_notification", "recipient_present": False}, "effect": {"decision": "deny", "reason": "recipient_required", "required_action": "provide_recipient"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "case_batch_requires_bytewax", "condition": {"operation": "case_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_case_batch_to_bytewax"}},
	{"name": "case_agent_runtime_supported", "condition": {"operation": "register_case_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "case_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "case_agent_role_supported", "condition": {"operation": "register_case_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "case_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "case_agent_name_required", "condition": {"operation": "register_case_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "case_agent_name_required", "required_action": "name_case_agent"}},
	{"name": "case_agent_scope_required", "condition": {"operation": "register_case_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "case_agent_scope_required", "required_action": "bound_case_agent_scope"}},
	{"name": "privileged_case_agent_action_requires_human_approval", "condition": {"operation": "case_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "evidence_fabrication_denied", "condition": {"operation": "case_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_denied", "required_action": "remove_evidence_fabrication_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"name": CAPABILITY_NAME,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": list(configuration),
			"properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/government-cas/api/v1",
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
			"routes": deepcopy(UI_ROUTES),
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


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
