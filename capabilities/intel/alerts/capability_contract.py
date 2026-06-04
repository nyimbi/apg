"""Executable capability contract for APG Alert Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_alerts"
CAPABILITY_NAME = "Alert Management"
CAPABILITY_VERSION = "1.1.0"
ALERT_EVENT_STREAM = "apg.intel.alerts.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "legal_mandate", "partner_authority", "consent", "incident_response_authority", "public_interest_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_WORKSPACE_TYPES = ["watch_center", "threat_alerting", "incident_alerting", "fraud_alerting", "public_safety_alerting", "executive_alerting", "partner_alerting"]
SUPPORTED_RULE_TYPES = ["threshold", "anomaly", "watchlist", "correlation", "prediction", "geofence", "case_trigger", "manual"]
SUPPORTED_SIGNAL_TYPES = ["metric", "indicator", "event", "forecast", "threat", "case_update", "geospatial", "partner_notice"]
SUPPORTED_ALERT_TYPES = ["early_warning", "critical_alert", "watchlist_hit", "incident_alert", "fraud_alert", "threat_alert", "system_alert"]
SUPPORTED_ESCALATION_TYPES = ["supervisor", "incident_team", "case_team", "executive", "partner", "field_team", "watch_center"]
SUPPORTED_NOTIFICATION_TYPES = ["in_app", "email", "sms", "secure_message", "webhook", "case_note", "briefing_queue"]
SUPPORTED_ASSIGNMENT_TYPES = ["analyst", "supervisor", "incident_commander", "case_owner", "field_team", "partner_owner"]
SUPPORTED_RESOLUTION_TYPES = ["confirmed", "false_positive", "duplicate", "mitigated", "escalated", "closed", "monitoring"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["rule_steward", "signal_triage", "alert_reviewer", "escalation_reviewer", "notification_reviewer", "resolution_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"workspaces": {"supported_workspace_types": SUPPORTED_WORKSPACE_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "authority_required": True, "evidence_required": True},
	"rules": {"supported_rule_types": SUPPORTED_RULE_TYPES, "supported_severities": SUPPORTED_SEVERITIES, "workspace_required": True, "owner_required": True, "evidence_required": True},
	"signals": {"supported_signal_types": SUPPORTED_SIGNAL_TYPES, "rule_required": True, "confidence_required": True, "evidence_required": True},
	"alerts": {"supported_alert_types": SUPPORTED_ALERT_TYPES, "supported_severities": SUPPORTED_SEVERITIES, "signal_required": True, "evidence_required": True},
	"escalations": {"supported_escalation_types": SUPPORTED_ESCALATION_TYPES, "alert_required": True, "target_required": True, "approval_required": True, "evidence_required": True},
	"notifications": {"supported_notification_types": SUPPORTED_NOTIFICATION_TYPES, "alert_required": True, "recipient_required": True, "approval_required": True, "evidence_required": True},
	"assignments": {"supported_assignment_types": SUPPORTED_ASSIGNMENT_TYPES, "alert_required": True, "assignee_required": True, "evidence_required": True},
	"resolutions": {"supported_resolution_types": SUPPORTED_RESOLUTION_TYPES, "alert_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "name_required": True, "scope_required": True, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "cross_tenant_alert_denied": True, "unapproved_escalation_denied": True, "unapproved_notification_denied": True, "alert_suppression_denied": True, "evidence_fabrication_denied": True, "privacy_bypass_denied": True, "autonomous_closure_denied": True, "severity_downgrade_denied": True},
	"observability": {"event_stream": ALERT_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "geospatial": "geos", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_workspaces": True, "enable_rules": True, "enable_signals": True, "enable_alerts": True, "enable_escalations": True, "enable_notifications": True, "enable_assignments": True, "enable_resolutions": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_alerts_control", "allow_tenant_overrides": True},
}

PROVIDES = ["alert_authority_workflow", "alert_workspace_workflow", "alert_rule_workflow", "alert_signal_workflow", "alert_record_workflow", "alert_escalation_workflow", "alert_notification_workflow", "alert_assignment_workflow", "alert_resolution_workflow", "alert_review_workflow", "alert_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-alerts/dashboard", "component": "AlertDashboard", "permission": "intel_alerts:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-alerts/authorities", "component": "AlertAuthorityConsole", "permission": "intel_alerts:authorities", "nav_group": "Governance"},
	{"name": "workspaces", "path": "/intel-alerts/workspaces", "component": "AlertWorkspaceConsole", "permission": "intel_alerts:workspaces", "nav_group": "Planning"},
	{"name": "rules", "path": "/intel-alerts/rules", "component": "AlertRuleWorkbench", "permission": "intel_alerts:rules", "nav_group": "Configuration"},
	{"name": "signals", "path": "/intel-alerts/signals", "component": "AlertSignalLedger", "permission": "intel_alerts:signals", "nav_group": "Signals"},
	{"name": "alerts", "path": "/intel-alerts/alerts", "component": "AlertQueue", "permission": "intel_alerts:alerts", "nav_group": "Operations"},
	{"name": "escalations", "path": "/intel-alerts/escalations", "component": "AlertEscalationConsole", "permission": "intel_alerts:escalations", "nav_group": "Operations"},
	{"name": "notifications", "path": "/intel-alerts/notifications", "component": "AlertNotificationConsole", "permission": "intel_alerts:notifications", "nav_group": "Dissemination"},
	{"name": "assignments", "path": "/intel-alerts/assignments", "component": "AlertAssignmentConsole", "permission": "intel_alerts:assignments", "nav_group": "Operations"},
	{"name": "resolutions", "path": "/intel-alerts/resolutions", "component": "AlertResolutionConsole", "permission": "intel_alerts:resolutions", "nav_group": "Operations"},
	{"name": "reviews", "path": "/intel-alerts/reviews", "component": "AlertReviewConsole", "permission": "intel_alerts:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-alerts/agents", "component": "AlertAgentWorkbench", "permission": "intel_alerts:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-alerts/settings", "component": "AlertSettings", "permission": "intel_alerts:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_alerts_control",
	"tokens": {"color.primary": "#B91C1C", "color.accent": "#0F766E", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#991B1B", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "workspaces": {"icon": "layout-dashboard", "status_indicator": "workspace-chip"}, "rules": {"icon": "list-checks", "status_indicator": "rule-chip"}, "signals": {"icon": "activity", "status_indicator": "signal-chip"}, "alerts": {"icon": "bell-ring", "status_indicator": "severity-chip"}, "escalations": {"icon": "arrow-up-right", "status_indicator": "escalation-chip"}, "notifications": {"icon": "send", "status_indicator": "notification-chip"}, "assignments": {"icon": "user-check", "status_indicator": "assignment-chip"}, "resolutions": {"icon": "check-circle", "status_indicator": "resolution-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": ALERT_EVENT_STREAM, "key": "tenant_id", "events": ["alert_authority_recorded", "alert_workspace_recorded", "alert_rule_recorded", "alert_signal_recorded", "alert_recorded", "alert_escalation_recorded", "alert_notification_recorded", "alert_assignment_recorded", "alert_resolution_recorded", "alert_review_recorded", "alert_agent_registered"], "guardrails": ["alert_batch_requires_bytewax", "privileged_alert_agent_action_requires_human_approval", "unapproved_escalation_action_denied", "unapproved_notification_action_denied", "alert_suppression_action_denied", "evidence_fabrication_action_denied", "privacy_bypass_action_denied", "autonomous_closure_action_denied", "severity_downgrade_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "alert_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "alert_policy_required", "required_action": "attach_alert_policy"}},
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
	{"name": "rule_workspace_required", "condition": {"operation": "record_rule", "workspace_present": False}, "effect": {"decision": "deny", "reason": "workspace_required", "required_action": "select_workspace"}},
	{"name": "rule_type_supported", "condition": {"operation": "record_rule", "rule_type_supported": False}, "effect": {"decision": "deny", "reason": "rule_type_not_supported", "required_action": "select_supported_rule_type"}},
	{"name": "rule_reference_required", "condition": {"operation": "record_rule", "rule_reference_present": False}, "effect": {"decision": "deny", "reason": "rule_reference_required", "required_action": "attach_rule_reference"}},
	{"name": "rule_severity_supported", "condition": {"operation": "record_rule", "severity_supported": False}, "effect": {"decision": "deny", "reason": "severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "rule_owner_required", "condition": {"operation": "record_rule", "owner_present": False}, "effect": {"decision": "deny", "reason": "rule_owner_required", "required_action": "assign_rule_owner"}},
	{"name": "rule_evidence_required", "condition": {"operation": "record_rule", "evidence_present": False}, "effect": {"decision": "deny", "reason": "rule_evidence_required", "required_action": "attach_rule_evidence"}},
	{"name": "signal_rule_required", "condition": {"operation": "record_signal", "rule_present": False}, "effect": {"decision": "deny", "reason": "rule_required", "required_action": "select_rule"}},
	{"name": "signal_type_supported", "condition": {"operation": "record_signal", "signal_type_supported": False}, "effect": {"decision": "deny", "reason": "signal_type_not_supported", "required_action": "select_supported_signal_type"}},
	{"name": "signal_reference_required", "condition": {"operation": "record_signal", "signal_reference_present": False}, "effect": {"decision": "deny", "reason": "signal_reference_required", "required_action": "attach_signal_reference"}},
	{"name": "signal_confidence_valid", "condition": {"operation": "record_signal", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "signal_evidence_required", "condition": {"operation": "record_signal", "evidence_present": False}, "effect": {"decision": "deny", "reason": "signal_evidence_required", "required_action": "attach_signal_evidence"}},
	{"name": "alert_signal_required", "condition": {"operation": "record_alert", "signal_present": False}, "effect": {"decision": "deny", "reason": "signal_required", "required_action": "select_signal"}},
	{"name": "alert_type_supported", "condition": {"operation": "record_alert", "alert_type_supported": False}, "effect": {"decision": "deny", "reason": "alert_type_not_supported", "required_action": "select_supported_alert_type"}},
	{"name": "alert_severity_supported", "condition": {"operation": "record_alert", "severity_supported": False}, "effect": {"decision": "deny", "reason": "severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "alert_reference_required", "condition": {"operation": "record_alert", "alert_reference_present": False}, "effect": {"decision": "deny", "reason": "alert_reference_required", "required_action": "attach_alert_reference"}},
	{"name": "alert_evidence_required", "condition": {"operation": "record_alert", "evidence_present": False}, "effect": {"decision": "deny", "reason": "alert_evidence_required", "required_action": "attach_alert_evidence"}},
	{"name": "escalation_alert_required", "condition": {"operation": "record_escalation", "alert_present": False}, "effect": {"decision": "deny", "reason": "alert_required", "required_action": "select_alert"}},
	{"name": "escalation_type_supported", "condition": {"operation": "record_escalation", "escalation_type_supported": False}, "effect": {"decision": "deny", "reason": "escalation_type_not_supported", "required_action": "select_supported_escalation_type"}},
	{"name": "escalation_target_required", "condition": {"operation": "record_escalation", "target_present": False}, "effect": {"decision": "deny", "reason": "escalation_target_required", "required_action": "attach_escalation_target"}},
	{"name": "escalation_approval_required", "condition": {"operation": "record_escalation", "approval_present": False}, "effect": {"decision": "deny", "reason": "escalation_approval_required", "required_action": "attach_escalation_approval"}},
	{"name": "escalation_evidence_required", "condition": {"operation": "record_escalation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "escalation_evidence_required", "required_action": "attach_escalation_evidence"}},
	{"name": "notification_alert_required", "condition": {"operation": "record_notification", "alert_present": False}, "effect": {"decision": "deny", "reason": "alert_required", "required_action": "select_alert"}},
	{"name": "notification_type_supported", "condition": {"operation": "record_notification", "notification_type_supported": False}, "effect": {"decision": "deny", "reason": "notification_type_not_supported", "required_action": "select_supported_notification_type"}},
	{"name": "notification_recipient_required", "condition": {"operation": "record_notification", "recipient_present": False}, "effect": {"decision": "deny", "reason": "recipient_reference_required", "required_action": "attach_recipient_reference"}},
	{"name": "notification_approval_required", "condition": {"operation": "record_notification", "approval_present": False}, "effect": {"decision": "deny", "reason": "notification_approval_required", "required_action": "attach_notification_approval"}},
	{"name": "notification_evidence_required", "condition": {"operation": "record_notification", "evidence_present": False}, "effect": {"decision": "deny", "reason": "notification_evidence_required", "required_action": "attach_notification_evidence"}},
	{"name": "assignment_alert_required", "condition": {"operation": "record_assignment", "alert_present": False}, "effect": {"decision": "deny", "reason": "alert_required", "required_action": "select_alert"}},
	{"name": "assignment_type_supported", "condition": {"operation": "record_assignment", "assignment_type_supported": False}, "effect": {"decision": "deny", "reason": "assignment_type_not_supported", "required_action": "select_supported_assignment_type"}},
	{"name": "assignment_assignee_required", "condition": {"operation": "record_assignment", "assignee_present": False}, "effect": {"decision": "deny", "reason": "assignee_required", "required_action": "assign_alert_owner"}},
	{"name": "assignment_evidence_required", "condition": {"operation": "record_assignment", "evidence_present": False}, "effect": {"decision": "deny", "reason": "assignment_evidence_required", "required_action": "attach_assignment_evidence"}},
	{"name": "resolution_alert_required", "condition": {"operation": "record_resolution", "alert_present": False}, "effect": {"decision": "deny", "reason": "alert_required", "required_action": "select_alert"}},
	{"name": "resolution_type_supported", "condition": {"operation": "record_resolution", "resolution_type_supported": False}, "effect": {"decision": "deny", "reason": "resolution_type_not_supported", "required_action": "select_supported_resolution_type"}},
	{"name": "resolution_reference_required", "condition": {"operation": "record_resolution", "resolution_reference_present": False}, "effect": {"decision": "deny", "reason": "resolution_reference_required", "required_action": "attach_resolution_reference"}},
	{"name": "resolution_approval_required", "condition": {"operation": "record_resolution", "approval_present": False}, "effect": {"decision": "deny", "reason": "resolution_approval_required", "required_action": "attach_resolution_approval"}},
	{"name": "resolution_evidence_required", "condition": {"operation": "record_resolution", "evidence_present": False}, "effect": {"decision": "deny", "reason": "resolution_evidence_required", "required_action": "attach_resolution_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "alert_batch_requires_bytewax", "condition": {"operation": "alert_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_alert_batch_to_bytewax"}},
	{"name": "alert_agent_runtime_supported", "condition": {"operation": "register_alert_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "alert_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "alert_agent_role_supported", "condition": {"operation": "register_alert_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "alert_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "alert_agent_name_required", "condition": {"operation": "register_alert_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "alert_agent_name_required", "required_action": "name_alert_agent"}},
	{"name": "alert_agent_scope_required", "condition": {"operation": "register_alert_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "alert_agent_scope_required", "required_action": "bound_alert_agent_scope"}},
	{"name": "privileged_alert_agent_action_requires_human_approval", "condition": {"operation": "alert_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "unapproved_escalation_action_denied", "condition": {"operation": "alert_agent_action", "unapproved_escalation_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_escalation_scope_denied", "required_action": "remove_unapproved_escalation_scope"}},
	{"name": "unapproved_notification_action_denied", "condition": {"operation": "alert_agent_action", "unapproved_notification_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_notification_scope_denied", "required_action": "remove_unapproved_notification_scope"}},
	{"name": "alert_suppression_action_denied", "condition": {"operation": "alert_agent_action", "alert_suppression_scope": True}, "effect": {"decision": "deny", "reason": "alert_suppression_scope_denied", "required_action": "remove_alert_suppression_scope"}},
	{"name": "evidence_fabrication_action_denied", "condition": {"operation": "alert_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_scope_denied", "required_action": "remove_evidence_fabrication_scope"}},
	{"name": "privacy_bypass_action_denied", "condition": {"operation": "alert_agent_action", "privacy_bypass_scope": True}, "effect": {"decision": "deny", "reason": "privacy_bypass_scope_denied", "required_action": "remove_privacy_bypass_scope"}},
	{"name": "autonomous_closure_action_denied", "condition": {"operation": "alert_agent_action", "autonomous_closure_scope": True}, "effect": {"decision": "deny", "reason": "autonomous_closure_scope_denied", "required_action": "remove_autonomous_closure_scope"}},
	{"name": "severity_downgrade_action_denied", "condition": {"operation": "alert_agent_action", "severity_downgrade_scope": True}, "effect": {"decision": "deny", "reason": "severity_downgrade_scope_denied", "required_action": "remove_severity_downgrade_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-alerts/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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

