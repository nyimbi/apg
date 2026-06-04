"""Executable capability contract for APG Real-Time Monitoring."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_monitoring"
CAPABILITY_NAME = "Real-Time Monitoring"
CAPABILITY_VERSION = "1.1.0"
MONITORING_EVENT_STREAM = "apg.intel.monitoring.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["legal_mandate", "security_monitoring_authority", "incident_response_authority", "consent", "partner_authority", "public_safety_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_POLICY_TYPES = ["security", "fraud", "public_safety", "operations", "compliance", "brand", "availability", "threat"]
SUPPORTED_SOURCE_TYPES = ["event_stream", "log_stream", "metric_stream", "telemetry_feed", "partner_feed", "case_feed", "sensor_feed", "api_feed"]
SUPPORTED_WATCH_TYPES = ["threshold", "pattern", "correlation", "watchlist", "anomaly", "sla", "keyword", "risk_rule"]
SUPPORTED_EVENT_TYPES = ["log", "metric", "trace", "alert", "case_update", "sensor_event", "external_notice", "heartbeat"]
SUPPORTED_SIGNAL_TYPES = ["threshold_breach", "correlation", "anomaly", "watchlist_match", "availability_degradation", "fraud_signal", "threat_signal", "safety_signal"]
SUPPORTED_INCIDENT_TYPES = ["security_incident", "fraud_incident", "safety_incident", "availability_incident", "compliance_incident", "brand_incident", "operational_incident"]
SUPPORTED_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_RETENTION_CLASSES = ["short", "standard", "extended", "legal_hold"]
SUPPORTED_REFERRAL_TYPES = ["incident_response", "fraud_review", "public_safety_notice", "compliance_review", "partner_notice", "maintenance_ticket", "case_escalation"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["authority_reviewer", "policy_planner", "source_steward", "signal_analyst", "incident_analyst", "dissemination_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"policies": {"supported_policy_types": SUPPORTED_POLICY_TYPES, "supported_severities": SUPPORTED_SEVERITIES, "authority_required": True, "evidence_required": True},
	"sources": {"supported_source_types": SUPPORTED_SOURCE_TYPES, "owner_required": True, "authority_required": True, "access_review_required": True, "evidence_required": True},
	"watches": {"supported_watch_types": SUPPORTED_WATCH_TYPES, "supported_retention_classes": SUPPORTED_RETENTION_CLASSES, "policy_required": True, "source_required": True, "watch_expression_required": True, "evidence_required": True},
	"events": {"supported_event_types": SUPPORTED_EVENT_TYPES, "watch_required": True, "event_fingerprint_required": True, "confidence_required": True, "observed_at_required": True, "evidence_required": True},
	"signals": {"supported_signal_types": SUPPORTED_SIGNAL_TYPES, "supported_severities": SUPPORTED_SEVERITIES, "event_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"incidents": {"supported_incident_types": SUPPORTED_INCIDENT_TYPES, "supported_severities": SUPPORTED_SEVERITIES, "signal_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"referrals": {"supported_types": SUPPORTED_REFERRAL_TYPES, "incident_required": True, "recipient_required": True, "approval_required": True, "evidence_required": True},
	"dissemination": {"incident_required": True, "audience_required": True, "release_marking_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True, "destructive_action_denied": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "source_access_review_required": True, "destructive_action_denied": True, "autonomous_enforcement_denied": True, "privacy_bypass_denied": True, "data_exfiltration_denied": True, "unauthorized_expansion_denied": True, "account_action_denied": True, "takedown_denied": True},
	"observability": {"event_stream": MONITORING_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_policies": True, "enable_sources": True, "enable_watches": True, "enable_events": True, "enable_signals": True, "enable_incidents": True, "enable_referrals": True, "enable_dissemination": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_monitoring_control", "allow_tenant_overrides": True},
}

PROVIDES = ["monitoring_authority_workflow", "monitoring_policy_workflow", "monitoring_source_workflow", "monitoring_watch_workflow", "monitoring_event_workflow", "monitoring_signal_workflow", "monitoring_incident_workflow", "monitoring_referral_workflow", "monitoring_dissemination_workflow", "monitoring_review_workflow", "monitoring_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-monitoring/dashboard", "component": "MonitoringDashboard", "permission": "intel_monitoring:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-monitoring/authorities", "component": "MonitoringAuthorityConsole", "permission": "intel_monitoring:authorities", "nav_group": "Governance"},
	{"name": "policies", "path": "/intel-monitoring/policies", "component": "MonitoringPolicyPlanner", "permission": "intel_monitoring:policies", "nav_group": "Planning"},
	{"name": "sources", "path": "/intel-monitoring/sources", "component": "MonitoringSourceRegistry", "permission": "intel_monitoring:sources", "nav_group": "Sources"},
	{"name": "watches", "path": "/intel-monitoring/watches", "component": "MonitoringWatchConsole", "permission": "intel_monitoring:watches", "nav_group": "Detection"},
	{"name": "events", "path": "/intel-monitoring/events", "component": "MonitoringEventLedger", "permission": "intel_monitoring:events", "nav_group": "Detection"},
	{"name": "signals", "path": "/intel-monitoring/signals", "component": "MonitoringSignalWorkbench", "permission": "intel_monitoring:signals", "nav_group": "Analysis"},
	{"name": "incidents", "path": "/intel-monitoring/incidents", "component": "MonitoringIncidentWorkbench", "permission": "intel_monitoring:incidents", "nav_group": "Response"},
	{"name": "referrals", "path": "/intel-monitoring/referrals", "component": "MonitoringReferralConsole", "permission": "intel_monitoring:referrals", "nav_group": "Release"},
	{"name": "dissemination", "path": "/intel-monitoring/dissemination", "component": "MonitoringDisseminationConsole", "permission": "intel_monitoring:dissemination", "nav_group": "Release"},
	{"name": "reviews", "path": "/intel-monitoring/reviews", "component": "MonitoringReviewConsole", "permission": "intel_monitoring:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-monitoring/agents", "component": "MonitoringAgentWorkbench", "permission": "intel_monitoring:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-monitoring/settings", "component": "MonitoringSettings", "permission": "intel_monitoring:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_monitoring_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#2563EB", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "policies": {"icon": "sliders-horizontal", "status_indicator": "severity-chip"}, "sources": {"icon": "database", "status_indicator": "source-chip"}, "watches": {"icon": "radar", "status_indicator": "watch-chip"}, "events": {"icon": "activity", "status_indicator": "event-chip"}, "signals": {"icon": "bell-ring", "status_indicator": "signal-chip"}, "incidents": {"icon": "shield-alert", "status_indicator": "incident-chip"}, "referrals": {"icon": "file-output", "status_indicator": "referral-chip"}, "dissemination": {"icon": "send", "status_indicator": "release-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": MONITORING_EVENT_STREAM, "key": "tenant_id", "events": ["monitoring_authority_recorded", "monitoring_policy_recorded", "monitoring_source_registered", "monitoring_watch_recorded", "monitoring_event_recorded", "monitoring_signal_recorded", "monitoring_incident_recorded", "monitoring_referral_recorded", "monitoring_dissemination_recorded", "monitoring_review_recorded", "monitoring_agent_registered"], "guardrails": ["monitoring_batch_requires_bytewax", "privileged_monitoring_agent_action_requires_human_approval", "destructive_action_denied", "autonomous_enforcement_action_denied", "privacy_bypass_action_denied", "data_exfiltration_action_denied", "unauthorized_expansion_action_denied", "account_action_denied", "takedown_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "monitoring_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "monitoring_policy_required", "required_action": "attach_monitoring_policy"}},
	{"name": "authority_type_supported", "condition": {"operation": "record_authority", "authority_type_supported": False}, "effect": {"decision": "deny", "reason": "authority_type_not_supported", "required_action": "select_supported_authority_type"}},
	{"name": "authority_scope_required", "condition": {"operation": "record_authority", "scope_present": False}, "effect": {"decision": "deny", "reason": "authority_scope_required", "required_action": "attach_scope_reference"}},
	{"name": "authority_classification_supported", "condition": {"operation": "record_authority", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "authority_approver_required", "condition": {"operation": "record_authority", "approver_present": False}, "effect": {"decision": "deny", "reason": "authority_approver_required", "required_action": "record_approver"}},
	{"name": "authority_expiry_required", "condition": {"operation": "record_authority", "expiry_present": False}, "effect": {"decision": "deny", "reason": "authority_expiry_required", "required_action": "set_expiry"}},
	{"name": "authority_evidence_required", "condition": {"operation": "record_authority", "evidence_present": False}, "effect": {"decision": "deny", "reason": "authority_evidence_required", "required_action": "attach_authority_evidence"}},
	{"name": "policy_type_supported", "condition": {"operation": "record_policy", "policy_type_supported": False}, "effect": {"decision": "deny", "reason": "policy_type_not_supported", "required_action": "select_supported_policy_type"}},
	{"name": "policy_name_required", "condition": {"operation": "record_policy", "policy_name_present": False}, "effect": {"decision": "deny", "reason": "policy_name_required", "required_action": "name_policy"}},
	{"name": "policy_severity_supported", "condition": {"operation": "record_policy", "severity_supported": False}, "effect": {"decision": "deny", "reason": "severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "policy_authority_required", "condition": {"operation": "record_policy", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "policy_evidence_required", "condition": {"operation": "record_policy", "evidence_present": False}, "effect": {"decision": "deny", "reason": "policy_evidence_required", "required_action": "attach_policy_evidence"}},
	{"name": "source_type_supported", "condition": {"operation": "register_source", "source_type_supported": False}, "effect": {"decision": "deny", "reason": "source_type_not_supported", "required_action": "select_supported_source_type"}},
	{"name": "source_reference_required", "condition": {"operation": "register_source", "source_reference_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "source_owner_required", "condition": {"operation": "register_source", "owner_present": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_authority_required", "condition": {"operation": "register_source", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "source_access_review_required", "condition": {"operation": "register_source", "access_review_present": False}, "effect": {"decision": "deny", "reason": "source_access_review_required", "required_action": "record_source_access_review"}},
	{"name": "source_evidence_required", "condition": {"operation": "register_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "watch_policy_required", "condition": {"operation": "record_watch", "policy_present": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "select_policy"}},
	{"name": "watch_source_required", "condition": {"operation": "record_watch", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "watch_policy_source_authority_match", "condition": {"operation": "record_watch", "policy_source_authority_match": False}, "effect": {"decision": "deny", "reason": "authority_mismatch", "required_action": "align_policy_source_authority"}},
	{"name": "watch_type_supported", "condition": {"operation": "record_watch", "watch_type_supported": False}, "effect": {"decision": "deny", "reason": "watch_type_not_supported", "required_action": "select_supported_watch_type"}},
	{"name": "watch_expression_required", "condition": {"operation": "record_watch", "watch_expression_present": False}, "effect": {"decision": "deny", "reason": "watch_expression_required", "required_action": "attach_watch_expression"}},
	{"name": "watch_retention_supported", "condition": {"operation": "record_watch", "retention_supported": False}, "effect": {"decision": "deny", "reason": "retention_class_not_supported", "required_action": "select_supported_retention_class"}},
	{"name": "watch_evidence_required", "condition": {"operation": "record_watch", "evidence_present": False}, "effect": {"decision": "deny", "reason": "watch_evidence_required", "required_action": "attach_watch_evidence"}},
	{"name": "event_watch_required", "condition": {"operation": "record_event", "watch_present": False}, "effect": {"decision": "deny", "reason": "watch_required", "required_action": "select_watch"}},
	{"name": "event_type_supported", "condition": {"operation": "record_event", "event_type_supported": False}, "effect": {"decision": "deny", "reason": "event_type_not_supported", "required_action": "select_supported_event_type"}},
	{"name": "event_reference_required", "condition": {"operation": "record_event", "event_reference_present": False}, "effect": {"decision": "deny", "reason": "event_reference_required", "required_action": "attach_event_reference"}},
	{"name": "event_fingerprint_required", "condition": {"operation": "record_event", "fingerprint_present": False}, "effect": {"decision": "deny", "reason": "event_fingerprint_required", "required_action": "record_event_fingerprint"}},
	{"name": "event_observed_at_required", "condition": {"operation": "record_event", "observed_at_present": False}, "effect": {"decision": "deny", "reason": "observed_at_required", "required_action": "record_observed_at"}},
	{"name": "event_confidence_valid", "condition": {"operation": "record_event", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "event_evidence_required", "condition": {"operation": "record_event", "evidence_present": False}, "effect": {"decision": "deny", "reason": "event_evidence_required", "required_action": "attach_event_evidence"}},
	{"name": "signal_event_required", "condition": {"operation": "record_signal", "event_present": False}, "effect": {"decision": "deny", "reason": "event_required", "required_action": "select_event"}},
	{"name": "signal_type_supported", "condition": {"operation": "record_signal", "signal_type_supported": False}, "effect": {"decision": "deny", "reason": "signal_type_not_supported", "required_action": "select_supported_signal_type"}},
	{"name": "signal_severity_supported", "condition": {"operation": "record_signal", "severity_supported": False}, "effect": {"decision": "deny", "reason": "severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "signal_confidence_valid", "condition": {"operation": "record_signal", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "signal_analyst_required", "condition": {"operation": "record_signal", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "signal_evidence_required", "condition": {"operation": "record_signal", "evidence_present": False}, "effect": {"decision": "deny", "reason": "signal_evidence_required", "required_action": "attach_signal_evidence"}},
	{"name": "incident_signal_required", "condition": {"operation": "record_incident", "signal_present": False}, "effect": {"decision": "deny", "reason": "signal_required", "required_action": "select_signal"}},
	{"name": "incident_type_supported", "condition": {"operation": "record_incident", "incident_type_supported": False}, "effect": {"decision": "deny", "reason": "incident_type_not_supported", "required_action": "select_supported_incident_type"}},
	{"name": "incident_severity_supported", "condition": {"operation": "record_incident", "severity_supported": False}, "effect": {"decision": "deny", "reason": "severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "incident_confidence_valid", "condition": {"operation": "record_incident", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "incident_analyst_required", "condition": {"operation": "record_incident", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "incident_evidence_required", "condition": {"operation": "record_incident", "evidence_present": False}, "effect": {"decision": "deny", "reason": "incident_evidence_required", "required_action": "attach_incident_evidence"}},
	{"name": "referral_incident_required", "condition": {"operation": "record_referral", "incident_present": False}, "effect": {"decision": "deny", "reason": "incident_required", "required_action": "select_incident"}},
	{"name": "referral_type_supported", "condition": {"operation": "record_referral", "referral_type_supported": False}, "effect": {"decision": "deny", "reason": "referral_type_not_supported", "required_action": "select_supported_referral_type"}},
	{"name": "referral_recipient_required", "condition": {"operation": "record_referral", "recipient_present": False}, "effect": {"decision": "deny", "reason": "recipient_required", "required_action": "select_recipient"}},
	{"name": "referral_approval_required", "condition": {"operation": "record_referral", "approval_present": False}, "effect": {"decision": "deny", "reason": "referral_approval_required", "required_action": "attach_referral_approval"}},
	{"name": "referral_evidence_required", "condition": {"operation": "record_referral", "evidence_present": False}, "effect": {"decision": "deny", "reason": "referral_evidence_required", "required_action": "attach_referral_evidence"}},
	{"name": "dissemination_incident_required", "condition": {"operation": "record_dissemination", "incident_present": False}, "effect": {"decision": "deny", "reason": "incident_required", "required_action": "select_incident"}},
	{"name": "dissemination_audience_required", "condition": {"operation": "record_dissemination", "audience_present": False}, "effect": {"decision": "deny", "reason": "audience_required", "required_action": "select_audience"}},
	{"name": "dissemination_release_required", "condition": {"operation": "record_dissemination", "release_marking_present": False}, "effect": {"decision": "deny", "reason": "release_marking_required", "required_action": "set_release_marking"}},
	{"name": "dissemination_approval_required", "condition": {"operation": "record_dissemination", "approval_present": False}, "effect": {"decision": "deny", "reason": "dissemination_approval_required", "required_action": "attach_release_approval"}},
	{"name": "dissemination_evidence_required", "condition": {"operation": "record_dissemination", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dissemination_evidence_required", "required_action": "attach_dissemination_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "monitoring_batch_requires_bytewax", "condition": {"operation": "monitoring_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_monitoring_batch_to_bytewax"}},
	{"name": "monitoring_agent_runtime_supported", "condition": {"operation": "register_monitoring_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "monitoring_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "monitoring_agent_role_supported", "condition": {"operation": "register_monitoring_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "monitoring_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_monitoring_agent_action_requires_human_approval", "condition": {"operation": "monitoring_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "destructive_action_denied", "condition": {"operation": "monitoring_agent_action", "destructive_action_scope": True}, "effect": {"decision": "deny", "reason": "destructive_action_scope_denied", "required_action": "remove_destructive_scope"}},
	{"name": "autonomous_enforcement_action_denied", "condition": {"operation": "monitoring_agent_action", "autonomous_enforcement_scope": True}, "effect": {"decision": "deny", "reason": "autonomous_enforcement_scope_denied", "required_action": "remove_autonomous_enforcement_scope"}},
	{"name": "privacy_bypass_action_denied", "condition": {"operation": "monitoring_agent_action", "privacy_bypass_scope": True}, "effect": {"decision": "deny", "reason": "privacy_bypass_scope_denied", "required_action": "remove_privacy_bypass_scope"}},
	{"name": "data_exfiltration_action_denied", "condition": {"operation": "monitoring_agent_action", "data_exfiltration_scope": True}, "effect": {"decision": "deny", "reason": "data_exfiltration_scope_denied", "required_action": "remove_data_exfiltration_scope"}},
	{"name": "unauthorized_expansion_action_denied", "condition": {"operation": "monitoring_agent_action", "unauthorized_expansion_scope": True}, "effect": {"decision": "deny", "reason": "unauthorized_expansion_scope_denied", "required_action": "remove_unauthorized_expansion_scope"}},
	{"name": "account_action_denied", "condition": {"operation": "monitoring_agent_action", "account_action_scope": True}, "effect": {"decision": "deny", "reason": "account_action_scope_denied", "required_action": "remove_account_action_scope"}},
	{"name": "takedown_action_denied", "condition": {"operation": "monitoring_agent_action", "takedown_scope": True}, "effect": {"decision": "deny", "reason": "takedown_scope_denied", "required_action": "remove_takedown_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-monitoring/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
