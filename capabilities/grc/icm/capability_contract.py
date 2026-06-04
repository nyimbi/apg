"""Executable capability contract for GRC Incident and Case Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "grc_icm"
CAPABILITY_NAME = "Incident and Case Management"
CAPABILITY_VERSION = "1.0.0"
ICM_EVENT_STREAM = "apg.grc.icm.lifecycle"

SUPPORTED_INCIDENT_TYPES = [
	"security_breach", "data_loss", "service_disruption", "policy_violation",
	"fraud", "physical_security", "compliance_breach", "near_miss", "third_party",
]
SUPPORTED_INCIDENT_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_INCIDENT_STATUSES = [
	"new", "triaged", "in_investigation", "contained", "eradicated",
	"recovering", "post_incident_review", "closed", "false_positive",
]
SUPPORTED_CASE_TYPES = [
	"disciplinary", "regulatory_inquiry", "legal_hold", "fraud_investigation",
	"whistleblower", "data_subject_request",
]
SUPPORTED_CASE_STATUSES = [
	"open", "under_investigation", "pending_legal", "resolved", "closed", "appealed",
]
SUPPORTED_NOTIFICATION_CHANNELS = ["email", "sms", "webhook", "slack", "teams", "pagerduty"]
SUPPORTED_ICM_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ICM_AGENT_ROLES = [
	"triage_analyst",
	"incident_coordinator",
	"investigation_reviewer",
	"evidence_reviewer",
	"notification_reviewer",
	"post_incident_reviewer",
]
SUPPORTED_REGULATORY_WINDOWS = {
	"gdpr": 72,
	"hipaa": 60 * 24,
	"pci_dss": 24,
	"default": 72,
}


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"incidents": {
		"title_required": True,
		"type_required": True,
		"supported_types": SUPPORTED_INCIDENT_TYPES,
		"severity_required": True,
		"supported_severities": SUPPORTED_INCIDENT_SEVERITIES,
		"supported_statuses": SUPPORTED_INCIDENT_STATUSES,
		"reporter_required": True,
		"owner_required": True,
		"detection_time_required": True,
		"critical_requires_immediate_triage": True,
		"regulatory_notification_windows_hours": SUPPORTED_REGULATORY_WINDOWS,
	},
	"cases": {
		"title_required": True,
		"type_required": True,
		"supported_types": SUPPORTED_CASE_TYPES,
		"supported_statuses": SUPPORTED_CASE_STATUSES,
		"owner_required": True,
		"legal_review_for_regulatory": True,
		"confidential_by_default": True,
	},
	"evidence": {
		"linked_record_required": True,
		"encryption_required": True,
		"chain_of_custody_required": True,
		"minimum_retention_days": 365,
		"tamper_evident": True,
	},
	"notifications": {
		"supported_channels": SUPPORTED_NOTIFICATION_CHANNELS,
		"critical_incident_auto_notify": True,
		"regulatory_breach_auto_notify": True,
		"notification_log_required": True,
	},
	"icm_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_ICM_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_ICM_AGENT_ROLES,
		"max_autonomous_scope": "triage_and_recommend",
		"human_approval_required": True,
		"privileged_actions_logged": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_state_changes": True,
		"segregation_of_duties": True,
		"cross_tenant_access_denied": True,
	},
	"observability": {
		"event_stream": ICM_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_incident_events": True,
		"emit_case_events": True,
		"emit_evidence_events": True,
		"emit_notification_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit_log": "adapter",
		"notification": "adapter",
		"document_management": "adapter",
		"workflow_orchestration": "adapter",
		"policy_management": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
		"multi_tenancy": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_incidents": True,
		"enable_cases": True,
		"enable_evidence": True,
		"enable_notifications": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "grc_icm_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"incident_lifecycle_management",
	"case_management_workflow",
	"incident_evidence_workflow",
	"regulatory_notification_workflow",
	"post_incident_review_workflow",
	"icm_dashboard_service",
	"icm_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"mten",
	"conf",
	"ntfy",
	"grc_doc",
	"wflo",
	"grc_pol",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/grc-icm/dashboard", "component": "IncidentDashboard", "permission": "grc_icm:view", "nav_group": "Overview"},
	{"name": "incidents", "path": "/grc-icm/incidents", "component": "IncidentRegister", "permission": "grc_icm:manage_incidents", "nav_group": "Incidents"},
	{"name": "incident_detail", "path": "/grc-icm/incidents/:id", "component": "IncidentDetail", "permission": "grc_icm:view", "nav_group": "Incidents"},
	{"name": "cases", "path": "/grc-icm/cases", "component": "CaseWorkbench", "permission": "grc_icm:manage_cases", "nav_group": "Cases"},
	{"name": "case_detail", "path": "/grc-icm/cases/:id", "component": "CaseDetail", "permission": "grc_icm:view", "nav_group": "Cases"},
	{"name": "evidence", "path": "/grc-icm/evidence", "component": "IncidentEvidenceVault", "permission": "grc_icm:manage_evidence", "nav_group": "Evidence"},
	{"name": "notifications", "path": "/grc-icm/notifications", "component": "NotificationLog", "permission": "grc_icm:view", "nav_group": "Notifications"},
	{"name": "timeline", "path": "/grc-icm/timeline", "component": "IncidentTimeline", "permission": "grc_icm:view", "nav_group": "Investigation"},
	{"name": "agents", "path": "/grc-icm/agents", "component": "IcmAgentWorkbench", "permission": "grc_icm:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/grc-icm/settings", "component": "IcmSettings", "permission": "grc_icm:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "grc_icm_control",
	"tokens": {
		"color.primary": "#7B1C1C",
		"color.accent": "#C05621",
		"color.success": "#237A57",
		"color.warning": "#B7791F",
		"color.danger": "#B42318",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"incidents": {"icon": "alert-octagon", "status_indicator": "incident-pill", "visual": "incident-board"},
		"cases": {"icon": "briefcase", "visual": "case-list", "status_style": "case-chip"},
		"evidence": {"icon": "lock", "visual": "evidence-vault", "status_style": "custody-chip"},
		"notifications": {"icon": "bell", "visual": "notification-log", "status_style": "channel-chip"},
		"timeline": {"icon": "clock", "visual": "incident-timeline", "status_style": "timeline-chip"},
		"agents": {"icon": "bot", "visual": "agent-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": ICM_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"incident_reported",
		"incident_triaged",
		"incident_severity_upgraded",
		"incident_severity_downgraded",
		"incident_contained",
		"incident_eradicated",
		"incident_escalated",
		"incident_post_review_opened",
		"incident_closed",
		"incident_false_positive_marked",
		"regulatory_notification_sent",
		"regulatory_notification_overdue",
		"case_opened",
		"case_updated",
		"case_legal_review_requested",
		"case_resolved",
		"case_closed",
		"evidence_collected",
		"evidence_tampered_alert",
		"icm_agent_registered",
		"icm_agent_action_approved",
	],
	"states": SUPPORTED_INCIDENT_STATUSES + SUPPORTED_CASE_STATUSES + ["queued", "failed"],
	"guardrails": [
		"icm_batch_requires_bytewax",
		"icm_event_requires_bytewax",
		"privileged_icm_agent_action_requires_human_approval",
		"cross_tenant_event_denied",
		"evidence_mutation_denied",
	],
}


RULES: list[dict[str, Any]] = [
	# Tenant and policy governance
	{
		"name": "tenant_context_required",
		"description": "Incident operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
	},
	{
		"name": "cross_tenant_access_denied",
		"description": "Incident data may not be accessed across tenant boundaries.",
		"condition": {"cross_tenant_access": True},
		"effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_identity"},
	},
	{
		"name": "icm_write_requires_policy",
		"description": "Incident writes require policy attachment.",
		"condition": {"operation_type": "write", "policy_attached": False},
		"effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"},
	},
	{
		"name": "privilege_escalation_denied",
		"description": "Users may not grant incident permissions exceeding their own level.",
		"condition": {"operation": "grant_icm_permission", "grant_exceeds_grantor_permission": True},
		"effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "reduce_grant_to_grantor_level"},
	},
	{
		"name": "admin_operation_requires_mfa",
		"description": "Admin-level incident operations require MFA.",
		"condition": {"permission_required": "admin", "mfa_verified": False},
		"effect": {"decision": "deny", "reason": "mfa_required_for_admin", "required_action": "complete_mfa"},
	},
	# Incident — create
	{
		"name": "incident_requires_title",
		"description": "Incidents require a title.",
		"condition": {"operation": "report_incident", "title_present": False},
		"effect": {"decision": "deny", "reason": "incident_title_required", "required_action": "set_incident_title"},
	},
	{
		"name": "incident_type_supported",
		"description": "Incident type must be from the supported list.",
		"condition": {"operation": "report_incident", "incident_type_supported": False},
		"effect": {"decision": "deny", "reason": "incident_type_not_supported", "required_action": "select_supported_incident_type"},
	},
	{
		"name": "incident_severity_supported",
		"description": "Incident severity must be from the supported list.",
		"condition": {"operation": "report_incident", "incident_severity_supported": False},
		"effect": {"decision": "deny", "reason": "incident_severity_not_supported", "required_action": "select_supported_incident_severity"},
	},
	{
		"name": "incident_requires_reporter",
		"description": "Incidents require an identified reporter.",
		"condition": {"operation": "report_incident", "reporter_present": False},
		"effect": {"decision": "deny", "reason": "incident_reporter_required", "required_action": "identify_reporter"},
	},
	{
		"name": "incident_requires_owner",
		"description": "Incidents require an assigned owner.",
		"condition": {"operation": "report_incident", "owner_present": False},
		"effect": {"decision": "deny", "reason": "incident_owner_required", "required_action": "assign_incident_owner"},
	},
	{
		"name": "incident_requires_detection_time",
		"description": "Incidents require a detection timestamp.",
		"condition": {"operation": "report_incident", "detection_time_present": False},
		"effect": {"decision": "deny", "reason": "incident_detection_time_required", "required_action": "record_detection_time"},
	},
	{
		"name": "critical_incident_requires_immediate_triage",
		"description": "Critical incidents require immediate triage.",
		"condition": {"operation": "report_incident", "incident_severity": "critical", "triage_recorded": False},
		"effect": {"decision": "require_review", "reason": "critical_incident_immediate_triage_required", "required_action": "triage_incident_immediately"},
	},
	# Incident — update / escalate
	{
		"name": "closed_incident_update_denied",
		"description": "Closed incidents cannot be updated.",
		"condition": {"operation": "update_incident", "incident_status": "closed"},
		"effect": {"decision": "deny", "reason": "closed_incident_is_immutable", "required_action": "reopen_incident_to_update"},
	},
	{
		"name": "escalate_incident_requires_reason",
		"description": "Incident escalation requires a stated reason.",
		"condition": {"operation": "escalate_incident", "escalation_reason_present": False},
		"effect": {"decision": "deny", "reason": "escalation_reason_required", "required_action": "record_escalation_reason"},
	},
	{
		"name": "escalate_incident_requires_target",
		"description": "Incident escalation requires a target.",
		"condition": {"operation": "escalate_incident", "escalation_target_present": False},
		"effect": {"decision": "deny", "reason": "escalation_target_required", "required_action": "specify_escalation_target"},
	},
	# Regulatory notification
	{
		"name": "regulatory_breach_requires_notification",
		"description": "Confirmed regulatory breaches require timely notification.",
		"condition": {"operation": "close_incident", "regulatory_breach": True, "notification_sent": False},
		"effect": {"decision": "deny", "reason": "regulatory_notification_required", "required_action": "send_regulatory_notification"},
	},
	{
		"name": "gdpr_breach_notification_window",
		"description": "GDPR breaches must be notified within 72 hours.",
		"condition": {"operation": "close_incident", "framework": "gdpr", "notification_hours_gt": 72},
		"effect": {"decision": "deny", "reason": "gdpr_72_hour_notification_window_exceeded", "required_action": "send_gdpr_notification_immediately"},
	},
	# Incident — close
	{
		"name": "close_incident_requires_post_review",
		"description": "High or critical incidents cannot be closed without a post-incident review.",
		"condition": {"operation": "close_incident", "high_or_critical_incident": True, "post_review_recorded": False},
		"effect": {"decision": "deny", "reason": "post_incident_review_required", "required_action": "complete_post_incident_review"},
	},
	{
		"name": "close_incident_requires_root_cause",
		"description": "Incidents cannot be closed without a recorded root cause.",
		"condition": {"operation": "close_incident", "root_cause_present": False},
		"effect": {"decision": "deny", "reason": "root_cause_required_to_close_incident", "required_action": "record_root_cause"},
	},
	# Case — create
	{
		"name": "case_requires_title",
		"description": "Cases require a title.",
		"condition": {"operation": "open_case", "title_present": False},
		"effect": {"decision": "deny", "reason": "case_title_required", "required_action": "set_case_title"},
	},
	{
		"name": "case_type_supported",
		"description": "Case type must be from the supported list.",
		"condition": {"operation": "open_case", "case_type_supported": False},
		"effect": {"decision": "deny", "reason": "case_type_not_supported", "required_action": "select_supported_case_type"},
	},
	{
		"name": "case_requires_owner",
		"description": "Cases require an assigned owner.",
		"condition": {"operation": "open_case", "owner_present": False},
		"effect": {"decision": "deny", "reason": "case_owner_required", "required_action": "assign_case_owner"},
	},
	{
		"name": "regulatory_case_requires_legal_review",
		"description": "Regulatory inquiry and legal hold cases require legal review.",
		"condition": {"operation": "open_case", "regulatory_case": True, "legal_review_recorded": False},
		"effect": {"decision": "require_review", "reason": "legal_review_required_for_regulatory_case", "required_action": "request_legal_review"},
	},
	# Case — approve/reject
	{
		"name": "approve_case_closure_requires_approver",
		"description": "Case closure requires an approver distinct from the case owner.",
		"condition": {"operation": "close_case", "approver_is_case_owner": True},
		"effect": {"decision": "deny", "reason": "case_closure_segregation_required", "required_action": "assign_independent_approver"},
	},
	{
		"name": "reject_case_requires_reason",
		"description": "Case rejection requires a reason.",
		"condition": {"operation": "reject_case", "rejection_reason_present": False},
		"effect": {"decision": "deny", "reason": "case_rejection_reason_required", "required_action": "record_rejection_reason"},
	},
	# Evidence
	{
		"name": "evidence_requires_linked_record",
		"description": "Incident evidence must be linked to an incident or case.",
		"condition": {"operation": "collect_evidence", "linked_record_present": False},
		"effect": {"decision": "deny", "reason": "evidence_linked_record_required", "required_action": "link_evidence_to_record"},
	},
	{
		"name": "evidence_requires_encryption",
		"description": "Incident evidence must be encrypted.",
		"condition": {"operation": "collect_evidence", "encrypted": False},
		"effect": {"decision": "deny", "reason": "evidence_encryption_required", "required_action": "encrypt_evidence"},
	},
	{
		"name": "evidence_requires_chain_of_custody",
		"description": "Incident evidence requires chain of custody documentation.",
		"condition": {"operation": "collect_evidence", "chain_of_custody_present": False},
		"effect": {"decision": "deny", "reason": "chain_of_custody_required", "required_action": "document_chain_of_custody"},
	},
	{
		"name": "evidence_retention_minimum",
		"description": "Evidence retention must meet minimum.",
		"condition": {"operation": "collect_evidence", "retention_days_lt": 365},
		"effect": {"decision": "deny", "reason": "evidence_retention_too_short", "required_action": "increase_retention"},
	},
	{
		"name": "evidence_mutation_denied",
		"description": "Collected incident evidence is immutable.",
		"condition": {"operation": "update_evidence", "evidence_collected": True},
		"effect": {"decision": "deny", "reason": "evidence_is_immutable", "required_action": "create_new_evidence_record"},
	},
	# Streaming infrastructure
	{
		"name": "icm_batch_requires_bytewax",
		"description": "Incident batches require Bytewax coordination.",
		"condition": {"operation": "icm_batch", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_icm_batch_to_bytewax"},
	},
	{
		"name": "icm_event_requires_bytewax",
		"description": "Incident events require Bytewax.",
		"condition": {"operation": "icm_event", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_icm_event_to_bytewax"},
	},
	# Agent governance
	{
		"name": "icm_agent_runtime_supported",
		"description": "Incident agents must use an approved runtime.",
		"condition": {"operation": "register_icm_agent", "agent_runtime_supported": False},
		"effect": {"decision": "deny", "reason": "icm_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"},
	},
	{
		"name": "icm_agent_role_supported",
		"description": "Incident agents must use an approved role.",
		"condition": {"operation": "register_icm_agent", "agent_role_supported": False},
		"effect": {"decision": "deny", "reason": "icm_agent_role_not_supported", "required_action": "select_supported_agent_role"},
	},
	{
		"name": "privileged_icm_agent_action_requires_human_approval",
		"description": "Privileged incident actions proposed by agents require human approval.",
		"condition": {"operation": "icm_agent_action", "privileged_scope": True, "human_approval_recorded": False},
		"effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"},
	},
	# Domain-specific governance
	{
		"name": "data_loss_incident_requires_dpia",
		"description": "Data loss incidents require a Data Protection Impact Assessment reference.",
		"condition": {"operation": "close_incident", "incident_type": "data_loss", "dpia_reference_present": False},
		"effect": {"decision": "deny", "reason": "dpia_required_for_data_loss_incident", "required_action": "attach_dpia_reference"},
	},
	{
		"name": "false_positive_requires_evidence",
		"description": "Marking an incident as false positive requires supporting evidence.",
		"condition": {"operation": "mark_false_positive", "false_positive_evidence_present": False},
		"effect": {"decision": "deny", "reason": "false_positive_evidence_required", "required_action": "attach_false_positive_evidence"},
	},
	{
		"name": "whistleblower_case_requires_confidentiality",
		"description": "Whistleblower cases must be marked confidential.",
		"condition": {"operation": "open_case", "case_type": "whistleblower", "marked_confidential": False},
		"effect": {"decision": "deny", "reason": "whistleblower_case_must_be_confidential", "required_action": "mark_case_confidential"},
	},
	{
		"name": "fraud_investigation_requires_legal_hold",
		"description": "Fraud investigation cases must trigger a legal hold.",
		"condition": {"operation": "open_case", "case_type": "fraud_investigation", "legal_hold_placed": False},
		"effect": {"decision": "deny", "reason": "legal_hold_required_for_fraud_investigation", "required_action": "place_legal_hold"},
	},
	{
		"name": "notification_requires_channel",
		"description": "Incident notifications must specify a delivery channel.",
		"condition": {"operation": "send_notification", "notification_channel_present": False},
		"effect": {"decision": "deny", "reason": "notification_channel_required", "required_action": "specify_notification_channel"},
	},
	{
		"name": "notification_channel_supported",
		"description": "Notification channel must be from the supported list.",
		"condition": {"operation": "send_notification", "notification_channel_supported": False},
		"effect": {"decision": "deny", "reason": "notification_channel_not_supported", "required_action": "select_supported_notification_channel"},
	},
]


def _configuration_schema() -> dict[str, Any]:
	return {
		"type": "object",
		"required": ["tenant_id", "ui", "theme"],
		"properties": {
			key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"
		} | {"tenant_id": {"type": "string", "minLength": 1}},
	}


def _matches_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			if context.get(key[:-4]) is None or context[key[:-4]] > expected:
				return False
			continue
		if key.endswith("_lt"):
			if context.get(key[:-3]) is None or context[key[:-3]] >= expected:
				return False
			continue
		if key.endswith("_gte"):
			if context.get(key[:-4]) is None or context[key[:-4]] < expected:
				return False
			continue
		if key.endswith("_gt"):
			if context.get(key[:-3]) is None or context[key[:-3]] <= expected:
				return False
			continue
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	if overrides:
		for key, value in overrides.items():
			if isinstance(value, dict) and isinstance(configuration.get(key), dict):
				configuration[key].update(value)
			else:
				configuration[key] = value

	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"provides": PROVIDES,
		"requires": REQUIRES,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/grc-icm/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	contract = get_capability_contract(context.get("tenant_id", "default"))
	matched = [
		rule for rule in contract["rule_engine"]["rules"]
		if _matches_condition(rule["condition"], context)
	]
	decision = "allow"
	for rule in matched:
		rule_decision = rule["effect"]["decision"]
		if rule_decision == "deny":
			decision = "deny"
			break
		if rule_decision == "require_review" and decision == "allow":
			decision = "require_review"
	return {
		"decision": decision,
		"matched_rules": [rule["name"] for rule in matched],
		"effects": [rule["effect"] for rule in matched],
	}
