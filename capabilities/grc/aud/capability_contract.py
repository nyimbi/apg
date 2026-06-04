"""Executable capability contract for GRC Audit Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "grc_aud"
CAPABILITY_NAME = "Audit Management"
CAPABILITY_VERSION = "1.0.0"
AUD_EVENT_STREAM = "apg.grc.aud.lifecycle"

SUPPORTED_AUDIT_TYPES = [
	"internal", "external", "regulatory", "iso_certification",
	"soc2", "penetration_test", "supplier", "it_general_controls",
]
SUPPORTED_AUDIT_STATUSES = [
	"planned", "in_progress", "fieldwork", "review", "report_draft",
	"report_final", "closed", "cancelled",
]
SUPPORTED_FINDING_SEVERITIES = ["observation", "minor", "major", "critical"]
SUPPORTED_FINDING_STATUSES = [
	"open", "in_remediation", "remediated", "accepted", "closed", "disputed",
]
SUPPORTED_EVIDENCE_TYPES = [
	"document", "screenshot", "log_export", "interview_note",
	"system_report", "configuration_snapshot", "certificate",
]
SUPPORTED_AUDIT_SCOPES = [
	"process", "system", "organizational_unit", "product", "supplier", "facility",
]
SUPPORTED_AUD_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AUD_AGENT_ROLES = [
	"audit_planner",
	"fieldwork_reviewer",
	"finding_reviewer",
	"evidence_reviewer",
	"remediation_tracker",
	"report_drafter",
]
SUPPORTED_REPORT_FORMATS = ["pdf", "docx", "html", "json"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"audits": {
		"title_required": True,
		"auditor_required": True,
		"audit_type_required": True,
		"supported_audit_types": SUPPORTED_AUDIT_TYPES,
		"supported_statuses": SUPPORTED_AUDIT_STATUSES,
		"scope_required": True,
		"supported_scopes": SUPPORTED_AUDIT_SCOPES,
		"start_date_required": True,
		"end_date_required": True,
		"auditee_required": True,
	},
	"findings": {
		"title_required": True,
		"severity_required": True,
		"supported_severities": SUPPORTED_FINDING_SEVERITIES,
		"supported_statuses": SUPPORTED_FINDING_STATUSES,
		"linked_audit_required": True,
		"owner_required": True,
		"remediation_plan_required": True,
		"major_finding_review_required": True,
		"critical_finding_escalation_required": True,
	},
	"evidence": {
		"linked_finding_or_audit_required": True,
		"evidence_type_required": True,
		"supported_evidence_types": SUPPORTED_EVIDENCE_TYPES,
		"encryption_required": True,
		"minimum_retention_days": 365,
		"tamper_evident": True,
	},
	"reports": {
		"linked_audit_required": True,
		"author_required": True,
		"supported_formats": SUPPORTED_REPORT_FORMATS,
		"approval_required_for_final": True,
		"approver_cannot_be_author": True,
	},
	"aud_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AUD_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AUD_AGENT_ROLES,
		"max_autonomous_scope": "review_and_recommend",
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
		"event_stream": AUD_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_audit_events": True,
		"emit_finding_events": True,
		"emit_evidence_events": True,
		"emit_report_events": True,
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
		"enable_audits": True,
		"enable_findings": True,
		"enable_evidence": True,
		"enable_reports": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "grc_aud_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"audit_program_lifecycle",
	"audit_finding_lifecycle",
	"audit_evidence_workflow",
	"audit_report_workflow",
	"audit_dashboard_service",
	"audit_agents",
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
	{"name": "dashboard", "path": "/grc-aud/dashboard", "component": "AuditDashboard", "permission": "grc_aud:view", "nav_group": "Overview"},
	{"name": "audits", "path": "/grc-aud/audits", "component": "AuditProgramWorkbench", "permission": "grc_aud:manage_audits", "nav_group": "Audits"},
	{"name": "audit_detail", "path": "/grc-aud/audits/:id", "component": "AuditDetail", "permission": "grc_aud:view", "nav_group": "Audits"},
	{"name": "findings", "path": "/grc-aud/findings", "component": "AuditFindingBoard", "permission": "grc_aud:manage_findings", "nav_group": "Findings"},
	{"name": "finding_detail", "path": "/grc-aud/findings/:id", "component": "FindingDetail", "permission": "grc_aud:view", "nav_group": "Findings"},
	{"name": "evidence", "path": "/grc-aud/evidence", "component": "AuditEvidenceVault", "permission": "grc_aud:manage_evidence", "nav_group": "Evidence"},
	{"name": "reports", "path": "/grc-aud/reports", "component": "AuditReportWorkbench", "permission": "grc_aud:manage_reports", "nav_group": "Reports"},
	{"name": "calendar", "path": "/grc-aud/calendar", "component": "AuditCalendar", "permission": "grc_aud:view", "nav_group": "Planning"},
	{"name": "agents", "path": "/grc-aud/agents", "component": "AuditAgentWorkbench", "permission": "grc_aud:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/grc-aud/settings", "component": "AuditSettings", "permission": "grc_aud:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "grc_aud_control",
	"tokens": {
		"color.primary": "#1B3A4B",
		"color.accent": "#5E6AD2",
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
		"audits": {"icon": "clipboard-list", "status_indicator": "audit-pill", "visual": "audit-timeline"},
		"findings": {"icon": "alert-circle", "visual": "finding-board", "status_style": "severity-chip"},
		"evidence": {"icon": "lock", "visual": "evidence-vault", "status_style": "retention-chip"},
		"reports": {"icon": "file-text", "visual": "report-list", "status_style": "report-chip"},
		"calendar": {"icon": "calendar", "visual": "audit-calendar", "status_style": "schedule-chip"},
		"agents": {"icon": "bot", "visual": "agent-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": AUD_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"audit_planned",
		"audit_started",
		"audit_fieldwork_completed",
		"audit_finding_raised",
		"audit_finding_updated",
		"audit_finding_escalated",
		"audit_finding_remediated",
		"audit_finding_closed",
		"audit_evidence_collected",
		"audit_evidence_tampered_alert",
		"audit_report_drafted",
		"audit_report_approved",
		"audit_report_published",
		"audit_closed",
		"audit_cancelled",
		"aud_agent_registered",
		"aud_agent_action_approved",
	],
	"states": SUPPORTED_AUDIT_STATUSES + ["queued", "failed", "expired"],
	"guardrails": [
		"aud_batch_requires_bytewax",
		"aud_event_requires_bytewax",
		"privileged_aud_agent_action_requires_human_approval",
		"cross_tenant_event_denied",
		"evidence_mutation_denied",
	],
}


RULES: list[dict[str, Any]] = [
	# Tenant and policy governance
	{
		"name": "tenant_context_required",
		"description": "Audit operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
	},
	{
		"name": "cross_tenant_access_denied",
		"description": "Audit data may not be accessed across tenant boundaries.",
		"condition": {"cross_tenant_access": True},
		"effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_identity"},
	},
	{
		"name": "aud_write_requires_policy",
		"description": "Audit writes require policy attachment.",
		"condition": {"operation_type": "write", "policy_attached": False},
		"effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"},
	},
	{
		"name": "privilege_escalation_denied",
		"description": "Users may not grant audit permissions exceeding their own level.",
		"condition": {"operation": "grant_audit_permission", "grant_exceeds_grantor_permission": True},
		"effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "reduce_grant_to_grantor_level"},
	},
	{
		"name": "admin_operation_requires_mfa",
		"description": "Admin-level audit operations require MFA.",
		"condition": {"permission_required": "admin", "mfa_verified": False},
		"effect": {"decision": "deny", "reason": "mfa_required_for_admin", "required_action": "complete_mfa"},
	},
	# Audit — create
	{
		"name": "audit_requires_title",
		"description": "Audits require a title.",
		"condition": {"operation": "create_audit", "title_present": False},
		"effect": {"decision": "deny", "reason": "audit_title_required", "required_action": "set_audit_title"},
	},
	{
		"name": "audit_requires_auditor",
		"description": "Audits require an assigned auditor.",
		"condition": {"operation": "create_audit", "auditor_present": False},
		"effect": {"decision": "deny", "reason": "audit_auditor_required", "required_action": "assign_auditor"},
	},
	{
		"name": "audit_type_supported",
		"description": "Audit type must be from the supported list.",
		"condition": {"operation": "create_audit", "audit_type_supported": False},
		"effect": {"decision": "deny", "reason": "audit_type_not_supported", "required_action": "select_supported_audit_type"},
	},
	{
		"name": "audit_requires_scope",
		"description": "Audits require a defined scope.",
		"condition": {"operation": "create_audit", "scope_present": False},
		"effect": {"decision": "deny", "reason": "audit_scope_required", "required_action": "define_audit_scope"},
	},
	{
		"name": "audit_scope_supported",
		"description": "Audit scope type must be from the supported list.",
		"condition": {"operation": "create_audit", "scope_type_supported": False},
		"effect": {"decision": "deny", "reason": "audit_scope_not_supported", "required_action": "select_supported_scope"},
	},
	{
		"name": "audit_requires_start_date",
		"description": "Audits require a start date.",
		"condition": {"operation": "create_audit", "start_date_present": False},
		"effect": {"decision": "deny", "reason": "audit_start_date_required", "required_action": "set_start_date"},
	},
	{
		"name": "audit_requires_end_date",
		"description": "Audits require an end date.",
		"condition": {"operation": "create_audit", "end_date_present": False},
		"effect": {"decision": "deny", "reason": "audit_end_date_required", "required_action": "set_end_date"},
	},
	{
		"name": "audit_end_date_after_start",
		"description": "Audit end date must be after start date.",
		"condition": {"operation": "create_audit", "end_before_start": True},
		"effect": {"decision": "deny", "reason": "audit_end_date_before_start", "required_action": "correct_audit_dates"},
	},
	{
		"name": "audit_requires_auditee",
		"description": "Audits require an identified auditee.",
		"condition": {"operation": "create_audit", "auditee_present": False},
		"effect": {"decision": "deny", "reason": "audit_auditee_required", "required_action": "assign_auditee"},
	},
	{
		"name": "auditor_cannot_be_auditee",
		"description": "Auditor and auditee must be different principals (segregation of duties).",
		"condition": {"operation": "create_audit", "auditor_is_auditee": True},
		"effect": {"decision": "deny", "reason": "auditor_auditee_segregation_required", "required_action": "assign_independent_auditor"},
	},
	# Audit — update
	{
		"name": "closed_audit_update_denied",
		"description": "Closed audits cannot be updated.",
		"condition": {"operation": "update_audit", "audit_status": "closed"},
		"effect": {"decision": "deny", "reason": "closed_audit_is_immutable", "required_action": "reopen_audit_to_update"},
	},
	# Audit — cancel
	{
		"name": "cancel_audit_requires_reason",
		"description": "Cancelling an audit requires a stated reason.",
		"condition": {"operation": "cancel_audit", "cancellation_reason_present": False},
		"effect": {"decision": "deny", "reason": "cancellation_reason_required", "required_action": "record_cancellation_reason"},
	},
	# Findings — create
	{
		"name": "finding_requires_title",
		"description": "Findings require a title.",
		"condition": {"operation": "raise_finding", "title_present": False},
		"effect": {"decision": "deny", "reason": "finding_title_required", "required_action": "set_finding_title"},
	},
	{
		"name": "finding_severity_supported",
		"description": "Finding severity must be from the supported list.",
		"condition": {"operation": "raise_finding", "finding_severity_supported": False},
		"effect": {"decision": "deny", "reason": "finding_severity_not_supported", "required_action": "select_supported_finding_severity"},
	},
	{
		"name": "finding_requires_linked_audit",
		"description": "Findings must be linked to an audit.",
		"condition": {"operation": "raise_finding", "linked_audit_present": False},
		"effect": {"decision": "deny", "reason": "finding_linked_audit_required", "required_action": "link_finding_to_audit"},
	},
	{
		"name": "finding_requires_owner",
		"description": "Findings require an owner for remediation.",
		"condition": {"operation": "raise_finding", "owner_present": False},
		"effect": {"decision": "deny", "reason": "finding_owner_required", "required_action": "assign_finding_owner"},
	},
	{
		"name": "finding_requires_remediation_plan",
		"description": "Findings require a remediation plan.",
		"condition": {"operation": "raise_finding", "remediation_plan_present": False},
		"effect": {"decision": "deny", "reason": "finding_remediation_plan_required", "required_action": "add_remediation_plan"},
	},
	{
		"name": "major_finding_requires_review",
		"description": "Major or critical findings require mandatory review.",
		"condition": {"operation": "raise_finding", "major_or_critical_finding": True, "review_recorded": False},
		"effect": {"decision": "require_review", "reason": "major_finding_review_required", "required_action": "record_finding_review"},
	},
	{
		"name": "critical_finding_requires_escalation",
		"description": "Critical findings require immediate escalation.",
		"condition": {"operation": "raise_finding", "finding_severity": "critical", "escalation_recorded": False},
		"effect": {"decision": "require_review", "reason": "critical_finding_escalation_required", "required_action": "escalate_critical_finding"},
	},
	# Findings — approve/reject/escalate
	{
		"name": "approve_finding_requires_approver",
		"description": "Finding approval requires an approver distinct from the raiser.",
		"condition": {"operation": "approve_finding", "approver_is_raiser": True},
		"effect": {"decision": "deny", "reason": "finding_approval_segregation_required", "required_action": "assign_independent_approver"},
	},
	{
		"name": "reject_finding_requires_reason",
		"description": "Finding rejection requires a reason.",
		"condition": {"operation": "reject_finding", "rejection_reason_present": False},
		"effect": {"decision": "deny", "reason": "finding_rejection_reason_required", "required_action": "record_rejection_reason"},
	},
	{
		"name": "escalate_finding_requires_target",
		"description": "Finding escalation requires a target.",
		"condition": {"operation": "escalate_finding", "escalation_target_present": False},
		"effect": {"decision": "deny", "reason": "escalation_target_required", "required_action": "specify_escalation_target"},
	},
	# Evidence
	{
		"name": "evidence_requires_linked_record",
		"description": "Audit evidence must be linked to a finding or audit.",
		"condition": {"operation": "collect_evidence", "linked_record_present": False},
		"effect": {"decision": "deny", "reason": "evidence_linked_record_required", "required_action": "link_evidence_to_record"},
	},
	{
		"name": "evidence_type_supported",
		"description": "Evidence type must be from the supported list.",
		"condition": {"operation": "collect_evidence", "evidence_type_supported": False},
		"effect": {"decision": "deny", "reason": "evidence_type_not_supported", "required_action": "select_supported_evidence_type"},
	},
	{
		"name": "evidence_requires_encryption",
		"description": "Audit evidence must be encrypted.",
		"condition": {"operation": "collect_evidence", "encrypted": False},
		"effect": {"decision": "deny", "reason": "evidence_encryption_required", "required_action": "encrypt_evidence"},
	},
	{
		"name": "evidence_retention_minimum",
		"description": "Evidence retention must meet minimum.",
		"condition": {"operation": "collect_evidence", "retention_days_lt": 365},
		"effect": {"decision": "deny", "reason": "evidence_retention_too_short", "required_action": "increase_retention"},
	},
	{
		"name": "evidence_mutation_denied",
		"description": "Collected audit evidence is immutable.",
		"condition": {"operation": "update_evidence", "evidence_collected": True},
		"effect": {"decision": "deny", "reason": "evidence_is_immutable", "required_action": "create_new_evidence_record"},
	},
	# Reports
	{
		"name": "report_requires_linked_audit",
		"description": "Audit reports must be linked to an audit.",
		"condition": {"operation": "create_report", "linked_audit_present": False},
		"effect": {"decision": "deny", "reason": "report_linked_audit_required", "required_action": "link_report_to_audit"},
	},
	{
		"name": "report_requires_author",
		"description": "Audit reports require an author.",
		"condition": {"operation": "create_report", "author_present": False},
		"effect": {"decision": "deny", "reason": "report_author_required", "required_action": "assign_report_author"},
	},
	{
		"name": "final_report_requires_approval",
		"description": "Final audit reports require approval.",
		"condition": {"operation": "finalize_report", "approval_recorded": False},
		"effect": {"decision": "deny", "reason": "final_report_approval_required", "required_action": "obtain_report_approval"},
	},
	{
		"name": "report_approver_cannot_be_author",
		"description": "Report approver must be independent from the author.",
		"condition": {"operation": "approve_report", "approver_is_author": True},
		"effect": {"decision": "deny", "reason": "report_approval_segregation_required", "required_action": "assign_independent_approver"},
	},
	{
		"name": "report_format_supported",
		"description": "Report export format must be from the supported list.",
		"condition": {"operation": "export_report", "format_supported": False},
		"effect": {"decision": "deny", "reason": "report_format_not_supported", "required_action": "select_supported_report_format"},
	},
	# Streaming infrastructure
	{
		"name": "aud_batch_requires_bytewax",
		"description": "Audit batches require Bytewax coordination.",
		"condition": {"operation": "aud_batch", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_aud_batch_to_bytewax"},
	},
	{
		"name": "aud_event_requires_bytewax",
		"description": "Audit events require Bytewax.",
		"condition": {"operation": "aud_event", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_aud_event_to_bytewax"},
	},
	# Agent governance
	{
		"name": "aud_agent_runtime_supported",
		"description": "Audit agents must use an approved runtime.",
		"condition": {"operation": "register_aud_agent", "agent_runtime_supported": False},
		"effect": {"decision": "deny", "reason": "aud_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"},
	},
	{
		"name": "aud_agent_role_supported",
		"description": "Audit agents must use an approved role.",
		"condition": {"operation": "register_aud_agent", "agent_role_supported": False},
		"effect": {"decision": "deny", "reason": "aud_agent_role_not_supported", "required_action": "select_supported_agent_role"},
	},
	{
		"name": "privileged_aud_agent_action_requires_human_approval",
		"description": "Privileged audit actions proposed by agents require human approval.",
		"condition": {"operation": "aud_agent_action", "privileged_scope": True, "human_approval_recorded": False},
		"effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"},
	},
	# Domain-specific governance
	{
		"name": "regulatory_audit_requires_external_auditor",
		"description": "Regulatory audits must use an external auditor.",
		"condition": {"operation": "create_audit", "audit_type": "regulatory", "auditor_is_internal": True},
		"effect": {"decision": "deny", "reason": "regulatory_audit_requires_external_auditor", "required_action": "assign_external_auditor"},
	},
	{
		"name": "audit_close_requires_all_findings_resolved",
		"description": "An audit cannot be closed while open critical or major findings remain.",
		"condition": {"operation": "close_audit", "open_critical_or_major_findings": True},
		"effect": {"decision": "deny", "reason": "open_findings_block_audit_close", "required_action": "resolve_open_findings"},
	},
	{
		"name": "finding_dispute_requires_evidence",
		"description": "Disputing a finding requires supporting evidence.",
		"condition": {"operation": "dispute_finding", "dispute_evidence_present": False},
		"effect": {"decision": "deny", "reason": "dispute_evidence_required", "required_action": "attach_dispute_evidence"},
	},
	{
		"name": "accepted_finding_requires_risk_acceptance",
		"description": "Accepting a finding without remediation requires a formal risk acceptance.",
		"condition": {"operation": "accept_finding", "risk_acceptance_recorded": False},
		"effect": {"decision": "deny", "reason": "risk_acceptance_required_to_accept_finding", "required_action": "record_risk_acceptance"},
	},
	{
		"name": "audit_scope_change_requires_approval",
		"description": "Changes to an in-progress audit scope require approval.",
		"condition": {"operation": "update_audit", "scope_changed": True, "audit_status": "in_progress", "approval_recorded": False},
		"effect": {"decision": "require_review", "reason": "in_progress_audit_scope_change_requires_approval", "required_action": "obtain_scope_change_approval"},
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
			"api_prefix": "/grc-aud/api/v1",
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
