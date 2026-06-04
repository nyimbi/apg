"""Executable capability contract for GRC Risk and Security Assessment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "grc_rsa"
CAPABILITY_NAME = "Risk and Security Assessment"
CAPABILITY_VERSION = "1.0.0"
RSA_EVENT_STREAM = "apg.grc.rsa.lifecycle"

SUPPORTED_ASSESSMENT_TYPES = [
	"vendor_risk", "penetration_test", "vulnerability_scan", "threat_modelling",
	"business_impact_analysis", "data_privacy_impact", "cloud_security_review",
	"third_party_audit", "physical_security_review",
]
SUPPORTED_ASSESSMENT_STATUSES = [
	"scoping", "in_progress", "findings_review", "remediation", "sign_off", "closed", "cancelled",
]
SUPPORTED_RISK_RATINGS = ["negligible", "low", "medium", "high", "critical"]
SUPPORTED_FINDING_TYPES = [
	"vulnerability", "misconfiguration", "policy_gap", "control_deficiency",
	"data_exposure", "access_control_weakness", "third_party_risk",
]
SUPPORTED_FINDING_STATUSES = [
	"open", "accepted", "in_remediation", "remediated", "false_positive", "closed",
]
SUPPORTED_CVSS_VERSIONS = ["3.1", "4.0"]
SUPPORTED_ASSET_TYPES = [
	"application", "infrastructure", "network_device", "cloud_service",
	"endpoint", "database", "third_party_service", "physical_facility",
]
SUPPORTED_REMEDIATION_STRATEGIES = ["fix", "mitigate", "accept", "transfer", "defer"]
SUPPORTED_RSA_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_RSA_AGENT_ROLES = [
	"scoping_analyst",
	"vulnerability_reviewer",
	"threat_modeller",
	"remediation_tracker",
	"risk_assessor",
	"report_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"assessments": {
		"title_required": True,
		"type_required": True,
		"supported_types": SUPPORTED_ASSESSMENT_TYPES,
		"supported_statuses": SUPPORTED_ASSESSMENT_STATUSES,
		"lead_assessor_required": True,
		"asset_scope_required": True,
		"supported_asset_types": SUPPORTED_ASSET_TYPES,
		"start_date_required": True,
		"end_date_required": True,
		"vendor_required_for_vendor_risk": True,
		"high_risk_assessment_review_required": True,
	},
	"findings": {
		"title_required": True,
		"finding_type_required": True,
		"supported_finding_types": SUPPORTED_FINDING_TYPES,
		"supported_finding_statuses": SUPPORTED_FINDING_STATUSES,
		"risk_rating_required": True,
		"supported_risk_ratings": SUPPORTED_RISK_RATINGS,
		"cvss_score_required_for_vulnerability": True,
		"supported_cvss_versions": SUPPORTED_CVSS_VERSIONS,
		"remediation_strategy_required": True,
		"supported_remediation_strategies": SUPPORTED_REMEDIATION_STRATEGIES,
		"owner_required": True,
		"critical_finding_escalation_required": True,
	},
	"remediation": {
		"due_date_required": True,
		"evidence_required_for_closure": True,
		"acceptance_requires_risk_sign_off": True,
		"defer_requires_approval": True,
		"max_defer_days": 180,
	},
	"vendor_risk": {
		"vendor_name_required": True,
		"vendor_tier_required": True,
		"contract_reference_required": True,
		"reassessment_frequency_days": 365,
		"critical_vendor_annual_assessment": True,
	},
	"rsa_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_RSA_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_RSA_AGENT_ROLES,
		"max_autonomous_scope": "scan_and_recommend",
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
		"event_stream": RSA_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_assessment_events": True,
		"emit_finding_events": True,
		"emit_remediation_events": True,
		"emit_vendor_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit_log": "adapter",
		"notification": "adapter",
		"document_management": "adapter",
		"workflow_orchestration": "adapter",
		"policy_management": "adapter",
		"risk_management": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
		"multi_tenancy": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_assessments": True,
		"enable_findings": True,
		"enable_remediation": True,
		"enable_vendor_risk": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "grc_rsa_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"security_assessment_lifecycle",
	"vulnerability_finding_workflow",
	"remediation_tracking_workflow",
	"vendor_risk_assessment_workflow",
	"threat_modelling_workflow",
	"rsa_dashboard_service",
	"rsa_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"mten",
	"conf",
	"ntfy",
	"grc_rcm",
	"grc_doc",
	"wflo",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/grc-rsa/dashboard", "component": "RsaDashboard", "permission": "grc_rsa:view", "nav_group": "Overview"},
	{"name": "assessments", "path": "/grc-rsa/assessments", "component": "AssessmentWorkbench", "permission": "grc_rsa:manage_assessments", "nav_group": "Assessments"},
	{"name": "assessment_detail", "path": "/grc-rsa/assessments/:id", "component": "AssessmentDetail", "permission": "grc_rsa:view", "nav_group": "Assessments"},
	{"name": "findings", "path": "/grc-rsa/findings", "component": "FindingRegister", "permission": "grc_rsa:manage_findings", "nav_group": "Findings"},
	{"name": "finding_detail", "path": "/grc-rsa/findings/:id", "component": "FindingDetail", "permission": "grc_rsa:view", "nav_group": "Findings"},
	{"name": "remediation", "path": "/grc-rsa/remediation", "component": "RemediationTracker", "permission": "grc_rsa:manage_remediation", "nav_group": "Remediation"},
	{"name": "vendor_risk", "path": "/grc-rsa/vendor-risk", "component": "VendorRiskRegister", "permission": "grc_rsa:manage_vendor_risk", "nav_group": "Vendor Risk"},
	{"name": "threat_model", "path": "/grc-rsa/threat-model", "component": "ThreatModelWorkbench", "permission": "grc_rsa:view", "nav_group": "Threat Intelligence"},
	{"name": "agents", "path": "/grc-rsa/agents", "component": "RsaAgentWorkbench", "permission": "grc_rsa:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/grc-rsa/settings", "component": "RsaSettings", "permission": "grc_rsa:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "grc_rsa_control",
	"tokens": {
		"color.primary": "#2D3748",
		"color.accent": "#E53E3E",
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
		"assessments": {"icon": "search", "status_indicator": "assessment-pill", "visual": "assessment-list"},
		"findings": {"icon": "bug", "visual": "finding-register", "status_style": "severity-chip"},
		"remediation": {"icon": "tool", "visual": "remediation-board", "status_style": "due-date-chip"},
		"vendor_risk": {"icon": "truck", "visual": "vendor-heatmap", "status_style": "vendor-tier-chip"},
		"threat_model": {"icon": "target", "visual": "threat-canvas", "status_style": "threat-chip"},
		"agents": {"icon": "bot", "visual": "agent-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": RSA_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"assessment_scoped",
		"assessment_started",
		"assessment_finding_raised",
		"assessment_finding_severity_upgraded",
		"assessment_finding_accepted",
		"assessment_finding_false_positive",
		"assessment_findings_reviewed",
		"assessment_signed_off",
		"assessment_closed",
		"assessment_cancelled",
		"remediation_started",
		"remediation_completed",
		"remediation_deferred",
		"remediation_overdue",
		"vendor_risk_assessed",
		"vendor_risk_upgraded",
		"vendor_risk_reassessment_due",
		"threat_model_created",
		"threat_model_updated",
		"rsa_agent_registered",
		"rsa_agent_action_approved",
	],
	"states": SUPPORTED_ASSESSMENT_STATUSES + SUPPORTED_FINDING_STATUSES + ["queued", "failed", "overdue"],
	"guardrails": [
		"rsa_batch_requires_bytewax",
		"rsa_event_requires_bytewax",
		"privileged_rsa_agent_action_requires_human_approval",
		"cross_tenant_event_denied",
	],
}


RULES: list[dict[str, Any]] = [
	# Tenant and policy governance
	{
		"name": "tenant_context_required",
		"description": "RSA operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
	},
	{
		"name": "cross_tenant_access_denied",
		"description": "Assessment data may not be accessed across tenant boundaries.",
		"condition": {"cross_tenant_access": True},
		"effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_identity"},
	},
	{
		"name": "rsa_write_requires_policy",
		"description": "RSA writes require policy attachment.",
		"condition": {"operation_type": "write", "policy_attached": False},
		"effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"},
	},
	{
		"name": "privilege_escalation_denied",
		"description": "Users may not grant RSA permissions exceeding their own level.",
		"condition": {"operation": "grant_rsa_permission", "grant_exceeds_grantor_permission": True},
		"effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "reduce_grant_to_grantor_level"},
	},
	{
		"name": "admin_operation_requires_mfa",
		"description": "Admin-level RSA operations require MFA.",
		"condition": {"permission_required": "admin", "mfa_verified": False},
		"effect": {"decision": "deny", "reason": "mfa_required_for_admin", "required_action": "complete_mfa"},
	},
	# Assessment — create
	{
		"name": "assessment_requires_title",
		"description": "Assessments require a title.",
		"condition": {"operation": "create_assessment", "title_present": False},
		"effect": {"decision": "deny", "reason": "assessment_title_required", "required_action": "set_assessment_title"},
	},
	{
		"name": "assessment_type_supported",
		"description": "Assessment type must be from the supported list.",
		"condition": {"operation": "create_assessment", "assessment_type_supported": False},
		"effect": {"decision": "deny", "reason": "assessment_type_not_supported", "required_action": "select_supported_assessment_type"},
	},
	{
		"name": "assessment_requires_lead_assessor",
		"description": "Assessments require a lead assessor.",
		"condition": {"operation": "create_assessment", "lead_assessor_present": False},
		"effect": {"decision": "deny", "reason": "assessment_lead_assessor_required", "required_action": "assign_lead_assessor"},
	},
	{
		"name": "assessment_requires_asset_scope",
		"description": "Assessments require a defined asset scope.",
		"condition": {"operation": "create_assessment", "asset_scope_present": False},
		"effect": {"decision": "deny", "reason": "assessment_asset_scope_required", "required_action": "define_asset_scope"},
	},
	{
		"name": "assessment_asset_type_supported",
		"description": "Assessment asset type must be from the supported list.",
		"condition": {"operation": "create_assessment", "asset_type_supported": False},
		"effect": {"decision": "deny", "reason": "assessment_asset_type_not_supported", "required_action": "select_supported_asset_type"},
	},
	{
		"name": "assessment_requires_start_date",
		"description": "Assessments require a start date.",
		"condition": {"operation": "create_assessment", "start_date_present": False},
		"effect": {"decision": "deny", "reason": "assessment_start_date_required", "required_action": "set_start_date"},
	},
	{
		"name": "assessment_requires_end_date",
		"description": "Assessments require an end date.",
		"condition": {"operation": "create_assessment", "end_date_present": False},
		"effect": {"decision": "deny", "reason": "assessment_end_date_required", "required_action": "set_end_date"},
	},
	{
		"name": "assessment_end_date_after_start",
		"description": "Assessment end date must be after start date.",
		"condition": {"operation": "create_assessment", "end_before_start": True},
		"effect": {"decision": "deny", "reason": "assessment_end_date_before_start", "required_action": "correct_assessment_dates"},
	},
	{
		"name": "vendor_risk_requires_vendor",
		"description": "Vendor risk assessments require a vendor reference.",
		"condition": {"operation": "create_assessment", "assessment_type": "vendor_risk", "vendor_present": False},
		"effect": {"decision": "deny", "reason": "vendor_required_for_vendor_risk_assessment", "required_action": "specify_vendor"},
	},
	{
		"name": "high_risk_assessment_requires_review",
		"description": "High-risk assessments require a mandatory review.",
		"condition": {"operation": "create_assessment", "high_risk_assessment": True, "review_recorded": False},
		"effect": {"decision": "require_review", "reason": "high_risk_assessment_review_required", "required_action": "record_assessment_review"},
	},
	# Assessment — update
	{
		"name": "closed_assessment_update_denied",
		"description": "Closed assessments cannot be updated.",
		"condition": {"operation": "update_assessment", "assessment_status": "closed"},
		"effect": {"decision": "deny", "reason": "closed_assessment_is_immutable", "required_action": "reopen_assessment_to_update"},
	},
	# Assessment — cancel
	{
		"name": "cancel_assessment_requires_reason",
		"description": "Cancelling an assessment requires a stated reason.",
		"condition": {"operation": "cancel_assessment", "cancellation_reason_present": False},
		"effect": {"decision": "deny", "reason": "cancellation_reason_required", "required_action": "record_cancellation_reason"},
	},
	# Findings — create
	{
		"name": "finding_requires_title",
		"description": "Assessment findings require a title.",
		"condition": {"operation": "raise_finding", "title_present": False},
		"effect": {"decision": "deny", "reason": "finding_title_required", "required_action": "set_finding_title"},
	},
	{
		"name": "finding_type_supported",
		"description": "Finding type must be from the supported list.",
		"condition": {"operation": "raise_finding", "finding_type_supported": False},
		"effect": {"decision": "deny", "reason": "finding_type_not_supported", "required_action": "select_supported_finding_type"},
	},
	{
		"name": "finding_risk_rating_supported",
		"description": "Finding risk rating must be from the supported list.",
		"condition": {"operation": "raise_finding", "risk_rating_supported": False},
		"effect": {"decision": "deny", "reason": "finding_risk_rating_not_supported", "required_action": "select_supported_risk_rating"},
	},
	{
		"name": "vulnerability_requires_cvss_score",
		"description": "Vulnerability findings require a CVSS score.",
		"condition": {"operation": "raise_finding", "finding_type": "vulnerability", "cvss_score_present": False},
		"effect": {"decision": "deny", "reason": "vulnerability_cvss_score_required", "required_action": "assign_cvss_score"},
	},
	{
		"name": "cvss_version_supported",
		"description": "CVSS version must be from the supported list.",
		"condition": {"operation": "raise_finding", "cvss_version_supported": False},
		"effect": {"decision": "deny", "reason": "cvss_version_not_supported", "required_action": "select_supported_cvss_version"},
	},
	{
		"name": "finding_requires_remediation_strategy",
		"description": "Findings require a remediation strategy.",
		"condition": {"operation": "raise_finding", "remediation_strategy_present": False},
		"effect": {"decision": "deny", "reason": "remediation_strategy_required", "required_action": "select_remediation_strategy"},
	},
	{
		"name": "remediation_strategy_supported",
		"description": "Remediation strategy must be from the supported list.",
		"condition": {"operation": "raise_finding", "remediation_strategy_supported": False},
		"effect": {"decision": "deny", "reason": "remediation_strategy_not_supported", "required_action": "select_supported_remediation_strategy"},
	},
	{
		"name": "finding_requires_owner",
		"description": "Findings require an owner.",
		"condition": {"operation": "raise_finding", "owner_present": False},
		"effect": {"decision": "deny", "reason": "finding_owner_required", "required_action": "assign_finding_owner"},
	},
	{
		"name": "critical_finding_requires_escalation",
		"description": "Critical findings require immediate escalation.",
		"condition": {"operation": "raise_finding", "risk_rating": "critical", "escalation_recorded": False},
		"effect": {"decision": "require_review", "reason": "critical_finding_escalation_required", "required_action": "escalate_critical_finding"},
	},
	# Findings — approve/reject/escalate
	{
		"name": "accept_finding_requires_risk_sign_off",
		"description": "Accepting a finding (no remediation) requires formal risk sign-off.",
		"condition": {"operation": "accept_finding", "risk_sign_off_present": False},
		"effect": {"decision": "deny", "reason": "risk_sign_off_required_to_accept_finding", "required_action": "record_risk_sign_off"},
	},
	{
		"name": "reject_finding_requires_evidence",
		"description": "Rejecting (false positive) a finding requires evidence.",
		"condition": {"operation": "reject_finding", "false_positive_evidence_present": False},
		"effect": {"decision": "deny", "reason": "evidence_required_to_reject_finding", "required_action": "attach_false_positive_evidence"},
	},
	{
		"name": "escalate_finding_requires_target",
		"description": "Finding escalation requires a target.",
		"condition": {"operation": "escalate_finding", "escalation_target_present": False},
		"effect": {"decision": "deny", "reason": "escalation_target_required", "required_action": "specify_escalation_target"},
	},
	# Remediation
	{
		"name": "remediation_requires_due_date",
		"description": "Remediation plans require a due date.",
		"condition": {"operation": "create_remediation_plan", "due_date_present": False},
		"effect": {"decision": "deny", "reason": "remediation_due_date_required", "required_action": "set_remediation_due_date"},
	},
	{
		"name": "remediation_closure_requires_evidence",
		"description": "Closing remediation requires supporting evidence.",
		"condition": {"operation": "close_remediation", "evidence_present": False},
		"effect": {"decision": "deny", "reason": "remediation_closure_evidence_required", "required_action": "attach_remediation_evidence"},
	},
	{
		"name": "defer_remediation_requires_approval",
		"description": "Deferring remediation requires approval.",
		"condition": {"operation": "defer_remediation", "approval_recorded": False},
		"effect": {"decision": "deny", "reason": "defer_approval_required", "required_action": "obtain_defer_approval"},
	},
	{
		"name": "defer_remediation_max_days",
		"description": "Remediation deferral cannot exceed the maximum.",
		"condition": {"operation": "defer_remediation", "defer_days_gt": 180},
		"effect": {"decision": "deny", "reason": "remediation_defer_exceeds_maximum", "required_action": "shorten_defer_duration"},
	},
	# Streaming infrastructure
	{
		"name": "rsa_batch_requires_bytewax",
		"description": "RSA batches require Bytewax coordination.",
		"condition": {"operation": "rsa_batch", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_rsa_batch_to_bytewax"},
	},
	{
		"name": "rsa_event_requires_bytewax",
		"description": "RSA events require Bytewax.",
		"condition": {"operation": "rsa_event", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_rsa_event_to_bytewax"},
	},
	# Agent governance
	{
		"name": "rsa_agent_runtime_supported",
		"description": "RSA agents must use an approved runtime.",
		"condition": {"operation": "register_rsa_agent", "agent_runtime_supported": False},
		"effect": {"decision": "deny", "reason": "rsa_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"},
	},
	{
		"name": "rsa_agent_role_supported",
		"description": "RSA agents must use an approved role.",
		"condition": {"operation": "register_rsa_agent", "agent_role_supported": False},
		"effect": {"decision": "deny", "reason": "rsa_agent_role_not_supported", "required_action": "select_supported_agent_role"},
	},
	{
		"name": "privileged_rsa_agent_action_requires_human_approval",
		"description": "Privileged RSA actions proposed by agents require human approval.",
		"condition": {"operation": "rsa_agent_action", "privileged_scope": True, "human_approval_recorded": False},
		"effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"},
	},
	# Domain-specific governance
	{
		"name": "dpia_required_for_high_risk_data_processing",
		"description": "High-risk data privacy assessments require a DPIA.",
		"condition": {"operation": "create_assessment", "assessment_type": "data_privacy_impact", "risk_rating": "high", "dpia_recorded": False},
		"effect": {"decision": "require_review", "reason": "dpia_required_for_high_risk_data_processing", "required_action": "conduct_dpia"},
	},
	{
		"name": "vendor_critical_tier_requires_annual_reassessment",
		"description": "Critical-tier vendors must be reassessed annually.",
		"condition": {"operation": "approve_vendor", "vendor_tier": "critical", "days_since_last_assessment_gt": 365},
		"effect": {"decision": "deny", "reason": "critical_vendor_annual_reassessment_required", "required_action": "schedule_vendor_reassessment"},
	},
	{
		"name": "pentest_findings_linked_to_risk_register",
		"description": "Penetration test critical findings must be linked to the risk register.",
		"condition": {"operation": "close_assessment", "assessment_type": "penetration_test", "critical_findings_linked_to_risk_register": False},
		"effect": {"decision": "deny", "reason": "pentest_critical_findings_must_link_risk_register", "required_action": "link_findings_to_risk_register"},
	},
	{
		"name": "sign_off_requires_independent_approver",
		"description": "Assessment sign-off requires an approver independent of the lead assessor.",
		"condition": {"operation": "sign_off_assessment", "approver_is_lead_assessor": True},
		"effect": {"decision": "deny", "reason": "assessment_sign_off_segregation_required", "required_action": "assign_independent_sign_off_approver"},
	},
	{
		"name": "cloud_security_review_requires_csp_report",
		"description": "Cloud security reviews require a Cloud Service Provider report.",
		"condition": {"operation": "create_assessment", "assessment_type": "cloud_security_review", "csp_report_present": False},
		"effect": {"decision": "deny", "reason": "csp_report_required_for_cloud_security_review", "required_action": "obtain_csp_report"},
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
			"api_prefix": "/grc-rsa/api/v1",
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
