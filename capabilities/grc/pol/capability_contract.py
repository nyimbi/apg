"""Executable capability contract for GRC Policy Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "grc_pol"
CAPABILITY_NAME = "Policy Management"
CAPABILITY_VERSION = "1.0.0"
POL_EVENT_STREAM = "apg.grc.pol.lifecycle"

SUPPORTED_POLICY_TYPES = [
	"corporate", "information_security", "data_privacy", "hr", "finance",
	"operational", "compliance", "acceptable_use", "third_party", "bcdr",
]
SUPPORTED_POLICY_STATUSES = [
	"draft", "in_review", "approved", "published", "superseded",
	"archived", "withdrawn", "under_revision",
]
SUPPORTED_REVIEW_FREQUENCIES = ["annual", "biannual", "quarterly", "ad_hoc", "triggered"]
SUPPORTED_POLICY_SCOPES = [
	"organization_wide", "department", "system", "product", "jurisdiction", "third_party",
]
SUPPORTED_ACKNOWLEDGEMENT_METHODS = [
	"electronic_signature", "checkbox_acceptance", "training_completion", "manager_attestation",
]
SUPPORTED_POL_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_POL_AGENT_ROLES = [
	"policy_drafter",
	"policy_reviewer",
	"gap_analyst",
	"acknowledgement_tracker",
	"exception_reviewer",
	"publication_reviewer",
]
SUPPORTED_EXCEPTION_TYPES = [
	"temporary_exemption", "scope_exclusion", "deadline_extension", "control_alternative",
]
SUPPORTED_EXCEPTION_STATUSES = ["pending", "approved", "rejected", "expired", "revoked"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"policies": {
		"title_required": True,
		"type_required": True,
		"supported_types": SUPPORTED_POLICY_TYPES,
		"supported_statuses": SUPPORTED_POLICY_STATUSES,
		"owner_required": True,
		"scope_required": True,
		"supported_scopes": SUPPORTED_POLICY_SCOPES,
		"review_frequency_required": True,
		"supported_review_frequencies": SUPPORTED_REVIEW_FREQUENCIES,
		"effective_date_required": True,
		"review_date_required": True,
		"version_required": True,
		"approval_required": True,
		"segregation_of_duties": True,
	},
	"acknowledgements": {
		"method_required": True,
		"supported_methods": SUPPORTED_ACKNOWLEDGEMENT_METHODS,
		"deadline_required": True,
		"reminder_frequency_days": 7,
		"overdue_escalation_enabled": True,
	},
	"exceptions": {
		"type_required": True,
		"supported_types": SUPPORTED_EXCEPTION_TYPES,
		"supported_statuses": SUPPORTED_EXCEPTION_STATUSES,
		"requestor_required": True,
		"approver_required": True,
		"rationale_required": True,
		"expiration_required": True,
		"max_exception_days": 365,
		"risk_assessment_required": True,
	},
	"reviews": {
		"reviewer_required": True,
		"review_note_required": True,
		"outcome_required": True,
		"reviewer_cannot_be_owner": True,
	},
	"pol_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_POL_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_POL_AGENT_ROLES,
		"max_autonomous_scope": "draft_and_recommend",
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
		"event_stream": POL_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_policy_events": True,
		"emit_acknowledgement_events": True,
		"emit_exception_events": True,
		"emit_review_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit_log": "adapter",
		"notification": "adapter",
		"document_management": "adapter",
		"workflow_orchestration": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
		"multi_tenancy": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_policies": True,
		"enable_acknowledgements": True,
		"enable_exceptions": True,
		"enable_reviews": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "grc_pol_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"policy_lifecycle_management",
	"policy_acknowledgement_workflow",
	"policy_exception_workflow",
	"policy_review_workflow",
	"policy_publication_workflow",
	"policy_dashboard_service",
	"pol_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"mten",
	"conf",
	"ntfy",
	"wflo",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/grc-pol/dashboard", "component": "PolicyDashboard", "permission": "grc_pol:view", "nav_group": "Overview"},
	{"name": "policies", "path": "/grc-pol/policies", "component": "PolicyLibrary", "permission": "grc_pol:manage_policies", "nav_group": "Policies"},
	{"name": "policy_detail", "path": "/grc-pol/policies/:id", "component": "PolicyDetail", "permission": "grc_pol:view", "nav_group": "Policies"},
	{"name": "acknowledgements", "path": "/grc-pol/acknowledgements", "component": "AcknowledgementTracker", "permission": "grc_pol:manage_acknowledgements", "nav_group": "Compliance"},
	{"name": "exceptions", "path": "/grc-pol/exceptions", "component": "PolicyExceptionWorkbench", "permission": "grc_pol:manage_exceptions", "nav_group": "Governance"},
	{"name": "reviews", "path": "/grc-pol/reviews", "component": "PolicyReviewQueue", "permission": "grc_pol:review", "nav_group": "Governance"},
	{"name": "review_calendar", "path": "/grc-pol/review-calendar", "component": "ReviewCalendar", "permission": "grc_pol:view", "nav_group": "Planning"},
	{"name": "gap_analysis", "path": "/grc-pol/gap-analysis", "component": "PolicyGapAnalysis", "permission": "grc_pol:view", "nav_group": "Analysis"},
	{"name": "agents", "path": "/grc-pol/agents", "component": "PolicyAgentWorkbench", "permission": "grc_pol:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/grc-pol/settings", "component": "PolicySettings", "permission": "grc_pol:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "grc_pol_control",
	"tokens": {
		"color.primary": "#1A3C34",
		"color.accent": "#3182CE",
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
		"policies": {"icon": "book-open", "status_indicator": "policy-pill", "visual": "policy-library"},
		"acknowledgements": {"icon": "user-check", "visual": "acknowledgement-tracker", "status_style": "ack-chip"},
		"exceptions": {"icon": "shield-off", "visual": "exception-register", "status_style": "expiry-chip"},
		"reviews": {"icon": "clipboard-check", "visual": "review-queue", "status_style": "review-chip"},
		"gap_analysis": {"icon": "bar-chart-2", "visual": "gap-heatmap", "status_style": "gap-chip"},
		"agents": {"icon": "bot", "visual": "agent-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": POL_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"policy_drafted",
		"policy_submitted_for_review",
		"policy_review_completed",
		"policy_approved",
		"policy_rejected",
		"policy_published",
		"policy_superseded",
		"policy_archived",
		"policy_withdrawn",
		"policy_review_due",
		"policy_review_overdue",
		"acknowledgement_requested",
		"acknowledgement_completed",
		"acknowledgement_overdue",
		"exception_requested",
		"exception_approved",
		"exception_rejected",
		"exception_expired",
		"pol_agent_registered",
		"pol_agent_action_approved",
	],
	"states": SUPPORTED_POLICY_STATUSES + SUPPORTED_EXCEPTION_STATUSES + ["queued", "failed"],
	"guardrails": [
		"pol_batch_requires_bytewax",
		"pol_event_requires_bytewax",
		"privileged_pol_agent_action_requires_human_approval",
		"cross_tenant_event_denied",
	],
}


RULES: list[dict[str, Any]] = [
	# Tenant and policy governance
	{
		"name": "tenant_context_required",
		"description": "Policy operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
	},
	{
		"name": "cross_tenant_access_denied",
		"description": "Policy data may not be accessed across tenant boundaries.",
		"condition": {"cross_tenant_access": True},
		"effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_identity"},
	},
	{
		"name": "pol_write_requires_policy",
		"description": "Policy management writes require policy attachment.",
		"condition": {"operation_type": "write", "policy_attached": False},
		"effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"},
	},
	{
		"name": "privilege_escalation_denied",
		"description": "Users may not grant policy permissions exceeding their own level.",
		"condition": {"operation": "grant_pol_permission", "grant_exceeds_grantor_permission": True},
		"effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "reduce_grant_to_grantor_level"},
	},
	{
		"name": "admin_operation_requires_mfa",
		"description": "Admin-level policy operations require MFA.",
		"condition": {"permission_required": "admin", "mfa_verified": False},
		"effect": {"decision": "deny", "reason": "mfa_required_for_admin", "required_action": "complete_mfa"},
	},
	# Policy — create
	{
		"name": "policy_requires_title",
		"description": "Policies require a title.",
		"condition": {"operation": "create_policy", "title_present": False},
		"effect": {"decision": "deny", "reason": "policy_title_required", "required_action": "set_policy_title"},
	},
	{
		"name": "policy_type_supported",
		"description": "Policy type must be from the supported list.",
		"condition": {"operation": "create_policy", "policy_type_supported": False},
		"effect": {"decision": "deny", "reason": "policy_type_not_supported", "required_action": "select_supported_policy_type"},
	},
	{
		"name": "policy_requires_owner",
		"description": "Policies require an owner.",
		"condition": {"operation": "create_policy", "owner_present": False},
		"effect": {"decision": "deny", "reason": "policy_owner_required", "required_action": "assign_policy_owner"},
	},
	{
		"name": "policy_scope_supported",
		"description": "Policy scope must be from the supported list.",
		"condition": {"operation": "create_policy", "scope_type_supported": False},
		"effect": {"decision": "deny", "reason": "policy_scope_not_supported", "required_action": "select_supported_policy_scope"},
	},
	{
		"name": "policy_requires_effective_date",
		"description": "Policies require an effective date.",
		"condition": {"operation": "create_policy", "effective_date_present": False},
		"effect": {"decision": "deny", "reason": "policy_effective_date_required", "required_action": "set_effective_date"},
	},
	{
		"name": "policy_requires_review_date",
		"description": "Policies require a scheduled review date.",
		"condition": {"operation": "create_policy", "review_date_present": False},
		"effect": {"decision": "deny", "reason": "policy_review_date_required", "required_action": "set_review_date"},
	},
	{
		"name": "policy_review_frequency_supported",
		"description": "Policy review frequency must be from the supported list.",
		"condition": {"operation": "create_policy", "review_frequency_supported": False},
		"effect": {"decision": "deny", "reason": "policy_review_frequency_not_supported", "required_action": "select_supported_review_frequency"},
	},
	{
		"name": "policy_requires_version",
		"description": "Policies require a version identifier.",
		"condition": {"operation": "create_policy", "version_present": False},
		"effect": {"decision": "deny", "reason": "policy_version_required", "required_action": "set_policy_version"},
	},
	# Policy — update
	{
		"name": "published_policy_update_denied",
		"description": "Published policies cannot be directly updated; a new revision must be initiated.",
		"condition": {"operation": "update_policy", "policy_status": "published"},
		"effect": {"decision": "deny", "reason": "published_policy_requires_revision_workflow", "required_action": "initiate_policy_revision"},
	},
	{
		"name": "archived_policy_update_denied",
		"description": "Archived policies cannot be updated.",
		"condition": {"operation": "update_policy", "policy_status": "archived"},
		"effect": {"decision": "deny", "reason": "archived_policy_is_immutable", "required_action": "create_new_policy_version"},
	},
	# Policy — approve
	{
		"name": "publish_requires_approval",
		"description": "Policies must be approved before publication.",
		"condition": {"operation": "publish_policy", "approved": False},
		"effect": {"decision": "deny", "reason": "policy_approval_required_for_publication", "required_action": "approve_policy"},
	},
	{
		"name": "approval_segregation_required",
		"description": "Policy owner cannot approve their own policy.",
		"condition": {"operation": "approve_policy", "approver_is_owner": True},
		"effect": {"decision": "deny", "reason": "policy_approval_segregation_required", "required_action": "assign_independent_approver"},
	},
	# Policy — reject
	{
		"name": "reject_policy_requires_reason",
		"description": "Policy rejection requires a reason.",
		"condition": {"operation": "reject_policy", "rejection_reason_present": False},
		"effect": {"decision": "deny", "reason": "policy_rejection_reason_required", "required_action": "record_rejection_reason"},
	},
	# Policy — supersede / archive
	{
		"name": "supersede_requires_successor",
		"description": "Superseding a policy requires linking a successor policy.",
		"condition": {"operation": "supersede_policy", "successor_id_present": False},
		"effect": {"decision": "deny", "reason": "successor_policy_required_for_supersede", "required_action": "link_successor_policy"},
	},
	{
		"name": "archive_requires_reason",
		"description": "Archiving a policy requires a stated reason.",
		"condition": {"operation": "archive_policy", "archive_reason_present": False},
		"effect": {"decision": "deny", "reason": "archive_reason_required", "required_action": "record_archive_reason"},
	},
	# Review workflow
	{
		"name": "review_requires_reviewer",
		"description": "Policy reviews require a reviewer.",
		"condition": {"operation": "submit_review", "reviewer_present": False},
		"effect": {"decision": "deny", "reason": "policy_reviewer_required", "required_action": "assign_reviewer"},
	},
	{
		"name": "review_requires_note",
		"description": "Policy reviews require a review note.",
		"condition": {"operation": "submit_review", "review_note_present": False},
		"effect": {"decision": "deny", "reason": "policy_review_note_required", "required_action": "record_review_note"},
	},
	{
		"name": "reviewer_cannot_be_owner",
		"description": "Policy reviewer cannot be the policy owner.",
		"condition": {"operation": "submit_review", "reviewer_is_owner": True},
		"effect": {"decision": "deny", "reason": "reviewer_owner_segregation_required", "required_action": "assign_independent_reviewer"},
	},
	# Acknowledgements
	{
		"name": "acknowledgement_method_supported",
		"description": "Acknowledgement method must be from the supported list.",
		"condition": {"operation": "request_acknowledgement", "ack_method_supported": False},
		"effect": {"decision": "deny", "reason": "acknowledgement_method_not_supported", "required_action": "select_supported_acknowledgement_method"},
	},
	{
		"name": "acknowledgement_requires_deadline",
		"description": "Acknowledgement requests require a deadline.",
		"condition": {"operation": "request_acknowledgement", "deadline_present": False},
		"effect": {"decision": "deny", "reason": "acknowledgement_deadline_required", "required_action": "set_acknowledgement_deadline"},
	},
	{
		"name": "overdue_acknowledgement_triggers_escalation",
		"description": "Overdue acknowledgements trigger escalation.",
		"condition": {"operation": "check_acknowledgements", "acknowledgement_overdue": True, "escalation_recorded": False},
		"effect": {"decision": "require_review", "reason": "overdue_acknowledgement_escalation_required", "required_action": "escalate_overdue_acknowledgement"},
	},
	# Exceptions
	{
		"name": "exception_type_supported",
		"description": "Policy exception type must be from the supported list.",
		"condition": {"operation": "request_exception", "exception_type_supported": False},
		"effect": {"decision": "deny", "reason": "exception_type_not_supported", "required_action": "select_supported_exception_type"},
	},
	{
		"name": "exception_requires_rationale",
		"description": "Policy exceptions require a written rationale.",
		"condition": {"operation": "request_exception", "rationale_present": False},
		"effect": {"decision": "deny", "reason": "exception_rationale_required", "required_action": "record_exception_rationale"},
	},
	{
		"name": "exception_requires_expiration",
		"description": "Policy exceptions require an expiration date.",
		"condition": {"operation": "request_exception", "expiration_present": False},
		"effect": {"decision": "deny", "reason": "exception_expiration_required", "required_action": "set_exception_expiration"},
	},
	{
		"name": "exception_max_duration",
		"description": "Exception duration must not exceed the maximum.",
		"condition": {"operation": "request_exception", "exception_days_gt": 365},
		"effect": {"decision": "deny", "reason": "exception_duration_exceeds_maximum", "required_action": "shorten_exception_duration"},
	},
	{
		"name": "exception_requires_risk_assessment",
		"description": "Policy exceptions require an associated risk assessment.",
		"condition": {"operation": "request_exception", "risk_assessment_present": False},
		"effect": {"decision": "deny", "reason": "risk_assessment_required_for_exception", "required_action": "attach_risk_assessment"},
	},
	{
		"name": "exception_requires_approver",
		"description": "Policy exceptions require an independent approver.",
		"condition": {"operation": "approve_exception", "approver_is_requestor": True},
		"effect": {"decision": "deny", "reason": "exception_approval_segregation_required", "required_action": "assign_independent_approver"},
	},
	# Streaming infrastructure
	{
		"name": "pol_batch_requires_bytewax",
		"description": "Policy batches require Bytewax coordination.",
		"condition": {"operation": "pol_batch", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_pol_batch_to_bytewax"},
	},
	{
		"name": "pol_event_requires_bytewax",
		"description": "Policy events require Bytewax.",
		"condition": {"operation": "pol_event", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_pol_event_to_bytewax"},
	},
	# Agent governance
	{
		"name": "pol_agent_runtime_supported",
		"description": "Policy agents must use an approved runtime.",
		"condition": {"operation": "register_pol_agent", "agent_runtime_supported": False},
		"effect": {"decision": "deny", "reason": "pol_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"},
	},
	{
		"name": "pol_agent_role_supported",
		"description": "Policy agents must use an approved role.",
		"condition": {"operation": "register_pol_agent", "agent_role_supported": False},
		"effect": {"decision": "deny", "reason": "pol_agent_role_not_supported", "required_action": "select_supported_agent_role"},
	},
	{
		"name": "privileged_pol_agent_action_requires_human_approval",
		"description": "Privileged policy actions proposed by agents require human approval.",
		"condition": {"operation": "pol_agent_action", "privileged_scope": True, "human_approval_recorded": False},
		"effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"},
	},
	# Domain-specific governance
	{
		"name": "regulatory_policy_requires_legal_sign_off",
		"description": "Compliance and regulatory policies require legal sign-off before publication.",
		"condition": {"operation": "publish_policy", "policy_type": "compliance", "legal_sign_off_present": False},
		"effect": {"decision": "deny", "reason": "legal_sign_off_required_for_compliance_policy", "required_action": "obtain_legal_sign_off"},
	},
	{
		"name": "overdue_review_blocks_publication",
		"description": "A policy with an overdue review date cannot be published.",
		"condition": {"operation": "publish_policy", "review_date_overdue": True},
		"effect": {"decision": "deny", "reason": "overdue_review_blocks_publication", "required_action": "complete_overdue_review"},
	},
	{
		"name": "gap_analysis_required_for_new_framework",
		"description": "Introducing a new framework-scoped policy requires a gap analysis.",
		"condition": {"operation": "create_policy", "new_framework_scope": True, "gap_analysis_present": False},
		"effect": {"decision": "require_review", "reason": "gap_analysis_required_for_new_framework_policy", "required_action": "conduct_gap_analysis"},
	},
	{
		"name": "mandatory_policy_acknowledgement_all_in_scope",
		"description": "Published mandatory policies must have acknowledgements requested for all in-scope principals.",
		"condition": {"operation": "publish_policy", "mandatory_policy": True, "acknowledgement_requested_all_in_scope": False},
		"effect": {"decision": "require_review", "reason": "mandatory_policy_acknowledgement_required_all_in_scope", "required_action": "request_acknowledgement_for_all_in_scope"},
	},
	{
		"name": "bcdr_policy_requires_annual_test",
		"description": "BCDR policies must reference an annual test result.",
		"condition": {"operation": "publish_policy", "policy_type": "bcdr", "annual_test_reference_present": False},
		"effect": {"decision": "deny", "reason": "bcdr_policy_requires_annual_test_reference", "required_action": "attach_bcdr_test_reference"},
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
			"api_prefix": "/grc-pol/api/v1",
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
