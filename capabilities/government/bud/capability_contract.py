"""Executable capability contract for APG Budget Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "government_bud"
CAPABILITY_NAME = "Budget Management"
CAPABILITY_VERSION = "1.0.0"
BUD_EVENT_STREAM = "apg.government.bud.lifecycle"

SUPPORTED_BUDGET_TYPES = ["recurrent", "development", "supplementary", "emergency", "donor_funded", "internal_generation"]
SUPPORTED_VOTE_TYPES = ["programme", "project", "administrative", "statutory", "conditional_grant"]
SUPPORTED_REVISION_TYPES = ["reallocation", "supplementary_estimate", "virements", "donor_amendment", "treasury_directive"]
SUPPORTED_COMMITMENT_TYPES = ["lpo", "contract", "payroll", "utility", "grant", "pension", "standing_order"]
SUPPORTED_EXPENDITURE_TYPES = ["goods_services", "personnel_emoluments", "development_expenditure", "transfers", "debt_service"]
SUPPORTED_FUND_SOURCES = ["exchequer", "aia", "donor_grant", "donor_loan", "conditional_grant", "own_revenue"]
SUPPORTED_FISCAL_PERIODS = ["q1", "q2", "q3", "q4", "annual", "mid_year"]
SUPPORTED_APPROVAL_STATUSES = ["draft", "submitted", "under_review", "approved", "rejected", "returned", "withdrawn"]
SUPPORTED_REPORT_TYPES = ["budget_outturn", "commitment_report", "variance_analysis", "treasury_submission", "audit_support", "mid_year_review"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["budget_analyst", "vote_controller", "commitment_checker", "report_generator", "revision_reviewer"]
SUPPORTED_CLASSIFICATIONS = ["public", "restricted", "confidential"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"budget": {
		"supported_budget_types": SUPPORTED_BUDGET_TYPES,
		"supported_fund_sources": SUPPORTED_FUND_SOURCES,
		"vote_required": True,
		"approver_required": True,
		"evidence_required": True,
	},
	"votes": {
		"supported_vote_types": SUPPORTED_VOTE_TYPES,
		"budget_required": True,
		"vote_code_required": True,
		"evidence_required": True,
	},
	"revisions": {
		"supported_revision_types": SUPPORTED_REVISION_TYPES,
		"budget_required": True,
		"approval_required": True,
		"evidence_required": True,
		"treasury_notification_required": True,
	},
	"commitments": {
		"supported_commitment_types": SUPPORTED_COMMITMENT_TYPES,
		"vote_required": True,
		"sufficient_balance_required": True,
		"approval_required": True,
		"evidence_required": True,
	},
	"expenditures": {
		"supported_expenditure_types": SUPPORTED_EXPENDITURE_TYPES,
		"commitment_required": True,
		"approval_required": True,
		"evidence_required": True,
	},
	"reports": {
		"supported_report_types": SUPPORTED_REPORT_TYPES,
		"supported_fiscal_periods": SUPPORTED_FISCAL_PERIODS,
		"budget_required": True,
		"evidence_required": True,
	},
	"approvals": {
		"supported_statuses": SUPPORTED_APPROVAL_STATUSES,
		"approver_required": True,
		"evidence_required": True,
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
		"commitment_without_balance_denied": True,
		"expenditure_without_commitment_denied": True,
		"revision_without_treasury_approval_denied": True,
		"cross_vote_reallocation_requires_approval": True,
		"negative_vote_balance_denied": True,
		"unapproved_supplementary_denied": True,
		"evidence_fabrication_denied": True,
	},
	"observability": {"event_stream": BUD_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"workflow": "wflo",
		"compliance": "comp",
		"monitoring": "moni",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_budgets": True,
		"enable_votes": True,
		"enable_revisions": True,
		"enable_commitments": True,
		"enable_expenditures": True,
		"enable_reports": True,
		"enable_approvals": True,
		"enable_reviews": True,
		"enable_agents": True,
	},
	"theme": {"default_theme": "government_bud_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"budget_programme_workflow",
	"vote_accounting_workflow",
	"budget_revision_workflow",
	"commitment_control_workflow",
	"expenditure_recording_workflow",
	"fiscal_reporting_workflow",
	"budget_approval_workflow",
	"budget_review_workflow",
	"budget_agent_workflow",
	"treasury_submission_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/government-bud/dashboard", "component": "BudgetDashboard", "permission": "government_bud:view", "nav_group": "Overview"},
	{"name": "budgets", "path": "/government-bud/budgets", "component": "BudgetProgrammeConsole", "permission": "government_bud:budgets", "nav_group": "Planning"},
	{"name": "votes", "path": "/government-bud/votes", "component": "VoteAccountingLedger", "permission": "government_bud:votes", "nav_group": "Planning"},
	{"name": "revisions", "path": "/government-bud/revisions", "component": "BudgetRevisionConsole", "permission": "government_bud:revisions", "nav_group": "Revisions"},
	{"name": "commitments", "path": "/government-bud/commitments", "component": "CommitmentControlQueue", "permission": "government_bud:commitments", "nav_group": "Execution"},
	{"name": "expenditures", "path": "/government-bud/expenditures", "component": "ExpenditureLedger", "permission": "government_bud:expenditures", "nav_group": "Execution"},
	{"name": "reports", "path": "/government-bud/reports", "component": "FiscalReportingConsole", "permission": "government_bud:reports", "nav_group": "Reporting"},
	{"name": "approvals", "path": "/government-bud/approvals", "component": "BudgetApprovalQueue", "permission": "government_bud:approvals", "nav_group": "Governance"},
	{"name": "reviews", "path": "/government-bud/reviews", "component": "BudgetReviewConsole", "permission": "government_bud:reviews", "nav_group": "Governance"},
	{"name": "treasury", "path": "/government-bud/treasury", "component": "TreasurySubmissionConsole", "permission": "government_bud:treasury", "nav_group": "Reporting"},
	{"name": "agents", "path": "/government-bud/agents", "component": "BudgetAgentWorkbench", "permission": "government_bud:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/government-bud/settings", "component": "BudgetSettings", "permission": "government_bud:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "government_bud_control",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#0F766E",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#991B1B",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"budgets": {"icon": "calculator", "status_indicator": "budget-status-chip"},
		"votes": {"icon": "list-ordered", "status_indicator": "vote-balance-chip"},
		"revisions": {"icon": "git-branch", "status_indicator": "revision-status-chip"},
		"commitments": {"icon": "lock", "status_indicator": "commitment-status-chip"},
		"expenditures": {"icon": "credit-card", "status_indicator": "expenditure-status-chip"},
		"reports": {"icon": "file-text", "status_indicator": "report-type-chip"},
		"approvals": {"icon": "check-circle", "status_indicator": "approval-status-chip"},
		"reviews": {"icon": "clipboard-check", "status_indicator": "review-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": BUD_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"budget_recorded",
		"vote_recorded",
		"budget_revision_recorded",
		"commitment_recorded",
		"expenditure_recorded",
		"fiscal_report_generated",
		"budget_approved",
		"budget_reviewed",
		"budget_agent_registered",
		"treasury_submission_recorded",
	],
	"guardrails": [
		"budget_batch_requires_bytewax",
		"commitment_without_balance_denied",
		"expenditure_without_commitment_denied",
		"revision_without_treasury_approval_denied",
		"negative_vote_balance_denied",
		"unapproved_supplementary_denied",
		"evidence_fabrication_denied",
		"privileged_budget_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "budget_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "budget_policy_required", "required_action": "attach_budget_policy"}},
	{"name": "budget_type_supported", "condition": {"operation": "record_budget", "budget_type_supported": False}, "effect": {"decision": "deny", "reason": "budget_type_not_supported", "required_action": "select_supported_budget_type"}},
	{"name": "budget_vote_required", "condition": {"operation": "record_budget", "vote_present": False}, "effect": {"decision": "deny", "reason": "vote_required", "required_action": "attach_vote_reference"}},
	{"name": "budget_fund_source_supported", "condition": {"operation": "record_budget", "fund_source_supported": False}, "effect": {"decision": "deny", "reason": "fund_source_not_supported", "required_action": "select_supported_fund_source"}},
	{"name": "budget_approver_required", "condition": {"operation": "record_budget", "approver_present": False}, "effect": {"decision": "deny", "reason": "approver_required", "required_action": "assign_approver"}},
	{"name": "budget_evidence_required", "condition": {"operation": "record_budget", "evidence_present": False}, "effect": {"decision": "deny", "reason": "budget_evidence_required", "required_action": "attach_budget_evidence"}},
	{"name": "vote_type_supported", "condition": {"operation": "record_vote", "vote_type_supported": False}, "effect": {"decision": "deny", "reason": "vote_type_not_supported", "required_action": "select_supported_vote_type"}},
	{"name": "vote_code_required", "condition": {"operation": "record_vote", "vote_code_present": False}, "effect": {"decision": "deny", "reason": "vote_code_required", "required_action": "assign_vote_code"}},
	{"name": "vote_budget_required", "condition": {"operation": "record_vote", "budget_present": False}, "effect": {"decision": "deny", "reason": "budget_reference_required", "required_action": "select_budget"}},
	{"name": "vote_evidence_required", "condition": {"operation": "record_vote", "evidence_present": False}, "effect": {"decision": "deny", "reason": "vote_evidence_required", "required_action": "attach_vote_evidence"}},
	{"name": "revision_type_supported", "condition": {"operation": "record_revision", "revision_type_supported": False}, "effect": {"decision": "deny", "reason": "revision_type_not_supported", "required_action": "select_supported_revision_type"}},
	{"name": "revision_budget_required", "condition": {"operation": "record_revision", "budget_present": False}, "effect": {"decision": "deny", "reason": "budget_reference_required", "required_action": "select_budget"}},
	{"name": "revision_approval_required", "condition": {"operation": "record_revision", "approval_present": False}, "effect": {"decision": "deny", "reason": "revision_approval_required", "required_action": "attach_revision_approval"}},
	{"name": "revision_treasury_required", "condition": {"operation": "record_revision", "revision_type_supported": True, "treasury_notification_present": False}, "effect": {"decision": "deny", "reason": "treasury_notification_required", "required_action": "notify_treasury"}},
	{"name": "commitment_type_supported", "condition": {"operation": "record_commitment", "commitment_type_supported": False}, "effect": {"decision": "deny", "reason": "commitment_type_not_supported", "required_action": "select_supported_commitment_type"}},
	{"name": "commitment_vote_required", "condition": {"operation": "record_commitment", "vote_present": False}, "effect": {"decision": "deny", "reason": "vote_required", "required_action": "select_vote"}},
	{"name": "commitment_balance_required", "condition": {"operation": "record_commitment", "sufficient_balance": False}, "effect": {"decision": "deny", "reason": "insufficient_vote_balance", "required_action": "check_vote_balance"}},
	{"name": "commitment_approval_required", "condition": {"operation": "record_commitment", "approval_present": False}, "effect": {"decision": "deny", "reason": "commitment_approval_required", "required_action": "attach_commitment_approval"}},
	{"name": "commitment_evidence_required", "condition": {"operation": "record_commitment", "evidence_present": False}, "effect": {"decision": "deny", "reason": "commitment_evidence_required", "required_action": "attach_commitment_evidence"}},
	{"name": "expenditure_type_supported", "condition": {"operation": "record_expenditure", "expenditure_type_supported": False}, "effect": {"decision": "deny", "reason": "expenditure_type_not_supported", "required_action": "select_supported_expenditure_type"}},
	{"name": "expenditure_commitment_required", "condition": {"operation": "record_expenditure", "commitment_present": False}, "effect": {"decision": "deny", "reason": "commitment_required", "required_action": "select_commitment"}},
	{"name": "expenditure_approval_required", "condition": {"operation": "record_expenditure", "approval_present": False}, "effect": {"decision": "deny", "reason": "expenditure_approval_required", "required_action": "attach_expenditure_approval"}},
	{"name": "expenditure_evidence_required", "condition": {"operation": "record_expenditure", "evidence_present": False}, "effect": {"decision": "deny", "reason": "expenditure_evidence_required", "required_action": "attach_expenditure_evidence"}},
	{"name": "report_type_supported", "condition": {"operation": "generate_report", "report_type_supported": False}, "effect": {"decision": "deny", "reason": "report_type_not_supported", "required_action": "select_supported_report_type"}},
	{"name": "report_fiscal_period_supported", "condition": {"operation": "generate_report", "fiscal_period_supported": False}, "effect": {"decision": "deny", "reason": "fiscal_period_not_supported", "required_action": "select_supported_fiscal_period"}},
	{"name": "report_budget_required", "condition": {"operation": "generate_report", "budget_present": False}, "effect": {"decision": "deny", "reason": "budget_reference_required", "required_action": "select_budget"}},
	{"name": "negative_vote_balance_denied", "condition": {"operation": "record_commitment", "negative_balance": True}, "effect": {"decision": "deny", "reason": "negative_vote_balance_denied", "required_action": "reduce_commitment_amount"}},
	{"name": "cross_vote_reallocation_requires_approval", "condition": {"operation": "record_revision", "cross_vote_reallocation": True, "approval_present": False}, "effect": {"decision": "deny", "reason": "cross_vote_reallocation_requires_approval", "required_action": "obtain_treasury_approval"}},
	{"name": "budget_batch_requires_bytewax", "condition": {"operation": "budget_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_budget_batch_to_bytewax"}},
	{"name": "budget_agent_runtime_supported", "condition": {"operation": "register_budget_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "budget_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "budget_agent_role_supported", "condition": {"operation": "register_budget_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "budget_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "budget_agent_name_required", "condition": {"operation": "register_budget_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "budget_agent_name_required", "required_action": "name_budget_agent"}},
	{"name": "budget_agent_scope_required", "condition": {"operation": "register_budget_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "budget_agent_scope_required", "required_action": "bound_budget_agent_scope"}},
	{"name": "privileged_budget_agent_action_requires_human_approval", "condition": {"operation": "budget_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "evidence_fabrication_denied", "condition": {"operation": "budget_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_denied", "required_action": "remove_evidence_fabrication_scope"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
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
			"api_prefix": "/government-bud/api/v1",
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
