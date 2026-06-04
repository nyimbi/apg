"""Executable capability contract for APG Time & Expense Management (tex)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "ppm_tex"
CAPABILITY_NAME = "Time & Expense Management"
CAPABILITY_VERSION = "1.0.0"
TEX_EVENT_STREAM = "apg.ppm.tex.lifecycle"

# ── Supported enum values ────────────────────────────────────────────────────
SUPPORTED_TIMESHEET_STATUSES = ["draft", "submitted", "under_review", "approved", "rejected", "paid", "cancelled"]
SUPPORTED_TIME_ENTRY_TYPES = ["regular", "overtime", "holiday", "sick_leave", "training", "travel", "admin", "bench"]
SUPPORTED_BILLABLE_STATUSES = ["billable", "non_billable", "not_to_exceed", "pro_bono", "internal"]
SUPPORTED_EXPENSE_STATUSES = ["draft", "submitted", "under_review", "approved", "rejected", "reimbursed", "cancelled"]
SUPPORTED_EXPENSE_CATEGORIES = ["travel_airfare", "travel_accommodation", "travel_ground", "meals_entertainment", "client_entertainment", "office_supplies", "software_licenses", "training_conference", "communications", "professional_fees", "other"]
SUPPORTED_REIMBURSEMENT_METHODS = ["payroll", "direct_bank_transfer", "expense_card_credit", "cheque", "petty_cash"]
SUPPORTED_APPROVAL_WORKFLOWS = ["single_approver", "project_manager_first", "finance_then_pm", "committee", "auto_approve_below_threshold"]
SUPPORTED_PERIOD_TYPES = ["daily", "weekly", "bi_weekly", "semi_monthly", "monthly"]
SUPPORTED_BILLING_RATE_TYPES = ["standard", "preferential", "overtime", "blended", "not_to_exceed"]
SUPPORTED_MILEAGE_UNITS = ["km", "miles"]
SUPPORTED_RECEIPT_STATUSES = ["not_required", "pending_upload", "uploaded", "verified", "rejected"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["timesheet_reviewer", "expense_auditor", "billing_analyst", "reimbursement_processor", "compliance_monitor"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS", "ETB"]

PROVIDES = [
	"timesheet_entry_and_management",
	"expense_claim_workflow",
	"approval_workflow_engine",
	"billable_hour_tracking",
	"reimbursement_processing",
	"project_time_reporting",
	"billing_rate_management",
	"compliance_and_policy_enforcement",
	"multi_currency_expense_management",
	"audit_trail_for_time_and_expenses",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/ppm-tex/dashboard", "component": "TexDashboard", "permission": "ppm_tex:view", "nav_group": "Overview"},
	{"name": "my_timesheets", "path": "/ppm-tex/timesheets/my", "component": "MyTimesheetList", "permission": "ppm_tex:timesheets", "nav_group": "Timesheets"},
	{"name": "timesheet_entry", "path": "/ppm-tex/timesheets/entry", "component": "TimesheetEntryForm", "permission": "ppm_tex:timesheets", "nav_group": "Timesheets"},
	{"name": "timesheet_approvals", "path": "/ppm-tex/timesheets/approvals", "component": "TimesheetApprovalQueue", "permission": "ppm_tex:approve_timesheets", "nav_group": "Approvals"},
	{"name": "my_expenses", "path": "/ppm-tex/expenses/my", "component": "MyExpenseList", "permission": "ppm_tex:expenses", "nav_group": "Expenses"},
	{"name": "expense_claim", "path": "/ppm-tex/expenses/claim", "component": "ExpenseClaimForm", "permission": "ppm_tex:expenses", "nav_group": "Expenses"},
	{"name": "expense_approvals", "path": "/ppm-tex/expenses/approvals", "component": "ExpenseApprovalQueue", "permission": "ppm_tex:approve_expenses", "nav_group": "Approvals"},
	{"name": "billable_hours", "path": "/ppm-tex/billable", "component": "BillableHourTracker", "permission": "ppm_tex:billing", "nav_group": "Billing"},
	{"name": "billing_rates", "path": "/ppm-tex/rates", "component": "BillingRateTable", "permission": "ppm_tex:rates", "nav_group": "Billing"},
	{"name": "reimbursements", "path": "/ppm-tex/reimbursements", "component": "ReimbursementConsole", "permission": "ppm_tex:reimburse", "nav_group": "Finance"},
	{"name": "reports", "path": "/ppm-tex/reports", "component": "TimeExpenseReportBuilder", "permission": "ppm_tex:reports", "nav_group": "Reports"},
	{"name": "policy", "path": "/ppm-tex/policy", "component": "ExpensePolicyManager", "permission": "ppm_tex:admin", "nav_group": "Configuration"},
	{"name": "agents", "path": "/ppm-tex/agents", "component": "TexAgentWorkbench", "permission": "ppm_tex:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/ppm-tex/settings", "component": "TexSettings", "permission": "ppm_tex:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "ppm_tex_control",
	"tokens": {
		"color.primary": "#0F766E",
		"color.accent": "#1D4ED8",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"timesheet": {"icon": "clock", "status_indicator": "timesheet-status-chip"},
		"time_entry": {"icon": "edit-3", "status_indicator": "billable-status-chip"},
		"expense_claim": {"icon": "credit-card", "status_indicator": "expense-status-chip"},
		"expense_category": {"icon": "tag", "status_indicator": "category-chip"},
		"reimbursement": {"icon": "refresh-cw", "status_indicator": "reimbursement-chip"},
		"billing_rate": {"icon": "dollar-sign", "status_indicator": "rate-type-chip"},
		"approval": {"icon": "clipboard-check", "status_indicator": "approval-status-chip"},
		"receipt": {"icon": "file-text", "status_indicator": "receipt-status-chip"},
		"agent": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": TEX_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"timesheet_submitted",
		"timesheet_approved",
		"timesheet_rejected",
		"time_entry_recorded",
		"expense_claim_submitted",
		"expense_approved",
		"expense_rejected",
		"reimbursement_processed",
		"billable_hours_exported",
		"billing_rate_updated",
		"policy_violation_detected",
		"agent_registered",
	],
	"guardrails": [
		"tex_batch_requires_bytewax",
		"timesheet_submission_requires_project",
		"expense_above_threshold_requires_receipt",
		"backdated_entry_requires_justification",
		"cross_tenant_time_access_denied",
		"duplicate_expense_submission_denied",
		"personal_expense_reimbursement_requires_approval",
		"privileged_agent_action_requires_human_approval",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"timesheets": {
		"supported_statuses": SUPPORTED_TIMESHEET_STATUSES,
		"supported_period_types": SUPPORTED_PERIOD_TYPES,
		"supported_entry_types": SUPPORTED_TIME_ENTRY_TYPES,
		"supported_billable_statuses": SUPPORTED_BILLABLE_STATUSES,
		"project_required": True,
		"approval_required": True,
		"evidence_required": True,
	},
	"expenses": {
		"supported_statuses": SUPPORTED_EXPENSE_STATUSES,
		"supported_categories": SUPPORTED_EXPENSE_CATEGORIES,
		"supported_currencies": SUPPORTED_CURRENCIES,
		"supported_receipt_statuses": SUPPORTED_RECEIPT_STATUSES,
		"supported_mileage_units": SUPPORTED_MILEAGE_UNITS,
		"receipt_threshold_amount": 25.00,
		"approval_required": True,
		"evidence_required": True,
	},
	"reimbursements": {
		"supported_methods": SUPPORTED_REIMBURSEMENT_METHODS,
		"approval_required": True,
		"evidence_required": True,
	},
	"billing": {
		"supported_rate_types": SUPPORTED_BILLING_RATE_TYPES,
		"approval_required": True,
		"effective_date_required": True,
	},
	"approvals": {
		"supported_workflows": SUPPORTED_APPROVAL_WORKFLOWS,
		"supported_review_statuses": SUPPORTED_REVIEW_STATUSES,
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
		"timesheet_submission_requires_project": True,
		"expense_above_threshold_requires_receipt": True,
		"backdated_entry_requires_justification": True,
		"cross_tenant_time_access_denied": True,
		"duplicate_expense_submission_denied": True,
		"personal_expense_reimbursement_requires_approval": True,
	},
	"observability": {"event_stream": TEX_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_timesheets": True, "enable_expenses": True, "enable_billing": True, "enable_reimbursements": True, "enable_reports": True, "enable_policy": True, "enable_agents": True},
	"theme": {"default_theme": "ppm_tex_control", "allow_tenant_overrides": True},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "tex_policy_required", "required_action": "attach_tex_policy"}},
	{"name": "timesheet_status_supported", "condition": {"operation": "submit_timesheet", "status_supported": False}, "effect": {"decision": "deny", "reason": "timesheet_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "timesheet_project_required", "condition": {"operation": "submit_timesheet", "project_present": False}, "effect": {"decision": "deny", "reason": "timesheet_submission_requires_project", "required_action": "select_project"}},
	{"name": "timesheet_period_supported", "condition": {"operation": "submit_timesheet", "period_supported": False}, "effect": {"decision": "deny", "reason": "timesheet_period_not_supported", "required_action": "select_supported_period"}},
	{"name": "timesheet_approval_required", "condition": {"operation": "submit_timesheet", "approval_workflow_present": False}, "effect": {"decision": "deny", "reason": "timesheet_approval_required", "required_action": "configure_approval_workflow"}},
	{"name": "time_entry_type_supported", "condition": {"operation": "record_time_entry", "entry_type_supported": False}, "effect": {"decision": "deny", "reason": "time_entry_type_not_supported", "required_action": "select_supported_entry_type"}},
	{"name": "time_entry_billable_status_supported", "condition": {"operation": "record_time_entry", "billable_status_supported": False}, "effect": {"decision": "deny", "reason": "billable_status_not_supported", "required_action": "select_supported_billable_status"}},
	{"name": "time_entry_hours_positive", "condition": {"operation": "record_time_entry", "hours_positive": False}, "effect": {"decision": "deny", "reason": "time_entry_hours_must_be_positive", "required_action": "correct_hours_value"}},
	{"name": "backdated_entry_requires_justification", "condition": {"operation": "record_time_entry", "backdated": True, "justification_present": False}, "effect": {"decision": "deny", "reason": "backdated_entry_requires_justification", "required_action": "attach_backdating_justification"}},
	{"name": "expense_status_supported", "condition": {"operation": "submit_expense", "status_supported": False}, "effect": {"decision": "deny", "reason": "expense_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "expense_category_supported", "condition": {"operation": "submit_expense", "category_supported": False}, "effect": {"decision": "deny", "reason": "expense_category_not_supported", "required_action": "select_supported_category"}},
	{"name": "expense_currency_supported", "condition": {"operation": "submit_expense", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "expense_amount_positive", "condition": {"operation": "submit_expense", "amount_positive": False}, "effect": {"decision": "deny", "reason": "expense_amount_must_be_positive", "required_action": "correct_expense_amount"}},
	{"name": "expense_receipt_required_above_threshold", "condition": {"operation": "submit_expense", "above_receipt_threshold": True, "receipt_present": False}, "effect": {"decision": "deny", "reason": "expense_above_threshold_requires_receipt", "required_action": "attach_receipt"}},
	{"name": "expense_approval_required", "condition": {"operation": "submit_expense", "approval_present": False}, "effect": {"decision": "deny", "reason": "expense_approval_required", "required_action": "obtain_expense_approval"}},
	{"name": "duplicate_expense_denied", "condition": {"duplicate_expense_submission": True}, "effect": {"decision": "deny", "reason": "duplicate_expense_submission_denied", "required_action": "remove_duplicate_submission"}},
	{"name": "reimbursement_method_supported", "condition": {"operation": "process_reimbursement", "method_supported": False}, "effect": {"decision": "deny", "reason": "reimbursement_method_not_supported", "required_action": "select_supported_reimbursement_method"}},
	{"name": "reimbursement_approval_required", "condition": {"operation": "process_reimbursement", "approval_present": False}, "effect": {"decision": "deny", "reason": "personal_expense_reimbursement_requires_approval", "required_action": "obtain_reimbursement_approval"}},
	{"name": "billing_rate_type_supported", "condition": {"operation": "set_billing_rate", "rate_type_supported": False}, "effect": {"decision": "deny", "reason": "billing_rate_type_not_supported", "required_action": "select_supported_billing_rate_type"}},
	{"name": "billing_rate_approval_required", "condition": {"operation": "set_billing_rate", "approval_present": False}, "effect": {"decision": "deny", "reason": "billing_rate_change_requires_approval", "required_action": "obtain_billing_rate_approval"}},
	{"name": "billing_rate_effective_date_required", "condition": {"operation": "set_billing_rate", "effective_date_present": False}, "effect": {"decision": "deny", "reason": "effective_date_required", "required_action": "set_effective_date"}},
	{"name": "approval_reviewer_required", "condition": {"operation": "approve_timesheet", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "cross_tenant_time_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_time_access_denied", "required_action": "use_own_tenant_context"}},
	{"name": "tex_batch_requires_bytewax", "condition": {"operation": "tex_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_tex_batch_to_bytewax"}},
	{"name": "agent_runtime_supported", "condition": {"operation": "register_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "agent_role_supported", "condition": {"operation": "register_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "agent_name_required", "condition": {"operation": "register_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "agent_name_required", "required_action": "name_agent"}},
	{"name": "agent_scope_required", "condition": {"operation": "register_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "agent_scope_required", "required_action": "bound_agent_scope"}},
	{"name": "privileged_agent_action_requires_human_approval", "condition": {"operation": "agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
			"required": ["tenant_id", "ui", "theme"],
			"properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/ppm-tex/api/v1",
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
