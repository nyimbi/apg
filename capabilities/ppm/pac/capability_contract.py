"""Executable capability contract for APG Project Accounting (pac)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "ppm_pac"
CAPABILITY_NAME = "Project Accounting"
CAPABILITY_VERSION = "1.0.0"
PAC_EVENT_STREAM = "apg.ppm.pac.lifecycle"

# ── Supported enum values ────────────────────────────────────────────────────
SUPPORTED_COST_TYPES = ["labour", "materials", "subcontractor", "equipment", "travel", "overhead", "contingency", "other"]
SUPPORTED_REVENUE_TYPES = ["fixed_fee", "time_and_materials", "milestone", "retainer", "cost_plus", "performance_incentive"]
SUPPORTED_WIP_METHODS = ["percentage_completion", "completed_contract", "cost_to_cost", "units_delivered", "earned_value"]
SUPPORTED_BILLING_TYPES = ["milestone", "progress", "time_and_materials", "fixed_price", "retainer", "advance"]
SUPPORTED_ACCOUNT_STATUSES = ["active", "on_hold", "closed", "pending_approval", "over_budget", "under_review"]
SUPPORTED_TRANSACTION_TYPES = ["actual_cost", "committed_cost", "forecast_cost", "budget_transfer", "revenue_recognition", "wip_adjustment", "invoice_posting", "payment_received"]
SUPPORTED_APPROVAL_STATUSES = ["draft", "submitted", "under_review", "approved", "rejected", "cancelled"]
SUPPORTED_PERIOD_TYPES = ["weekly", "bi_weekly", "monthly", "quarterly", "annual", "project_to_date", "custom"]
SUPPORTED_PROFITABILITY_METHODS = ["gross_margin", "contribution_margin", "net_margin", "earned_value_margin"]
SUPPORTED_REPORT_TYPES = ["cost_summary", "revenue_summary", "wip_report", "profitability_report", "variance_report", "billing_summary", "cash_flow_forecast"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["cost_analyst", "billing_reviewer", "wip_auditor", "revenue_recogniser", "budget_controller"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS", "ETB"]

PROVIDES = [
	"project_cost_tracking",
	"revenue_recognition_workflow",
	"wip_accounting_workflow",
	"milestone_billing_workflow",
	"project_profitability_reporting",
	"budget_vs_actual_analysis",
	"cost_variance_alerts",
	"cash_flow_forecasting",
	"multi_currency_project_accounting",
	"audit_trail_maintenance",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "comp", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/ppm-pac/dashboard", "component": "PacDashboard", "permission": "ppm_pac:view", "nav_group": "Overview"},
	{"name": "project_accounts", "path": "/ppm-pac/accounts", "component": "ProjectAccountList", "permission": "ppm_pac:accounts", "nav_group": "Accounts"},
	{"name": "account_detail", "path": "/ppm-pac/accounts/<id>", "component": "ProjectAccountDetail", "permission": "ppm_pac:accounts", "nav_group": "Accounts"},
	{"name": "cost_transactions", "path": "/ppm-pac/costs", "component": "CostTransactionLedger", "permission": "ppm_pac:costs", "nav_group": "Costs"},
	{"name": "revenue_recognition", "path": "/ppm-pac/revenue", "component": "RevenueRecognitionConsole", "permission": "ppm_pac:revenue", "nav_group": "Revenue"},
	{"name": "wip_accounting", "path": "/ppm-pac/wip", "component": "WipAccountingWorkbench", "permission": "ppm_pac:wip", "nav_group": "WIP"},
	{"name": "billing", "path": "/ppm-pac/billing", "component": "MilestoneBillingConsole", "permission": "ppm_pac:billing", "nav_group": "Billing"},
	{"name": "budget_control", "path": "/ppm-pac/budgets", "component": "BudgetControlConsole", "permission": "ppm_pac:budgets", "nav_group": "Budgets"},
	{"name": "profitability", "path": "/ppm-pac/profitability", "component": "ProfitabilityReportView", "permission": "ppm_pac:reports", "nav_group": "Reports"},
	{"name": "variance_analysis", "path": "/ppm-pac/variance", "component": "VarianceAnalysisView", "permission": "ppm_pac:reports", "nav_group": "Reports"},
	{"name": "cash_flow", "path": "/ppm-pac/cashflow", "component": "CashFlowForecastView", "permission": "ppm_pac:reports", "nav_group": "Reports"},
	{"name": "approvals", "path": "/ppm-pac/approvals", "component": "AccountingApprovalQueue", "permission": "ppm_pac:approve", "nav_group": "Governance"},
	{"name": "agents", "path": "/ppm-pac/agents", "component": "PacAgentWorkbench", "permission": "ppm_pac:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/ppm-pac/settings", "component": "PacSettings", "permission": "ppm_pac:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "ppm_pac_control",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#0891B2",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"project_account": {"icon": "briefcase", "status_indicator": "account-status-chip"},
		"cost_transaction": {"icon": "receipt", "status_indicator": "cost-type-chip"},
		"revenue_recognition": {"icon": "trending-up", "status_indicator": "revenue-status-chip"},
		"wip": {"icon": "layers", "status_indicator": "wip-method-chip"},
		"billing": {"icon": "file-invoice", "status_indicator": "billing-status-chip"},
		"budget": {"icon": "target", "status_indicator": "budget-health-chip"},
		"profitability": {"icon": "bar-chart-2", "status_indicator": "margin-chip"},
		"approval": {"icon": "clipboard-check", "status_indicator": "approval-status-chip"},
		"agent": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": PAC_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"project_account_created",
		"cost_transaction_recorded",
		"revenue_recognised",
		"wip_adjustment_posted",
		"milestone_invoice_raised",
		"budget_variance_detected",
		"approval_submitted",
		"approval_completed",
		"profitability_report_generated",
		"agent_registered",
	],
	"guardrails": [
		"cost_batch_requires_bytewax",
		"revenue_recognition_requires_approval",
		"wip_adjustment_requires_auditor",
		"budget_override_requires_controller",
		"privileged_agent_action_requires_human_approval",
		"cross_tenant_cost_access_denied",
		"backdated_transaction_requires_justification",
		"negative_revenue_recognition_denied",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"project_accounts": {
		"supported_statuses": SUPPORTED_ACCOUNT_STATUSES,
		"supported_currencies": SUPPORTED_CURRENCIES,
		"owner_required": True,
		"budget_required": True,
		"evidence_required": True,
	},
	"costs": {
		"supported_cost_types": SUPPORTED_COST_TYPES,
		"supported_transaction_types": SUPPORTED_TRANSACTION_TYPES,
		"account_required": True,
		"amount_positive_required": True,
		"evidence_required": True,
	},
	"revenue": {
		"supported_revenue_types": SUPPORTED_REVENUE_TYPES,
		"supported_wip_methods": SUPPORTED_WIP_METHODS,
		"account_required": True,
		"approval_required": True,
		"evidence_required": True,
	},
	"billing": {
		"supported_billing_types": SUPPORTED_BILLING_TYPES,
		"account_required": True,
		"amount_positive_required": True,
		"approval_required": True,
		"evidence_required": True,
	},
	"reports": {
		"supported_report_types": SUPPORTED_REPORT_TYPES,
		"supported_period_types": SUPPORTED_PERIOD_TYPES,
		"supported_profitability_methods": SUPPORTED_PROFITABILITY_METHODS,
	},
	"approvals": {
		"supported_statuses": SUPPORTED_APPROVAL_STATUSES,
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
		"cross_tenant_cost_access_denied": True,
		"backdated_transaction_requires_justification": True,
		"revenue_recognition_requires_approval": True,
		"wip_adjustment_requires_auditor": True,
		"budget_override_requires_controller": True,
		"negative_revenue_recognition_denied": True,
	},
	"observability": {"event_stream": PAC_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_project_accounts": True, "enable_costs": True, "enable_revenue": True, "enable_wip": True, "enable_billing": True, "enable_budgets": True, "enable_reports": True, "enable_approvals": True, "enable_agents": True},
	"theme": {"default_theme": "ppm_pac_control", "allow_tenant_overrides": True},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "accounting_policy_required", "required_action": "attach_accounting_policy"}},
	{"name": "account_status_supported", "condition": {"operation": "create_account", "status_supported": False}, "effect": {"decision": "deny", "reason": "account_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "account_owner_required", "condition": {"operation": "create_account", "owner_present": False}, "effect": {"decision": "deny", "reason": "account_owner_required", "required_action": "assign_account_owner"}},
	{"name": "account_budget_required", "condition": {"operation": "create_account", "budget_present": False}, "effect": {"decision": "deny", "reason": "account_budget_required", "required_action": "set_account_budget"}},
	{"name": "account_currency_supported", "condition": {"operation": "create_account", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "account_evidence_required", "condition": {"operation": "create_account", "evidence_present": False}, "effect": {"decision": "deny", "reason": "account_evidence_required", "required_action": "attach_account_evidence"}},
	{"name": "cost_type_supported", "condition": {"operation": "record_cost", "cost_type_supported": False}, "effect": {"decision": "deny", "reason": "cost_type_not_supported", "required_action": "select_supported_cost_type"}},
	{"name": "cost_transaction_type_supported", "condition": {"operation": "record_cost", "transaction_type_supported": False}, "effect": {"decision": "deny", "reason": "transaction_type_not_supported", "required_action": "select_supported_transaction_type"}},
	{"name": "cost_account_required", "condition": {"operation": "record_cost", "account_present": False}, "effect": {"decision": "deny", "reason": "project_account_required", "required_action": "select_project_account"}},
	{"name": "cost_amount_positive", "condition": {"operation": "record_cost", "amount_positive": False}, "effect": {"decision": "deny", "reason": "cost_amount_must_be_positive", "required_action": "correct_cost_amount"}},
	{"name": "cost_evidence_required", "condition": {"operation": "record_cost", "evidence_present": False}, "effect": {"decision": "deny", "reason": "cost_evidence_required", "required_action": "attach_cost_evidence"}},
	{"name": "backdated_cost_requires_justification", "condition": {"operation": "record_cost", "backdated": True, "justification_present": False}, "effect": {"decision": "deny", "reason": "backdated_transaction_requires_justification", "required_action": "attach_backdating_justification"}},
	{"name": "revenue_type_supported", "condition": {"operation": "recognise_revenue", "revenue_type_supported": False}, "effect": {"decision": "deny", "reason": "revenue_type_not_supported", "required_action": "select_supported_revenue_type"}},
	{"name": "revenue_wip_method_supported", "condition": {"operation": "recognise_revenue", "wip_method_supported": False}, "effect": {"decision": "deny", "reason": "wip_method_not_supported", "required_action": "select_supported_wip_method"}},
	{"name": "revenue_account_required", "condition": {"operation": "recognise_revenue", "account_present": False}, "effect": {"decision": "deny", "reason": "project_account_required", "required_action": "select_project_account"}},
	{"name": "revenue_approval_required", "condition": {"operation": "recognise_revenue", "approval_present": False}, "effect": {"decision": "deny", "reason": "revenue_recognition_requires_approval", "required_action": "obtain_revenue_approval"}},
	{"name": "negative_revenue_denied", "condition": {"operation": "recognise_revenue", "amount_positive": False}, "effect": {"decision": "deny", "reason": "negative_revenue_recognition_denied", "required_action": "correct_revenue_amount"}},
	{"name": "revenue_evidence_required", "condition": {"operation": "recognise_revenue", "evidence_present": False}, "effect": {"decision": "deny", "reason": "revenue_evidence_required", "required_action": "attach_revenue_evidence"}},
	{"name": "wip_account_required", "condition": {"operation": "post_wip_adjustment", "account_present": False}, "effect": {"decision": "deny", "reason": "project_account_required", "required_action": "select_project_account"}},
	{"name": "wip_auditor_required", "condition": {"operation": "post_wip_adjustment", "auditor_present": False}, "effect": {"decision": "deny", "reason": "wip_adjustment_requires_auditor", "required_action": "assign_wip_auditor"}},
	{"name": "wip_evidence_required", "condition": {"operation": "post_wip_adjustment", "evidence_present": False}, "effect": {"decision": "deny", "reason": "wip_evidence_required", "required_action": "attach_wip_evidence"}},
	{"name": "billing_type_supported", "condition": {"operation": "raise_invoice", "billing_type_supported": False}, "effect": {"decision": "deny", "reason": "billing_type_not_supported", "required_action": "select_supported_billing_type"}},
	{"name": "billing_account_required", "condition": {"operation": "raise_invoice", "account_present": False}, "effect": {"decision": "deny", "reason": "project_account_required", "required_action": "select_project_account"}},
	{"name": "billing_amount_positive", "condition": {"operation": "raise_invoice", "amount_positive": False}, "effect": {"decision": "deny", "reason": "invoice_amount_must_be_positive", "required_action": "correct_invoice_amount"}},
	{"name": "billing_approval_required", "condition": {"operation": "raise_invoice", "approval_present": False}, "effect": {"decision": "deny", "reason": "invoice_approval_required", "required_action": "obtain_invoice_approval"}},
	{"name": "billing_evidence_required", "condition": {"operation": "raise_invoice", "evidence_present": False}, "effect": {"decision": "deny", "reason": "billing_evidence_required", "required_action": "attach_billing_evidence"}},
	{"name": "budget_override_requires_controller", "condition": {"operation": "override_budget", "controller_approval_present": False}, "effect": {"decision": "deny", "reason": "budget_override_requires_controller", "required_action": "obtain_controller_approval"}},
	{"name": "cross_tenant_cost_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_cost_access_denied", "required_action": "use_own_tenant_context"}},
	{"name": "cost_batch_requires_bytewax", "condition": {"operation": "cost_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_cost_batch_to_bytewax"}},
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
			"api_prefix": "/ppm-pac/api/v1",
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
