"""Executable capability contract for APG Real Estate Accounting."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "realestate_acc"
CAPABILITY_NAME = "Real Estate Accounting"
CAPABILITY_VERSION = "1.0.0"
ACC_EVENT_STREAM = "apg.realestate.acc.lifecycle"

SUPPORTED_LEDGER_TYPES = ["property_ledger", "service_charge", "cam_reconciliation", "rental_income", "security_deposit", "capex", "opex", "inter_company"]
SUPPORTED_ACCOUNT_TYPES = ["asset", "liability", "equity", "revenue", "expense", "contra"]
SUPPORTED_JOURNAL_TYPES = ["manual", "automatic", "recurring", "reversing", "closing", "accrual", "prepayment"]
SUPPORTED_CHARGE_TYPES = ["base_rent", "service_charge", "insurance", "utilities", "management_fee", "parking", "storage", "ad_hoc"]
SUPPORTED_CAM_METHODS = ["pro_rata", "fixed_share", "gross_leasable_area", "occupied_area", "metered"]
SUPPORTED_IFRS16_CATEGORIES = ["finance_lease", "operating_lease", "short_term_exemption", "low_value_exemption"]
SUPPORTED_REVENUE_METHODS = ["straight_line", "escalation_linked", "percentage_rent", "hybrid"]
SUPPORTED_RECONCILIATION_STATUSES = ["draft", "in_review", "approved", "posted", "disputed", "settled"]
SUPPORTED_PERIOD_TYPES = ["monthly", "quarterly", "semi_annual", "annual"]
SUPPORTED_CURRENCY_CODES = ["KES", "USD", "EUR", "GBP", "ZAR", "NGN", "UGX", "TZS"]
SUPPORTED_TAX_TYPES = ["vat", "withholding_tax", "stamp_duty", "capital_gains", "income_tax"]
SUPPORTED_APPROVAL_LEVELS = ["supervisor", "finance_manager", "cfo", "board"]
SUPPORTED_REPORT_TYPES = ["trial_balance", "income_statement", "balance_sheet", "cash_flow", "cam_statement", "rent_roll_summary", "variance_report"]
SUPPORTED_ALLOCATION_METHODS = ["direct", "proportional", "stepped", "capped", "base_year"]
SUPPORTED_POSTING_STATUSES = ["draft", "pending_approval", "approved", "posted", "reversed", "void"]

PROVIDES = [
	"property_ledger_management",
	"service_charge_accounting",
	"cam_reconciliation_workflow",
	"ifrs16_lease_accounting",
	"revenue_recognition_engine",
	"journal_entry_management",
	"period_close_workflow",
	"tenant_statement_generation",
	"tax_calculation_engine",
	"financial_report_generation",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/realestate/acc/dashboard", "component": "AccDashboard", "permission": "realestate_acc:view", "nav_group": "Overview"},
	{"name": "ledger", "path": "/realestate/acc/ledger", "component": "PropertyLedger", "permission": "realestate_acc:ledger", "nav_group": "Ledger"},
	{"name": "journal-entries", "path": "/realestate/acc/journals", "component": "JournalEntryWorkbench", "permission": "realestate_acc:journals", "nav_group": "Ledger"},
	{"name": "service-charges", "path": "/realestate/acc/service-charges", "component": "ServiceChargeConsole", "permission": "realestate_acc:service_charges", "nav_group": "Charges"},
	{"name": "cam-reconciliation", "path": "/realestate/acc/cam", "component": "CamReconciliationWorkbench", "permission": "realestate_acc:cam", "nav_group": "Charges"},
	{"name": "ifrs16", "path": "/realestate/acc/ifrs16", "component": "Ifrs16LeaseSchedule", "permission": "realestate_acc:ifrs16", "nav_group": "Compliance"},
	{"name": "revenue", "path": "/realestate/acc/revenue", "component": "RevenueRecognitionConsole", "permission": "realestate_acc:revenue", "nav_group": "Revenue"},
	{"name": "period-close", "path": "/realestate/acc/period-close", "component": "PeriodCloseWorkflow", "permission": "realestate_acc:period_close", "nav_group": "Periods"},
	{"name": "tenant-statements", "path": "/realestate/acc/statements", "component": "TenantStatementQueue", "permission": "realestate_acc:statements", "nav_group": "Reporting"},
	{"name": "tax", "path": "/realestate/acc/tax", "component": "TaxCalcConsole", "permission": "realestate_acc:tax", "nav_group": "Compliance"},
	{"name": "reports", "path": "/realestate/acc/reports", "component": "FinancialReportBuilder", "permission": "realestate_acc:reports", "nav_group": "Reporting"},
	{"name": "allocations", "path": "/realestate/acc/allocations", "component": "CostAllocationConsole", "permission": "realestate_acc:allocations", "nav_group": "Charges"},
	{"name": "approvals", "path": "/realestate/acc/approvals", "component": "AccountingApprovalQueue", "permission": "realestate_acc:approvals", "nav_group": "Governance"},
	{"name": "settings", "path": "/realestate/acc/settings", "component": "AccSettings", "permission": "realestate_acc:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "realestate_acc_ledger",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#0891B2",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F1F5F9",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0F172A",
		"text.secondary": "#475569",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"ledger": {"icon": "book-open", "status_indicator": "account-type-chip"},
		"journal_entries": {"icon": "file-text", "status_indicator": "posting-status-chip"},
		"service_charges": {"icon": "receipt", "status_indicator": "charge-type-chip"},
		"cam_reconciliation": {"icon": "calculator", "status_indicator": "reconciliation-status-chip"},
		"ifrs16": {"icon": "layers", "status_indicator": "ifrs16-category-chip"},
		"revenue": {"icon": "trending-up", "status_indicator": "revenue-method-chip"},
		"period_close": {"icon": "calendar-check", "status_indicator": "period-status-chip"},
		"tenant_statements": {"icon": "mail", "status_indicator": "statement-status-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": ACC_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"journal_entry_created", "journal_entry_posted", "journal_entry_reversed",
		"service_charge_raised", "service_charge_approved", "service_charge_posted",
		"cam_reconciliation_started", "cam_reconciliation_approved", "cam_reconciliation_settled",
		"ifrs16_schedule_generated", "revenue_recognised", "period_opened", "period_closed",
		"tenant_statement_generated", "tax_calculated", "allocation_run_completed",
	],
	"guardrails": [
		"journal_batch_requires_bytewax",
		"period_close_requires_approval",
		"cam_reconciliation_requires_evidence",
		"revenue_recognition_requires_lease_link",
		"ifrs16_reclassification_requires_audit",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ledger": {"supported_ledger_types": SUPPORTED_LEDGER_TYPES, "supported_account_types": SUPPORTED_ACCOUNT_TYPES, "supported_currencies": SUPPORTED_CURRENCY_CODES},
	"journals": {"supported_journal_types": SUPPORTED_JOURNAL_TYPES, "approval_required_above_amount": 50000, "supported_posting_statuses": SUPPORTED_POSTING_STATUSES},
	"service_charges": {"supported_charge_types": SUPPORTED_CHARGE_TYPES, "supported_cam_methods": SUPPORTED_CAM_METHODS, "cap_required": False},
	"ifrs16": {"supported_categories": SUPPORTED_IFRS16_CATEGORIES, "discount_rate_required": True, "commencement_date_required": True},
	"revenue": {"supported_methods": SUPPORTED_REVENUE_METHODS, "period_types": SUPPORTED_PERIOD_TYPES},
	"tax": {"supported_tax_types": SUPPORTED_TAX_TYPES, "auto_calculate": True},
	"approvals": {"supported_approval_levels": SUPPORTED_APPROVAL_LEVELS, "escalation_enabled": True},
	"reports": {"supported_report_types": SUPPORTED_REPORT_TYPES},
	"ui": {"enable_dashboard": True, "enable_ledger": True, "enable_journals": True, "enable_cam": True, "enable_ifrs16": True, "enable_revenue": True, "enable_reports": True},
	"theme": {"default_theme": "realestate_acc_ledger", "allow_tenant_overrides": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "dual_control_for_period_close": True, "cross_tenant_posting_denied": True},
	"observability": {"event_stream": ACC_EVENT_STREAM, "stream_processor": "bytewax"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "accounting_policy_required", "required_action": "attach_accounting_policy"}},
	{"name": "journal_requires_balanced_entries", "condition": {"operation": "post_journal", "entries_balanced": False}, "effect": {"decision": "deny", "reason": "journal_must_balance", "required_action": "balance_debit_credit"}},
	{"name": "journal_requires_period_open", "condition": {"operation": "post_journal", "period_open": False}, "effect": {"decision": "deny", "reason": "accounting_period_closed", "required_action": "open_period_or_use_open_period"}},
	{"name": "journal_above_threshold_requires_approval", "condition": {"operation": "post_journal", "amount_above_threshold": True, "approved": False}, "effect": {"decision": "deny", "reason": "approval_required_for_large_journal", "required_action": "submit_for_approval"}},
	{"name": "journal_reversal_requires_original", "condition": {"operation": "reverse_journal", "original_journal_present": False}, "effect": {"decision": "deny", "reason": "original_journal_required_for_reversal", "required_action": "link_original_journal"}},
	{"name": "service_charge_requires_property", "condition": {"operation": "raise_service_charge", "property_present": False}, "effect": {"decision": "deny", "reason": "property_required_for_service_charge", "required_action": "link_property"}},
	{"name": "service_charge_type_supported", "condition": {"operation": "raise_service_charge", "charge_type_supported": False}, "effect": {"decision": "deny", "reason": "charge_type_not_supported", "required_action": "select_supported_charge_type"}},
	{"name": "cam_requires_lease_links", "condition": {"operation": "start_cam_reconciliation", "leases_linked": False}, "effect": {"decision": "deny", "reason": "lease_links_required_for_cam", "required_action": "link_leases_to_cam"}},
	{"name": "cam_requires_actual_costs", "condition": {"operation": "start_cam_reconciliation", "actual_costs_present": False}, "effect": {"decision": "deny", "reason": "actual_costs_required_for_cam", "required_action": "record_actual_costs"}},
	{"name": "cam_approval_required_before_settlement", "condition": {"operation": "settle_cam", "cam_approved": False}, "effect": {"decision": "deny", "reason": "cam_approval_required", "required_action": "approve_cam_reconciliation"}},
	{"name": "ifrs16_requires_lease_term", "condition": {"operation": "create_ifrs16_schedule", "lease_term_present": False}, "effect": {"decision": "deny", "reason": "lease_term_required_for_ifrs16", "required_action": "record_lease_term"}},
	{"name": "ifrs16_requires_discount_rate", "condition": {"operation": "create_ifrs16_schedule", "discount_rate_present": False}, "effect": {"decision": "deny", "reason": "discount_rate_required_for_ifrs16", "required_action": "set_discount_rate"}},
	{"name": "ifrs16_reclassification_requires_auditor", "condition": {"operation": "reclassify_ifrs16_lease", "auditor_approved": False}, "effect": {"decision": "deny", "reason": "auditor_approval_required_for_reclassification", "required_action": "obtain_auditor_approval"}},
	{"name": "revenue_requires_lease_link", "condition": {"operation": "recognise_revenue", "lease_linked": False}, "effect": {"decision": "deny", "reason": "lease_link_required_for_revenue", "required_action": "link_lease_to_revenue_schedule"}},
	{"name": "revenue_method_supported", "condition": {"operation": "recognise_revenue", "method_supported": False}, "effect": {"decision": "deny", "reason": "revenue_method_not_supported", "required_action": "select_supported_revenue_method"}},
	{"name": "period_close_requires_dual_control", "condition": {"operation": "close_period", "dual_control_satisfied": False}, "effect": {"decision": "deny", "reason": "dual_control_required_for_period_close", "required_action": "obtain_second_approver"}},
	{"name": "period_close_requires_reconciliations_complete", "condition": {"operation": "close_period", "reconciliations_complete": False}, "effect": {"decision": "deny", "reason": "all_reconciliations_must_be_complete", "required_action": "complete_pending_reconciliations"}},
	{"name": "cross_tenant_posting_denied", "condition": {"operation": "post_journal", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_posting_not_allowed", "required_action": "use_intercompany_workflow"}},
	{"name": "tax_type_supported", "condition": {"operation": "calculate_tax", "tax_type_supported": False}, "effect": {"decision": "deny", "reason": "tax_type_not_supported", "required_action": "select_supported_tax_type"}},
	{"name": "delete_posted_journal_denied", "condition": {"operation": "delete_journal", "journal_status": "posted"}, "effect": {"decision": "deny", "reason": "posted_journals_cannot_be_deleted", "required_action": "reverse_journal_instead"}},
	{"name": "allocation_method_supported", "condition": {"operation": "run_allocation", "method_supported": False}, "effect": {"decision": "deny", "reason": "allocation_method_not_supported", "required_action": "select_supported_allocation_method"}},
	{"name": "statement_requires_tenant", "condition": {"operation": "generate_statement", "tenant_linked": False}, "effect": {"decision": "deny", "reason": "tenant_required_for_statement", "required_action": "link_tenant"}},
	{"name": "report_period_required", "condition": {"operation": "generate_report", "period_present": False}, "effect": {"decision": "deny", "reason": "reporting_period_required", "required_action": "specify_reporting_period"}},
	{"name": "currency_supported", "condition": {"operation_type": "write", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	"""Return the full capability contract for the given tenant."""
	cfg = deepcopy(DEFAULT_CONFIGURATION)
	cfg["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": cfg,
		"configuration_schema": {
			"required": ["tenant_id", "ui", "theme"],
			"properties": {
				"tenant_id": {"type": "string"},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
			},
		},
		"rule_engine": {
			"type": "deterministic",
			"default_decision": "allow",
			"rules": RULES,
		},
		"ui": {
			"shell": "apg_python",
			"requires_theme": True,
			"template_roots": ["realestate/acc/templates"],
			"routes": UI_ROUTES,
		},
		"theme": THEME,
		"streaming": STREAMING,
		"provides": PROVIDES,
		"requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate all rules against the given context. Returns first denial or allow."""
	for rule in RULES:
		cond = rule["condition"]
		match = all(context.get(k) == v for k, v in cond.items())
		if match:
			effect = rule["effect"]
			if effect["decision"] == "deny":
				return {"decision": "deny", "rule": rule["name"], "reason": effect["reason"], "required_action": effect.get("required_action")}
	return {"decision": "allow", "rule": None, "reason": None}
