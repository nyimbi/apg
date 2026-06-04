"""Executable capability contract for HCM Payroll."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "pay_payroll"
CAPABILITY_NAME = "Payroll Management"
CAPABILITY_VERSION = "2.2.0"
PAYROLL_EVENT_STREAM = "apg.hcm.pay.payroll.lifecycle"

SUPPORTED_PAY_FREQUENCIES = ["weekly", "biweekly", "semimonthly", "monthly", "quarterly", "annual"]
SUPPORTED_PERIOD_STATUSES = ["draft", "open", "locked", "closed", "voided"]
SUPPORTED_RUN_STATUSES = ["draft", "calculated", "review", "approved", "posted", "paid", "voided", "reversed"]
SUPPORTED_COMPONENT_TYPES = ["earning", "deduction", "tax", "benefit", "reimbursement", "garnishment", "advance", "loan_repayment"]
SUPPORTED_PAYMENT_METHODS = ["bank_transfer", "check", "cash", "mobile_money", "pay_card", "crypto_wallet"]
SUPPORTED_CURRENCIES = ["USD", "KES", "EUR", "GBP", "NGN", "ZAR", "GHS", "UGX", "TZS", "ETB", "RWF", "XOF"]
SUPPORTED_TAX_SCOPES = ["employee", "employer", "statutory", "local", "social_security", "pension", "health_levy", "withholding"]
SUPPORTED_TAX_BRACKET_TYPES = ["flat_rate", "progressive", "exempt", "regressive", "cap_based"]
SUPPORTED_PAYROLL_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_PAYROLL_AGENT_ROLES = [
	"payroll_reviewer",
	"tax_reviewer",
	"compliance_reviewer",
	"payment_reviewer",
	"variance_reviewer",
	"employee_query_reviewer",
]
SUPPORTED_REVERSAL_REASONS = [
	"duplicate_run",
	"incorrect_employee",
	"calculation_error",
	"wrong_period",
	"system_error",
	"authorized_correction",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"periods": {
		"name_required": True,
		"frequency_required": True,
		"supported_frequencies": SUPPORTED_PAY_FREQUENCIES,
		"start_date_required": True,
		"end_date_required": True,
		"pay_date_required": True,
		"currency_required": True,
		"supported_currencies": SUPPORTED_CURRENCIES,
		"prevent_overlapping_periods": True,
	},
	"pay_groups": {
		"code_required": True,
		"name_required": True,
		"frequency_required": True,
		"currency_required": True,
		"country_required": True,
		"owner_required": True,
		"tax_jurisdiction_required": True,
	},
	"employee_pay_profiles": {
		"employee_required": True,
		"pay_group_required": True,
		"payment_method_required": True,
		"supported_payment_methods": SUPPORTED_PAYMENT_METHODS,
		"tax_id_required": True,
		"currency_required": True,
		"bank_review_required": True,
		"pension_scheme_code_required": True,
	},
	"components": {
		"code_required": True,
		"name_required": True,
		"supported_types": SUPPORTED_COMPONENT_TYPES,
		"currency_required": True,
		"taxable_flag_required": True,
		"pensionable_flag_required": True,
		"gl_account_required": True,
	},
	"tax_rules": {
		"jurisdiction_required": True,
		"effective_date_required": True,
		"supported_bracket_types": SUPPORTED_TAX_BRACKET_TYPES,
		"brackets_required_for_progressive": True,
		"flat_rate_required_for_flat": True,
		"cap_amount_required_for_cap_based": True,
		"employer_rate_required": True,
		"employee_rate_required": True,
		"personal_relief_configurable": True,
		"annual_review_required": True,
	},
	"time_imports": {
		"period_required": True,
		"employee_required": True,
		"hours_nonnegative": True,
		"source_required": True,
		"approval_required_for_overtime": True,
		"overtime_rate_multiplier_required": True,
		"double_time_threshold_hours": 60,
	},
	"runs": {
		"period_required": True,
		"pay_group_required": True,
		"initiator_required": True,
		"variance_review_threshold_percent": 10,
		"approval_required_before_posting": True,
		"prevent_duplicate_run_same_period": True,
		"reversal_requires_reason": True,
		"supported_reversal_reasons": SUPPORTED_REVERSAL_REASONS,
	},
	"line_items": {
		"run_required": True,
		"employee_required": True,
		"component_required": True,
		"amount_required": True,
		"negative_amount_review_required": True,
		"gl_account_required": True,
	},
	"taxes": {
		"run_required": True,
		"employee_required": True,
		"supported_scopes": SUPPORTED_TAX_SCOPES,
		"tax_authority_required": True,
		"amount_required": True,
		"tax_rule_reference_required": True,
		"effective_tax_rate_recorded": True,
	},
	"adjustments": {
		"run_required": True,
		"employee_required": True,
		"reason_required": True,
		"approval_required": True,
		"max_adjustment_pct_without_executive_approval": 50,
	},
	"payments": {
		"run_required": True,
		"approval_required": True,
		"payment_date_required": True,
		"positive_net_pay_required": True,
		"bank_details_verified_required": True,
	},
	"payslips": {
		"run_required": True,
		"employee_required": True,
		"posting_required": True,
		"privacy_basis_required": True,
		"ytd_figures_included": True,
	},
	"tax_filings": {
		"run_required": True,
		"authority_required": True,
		"period_required": True,
		"approval_required": True,
		"deadline_tracking_enabled": True,
	},
	"payroll_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_PAYROLL_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_PAYROLL_AGENT_ROLES,
		"max_autonomous_scope": "inspect_prepare_and_recommend",
		"human_approval_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_state_changes": True,
		"segregation_of_duties": True,
		"approval_before_payment": True,
		"cross_tenant_access_denied": True,
		"privilege_escalation_denied": True,
		"initiator_cannot_approve_own_run": True,
		"tax_calculation_rules_enforced": True,
	},
	"observability": {
		"event_stream": PAYROLL_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_period_events": True,
		"emit_pay_group_events": True,
		"emit_profile_events": True,
		"emit_component_events": True,
		"emit_run_events": True,
		"emit_payment_events": True,
		"emit_tax_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"workflow": "adapter",
		"employee_data": "adapter",
		"time_attendance": "adapter",
		"benefits": "adapter",
		"general_ledger": "adapter",
		"banking": "adapter",
		"tax_authority": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_periods": True,
		"enable_pay_groups": True,
		"enable_profiles": True,
		"enable_components": True,
		"enable_tax_rules": True,
		"enable_time_imports": True,
		"enable_runs": True,
		"enable_line_items": True,
		"enable_taxes": True,
		"enable_adjustments": True,
		"enable_payments": True,
		"enable_payslips": True,
		"enable_filings": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "payroll_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"payroll_period_lifecycle",
	"pay_group_lifecycle",
	"employee_pay_profile_lifecycle",
	"pay_component_lifecycle",
	"payroll_tax_rule_lifecycle",
	"time_import_lifecycle",
	"payroll_run_lifecycle",
	"payroll_line_item_lifecycle",
	"payroll_tax_lifecycle",
	"payroll_adjustment_lifecycle",
	"payroll_payment_workflow",
	"payslip_lifecycle",
	"payroll_tax_filing_lifecycle",
	"payroll_dashboard_service",
	"payroll_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"mten",
	"conf",
	"ntfy",
	"wflo",
	"mqeb",
	"comp",
	"schd",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/hcm/payroll/dashboard", "component": "PayrollDashboard", "permission": "pay_payroll:view", "nav_group": "Overview"},
	{"name": "periods", "path": "/hcm/payroll/periods", "component": "PayrollPeriodWorkbench", "permission": "pay_payroll:manage_periods", "nav_group": "Setup"},
	{"name": "pay_groups", "path": "/hcm/payroll/pay-groups", "component": "PayGroupWorkbench", "permission": "pay_payroll:manage_setup", "nav_group": "Setup"},
	{"name": "profiles", "path": "/hcm/payroll/profiles", "component": "EmployeePayProfileWorkbench", "permission": "pay_payroll:manage_profiles", "nav_group": "Employees"},
	{"name": "components", "path": "/hcm/payroll/components", "component": "PayComponentWorkbench", "permission": "pay_payroll:manage_setup", "nav_group": "Setup"},
	{"name": "tax_rules", "path": "/hcm/payroll/tax-rules", "component": "TaxRuleWorkbench", "permission": "pay_payroll:manage_tax_rules", "nav_group": "Compliance"},
	{"name": "time_imports", "path": "/hcm/payroll/time-imports", "component": "PayrollTimeImportWorkbench", "permission": "pay_payroll:manage_runs", "nav_group": "Processing"},
	{"name": "runs", "path": "/hcm/payroll/runs", "component": "PayrollRunWorkbench", "permission": "pay_payroll:manage_runs", "nav_group": "Processing"},
	{"name": "line_items", "path": "/hcm/payroll/line-items", "component": "PayrollLineItemWorkbench", "permission": "pay_payroll:review", "nav_group": "Processing"},
	{"name": "taxes", "path": "/hcm/payroll/taxes", "component": "PayrollTaxWorkbench", "permission": "pay_payroll:review_tax", "nav_group": "Compliance"},
	{"name": "adjustments", "path": "/hcm/payroll/adjustments", "component": "PayrollAdjustmentWorkbench", "permission": "pay_payroll:adjust", "nav_group": "Processing"},
	{"name": "payments", "path": "/hcm/payroll/payments", "component": "PayrollPaymentWorkbench", "permission": "pay_payroll:pay", "nav_group": "Payments"},
	{"name": "payslips", "path": "/hcm/payroll/payslips", "component": "PayslipWorkbench", "permission": "pay_payroll:view_payslips", "nav_group": "Employees"},
	{"name": "filings", "path": "/hcm/payroll/tax-filings", "component": "PayrollTaxFilingWorkbench", "permission": "pay_payroll:file_tax", "nav_group": "Compliance"},
	{"name": "agents", "path": "/hcm/payroll/agents", "component": "PayrollAgentWorkbench", "permission": "pay_payroll:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/hcm/payroll/settings", "component": "PayrollSettings", "permission": "pay_payroll:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "payroll_control",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#5B6C8C",
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
		"periods": {"icon": "calendar-days", "status_indicator": "period-pill", "visual": "calendar-list"},
		"pay_groups": {"icon": "users", "visual": "pay-group-table", "status_style": "group-chip"},
		"profiles": {"icon": "id-badge", "visual": "profile-ledger", "status_style": "profile-chip"},
		"components": {"icon": "puzzle", "visual": "component-list", "status_style": "component-chip"},
		"tax_rules": {"icon": "receipt-tax", "visual": "tax-rule-matrix", "status_style": "rule-chip"},
		"time_imports": {"icon": "clock-arrow-up", "visual": "time-import-board", "status_style": "source-chip"},
		"runs": {"icon": "play-circle", "visual": "run-board", "status_style": "run-chip"},
		"line_items": {"icon": "list", "visual": "line-ledger", "status_style": "line-chip"},
		"taxes": {"icon": "landmark", "visual": "tax-ledger", "status_style": "tax-chip"},
		"adjustments": {"icon": "sliders", "visual": "adjustment-list", "status_style": "adjustment-chip"},
		"payments": {"icon": "banknote", "visual": "payment-lane", "status_style": "payment-chip"},
		"payslips": {"icon": "file-text", "visual": "payslip-list", "status_style": "privacy-chip"},
		"filings": {"icon": "file-check", "visual": "filing-calendar", "status_style": "filing-chip"},
		"agents": {"icon": "bot", "visual": "review-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": PAYROLL_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"payroll_period_created",
		"payroll_period_locked",
		"pay_group_created",
		"employee_pay_profile_created",
		"pay_component_created",
		"tax_rule_created",
		"tax_rule_updated",
		"time_import_recorded",
		"overtime_import_approved",
		"payroll_run_started",
		"payroll_run_calculated",
		"payroll_line_item_added",
		"payroll_tax_recorded",
		"payroll_adjustment_recorded",
		"payroll_run_approved",
		"payroll_run_posted",
		"payroll_run_reversed",
		"payment_batch_created",
		"payslip_published",
		"tax_filing_created",
		"payroll_agent_registered",
		"cross_tenant_access_blocked",
	],
	"states": ["draft", "open", "calculated", "review", "approved", "posted", "paid", "voided", "reversed", "blocked"],
	"guardrails": [
		"payroll_batch_requires_bytewax",
		"payroll_event_requires_bytewax",
		"privileged_payroll_agent_action_requires_human_approval",
		"cross_tenant_access_denied",
		"initiator_cannot_approve_own_run",
	],
}


RULES: list[dict[str, Any]] = [
	# --- Tenant context and write policy (mandatory gates) ---
	{"name": "tenant_context_required", "description": "All payroll operations require tenant context; deny if missing.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "payroll_write_requires_policy", "description": "Payroll writes require an attached operation policy.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},

	# --- Cross-tenant access prevention ---
	{"name": "cross_tenant_employee_access_denied", "description": "Payroll operations referencing an employee from a different tenant are denied.", "condition": {"operation_type": "write", "employee_tenant_mismatch": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_same_tenant_employee"}},
	{"name": "cross_tenant_period_access_denied", "description": "Payroll operations referencing a period from a different tenant are denied.", "condition": {"operation_type": "write", "period_tenant_mismatch": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_same_tenant_period"}},
	{"name": "cross_tenant_pay_group_access_denied", "description": "Payroll operations referencing a pay group from a different tenant are denied.", "condition": {"operation_type": "write", "pay_group_tenant_mismatch": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_same_tenant_pay_group"}},

	# --- Privilege escalation prevention ---
	{"name": "initiator_cannot_approve_own_run", "description": "The user who initiated a payroll run cannot be its approver (SoD).", "condition": {"operation": "approve_payroll_run", "initiator_equals_approver": True}, "effect": {"decision": "deny", "reason": "segregation_of_duties_violation", "required_action": "assign_independent_approver"}},
	{"name": "non_admin_void_run_denied", "description": "Voiding a posted payroll run requires admin or finance-director role.", "condition": {"operation": "void_payroll_run", "actor_is_admin_or_finance_director": False}, "effect": {"decision": "deny", "reason": "insufficient_role_for_void", "required_action": "elevate_role_or_request_admin_void"}},
	{"name": "self_payment_creation_denied", "description": "A payroll processor cannot create a payment batch that includes their own pay line.", "condition": {"operation": "create_payment_batch", "includes_own_pay": True}, "effect": {"decision": "deny", "reason": "self_payment_privilege_escalation", "required_action": "assign_independent_payment_processor"}},

	# --- Payroll period ---
	{"name": "period_requires_name", "description": "Payroll periods require a name.", "condition": {"operation": "create_payroll_period", "name_present": False}, "effect": {"decision": "deny", "reason": "period_name_required", "required_action": "set_period_name"}},
	{"name": "period_frequency_supported", "description": "Payroll period frequency must be from the supported set.", "condition": {"operation": "create_payroll_period", "frequency_supported": False}, "effect": {"decision": "deny", "reason": "pay_frequency_not_supported", "required_action": "select_supported_frequency"}},
	{"name": "period_requires_start_date", "description": "Payroll periods require a start date.", "condition": {"operation": "create_payroll_period", "start_date_present": False}, "effect": {"decision": "deny", "reason": "period_start_date_required", "required_action": "set_start_date"}},
	{"name": "period_requires_end_date", "description": "Payroll periods require an end date.", "condition": {"operation": "create_payroll_period", "end_date_present": False}, "effect": {"decision": "deny", "reason": "period_end_date_required", "required_action": "set_end_date"}},
	{"name": "period_requires_pay_date", "description": "Payroll periods require a pay date.", "condition": {"operation": "create_payroll_period", "pay_date_present": False}, "effect": {"decision": "deny", "reason": "pay_date_required", "required_action": "set_pay_date"}},
	{"name": "period_currency_supported", "description": "Payroll period currency must be from the supported set.", "condition": {"operation": "create_payroll_period", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "period_overlap_denied", "description": "Two payroll periods for the same pay group cannot overlap.", "condition": {"operation": "create_payroll_period", "period_overlaps_existing": True}, "effect": {"decision": "deny", "reason": "overlapping_payroll_period", "required_action": "adjust_period_dates_to_remove_overlap"}},
	{"name": "locked_period_modification_denied", "description": "Locked payroll periods cannot be modified.", "condition": {"operation": "update_payroll_period", "period_status": "locked"}, "effect": {"decision": "deny", "reason": "locked_period_is_immutable", "required_action": "unlock_period_with_approval_before_modifying"}},

	# --- Pay group ---
	{"name": "pay_group_requires_code", "description": "Pay groups require a code.", "condition": {"operation": "create_pay_group", "code_present": False}, "effect": {"decision": "deny", "reason": "pay_group_code_required", "required_action": "set_pay_group_code"}},
	{"name": "pay_group_requires_name", "description": "Pay groups require a name.", "condition": {"operation": "create_pay_group", "name_present": False}, "effect": {"decision": "deny", "reason": "pay_group_name_required", "required_action": "set_pay_group_name"}},
	{"name": "pay_group_frequency_supported", "description": "Pay group frequency must be from the supported set.", "condition": {"operation": "create_pay_group", "frequency_supported": False}, "effect": {"decision": "deny", "reason": "pay_frequency_not_supported", "required_action": "select_supported_frequency"}},
	{"name": "pay_group_currency_supported", "description": "Pay group currency must be from the supported set.", "condition": {"operation": "create_pay_group", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "pay_group_requires_country", "description": "Pay groups require a country (drives tax jurisdiction selection).", "condition": {"operation": "create_pay_group", "country_present": False}, "effect": {"decision": "deny", "reason": "country_required", "required_action": "set_country"}},
	{"name": "pay_group_requires_owner", "description": "Pay groups require an owner.", "condition": {"operation": "create_pay_group", "owner_present": False}, "effect": {"decision": "deny", "reason": "pay_group_owner_required", "required_action": "assign_pay_group_owner"}},
	{"name": "pay_group_requires_tax_jurisdiction", "description": "Pay groups require a tax jurisdiction (determines applicable tax rules).", "condition": {"operation": "create_pay_group", "tax_jurisdiction_present": False}, "effect": {"decision": "deny", "reason": "tax_jurisdiction_required", "required_action": "set_tax_jurisdiction"}},

	# --- Employee pay profile ---
	{"name": "profile_requires_employee", "description": "Employee pay profiles require an employee reference.", "condition": {"operation": "create_employee_pay_profile", "employee_present": False}, "effect": {"decision": "deny", "reason": "employee_required", "required_action": "select_employee"}},
	{"name": "profile_requires_pay_group", "description": "Employee pay profiles require a pay group.", "condition": {"operation": "create_employee_pay_profile", "pay_group_present": False}, "effect": {"decision": "deny", "reason": "pay_group_required", "required_action": "select_pay_group"}},
	{"name": "profile_payment_method_supported", "description": "Employee pay profile payment method must be from the supported set.", "condition": {"operation": "create_employee_pay_profile", "payment_method_supported": False}, "effect": {"decision": "deny", "reason": "payment_method_not_supported", "required_action": "select_supported_payment_method"}},
	{"name": "profile_requires_tax_id", "description": "Employee pay profiles require a tax ID.", "condition": {"operation": "create_employee_pay_profile", "tax_id_present": False}, "effect": {"decision": "deny", "reason": "tax_id_required", "required_action": "set_tax_id"}},
	{"name": "profile_currency_supported", "description": "Employee pay profile currency must be from the supported set.", "condition": {"operation": "create_employee_pay_profile", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "bank_profile_requires_review", "description": "Bank-transfer pay profiles require verification before activation.", "condition": {"operation": "create_employee_pay_profile", "bank_payment": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "bank_profile_review_required", "required_action": "record_bank_profile_review"}},
	{"name": "duplicate_active_profile_denied", "description": "An employee cannot have two active profiles in the same pay group.", "condition": {"operation": "create_employee_pay_profile", "duplicate_active_profile": True}, "effect": {"decision": "deny", "reason": "duplicate_active_pay_profile", "required_action": "deactivate_existing_profile_first"}},

	# --- Pay component ---
	{"name": "component_requires_code", "description": "Pay components require a code.", "condition": {"operation": "create_pay_component", "code_present": False}, "effect": {"decision": "deny", "reason": "component_code_required", "required_action": "set_component_code"}},
	{"name": "component_requires_name", "description": "Pay components require a name.", "condition": {"operation": "create_pay_component", "name_present": False}, "effect": {"decision": "deny", "reason": "component_name_required", "required_action": "set_component_name"}},
	{"name": "component_type_supported", "description": "Pay component type must be from the supported set.", "condition": {"operation": "create_pay_component", "component_type_supported": False}, "effect": {"decision": "deny", "reason": "component_type_not_supported", "required_action": "select_supported_component_type"}},
	{"name": "component_currency_supported", "description": "Pay component currency must be from the supported set.", "condition": {"operation": "create_pay_component", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "component_requires_taxable_flag", "description": "Pay components must declare their taxability.", "condition": {"operation": "create_pay_component", "taxable_flag_present": False}, "effect": {"decision": "deny", "reason": "taxable_flag_required", "required_action": "set_taxable_flag"}},
	{"name": "component_requires_gl_account", "description": "Pay components require a GL account code for ledger integration.", "condition": {"operation": "create_pay_component", "gl_account_present": False}, "effect": {"decision": "deny", "reason": "gl_account_required", "required_action": "set_gl_account"}},

	# --- Tax calculation rules ---
	{"name": "tax_rule_requires_jurisdiction", "description": "Tax calculation rules require a jurisdiction.", "condition": {"operation": "create_tax_rule", "jurisdiction_present": False}, "effect": {"decision": "deny", "reason": "tax_jurisdiction_required", "required_action": "set_tax_jurisdiction"}},
	{"name": "tax_rule_requires_effective_date", "description": "Tax rules require an effective date to support rate changes over time.", "condition": {"operation": "create_tax_rule", "effective_date_present": False}, "effect": {"decision": "deny", "reason": "tax_rule_effective_date_required", "required_action": "set_tax_rule_effective_date"}},
	{"name": "tax_rule_bracket_type_supported", "description": "Tax rule bracket type must be from the supported set.", "condition": {"operation": "create_tax_rule", "bracket_type_supported": False}, "effect": {"decision": "deny", "reason": "tax_bracket_type_not_supported", "required_action": "select_supported_bracket_type"}},
	{"name": "progressive_tax_requires_brackets", "description": "Progressive tax rules must define income brackets with rates.", "condition": {"operation": "create_tax_rule", "bracket_type": "progressive", "brackets_present": False}, "effect": {"decision": "deny", "reason": "progressive_tax_brackets_required", "required_action": "define_tax_brackets"}},
	{"name": "flat_rate_tax_requires_rate", "description": "Flat-rate tax rules must specify the rate.", "condition": {"operation": "create_tax_rule", "bracket_type": "flat_rate", "flat_rate_present": False}, "effect": {"decision": "deny", "reason": "flat_rate_required", "required_action": "set_flat_tax_rate"}},
	{"name": "cap_based_tax_requires_cap_amount", "description": "Cap-based tax rules must specify the cap amount.", "condition": {"operation": "create_tax_rule", "bracket_type": "cap_based", "cap_amount_present": False}, "effect": {"decision": "deny", "reason": "cap_amount_required", "required_action": "set_tax_cap_amount"}},
	{"name": "tax_rule_requires_employer_rate", "description": "Tax rules must declare the employer contribution rate.", "condition": {"operation": "create_tax_rule", "employer_rate_present": False}, "effect": {"decision": "deny", "reason": "employer_tax_rate_required", "required_action": "set_employer_tax_rate"}},
	{"name": "tax_rule_requires_employee_rate", "description": "Tax rules must declare the employee contribution rate.", "condition": {"operation": "create_tax_rule", "employee_rate_present": False}, "effect": {"decision": "deny", "reason": "employee_tax_rate_required", "required_action": "set_employee_tax_rate"}},
	{"name": "tax_rate_exceeds_100_denied", "description": "Tax rate (employee + employer combined) cannot exceed 100%.", "condition": {"operation": "create_tax_rule", "combined_rate_gt": 100}, "effect": {"decision": "deny", "reason": "combined_tax_rate_exceeds_100_percent", "required_action": "reduce_tax_rates"}},
	{"name": "tax_rule_annual_review_required", "description": "Tax rules older than 12 months require an annual review before being applied.", "condition": {"operation": "apply_tax_rule", "months_since_review_gt": 12}, "effect": {"decision": "require_review", "reason": "annual_tax_rule_review_required", "required_action": "conduct_annual_tax_rule_review"}},
	{"name": "tax_rule_deletion_requires_no_open_runs", "description": "A tax rule in use by open payroll runs cannot be deleted.", "condition": {"operation": "delete_tax_rule", "used_in_open_runs": True}, "effect": {"decision": "deny", "reason": "tax_rule_in_use_by_open_runs", "required_action": "complete_or_void_open_runs_first"}},

	# --- Time imports ---
	{"name": "time_import_requires_period", "description": "Time imports require a payroll period.", "condition": {"operation": "record_time_import", "period_present": False}, "effect": {"decision": "deny", "reason": "period_required", "required_action": "select_period"}},
	{"name": "time_import_requires_employee", "description": "Time imports require an employee pay profile.", "condition": {"operation": "record_time_import", "profile_present": False}, "effect": {"decision": "deny", "reason": "employee_profile_required", "required_action": "select_employee_profile"}},
	{"name": "time_import_hours_nonnegative", "description": "Time import hours cannot be negative.", "condition": {"operation": "record_time_import", "hours_lt": 0}, "effect": {"decision": "deny", "reason": "hours_invalid", "required_action": "set_valid_hours"}},
	{"name": "time_import_requires_source", "description": "Time imports require a source system reference.", "condition": {"operation": "record_time_import", "source_present": False}, "effect": {"decision": "deny", "reason": "time_source_required", "required_action": "set_time_source"}},
	{"name": "overtime_requires_approval", "description": "Overtime time imports require prior approval.", "condition": {"operation": "record_time_import", "overtime": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "overtime_approval_required", "required_action": "record_overtime_approval"}},
	{"name": "overtime_rate_required_for_overtime_import", "description": "Overtime imports require the overtime rate multiplier.", "condition": {"operation": "record_time_import", "overtime": True, "overtime_rate_present": False}, "effect": {"decision": "deny", "reason": "overtime_rate_multiplier_required", "required_action": "set_overtime_rate_multiplier"}},

	# --- Payroll runs ---
	{"name": "run_requires_period", "description": "Payroll runs require a period.", "condition": {"operation": "start_payroll_run", "period_present": False}, "effect": {"decision": "deny", "reason": "period_required", "required_action": "select_period"}},
	{"name": "run_requires_pay_group", "description": "Payroll runs require a pay group.", "condition": {"operation": "start_payroll_run", "pay_group_present": False}, "effect": {"decision": "deny", "reason": "pay_group_required", "required_action": "select_pay_group"}},
	{"name": "run_requires_initiator", "description": "Payroll runs require an initiator.", "condition": {"operation": "start_payroll_run", "initiator_present": False}, "effect": {"decision": "deny", "reason": "run_initiator_required", "required_action": "set_run_initiator"}},
	{"name": "duplicate_run_same_period_denied", "description": "Cannot start a new payroll run for a pay group that already has an active run in the same period.", "condition": {"operation": "start_payroll_run", "active_run_exists": True}, "effect": {"decision": "deny", "reason": "duplicate_payroll_run_same_period", "required_action": "void_or_complete_existing_run_first"}},
	{"name": "variance_exceeding_threshold_requires_review", "description": "Payroll runs where gross variance exceeds threshold require variance review.", "condition": {"operation": "approve_payroll_run", "variance_pct_gt": 10, "variance_review_recorded": False}, "effect": {"decision": "require_review", "reason": "variance_review_required", "required_action": "record_variance_review"}},
	{"name": "run_reversal_requires_reason", "description": "Reversing a posted payroll run requires a documented reason.", "condition": {"operation": "reverse_payroll_run", "reversal_reason_present": False}, "effect": {"decision": "deny", "reason": "reversal_reason_required", "required_action": "set_reversal_reason"}},
	{"name": "run_reversal_reason_supported", "description": "Reversal reason must be from the supported set.", "condition": {"operation": "reverse_payroll_run", "reversal_reason_supported": False}, "effect": {"decision": "deny", "reason": "reversal_reason_not_supported", "required_action": "select_supported_reversal_reason"}},

	# --- Line items ---
	{"name": "line_requires_run", "description": "Payroll line items require a run.", "condition": {"operation": "add_line_item", "run_present": False}, "effect": {"decision": "deny", "reason": "run_required", "required_action": "select_run"}},
	{"name": "line_requires_profile", "description": "Payroll line items require an employee pay profile.", "condition": {"operation": "add_line_item", "profile_present": False}, "effect": {"decision": "deny", "reason": "employee_profile_required", "required_action": "select_employee_profile"}},
	{"name": "line_requires_component", "description": "Payroll line items require a pay component.", "condition": {"operation": "add_line_item", "component_present": False}, "effect": {"decision": "deny", "reason": "component_required", "required_action": "select_component"}},
	{"name": "line_requires_amount", "description": "Payroll line items require an amount.", "condition": {"operation": "add_line_item", "amount_present": False}, "effect": {"decision": "deny", "reason": "amount_required", "required_action": "set_amount"}},
	{"name": "negative_line_requires_review", "description": "Negative payroll line amounts require review.", "condition": {"operation": "add_line_item", "negative_amount": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "negative_amount_review_required", "required_action": "record_negative_amount_review"}},
	{"name": "line_item_on_posted_run_denied", "description": "Line items cannot be added to an already-posted run.", "condition": {"operation": "add_line_item", "run_status": "posted"}, "effect": {"decision": "deny", "reason": "cannot_modify_posted_run", "required_action": "create_adjustment_instead"}},

	# --- Taxes ---
	{"name": "tax_requires_run", "description": "Payroll tax records require a run.", "condition": {"operation": "record_tax", "run_present": False}, "effect": {"decision": "deny", "reason": "run_required", "required_action": "select_run"}},
	{"name": "tax_requires_profile", "description": "Payroll tax records require an employee pay profile.", "condition": {"operation": "record_tax", "profile_present": False}, "effect": {"decision": "deny", "reason": "employee_profile_required", "required_action": "select_employee_profile"}},
	{"name": "tax_scope_supported", "description": "Payroll tax scope must be from the supported set.", "condition": {"operation": "record_tax", "tax_scope_supported": False}, "effect": {"decision": "deny", "reason": "tax_scope_not_supported", "required_action": "select_supported_tax_scope"}},
	{"name": "tax_requires_authority", "description": "Payroll tax records require a tax authority.", "condition": {"operation": "record_tax", "authority_present": False}, "effect": {"decision": "deny", "reason": "tax_authority_required", "required_action": "set_tax_authority"}},
	{"name": "tax_requires_amount", "description": "Payroll tax records require an amount.", "condition": {"operation": "record_tax", "amount_present": False}, "effect": {"decision": "deny", "reason": "amount_required", "required_action": "set_amount"}},
	{"name": "tax_requires_rule_reference", "description": "Payroll tax records must reference the tax rule used for calculation.", "condition": {"operation": "record_tax", "tax_rule_reference_present": False}, "effect": {"decision": "deny", "reason": "tax_rule_reference_required", "required_action": "set_tax_rule_reference"}},
	{"name": "tax_without_rule_blocks_run_approval", "description": "Payroll runs with tax lines lacking rule references cannot be approved.", "condition": {"operation": "approve_payroll_run", "tax_lines_missing_rules": True}, "effect": {"decision": "deny", "reason": "unlinked_tax_lines_block_approval", "required_action": "link_all_tax_lines_to_rules"}},

	# --- Adjustments ---
	{"name": "adjustment_requires_run", "description": "Payroll adjustments require a run.", "condition": {"operation": "record_adjustment", "run_present": False}, "effect": {"decision": "deny", "reason": "run_required", "required_action": "select_run"}},
	{"name": "adjustment_requires_profile", "description": "Payroll adjustments require an employee pay profile.", "condition": {"operation": "record_adjustment", "profile_present": False}, "effect": {"decision": "deny", "reason": "employee_profile_required", "required_action": "select_employee_profile"}},
	{"name": "adjustment_requires_reason", "description": "Payroll adjustments require a documented reason.", "condition": {"operation": "record_adjustment", "reason_present": False}, "effect": {"decision": "deny", "reason": "adjustment_reason_required", "required_action": "set_adjustment_reason"}},
	{"name": "adjustment_requires_approval", "description": "Payroll adjustments require approval.", "condition": {"operation": "record_adjustment", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "adjustment_approval_required", "required_action": "record_adjustment_approval"}},
	{"name": "large_adjustment_requires_executive_approval", "description": "Adjustments exceeding 50% of base pay require executive approval.", "condition": {"operation": "record_adjustment", "adjustment_pct_gt": 50, "executive_approval_recorded": False}, "effect": {"decision": "deny", "reason": "large_adjustment_requires_executive_approval", "required_action": "obtain_executive_approval"}},

	# --- Approval and posting ---
	{"name": "approval_requires_run", "description": "Payroll approval requires a run reference.", "condition": {"operation": "approve_payroll_run", "run_present": False}, "effect": {"decision": "deny", "reason": "run_required", "required_action": "select_run"}},
	{"name": "approval_requires_approver", "description": "Payroll approval requires an approver.", "condition": {"operation": "approve_payroll_run", "approver_present": False}, "effect": {"decision": "deny", "reason": "payroll_approver_required", "required_action": "assign_payroll_approver"}},
	{"name": "posting_requires_approval", "description": "Payroll posting requires prior approval.", "condition": {"operation": "post_payroll_run", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "payroll_approval_required", "required_action": "approve_payroll_run"}},

	# --- Payments ---
	{"name": "payment_requires_run", "description": "Payment batches require a run reference.", "condition": {"operation": "create_payment_batch", "run_present": False}, "effect": {"decision": "deny", "reason": "run_required", "required_action": "select_run"}},
	{"name": "payment_requires_approval", "description": "Payment batches require an approved and posted run.", "condition": {"operation": "create_payment_batch", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "payment_approval_required", "required_action": "approve_payroll_run"}},
	{"name": "payment_requires_date", "description": "Payment batches require a payment date.", "condition": {"operation": "create_payment_batch", "payment_date_present": False}, "effect": {"decision": "deny", "reason": "payment_date_required", "required_action": "set_payment_date"}},
	{"name": "payment_net_positive", "description": "Payment batches require positive net pay.", "condition": {"operation": "create_payment_batch", "net_pay_lte": 0}, "effect": {"decision": "deny", "reason": "net_pay_invalid", "required_action": "calculate_positive_net_pay"}},
	{"name": "payment_bank_details_verified", "description": "Bank-transfer payments require verified bank details.", "condition": {"operation": "create_payment_batch", "bank_payment": True, "bank_details_verified": False}, "effect": {"decision": "deny", "reason": "bank_details_not_verified", "required_action": "verify_bank_details"}},

	# --- Payslips ---
	{"name": "payslip_requires_run", "description": "Payslips require a run reference.", "condition": {"operation": "publish_payslip", "run_present": False}, "effect": {"decision": "deny", "reason": "run_required", "required_action": "select_run"}},
	{"name": "payslip_requires_profile", "description": "Payslips require an employee pay profile.", "condition": {"operation": "publish_payslip", "profile_present": False}, "effect": {"decision": "deny", "reason": "employee_profile_required", "required_action": "select_employee_profile"}},
	{"name": "payslip_requires_posted_run", "description": "Payslips can only be published for posted runs.", "condition": {"operation": "publish_payslip", "posted_run": False}, "effect": {"decision": "deny", "reason": "posted_run_required", "required_action": "post_payroll_run"}},
	{"name": "payslip_requires_privacy_basis", "description": "Payslips require a privacy basis declaration.", "condition": {"operation": "publish_payslip", "privacy_basis_present": False}, "effect": {"decision": "deny", "reason": "privacy_basis_required", "required_action": "set_privacy_basis"}},

	# --- Tax filings ---
	{"name": "filing_requires_run", "description": "Tax filings require a run reference.", "condition": {"operation": "create_tax_filing", "run_present": False}, "effect": {"decision": "deny", "reason": "run_required", "required_action": "select_run"}},
	{"name": "filing_requires_authority", "description": "Tax filings require a tax authority.", "condition": {"operation": "create_tax_filing", "authority_present": False}, "effect": {"decision": "deny", "reason": "tax_authority_required", "required_action": "set_tax_authority"}},
	{"name": "filing_requires_period", "description": "Tax filings require a period reference.", "condition": {"operation": "create_tax_filing", "period_present": False}, "effect": {"decision": "deny", "reason": "period_required", "required_action": "select_period"}},
	{"name": "filing_requires_approval", "description": "Tax filings require approval before submission.", "condition": {"operation": "create_tax_filing", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "tax_filing_approval_required", "required_action": "record_tax_filing_approval"}},
	{"name": "late_filing_requires_penalty_assessment", "description": "Tax filings submitted after the statutory deadline must include a penalty assessment.", "condition": {"operation": "create_tax_filing", "past_deadline": True, "penalty_assessed": False}, "effect": {"decision": "require_review", "reason": "late_filing_penalty_assessment_required", "required_action": "assess_late_filing_penalty"}},

	# --- Streaming / agents ---
	{"name": "payroll_batch_requires_bytewax", "description": "Payroll batch operations must be routed through the Bytewax event stream.", "condition": {"operation": "payroll_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_payroll_batch_to_bytewax"}},
	{"name": "payroll_event_requires_bytewax", "description": "Payroll lifecycle events must be published to the Bytewax stream.", "condition": {"operation": "payroll_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_payroll_event_to_bytewax"}},
	{"name": "payroll_agent_runtime_supported", "description": "Payroll agents must use an approved runtime.", "condition": {"operation": "register_payroll_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "payroll_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "payroll_agent_role_supported", "description": "Payroll agents must use an approved role.", "condition": {"operation": "register_payroll_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "payroll_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_payroll_agent_action_requires_human_approval", "description": "Privileged payroll actions proposed by agents require human approval before execution.", "condition": {"operation": "payroll_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def _configuration_schema() -> dict[str, Any]:
	return {
		"type": "object",
		"required": ["tenant_id", "ui", "theme"],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
		} | {
			key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"
		},
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
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"provides": PROVIDES,
		"requires": REQUIRES,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/hcm/payroll/api/v1",
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
