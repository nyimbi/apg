"""Executable capability contract for APG accounts payable."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_AP_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AP_AGENT_ROLES = [
	"vendor_risk_reviewer",
	"invoice_exception_reviewer",
	"matching_reviewer",
	"payment_run_reviewer",
	"cash_flow_reviewer",
	"close_reviewer",
]
AP_EVENT_STREAM = "apg.fin.apy.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"vendors": {
		"owner_required": True,
		"tax_profile_required": True,
		"payment_method_required": True,
		"bank_review_required": True,
	},
	"invoices": {
		"vendor_required": True,
		"invoice_number_required": True,
		"currency_required": True,
		"positive_amount_required": True,
		"duplicate_check_required": True,
		"document_reference_required": True,
	},
	"matching": {
		"three_way_match_enabled": True,
		"variance_review_threshold": 0.03,
		"receipt_required_for_po_invoice": True,
	},
	"approvals": {
		"approval_required_for_high_value": True,
		"high_value_threshold": 10000,
		"separation_of_duties_required": True,
	},
	"payments": {
		"approved_invoice_required": True,
		"positive_amount_required": True,
		"cash_account_required": True,
		"payment_batch_review_required": True,
	},
	"holds": {"reason_required": True, "release_approval_required": True},
	"expenses": {"employee_required": True, "receipt_required": True, "policy_review_required": True},
	"close": {"open_exception_block": True, "unposted_invoice_block": True, "aging_review_required": True},
	"ap_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AP_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AP_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "recommend_validate_and_prepare",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"policy_attached_for_writes": True,
		"segregation_of_duties": True,
	},
	"observability": {
		"event_stream": AP_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_vendor_events": True,
		"emit_invoice_events": True,
		"emit_matching_events": True,
		"emit_payment_events": True,
		"emit_close_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"event_stream": "bytewax",
		"notification": "adapter",
		"general_ledger": "adapter",
		"cash_management": "adapter",
		"document_management": "adapter",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_vendors": True,
		"enable_invoices": True,
		"enable_matching": True,
		"enable_approvals": True,
		"enable_payments": True,
		"enable_expenses": True,
		"enable_aging": True,
		"enable_close": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "apy_accounts_payable_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"vendors",
		"invoices",
		"matching",
		"approvals",
		"payments",
		"holds",
		"expenses",
		"close",
		"ap_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"vendors": {"type": "object"},
		"invoices": {"type": "object"},
		"matching": {"type": "object"},
		"approvals": {"type": "object"},
		"payments": {"type": "object"},
		"holds": {"type": "object"},
		"expenses": {"type": "object"},
		"close": {"type": "object"},
		"ap_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "AP operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "ap_write_requires_policy", "description": "AP writes require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "vendor_requires_owner", "description": "Vendors require an accountable owner.", "condition": {"operation": "register_vendor", "vendor_owner_assigned": False}, "effect": {"decision": "deny", "reason": "vendor_owner_required", "required_action": "assign_vendor_owner"}},
	{"name": "vendor_requires_tax_profile", "description": "Vendors require tax profile evidence.", "condition": {"operation": "register_vendor", "tax_profile_present": False}, "effect": {"decision": "deny", "reason": "vendor_tax_profile_required", "required_action": "attach_tax_profile"}},
	{"name": "vendor_requires_payment_method", "description": "Vendors require payment method setup.", "condition": {"operation": "register_vendor", "payment_method_present": False}, "effect": {"decision": "deny", "reason": "vendor_payment_method_required", "required_action": "attach_payment_method"}},
	{"name": "vendor_bank_change_requires_review", "description": "Vendor bank changes require independent review.", "condition": {"operation": "register_vendor", "bank_change": True, "bank_review_recorded": False}, "effect": {"decision": "require_review", "reason": "vendor_bank_review_required", "required_action": "record_bank_review"}},
	{"name": "invoice_requires_vendor", "description": "Invoices require a registered vendor.", "condition": {"operation": "record_invoice", "vendor_present": False}, "effect": {"decision": "deny", "reason": "invoice_vendor_required", "required_action": "attach_vendor"}},
	{"name": "invoice_requires_number", "description": "Invoices require invoice number.", "condition": {"operation": "record_invoice", "invoice_number_present": False}, "effect": {"decision": "deny", "reason": "invoice_number_required", "required_action": "record_invoice_number"}},
	{"name": "invoice_requires_currency", "description": "Invoices require currency.", "condition": {"operation": "record_invoice", "currency_present": False}, "effect": {"decision": "deny", "reason": "invoice_currency_required", "required_action": "set_currency"}},
	{"name": "invoice_amount_positive", "description": "Invoice amount must be positive.", "condition": {"operation": "record_invoice", "amount_lte": 0}, "effect": {"decision": "deny", "reason": "invoice_amount_must_be_positive", "required_action": "set_positive_amount"}},
	{"name": "invoice_requires_document", "description": "Invoices require document reference.", "condition": {"operation": "record_invoice", "document_reference_present": False}, "effect": {"decision": "deny", "reason": "invoice_document_required", "required_action": "attach_invoice_document"}},
	{"name": "duplicate_invoice_requires_review", "description": "Potential duplicate invoices require review.", "condition": {"operation": "record_invoice", "duplicate_detected": True, "duplicate_review_recorded": False}, "effect": {"decision": "require_review", "reason": "duplicate_invoice_review_required", "required_action": "record_duplicate_review"}},
	{"name": "po_invoice_requires_receipt", "description": "PO-backed invoices require receipt evidence.", "condition": {"operation": "match_invoice", "po_backed": True, "receipt_present": False}, "effect": {"decision": "deny", "reason": "receipt_required_for_po_invoice", "required_action": "attach_receipt"}},
	{"name": "matching_variance_requires_review", "description": "Invoice match variance above threshold requires review.", "condition": {"operation": "match_invoice", "variance_rate_gt": 0.03, "variance_review_recorded": False}, "effect": {"decision": "require_review", "reason": "matching_variance_review_required", "required_action": "record_variance_review"}},
	{"name": "approval_requires_invoice", "description": "Approvals require an invoice.", "condition": {"operation": "approve_invoice", "invoice_present": False}, "effect": {"decision": "deny", "reason": "approval_invoice_required", "required_action": "attach_invoice"}},
	{"name": "high_value_invoice_requires_approval", "description": "High value invoices require approval.", "condition": {"operation": "approve_invoice", "amount_gt": 10000, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "high_value_invoice_approval_required", "required_action": "record_approval"}},
	{"name": "approval_requires_separation", "description": "Invoice requester cannot self-approve.", "condition": {"operation": "approve_invoice", "separation_of_duties_passed": False}, "effect": {"decision": "deny", "reason": "separation_of_duties_required", "required_action": "select_independent_approver"}},
	{"name": "payment_requires_approved_invoice", "description": "Payments require approved invoices.", "condition": {"operation": "schedule_payment", "invoice_approved": False}, "effect": {"decision": "deny", "reason": "approved_invoice_required", "required_action": "approve_invoice"}},
	{"name": "payment_amount_positive", "description": "Payment amount must be positive.", "condition": {"operation": "schedule_payment", "payment_amount_lte": 0}, "effect": {"decision": "deny", "reason": "payment_amount_must_be_positive", "required_action": "set_positive_payment_amount"}},
	{"name": "payment_requires_cash_account", "description": "Payments require cash account.", "condition": {"operation": "schedule_payment", "cash_account_present": False}, "effect": {"decision": "deny", "reason": "cash_account_required", "required_action": "attach_cash_account"}},
	{"name": "payment_batch_requires_review", "description": "Payment batches require review.", "condition": {"operation": "release_payment_batch", "batch_review_recorded": False}, "effect": {"decision": "deny", "reason": "payment_batch_review_required", "required_action": "record_batch_review"}},
	{"name": "hold_requires_reason", "description": "Invoice holds require a reason.", "condition": {"operation": "place_invoice_hold", "hold_reason_present": False}, "effect": {"decision": "deny", "reason": "hold_reason_required", "required_action": "record_hold_reason"}},
	{"name": "hold_release_requires_approval", "description": "Hold release requires approval.", "condition": {"operation": "release_invoice_hold", "release_approval_recorded": False}, "effect": {"decision": "deny", "reason": "hold_release_approval_required", "required_action": "record_release_approval"}},
	{"name": "expense_requires_employee", "description": "Expense reports require employee identity.", "condition": {"operation": "record_expense_report", "employee_present": False}, "effect": {"decision": "deny", "reason": "expense_employee_required", "required_action": "attach_employee"}},
	{"name": "expense_amount_positive", "description": "Expense report amount must be positive.", "condition": {"operation": "record_expense_report", "expense_amount_lte": 0}, "effect": {"decision": "deny", "reason": "expense_amount_must_be_positive", "required_action": "set_positive_expense_amount"}},
	{"name": "expense_requires_receipt", "description": "Expense reports require receipts.", "condition": {"operation": "record_expense_report", "receipt_present": False}, "effect": {"decision": "deny", "reason": "expense_receipt_required", "required_action": "attach_receipt"}},
	{"name": "expense_policy_exception_requires_review", "description": "Expense policy exceptions require review.", "condition": {"operation": "record_expense_report", "policy_exception": True, "policy_review_recorded": False}, "effect": {"decision": "require_review", "reason": "expense_policy_review_required", "required_action": "record_policy_review"}},
	{"name": "period_close_blocks_open_exceptions", "description": "AP period close blocks open exceptions.", "condition": {"operation": "close_period", "open_exception_count_gt": 0}, "effect": {"decision": "deny", "reason": "open_ap_exceptions_block_close", "required_action": "resolve_open_exceptions"}},
	{"name": "period_close_blocks_unposted_invoices", "description": "AP period close blocks unposted invoices.", "condition": {"operation": "close_period", "unposted_invoice_count_gt": 0}, "effect": {"decision": "deny", "reason": "unposted_invoices_block_close", "required_action": "post_or_hold_invoices"}},
	{"name": "period_close_requires_aging_review", "description": "AP period close requires aging review.", "condition": {"operation": "close_period", "aging_review_recorded": False}, "effect": {"decision": "deny", "reason": "aging_review_required", "required_action": "record_aging_review"}},
	{"name": "ap_batch_requires_bytewax", "description": "AP batches require Bytewax coordination.", "condition": {"operation": "ap_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_ap_batch_to_bytewax"}},
	{"name": "ap_event_requires_bytewax", "description": "AP lifecycle events require Bytewax.", "condition": {"operation": "ap_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_ap_event_to_bytewax"}},
	{"name": "ap_agent_runtime_supported", "description": "AP agents must use an approved runtime.", "condition": {"operation": "register_ap_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "ap_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "ap_agent_role_supported", "description": "AP agents must use an approved role.", "condition": {"operation": "register_ap_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "ap_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_ap_action_requires_human_approval", "description": "Privileged AP actions proposed by agents require human approval.", "condition": {"operation": "agent_ap_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/apy-accounts-payable/dashboard", "component": "APDashboard", "permission": "apy_accounts_payable:view", "nav_group": "Overview"},
	{"name": "vendors", "path": "/apy-accounts-payable/vendors", "component": "VendorRegistry", "permission": "apy_accounts_payable:manage_vendors", "nav_group": "Vendors"},
	{"name": "invoices", "path": "/apy-accounts-payable/invoices", "component": "InvoiceWorkbench", "permission": "apy_accounts_payable:manage_invoices", "nav_group": "Invoices"},
	{"name": "matching", "path": "/apy-accounts-payable/matching", "component": "InvoiceMatchingConsole", "permission": "apy_accounts_payable:match", "nav_group": "Invoices"},
	{"name": "approvals", "path": "/apy-accounts-payable/approvals", "component": "APApprovalQueue", "permission": "apy_accounts_payable:approve", "nav_group": "Approvals"},
	{"name": "payments", "path": "/apy-accounts-payable/payments", "component": "PaymentRunConsole", "permission": "apy_accounts_payable:pay", "nav_group": "Payments"},
	{"name": "expenses", "path": "/apy-accounts-payable/expenses", "component": "ExpenseReportConsole", "permission": "apy_accounts_payable:expenses", "nav_group": "Expenses"},
	{"name": "aging", "path": "/apy-accounts-payable/aging", "component": "APAgingAnalysis", "permission": "apy_accounts_payable:view", "nav_group": "Close"},
	{"name": "close", "path": "/apy-accounts-payable/close", "component": "APCloseWorkbench", "permission": "apy_accounts_payable:close", "nav_group": "Close"},
	{"name": "agents", "path": "/apy-accounts-payable/agents", "component": "APAgentWorkbench", "permission": "apy_accounts_payable:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/apy-accounts-payable/settings", "component": "APSettings", "permission": "apy_accounts_payable:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "apy_accounts_payable_control",
	"tokens": {"color.primary": "#28536B", "color.accent": "#C44536", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"vendors": {"icon": "building-2", "status_indicator": "vendor-pill", "risk_style": "risk-band"},
		"invoices": {"visual": "invoice-table", "status_style": "invoice-chip"},
		"matching": {"visual": "match-grid", "status_style": "variance-chip"},
		"approvals": {"visual": "approval-queue", "status_style": "approval-chip"},
		"payments": {"visual": "payment-run", "status_style": "cash-chip"},
		"expenses": {"visual": "receipt-list", "status_style": "policy-chip"},
		"aging": {"visual": "aging-buckets", "status_style": "aging-chip"},
		"close": {"visual": "close-checklist", "status_style": "period-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "apy_accounts_payable",
		"display_name": "Accounts Payable",
		"provides": [
			"vendor_payables_lifecycle",
			"invoice_capture_and_matching",
			"approval_workflow",
			"payment_run_lifecycle",
			"expense_reimbursement_lifecycle",
			"ap_aging_and_close",
			"ap_agents",
		],
		"requires": ["auth", "audl", "ntfy", "composition_events", "composition_config", "glr_general_ledger", "cbm_cash_management", "grc_doc"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/apy-accounts-payable/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": AP_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"vendor_registered",
			"invoice_recorded",
			"invoice_matched",
			"invoice_approved",
			"invoice_hold_placed",
			"payment_scheduled",
			"payment_batch_released",
			"expense_report_recorded",
			"period_closed",
			"ap_agent_registered",
		],
		"states": ["draft", "active", "captured", "matched", "approved", "held", "scheduled", "paid", "closed", "blocked"],
		"guardrails": [
			"ap_batch_requires_bytewax",
			"ap_event_requires_bytewax",
			"privileged_agent_ap_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return AP_EVENT_STREAM


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			if not context.get(key[:-4], 0) <= expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gte"):
			if not context.get(key[:-4], 0) >= expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
