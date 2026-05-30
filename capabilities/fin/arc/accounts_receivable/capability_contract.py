"""Executable capability contract for Accounts Receivable."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "arc_accounts_receivable"
CAPABILITY_NAME = "Accounts Receivable"
CAPABILITY_VERSION = "2.1.0"
ARC_EVENT_STREAM = "apg.fin.arc.lifecycle"

SUPPORTED_CUSTOMER_TYPES = ["business", "government", "individual", "intercompany"]
SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_INVOICE_STATUSES = ["draft", "approved", "issued", "partially_paid", "paid", "void"]
SUPPORTED_PAYMENT_METHODS = ["bank_transfer", "card", "cash", "check", "mobile_money", "lockbox"]
SUPPORTED_DISPUTE_REASONS = ["pricing", "quantity", "tax", "delivery", "quality", "duplicate", "other"]
SUPPORTED_ARC_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ARC_AGENT_ROLES = [
	"credit_reviewer",
	"invoice_reviewer",
	"cash_application_reviewer",
	"collections_reviewer",
	"dispute_reviewer",
	"revenue_recognition_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"customers": {
		"customer_code_required": True,
		"legal_name_required": True,
		"customer_type_required": True,
		"supported_customer_types": SUPPORTED_CUSTOMER_TYPES,
		"credit_profile_required_for_invoicing": True,
	},
	"credit": {
		"credit_limit_required": True,
		"credit_score_review_threshold": 0.6,
		"credit_hold_blocks_invoice_issue": True,
		"review_required_for_limit_increase": True,
	},
	"invoices": {
		"customer_required": True,
		"invoice_number_required": True,
		"invoice_date_required": True,
		"due_date_required": True,
		"line_required": True,
		"positive_total_required": True,
		"approval_required_for_issue": True,
		"supported_statuses": SUPPORTED_INVOICE_STATUSES,
	},
	"invoice_lines": {
		"description_required": True,
		"quantity_positive_required": True,
		"unit_price_nonnegative_required": True,
		"revenue_account_required": True,
	},
	"payments": {
		"customer_required": True,
		"payment_reference_required": True,
		"payment_date_required": True,
		"amount_positive_required": True,
		"supported_methods": SUPPORTED_PAYMENT_METHODS,
		"cash_account_required": True,
	},
	"cash_application": {
		"payment_required": True,
		"invoice_required": True,
		"allocation_amount_positive_required": True,
		"overapplication_blocked": True,
		"unapplied_cash_requires_review": True,
	},
	"collections": {
		"overdue_invoice_required": True,
		"contact_method_required": True,
		"priority_required": True,
		"promise_to_pay_review_required": True,
	},
	"disputes": {
		"invoice_required": True,
		"reason_required": True,
		"supported_reasons": SUPPORTED_DISPUTE_REASONS,
		"owner_required": True,
		"resolution_review_required": True,
	},
	"arc_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_ARC_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_ARC_AGENT_ROLES,
		"max_autonomous_scope": "recommend_validate_and_prepare",
		"human_approval_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_state_changes": True,
		"segregation_of_duties": True,
	},
	"observability": {
		"event_stream": ARC_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_customer_events": True,
		"emit_credit_events": True,
		"emit_invoice_events": True,
		"emit_payment_events": True,
		"emit_cash_application_events": True,
		"emit_collection_events": True,
		"emit_dispute_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"document_management": "adapter",
		"business_intelligence": "adapter",
		"general_ledger": "adapter",
		"cash_management": "adapter",
		"customer_relationship_management": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_customers": True,
		"enable_credit": True,
		"enable_invoices": True,
		"enable_payments": True,
		"enable_cash_application": True,
		"enable_collections": True,
		"enable_disputes": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "arc_accounts_receivable_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"customer_receivable_lifecycle",
	"credit_assessment_workflow",
	"invoice_lifecycle",
	"invoice_line_management",
	"payment_receipt_lifecycle",
	"cash_application_workflow",
	"collections_workflow",
	"dispute_resolution_workflow",
	"receivables_aging_service",
	"arc_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"composition_events",
	"composition_config",
	"general_ledger",
	"cash_management",
	"document_management",
	"business_intelligence",
	"customer_relationship_management",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/arc-accounts-receivable/dashboard", "component": "AccountsReceivableDashboard", "permission": "arc_accounts_receivable:view", "nav_group": "Overview"},
	{"name": "customers", "path": "/arc-accounts-receivable/customers", "component": "CustomerReceivablesWorkbench", "permission": "arc_accounts_receivable:manage_customers", "nav_group": "Customers"},
	{"name": "credit", "path": "/arc-accounts-receivable/credit", "component": "CreditAssessmentWorkbench", "permission": "arc_accounts_receivable:credit", "nav_group": "Customers"},
	{"name": "invoices", "path": "/arc-accounts-receivable/invoices", "component": "InvoiceWorkbench", "permission": "arc_accounts_receivable:invoice", "nav_group": "Invoices"},
	{"name": "payments", "path": "/arc-accounts-receivable/payments", "component": "PaymentReceiptConsole", "permission": "arc_accounts_receivable:receive_payments", "nav_group": "Cash"},
	{"name": "cash_application", "path": "/arc-accounts-receivable/cash-application", "component": "CashApplicationWorkbench", "permission": "arc_accounts_receivable:apply_cash", "nav_group": "Cash"},
	{"name": "collections", "path": "/arc-accounts-receivable/collections", "component": "CollectionsWorkbench", "permission": "arc_accounts_receivable:collect", "nav_group": "Collections"},
	{"name": "disputes", "path": "/arc-accounts-receivable/disputes", "component": "DisputeResolutionWorkbench", "permission": "arc_accounts_receivable:resolve_disputes", "nav_group": "Collections"},
	{"name": "aging", "path": "/arc-accounts-receivable/aging", "component": "ReceivablesAgingConsole", "permission": "arc_accounts_receivable:report", "nav_group": "Reports"},
	{"name": "agents", "path": "/arc-accounts-receivable/agents", "component": "ARCAgentWorkbench", "permission": "arc_accounts_receivable:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/arc-accounts-receivable/settings", "component": "AccountsReceivableSettings", "permission": "arc_accounts_receivable:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "arc_accounts_receivable_control",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#C44536",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"customers": {"icon": "users", "status_indicator": "customer-pill", "risk_style": "credit-band"},
		"credit": {"visual": "score-lane", "status_style": "risk-chip"},
		"invoices": {"visual": "invoice-grid", "status_style": "invoice-chip"},
		"payments": {"visual": "receipt-list", "status_style": "payment-chip"},
		"cash_application": {"visual": "allocation-grid", "status_style": "match-chip"},
		"collections": {"visual": "collection-queue", "status_style": "priority-chip"},
		"disputes": {"visual": "resolution-board", "status_style": "dispute-chip"},
		"agents": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": ARC_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"customer_created",
		"credit_assessed",
		"invoice_created",
		"invoice_issued",
		"payment_recorded",
		"cash_applied",
		"collection_activity_recorded",
		"dispute_opened",
		"dispute_resolved",
		"arc_agent_registered",
	],
	"states": ["draft", "active", "assessed", "approved", "issued", "paid", "applied", "overdue", "disputed", "resolved", "blocked"],
	"guardrails": [
		"arc_batch_requires_bytewax",
		"arc_event_requires_bytewax",
		"privileged_agent_arc_action_requires_human_approval",
	],
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Accounts receivable operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "arc_write_requires_policy", "description": "Accounts receivable writes require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "customer_requires_code", "description": "Customers require a customer code.", "condition": {"operation": "create_customer", "customer_code_present": False}, "effect": {"decision": "deny", "reason": "customer_code_required", "required_action": "set_customer_code"}},
	{"name": "customer_requires_legal_name", "description": "Customers require a legal name.", "condition": {"operation": "create_customer", "legal_name_present": False}, "effect": {"decision": "deny", "reason": "legal_name_required", "required_action": "set_legal_name"}},
	{"name": "customer_type_supported", "description": "Customer type must be supported.", "condition": {"operation": "create_customer", "customer_type_supported": False}, "effect": {"decision": "deny", "reason": "customer_type_not_supported", "required_action": "select_supported_customer_type"}},
	{"name": "credit_requires_customer", "description": "Credit assessments require customer.", "condition": {"operation": "assess_credit", "customer_present": False}, "effect": {"decision": "deny", "reason": "customer_required", "required_action": "attach_customer"}},
	{"name": "credit_limit_required", "description": "Credit assessments require credit limit.", "condition": {"operation": "assess_credit", "credit_limit_present": False}, "effect": {"decision": "deny", "reason": "credit_limit_required", "required_action": "set_credit_limit"}},
	{"name": "low_credit_score_requires_review", "description": "Low credit score requires review.", "condition": {"operation": "assess_credit", "credit_score_lt": 0.6, "credit_review_recorded": False}, "effect": {"decision": "require_review", "reason": "credit_review_required", "required_action": "record_credit_review"}},
	{"name": "invoice_requires_customer", "description": "Invoices require customer.", "condition": {"operation": "create_invoice", "customer_present": False}, "effect": {"decision": "deny", "reason": "invoice_customer_required", "required_action": "attach_customer"}},
	{"name": "invoice_requires_number", "description": "Invoices require invoice number.", "condition": {"operation": "create_invoice", "invoice_number_present": False}, "effect": {"decision": "deny", "reason": "invoice_number_required", "required_action": "set_invoice_number"}},
	{"name": "invoice_requires_dates", "description": "Invoices require invoice and due dates.", "condition": {"operation": "create_invoice", "invoice_dates_present": False}, "effect": {"decision": "deny", "reason": "invoice_dates_required", "required_action": "set_invoice_dates"}},
	{"name": "invoice_due_after_invoice_date", "description": "Invoice due date must not precede invoice date.", "condition": {"operation": "create_invoice", "due_date_valid": False}, "effect": {"decision": "deny", "reason": "invoice_due_date_invalid", "required_action": "set_valid_due_date"}},
	{"name": "invoice_requires_lines", "description": "Invoices require line items.", "condition": {"operation": "create_invoice", "invoice_line_count_lte": 0}, "effect": {"decision": "deny", "reason": "invoice_lines_required", "required_action": "add_invoice_lines"}},
	{"name": "invoice_total_positive", "description": "Invoice total must be positive.", "condition": {"operation": "create_invoice", "invoice_total_lte": 0}, "effect": {"decision": "deny", "reason": "invoice_total_positive_required", "required_action": "set_positive_invoice_total"}},
	{"name": "invoice_blocks_credit_hold", "description": "Credit hold blocks invoice issue.", "condition": {"operation": "issue_invoice", "credit_hold": True}, "effect": {"decision": "deny", "reason": "customer_credit_hold", "required_action": "release_credit_hold"}},
	{"name": "invoice_issue_requires_approval", "description": "Invoice issue requires approval.", "condition": {"operation": "issue_invoice", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "invoice_approval_required", "required_action": "approve_invoice"}},
	{"name": "payment_requires_customer", "description": "Payments require customer.", "condition": {"operation": "record_payment", "customer_present": False}, "effect": {"decision": "deny", "reason": "payment_customer_required", "required_action": "attach_customer"}},
	{"name": "payment_requires_reference", "description": "Payments require reference.", "condition": {"operation": "record_payment", "payment_reference_present": False}, "effect": {"decision": "deny", "reason": "payment_reference_required", "required_action": "set_payment_reference"}},
	{"name": "payment_requires_date", "description": "Payments require payment date.", "condition": {"operation": "record_payment", "payment_date_present": False}, "effect": {"decision": "deny", "reason": "payment_date_required", "required_action": "set_payment_date"}},
	{"name": "payment_amount_positive", "description": "Payment amount must be positive.", "condition": {"operation": "record_payment", "payment_amount_lte": 0}, "effect": {"decision": "deny", "reason": "payment_amount_positive_required", "required_action": "set_positive_payment_amount"}},
	{"name": "payment_method_supported", "description": "Payment method must be supported.", "condition": {"operation": "record_payment", "payment_method_supported": False}, "effect": {"decision": "deny", "reason": "payment_method_not_supported", "required_action": "select_supported_payment_method"}},
	{"name": "payment_requires_cash_account", "description": "Payments require cash account.", "condition": {"operation": "record_payment", "cash_account_present": False}, "effect": {"decision": "deny", "reason": "cash_account_required", "required_action": "attach_cash_account"}},
	{"name": "cash_application_requires_payment", "description": "Cash application requires payment.", "condition": {"operation": "apply_cash", "payment_present": False}, "effect": {"decision": "deny", "reason": "payment_required", "required_action": "select_payment"}},
	{"name": "cash_application_requires_invoice", "description": "Cash application requires invoice.", "condition": {"operation": "apply_cash", "invoice_present": False}, "effect": {"decision": "deny", "reason": "invoice_required", "required_action": "select_invoice"}},
	{"name": "cash_application_amount_positive", "description": "Cash application amount must be positive.", "condition": {"operation": "apply_cash", "allocation_amount_lte": 0}, "effect": {"decision": "deny", "reason": "cash_application_amount_positive_required", "required_action": "set_positive_allocation"}},
	{"name": "cash_application_blocks_overapplication", "description": "Cash application cannot exceed invoice outstanding balance.", "condition": {"operation": "apply_cash", "overapplication": True}, "effect": {"decision": "deny", "reason": "cash_overapplication_blocked", "required_action": "reduce_allocation"}},
	{"name": "unapplied_cash_requires_review", "description": "Unapplied cash requires review.", "condition": {"operation": "apply_cash", "unapplied_amount_gt": 0, "cash_application_review_recorded": False}, "effect": {"decision": "require_review", "reason": "unapplied_cash_review_required", "required_action": "record_cash_application_review"}},
	{"name": "collection_requires_overdue_invoice", "description": "Collection activity requires overdue invoice.", "condition": {"operation": "record_collection_activity", "overdue_invoice_present": False}, "effect": {"decision": "deny", "reason": "overdue_invoice_required", "required_action": "select_overdue_invoice"}},
	{"name": "collection_requires_contact_method", "description": "Collection activity requires contact method.", "condition": {"operation": "record_collection_activity", "contact_method_present": False}, "effect": {"decision": "deny", "reason": "contact_method_required", "required_action": "set_contact_method"}},
	{"name": "collection_requires_priority", "description": "Collection activity requires priority.", "condition": {"operation": "record_collection_activity", "priority_present": False}, "effect": {"decision": "deny", "reason": "collection_priority_required", "required_action": "set_collection_priority"}},
	{"name": "dispute_requires_invoice", "description": "Disputes require invoice.", "condition": {"operation": "open_dispute", "invoice_present": False}, "effect": {"decision": "deny", "reason": "dispute_invoice_required", "required_action": "select_invoice"}},
	{"name": "dispute_reason_supported", "description": "Dispute reason must be supported.", "condition": {"operation": "open_dispute", "dispute_reason_supported": False}, "effect": {"decision": "deny", "reason": "dispute_reason_not_supported", "required_action": "select_supported_reason"}},
	{"name": "dispute_requires_owner", "description": "Disputes require owner.", "condition": {"operation": "open_dispute", "owner_present": False}, "effect": {"decision": "deny", "reason": "dispute_owner_required", "required_action": "assign_owner"}},
	{"name": "dispute_resolution_requires_review", "description": "Dispute resolution requires review.", "condition": {"operation": "resolve_dispute", "resolution_review_recorded": False}, "effect": {"decision": "deny", "reason": "dispute_resolution_review_required", "required_action": "record_resolution_review"}},
	{"name": "arc_batch_requires_bytewax", "description": "Accounts receivable batches require Bytewax coordination.", "condition": {"operation": "arc_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_arc_batch_to_bytewax"}},
	{"name": "arc_event_requires_bytewax", "description": "Accounts receivable lifecycle events require Bytewax.", "condition": {"operation": "arc_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_arc_event_to_bytewax"}},
	{"name": "arc_agent_runtime_supported", "description": "ARC agents must use an approved runtime.", "condition": {"operation": "register_arc_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "arc_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "arc_agent_role_supported", "description": "ARC agents must use an approved role.", "condition": {"operation": "register_arc_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "arc_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_arc_action_requires_human_approval", "description": "Privileged ARC actions proposed by agents require human approval.", "condition": {"operation": "agent_arc_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def _configuration_schema() -> dict[str, Any]:
	return {
		"type": "object",
		"required": list(DEFAULT_CONFIGURATION),
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
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"provides": PROVIDES,
		"requires": REQUIRES,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/arc-accounts-receivable/api/v1",
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
