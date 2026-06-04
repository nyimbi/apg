"""Executable capability contract for APG Telecom Billing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "telecom_bil"
CAPABILITY_NAME = "Telecom Billing"
CAPABILITY_VERSION = "1.0.0"
BIL_EVENT_STREAM = "apg.telecom.bil.lifecycle"

SUPPORTED_BILL_CYCLE_TYPES = ["monthly", "bimonthly", "quarterly", "annual", "event_based", "prepaid_top_up", "custom"]
SUPPORTED_CHARGE_TYPES = ["recurring", "one_time", "usage_based", "overage", "roaming", "interconnect", "penalty", "credit", "adjustment", "tax"]
SUPPORTED_MEDIATION_STATUSES = ["raw", "normalised", "rated", "aggregated", "billed", "rejected", "held"]
SUPPORTED_INVOICE_STATUSES = ["draft", "pending_approval", "approved", "sent", "paid", "partially_paid", "overdue", "disputed", "cancelled", "written_off"]
SUPPORTED_DUNNING_STEPS = ["reminder_1", "reminder_2", "suspension_warning", "service_suspended", "legal_notice", "collections", "write_off"]
SUPPORTED_PAYMENT_METHODS = ["bank_transfer", "mobile_money", "credit_card", "debit_card", "direct_debit", "cheque", "cash", "voucher", "crypto"]
SUPPORTED_RATING_TYPES = ["flat_rate", "tiered", "volume", "stepped", "time_of_day", "geo_based", "contract_rate", "promotional"]
SUPPORTED_DISCOUNT_TYPES = ["loyalty", "promotional", "bulk", "bundle", "retention", "corporate", "staff", "seasonal"]
SUPPORTED_TAX_TYPES = ["vat", "withholding", "excise", "regulatory_levy", "universal_service_fund", "spectrum_fee"]
SUPPORTED_CONVERGENT_MODES = ["single_bill", "multi_account", "household", "corporate_group", "mvno_wholesale"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["mediation_processor", "rating_engine", "invoice_generator", "dunning_manager", "dispute_resolver"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"mediation": {"supported_statuses": SUPPORTED_MEDIATION_STATUSES, "normalisation_required": True, "duplicate_detection": True},
	"rating": {"supported_rating_types": SUPPORTED_RATING_TYPES, "charge_types": SUPPORTED_CHARGE_TYPES, "tax_types": SUPPORTED_TAX_TYPES, "contract_required": True},
	"bill_cycles": {"supported_types": SUPPORTED_BILL_CYCLE_TYPES, "cutoff_required": True, "grace_period_days": 5},
	"invoices": {"supported_statuses": SUPPORTED_INVOICE_STATUSES, "approval_required": True, "electronic_delivery": True},
	"dunning": {"supported_steps": SUPPORTED_DUNNING_STEPS, "escalation_days": [7, 14, 21, 30, 45, 60], "suspension_enabled": True},
	"payments": {"supported_methods": SUPPORTED_PAYMENT_METHODS, "reconciliation_required": True, "partial_payment_allowed": True},
	"discounts": {"supported_types": SUPPORTED_DISCOUNT_TYPES, "approval_required": True, "max_discount_pct": 50},
	"convergent": {"supported_modes": SUPPORTED_CONVERGENT_MODES, "cross_account_credits": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "bill_suppression_denied": True, "unapproved_write_off_denied": True, "cross_tenant_billing_denied": True},
	"observability": {"event_stream": BIL_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_mediation": True, "enable_rating": True, "enable_bill_cycles": True, "enable_invoices": True, "enable_dunning": True, "enable_payments": True, "enable_discounts": True, "enable_agents": True},
	"theme": {"default_theme": "telecom_bil_control", "allow_tenant_overrides": True},
}

PROVIDES = ["mediation_workflow", "rating_workflow", "charging_workflow", "invoice_workflow", "bill_cycle_management", "dunning_workflow", "payment_reconciliation_workflow", "discount_workflow", "convergent_billing_workflow", "billing_agent_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "mqeb", "schd", "comp"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/telecom-bil/dashboard", "component": "BilDashboard", "permission": "telecom_bil:view", "nav_group": "Overview"},
	{"name": "mediation", "path": "/telecom-bil/mediation", "component": "BilMediationConsole", "permission": "telecom_bil:mediation", "nav_group": "Processing"},
	{"name": "rating", "path": "/telecom-bil/rating", "component": "BilRatingEngine", "permission": "telecom_bil:rating", "nav_group": "Processing"},
	{"name": "bill_cycles", "path": "/telecom-bil/bill-cycles", "component": "BilCycleManager", "permission": "telecom_bil:bill_cycles", "nav_group": "Billing"},
	{"name": "invoices", "path": "/telecom-bil/invoices", "component": "BilInvoiceConsole", "permission": "telecom_bil:invoices", "nav_group": "Billing"},
	{"name": "dunning", "path": "/telecom-bil/dunning", "component": "BilDunningConsole", "permission": "telecom_bil:dunning", "nav_group": "Collections"},
	{"name": "payments", "path": "/telecom-bil/payments", "component": "BilPaymentLedger", "permission": "telecom_bil:payments", "nav_group": "Payments"},
	{"name": "discounts", "path": "/telecom-bil/discounts", "component": "BilDiscountWorkbench", "permission": "telecom_bil:discounts", "nav_group": "Promotions"},
	{"name": "convergent", "path": "/telecom-bil/convergent", "component": "BilConvergentConsole", "permission": "telecom_bil:convergent", "nav_group": "Billing"},
	{"name": "disputes", "path": "/telecom-bil/disputes", "component": "BilDisputeConsole", "permission": "telecom_bil:disputes", "nav_group": "Support"},
	{"name": "agents", "path": "/telecom-bil/agents", "component": "BilAgentWorkbench", "permission": "telecom_bil:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/telecom-bil/settings", "component": "BilSettings", "permission": "telecom_bil:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "telecom_bil_control",
	"tokens": {"color.primary": "#065F46", "color.accent": "#0369A1", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"mediation": {"icon": "filter", "status_indicator": "mediation-status-chip"}, "rating": {"icon": "calculator", "status_indicator": "rating-chip"}, "bill_cycles": {"icon": "calendar", "status_indicator": "cycle-chip"}, "invoices": {"icon": "file-invoice", "status_indicator": "invoice-status-chip"}, "dunning": {"icon": "alert-circle", "status_indicator": "dunning-step-chip"}, "payments": {"icon": "credit-card", "status_indicator": "payment-chip"}, "discounts": {"icon": "tag", "status_indicator": "discount-chip"}, "convergent": {"icon": "merge", "status_indicator": "convergent-chip"}, "disputes": {"icon": "message-square", "status_indicator": "dispute-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": BIL_EVENT_STREAM, "key": "tenant_id", "events": ["cdr_mediated", "charge_rated", "invoice_generated", "invoice_approved", "invoice_sent", "payment_received", "dunning_step_triggered", "discount_applied", "write_off_recorded", "bil_agent_registered"], "guardrails": ["bil_batch_requires_bytewax", "privileged_bil_agent_action_requires_human_approval", "bill_suppression_denied", "unapproved_write_off_denied", "cross_tenant_billing_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "bil_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "bil_policy_required", "required_action": "attach_bil_policy"}},
	{"name": "mediation_status_supported", "condition": {"operation": "record_cdr", "mediation_status_supported": False}, "effect": {"decision": "deny", "reason": "mediation_status_not_supported", "required_action": "select_supported_mediation_status"}},
	{"name": "cdr_source_required", "condition": {"operation": "record_cdr", "source_present": False}, "effect": {"decision": "deny", "reason": "cdr_source_required", "required_action": "set_cdr_source"}},
	{"name": "rating_type_supported", "condition": {"operation": "record_charge", "rating_type_supported": False}, "effect": {"decision": "deny", "reason": "rating_type_not_supported", "required_action": "select_supported_rating_type"}},
	{"name": "charge_type_supported", "condition": {"operation": "record_charge", "charge_type_supported": False}, "effect": {"decision": "deny", "reason": "charge_type_not_supported", "required_action": "select_supported_charge_type"}},
	{"name": "charge_amount_positive", "condition": {"operation": "record_charge", "amount_positive": False}, "effect": {"decision": "deny", "reason": "charge_amount_must_be_positive", "required_action": "set_positive_amount"}},
	{"name": "bill_cycle_type_supported", "condition": {"operation": "create_bill_cycle", "cycle_type_supported": False}, "effect": {"decision": "deny", "reason": "bill_cycle_type_not_supported", "required_action": "select_supported_cycle_type"}},
	{"name": "bill_cycle_cutoff_required", "condition": {"operation": "create_bill_cycle", "cutoff_present": False}, "effect": {"decision": "deny", "reason": "bill_cycle_cutoff_required", "required_action": "set_cycle_cutoff"}},
	{"name": "invoice_approval_required", "condition": {"operation": "approve_invoice", "approval_present": False}, "effect": {"decision": "deny", "reason": "invoice_approval_required", "required_action": "attach_invoice_approval"}},
	{"name": "invoice_status_supported", "condition": {"operation": "update_invoice_status", "status_supported": False}, "effect": {"decision": "deny", "reason": "invoice_status_not_supported", "required_action": "select_supported_invoice_status"}},
	{"name": "dunning_step_supported", "condition": {"operation": "trigger_dunning", "dunning_step_supported": False}, "effect": {"decision": "deny", "reason": "dunning_step_not_supported", "required_action": "select_supported_dunning_step"}},
	{"name": "payment_method_supported", "condition": {"operation": "record_payment", "payment_method_supported": False}, "effect": {"decision": "deny", "reason": "payment_method_not_supported", "required_action": "select_supported_payment_method"}},
	{"name": "payment_amount_positive", "condition": {"operation": "record_payment", "amount_positive": False}, "effect": {"decision": "deny", "reason": "payment_amount_must_be_positive", "required_action": "set_positive_amount"}},
	{"name": "discount_type_supported", "condition": {"operation": "apply_discount", "discount_type_supported": False}, "effect": {"decision": "deny", "reason": "discount_type_not_supported", "required_action": "select_supported_discount_type"}},
	{"name": "discount_approval_required", "condition": {"operation": "apply_discount", "approval_present": False}, "effect": {"decision": "deny", "reason": "discount_approval_required", "required_action": "attach_discount_approval"}},
	{"name": "discount_max_exceeded", "condition": {"operation": "apply_discount", "max_discount_exceeded": True}, "effect": {"decision": "deny", "reason": "discount_exceeds_max_allowed", "required_action": "reduce_discount_pct"}},
	{"name": "convergent_mode_supported", "condition": {"operation": "setup_convergent", "convergent_mode_supported": False}, "effect": {"decision": "deny", "reason": "convergent_mode_not_supported", "required_action": "select_supported_convergent_mode"}},
	{"name": "write_off_requires_approval", "condition": {"operation": "write_off_invoice", "approval_present": False}, "effect": {"decision": "deny", "reason": "write_off_approval_required", "required_action": "attach_write_off_approval"}},
	{"name": "bill_suppression_denied", "condition": {"operation": "bil_agent_action", "bill_suppression_scope": True}, "effect": {"decision": "deny", "reason": "bill_suppression_scope_denied", "required_action": "remove_bill_suppression_scope"}},
	{"name": "cross_tenant_billing_denied", "condition": {"operation": "bil_agent_action", "cross_tenant_billing_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_billing_scope_denied", "required_action": "remove_cross_tenant_billing_scope"}},
	{"name": "bil_batch_requires_bytewax", "condition": {"operation": "bil_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_bil_batch_to_bytewax"}},
	{"name": "bil_agent_runtime_supported", "condition": {"operation": "register_bil_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "bil_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "bil_agent_role_supported", "condition": {"operation": "register_bil_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "bil_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "bil_agent_name_required", "condition": {"operation": "register_bil_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "bil_agent_name_required", "required_action": "name_bil_agent"}},
	{"name": "bil_agent_scope_required", "condition": {"operation": "register_bil_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "bil_agent_scope_required", "required_action": "bound_bil_agent_scope"}},
	{"name": "privileged_bil_agent_action_requires_human_approval", "condition": {"operation": "bil_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/telecom-bil/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
