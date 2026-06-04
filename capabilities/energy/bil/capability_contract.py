"""Executable capability contract for APG Energy Billing & Tariffs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "energy_bil"
CAPABILITY_NAME = "Energy Billing & Tariffs"
CAPABILITY_VERSION = "1.0.0"
BIL_EVENT_STREAM = "apg.energy.bil.lifecycle"

SUPPORTED_TARIFF_TYPES = ["flat_rate", "tiered_block", "time_of_use", "real_time_pricing", "demand_charge", "demand_plus_energy", "net_metering", "feed_in_tariff", "critical_peak_pricing", "inclining_block", "declining_block", "lifeline", "prepayment"]
SUPPORTED_CUSTOMER_CLASSES = ["residential", "small_commercial", "medium_commercial", "large_commercial", "industrial", "agricultural", "government", "streetlighting", "temporary"]
SUPPORTED_BILLING_CYCLES = ["monthly", "bi_monthly", "quarterly", "annual", "weekly", "on_demand"]
SUPPORTED_PAYMENT_METHODS = ["direct_debit", "credit_card", "mobile_money", "bank_transfer", "cash", "prepayment_token", "standing_order", "third_party"]
SUPPORTED_BILL_STATUSES = ["draft", "issued", "sent", "partially_paid", "paid", "overdue", "disputed", "written_off", "reversed"]
SUPPORTED_CREDIT_TYPES = ["renewable_energy_credit", "low_income_assistance", "green_tariff_credit", "demand_response_incentive", "early_payment_discount", "loyalty_discount", "carbon_offset_credit"]
SUPPORTED_CHARGE_TYPES = ["energy_charge", "demand_charge", "fixed_charge", "capacity_charge", "distribution_charge", "transmission_charge", "tax", "levy", "surcharge", "credit"]
SUPPORTED_REVENUE_ASSURANCE_TYPES = ["unbilled_energy", "billing_exception", "tariff_error", "payment_reconciliation", "meter_read_gap", "estimation_variance"]
SUPPORTED_DISPUTE_STATUSES = ["open", "under_review", "resolved_accepted", "resolved_rejected", "escalated", "closed"]
SUPPORTED_APPROVAL_STATUSES = ["pending", "approved", "rejected", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["tariff_calculator", "bill_validator", "revenue_analyst", "dispute_resolver", "credit_manager"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ui": {"enable_dashboard": True, "enable_tariffs": True, "enable_billing": True, "enable_payments": True, "enable_credits": True, "enable_disputes": True, "enable_revenue_assurance": True},
	"theme": {"default_theme": "energy_bil_ops", "allow_tenant_overrides": True},
	"tariffs": {"supported_types": SUPPORTED_TARIFF_TYPES, "supported_customer_classes": SUPPORTED_CUSTOMER_CLASSES, "approval_required": True, "effective_date_required": True},
	"billing": {"supported_cycles": SUPPORTED_BILLING_CYCLES, "supported_charge_types": SUPPORTED_CHARGE_TYPES, "supported_statuses": SUPPORTED_BILL_STATUSES, "auto_generate": True},
	"payments": {"supported_methods": SUPPORTED_PAYMENT_METHODS, "reconciliation_required": True, "receipt_generation": True},
	"credits": {"supported_types": SUPPORTED_CREDIT_TYPES, "approval_required": True, "expiry_required": True},
	"disputes": {"supported_statuses": SUPPORTED_DISPUTE_STATUSES, "resolution_deadline_days": 30, "evidence_required": True},
	"revenue_assurance": {"supported_types": SUPPORTED_REVENUE_ASSURANCE_TYPES, "auto_flag": True, "investigation_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_write_off": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_denied": True, "unapproved_tariff_change_denied": True},
	"observability": {"event_stream": BIL_EVENT_STREAM, "stream_processor": "bytewax"},
}

PROVIDES = [
	"tariff_management",
	"consumption_billing",
	"demand_charge_calculation",
	"renewable_credits_management",
	"revenue_assurance",
	"payment_processing",
	"dispute_management",
	"billing_analytics",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/energy-bil/dashboard", "component": "BilDashboard", "permission": "energy_bil:view", "nav_group": "Overview"},
	{"name": "tariffs", "path": "/energy-bil/tariffs", "component": "TariffManager", "permission": "energy_bil:tariffs", "nav_group": "Configuration"},
	{"name": "tariff_detail", "path": "/energy-bil/tariffs/<id>", "component": "TariffDetail", "permission": "energy_bil:tariffs", "nav_group": "Configuration"},
	{"name": "bills", "path": "/energy-bil/bills", "component": "BillingConsole", "permission": "energy_bil:billing", "nav_group": "Billing"},
	{"name": "bill_detail", "path": "/energy-bil/bills/<id>", "component": "BillDetail", "permission": "energy_bil:billing", "nav_group": "Billing"},
	{"name": "payments", "path": "/energy-bil/payments", "component": "PaymentConsole", "permission": "energy_bil:payments", "nav_group": "Payments"},
	{"name": "credits", "path": "/energy-bil/credits", "component": "CreditManager", "permission": "energy_bil:credits", "nav_group": "Credits"},
	{"name": "disputes", "path": "/energy-bil/disputes", "component": "DisputeConsole", "permission": "energy_bil:disputes", "nav_group": "Customer Service"},
	{"name": "revenue_assurance", "path": "/energy-bil/revenue-assurance", "component": "RevenueAssuranceConsole", "permission": "energy_bil:revenue_assurance", "nav_group": "Assurance"},
	{"name": "reports", "path": "/energy-bil/reports", "component": "BillingReports", "permission": "energy_bil:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/energy-bil/agents", "component": "BilAgentWorkbench", "permission": "energy_bil:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/energy-bil/settings", "component": "BilSettings", "permission": "energy_bil:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "energy_bil_ops",
	"tokens": {
		"color.primary": "#065F46",
		"color.accent": "#0369A1",
		"color.success": "#166534",
		"color.warning": "#92400E",
		"color.danger": "#991B1B",
		"surface.canvas": "#ECFDF5",
		"surface.panel": "#FFFFFF",
		"text.primary": "#064E3B",
		"text.secondary": "#065F46",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"tariffs": {"icon": "tag", "status_indicator": "tariff-type-chip"},
		"bills": {"icon": "file-text", "status_indicator": "bill-status-chip"},
		"payments": {"icon": "credit-card", "status_indicator": "payment-method-chip"},
		"credits": {"icon": "gift", "status_indicator": "credit-type-chip"},
		"disputes": {"icon": "message-square", "status_indicator": "dispute-status-chip"},
		"revenue_assurance": {"icon": "trending-up", "status_indicator": "ra-type-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": BIL_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"tariff_created", "tariff_approved", "tariff_activated",
		"bill_generated", "bill_issued", "payment_received",
		"payment_reconciled", "credit_applied", "dispute_opened",
		"dispute_resolved", "revenue_assurance_flag_raised",
	],
	"guardrails": [
		"unapproved_tariff_change_denied",
		"cross_tenant_billing_data_denied",
		"privileged_bil_agent_requires_human_approval",
		"bill_write_off_requires_approval",
		"credit_issuance_requires_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "tariff_type_supported", "condition": {"operation": "create_tariff", "tariff_type_supported": False}, "effect": {"decision": "deny", "reason": "tariff_type_not_supported", "required_action": "select_supported_tariff_type"}},
	{"name": "tariff_customer_class_supported", "condition": {"operation": "create_tariff", "customer_class_supported": False}, "effect": {"decision": "deny", "reason": "customer_class_not_supported", "required_action": "select_supported_customer_class"}},
	{"name": "tariff_effective_date_required", "condition": {"operation": "create_tariff", "effective_date_present": False}, "effect": {"decision": "deny", "reason": "tariff_effective_date_required", "required_action": "set_effective_date"}},
	{"name": "tariff_approval_required", "condition": {"operation": "activate_tariff", "approval_present": False}, "effect": {"decision": "deny", "reason": "tariff_activation_approval_required", "required_action": "obtain_tariff_approval"}},
	{"name": "tariff_rate_positive", "condition": {"operation": "create_tariff", "rate_positive": False}, "effect": {"decision": "deny", "reason": "tariff_rate_must_be_positive", "required_action": "set_positive_rate"}},
	{"name": "bill_cycle_supported", "condition": {"operation": "generate_bill", "billing_cycle_supported": False}, "effect": {"decision": "deny", "reason": "billing_cycle_not_supported", "required_action": "select_supported_billing_cycle"}},
	{"name": "bill_tariff_exists", "condition": {"operation": "generate_bill", "tariff_exists": False}, "effect": {"decision": "deny", "reason": "active_tariff_not_found", "required_action": "create_tariff_first"}},
	{"name": "bill_meter_reading_required", "condition": {"operation": "generate_bill", "meter_reading_present": False}, "effect": {"decision": "deny", "reason": "meter_reading_required_for_billing", "required_action": "provide_meter_reading"}},
	{"name": "payment_method_supported", "condition": {"operation": "record_payment", "payment_method_supported": False}, "effect": {"decision": "deny", "reason": "payment_method_not_supported", "required_action": "select_supported_payment_method"}},
	{"name": "payment_amount_positive", "condition": {"operation": "record_payment", "amount_positive": False}, "effect": {"decision": "deny", "reason": "payment_amount_must_be_positive", "required_action": "set_positive_amount"}},
	{"name": "payment_bill_exists", "condition": {"operation": "record_payment", "bill_exists": False}, "effect": {"decision": "deny", "reason": "bill_not_found_for_payment", "required_action": "reference_valid_bill"}},
	{"name": "credit_type_supported", "condition": {"operation": "issue_credit", "credit_type_supported": False}, "effect": {"decision": "deny", "reason": "credit_type_not_supported", "required_action": "select_supported_credit_type"}},
	{"name": "credit_approval_required", "condition": {"operation": "issue_credit", "approval_present": False}, "effect": {"decision": "deny", "reason": "credit_issuance_requires_approval", "required_action": "obtain_credit_approval"}},
	{"name": "credit_expiry_required", "condition": {"operation": "issue_credit", "expiry_present": False}, "effect": {"decision": "deny", "reason": "credit_expiry_date_required", "required_action": "set_credit_expiry"}},
	{"name": "dispute_evidence_required", "condition": {"operation": "open_dispute", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dispute_evidence_required", "required_action": "attach_dispute_evidence"}},
	{"name": "dispute_bill_exists", "condition": {"operation": "open_dispute", "bill_exists": False}, "effect": {"decision": "deny", "reason": "bill_not_found_for_dispute", "required_action": "reference_valid_bill"}},
	{"name": "write_off_approval_required", "condition": {"operation": "write_off_bill", "approval_present": False}, "effect": {"decision": "deny", "reason": "bill_write_off_requires_approval", "required_action": "obtain_write_off_approval"}},
	{"name": "revenue_assurance_type_supported", "condition": {"operation": "flag_revenue_issue", "ra_type_supported": False}, "effect": {"decision": "deny", "reason": "revenue_assurance_type_not_supported", "required_action": "select_supported_ra_type"}},
	{"name": "cross_tenant_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "bil_agent_runtime_supported", "condition": {"operation": "register_bil_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "bil_agent_role_supported", "condition": {"operation": "register_bil_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_bil_agent_requires_human_approval", "condition": {"operation": "bil_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required_for_write_off_or_credit", "required_action": "record_human_approval"}},
	{"name": "charge_type_supported", "condition": {"operation": "add_bill_charge", "charge_type_supported": False}, "effect": {"decision": "deny", "reason": "charge_type_not_supported", "required_action": "select_supported_charge_type"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
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
			"api_prefix": "/energy-bil/api/v1",
			"requires_theme": True,
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
