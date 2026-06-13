"""Executable capability contract for APG SaaS Billing Engine."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "common_sbl"
CAPABILITY_NAME = "SaaS Billing Engine"
CAPABILITY_VERSION = "1.0.0"
BILLING_EVENT_STREAM = "apg.common.sbl.lifecycle"

SUPPORTED_PLAN_TIERS = ["free", "starter", "professional", "enterprise"]
SUPPORTED_BILLING_CYCLES = ["monthly", "annual"]
SUPPORTED_TENANT_STATUSES = ["active", "suspended", "cancelled", "trial", "past_due"]
SUPPORTED_SUBSCRIPTION_STATUSES = ["active", "cancelled", "past_due", "trialing", "paused"]
SUPPORTED_INVOICE_STATUSES = ["draft", "open", "paid", "void", "uncollectible"]
SUPPORTED_PAYMENT_METHOD_TYPES = ["card", "bank_transfer", "mpesa", "paypal", "stripe_token"]
SUPPORTED_USAGE_METRICS = ["api_calls", "storage_gb", "users", "transactions", "exports", "webhooks", "seats"]
SUPPORTED_CREDIT_NOTE_REASONS = ["duplicate", "fraudulent", "customer_request", "adjustment", "subscription_upgrade"]
SUPPORTED_PRORATION_MODES = ["immediate", "next_cycle", "prorated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["billing_analyst", "usage_auditor", "invoice_reviewer", "churn_predictor", "dunning_agent"]

# Plan definitions: metric limits and monthly price (USD cents)
PLAN_DEFINITIONS: dict[str, dict[str, Any]] = {
	"free": {
		"display_name": "Free",
		"price_monthly_cents": 0,
		"price_annual_cents": 0,
		"limits": {
			"api_calls": 100,
			"users": 5,
			"storage_gb": 1,
			"transactions": 50,
			"exports": 10,
			"webhooks": 0,
			"seats": 5,
		},
		"features": ["basic_api", "community_support"],
		"overage_allowed": False,
	},
	"starter": {
		"display_name": "Starter",
		"price_monthly_cents": 4900,
		"price_annual_cents": 47040,  # 20% annual discount
		"limits": {
			"api_calls": 10000,
			"users": 25,
			"storage_gb": 10,
			"transactions": 5000,
			"exports": 500,
			"webhooks": 10,
			"seats": 25,
		},
		"features": ["basic_api", "email_support", "webhooks", "exports"],
		"overage_allowed": True,
		"overage_rates": {
			"api_calls": 0.001,          # $0.001 per extra call
			"users": 200,                 # $2 per extra user
			"storage_gb": 50,             # $0.50 per extra GB
			"transactions": 0.005,        # $0.005 per extra tx
		},
	},
	"professional": {
		"display_name": "Professional",
		"price_monthly_cents": 19900,
		"price_annual_cents": 191040,  # 20% annual discount
		"limits": {
			"api_calls": 100000,
			"users": 100,
			"storage_gb": 100,
			"transactions": 50000,
			"exports": 5000,
			"webhooks": 100,
			"seats": 100,
		},
		"features": ["full_api", "priority_support", "webhooks", "exports", "audit_log", "sso", "analytics"],
		"overage_allowed": True,
		"overage_rates": {
			"api_calls": 0.0008,
			"users": 150,
			"storage_gb": 40,
			"transactions": 0.003,
		},
	},
	"enterprise": {
		"display_name": "Enterprise",
		"price_monthly_cents": 0,       # custom pricing
		"price_annual_cents": 0,
		"limits": {
			"api_calls": -1,             # -1 = unlimited
			"users": -1,
			"storage_gb": -1,
			"transactions": -1,
			"exports": -1,
			"webhooks": -1,
			"seats": -1,
		},
		"features": ["full_api", "dedicated_support", "webhooks", "exports", "audit_log", "sso", "analytics", "custom_integrations", "sla", "white_label"],
		"overage_allowed": False,
	},
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"plans": {
		"supported_tiers": SUPPORTED_PLAN_TIERS,
		"supported_billing_cycles": SUPPORTED_BILLING_CYCLES,
		"definitions": PLAN_DEFINITIONS,
		"default_trial_days": 14,
		"currency": "USD",
	},
	"subscriptions": {
		"supported_statuses": SUPPORTED_SUBSCRIPTION_STATUSES,
		"proration_modes": SUPPORTED_PRORATION_MODES,
		"grace_period_days": 7,
		"renewal_reminder_days": [7, 3, 1],
	},
	"usage": {
		"supported_metrics": SUPPORTED_USAGE_METRICS,
		"aggregation_interval_minutes": 60,
		"retention_days": 90,
		"limit_enforcement": "hard",   # hard | soft | warn
	},
	"invoicing": {
		"supported_statuses": SUPPORTED_INVOICE_STATUSES,
		"auto_generate": True,
		"net_days": 30,
		"dunning_schedule_days": [1, 7, 14, 21],
	},
	"payments": {
		"supported_methods": SUPPORTED_PAYMENT_METHOD_TYPES,
		"tokenization_required": True,
		"raw_card_storage_denied": True,
	},
	"credit_notes": {
		"supported_reasons": SUPPORTED_CREDIT_NOTE_REASONS,
		"approval_required": True,
	},
	"provisioning": {
		"self_service_allowed": True,
		"kyc_required_for_paid": False,
		"instant_activation": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"cross_tenant_billing_denied": True,
		"raw_card_storage_denied": True,
		"invoice_tampering_denied": True,
		"usage_fabrication_denied": True,
		"downgrade_during_active_period_denied": False,
	},
	"observability": {
		"event_stream": BILLING_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"gl": "fin_gl",
		"payments": "fintech_payments",
	},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"name_required": True,
		"scope_required": True,
		"human_approval_required_for_refunds": True,
	},
	"ui": {
		"enable_dashboard": True,
		"enable_plans": True,
		"enable_tenants": True,
		"enable_subscriptions": True,
		"enable_usage": True,
		"enable_invoices": True,
		"enable_payments": True,
		"enable_credit_notes": True,
		"enable_analytics": True,
		"enable_agents": True,
	},
	"theme": {
		"default_theme": "sbl_billing_console",
		"allow_tenant_overrides": True,
	},
}

PROVIDES = [
	"subscription_management",
	"usage_metering",
	"invoice_generation",
	"tenant_provisioning",
	"billing_analytics",
]
REQUIRES = ["auth", "audl", "ntfy", "fin_gl", "fintech_payments"]

UI_ROUTES = [
	{"name": "dashboard",      "path": "/sbl/dashboard",      "component": "BillingDashboard",      "permission": "sbl:view",          "nav_group": "Overview"},
	{"name": "tenants",        "path": "/sbl/tenants",        "component": "TenantConsole",          "permission": "sbl:tenants",       "nav_group": "Tenants"},
	{"name": "plans",          "path": "/sbl/plans",          "component": "PlanCatalog",            "permission": "sbl:plans",         "nav_group": "Configuration"},
	{"name": "subscriptions",  "path": "/sbl/subscriptions",  "component": "SubscriptionLedger",     "permission": "sbl:subscriptions", "nav_group": "Billing"},
	{"name": "usage",          "path": "/sbl/usage",          "component": "UsageMeter",             "permission": "sbl:usage",         "nav_group": "Metering"},
	{"name": "invoices",       "path": "/sbl/invoices",       "component": "InvoiceQueue",           "permission": "sbl:invoices",      "nav_group": "Billing"},
	{"name": "payments",       "path": "/sbl/payments",       "component": "PaymentConsole",         "permission": "sbl:payments",      "nav_group": "Billing"},
	{"name": "credit_notes",   "path": "/sbl/credit-notes",   "component": "CreditNoteConsole",      "permission": "sbl:credit_notes",  "nav_group": "Adjustments"},
	{"name": "analytics",      "path": "/sbl/analytics",      "component": "BillingAnalytics",       "permission": "sbl:analytics",     "nav_group": "Analytics"},
	{"name": "agents",         "path": "/sbl/agents",         "component": "BillingAgentWorkbench",  "permission": "sbl:admin",         "nav_group": "Automation"},
	{"name": "settings",       "path": "/sbl/settings",       "component": "BillingSettings",        "permission": "sbl:admin",         "nav_group": "Administration"},
]

THEME = {
	"name": "sbl_billing_console",
	"tokens": {
		"color.primary":   "#1D4ED8",
		"color.accent":    "#0891B2",
		"color.success":   "#15803D",
		"color.warning":   "#B45309",
		"color.danger":    "#B91C1C",
		"surface.canvas":  "#F8FAFC",
		"surface.panel":   "#FFFFFF",
		"text.primary":    "#0F172A",
		"text.secondary":  "#475569",
		"border.radius":   "8px",
		"density":         "comfortable",
	},
	"components": {
		"plans":         {"icon": "layers",          "status_indicator": "plan-tier-chip"},
		"tenants":       {"icon": "building-2",      "status_indicator": "tenant-status-chip"},
		"subscriptions": {"icon": "repeat",          "status_indicator": "subscription-status-chip"},
		"usage":         {"icon": "bar-chart-2",     "status_indicator": "metric-chip"},
		"invoices":      {"icon": "file-text",       "status_indicator": "invoice-status-chip"},
		"payments":      {"icon": "credit-card",     "status_indicator": "payment-chip"},
		"credit_notes":  {"icon": "receipt",         "status_indicator": "credit-note-chip"},
		"analytics":     {"icon": "trending-up",     "status_indicator": "analytics-chip"},
		"agents":        {"icon": "bot",             "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": BILLING_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"tenant_created",
		"subscription_created",
		"subscription_upgraded",
		"subscription_cancelled",
		"usage_recorded",
		"invoice_generated",
		"invoice_paid",
		"payment_method_attached",
		"credit_note_issued",
		"usage_limit_approaching",
		"usage_limit_exceeded",
	],
	"guardrails": [
		"raw_card_storage_denied",
		"cross_tenant_billing_denied",
		"invoice_tampering_denied",
		"usage_fabrication_denied",
		"unauthorized_refund_denied",
	],
}

RULES: list[dict[str, Any]] = [
	# Governance basics
	{"name": "tenant_context_required",          "condition": {"tenant_context_present": False},                                                          "effect": {"decision": "deny", "reason": "tenant_context_required",            "required_action": "attach_tenant_context"}},
	{"name": "billing_write_requires_policy",    "condition": {"operation_type": "write", "policy_attached": False},                                      "effect": {"decision": "deny", "reason": "billing_policy_required",            "required_action": "attach_billing_policy"}},
	{"name": "cross_tenant_billing_denied",      "condition": {"cross_tenant_operation": True},                                                           "effect": {"decision": "deny", "reason": "cross_tenant_billing_denied",        "required_action": "use_tenant_scoped_operation"}},
	# Raw card storage
	{"name": "raw_card_storage_denied",          "condition": {"operation": "store_payment_method", "raw_card_number_present": True},                     "effect": {"decision": "deny", "reason": "raw_card_storage_denied",           "required_action": "tokenize_card_before_storage"}},
	# Plan operations
	{"name": "plan_tier_supported",              "condition": {"operation": "create_plan", "plan_tier_supported": False},                                 "effect": {"decision": "deny", "reason": "plan_tier_not_supported",           "required_action": "select_supported_plan_tier"}},
	{"name": "plan_price_required",              "condition": {"operation": "create_plan", "plan_price_present": False},                                  "effect": {"decision": "deny", "reason": "plan_price_required",               "required_action": "set_plan_price"}},
	# Tenant operations
	{"name": "tenant_email_required",            "condition": {"operation": "create_tenant", "tenant_email_present": False},                              "effect": {"decision": "deny", "reason": "tenant_email_required",             "required_action": "provide_tenant_email"}},
	{"name": "tenant_plan_required",             "condition": {"operation": "create_tenant", "tenant_plan_present": False},                               "effect": {"decision": "deny", "reason": "tenant_plan_required",              "required_action": "select_plan"}},
	# Subscription operations
	{"name": "subscription_tenant_required",     "condition": {"operation": "create_subscription", "subscription_tenant_present": False},                 "effect": {"decision": "deny", "reason": "subscription_tenant_required",      "required_action": "provide_tenant_id"}},
	{"name": "subscription_plan_required",       "condition": {"operation": "create_subscription", "subscription_plan_present": False},                   "effect": {"decision": "deny", "reason": "subscription_plan_required",        "required_action": "select_plan"}},
	{"name": "subscription_cycle_supported",     "condition": {"operation": "create_subscription", "billing_cycle_supported": False},                     "effect": {"decision": "deny", "reason": "billing_cycle_not_supported",       "required_action": "select_supported_billing_cycle"}},
	# Usage recording
	{"name": "usage_metric_supported",           "condition": {"operation": "record_usage", "usage_metric_supported": False},                             "effect": {"decision": "deny", "reason": "usage_metric_not_supported",        "required_action": "select_supported_metric"}},
	{"name": "usage_quantity_positive",          "condition": {"operation": "record_usage", "usage_quantity_positive": False},                            "effect": {"decision": "deny", "reason": "usage_quantity_must_be_positive",   "required_action": "provide_positive_quantity"}},
	{"name": "usage_fabrication_denied",         "condition": {"operation": "record_usage", "backdated_beyond_window": True},                             "effect": {"decision": "deny", "reason": "usage_fabrication_denied",          "required_action": "use_current_period_timestamp"}},
	# Invoice operations
	{"name": "invoice_period_required",          "condition": {"operation": "generate_invoice", "invoice_period_present": False},                         "effect": {"decision": "deny", "reason": "invoice_period_required",           "required_action": "provide_invoice_period"}},
	{"name": "invoice_tampering_denied",         "condition": {"operation": "modify_invoice", "invoice_status": "paid"},                                  "effect": {"decision": "deny", "reason": "invoice_tampering_denied",          "required_action": "issue_credit_note_instead"}},
	# Credit notes
	{"name": "credit_note_reason_required",      "condition": {"operation": "issue_credit_note", "credit_note_reason_present": False},                    "effect": {"decision": "deny", "reason": "credit_note_reason_required",       "required_action": "provide_credit_note_reason"}},
	{"name": "credit_note_approval_required",    "condition": {"operation": "issue_credit_note", "approval_present": False},                              "effect": {"decision": "deny", "reason": "credit_note_approval_required",     "required_action": "obtain_approval"}},
	{"name": "credit_note_reason_supported",     "condition": {"operation": "issue_credit_note", "credit_note_reason_supported": False},                  "effect": {"decision": "deny", "reason": "credit_note_reason_not_supported",  "required_action": "select_supported_reason"}},
	# Payment methods
	{"name": "payment_method_type_supported",    "condition": {"operation": "attach_payment_method", "payment_method_type_supported": False},             "effect": {"decision": "deny", "reason": "payment_method_type_not_supported", "required_action": "select_supported_payment_method_type"}},
	{"name": "payment_method_token_required",    "condition": {"operation": "attach_payment_method", "token_present": False},                             "effect": {"decision": "deny", "reason": "payment_token_required",            "required_action": "provide_payment_token"}},
	# Upgrade/downgrade
	{"name": "upgrade_plan_required",            "condition": {"operation": "upgrade_plan", "new_plan_present": False},                                   "effect": {"decision": "deny", "reason": "new_plan_required",                  "required_action": "specify_new_plan"}},
	{"name": "upgrade_active_subscription_required", "condition": {"operation": "upgrade_plan", "active_subscription_present": False},                    "effect": {"decision": "deny", "reason": "active_subscription_required",      "required_action": "ensure_active_subscription"}},
	# Agent guardrails
	{"name": "refund_agent_action_requires_human_approval", "condition": {"operation": "agent_action", "action_type": "refund", "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required_for_refund", "required_action": "record_human_approval"}},
	{"name": "unauthorized_refund_denied",       "condition": {"operation": "agent_action", "unauthorized_refund": True},                                 "effect": {"decision": "deny", "reason": "unauthorized_refund_denied",        "required_action": "obtain_authorization"}},
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
		"rule_engine": {
			"type": "deterministic",
			"default_decision": "allow",
			"rules": deepcopy(RULES),
		},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/sbl/api/v1",
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
