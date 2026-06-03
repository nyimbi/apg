"""Executable capability contract for APG Digital Payments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_payments"
CAPABILITY_NAME = "Digital Payments"
CAPABILITY_VERSION = "1.1.0"
PAYMENTS_EVENT_STREAM = "apg.fintech.payments.lifecycle"

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_INSTRUMENT_TYPES = ["card", "bank_account", "mobile_money", "wallet", "qr", "voucher"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "blocked"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = [
	"payment_ops_reviewer",
	"risk_reviewer",
	"settlement_reviewer",
	"dispute_reviewer",
	"provider_reconciliation_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"accounts": {
		"owner_required": True,
		"currency_supported_required": True,
		"default_status": "active",
	},
	"instruments": {
		"account_required": True,
		"token_reference_required": True,
		"supported_types": SUPPORTED_INSTRUMENT_TYPES,
		"vault_reference_required": True,
	},
	"orders": {
		"amount_positive_required": True,
		"currency_supported_required": True,
		"account_required": True,
		"instrument_required": True,
		"risk_screening_required": True,
		"high_value_threshold": 100000,
	},
	"risk": {
		"supported_levels": SUPPORTED_RISK_LEVELS,
		"review_required_for_high_risk": True,
		"blocked_risk_denies_authorization": True,
	},
	"money_movement": {
		"authorization_provider_required": True,
		"capture_requires_authorization": True,
		"refund_requires_capture": True,
		"payout_destination_required": True,
		"settlement_variance_review_required": True,
	},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"human_approval_required_for_privileged_actions": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_state_changes": True,
		"segregation_of_duties": True,
	},
	"observability": {
		"event_stream": PAYMENTS_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_payment_events": True,
		"emit_risk_events": True,
		"emit_settlement_events": True,
		"emit_dispute_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"keys": "keym",
		"encryption": "encr",
		"gateway": "fintech_gateway",
		"cash_management": "fin_cbm",
		"accounts_receivable": "fin_arc",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_accounts": True,
		"enable_instruments": True,
		"enable_orders": True,
		"enable_risk": True,
		"enable_settlement": True,
		"enable_disputes": True,
		"enable_agents": True,
	},
	"theme": {
		"default_theme": "fintech_payments_control",
		"allow_tenant_overrides": True,
	},
}

PROVIDES = [
	"payment_account_lifecycle",
	"payment_instrument_vault",
	"payment_order_lifecycle",
	"risk_screening_workflow",
	"authorization_capture_refund_workflow",
	"payout_workflow",
	"settlement_reconciliation_workflow",
	"payment_dispute_workflow",
	"payment_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"keym",
	"encr",
	"fintech_gateway",
	"cbm_cash_management",
	"arc_accounts_receivable",
]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-payments/dashboard", "component": "PaymentsDashboard", "permission": "fintech_payments:view", "nav_group": "Overview"},
	{"name": "accounts", "path": "/fintech-payments/accounts", "component": "PaymentAccountWorkbench", "permission": "fintech_payments:manage_accounts", "nav_group": "Accounts"},
	{"name": "instruments", "path": "/fintech-payments/instruments", "component": "PaymentInstrumentVault", "permission": "fintech_payments:manage_instruments", "nav_group": "Payments"},
	{"name": "orders", "path": "/fintech-payments/orders", "component": "PaymentOrderConsole", "permission": "fintech_payments:operate", "nav_group": "Payments"},
	{"name": "risk", "path": "/fintech-payments/risk", "component": "PaymentRiskQueue", "permission": "fintech_payments:risk", "nav_group": "Risk"},
	{"name": "settlement", "path": "/fintech-payments/settlement", "component": "PaymentSettlementConsole", "permission": "fintech_payments:settle", "nav_group": "Finance"},
	{"name": "disputes", "path": "/fintech-payments/disputes", "component": "PaymentDisputeWorkbench", "permission": "fintech_payments:disputes", "nav_group": "Risk"},
	{"name": "agents", "path": "/fintech-payments/agents", "component": "PaymentAgentWorkbench", "permission": "fintech_payments:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-payments/settings", "component": "PaymentSettings", "permission": "fintech_payments:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_payments_control",
	"tokens": {
		"color.primary": "#14532D",
		"color.accent": "#0F766E",
		"color.success": "#15803D",
		"color.warning": "#A16207",
		"color.danger": "#B42318",
		"surface.canvas": "#F7FAF8",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"accounts": {"icon": "landmark", "status_indicator": "account-pill"},
		"instruments": {"icon": "credit-card", "status_indicator": "instrument-chip"},
		"orders": {"visual": "payment-timeline", "status_style": "order-chip"},
		"risk": {"visual": "risk-queue", "status_style": "risk-band"},
		"settlement": {"visual": "settlement-grid", "status_style": "variance-chip"},
		"disputes": {"visual": "case-board", "status_style": "dispute-chip"},
		"agents": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": PAYMENTS_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"payment_account_opened",
		"payment_instrument_registered",
		"payment_order_created",
		"payment_risk_screened",
		"payment_authorized",
		"payment_captured",
		"payment_refunded",
		"payout_scheduled",
		"settlement_recorded",
		"payment_dispute_opened",
		"payment_agent_registered",
	],
	"guardrails": [
		"payment_batch_requires_bytewax",
		"payment_event_requires_bytewax",
		"privileged_payment_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Digital payment operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "payment_write_requires_policy", "description": "Payment writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "payment_policy_required", "required_action": "attach_payment_policy"}},
	{"name": "account_owner_required", "description": "Payment accounts require an owner reference.", "condition": {"operation": "open_payment_account", "owner_present": False}, "effect": {"decision": "deny", "reason": "account_owner_required", "required_action": "attach_owner_reference"}},
	{"name": "account_currency_supported", "description": "Payment accounts require a supported currency.", "condition": {"operation": "open_payment_account", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "instrument_account_required", "description": "Payment instruments require an account.", "condition": {"operation": "register_instrument", "account_present": False}, "effect": {"decision": "deny", "reason": "payment_account_required", "required_action": "select_payment_account"}},
	{"name": "instrument_type_supported", "description": "Payment instrument type must be supported.", "condition": {"operation": "register_instrument", "instrument_type_supported": False}, "effect": {"decision": "deny", "reason": "instrument_type_not_supported", "required_action": "select_supported_instrument_type"}},
	{"name": "instrument_token_required", "description": "Payment instruments require a vault token reference.", "condition": {"operation": "register_instrument", "token_reference_present": False}, "effect": {"decision": "deny", "reason": "instrument_token_required", "required_action": "attach_token_reference"}},
	{"name": "payment_amount_positive", "description": "Payment orders require a positive amount.", "condition": {"operation": "create_payment_order", "amount_lte": 0}, "effect": {"decision": "deny", "reason": "payment_amount_positive_required", "required_action": "set_positive_amount"}},
	{"name": "payment_currency_supported", "description": "Payment order currency must be supported.", "condition": {"operation": "create_payment_order", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "payment_account_required", "description": "Payment orders require an account.", "condition": {"operation": "create_payment_order", "account_present": False}, "effect": {"decision": "deny", "reason": "payment_account_required", "required_action": "select_payment_account"}},
	{"name": "payment_instrument_required", "description": "Payment orders require an instrument.", "condition": {"operation": "create_payment_order", "instrument_present": False}, "effect": {"decision": "deny", "reason": "payment_instrument_required", "required_action": "select_payment_instrument"}},
	{"name": "high_risk_payment_requires_review", "description": "High-risk payments require recorded review.", "condition": {"operation": "screen_payment_risk", "risk_level": "high", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "payment_risk_review_required", "required_action": "record_payment_risk_review"}},
	{"name": "blocked_risk_denies_authorization", "description": "Blocked risk decisions deny authorization.", "condition": {"operation": "authorize_payment", "risk_level": "blocked"}, "effect": {"decision": "deny", "reason": "payment_risk_blocked", "required_action": "resolve_payment_risk"}},
	{"name": "authorization_provider_required", "description": "Payment authorization requires a provider reference.", "condition": {"operation": "authorize_payment", "provider_present": False}, "effect": {"decision": "deny", "reason": "authorization_provider_required", "required_action": "select_payment_provider"}},
	{"name": "high_value_authorization_requires_approval", "description": "High-value authorization requires approval evidence.", "condition": {"operation": "authorize_payment", "high_value": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "high_value_payment_approval_required", "required_action": "record_payment_approval"}},
	{"name": "capture_requires_authorization", "description": "Captures require an authorized payment.", "condition": {"operation": "capture_payment", "authorized_payment_present": False}, "effect": {"decision": "deny", "reason": "authorized_payment_required", "required_action": "authorize_payment"}},
	{"name": "capture_blocks_overcapture", "description": "Capture cannot exceed authorized amount.", "condition": {"operation": "capture_payment", "overcapture": True}, "effect": {"decision": "deny", "reason": "overcapture_blocked", "required_action": "reduce_capture_amount"}},
	{"name": "refund_requires_capture", "description": "Refunds require captured funds.", "condition": {"operation": "refund_payment", "captured_payment_present": False}, "effect": {"decision": "deny", "reason": "captured_payment_required", "required_action": "capture_payment"}},
	{"name": "refund_blocks_overrefund", "description": "Refund cannot exceed captured balance.", "condition": {"operation": "refund_payment", "overrefund": True}, "effect": {"decision": "deny", "reason": "overrefund_blocked", "required_action": "reduce_refund_amount"}},
	{"name": "payout_destination_required", "description": "Payouts require a destination reference.", "condition": {"operation": "schedule_payout", "destination_present": False}, "effect": {"decision": "deny", "reason": "payout_destination_required", "required_action": "attach_payout_destination"}},
	{"name": "settlement_variance_requires_review", "description": "Settlement variance requires review.", "condition": {"operation": "record_settlement", "variance_detected": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "settlement_variance_review_required", "required_action": "record_settlement_review"}},
	{"name": "dispute_owner_required", "description": "Payment disputes require an owner.", "condition": {"operation": "open_dispute", "owner_present": False}, "effect": {"decision": "deny", "reason": "dispute_owner_required", "required_action": "assign_dispute_owner"}},
	{"name": "payment_batch_requires_bytewax", "description": "Payment lifecycle batches require Bytewax.", "condition": {"operation": "payment_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_payment_batch_to_bytewax"}},
	{"name": "payment_event_requires_bytewax", "description": "Payment lifecycle events require Bytewax.", "condition": {"operation": "payment_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_payment_event_to_bytewax"}},
	{"name": "payment_agent_runtime_supported", "description": "Payment agents must use a supported runtime.", "condition": {"operation": "register_payment_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "payment_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "payment_agent_role_supported", "description": "Payment agents must use a supported role.", "condition": {"operation": "register_payment_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "payment_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_payment_agent_action_requires_human_approval", "description": "Privileged payment-agent actions require human approval.", "condition": {"operation": "payment_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},

	# Cross-tenant and privilege escalation guards
	{"name": "cross_tenant_payment_access_denied", "description": "Payment resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Payment privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Africa-specific mobile money rules
	{"name": "mpesa_stk_push_shortcode_required", "description": "M-Pesa STK push requires a registered Safaricom shortcode.", "condition": {"operation": "mpesa_stk_push", "mpesa_shortcode_present": False}, "effect": {"decision": "deny", "reason": "mpesa_shortcode_required", "required_action": "register_mpesa_shortcode"}},
	{"name": "mpesa_b2c_approval_required", "description": "M-Pesa B2C disbursements above threshold require human approval.", "condition": {"operation": "mpesa_b2c", "high_value": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "mpesa_b2c_approval_required", "required_action": "record_mpesa_b2c_approval"}},
	{"name": "mpesa_b2b_paybill_required", "description": "M-Pesa B2B transactions require a registered paybill number.", "condition": {"operation": "mpesa_b2b", "mpesa_paybill_present": False}, "effect": {"decision": "deny", "reason": "mpesa_paybill_required", "required_action": "register_mpesa_paybill"}},
	{"name": "mobile_money_kyc_required", "description": "Mobile money payment instruments require linked KYC evidence.", "condition": {"operation": "register_instrument", "instrument_type": "mobile_money", "kyc_present": False}, "effect": {"decision": "deny", "reason": "mobile_money_kyc_required", "required_action": "attach_kyc_profile"}},
	{"name": "mobile_money_phone_number_required", "description": "Mobile money instruments require a verified phone number.", "condition": {"operation": "register_instrument", "instrument_type": "mobile_money", "phone_number_verified": False}, "effect": {"decision": "deny", "reason": "verified_phone_required", "required_action": "verify_phone_number"}},
	{"name": "airtel_money_msisdn_required", "description": "Airtel Money payments require a registered MSISDN.", "condition": {"operation": "airtel_money_payment", "msisdn_registered": False}, "effect": {"decision": "deny", "reason": "airtel_money_msisdn_required", "required_action": "register_airtel_money_msisdn"}},
	{"name": "pesalink_account_required", "description": "PesaLink transfers require a registered bank account.", "condition": {"operation": "pesalink_transfer", "bank_account_registered": False}, "effect": {"decision": "deny", "reason": "pesalink_account_required", "required_action": "register_pesalink_account"}},
	{"name": "pesalink_daily_limit_enforced", "description": "PesaLink single transfer limit is enforced (KES 999,999).", "condition": {"operation": "pesalink_transfer", "exceeds_pesalink_limit": True}, "effect": {"decision": "deny", "reason": "pesalink_limit_exceeded", "required_action": "reduce_transfer_amount"}},
	{"name": "ussd_payment_session_required", "description": "USSD-initiated payments require an active session token.", "condition": {"operation": "ussd_payment", "ussd_session_present": False}, "effect": {"decision": "deny", "reason": "ussd_session_required", "required_action": "initiate_ussd_session"}},
	{"name": "cbk_large_cash_reporting_required", "description": "Cash payments above CBK reporting threshold require declaration.", "condition": {"operation": "create_payment_order", "instrument_type": "cash", "exceeds_cbk_reporting_threshold": True, "declaration_present": False}, "effect": {"decision": "require_review", "reason": "cbk_large_cash_reporting_required", "required_action": "file_cbk_cash_declaration"}},
	{"name": "ke_kes_payment_requires_cbk_compliance", "description": "KES-denominated payments require CBK compliance evidence.", "condition": {"operation": "create_payment_order", "currency": "KES", "cbk_compliant": False}, "effect": {"decision": "deny", "reason": "cbk_compliance_required", "required_action": "attach_cbk_compliance_evidence"}},
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
	"""Return the executable APG capability contract."""
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
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/fintech-payments/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic payment guardrails."""
	contract = get_capability_contract(str(context.get("tenant_id") or "default"))
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
		"actions": [rule["effect"] for rule in matched],
		"effects": [rule["effect"] for rule in matched],
	}
