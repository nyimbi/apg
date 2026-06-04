"""Executable capability contract for APG Cross-Border Remittance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_remittance"
CAPABILITY_NAME = "Cross-Border Remittance"
CAPABILITY_VERSION = "1.1.0"
REMITTANCE_EVENT_STREAM = "apg.fintech.remittance.lifecycle"

SUPPORTED_COUNTRIES = ["KE", "UG", "TZ", "RW", "GH", "NG", "ZA", "GB", "US", "AE", "IN"]
SUPPORTED_CURRENCIES = ["KES", "UGX", "TZS", "RWF", "GHS", "NGN", "ZAR", "GBP", "USD", "AED", "INR", "EUR"]
SUPPORTED_PAYOUT_METHODS = ["bank_account", "mobile_money", "wallet", "cash_pickup", "card_push"]
SUPPORTED_PURPOSE_CODES = ["family_support", "education", "medical", "trade", "salary", "savings", "emergency"]
SUPPORTED_FRAUD_DECISIONS = ["clear", "review", "hold", "block"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["remittance_ops_reviewer", "compliance_reviewer", "payout_reviewer", "treasury_reviewer", "customer_support_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"corridors": {"supported_countries": SUPPORTED_COUNTRIES, "same_country_blocked": True, "corridor_policy_required": True},
	"quotes": {"supported_currencies": SUPPORTED_CURRENCIES, "positive_amount_required": True, "positive_fx_rate_required": True, "fee_non_negative_required": True, "expiry_required": True},
	"transfers": {"quote_lock_required": True, "sender_required": True, "beneficiary_required": True, "kyc_required": True, "funding_required": True, "source_of_funds_required": True, "high_value_threshold": 100000},
	"compliance": {"aml_screen_required": True, "sanctions_hit_blocks": True, "aml_review_requires_human_approval": True, "fraud_review_requires_human_approval": True},
	"payouts": {"supported_methods": SUPPORTED_PAYOUT_METHODS, "settlement_reference_required": True, "provider_receipt_required": True},
	"refunds": {"reason_required": True, "reviewer_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_remittance_events": True, "customer_consent_required": True},
	"observability": {"event_stream": REMITTANCE_EVENT_STREAM, "stream_processor": "bytewax", "emit_quote_events": True, "emit_transfer_events": True, "emit_payout_events": True, "emit_refund_events": True, "emit_agent_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_corridors": True, "enable_quotes": True, "enable_transfers": True, "enable_payouts": True, "enable_refunds": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_remittance_control", "allow_tenant_overrides": True},
}

PROVIDES = ["remittance_corridor_governance", "remittance_quote_lifecycle", "cross_border_transfer_workflow", "remittance_payout_workflow", "remittance_refund_workflow", "remittance_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_payments", "fintech_wallets", "fintech_kyc", "fintech_aml", "fintech_fraud"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-remittance/dashboard", "component": "RemittanceDashboard", "permission": "fintech_remittance:view", "nav_group": "Overview"},
	{"name": "corridors", "path": "/fintech-remittance/corridors", "component": "CorridorConsole", "permission": "fintech_remittance:govern_corridors", "nav_group": "Corridors"},
	{"name": "quotes", "path": "/fintech-remittance/quotes", "component": "QuoteWorkbench", "permission": "fintech_remittance:quote", "nav_group": "Quotes"},
	{"name": "transfers", "path": "/fintech-remittance/transfers", "component": "TransferWorkbench", "permission": "fintech_remittance:transfer", "nav_group": "Transfers"},
	{"name": "payouts", "path": "/fintech-remittance/payouts", "component": "PayoutConsole", "permission": "fintech_remittance:payout", "nav_group": "Payouts"},
	{"name": "refunds", "path": "/fintech-remittance/refunds", "component": "RefundConsole", "permission": "fintech_remittance:refund", "nav_group": "Exceptions"},
	{"name": "agents", "path": "/fintech-remittance/agents", "component": "RemittanceAgentWorkbench", "permission": "fintech_remittance:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-remittance/settings", "component": "RemittanceSettings", "permission": "fintech_remittance:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_remittance_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#2563EB", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"corridors": {"icon": "route", "status_indicator": "corridor-chip"}, "quotes": {"icon": "badge-dollar-sign", "status_indicator": "quote-chip"}, "transfers": {"icon": "send", "status_indicator": "transfer-status-chip"}, "payouts": {"icon": "landmark", "status_indicator": "payout-chip"}, "refunds": {"icon": "undo-2", "status_indicator": "refund-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": REMITTANCE_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["remittance_quote_created", "remittance_transfer_created", "remittance_payout_released", "remittance_refund_filed", "remittance_agent_registered"],
	"guardrails": ["remittance_batch_requires_bytewax", "remittance_event_requires_bytewax", "privileged_remittance_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Remittance operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "remittance_write_requires_policy", "description": "Remittance writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "remittance_policy_required", "required_action": "attach_remittance_policy"}},
	{"name": "corridor_supported", "description": "Remittance corridor must be supported.", "condition": {"operation": "quote_transfer", "corridor_supported": False}, "effect": {"decision": "deny", "reason": "corridor_not_supported", "required_action": "select_supported_corridor"}},
	{"name": "same_country_blocked", "description": "Cross-border remittance cannot use the same source and destination country.", "condition": {"operation": "quote_transfer", "same_country": True}, "effect": {"decision": "deny", "reason": "cross_border_corridor_required", "required_action": "select_cross_border_corridor"}},
	{"name": "source_currency_supported", "description": "Source currency must be supported.", "condition": {"operation": "quote_transfer", "source_currency_supported": False}, "effect": {"decision": "deny", "reason": "source_currency_not_supported", "required_action": "select_supported_source_currency"}},
	{"name": "destination_currency_supported", "description": "Destination currency must be supported.", "condition": {"operation": "quote_transfer", "destination_currency_supported": False}, "effect": {"decision": "deny", "reason": "destination_currency_not_supported", "required_action": "select_supported_destination_currency"}},
	{"name": "send_amount_positive", "description": "Quote send amount must be positive.", "condition": {"operation": "quote_transfer", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_amount_required", "required_action": "set_positive_send_amount"}},
	{"name": "fx_rate_positive", "description": "Quote FX rate must be positive.", "condition": {"operation": "quote_transfer", "positive_fx_rate": False}, "effect": {"decision": "deny", "reason": "positive_fx_rate_required", "required_action": "set_positive_fx_rate"}},
	{"name": "fee_non_negative", "description": "Quote fee cannot be negative.", "condition": {"operation": "quote_transfer", "fee_non_negative": False}, "effect": {"decision": "deny", "reason": "non_negative_fee_required", "required_action": "set_non_negative_fee"}},
	{"name": "quote_expiry_required", "description": "Quote expiry is required.", "condition": {"operation": "quote_transfer", "expiry_present": False}, "effect": {"decision": "deny", "reason": "quote_expiry_required", "required_action": "set_quote_expiry"}},
	{"name": "quote_lock_required", "description": "Transfer creation requires a quote lock.", "condition": {"operation": "create_transfer", "quote_present": False}, "effect": {"decision": "deny", "reason": "quote_lock_required", "required_action": "attach_quote_lock"}},
	{"name": "sender_required", "description": "Transfer requires sender reference.", "condition": {"operation": "create_transfer", "sender_present": False}, "effect": {"decision": "deny", "reason": "sender_required", "required_action": "attach_sender"}},
	{"name": "beneficiary_required", "description": "Transfer requires beneficiary reference.", "condition": {"operation": "create_transfer", "beneficiary_present": False}, "effect": {"decision": "deny", "reason": "beneficiary_required", "required_action": "attach_beneficiary"}},
	{"name": "sender_kyc_required", "description": "Transfer requires sender KYC.", "condition": {"operation": "create_transfer", "sender_kyc_present": False}, "effect": {"decision": "deny", "reason": "sender_kyc_required", "required_action": "attach_sender_kyc"}},
	{"name": "beneficiary_kyc_required", "description": "Transfer requires beneficiary KYC.", "condition": {"operation": "create_transfer", "beneficiary_kyc_present": False}, "effect": {"decision": "deny", "reason": "beneficiary_kyc_required", "required_action": "attach_beneficiary_kyc"}},
	{"name": "funding_reference_required", "description": "Transfer requires funding reference.", "condition": {"operation": "create_transfer", "funding_present": False}, "effect": {"decision": "deny", "reason": "funding_reference_required", "required_action": "attach_funding_reference"}},
	{"name": "payout_method_supported", "description": "Transfer payout method must be supported.", "condition": {"operation": "create_transfer", "payout_method_supported": False}, "effect": {"decision": "deny", "reason": "payout_method_not_supported", "required_action": "select_supported_payout_method"}},
	{"name": "purpose_code_supported", "description": "Transfer purpose code must be supported.", "condition": {"operation": "create_transfer", "purpose_code_supported": False}, "effect": {"decision": "deny", "reason": "purpose_code_not_supported", "required_action": "select_supported_purpose"}},
	{"name": "source_of_funds_required", "description": "Transfer requires source-of-funds evidence.", "condition": {"operation": "create_transfer", "source_of_funds_present": False}, "effect": {"decision": "deny", "reason": "source_of_funds_required", "required_action": "attach_source_of_funds"}},
	{"name": "aml_screen_required", "description": "Transfer requires AML screen evidence.", "condition": {"operation": "create_transfer", "aml_screen_present": False}, "effect": {"decision": "deny", "reason": "aml_screen_required", "required_action": "attach_aml_screen"}},
	{"name": "sanctions_hit_blocks_transfer", "description": "Sanctions hits block transfer creation.", "condition": {"operation": "create_transfer", "sanctions_hit": True}, "effect": {"decision": "deny", "reason": "sanctions_hit_blocked", "required_action": "resolve_sanctions_hit"}},
	{"name": "fraud_decision_supported", "description": "Fraud decision must be supported.", "condition": {"operation": "create_transfer", "fraud_decision_supported": False}, "effect": {"decision": "deny", "reason": "fraud_decision_not_supported", "required_action": "attach_supported_fraud_decision"}},
	{"name": "fraud_block_denies_transfer", "description": "Blocked fraud decisions deny transfer creation.", "condition": {"operation": "create_transfer", "fraud_blocked": True}, "effect": {"decision": "deny", "reason": "fraud_blocked_transfer", "required_action": "resolve_fraud_block"}},
	{"name": "aml_review_requires_approval", "description": "AML review outcomes require human approval.", "condition": {"operation": "create_transfer", "aml_review": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "aml_review_approval_required", "required_action": "record_aml_approval"}},
	{"name": "fraud_review_requires_approval", "description": "Fraud review or hold outcomes require human approval.", "condition": {"operation": "create_transfer", "fraud_review": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "fraud_review_approval_required", "required_action": "record_fraud_approval"}},
	{"name": "high_value_requires_approval", "description": "High-value transfers require approval.", "condition": {"operation": "create_transfer", "high_value": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "high_value_approval_required", "required_action": "record_high_value_approval"}},
	{"name": "payout_transfer_required", "description": "Payout release requires transfer reference.", "condition": {"operation": "release_payout", "transfer_present": False}, "effect": {"decision": "deny", "reason": "transfer_required", "required_action": "select_transfer"}},
	{"name": "provider_receipt_required", "description": "Payout release requires provider receipt.", "condition": {"operation": "release_payout", "provider_receipt_present": False}, "effect": {"decision": "deny", "reason": "provider_receipt_required", "required_action": "attach_provider_receipt"}},
	{"name": "settlement_reference_required", "description": "Payout release requires settlement reference.", "condition": {"operation": "release_payout", "settlement_reference_present": False}, "effect": {"decision": "deny", "reason": "settlement_reference_required", "required_action": "attach_settlement_reference"}},
	{"name": "refund_reason_required", "description": "Refund filing requires reason.", "condition": {"operation": "file_refund", "reason_present": False}, "effect": {"decision": "deny", "reason": "refund_reason_required", "required_action": "record_refund_reason"}},
	{"name": "refund_reviewer_required", "description": "Refund filing requires reviewer.", "condition": {"operation": "file_refund", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "refund_reviewer_required", "required_action": "assign_refund_reviewer"}},
	{"name": "remittance_batch_requires_bytewax", "description": "Remittance batches require Bytewax.", "condition": {"operation": "remittance_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_remittance_batch_to_bytewax"}},
	{"name": "remittance_agent_runtime_supported", "description": "Remittance agents must use a supported runtime.", "condition": {"operation": "register_remittance_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "remittance_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "remittance_agent_role_supported", "description": "Remittance agents must use a supported role.", "condition": {"operation": "register_remittance_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "remittance_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_remittance_agent_action_requires_human_approval", "description": "Privileged remittance-agent actions require human approval.", "condition": {"operation": "remittance_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},

	# Cross-tenant and privilege escalation guards
	{"name": "cross_tenant_remittance_access_denied", "description": "Remittance resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Remittance privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Africa-specific remittance rules (mobile money payout corridors)
	{"name": "mpesa_payout_shortcode_required", "description": "M-Pesa mobile money payout requires a registered Safaricom shortcode.", "condition": {"operation": "release_payout", "payout_method": "mobile_money", "provider": "mpesa", "mpesa_shortcode_present": False}, "effect": {"decision": "deny", "reason": "mpesa_shortcode_required", "required_action": "register_mpesa_shortcode"}},
	{"name": "mpesa_b2c_payout_approval_required", "description": "M-Pesa B2C remittance payouts above threshold require approval.", "condition": {"operation": "release_payout", "payout_method": "mobile_money", "provider": "mpesa", "high_value": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "mpesa_b2c_payout_approval_required", "required_action": "record_mpesa_b2c_approval"}},
	{"name": "airtel_money_payout_msisdn_required", "description": "Airtel Money payout requires a verified beneficiary MSISDN.", "condition": {"operation": "release_payout", "payout_method": "mobile_money", "provider": "airtel_money", "msisdn_verified": False}, "effect": {"decision": "deny", "reason": "airtel_money_msisdn_required", "required_action": "verify_beneficiary_msisdn"}},
	{"name": "mtn_momo_payout_registered_required", "description": "MTN Mobile Money payout requires registered beneficiary account.", "condition": {"operation": "release_payout", "payout_method": "mobile_money", "provider": "mtn_momo", "account_registered": False}, "effect": {"decision": "deny", "reason": "mtn_momo_account_required", "required_action": "register_mtn_momo_account"}},
	{"name": "ke_forex_bureau_licence_required", "description": "KES inbound remittance corridors require CBK-authorised forex bureau or PSP licence.", "condition": {"operation": "quote_transfer", "destination_country": "KE", "cbk_forex_licence_present": False}, "effect": {"decision": "deny", "reason": "cbk_forex_licence_required", "required_action": "attach_cbk_forex_licence"}},
	{"name": "ea_corridor_reporting_required", "description": "East African remittance corridors require CBK/BOK/BOT regulatory reporting.", "condition": {"operation": "create_transfer", "ea_corridor": True, "regulatory_report_filed": False}, "effect": {"decision": "require_review", "reason": "ea_corridor_regulatory_reporting_required", "required_action": "file_regulatory_report"}},
	{"name": "ng_cbn_remittance_compliance_required", "description": "Nigeria-bound remittances require CBN-compliant payout partner.", "condition": {"operation": "release_payout", "destination_country": "NG", "cbn_compliant_partner": False}, "effect": {"decision": "deny", "reason": "cbn_compliant_payout_partner_required", "required_action": "select_cbn_compliant_partner"}},
	{"name": "remittance_event_requires_bytewax", "description": "Remittance events require Bytewax stream processor.", "condition": {"operation": "remittance_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_required", "required_action": "route_remittance_event_to_bytewax"}},
	{"name": "cash_pickup_identity_verification_required", "description": "Cash pickup payouts require in-person identity verification.", "condition": {"operation": "release_payout", "payout_method": "cash_pickup", "identity_verified": False}, "effect": {"decision": "deny", "reason": "cash_pickup_identity_verification_required", "required_action": "verify_beneficiary_identity"}},
	{"name": "transfer_quote_expiry_enforced", "description": "Transfers using expired quotes are denied.", "condition": {"operation": "create_transfer", "quote_expired": True}, "effect": {"decision": "deny", "reason": "quote_expired", "required_action": "request_new_quote"}},
]


def _configuration_schema() -> dict[str, Any]:
	return {"type": "object", "required": list(DEFAULT_CONFIGURATION), "properties": {key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}


def _matches_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
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
	return {"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "configuration": configuration, "configuration_schema": _configuration_schema(), "provides": PROVIDES, "requires": REQUIRES, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-remittance/api/v1", "routes": deepcopy(UI_ROUTES), "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"]}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	contract = get_capability_contract(str(context.get("tenant_id") or "default"))
	matched = [rule for rule in contract["rule_engine"]["rules"] if _matches_condition(rule["condition"], context)]
	decision = "allow"
	for rule in matched:
		effect = rule["effect"]["decision"]
		if effect == "deny":
			decision = "deny"
			break
		if effect == "require_review" and decision == "allow":
			decision = "require_review"
	return {"decision": decision, "matched_rules": [rule["name"] for rule in matched], "actions": [rule["effect"] for rule in matched], "effects": [rule["effect"] for rule in matched]}
