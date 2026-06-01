"""Executable capability contract for APG Digital Cards."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_cards"
CAPABILITY_NAME = "Digital Cards"
CAPABILITY_VERSION = "1.1.0"
CARDS_EVENT_STREAM = "apg.fintech.cards.lifecycle"

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_COUNTRIES = ["KE", "UG", "TZ", "RW", "GH", "NG", "ZA", "GB", "US", "AE"]
SUPPORTED_CARD_TYPES = ["virtual", "physical"]
SUPPORTED_PRODUCTS = ["debit", "prepaid", "expense", "fleet", "merchant"]
SUPPORTED_TOKEN_TYPES = ["wallet", "device", "merchant", "network"]
SUPPORTED_MERCHANT_CATEGORIES = ["grocery", "fuel", "travel", "education", "medical", "commerce", "cash", "restricted"]
SUPPORTED_FRAUD_DECISIONS = ["clear", "review", "hold", "block"]
SUPPORTED_AML_RESULTS = ["clear", "review", "blocked"]
SUPPORTED_DISPUTE_REASONS = ["fraud", "duplicate", "goods_not_received", "service_not_provided", "cash_not_dispensed", "processing_error"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["card_ops_reviewer", "authorization_reviewer", "token_reviewer", "dispute_reviewer", "issuer_processor_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"programs": {"owner_required": True, "bin_range_required": True, "supported_currencies": SUPPORTED_CURRENCIES, "settlement_account_required": True},
	"cardholders": {"customer_reference_required": True, "kyc_required": True, "supported_countries": SUPPORTED_COUNTRIES},
	"cards": {"supported_types": SUPPORTED_CARD_TYPES, "supported_products": SUPPORTED_PRODUCTS, "wallet_required": True, "funding_account_required": True, "consent_required": True, "shipping_required_for_physical": True},
	"tokens": {"supported_types": SUPPORTED_TOKEN_TYPES, "token_reference_required": True, "key_domain_required": True, "device_or_merchant_reference_required": True},
	"authorizations": {"supported_currencies": SUPPORTED_CURRENCIES, "supported_merchant_categories": SUPPORTED_MERCHANT_CATEGORIES, "positive_amount_required": True, "fraud_decision_required": True, "aml_result_required": True, "high_value_threshold": 100000, "human_approval_required_for_high_impact": True},
	"disputes": {"supported_reasons": SUPPORTED_DISPUTE_REASONS, "evidence_required": True, "reviewer_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_card_events": True, "customer_consent_required": True},
	"observability": {"event_stream": CARDS_EVENT_STREAM, "stream_processor": "bytewax", "emit_program_events": True, "emit_card_events": True, "emit_token_events": True, "emit_authorization_events": True, "emit_dispute_events": True, "emit_agent_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "encryption": "encr", "payments": "fintech_payments", "wallets": "fintech_wallets", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_programs": True, "enable_cardholders": True, "enable_cards": True, "enable_tokens": True, "enable_authorizations": True, "enable_disputes": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_cards_control", "allow_tenant_overrides": True},
}

PROVIDES = ["card_program_governance", "cardholder_card_lifecycle", "tokenized_card_credentialing", "card_authorization_control", "card_dispute_workflow", "card_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "encr", "fintech_payments", "fintech_wallets", "fintech_kyc", "fintech_aml", "fintech_fraud"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-cards/dashboard", "component": "CardsDashboard", "permission": "fintech_cards:view", "nav_group": "Overview"},
	{"name": "programs", "path": "/fintech-cards/programs", "component": "CardProgramConsole", "permission": "fintech_cards:manage_programs", "nav_group": "Programs"},
	{"name": "cardholders", "path": "/fintech-cards/cardholders", "component": "CardholderWorkbench", "permission": "fintech_cards:manage_cardholders", "nav_group": "Cards"},
	{"name": "cards", "path": "/fintech-cards/cards", "component": "CardWorkbench", "permission": "fintech_cards:issue", "nav_group": "Cards"},
	{"name": "tokens", "path": "/fintech-cards/tokens", "component": "CardTokenConsole", "permission": "fintech_cards:tokenize", "nav_group": "Tokens"},
	{"name": "authorizations", "path": "/fintech-cards/authorizations", "component": "AuthorizationConsole", "permission": "fintech_cards:authorize", "nav_group": "Controls"},
	{"name": "disputes", "path": "/fintech-cards/disputes", "component": "CardDisputeWorkbench", "permission": "fintech_cards:dispute", "nav_group": "Exceptions"},
	{"name": "agents", "path": "/fintech-cards/agents", "component": "CardAgentWorkbench", "permission": "fintech_cards:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-cards/settings", "component": "CardSettings", "permission": "fintech_cards:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_cards_control",
	"tokens": {"color.primary": "#1D4ED8", "color.accent": "#0F766E", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"programs": {"icon": "badge-credit-card", "status_indicator": "program-chip"}, "cards": {"icon": "credit-card", "status_indicator": "card-status-chip"}, "tokens": {"icon": "key-round", "status_indicator": "token-chip"}, "authorizations": {"icon": "shield-check", "status_indicator": "auth-chip"}, "disputes": {"icon": "receipt-text", "status_indicator": "dispute-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": CARDS_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["card_program_registered", "cardholder_onboarded", "card_issued", "card_token_provisioned", "card_authorization_decided", "card_dispute_filed", "card_agent_registered"],
	"guardrails": ["card_batch_requires_bytewax", "privileged_card_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Card operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "card_write_requires_policy", "description": "Card writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "card_policy_required", "required_action": "attach_card_policy"}},
	{"name": "program_owner_required", "description": "Card programs require owner.", "condition": {"operation": "register_program", "program_owner_present": False}, "effect": {"decision": "deny", "reason": "program_owner_required", "required_action": "assign_program_owner"}},
	{"name": "program_bin_range_required", "description": "Card programs require BIN range.", "condition": {"operation": "register_program", "bin_range_present": False}, "effect": {"decision": "deny", "reason": "bin_range_required", "required_action": "attach_bin_range"}},
	{"name": "program_currency_supported", "description": "Card program currency must be supported.", "condition": {"operation": "register_program", "currency_supported": False}, "effect": {"decision": "deny", "reason": "card_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "program_settlement_required", "description": "Card programs require settlement account.", "condition": {"operation": "register_program", "settlement_account_present": False}, "effect": {"decision": "deny", "reason": "settlement_account_required", "required_action": "attach_settlement_account"}},
	{"name": "cardholder_customer_required", "description": "Cardholders require customer reference.", "condition": {"operation": "onboard_cardholder", "customer_present": False}, "effect": {"decision": "deny", "reason": "customer_reference_required", "required_action": "attach_customer_reference"}},
	{"name": "cardholder_kyc_required", "description": "Cardholders require KYC evidence.", "condition": {"operation": "onboard_cardholder", "kyc_present": False}, "effect": {"decision": "deny", "reason": "cardholder_kyc_required", "required_action": "attach_kyc_profile"}},
	{"name": "cardholder_country_supported", "description": "Cardholder country must be supported.", "condition": {"operation": "onboard_cardholder", "country_supported": False}, "effect": {"decision": "deny", "reason": "cardholder_country_not_supported", "required_action": "select_supported_country"}},
	{"name": "card_program_required", "description": "Card issuance requires program.", "condition": {"operation": "issue_card", "program_present": False}, "effect": {"decision": "deny", "reason": "card_program_required", "required_action": "select_card_program"}},
	{"name": "cardholder_required", "description": "Card issuance requires cardholder.", "condition": {"operation": "issue_card", "cardholder_present": False}, "effect": {"decision": "deny", "reason": "cardholder_required", "required_action": "select_cardholder"}},
	{"name": "card_type_supported", "description": "Card type must be supported.", "condition": {"operation": "issue_card", "card_type_supported": False}, "effect": {"decision": "deny", "reason": "card_type_not_supported", "required_action": "select_supported_card_type"}},
	{"name": "card_product_supported", "description": "Card product must be supported.", "condition": {"operation": "issue_card", "card_product_supported": False}, "effect": {"decision": "deny", "reason": "card_product_not_supported", "required_action": "select_supported_card_product"}},
	{"name": "wallet_reference_required", "description": "Card issuance requires wallet reference.", "condition": {"operation": "issue_card", "wallet_present": False}, "effect": {"decision": "deny", "reason": "wallet_reference_required", "required_action": "attach_wallet_reference"}},
	{"name": "funding_account_required", "description": "Card issuance requires funding account.", "condition": {"operation": "issue_card", "funding_account_present": False}, "effect": {"decision": "deny", "reason": "funding_account_required", "required_action": "attach_funding_account"}},
	{"name": "card_consent_required", "description": "Card issuance requires consent evidence.", "condition": {"operation": "issue_card", "consent_present": False}, "effect": {"decision": "deny", "reason": "cardholder_consent_required", "required_action": "attach_cardholder_consent"}},
	{"name": "physical_shipping_required", "description": "Physical card issuance requires shipping address.", "condition": {"operation": "issue_card", "physical_card": True, "shipping_present": False}, "effect": {"decision": "deny", "reason": "shipping_address_required", "required_action": "attach_shipping_address"}},
	{"name": "token_card_required", "description": "Token provisioning requires card.", "condition": {"operation": "provision_token", "card_present": False}, "effect": {"decision": "deny", "reason": "card_required", "required_action": "select_card"}},
	{"name": "token_type_supported", "description": "Token type must be supported.", "condition": {"operation": "provision_token", "token_type_supported": False}, "effect": {"decision": "deny", "reason": "token_type_not_supported", "required_action": "select_supported_token_type"}},
	{"name": "token_reference_required", "description": "Token provisioning requires token reference.", "condition": {"operation": "provision_token", "token_reference_present": False}, "effect": {"decision": "deny", "reason": "token_reference_required", "required_action": "attach_token_reference"}},
	{"name": "token_key_domain_required", "description": "Token provisioning requires key domain.", "condition": {"operation": "provision_token", "key_domain_present": False}, "effect": {"decision": "deny", "reason": "key_domain_required", "required_action": "attach_key_domain"}},
	{"name": "token_device_or_merchant_required", "description": "Token provisioning requires device or merchant reference.", "condition": {"operation": "provision_token", "device_or_merchant_present": False}, "effect": {"decision": "deny", "reason": "device_or_merchant_reference_required", "required_action": "attach_device_or_merchant_reference"}},
	{"name": "authorization_card_required", "description": "Authorization requires card.", "condition": {"operation": "authorize_transaction", "card_present": False}, "effect": {"decision": "deny", "reason": "card_required", "required_action": "select_card"}},
	{"name": "authorization_amount_positive", "description": "Authorization amount must be positive.", "condition": {"operation": "authorize_transaction", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_amount_required", "required_action": "set_positive_amount"}},
	{"name": "authorization_currency_supported", "description": "Authorization currency must be supported.", "condition": {"operation": "authorize_transaction", "currency_supported": False}, "effect": {"decision": "deny", "reason": "card_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "merchant_category_supported", "description": "Merchant category must be supported.", "condition": {"operation": "authorize_transaction", "merchant_category_supported": False}, "effect": {"decision": "deny", "reason": "merchant_category_not_supported", "required_action": "select_supported_merchant_category"}},
	{"name": "fraud_decision_supported", "description": "Fraud decision must be supported.", "condition": {"operation": "authorize_transaction", "fraud_decision_supported": False}, "effect": {"decision": "deny", "reason": "fraud_decision_not_supported", "required_action": "attach_supported_fraud_decision"}},
	{"name": "fraud_block_denies_authorization", "description": "Blocked fraud decisions deny authorization.", "condition": {"operation": "authorize_transaction", "fraud_blocked": True}, "effect": {"decision": "deny", "reason": "fraud_blocked_authorization", "required_action": "resolve_fraud_block"}},
	{"name": "aml_result_supported", "description": "AML result must be supported.", "condition": {"operation": "authorize_transaction", "aml_result_supported": False}, "effect": {"decision": "deny", "reason": "aml_result_not_supported", "required_action": "attach_supported_aml_result"}},
	{"name": "aml_block_denies_authorization", "description": "Blocked AML outcomes deny authorization.", "condition": {"operation": "authorize_transaction", "aml_blocked": True}, "effect": {"decision": "deny", "reason": "aml_blocked_authorization", "required_action": "resolve_aml_block"}},
	{"name": "high_impact_authorization_requires_approval", "description": "High-impact authorizations require approval.", "condition": {"operation": "authorize_transaction", "high_impact": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "authorization_approval_required", "required_action": "record_authorization_approval"}},
	{"name": "dispute_transaction_required", "description": "Disputes require transaction reference.", "condition": {"operation": "file_dispute", "transaction_present": False}, "effect": {"decision": "deny", "reason": "transaction_reference_required", "required_action": "attach_transaction_reference"}},
	{"name": "dispute_reason_supported", "description": "Dispute reason must be supported.", "condition": {"operation": "file_dispute", "dispute_reason_supported": False}, "effect": {"decision": "deny", "reason": "dispute_reason_not_supported", "required_action": "select_supported_dispute_reason"}},
	{"name": "dispute_evidence_required", "description": "Disputes require evidence.", "condition": {"operation": "file_dispute", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dispute_evidence_required", "required_action": "attach_dispute_evidence"}},
	{"name": "dispute_reviewer_required", "description": "Disputes require reviewer.", "condition": {"operation": "file_dispute", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "dispute_reviewer_required", "required_action": "assign_dispute_reviewer"}},
	{"name": "card_batch_requires_bytewax", "description": "Card batches require Bytewax.", "condition": {"operation": "card_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_card_batch_to_bytewax"}},
	{"name": "card_agent_runtime_supported", "description": "Card agents must use a supported runtime.", "condition": {"operation": "register_card_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "card_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "card_agent_role_supported", "description": "Card agents must use a supported role.", "condition": {"operation": "register_card_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "card_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_card_agent_action_requires_human_approval", "description": "Privileged card-agent actions require human approval.", "condition": {"operation": "card_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
	return {"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "configuration": configuration, "configuration_schema": _configuration_schema(), "provides": PROVIDES, "requires": REQUIRES, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-cards/api/v1", "routes": deepcopy(UI_ROUTES), "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"]}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
