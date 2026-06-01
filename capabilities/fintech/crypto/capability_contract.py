"""Executable capability contract for APG Cryptocurrency Services."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_crypto"
CAPABILITY_NAME = "Cryptocurrency Services"
CAPABILITY_VERSION = "1.1.0"
CRYPTO_EVENT_STREAM = "apg.fintech.crypto.lifecycle"

SUPPORTED_ASSET_TYPES = ["native_coin", "stablecoin", "utility_token", "security_token", "governance_token", "tokenized_deposit"]
SUPPORTED_CUSTODY_MODELS = ["self_custody", "mpc", "hsm", "exchange_custody", "smart_contract", "custodial"]
SUPPORTED_ORDER_SIDES = ["buy", "sell", "swap"]
SUPPORTED_ORDER_TYPES = ["market", "limit", "stop_limit", "rfq", "rebalance"]
SUPPORTED_TRADE_STATUSES = ["requested", "approved", "executed", "settled", "failed", "cancelled"]
SUPPORTED_TRANSFER_TYPES = ["deposit", "withdrawal", "internal", "settlement", "treasury_rebalance"]
SUPPORTED_TRANSFER_STATUSES = ["requested", "approved", "broadcast", "confirmed", "failed", "cancelled"]
SUPPORTED_SCREENING_TYPES = ["wallet", "transaction", "asset", "counterparty", "sanctions", "travel_rule"]
SUPPORTED_SCREENING_STATUSES = ["clear", "review", "blocked", "escalated"]
SUPPORTED_PRICE_SOURCES = ["exchange", "oracle", "custodian", "manual", "aggregator"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["portfolio_monitor", "trade_reviewer", "custody_reconciler", "compliance_screening_agent", "treasury_rebalancer", "market_data_agent"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"assets": {"supported_types": SUPPORTED_ASSET_TYPES, "symbol_required": True, "network_reference_required": True, "precision_non_negative": True, "owner_required": True, "evidence_required": True},
	"custody_accounts": {"supported_custody_models": SUPPORTED_CUSTODY_MODELS, "provider_required": True, "policy_required": True, "owner_required": True, "evidence_required": True},
	"balances": {"account_required": True, "asset_required": True, "amount_non_negative": True, "valuation_non_negative": True, "currency_required": True, "evidence_required": True},
	"orders": {"account_required": True, "asset_required": True, "supported_sides": SUPPORTED_ORDER_SIDES, "supported_types": SUPPORTED_ORDER_TYPES, "quantity_positive": True, "limit_price_required_for_limit_orders": True, "policy_required": True, "requester_required": True, "evidence_required": True},
	"trades": {"order_required": True, "venue_required": True, "execution_price_non_negative": True, "quantity_positive": True, "fee_non_negative": True, "supported_statuses": SUPPORTED_TRADE_STATUSES, "settlement_reference_required": True},
	"transfers": {"account_required": True, "asset_required": True, "supported_types": SUPPORTED_TRANSFER_TYPES, "destination_required": True, "amount_positive": True, "approval_required": True, "evidence_required": True, "supported_statuses": SUPPORTED_TRANSFER_STATUSES},
	"screening": {"supported_types": SUPPORTED_SCREENING_TYPES, "supported_statuses": SUPPORTED_SCREENING_STATUSES, "reference_required": True, "reviewer_required_for_non_clear": True, "evidence_required": True},
	"prices": {"asset_required": True, "supported_sources": SUPPORTED_PRICE_SOURCES, "price_non_negative": True, "currency_required": True, "observed_at_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "large_transfer_requires_approval": True},
	"observability": {"event_stream": CRYPTO_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "blockchain": "fintech_blockchain", "wallets": "fintech_wallets", "risk": "fintech_risk", "compliance": "fintech_compliance", "regtech": "fintech_regtech", "aml": "fintech_aml", "kyc": "fintech_kyc", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_assets": True, "enable_custody": True, "enable_balances": True, "enable_orders": True, "enable_trades": True, "enable_transfers": True, "enable_screening": True, "enable_prices": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_crypto_control", "allow_tenant_overrides": True},
}

PROVIDES = ["crypto_asset_workflow", "crypto_custody_workflow", "crypto_balance_workflow", "crypto_order_workflow", "crypto_trade_workflow", "crypto_transfer_workflow", "crypto_screening_workflow", "crypto_price_workflow", "crypto_review_workflow", "crypto_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_blockchain", "fintech_wallets", "fintech_risk", "fintech_compliance", "fintech_regtech", "fintech_aml", "fintech_kyc"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-crypto/dashboard", "component": "CryptoDashboard", "permission": "fintech_crypto:view", "nav_group": "Overview"},
	{"name": "assets", "path": "/fintech-crypto/assets", "component": "CryptoAssetRegistry", "permission": "fintech_crypto:assets", "nav_group": "Assets"},
	{"name": "custody", "path": "/fintech-crypto/custody", "component": "CustodyAccountConsole", "permission": "fintech_crypto:custody", "nav_group": "Custody"},
	{"name": "balances", "path": "/fintech-crypto/balances", "component": "CryptoBalanceLedger", "permission": "fintech_crypto:balances", "nav_group": "Portfolio"},
	{"name": "orders", "path": "/fintech-crypto/orders", "component": "CryptoOrderWorkbench", "permission": "fintech_crypto:orders", "nav_group": "Trading"},
	{"name": "trades", "path": "/fintech-crypto/trades", "component": "CryptoTradeBlotter", "permission": "fintech_crypto:trades", "nav_group": "Trading"},
	{"name": "transfers", "path": "/fintech-crypto/transfers", "component": "CryptoTransferQueue", "permission": "fintech_crypto:transfers", "nav_group": "Treasury"},
	{"name": "screening", "path": "/fintech-crypto/screening", "component": "CryptoScreeningConsole", "permission": "fintech_crypto:screening", "nav_group": "Compliance"},
	{"name": "prices", "path": "/fintech-crypto/prices", "component": "CryptoPriceConsole", "permission": "fintech_crypto:prices", "nav_group": "Market Data"},
	{"name": "reviews", "path": "/fintech-crypto/reviews", "component": "CryptoReviewConsole", "permission": "fintech_crypto:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/fintech-crypto/agents", "component": "CryptoAgentWorkbench", "permission": "fintech_crypto:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-crypto/settings", "component": "CryptoSettings", "permission": "fintech_crypto:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_crypto_control",
	"tokens": {"color.primary": "#0E7490", "color.accent": "#7C3AED", "color.success": "#15803D", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"assets": {"icon": "coins", "status_indicator": "asset-chip"}, "custody": {"icon": "shield-keyhole", "status_indicator": "custody-chip"}, "balances": {"icon": "wallet", "status_indicator": "balance-chip"}, "orders": {"icon": "list-plus", "status_indicator": "order-chip"}, "trades": {"icon": "candlestick-chart", "status_indicator": "trade-chip"}, "transfers": {"icon": "send", "status_indicator": "transfer-chip"}, "screening": {"icon": "shield-alert", "status_indicator": "screening-chip"}, "prices": {"icon": "chart-line", "status_indicator": "price-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": CRYPTO_EVENT_STREAM, "key": "tenant_id", "events": ["crypto_asset_registered", "crypto_custody_account_opened", "crypto_balance_recorded", "crypto_order_created", "crypto_trade_recorded", "crypto_transfer_requested", "crypto_screening_recorded", "crypto_price_recorded", "crypto_review_recorded", "crypto_agent_registered"], "guardrails": ["crypto_batch_requires_bytewax", "privileged_crypto_agent_action_requires_human_approval"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "crypto_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "crypto_policy_required", "required_action": "attach_crypto_policy"}},
	{"name": "asset_symbol_required", "condition": {"operation": "register_asset", "symbol_present": False}, "effect": {"decision": "deny", "reason": "asset_symbol_required", "required_action": "attach_symbol"}},
	{"name": "asset_type_supported", "condition": {"operation": "register_asset", "asset_type_supported": False}, "effect": {"decision": "deny", "reason": "asset_type_not_supported", "required_action": "select_supported_asset_type"}},
	{"name": "asset_network_required", "condition": {"operation": "register_asset", "network_present": False}, "effect": {"decision": "deny", "reason": "asset_network_required", "required_action": "attach_network_reference"}},
	{"name": "asset_precision_valid", "condition": {"operation": "register_asset", "precision_valid": False}, "effect": {"decision": "deny", "reason": "asset_precision_invalid", "required_action": "set_non_negative_precision"}},
	{"name": "asset_owner_required", "condition": {"operation": "register_asset", "owner_present": False}, "effect": {"decision": "deny", "reason": "asset_owner_required", "required_action": "assign_asset_owner"}},
	{"name": "asset_evidence_required", "condition": {"operation": "register_asset", "evidence_present": False}, "effect": {"decision": "deny", "reason": "asset_evidence_required", "required_action": "attach_asset_evidence"}},
	{"name": "custody_model_supported", "condition": {"operation": "open_custody_account", "custody_model_supported": False}, "effect": {"decision": "deny", "reason": "custody_model_not_supported", "required_action": "select_supported_custody_model"}},
	{"name": "custody_provider_required", "condition": {"operation": "open_custody_account", "provider_present": False}, "effect": {"decision": "deny", "reason": "custody_provider_required", "required_action": "attach_provider_reference"}},
	{"name": "custody_policy_required", "condition": {"operation": "open_custody_account", "policy_present": False}, "effect": {"decision": "deny", "reason": "custody_policy_required", "required_action": "attach_policy_reference"}},
	{"name": "custody_owner_required", "condition": {"operation": "open_custody_account", "owner_present": False}, "effect": {"decision": "deny", "reason": "custody_owner_required", "required_action": "assign_owner"}},
	{"name": "custody_evidence_required", "condition": {"operation": "open_custody_account", "evidence_present": False}, "effect": {"decision": "deny", "reason": "custody_evidence_required", "required_action": "attach_evidence"}},
	{"name": "balance_account_required", "condition": {"operation": "record_balance", "account_present": False}, "effect": {"decision": "deny", "reason": "custody_account_required", "required_action": "select_custody_account"}},
	{"name": "balance_asset_required", "condition": {"operation": "record_balance", "asset_present": False}, "effect": {"decision": "deny", "reason": "asset_required", "required_action": "select_asset"}},
	{"name": "balance_amount_valid", "condition": {"operation": "record_balance", "amount_valid": False}, "effect": {"decision": "deny", "reason": "balance_amount_invalid", "required_action": "set_non_negative_amount"}},
	{"name": "balance_valuation_valid", "condition": {"operation": "record_balance", "valuation_valid": False}, "effect": {"decision": "deny", "reason": "valuation_amount_invalid", "required_action": "set_non_negative_valuation"}},
	{"name": "balance_currency_required", "condition": {"operation": "record_balance", "currency_present": False}, "effect": {"decision": "deny", "reason": "valuation_currency_required", "required_action": "set_currency"}},
	{"name": "balance_evidence_required", "condition": {"operation": "record_balance", "evidence_present": False}, "effect": {"decision": "deny", "reason": "balance_evidence_required", "required_action": "attach_balance_evidence"}},
	{"name": "order_account_required", "condition": {"operation": "create_order", "account_present": False}, "effect": {"decision": "deny", "reason": "custody_account_required", "required_action": "select_custody_account"}},
	{"name": "order_asset_required", "condition": {"operation": "create_order", "asset_present": False}, "effect": {"decision": "deny", "reason": "asset_required", "required_action": "select_asset"}},
	{"name": "order_side_supported", "condition": {"operation": "create_order", "side_supported": False}, "effect": {"decision": "deny", "reason": "order_side_not_supported", "required_action": "select_supported_side"}},
	{"name": "order_type_supported", "condition": {"operation": "create_order", "order_type_supported": False}, "effect": {"decision": "deny", "reason": "order_type_not_supported", "required_action": "select_supported_order_type"}},
	{"name": "order_quantity_valid", "condition": {"operation": "create_order", "quantity_valid": False}, "effect": {"decision": "deny", "reason": "order_quantity_invalid", "required_action": "set_positive_quantity"}},
	{"name": "limit_order_requires_price", "condition": {"operation": "create_order", "limit_price_required": True, "limit_price_present": False}, "effect": {"decision": "deny", "reason": "limit_price_required", "required_action": "set_limit_price"}},
	{"name": "order_policy_required", "condition": {"operation": "create_order", "policy_present": False}, "effect": {"decision": "deny", "reason": "order_policy_required", "required_action": "attach_order_policy"}},
	{"name": "order_requester_required", "condition": {"operation": "create_order", "requester_present": False}, "effect": {"decision": "deny", "reason": "order_requester_required", "required_action": "record_requester"}},
	{"name": "order_evidence_required", "condition": {"operation": "create_order", "evidence_present": False}, "effect": {"decision": "deny", "reason": "order_evidence_required", "required_action": "attach_order_evidence"}},
	{"name": "trade_order_required", "condition": {"operation": "record_trade", "order_present": False}, "effect": {"decision": "deny", "reason": "order_required", "required_action": "select_order"}},
	{"name": "trade_venue_required", "condition": {"operation": "record_trade", "venue_present": False}, "effect": {"decision": "deny", "reason": "trade_venue_required", "required_action": "attach_venue"}},
	{"name": "trade_price_valid", "condition": {"operation": "record_trade", "execution_price_valid": False}, "effect": {"decision": "deny", "reason": "execution_price_invalid", "required_action": "set_non_negative_price"}},
	{"name": "trade_quantity_valid", "condition": {"operation": "record_trade", "quantity_valid": False}, "effect": {"decision": "deny", "reason": "trade_quantity_invalid", "required_action": "set_positive_quantity"}},
	{"name": "trade_fee_valid", "condition": {"operation": "record_trade", "fee_valid": False}, "effect": {"decision": "deny", "reason": "trade_fee_invalid", "required_action": "set_non_negative_fee"}},
	{"name": "trade_status_supported", "condition": {"operation": "record_trade", "status_supported": False}, "effect": {"decision": "deny", "reason": "trade_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "trade_settlement_required", "condition": {"operation": "record_trade", "settlement_present": False}, "effect": {"decision": "deny", "reason": "trade_settlement_required", "required_action": "attach_settlement_reference"}},
	{"name": "transfer_account_required", "condition": {"operation": "request_transfer", "account_present": False}, "effect": {"decision": "deny", "reason": "custody_account_required", "required_action": "select_custody_account"}},
	{"name": "transfer_asset_required", "condition": {"operation": "request_transfer", "asset_present": False}, "effect": {"decision": "deny", "reason": "asset_required", "required_action": "select_asset"}},
	{"name": "transfer_type_supported", "condition": {"operation": "request_transfer", "transfer_type_supported": False}, "effect": {"decision": "deny", "reason": "transfer_type_not_supported", "required_action": "select_supported_transfer_type"}},
	{"name": "transfer_destination_required", "condition": {"operation": "request_transfer", "destination_present": False}, "effect": {"decision": "deny", "reason": "transfer_destination_required", "required_action": "attach_destination"}},
	{"name": "transfer_amount_valid", "condition": {"operation": "request_transfer", "amount_valid": False}, "effect": {"decision": "deny", "reason": "transfer_amount_invalid", "required_action": "set_positive_amount"}},
	{"name": "transfer_approval_required", "condition": {"operation": "request_transfer", "approval_present": False}, "effect": {"decision": "deny", "reason": "transfer_approval_required", "required_action": "attach_transfer_approval"}},
	{"name": "transfer_evidence_required", "condition": {"operation": "request_transfer", "evidence_present": False}, "effect": {"decision": "deny", "reason": "transfer_evidence_required", "required_action": "attach_transfer_evidence"}},
	{"name": "transfer_status_supported", "condition": {"operation": "request_transfer", "status_supported": False}, "effect": {"decision": "deny", "reason": "transfer_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "screening_reference_required", "condition": {"operation": "record_screening", "reference_present": False}, "effect": {"decision": "deny", "reason": "screening_reference_required", "required_action": "attach_reference"}},
	{"name": "screening_type_supported", "condition": {"operation": "record_screening", "screening_type_supported": False}, "effect": {"decision": "deny", "reason": "screening_type_not_supported", "required_action": "select_supported_screening_type"}},
	{"name": "screening_status_supported", "condition": {"operation": "record_screening", "status_supported": False}, "effect": {"decision": "deny", "reason": "screening_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "screening_evidence_required", "condition": {"operation": "record_screening", "evidence_present": False}, "effect": {"decision": "deny", "reason": "screening_evidence_required", "required_action": "attach_screening_evidence"}},
	{"name": "non_clear_screening_requires_reviewer", "condition": {"operation": "record_screening", "reviewer_required": True, "reviewer_present": False}, "effect": {"decision": "deny", "reason": "screening_reviewer_required", "required_action": "assign_screening_reviewer"}},
	{"name": "price_asset_required", "condition": {"operation": "record_price", "asset_present": False}, "effect": {"decision": "deny", "reason": "asset_required", "required_action": "select_asset"}},
	{"name": "price_source_supported", "condition": {"operation": "record_price", "source_supported": False}, "effect": {"decision": "deny", "reason": "price_source_not_supported", "required_action": "select_supported_price_source"}},
	{"name": "price_amount_valid", "condition": {"operation": "record_price", "price_valid": False}, "effect": {"decision": "deny", "reason": "price_amount_invalid", "required_action": "set_non_negative_price"}},
	{"name": "price_currency_required", "condition": {"operation": "record_price", "currency_present": False}, "effect": {"decision": "deny", "reason": "price_currency_required", "required_action": "set_currency"}},
	{"name": "price_observed_at_required", "condition": {"operation": "record_price", "observed_at_present": False}, "effect": {"decision": "deny", "reason": "price_observed_at_required", "required_action": "record_observed_at"}},
	{"name": "price_evidence_required", "condition": {"operation": "record_price", "evidence_present": False}, "effect": {"decision": "deny", "reason": "price_evidence_required", "required_action": "attach_price_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "crypto_batch_requires_bytewax", "condition": {"operation": "crypto_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_crypto_batch_to_bytewax"}},
	{"name": "crypto_agent_runtime_supported", "condition": {"operation": "register_crypto_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "crypto_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "crypto_agent_role_supported", "condition": {"operation": "register_crypto_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "crypto_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_crypto_agent_action_requires_human_approval", "condition": {"operation": "crypto_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-crypto/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
