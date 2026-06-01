"""Executable capability contract for APG Digital Wallets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_wallets"
CAPABILITY_NAME = "Digital Wallets"
CAPABILITY_VERSION = "1.1.0"
WALLETS_EVENT_STREAM = "apg.fintech.wallets.lifecycle"

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_WALLET_TYPES = ["consumer", "merchant", "agent", "escrow", "treasury"]
SUPPORTED_INSTRUMENT_TYPES = ["bank_account", "card", "mobile_money", "wallet", "voucher"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["wallet_ops_reviewer", "risk_reviewer", "limits_reviewer", "settlement_reviewer", "dispute_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"wallets": {
		"owner_required": True,
		"supported_types": SUPPORTED_WALLET_TYPES,
		"supported_currencies": SUPPORTED_CURRENCIES,
		"negative_balance_blocked": True,
	},
	"instruments": {
		"supported_types": SUPPORTED_INSTRUMENT_TYPES,
		"token_reference_required": True,
		"verified_instrument_required": True,
	},
	"ledger": {
		"double_entry_required": True,
		"hold_balance_required": True,
		"idempotency_required": True,
	},
	"limits": {
		"daily_debit_limit_minor": 500000,
		"single_transfer_limit_minor": 250000,
		"review_required_for_limit_override": True,
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
		"audit_financial_events": True,
		"customer_consent_required": True,
	},
	"observability": {
		"event_stream": WALLETS_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_wallet_events": True,
		"emit_ledger_events": True,
		"emit_limit_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"wallet_core": "walt",
		"payments": "fintech_payments",
		"gateway": "fintech_gateway",
		"keys": "keym",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_wallets": True,
		"enable_instruments": True,
		"enable_ledger": True,
		"enable_limits": True,
		"enable_holds": True,
		"enable_agents": True,
	},
	"theme": {"default_theme": "fintech_wallets_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"wallet_lifecycle",
	"stored_value_ledger",
	"wallet_instrument_registry",
	"wallet_transfer_workflow",
	"wallet_hold_workflow",
	"wallet_limit_governance",
	"wallet_agent_workflow",
]

REQUIRES = ["auth", "audl", "ntfy", "walt", "fintech_payments", "fintech_gateway", "keym"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-wallets/dashboard", "component": "WalletsDashboard", "permission": "fintech_wallets:view", "nav_group": "Overview"},
	{"name": "wallets", "path": "/fintech-wallets/wallets", "component": "WalletWorkbench", "permission": "fintech_wallets:manage_wallets", "nav_group": "Wallets"},
	{"name": "instruments", "path": "/fintech-wallets/instruments", "component": "WalletInstrumentVault", "permission": "fintech_wallets:manage_instruments", "nav_group": "Wallets"},
	{"name": "ledger", "path": "/fintech-wallets/ledger", "component": "WalletLedger", "permission": "fintech_wallets:view_ledger", "nav_group": "Ledger"},
	{"name": "limits", "path": "/fintech-wallets/limits", "component": "WalletLimitConsole", "permission": "fintech_wallets:govern_limits", "nav_group": "Governance"},
	{"name": "holds", "path": "/fintech-wallets/holds", "component": "WalletHoldConsole", "permission": "fintech_wallets:operate", "nav_group": "Operations"},
	{"name": "agents", "path": "/fintech-wallets/agents", "component": "WalletAgentWorkbench", "permission": "fintech_wallets:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-wallets/settings", "component": "WalletSettings", "permission": "fintech_wallets:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_wallets_control",
	"tokens": {
		"color.primary": "#1E3A5F",
		"color.accent": "#0F766E",
		"color.success": "#15803D",
		"color.warning": "#A16207",
		"color.danger": "#B42318",
		"surface.canvas": "#F7F9FC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"wallets": {"icon": "wallet", "status_indicator": "balance-pill"},
		"instruments": {"icon": "credit-card", "status_indicator": "verified-chip"},
		"ledger": {"visual": "ledger-grid", "status_style": "entry-chip"},
		"limits": {"visual": "limit-band", "status_style": "threshold-chip"},
		"holds": {"visual": "hold-lane", "status_style": "reserve-chip"},
		"agents": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": WALLETS_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"wallet_opened",
		"wallet_instrument_registered",
		"wallet_credited",
		"wallet_debited",
		"wallet_transfer_posted",
		"wallet_hold_placed",
		"wallet_hold_released",
		"wallet_limit_updated",
		"wallet_agent_registered",
	],
	"guardrails": ["wallet_batch_requires_bytewax", "wallet_event_requires_bytewax", "privileged_wallet_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Wallet operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "wallet_write_requires_policy", "description": "Wallet writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "wallet_policy_required", "required_action": "attach_wallet_policy"}},
	{"name": "wallet_owner_required", "description": "Wallets require an owner reference.", "condition": {"operation": "open_wallet", "owner_present": False}, "effect": {"decision": "deny", "reason": "wallet_owner_required", "required_action": "attach_owner_reference"}},
	{"name": "wallet_type_supported", "description": "Wallet type must be supported.", "condition": {"operation": "open_wallet", "wallet_type_supported": False}, "effect": {"decision": "deny", "reason": "wallet_type_not_supported", "required_action": "select_supported_wallet_type"}},
	{"name": "wallet_currency_supported", "description": "Wallet currency must be supported.", "condition": {"operation": "open_wallet", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "instrument_wallet_required", "description": "Wallet instruments require an existing wallet.", "condition": {"operation": "register_instrument", "wallet_present": False}, "effect": {"decision": "deny", "reason": "wallet_required", "required_action": "select_wallet"}},
	{"name": "instrument_type_supported", "description": "Wallet instrument type must be supported.", "condition": {"operation": "register_instrument", "instrument_type_supported": False}, "effect": {"decision": "deny", "reason": "instrument_type_not_supported", "required_action": "select_supported_instrument_type"}},
	{"name": "instrument_token_required", "description": "Wallet instruments require token reference.", "condition": {"operation": "register_instrument", "token_reference_present": False}, "effect": {"decision": "deny", "reason": "instrument_token_required", "required_action": "attach_token_reference"}},
	{"name": "instrument_verification_required", "description": "Wallet instruments require verification evidence.", "condition": {"operation": "register_instrument", "verified": False}, "effect": {"decision": "deny", "reason": "instrument_verification_required", "required_action": "verify_instrument"}},
	{"name": "credit_amount_positive", "description": "Wallet credits require positive amounts.", "condition": {"operation": "credit_wallet", "amount_lte": 0}, "effect": {"decision": "deny", "reason": "credit_amount_positive_required", "required_action": "set_positive_amount"}},
	{"name": "debit_amount_positive", "description": "Wallet debits require positive amounts.", "condition": {"operation": "debit_wallet", "amount_lte": 0}, "effect": {"decision": "deny", "reason": "debit_amount_positive_required", "required_action": "set_positive_amount"}},
	{"name": "debit_blocks_negative_balance", "description": "Wallet debits cannot create negative available balance.", "condition": {"operation": "debit_wallet", "insufficient_available_balance": True}, "effect": {"decision": "deny", "reason": "insufficient_available_balance", "required_action": "fund_wallet_or_reduce_amount"}},
	{"name": "transfer_requires_distinct_wallets", "description": "Wallet transfers require distinct wallets.", "condition": {"operation": "transfer", "same_wallet": True}, "effect": {"decision": "deny", "reason": "distinct_wallets_required", "required_action": "select_different_wallets"}},
	{"name": "transfer_requires_matching_currency", "description": "Wallet transfers require matching wallet currencies.", "condition": {"operation": "transfer", "currency_mismatch": True}, "effect": {"decision": "deny", "reason": "wallet_currency_mismatch", "required_action": "route_through_fx_or_select_matching_wallet"}},
	{"name": "transfer_limit_requires_review", "description": "Large wallet transfers require review.", "condition": {"operation": "transfer", "limit_exceeded": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "wallet_limit_review_required", "required_action": "record_limit_review"}},
	{"name": "hold_amount_positive", "description": "Wallet holds require positive amounts.", "condition": {"operation": "place_hold", "amount_lte": 0}, "effect": {"decision": "deny", "reason": "hold_amount_positive_required", "required_action": "set_positive_hold_amount"}},
	{"name": "hold_blocks_negative_available", "description": "Wallet holds cannot exceed available balance.", "condition": {"operation": "place_hold", "insufficient_available_balance": True}, "effect": {"decision": "deny", "reason": "hold_exceeds_available_balance", "required_action": "reduce_hold_amount"}},
	{"name": "hold_release_amount_positive", "description": "Wallet hold releases require positive amounts.", "condition": {"operation": "release_hold", "amount_lte": 0}, "effect": {"decision": "deny", "reason": "hold_release_amount_positive_required", "required_action": "set_positive_release_amount"}},
	{"name": "hold_release_blocks_overrelease", "description": "Wallet hold releases cannot exceed held balance.", "condition": {"operation": "release_hold", "release_exceeds_held_balance": True}, "effect": {"decision": "deny", "reason": "hold_release_exceeds_held_balance", "required_action": "reduce_release_amount"}},
	{"name": "wallet_batch_requires_bytewax", "description": "Wallet batches require Bytewax.", "condition": {"operation": "wallet_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_wallet_batch_to_bytewax"}},
	{"name": "wallet_event_requires_bytewax", "description": "Wallet events require Bytewax.", "condition": {"operation": "wallet_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_wallet_event_to_bytewax"}},
	{"name": "wallet_agent_runtime_supported", "description": "Wallet agents must use a supported runtime.", "condition": {"operation": "register_wallet_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "wallet_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "wallet_agent_role_supported", "description": "Wallet agents must use a supported role.", "condition": {"operation": "register_wallet_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "wallet_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_wallet_agent_action_requires_human_approval", "description": "Privileged wallet-agent actions require human approval.", "condition": {"operation": "wallet_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def _configuration_schema() -> dict[str, Any]:
	return {"type": "object", "required": list(DEFAULT_CONFIGURATION), "properties": {key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}


def _matches_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			if context.get(key[:-4]) is None or context[key[:-4]] > expected:
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
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "api_prefix": "/fintech-wallets/api/v1", "routes": deepcopy(UI_ROUTES), "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"]},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


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
