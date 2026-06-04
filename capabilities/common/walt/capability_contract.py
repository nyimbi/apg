"""Executable capability contract for APG Wallet and Payment Core."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_WALT_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_WALT_AGENT_ROLES = [
	"payment_reviewer",
	"risk_reviewer",
	"settlement_reviewer",
	"reconciliation_reviewer",
	"instrument_reviewer",
	"chargeback_reviewer",
]
WALT_EVENT_STREAM = "apg.walt.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"wallets": {
		"wallet_owner_required": True,
		"ledger_integrity_required": True,
		"multi_currency_enabled": True,
		"negative_balance_blocking": True,
	},
	"payments": {
		"instrument_tokenization_required": True,
		"instrument_verification_required": True,
		"transaction_limits_required": True,
		"mfa_for_high_value": True,
		"risk_scoring_required": True,
	},
	"settlement": {
		"settlement_approval_required": True,
		"reconciliation_required": True,
		"exception_queue_enabled": True,
		"chargeback_supported": True,
		"settlement_stream_required": True,
	},
	"walt_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_WALT_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_WALT_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "non_privileged",
		"disclose_agent_recommendations": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_financial_events": True,
		"encrypted_instruments_required": True,
		"compliance_policy_required": True,
		"state_change_audit_required": True,
	},
	"observability": {
		"event_stream": WALT_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_wallet_events": True,
		"emit_transaction_events": True,
		"emit_settlement_events": True,
	},
	"adapters": {
		"encryption": "adapter",
		"authorization": "adapter",
		"compliance": "adapter",
		"ledger": "adapter",
		"audit": "adapter",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_wallet_dashboard": True,
		"enable_transaction_console": True,
		"enable_settlement_center": True,
		"enable_reconciliation_queue": True,
		"enable_agent_workbench": True,
		"enable_policy_center": True,
	},
	"theme": {"default_theme": "walt_wallet_ops", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"wallets",
		"payments",
		"settlement",
		"walt_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"wallets",
			"payments",
			"settlement",
			"walt_agents",
			"governance",
			"observability",
			"adapters",
			"ui",
			"theme",
		]
	} | {"tenant_id": {"type": "string", "minLength": 1}},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All wallet operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "wallet_requires_owner", "description": "Wallets require an accountable owner.", "condition": {"operation": "create_wallet", "wallet_owner_assigned": False}, "effect": {"decision": "deny", "reason": "wallet_owner_required", "required_action": "assign_wallet_owner"}},
	{"name": "wallet_requires_ledger", "description": "Wallets require ledger integrity evidence.", "condition": {"operation": "create_wallet", "ledger_ref_present": False}, "effect": {"decision": "deny", "reason": "ledger_integrity_required", "required_action": "attach_ledger_reference"}},
	{"name": "wallet_requires_compliance_policy", "description": "Wallets require a compliance policy reference.", "condition": {"operation": "create_wallet", "compliance_policy_present": False}, "effect": {"decision": "deny", "reason": "compliance_policy_required", "required_action": "attach_compliance_policy"}},
	{"name": "instrument_requires_encryption", "description": "Payment instruments must be encrypted or tokenized.", "condition": {"operation": "register_instrument", "payment_instrument_present": True, "instrument_encrypted": False}, "effect": {"decision": "deny", "reason": "instrument_encryption_required", "required_action": "encrypt_or_tokenize_instrument"}},
	{"name": "instrument_requires_token", "description": "Payment instruments require a token reference.", "condition": {"operation": "register_instrument", "instrument_token_present": False}, "effect": {"decision": "deny", "reason": "instrument_tokenization_required", "required_action": "attach_instrument_token"}},
	{"name": "instrument_requires_verification", "description": "Payment instruments require verification attribution.", "condition": {"operation": "register_instrument", "instrument_verifier_present": False}, "effect": {"decision": "deny", "reason": "instrument_verification_required", "required_action": "record_instrument_verifier"}},
	{"name": "high_value_requires_mfa", "description": "High-value transactions require MFA.", "condition": {"transaction_amount_gt": 10000, "mfa_completed": False}, "effect": {"decision": "deny", "reason": "high_value_mfa_required", "required_action": "complete_mfa"}},
	{"name": "transaction_requires_risk_score", "description": "Transaction authorization requires risk score evidence.", "condition": {"operation": "authorize_transaction", "risk_score_present": False}, "effect": {"decision": "deny", "reason": "risk_score_required", "required_action": "attach_transaction_risk_score"}},
	{"name": "transaction_requires_bytewax_stream", "description": "Transaction lifecycle events must be emitted through Bytewax.", "condition": {"operation": "authorize_transaction", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_transaction_lifecycle_to_bytewax"}},
	{"name": "settlement_requires_reconciliation", "description": "Settlement requires reconciliation evidence.", "condition": {"operation": "settle_batch", "reconciliation_completed": False}, "effect": {"decision": "deny", "reason": "reconciliation_required", "required_action": "complete_reconciliation"}},
	{"name": "settlement_requires_approval", "description": "Settlement requires approval evidence.", "condition": {"operation": "settle_batch", "settlement_approval_recorded": False}, "effect": {"decision": "deny", "reason": "settlement_approval_required", "required_action": "record_settlement_approval"}},
	{"name": "settlement_requires_bytewax_stream", "description": "Settlement lifecycle events must be emitted through Bytewax.", "condition": {"operation": "settle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_settlement_lifecycle_to_bytewax"}},
	{"name": "reconciliation_requires_evidence", "description": "Reconciliation requires a durable evidence reference.", "condition": {"operation": "record_reconciliation", "reconciliation_evidence_present": False}, "effect": {"decision": "deny", "reason": "reconciliation_evidence_required", "required_action": "attach_reconciliation_evidence"}},
	{"name": "high_risk_transaction_requires_review", "description": "High-risk transactions require review.", "condition": {"transaction_risk_score_gt": 0.8, "risk_review_recorded": False}, "effect": {"decision": "require_review", "reason": "risk_review_required", "required_action": "review_transaction_risk"}},
	{"name": "batch_settlement_requires_bytewax", "description": "Batch settlement mutation requires Bytewax stream coordination.", "condition": {"operation": "batch_settlement", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_batch_settlement_to_bytewax"}},
	{"name": "walt_agent_runtime_supported", "description": "Wallet/payment agents must use an approved runtime.", "condition": {"operation": "register_walt_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "walt_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "walt_agent_role_supported", "description": "Wallet/payment agents must use an approved role.", "condition": {"operation": "register_walt_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "walt_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_payment_action_requires_human_approval", "description": "Privileged payment actions proposed by agents require human approval.", "condition": {"operation": "agent_payment_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "write_requires_policy", "description": "Wallet and payment write operations require an explicit authorization policy.", "condition": {"operation_type": "write", "write_policy_present": False}, "effect": {"decision": "deny", "reason": "walt_write_policy_required", "required_action": "attach_write_policy"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/walt/dashboard", "component": "WALTDashboard", "permission": "walt:view", "nav_group": "Overview"},
	{"name": "wallets", "path": "/walt/wallets", "component": "WalletConsole", "permission": "walt:manage_wallets", "nav_group": "Wallets"},
	{"name": "transactions", "path": "/walt/transactions", "component": "TransactionConsole", "permission": "walt:authorize", "nav_group": "Transactions"},
	{"name": "instruments", "path": "/walt/instruments", "component": "PaymentInstruments", "permission": "walt:manage_wallets", "nav_group": "Payments"},
	{"name": "settlement", "path": "/walt/settlement", "component": "SettlementCenter", "permission": "walt:settle", "nav_group": "Settlement"},
	{"name": "reconciliation", "path": "/walt/reconciliation", "component": "ReconciliationQueue", "permission": "walt:settle", "nav_group": "Settlement"},
	{"name": "risk", "path": "/walt/risk", "component": "PaymentRisk", "permission": "walt:view", "nav_group": "Governance"},
	{"name": "agents", "path": "/walt/agents", "component": "WALTAgentWorkbench", "permission": "walt:admin", "nav_group": "Automation"},
	{"name": "policy", "path": "/walt/policy", "component": "WALTPolicyCenter", "permission": "walt:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/walt/settings", "component": "WALTSettings", "permission": "walt:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "walt_wallet_ops",
	"tokens": {"color.primary": "#214E34", "color.accent": "#B7791F", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"wallet_grid": {"icon": "wallet", "status_indicator": "balance-pill", "risk_style": "ledger-band"},
		"transaction_table": {"visual": "transaction-list", "highlight": "risk-chip"},
		"settlement_center": {"visual": "batch-timeline", "status_style": "reconciliation-chip"},
		"instrument_vault": {"visual": "tokenized-card-list", "status_style": "encryption-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
		"policy_center": {"visual": "rule-grid", "status_style": "guardrail-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "walt",
		"display_name": "Wallet and Payment Core",
		"provides": [
			"wallet_ledger",
			"payment_instruments",
			"transaction_authorization",
			"settlement",
			"reconciliation",
			"payment_risk_governance",
			"walt_agents",
		],
		"requires": ["encr", "auth", "comp", "audl", "wflo"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/walt/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": WALT_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"wallet_created",
			"instrument_registered",
			"transaction_authorized",
			"transaction_captured",
			"settlement_batch_created",
			"reconciliation_recorded",
			"walt_agent_registered",
		],
		"states": ["active", "authorized", "captured", "review_required", "settled", "reconciled", "exception_review", "blocked"],
		"guardrails": [
			"transaction_requires_bytewax_stream",
			"settlement_requires_bytewax_stream",
			"batch_settlement_requires_bytewax",
			"privileged_agent_payment_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return WALT_EVENT_STREAM


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
