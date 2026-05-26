"""Executable capability contract for APG Wallet and Payment Core."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"wallets": {"wallet_owner_required": True, "ledger_integrity_required": True, "multi_currency_enabled": True, "negative_balance_blocking": True},
	"payments": {"instrument_tokenization_required": True, "transaction_limits_required": True, "mfa_for_high_value": True, "risk_scoring_required": True},
	"settlement": {"settlement_approval_required": True, "reconciliation_required": True, "exception_queue_enabled": True, "chargeback_supported": True},
	"governance": {"require_tenant_context": True, "audit_financial_events": True, "encrypted_instruments_required": True, "compliance_policy_required": True},
	"ui": {"enable_wallet_dashboard": True, "enable_transaction_console": True, "enable_settlement_center": True, "enable_reconciliation_queue": True},
	"theme": {"default_theme": "walt_wallet_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "wallets", "payments", "settlement", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["wallets", "payments", "settlement", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All wallet operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "wallet_requires_owner", "description": "Wallets require an accountable owner.", "condition": {"operation": "create_wallet", "wallet_owner_assigned": False}, "effect": {"decision": "deny", "reason": "wallet_owner_required", "required_action": "assign_wallet_owner"}},
	{"name": "instrument_requires_encryption", "description": "Payment instruments must be encrypted or tokenized.", "condition": {"payment_instrument_present": True, "instrument_encrypted": False}, "effect": {"decision": "deny", "reason": "instrument_encryption_required", "required_action": "encrypt_or_tokenize_instrument"}},
	{"name": "high_value_requires_mfa", "description": "High-value transactions require MFA.", "condition": {"transaction_amount_gt": 10000, "mfa_completed": False}, "effect": {"decision": "deny", "reason": "high_value_mfa_required", "required_action": "complete_mfa"}},
	{"name": "settlement_requires_reconciliation", "description": "Settlement requires reconciliation evidence.", "condition": {"operation": "settle_batch", "reconciliation_completed": False}, "effect": {"decision": "deny", "reason": "reconciliation_required", "required_action": "complete_reconciliation"}},
	{"name": "high_risk_transaction_requires_review", "description": "High-risk transactions require review.", "condition": {"transaction_risk_score_gt": 0.8, "risk_review_recorded": False}, "effect": {"decision": "require_review", "reason": "risk_review_required", "required_action": "review_transaction_risk"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/walt/dashboard", "component": "WALTDashboard", "permission": "walt:view", "nav_group": "Overview"},
	{"name": "wallets", "path": "/walt/wallets", "component": "WalletConsole", "permission": "walt:manage_wallets", "nav_group": "Wallets"},
	{"name": "transactions", "path": "/walt/transactions", "component": "TransactionConsole", "permission": "walt:authorize", "nav_group": "Transactions"},
	{"name": "instruments", "path": "/walt/instruments", "component": "PaymentInstruments", "permission": "walt:manage_wallets", "nav_group": "Payments"},
	{"name": "settlement", "path": "/walt/settlement", "component": "SettlementCenter", "permission": "walt:settle", "nav_group": "Settlement"},
	{"name": "reconciliation", "path": "/walt/reconciliation", "component": "ReconciliationQueue", "permission": "walt:settle", "nav_group": "Settlement"},
	{"name": "risk", "path": "/walt/risk", "component": "PaymentRisk", "permission": "walt:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/walt/settings", "component": "WALTSettings", "permission": "walt:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "walt_wallet_ops", "tokens": {"color.primary": "#214E34", "color.accent": "#B7791F", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"wallet_grid": {"icon": "wallet", "status_indicator": "balance-pill", "risk_style": "ledger-band"}, "transaction_table": {"visual": "transaction-list", "highlight": "risk-chip"}, "settlement_center": {"visual": "batch-timeline", "status_style": "reconciliation-chip"}, "instrument_vault": {"visual": "tokenized-card-list", "status_style": "encryption-chip"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "walt", "display_name": "Wallet and Payment Core", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/walt/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
