"""Executable capability contract for APG Blockchain Ledger Services."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ledgers": {"ledger_owner_required": True, "network_policy_required": True, "consensus_profile_required": True, "fork_monitoring_enabled": True},
	"transactions": {"signature_required": True, "key_custody_required": True, "compliance_mapping_required": True, "high_value_review_threshold": 100000},
	"smart_contracts": {"contract_review_required": True, "artifact_hash_required": True, "deployment_approval_required": True, "rollback_plan_required": True},
	"governance": {"require_tenant_context": True, "audit_ledger_changes": True, "key_rotation_policy_required": True, "chain_fork_review_required": True},
	"ui": {"enable_ledger_console": True, "enable_transaction_monitor": True, "enable_contract_registry": True, "enable_key_custody_view": True},
	"theme": {"default_theme": "bclg_ledger_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "ledgers", "transactions", "smart_contracts", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["ledgers", "transactions", "smart_contracts", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All ledger operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "ledger_requires_owner", "description": "Ledgers require an accountable owner.", "condition": {"operation": "create_ledger", "ledger_owner_assigned": False}, "effect": {"decision": "deny", "reason": "ledger_owner_required", "required_action": "assign_ledger_owner"}},
	{"name": "transaction_requires_signature", "description": "Ledger transactions require signatures.", "condition": {"operation": "submit_transaction", "signature_present": False}, "effect": {"decision": "deny", "reason": "transaction_signature_required", "required_action": "sign_transaction"}},
	{"name": "key_custody_required", "description": "Ledger operations require managed key custody.", "condition": {"key_custody_bound": False}, "effect": {"decision": "deny", "reason": "key_custody_required", "required_action": "bind_key_custody"}},
	{"name": "contract_requires_review", "description": "Smart contract deployment requires review.", "condition": {"operation": "deploy_contract", "contract_review_recorded": False}, "effect": {"decision": "deny", "reason": "contract_review_required", "required_action": "review_contract"}},
	{"name": "high_value_transaction_requires_review", "description": "High-value transactions require review.", "condition": {"transaction_value_gt": 100000, "transaction_review_recorded": False}, "effect": {"decision": "require_review", "reason": "high_value_transaction_review_required", "required_action": "review_transaction"}},
	{"name": "transaction_review_requires_independent_reviewer", "description": "High-value transaction reviews require an independent reviewer.", "condition": {"operation": "approve_transaction_review", "reviewer_same_as_submitter": True}, "effect": {"decision": "deny", "reason": "independent_transaction_reviewer_required", "required_action": "route_to_independent_transaction_reviewer"}},
	{"name": "contract_deployment_review_requires_independent_reviewer", "description": "Smart contract deployment reviews require an independent reviewer.", "condition": {"operation": "approve_contract_deployment", "reviewer_same_as_requester": True}, "effect": {"decision": "deny", "reason": "independent_contract_reviewer_required", "required_action": "route_to_independent_contract_reviewer"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/bclg/dashboard", "component": "BCLGDashboard", "permission": "bclg:view", "nav_group": "Overview"},
	{"name": "ledgers", "path": "/bclg/ledgers", "component": "LedgerConsole", "permission": "bclg:manage_ledgers", "nav_group": "Ledgers"},
	{"name": "transactions", "path": "/bclg/transactions", "component": "TransactionMonitor", "permission": "bclg:transact", "nav_group": "Transactions"},
	{"name": "transaction_reviews", "path": "/bclg/transactions/reviews", "component": "TransactionReviewQueue", "permission": "bclg:review_transactions", "nav_group": "Transactions"},
	{"name": "contracts", "path": "/bclg/contracts", "component": "SmartContractRegistry", "permission": "bclg:manage_contracts", "nav_group": "Contracts"},
	{"name": "contract_reviews", "path": "/bclg/contracts/reviews", "component": "ContractDeploymentReviewQueue", "permission": "bclg:review_contracts", "nav_group": "Contracts"},
	{"name": "keys", "path": "/bclg/keys", "component": "KeyCustodyView", "permission": "bclg:admin", "nav_group": "Security"},
	{"name": "audit", "path": "/bclg/audit", "component": "LedgerAudit", "permission": "bclg:view", "nav_group": "Governance"},
	{"name": "compliance", "path": "/bclg/compliance", "component": "LedgerCompliance", "permission": "bclg:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/bclg/settings", "component": "BCLGSettings", "permission": "bclg:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "bclg_ledger_ops",
	"tokens": {"color.primary": "#2A4365", "color.accent": "#805AD5", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"ledger_card": {"icon": "blocks", "status_indicator": "chain-pill", "risk_style": "fork-band"}, "transaction_monitor": {"visual": "signed-ledger-table", "highlight": "review-chip"}, "transaction_review_queue": {"visual": "approval-lane", "status_style": "risk-review-chip"}, "contract_registry": {"visual": "artifact-list", "status_style": "hash-chip"}, "contract_review_queue": {"visual": "deployment-approval-lane", "status_style": "artifact-review-chip"}, "key_custody": {"visual": "custody-matrix", "status_style": "rotation-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "bclg", "display_name": "Blockchain Ledger Services", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/bclg/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
