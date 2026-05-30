"""Executable capability contract for APG Blockchain Ledger Services."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any

SUPPORTED_LEDGER_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_LEDGER_AGENT_ROLES = ["ledger_reviewer", "transaction_reviewer", "contract_reviewer", "custody_reviewer", "fork_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ledgers": {"ledger_owner_required": True, "network_policy_required": True, "consensus_profile_required": True, "fork_monitoring_enabled": True},
	"transactions": {"signature_required": True, "key_custody_required": True, "compliance_mapping_required": True, "high_value_review_threshold": 100000},
	"smart_contracts": {"contract_review_required": True, "artifact_hash_required": True, "deployment_approval_required": True, "rollback_plan_required": True},
	"ledger_agents": {"agent_assist_enabled": True, "agent_registration_required": True, "agent_runtime_required": True, "agent_scope_required": True, "agent_contribution_disclosure_required": True, "supported_runtimes": SUPPORTED_LEDGER_AGENT_RUNTIMES, "allowed_roles": SUPPORTED_LEDGER_AGENT_ROLES},
	"governance": {"require_tenant_context": True, "audit_ledger_changes": True, "key_rotation_policy_required": True, "chain_fork_review_required": True, "batch_event_stream": "bytewax"},
	"observability": {"audit_required": True, "trace_required": True, "ledger_metrics_required": True, "agent_activity_required": True, "event_stream": "bytewax"},
	"adapters": {"generated_app_runtime": "service.BclgService", "api_helpers": "api.py", "view_models": "views.py", "event_stream": "bytewax", "audit_sink": "audl", "key_management": "keym", "encryption": "encr", "compliance": "comp", "wallet": "walt", "security": "secu"},
	"ui": {"enable_ledger_console": True, "enable_transaction_monitor": True, "enable_contract_registry": True, "enable_key_custody_view": True, "enable_agent_panel": True, "enable_audit": True, "enable_analytics": True},
	"theme": {"default_theme": "bclg_ledger_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "ledgers", "transactions", "smart_contracts", "ledger_agents", "governance", "observability", "adapters", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["ledgers", "transactions", "smart_contracts", "ledger_agents", "governance", "observability", "adapters", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All ledger operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "ledger_requires_owner", "description": "Ledgers require an accountable owner.", "condition": {"operation": "create_ledger", "ledger_owner_assigned": False}, "effect": {"decision": "deny", "reason": "ledger_owner_required", "required_action": "assign_ledger_owner"}},
	{"name": "transaction_requires_signature", "description": "Ledger transactions require signatures.", "condition": {"operation": "submit_transaction", "signature_present": False}, "effect": {"decision": "deny", "reason": "transaction_signature_required", "required_action": "sign_transaction"}},
	{"name": "key_custody_required", "description": "Ledger operations require managed key custody.", "condition": {"key_custody_bound": False}, "effect": {"decision": "deny", "reason": "key_custody_required", "required_action": "bind_key_custody"}},
	{"name": "contract_requires_review", "description": "Smart contract deployment requires review.", "condition": {"operation": "deploy_contract", "contract_review_recorded": False}, "effect": {"decision": "deny", "reason": "contract_review_required", "required_action": "review_contract"}},
	{"name": "high_value_transaction_requires_review", "description": "High-value transactions require review.", "condition": {"transaction_value_gt": 100000, "transaction_review_recorded": False}, "effect": {"decision": "require_review", "reason": "high_value_transaction_review_required", "required_action": "review_transaction"}},
	{"name": "transaction_review_requires_independent_reviewer", "description": "High-value transaction reviews require an independent reviewer.", "condition": {"operation": "approve_transaction_review", "reviewer_same_as_submitter": True}, "effect": {"decision": "deny", "reason": "independent_transaction_reviewer_required", "required_action": "route_to_independent_transaction_reviewer"}},
	{"name": "contract_deployment_review_requires_independent_reviewer", "description": "Smart contract deployment reviews require an independent reviewer.", "condition": {"operation": "approve_contract_deployment", "reviewer_same_as_requester": True}, "effect": {"decision": "deny", "reason": "independent_contract_reviewer_required", "required_action": "route_to_independent_contract_reviewer"}},
	{"name": "ledger_agent_requires_registration", "description": "AI ledger agents must be registered.", "condition": {"ledger_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "ledger_agent_registration_required", "required_action": "register_ledger_agent"}},
	{"name": "ledger_agent_runtime_supported", "description": "AI ledger agents must use a supported runtime.", "condition": {"ledger_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "ledger_agent_runtime_not_supported", "required_action": "choose_supported_ledger_agent_runtime"}},
	{"name": "ledger_agent_role_supported", "description": "AI ledger agents must use a supported role.", "condition": {"ledger_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "ledger_agent_role_not_supported", "required_action": "choose_supported_ledger_agent_role"}},
	{"name": "ledger_agent_requires_scope", "description": "AI ledger agents require explicit scope.", "condition": {"ledger_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "ledger_agent_scope_required", "required_action": "set_ledger_agent_scope"}},
	{"name": "ledger_agent_requires_disclosure", "description": "AI ledger-agent contributions require disclosure.", "condition": {"ledger_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "ledger_agent_disclosure_required", "required_action": "disclose_ledger_agent"}},
	{"name": "ledger_state_change_requires_audit", "description": "BCLG lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "ledger_audit_event_required", "required_action": "record_ledger_audit_event"}},
	{"name": "batch_ledger_mutation_requires_bytewax", "description": "Batch BCLG mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_ledger_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/bclg/dashboard", "component": "BCLGDashboard", "permission": "bclg:view", "nav_group": "Overview"},
	{"name": "ledgers", "path": "/bclg/ledgers", "component": "LedgerConsole", "permission": "bclg:manage_ledgers", "nav_group": "Ledgers"},
	{"name": "transactions", "path": "/bclg/transactions", "component": "TransactionMonitor", "permission": "bclg:transact", "nav_group": "Transactions"},
	{"name": "transaction_reviews", "path": "/bclg/transactions/reviews", "component": "TransactionReviewQueue", "permission": "bclg:review_transactions", "nav_group": "Transactions"},
	{"name": "contracts", "path": "/bclg/contracts", "component": "SmartContractRegistry", "permission": "bclg:manage_contracts", "nav_group": "Contracts"},
	{"name": "contract_reviews", "path": "/bclg/contracts/reviews", "component": "ContractDeploymentReviewQueue", "permission": "bclg:review_contracts", "nav_group": "Contracts"},
	{"name": "keys", "path": "/bclg/keys", "component": "KeyCustodyView", "permission": "bclg:admin", "nav_group": "Security"},
	{"name": "ledger_agents", "path": "/bclg/agents", "component": "LedgerAgentPanel", "permission": "bclg:review_transactions", "nav_group": "Governance"},
	{"name": "audit", "path": "/bclg/audit", "component": "LedgerAudit", "permission": "bclg:view", "nav_group": "Governance"},
	{"name": "analytics", "path": "/bclg/analytics", "component": "LedgerAnalytics", "permission": "bclg:view", "nav_group": "Operations"},
	{"name": "compliance", "path": "/bclg/compliance", "component": "LedgerCompliance", "permission": "bclg:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/bclg/settings", "component": "BCLGSettings", "permission": "bclg:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "bclg_ledger_ops",
	"tokens": {"color.primary": "#2A4365", "color.accent": "#805AD5", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"ledger_card": {"icon": "blocks", "status_indicator": "chain-pill", "risk_style": "fork-band"}, "transaction_monitor": {"visual": "signed-ledger-table", "highlight": "review-chip"}, "transaction_review_queue": {"visual": "approval-lane", "status_style": "risk-review-chip"}, "contract_registry": {"visual": "artifact-list", "status_style": "hash-chip"}, "contract_review_queue": {"visual": "deployment-approval-lane", "status_style": "artifact-review-chip"}, "key_custody": {"visual": "custody-matrix", "status_style": "rotation-chip"}, "ledger_agent_panel": {"icon": "bot", "status_style": "scope-chip"}, "stream_health": {"visual": "event-lane", "status_style": "stream-chip"}}
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.bclg.lifecycle",
		"state": ["ledgers", "key_custody", "transactions", "transaction_reviews", "contract_reviews", "contracts", "ledger_agents", "audit_events", "ledger_heads"],
		"events": ["ledger_registered", "key_custody_bound", "transaction_submitted", "transaction_review_requested", "transaction_review_decided", "contract_deployment_review_requested", "contract_deployment_review_decided", "contract_deployed", "ledger_agent_registered"],
		"batch_mutation_guardrail": "batch_ledger_mutation_requires_bytewax"
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "bclg", "display_name": "Blockchain Ledger Services", "provides": ["ledger_registry", "transaction_governance", "smart_contract_governance", "key_custody_governance", "ledger_audit", "ledger_agents"], "requires": ["encr", "keym", "comp"], "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/bclg/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME), "streaming": streaming_manifest()}


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
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
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
