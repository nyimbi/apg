"""Executable capability contract for APG Decentralized Finance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_defi"
CAPABILITY_NAME = "Decentralized Finance"
CAPABILITY_VERSION = "1.1.0"
DEFI_EVENT_STREAM = "apg.fintech.defi.lifecycle"

SUPPORTED_PROTOCOL_TYPES = ["lending_pool", "liquidity_pool", "staking", "yield_vault", "dex", "bridge", "derivatives", "insurance_pool"]
SUPPORTED_POSITION_TYPES = ["supply", "borrow", "liquidity", "stake", "vault_share", "short", "long", "cover"]
SUPPORTED_RISK_TIERS = ["low", "medium", "high", "critical"]
SUPPORTED_ACTION_TYPES = ["deposit", "withdraw", "borrow", "repay", "swap", "stake", "unstake", "claim", "rebalance"]
SUPPORTED_REWARD_TYPES = ["interest", "fee_share", "staking_reward", "liquidity_mining", "governance"]
SUPPORTED_GOVERNANCE_VOTES = ["for", "against", "abstain"]
SUPPORTED_ACTION_STATUSES = ["requested", "approved", "submitted", "confirmed", "failed", "cancelled"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["protocol_monitor", "position_reconciler", "liquidity_risk_agent", "governance_reviewer", "yield_strategy_agent", "treasury_rebalancer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"protocols": {"supported_protocol_types": SUPPORTED_PROTOCOL_TYPES, "supported_risk_tiers": SUPPORTED_RISK_TIERS, "network_reference_required": True, "protocol_reference_required": True, "owner_required": True, "evidence_required": True},
	"positions": {"supported_position_types": SUPPORTED_POSITION_TYPES, "protocol_required": True, "account_required": True, "asset_pair_required": True, "amount_positive": True, "collateral_non_negative": True, "health_factor_positive": True, "evidence_required": True},
	"actions": {"supported_action_types": SUPPORTED_ACTION_TYPES, "supported_statuses": SUPPORTED_ACTION_STATUSES, "protocol_required": True, "position_required": True, "position_protocol_match_required": True, "amount_positive": True, "requester_required": True, "approval_required": True, "evidence_required": True},
	"yield_strategies": {"target_apy_non_negative": True, "supported_max_risk_tiers": SUPPORTED_RISK_TIERS, "owner_required": True, "evidence_required": True},
	"rewards": {"supported_reward_types": SUPPORTED_REWARD_TYPES, "position_required": True, "asset_reference_required": True, "amount_positive": True, "evidence_required": True},
	"governance": {"supported_vote_choices": SUPPORTED_GOVERNANCE_VOTES, "proposal_reference_required": True, "voter_required": True, "evidence_required": True},
	"risk": {"supported_risk_tiers": SUPPORTED_RISK_TIERS, "reviewer_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"control": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": DEFI_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "blockchain": "fintech_blockchain", "crypto": "fintech_crypto", "wallets": "fintech_wallets", "risk": "fintech_risk", "compliance": "fintech_compliance", "regtech": "fintech_regtech", "aml": "fintech_aml", "kyc": "fintech_kyc", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_protocols": True, "enable_positions": True, "enable_actions": True, "enable_yield_strategies": True, "enable_rewards": True, "enable_governance": True, "enable_risk": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_defi_control", "allow_tenant_overrides": True},
}

PROVIDES = ["defi_protocol_workflow", "defi_position_workflow", "defi_action_workflow", "defi_yield_strategy_workflow", "defi_reward_workflow", "defi_governance_workflow", "defi_risk_workflow", "defi_review_workflow", "defi_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_blockchain", "fintech_crypto", "fintech_wallets", "fintech_risk", "fintech_compliance", "fintech_regtech", "fintech_aml", "fintech_kyc"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-defi/dashboard", "component": "DeFiDashboard", "permission": "fintech_defi:view", "nav_group": "Overview"},
	{"name": "protocols", "path": "/fintech-defi/protocols", "component": "DeFiProtocolRegistry", "permission": "fintech_defi:protocols", "nav_group": "Protocols"},
	{"name": "positions", "path": "/fintech-defi/positions", "component": "DeFiPositionConsole", "permission": "fintech_defi:positions", "nav_group": "Portfolio"},
	{"name": "actions", "path": "/fintech-defi/actions", "component": "DeFiActionQueue", "permission": "fintech_defi:actions", "nav_group": "Operations"},
	{"name": "yield_strategies", "path": "/fintech-defi/yield-strategies", "component": "DeFiYieldStrategyWorkbench", "permission": "fintech_defi:yield", "nav_group": "Strategies"},
	{"name": "rewards", "path": "/fintech-defi/rewards", "component": "DeFiRewardLedger", "permission": "fintech_defi:rewards", "nav_group": "Portfolio"},
	{"name": "governance", "path": "/fintech-defi/governance", "component": "DeFiGovernanceConsole", "permission": "fintech_defi:governance", "nav_group": "Governance"},
	{"name": "risk", "path": "/fintech-defi/risk", "component": "DeFiRiskConsole", "permission": "fintech_defi:risk", "nav_group": "Risk"},
	{"name": "reviews", "path": "/fintech-defi/reviews", "component": "DeFiReviewConsole", "permission": "fintech_defi:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/fintech-defi/agents", "component": "DeFiAgentWorkbench", "permission": "fintech_defi:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-defi/settings", "component": "DeFiSettings", "permission": "fintech_defi:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_defi_control",
	"tokens": {"color.primary": "#047857", "color.accent": "#2563EB", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"protocols": {"icon": "network", "status_indicator": "protocol-chip"}, "positions": {"icon": "wallet-cards", "status_indicator": "position-chip"}, "actions": {"icon": "workflow", "status_indicator": "action-chip"}, "yield_strategies": {"icon": "line-chart", "status_indicator": "strategy-chip"}, "rewards": {"icon": "badge-dollar-sign", "status_indicator": "reward-chip"}, "governance": {"icon": "vote", "status_indicator": "vote-chip"}, "risk": {"icon": "shield-alert", "status_indicator": "risk-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": DEFI_EVENT_STREAM, "key": "tenant_id", "events": ["defi_protocol_registered", "defi_position_opened", "defi_action_recorded", "defi_yield_strategy_registered", "defi_reward_recorded", "defi_governance_vote_recorded", "defi_risk_assessment_recorded", "defi_review_recorded", "defi_agent_registered"], "guardrails": ["defi_batch_requires_bytewax", "privileged_defi_agent_action_requires_human_approval"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "defi_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "defi_policy_required", "required_action": "attach_defi_policy"}},
	{"name": "protocol_type_supported", "condition": {"operation": "register_protocol", "protocol_type_supported": False}, "effect": {"decision": "deny", "reason": "protocol_type_not_supported", "required_action": "select_supported_protocol_type"}},
	{"name": "protocol_network_required", "condition": {"operation": "register_protocol", "network_present": False}, "effect": {"decision": "deny", "reason": "protocol_network_required", "required_action": "attach_network_reference"}},
	{"name": "protocol_reference_required", "condition": {"operation": "register_protocol", "protocol_reference_present": False}, "effect": {"decision": "deny", "reason": "protocol_reference_required", "required_action": "attach_protocol_reference"}},
	{"name": "protocol_owner_required", "condition": {"operation": "register_protocol", "owner_present": False}, "effect": {"decision": "deny", "reason": "protocol_owner_required", "required_action": "assign_protocol_owner"}},
	{"name": "protocol_evidence_required", "condition": {"operation": "register_protocol", "evidence_present": False}, "effect": {"decision": "deny", "reason": "protocol_evidence_required", "required_action": "attach_protocol_evidence"}},
	{"name": "protocol_risk_supported", "condition": {"operation": "register_protocol", "risk_tier_supported": False}, "effect": {"decision": "deny", "reason": "risk_tier_not_supported", "required_action": "select_supported_risk_tier"}},
	{"name": "position_protocol_required", "condition": {"operation": "open_position", "protocol_present": False}, "effect": {"decision": "deny", "reason": "protocol_required", "required_action": "select_protocol"}},
	{"name": "position_account_required", "condition": {"operation": "open_position", "account_present": False}, "effect": {"decision": "deny", "reason": "account_reference_required", "required_action": "attach_account_reference"}},
	{"name": "position_asset_pair_required", "condition": {"operation": "open_position", "asset_pair_present": False}, "effect": {"decision": "deny", "reason": "asset_pair_reference_required", "required_action": "attach_asset_pair_reference"}},
	{"name": "position_type_supported", "condition": {"operation": "open_position", "position_type_supported": False}, "effect": {"decision": "deny", "reason": "position_type_not_supported", "required_action": "select_supported_position_type"}},
	{"name": "position_amount_valid", "condition": {"operation": "open_position", "amount_valid": False}, "effect": {"decision": "deny", "reason": "position_amount_invalid", "required_action": "set_positive_amount"}},
	{"name": "position_collateral_valid", "condition": {"operation": "open_position", "collateral_valid": False}, "effect": {"decision": "deny", "reason": "collateral_amount_invalid", "required_action": "set_non_negative_collateral"}},
	{"name": "position_health_factor_valid", "condition": {"operation": "open_position", "health_factor_valid": False}, "effect": {"decision": "deny", "reason": "health_factor_invalid", "required_action": "set_positive_health_factor"}},
	{"name": "position_evidence_required", "condition": {"operation": "open_position", "evidence_present": False}, "effect": {"decision": "deny", "reason": "position_evidence_required", "required_action": "attach_position_evidence"}},
	{"name": "action_protocol_required", "condition": {"operation": "record_action", "protocol_present": False}, "effect": {"decision": "deny", "reason": "protocol_required", "required_action": "select_protocol"}},
	{"name": "action_position_required", "condition": {"operation": "record_action", "position_present": False}, "effect": {"decision": "deny", "reason": "position_required", "required_action": "select_position"}},
	{"name": "action_position_protocol_match", "condition": {"operation": "record_action", "position_protocol_match": False}, "effect": {"decision": "deny", "reason": "position_protocol_mismatch", "required_action": "select_position_for_protocol"}},
	{"name": "action_type_supported", "condition": {"operation": "record_action", "action_type_supported": False}, "effect": {"decision": "deny", "reason": "action_type_not_supported", "required_action": "select_supported_action_type"}},
	{"name": "action_amount_valid", "condition": {"operation": "record_action", "amount_valid": False}, "effect": {"decision": "deny", "reason": "action_amount_invalid", "required_action": "set_positive_amount"}},
	{"name": "action_requester_required", "condition": {"operation": "record_action", "requester_present": False}, "effect": {"decision": "deny", "reason": "action_requester_required", "required_action": "record_requester"}},
	{"name": "action_approval_required", "condition": {"operation": "record_action", "approval_present": False}, "effect": {"decision": "deny", "reason": "action_approval_required", "required_action": "attach_approval_reference"}},
	{"name": "action_evidence_required", "condition": {"operation": "record_action", "evidence_present": False}, "effect": {"decision": "deny", "reason": "action_evidence_required", "required_action": "attach_action_evidence"}},
	{"name": "action_status_supported", "condition": {"operation": "record_action", "status_supported": False}, "effect": {"decision": "deny", "reason": "action_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "strategy_protocol_required", "condition": {"operation": "register_yield_strategy", "protocol_present": False}, "effect": {"decision": "deny", "reason": "protocol_required", "required_action": "select_protocol"}},
	{"name": "strategy_reference_required", "condition": {"operation": "register_yield_strategy", "strategy_reference_present": False}, "effect": {"decision": "deny", "reason": "strategy_reference_required", "required_action": "attach_strategy_reference"}},
	{"name": "strategy_target_apy_valid", "condition": {"operation": "register_yield_strategy", "target_apy_valid": False}, "effect": {"decision": "deny", "reason": "target_apy_invalid", "required_action": "set_non_negative_target_apy"}},
	{"name": "strategy_max_risk_supported", "condition": {"operation": "register_yield_strategy", "max_risk_supported": False}, "effect": {"decision": "deny", "reason": "max_risk_tier_not_supported", "required_action": "select_supported_risk_tier"}},
	{"name": "strategy_owner_required", "condition": {"operation": "register_yield_strategy", "owner_present": False}, "effect": {"decision": "deny", "reason": "strategy_owner_required", "required_action": "assign_strategy_owner"}},
	{"name": "strategy_evidence_required", "condition": {"operation": "register_yield_strategy", "evidence_present": False}, "effect": {"decision": "deny", "reason": "strategy_evidence_required", "required_action": "attach_strategy_evidence"}},
	{"name": "reward_position_required", "condition": {"operation": "record_reward", "position_present": False}, "effect": {"decision": "deny", "reason": "position_required", "required_action": "select_position"}},
	{"name": "reward_type_supported", "condition": {"operation": "record_reward", "reward_type_supported": False}, "effect": {"decision": "deny", "reason": "reward_type_not_supported", "required_action": "select_supported_reward_type"}},
	{"name": "reward_asset_required", "condition": {"operation": "record_reward", "asset_present": False}, "effect": {"decision": "deny", "reason": "reward_asset_required", "required_action": "attach_reward_asset"}},
	{"name": "reward_amount_valid", "condition": {"operation": "record_reward", "amount_valid": False}, "effect": {"decision": "deny", "reason": "reward_amount_invalid", "required_action": "set_positive_reward_amount"}},
	{"name": "reward_evidence_required", "condition": {"operation": "record_reward", "evidence_present": False}, "effect": {"decision": "deny", "reason": "reward_evidence_required", "required_action": "attach_reward_evidence"}},
	{"name": "governance_protocol_required", "condition": {"operation": "record_governance_vote", "protocol_present": False}, "effect": {"decision": "deny", "reason": "protocol_required", "required_action": "select_protocol"}},
	{"name": "governance_proposal_required", "condition": {"operation": "record_governance_vote", "proposal_present": False}, "effect": {"decision": "deny", "reason": "proposal_reference_required", "required_action": "attach_proposal_reference"}},
	{"name": "governance_vote_supported", "condition": {"operation": "record_governance_vote", "vote_supported": False}, "effect": {"decision": "deny", "reason": "vote_choice_not_supported", "required_action": "select_supported_vote_choice"}},
	{"name": "governance_voter_required", "condition": {"operation": "record_governance_vote", "voter_present": False}, "effect": {"decision": "deny", "reason": "voter_required", "required_action": "record_voter"}},
	{"name": "governance_evidence_required", "condition": {"operation": "record_governance_vote", "evidence_present": False}, "effect": {"decision": "deny", "reason": "governance_evidence_required", "required_action": "attach_governance_evidence"}},
	{"name": "risk_reference_required", "condition": {"operation": "record_risk_assessment", "reference_present": False}, "effect": {"decision": "deny", "reason": "risk_reference_required", "required_action": "attach_reference"}},
	{"name": "risk_tier_supported", "condition": {"operation": "record_risk_assessment", "risk_tier_supported": False}, "effect": {"decision": "deny", "reason": "risk_tier_not_supported", "required_action": "select_supported_risk_tier"}},
	{"name": "risk_reviewer_required", "condition": {"operation": "record_risk_assessment", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "risk_reviewer_required", "required_action": "assign_risk_reviewer"}},
	{"name": "risk_evidence_required", "condition": {"operation": "record_risk_assessment", "evidence_present": False}, "effect": {"decision": "deny", "reason": "risk_evidence_required", "required_action": "attach_risk_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "defi_batch_requires_bytewax", "condition": {"operation": "defi_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_defi_batch_to_bytewax"}},
	{"name": "defi_agent_runtime_supported", "condition": {"operation": "register_defi_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "defi_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "defi_agent_role_supported", "condition": {"operation": "register_defi_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "defi_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_defi_agent_action_requires_human_approval", "condition": {"operation": "defi_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-defi/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
