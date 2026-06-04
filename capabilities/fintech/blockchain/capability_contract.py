"""Executable capability contract for APG Blockchain Services."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_blockchain"
CAPABILITY_NAME = "Blockchain Services"
CAPABILITY_VERSION = "1.1.0"
BLOCKCHAIN_EVENT_STREAM = "apg.fintech.blockchain.lifecycle"

SUPPORTED_NETWORK_TYPES = ["ethereum", "polygon", "solana", "bitcoin", "hyperledger_fabric", "private_evm"]
SUPPORTED_ENVIRONMENTS = ["mainnet", "testnet", "sandbox", "consortium", "private"]
SUPPORTED_CUSTODY_MODELS = ["self_custody", "mpc", "hsm", "smart_contract", "custodial"]
SUPPORTED_CONTRACT_TYPES = ["token", "multisig", "settlement", "identity", "oracle", "bridge", "escrow"]
SUPPORTED_TRANSACTION_TYPES = ["transfer", "mint", "burn", "settlement", "contract_call", "anchoring", "bridge"]
SUPPORTED_SETTLEMENT_STATUSES = ["pending", "confirmed", "finalized", "failed", "reversed"]
SUPPORTED_ORACLE_FEED_TYPES = ["price", "identity", "compliance", "fx_rate", "proof_of_reserve", "risk_signal"]
SUPPORTED_NODE_STATUSES = ["healthy", "degraded", "offline", "catching_up"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["network_operator", "contract_reviewer", "transaction_reconciler", "oracle_monitor", "custody_policy_agent", "evidence_anchor_agent"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"networks": {"supported_types": SUPPORTED_NETWORK_TYPES, "supported_environments": SUPPORTED_ENVIRONMENTS, "chain_id_required": True, "rpc_reference_required": True, "owner_required": True, "evidence_required": True},
	"wallets": {"network_required": True, "supported_custody_models": SUPPORTED_CUSTODY_MODELS, "wallet_reference_required": True, "key_policy_required": True, "owner_required": True, "evidence_required": True},
	"contracts": {"network_required": True, "supported_types": SUPPORTED_CONTRACT_TYPES, "artifact_required": True, "owner_required": True, "approval_required": True, "evidence_required": True},
	"transactions": {"network_required": True, "supported_types": SUPPORTED_TRANSACTION_TYPES, "transaction_hash_required": True, "asset_reference_required": True, "amount_non_negative": True, "signer_required": True, "evidence_required": True, "supported_settlement_statuses": SUPPORTED_SETTLEMENT_STATUSES},
	"anchors": {"network_required": True, "payload_hash_required": True, "reference_required": True, "anchored_at_required": True, "evidence_required": True},
	"oracles": {"network_required": True, "supported_feed_types": SUPPORTED_ORACLE_FEED_TYPES, "source_required": True, "owner_required": True, "evidence_required": True},
	"nodes": {"network_required": True, "endpoint_required": True, "supported_statuses": SUPPORTED_NODE_STATUSES, "block_height_non_negative": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "high_value_transaction_requires_approval": True},
	"observability": {"event_stream": BLOCKCHAIN_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "risk": "fintech_risk", "compliance": "fintech_compliance", "regtech": "fintech_regtech", "wallets": "fintech_wallets", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_networks": True, "enable_wallets": True, "enable_contracts": True, "enable_transactions": True, "enable_anchors": True, "enable_oracles": True, "enable_nodes": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_blockchain_control", "allow_tenant_overrides": True},
}

PROVIDES = ["blockchain_network_workflow", "blockchain_wallet_workflow", "smart_contract_workflow", "chain_transaction_workflow", "evidence_anchor_workflow", "oracle_feed_workflow", "node_health_workflow", "blockchain_review_workflow", "blockchain_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_risk", "fintech_compliance", "fintech_regtech", "fintech_wallets"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-blockchain/dashboard", "component": "BlockchainDashboard", "permission": "fintech_blockchain:view", "nav_group": "Overview"},
	{"name": "networks", "path": "/fintech-blockchain/networks", "component": "NetworkConsole", "permission": "fintech_blockchain:networks", "nav_group": "Networks"},
	{"name": "wallets", "path": "/fintech-blockchain/wallets", "component": "WalletRegistry", "permission": "fintech_blockchain:wallets", "nav_group": "Custody"},
	{"name": "contracts", "path": "/fintech-blockchain/contracts", "component": "ContractDeploymentConsole", "permission": "fintech_blockchain:contracts", "nav_group": "Contracts"},
	{"name": "transactions", "path": "/fintech-blockchain/transactions", "component": "TransactionLedger", "permission": "fintech_blockchain:transactions", "nav_group": "Ledger"},
	{"name": "anchors", "path": "/fintech-blockchain/anchors", "component": "EvidenceAnchorConsole", "permission": "fintech_blockchain:anchors", "nav_group": "Evidence"},
	{"name": "oracles", "path": "/fintech-blockchain/oracles", "component": "OracleFeedMonitor", "permission": "fintech_blockchain:oracles", "nav_group": "Data"},
	{"name": "nodes", "path": "/fintech-blockchain/nodes", "component": "NodeHealthConsole", "permission": "fintech_blockchain:nodes", "nav_group": "Operations"},
	{"name": "reviews", "path": "/fintech-blockchain/reviews", "component": "BlockchainReviewConsole", "permission": "fintech_blockchain:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/fintech-blockchain/agents", "component": "BlockchainAgentWorkbench", "permission": "fintech_blockchain:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-blockchain/settings", "component": "BlockchainSettings", "permission": "fintech_blockchain:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_blockchain_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#2563EB", "color.success": "#15803D", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"networks": {"icon": "network", "status_indicator": "network-chip"}, "wallets": {"icon": "wallet-cards", "status_indicator": "custody-chip"}, "contracts": {"icon": "file-code-2", "status_indicator": "contract-chip"}, "transactions": {"icon": "blocks", "status_indicator": "transaction-chip"}, "anchors": {"icon": "fingerprint", "status_indicator": "anchor-chip"}, "oracles": {"icon": "radio-tower", "status_indicator": "oracle-chip"}, "nodes": {"icon": "server-cog", "status_indicator": "node-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": BLOCKCHAIN_EVENT_STREAM, "key": "tenant_id", "events": ["blockchain_network_registered", "blockchain_wallet_registered", "smart_contract_deployed", "chain_transaction_recorded", "evidence_anchor_recorded", "oracle_feed_registered", "node_health_recorded", "blockchain_review_recorded", "blockchain_agent_registered"], "guardrails": ["blockchain_batch_requires_bytewax", "privileged_blockchain_agent_action_requires_human_approval"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "blockchain_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "blockchain_policy_required", "required_action": "attach_blockchain_policy"}},
	{"name": "network_type_supported", "condition": {"operation": "register_network", "network_type_supported": False}, "effect": {"decision": "deny", "reason": "network_type_not_supported", "required_action": "select_supported_network_type"}},
	{"name": "network_environment_supported", "condition": {"operation": "register_network", "environment_supported": False}, "effect": {"decision": "deny", "reason": "network_environment_not_supported", "required_action": "select_supported_environment"}},
	{"name": "network_chain_id_required", "condition": {"operation": "register_network", "chain_id_present": False}, "effect": {"decision": "deny", "reason": "chain_id_required", "required_action": "attach_chain_id"}},
	{"name": "network_rpc_required", "condition": {"operation": "register_network", "rpc_present": False}, "effect": {"decision": "deny", "reason": "rpc_reference_required", "required_action": "attach_rpc_reference"}},
	{"name": "network_owner_required", "condition": {"operation": "register_network", "owner_present": False}, "effect": {"decision": "deny", "reason": "network_owner_required", "required_action": "assign_network_owner"}},
	{"name": "network_evidence_required", "condition": {"operation": "register_network", "evidence_present": False}, "effect": {"decision": "deny", "reason": "network_evidence_required", "required_action": "attach_network_evidence"}},
	{"name": "wallet_network_required", "condition": {"operation": "register_wallet", "network_present": False}, "effect": {"decision": "deny", "reason": "network_required", "required_action": "select_network"}},
	{"name": "wallet_reference_required", "condition": {"operation": "register_wallet", "wallet_present": False}, "effect": {"decision": "deny", "reason": "wallet_reference_required", "required_action": "attach_wallet_reference"}},
	{"name": "custody_model_supported", "condition": {"operation": "register_wallet", "custody_model_supported": False}, "effect": {"decision": "deny", "reason": "custody_model_not_supported", "required_action": "select_supported_custody_model"}},
	{"name": "wallet_key_policy_required", "condition": {"operation": "register_wallet", "key_policy_present": False}, "effect": {"decision": "deny", "reason": "key_policy_required", "required_action": "attach_key_policy"}},
	{"name": "wallet_owner_required", "condition": {"operation": "register_wallet", "owner_present": False}, "effect": {"decision": "deny", "reason": "wallet_owner_required", "required_action": "assign_wallet_owner"}},
	{"name": "wallet_evidence_required", "condition": {"operation": "register_wallet", "evidence_present": False}, "effect": {"decision": "deny", "reason": "wallet_evidence_required", "required_action": "attach_wallet_evidence"}},
	{"name": "contract_network_required", "condition": {"operation": "deploy_contract", "network_present": False}, "effect": {"decision": "deny", "reason": "network_required", "required_action": "select_network"}},
	{"name": "contract_type_supported", "condition": {"operation": "deploy_contract", "contract_type_supported": False}, "effect": {"decision": "deny", "reason": "contract_type_not_supported", "required_action": "select_supported_contract_type"}},
	{"name": "contract_artifact_required", "condition": {"operation": "deploy_contract", "artifact_present": False}, "effect": {"decision": "deny", "reason": "contract_artifact_required", "required_action": "attach_contract_artifact"}},
	{"name": "contract_owner_required", "condition": {"operation": "deploy_contract", "owner_present": False}, "effect": {"decision": "deny", "reason": "contract_owner_required", "required_action": "assign_contract_owner"}},
	{"name": "contract_approval_required", "condition": {"operation": "deploy_contract", "approval_present": False}, "effect": {"decision": "deny", "reason": "contract_approval_required", "required_action": "attach_contract_approval"}},
	{"name": "contract_evidence_required", "condition": {"operation": "deploy_contract", "evidence_present": False}, "effect": {"decision": "deny", "reason": "contract_evidence_required", "required_action": "attach_contract_evidence"}},
	{"name": "transaction_network_required", "condition": {"operation": "record_transaction", "network_present": False}, "effect": {"decision": "deny", "reason": "network_required", "required_action": "select_network"}},
	{"name": "transaction_hash_required", "condition": {"operation": "record_transaction", "transaction_hash_present": False}, "effect": {"decision": "deny", "reason": "transaction_hash_required", "required_action": "attach_transaction_hash"}},
	{"name": "transaction_type_supported", "condition": {"operation": "record_transaction", "transaction_type_supported": False}, "effect": {"decision": "deny", "reason": "transaction_type_not_supported", "required_action": "select_supported_transaction_type"}},
	{"name": "transaction_asset_required", "condition": {"operation": "record_transaction", "asset_present": False}, "effect": {"decision": "deny", "reason": "asset_reference_required", "required_action": "attach_asset_reference"}},
	{"name": "transaction_amount_required", "condition": {"operation": "record_transaction", "amount_valid": False}, "effect": {"decision": "deny", "reason": "transaction_amount_invalid", "required_action": "set_non_negative_amount"}},
	{"name": "transaction_signer_required", "condition": {"operation": "record_transaction", "signer_present": False}, "effect": {"decision": "deny", "reason": "transaction_signer_required", "required_action": "record_signer"}},
	{"name": "transaction_evidence_required", "condition": {"operation": "record_transaction", "evidence_present": False}, "effect": {"decision": "deny", "reason": "transaction_evidence_required", "required_action": "attach_transaction_evidence"}},
	{"name": "settlement_status_supported", "condition": {"operation": "record_transaction", "settlement_status_supported": False}, "effect": {"decision": "deny", "reason": "settlement_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "high_value_transaction_requires_approval", "condition": {"operation": "record_transaction", "high_value": True, "approval_present": False}, "effect": {"decision": "deny", "reason": "transaction_approval_required", "required_action": "attach_transaction_approval"}},
	{"name": "anchor_network_required", "condition": {"operation": "anchor_evidence", "network_present": False}, "effect": {"decision": "deny", "reason": "network_required", "required_action": "select_network"}},
	{"name": "anchor_payload_required", "condition": {"operation": "anchor_evidence", "payload_hash_present": False}, "effect": {"decision": "deny", "reason": "payload_hash_required", "required_action": "attach_payload_hash"}},
	{"name": "anchor_reference_required", "condition": {"operation": "anchor_evidence", "reference_present": False}, "effect": {"decision": "deny", "reason": "anchor_reference_required", "required_action": "attach_reference"}},
	{"name": "anchor_timestamp_required", "condition": {"operation": "anchor_evidence", "anchored_at_present": False}, "effect": {"decision": "deny", "reason": "anchored_at_required", "required_action": "record_anchor_time"}},
	{"name": "anchor_evidence_required", "condition": {"operation": "anchor_evidence", "evidence_present": False}, "effect": {"decision": "deny", "reason": "anchor_evidence_required", "required_action": "attach_anchor_evidence"}},
	{"name": "oracle_network_required", "condition": {"operation": "register_oracle_feed", "network_present": False}, "effect": {"decision": "deny", "reason": "network_required", "required_action": "select_network"}},
	{"name": "oracle_feed_type_supported", "condition": {"operation": "register_oracle_feed", "feed_type_supported": False}, "effect": {"decision": "deny", "reason": "oracle_feed_type_not_supported", "required_action": "select_supported_feed_type"}},
	{"name": "oracle_source_required", "condition": {"operation": "register_oracle_feed", "source_present": False}, "effect": {"decision": "deny", "reason": "oracle_source_required", "required_action": "attach_source"}},
	{"name": "oracle_owner_required", "condition": {"operation": "register_oracle_feed", "owner_present": False}, "effect": {"decision": "deny", "reason": "oracle_owner_required", "required_action": "assign_owner"}},
	{"name": "oracle_evidence_required", "condition": {"operation": "register_oracle_feed", "evidence_present": False}, "effect": {"decision": "deny", "reason": "oracle_evidence_required", "required_action": "attach_evidence"}},
	{"name": "node_network_required", "condition": {"operation": "record_node_health", "network_present": False}, "effect": {"decision": "deny", "reason": "network_required", "required_action": "select_network"}},
	{"name": "node_endpoint_required", "condition": {"operation": "record_node_health", "endpoint_present": False}, "effect": {"decision": "deny", "reason": "node_endpoint_required", "required_action": "attach_endpoint"}},
	{"name": "node_status_supported", "condition": {"operation": "record_node_health", "node_status_supported": False}, "effect": {"decision": "deny", "reason": "node_status_not_supported", "required_action": "select_supported_node_status"}},
	{"name": "node_block_height_valid", "condition": {"operation": "record_node_health", "block_height_valid": False}, "effect": {"decision": "deny", "reason": "block_height_invalid", "required_action": "set_non_negative_block_height"}},
	{"name": "node_evidence_required", "condition": {"operation": "record_node_health", "evidence_present": False}, "effect": {"decision": "deny", "reason": "node_evidence_required", "required_action": "attach_node_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "blockchain_batch_requires_bytewax", "condition": {"operation": "blockchain_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_blockchain_batch_to_bytewax"}},
	{"name": "blockchain_agent_runtime_supported", "condition": {"operation": "register_blockchain_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "blockchain_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "blockchain_agent_role_supported", "condition": {"operation": "register_blockchain_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "blockchain_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_blockchain_agent_action_requires_human_approval", "condition": {"operation": "blockchain_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},

	# Cross-tenant and privilege escalation guards
	{"name": "cross_tenant_blockchain_access_denied", "description": "Blockchain resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Blockchain privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Africa-specific blockchain rules
	{"name": "ke_cma_virtual_asset_licence", "description": "Kenya CMA requires virtual asset service provider licence for blockchain token issuance.", "condition": {"operation": "issue_token", "country": "KE", "cma_vasp_licence_present": False}, "effect": {"decision": "deny", "reason": "ke_cma_vasp_licence_required", "required_action": "obtain_cma_vasp_licence"}},
	{"name": "ke_cbk_digital_currency_approval", "description": "Kenya CBK approval required for digital currency issuance.", "condition": {"operation": "issue_digital_currency", "country": "KE", "cbk_approval_present": False}, "effect": {"decision": "deny", "reason": "ke_cbk_digital_currency_approval_required", "required_action": "obtain_cbk_approval"}},
	{"name": "ng_sec_blockchain_compliance", "description": "Nigeria SEC digital assets framework compliance required for token offerings.", "condition": {"operation": "issue_token", "country": "NG", "ng_sec_compliant": False}, "effect": {"decision": "deny", "reason": "ng_sec_compliance_required", "required_action": "comply_with_ng_sec_framework"}},
	{"name": "mobile_money_blockchain_bridge_kyc", "description": "Mobile money to blockchain bridge transactions require KYC verification.", "condition": {"operation": "mobile_money_to_blockchain", "kyc_verified": False}, "effect": {"decision": "deny", "reason": "mobile_money_blockchain_bridge_kyc_required", "required_action": "verify_kyc_before_bridge"}},
	{"name": "blockchain_aml_screening_required", "description": "Blockchain wallet addresses must be screened against AML watchlists.", "condition": {"operation": "blockchain_transfer", "aml_screened": False}, "effect": {"decision": "deny", "reason": "blockchain_aml_screening_required", "required_action": "screen_blockchain_address"}},
	{"name": "cbdc_cbk_approval_required", "description": "CBDC (e-Shilling) integrations require CBK pilot programme approval.", "condition": {"operation": "integrate_cbdc", "cbk_cbdc_approval_present": False}, "effect": {"decision": "deny", "reason": "cbk_cbdc_approval_required", "required_action": "obtain_cbk_cbdc_approval"}},
]



def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-blockchain/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
