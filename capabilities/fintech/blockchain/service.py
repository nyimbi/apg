"""Executable service layer for APG Blockchain Services."""

from __future__ import annotations

from typing import Any

try:
	from .blockchain_runtime import non_negative_int, normalize_code, present
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CONTRACT_TYPES, SUPPORTED_CUSTODY_MODELS, SUPPORTED_ENVIRONMENTS, SUPPORTED_NETWORK_TYPES, SUPPORTED_NODE_STATUSES, SUPPORTED_ORACLE_FEED_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SETTLEMENT_STATUSES, SUPPORTED_TRANSACTION_TYPES, evaluate_capability_rules, get_capability_contract
	from .models import BlockchainAgent, BlockchainNetwork, BlockchainReview, BlockchainWallet, ChainTransaction, EvidenceAnchor, NodeHealth, OracleFeed, SmartContractDeployment
except ImportError:  # pragma: no cover
	from blockchain_runtime import non_negative_int, normalize_code, present  # type: ignore
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CONTRACT_TYPES, SUPPORTED_CUSTODY_MODELS, SUPPORTED_ENVIRONMENTS, SUPPORTED_NETWORK_TYPES, SUPPORTED_NODE_STATUSES, SUPPORTED_ORACLE_FEED_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SETTLEMENT_STATUSES, SUPPORTED_TRANSACTION_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import BlockchainAgent, BlockchainNetwork, BlockchainReview, BlockchainWallet, ChainTransaction, EvidenceAnchor, NodeHealth, OracleFeed, SmartContractDeployment  # type: ignore


class BlockchainServicesService:
	"""Dependency-light blockchain runtime for generated APG applications."""

	def __init__(self) -> None:
		self.networks: dict[str, BlockchainNetwork] = {}
		self.wallets: dict[str, BlockchainWallet] = {}
		self.contracts: dict[str, SmartContractDeployment] = {}
		self.transactions: dict[str, ChainTransaction] = {}
		self.anchors: dict[str, EvidenceAnchor] = {}
		self.oracles: dict[str, OracleFeed] = {}
		self.nodes: dict[str, NodeHealth] = {}
		self.reviews: dict[str, BlockchainReview] = {}
		self.agents: dict[str, BlockchainAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_network(self, network_id: str, tenant_id: str, network_type: str, environment: str, chain_id: str, rpc_reference: str, owner_id: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		network_type = normalize_code(network_type)
		environment = normalize_code(environment)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_network", "network_type_supported": network_type in SUPPORTED_NETWORK_TYPES, "environment_supported": environment in SUPPORTED_ENVIRONMENTS, "chain_id_present": present(chain_id), "rpc_present": present(rpc_reference), "owner_present": present(owner_id), "evidence_present": present(evidence_reference)})
		item = BlockchainNetwork(network_id, tenant_id, network_type, environment, chain_id, rpc_reference, owner_id, evidence_reference)
		self.networks[network_id] = item
		self._audit(tenant_id, "blockchain_network_registered", network_id)
		return item.to_dict()

	def register_wallet(self, wallet_id: str, tenant_id: str, network_id: str, wallet_reference: str, custody_model: str, key_policy_reference: str, owner_id: str, evidence_reference: str) -> dict[str, Any]:
		network = self._tenant_network_or_none(network_id, tenant_id)
		custody_model = normalize_code(custody_model)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_wallet", "network_present": network is not None, "wallet_present": present(wallet_reference), "custody_model_supported": custody_model in SUPPORTED_CUSTODY_MODELS, "key_policy_present": present(key_policy_reference), "owner_present": present(owner_id), "evidence_present": present(evidence_reference)})
		item = BlockchainWallet(wallet_id, tenant_id, network_id, wallet_reference, custody_model, key_policy_reference, owner_id, evidence_reference)
		self.wallets[wallet_id] = item
		self._audit(tenant_id, "blockchain_wallet_registered", wallet_id)
		return item.to_dict()

	def deploy_contract(self, contract_id: str, tenant_id: str, network_id: str, contract_type: str, artifact_reference: str, owner_id: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		network = self._tenant_network_or_none(network_id, tenant_id)
		contract_type = normalize_code(contract_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "deploy_contract", "network_present": network is not None, "contract_type_supported": contract_type in SUPPORTED_CONTRACT_TYPES, "artifact_present": present(artifact_reference), "owner_present": present(owner_id), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = SmartContractDeployment(contract_id, tenant_id, network_id, contract_type, artifact_reference, owner_id, approval_reference, evidence_reference, "deployed")
		self.contracts[contract_id] = item
		self._audit(tenant_id, "smart_contract_deployed", contract_id)
		return item.to_dict()

	def record_transaction(self, transaction_id: str, tenant_id: str, network_id: str, transaction_hash: str, transaction_type: str, asset_reference: str, amount_minor: int, signer_id: str, evidence_reference: str, settlement_status: str, approval_reference: str = "") -> dict[str, Any]:
		network = self._tenant_network_or_none(network_id, tenant_id)
		transaction_type = normalize_code(transaction_type)
		settlement_status = normalize_code(settlement_status)
		high_value = amount_minor >= 100000000
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_transaction", "network_present": network is not None, "transaction_hash_present": present(transaction_hash), "transaction_type_supported": transaction_type in SUPPORTED_TRANSACTION_TYPES, "asset_present": present(asset_reference), "amount_valid": non_negative_int(amount_minor), "signer_present": present(signer_id), "evidence_present": present(evidence_reference), "settlement_status_supported": settlement_status in SUPPORTED_SETTLEMENT_STATUSES, "high_value": high_value, "approval_present": present(approval_reference)})
		item = ChainTransaction(transaction_id, tenant_id, network_id, transaction_hash, transaction_type, asset_reference, int(amount_minor), signer_id, evidence_reference, settlement_status)
		self.transactions[transaction_id] = item
		self._audit(tenant_id, "chain_transaction_recorded", transaction_id)
		return item.to_dict()

	def anchor_evidence(self, anchor_id: str, tenant_id: str, network_id: str, payload_hash: str, reference_id: str, anchored_at: str, evidence_reference: str) -> dict[str, Any]:
		network = self._tenant_network_or_none(network_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "anchor_evidence", "network_present": network is not None, "payload_hash_present": present(payload_hash), "reference_present": present(reference_id), "anchored_at_present": present(anchored_at), "evidence_present": present(evidence_reference)})
		item = EvidenceAnchor(anchor_id, tenant_id, network_id, payload_hash, reference_id, anchored_at, evidence_reference)
		self.anchors[anchor_id] = item
		self._audit(tenant_id, "evidence_anchor_recorded", anchor_id)
		return item.to_dict()

	def register_oracle_feed(self, oracle_id: str, tenant_id: str, network_id: str, feed_type: str, source_reference: str, owner_id: str, evidence_reference: str) -> dict[str, Any]:
		network = self._tenant_network_or_none(network_id, tenant_id)
		feed_type = normalize_code(feed_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_oracle_feed", "network_present": network is not None, "feed_type_supported": feed_type in SUPPORTED_ORACLE_FEED_TYPES, "source_present": present(source_reference), "owner_present": present(owner_id), "evidence_present": present(evidence_reference)})
		item = OracleFeed(oracle_id, tenant_id, network_id, feed_type, source_reference, owner_id, evidence_reference)
		self.oracles[oracle_id] = item
		self._audit(tenant_id, "oracle_feed_registered", oracle_id)
		return item.to_dict()

	def record_node_health(self, node_id: str, tenant_id: str, network_id: str, endpoint_reference: str, status: str, block_height: int, evidence_reference: str) -> dict[str, Any]:
		network = self._tenant_network_or_none(network_id, tenant_id)
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_node_health", "network_present": network is not None, "endpoint_present": present(endpoint_reference), "node_status_supported": status in SUPPORTED_NODE_STATUSES, "block_height_valid": non_negative_int(block_height), "evidence_present": present(evidence_reference)})
		item = NodeHealth(node_id, tenant_id, network_id, endpoint_reference, status, int(block_height), evidence_reference)
		self.nodes[node_id] = item
		self._audit(tenant_id, "node_health_recorded", node_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = BlockchainReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = item
		self._audit(tenant_id, "blockchain_review_recorded", review_id)
		return item.to_dict()

	def register_blockchain_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_blockchain_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = BlockchainAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[agent_id] = item
		self._audit(tenant_id, "blockchain_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "blockchain_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "blockchain_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.blockchain.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "network_count": self._count(self.networks, tenant_id), "wallet_count": self._count(self.wallets, tenant_id), "contract_count": self._count(self.contracts, tenant_id), "transaction_count": self._count(self.transactions, tenant_id), "anchor_count": self._count(self.anchors, tenant_id), "oracle_count": self._count(self.oracles, tenant_id), "node_count": self._count(self.nodes, tenant_id), "degraded_node_count": sum(1 for item in self.nodes.values() if item.tenant_id == tenant_id and item.status != "healthy"), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_network_or_none(self, item_id: str, tenant_id: str) -> BlockchainNetwork | None:
		item = self.networks.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "blockchain_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "blockchain_policy_denied")


FintechBlockchainService = BlockchainServicesService
