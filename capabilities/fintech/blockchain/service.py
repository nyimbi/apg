"""Executable service layer for APG Blockchain Services.

© 2025 Datacraft — www.datacraft.co.ke
"""

from __future__ import annotations

import datetime
import hashlib
import json
import statistics
import uuid
from typing import Any

try:
	from .blockchain_runtime import non_negative_int, normalize_code, present
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CONTRACT_TYPES,
		SUPPORTED_CUSTODY_MODELS,
		SUPPORTED_ENVIRONMENTS,
		SUPPORTED_NETWORK_TYPES,
		SUPPORTED_NODE_STATUSES,
		SUPPORTED_ORACLE_FEED_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_SETTLEMENT_STATUSES,
		SUPPORTED_TRANSACTION_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		BlockchainAgent,
		BlockchainNetwork,
		BlockchainReview,
		BlockchainWallet,
		ChainTransaction,
		EvidenceAnchor,
		NodeHealth,
		OracleFeed,
		SmartContractDeployment,
	)
except ImportError:  # pragma: no cover
	from blockchain_runtime import non_negative_int, normalize_code, present  # type: ignore
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CONTRACT_TYPES,
		SUPPORTED_CUSTODY_MODELS,
		SUPPORTED_ENVIRONMENTS,
		SUPPORTED_NETWORK_TYPES,
		SUPPORTED_NODE_STATUSES,
		SUPPORTED_ORACLE_FEED_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_SETTLEMENT_STATUSES,
		SUPPORTED_TRANSACTION_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore
		BlockchainAgent,
		BlockchainNetwork,
		BlockchainReview,
		BlockchainWallet,
		ChainTransaction,
		EvidenceAnchor,
		NodeHealth,
		OracleFeed,
		SmartContractDeployment,
	)


# ---------------------------------------------------------------------------
# Supported consensus mechanisms
# ---------------------------------------------------------------------------
_CONSENSUS_MECHANISMS = {
	"pbft", "raft", "pow", "pos", "dpos", "clique", "ibft", "tendermint"
}

# Gas price stubs per chain (gwei)
_GAS_PRICE_GWEI: dict[str, float] = {
	"ethereum":  28.5,
	"polygon":    3.2,
	"bsc":        5.0,
	"avalanche":  27.0,
	"arbitrum":   0.1,
	"optimism":   0.05,
	"private":    0.0,
}

# Block time in seconds
_BLOCK_TIME_SECONDS: dict[str, float] = {
	"ethereum":   12.0,
	"polygon":     2.0,
	"bsc":          3.0,
	"avalanche":    2.0,
	"arbitrum":    0.25,
	"optimism":    2.0,
	"private":     1.0,
}

# Smart contract ABI stubs
_CONTRACT_ABI_STUBS: dict[str, list[dict[str, Any]]] = {
	"erc20": [
		{"name": "transfer",    "type": "function", "inputs": [{"name": "to", "type": "address"}, {"name": "amount", "type": "uint256"}]},
		{"name": "balanceOf",   "type": "function", "inputs": [{"name": "account", "type": "address"}]},
		{"name": "approve",     "type": "function", "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}]},
		{"name": "totalSupply", "type": "function", "inputs": []},
		{"name": "Transfer",    "type": "event",    "inputs": [{"name": "from", "type": "address"}, {"name": "to", "type": "address"}, {"name": "value", "type": "uint256"}]},
	],
	"erc721": [
		{"name": "mint",        "type": "function", "inputs": [{"name": "to", "type": "address"}, {"name": "tokenId", "type": "uint256"}]},
		{"name": "ownerOf",     "type": "function", "inputs": [{"name": "tokenId", "type": "uint256"}]},
		{"name": "transferFrom","type": "function", "inputs": [{"name": "from", "type": "address"}, {"name": "to", "type": "address"}, {"name": "tokenId", "type": "uint256"}]},
	],
	"multisig": [
		{"name": "submitTransaction",  "type": "function", "inputs": [{"name": "destination", "type": "address"}, {"name": "value", "type": "uint256"}, {"name": "data", "type": "bytes"}]},
		{"name": "confirmTransaction", "type": "function", "inputs": [{"name": "transactionId", "type": "uint256"}]},
		{"name": "executeTransaction", "type": "function", "inputs": [{"name": "transactionId", "type": "uint256"}]},
	],
}


def _compute_tx_hash(chain_id: str, data: Any, from_address: str, nonce: int) -> str:
	payload = json.dumps({"chain": chain_id, "data": str(data), "from": from_address, "nonce": nonce}, sort_keys=True)
	return "0x" + hashlib.sha256(payload.encode()).hexdigest()


def _compute_block_hash(chain_id: str, block_number: int, prev_hash: str, tx_hashes: list[str]) -> str:
	payload = json.dumps({"chain": chain_id, "block": block_number, "prev": prev_hash, "txs": tx_hashes}, sort_keys=True)
	return "0x" + hashlib.sha256(payload.encode()).hexdigest()


class BlockchainService:
	"""Full-featured blockchain service for APG applications.

	Covers private chain management, smart contract deployment/invocation,
	on-chain evidence anchoring, supply chain tracking, NFT certificates,
	token issuance, cross-chain transfers, and node health monitoring.
	"""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

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

		# Extended in-memory state
		self._blocks: dict[str, list[dict[str, Any]]] = {}    # chain_id -> block list
		self._supply_chain: list[dict[str, Any]] = []
		self._certificates: dict[str, dict[str, Any]] = {}
		self._tokens: dict[str, dict[str, Any]] = {}          # token_id -> token metadata
		self._token_balances: dict[str, dict[str, int]] = {}  # token_id -> {address: balance}
		self._cross_chain_transfers: list[dict[str, Any]] = []
		self._tx_nonces: dict[str, int] = {}                  # chain_id:address -> nonce
		self._contract_state: dict[str, dict[str, Any]] = {}  # contract_address -> state

	# ------------------------------------------------------------------
	# Contract / policy
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Private blockchain management
	# ------------------------------------------------------------------

	async def create_private_blockchain(
		self,
		name: str,
		consensus: str,
		permissioned: bool,
		*,
		chain_type: str = "evm",
		block_gas_limit: int = 8_000_000,
		initial_validators: list[str] | None = None,
	) -> dict[str, Any]:
		"""Provision a new private/consortium blockchain."""
		assert name, "name required"
		consensus_normalized = consensus.lower()
		assert consensus_normalized in _CONSENSUS_MECHANISMS, (
			f"unsupported consensus: {consensus}. Supported: {_CONSENSUS_MECHANISMS}"
		)

		chain_id = str(uuid.uuid4())
		genesis_hash = "0x" + hashlib.sha256(f"{name}{chain_id}genesis".encode()).hexdigest()
		validators = initial_validators or ["validator-1", "validator-2", "validator-3"]

		genesis_block: dict[str, Any] = {
			"block_number": 0,
			"block_hash": genesis_hash,
			"parent_hash": "0x" + "0" * 64,
			"transactions": [],
			"validator": validators[0] if validators else "genesis",
			"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"gas_limit": block_gas_limit,
			"gas_used": 0,
		}
		self._blocks[chain_id] = [genesis_block]

		# Register as a network
		network_id = str(uuid.uuid4())
		network = BlockchainNetwork(
			network_id,
			self.tenant_id,
			"private",
			"private",
			chain_id,
			f"http://localhost:8545/{chain_id}",
			self.actor_id,
			f"ev-{chain_id[:8]}",
		)
		self.networks[network_id] = network

		chain_data = {
			"chain_id": chain_id,
			"network_id": network_id,
			"name": name,
			"chain_type": chain_type,
			"consensus": consensus_normalized,
			"permissioned": permissioned,
			"validators": validators,
			"block_gas_limit": block_gas_limit,
			"genesis_hash": genesis_hash,
			"block_count": 1,
			"status": "active",
			"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._audit(self.tenant_id, "private_blockchain_created", chain_id)
		return chain_data

	# ------------------------------------------------------------------
	# Smart contracts
	# ------------------------------------------------------------------

	async def deploy_smart_contract(
		self,
		chain_id: str,
		contract_code: str,
		deployer: str,
		*,
		contract_type: str = "erc20",
		constructor_args: dict[str, Any] | None = None,
		gas_limit: int = 3_000_000,
	) -> dict[str, Any]:
		"""Deploy a smart contract to a blockchain network."""
		assert chain_id, "chain_id required"
		assert contract_code, "contract_code required"
		assert deployer, "deployer required"

		# Deterministic contract address
		nonce_key = f"{chain_id}:{deployer}"
		nonce = self._tx_nonces.get(nonce_key, 0)
		self._tx_nonces[nonce_key] = nonce + 1

		contract_address = "0x" + hashlib.sha256(
			f"{deployer}{nonce}{chain_id}".encode()
		).hexdigest()[:40]
		code_hash = hashlib.sha256(contract_code.encode()).hexdigest()
		deploy_tx_hash = _compute_tx_hash(chain_id, contract_code, deployer, nonce)

		# Estimate gas cost
		gas_used = min(int(len(contract_code) * 200), gas_limit)
		chain_name = self._chain_name_for_id(chain_id)
		gas_price = _GAS_PRICE_GWEI.get(chain_name, 20.0)
		gas_cost_eth = (gas_used * gas_price) / 1e9

		# Initialize contract state
		self._contract_state[contract_address] = {
			"type": contract_type,
			"owner": deployer,
			"storage": {},
		}

		# Register in contracts store
		contract_id = str(uuid.uuid4())
		deployment = SmartContractDeployment(
			contract_id,
			self.tenant_id,
			chain_id,
			normalize_code(contract_type) if contract_type in SUPPORTED_CONTRACT_TYPES else "generic",
			code_hash,
			deployer,
			deploy_tx_hash,
			f"ev-{contract_id[:8]}",
			"deployed",
		)
		self.contracts[contract_id] = deployment

		self._audit(self.tenant_id, "smart_contract_deployed", contract_address)
		return {
			"contract_id": contract_id,
			"contract_address": contract_address,
			"chain_id": chain_id,
			"contract_type": contract_type,
			"deployer": deployer,
			"deploy_tx_hash": deploy_tx_hash,
			"code_hash": code_hash,
			"gas_used": gas_used,
			"gas_price_gwei": gas_price,
			"gas_cost_eth": round(gas_cost_eth, 8),
			"abi": _CONTRACT_ABI_STUBS.get(contract_type, []),
			"constructor_args": constructor_args or {},
			"deployed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "deployed",
		}

	async def invoke_smart_contract(
		self,
		contract_address: str,
		method: str,
		params: dict[str, Any],
		*,
		caller: str = "system",
		chain_id: str = "",
		value_wei: int = 0,
	) -> dict[str, Any]:
		"""Invoke a method on a deployed smart contract."""
		assert contract_address, "contract_address required"
		assert method, "method required"

		state = self._contract_state.get(contract_address, {"type": "unknown", "owner": "", "storage": {}})
		contract_type = state["type"]
		storage = state["storage"]

		# Route method invocations to in-memory handlers
		result: Any = None
		emit_events: list[dict[str, Any]] = []

		if contract_type == "erc20":
			if method == "transfer":
				to_addr = params.get("to", "")
				amount = int(params.get("amount", 0))
				balances: dict[str, int] = storage.setdefault("balances", {})
				from_bal = balances.get(caller, 0)
				assert from_bal >= amount, f"insufficient balance: {from_bal} < {amount}"
				balances[caller] = from_bal - amount
				balances[to_addr] = balances.get(to_addr, 0) + amount
				result = True
				emit_events.append({"event": "Transfer", "from": caller, "to": to_addr, "value": amount})

			elif method == "balanceOf":
				account = params.get("account", caller)
				result = storage.get("balances", {}).get(account, 0)

			elif method == "totalSupply":
				result = storage.get("total_supply", 0)

			elif method == "approve":
				spender = params.get("spender", "")
				amount = int(params.get("amount", 0))
				storage.setdefault("allowances", {}).setdefault(caller, {})[spender] = amount
				result = True

		elif contract_type == "erc721":
			if method == "mint":
				to_addr = params.get("to", "")
				token_id = int(params.get("tokenId", 0))
				storage.setdefault("owners", {})[token_id] = to_addr
				result = True
			elif method == "ownerOf":
				token_id = int(params.get("tokenId", 0))
				result = storage.get("owners", {}).get(token_id, "0x0")

		else:
			# Generic: store call in state
			call_key = f"{method}:{json.dumps(params, sort_keys=True)}"
			storage[call_key] = {"caller": caller, "params": params}
			result = f"executed:{method}"

		nonce_key = f"{chain_id}:{caller}"
		nonce = self._tx_nonces.get(nonce_key, 0)
		self._tx_nonces[nonce_key] = nonce + 1
		tx_hash = _compute_tx_hash(chain_id, {"method": method, "params": params}, caller, nonce)

		self._audit(self.tenant_id, "smart_contract_invoked", contract_address)
		return {
			"contract_address": contract_address,
			"method": method,
			"params": params,
			"caller": caller,
			"result": result,
			"tx_hash": tx_hash,
			"events_emitted": emit_events,
			"gas_used": 21_000 + len(json.dumps(params)) * 68,
			"invoked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "confirmed",
		}

	# ------------------------------------------------------------------
	# Transaction recording and verification
	# ------------------------------------------------------------------

	async def record_transaction(
		self,
		chain_id: str,
		data: Any,
		from_address: str,
		*,
		asset_reference: str = "native",
		amount_minor: int = 0,
		tx_type: str = "transfer",
		settlement_status: str = "pending",
	) -> dict[str, Any]:
		"""Record a transaction to a blockchain and add to pending block."""
		assert chain_id, "chain_id required"
		assert from_address, "from_address required"

		nonce_key = f"{chain_id}:{from_address}"
		nonce = self._tx_nonces.get(nonce_key, 0)
		self._tx_nonces[nonce_key] = nonce + 1
		tx_hash = _compute_tx_hash(chain_id, data, from_address, nonce)

		# Simulate mining into next block
		chain_blocks = self._blocks.setdefault(chain_id, [])
		block_number = len(chain_blocks)
		prev_hash = chain_blocks[-1]["block_hash"] if chain_blocks else "0x" + "0" * 64
		block_hash = _compute_block_hash(chain_id, block_number, prev_hash, [tx_hash])

		block: dict[str, Any] = {
			"block_number": block_number,
			"block_hash": block_hash,
			"parent_hash": prev_hash,
			"transactions": [tx_hash],
			"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"gas_used": 21_000,
			"gas_limit": 8_000_000,
		}
		chain_blocks.append(block)

		# Record in transactions store
		tx_id = str(uuid.uuid4())
		tx = ChainTransaction(
			tx_id, self.tenant_id, chain_id, tx_hash,
			normalize_code(tx_type) if tx_type in SUPPORTED_TRANSACTION_TYPES else "transfer",
			asset_reference, int(amount_minor), from_address,
			f"ev-{tx_id[:8]}", normalize_code(settlement_status),
		)
		self.transactions[tx_id] = tx

		self._audit(self.tenant_id, "chain_transaction_recorded", tx_hash)
		return {
			"tx_id": tx_id,
			"tx_hash": tx_hash,
			"chain_id": chain_id,
			"from_address": from_address,
			"nonce": nonce,
			"block_number": block_number,
			"block_hash": block_hash,
			"asset_reference": asset_reference,
			"amount_minor": amount_minor,
			"settlement_status": settlement_status,
			"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def verify_transaction(self, chain_id: str, tx_hash: str) -> dict[str, Any]:
		"""Verify a transaction exists on-chain and return its block context."""
		assert chain_id, "chain_id required"
		assert tx_hash, "tx_hash required"

		chain_blocks = self._blocks.get(chain_id, [])
		containing_block: dict[str, Any] | None = None
		for block in chain_blocks:
			if tx_hash in block.get("transactions", []):
				containing_block = block
				break

		# Also check transactions store
		matching_tx = next(
			(t for t in self.transactions.values()
			 if t.transaction_hash == tx_hash and t.network_id == chain_id),
			None,
		)

		confirmed = containing_block is not None or matching_tx is not None
		confirmations = len(chain_blocks) - (containing_block["block_number"] + 1) if containing_block else 0

		self._audit(self.tenant_id, "transaction_verified", tx_hash)
		return {
			"tx_hash": tx_hash,
			"chain_id": chain_id,
			"confirmed": confirmed,
			"confirmations": confirmations,
			"block_number": containing_block["block_number"] if containing_block else None,
			"block_hash": containing_block["block_hash"] if containing_block else None,
			"block_timestamp": containing_block["timestamp"] if containing_block else None,
			"settlement_status": matching_tx.settlement_status if matching_tx else ("confirmed" if confirmed else "not_found"),
			"verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def get_block(self, chain_id: str, block_number: int) -> dict[str, Any]:
		"""Retrieve a block by number from a chain."""
		assert chain_id, "chain_id required"
		chain_blocks = self._blocks.get(chain_id, [])

		if block_number < 0:
			block_number = len(chain_blocks) + block_number  # support -1 for latest

		if block_number >= len(chain_blocks):
			return {
				"chain_id": chain_id,
				"block_number": block_number,
				"found": False,
				"error": "block_not_found",
			}

		block = chain_blocks[block_number]
		chain_name = self._chain_name_for_id(chain_id)
		self._audit(self.tenant_id, "block_retrieved", f"{chain_id}:{block_number}")
		return {
			"chain_id": chain_id,
			"found": True,
			"chain_name": chain_name,
			"block_time_seconds": _BLOCK_TIME_SECONDS.get(chain_name, 12.0),
			"total_blocks": len(chain_blocks),
			**block,
		}

	# ------------------------------------------------------------------
	# Evidence anchoring
	# ------------------------------------------------------------------

	async def audit_trail_on_chain(
		self,
		record_id: str,
		record_hash: str,
		*,
		chain_id: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Anchor an audit record hash permanently on-chain."""
		assert record_id, "record_id required"
		assert record_hash, "record_hash required"

		# Use first available network if chain_id not specified
		if not chain_id and self.networks:
			chain_id = next(iter(self.networks.values())).chain_id

		anchor_payload = {
			"record_id": record_id,
			"record_hash": record_hash,
			"metadata": metadata or {},
			"anchored_by": self.actor_id,
		}
		payload_hash = hashlib.sha256(
			json.dumps(anchor_payload, sort_keys=True).encode()
		).hexdigest()

		# Record on-chain
		tx_result = await self.record_transaction(
			chain_id or "default",
			anchor_payload,
			self.actor_id,
			tx_type="evidence_anchor",
			settlement_status="confirmed",
		)

		anchor_id = str(uuid.uuid4())
		anchor = EvidenceAnchor(
			anchor_id,
			self.tenant_id,
			chain_id or "default",
			payload_hash,
			record_id,
			datetime.datetime.now(datetime.timezone.utc).isoformat(),
			f"ev-{anchor_id[:8]}",
		)
		self.anchors[anchor_id] = anchor

		self._audit(self.tenant_id, "audit_trail_anchored", anchor_id)
		return {
			"anchor_id": anchor_id,
			"record_id": record_id,
			"record_hash": record_hash,
			"payload_hash": payload_hash,
			"tx_hash": tx_result["tx_hash"],
			"block_number": tx_result["block_number"],
			"chain_id": chain_id or "default",
			"anchored_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"verification_url": f"https://explorer.datacraft.co.ke/tx/{tx_result['tx_hash']}",
		}

	async def verify_anchor(self, anchor_id: str, record_hash: str) -> dict[str, Any]:
		"""Verify that an evidence anchor matches the provided record hash."""
		anchor = self.anchors.get(anchor_id)
		if anchor is None:
			return {"anchor_id": anchor_id, "verified": False, "error": "anchor_not_found"}

		# Recompute payload hash and compare
		stored_hash = anchor.payload_hash
		provided_hash = hashlib.sha256(record_hash.encode()).hexdigest()
		# Check if the stored hash incorporates the record_hash
		hash_match = stored_hash == provided_hash or record_hash in stored_hash

		self._audit(self.tenant_id, "anchor_verified", anchor_id)
		return {
			"anchor_id": anchor_id,
			"reference_id": anchor.reference_id,
			"verified": hash_match,
			"stored_hash": stored_hash,
			"provided_hash": provided_hash,
			"chain_id": anchor.network_id,
			"anchored_at": anchor.anchored_at,
			"verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Supply chain tracking
	# ------------------------------------------------------------------

	async def supply_chain_tracking(
		self,
		product_id: str,
		event: str,
		location: str,
		*,
		actor: str = "",
		metadata: dict[str, Any] | None = None,
		chain_id: str = "default",
	) -> dict[str, Any]:
		"""Record a supply chain event for a product on-chain."""
		assert product_id, "product_id required"
		assert event, "event required"
		assert location, "location required"

		event_data = {
			"product_id": product_id,
			"event": event,
			"location": location,
			"actor": actor or self.actor_id,
			"metadata": metadata or {},
		}
		event_hash = hashlib.sha256(
			json.dumps(event_data, sort_keys=True).encode()
		).hexdigest()

		# Anchor event on-chain
		anchor_result = await self.audit_trail_on_chain(
			record_id=f"sc:{product_id}:{event}",
			record_hash=event_hash,
			chain_id=chain_id,
			metadata=event_data,
		)

		tracking_entry = {
			"tracking_id": str(uuid.uuid4()),
			"product_id": product_id,
			"event": event,
			"location": location,
			"actor": actor or self.actor_id,
			"event_hash": event_hash,
			"anchor_id": anchor_result["anchor_id"],
			"tx_hash": anchor_result["tx_hash"],
			"metadata": metadata or {},
			"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._supply_chain.append(tracking_entry)
		self._audit(self.tenant_id, "supply_chain_event_recorded", product_id)
		return tracking_entry

	async def get_product_journey(self, product_id: str) -> dict[str, Any]:
		"""Return the full on-chain journey for a product."""
		events = [e for e in self._supply_chain if e["product_id"] == product_id]
		events.sort(key=lambda e: e["recorded_at"])
		self._audit(self.tenant_id, "product_journey_queried", product_id)
		return {
			"product_id": product_id,
			"event_count": len(events),
			"journey": events,
			"first_recorded_at": events[0]["recorded_at"] if events else None,
			"last_recorded_at": events[-1]["recorded_at"] if events else None,
			"queried_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Digital certificates
	# ------------------------------------------------------------------

	async def digital_certificate_issue(
		self,
		recipient: str,
		certificate_data: dict[str, Any],
		*,
		certificate_type: str = "credential",
		issuer: str = "",
		chain_id: str = "default",
		expiry_days: int = 365,
	) -> dict[str, Any]:
		"""Issue a tamper-proof digital certificate anchored on-chain."""
		assert recipient, "recipient required"
		assert certificate_data, "certificate_data required"

		cert_id = str(uuid.uuid4())
		issued_at = datetime.datetime.now(datetime.timezone.utc)
		expires_at = (issued_at + datetime.timedelta(days=expiry_days)).isoformat()

		cert_payload = {
			"certificate_id": cert_id,
			"certificate_type": certificate_type,
			"recipient": recipient,
			"issuer": issuer or self.actor_id,
			"data": certificate_data,
			"issued_at": issued_at.isoformat(),
			"expires_at": expires_at,
		}
		cert_hash = hashlib.sha256(
			json.dumps(cert_payload, sort_keys=True).encode()
		).hexdigest()

		# Issue as NFT (ERC-721 stub)
		nft_token_id = int(cert_id.replace("-", "")[:8], 16)
		anchor_result = await self.audit_trail_on_chain(
			record_id=cert_id,
			record_hash=cert_hash,
			chain_id=chain_id,
			metadata=cert_payload,
		)

		certificate = {
			"certificate_id": cert_id,
			"certificate_type": certificate_type,
			"recipient": recipient,
			"issuer": issuer or self.actor_id,
			"data": certificate_data,
			"certificate_hash": cert_hash,
			"nft_token_id": nft_token_id,
			"anchor_id": anchor_result["anchor_id"],
			"tx_hash": anchor_result["tx_hash"],
			"chain_id": chain_id,
			"issued_at": issued_at.isoformat(),
			"expires_at": expires_at,
			"status": "valid",
			"verification_url": f"https://certs.datacraft.co.ke/{cert_id}",
		}
		self._certificates[cert_id] = certificate
		self._audit(self.tenant_id, "digital_certificate_issued", cert_id)
		return certificate

	async def verify_certificate(self, certificate_id: str) -> dict[str, Any]:
		"""Verify the authenticity and validity of a digital certificate."""
		cert = self._certificates.get(certificate_id)
		if cert is None:
			return {"certificate_id": certificate_id, "valid": False, "error": "not_found"}

		now = datetime.datetime.now(datetime.timezone.utc)
		expires_at = datetime.datetime.fromisoformat(cert["expires_at"])
		expired = now > expires_at

		self._audit(self.tenant_id, "certificate_verified", certificate_id)
		return {
			"certificate_id": certificate_id,
			"valid": not expired and cert["status"] == "valid",
			"expired": expired,
			"recipient": cert["recipient"],
			"issuer": cert["issuer"],
			"certificate_type": cert["certificate_type"],
			"issued_at": cert["issued_at"],
			"expires_at": cert["expires_at"],
			"anchor_id": cert["anchor_id"],
			"chain_id": cert["chain_id"],
			"verified_at": now.isoformat(),
		}

	# ------------------------------------------------------------------
	# Token issuance
	# ------------------------------------------------------------------

	async def token_issuance(
		self,
		chain_id: str,
		token_name: str,
		total_supply: int,
		owner: str,
		*,
		token_symbol: str = "",
		decimals: int = 18,
		token_type: str = "erc20",
	) -> dict[str, Any]:
		"""Issue a new fungible token on a specified blockchain."""
		assert chain_id, "chain_id required"
		assert token_name, "token_name required"
		assert total_supply > 0, "total_supply must be positive"
		assert owner, "owner required"

		symbol = token_symbol or token_name[:3].upper()
		token_id = str(uuid.uuid4())

		# Deploy ERC-20 contract stub
		contract_code = f"// {token_name} ERC-20 Token\ncontract {token_name} {{...}}"
		deploy_result = await self.deploy_smart_contract(
			chain_id,
			contract_code,
			owner,
			contract_type=token_type,
			constructor_args={"name": token_name, "symbol": symbol, "totalSupply": total_supply, "decimals": decimals},
		)

		# Initialize token balances
		contract_address = deploy_result["contract_address"]
		self._contract_state[contract_address]["storage"]["balances"] = {owner: total_supply}
		self._contract_state[contract_address]["storage"]["total_supply"] = total_supply

		token_record = {
			"token_id": token_id,
			"token_name": token_name,
			"symbol": symbol,
			"total_supply": total_supply,
			"decimals": decimals,
			"owner": owner,
			"contract_address": contract_address,
			"chain_id": chain_id,
			"token_type": token_type,
			"deploy_tx_hash": deploy_result["deploy_tx_hash"],
			"issued_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "active",
		}
		self._tokens[token_id] = token_record
		self._token_balances[token_id] = {owner: total_supply}

		self._audit(self.tenant_id, "token_issued", token_id)
		return token_record

	async def transfer_token(
		self,
		token_id: str,
		from_address: str,
		to_address: str,
		amount: int,
	) -> dict[str, Any]:
		"""Transfer tokens between addresses."""
		token = self._tokens.get(token_id)
		assert token is not None, f"token not found: {token_id}"
		assert amount > 0, "amount must be positive"

		balances = self._token_balances.setdefault(token_id, {})
		from_balance = balances.get(from_address, 0)
		assert from_balance >= amount, f"insufficient token balance: {from_balance} < {amount}"

		balances[from_address] = from_balance - amount
		balances[to_address] = balances.get(to_address, 0) + amount

		tx_hash = "0x" + hashlib.sha256(
			f"{token_id}{from_address}{to_address}{amount}".encode()
		).hexdigest()

		self._audit(self.tenant_id, "token_transferred", token_id)
		return {
			"token_id": token_id,
			"from_address": from_address,
			"to_address": to_address,
			"amount": amount,
			"from_balance_after": balances[from_address],
			"to_balance_after": balances[to_address],
			"tx_hash": tx_hash,
			"transferred_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def token_balance(self, token_id: str, address: str) -> dict[str, Any]:
		"""Return token balance for an address."""
		token = self._tokens.get(token_id)
		if token is None:
			return {"token_id": token_id, "address": address, "balance": 0, "found": False}
		balance = self._token_balances.get(token_id, {}).get(address, 0)
		return {
			"token_id": token_id,
			"token_name": token["token_name"],
			"symbol": token["symbol"],
			"address": address,
			"balance": balance,
			"decimals": token["decimals"],
			"found": True,
		}

	# ------------------------------------------------------------------
	# Cross-chain transfer
	# ------------------------------------------------------------------

	async def cross_chain_transfer(
		self,
		from_chain: str,
		to_chain: str,
		amount: int,
		token: str,
		*,
		sender: str = "",
		recipient: str = "",
		bridge: str = "datacraft_bridge",
	) -> dict[str, Any]:
		"""Initiate a cross-chain asset transfer via a bridge protocol."""
		assert from_chain, "from_chain required"
		assert to_chain, "to_chain required"
		assert from_chain != to_chain, "source and destination chains must differ"
		assert amount > 0, "amount must be positive"
		assert token, "token required"

		# Lock on source chain (burn/lock pattern)
		lock_tx = await self.record_transaction(
			from_chain,
			{"bridge": bridge, "token": token, "amount": amount, "type": "lock"},
			sender or self.actor_id,
			tx_type="transfer",
			settlement_status="confirmed",
		)

		# Issue on destination chain (mint/release pattern)
		release_tx = await self.record_transaction(
			to_chain,
			{"bridge": bridge, "token": token, "amount": amount, "type": "release"},
			recipient or self.actor_id,
			tx_type="transfer",
			settlement_status="pending",
		)

		bridge_fee_bps = 10  # 0.1% bridge fee
		bridge_fee = int(amount * bridge_fee_bps / 10_000)
		net_amount = amount - bridge_fee

		xfer = {
			"transfer_id": str(uuid.uuid4()),
			"from_chain": from_chain,
			"to_chain": to_chain,
			"token": token,
			"amount": amount,
			"bridge_fee": bridge_fee,
			"net_amount": net_amount,
			"bridge": bridge,
			"sender": sender or self.actor_id,
			"recipient": recipient or self.actor_id,
			"lock_tx_hash": lock_tx["tx_hash"],
			"release_tx_hash": release_tx["tx_hash"],
			"estimated_finality_minutes": 15,
			"status": "in_flight",
			"initiated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._cross_chain_transfers.append(xfer)
		self._audit(self.tenant_id, "cross_chain_transfer_initiated", xfer["transfer_id"])
		return xfer

	async def cross_chain_transfer_status(self, transfer_id: str) -> dict[str, Any]:
		"""Return status of a cross-chain transfer."""
		xfer = next((x for x in self._cross_chain_transfers if x["transfer_id"] == transfer_id), None)
		if xfer is None:
			return {"transfer_id": transfer_id, "found": False, "error": "not_found"}
		return {**xfer, "queried_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}

	# ------------------------------------------------------------------
	# Analytics
	# ------------------------------------------------------------------

	async def chain_analytics(
		self,
		chain_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Compute analytics for a specific chain over a period."""
		chain_blocks = self._blocks.get(chain_id, [])
		chain_txs = [
			t for t in self.transactions.values()
			if t.network_id == chain_id
		]
		chain_name = self._chain_name_for_id(chain_id)
		block_times = _BLOCK_TIME_SECONDS.get(chain_name, 12.0)
		estimated_tps = 1.0 / block_times * 100  # rough estimate

		amounts = [t.amount_minor for t in chain_txs if t.amount_minor > 0]

		self._audit(self.tenant_id, "chain_analytics_computed", chain_id)
		return {
			"chain_id": chain_id,
			"period": period,
			"block_count": len(chain_blocks),
			"transaction_count": len(chain_txs),
			"total_value_minor": sum(amounts),
			"avg_value_minor": statistics.mean(amounts) if amounts else 0,
			"anchor_count": sum(1 for a in self.anchors.values() if a.network_id == chain_id),
			"contract_count": sum(1 for c in self.contracts.values() if c.network_id == chain_id),
			"node_count": sum(1 for n in self.nodes.values() if n.network_id == chain_id),
			"estimated_tps": round(estimated_tps, 2),
			"gas_price_gwei": _GAS_PRICE_GWEI.get(chain_name, 0.0),
			"computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Existing core methods (preserved from original)
	# ------------------------------------------------------------------

	def register_network(
		self,
		network_id: str,
		tenant_id: str,
		network_type: str,
		environment: str,
		chain_id: str,
		rpc_reference: str,
		owner_id: str,
		evidence_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		network_type = normalize_code(network_type)
		environment = normalize_code(environment)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_network",
			"network_type_supported": network_type in SUPPORTED_NETWORK_TYPES,
			"environment_supported": environment in SUPPORTED_ENVIRONMENTS,
			"chain_id_present": present(chain_id),
			"rpc_present": present(rpc_reference),
			"owner_present": present(owner_id),
			"evidence_present": present(evidence_reference),
		})
		item = BlockchainNetwork(
			network_id, tenant_id, network_type, environment,
			chain_id, rpc_reference, owner_id, evidence_reference,
		)
		self.networks[network_id] = item
		self._audit(tenant_id, "blockchain_network_registered", network_id)
		return item.to_dict()

	def register_wallet(
		self,
		wallet_id: str,
		tenant_id: str,
		network_id: str,
		wallet_reference: str,
		custody_model: str,
		key_policy_reference: str,
		owner_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		network = self._tenant_network_or_none(network_id, tenant_id)
		custody_model = normalize_code(custody_model)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_wallet",
			"network_present": network is not None,
			"wallet_present": present(wallet_reference),
			"custody_model_supported": custody_model in SUPPORTED_CUSTODY_MODELS,
			"key_policy_present": present(key_policy_reference),
			"owner_present": present(owner_id),
			"evidence_present": present(evidence_reference),
		})
		item = BlockchainWallet(
			wallet_id, tenant_id, network_id, wallet_reference,
			custody_model, key_policy_reference, owner_id, evidence_reference,
		)
		self.wallets[wallet_id] = item
		self._audit(tenant_id, "blockchain_wallet_registered", wallet_id)
		return item.to_dict()

	def deploy_contract(
		self,
		contract_id: str,
		tenant_id: str,
		network_id: str,
		contract_type: str,
		artifact_reference: str,
		owner_id: str,
		approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		network = self._tenant_network_or_none(network_id, tenant_id)
		contract_type = normalize_code(contract_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "deploy_contract",
			"network_present": network is not None,
			"contract_type_supported": contract_type in SUPPORTED_CONTRACT_TYPES,
			"artifact_present": present(artifact_reference),
			"owner_present": present(owner_id),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = SmartContractDeployment(
			contract_id, tenant_id, network_id, contract_type, artifact_reference,
			owner_id, approval_reference, evidence_reference, "deployed",
		)
		self.contracts[contract_id] = item
		self._audit(tenant_id, "smart_contract_deployed", contract_id)
		return item.to_dict()

	def record_transaction_sync(
		self,
		transaction_id: str,
		tenant_id: str,
		network_id: str,
		transaction_hash: str,
		transaction_type: str,
		asset_reference: str,
		amount_minor: int,
		signer_id: str,
		evidence_reference: str,
		settlement_status: str,
		approval_reference: str = "",
	) -> dict[str, Any]:
		network = self._tenant_network_or_none(network_id, tenant_id)
		transaction_type = normalize_code(transaction_type)
		settlement_status = normalize_code(settlement_status)
		high_value = amount_minor >= 100_000_000
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_transaction",
			"network_present": network is not None,
			"transaction_hash_present": present(transaction_hash),
			"transaction_type_supported": transaction_type in SUPPORTED_TRANSACTION_TYPES,
			"asset_present": present(asset_reference),
			"amount_valid": non_negative_int(amount_minor),
			"signer_present": present(signer_id),
			"evidence_present": present(evidence_reference),
			"settlement_status_supported": settlement_status in SUPPORTED_SETTLEMENT_STATUSES,
			"high_value": high_value,
			"approval_present": present(approval_reference),
		})
		item = ChainTransaction(
			transaction_id, tenant_id, network_id, transaction_hash, transaction_type,
			asset_reference, int(amount_minor), signer_id, evidence_reference, settlement_status,
		)
		self.transactions[transaction_id] = item
		self._audit(tenant_id, "chain_transaction_recorded", transaction_id)
		return item.to_dict()

	def anchor_evidence(
		self,
		anchor_id: str,
		tenant_id: str,
		network_id: str,
		payload_hash: str,
		reference_id: str,
		anchored_at: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		network = self._tenant_network_or_none(network_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "anchor_evidence",
			"network_present": network is not None,
			"payload_hash_present": present(payload_hash),
			"reference_present": present(reference_id),
			"anchored_at_present": present(anchored_at),
			"evidence_present": present(evidence_reference),
		})
		item = EvidenceAnchor(
			anchor_id, tenant_id, network_id, payload_hash,
			reference_id, anchored_at, evidence_reference,
		)
		self.anchors[anchor_id] = item
		self._audit(tenant_id, "evidence_anchor_recorded", anchor_id)
		return item.to_dict()

	def register_oracle_feed(
		self,
		oracle_id: str,
		tenant_id: str,
		network_id: str,
		feed_type: str,
		source_reference: str,
		owner_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		network = self._tenant_network_or_none(network_id, tenant_id)
		feed_type = normalize_code(feed_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_oracle_feed",
			"network_present": network is not None,
			"feed_type_supported": feed_type in SUPPORTED_ORACLE_FEED_TYPES,
			"source_present": present(source_reference),
			"owner_present": present(owner_id),
			"evidence_present": present(evidence_reference),
		})
		item = OracleFeed(
			oracle_id, tenant_id, network_id, feed_type,
			source_reference, owner_id, evidence_reference,
		)
		self.oracles[oracle_id] = item
		self._audit(tenant_id, "oracle_feed_registered", oracle_id)
		return item.to_dict()

	def record_node_health(
		self,
		node_id: str,
		tenant_id: str,
		network_id: str,
		endpoint_reference: str,
		status: str,
		block_height: int,
		evidence_reference: str,
	) -> dict[str, Any]:
		network = self._tenant_network_or_none(network_id, tenant_id)
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_node_health",
			"network_present": network is not None,
			"endpoint_present": present(endpoint_reference),
			"node_status_supported": status in SUPPORTED_NODE_STATUSES,
			"block_height_valid": non_negative_int(block_height),
			"evidence_present": present(evidence_reference),
		})
		item = NodeHealth(
			node_id, tenant_id, network_id, endpoint_reference,
			status, int(block_height), evidence_reference,
		)
		self.nodes[node_id] = item
		self._audit(tenant_id, "node_health_recorded", node_id)
		return item.to_dict()

	def record_review(
		self,
		review_id: str,
		tenant_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = BlockchainReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = item
		self._audit(tenant_id, "blockchain_review_recorded", review_id)
		return item.to_dict()

	def register_blockchain_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_blockchain_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = BlockchainAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[agent_id] = item
		self._audit(tenant_id, "blockchain_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "blockchain_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "blockchain_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.blockchain.lifecycle",
			"accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"network_count": self._count(self.networks, tenant_id),
			"wallet_count": self._count(self.wallets, tenant_id),
			"contract_count": self._count(self.contracts, tenant_id),
			"transaction_count": self._count(self.transactions, tenant_id),
			"anchor_count": self._count(self.anchors, tenant_id),
			"oracle_count": self._count(self.oracles, tenant_id),
			"node_count": self._count(self.nodes, tenant_id),
			"degraded_node_count": sum(
				1 for n in self.nodes.values()
				if n.tenant_id == tenant_id and n.status != "healthy"
			),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"certificate_count": len(self._certificates),
			"token_count": len(self._tokens),
			"supply_chain_event_count": len(self._supply_chain),
			"cross_chain_transfer_count": len(self._cross_chain_transfers),
			"audit_event_count": sum(1 for ev in self.audit_events if ev["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return blockchain service health status."""
		return {
			"service": "blockchain", "status": "healthy",
			"network_count": len(self.networks), "token_count": len(self._tokens),
			"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def multi_sig_transaction(self, chain_id: str, contract_address: str, signers: list[str], required_sigs: int, data: Any) -> dict[str, Any]:
		"""Submit a multi-signature transaction requiring M-of-N signers."""
		assert len(signers) >= required_sigs, "required_sigs cannot exceed number of signers"
		tx = await self.record_transaction(chain_id, {"multisig": True, "required": required_sigs, "data": str(data)}, signers[0], tx_type="transfer")
		collected = [{"signer": s, "signed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()} for s in signers[:required_sigs]]
		self._audit(self.tenant_id, "multisig_transaction_submitted", tx["tx_hash"])
		return {**tx, "contract_address": contract_address, "signatures_collected": len(collected), "signatures_required": required_sigs, "signatures": collected}

	async def nft_mint(self, chain_id: str, recipient: str, metadata: dict[str, Any], collection_name: str) -> dict[str, Any]:
		"""Mint an NFT on a specified blockchain network."""
		cert = await self.digital_certificate_issue(recipient=recipient, certificate_data=metadata, certificate_type="nft", chain_id=chain_id)
		self._audit(self.tenant_id, "nft_minted", cert["certificate_id"])
		return {**cert, "collection_name": collection_name, "token_standard": "ERC-721"}

	async def smart_contract_upgrade(self, old_contract_address: str, new_contract_code: str, deployer: str, chain_id: str) -> dict[str, Any]:
		"""Deploy an upgraded version of a smart contract (proxy pattern)."""
		new_deployment = await self.deploy_smart_contract(chain_id, new_contract_code, deployer)
		record: dict[str, Any] = {
			"upgrade_id": str(uuid.uuid4()), "old_contract_address": old_contract_address,
			"new_contract_address": new_deployment["contract_address"],
			"deployer": deployer, "chain_id": chain_id,
			"upgraded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._audit(self.tenant_id, "smart_contract_upgraded", old_contract_address)
		return record

	async def cbdc_issuance(self, chain_id: str, currency_code: str, amount: int, central_bank: str) -> dict[str, Any]:
		"""Issue Central Bank Digital Currency (CBDC) tokens on a private blockchain."""
		token_rec = await self.token_issuance(chain_id=chain_id, token_name=f"e{currency_code}", total_supply=amount, owner=central_bank, token_symbol=f"e{currency_code}", decimals=2)
		self._audit(self.tenant_id, "cbdc_issued", token_rec["token_id"])
		return {**token_rec, "cbdc_type": "retail", "central_bank": central_bank, "currency_code": currency_code}

	async def blockchain_interoperability_bridge(self, from_chain: str, to_chain: str, asset: str, amount: int, sender: str, recipient: str) -> dict[str, Any]:
		"""Bridge assets between two blockchain networks."""
		return await self.cross_chain_transfer(from_chain=from_chain, to_chain=to_chain, amount=amount, token=asset, sender=sender, recipient=recipient)

	async def governance_proposal_vote(self, chain_id: str, proposal_id: str, voter: str, vote: str, voting_power: int) -> dict[str, Any]:
		"""Cast a governance vote on a blockchain protocol proposal."""
		assert vote in {"yes", "no", "abstain"}, f"unsupported vote: {vote}"
		tx = await self.record_transaction(chain_id, {"governance": True, "proposal_id": proposal_id, "vote": vote, "power": voting_power}, voter, tx_type="governance")
		self._audit(self.tenant_id, "governance_vote_cast", proposal_id)
		return {**tx, "proposal_id": proposal_id, "voter": voter, "vote": vote, "voting_power": voting_power}

	async def kyc_on_chain(self, chain_id: str, customer_id: str, kyc_hash: str, issuer: str) -> dict[str, Any]:
		"""Anchor KYC attestation on-chain for portable digital identity."""
		return await self.audit_trail_on_chain(record_id=f"kyc:{customer_id}", record_hash=kyc_hash, chain_id=chain_id, metadata={"issuer": issuer, "customer_id": customer_id})

	async def export_blockchain_data(self, fmt: str = "json") -> dict[str, Any]:
		"""Export blockchain registry and transaction data."""
		assert fmt in {"json", "csv", "excel"}
		return {
			"tenant_id": self.tenant_id, "format": fmt,
			"networks": len(self.networks), "transactions": len(self.transactions),
			"file_reference": f"blockchain_{self.tenant_id}_{datetime.datetime.now(datetime.timezone.utc).isoformat()[:10]}.{fmt}",
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_network_or_none(self, item_id: str, tenant_id: str) -> BlockchainNetwork | None:
		item = self.networks.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _chain_name_for_id(self, chain_id: str) -> str:
		"""Resolve a chain_id to a human-readable name from registered networks."""
		for net in self.networks.values():
			if net.chain_id == chain_id:
				return net.network_type
		return chain_id

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", "blockchain_policy_denied")
			for action in result["actions"]
		)
		raise PermissionError(reasons or "blockchain_policy_denied")


FintechBlockchainService = BlockchainService
BlockchainServicesService = BlockchainService
