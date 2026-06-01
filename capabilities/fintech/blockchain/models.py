"""In-memory models for APG Blockchain Services."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BlockchainNetwork:
	id: str
	tenant_id: str
	network_type: str
	environment: str
	chain_id: str
	rpc_reference: str
	owner_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BlockchainWallet:
	id: str
	tenant_id: str
	network_id: str
	wallet_reference: str
	custody_model: str
	key_policy_reference: str
	owner_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SmartContractDeployment:
	id: str
	tenant_id: str
	network_id: str
	contract_type: str
	artifact_reference: str
	owner_id: str
	approval_reference: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ChainTransaction:
	id: str
	tenant_id: str
	network_id: str
	transaction_hash: str
	transaction_type: str
	asset_reference: str
	amount_minor: int
	signer_id: str
	evidence_reference: str
	settlement_status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class EvidenceAnchor:
	id: str
	tenant_id: str
	network_id: str
	payload_hash: str
	reference_id: str
	anchored_at: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class OracleFeed:
	id: str
	tenant_id: str
	network_id: str
	feed_type: str
	source_reference: str
	owner_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class NodeHealth:
	id: str
	tenant_id: str
	network_id: str
	endpoint_reference: str
	status: str
	block_height: int
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BlockchainReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BlockchainAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
