"""In-memory models for APG Decentralized Finance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DeFiProtocol:
	id: str
	tenant_id: str
	protocol_type: str
	network_reference: str
	protocol_reference: str
	owner_id: str
	evidence_reference: str
	risk_tier: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class LiquidityPosition:
	id: str
	tenant_id: str
	protocol_id: str
	account_reference: str
	asset_pair_reference: str
	position_type: str
	amount_minor: int
	collateral_minor: int
	health_factor_bps: int
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DeFiAction:
	id: str
	tenant_id: str
	protocol_id: str
	position_id: str
	action_type: str
	amount_minor: int
	requester_id: str
	approval_reference: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class YieldStrategy:
	id: str
	tenant_id: str
	protocol_id: str
	strategy_reference: str
	target_apy_bps: int
	max_risk_tier: str
	owner_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RewardAccrual:
	id: str
	tenant_id: str
	position_id: str
	reward_type: str
	asset_reference: str
	amount_minor: int
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class GovernanceProposal:
	id: str
	tenant_id: str
	protocol_id: str
	proposal_reference: str
	vote_choice: str
	voter_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RiskAssessment:
	id: str
	tenant_id: str
	reference_id: str
	risk_tier: str
	reviewer_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DeFiReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DeFiAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
