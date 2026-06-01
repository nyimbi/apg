"""In-memory models for APG Cryptocurrency Services."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CryptoAsset:
	id: str
	tenant_id: str
	symbol: str
	asset_type: str
	network_reference: str
	contract_reference: str
	precision: int
	owner_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CustodyAccount:
	id: str
	tenant_id: str
	provider_reference: str
	custody_model: str
	policy_reference: str
	owner_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CryptoBalance:
	id: str
	tenant_id: str
	account_id: str
	asset_id: str
	amount_minor: int
	valuation_minor: int
	valuation_currency: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CryptoOrder:
	id: str
	tenant_id: str
	account_id: str
	asset_id: str
	side: str
	order_type: str
	quantity_minor: int
	limit_price_minor: int
	policy_reference: str
	requester_id: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CryptoTrade:
	id: str
	tenant_id: str
	order_id: str
	venue_reference: str
	execution_price_minor: int
	quantity_minor: int
	fee_minor: int
	status: str
	settlement_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CryptoTransfer:
	id: str
	tenant_id: str
	account_id: str
	asset_id: str
	transfer_type: str
	destination_reference: str
	amount_minor: int
	approval_reference: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ComplianceScreening:
	id: str
	tenant_id: str
	reference_id: str
	screening_type: str
	status: str
	evidence_reference: str
	reviewer_id: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PriceSnapshot:
	id: str
	tenant_id: str
	asset_id: str
	source: str
	price_minor: int
	currency: str
	observed_at: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CryptoReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CryptoAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
