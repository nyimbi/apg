"""Dependency-light data models for APG Embedded Finance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PartnerProgram:
	id: str
	tenant_id: str
	name: str
	kyb_reference: str
	contract_reference: str
	risk_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class HostApplication:
	id: str
	tenant_id: str
	program_id: str
	name: str
	environment: str
	domain: str
	terms_reference: str
	status: str = "registered"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class ProductPlacement:
	id: str
	tenant_id: str
	application_id: str
	product_type: str
	channel: str
	scopes: list[str]
	risk_policy_reference: str
	status: str = "published"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__ | {"scopes": list(self.scopes)}


@dataclass
class CustomerConsent:
	id: str
	tenant_id: str
	application_id: str
	customer_reference: str
	scopes: list[str]
	expiry_date: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__ | {"scopes": list(self.scopes)}


@dataclass
class EmbeddedAccount:
	id: str
	tenant_id: str
	application_id: str
	customer_reference: str
	wallet_reference: str
	kyc_reference: str
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class EmbeddedPayment:
	id: str
	tenant_id: str
	application_id: str
	placement_id: str
	consent_id: str
	source_reference: str
	destination_reference: str
	amount_minor: int
	currency: str
	risk_reference: str
	status: str = "initiated"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class EmbeddedCardOffer:
	id: str
	tenant_id: str
	application_id: str
	customer_reference: str
	limit_minor: int
	risk_reference: str
	status: str = "offered"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class EmbeddedLendingOffer:
	id: str
	tenant_id: str
	application_id: str
	customer_reference: str
	amount_minor: int
	affordability_reference: str
	underwriting_reference: str
	status: str = "offered"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class SettlementBatch:
	id: str
	tenant_id: str
	program_id: str
	amount_minor: int
	currency: str
	reconciliation_reference: str
	status: str = "closed"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class RevenueShare:
	id: str
	tenant_id: str
	program_id: str
	percent: float
	contract_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class EmbeddedEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
