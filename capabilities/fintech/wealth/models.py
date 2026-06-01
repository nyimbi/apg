"""Dependency-light data models for APG Wealth Management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClientProfile:
	id: str
	tenant_id: str
	name: str
	kyc_reference: str
	tax_reference: str
	risk_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class SuitabilityProfile:
	id: str
	tenant_id: str
	client_id: str
	risk_profile: str
	risk_tolerance: str
	horizon: str
	goals: list[str]
	status: str = "current"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__ | {"goals": list(self.goals)}


@dataclass
class Portfolio:
	id: str
	tenant_id: str
	client_id: str
	name: str
	base_currency: str
	advisor_id: str
	policy_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class AdvisoryMandate:
	id: str
	tenant_id: str
	portfolio_id: str
	suitability_id: str
	mandate_type: str
	policy_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class RebalanceProposal:
	id: str
	tenant_id: str
	portfolio_id: str
	mandate_id: str
	target_allocation: dict[str, float]
	analysis_reference: str
	status: str = "proposed"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__ | {"target_allocation": dict(self.target_allocation)}


@dataclass
class WealthOrder:
	id: str
	tenant_id: str
	portfolio_id: str
	instrument_id: str
	side: str
	quantity: float
	notional_minor: int
	risk_reference: str
	human_approval: str = ""
	status: str = "staged"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class PerformanceSnapshot:
	id: str
	tenant_id: str
	portfolio_id: str
	period: str
	valuation_reference: str
	benchmark_reference: str
	return_percent: float
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class FeeSchedule:
	id: str
	tenant_id: str
	portfolio_id: str
	advisory_percent: float
	performance_percent: float
	platform_percent: float
	contract_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class WealthEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
