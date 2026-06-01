"""Dependency-light data models for APG Portfolio Management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PortfolioBook:
	id: str
	tenant_id: str
	owner_id: str
	name: str
	portfolio_type: str
	base_currency: str
	policy_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class HoldingRecord:
	id: str
	tenant_id: str
	portfolio_id: str
	instrument_id: str
	quantity: float
	cost_minor: int
	currency: str
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class AllocationPolicy:
	id: str
	tenant_id: str
	portfolio_id: str
	target_allocation: dict[str, float]
	policy_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__ | {"target_allocation": dict(self.target_allocation)}


@dataclass
class PortfolioValuation:
	id: str
	tenant_id: str
	portfolio_id: str
	market_value_minor: int
	currency: str
	valuation_date: str
	source_reference: str
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class BenchmarkAssignment:
	id: str
	tenant_id: str
	portfolio_id: str
	index_id: str
	policy_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class RiskExposure:
	id: str
	tenant_id: str
	portfolio_id: str
	metric: str
	value: float
	as_of_date: str
	source_reference: str
	limit_reference: str
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class PerformanceAttribution:
	id: str
	tenant_id: str
	portfolio_id: str
	period: str
	benchmark_id: str
	source_reference: str
	contributions: dict[str, float]
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__ | {"contributions": dict(self.contributions)}


@dataclass
class CashMovement:
	id: str
	tenant_id: str
	portfolio_id: str
	amount_minor: int
	currency: str
	reference: str
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class CorporateAction:
	id: str
	tenant_id: str
	instrument_id: str
	action_type: str
	effective_date: str
	evidence_reference: str
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class ComplianceBreach:
	id: str
	tenant_id: str
	portfolio_id: str
	severity: str
	evidence_reference: str
	review_required: bool = True
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class PortfolioReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class PortfolioEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
