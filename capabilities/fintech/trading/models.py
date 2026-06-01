"""Dependency-light data models for APG Algorithmic Trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TradingStrategy:
	id: str
	tenant_id: str
	owner_id: str
	name: str
	strategy_type: str
	asset_class: str
	policy_reference: str
	status: str = "registered"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class SignalSource:
	id: str
	tenant_id: str
	strategy_id: str
	source_reference: str
	freshness_sla: str
	lineage_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class BacktestRun:
	id: str
	tenant_id: str
	strategy_id: str
	period: str
	trade_count: int
	data_source_reference: str
	metrics: dict[str, float]
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__ | {"metrics": dict(self.metrics)}


@dataclass
class RiskLimit:
	id: str
	tenant_id: str
	strategy_id: str
	metric: str
	limit_value: float
	approval_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class OrderIntent:
	id: str
	tenant_id: str
	strategy_id: str
	risk_limit_id: str
	instrument_id: str
	order_type: str
	quantity: float
	approval_reference: str
	status: str = "staged"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class ExecutionRecord:
	id: str
	tenant_id: str
	order_id: str
	venue: str
	filled_quantity: float
	source_reference: str
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class PositionSnapshot:
	id: str
	tenant_id: str
	strategy_id: str
	as_of_date: str
	gross_exposure_minor: int
	net_exposure_minor: int
	source_reference: str
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class SurveillanceAlert:
	id: str
	tenant_id: str
	strategy_id: str
	severity: str
	evidence_reference: str
	review_required: bool = True
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class TradingReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class TradingEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
