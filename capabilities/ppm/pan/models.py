"""In-memory models for APG Portfolio Analytics (pan)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Portfolio:
	id: str
	tenant_id: str
	name: str
	status: str
	classification: str
	owner_id: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AlignmentScore:
	id: str
	tenant_id: str
	portfolio_id: str
	dimension: str
	scoring_method: str
	score: float
	rationale: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RiskReturnAnalysis:
	id: str
	tenant_id: str
	portfolio_id: str
	risk_category: str
	return_metric: str
	risk_score: float
	return_value: float
	analysis_period: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CapacityHeatMap:
	id: str
	tenant_id: str
	portfolio_id: str
	dimension: str
	snapshot_period: str
	heat_map_data: str  # JSON serialised grid
	generated_by: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PerformanceSnapshot:
	id: str
	tenant_id: str
	portfolio_id: str
	period: str
	metrics: str  # JSON serialised KPI map
	benchmark_type: str
	benchmark_value: float
	actual_value: float

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ScenarioAnalysis:
	id: str
	tenant_id: str
	portfolio_id: str
	scenario_name: str
	assumptions: str  # JSON serialised
	projected_outcome: str  # JSON serialised
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PortfolioReport:
	id: str
	tenant_id: str
	portfolio_id: str
	dashboard_type: str
	format: str
	generated_by: str
	report_data: str  # JSON serialised

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PortfolioAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
