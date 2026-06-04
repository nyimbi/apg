"""In-memory models for APG Performance Management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PerKpi:
	id: str
	tenant_id: str
	kpi_category: str
	kpi_name: str
	value: float
	baseline_value: float
	unit: str
	status: str
	network_layer: str
	recorded_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PerSlaCompliance:
	id: str
	tenant_id: str
	sla_type: str
	customer_id: str | None
	target_value: float
	actual_value: float
	status: str
	period: str
	notification_sent: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PerCapacityRecord:
	id: str
	tenant_id: str
	resource_reference: str
	capacity_state: str
	utilisation_pct: float
	forecast_horizon_days: int
	recorded_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PerTrend:
	id: str
	tenant_id: str
	kpi_id: str
	trend_direction: str
	lookback_days: int
	forecast_value: float | None
	recorded_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PerThreshold:
	id: str
	tenant_id: str
	kpi_name: str
	network_layer: str
	warning_value: float
	critical_value: float
	action: str
	approval_reference: str
	set_by: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PerBenchmark:
	id: str
	tenant_id: str
	benchmark_type: str
	kpi_name: str
	benchmark_value: float
	current_value: float
	gap_pct: float
	recorded_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PerReport:
	id: str
	tenant_id: str
	report_period: str
	format: str
	approval_reference: str
	generated_by: str
	generated_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PerAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
