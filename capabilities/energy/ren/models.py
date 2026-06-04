"""In-memory models for APG Renewable Energy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RenewableAsset:
	id: str
	tenant_id: str
	name: str
	renewable_type: str
	capacity_mw: float
	status: str
	owner_id: str
	commissioning_date: str
	location_reference: str
	grid_connection_point: str = ""
	installed_capacity_kwp: float = 0.0
	annual_yield_estimate_mwh: float = 0.0
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CurtailmentEvent:
	id: str
	tenant_id: str
	asset_id: str
	reason: str
	curtailed_mwh: float
	start_time: str
	end_time: str
	revenue_loss: float
	currency: str
	approved_by: str = ""
	status: str = "pending"
	operator_reference: str = ""
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RecCertificate:
	id: str
	tenant_id: str
	asset_id: str
	rec_type: str
	quantity_mwh: float
	vintage_year: int
	registry: str
	status: str
	issued_at: str
	serial_number: str = ""
	transferred_to: str = ""
	transferred_at: str = ""
	retired_at: str = ""
	expires_at: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CarbonCredit:
	id: str
	tenant_id: str
	asset_id: str
	credit_type: str
	quantity_tco2e: float
	vintage_year: int
	standard: str
	verification_reference: str
	status: str
	issued_at: str
	serial_number: str = ""
	retired_at: str = ""
	project_id: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FeedInTariff:
	id: str
	tenant_id: str
	asset_id: str
	fit_type: str
	rate_per_kwh: float
	currency: str
	effective_date: str
	status: str
	approved_by: str
	end_date: str = ""
	minimum_export_kw: float = 0.0
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class GenerationForecast:
	id: str
	tenant_id: str
	asset_id: str
	forecast_type: str
	horizon: str
	forecast_start: str
	forecast_end: str
	values: list[dict[str, Any]]
	model_version: str
	published_at: str
	rmse: float = 0.0
	mae: float = 0.0

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PerformanceMetric:
	id: str
	tenant_id: str
	asset_id: str
	metric_type: str
	period_start: str
	period_end: str
	value: float
	unit: str
	benchmark_value: float = 0.0
	calculated_at: str = ""

	def deviation_from_benchmark(self) -> float:
		return round(self.value - self.benchmark_value, 4)

	def to_dict(self) -> dict[str, Any]:
		d = asdict(self)
		d["deviation_from_benchmark"] = self.deviation_from_benchmark()
		return d


@dataclass
class RenAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered_at: str
	active: bool = True

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AuditEvent:
	id: str
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	actor: str
	occurred_at: str
	payload: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
