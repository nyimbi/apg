"""In-memory models for APG Generation Management."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GenPlant:
	id: str
	tenant_id: str
	name: str
	plant_type: str
	fuel_type: str
	capacity_mw: float
	status: str
	owner_id: str
	commissioning_date: str
	location_reference: str
	derating_mw: float = 0.0
	min_stable_load_mw: float = 0.0
	notes: str = ""

	def available_mw(self) -> float:
		return max(0.0, self.capacity_mw - self.derating_mw)

	def to_dict(self) -> dict[str, Any]:
		d = asdict(self)
		d["available_mw"] = self.available_mw()
		return d


@dataclass
class DispatchSchedule:
	id: str
	tenant_id: str
	plant_id: str
	dispatch_mode: str
	scheduled_mw: float
	start_time: str
	end_time: str
	status: str
	approved_by: str = ""
	approved_at: str = ""
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PlantOutage:
	id: str
	tenant_id: str
	plant_id: str
	outage_type: str
	status: str
	planned_start: str
	planned_end: str
	actual_start: str = ""
	actual_end: str = ""
	approved_by: str = ""
	derated_mw: float = 0.0
	reason: str = ""
	evidence_reference: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class GenerationKPI:
	id: str
	tenant_id: str
	plant_id: str
	kpi_type: str
	period: str
	period_start: str
	period_end: str
	value: float
	unit: str
	calculated_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CapacityPlan:
	id: str
	tenant_id: str
	plan_name: str
	horizon_years: int
	base_year: int
	total_existing_mw: float
	total_planned_mw: float
	peak_demand_mw: float
	reserve_margin_pct: float
	created_by: str
	approved_by: str = ""
	status: str = "draft"
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FuelStock:
	id: str
	tenant_id: str
	plant_id: str
	fuel_type: str
	quantity: float
	unit: str
	days_of_supply: float
	last_updated: str
	alert_threshold_days: float = 7.0
	supplier_reference: str = ""

	def is_low(self) -> bool:
		return self.days_of_supply <= self.alert_threshold_days

	def to_dict(self) -> dict[str, Any]:
		d = asdict(self)
		d["is_low"] = self.is_low()
		return d


@dataclass
class GenAgent:
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
