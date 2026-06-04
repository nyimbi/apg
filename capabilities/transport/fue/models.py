"""In-memory models for APG Fuel Management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FuelProcurement:
	id: str
	tenant_id: str
	procurement_type: str
	supplier_id: str
	fuel_type: str
	quantity_litres: float
	unit_price: float
	currency: str
	purchase_order_ref: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FuelTransaction:
	id: str
	tenant_id: str
	transaction_type: str
	vehicle_id: str
	driver_id: str
	fuel_type: str
	quantity_litres: float
	odometer_km: float
	unit_price: float
	currency: str
	transaction_at: str
	card_id: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FuelCard:
	id: str
	tenant_id: str
	provider: str
	card_number_masked: str
	vehicle_id: str | None
	driver_id: str | None
	active: bool
	pin_set: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FuelCardReconciliation:
	id: str
	tenant_id: str
	card_id: str
	period_start: str
	period_end: str
	expected_total: float
	actual_total: float
	discrepancy: float
	currency: str
	reconciled: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CarbonEmissionRecord:
	id: str
	tenant_id: str
	vehicle_id: str
	standard: str
	fuel_type: str
	quantity_litres: float
	co2_kg: float
	period_start: str
	period_end: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FuelStorageTank:
	id: str
	tenant_id: str
	storage_type: str
	location: str
	capacity_litres: float
	current_level_litres: float
	fuel_type: str
	last_calibrated: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FuelAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
