"""In-memory models for APG Vehicle Maintenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MaintenanceJob:
	id: str
	tenant_id: str
	vehicle_id: str
	maintenance_type: str
	status: str
	priority: str
	technician_id: str
	workshop_type: str
	estimated_hours: float
	actual_hours: float | None
	job_card_ref: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class WorkshopAllocation:
	id: str
	tenant_id: str
	workshop_type: str
	location: str
	bay_number: str
	job_id: str
	allocated_at: str
	released_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PartsOrder:
	id: str
	tenant_id: str
	job_id: str
	parts_category: str
	part_number: str
	description: str
	quantity: int
	supplier_id: str
	ordered_at: str
	received_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class WarrantyRecord:
	id: str
	tenant_id: str
	vehicle_id: str
	warranty_type: str
	provider: str
	start_date: str
	expiry_date: str
	claim_ref: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class VehicleInspection:
	id: str
	tenant_id: str
	vehicle_id: str
	inspection_type: str
	inspector_id: str
	conducted_at: str
	defects_found: bool
	digital_signature: str
	passed: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RoadworthinessRecord:
	id: str
	tenant_id: str
	vehicle_id: str
	standard: str
	certificate_number: str
	issued_at: str
	expires_at: str
	issuing_authority: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MaintenanceSchedule:
	id: str
	tenant_id: str
	vehicle_id: str
	maintenance_type: str
	scheduled_at: str
	interval_km: int | None
	interval_days: int | None
	last_completed_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MaintenanceAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
