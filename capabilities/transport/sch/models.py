"""In-memory models for APG Transport Scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Schedule:
	id: str
	tenant_id: str
	schedule_type: str
	status: str
	start_date: str
	end_date: str
	optimisation_mode: str
	created_by: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DriverShift:
	id: str
	tenant_id: str
	schedule_id: str
	driver_id: str
	shift_type: str
	start_time: str
	end_time: str
	hours: float
	tacho_compliant: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class VehicleAssignment:
	id: str
	tenant_id: str
	schedule_id: str
	vehicle_id: str
	route_id: str
	assigned_from: str
	assigned_until: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Charter:
	id: str
	tenant_id: str
	schedule_id: str
	charter_type: str
	customer_id: str
	vehicle_id: str
	driver_id: str
	pickup_location: str
	destination: str
	charter_date: str
	customer_confirmed: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ScheduleConflict:
	id: str
	tenant_id: str
	schedule_id: str
	conflict_type: str
	resource_id: str
	detected_at: str
	resolved_at: str | None
	resolution_notes: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ScheduleNotification:
	id: str
	tenant_id: str
	schedule_id: str
	notification_type: str
	recipient_id: str
	channel: str
	sent_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SchedulingAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
