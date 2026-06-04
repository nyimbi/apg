"""In-memory models for APG Dispatch Operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class LoadPlan:
	id: str
	tenant_id: str
	load_type: str
	vehicle_id: str
	total_weight_kg: float
	total_volume_cbm: float
	stop_count: int
	optimisation_mode: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DriverAssignment:
	id: str
	tenant_id: str
	dispatch_id: str
	driver_id: str
	assignment_type: str
	assigned_at: str
	hours_available: float

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Dispatch:
	id: str
	tenant_id: str
	load_plan_id: str
	vehicle_id: str
	driver_id: str
	route_id: str
	status: str
	dispatched_at: str | None
	completed_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DispatchTrackingUpdate:
	id: str
	tenant_id: str
	dispatch_id: str
	update_type: str
	location: str
	timestamp: str
	eta_minutes: int | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DispatchException:
	id: str
	tenant_id: str
	dispatch_id: str
	exception_type: str
	raised_at: str
	resolved_at: str | None
	resolution_notes: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DispatchCommunication:
	id: str
	tenant_id: str
	dispatch_id: str
	channel: str
	recipient_id: str
	message: str
	sent_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DispatchAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
