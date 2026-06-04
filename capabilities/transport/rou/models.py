"""In-memory models for APG Route Optimisation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Route:
	id: str
	tenant_id: str
	route_type: str
	origin: str
	destination: str
	vehicle_id: str
	transport_mode: str
	stop_count: int
	total_distance_km: float
	estimated_duration_minutes: int
	optimisation_objective: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RouteStop:
	id: str
	tenant_id: str
	route_id: str
	sequence: int
	location: str
	address: str
	time_window_start: str
	time_window_end: str
	service_time_minutes: int
	completed: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RouteConstraint:
	id: str
	tenant_id: str
	route_id: str
	constraint_type: str
	parameters: str
	active: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class TrafficIntegration:
	id: str
	tenant_id: str
	provider: str
	route_id: str
	incident_type: str | None
	delay_minutes: int
	recorded_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RerouteEvent:
	id: str
	tenant_id: str
	original_route_id: str
	new_route_id: str
	trigger: str
	triggered_at: str
	completed_at: str | None
	distance_delta_km: float

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MultimodalSegment:
	id: str
	tenant_id: str
	route_id: str
	transport_mode: str
	segment_origin: str
	segment_destination: str
	carrier_ref: str
	estimated_duration_minutes: int

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RouteAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
