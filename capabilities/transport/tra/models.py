"""In-memory models for APG Asset Tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrackedAsset:
	id: str
	tenant_id: str
	asset_type: str
	unique_id: str
	owner_id: str
	registration: str
	tracking_technology: str
	active: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AssetLocationUpdate:
	id: str
	tenant_id: str
	asset_id: str
	latitude: float
	longitude: float
	speed_kmh: float
	heading_degrees: float
	timestamp: str
	source: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Geofence:
	id: str
	tenant_id: str
	geofence_type: str
	name: str
	boundary_definition: str
	active: bool
	alert_on_entry: bool
	alert_on_exit: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class TrackingAlert:
	id: str
	tenant_id: str
	asset_id: str
	alert_type: str
	severity: str
	raised_at: str
	acknowledged_at: str | None
	resolved_at: str | None
	details: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ColdChainRecord:
	id: str
	tenant_id: str
	asset_id: str
	standard: str
	min_temp_c: float
	max_temp_c: float
	recorded_temp_c: float
	timestamp: str
	breached: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Container:
	id: str
	tenant_id: str
	iso_number: str
	seal_number: str
	status: str
	owner_id: str
	current_location: str
	last_updated: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AssetUtilisationRecord:
	id: str
	tenant_id: str
	asset_id: str
	period: str
	period_start: str
	period_end: str
	idle_time_minutes: int
	active_time_minutes: int
	distance_km: float
	utilisation_pct: float

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class TrackingAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
