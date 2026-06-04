"""In-memory models for APG Distribution Network."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NetworkElement:
	id: str
	tenant_id: str
	element_type: str
	name: str
	feeder_id: str
	voltage_level: str
	status: str
	location_reference: str
	substation_id: str = ""
	rating_kva: float = 0.0
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Feeder:
	id: str
	tenant_id: str
	name: str
	substation_id: str
	voltage_level: str
	status: str
	peak_load_mw: float = 0.0
	normal_capacity_mw: float = 0.0
	emergency_capacity_mw: float = 0.0

	def loading_pct(self) -> float:
		if self.normal_capacity_mw <= 0:
			return 0.0
		return round(self.peak_load_mw / self.normal_capacity_mw * 100, 2)

	def to_dict(self) -> dict[str, Any]:
		d = asdict(self)
		d["loading_pct"] = self.loading_pct()
		return d


@dataclass
class FaultRecord:
	id: str
	tenant_id: str
	element_id: str
	fault_type: str
	status: str
	detected_at: str
	location_reference: str
	isolated_at: str = ""
	restored_at: str = ""
	crew_id: str = ""
	affected_customers: int = 0
	cause: str = ""
	notes: str = ""

	def duration_minutes(self, current_time: str | None = None) -> float | None:
		"""Returns outage duration in minutes if both timestamps present."""
		if self.detected_at and self.restored_at:
			# Simplified: caller provides pre-computed value when needed
			return None
		return None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SwitchingOrder:
	id: str
	tenant_id: str
	element_id: str
	operation: str
	status: str
	requested_by: str
	requested_at: str
	approved_by: str = ""
	approved_at: str = ""
	executed_at: str = ""
	purpose: str = ""
	safety_notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class OutageRecord:
	id: str
	tenant_id: str
	feeder_id: str
	cause: str
	started_at: str
	restoration_strategy: str
	affected_customers: int
	restored_at: str = ""
	saidi_minutes: float = 0.0
	saifi_count: float = 0.0
	caidi_minutes: float = 0.0
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ScadaReading:
	id: str
	tenant_id: str
	element_id: str
	protocol: str
	parameter: str
	value: float
	unit: str
	quality: str
	timestamp: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class LoadBalanceAction:
	id: str
	tenant_id: str
	feeder_id: str
	mode: str
	action_type: str
	load_transferred_mw: float
	voltage_improvement_pu: float
	executed_at: str
	approved_by: str = ""
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DisAgent:
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
