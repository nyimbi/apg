"""In-memory models for APG Smart Metering & AMI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SmartMeter:
	id: str
	tenant_id: str
	serial_number: str
	meter_type: str
	communication_technology: str
	status: str
	customer_id: str
	location_reference: str
	installed_at: str
	firmware_version: str = ""
	last_communication_at: str = ""
	load_limit_kw: float = 0.0
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class IntervalReading:
	id: str
	tenant_id: str
	meter_id: str
	reading_type: str
	interval_length: str
	interval_start: str
	interval_end: str
	value: float
	unit: str
	quality_flag: str
	received_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class TamperEvent:
	id: str
	tenant_id: str
	meter_id: str
	tamper_type: str
	detected_at: str
	evidence_reference: str
	status: str = "open"
	investigated_by: str = ""
	resolved_at: str = ""
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RemoteCommand:
	id: str
	tenant_id: str
	meter_id: str
	command_type: str
	status: str
	issued_by: str
	issued_at: str
	approved_by: str = ""
	sent_at: str = ""
	executed_at: str = ""
	failed_reason: str = ""
	retry_count: int = 0
	parameters: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DemandResponseEvent:
	id: str
	tenant_id: str
	event_type: str
	status: str
	target_reduction_kw: float
	actual_reduction_kw: float
	start_time: str
	end_time: str
	notification_sent_at: str = ""
	created_by: str = ""
	meter_ids: list[str] = field(default_factory=list)
	opt_out_meter_ids: list[str] = field(default_factory=list)

	def participation_count(self) -> int:
		return len(self.meter_ids) - len(self.opt_out_meter_ids)

	def to_dict(self) -> dict[str, Any]:
		d = asdict(self)
		d["participation_count"] = self.participation_count()
		return d


@dataclass
class DataQualityFlag:
	id: str
	tenant_id: str
	reading_id: str
	meter_id: str
	quality_flag: str
	reason: str
	flagged_at: str
	flagged_by: str
	resolved_at: str = ""
	substitute_value: float | None = None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AmiHeadEndStatus:
	id: str
	tenant_id: str
	head_end_name: str
	protocol: str
	connected_meters: int
	total_meters: int
	last_heartbeat_at: str
	status: str = "healthy"
	error_message: str = ""

	def communication_ratio(self) -> float:
		if self.total_meters <= 0:
			return 0.0
		return round(self.connected_meters / self.total_meters, 4)

	def to_dict(self) -> dict[str, Any]:
		d = asdict(self)
		d["communication_ratio"] = self.communication_ratio()
		return d


@dataclass
class MetAgent:
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
