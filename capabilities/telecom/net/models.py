"""In-memory models for APG Network Management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class NetAlarm:
	id: str
	tenant_id: str
	ne_reference: str
	severity: str
	category: str
	status: str
	description: str
	raised_at: str
	cleared_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class NetFaultTicket:
	id: str
	tenant_id: str
	alarm_id: str
	title: str
	severity: str
	assigned_to: str | None
	escalation_level: str
	status: str
	opened_at: str
	resolved_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class NetPerformanceRecord:
	id: str
	tenant_id: str
	ne_reference: str
	metric_type: str
	value: float
	threshold: float
	domain: str
	recorded_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class NetConfigChange:
	id: str
	tenant_id: str
	ne_reference: str
	change_type: str
	description: str
	status: str
	approval_reference: str
	submitted_by: str
	submitted_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class NetSlaRecord:
	id: str
	tenant_id: str
	sla_type: str
	customer_id: str | None
	target_value: float
	actual_value: float
	period: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class NetNocHandover:
	id: str
	tenant_id: str
	shift: str
	handing_over_operator: str
	taking_over_operator: str
	notes: str
	open_alarms_count: int
	handover_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class NetAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
