"""In-memory models for APG Delivery Management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Delivery:
	id: str
	tenant_id: str
	delivery_type: str
	recipient_name: str
	delivery_address: str
	time_window_start: str
	time_window_end: str
	status: str
	sla_tier: str
	attempt_count: int

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ProofOfDelivery:
	id: str
	tenant_id: str
	delivery_id: str
	pod_type: str
	geo_stamp: str
	captured_at: str
	signatory_name: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FailedDelivery:
	id: str
	tenant_id: str
	delivery_id: str
	failure_reason: str
	failed_at: str
	notes: str
	rescheduled: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DeliveryReschedule:
	id: str
	tenant_id: str
	delivery_id: str
	source: str
	new_time_window_start: str
	new_time_window_end: str
	reschedule_count: int

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SlaRecord:
	id: str
	tenant_id: str
	delivery_id: str
	sla_tier: str
	committed_at: str
	actual_at: str | None
	breached: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DeliveryNotification:
	id: str
	tenant_id: str
	delivery_id: str
	channel: str
	recipient_contact: str
	sent_at: str
	notification_type: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DeliveryReturn:
	id: str
	tenant_id: str
	delivery_id: str
	return_reason: str
	rma_number: str
	initiated_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DeliveryAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
