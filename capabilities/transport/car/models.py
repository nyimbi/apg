"""In-memory models for APG Cargo Management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CargoBooking:
	id: str
	tenant_id: str
	cargo_type: str
	shipper_id: str
	consignee_id: str
	origin: str
	destination: str
	weight_kg: float
	volume_cbm: float
	incoterm: str
	status: str
	packaging_type: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CargoManifest:
	id: str
	tenant_id: str
	booking_id: str
	status: str
	customs_declaration_ref: str
	submitted_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DangerousGoodsDeclaration:
	id: str
	tenant_id: str
	booking_id: str
	dg_class: str
	un_number: str
	packing_group: str
	emergency_contact: str
	compliance_standard: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CargoTrackingEvent:
	id: str
	tenant_id: str
	booking_id: str
	event_type: str
	location: str
	timestamp: str
	notes: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CargoRevenueRecord:
	id: str
	tenant_id: str
	booking_id: str
	revenue_type: str
	amount: float
	currency: str
	reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CargoComplianceRecord:
	id: str
	tenant_id: str
	booking_id: str
	standard: str
	certificate_ref: str
	checked_at: str
	passed: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CargoAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
