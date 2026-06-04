"""In-memory models for APG Customer Management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CusCustomer:
	id: str
	tenant_id: str
	customer_type: str
	msisdn: str
	name: str
	status: str
	kyc_status: str
	created_by: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CusKycDocument:
	id: str
	tenant_id: str
	customer_id: str
	document_type: str
	document_reference: str
	status: str
	verified_by: str | None
	expires_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CusPlan:
	id: str
	tenant_id: str
	customer_id: str
	plan_type: str
	plan_name: str
	plan_reference: str
	activated_at: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CusSim:
	id: str
	tenant_id: str
	customer_id: str
	iccid: str
	imsi: str
	msisdn: str
	status: str
	provisioned_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CusDevice:
	id: str
	tenant_id: str
	customer_id: str
	device_type: str
	imei: str
	model: str
	blacklist_checked: bool
	registered_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CusCase:
	id: str
	tenant_id: str
	customer_id: str
	case_type: str
	status: str
	description: str
	assigned_to: str | None
	opened_at: str
	resolved_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CusLifecycleEvent:
	id: str
	tenant_id: str
	customer_id: str
	event_type: str
	event_reference: str
	occurred_at: str
	recorded_by: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CusAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
