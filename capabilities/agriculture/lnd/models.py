"""Land Management models — Pydantic v2."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str


class TenureType(str, Enum):
	FREEHOLD = "freehold"
	LEASEHOLD = "leasehold"
	CUSTOMARY = "customary"
	COMMUNAL = "communal"
	GOVERNMENT = "government"


class ParcelStatus(str, Enum):
	REGISTERED = "registered"
	DISPUTED = "disputed"
	UNDER_TRANSFER = "under_transfer"
	ENCUMBERED = "encumbered"
	CANCELLED = "cancelled"


class TransferStatus(str, Enum):
	INITIATED = "initiated"
	PENDING_APPROVAL = "pending_approval"
	APPROVED = "approved"
	REGISTERED = "registered"
	REJECTED = "rejected"


class LandParcelCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	parcel_number: str
	area_ha: float
	tenure_type: TenureType
	owner_id: str
	owner_name: str
	location_county: str
	location_sub_county: str | None = None
	location_ward: str | None = None
	coordinates: list[dict[str, float]] = Field(default_factory=list)
	land_use: str | None = None
	notes: str | None = None


class LandParcelUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	owner_id: str | None = None
	owner_name: str | None = None
	tenure_type: TenureType | None = None
	land_use: str | None = None
	status: ParcelStatus | None = None
	notes: str | None = None


class LandParcelResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	parcel_number: str
	area_ha: float
	tenure_type: TenureType
	owner_id: str
	owner_name: str
	location_county: str
	location_sub_county: str | None = None
	location_ward: str | None = None
	coordinates: list[dict[str, float]] = Field(default_factory=list)
	land_use: str | None = None
	status: ParcelStatus = ParcelStatus.REGISTERED
	title_number: str | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


class GPSBoundaryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	parcel_id: str
	captured_by: str
	device_id: str | None = None
	waypoints: list[dict[str, float]]
	accuracy_m: float | None = None
	notes: str | None = None


class GPSBoundaryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	parcel_id: str
	captured_by: str
	device_id: str | None = None
	waypoints: list[dict[str, float]]
	computed_area_ha: float | None = None
	accuracy_m: float | None = None
	notes: str | None = None
	captured_at: str
	created_at: str


class TitleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	parcel_id: str
	title_number: str
	issued_by: str
	issue_date: str
	tenure_type: TenureType
	notes: str | None = None


class TitleResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	parcel_id: str
	title_number: str
	issued_by: str
	issue_date: str
	tenure_type: TenureType
	valid: bool = True
	notes: str | None = None
	created_at: str


class TransferCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	parcel_id: str
	from_owner_id: str
	to_owner_id: str
	to_owner_name: str
	transfer_value: float | None = None
	currency: str = "KES"
	reason: str | None = None
	initiated_at: str | None = None
	notes: str | None = None


class TransferResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	parcel_id: str
	from_owner_id: str
	to_owner_id: str
	to_owner_name: str
	transfer_value: float | None = None
	currency: str
	reason: str | None = None
	status: TransferStatus = TransferStatus.INITIATED
	approved_at: str | None = None
	registered_at: str | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


class AuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	payload: dict[str, Any] = Field(default_factory=dict)
	occurred_at: str
