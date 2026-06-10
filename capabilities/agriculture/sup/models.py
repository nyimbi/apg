"""Agricultural Supply Chain models — Pydantic v2."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str


class TraceabilityStatus(str, Enum):
	FARM = "farm"
	COLLECTION = "collection"
	PROCESSING = "processing"
	STORAGE = "storage"
	TRANSPORT = "transport"
	EXPORT = "export"
	DELIVERED = "delivered"


class ProcurementStatus(str, Enum):
	REQUESTED = "requested"
	QUOTED = "quoted"
	ORDERED = "ordered"
	DELIVERED = "delivered"
	INVOICED = "invoiced"
	PAID = "paid"


class ColdChainStatus(str, Enum):
	NORMAL = "normal"
	BREACH = "breach"
	CRITICAL = "critical"


class TraceabilityCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	batch_id: str
	product_type: str
	farm_parcel_id: str
	farmer_id: str
	harvest_date: str
	weight_kg: float
	quality_grade: str | None = None
	buyer_id: str | None = None
	notes: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class TraceabilityUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: TraceabilityStatus | None = None
	buyer_id: str | None = None
	current_location: str | None = None
	weight_kg: float | None = None
	notes: str | None = None
	metadata: dict[str, Any] | None = None


class TraceabilityResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	batch_id: str
	product_type: str
	farm_parcel_id: str
	farmer_id: str
	harvest_date: str
	weight_kg: float
	quality_grade: str | None = None
	status: TraceabilityStatus = TraceabilityStatus.FARM
	buyer_id: str | None = None
	current_location: str | None = None
	notes: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: str
	updated_at: str


class ProcurementCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	supplier_id: str
	product_name: str
	quantity: float
	unit: str
	unit_price: float
	required_date: str
	notes: str | None = None


class ProcurementUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: ProcurementStatus | None = None
	actual_delivery_date: str | None = None
	quantity_received: float | None = None
	invoice_reference: str | None = None
	notes: str | None = None


class ProcurementResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	supplier_id: str
	product_name: str
	quantity: float
	unit: str
	unit_price: float
	total_value: float
	required_date: str
	status: ProcurementStatus = ProcurementStatus.REQUESTED
	actual_delivery_date: str | None = None
	quantity_received: float | None = None
	invoice_reference: str | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


class ColdChainLogCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	batch_id: str
	location: str
	temperature_c: float
	humidity_pct: float | None = None
	recorded_at: str | None = None


class ColdChainLogResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	batch_id: str
	location: str
	temperature_c: float
	humidity_pct: float | None = None
	status: ColdChainStatus
	recorded_at: str
	created_at: str


class ExportDocCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	batch_id: str
	document_type: str
	issuing_authority: str | None = None
	issue_date: str
	expiry_date: str | None = None
	reference_number: str
	file_url: str | None = None
	notes: str | None = None


class ExportDocResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	batch_id: str
	document_type: str
	issuing_authority: str | None = None
	issue_date: str
	expiry_date: str | None = None
	reference_number: str
	file_url: str | None = None
	notes: str | None = None
	created_at: str


class AuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	payload: dict[str, Any] = Field(default_factory=dict)
	occurred_at: str
