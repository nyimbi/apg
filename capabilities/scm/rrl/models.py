"""Pydantic v2 models for Returns & Reverse Logistics (scm_rrl)."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


class RMACreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	order_id: str
	customer_id: str
	items: list[dict[str, Any]]
	reason_code: str  # defective | wrong_item | not_as_described | changed_mind | damaged_in_transit
	description: str | None = None
	requested_resolution: str = "refund"  # refund | replacement | credit | repair


class RMAUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: str | None = None
	resolution: str | None = None
	notes: str | None = None


class RMAResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	rma_number: str
	order_id: str
	customer_id: str
	items: list[dict[str, Any]]
	reason_code: str
	description: str | None
	requested_resolution: str
	resolution: str | None
	status: str
	created_at: str
	updated_at: str | None = None


class RefurbishmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	rma_id: str
	sku: str
	condition_received: str  # like_new | good | fair | poor | scrap
	refurbishment_actions: list[str]
	assigned_to: str | None = None
	estimated_cost: float | None = None


class RefurbishmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	rma_id: str
	sku: str
	condition_received: str
	condition_after: str | None
	refurbishment_actions: list[str]
	assigned_to: str | None
	estimated_cost: float | None
	actual_cost: float | None
	status: str
	created_at: str
	completed_at: str | None = None


class DisposalCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	rma_id: str
	sku: str
	quantity: float
	disposal_method: str  # recycle | destroy | donate | auction | landfill
	reason: str
	authorised_by: str


class DisposalResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	rma_id: str
	sku: str
	quantity: float
	disposal_method: str
	reason: str
	authorised_by: str
	disposal_cost: float | None
	status: str
	created_at: str
	disposed_at: str | None = None


class CreditNoteCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	rma_id: str
	customer_id: str
	amount: float
	currency: str = "USD"
	reason: str
	issued_by: str


class CreditNoteResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	credit_note_number: str
	rma_id: str
	customer_id: str
	amount: float
	currency: str
	reason: str
	issued_by: str
	status: str
	issued_at: str


class ReverseShipmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	rma_id: str
	carrier_id: str
	pickup_address: dict[str, Any]
	destination_address: dict[str, Any]
	weight_kg: float | None = None


class ReverseShipmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	rma_id: str
	carrier_id: str
	pickup_address: dict[str, Any]
	destination_address: dict[str, Any]
	weight_kg: float | None
	tracking_number: str | None
	status: str
	created_at: str


class RrlAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	status: str
	emitted_at: str
