"""Pydantic v2 models for Procurement Management (scm_prc)."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


class RFQLineCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	sku: str
	description: str | None = None
	quantity: float
	unit_of_measure: str = "EA"
	required_by: str | None = None


class RFQCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str
	lines: list[RFQLineCreate]
	vendor_ids: list[str] = Field(default_factory=list)
	deadline: str | None = None
	notes: str | None = None


class RFQUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: str | None = None
	deadline: str | None = None
	notes: str | None = None


class RFQResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	rfq_number: str
	title: str
	lines: list[dict[str, Any]]
	vendor_ids: list[str]
	deadline: str | None
	notes: str | None
	status: str
	created_at: str
	updated_at: str | None = None


class PurchaseOrderLineCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	sku: str
	description: str | None = None
	quantity: float
	unit_price: float
	currency: str = "USD"
	unit_of_measure: str = "EA"
	required_by: str | None = None


class PurchaseOrderCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	vendor_id: str
	lines: list[PurchaseOrderLineCreate]
	rfq_id: str | None = None
	payment_terms: str = "NET30"
	delivery_address: dict[str, Any] = Field(default_factory=dict)
	notes: str | None = None


class PurchaseOrderUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: str | None = None
	payment_terms: str | None = None
	notes: str | None = None


class PurchaseOrderResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	po_number: str
	vendor_id: str
	lines: list[dict[str, Any]]
	total_value: float
	currency: str
	rfq_id: str | None
	payment_terms: str
	delivery_address: dict[str, Any]
	notes: str | None
	status: str
	created_at: str
	updated_at: str | None = None


class ThreeWayMatchCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	po_id: str
	receipt_id: str
	invoice_number: str
	invoiced_amount: float
	currency: str = "USD"


class ThreeWayMatchResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	po_id: str
	receipt_id: str
	invoice_number: str
	po_amount: float
	received_amount: float
	invoiced_amount: float
	variance: float
	currency: str
	match_result: str  # matched | partial | disputed
	status: str
	created_at: str


class VendorEvaluationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	vendor_id: str
	period: str
	quality_score: float
	delivery_score: float
	price_score: float
	service_score: float
	evaluated_by: str
	notes: str | None = None


class VendorEvaluationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	vendor_id: str
	period: str
	quality_score: float
	delivery_score: float
	price_score: float
	service_score: float
	overall_score: float
	evaluated_by: str
	notes: str | None
	status: str
	created_at: str


class ContractCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	vendor_id: str
	contract_reference: str
	start_date: str
	end_date: str
	value: float
	currency: str = "USD"
	terms: dict[str, Any] = Field(default_factory=dict)


class ContractResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	vendor_id: str
	contract_reference: str
	start_date: str
	end_date: str
	value: float
	currency: str
	terms: dict[str, Any]
	status: str
	created_at: str


class PrcAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	status: str
	emitted_at: str
