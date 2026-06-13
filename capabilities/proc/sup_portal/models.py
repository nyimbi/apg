"""Pydantic v2 models for APG Supplier Self-Service Portal."""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

try:
	from situ_cloudevents._uuid7 import uuid7str  # type: ignore[import]
except ImportError:
	from uuid6 import uuid7  # type: ignore[import]
	def uuid7str() -> str:
		return str(uuid7())


class SpSupplierProfile(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	supplier_id: str
	company_name: str
	contact_name: str
	email: str
	phone: str = ""
	tax_number: str = ""
	bank_details: dict[str, Any] = Field(default_factory=dict)
	status: str = "pending"
	performance_score: float | None = None
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SpQuote(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	supplier_id: str
	rfq_id: str
	line_items: list[dict[str, Any]] = Field(default_factory=list)
	total_amount: float = 0.0
	currency: str = "KES"
	valid_until: datetime | None = None
	terms: str = ""
	status: str = "submitted"
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SpInvoice(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	supplier_id: str
	po_number: str
	invoice_number: str
	line_items: list[dict[str, Any]] = Field(default_factory=list)
	total_amount: float = 0.0
	currency: str = "KES"
	invoice_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	due_date: datetime | None = None
	status: str = "submitted"
	payment_status: str = "unpaid"
	ocr_extracted: bool = False
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SpDeliveryConfirmation(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	supplier_id: str
	po_number: str
	delivery_date: datetime
	delivery_note: str = ""
	items_delivered: list[dict[str, Any]] = Field(default_factory=list)
	delivery_location: str = ""
	received_by: str = ""
	status: str = "confirmed"
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SpDispute(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	supplier_id: str
	reference_id: str
	reference_type: str
	dispute_reason: str
	details: str = ""
	resolution: str | None = None
	status: str = "open"
	sla_due_at: datetime | None = None
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	resolved_at: datetime | None = None
