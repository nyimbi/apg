"""Document Intelligence — Pydantic v2 models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid4())


class DocumentSubmitRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	document_type: str  # invoice | contract | id_document | form | receipt | other
	file_name: str
	file_content_base64: str = ""
	file_url: str = ""
	mime_type: str = "application/pdf"
	metadata: dict[str, Any] = Field(default_factory=dict)
	pipeline: str = "ocr_llm"  # ocr_only | llm_only | ocr_llm

class DocumentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	document_type: str
	file_name: str
	mime_type: str
	pipeline: str
	status: str = "pending"
	submitted_at: str

class ExtractionResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	document_id: str
	document_type: str
	extracted_fields: dict[str, Any] = Field(default_factory=dict)
	confidence: float = 0.0
	ocr_text: str = ""
	pages: int = 0
	processing_ms: float = 0.0
	status: str = "completed"
	extracted_at: str

class InvoiceFields(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	invoice_number: str | None = None
	invoice_date: str | None = None
	due_date: str | None = None
	vendor_name: str | None = None
	vendor_address: str | None = None
	buyer_name: str | None = None
	currency: str | None = None
	subtotal: float | None = None
	tax_amount: float | None = None
	total_amount: float | None = None
	line_items: list[dict[str, Any]] = Field(default_factory=list)

class ContractFields(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	contract_title: str | None = None
	parties: list[str] = Field(default_factory=list)
	effective_date: str | None = None
	expiry_date: str | None = None
	governing_law: str | None = None
	payment_terms: str | None = None
	termination_clause: str | None = None
	key_obligations: list[str] = Field(default_factory=list)

class IDDocumentFields(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	document_number: str | None = None
	full_name: str | None = None
	date_of_birth: str | None = None
	nationality: str | None = None
	issue_date: str | None = None
	expiry_date: str | None = None
	issuing_country: str | None = None
	mrz_line1: str | None = None
	mrz_line2: str | None = None
	verification_status: str = "unverified"

class DocintAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	document_id: str
	document_type: str = ""
	payload: dict[str, Any] = Field(default_factory=dict)
	created_at: str

class DocintFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	document_type: str | None = None
	status: str | None = None
	pipeline: str | None = None
