"""Grant Management — Pydantic v2 models."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
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


_cfg = ConfigDict(extra="forbid", validate_by_name=True)


class GrnGrantCreate(BaseModel):
	model_config = _cfg
	title: str
	donor_reference: str
	currency: str = "KES"
	amount: Decimal
	start_date: str
	end_date: str
	sector: str = ""
	country: str = "KE"
	programme_id: str | None = None
	contact_person: str = ""
	notes: str = ""


class GrnGrantUpdate(BaseModel):
	model_config = _cfg
	title: str | None = None
	amount: Decimal | None = None
	end_date: str | None = None
	status: str | None = None
	contact_person: str | None = None
	notes: str | None = None


class GrnGrantResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	title: str
	donor_reference: str
	currency: str
	amount: Decimal
	disbursed_amount: Decimal = Decimal("0")
	start_date: str
	end_date: str
	sector: str
	country: str
	programme_id: str | None
	contact_person: str
	status: str
	tenant_id: str
	created_at: str
	updated_at: str | None = None


class GrnProposalCreate(BaseModel):
	model_config = _cfg
	grant_id: str
	title: str
	narrative: str
	budget: Decimal
	currency: str = "KES"
	submitted_by: str
	deadline: str


class GrnProposalResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	grant_id: str
	title: str
	narrative: str
	budget: Decimal
	currency: str
	submitted_by: str
	deadline: str
	status: str
	tenant_id: str
	created_at: str


class GrnBudgetLineCreate(BaseModel):
	model_config = _cfg
	grant_id: str
	category: str
	description: str
	amount: Decimal
	currency: str = "KES"
	period: str = ""


class GrnBudgetLineResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	grant_id: str
	category: str
	description: str
	amount: Decimal
	spent_amount: Decimal = Decimal("0")
	currency: str
	period: str
	status: str
	tenant_id: str
	created_at: str


class GrnDisbursementCreate(BaseModel):
	model_config = _cfg
	grant_id: str
	amount: Decimal
	currency: str = "KES"
	disbursement_date: str
	reference: str
	payment_method: str = "bank_transfer"
	approved_by: str
	notes: str = ""


class GrnDisbursementResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	grant_id: str
	amount: Decimal
	currency: str
	disbursement_date: str
	reference: str
	payment_method: str
	approved_by: str
	notes: str
	status: str
	tenant_id: str
	created_at: str


class GrnComplianceReportCreate(BaseModel):
	model_config = _cfg
	grant_id: str
	report_type: str
	period_start: str
	period_end: str
	submitted_by: str
	narrative: str = ""
	attachments: list[str] = Field(default_factory=list)


class GrnComplianceReportResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	grant_id: str
	report_type: str
	period_start: str
	period_end: str
	submitted_by: str
	narrative: str
	attachments: list[str]
	status: str
	tenant_id: str
	created_at: str


class GrnAuditFindingCreate(BaseModel):
	model_config = _cfg
	grant_id: str
	finding_type: str
	severity: str = "medium"
	description: str
	auditor: str
	audit_date: str
	recommendations: str = ""


class GrnAuditFindingResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	grant_id: str
	finding_type: str
	severity: str
	description: str
	auditor: str
	audit_date: str
	recommendations: str
	status: str
	tenant_id: str
	created_at: str


class GrnGrantFilter(BaseModel):
	model_config = _cfg
	status: str | None = None
	sector: str | None = None
	donor_reference: str | None = None
	country: str | None = None
	currency: str | None = None


class GrnAuditEvent(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	actor: str | None = None
	details: dict[str, Any] = Field(default_factory=dict)
	emitted_at: str


__all__ = [
	"GrnGrantCreate", "GrnGrantUpdate", "GrnGrantResponse",
	"GrnProposalCreate", "GrnProposalResponse",
	"GrnBudgetLineCreate", "GrnBudgetLineResponse",
	"GrnDisbursementCreate", "GrnDisbursementResponse",
	"GrnComplianceReportCreate", "GrnComplianceReportResponse",
	"GrnAuditFindingCreate", "GrnAuditFindingResponse",
	"GrnGrantFilter", "GrnAuditEvent",
]
