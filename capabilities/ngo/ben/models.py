"""Beneficiary Registry — Pydantic v2 models."""
from __future__ import annotations

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


class BenBeneficiaryCreate(BaseModel):
	model_config = _cfg
	first_name: str
	last_name: str
	national_id: str = ""
	date_of_birth: str = ""
	gender: str = "unknown"
	phone: str = ""
	location: str = ""
	county: str = ""
	household_size: int = 1
	vulnerability_category: str = ""
	notes: str = ""


class BenBeneficiaryUpdate(BaseModel):
	model_config = _cfg
	first_name: str | None = None
	last_name: str | None = None
	phone: str | None = None
	location: str | None = None
	county: str | None = None
	household_size: int | None = None
	vulnerability_category: str | None = None
	status: str | None = None
	notes: str | None = None


class BenBeneficiaryResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	first_name: str
	last_name: str
	national_id: str
	date_of_birth: str
	gender: str
	phone: str
	location: str
	county: str
	household_size: int
	vulnerability_category: str
	vulnerability_score: float = 0.0
	status: str
	tenant_id: str
	created_at: str
	updated_at: str | None = None


class BenEnrolmentCreate(BaseModel):
	model_config = _cfg
	beneficiary_id: str
	programme_id: str
	enrolment_date: str
	enrolled_by: str
	notes: str = ""


class BenEnrolmentResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	beneficiary_id: str
	programme_id: str
	enrolment_date: str
	enrolled_by: str
	notes: str
	status: str
	tenant_id: str
	created_at: str


class BenVulnerabilityAssessmentCreate(BaseModel):
	model_config = _cfg
	beneficiary_id: str
	assessor: str
	assessment_date: str
	food_security_score: float = 0.0
	shelter_score: float = 0.0
	health_score: float = 0.0
	income_score: float = 0.0
	protection_score: float = 0.0
	notes: str = ""


class BenVulnerabilityAssessmentResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	beneficiary_id: str
	assessor: str
	assessment_date: str
	food_security_score: float
	shelter_score: float
	health_score: float
	income_score: float
	protection_score: float
	composite_score: float
	category: str
	notes: str
	tenant_id: str
	created_at: str


class BenTransferCreate(BaseModel):
	model_config = _cfg
	beneficiary_id: str
	programme_id: str
	amount: Decimal
	currency: str = "KES"
	transfer_date: str
	payment_method: str = "mpesa"
	reference: str
	approved_by: str
	notes: str = ""


class BenTransferResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	beneficiary_id: str
	programme_id: str
	amount: Decimal
	currency: str
	transfer_date: str
	payment_method: str
	reference: str
	approved_by: str
	notes: str
	status: str
	tenant_id: str
	created_at: str


class BenDeduplicationResult(BaseModel):
	model_config = _cfg
	beneficiary_id: str
	duplicate_candidates: list[dict[str, Any]] = Field(default_factory=list)
	match_score: float = 0.0
	is_duplicate: bool = False


class BenBeneficiaryFilter(BaseModel):
	model_config = _cfg
	status: str | None = None
	county: str | None = None
	vulnerability_category: str | None = None
	gender: str | None = None
	programme_id: str | None = None


class BenAuditEvent(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	details: dict[str, Any] = Field(default_factory=dict)
	emitted_at: str


__all__ = [
	"BenBeneficiaryCreate", "BenBeneficiaryUpdate", "BenBeneficiaryResponse",
	"BenEnrolmentCreate", "BenEnrolmentResponse",
	"BenVulnerabilityAssessmentCreate", "BenVulnerabilityAssessmentResponse",
	"BenTransferCreate", "BenTransferResponse",
	"BenDeduplicationResult", "BenBeneficiaryFilter", "BenAuditEvent",
]
