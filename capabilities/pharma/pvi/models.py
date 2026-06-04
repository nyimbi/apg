"""Pydantic v2 models for APG Pharma Pharmacovigilance."""

from __future__ import annotations

from datetime import datetime
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _uuid7str() -> str:
	return str(uuid7())


class PviBase(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class AdvEventCase(PviBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	case_number: str
	source: str
	case_type: str
	product_id: str
	suspect_drug: str
	patient_age: int | None = None
	patient_sex: str | None = None
	patient_weight: float | None = None
	onset_date: datetime | None = None
	report_date: datetime
	status: str = "new"
	serious: bool = False
	seriousness_criteria: list[str] = Field(default_factory=list)
	causality: str | None = None
	meddra_pt: str | None = None
	meddra_soc: str | None = None
	meddra_coded: bool = False
	narrative: str | None = None
	duplicate_of: str | None = None
	follow_up_required: bool = False
	medical_reviewed: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("source")
	@classmethod
	def validate_source(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_AE_SOURCES
		if v not in SUPPORTED_AE_SOURCES:
			raise ValueError(f"source must be one of {SUPPORTED_AE_SOURCES}")
		return v

	@field_validator("case_type")
	@classmethod
	def validate_case_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_CASE_TYPES
		if v not in SUPPORTED_CASE_TYPES:
			raise ValueError(f"case_type must be one of {SUPPORTED_CASE_TYPES}")
		return v

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_CASE_STATUSES
		if v not in SUPPORTED_CASE_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_CASE_STATUSES}")
		return v


class AdvEventCaseCreate(PviBase):
	tenant_id: str
	case_number: str
	source: str
	case_type: str
	product_id: str
	suspect_drug: str
	report_date: datetime
	serious: bool = False
	onset_date: datetime | None = None
	created_by: str


class IcsrSubmission(PviBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	case_id: str
	regulatory_database: str
	submission_type: str
	e2b_r3_message_id: str | None = None
	submission_date: datetime | None = None
	acknowledgement_date: datetime | None = None
	due_date: datetime
	status: str = "pending"
	follow_up_number: int = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("regulatory_database")
	@classmethod
	def validate_database(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_REGULATORY_DATABASES
		if v not in SUPPORTED_REGULATORY_DATABASES:
			raise ValueError(f"regulatory_database must be one of {SUPPORTED_REGULATORY_DATABASES}")
		return v


class SafetySignal(PviBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	signal_number: str
	product_id: str
	signal_type: str
	meddra_pt: str
	description: str
	detected_by: str
	detection_method: str
	detection_date: datetime
	status: str = "new"
	strength_of_evidence: str | None = None
	clinical_review_reference: str | None = None
	phvwp_submission_reference: str | None = None
	closed_date: datetime | None = None
	closure_reason: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("signal_type")
	@classmethod
	def validate_signal_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_SIGNAL_TYPES
		if v not in SUPPORTED_SIGNAL_TYPES:
			raise ValueError(f"signal_type must be one of {SUPPORTED_SIGNAL_TYPES}")
		return v


class PsurReport(PviBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	report_number: str
	product_id: str
	report_type: str
	data_lock_point: datetime
	international_birth_date: datetime
	period_start: datetime
	period_end: datetime
	ibrd_reference: str | None = None
	benefit_risk_assessed: bool = False
	signal_evaluation_reference: str | None = None
	submission_date: datetime | None = None
	status: str = "draft"
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("report_type")
	@classmethod
	def validate_report_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_PSUR_TYPES
		if v not in SUPPORTED_PSUR_TYPES:
			raise ValueError(f"report_type must be one of {SUPPORTED_PSUR_TYPES}")
		return v


class LiteratureRecord(PviBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	database_source: str
	article_reference: str
	title: str
	authors: str | None = None
	publication_date: datetime | None = None
	screened_at: datetime = Field(default_factory=datetime.utcnow)
	relevant: bool | None = None
	assessed_by: str | None = None
	product_id: str | None = None
	case_created: bool = False
	case_id: str | None = None
	duplicate: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


class FollowUpRequest(PviBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	case_id: str
	follow_up_type: str
	requested_from: str
	request_date: datetime = Field(default_factory=datetime.utcnow)
	due_date: datetime
	status: str = "requested"
	response_date: datetime | None = None
	response_reference: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("follow_up_type")
	@classmethod
	def validate_follow_up_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_FOLLOW_UP_TYPES
		if v not in SUPPORTED_FOLLOW_UP_TYPES:
			raise ValueError(f"follow_up_type must be one of {SUPPORTED_FOLLOW_UP_TYPES}")
		return v
