"""Pydantic v2 models for APG Pharma Quality Management System."""

from __future__ import annotations

from datetime import datetime
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _uuid7str() -> str:
	return str(uuid7())


class QmsBase(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class ChangeControl(QmsBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	change_number: str
	title: str
	change_type: str
	description: str
	status: str = "draft"
	gmp_impact: bool = False
	regulatory_impact: bool = False
	impact_assessment_reference: str | None = None
	risk_assessment_reference: str | None = None
	approval_reference: str | None = None
	implementation_date: datetime | None = None
	effectiveness_check_date: datetime | None = None
	effectiveness_check_reference: str | None = None
	raised_by: str
	raised_date: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("change_type")
	@classmethod
	def validate_change_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_CHANGE_TYPES
		if v not in SUPPORTED_CHANGE_TYPES:
			raise ValueError(f"change_type must be one of {SUPPORTED_CHANGE_TYPES}")
		return v

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_CHANGE_STATUSES
		if v not in SUPPORTED_CHANGE_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_CHANGE_STATUSES}")
		return v


class ChangeControlCreate(QmsBase):
	tenant_id: str
	change_number: str
	title: str
	change_type: str
	description: str
	raised_by: str
	created_by: str


class CapaRecord(QmsBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	capa_number: str
	capa_type: str
	title: str
	description: str
	source_reference: str
	status: str = "open"
	root_cause: str | None = None
	root_cause_method: str | None = None
	action_plan: str | None = None
	target_completion_date: datetime | None = None
	actual_completion_date: datetime | None = None
	effectiveness_check_date: datetime | None = None
	effectiveness_result: str | None = None
	owner_id: str
	overdue: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("capa_type")
	@classmethod
	def validate_capa_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_CAPA_TYPES
		if v not in SUPPORTED_CAPA_TYPES:
			raise ValueError(f"capa_type must be one of {SUPPORTED_CAPA_TYPES}")
		return v

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_CAPA_STATUSES
		if v not in SUPPORTED_CAPA_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_CAPA_STATUSES}")
		return v


class CapaCreate(QmsBase):
	tenant_id: str
	capa_number: str
	capa_type: str
	title: str
	description: str
	source_reference: str
	owner_id: str
	target_completion_date: datetime | None = None
	created_by: str


class QmsDeviation(QmsBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	deviation_number: str
	deviation_type: str
	severity: str
	description: str
	status: str = "open"
	affected_products: list[str] = Field(default_factory=list)
	affected_batches: list[str] = Field(default_factory=list)
	root_cause: str | None = None
	capa_reference: str | None = None
	gmp_impact: bool = False
	raised_by: str
	raised_date: datetime = Field(default_factory=datetime.utcnow)
	closed_date: datetime | None = None
	recurring: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("deviation_type")
	@classmethod
	def validate_deviation_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_DEVIATION_TYPES
		if v not in SUPPORTED_DEVIATION_TYPES:
			raise ValueError(f"deviation_type must be one of {SUPPORTED_DEVIATION_TYPES}")
		return v


class ControlledDocument(QmsBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	document_number: str
	title: str
	document_type: str
	version: str
	status: str = "draft"
	department: str
	owner_id: str
	approver_id: str | None = None
	effective_date: datetime | None = None
	next_review_date: datetime | None = None
	superseded_by: str | None = None
	storage_reference: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("document_type")
	@classmethod
	def validate_document_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_DOCUMENT_TYPES
		if v not in SUPPORTED_DOCUMENT_TYPES:
			raise ValueError(f"document_type must be one of {SUPPORTED_DOCUMENT_TYPES}")
		return v

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_DOCUMENT_STATUSES
		if v not in SUPPORTED_DOCUMENT_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_DOCUMENT_STATUSES}")
		return v


class QualityAudit(QmsBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	audit_number: str
	audit_type: str
	auditee: str
	auditor_ids: list[str]
	status: str = "planned"
	planned_date: datetime | None = None
	conducted_date: datetime | None = None
	report_reference: str | None = None
	findings_count: int = 0
	capa_references: list[str] = Field(default_factory=list)
	scope: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("audit_type")
	@classmethod
	def validate_audit_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_AUDIT_TYPES
		if v not in SUPPORTED_AUDIT_TYPES:
			raise ValueError(f"audit_type must be one of {SUPPORTED_AUDIT_TYPES}")
		return v


class ValidationRecord(QmsBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	validation_number: str
	validation_type: str
	subject: str
	status: str = "planned"
	protocol_reference: str | None = None
	protocol_approved_by: str | None = None
	protocol_approval_date: datetime | None = None
	report_reference: str | None = None
	report_approved_by: str | None = None
	report_approval_date: datetime | None = None
	revalidation_due: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("validation_type")
	@classmethod
	def validate_validation_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_VALIDATION_TYPES
		if v not in SUPPORTED_VALIDATION_TYPES:
			raise ValueError(f"validation_type must be one of {SUPPORTED_VALIDATION_TYPES}")
		return v


class RiskAssessment(QmsBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	assessment_number: str
	subject: str
	risk_level: str
	likelihood: str
	impact: str
	risk_score: float | None = None
	mitigation_required: bool = False
	mitigation_actions: list[str] = Field(default_factory=list)
	residual_risk_level: str | None = None
	residual_risk_assessed: bool = False
	owner_id: str
	review_date: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("risk_level")
	@classmethod
	def validate_risk_level(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_RISK_LEVELS
		if v not in SUPPORTED_RISK_LEVELS:
			raise ValueError(f"risk_level must be one of {SUPPORTED_RISK_LEVELS}")
		return v
