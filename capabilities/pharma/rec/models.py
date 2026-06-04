"""Pydantic v2 models for APG Pharma Regulatory Compliance."""

from __future__ import annotations

from datetime import datetime
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _uuid7str() -> str:
	return str(uuid7())


class RecBase(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class ComplianceFrameworkRecord(RecBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	framework: str
	title: str
	applicable_sites: list[str]
	applicable_products: list[str] = Field(default_factory=list)
	status: str = "active"
	gap_assessment_reference: str | None = None
	implementation_plan_reference: str | None = None
	last_review_date: datetime | None = None
	next_review_date: datetime | None = None
	owner_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("framework")
	@classmethod
	def validate_framework(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_REGULATORY_FRAMEWORKS
		if v not in SUPPORTED_REGULATORY_FRAMEWORKS:
			raise ValueError(f"framework must be one of {SUPPORTED_REGULATORY_FRAMEWORKS}")
		return v


class InspectionRecord(RecBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	inspection_number: str
	inspection_type: str
	authority: str
	site: str
	announced: bool = True
	start_date: datetime | None = None
	end_date: datetime | None = None
	status: str = "planned"
	outcome: str | None = None
	findings_count: int = 0
	capa_references: list[str] = Field(default_factory=list)
	response_deadline: datetime | None = None
	response_submitted_date: datetime | None = None
	lead_inspector: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("inspection_type")
	@classmethod
	def validate_inspection_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_AUDIT_TYPES
		if v not in SUPPORTED_AUDIT_TYPES:
			raise ValueError(f"inspection_type must be one of {SUPPORTED_AUDIT_TYPES}")
		return v


class LabelRecord(RecBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	label_number: str
	product_id: str
	market: str
	language: str
	version: str
	change_type: str
	status: str = "draft"
	qp_approved: bool = False
	qp_approval_date: datetime | None = None
	artwork_approved: bool = False
	artwork_approval_date: datetime | None = None
	effective_date: datetime | None = None
	superseded_by: str | None = None
	storage_reference: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("change_type")
	@classmethod
	def validate_change_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_LABEL_CHANGE_TYPES
		if v not in SUPPORTED_LABEL_CHANGE_TYPES:
			raise ValueError(f"change_type must be one of {SUPPORTED_LABEL_CHANGE_TYPES}")
		return v


class PostMarketSurveillanceRecord(RecBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	pms_number: str
	product_id: str
	pms_type: str
	protocol_reference: str | None = None
	protocol_approved: bool = False
	report_reference: str | None = None
	report_submitted_date: datetime | None = None
	status: str = "planned"
	start_date: datetime | None = None
	end_date: datetime | None = None
	signals_identified: int = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("pms_type")
	@classmethod
	def validate_pms_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_PMS_TYPES
		if v not in SUPPORTED_PMS_TYPES:
			raise ValueError(f"pms_type must be one of {SUPPORTED_PMS_TYPES}")
		return v


class RegulatoryIntelligenceRecord(RecBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	intel_number: str
	intel_type: str
	region: str
	title: str
	description: str
	source_url: str | None = None
	published_date: datetime | None = None
	effective_date: datetime | None = None
	impact_assessed: bool = False
	impact_assessment_reference: str | None = None
	disseminated: bool = False
	dissemination_reference: str | None = None
	products_affected: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("intel_type")
	@classmethod
	def validate_intel_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_INTEL_TYPES
		if v not in SUPPORTED_INTEL_TYPES:
			raise ValueError(f"intel_type must be one of {SUPPORTED_INTEL_TYPES}")
		return v


class RegulatoryCommitment(RecBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	commitment_number: str
	product_id: str
	authority: str
	description: str
	status: str = "open"
	milestones: list[dict] = Field(default_factory=list)
	due_date: datetime
	completed_date: datetime | None = None
	submission_reference: str | None = None
	overdue: bool = False
	escalated: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_COMMITMENT_STATUSES
		if v not in SUPPORTED_COMMITMENT_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_COMMITMENT_STATUSES}")
		return v


class GapAssessment(RecBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	assessment_number: str
	framework: str
	site: str
	conducted_date: datetime
	conducted_by: str
	gaps_identified: int = 0
	critical_gaps: int = 0
	major_gaps: int = 0
	minor_gaps: int = 0
	implementation_plan_reference: str | None = None
	capa_references: list[str] = Field(default_factory=list)
	next_assessment_date: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
