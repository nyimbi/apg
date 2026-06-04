"""Pydantic v2 models for APG Pharma Product Registration."""

from __future__ import annotations

from datetime import datetime
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _uuid7str() -> str:
	return str(uuid7())


class RegBase(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class ProductRegistration(RegBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	registration_number: str | None = None
	product_id: str
	product_name: str
	product_type: str
	registration_type: str
	region: str
	status: str = "not_submitted"
	dossier_id: str | None = None
	qp_signed_off: bool = False
	local_representative_id: str | None = None
	submission_date: datetime | None = None
	approval_date: datetime | None = None
	expiry_date: datetime | None = None
	renewal_initiated: bool = False
	conditions_of_approval: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("registration_type")
	@classmethod
	def validate_registration_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_REGISTRATION_TYPES
		if v not in SUPPORTED_REGISTRATION_TYPES:
			raise ValueError(f"registration_type must be one of {SUPPORTED_REGISTRATION_TYPES}")
		return v

	@field_validator("region")
	@classmethod
	def validate_region(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_REGULATORY_REGIONS
		if v not in SUPPORTED_REGULATORY_REGIONS:
			raise ValueError(f"region must be one of {SUPPORTED_REGULATORY_REGIONS}")
		return v

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_APPROVAL_STATUSES
		if v not in SUPPORTED_APPROVAL_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_APPROVAL_STATUSES}")
		return v

	@field_validator("product_type")
	@classmethod
	def validate_product_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_PRODUCT_TYPES
		if v not in SUPPORTED_PRODUCT_TYPES:
			raise ValueError(f"product_type must be one of {SUPPORTED_PRODUCT_TYPES}")
		return v


class ProductRegistrationCreate(RegBase):
	tenant_id: str
	product_id: str
	product_name: str
	product_type: str
	registration_type: str
	region: str
	created_by: str


class RegistrationDossier(RegBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	dossier_number: str
	product_id: str
	format: str
	version: str
	modules_present: list[str] = Field(default_factory=list)
	ectd_validated: bool = False
	completeness_checked: bool = False
	storage_reference: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("format")
	@classmethod
	def validate_format(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_DOSSIER_FORMATS
		if v not in SUPPORTED_DOSSIER_FORMATS:
			raise ValueError(f"format must be one of {SUPPORTED_DOSSIER_FORMATS}")
		return v


class AuthorityInteraction(RegBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	registration_id: str
	interaction_type: str
	authority: str
	interaction_date: datetime
	minutes_reference: str | None = None
	action_items: list[str] = Field(default_factory=list)
	follow_up_required: bool = False
	follow_up_due: datetime | None = None
	participants: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("interaction_type")
	@classmethod
	def validate_interaction_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_AUTHORITY_INTERACTIONS
		if v not in SUPPORTED_AUTHORITY_INTERACTIONS:
			raise ValueError(f"interaction_type must be one of {SUPPORTED_AUTHORITY_INTERACTIONS}")
		return v


class RegistrationVariation(RegBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	variation_number: str
	registration_id: str
	variation_type: str
	description: str
	impact_assessed: bool = False
	dossier_supplement_reference: str | None = None
	submission_date: datetime | None = None
	approval_date: datetime | None = None
	status: str = "draft"
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("variation_type")
	@classmethod
	def validate_variation_type(cls, v: str) -> str:
		valid = ["variation_type_ia", "variation_type_ib", "variation_type_ii", "extension"]
		if v not in valid:
			raise ValueError(f"variation_type must be one of {valid}")
		return v


class RegistrationCertificate(RegBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	certificate_number: str
	registration_id: str
	product_id: str
	region: str
	authority: str
	issued_date: datetime
	expiry_date: datetime | None = None
	storage_reference: str
	conditions: list[str] = Field(default_factory=list)
	active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


class RegistrationProcedure(RegBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	procedure_number: str
	registration_id: str
	procedure_type: str
	reference_member_state: str | None = None
	concerned_member_states: list[str] = Field(default_factory=list)
	status: str = "initiated"
	start_date: datetime | None = None
	end_date: datetime | None = None
	concerns: list[str] = Field(default_factory=list)
	outcome: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("procedure_type")
	@classmethod
	def validate_procedure_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_PROCEDURE_TYPES
		if v not in SUPPORTED_PROCEDURE_TYPES:
			raise ValueError(f"procedure_type must be one of {SUPPORTED_PROCEDURE_TYPES}")
		return v
