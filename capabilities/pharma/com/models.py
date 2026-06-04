"""Pydantic v2 models for APG Pharma Commercial Operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _uuid7str() -> str:
	return str(uuid7())


class ComBase(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class Territory(ComBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	territory_type: str
	name: str
	owner_id: str
	product_ids: list[str] = Field(default_factory=list)
	approval_reference: str
	region: str | None = None
	status: str = "active"
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("territory_type")
	@classmethod
	def validate_territory_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_TERRITORY_TYPES
		if v not in SUPPORTED_TERRITORY_TYPES:
			raise ValueError(f"territory_type must be one of {SUPPORTED_TERRITORY_TYPES}")
		return v


class TerritoryCreate(ComBase):
	tenant_id: str
	territory_type: str
	name: str
	owner_id: str
	product_ids: list[str] = Field(default_factory=list)
	approval_reference: str
	region: str | None = None
	created_by: str


class TerritoryUpdate(ComBase):
	name: str | None = None
	owner_id: str | None = None
	product_ids: list[str] | None = None
	region: str | None = None
	status: str | None = None


class SalesRep(ComBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	rep_type: str
	employee_id: str
	name: str
	territory_id: str
	quota: float
	certification_reference: str
	active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("rep_type")
	@classmethod
	def validate_rep_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_REP_TYPES
		if v not in SUPPORTED_REP_TYPES:
			raise ValueError(f"rep_type must be one of {SUPPORTED_REP_TYPES}")
		return v

	@field_validator("quota")
	@classmethod
	def validate_quota(cls, v: float) -> float:
		if v < 0:
			raise ValueError("quota must be non-negative")
		return v


class SalesRepCreate(ComBase):
	tenant_id: str
	rep_type: str
	employee_id: str
	name: str
	territory_id: str
	quota: float
	certification_reference: str
	created_by: str


class CallRecord(ComBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	rep_id: str
	physician_id: str
	call_type: str
	products_discussed: list[str]
	outcome: str
	call_date: datetime
	duration_minutes: int | None = None
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("call_type")
	@classmethod
	def validate_call_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_CALL_TYPES
		if v not in SUPPORTED_CALL_TYPES:
			raise ValueError(f"call_type must be one of {SUPPORTED_CALL_TYPES}")
		return v


class CallRecordCreate(ComBase):
	tenant_id: str
	rep_id: str
	physician_id: str
	call_type: str
	products_discussed: list[str]
	outcome: str
	call_date: datetime
	duration_minutes: int | None = None
	notes: str | None = None
	created_by: str


class SampleDispensing(ComBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	rep_id: str
	physician_id: str
	sample_type: str
	product_id: str
	lot_number: str
	expiry_date: str
	quantity: int
	hcp_signature_reference: str
	pdma_compliant: bool = False
	dispensed_at: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("sample_type")
	@classmethod
	def validate_sample_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_SAMPLE_TYPES
		if v not in SUPPORTED_SAMPLE_TYPES:
			raise ValueError(f"sample_type must be one of {SUPPORTED_SAMPLE_TYPES}")
		return v

	@field_validator("quantity")
	@classmethod
	def validate_quantity(cls, v: int) -> int:
		if v <= 0:
			raise ValueError("quantity must be positive")
		return v


class SampleDispensingCreate(ComBase):
	tenant_id: str
	rep_id: str
	physician_id: str
	sample_type: str
	product_id: str
	lot_number: str
	expiry_date: str
	quantity: int
	hcp_signature_reference: str
	pdma_compliant: bool = False
	created_by: str


class HcpInteraction(ComBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	rep_id: str
	hcp_id: str
	interaction_type: str
	products_discussed: list[str] = Field(default_factory=list)
	spend_amount: float = 0.0
	spend_category: str | None = None
	interaction_date: datetime
	venue: str | None = None
	pre_approval_reference: str | None = None
	receipt_reference: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("interaction_type")
	@classmethod
	def validate_interaction_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_INTERACTION_TYPES
		if v not in SUPPORTED_INTERACTION_TYPES:
			raise ValueError(f"interaction_type must be one of {SUPPORTED_INTERACTION_TYPES}")
		return v

	@field_validator("spend_amount")
	@classmethod
	def validate_spend_amount(cls, v: float) -> float:
		if v < 0:
			raise ValueError("spend_amount must be non-negative")
		return v


class HcpInteractionCreate(ComBase):
	tenant_id: str
	rep_id: str
	hcp_id: str
	interaction_type: str
	products_discussed: list[str] = Field(default_factory=list)
	spend_amount: float = 0.0
	spend_category: str | None = None
	interaction_date: datetime
	venue: str | None = None
	pre_approval_reference: str | None = None
	receipt_reference: str | None = None
	created_by: str


class CommercialPlan(ComBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	plan_name: str
	plan_period: str
	territory_ids: list[str]
	product_ids: list[str]
	status: str = "draft"
	approval_reference: str | None = None
	total_quota: float
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_PLAN_STATUSES
		if v not in SUPPORTED_PLAN_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_PLAN_STATUSES}")
		return v


class CommercialPlanCreate(ComBase):
	tenant_id: str
	plan_name: str
	plan_period: str
	territory_ids: list[str]
	product_ids: list[str]
	total_quota: float
	created_by: str


class TargetPhysician(ComBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	physician_id: str
	tier: str
	territory_id: str
	product_ids: list[str]
	call_frequency_per_quarter: int
	segmentation_reference: str
	active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("tier")
	@classmethod
	def validate_tier(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_TARGET_TIERS
		if v not in SUPPORTED_TARGET_TIERS:
			raise ValueError(f"tier must be one of {SUPPORTED_TARGET_TIERS}")
		return v


class AggregateSpendRecord(ComBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	hcp_id: str
	category: str
	amount: float
	currency: str = "USD"
	fiscal_year: str
	quarter: str | None = None
	receipt_reference: str | None = None
	pre_approval_reference: str | None = None
	hcp_consent_reference: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("category")
	@classmethod
	def validate_category(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_SPEND_CATEGORIES
		if v not in SUPPORTED_SPEND_CATEGORIES:
			raise ValueError(f"category must be one of {SUPPORTED_SPEND_CATEGORIES}")
		return v

	@field_validator("amount")
	@classmethod
	def validate_amount(cls, v: float) -> float:
		if v < 0:
			raise ValueError("amount must be non-negative")
		return v
