"""Agricultural Credit Scoring models — Pydantic v2."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str


class CreditRating(str, Enum):
	AAA = "AAA"
	AA = "AA"
	A = "A"
	BBB = "BBB"
	BB = "BB"
	B = "B"
	CCC = "CCC"
	D = "D"


class LoanStatus(str, Enum):
	APPLIED = "applied"
	SCORED = "scored"
	APPROVED = "approved"
	DISBURSED = "disbursed"
	REPAYING = "repaying"
	SETTLED = "settled"
	DEFAULTED = "defaulted"
	REJECTED = "rejected"


class CreditProfileCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farmer_id: str
	farm_parcel_ids: list[str] = Field(default_factory=list)
	years_farming: int | None = None
	crop_types: list[str] = Field(default_factory=list)
	avg_annual_yield_kg: float | None = None
	avg_annual_revenue: float | None = None
	mobile_money_account: str | None = None
	cooperative_member: bool = False
	cooperative_id: str | None = None
	notes: str | None = None


class CreditProfileResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farmer_id: str
	farm_parcel_ids: list[str]
	years_farming: int | None = None
	crop_types: list[str]
	avg_annual_yield_kg: float | None = None
	avg_annual_revenue: float | None = None
	mobile_money_account: str | None = None
	cooperative_member: bool
	cooperative_id: str | None = None
	credit_score: float | None = None
	rating: CreditRating | None = None
	last_scored_at: str | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


class CreditScoreResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farmer_id: str
	credit_score: float
	rating: CreditRating
	max_loan_amount: float
	recommended_rate_pct: float
	factors: dict[str, float] = Field(default_factory=dict)
	scored_at: str


class LoanApplicationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farmer_id: str
	amount: float
	currency: str = "KES"
	purpose: str
	season: str
	duration_months: int
	collateral_description: str | None = None
	guarantor_id: str | None = None
	group_id: str | None = None
	notes: str | None = None


class LoanApplicationUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: LoanStatus | None = None
	approved_amount: float | None = None
	interest_rate_pct: float | None = None
	disbursed_at: str | None = None
	notes: str | None = None


class LoanApplicationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farmer_id: str
	amount: float
	currency: str
	purpose: str
	season: str
	duration_months: int
	collateral_description: str | None = None
	guarantor_id: str | None = None
	group_id: str | None = None
	status: LoanStatus = LoanStatus.APPLIED
	credit_score: float | None = None
	approved_amount: float | None = None
	interest_rate_pct: float | None = None
	disbursed_at: str | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


class CollateralCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farmer_id: str
	description: str
	estimated_value: float
	currency: str = "KES"
	asset_type: str
	reference_number: str | None = None
	notes: str | None = None


class CollateralResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farmer_id: str
	description: str
	estimated_value: float
	currency: str
	asset_type: str
	reference_number: str | None = None
	pledged_to_loan: str | None = None
	notes: str | None = None
	created_at: str


class GroupLendingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	group_name: str
	member_ids: list[str]
	loan_amount: float
	currency: str = "KES"
	season: str
	duration_months: int
	purpose: str


class GroupLendingResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	group_name: str
	member_ids: list[str]
	loan_amount: float
	per_member_amount: float
	currency: str
	season: str
	duration_months: int
	purpose: str
	status: LoanStatus = LoanStatus.APPLIED
	created_at: str
	updated_at: str


class AuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	payload: dict[str, Any] = Field(default_factory=dict)
	occurred_at: str
