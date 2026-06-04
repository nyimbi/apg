"""Pydantic v2 models for Property Valuation (val)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class ValuationMethod(str, Enum):
	dcf = "dcf"
	comparable_sales = "comparable_sales"
	investment_method = "investment_method"
	residual_method = "residual_method"
	cost_method = "cost_method"
	profits_method = "profits_method"
	mass_appraisal = "mass_appraisal"
	desk_review = "desk_review"
	drive_by = "drive_by"
	full_inspection = "full_inspection"


class ValuationPurpose(str, Enum):
	mortgage_security = "mortgage_security"
	insurance_reinstatement = "insurance_reinstatement"
	purchase = "purchase"
	sale = "sale"
	financial_reporting = "financial_reporting"
	ifrs16_commencement = "ifrs16_commencement"
	rating_appeal = "rating_appeal"
	compulsory_purchase = "compulsory_purchase"
	inheritance_tax = "inheritance_tax"
	rental_review = "rental_review"


class ValuationStatus(str, Enum):
	instructed = "instructed"
	in_progress = "in_progress"
	draft_issued = "draft_issued"
	under_review = "under_review"
	approved = "approved"
	signed_off = "signed_off"
	published = "published"
	superseded = "superseded"
	challenged = "challenged"


class ComparableType(str, Enum):
	sale = "sale"
	lease = "lease"
	letting = "letting"
	auction = "auction"
	off_market = "off_market"
	distressed = "distressed"


class YieldType(str, Enum):
	net_initial_yield = "net_initial_yield"
	equivalent_yield = "equivalent_yield"
	reversionary_yield = "reversionary_yield"
	running_yield = "running_yield"
	true_equivalent_yield = "true_equivalent_yield"
	net_income_yield = "net_income_yield"


class ValuerGrade(str, Enum):
	rics_registered = "rics_registered"
	rics_fellow = "rics_fellow"
	api_registered = "api_registered"
	internal_valuer = "internal_valuer"
	external_valuer = "external_valuer"
	independent_valuer = "independent_valuer"


class ReportType(str, Enum):
	desktop_valuation = "desktop_valuation"
	restricted_report = "restricted_report"
	full_red_book = "full_red_book"
	market_appraisal = "market_appraisal"
	reinstatement_cost_assessment = "reinstatement_cost_assessment"
	schedule_of_condition = "schedule_of_condition"
	mass_appraisal_report = "mass_appraisal_report"


class MassAppraisalModel(str, Enum):
	regression = "regression"
	spatial_interpolation = "spatial_interpolation"
	hedonic_pricing = "hedonic_pricing"
	ai_avms = "ai_avms"
	comparable_grid = "comparable_grid"


# ── Valuer ────────────────────────────────────────────────────────────────────

class ValuerCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	grade: ValuerGrade
	registration_number: str | None = None
	firm_name: str | None = None
	email: str
	phone: str | None = None
	is_independent: bool = False
	specialisms: list[str] = Field(default_factory=list)
	created_by: str


class ValuerResponse(ValuerCreate):
	id: str = Field(default_factory=uuid7str)
	active_instructions: int = 0
	completed_valuations: int = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Comparable ────────────────────────────────────────────────────────────────

class ComparableCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	comparable_type: ComparableType
	address: str
	transaction_date: date
	price: Decimal
	currency: str = "KES"
	area: Decimal | None = None
	area_unit: str = "sqm"
	price_per_sqm: Decimal | None = None
	property_type: str | None = None
	adjustments: dict[str, Decimal] = Field(default_factory=dict)
	adjusted_price: Decimal | None = None
	source: str | None = None
	verified: bool = False
	created_by: str

	@field_validator("price")
	@classmethod
	def _positive_price(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("price must be positive")
		return v


class ComparableResponse(ComparableCreate):
	id: str = Field(default_factory=uuid7str)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Valuation ─────────────────────────────────────────────────────────────────

class ValuationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	valuation_method: ValuationMethod
	purpose: ValuationPurpose
	report_type: ReportType
	valuer_id: str
	instruction_date: date
	inspection_date: date | None = None
	currency: str = "KES"
	instructions: str | None = None
	created_by: str


class ValuationResponse(ValuationCreate):
	id: str = Field(default_factory=uuid7str)
	ref: str = ""
	status: ValuationStatus = ValuationStatus.instructed
	valuation_figure: Decimal | None = None
	capital_value: Decimal | None = None
	rental_value: Decimal | None = None
	comparable_ids: list[str] = Field(default_factory=list)
	signed_off_by: str | None = None
	published_at: datetime | None = None
	valuer_independent: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class ValuationUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: ValuationStatus | None = None
	valuation_figure: Decimal | None = None
	capital_value: Decimal | None = None
	rental_value: Decimal | None = None
	inspection_date: date | None = None


# ── DCF Model ─────────────────────────────────────────────────────────────────

class DcfModelCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	valuation_id: str
	property_id: str
	discount_rate: Decimal  # 0.03 – 0.30
	holding_period_years: int
	exit_yield: Decimal
	annual_rental_income: Decimal
	rental_growth_rate: Decimal = Decimal("0")
	void_period_months: int = 0
	capex_allowance: Decimal = Decimal("0")
	purchasers_costs_pct: Decimal = Decimal("0.05")
	currency: str = "KES"
	created_by: str

	@field_validator("discount_rate")
	@classmethod
	def _rate_range(cls, v: Decimal) -> Decimal:
		if not (Decimal("0.03") <= v <= Decimal("0.30")):
			raise ValueError("discount_rate must be between 0.03 and 0.30")
		return v

	@field_validator("holding_period_years")
	@classmethod
	def _positive_period(cls, v: int) -> int:
		if v < 1:
			raise ValueError("holding_period_years must be at least 1")
		return v


class DcfModelResponse(DcfModelCreate):
	id: str = Field(default_factory=uuid7str)
	npv: Decimal = Decimal("0")
	irr: Decimal | None = None
	capital_value: Decimal = Decimal("0")
	cash_flow_schedule: list[dict[str, Any]] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Valuation Roll Entry ───────────────────────────────────────────────────────

class ValuationRollEntryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	valuation_id: str
	effective_date: date
	valuation_figure: Decimal
	currency: str = "KES"
	next_review_date: date | None = None
	revaluation_trigger: str | None = None
	created_by: str

	@field_validator("valuation_figure")
	@classmethod
	def _positive_figure(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("valuation_figure must be positive")
		return v


class ValuationRollEntryResponse(ValuationRollEntryCreate):
	id: str = Field(default_factory=uuid7str)
	superseded: bool = False
	superseded_by_id: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Mass Appraisal Run ────────────────────────────────────────────────────────

class MassAppraisalRunCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	model_type: MassAppraisalModel
	run_date: date
	property_ids: list[str]
	model_calibrated: bool = False
	calibration_r_squared: Decimal | None = None
	created_by: str


class MassAppraisalRunResponse(MassAppraisalRunCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "pending"  # pending | running | completed | failed
	results: list[dict[str, Any]] = Field(default_factory=list)
	completed_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Valuation Challenge ───────────────────────────────────────────────────────

class ValuationChallengeCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	valuation_id: str
	raised_by: str
	grounds: str
	counter_evidence_document_ids: list[str] = Field(default_factory=list)
	counter_valuation_figure: Decimal | None = None
	currency: str = "KES"
	created_by: str

	@field_validator("counter_evidence_document_ids")
	@classmethod
	def _must_have_evidence(cls, v: list[str]) -> list[str]:
		if not v:
			raise ValueError("at least one counter_evidence document is required")
		return v


class ValuationChallengeResponse(ValuationChallengeCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "open"  # open | under_review | upheld | rejected | withdrawn
	reviewed_by: str | None = None
	resolution_notes: str | None = None
	resolved_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
