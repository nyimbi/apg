"""
Pydantic v2 models for APG Digital Lending.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import math
from datetime import datetime, date
from enum import Enum
from typing import Any, Annotated

from pydantic import BaseModel, ConfigDict, Field, AfterValidator, field_validator, model_validator

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


def _positive(v: float) -> float:
	assert v > 0, "must be positive"
	return v


def _non_negative(v: float) -> float:
	assert v >= 0, "must be non-negative"
	return v


def _rate(v: float) -> float:
	assert 0 < v <= 1.0, f"rate must be in (0, 1.0], got {v}"
	return v


def _score(v: int) -> int:
	assert 300 <= v <= 850, f"credit score must be 300–850, got {v}"
	return v


PositiveFloat = Annotated[float, AfterValidator(_positive)]
NonNegativeFloat = Annotated[float, AfterValidator(_non_negative)]
RateFloat = Annotated[float, AfterValidator(_rate)]
CreditScoreInt = Annotated[int, AfterValidator(_score)]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LoanApplicationStatus(str, Enum):
	DRAFT = "draft"
	SUBMITTED = "submitted"
	UNDER_REVIEW = "under_review"
	REFERRED = "referred"
	CONDITIONALLY_APPROVED = "conditionally_approved"
	APPROVED = "approved"
	DECLINED = "declined"
	WITHDRAWN = "withdrawn"
	DISBURSED = "disbursed"
	EXPIRED = "expired"


class LoanStatus(str, Enum):
	ACTIVE = "active"
	SETTLED = "settled"
	WRITTEN_OFF = "written_off"
	RESTRUCTURED = "restructured"
	CLOSED = "closed"
	CANCELLED = "cancelled"


class LoanProductType(str, Enum):
	TERM_LOAN = "term_loan"
	REVOLVING = "revolving"
	OVERDRAFT = "overdraft"
	MICROFINANCE = "microfinance"
	MORTGAGE = "mortgage"
	ASSET_FINANCE = "asset_finance"
	INVOICE_DISCOUNTING = "invoice_discounting"
	BNPL = "bnpl"
	SALARY_ADVANCE = "salary_advance"
	EMERGENCY = "emergency"
	AGRI = "agri"
	SME = "sme"
	GROUP = "group"


class RepaymentFrequency(str, Enum):
	DAILY = "daily"
	WEEKLY = "weekly"
	BIWEEKLY = "biweekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	BULLET = "bullet"


class ScheduleType(str, Enum):
	REDUCING_BALANCE = "reducing_balance"
	FLAT_RATE = "flat_rate"
	BULLET = "bullet"
	INTEREST_ONLY = "interest_only"


class CreditRiskGrade(str, Enum):
	A = "A"
	B = "B"
	C = "C"
	D = "D"
	E = "E"
	F = "F"


class OfferStatus(str, Enum):
	DRAFT = "draft"
	ISSUED = "issued"
	ACCEPTED = "accepted"
	REJECTED = "rejected"
	EXPIRED = "expired"
	WITHDRAWN = "withdrawn"


class RepaymentStatus(str, Enum):
	PENDING = "pending"
	PARTIAL = "partial"
	PAID = "paid"
	OVERDUE = "overdue"
	WAIVED = "waived"
	WRITTEN_OFF = "written_off"


class DelinquencyStatus(str, Enum):
	OPEN = "open"
	MONITORING = "monitoring"
	COLLECTIONS = "collections"
	LEGAL = "legal"
	RESOLVED = "resolved"
	WRITTEN_OFF = "written_off"


class RestructureType(str, Enum):
	TENOR_EXTENSION = "tenor_extension"
	RATE_REDUCTION = "rate_reduction"
	CAPITALISE_ARREARS = "capitalise_arrears"
	PAYMENT_HOLIDAY = "payment_holiday"
	FULL_RESTRUCTURE = "full_restructure"


class CollateralType(str, Enum):
	PROPERTY = "property"
	VEHICLE = "vehicle"
	CASH = "cash"
	SHARES = "shares"
	INVENTORY = "inventory"
	MACHINERY = "machinery"
	LAND = "land"
	OTHER = "other"


class CollateralStatus(str, Enum):
	PLEDGED = "pledged"
	HELD = "held"
	RELEASED = "released"
	FORECLOSED = "foreclosed"


class WriteOffReason(str, Enum):
	NON_PERFORMING = "non_performing"
	BANKRUPTCY = "bankruptcy"
	DEATH = "death"
	FRAUD = "fraud"
	STATUTE_OF_LIMITATIONS = "statute_of_limitations"
	REGULATORY = "regulatory"
	OTHER = "other"


class IncomeSource(str, Enum):
	EMPLOYED = "employed"
	SELF_EMPLOYED = "self_employed"
	BUSINESS = "business"
	MOBILE_MONEY = "mobile_money"
	RENTAL = "rental"
	PENSION = "pension"
	AGRICULTURE = "agriculture"
	GIGS = "gigs"
	OTHER = "other"


class UnderwritingDecision(str, Enum):
	APPROVE = "approve"
	DECLINE = "decline"
	REFER = "refer"
	CONDITIONAL_APPROVE = "conditional_approve"


class ProvisionStage(str, Enum):
	STAGE1 = "stage1"
	STAGE2 = "stage2"
	STAGE3 = "stage3"


class DisbursementRail(str, Enum):
	BANK_TRANSFER = "bank_transfer"
	MOBILE_MONEY = "mobile_money"
	CASH = "cash"
	CHEQUE = "cheque"
	INTERNAL = "internal"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class LendingBase(BaseModel):
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
		use_enum_values=True,
	)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"
	is_deleted: bool = False


# ---------------------------------------------------------------------------
# Loan Product
# ---------------------------------------------------------------------------

class LoanProductCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	code: str
	name: str
	product_type: LoanProductType
	currency: str = "KES"
	min_amount: PositiveFloat
	max_amount: PositiveFloat
	min_tenor_months: int = Field(ge=1)
	max_tenor_months: int = Field(ge=1)
	base_annual_rate: RateFloat
	repayment_frequency: RepaymentFrequency = RepaymentFrequency.MONTHLY
	schedule_type: ScheduleType = ScheduleType.REDUCING_BALANCE
	processing_fee_pct: float = Field(default=0.0, ge=0, le=0.10)
	insurance_fee_pct: float = Field(default=0.0, ge=0, le=0.05)
	late_penalty_pct: float = Field(default=0.02, ge=0, le=0.10)
	early_settlement_fee_pct: float = Field(default=0.01, ge=0, le=0.05)
	max_dsr: float = Field(default=0.40, ge=0.10, le=0.70)
	requires_collateral: bool = False
	requires_guarantor: bool = False
	is_active: bool = True

	@model_validator(mode="after")
	def _check_tenor(self) -> LoanProductCreate:
		assert self.max_tenor_months >= self.min_tenor_months, "max_tenor_months must be >= min_tenor_months"
		assert self.max_amount >= self.min_amount, "max_amount must be >= min_amount"
		return self


class LoanProductUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name: str | None = None
	base_annual_rate: float | None = None
	max_amount: float | None = None
	is_active: bool | None = None
	late_penalty_pct: float | None = None
	processing_fee_pct: float | None = None


class LoanProductResponse(LendingBase):
	code: str
	name: str
	product_type: str
	currency: str
	min_amount: float
	max_amount: float
	min_tenor_months: int
	max_tenor_months: int
	base_annual_rate: float
	repayment_frequency: str
	schedule_type: str
	processing_fee_pct: float
	insurance_fee_pct: float
	late_penalty_pct: float
	early_settlement_fee_pct: float
	max_dsr: float
	requires_collateral: bool
	requires_guarantor: bool
	is_active: bool


# ---------------------------------------------------------------------------
# Loan Application
# ---------------------------------------------------------------------------

class LoanApplicationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	borrower_id: str
	product_id: str
	requested_amount: PositiveFloat
	requested_tenor_months: int = Field(ge=1)
	currency: str = "KES"
	purpose: str
	income_source: IncomeSource
	monthly_income: PositiveFloat
	employer_or_business: str = ""
	bank_statement_ref: str = ""
	payslip_ref: str = ""
	kyc_ref: str
	aml_ref: str = ""
	fraud_ref: str = ""
	notes: str = ""


class LoanApplicationUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: LoanApplicationStatus | None = None
	notes: str | None = None
	requested_amount: float | None = None
	requested_tenor_months: int | None = None


class LoanApplicationResponse(LendingBase):
	borrower_id: str
	product_id: str
	requested_amount: float
	requested_tenor_months: int
	currency: str
	purpose: str
	income_source: str
	monthly_income: float
	employer_or_business: str
	bank_statement_ref: str
	payslip_ref: str
	kyc_ref: str
	aml_ref: str
	fraud_ref: str
	status: str
	notes: str
	underwriter_id: str | None = None
	decision_date: date | None = None
	decline_reason: str | None = None


# ---------------------------------------------------------------------------
# Credit Score
# ---------------------------------------------------------------------------

class CreditScoreCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	borrower_id: str
	application_id: str | None = None
	bureau_score: int = Field(ge=300, le=900)
	bureau_name: str = ""
	behavioural_score: int = Field(ge=300, le=850)
	demographic_score: int = Field(ge=300, le=850)
	payment_ratio: float = Field(ge=0, le=1)
	utilisation_ratio: float = Field(ge=0, le=1)
	defaults_count: int = Field(ge=0, default=0)
	fraud_flags: list[str] = Field(default_factory=list)
	income_verified: bool = False
	components: dict[str, Any] = Field(default_factory=dict)


class CreditScoreResponse(LendingBase):
	borrower_id: str
	application_id: str | None
	composite_score: CreditScoreInt
	risk_grade: str
	probability_of_default: float
	bureau_score: int
	bureau_name: str
	behavioural_score: int
	demographic_score: int
	payment_ratio: float
	utilisation_ratio: float
	defaults_count: int
	fraud_flags: list[str]
	income_verified: bool
	components: dict[str, Any]
	computed_at: date


# ---------------------------------------------------------------------------
# Loan Offer
# ---------------------------------------------------------------------------

class LoanOfferCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	application_id: str
	credit_score_id: str
	tier: str = "standard"  # conservative | standard | aggressive
	offered_amount: PositiveFloat
	currency: str = "KES"
	annual_rate: RateFloat
	tenor_months: int = Field(ge=1)
	schedule_type: ScheduleType = ScheduleType.REDUCING_BALANCE
	repayment_frequency: RepaymentFrequency = RepaymentFrequency.MONTHLY
	processing_fee: float = Field(default=0.0, ge=0)
	insurance_fee: float = Field(default=0.0, ge=0)
	conditions: list[str] = Field(default_factory=list)
	expiry_date: date


class LoanOfferUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: OfferStatus | None = None
	conditions: list[str] | None = None


class LoanOfferResponse(LendingBase):
	application_id: str
	credit_score_id: str
	tier: str
	offered_amount: float
	currency: str
	annual_rate: float
	monthly_rate: float
	tenor_months: int
	monthly_emi: float
	total_repayable: float
	total_interest: float
	processing_fee: float
	insurance_fee: float
	total_cost: float
	schedule_type: str
	repayment_frequency: str
	conditions: list[str]
	status: str
	expiry_date: date
	accepted_at: datetime | None = None


# ---------------------------------------------------------------------------
# Loan
# ---------------------------------------------------------------------------

class LoanCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	application_id: str
	offer_id: str
	borrower_id: str
	product_id: str
	principal: PositiveFloat
	currency: str
	annual_rate: RateFloat
	tenor_months: int = Field(ge=1)
	schedule_type: ScheduleType = ScheduleType.REDUCING_BALANCE
	repayment_frequency: RepaymentFrequency = RepaymentFrequency.MONTHLY
	disbursement_date: date
	bank_account: str
	disbursement_rail: DisbursementRail = DisbursementRail.BANK_TRANSFER


class LoanResponse(LendingBase):
	application_id: str
	offer_id: str
	borrower_id: str
	product_id: str
	principal: float
	outstanding_principal: float
	currency: str
	annual_rate: float
	tenor_months: int
	schedule_type: str
	repayment_frequency: str
	disbursement_date: date
	bank_account: str
	disbursement_rail: str
	status: str
	max_dpd: int = 0
	delinquency_stage: str | None = None
	provision_stage: str | None = None
	total_repaid: float = 0.0
	total_interest_paid: float = 0.0
	collateral_ids: list[str] = Field(default_factory=list)
	guarantor_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Loan Schedule (individual installment)
# ---------------------------------------------------------------------------

class LoanScheduleItem(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	loan_id: str
	tenant_id: str
	installment_no: int
	due_date: date
	emi: float
	principal_portion: float
	interest_portion: float
	opening_balance: float
	closing_balance: float
	status: RepaymentStatus = RepaymentStatus.PENDING
	paid_amount: float = 0.0
	paid_date: date | None = None
	dpd: int = 0


class LoanScheduleResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	loan_id: str
	tenant_id: str
	schedule_type: str
	principal: float
	annual_rate: float
	tenor_months: int
	monthly_emi: float
	total_repayable: float
	total_interest: float
	currency: str
	installments: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Repayment Transaction
# ---------------------------------------------------------------------------

class RepaymentTransactionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	loan_id: str
	amount: PositiveFloat
	payment_date: date
	payment_method: str
	reference: str
	channel: str = "branch"
	notes: str = ""


class RepaymentTransactionResponse(LendingBase):
	loan_id: str
	amount: float
	payment_date: date
	payment_method: str
	reference: str
	channel: str
	fees_cleared: float
	interest_cleared: float
	principal_cleared: float
	overpayment: float
	outstanding_principal_after: float
	loan_status_after: str
	allocations: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Delinquency
# ---------------------------------------------------------------------------

class DelinquencyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	loan_id: str
	borrower_id: str
	dpd_days: int = Field(ge=1)
	overdue_amount: NonNegativeFloat
	currency: str


class DelinquencyResponse(LendingBase):
	loan_id: str
	borrower_id: str
	dpd_days: int
	delinquency_bucket: str
	overdue_amount: float
	currency: str
	status: str
	assigned_collector_id: str | None = None
	collection_activities: list[dict[str, Any]] = Field(default_factory=list)
	demand_notices: list[dict[str, Any]] = Field(default_factory=list)
	legal_actions: list[dict[str, Any]] = Field(default_factory=list)
	opened_at: date
	resolved_at: date | None = None


# ---------------------------------------------------------------------------
# Restructure
# ---------------------------------------------------------------------------

class RestructureCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	loan_id: str
	restructure_type: RestructureType
	new_annual_rate: float | None = None
	new_tenor_months: int | None = None
	capitalise_arrears: bool = False
	payment_holiday_months: int = 0
	reason: str
	approved_by: str
	conditions: list[str] = Field(default_factory=list)


class RestructureResponse(LendingBase):
	loan_id: str
	restructure_type: str
	old_annual_rate: float
	new_annual_rate: float
	old_tenor_months: int
	new_tenor_months: int
	old_outstanding: float
	new_outstanding: float
	capitalise_arrears: bool
	payment_holiday_months: int
	arrears_capitalised: float
	reason: str
	approved_by: str
	conditions: list[str]
	new_monthly_emi: float
	effective_date: date


# ---------------------------------------------------------------------------
# Write-Off
# ---------------------------------------------------------------------------

class WriteOffCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	loan_id: str
	reason: WriteOffReason
	write_off_date: date
	approved_by: str
	recovery_prospect: float = Field(default=0.0, ge=0, le=1)
	notes: str = ""


class WriteOffResponse(LendingBase):
	loan_id: str
	write_off_amount: float
	fees_written_off: float
	total_written_off: float
	reason: str
	write_off_date: date
	approved_by: str
	recovery_prospect: float
	currency: str
	notes: str


# ---------------------------------------------------------------------------
# Collateral
# ---------------------------------------------------------------------------

class CollateralItemCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	loan_id: str
	collateral_type: CollateralType
	description: str
	market_value: PositiveFloat
	currency: str = "KES"
	registration_number: str = ""
	valuation_date: date | None = None
	valuer_name: str = ""
	location: str = ""
	insurance_policy_ref: str = ""


class CollateralItemResponse(LendingBase):
	loan_id: str
	collateral_type: str
	description: str
	market_value: float
	forced_sale_value: float
	haircut_pct: float
	currency: str
	registration_number: str
	valuation_date: date | None
	valuer_name: str
	location: str
	insurance_policy_ref: str
	status: str
	released_by: str | None = None
	release_date: date | None = None
	release_reason: str | None = None


# ---------------------------------------------------------------------------
# Guarantor
# ---------------------------------------------------------------------------

class GuarantorRecordCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	loan_id: str
	guarantor_name: str
	id_number: str
	phone: str
	email: str = ""
	relationship: str
	guaranteed_amount: PositiveFloat
	currency: str = "KES"
	consent_ref: str
	kyc_ref: str = ""


class GuarantorRecordResponse(LendingBase):
	loan_id: str
	guarantor_name: str
	id_number: str
	phone: str
	email: str
	relationship: str
	guaranteed_amount: float
	currency: str
	consent_ref: str
	kyc_ref: str
	is_active: bool = True


# ---------------------------------------------------------------------------
# Portfolio Analysis
# ---------------------------------------------------------------------------

class PortfolioAnalysisResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	as_of_date: date
	tenant_id: str
	total_active_loans: int
	total_book: float
	total_disbursed: float
	average_ticket: float
	portfolio_yield: float
	par_30: float
	par_60: float
	par_90: float
	npl_ratio: float
	npl_balance: float
	written_off_total: float
	stage1_ecl: float
	stage2_ecl: float
	stage3_ecl: float
	total_ecl: float
	provision_coverage_ratio: float
	currency: str = "KES"


class VintageCohort(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	cohort: str
	disbursed_count: int
	written_off_count: int
	total_principal: float
	outstanding: float
	default_rate: float


class VintageAnalysisResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	cohort_months: int
	as_of_date: date
	cohorts: list[VintageCohort]
	total_cohort_count: int


class EarlySettlementResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	loan_id: str
	settlement_date: date
	outstanding_principal: float
	accrued_interest: float
	early_settlement_fee: float
	total_settlement_amount: float
	currency: str
	saving_vs_full_term: float


class IFRS9ProvisionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	method: str
	as_of_date: date
	stage1: dict[str, Any]
	stage2: dict[str, Any]
	stage3: dict[str, Any]
	total_ecl: float
	total_outstanding: float
	provision_coverage_ratio: float


class DisbursementResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	loan_id: str
	application_id: str
	offer_id: str
	principal: float
	currency: str
	disbursement_date: str
	bank_account: str
	rail: str
	status: str
	schedule_summary: dict[str, Any]


class AmortisationScheduleRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	principal: PositiveFloat
	annual_rate: RateFloat
	tenor_months: int = Field(ge=1)
	start_date: date
	schedule_type: ScheduleType = ScheduleType.REDUCING_BALANCE
	currency: str = "KES"


class PaginatedResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	items: list[dict[str, Any]]
	total: int
	page: int
	page_size: int
	pages: int


class ErrorResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	error: str
	detail: str | None = None
	code: str | None = None
