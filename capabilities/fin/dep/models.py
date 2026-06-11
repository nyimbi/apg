"""Deposit Products Engine — data models.

All monetary values are Decimal.  Pydantic v2, modern typing, tabs.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.functional_validators import AfterValidator
from typing import Annotated


# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────

class ProductType(str, Enum):
	CURRENT        = "CURRENT"
	SAVINGS        = "SAVINGS"
	TERM_DEPOSIT   = "TERM_DEPOSIT"
	CALL_DEPOSIT   = "CALL_DEPOSIT"
	NOTICE_DEPOSIT = "NOTICE_DEPOSIT"


class InterestCalculationType(str, Enum):
	SIMPLE        = "SIMPLE"
	COMPOUND      = "COMPOUND"
	DAILY_ACCRUAL = "DAILY_ACCRUAL"


class CompoundingFrequency(str, Enum):
	DAILY    = "DAILY"
	MONTHLY  = "MONTHLY"
	ANNUALLY = "ANNUALLY"


class FeeFrequency(str, Enum):
	MONTHLY   = "MONTHLY"
	QUARTERLY = "QUARTERLY"


class MaturityInstruction(str, Enum):
	ROLLOVER = "ROLLOVER"
	PAYOUT   = "PAYOUT"
	PARTIAL  = "PARTIAL"


class ProductStatus(str, Enum):
	ACTIVE   = "ACTIVE"
	INACTIVE = "INACTIVE"
	DRAFT    = "DRAFT"


# ─────────────────────────────────────────────────────────────
# Validators
# ─────────────────────────────────────────────────────────────

def _positive_decimal(v: Decimal) -> Decimal:
	if v < Decimal("0"):
		raise ValueError("Value must be non-negative")
	return v


def _rate_range(v: Decimal) -> Decimal:
	if not (Decimal("0") <= v <= Decimal("100")):
		raise ValueError("Rate must be between 0 and 100")
	return v


PositiveDecimal = Annotated[Decimal, AfterValidator(_positive_decimal)]
RateDecimal     = Annotated[Decimal, AfterValidator(_rate_range)]


# ─────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────

class InterestTier(BaseModel):
	"""Tiered interest rate band: applies when balance >= min_balance."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	min_balance: PositiveDecimal = Field(description="Lower balance boundary (inclusive)")
	rate:        RateDecimal     = Field(description="Annual rate % for this tier")
	description: str             = Field(default="", description="Human-readable label")


class InterestConfig(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	rate:             RateDecimal              = Field(description="Base annual rate %")
	calculation:      InterestCalculationType  = Field(default=InterestCalculationType.DAILY_ACCRUAL)
	compounding:      CompoundingFrequency     = Field(default=CompoundingFrequency.MONTHLY)
	tiers:            list[InterestTier]       = Field(default_factory=list, description="Optional tiered rates, sorted asc by min_balance")
	withholding_rate: RateDecimal              = Field(default=Decimal("0"), description="WHT % deducted at source")


class FeeConfig(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	maintenance_fee:   PositiveDecimal = Field(default=Decimal("0"))
	fee_frequency:     FeeFrequency    = Field(default=FeeFrequency.MONTHLY)
	minimum_balance:   PositiveDecimal = Field(default=Decimal("0"))
	below_minimum_fee: PositiveDecimal = Field(default=Decimal("0"))


class ProductTerms(BaseModel):
	"""Flexible term constraints per product type."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	min_tenor_days:     int             = Field(default=0,  ge=0)
	max_tenor_days:     int             = Field(default=0,  ge=0, description="0 = no max")
	notice_period_days: int             = Field(default=0,  ge=0)
	auto_rollover:      bool            = Field(default=False)
	rollover_rate_delta: Decimal        = Field(default=Decimal("0"), description="Rate adjustment on rollover")
	break_penalty_rate: RateDecimal     = Field(default=Decimal("0"), description="% of interest forfeited on early break")
	tax_exempt:         bool            = Field(default=False)
	allowed_currencies: list[str]       = Field(default_factory=list)
	max_balance:        Decimal | None  = Field(default=None)
	min_opening_amount: PositiveDecimal = Field(default=Decimal("0"))


# ─────────────────────────────────────────────────────────────
# Core models
# ─────────────────────────────────────────────────────────────

class DepositProduct(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id:              str            = Field(description="UUID7 primary key")
	tenant_id:       str
	code:            str            = Field(description="Unique product code within tenant")
	name:            str
	product_type:    ProductType
	currency:        str            = Field(description="ISO 4217 3-letter code")
	interest_config: InterestConfig
	fee_config:      FeeConfig
	terms:           ProductTerms
	status:          ProductStatus  = Field(default=ProductStatus.ACTIVE)
	created_at:      datetime
	updated_at:      datetime
	created_by:      str            = Field(default="system")
	gl_interest_income_account: str = Field(default="", description="GL account for interest income")
	gl_interest_payable_account: str = Field(default="", description="GL accrual account")
	gl_wht_payable_account:     str = Field(default="", description="GL WHT liability account")

	@field_validator("currency")
	@classmethod
	def _validate_currency(cls, v: str) -> str:
		if len(v) != 3 or not v.isupper():
			raise ValueError("Currency must be ISO 4217 3-letter uppercase code")
		return v


class RateHistoryEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id:             str
	tenant_id:      str
	product_code:   str
	old_rate:       RateDecimal
	new_rate:       RateDecimal
	effective_date: date
	reason:         str
	changed_by:     str
	changed_at:     datetime


class AccrualEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id:              str
	tenant_id:       str
	account_id:      str
	product_code:    str
	accrual_date:    date
	gross_amount:    PositiveDecimal
	wht_amount:      PositiveDecimal
	net_amount:      PositiveDecimal
	posted:          bool     = False
	posting_ref:     str      = ""
	batch_ref:       str      = ""


class InterestPostingEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id:             str
	tenant_id:      str
	account_id:     str
	product_code:   str
	value_date:     date
	gross_interest: PositiveDecimal
	wht_amount:     PositiveDecimal
	net_interest:   PositiveDecimal
	posting_ref:    str
	gl_ref:         str      = ""
	posted_at:      datetime


class MaturityRecord(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id:              str
	tenant_id:       str
	account_id:      str
	product_code:    str
	maturity_date:   date
	principal:       PositiveDecimal
	interest_earned: PositiveDecimal
	instruction:     MaturityInstruction
	rollover_ref:    str = ""
	payout_ref:      str = ""
	processed_at:    datetime


# ─────────────────────────────────────────────────────────────
# Response / result models (returned to callers)
# ─────────────────────────────────────────────────────────────

class InterestCalculationResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	gross_interest:  Decimal
	withholding_tax: Decimal
	net_interest:    Decimal
	accrual_days:    int
	rate_applied:    Decimal
	calculation_type: str
	tier_breakdown:  list[dict[str, Any]] = Field(default_factory=list)


class MinimumBalanceCheck(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	account_id:     str
	meets_minimum:  bool
	current_balance: Decimal
	minimum_required: Decimal
	shortfall:      Decimal
	fee_applicable: bool


class BatchAccrualResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id:         str
	accrual_date:      date
	accounts_processed: int
	total_accrued:     Decimal
	entries_posted:    int
	errors:            list[str] = Field(default_factory=list)
	idempotent_hit:    bool      = False


class SimulationResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	product_code:    str
	principal:       Decimal
	tenor_days:      int
	gross_interest:  Decimal
	withholding_tax: Decimal
	net_interest:    Decimal
	maturity_amount: Decimal
	effective_rate:  Decimal
	annual_rate:     Decimal


class WithholdingTaxEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	account_id:   str
	product_code: str
	period_start: date
	period_end:   date
	gross_amount: Decimal
	wht_amount:   Decimal
	posted_at:    datetime
