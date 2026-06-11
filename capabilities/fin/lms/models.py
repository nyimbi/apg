"""Loan Management System — Domain Models.
© 2025 Datacraft. Author: Nyimbi Odero

Pydantic v2 models for the post-origination loan lifecycle:
disbursement → repayment → arrears → restructuring → write-off → recovery.

All monetary fields are Decimal. CBK/Basel classification thresholds enforced.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())

except ImportError:
	import uuid

	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


# ── Enumerations ──────────────────────────────────────────────────────────────

class LoanStatus(str, Enum):
	PENDING_DISBURSEMENT = "pending_disbursement"
	ACTIVE               = "active"
	IN_ARREARS           = "in_arrears"
	NPA                  = "npa"                  # Non-Performing Asset
	MORATORIUM           = "moratorium"
	RESTRUCTURED         = "restructured"
	WRITTEN_OFF          = "written_off"
	CLOSED               = "closed"
	RECOVERED            = "recovered"


class AmortisationMethod(str, Enum):
	REDUCING_BALANCE = "reducing_balance"   # Equal instalment, diminishing interest
	FLAT_RATE        = "flat_rate"           # Interest on original principal throughout
	FRENCH_ANNUITY   = "french_annuity"      # Constant payment (PMT formula)
	BULLET           = "bullet"              # Principal at maturity, periodic interest
	INTEREST_ONLY    = "interest_only"       # Interest only, principal at end


class RestructureType(str, Enum):
	EXTEND_TENOR      = "extend_tenor"       # Push out maturity
	REDUCE_RATE       = "reduce_rate"        # Lower interest rate
	CAPITALISE_ARREARS = "capitalise_arrears" # Roll arrears into principal
	CONVERT_TO_TERM   = "convert_to_term"    # Convert revolving to term


class MoratoriumType(str, Enum):
	FULL           = "full"           # No payments (interest may still accrue)
	PRINCIPAL_ONLY = "principal_only" # Interest still paid, principal deferred


class PenaltyType(str, Enum):
	LATE_FEE      = "late_fee"       # One-time flat charge
	DAILY_PENALTY = "daily_penalty"  # Per-day accrual on overdue balance


class PaymentMethod(str, Enum):
	MOBILE_MONEY = "mobile_money"
	BANK_TRANSFER = "bank_transfer"
	CASH         = "cash"
	CHEQUE       = "cheque"
	DIRECT_DEBIT = "direct_debit"
	STANDING_ORDER = "standing_order"


class LoanClassification(str, Enum):
	"""CBK/Basel II loan classification by days past due."""
	PERFORMING   = "performing"    # 0–29 DPD
	WATCH        = "watch"         # 30–89 DPD
	SUBSTANDARD  = "substandard"   # 90–179 DPD
	DOUBTFUL     = "doubtful"      # 180–359 DPD
	LOSS         = "loss"          # 360+ DPD


class ClosureReason(str, Enum):
	FULLY_PAID   = "fully_paid"
	WRITTEN_OFF  = "written_off"
	RESTRUCTURED = "restructured"


class DemandNoticeType(str, Enum):
	REMINDER       = "reminder"
	FORMAL_DEMAND  = "formal_demand"
	LEGAL          = "legal"


# ── CBK Provisioning Matrix ───────────────────────────────────────────────────

# CBK Prudential Guideline CBK/PG/15 — minimum provision rates
CBK_PROVISION_RATES: dict[LoanClassification, Decimal] = {
	LoanClassification.PERFORMING:  Decimal("0.01"),   # 1% general provision
	LoanClassification.WATCH:       Decimal("0.03"),   # 3%
	LoanClassification.SUBSTANDARD: Decimal("0.20"),   # 20%
	LoanClassification.DOUBTFUL:    Decimal("0.50"),   # 50%
	LoanClassification.LOSS:        Decimal("1.00"),   # 100%
}

# DPD thresholds per CBK classification
CBK_DPD_THRESHOLDS: list[tuple[int, LoanClassification]] = [
	(360, LoanClassification.LOSS),
	(180, LoanClassification.DOUBTFUL),
	(90,  LoanClassification.SUBSTANDARD),
	(30,  LoanClassification.WATCH),
	(0,   LoanClassification.PERFORMING),
]


# ── Core Pydantic Models ──────────────────────────────────────────────────────

class Installment(BaseModel):
	"""Single row of an amortisation schedule."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	installment_no: int
	due_date: date
	principal:  Decimal = Field(ge=Decimal("0"))
	interest:   Decimal = Field(ge=Decimal("0"))
	total:      Decimal = Field(ge=Decimal("0"))
	balance:    Decimal = Field(ge=Decimal("0"))   # outstanding after payment
	# Actuals (populated once repayment posted)
	paid_date:      date | None = None
	paid_amount:    Decimal | None = None
	paid_principal: Decimal | None = None
	paid_interest:  Decimal | None = None
	status:         str = "pending"   # pending / partial / paid / overdue


class Repayment(BaseModel):
	"""A payment received against a loan."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id:             str = Field(default_factory=uuid7str)
	loan_id:        str
	tenant_id:      str
	amount:         Decimal = Field(gt=Decimal("0"))
	payment_date:   date
	payment_ref:    str
	payment_method: PaymentMethod
	# Waterfall allocation
	penalty_cleared: Decimal = Decimal("0")
	fees_cleared:    Decimal = Decimal("0")
	interest_cleared: Decimal = Decimal("0")
	principal_cleared: Decimal = Decimal("0")
	unallocated:     Decimal = Decimal("0")
	gl_entry_id:     str | None = None
	created_at:      str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")


class ArrearsPosition(BaseModel):
	"""Current arrears snapshot for a loan."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	loan_id:             str
	tenant_id:           str
	as_of_date:          date
	days_past_due:       int = 0
	amount_in_arrears:   Decimal = Decimal("0")
	installments_missed: int = 0
	penalty_accrued:     Decimal = Decimal("0")
	total_overdue:       Decimal = Decimal("0")   # arrears + penalties
	npa_status:          bool = False
	classification:      LoanClassification = LoanClassification.PERFORMING
	calculated_at:       str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")


class Moratorium(BaseModel):
	"""Moratorium (payment holiday) record."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id:             str = Field(default_factory=uuid7str)
	loan_id:        str
	tenant_id:      str
	from_date:      date
	to_date:        date
	moratorium_type: MoratoriumType
	interest_accrues: bool = True   # configurable
	reason:         str
	approved_by:    str
	created_at:     str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")


class Restructure(BaseModel):
	"""Loan restructuring event."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id:              str = Field(default_factory=uuid7str)
	loan_id:         str
	tenant_id:       str
	restructure_type: RestructureType
	new_terms:       dict[str, Any]     # tenor_months, rate, etc.
	effective_date:  date
	approved_by:     str
	gl_entry_id:     str | None = None
	created_at:      str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")


class WriteOff(BaseModel):
	"""Write-off record."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id:              str = Field(default_factory=uuid7str)
	loan_id:         str
	tenant_id:       str
	write_off_date:  date
	reason:          str
	approved_by:     str
	write_off_amount: Decimal = Field(gt=Decimal("0"))
	gl_entry_id:     str | None = None
	created_at:      str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")


class Recovery(BaseModel):
	"""Post write-off recovery record."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id:            str = Field(default_factory=uuid7str)
	loan_id:       str
	tenant_id:     str
	amount:        Decimal = Field(gt=Decimal("0"))
	recovery_date: date
	method:        str
	gl_entry_id:   str | None = None
	created_at:    str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")


class LoanProvision(BaseModel):
	"""Provision entry for a loan."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id:               str = Field(default_factory=uuid7str)
	loan_id:          str
	tenant_id:        str
	classification:   LoanClassification
	outstanding_balance: Decimal
	provision_rate:   Decimal
	required_provision: Decimal
	posted_provision:   Decimal = Decimal("0")
	posting_date:     date | None = None
	gl_entry_id:      str | None = None
	created_at:       str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")


class Loan(BaseModel):
	"""Full loan record — lifecycle state machine."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id:                  str = Field(default_factory=uuid7str)
	tenant_id:           str
	customer_id:         str
	product_code:        str
	principal:           Decimal = Field(gt=Decimal("0"))
	disbursed_amount:    Decimal = Field(default=Decimal("0"))
	outstanding_balance: Decimal = Field(default=Decimal("0"))
	rate:                Decimal = Field(gt=Decimal("0"))  # annual % as decimal e.g. 0.14
	tenor_months:        int = Field(gt=0)
	method:              AmortisationMethod = AmortisationMethod.REDUCING_BALANCE
	disbursement_date:   date | None = None
	first_payment_date:  date | None = None
	maturity_date:       date | None = None
	account_id:          str | None = None
	disbursement_ref:    str | None = None
	status:              LoanStatus = LoanStatus.PENDING_DISBURSEMENT
	currency:            str = "KES"
	# Penalty config
	late_fee_amount:     Decimal = Decimal("0")
	daily_penalty_rate:  Decimal = Decimal("0.001")  # 0.1% per day default
	# Tracking
	days_past_due:       int = 0
	total_penalties:     Decimal = Decimal("0")
	total_fees:          Decimal = Decimal("0")
	total_interest_paid: Decimal = Decimal("0")
	total_principal_paid: Decimal = Decimal("0")
	write_off_amount:    Decimal = Decimal("0")
	recovered_amount:    Decimal = Decimal("0")
	classification:      LoanClassification = LoanClassification.PERFORMING
	# Collections
	referred_to_collections: bool = False
	collections_referred_at: str | None = None
	collections_referred_by: str | None = None
	collections_notes:        str | None = None
	# Demand notices
	last_notice_type:    DemandNoticeType | None = None
	last_notice_date:    str | None = None
	# Closure
	closure_date:        date | None = None
	closure_reason:      ClosureReason | None = None
	# Audit
	created_at:          str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
	updated_at:          str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")

	@field_validator("rate")
	@classmethod
	def rate_reasonable(cls, v: Decimal) -> Decimal:
		assert Decimal("0") < v <= Decimal("1"), "rate must be between 0 and 1 (e.g. 0.14 for 14%)"
		return v


class StatementLine(BaseModel):
	"""Single line in a loan statement."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	date:        date
	description: str
	debit:       Decimal = Decimal("0")
	credit:      Decimal = Decimal("0")
	balance:     Decimal = Decimal("0")
	ref:         str | None = None


class PortfolioQuality(BaseModel):
	"""Portfolio quality metrics for a tenant at a given date."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id:          str
	as_of_date:         date
	total_loans:        int
	total_portfolio:    Decimal
	npl_amount:         Decimal   # Non-performing loans (NPA)
	npl_ratio:          Decimal   # npl/total
	par_30_amount:      Decimal   # Portfolio At Risk > 30 DPD
	par_30_ratio:       Decimal
	par_90_amount:      Decimal   # Portfolio At Risk > 90 DPD
	par_90_ratio:       Decimal
	total_provisions:   Decimal
	provision_coverage: Decimal   # provisions / npl
	written_off_amount: Decimal
	recovered_amount:   Decimal
	by_classification:  dict[str, dict[str, Any]] = Field(default_factory=dict)
	calculated_at:      str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")


class GLEntry(BaseModel):
	"""Lightweight GL entry record posted by LMS operations."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id:          str = Field(default_factory=uuid7str)
	tenant_id:   str
	loan_id:     str
	entry_type:  str    # disbursement / repayment / penalty / write_off / recovery / provision / restructure
	description: str
	dr_account:  str
	cr_account:  str
	amount:      Decimal = Field(gt=Decimal("0"))
	currency:    str = "KES"
	posting_date: date
	ref:         str | None = None
	created_at:  str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
