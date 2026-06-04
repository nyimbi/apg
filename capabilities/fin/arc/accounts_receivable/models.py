"""
APG Accounts Receivable — Pydantic v2 Data Models

Complete domain model covering every entity, enum, status lifecycle, and
aggregation shape required by the AR capability.

© 2025 Datacraft · Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated
from uuid6 import uuid7


# ---------------------------------------------------------------------------
# UUID helper
# ---------------------------------------------------------------------------

def uuid7str() -> str:
	return str(uuid7())


# ---------------------------------------------------------------------------
# Validated scalar types
# ---------------------------------------------------------------------------

def _validate_positive(v: Decimal) -> Decimal:
	if v < Decimal("0"):
		raise ValueError("amount must be >= 0")
	return v

def _validate_currency(v: str) -> str:
	v = v.strip().upper()
	if len(v) != 3 or not v.isalpha():
		raise ValueError(f"Invalid ISO-4217 currency code: {v!r}")
	return v

def _validate_rate(v: Decimal) -> Decimal:
	if v <= Decimal("0"):
		raise ValueError("exchange rate must be > 0")
	return v


NonNegativeAmount = Annotated[Decimal, AfterValidator(_validate_positive)]
CurrencyCode      = Annotated[str,     AfterValidator(_validate_currency)]
PositiveRate      = Annotated[Decimal, AfterValidator(_validate_rate)]


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class ARBase(BaseModel):
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
		populate_by_name=True,
	)

	id:         str      = Field(default_factory=uuid7str)
	tenant_id:  str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	is_deleted: bool     = False


# ===========================================================================
# Enumerations
# ===========================================================================

class CreditStatus(str, Enum):
	ACTIVE      = "active"
	ON_HOLD     = "on_hold"
	SUSPENDED   = "suspended"
	BLACKLISTED = "blacklisted"


class CreditTerms(str, Enum):
	NET_30       = "Net30"
	NET_60       = "Net60"
	NET_90       = "Net90"
	EARLY_10_30  = "2_10_Net30"   # 2% discount if paid within 10 days
	COD          = "COD"
	CIA          = "CIA"           # Cash in advance
	NET_7        = "Net7"
	NET_14       = "Net14"
	NET_45       = "Net45"
	IMMEDIATE    = "Immediate"


class InvoiceStatus(str, Enum):
	DRAFT           = "draft"
	SUBMITTED       = "submitted"
	APPROVED        = "approved"
	POSTED          = "posted"
	PARTIALLY_PAID  = "partially_paid"
	FULLY_PAID      = "fully_paid"
	DISPUTED        = "disputed"
	WRITTEN_OFF     = "written_off"
	CANCELLED       = "cancelled"
	CORRECTION_PENDING = "correction_pending"  # period-close correction in progress


class InvoiceType(str, Enum):
	STANDARD      = "standard"
	CREDIT_NOTE   = "credit_note"
	DEBIT_NOTE    = "debit_note"
	PROFORMA      = "proforma"
	INTERCOMPANY  = "intercompany"
	PREPAYMENT    = "prepayment"    # advance invoice


class PaymentStatus(str, Enum):
	UNALLOCATED         = "unallocated"
	PARTIALLY_ALLOCATED = "partially_allocated"
	FULLY_ALLOCATED     = "fully_allocated"
	BOUNCED             = "bounced"
	REVERSED            = "reversed"
	CANCELLED           = "cancelled"
	SUSPENSE            = "suspense"   # received before invoice exists


class PaymentMethod(str, Enum):
	BANK_TRANSFER  = "bank_transfer"
	CHEQUE         = "cheque"
	CASH           = "cash"
	MPESA          = "mpesa"
	AIRTEL_MONEY   = "airtel_money"
	CARD           = "card"
	DIRECT_DEBIT   = "direct_debit"
	RTGS           = "rtgs"
	SWIFT          = "swift"
	CRYPTO         = "crypto"


class DisputeType(str, Enum):
	PRICING     = "pricing"
	QUANTITY    = "quantity"
	QUALITY     = "quality"
	DELIVERY    = "delivery"
	DUPLICATE   = "duplicate"
	ALREADY_PAID = "already_paid"
	OTHER       = "other"


class DisputeStatus(str, Enum):
	OPEN               = "open"
	UNDER_INVESTIGATION = "under_investigation"
	RESOLVED           = "resolved"
	REJECTED           = "rejected"
	ESCALATED          = "escalated"


class DisputeResolution(str, Enum):
	CREDIT_NOTE = "credit_note"
	PAYMENT     = "payment"
	WRITE_OFF   = "write_off"
	NO_ACTION   = "no_action"
	PARTIAL_CREDIT = "partial_credit"


class WriteOffApprovalStatus(str, Enum):
	PENDING   = "pending"
	APPROVED  = "approved"
	REJECTED  = "rejected"


class DunningLevel(int, Enum):
	REMINDER    = 1
	FIRST       = 2
	SECOND      = 3
	FINAL       = 4
	LEGAL       = 5


class RevenueRecognitionMethod(str, Enum):
	POINT_IN_TIME = "point_in_time"
	OVER_TIME     = "over_time"
	MILESTONE     = "milestone"
	USAGE_BASED   = "usage_based"


class TaxType(str, Enum):
	VAT           = "VAT"
	GST           = "GST"
	WHT           = "WHT"   # Withholding Tax
	REVERSE_CHARGE = "reverse_charge"
	EXEMPT        = "exempt"
	ZERO_RATED    = "zero_rated"


class AllocationMethod(str, Enum):
	FIFO       = "fifo"       # oldest invoice first
	SPECIFIC   = "specific"   # operator-chosen
	PRO_RATA   = "pro_rata"   # proportional to outstanding
	LIFO       = "lifo"       # newest first


class CustomerInsolvencyStatus(str, Enum):
	NONE             = "none"
	UNDER_REVIEW     = "under_review"
	LIQUIDATION      = "liquidation"
	RESTRUCTURING    = "restructuring"
	ADMINISTRATION   = "administration"
	DISCHARGED       = "discharged"


# ===========================================================================
# Address / Contact embedded models
# ===========================================================================

class ARAddress(BaseModel):
	model_config = ConfigDict(extra="forbid")

	line1:       str
	line2:       str | None = None
	city:        str
	state:       str | None = None
	postal_code: str | None = None
	country:     str                # ISO-3166-1 alpha-2


class ARContactPerson(BaseModel):
	model_config = ConfigDict(extra="forbid")

	name:  str
	email: str | None = None
	phone: str | None = None
	role:  str | None = None       # e.g. "accounts payable"


# ===========================================================================
# ARCustomer
# ===========================================================================

class ARCustomerBase(BaseModel):
	model_config = ConfigDict(extra="forbid")

	customer_number:           str
	legal_name:                str
	trading_name:              str | None = None
	credit_limit:              NonNegativeAmount = Decimal("0")
	credit_terms:              CreditTerms       = CreditTerms.NET_30
	credit_status:             CreditStatus      = CreditStatus.ACTIVE
	credit_score:              int               = Field(default=500, ge=0, le=1000)
	payment_behaviour_score:   int               = Field(default=500, ge=0, le=1000)
	currency_code:             CurrencyCode      = "USD"
	tax_registration:          str | None        = None
	address:                   ARAddress | None  = None
	billing_address:           ARAddress | None  = None
	contact_person:            ARContactPerson | None = None
	ar_account_code:           str               = "1200"
	dunning_group:             str               = "standard"
	dispute_threshold:         NonNegativeAmount = Decimal("0")
	withholding_tax_applicable: bool             = False
	withholding_tax_rate:      Decimal           = Field(default=Decimal("0"), ge=0, le=100)
	industry_code:             str | None        = None
	country_of_incorporation:  str | None        = None    # ISO-3166-1 alpha-2
	insolvency_status:         CustomerInsolvencyStatus = CustomerInsolvencyStatus.NONE
	insolvency_date:           date | None       = None
	statute_limitation_years:  int               = 6       # jurisdiction-specific
	intercompany:              bool              = False
	counterpart_entity_id:     str | None        = None    # for intercompany matching


class ARCustomerCreate(ARCustomerBase):
	tenant_id:  str
	created_by: str


class ARCustomerUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	legal_name:                str | None = None
	trading_name:              str | None = None
	credit_limit:              NonNegativeAmount | None = None
	credit_terms:              CreditTerms | None = None
	credit_status:             CreditStatus | None = None
	credit_score:              int | None = None
	payment_behaviour_score:   int | None = None
	currency_code:             CurrencyCode | None = None
	address:                   ARAddress | None = None
	billing_address:           ARAddress | None = None
	contact_person:            ARContactPerson | None = None
	ar_account_code:           str | None = None
	dunning_group:             str | None = None
	dispute_threshold:         NonNegativeAmount | None = None
	withholding_tax_applicable: bool | None = None
	withholding_tax_rate:      Decimal | None = None
	insolvency_status:         CustomerInsolvencyStatus | None = None
	insolvency_date:           date | None = None


class ARCustomerResponse(ARBase, ARCustomerBase):
	pass


# ===========================================================================
# ARInvoiceLine
# ===========================================================================

class ARInvoiceLineBase(BaseModel):
	model_config = ConfigDict(extra="forbid")

	line_number:               int
	description:               str
	quantity:                  Decimal  = Decimal("1")
	unit_price:                Decimal  = Decimal("0")
	discount_pct:              Decimal  = Field(default=Decimal("0"), ge=0, le=100)
	line_subtotal:             Decimal  = Decimal("0")   # computed: qty * unit_price * (1 - discount/100)
	tax_code:                  str      = ""
	tax_type:                  TaxType  = TaxType.VAT
	tax_rate:                  Decimal  = Field(default=Decimal("0"), ge=0, le=100)
	tax_amount:                Decimal  = Decimal("0")
	gl_account:                str      = ""
	cost_center:               str | None = None
	project_ref:               str | None = None
	revenue_recognition_method: RevenueRecognitionMethod = RevenueRecognitionMethod.POINT_IN_TIME
	delivery_period_start:     date | None = None
	delivery_period_end:       date | None = None
	performance_obligation_id: str | None = None

	@model_validator(mode="after")
	def _compute_subtotal(self) -> "ARInvoiceLineBase":
		if self.line_subtotal == Decimal("0"):
			self.line_subtotal = (
				self.quantity * self.unit_price * (1 - self.discount_pct / 100)
			).quantize(Decimal("0.01"))
		if self.tax_amount == Decimal("0") and self.tax_rate > 0:
			self.tax_amount = (self.line_subtotal * self.tax_rate / 100).quantize(Decimal("0.01"))
		return self


class ARInvoiceLineCreate(ARInvoiceLineBase):
	invoice_id: str


class ARInvoiceLineResponse(ARInvoiceLineBase):
	id:         str = Field(default_factory=uuid7str)
	invoice_id: str


# ===========================================================================
# ARTaxLine
# ===========================================================================

class ARTaxLine(BaseModel):
	model_config = ConfigDict(extra="forbid")

	tax_code:    str
	tax_type:    TaxType
	tax_rate:    Decimal
	taxable_amount: Decimal
	tax_amount:  Decimal
	gl_account:  str = ""
	jurisdiction: str | None = None  # e.g. "KE", "NG", "EU"


# ===========================================================================
# ARInvoice
# ===========================================================================

class ARInvoiceBase(BaseModel):
	model_config = ConfigDict(extra="forbid")

	invoice_number:        str
	customer_id:           str
	invoice_date:          date
	due_date:              date
	posting_date:          date | None  = None
	period:                str | None   = None    # "2025-01" — accounting period
	currency:              CurrencyCode = "USD"
	exchange_rate:         PositiveRate = Decimal("1")    # to functional currency
	subtotal:              NonNegativeAmount = Decimal("0")
	tax_amount:            NonNegativeAmount = Decimal("0")
	withholding_tax_amount: NonNegativeAmount = Decimal("0")
	total:                 NonNegativeAmount = Decimal("0")
	allocated_amount:      NonNegativeAmount = Decimal("0")
	outstanding_amount:    NonNegativeAmount = Decimal("0")
	functional_total:      NonNegativeAmount = Decimal("0")  # total * exchange_rate
	status:                InvoiceStatus     = InvoiceStatus.DRAFT
	payment_terms:         CreditTerms       = CreditTerms.NET_30
	invoice_type:          InvoiceType       = InvoiceType.STANDARD
	purchase_order_ref:    str | None        = None
	delivery_ref:          str | None        = None
	reversal_of:           str | None        = None   # original invoice id (credit note)
	original_invoice_id:   str | None        = None   # for debit notes / corrections
	notes:                 str | None        = None
	internal_notes:        str | None        = None
	counterpart_entity_id: str | None        = None   # intercompany AP invoice id
	suspense_account:      str | None        = None   # when payment pre-dates invoice
	line_items:            list[ARInvoiceLineBase] = Field(default_factory=list)
	tax_lines:             list[ARTaxLine]          = Field(default_factory=list)

	@field_validator("due_date", mode="after")
	@classmethod
	def _due_after_invoice(cls, v: date, info: Any) -> date:
		# info.data may not contain invoice_date if validation failed earlier
		inv = info.data.get("invoice_date")
		if inv and v < inv and info.data.get("invoice_type") not in (
			InvoiceType.CREDIT_NOTE, InvoiceType.DEBIT_NOTE
		):
			raise ValueError("due_date must be >= invoice_date")
		return v


class ARInvoiceCreate(ARInvoiceBase):
	tenant_id:  str
	created_by: str


class ARInvoiceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	invoice_date:       date | None = None
	due_date:           date | None = None
	purchase_order_ref: str | None  = None
	delivery_ref:       str | None  = None
	notes:              str | None  = None
	internal_notes:     str | None  = None
	line_items:         list[ARInvoiceLineBase] | None = None


class ARInvoiceResponse(ARBase, ARInvoiceBase):
	approved_by:  str | None = None
	approved_at:  datetime | None = None
	posted_by:    str | None = None
	posted_at:    datetime | None = None
	cancelled_by: str | None = None
	cancelled_at: datetime | None = None
	write_off_id: str | None = None


# ===========================================================================
# ARPayment
# ===========================================================================

class ARPaymentBase(BaseModel):
	model_config = ConfigDict(extra="forbid")

	payment_number:     str
	payment_date:       date
	payment_method:     PaymentMethod
	bank_account:       str | None          = None
	amount:             NonNegativeAmount   = Decimal("0")
	currency:           CurrencyCode        = "USD"
	exchange_rate:      PositiveRate        = Decimal("1")
	functional_amount:  NonNegativeAmount   = Decimal("0")
	status:             PaymentStatus       = PaymentStatus.UNALLOCATED
	customer_id:        str
	reference:          str | None          = None
	bank_statement_ref: str | None          = None
	cheque_number:      str | None          = None
	reversal_of:        str | None          = None   # original payment id (bounced)
	suspense_account:   str | None          = None   # when customer not yet identified
	notes:              str | None          = None

	@model_validator(mode="after")
	def _set_functional(self) -> "ARPaymentBase":
		if self.functional_amount == Decimal("0") and self.amount > 0:
			self.functional_amount = (self.amount * self.exchange_rate).quantize(Decimal("0.01"))
		return self


class ARPaymentCreate(ARPaymentBase):
	tenant_id:  str
	created_by: str


class ARPaymentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	bank_account:       str | None = None
	reference:          str | None = None
	bank_statement_ref: str | None = None
	notes:              str | None = None


class ARPaymentResponse(ARBase, ARPaymentBase):
	unallocated_amount: NonNegativeAmount = Decimal("0")
	allocations:        list["ARAllocationResponse"] = Field(default_factory=list)
	reversed_by:        str | None = None


# ===========================================================================
# ARAllocation
# ===========================================================================

class ARAllocationBase(BaseModel):
	model_config = ConfigDict(extra="forbid")

	payment_id:            str
	invoice_id:            str
	allocated_amount:      NonNegativeAmount
	allocation_date:       date
	exchange_gain_loss:    Decimal           = Decimal("0")  # +ve gain, -ve loss
	discount_taken:        NonNegativeAmount = Decimal("0")
	withholding_tax_applied: NonNegativeAmount = Decimal("0")
	notes:                 str | None        = None


class ARAllocationCreate(ARAllocationBase):
	tenant_id:  str
	created_by: str


class ARAllocationResponse(ARBase, ARAllocationBase):
	pass


# ===========================================================================
# ARDispute
# ===========================================================================

class ARDisputeBase(BaseModel):
	model_config = ConfigDict(extra="forbid")

	dispute_number:    str
	invoice_id:        str
	dispute_type:      DisputeType
	dispute_amount:    NonNegativeAmount
	dispute_date:      date
	description:       str
	status:            DisputeStatus     = DisputeStatus.OPEN
	resolution_type:   DisputeResolution | None = None
	resolution_date:   date | None       = None
	resolution_notes:  str | None        = None
	credit_note_id:    str | None        = None  # if resolved by credit note
	assigned_to:       str | None        = None


class ARDisputeCreate(ARDisputeBase):
	tenant_id:  str
	created_by: str


class ARDisputeUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	status:           DisputeStatus | None   = None
	resolution_type:  DisputeResolution | None = None
	resolution_date:  date | None            = None
	resolution_notes: str | None             = None
	credit_note_id:   str | None             = None
	assigned_to:      str | None             = None


class ARDisputeResponse(ARBase, ARDisputeBase):
	resolved_by: str | None  = None
	resolved_at: datetime | None = None


# ===========================================================================
# ARDunningRun
# ===========================================================================

class ARDunningRunBase(BaseModel):
	model_config = ConfigDict(extra="forbid")

	run_date:           date
	dunning_level:      DunningLevel
	dunning_group:      str | None = None
	customer_ids:       list[str]  = Field(default_factory=list)
	total_overdue:      NonNegativeAmount = Decimal("0")
	letters_generated:  int        = 0
	emails_sent:        int        = 0
	calls_scheduled:    int        = 0
	sms_sent:           int        = 0
	status:             str        = "pending"   # pending / completed / failed


class ARDunningRunCreate(ARDunningRunBase):
	tenant_id:  str
	created_by: str


class ARDunningRunResponse(ARBase, ARDunningRunBase):
	completed_at: datetime | None = None
	error_log:    list[str]       = Field(default_factory=list)


# ===========================================================================
# ARDunningAction  (per-customer record within a run)
# ===========================================================================

class ARDunningAction(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id:               str = Field(default_factory=uuid7str)
	run_id:           str
	customer_id:      str
	tenant_id:        str
	dunning_level:    DunningLevel
	overdue_amount:   NonNegativeAmount
	oldest_due_date:  date
	action_taken:     str    # "email" | "letter" | "sms" | "call"
	sent_at:          datetime | None = None
	contact_email:    str | None = None
	letter_path:      str | None = None
	response:         str | None = None


# ===========================================================================
# ARWriteOff
# ===========================================================================

class ARWriteOffBase(BaseModel):
	model_config = ConfigDict(extra="forbid")

	write_off_number:    str
	invoice_ids:         list[str]
	write_off_amount:    NonNegativeAmount
	write_off_date:      date
	write_off_reason:    str
	provision_reversal:  bool              = False
	gl_account:          str               = "7000"   # bad debt expense
	tax_adjustment_required: bool          = False    # GST/VAT write-off adjustment
	tax_adjustment_amount:   NonNegativeAmount = Decimal("0")
	approval_status:     WriteOffApprovalStatus = WriteOffApprovalStatus.PENDING
	approved_by:         str | None        = None
	notes:               str | None        = None


class ARWriteOffCreate(ARWriteOffBase):
	tenant_id:  str
	created_by: str


class ARWriteOffResponse(ARBase, ARWriteOffBase):
	approved_at:  datetime | None = None
	rejected_by:  str | None      = None
	rejected_at:  datetime | None = None
	rejection_reason: str | None  = None


# ===========================================================================
# ARAgingBucket
# ===========================================================================

class ARAgingBucket(BaseModel):
	"""Aging snapshot for a single customer."""
	model_config = ConfigDict(extra="forbid")

	customer_id:        str
	customer_number:    str
	legal_name:         str
	currency:           CurrencyCode
	current_amount:     NonNegativeAmount  = Decimal("0")
	days_1_30:          NonNegativeAmount  = Decimal("0")
	days_31_60:         NonNegativeAmount  = Decimal("0")
	days_61_90:         NonNegativeAmount  = Decimal("0")
	days_91_120:        NonNegativeAmount  = Decimal("0")
	over_120:           NonNegativeAmount  = Decimal("0")
	total_outstanding:  NonNegativeAmount  = Decimal("0")
	functional_total:   NonNegativeAmount  = Decimal("0")
	as_of_date:         date
	credit_limit:       NonNegativeAmount  = Decimal("0")
	credit_utilisation: Decimal            = Decimal("0")  # %


class ARAgingReport(BaseModel):
	model_config = ConfigDict(extra="forbid")

	tenant_id:        str
	as_of_date:       date
	functional_currency: CurrencyCode
	buckets:          list[ARAgingBucket] = Field(default_factory=list)
	total_current:    NonNegativeAmount   = Decimal("0")
	total_1_30:       NonNegativeAmount   = Decimal("0")
	total_31_60:      NonNegativeAmount   = Decimal("0")
	total_61_90:      NonNegativeAmount   = Decimal("0")
	total_91_120:     NonNegativeAmount   = Decimal("0")
	total_over_120:   NonNegativeAmount   = Decimal("0")
	grand_total:      NonNegativeAmount   = Decimal("0")
	generated_at:     datetime            = Field(default_factory=datetime.utcnow)


# ===========================================================================
# ARRevenueRecognition
# ===========================================================================

class ARRevenueRecognitionBase(BaseModel):
	model_config = ConfigDict(extra="forbid")

	invoice_id:           str
	invoice_line_id:      str
	recognition_method:   RevenueRecognitionMethod
	performance_obligation: str
	transaction_price:    NonNegativeAmount
	recognised_amount:    NonNegativeAmount  = Decimal("0")
	deferred_amount:      NonNegativeAmount  = Decimal("0")
	recognition_date:     date
	period:               str
	gl_revenue_account:   str               = "4000"
	gl_deferred_account:  str               = "2400"


class ARRevenueRecognitionCreate(ARRevenueRecognitionBase):
	tenant_id:  str
	created_by: str


class ARRevenueRecognitionResponse(ARBase, ARRevenueRecognitionBase):
	journal_entry_id: str | None = None


# ===========================================================================
# ARBadDebtProvision  (IFRS 9 ECL)
# ===========================================================================

class ARBadDebtProvision(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id:                str  = Field(default_factory=uuid7str)
	tenant_id:         str
	period:            str
	provision_method:  str                 = "ecl"    # ecl | percentage_of_balance | specific
	customer_id:       str | None          = None     # None = portfolio-level
	outstanding_amount: NonNegativeAmount
	provision_rate:    Decimal             = Decimal("0")  # %
	provision_amount:  NonNegativeAmount   = Decimal("0")
	probability_of_default: Decimal        = Decimal("0")
	loss_given_default:     Decimal        = Decimal("0")
	exposure_at_default:    NonNegativeAmount = Decimal("0")
	days_overdue:      int                 = 0
	stage:             int                 = Field(default=1, ge=1, le=3)  # IFRS 9 stages
	created_at:        datetime            = Field(default_factory=datetime.utcnow)
	created_by:        str                 = ""


# ===========================================================================
# ARFxRevaluation  (unrealised gain/loss)
# ===========================================================================

class ARFxRevaluation(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id:                   str  = Field(default_factory=uuid7str)
	tenant_id:            str
	period:               str
	revaluation_date:     date
	invoice_id:           str
	customer_id:          str
	original_currency:    CurrencyCode
	booking_rate:         PositiveRate
	current_rate:         PositiveRate
	outstanding_foreign:  NonNegativeAmount
	original_functional:  NonNegativeAmount
	revalued_functional:  NonNegativeAmount
	unrealised_gain_loss: Decimal            # +ve = gain, -ve = loss
	gl_account:           str               = "7500"
	journal_entry_id:     str | None        = None
	created_at:           datetime          = Field(default_factory=datetime.utcnow)


# ===========================================================================
# ARCreditCheck result
# ===========================================================================

class ARCreditCheckResult(BaseModel):
	model_config = ConfigDict(extra="forbid")

	customer_id:         str
	requested_amount:    NonNegativeAmount
	currency:            CurrencyCode
	credit_limit:        NonNegativeAmount
	current_outstanding: NonNegativeAmount
	available_credit:    NonNegativeAmount
	utilisation_pct:     Decimal
	approved:            bool
	hold_reason:         str | None = None   # if not approved
	checked_at:          datetime   = Field(default_factory=datetime.utcnow)


# ===========================================================================
# ARDSOReport
# ===========================================================================

class ARDSOReport(BaseModel):
	model_config = ConfigDict(extra="forbid")

	tenant_id:    str
	period:       str
	dso_days:     Decimal    # Days Sales Outstanding
	best_possible_dso: Decimal
	delinquency_dso:   Decimal
	net_credit_sales:  NonNegativeAmount
	average_ar:        NonNegativeAmount
	generated_at:  datetime = Field(default_factory=datetime.utcnow)


# ===========================================================================
# ARCustomerStatement
# ===========================================================================

class ARStatementLine(BaseModel):
	model_config = ConfigDict(extra="forbid")

	transaction_date: date
	transaction_type: str   # "invoice" | "payment" | "credit_note" | "debit_note" | "write_off"
	reference:        str
	description:      str
	debit:            NonNegativeAmount = Decimal("0")
	credit:           NonNegativeAmount = Decimal("0")
	balance:          Decimal           = Decimal("0")
	currency:         CurrencyCode


class ARCustomerStatement(BaseModel):
	model_config = ConfigDict(extra="forbid")

	customer_id:     str
	customer_number: str
	legal_name:      str
	period_from:     date
	period_to:       date
	opening_balance: Decimal
	closing_balance: Decimal
	currency:        CurrencyCode
	lines:           list[ARStatementLine] = Field(default_factory=list)
	generated_at:    datetime             = Field(default_factory=datetime.utcnow)


# ===========================================================================
# ARPaymentForecast  (ML-based)
# ===========================================================================

class ARPaymentForecastLine(BaseModel):
	model_config = ConfigDict(extra="forbid")

	customer_id:      str
	invoice_id:       str
	due_date:         date
	outstanding:      NonNegativeAmount
	predicted_date:   date
	confidence:       Decimal  = Field(ge=0, le=1)
	days_late_predicted: int  = 0


class ARPaymentForecast(BaseModel):
	model_config = ConfigDict(extra="forbid")

	tenant_id:       str
	forecast_date:   date
	horizon_days:    int
	functional_currency: CurrencyCode
	lines:           list[ARPaymentForecastLine] = Field(default_factory=list)
	total_expected:  NonNegativeAmount  = Decimal("0")
	total_at_risk:   NonNegativeAmount  = Decimal("0")
	model_version:   str                = "v1"
	generated_at:    datetime           = Field(default_factory=datetime.utcnow)


# ===========================================================================
# ARCollectionQueue entry
# ===========================================================================

class ARCollectionQueueEntry(BaseModel):
	model_config = ConfigDict(extra="forbid")

	customer_id:         str
	customer_number:     str
	legal_name:          str
	total_overdue:       NonNegativeAmount
	oldest_due_date:     date
	days_overdue_max:    int
	risk_score:          int = Field(ge=0, le=100)   # higher = more urgent
	dunning_level:       DunningLevel
	disputed_amount:     NonNegativeAmount = Decimal("0")
	insolvency_status:   CustomerInsolvencyStatus
	recommended_action:  str   # "call" | "email" | "legal" | "write_off" | "escalate"
	last_contact_date:   date | None = None
	assigned_collector:  str | None  = None


# ===========================================================================
# ARBankStatement (for auto-reconciliation)
# ===========================================================================

class ARBankStatementLine(BaseModel):
	model_config = ConfigDict(extra="forbid")

	line_number:     int
	transaction_date: date
	value_date:      date | None = None
	description:     str
	amount:          Decimal     # +ve = credit (receipt), -ve = debit
	currency:        CurrencyCode
	reference:       str | None  = None
	matched:         bool        = False
	payment_id:      str | None  = None   # linked AR payment after matching


class ARBankStatement(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id:           str           = Field(default_factory=uuid7str)
	tenant_id:    str
	bank_account: str
	currency:     CurrencyCode
	period_from:  date
	period_to:    date
	lines:        list[ARBankStatementLine] = Field(default_factory=list)
	uploaded_at:  datetime = Field(default_factory=datetime.utcnow)
	uploaded_by:  str


# ===========================================================================
# ARPrepayment
# ===========================================================================

class ARPrepaymentBase(BaseModel):
	model_config = ConfigDict(extra="forbid")

	prepayment_number:  str
	customer_id:        str
	receipt_date:       date
	amount:             NonNegativeAmount
	currency:           CurrencyCode
	exchange_rate:      PositiveRate       = Decimal("1")
	functional_amount:  NonNegativeAmount  = Decimal("0")
	gl_account:         str               = "2100"   # advances received
	applied_amount:     NonNegativeAmount  = Decimal("0")
	unapplied_amount:   NonNegativeAmount  = Decimal("0")
	payment_reference:  str | None        = None
	notes:              str | None        = None

	@model_validator(mode="after")
	def _set_functional_and_unapplied(self) -> "ARPrepaymentBase":
		if self.functional_amount == Decimal("0"):
			self.functional_amount = (self.amount * self.exchange_rate).quantize(Decimal("0.01"))
		if self.unapplied_amount == Decimal("0"):
			self.unapplied_amount = self.amount
		return self


class ARPrepaymentCreate(ARPrepaymentBase):
	tenant_id:  str
	created_by: str


class ARPrepaymentResponse(ARBase, ARPrepaymentBase):
	pass


# ===========================================================================
# ARAuditEvent
# ===========================================================================

class ARAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id:          str      = Field(default_factory=uuid7str)
	tenant_id:   str
	entity_type: str
	entity_id:   str
	event_type:  str
	actor_id:    str
	payload:     dict[str, Any] = Field(default_factory=dict)
	emitted_at:  datetime = Field(default_factory=datetime.utcnow)


# ===========================================================================
# ARIntercompanyMatch
# ===========================================================================

class ARIntercompanyMatch(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id:                str = Field(default_factory=uuid7str)
	tenant_id:         str
	ar_invoice_id:     str
	ap_invoice_id:     str    # counterpart entity AP invoice
	counterpart_entity: str
	matched_amount:    NonNegativeAmount
	currency:          CurrencyCode
	match_date:        date
	status:            str = "matched"   # matched | disputed | cleared
	created_at:        datetime = Field(default_factory=datetime.utcnow)
	created_by:        str
