"""Pydantic v2 data models for APG Digital Payments — Africa-first."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


def utcnow() -> datetime:
	return datetime.now(timezone.utc)


def money(value: Decimal | int | str) -> str:
	"""Stable JSON money text without floating-point drift."""
	return str(Decimal(str(value)))


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PaymentStatus(str, Enum):
	pending     = "pending"
	initiated   = "initiated"
	processing  = "processing"
	completed   = "completed"
	failed      = "failed"
	reversed    = "reversed"
	refunded    = "refunded"
	disputed    = "disputed"
	expired     = "expired"


class PaymentMethod(str, Enum):
	mpesa_stk       = "mpesa_stk"
	mpesa_b2c       = "mpesa_b2c"
	mpesa_b2b       = "mpesa_b2b"
	mtn_momo        = "mtn_momo"
	airtel_money    = "airtel_money"
	tigo_pesa       = "tigo_pesa"
	card_visa       = "card_visa"
	card_mastercard = "card_mastercard"
	bank_eft        = "bank_eft"
	swift           = "swift"
	rtgs            = "rtgs"
	cash            = "cash"
	ussd            = "ussd"
	qr_code         = "qr_code"
	pesalink        = "pesalink"


class TransactionType(str, Enum):
	payment    = "payment"
	refund     = "refund"
	reversal   = "reversal"
	top_up     = "top_up"
	withdrawal = "withdrawal"
	transfer   = "transfer"
	settlement = "settlement"
	charge     = "charge"


class CurrencyCode(str, Enum):
	KES = "KES"
	UGX = "UGX"
	TZS = "TZS"
	RWF = "RWF"
	GHS = "GHS"
	NGN = "NGN"
	ZAR = "ZAR"
	USD = "USD"
	EUR = "EUR"
	GBP = "GBP"
	XOF = "XOF"
	XAF = "XAF"


class KYCTier(str, Enum):
	basic    = "basic"      # CBK tier 1 — KES 300k/day
	standard = "standard"  # CBK tier 2 — KES 1M/day
	full_kyc = "full_kyc"  # CBK tier 3 — KES 5M/day
	enhanced = "enhanced"  # institutional — no retail limit


class RiskLevel(str, Enum):
	low     = "low"
	medium  = "medium"
	high    = "high"
	blocked = "blocked"


class DisputeStatus(str, Enum):
	opened     = "opened"
	under_review = "under_review"
	resolved   = "resolved"
	closed     = "closed"


class WebhookEventType(str, Enum):
	payment_initiated   = "payment.initiated"
	payment_completed   = "payment.completed"
	payment_failed      = "payment.failed"
	payment_refunded    = "payment.refunded"
	payment_reversed    = "payment.reversed"
	settlement_complete = "settlement.complete"
	dispute_opened      = "dispute.opened"
	dispute_resolved    = "dispute.resolved"


class FXRateType(str, Enum):
	spot      = "spot"
	forward   = "forward"
	indicative = "indicative"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

_BASE_CFG = ConfigDict(
	extra="forbid",
	validate_by_name=True,
	validate_by_alias=True,
	arbitrary_types_allowed=True,
)


class _Base(BaseModel):
	model_config = _BASE_CFG


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class PaymentAccount(_Base):
	id:              str           = Field(default_factory=uuid7str)
	tenant_id:       str
	owner_reference: str
	currency:        CurrencyCode
	status:          str           = "active"
	kyc_tier:        KYCTier       = KYCTier.basic
	balance:         Decimal       = Decimal("0")
	reserved:        Decimal       = Decimal("0")
	created_at:      datetime      = Field(default_factory=utcnow)
	metadata:        dict[str, Any] = Field(default_factory=dict)

	@field_validator("owner_reference")
	@classmethod
	def owner_not_empty(cls, v: str) -> str:
		assert v.strip(), "owner_reference must not be blank"
		return v


class PaymentInstrument(_Base):
	id:              str          = Field(default_factory=uuid7str)
	tenant_id:       str
	account_id:      str
	instrument_type: str
	token_reference: str
	provider:        str          = ""
	phone_number:    str | None   = None
	card_last4:      str | None   = None
	bank_code:       str | None   = None
	status:          str          = "active"
	verified:        bool         = False
	created_at:      datetime     = Field(default_factory=utcnow)


class PaymentOrder(_Base):
	id:                   str          = Field(default_factory=uuid7str)
	tenant_id:            str
	account_id:           str
	instrument_id:        str
	amount:               Decimal
	currency:             CurrencyCode
	counterparty_reference: str
	purpose:              str          = "payment"
	status:               PaymentStatus = PaymentStatus.pending
	authorized_amount:    Decimal       = Decimal("0")
	captured_amount:      Decimal       = Decimal("0")
	refunded_amount:      Decimal       = Decimal("0")
	risk_level:           RiskLevel     = RiskLevel.medium
	risk_score:           Decimal       = Decimal("0")
	fee_amount:           Decimal       = Decimal("0")
	excise_tax:           Decimal       = Decimal("0")
	net_amount:           Decimal       = Decimal("0")
	created_at:           datetime      = Field(default_factory=utcnow)
	expires_at:           datetime | None = None
	metadata:             dict[str, Any] = Field(default_factory=dict)

	@field_validator("amount")
	@classmethod
	def amount_positive(cls, v: Decimal) -> Decimal:
		assert v > 0, "amount must be positive"
		return v


class PaymentTransaction(_Base):
	id:               str              = Field(default_factory=uuid7str)
	tenant_id:        str
	order_id:         str
	transaction_type: TransactionType  = TransactionType.payment
	method:           PaymentMethod
	amount:           Decimal
	currency:         CurrencyCode
	status:           PaymentStatus    = PaymentStatus.initiated
	provider_ref:     str | None       = None
	provider_status:  str | None       = None
	recipient:        str              = ""
	sender:           str              = ""
	reference:        str              = ""
	fee_amount:       Decimal          = Decimal("0")
	excise_tax:       Decimal          = Decimal("0")
	fx_rate:          Decimal | None   = None
	idempotency_key:  str              = ""
	retry_count:      int              = 0
	created_at:       datetime         = Field(default_factory=utcnow)
	updated_at:       datetime         = Field(default_factory=utcnow)
	completed_at:     datetime | None  = None
	metadata:         dict[str, Any]   = Field(default_factory=dict)


class PaymentLeg(_Base):
	"""One split leg of a marketplace transaction."""
	id:             str        = Field(default_factory=uuid7str)
	transaction_id: str
	merchant_id:    str
	amount:         Decimal
	currency:       CurrencyCode
	percentage:     Decimal    = Decimal("0")
	purpose:        str        = ""
	settled:        bool       = False
	created_at:     datetime   = Field(default_factory=utcnow)


class MobileMoneyPayment(_Base):
	id:             str          = Field(default_factory=uuid7str)
	tenant_id:      str
	provider:       str          # mpesa / mtn_momo / airtel / tigo
	msisdn:         str
	amount:         Decimal
	currency:       CurrencyCode
	external_id:    str          = Field(default_factory=uuid7str)
	status:         PaymentStatus = PaymentStatus.initiated
	provider_ref:   str | None   = None
	callback_url:   str          = ""
	narration:      str          = ""
	created_at:     datetime     = Field(default_factory=utcnow)
	updated_at:     datetime     = Field(default_factory=utcnow)


class CardPayment(_Base):
	id:              str        = Field(default_factory=uuid7str)
	tenant_id:       str
	card_token:      str
	amount:          Decimal
	currency:        CurrencyCode
	merchant_id:     str
	cvv_result:      str        = "M"   # M=match, N=no-match, P=not-processed
	avs_result:      str        = "Y"
	auth_code:       str | None = None
	rrn:             str | None = None   # retrieval reference number
	status:          PaymentStatus = PaymentStatus.initiated
	three_ds_result: str | None = None
	created_at:      datetime   = Field(default_factory=utcnow)


class BankTransfer(_Base):
	id:             str        = Field(default_factory=uuid7str)
	tenant_id:      str
	from_account:   str
	to_account:     str
	bank_code:      str        = ""
	amount:         Decimal
	currency:       CurrencyCode
	reference:      str
	narration:      str        = ""
	clearing_type:  str        = "eft"   # eft / rtgs / pesalink
	status:         PaymentStatus = PaymentStatus.initiated
	value_date:     str | None = None    # YYYY-MM-DD
	created_at:     datetime   = Field(default_factory=utcnow)


class SWIFTPayment(_Base):
	id:           str        = Field(default_factory=uuid7str)
	tenant_id:    str
	sender_bic:   str
	receiver_bic: str
	iban:         str
	amount:       Decimal
	currency:     CurrencyCode
	purpose_code: str        = "OTH"
	charges:      str        = "SHA"    # SHA / OUR / BEN
	uetr:         str        = Field(default_factory=uuid7str)   # unique end-to-end transaction reference
	status:       PaymentStatus = PaymentStatus.initiated
	created_at:   datetime   = Field(default_factory=utcnow)


class PaymentRefund(_Base):
	id:                 str          = Field(default_factory=uuid7str)
	tenant_id:          str
	original_txn_id:    str
	amount:             Decimal
	reason:             str
	refund_to_original: bool         = True
	status:             PaymentStatus = PaymentStatus.initiated
	provider_ref:       str | None   = None
	initiated_at:       datetime     = Field(default_factory=utcnow)
	completed_at:       datetime | None = None


class PaymentReversal(_Base):
	id:              str          = Field(default_factory=uuid7str)
	tenant_id:       str
	original_txn_id: str
	reason:          str
	reversal_code:   str          = ""
	amount:          Decimal
	status:          PaymentStatus = PaymentStatus.initiated
	window_expires:  datetime | None = None   # 24-hour window for wrong-number
	created_at:      datetime     = Field(default_factory=utcnow)


class FXConversion(_Base):
	id:            str        = Field(default_factory=uuid7str)
	tenant_id:     str
	from_currency: CurrencyCode
	to_currency:   CurrencyCode
	from_amount:   Decimal
	to_amount:     Decimal
	rate:          Decimal
	rate_type:     FXRateType = FXRateType.spot
	provider:      str        = "CBK"
	quoted_at:     datetime   = Field(default_factory=utcnow)
	executed_at:   datetime | None = None
	rate_expires:  datetime | None = None
	spread_bps:    int        = 150    # basis points


class SettlementBatch(_Base):
	id:               str        = Field(default_factory=uuid7str)
	tenant_id:        str
	settlement_date:  str        # YYYY-MM-DD
	bank_account:     str
	total_amount:     Decimal    = Decimal("0")
	currency:         CurrencyCode = CurrencyCode.KES
	transaction_ids:  list[str]  = Field(default_factory=list)
	status:           str        = "pending"
	variance_amount:  Decimal    = Decimal("0")
	review_id:        str        = ""
	created_at:       datetime   = Field(default_factory=utcnow)
	completed_at:     datetime | None = None


class MerchantAccount(_Base):
	id:                 str        = Field(default_factory=uuid7str)
	tenant_id:          str
	name:               str
	category_code:      str        = "7372"   # MCC
	settlement_account: str
	paybill_number:     str | None = None
	till_number:        str | None = None
	status:             str        = "active"
	daily_limit:        Decimal    = Decimal("5000000")
	created_at:         datetime   = Field(default_factory=utcnow)
	metadata:           dict[str, Any] = Field(default_factory=dict)


class VirtualAccount(_Base):
	id:          str        = Field(default_factory=uuid7str)
	tenant_id:   str
	owner_id:    str
	currency:    CurrencyCode
	balance:     Decimal    = Decimal("0")
	reserved:    Decimal    = Decimal("0")
	status:      str        = "active"
	created_at:  datetime   = Field(default_factory=utcnow)

	@property
	def available(self) -> Decimal:
		return self.balance - self.reserved


class PaymentReceipt(_Base):
	id:              str        = Field(default_factory=uuid7str)
	tenant_id:       str
	transaction_id:  str
	amount:          Decimal
	currency:        CurrencyCode
	method:          PaymentMethod
	recipient:       str
	reference:       str
	status:          PaymentStatus
	fee_amount:      Decimal    = Decimal("0")
	excise_tax:      Decimal    = Decimal("0")
	issued_at:       datetime   = Field(default_factory=utcnow)
	sms_sent:        bool       = False
	email_sent:      bool       = False


class ChargebackCase(_Base):
	id:           str           = Field(default_factory=uuid7str)
	tenant_id:    str
	dispute_id:   str
	transaction_id: str
	amount:       Decimal
	decision:     str           = ""    # accept / reject / partial
	settled_amount: Decimal     = Decimal("0")
	reason_code:  str           = ""
	created_at:   datetime      = Field(default_factory=utcnow)
	resolved_at:  datetime | None = None


class PaymentLimit(_Base):
	"""KYC-tier transaction limits (Kenya CBK framework)."""
	kyc_tier:         KYCTier
	daily_limit:      Decimal
	monthly_limit:    Decimal
	per_txn_limit:    Decimal
	currency:         CurrencyCode = CurrencyCode.KES

	@model_validator(mode="after")
	def limits_consistent(self) -> "PaymentLimit":
		assert self.per_txn_limit <= self.daily_limit, "per_txn_limit must not exceed daily_limit"
		assert self.daily_limit <= self.monthly_limit, "daily_limit must not exceed monthly_limit"
		return self


class WebhookEvent(_Base):
	id:          str             = Field(default_factory=uuid7str)
	tenant_id:   str
	event_types: list[str]
	url:         str
	secret:      str             = Field(default_factory=uuid7str)
	active:      bool            = True
	created_at:  datetime        = Field(default_factory=utcnow)


class PaymentNotification(_Base):
	id:             str        = Field(default_factory=uuid7str)
	tenant_id:      str
	transaction_id: str
	channel:        str        = "sms"   # sms / email / push
	recipient:      str
	message:        str
	sent:           bool       = False
	sent_at:        datetime | None = None
	created_at:     datetime   = Field(default_factory=utcnow)


class ReconciliationRecord(_Base):
	id:                str        = Field(default_factory=uuid7str)
	tenant_id:         str
	settlement_id:     str
	transaction_id:    str
	expected_amount:   Decimal
	actual_amount:     Decimal
	variance:          Decimal    = Decimal("0")
	status:            str        = "matched"   # matched / variance / missing
	note:              str        = ""
	reconciled_at:     datetime   = Field(default_factory=utcnow)

	@model_validator(mode="after")
	def compute_variance(self) -> "ReconciliationRecord":
		self.variance = self.actual_amount - self.expected_amount
		if self.variance != 0:
			self.status = "variance"
		return self


class PaymentFee(_Base):
	id:            str        = Field(default_factory=uuid7str)
	method:        PaymentMethod
	amount:        Decimal
	currency:      CurrencyCode
	fee_amount:    Decimal
	excise_tax:    Decimal    = Decimal("0")   # Kenya 15% on fee
	total_charge:  Decimal    = Decimal("0")
	tier:          str        = ""
	computed_at:   datetime   = Field(default_factory=utcnow)

	@model_validator(mode="after")
	def sum_total(self) -> "PaymentFee":
		self.total_charge = self.fee_amount + self.excise_tax
		return self


class SplitPayment(_Base):
	id:             str        = Field(default_factory=uuid7str)
	transaction_id: str
	tenant_id:      str
	legs:           list[PaymentLeg] = Field(default_factory=list)
	total_amount:   Decimal    = Decimal("0")
	currency:       CurrencyCode = CurrencyCode.KES
	created_at:     datetime   = Field(default_factory=utcnow)

	@model_validator(mode="after")
	def legs_sum_check(self) -> "SplitPayment":
		if self.legs:
			leg_total = sum(leg.amount for leg in self.legs)
			assert abs(leg_total - self.total_amount) < Decimal("0.01"), (
				f"Legs sum {leg_total} != total_amount {self.total_amount}"
			)
		return self


class BulkPaymentBatch(_Base):
	id:            str        = Field(default_factory=uuid7str)
	tenant_id:     str
	payment_date:  str        # YYYY-MM-DD
	method:        PaymentMethod
	recipients:    list[str]
	amounts:       list[Decimal]
	references:    list[str]
	currency:      CurrencyCode
	status:        str        = "queued"
	processed:     int        = 0
	failed:        int        = 0
	total_amount:  Decimal    = Decimal("0")
	created_at:    datetime   = Field(default_factory=utcnow)
	completed_at:  datetime | None = None

	@model_validator(mode="after")
	def lists_aligned(self) -> "BulkPaymentBatch":
		n = len(self.recipients)
		assert len(self.amounts) == n and len(self.references) == n, (
			"recipients, amounts, references must have equal length"
		)
		if not self.total_amount:
			self.total_amount = sum(self.amounts)
		return self


class PaymentDispute(_Base):
	id:             str           = Field(default_factory=uuid7str)
	tenant_id:      str
	transaction_id: str
	raised_by:      str
	reason:         str
	evidence:       dict[str, Any] = Field(default_factory=dict)
	status:         DisputeStatus = DisputeStatus.opened
	amount:         Decimal
	created_at:     datetime      = Field(default_factory=utcnow)
	resolved_at:    datetime | None = None


# Canonical KYC-tier limits for Kenya (CBK Prudential Guidelines)
KYC_LIMITS: dict[KYCTier, PaymentLimit] = {
	KYCTier.basic: PaymentLimit(
		kyc_tier=KYCTier.basic,
		daily_limit=Decimal("300000"),
		monthly_limit=Decimal("3000000"),
		per_txn_limit=Decimal("150000"),
	),
	KYCTier.standard: PaymentLimit(
		kyc_tier=KYCTier.standard,
		daily_limit=Decimal("1000000"),
		monthly_limit=Decimal("10000000"),
		per_txn_limit=Decimal("500000"),
	),
	KYCTier.full_kyc: PaymentLimit(
		kyc_tier=KYCTier.full_kyc,
		daily_limit=Decimal("5000000"),
		monthly_limit=Decimal("50000000"),
		per_txn_limit=Decimal("1000000"),
	),
	KYCTier.enhanced: PaymentLimit(
		kyc_tier=KYCTier.enhanced,
		daily_limit=Decimal("999999999"),
		monthly_limit=Decimal("999999999"),
		per_txn_limit=Decimal("999999999"),
	),
}

# Safaricom Daraja M-Pesa fee schedule (KES, 2025)
MPESA_FEE_TIERS: list[tuple[Decimal, Decimal, Decimal]] = [
	# (min, max, fee)
	(Decimal("1"),    Decimal("100"),    Decimal("0")),
	(Decimal("101"),  Decimal("500"),    Decimal("7")),
	(Decimal("501"),  Decimal("1000"),   Decimal("13")),
	(Decimal("1001"), Decimal("1500"),   Decimal("23")),
	(Decimal("1501"), Decimal("2500"),   Decimal("33")),
	(Decimal("2501"), Decimal("3500"),   Decimal("53")),
	(Decimal("3501"), Decimal("5000"),   Decimal("57")),
	(Decimal("5001"), Decimal("7500"),   Decimal("78")),
	(Decimal("7501"), Decimal("10000"),  Decimal("90")),
	(Decimal("10001"),Decimal("15000"),  Decimal("100")),
	(Decimal("15001"),Decimal("20000"),  Decimal("105")),
	(Decimal("20001"),Decimal("35000"),  Decimal("108")),
	(Decimal("35001"),Decimal("50000"),  Decimal("108")),
	(Decimal("50001"),Decimal("150000"), Decimal("108")),
	(Decimal("150001"),Decimal("250000"),Decimal("108")),
	(Decimal("250001"),Decimal("999999"),Decimal("108")),
]
