"""Pydantic v2 models for APG Telecom Billing (telecom/bil).

Entities: CDR, UsageEvent, RatingResult, Invoice, BillingAccount,
Bundle, TariffPlan, Discount, Promotion, Roaming, InterconnectSettlement,
Dispute, PaymentAllocation, CreditLimit.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _non_empty(v: str) -> str:
	assert v and v.strip(), "must not be blank"
	return v.strip()


def _positive_decimal(v: Decimal) -> Decimal:
	assert v > 0, "must be positive"
	return v


def _non_negative_decimal(v: Decimal) -> Decimal:
	assert v >= 0, "must be >= 0"
	return v


NonEmpty = Annotated[str, AfterValidator(_non_empty)]
PositiveDecimal = Annotated[Decimal, AfterValidator(_positive_decimal)]
NonNegativeDecimal = Annotated[Decimal, AfterValidator(_non_negative_decimal)]


# ---------------------------------------------------------------------------
# Status / Type Enums
# ---------------------------------------------------------------------------

class CDRStatus(str, Enum):
	RAW = "raw"
	NORMALISED = "normalised"
	RATED = "rated"
	AGGREGATED = "aggregated"
	BILLED = "billed"
	REJECTED = "rejected"
	HELD = "held"
	DUPLICATE = "duplicate"


class CDRType(str, Enum):
	VOICE = "voice"
	SMS = "sms"
	DATA = "data"
	MMS = "mms"
	VIDEO_CALL = "video_call"
	ROAMING = "roaming"
	INTERCONNECT = "interconnect"
	SHORT_CODE = "short_code"


class CallDirection(str, Enum):
	ORIGINATING = "originating"
	TERMINATING = "terminating"
	TRANSIT = "transit"
	FORWARDED = "forwarded"


class InvoiceStatus(str, Enum):
	DRAFT = "draft"
	PENDING_APPROVAL = "pending_approval"
	APPROVED = "approved"
	SENT = "sent"
	PAID = "paid"
	PARTIALLY_PAID = "partially_paid"
	OVERDUE = "overdue"
	DISPUTED = "disputed"
	CANCELLED = "cancelled"
	WRITTEN_OFF = "written_off"


class BillingAccountStatus(str, Enum):
	ACTIVE = "active"
	SUSPENDED = "suspended"
	CLOSED = "closed"
	PENDING = "pending"
	BARRED = "barred"


class BillingAccountType(str, Enum):
	POSTPAID = "postpaid"
	PREPAID = "prepaid"
	HYBRID = "hybrid"
	WHOLESALE = "wholesale"
	MVNO = "mvno"
	ROAMING_PARTNER = "roaming_partner"


class TariffPlanType(str, Enum):
	FLAT_RATE = "flat_rate"
	TIERED = "tiered"
	VOLUME = "volume"
	STEPPED = "stepped"
	TIME_OF_DAY = "time_of_day"
	GEO_BASED = "geo_based"
	CONTRACT = "contract"
	PROMOTIONAL = "promotional"
	PAY_AS_YOU_GO = "pay_as_you_go"


class BundleType(str, Enum):
	VOICE = "voice"
	DATA = "data"
	SMS = "sms"
	COMBO = "combo"
	UNLIMITED = "unlimited"
	FAMILY = "family"
	CORPORATE = "corporate"
	ROAMING = "roaming"


class BundleStatus(str, Enum):
	ACTIVE = "active"
	EXHAUSTED = "exhausted"
	EXPIRED = "expired"
	SUSPENDED = "suspended"
	PENDING = "pending"


class DiscountType(str, Enum):
	LOYALTY = "loyalty"
	PROMOTIONAL = "promotional"
	BULK = "bulk"
	BUNDLE = "bundle"
	RETENTION = "retention"
	CORPORATE = "corporate"
	STAFF = "staff"
	SEASONAL = "seasonal"
	REGULATORY = "regulatory"


class PromotionStatus(str, Enum):
	DRAFT = "draft"
	ACTIVE = "active"
	PAUSED = "paused"
	EXPIRED = "expired"
	CANCELLED = "cancelled"


class DisputeStatus(str, Enum):
	OPEN = "open"
	UNDER_REVIEW = "under_review"
	EVIDENCE_REQUESTED = "evidence_requested"
	RESOLVED_UPHELD = "resolved_upheld"
	RESOLVED_REJECTED = "resolved_rejected"
	ESCALATED = "escalated"
	WITHDRAWN = "withdrawn"
	ARBITRATION = "arbitration"


class DisputeType(str, Enum):
	BILLING_ERROR = "billing_error"
	SERVICE_QUALITY = "service_quality"
	UNAUTHORISED_CHARGE = "unauthorised_charge"
	ROAMING_DISPUTE = "roaming_dispute"
	INTERCONNECT_DISPUTE = "interconnect_dispute"
	FRAUD = "fraud"
	OTHER = "other"


class RoamingZone(str, Enum):
	DOMESTIC = "domestic"
	ZONE_A = "zone_a"
	ZONE_B = "zone_b"
	ZONE_C = "zone_c"
	PREMIUM = "premium"
	GLOBAL = "global"


class SettlementStatus(str, Enum):
	DRAFT = "draft"
	SUBMITTED = "submitted"
	ACKNOWLEDGED = "acknowledged"
	DISPUTED = "disputed"
	AGREED = "agreed"
	PAID = "paid"
	OVERDUE = "overdue"


class PaymentMethod(str, Enum):
	BANK_TRANSFER = "bank_transfer"
	MOBILE_MONEY = "mobile_money"
	CREDIT_CARD = "credit_card"
	DEBIT_CARD = "debit_card"
	DIRECT_DEBIT = "direct_debit"
	CHEQUE = "cheque"
	CASH = "cash"
	VOUCHER = "voucher"
	CRYPTO = "crypto"


class TaxType(str, Enum):
	VAT = "vat"
	WITHHOLDING = "withholding"
	EXCISE = "excise"
	REGULATORY_LEVY = "regulatory_levy"
	USF = "universal_service_fund"
	SPECTRUM_FEE = "spectrum_fee"


class DunningStep(str, Enum):
	REMINDER_1 = "reminder_1"
	REMINDER_2 = "reminder_2"
	SUSPENSION_WARNING = "suspension_warning"
	SERVICE_SUSPENDED = "service_suspended"
	LEGAL_NOTICE = "legal_notice"
	COLLECTIONS = "collections"
	WRITE_OFF = "write_off"


class ConvergentMode(str, Enum):
	SINGLE_BILL = "single_bill"
	MULTI_ACCOUNT = "multi_account"
	HOUSEHOLD = "household"
	CORPORATE_GROUP = "corporate_group"
	MVNO_WHOLESALE = "mvno_wholesale"


class ChargeType(str, Enum):
	RECURRING = "recurring"
	ONE_TIME = "one_time"
	USAGE_BASED = "usage_based"
	OVERAGE = "overage"
	ROAMING = "roaming"
	INTERCONNECT = "interconnect"
	PENALTY = "penalty"
	CREDIT = "credit"
	ADJUSTMENT = "adjustment"
	TAX = "tax"


# ---------------------------------------------------------------------------
# Base Model
# ---------------------------------------------------------------------------

class BilBase(BaseModel):
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
		use_enum_values=True,
	)
	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmpty
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(default="system")
	is_deleted: bool = Field(default=False)


# ---------------------------------------------------------------------------
# CDR (Call Detail Record)
# ---------------------------------------------------------------------------

class CDRCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	cdr_type: CDRType
	direction: CallDirection
	source: NonEmpty
	msisdn: NonEmpty
	called_number: str | None = None
	imsi: str | None = None
	imei: str | None = None
	cell_id: str | None = None
	duration_seconds: int = Field(default=0, ge=0)
	data_volume_bytes: int = Field(default=0, ge=0)
	sms_count: int = Field(default=0, ge=0)
	recorded_at: datetime
	network_id: str | None = None
	roaming_network: str | None = None
	interconnect_carrier: str | None = None
	raw_record: dict[str, Any] = Field(default_factory=dict)
	created_by: str = Field(default="system")


class CDRUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	mediation_status: CDRStatus | None = None
	rating_result_id: str | None = None
	duplicate_of: str | None = None


class CDR(BilBase):
	"""A single usage event as received from a network element, post-normalisation."""
	cdr_type: CDRType
	direction: CallDirection
	source: NonEmpty
	msisdn: NonEmpty
	called_number: str | None = None
	imsi: str | None = None
	imei: str | None = None
	cell_id: str | None = None
	duration_seconds: int = Field(default=0, ge=0)
	data_volume_bytes: int = Field(default=0, ge=0)
	sms_count: int = Field(default=0, ge=0)
	recorded_at: datetime
	network_id: str | None = None
	roaming_network: str | None = None
	interconnect_carrier: str | None = None
	mediation_status: CDRStatus = CDRStatus.RAW
	rating_result_id: str | None = None
	duplicate_of: str | None = None
	raw_record: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# UsageEvent (real-time charging trigger)
# ---------------------------------------------------------------------------

class UsageEventCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	account_id: NonEmpty
	service_id: NonEmpty
	event_type: str
	quantity: NonNegativeDecimal
	unit: str
	occurred_at: datetime
	session_id: str | None = None
	network_element: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_by: str = Field(default="system")


class UsageEvent(BilBase):
	"""A discrete unit of consumption driving real-time charging."""
	account_id: NonEmpty
	service_id: NonEmpty
	event_type: str
	quantity: NonNegativeDecimal
	unit: str
	occurred_at: datetime
	session_id: str | None = None
	network_element: str | None = None
	rated: bool = False
	rating_result_id: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RatingResult
# ---------------------------------------------------------------------------

class RatingResultCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	cdr_id: str | None = None
	usage_event_id: str | None = None
	account_id: NonEmpty
	tariff_plan_id: NonEmpty
	rated_amount: NonNegativeDecimal
	tax_amount: NonNegativeDecimal
	currency: str = Field(default="KES")
	rating_type: TariffPlanType
	bundle_id: str | None = None
	bundle_consumed_units: Decimal = Field(default=Decimal("0"))
	discount_id: str | None = None
	discount_amount: NonNegativeDecimal = Field(default=Decimal("0"))
	rated_at: datetime = Field(default_factory=datetime.utcnow)
	breakdown: dict[str, Any] = Field(default_factory=dict)
	created_by: str = Field(default="system")


class RatingResult(BilBase):
	"""Output of rating engine for a single CDR or UsageEvent."""
	cdr_id: str | None = None
	usage_event_id: str | None = None
	account_id: NonEmpty
	tariff_plan_id: NonEmpty
	rated_amount: NonNegativeDecimal
	tax_amount: NonNegativeDecimal
	currency: str = "KES"
	rating_type: TariffPlanType
	bundle_id: str | None = None
	bundle_consumed_units: Decimal = Decimal("0")
	discount_id: str | None = None
	discount_amount: NonNegativeDecimal = Decimal("0")
	rated_at: datetime = Field(default_factory=datetime.utcnow)
	breakdown: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# BillingAccount
# ---------------------------------------------------------------------------

class BillingAccountCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	account_type: BillingAccountType
	customer_id: NonEmpty
	currency: str = "KES"
	billing_day: int = Field(default=1, ge=1, le=28)
	payment_terms_days: int = Field(default=30, ge=0, le=365)
	parent_account_id: str | None = None
	credit_limit_id: str | None = None
	tax_id: str | None = None
	contact_email: str | None = None
	created_by: str = Field(default="system")


class BillingAccountUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: BillingAccountStatus | None = None
	credit_limit_id: str | None = None
	payment_terms_days: int | None = None
	contact_email: str | None = None


class BillingAccount(BilBase):
	"""A customer billing relationship (prepaid, postpaid, wholesale, MVNO)."""
	account_type: BillingAccountType
	customer_id: NonEmpty
	status: BillingAccountStatus = BillingAccountStatus.ACTIVE
	currency: str = "KES"
	billing_day: int = Field(default=1, ge=1, le=28)
	payment_terms_days: int = Field(default=30, ge=0, le=365)
	parent_account_id: str | None = None
	credit_limit_id: str | None = None
	tax_id: str | None = None
	contact_email: str | None = None
	outstanding_balance: NonNegativeDecimal = Field(default=Decimal("0"))
	last_invoice_id: str | None = None


# ---------------------------------------------------------------------------
# TariffPlan
# ---------------------------------------------------------------------------

class TariffPlanCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	name: NonEmpty
	plan_type: TariffPlanType
	currency: str = "KES"
	base_rate: NonNegativeDecimal
	rate_per_second: NonNegativeDecimal = Field(default=Decimal("0"))
	rate_per_kb: NonNegativeDecimal = Field(default=Decimal("0"))
	rate_per_sms: NonNegativeDecimal = Field(default=Decimal("0"))
	minimum_charge: NonNegativeDecimal = Field(default=Decimal("0"))
	tiers: list[dict[str, Any]] = Field(default_factory=list)
	time_bands: list[dict[str, Any]] = Field(default_factory=list)
	valid_from: datetime
	valid_to: datetime | None = None
	applicable_cdr_types: list[str] = Field(default_factory=list)
	created_by: str = Field(default="system")


class TariffPlan(BilBase):
	"""A pricing template for rating CDRs and usage events."""
	name: NonEmpty
	plan_type: TariffPlanType
	currency: str = "KES"
	base_rate: NonNegativeDecimal
	rate_per_second: NonNegativeDecimal = Decimal("0")
	rate_per_kb: NonNegativeDecimal = Decimal("0")
	rate_per_sms: NonNegativeDecimal = Decimal("0")
	minimum_charge: NonNegativeDecimal = Decimal("0")
	tiers: list[dict[str, Any]] = Field(default_factory=list)
	time_bands: list[dict[str, Any]] = Field(default_factory=list)
	valid_from: datetime
	valid_to: datetime | None = None
	is_active: bool = True
	applicable_cdr_types: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

class BundleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	account_id: NonEmpty
	bundle_type: BundleType
	name: NonEmpty
	total_units: PositiveDecimal
	unit: str
	price: NonNegativeDecimal
	currency: str = "KES"
	valid_from: datetime
	valid_to: datetime
	rollover_allowed: bool = False
	shared: bool = False
	created_by: str = Field(default="system")


class BundleUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: BundleStatus | None = None
	consumed_units: NonNegativeDecimal | None = None


class Bundle(BilBase):
	"""A prepaid allowance (voice minutes, data GB, SMS count) for an account."""
	account_id: NonEmpty
	bundle_type: BundleType
	name: NonEmpty
	total_units: PositiveDecimal
	consumed_units: NonNegativeDecimal = Decimal("0")
	unit: str
	price: NonNegativeDecimal
	currency: str = "KES"
	valid_from: datetime
	valid_to: datetime
	status: BundleStatus = BundleStatus.ACTIVE
	rollover_allowed: bool = False
	rollover_units: NonNegativeDecimal = Decimal("0")
	shared: bool = False
	shared_with: list[str] = Field(default_factory=list)

	@property
	def remaining_units(self) -> Decimal:
		return max(Decimal("0"), self.total_units - self.consumed_units + self.rollover_units)


# ---------------------------------------------------------------------------
# Discount
# ---------------------------------------------------------------------------

class DiscountCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	account_id: NonEmpty
	discount_type: DiscountType
	discount_pct: Decimal = Field(ge=Decimal("0"), le=Decimal("50"))
	flat_amount: NonNegativeDecimal = Field(default=Decimal("0"))
	currency: str = "KES"
	approval_reference: NonEmpty
	valid_from: datetime
	valid_to: datetime
	applicable_charge_types: list[str] = Field(default_factory=list)
	max_applications: int | None = None
	created_by: str = Field(default="system")


class Discount(BilBase):
	"""An approved price reduction applied to a billing account."""
	account_id: NonEmpty
	discount_type: DiscountType
	discount_pct: Decimal
	flat_amount: NonNegativeDecimal = Decimal("0")
	currency: str = "KES"
	approval_reference: NonEmpty
	valid_from: datetime
	valid_to: datetime
	applicable_charge_types: list[str] = Field(default_factory=list)
	applications_count: int = 0
	max_applications: int | None = None
	is_active: bool = True


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------

class PromotionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	name: NonEmpty
	description: str = ""
	discount_pct: Decimal = Field(ge=Decimal("0"), le=Decimal("100"))
	bonus_units: NonNegativeDecimal = Decimal("0")
	bonus_unit_type: str | None = None
	eligible_account_types: list[str] = Field(default_factory=list)
	promo_code: str | None = None
	valid_from: datetime
	valid_to: datetime
	max_redemptions: int | None = None
	budget_cap: NonNegativeDecimal | None = None
	currency: str = "KES"
	created_by: str = Field(default="system")


class Promotion(BilBase):
	"""A time-limited pricing campaign with optional budget and redemption caps."""
	name: NonEmpty
	description: str = ""
	status: PromotionStatus = PromotionStatus.DRAFT
	discount_pct: Decimal
	bonus_units: NonNegativeDecimal = Decimal("0")
	bonus_unit_type: str | None = None
	eligible_account_types: list[str] = Field(default_factory=list)
	promo_code: str | None = None
	valid_from: datetime
	valid_to: datetime
	redemption_count: int = 0
	max_redemptions: int | None = None
	budget_cap: NonNegativeDecimal | None = None
	budget_consumed: NonNegativeDecimal = Decimal("0")
	currency: str = "KES"


# ---------------------------------------------------------------------------
# Roaming
# ---------------------------------------------------------------------------

class RoamingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	account_id: NonEmpty
	cdr_id: NonEmpty
	zone: RoamingZone
	visited_network: NonEmpty
	home_network: NonEmpty
	service_type: CDRType
	duration_seconds: int = Field(default=0, ge=0)
	data_volume_bytes: int = Field(default=0, ge=0)
	base_charge: NonNegativeDecimal
	surcharge: NonNegativeDecimal = Decimal("0")
	currency: str = "KES"
	tap_file_reference: str | None = None
	created_by: str = Field(default="system")


class Roaming(BilBase):
	"""Roaming charge record derived from a TAP/NRTRDE file or live network event."""
	account_id: NonEmpty
	cdr_id: NonEmpty
	zone: RoamingZone
	visited_network: NonEmpty
	home_network: NonEmpty
	service_type: CDRType
	duration_seconds: int = 0
	data_volume_bytes: int = 0
	base_charge: NonNegativeDecimal
	surcharge: NonNegativeDecimal = Decimal("0")
	total_charge: NonNegativeDecimal = Decimal("0")
	currency: str = "KES"
	tap_file_reference: str | None = None
	settled: bool = False
	settlement_id: str | None = None


# ---------------------------------------------------------------------------
# InterconnectSettlement
# ---------------------------------------------------------------------------

class InterconnectSettlementCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	carrier_id: NonEmpty
	carrier_name: NonEmpty
	period_start: datetime
	period_end: datetime
	originating_minutes: Decimal = Decimal("0")
	terminating_minutes: Decimal = Decimal("0")
	transit_minutes: Decimal = Decimal("0")
	data_gb: Decimal = Decimal("0")
	receivable_amount: NonNegativeDecimal
	payable_amount: NonNegativeDecimal
	currency: str = "KES"
	reference_number: NonEmpty
	created_by: str = Field(default="system")


class InterconnectSettlement(BilBase):
	"""Bilateral settlement between carriers for traffic exchange."""
	carrier_id: NonEmpty
	carrier_name: NonEmpty
	period_start: datetime
	period_end: datetime
	originating_minutes: Decimal = Decimal("0")
	terminating_minutes: Decimal = Decimal("0")
	transit_minutes: Decimal = Decimal("0")
	data_gb: Decimal = Decimal("0")
	receivable_amount: NonNegativeDecimal
	payable_amount: NonNegativeDecimal
	net_amount: Decimal = Decimal("0")
	currency: str = "KES"
	status: SettlementStatus = SettlementStatus.DRAFT
	reference_number: NonEmpty
	dispute_reference: str | None = None
	paid_at: datetime | None = None


class InterconnectSettlementUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: SettlementStatus | None = None
	dispute_reference: str | None = None
	paid_at: datetime | None = None


# ---------------------------------------------------------------------------
# Invoice
# ---------------------------------------------------------------------------

class InvoiceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	account_id: NonEmpty
	period_start: datetime
	period_end: datetime
	currency: str = "KES"
	due_date: datetime
	line_items: list[dict[str, Any]] = Field(default_factory=list)
	created_by: str = Field(default="system")


class InvoiceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: InvoiceStatus | None = None
	approval_reference: str | None = None
	notes: str | None = None


class Invoice(BilBase):
	"""A customer bill covering a billing period."""
	account_id: NonEmpty
	period_start: datetime
	period_end: datetime
	subtotal: NonNegativeDecimal = Decimal("0")
	tax_amount: NonNegativeDecimal = Decimal("0")
	discount_amount: NonNegativeDecimal = Decimal("0")
	total_amount: NonNegativeDecimal = Decimal("0")
	paid_amount: NonNegativeDecimal = Decimal("0")
	currency: str = "KES"
	status: InvoiceStatus = InvoiceStatus.DRAFT
	due_date: datetime
	approval_reference: str | None = None
	approved_at: datetime | None = None
	sent_at: datetime | None = None
	line_items: list[dict[str, Any]] = Field(default_factory=list)
	dunning_step: DunningStep | None = None
	last_dunning_at: datetime | None = None
	notes: str | None = None


# ---------------------------------------------------------------------------
# Dispute
# ---------------------------------------------------------------------------

class DisputeCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	account_id: NonEmpty
	invoice_id: str | None = None
	cdr_id: str | None = None
	dispute_type: DisputeType
	disputed_amount: NonNegativeDecimal
	currency: str = "KES"
	reason: NonEmpty
	evidence_refs: list[str] = Field(default_factory=list)
	created_by: str = Field(default="system")


class DisputeUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: DisputeStatus | None = None
	resolution_notes: str | None = None
	credit_amount: NonNegativeDecimal | None = None
	resolver_id: str | None = None


class Dispute(BilBase):
	"""A customer challenge to a charge, CDR, or invoice."""
	account_id: NonEmpty
	invoice_id: str | None = None
	cdr_id: str | None = None
	dispute_type: DisputeType
	disputed_amount: NonNegativeDecimal
	currency: str = "KES"
	reason: NonEmpty
	status: DisputeStatus = DisputeStatus.OPEN
	evidence_refs: list[str] = Field(default_factory=list)
	resolution_notes: str | None = None
	credit_amount: NonNegativeDecimal = Decimal("0")
	resolver_id: str | None = None
	resolved_at: datetime | None = None
	sla_deadline: datetime | None = None


# ---------------------------------------------------------------------------
# PaymentAllocation
# ---------------------------------------------------------------------------

class PaymentAllocationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	account_id: NonEmpty
	invoice_id: NonEmpty
	payment_method: PaymentMethod
	amount: PositiveDecimal
	currency: str = "KES"
	reference: NonEmpty
	paid_at: datetime
	gateway_reference: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_by: str = Field(default="system")


class PaymentAllocation(BilBase):
	"""A payment received and allocated to an invoice."""
	account_id: NonEmpty
	invoice_id: NonEmpty
	payment_method: PaymentMethod
	amount: PositiveDecimal
	currency: str = "KES"
	reference: NonEmpty
	paid_at: datetime
	gateway_reference: str | None = None
	allocated: bool = False
	allocated_at: datetime | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# CreditLimit
# ---------------------------------------------------------------------------

class CreditLimitCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmpty
	account_id: NonEmpty
	hard_limit: PositiveDecimal
	soft_limit: PositiveDecimal
	currency: str = "KES"
	approval_reference: NonEmpty
	review_date: datetime | None = None
	auto_suspend_at_hard: bool = True
	alert_at_soft: bool = True
	created_by: str = Field(default="system")


class CreditLimitUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	hard_limit: PositiveDecimal | None = None
	soft_limit: PositiveDecimal | None = None
	current_usage: NonNegativeDecimal | None = None
	approval_reference: str | None = None


class CreditLimit(BilBase):
	"""Spend ceiling for a billing account with soft (alert) and hard (block) thresholds."""
	account_id: NonEmpty
	hard_limit: PositiveDecimal
	soft_limit: PositiveDecimal
	current_usage: NonNegativeDecimal = Decimal("0")
	currency: str = "KES"
	approval_reference: NonEmpty
	review_date: datetime | None = None
	auto_suspend_at_hard: bool = True
	alert_at_soft: bool = True
	suspended_at: datetime | None = None


# ---------------------------------------------------------------------------
# Report / Aggregation models
# ---------------------------------------------------------------------------

class RevenueReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	period_start: datetime
	period_end: datetime
	total_revenue: Decimal
	voice_revenue: Decimal
	data_revenue: Decimal
	sms_revenue: Decimal
	roaming_revenue: Decimal
	interconnect_revenue: Decimal
	other_revenue: Decimal
	tax_collected: Decimal
	discounts_given: Decimal
	net_revenue: Decimal
	currency: str
	invoice_count: int
	paid_invoice_count: int
	disputed_amount: Decimal
	written_off_amount: Decimal
	generated_at: datetime = Field(default_factory=datetime.utcnow)


class BillingDashboardKPI(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	total_accounts: int
	active_accounts: int
	suspended_accounts: int
	total_invoices: int
	draft_invoices: int
	overdue_invoices: int
	open_disputes: int
	total_revenue_mtd: Decimal
	collection_rate_pct: Decimal
	average_revenue_per_account: Decimal
	credit_utilisation_pct: Decimal
	currency: str
	as_of: datetime = Field(default_factory=datetime.utcnow)


class CDRRatingReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	period_start: datetime
	period_end: datetime
	total_cdrs: int
	rated_cdrs: int
	rejected_cdrs: int
	held_cdrs: int
	duplicate_cdrs: int
	total_rated_amount: Decimal
	currency: str
	by_type: dict[str, int]
	generated_at: datetime = Field(default_factory=datetime.utcnow)


class SpendAlert(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	account_id: str
	tenant_id: str
	alert_type: str  # "soft_limit" | "hard_limit" | "spend_velocity"
	current_usage: Decimal
	limit: Decimal
	utilisation_pct: Decimal
	currency: str
	triggered_at: datetime = Field(default_factory=datetime.utcnow)


class ConvergentBillingSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	master_account_id: str
	tenant_id: str
	mode: ConvergentMode
	member_account_ids: list[str]
	total_fixed_charges: Decimal
	total_mobile_charges: Decimal
	total_data_charges: Decimal
	combined_total: Decimal
	currency: str
	period_start: datetime
	period_end: datetime
	generated_at: datetime = Field(default_factory=datetime.utcnow)


class TaxBreakdown(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	account_id: str
	invoice_id: str
	jurisdiction: str
	pre_tax_amount: Decimal
	tax_components: list[dict[str, Any]]
	total_tax: Decimal
	total_with_tax: Decimal
	currency: str
	calculated_at: datetime = Field(default_factory=datetime.utcnow)


class RevenueAssuranceResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	period_start: datetime
	period_end: datetime
	total_cdrs: int
	unrated_cdrs: int
	leakage_pct: Decimal
	collection_rate_pct: Decimal
	dso_days: Decimal
	arpu: Decimal
	anomalies: list[dict[str, Any]]
	currency: str
	generated_at: datetime = Field(default_factory=datetime.utcnow)
