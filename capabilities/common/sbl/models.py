"""Pydantic v2 models for APG SaaS Billing Engine.

All model names use the 'Sb' prefix per capability convention.
IDs use uuid7str for k-sortable, time-prefixed UUIDs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import AfterValidator, ConfigDict, Field

try:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())
except ImportError:  # pragma: no cover — optional dep during standalone execution
	import uuid

	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())

try:
	from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
	raise ImportError("pydantic>=2 is required for apg-common-sbl") from exc


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _non_empty_str(v: str) -> str:
	assert v and v.strip(), "must be a non-empty string"
	return v.strip()


def _positive_int(v: int) -> int:
	assert v > 0, "must be a positive integer"
	return v


def _non_negative_int(v: int) -> int:
	assert v >= 0, "must be non-negative"
	return v


def _non_negative_float(v: float) -> float:
	assert v >= 0.0, "must be non-negative"
	return v


NonEmptyStr   = Annotated[str,   AfterValidator(_non_empty_str)]
PositiveInt   = Annotated[int,   AfterValidator(_positive_int)]
NonNegInt     = Annotated[int,   AfterValidator(_non_negative_int)]
NonNegFloat   = Annotated[float, AfterValidator(_non_negative_float)]


# ---------------------------------------------------------------------------
# SbPlan — billing plan definition
# ---------------------------------------------------------------------------

class SbPlanLimits(BaseModel):
	"""Per-metric hard limits for a plan.  -1 means unlimited."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	api_calls:    int = Field(default=0, description="Monthly API call limit; -1=unlimited")
	storage_gb:   int = Field(default=0, description="Storage quota in GB; -1=unlimited")
	users:        int = Field(default=0, description="User seat limit; -1=unlimited")
	transactions: int = Field(default=0, description="Transaction count limit; -1=unlimited")
	exports:      int = Field(default=0, description="Export operation limit; -1=unlimited")
	webhooks:     int = Field(default=0, description="Active webhook endpoints; -1=unlimited")
	seats:        int = Field(default=0, description="Concurrent seat limit; -1=unlimited")


class SbPlan(BaseModel):
	"""A billing plan defining price, limits, and feature flags.

	Plans are global (not tenant-scoped) but subscriptions are tenant-scoped.
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id:                  str        = Field(default_factory=uuid7str)
	name:                NonEmptyStr                                     # e.g. "starter"
	display_name:        NonEmptyStr                                     # e.g. "Starter"
	tier:                NonEmptyStr                                     # free|starter|professional|enterprise
	price_monthly_cents: NonNegInt  = Field(default=0)
	price_annual_cents:  NonNegInt  = Field(default=0)
	currency:            str        = Field(default="USD")
	limits:              SbPlanLimits = Field(default_factory=SbPlanLimits)
	features:            list[str]  = Field(default_factory=list)
	overage_allowed:     bool       = Field(default=False)
	overage_rates:       dict[str, float] = Field(default_factory=dict,
		description="Metric → cost per unit overage (in USD cents)")
	is_active:           bool       = Field(default=True)
	created_at:          str        = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	metadata:            dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SbTenant — a billable tenant/organisation
# ---------------------------------------------------------------------------

class SbTenant(BaseModel):
	"""A tenant (company / organisation) subscribed to the APG platform."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id:                  str        = Field(default_factory=uuid7str)
	name:                NonEmptyStr
	email:               NonEmptyStr                                     # billing contact email
	plan_id:             str        = Field(default="")
	status:              str        = Field(default="trial")             # active|suspended|cancelled|trial|past_due
	trial_ends_at:       str | None = Field(default=None)
	created_at:          str        = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	updated_at:          str        = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	billing_address:     dict[str, str] = Field(default_factory=dict)
	tax_id:              str | None = Field(default=None)
	metadata:            dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SbSubscription — active subscription for a tenant
# ---------------------------------------------------------------------------

class SbSubscription(BaseModel):
	"""An active (or historical) subscription linking a tenant to a plan."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id:                  str        = Field(default_factory=uuid7str)
	tenant_id:           NonEmptyStr
	plan_id:             NonEmptyStr
	billing_cycle:       str        = Field(default="monthly")          # monthly|annual
	status:              str        = Field(default="active")           # active|cancelled|past_due|trialing|paused
	current_period_start: str       = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	current_period_end:  str        = Field(default="")
	next_renewal_at:     str | None = Field(default=None)
	cancelled_at:        str | None = Field(default=None)
	cancellation_reason: str | None = Field(default=None)
	proration_credit_cents: NonNegInt = Field(default=0,
		description="Unused credit from mid-cycle upgrade (in cents)")
	created_at:          str        = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	metadata:            dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SbUsageRecord — a single metered usage event
# ---------------------------------------------------------------------------

class SbUsageRecord(BaseModel):
	"""A single metered usage event (API call, storage write, transaction, etc.)."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id:                  str        = Field(default_factory=uuid7str)
	tenant_id:           NonEmptyStr
	subscription_id:     str        = Field(default="")
	metric:              NonEmptyStr                                     # one of SUPPORTED_USAGE_METRICS
	quantity:            PositiveInt
	timestamp:           str        = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	idempotency_key:     str | None = Field(default=None,
		description="Optional deduplication key; re-submission with same key is ignored")
	source:              str        = Field(default="api",
		description="Origin of the usage event: api|batch|webhook|internal")
	metadata:            dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SbInvoiceLineItem — a single line on an invoice
# ---------------------------------------------------------------------------

class SbInvoiceLineItem(BaseModel):
	"""A line item on an invoice (subscription fee, overage charge, adjustment)."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id:                  str        = Field(default_factory=uuid7str)
	invoice_id:          str        = Field(default="")
	description:         NonEmptyStr
	item_type:           str        = Field(default="subscription_fee") # subscription_fee|overage|adjustment|credit
	metric:              str | None = Field(default=None)               # for overage lines
	quantity:            float      = Field(default=1.0)
	unit_price_cents:    NonNegInt
	amount_cents:        NonNegInt
	period_start:        str | None = Field(default=None)
	period_end:          str | None = Field(default=None)
	metadata:            dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SbInvoice — a generated invoice
# ---------------------------------------------------------------------------

class SbInvoice(BaseModel):
	"""A tenant invoice covering a billing period."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id:                  str        = Field(default_factory=uuid7str)
	tenant_id:           NonEmptyStr
	subscription_id:     str        = Field(default="")
	invoice_number:      str        = Field(default="")                 # human-readable e.g. INV-2025-001
	status:              str        = Field(default="draft")            # draft|open|paid|void|uncollectible
	currency:            str        = Field(default="USD")
	period_start:        str        = Field(default="")
	period_end:          str        = Field(default="")
	subtotal_cents:      NonNegInt  = Field(default=0)
	tax_cents:           NonNegInt  = Field(default=0)
	discount_cents:      NonNegInt  = Field(default=0)
	total_cents:         NonNegInt  = Field(default=0)
	amount_due_cents:    NonNegInt  = Field(default=0)
	line_items:          list[SbInvoiceLineItem] = Field(default_factory=list)
	due_date:            str | None = Field(default=None)
	paid_at:             str | None = Field(default=None)
	voided_at:           str | None = Field(default=None)
	created_at:          str        = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	metadata:            dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SbPaymentMethod — tokenized payment instrument (never raw card data)
# ---------------------------------------------------------------------------

class SbPaymentMethod(BaseModel):
	"""A tokenized payment method.  Raw card numbers are NEVER stored here.

	The ``token`` field holds a processor token (e.g. Stripe PaymentMethod id,
	M-Pesa wallet reference, PayPal billing agreement id).
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id:                  str        = Field(default_factory=uuid7str)
	tenant_id:           NonEmptyStr
	method_type:         NonEmptyStr                                    # card|bank_transfer|mpesa|paypal|stripe_token
	token:               NonEmptyStr                                    # processor token — never a raw card number
	last_four:           str | None = Field(default=None,              # last 4 digits of card if applicable
		description="Display hint only — not a secret")
	brand:               str | None = Field(default=None)              # visa|mastercard|amex|mpesa…
	expiry_month:        int | None = Field(default=None)
	expiry_year:         int | None = Field(default=None)
	is_default:          bool       = Field(default=False)
	created_at:          str        = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	metadata:            dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SbCreditNote — credit note for refunds / adjustments
# ---------------------------------------------------------------------------

class SbCreditNote(BaseModel):
	"""A credit note issued against an invoice (refund, duplicate charge correction, etc.)."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id:                  str        = Field(default_factory=uuid7str)
	tenant_id:           NonEmptyStr
	invoice_id:          NonEmptyStr
	reason:              NonEmptyStr                                    # one of SUPPORTED_CREDIT_NOTE_REASONS
	amount_cents:        PositiveInt
	currency:            str        = Field(default="USD")
	description:         str        = Field(default="")
	approved_by:         str | None = Field(default=None)
	approved_at:         str | None = Field(default=None)
	issued_at:           str        = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	metadata:            dict[str, Any] = Field(default_factory=dict)
