"""Pydantic v2 models for APG Three-Way Match Engine (proc_twy).

Model prefix: Tw
Domain: procurement / accounts payable

A three-way match validates that a Vendor Invoice agrees with the originating
Purchase Order and the Goods Receipt.  Any discrepancy beyond configured
tolerance triggers an Exception that routes to an AP reviewer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())

except ImportError:  # pragma: no cover — shim for environments missing uuid6
	import uuid

	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TwDocumentType(StrEnum):
	PURCHASE_ORDER = "purchase_order"
	GOODS_RECEIPT = "goods_receipt"
	VENDOR_INVOICE = "vendor_invoice"


class TwMatchOutcome(StrEnum):
	MATCHED = "matched"
	PARTIAL_MATCH = "partial_match"
	EXCEPTION = "exception"


class TwMatchStatus(StrEnum):
	"""Lifecycle status of a TwMatchAttempt."""
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"


class TwExceptionStatus(StrEnum):
	OPEN = "open"
	PENDING_REVIEW = "pending_review"
	ESCALATED = "escalated"
	RESOLVED = "resolved"
	CANCELLED = "cancelled"


class TwExceptionResolutionType(StrEnum):
	APPROVED_WITH_VARIANCE = "approved_with_variance"
	REJECTED = "rejected"
	DUPLICATE_INVOICE = "duplicate_invoice"
	CANCELLED_PO = "cancelled_po"
	GOODS_NOT_RECEIVED = "goods_not_received"
	PRICE_CORRECTION = "price_correction"
	QUANTITY_CORRECTION = "quantity_correction"
	DATE_EXTENSION = "date_extension"
	MANUAL_OVERRIDE = "manual_override"


class TwVarianceType(StrEnum):
	PRICE = "price"
	QUANTITY = "quantity"
	DATE = "date"
	LINE_MISSING = "line_missing"
	DOCUMENT_MISSING = "document_missing"


class TwToleranceScope(StrEnum):
	GLOBAL = "global"
	VENDOR = "vendor"
	CATEGORY = "category"
	LINE_ITEM = "line_item"


class TwEscalationTarget(StrEnum):
	AP_MANAGER = "ap_manager"
	PROCUREMENT_MANAGER = "procurement_manager"
	CFO = "cfo"
	VENDOR_MANAGER = "vendor_manager"
	COMPLIANCE = "compliance"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _non_empty_str(v: str) -> str:
	assert v and v.strip(), "must be a non-empty string"
	return v.strip()


def _positive_decimal(v: Decimal) -> Decimal:
	assert v > 0, "must be positive"
	return v


def _non_negative_decimal(v: Decimal) -> Decimal:
	assert v >= 0, "must be >= 0"
	return v


def _pct_tolerance(v: float) -> float:
	assert 0.0 <= v <= 100.0, "tolerance percentage must be between 0 and 100"
	return v


NonEmptyStr = Annotated[str, AfterValidator(_non_empty_str)]
PositiveDecimal = Annotated[Decimal, AfterValidator(_positive_decimal)]
NonNegativeDecimal = Annotated[Decimal, AfterValidator(_non_negative_decimal)]
PctTolerance = Annotated[float, AfterValidator(_pct_tolerance)]


# ---------------------------------------------------------------------------
# Line-level sub-models
# ---------------------------------------------------------------------------


class TwDocumentLine(BaseModel):
	"""A single line on a PO, GR, or Invoice."""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	line_number: int = Field(..., description="1-based line number within the document")
	item_code: NonEmptyStr = Field(..., description="SKU / catalogue item code")
	description: str = Field(default="", description="Human-readable line description")
	quantity: NonNegativeDecimal = Field(..., description="Ordered / received / invoiced quantity")
	unit_price: NonNegativeDecimal = Field(..., description="Unit price in document currency")
	unit_of_measure: str = Field(default="EA", description="Unit of measure (EA, KG, L, …)")
	line_total: NonNegativeDecimal = Field(..., description="quantity × unit_price (may carry rounding)")
	tax_amount: NonNegativeDecimal = Field(default=Decimal("0.00"))
	discount_amount: NonNegativeDecimal = Field(default=Decimal("0.00"))
	account_code: str = Field(default="", description="GL account code for this line")
	metadata: dict[str, Any] = Field(default_factory=dict)


class TwVarianceDetail(BaseModel):
	"""A single variance identified during matching."""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	variance_type: TwVarianceType
	line_number: int | None = None
	field_name: str = Field(..., description="e.g. 'unit_price', 'quantity', 'invoice_date'")
	po_value: str | None = Field(default=None, description="String representation of PO value")
	gr_value: str | None = Field(default=None, description="String representation of GR value")
	invoice_value: str | None = Field(default=None, description="String representation of Invoice value")
	absolute_variance: Decimal | None = None
	percentage_variance: float | None = None
	within_tolerance: bool = False
	tolerance_rule_id: str | None = None
	note: str = ""


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


class TwMatchDocument(BaseModel):
	"""A procurement document — PO, GR, or Vendor Invoice — ingested into the engine."""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str, description="Internal UUID7 document ID")
	tenant_id: NonEmptyStr
	document_type: TwDocumentType
	external_ref: NonEmptyStr = Field(..., description="Source system reference (e.g. ERP PO number)")
	vendor_id: NonEmptyStr
	vendor_name: str = Field(default="")
	currency: str = Field(default="KES", description="ISO 4217 currency code")
	document_date: datetime
	delivery_date: datetime | None = None
	payment_terms_days: int = Field(default=30, ge=0)
	total_amount: NonNegativeDecimal
	tax_amount: NonNegativeDecimal = Field(default=Decimal("0.00"))
	lines: list[TwDocumentLine] = Field(default_factory=list)
	status: str = Field(default="active")
	raw_payload: dict[str, Any] = Field(default_factory=dict, description="Original source payload for audit")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	metadata: dict[str, Any] = Field(default_factory=dict)


class TwMatchAttempt(BaseModel):
	"""Records a single attempt to match a (PO, GR, Invoice) triple."""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	po_id: NonEmptyStr = Field(..., description="ID of the TwMatchDocument with type=purchase_order")
	gr_id: NonEmptyStr = Field(..., description="ID of the TwMatchDocument with type=goods_receipt")
	invoice_id: NonEmptyStr = Field(..., description="ID of the TwMatchDocument with type=vendor_invoice")
	status: TwMatchStatus = TwMatchStatus.PENDING
	variances: list[TwVarianceDetail] = Field(default_factory=list)
	tolerance_rules_applied: list[str] = Field(default_factory=list, description="IDs of TwMatchToleranceRule applied")
	initiated_by: str = Field(default="system")
	started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	completed_at: datetime | None = None
	error: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class TwMatchResult(BaseModel):
	"""Final outcome of a three-way match attempt."""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	match_attempt_id: NonEmptyStr
	po_id: NonEmptyStr
	gr_id: NonEmptyStr
	invoice_id: NonEmptyStr
	outcome: TwMatchOutcome
	# Summary amounts pulled from documents for quick reference
	po_total: Decimal
	gr_total: Decimal
	invoice_total: Decimal
	price_variance_pct: float = Field(default=0.0, description="Max price variance across lines (%)")
	quantity_variance_pct: float = Field(default=0.0, description="Max qty variance across lines (%)")
	date_variance_days: int = Field(default=0, description="Invoice date vs PO payment-terms deadline (days)")
	all_within_tolerance: bool
	variances: list[TwVarianceDetail] = Field(default_factory=list)
	exception_id: str | None = Field(default=None, description="Set when outcome == EXCEPTION")
	auto_approved: bool = False
	matched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	matched_by: str = Field(default="system")
	audit_trail: list[dict[str, Any]] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


class TwMatchException(BaseModel):
	"""An exception requiring human review, raised when match variances exceed tolerance."""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	match_result_id: NonEmptyStr
	po_id: NonEmptyStr
	gr_id: NonEmptyStr
	invoice_id: NonEmptyStr
	status: TwExceptionStatus = TwExceptionStatus.OPEN
	variance_summary: list[TwVarianceDetail] = Field(default_factory=list)
	# Lifecycle fields
	assigned_to: str | None = None
	escalated_to: str | None = None
	escalation_reason: str | None = None
	escalated_at: datetime | None = None
	resolution_type: TwExceptionResolutionType | None = None
	resolution_note: str = ""
	resolved_by: str | None = None
	resolved_at: datetime | None = None
	# Governance
	raised_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	due_at: datetime | None = None
	age_days: float = Field(default=0.0)
	audit_trail: list[dict[str, Any]] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


class TwMatchToleranceRule(BaseModel):
	"""Configurable tolerance rule — evaluated in specificity order (global < vendor < category < line_item)."""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	scope: TwToleranceScope = TwToleranceScope.GLOBAL
	# Scope selectors — only the relevant one is used based on scope
	vendor_id: str | None = None
	category_code: str | None = None
	item_code: str | None = None
	# Tolerance values
	price_tolerance_pct: PctTolerance = Field(default=2.0, description="Allowed price variance in %")
	quantity_tolerance_pct: PctTolerance = Field(default=5.0, description="Allowed quantity variance in %")
	date_tolerance_days: int = Field(default=30, ge=0, description="Days invoice date may exceed payment terms")
	# Amount-band overrides: apply tighter tolerance above a threshold
	amount_threshold: Decimal | None = Field(default=None, description="If set, use stricter tolerances above this amount")
	price_tolerance_pct_above_threshold: PctTolerance | None = None
	quantity_tolerance_pct_above_threshold: PctTolerance | None = None
	# Rule metadata
	active: bool = True
	priority: int = Field(default=100, ge=1, description="Lower = evaluated first; global rules typically 100")
	effective_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	effective_to: datetime | None = None
	created_by: NonEmptyStr = Field(default="system")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	notes: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)
