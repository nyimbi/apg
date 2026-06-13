"""Pydantic v2 models for APG Tax Calculation Engine.

Model prefix: Tx (Tax)
All monetary amounts in minor currency units unless explicitly noted.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from uuid_extensions import uuid7str as _uuid7str


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Core value objects
# ---------------------------------------------------------------------------

class TxMoney(BaseModel):
	"""Immutable monetary value with currency."""
	model_config = ConfigDict(extra="forbid", frozen=True)

	amount: Decimal = Field(..., description="Amount in major currency units (e.g. KES 1500.00)")
	currency: str = Field(..., min_length=3, max_length=3, description="ISO 4217 currency code")

	@field_validator("amount")
	@classmethod
	def amount_non_negative(cls, v: Decimal) -> Decimal:
		if v < 0:
			raise ValueError("amount must be non-negative")
		return v

	@field_validator("currency")
	@classmethod
	def currency_upper(cls, v: str) -> str:
		return v.upper()


class TxApplicableRate(BaseModel):
	"""Rate entry resolved for a specific calculation context."""
	model_config = ConfigDict(extra="forbid", frozen=True)

	rate_id: str
	tax_type: str
	country_code: str
	product_category: str
	rate_pct: Decimal = Field(..., description="Rate as a percentage, e.g. 16.0 means 16%")
	is_compound: bool = Field(default=False, description="True if applied on top of another tax")
	effective_from: date
	effective_to: date | None = None
	source: str = Field(..., description="Authoritative source: KRA, FIRS, etc.")


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------

class TxTaxRate(BaseModel):
	"""Master tax rate record.

	Keyed by (country_code, tax_type, product_category, effective_from).
	Multiple rates per country/type are allowed for compound taxes (e.g. Ghana VAT bundle).
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	country_code: str = Field(..., min_length=2, max_length=2)
	tax_type: str
	product_category: str
	rate_pct: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
	is_compound: bool = False
	compound_base: str | None = Field(default=None, description="Rate ID this is compounded on top of")
	authority_name: str = Field(..., description="Tax authority short name, e.g. KRA")
	authority_ref: str | None = Field(default=None, description="Official gazette / act reference")
	effective_from: date
	effective_to: date | None = None
	notes: str = ""
	created_at: datetime = Field(default_factory=_utcnow)
	created_by: str = "system"

	@field_validator("country_code")
	@classmethod
	def country_upper(cls, v: str) -> str:
		return v.upper()

	@field_validator("tax_type", "product_category")
	@classmethod
	def code_lower(cls, v: str) -> str:
		return v.lower().strip()

	def as_applicable_rate(self) -> TxApplicableRate:
		return TxApplicableRate(
			rate_id=self.id,
			tax_type=self.tax_type,
			country_code=self.country_code,
			product_category=self.product_category,
			rate_pct=self.rate_pct,
			is_compound=self.is_compound,
			effective_from=self.effective_from,
			effective_to=self.effective_to,
			source=self.authority_name,
		)


class TxTaxCalculation(BaseModel):
	"""Single tax calculation result — the primary output of calculate_tax().

	Retained permanently for audit. Each external tax-triggering event (invoice
	line, payroll run, purchase order approval) creates one or more of these.
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	reference_id: str = Field(..., description="ID of the source document (invoice_id, po_id, etc.)")
	reference_type: str = Field(..., description="Document type: invoice, payroll_run, purchase_order, etc.")
	tax_type: str
	country_code: str
	product_category: str
	entity_type: str = "company"
	treaty_status: str = "domestic"

	# Monetary inputs
	taxable_amount: Decimal = Field(..., ge=Decimal("0"))
	currency: str = Field(default="KES", min_length=3, max_length=3)

	# Resolved rates (may be multiple for compound taxes like Ghana)
	applicable_rates: list[TxApplicableRate] = Field(default_factory=list)

	# Computed outputs
	tax_amount: Decimal = Field(default=Decimal("0"))
	total_amount: Decimal = Field(default=Decimal("0"))  # taxable + tax

	# Breakdown for compound taxes (Ghana NHIL, GETFund etc.)
	tax_breakdown: list[dict[str, Any]] = Field(default_factory=list)

	# Override tracking
	rate_overridden: bool = False
	override_rate_pct: Decimal | None = None
	override_justification: str | None = None
	override_approved_by: str | None = None

	# Status
	status: str = "calculated"
	period_id: str | None = None
	calculated_at: datetime = Field(default_factory=_utcnow)
	calculated_by: str = "system"
	amended_from_id: str | None = None
	notes: str = ""

	@field_validator("currency")
	@classmethod
	def currency_upper(cls, v: str) -> str:
		return v.upper()

	@field_validator("country_code")
	@classmethod
	def country_upper(cls, v: str) -> str:
		return v.upper()

	@field_validator("tax_type", "product_category", "entity_type", "treaty_status", "status")
	@classmethod
	def code_lower(cls, v: str) -> str:
		return v.lower().strip()


class TxTaxPeriod(BaseModel):
	"""Tax filing period (monthly, quarterly, annual).

	Invoicing, payroll, and commerce caps attach calculations to open periods.
	When a period is closed/filed, no further calculations can be attached.
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	country_code: str
	tax_type: str
	period_name: str = Field(..., description="Human-readable label: e.g. 'Jan 2025', 'Q1 2025'")
	period_start: date
	period_end: date
	filing_due_date: date
	status: str = "open"  # open | closed | filed | paid | amended
	total_taxable_amount: Decimal = Field(default=Decimal("0"))
	total_tax_amount: Decimal = Field(default=Decimal("0"))
	currency: str = "KES"
	calculation_ids: list[str] = Field(default_factory=list)
	filed_at: datetime | None = None
	filed_by: str | None = None
	payment_reference: str | None = None
	created_at: datetime = Field(default_factory=_utcnow)

	@model_validator(mode="after")
	def period_dates_valid(self) -> "TxTaxPeriod":
		if self.period_end < self.period_start:
			raise ValueError("period_end must be >= period_start")
		if self.filing_due_date < self.period_end:
			raise ValueError("filing_due_date must be >= period_end")
		return self


class TxTaxAudit(BaseModel):
	"""Immutable audit record for every tax event.

	Modelled as append-only — once written, never mutated.
	Regulatory retention: 7 years minimum across all African jurisdictions.
	"""
	model_config = ConfigDict(extra="forbid", frozen=True)

	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	action: str = Field(..., description="Audit action from SUPPORTED_AUDIT_ACTIONS")
	reference_id: str = Field(..., description="ID of the affected entity (calc_id, rate_id, period_id)")
	reference_type: str = Field(..., description="Entity type: calculation, rate, period, override")
	actor: str = Field(default="system")
	country_code: str | None = None
	tax_type: str | None = None
	snapshot: dict[str, Any] = Field(default_factory=dict, description="JSON snapshot of state at time of event")
	occurred_at: datetime = Field(default_factory=_utcnow)
	notes: str = ""


class TxTaxResult(BaseModel):
	"""Return value of TaxCalcService.calculate_tax() — the primary API contract.

	Consumers (invoicing, payroll, commerce) call calculate_tax() and receive
	this object. They must persist the calculation_id for audit linkage.
	"""
	model_config = ConfigDict(extra="forbid", frozen=True)

	calculation_id: str
	tenant_id: str
	reference_id: str
	reference_type: str
	tax_type: str
	country_code: str
	product_category: str
	taxable_amount: Decimal
	tax_amount: Decimal
	total_amount: Decimal
	currency: str
	effective_rate_pct: Decimal = Field(..., description="Blended effective rate applied")
	tax_breakdown: list[dict[str, Any]]
	calculated_at: datetime
	period_id: str | None
	notes: str = ""


class TxRateLookupRequest(BaseModel):
	"""Input for get_rate() — used by caching layer and external callers."""
	model_config = ConfigDict(extra="forbid")

	country_code: str
	tax_type: str
	product_category: str
	as_of_date: date
	entity_type: str = "company"
	treaty_status: str = "domestic"

	@field_validator("country_code")
	@classmethod
	def country_upper(cls, v: str) -> str:
		return v.upper()

	@field_validator("tax_type", "product_category", "entity_type", "treaty_status")
	@classmethod
	def code_lower(cls, v: str) -> str:
		return v.lower().strip()


class TxCalculationRequest(BaseModel):
	"""Input for calculate_tax() — validated before service dispatch."""
	model_config = ConfigDict(extra="forbid")

	tenant_id: str
	reference_id: str
	reference_type: str
	tax_type: str
	country_code: str
	product_category: str
	taxable_amount: Decimal = Field(..., gt=Decimal("0"))
	currency: str = "KES"
	as_of_date: date | None = None
	entity_type: str = "company"
	treaty_status: str = "domestic"
	period_id: str | None = None
	notes: str = ""

	@field_validator("currency")
	@classmethod
	def currency_upper(cls, v: str) -> str:
		return v.upper()

	@field_validator("country_code")
	@classmethod
	def country_upper(cls, v: str) -> str:
		return v.upper()

	@field_validator("tax_type", "product_category", "entity_type", "treaty_status")
	@classmethod
	def code_lower(cls, v: str) -> str:
		return v.lower().strip()
