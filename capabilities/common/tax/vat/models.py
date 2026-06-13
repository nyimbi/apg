"""Pydantic v2 models for APG VAT/GST Country Rule Packs.

Model prefix: Tx (Tax)
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from uuid_extensions import uuid7str as _uuid7str


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


class TxVatRate(BaseModel):
	"""VAT rate entry for a specific country, category, and effective period.

	Ghana compound rates (NHIL + GETFund) are modelled as separate entries
	with is_levy=True so they render distinctly on tax invoices.
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=_uuid7str)
	country_code: str = Field(..., min_length=2, max_length=2)
	vat_category: str
	rate_pct: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
	is_levy: bool = Field(default=False, description="True for Ghana NHIL/GETFund-style levies")
	levy_name: str = Field(default="", description="Levy short label, e.g. 'NHIL', 'GETFund'")
	authority_name: str
	authority_ref: str = ""
	effective_from: date
	effective_to: date | None = None
	notes: str = ""
	created_at: datetime = Field(default_factory=_utcnow)

	@field_validator("country_code")
	@classmethod
	def country_upper(cls, v: str) -> str:
		return v.upper()

	@field_validator("vat_category")
	@classmethod
	def category_lower(cls, v: str) -> str:
		return v.lower().strip()

	@property
	def is_active(self) -> bool:
		today = datetime.now(timezone.utc).date()
		if self.effective_from > today:
			return False
		if self.effective_to and self.effective_to < today:
			return False
		return True


class TxVatCountryConfig(BaseModel):
	"""Per-country VAT configuration — registration thresholds, filing frequency, etc."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=_uuid7str)
	country_code: str = Field(..., min_length=2, max_length=2)
	authority_name: str
	authority_website: str = ""
	registration_threshold_local: Decimal | None = Field(
		default=None,
		description="Annual turnover threshold for mandatory VAT registration in local currency",
	)
	threshold_currency: str = Field(default="", min_length=0, max_length=3)
	filing_frequency: str = Field(default="monthly", description="monthly | quarterly | annual")
	standard_rate_pct: Decimal
	has_zero_rated: bool = True
	has_exempt: bool = True
	has_compound_levies: bool = Field(default=False, description="e.g. Ghana NHIL + GETFund")
	compound_levy_names: list[str] = Field(default_factory=list)
	digital_services_tax_pct: Decimal | None = None
	notes: str = ""
	updated_at: datetime = Field(default_factory=_utcnow)

	@field_validator("country_code")
	@classmethod
	def country_upper(cls, v: str) -> str:
		return v.upper()

	@field_validator("filing_frequency")
	@classmethod
	def freq_lower(cls, v: str) -> str:
		return v.lower().strip()


class TxVatReturn(BaseModel):
	"""VAT return filed with a tax authority for a given period."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	country_code: str
	tax_period_id: str = Field(..., description="Links to TxTaxPeriod in calc subcap")
	period_name: str
	period_start: date
	period_end: date
	filing_due_date: date
	output_vat: Decimal = Field(default=Decimal("0"), description="VAT collected on sales")
	input_vat: Decimal = Field(default=Decimal("0"), description="VAT paid on purchases (recoverable)")
	net_vat_payable: Decimal = Field(default=Decimal("0"), description="output - input; negative = refund")
	currency: str = "KES"
	status: str = "draft"
	submitted_at: datetime | None = None
	submitted_by: str | None = None
	authority_reference: str | None = Field(default=None, description="Reference number assigned by tax authority")
	notes: str = ""
	created_at: datetime = Field(default_factory=_utcnow)

	@model_validator(mode="after")
	def compute_net(self) -> "TxVatReturn":
		self.net_vat_payable = self.output_vat - self.input_vat
		return self

	@field_validator("country_code")
	@classmethod
	def country_upper(cls, v: str) -> str:
		return v.upper()

	@field_validator("status")
	@classmethod
	def status_lower(cls, v: str) -> str:
		return v.lower().strip()


class TxVatExemption(BaseModel):
	"""Registered VAT exemption for a product/entity/transaction type."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	country_code: str
	exemption_type: str
	entity_reference: str = Field(..., description="Product ID, supplier ID, or entity ID exempt")
	authority_ref: str = Field(default="", description="Official gazette or ruling reference")
	evidence_reference: str = Field(..., description="Document store reference for evidence")
	granted_from: date
	expires_at: date | None = None
	notes: str = ""
	created_at: datetime = Field(default_factory=_utcnow)
	created_by: str = "system"

	@field_validator("country_code")
	@classmethod
	def country_upper(cls, v: str) -> str:
		return v.upper()

	@field_validator("exemption_type")
	@classmethod
	def type_lower(cls, v: str) -> str:
		return v.lower().strip()

	@property
	def is_active(self) -> bool:
		today = datetime.now(timezone.utc).date()
		if self.granted_from > today:
			return False
		if self.expires_at and self.expires_at < today:
			return False
		return True
