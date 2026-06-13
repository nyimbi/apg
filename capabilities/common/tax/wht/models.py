"""Pydantic v2 models for APG Withholding Tax (WHT) Engine.

Model prefix: Tx (Tax)
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid_extensions import uuid7str as _uuid7str


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


class TxWhtRate(BaseModel):
	"""WHT rate for a country × payment type × treaty status × entity type.

	Kenya examples:
	  professional_fees / domestic / company  → 5%
	  rent             / domestic / company  → 3%
	  dividends        / domestic / company  → 10% (listed), 15% (unlisted)
	  dividends        / treaty_reduced / company → varies by treaty country
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=_uuid7str)
	country_code: str = Field(..., min_length=2, max_length=2)
	payment_type: str
	treaty_status: str = "domestic"
	entity_type: str = "company"
	treaty_country_code: str | None = Field(
		default=None,
		description="Counterparty country for reduced treaty rates; None for domestic",
	)
	rate_pct: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
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

	@field_validator("payment_type", "treaty_status", "entity_type")
	@classmethod
	def code_lower(cls, v: str) -> str:
		return v.lower().strip()

	@property
	def is_active(self) -> bool:
		today = datetime.now(timezone.utc).date()
		if self.effective_from > today:
			return False
		if self.effective_to and self.effective_to < today:
			return False
		return True


class TxWhtCertificate(BaseModel):
	"""WHT certificate issued to a payee confirming tax was withheld.

	In Kenya (M-Service / iTax), South Africa (SARS eFiling), and most African
	jurisdictions, WHT certificates must be issued to payees within a statutory
	period (e.g. 30 days of withholding in Kenya).
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	certificate_number: str = Field(..., description="Sequential number assigned by the system")
	country_code: str
	payer_id: str = Field(..., description="Entity that withheld the tax")
	payer_name: str
	payee_id: str = Field(..., description="Entity from whom tax was withheld")
	payee_name: str
	payee_tax_pin: str = Field(default="", description="Payee's tax PIN / TIN")
	payment_type: str
	gross_payment: Decimal = Field(..., ge=Decimal("0"))
	wht_rate_pct: Decimal
	wht_amount: Decimal
	currency: str = "KES"
	payment_date: date
	payment_reference: str = Field(default="", description="ID of the source payment")
	wht_return_id: str | None = None
	status: str = "issued"
	issued_at: datetime = Field(default_factory=_utcnow)
	issued_by: str = "system"
	notes: str = ""

	@field_validator("country_code")
	@classmethod
	def country_upper(cls, v: str) -> str:
		return v.upper()

	@field_validator("payment_type", "status")
	@classmethod
	def code_lower(cls, v: str) -> str:
		return v.lower().strip()

	@field_validator("currency")
	@classmethod
	def currency_upper(cls, v: str) -> str:
		return v.upper()


class TxWhtReturn(BaseModel):
	"""Quarterly (or monthly) WHT return filed with the tax authority.

	Aggregates all TxWhtPayments in the period; references all issued certificates.
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	country_code: str
	period_name: str
	period_start: date
	period_end: date
	filing_due_date: date
	total_gross_payments: Decimal = Field(default=Decimal("0"))
	total_wht_amount: Decimal = Field(default=Decimal("0"))
	currency: str = "KES"
	payment_ids: list[str] = Field(default_factory=list)
	certificate_ids: list[str] = Field(default_factory=list)
	status: str = "draft"
	submitted_at: datetime | None = None
	submitted_by: str | None = None
	authority_reference: str | None = None
	notes: str = ""
	created_at: datetime = Field(default_factory=_utcnow)

	@field_validator("country_code")
	@classmethod
	def country_upper(cls, v: str) -> str:
		return v.upper()

	@field_validator("status")
	@classmethod
	def status_lower(cls, v: str) -> str:
		return v.lower().strip()


class TxWhtPayment(BaseModel):
	"""Individual payment transaction that attracted WHT.

	One payment → one certificate (or one deferred certificate).
	Many payments aggregate into one TxWhtReturn per period.
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	country_code: str
	payer_id: str
	payee_id: str
	payment_type: str
	treaty_status: str = "domestic"
	treaty_country_code: str | None = None
	gross_amount: Decimal = Field(..., ge=Decimal("0"))
	wht_rate_pct: Decimal
	wht_amount: Decimal
	net_amount: Decimal
	currency: str = "KES"
	payment_date: date
	source_document_id: str = Field(default="", description="Invoice ID, payroll run ID, etc.")
	source_document_type: str = ""
	certificate_id: str | None = None
	return_id: str | None = None
	notes: str = ""
	created_at: datetime = Field(default_factory=_utcnow)

	@field_validator("country_code")
	@classmethod
	def country_upper(cls, v: str) -> str:
		return v.upper()

	@field_validator("payment_type", "treaty_status")
	@classmethod
	def code_lower(cls, v: str) -> str:
		return v.lower().strip()

	@field_validator("currency")
	@classmethod
	def currency_upper(cls, v: str) -> str:
		return v.upper()
