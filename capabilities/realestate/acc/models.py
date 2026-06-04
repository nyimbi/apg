"""Pydantic v2 models for Real Estate Accounting (acc)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.functional_validators import AfterValidator
from typing import Annotated

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


def _positive_amount(v: Decimal) -> Decimal:
	if v < 0:
		raise ValueError("amount must be non-negative")
	return v


PositiveDecimal = Annotated[Decimal, AfterValidator(_positive_amount)]


class LedgerType(str, Enum):
	property_ledger = "property_ledger"
	service_charge = "service_charge"
	cam_reconciliation = "cam_reconciliation"
	rental_income = "rental_income"
	security_deposit = "security_deposit"
	capex = "capex"
	opex = "opex"
	inter_company = "inter_company"


class AccountType(str, Enum):
	asset = "asset"
	liability = "liability"
	equity = "equity"
	revenue = "revenue"
	expense = "expense"
	contra = "contra"


class JournalType(str, Enum):
	manual = "manual"
	automatic = "automatic"
	recurring = "recurring"
	reversing = "reversing"
	closing = "closing"
	accrual = "accrual"
	prepayment = "prepayment"


class PostingStatus(str, Enum):
	draft = "draft"
	pending_approval = "pending_approval"
	approved = "approved"
	posted = "posted"
	reversed = "reversed"
	void = "void"


class ReconciliationStatus(str, Enum):
	draft = "draft"
	in_review = "in_review"
	approved = "approved"
	posted = "posted"
	disputed = "disputed"
	settled = "settled"


class RevenueMethod(str, Enum):
	straight_line = "straight_line"
	escalation_linked = "escalation_linked"
	percentage_rent = "percentage_rent"
	hybrid = "hybrid"


class Ifrs16Category(str, Enum):
	finance_lease = "finance_lease"
	operating_lease = "operating_lease"
	short_term_exemption = "short_term_exemption"
	low_value_exemption = "low_value_exemption"


class ChargeType(str, Enum):
	base_rent = "base_rent"
	service_charge = "service_charge"
	insurance = "insurance"
	utilities = "utilities"
	management_fee = "management_fee"
	parking = "parking"
	storage = "storage"
	ad_hoc = "ad_hoc"


# ── Account ──────────────────────────────────────────────────────────────────

class AccountCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str | None = None
	code: str
	name: str
	account_type: AccountType
	ledger_type: LedgerType
	currency: str = "KES"
	parent_account_id: str | None = None
	is_control_account: bool = False
	description: str | None = None
	created_by: str


class AccountResponse(AccountCreate):
	id: str = Field(default_factory=uuid7str)
	balance: Decimal = Decimal("0")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class AccountUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name: str | None = None
	description: str | None = None
	is_control_account: bool | None = None


# ── Journal Entry ────────────────────────────────────────────────────────────

class JournalLine(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	account_id: str
	account_code: str
	description: str
	debit: Decimal = Decimal("0")
	credit: Decimal = Decimal("0")
	property_id: str | None = None
	cost_centre: str | None = None

	@field_validator("debit", "credit")
	@classmethod
	def _non_negative(cls, v: Decimal) -> Decimal:
		if v < 0:
			raise ValueError("debit/credit must be non-negative")
		return v


class JournalEntryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	journal_type: JournalType
	reference: str
	period: str  # YYYY-MM
	journal_date: date
	description: str
	lines: list[JournalLine]
	currency: str = "KES"
	property_id: str | None = None
	created_by: str
	supporting_document_ids: list[str] = Field(default_factory=list)

	@field_validator("lines")
	@classmethod
	def _must_balance(cls, lines: list[JournalLine]) -> list[JournalLine]:
		total_debit = sum(l.debit for l in lines)
		total_credit = sum(l.credit for l in lines)
		if abs(total_debit - total_credit) > Decimal("0.01"):
			raise ValueError(f"journal must balance: debit={total_debit} credit={total_credit}")
		return lines


class JournalEntryResponse(JournalEntryCreate):
	id: str = Field(default_factory=uuid7str)
	status: PostingStatus = PostingStatus.draft
	total_debit: Decimal = Decimal("0")
	reversal_of_id: str | None = None
	approved_by: str | None = None
	posted_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Service Charge ───────────────────────────────────────────────────────────

class ServiceChargeCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	charge_type: ChargeType
	lease_id: str | None = None
	description: str
	amount: PositiveDecimal
	currency: str = "KES"
	period: str  # YYYY-MM
	due_date: date
	vat_rate: Decimal = Decimal("0")
	created_by: str


class ServiceChargeResponse(ServiceChargeCreate):
	id: str = Field(default_factory=uuid7str)
	status: PostingStatus = PostingStatus.draft
	vat_amount: Decimal = Decimal("0")
	total_amount: Decimal = Decimal("0")
	approved_by: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── CAM Reconciliation ───────────────────────────────────────────────────────

class CamReconciliationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	period_year: int
	estimated_costs: PositiveDecimal
	actual_costs: Decimal | None = None
	currency: str = "KES"
	lease_ids: list[str] = Field(default_factory=list)
	cam_method: str = "pro_rata"
	created_by: str
	notes: str | None = None


class CamReconciliationResponse(CamReconciliationCreate):
	id: str = Field(default_factory=uuid7str)
	status: ReconciliationStatus = ReconciliationStatus.draft
	variance: Decimal = Decimal("0")
	approved_by: str | None = None
	settled_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── IFRS 16 Schedule ─────────────────────────────────────────────────────────

class Ifrs16ScheduleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	category: Ifrs16Category
	commencement_date: date
	expiry_date: date
	annual_payment: PositiveDecimal
	discount_rate: Decimal  # e.g. 0.05 for 5%
	currency: str = "KES"
	created_by: str

	@field_validator("discount_rate")
	@classmethod
	def _rate_range(cls, v: Decimal) -> Decimal:
		if not (Decimal("0") < v < Decimal("1")):
			raise ValueError("discount_rate must be between 0 and 1 exclusive")
		return v


class Ifrs16ScheduleResponse(Ifrs16ScheduleCreate):
	id: str = Field(default_factory=uuid7str)
	rou_asset: Decimal = Decimal("0")
	lease_liability: Decimal = Decimal("0")
	schedule_lines: list[dict[str, Any]] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Revenue Schedule ─────────────────────────────────────────────────────────

class RevenueScheduleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	property_id: str
	method: RevenueMethod
	start_date: date
	end_date: date
	total_contract_value: PositiveDecimal
	currency: str = "KES"
	created_by: str


class RevenueScheduleResponse(RevenueScheduleCreate):
	id: str = Field(default_factory=uuid7str)
	recognised_to_date: Decimal = Decimal("0")
	deferred_revenue: Decimal = Decimal("0")
	schedule_lines: list[dict[str, Any]] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Period ───────────────────────────────────────────────────────────────────

class AccountingPeriodCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	period: str  # YYYY-MM
	property_ids: list[str] = Field(default_factory=list)
	opened_by: str


class AccountingPeriodResponse(AccountingPeriodCreate):
	id: str = Field(default_factory=uuid7str)
	is_open: bool = True
	closed_by: str | None = None
	closed_at: datetime | None = None
	second_approver: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Tenant Statement ─────────────────────────────────────────────────────────

class TenantStatementCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	lease_id: str
	statement_period: str  # YYYY-MM
	opening_balance: Decimal = Decimal("0")
	currency: str = "KES"
	created_by: str


class TenantStatementResponse(TenantStatementCreate):
	id: str = Field(default_factory=uuid7str)
	charges: list[dict[str, Any]] = Field(default_factory=list)
	payments: list[dict[str, Any]] = Field(default_factory=list)
	closing_balance: Decimal = Decimal("0")
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	sent_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
