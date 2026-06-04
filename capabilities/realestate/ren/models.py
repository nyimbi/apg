"""Pydantic v2 models for Rental Operations (ren)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class TenancyType(str, Enum):
	assured_shorthold = "assured_shorthold"
	fixed_term = "fixed_term"
	periodic = "periodic"
	licence = "licence"
	commercial = "commercial"
	regulated = "regulated"
	student = "student"
	social_housing = "social_housing"
	serviced_office = "serviced_office"


class TenancyStatus(str, Enum):
	application = "application"
	referencing = "referencing"
	approved = "approved"
	notice_signed = "notice_signed"
	active = "active"
	notice_served = "notice_served"
	holding_over = "holding_over"
	vacating = "vacating"
	vacated = "vacated"
	dispute = "dispute"


class RentFrequency(str, Enum):
	weekly = "weekly"
	fortnightly = "fortnightly"
	monthly = "monthly"
	quarterly = "quarterly"
	semi_annual = "semi_annual"
	annual = "annual"
	in_advance = "in_advance"


class PaymentMethod(str, Enum):
	bank_transfer = "bank_transfer"
	direct_debit = "direct_debit"
	standing_order = "standing_order"
	cheque = "cheque"
	cash = "cash"
	mpesa = "mpesa"
	credit_card = "credit_card"
	debit_card = "debit_card"


class ArrearsStatus(str, Enum):
	current = "current"
	days_1_30 = "1_30_days"
	days_31_60 = "31_60_days"
	days_61_90 = "61_90_days"
	days_90_plus = "90_plus_days"
	legal_action = "legal_action"
	write_off = "write_off"


class DepositType(str, Enum):
	cash_deposit = "cash_deposit"
	deposit_replacement_insurance = "deposit_replacement_insurance"
	guarantor_deposit = "guarantor_deposit"
	deed_of_guarantee = "deed_of_guarantee"
	zero_deposit = "zero_deposit"


class DepositStatus(str, Enum):
	held = "held"
	registered = "registered"
	released = "released"
	disputed = "disputed"
	deducted = "deducted"
	refunded = "refunded"


class NoticeType(str, Enum):
	section_21 = "section_21"
	section_8 = "section_8"
	notice_to_quit = "notice_to_quit"
	break_notice = "break_notice"
	forfeiture_notice = "forfeiture_notice"
	rent_increase_notice = "rent_increase_notice"


# ── Tenancy ───────────────────────────────────────────────────────────────────

class TenancyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	unit_id: str
	property_id: str
	tenant_entity_id: str
	tenancy_type: TenancyType
	start_date: date
	end_date: date | None = None
	rent_amount: Decimal
	rent_frequency: RentFrequency
	currency: str = "KES"
	payment_method: PaymentMethod = PaymentMethod.bank_transfer
	created_by: str

	@field_validator("rent_amount")
	@classmethod
	def _positive_rent(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("rent_amount must be positive")
		return v


class TenancyResponse(TenancyCreate):
	id: str = Field(default_factory=uuid7str)
	status: TenancyStatus = TenancyStatus.application
	deposit_id: str | None = None
	deposit_registered: bool = False
	referencing_complete: bool = False
	right_to_rent_checked: bool = False
	arrears_status: ArrearsStatus = ArrearsStatus.current
	total_arrears: Decimal = Decimal("0")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class TenancyUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: TenancyStatus | None = None
	rent_amount: Decimal | None = None
	end_date: date | None = None
	payment_method: PaymentMethod | None = None


# ── Rent Payment ──────────────────────────────────────────────────────────────

class RentPaymentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	tenancy_id: str
	amount: Decimal
	payment_date: date
	payment_method: PaymentMethod
	currency: str = "KES"
	period: str  # YYYY-MM
	reference: str | None = None
	created_by: str

	@field_validator("amount")
	@classmethod
	def _positive_amount(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("amount must be positive")
		return v


class RentPaymentResponse(RentPaymentCreate):
	id: str = Field(default_factory=uuid7str)
	is_short_payment: bool = False
	shortfall: Decimal = Decimal("0")
	receipt_number: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Arrears Record ────────────────────────────────────────────────────────────

class ArrearsRecordCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	tenancy_id: str
	amount_overdue: Decimal
	days_overdue: int
	currency: str = "KES"
	created_by: str


class ArrearsRecordResponse(ArrearsRecordCreate):
	id: str = Field(default_factory=uuid7str)
	status: ArrearsStatus = ArrearsStatus.days_1_30
	actions_taken: list[dict[str, Any]] = Field(default_factory=list)
	legal_action_commenced: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Deposit ───────────────────────────────────────────────────────────────────

class DepositCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	tenancy_id: str
	deposit_type: DepositType
	amount: Decimal
	currency: str = "KES"
	scheme: str | None = None
	scheme_reference: str | None = None
	registered_date: date | None = None
	created_by: str

	@field_validator("amount")
	@classmethod
	def _positive_amount(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("deposit amount must be positive")
		return v


class DepositResponse(DepositCreate):
	id: str = Field(default_factory=uuid7str)
	status: DepositStatus = DepositStatus.held
	deductions: list[dict[str, Any]] = Field(default_factory=list)
	total_deducted: Decimal = Decimal("0")
	refunded_amount: Decimal = Decimal("0")
	released_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class DepositDeductionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	deposit_id: str
	reason: str
	amount: Decimal
	evidence_document_ids: list[str] = Field(default_factory=list)
	created_by: str

	@field_validator("amount")
	@classmethod
	def _positive_deduction(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("deduction amount must be positive")
		return v


class DepositDeductionResponse(DepositDeductionCreate):
	id: str = Field(default_factory=uuid7str)
	approved_by: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Notice ────────────────────────────────────────────────────────────────────

class NoticeCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	tenancy_id: str
	notice_type: NoticeType
	served_date: date
	effective_date: date
	served_by: str
	method: str = "letter"
	created_by: str


class NoticeResponse(NoticeCreate):
	id: str = Field(default_factory=uuid7str)
	acknowledged_by_tenant: bool = False
	acknowledgement_date: date | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Tenancy Renewal ───────────────────────────────────────────────────────────

class TenancyRenewalCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	tenancy_id: str
	renewal_type: str
	new_start_date: date
	new_end_date: date
	new_rent: Decimal
	currency: str = "KES"
	created_by: str


class TenancyRenewalResponse(TenancyRenewalCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "pending"  # pending | offered | accepted | rejected
	offered_at: datetime | None = None
	accepted_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Referencing ───────────────────────────────────────────────────────────────

class ReferencingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	tenancy_id: str
	referencing_types: list[str]
	applicant_id: str
	created_by: str


class ReferencingResponse(ReferencingCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "pending"  # pending | in_progress | passed | failed | referred
	results: dict[str, Any] = Field(default_factory=dict)
	completed_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
