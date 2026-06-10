"""Pydantic v2 models for SACCO Deposits & Savings."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


# ── Savings Product ───────────────────────────────────────────────────────────

class SavingsProductCreateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	product_code: str
	product_name: str
	product_type: str  # regular | fixed_deposit | holiday | junior | institutional
	interest_rate_pa: Decimal  # % per annum
	min_balance: Decimal = Decimal("0")
	min_opening_balance: Decimal = Decimal("0")
	max_balance: Decimal | None = None
	lock_in_months: int = 0
	interest_posting_frequency: str = "monthly"  # daily | monthly | quarterly | annually
	withdrawal_notice_days: int = 0
	allow_overdraft: bool = False
	tax_exempt: bool = False
	description: str | None = None


class SavingsProductUpdateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	interest_rate_pa: Decimal | None = None
	min_balance: Decimal | None = None
	max_balance: Decimal | None = None
	description: str | None = None
	is_active: bool | None = None


class SavingsProductResponseModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	product_code: str
	product_name: str
	product_type: str
	interest_rate_pa: Decimal
	min_balance: Decimal
	min_opening_balance: Decimal
	max_balance: Decimal | None = None
	lock_in_months: int
	interest_posting_frequency: str
	withdrawal_notice_days: int
	is_active: bool
	created_at: str


# ── Account ───────────────────────────────────────────────────────────────────

class SavingsAccountCreateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	product_id: str
	opening_balance: Decimal = Decimal("0")
	currency: str = "KES"
	account_name: str | None = None
	maturity_date: str | None = None  # for fixed deposits


class SavingsAccountUpdateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_name: str | None = None
	status: str | None = None


class SavingsAccountResponseModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	account_number: str
	tenant_id: str
	member_id: str
	product_id: str
	balance: Decimal
	available_balance: Decimal
	currency: str
	status: str
	created_at: str
	updated_at: str | None = None


# ── Transaction ───────────────────────────────────────────────────────────────

class DepositModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	amount: Decimal
	payment_reference: str
	payment_method: str = "cash"
	narration: str | None = None
	recorded_by: str


class WithdrawalModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	amount: Decimal
	payment_method: str = "cash"
	narration: str | None = None
	approved_by: str
	payment_reference: str | None = None


# ── Interest Accrual ──────────────────────────────────────────────────────────

class InterestAccrualModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	period_start: str
	period_end: str
	posting_date: str
	run_by: str
	accounts: list[str] | None = None  # None = all accounts


# ── Filter ────────────────────────────────────────────────────────────────────

class SavingsFilterModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str | None = None
	product_id: str | None = None
	status: str | None = None
	min_balance: Decimal | None = None
	from_date: str | None = None
	to_date: str | None = None


# ── Audit ─────────────────────────────────────────────────────────────────────

class SavingsAuditModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	account_id: str | None = None
	member_id: str | None = None
	amount: Decimal | None = None
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: str


# ── List ──────────────────────────────────────────────────────────────────────

class SavingsListModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[dict[str, Any]] = Field(default_factory=list)
	total: int = 0
	page: int = 1
	page_size: int = 50
