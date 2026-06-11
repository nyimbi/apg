"""
Bank Account Management — Pydantic v2 models.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


_CFG = ConfigDict(
	extra="forbid",
	validate_by_name=True,
	validate_by_alias=True,
	populate_by_name=True,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AccountStatus(str, Enum):
	PENDING = "pending"
	ACTIVE = "active"
	FROZEN = "frozen"
	DORMANT = "dormant"
	CLOSED = "closed"


class AccountType(str, Enum):
	CURRENT = "current"
	SAVINGS = "savings"
	FIXED_DEPOSIT = "fixed_deposit"
	LOAN = "loan"
	OVERDRAFT = "overdraft"
	ESCROW = "escrow"


class TransactionType(str, Enum):
	DEPOSIT = "deposit"
	WITHDRAWAL = "withdrawal"
	TRANSFER_IN = "transfer_in"
	TRANSFER_OUT = "transfer_out"
	FEE = "fee"
	INTEREST = "interest"
	REVERSAL = "reversal"
	ADJUSTMENT = "adjustment"
	BULK_CREDIT = "bulk_credit"
	LOCK = "lock"
	RELEASE = "release"
	SWEEP = "sweep"


class TransactionDirection(str, Enum):
	CREDIT = "credit"
	DEBIT = "debit"


class SigningAuthority(str, Enum):
	SINGLE = "single"
	JOINT_ANY = "joint_any"
	JOINT_ALL = "joint_all"


class FundLockStatus(str, Enum):
	ACTIVE = "active"
	RELEASED = "released"
	EXPIRED = "expired"


class StatementFormat(str, Enum):
	JSON = "json"
	PDF = "pdf"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class BankAccount(BaseModel):
	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	customer_id: str
	account_number: str
	iban: str | None = None
	currency: str
	account_type: AccountType
	product_code: str
	status: AccountStatus = AccountStatus.PENDING
	overdraft_limit: Decimal = Decimal("0")
	book_balance: Decimal = Decimal("0")
	available_balance: Decimal = Decimal("0")
	locked_balance: Decimal = Decimal("0")
	overdraft_used: Decimal = Decimal("0")
	opening_deposit: Decimal = Decimal("0")
	opened_at: datetime = Field(default_factory=datetime.utcnow)
	closed_at: datetime | None = None
	frozen_at: datetime | None = None
	dormant_since: datetime | None = None
	last_transaction_at: datetime | None = None
	close_reason: str | None = None
	closed_by: str | None = None
	freeze_reason: str | None = None
	frozen_by: str | None = None
	unfreeze_reason: str | None = None
	unfrozen_by: str | None = None
	overdraft_approved_by: str | None = None
	gl_account_code: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)

	@field_validator("currency")
	@classmethod
	def _upper_currency(cls, v: str) -> str:
		return v.upper()

	@field_validator("overdraft_limit", "book_balance", "available_balance",
	                 "locked_balance", "overdraft_used", "opening_deposit", mode="before")
	@classmethod
	def _to_decimal(cls, v: Any) -> Decimal:
		return Decimal(str(v))


class AccountTransaction(BaseModel):
	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	account_id: str
	account_number: str
	currency: str
	amount: Decimal
	direction: TransactionDirection
	transaction_type: TransactionType
	reference: str
	description: str
	balance_before: Decimal
	balance_after: Decimal
	gl_journal_id: str | None = None
	reversal_of: str | None = None
	reversed_by: str | None = None
	posted_at: datetime = Field(default_factory=datetime.utcnow)
	value_date: date = Field(default_factory=date.today)
	metadata: dict[str, Any] = Field(default_factory=dict)

	@field_validator("amount", "balance_before", "balance_after", mode="before")
	@classmethod
	def _to_decimal(cls, v: Any) -> Decimal:
		return Decimal(str(v))


class AccountBalance(BaseModel):
	model_config = _CFG

	account_id: str
	account_number: str
	currency: str
	book_balance: Decimal
	available_balance: Decimal
	locked_balance: Decimal
	overdraft_limit: Decimal
	overdraft_used: Decimal
	overdraft_available: Decimal
	as_of: datetime = Field(default_factory=datetime.utcnow)

	@field_validator("book_balance", "available_balance", "locked_balance",
	                 "overdraft_limit", "overdraft_used", "overdraft_available", mode="before")
	@classmethod
	def _to_decimal(cls, v: Any) -> Decimal:
		return Decimal(str(v))


class FundLock(BaseModel):
	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	account_id: str
	amount: Decimal
	lock_reference: str
	reason: str | None = None
	status: FundLockStatus = FundLockStatus.ACTIVE
	locked_at: datetime = Field(default_factory=datetime.utcnow)
	released_at: datetime | None = None
	expires_at: datetime | None = None

	@field_validator("amount", mode="before")
	@classmethod
	def _to_decimal(cls, v: Any) -> Decimal:
		return Decimal(str(v))


class StatementEntry(BaseModel):
	model_config = _CFG

	transaction_id: str
	value_date: date
	posted_at: datetime
	description: str
	reference: str
	transaction_type: str
	debit: Decimal | None = None
	credit: Decimal | None = None
	running_balance: Decimal
	currency: str


class AccountSignatory(BaseModel):
	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	account_id: str
	customer_id: str
	signing_authority: SigningAuthority
	added_at: datetime = Field(default_factory=datetime.utcnow)
	added_by: str | None = None
	is_active: bool = True


class AccountHistoryEntry(BaseModel):
	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	account_id: str
	tenant_id: str
	event_type: str
	old_status: str | None = None
	new_status: str | None = None
	description: str
	performed_by: str | None = None
	occurred_at: datetime = Field(default_factory=datetime.utcnow)
	metadata: dict[str, Any] = Field(default_factory=dict)


class AccountProduct(BaseModel):
	model_config = _CFG

	product_code: str
	product_name: str
	account_type: AccountType
	currency: str
	min_balance: Decimal = Decimal("0")
	max_balance: Decimal | None = None
	interest_rate: Decimal = Decimal("0")
	overdraft_allowed: bool = False
	max_overdraft: Decimal = Decimal("0")
	monthly_fee: Decimal = Decimal("0")
	description: str = ""


class TransactionSummary(BaseModel):
	model_config = _CFG

	account_id: str
	period: str
	total_credits: Decimal
	total_debits: Decimal
	net_movement: Decimal
	transaction_count: int
	opening_balance: Decimal
	closing_balance: Decimal
	currency: str


class AccountStats(BaseModel):
	model_config = _CFG

	customer_id: str
	tenant_id: str
	total_accounts: int
	active_accounts: int
	frozen_accounts: int
	dormant_accounts: int
	closed_accounts: int
	total_book_balance: Decimal
	total_available_balance: Decimal
	currencies: list[str]
	as_of: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Request / command models  (placed in views.py per CLAUDE.md — duplicated
# here as pure data classes for service layer use)
# ---------------------------------------------------------------------------

class OpenAccountRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	customer_id: str
	product_code: str
	currency: str
	account_number: str | None = None
	opening_deposit: Decimal | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)

	@field_validator("opening_deposit", mode="before")
	@classmethod
	def _to_decimal(cls, v: Any) -> Decimal | None:
		if v is None:
			return None
		return Decimal(str(v))


class CloseAccountRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	account_id: str
	reason: str
	closed_by: str


class FreezeAccountRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	account_id: str
	reason: str
	frozen_by: str


class UnfreezeAccountRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	account_id: str
	reason: str
	unfrozen_by: str


class CreditRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	account_id: str
	amount: Decimal
	currency: str
	reference: str
	description: str
	transaction_type: TransactionType = TransactionType.DEPOSIT
	value_date: date | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)

	@field_validator("amount", mode="before")
	@classmethod
	def _to_decimal(cls, v: Any) -> Decimal:
		return Decimal(str(v))


class DebitRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	account_id: str
	amount: Decimal
	currency: str
	reference: str
	description: str
	transaction_type: TransactionType = TransactionType.WITHDRAWAL
	value_date: date | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)

	@field_validator("amount", mode="before")
	@classmethod
	def _to_decimal(cls, v: Any) -> Decimal:
		return Decimal(str(v))


class TransferRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	from_account_id: str
	to_account_id: str
	amount: Decimal
	reference: str
	description: str
	value_date: date | None = None

	@field_validator("amount", mode="before")
	@classmethod
	def _to_decimal(cls, v: Any) -> Decimal:
		return Decimal(str(v))


class BulkCreditItem(BaseModel):
	model_config = _CFG

	account_id: str
	amount: Decimal
	reference: str
	description: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)

	@field_validator("amount", mode="before")
	@classmethod
	def _to_decimal(cls, v: Any) -> Decimal:
		return Decimal(str(v))


class BulkCreditResult(BaseModel):
	model_config = _CFG

	succeeded: list[AccountTransaction] = Field(default_factory=list)
	failed: list[dict[str, Any]] = Field(default_factory=list)
	total: int
	success_count: int
	failure_count: int
