"""General Ledger — Pydantic v2 models.

All monetary values use Decimal with 4dp precision.
Immutability invariant: posted journal entries never mutate.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


# ── Shared config ────────────────────────────────────────────────────────────

_CFG = ConfigDict(
	extra="forbid",
	validate_by_name=True,
	validate_by_alias=True,
	populate_by_name=True,
)

FOUR_DP = Decimal("0.0001")


# ── Domain errors ────────────────────────────────────────────────────────────

class GLImbalanceError(ValueError):
	"""Raised when sum(debits) != sum(credits) in a journal entry."""

	def __init__(self, total_debits: Decimal, total_credits: Decimal) -> None:
		diff = total_debits - total_credits
		super().__init__(
			f"Journal imbalanced: debits={total_debits} credits={total_credits} diff={diff}"
		)
		self.total_debits = total_debits
		self.total_credits = total_credits


class PostingToClosedPeriodError(ValueError):
	"""Raised when attempting to post to a non-open accounting period."""

	def __init__(self, period_id: str, status: str) -> None:
		super().__init__(f"Period {period_id!r} is {status!r}; only OPEN periods accept postings")
		self.period_id = period_id
		self.status = status


class AccountNotFoundError(KeyError):
	def __init__(self, code: str, tenant_id: str) -> None:
		super().__init__(f"Account {code!r} not found for tenant {tenant_id!r}")


class DuplicateAccountError(ValueError):
	def __init__(self, code: str, tenant_id: str) -> None:
		super().__init__(f"Account {code!r} already exists for tenant {tenant_id!r}")


class JournalNotFoundError(KeyError):
	def __init__(self, journal_id: str) -> None:
		super().__init__(f"Journal entry {journal_id!r} not found")


# ── Enumerations ─────────────────────────────────────────────────────────────

class GLAccountType(str, Enum):
	ASSET = "ASSET"
	LIABILITY = "LIABILITY"
	EQUITY = "EQUITY"
	INCOME = "INCOME"
	EXPENSE = "EXPENSE"


class NormalBalance(str, Enum):
	DEBIT = "DEBIT"
	CREDIT = "CREDIT"


# Assets and expenses increase with debits; liabilities, equity, income with credits.
_NORMAL_BALANCE: dict[GLAccountType, NormalBalance] = {
	GLAccountType.ASSET: NormalBalance.DEBIT,
	GLAccountType.LIABILITY: NormalBalance.CREDIT,
	GLAccountType.EQUITY: NormalBalance.CREDIT,
	GLAccountType.INCOME: NormalBalance.CREDIT,
	GLAccountType.EXPENSE: NormalBalance.DEBIT,
}


def normal_balance_for(account_type: GLAccountType) -> NormalBalance:
	return _NORMAL_BALANCE[account_type]


class PeriodStatus(str, Enum):
	OPEN = "OPEN"
	CLOSED = "CLOSED"
	LOCKED = "LOCKED"


class JournalStatus(str, Enum):
	DRAFT = "DRAFT"
	POSTED = "POSTED"
	REVERSED = "REVERSED"
	PENDING_APPROVAL = "PENDING_APPROVAL"
	APPROVED = "APPROVED"
	REJECTED = "REJECTED"


class JournalType(str, Enum):
	STANDARD = "STANDARD"
	ADJUSTMENT = "ADJUSTMENT"
	REVERSAL = "REVERSAL"
	ACCRUAL = "ACCRUAL"
	FX_REVALUATION = "FX_REVALUATION"
	OPENING_BALANCE = "OPENING_BALANCE"
	INTERCOMPANY = "INTERCOMPANY"
	COST_ALLOCATION = "COST_ALLOCATION"


# ── Core models ───────────────────────────────────────────────────────────────

class GLAccount(BaseModel):
	"""A single account in the chart of accounts."""

	model_config = _CFG

	id: str = Field(default_factory=uuid7str, description="Surrogate UUID7 key")
	tenant_id: str = Field(..., description="Owning tenant")
	code: str = Field(..., description="Account code, e.g. '1001'", min_length=1, max_length=32)
	name: str = Field(..., description="Human-readable account name", min_length=1, max_length=255)
	account_type: GLAccountType = Field(..., description="One of ASSET/LIABILITY/EQUITY/INCOME/EXPENSE")
	normal_balance: NormalBalance = Field(..., description="DEBIT or CREDIT per accounting convention")
	currency: str = Field(default="USD", description="ISO 4217 currency code", min_length=3, max_length=3)
	balance: Decimal = Field(
		default=Decimal("0.0000"),
		description="Current running balance in account's normal balance direction (4dp)",
	)
	is_active: bool = Field(default=True, description="Inactive accounts reject new postings")
	parent_code: str | None = Field(default=None, description="Parent account code for hierarchy")
	level: int = Field(default=1, description="Depth in account hierarchy; root = 1", ge=1)
	description: str | None = Field(default=None, description="Optional narrative")
	cost_centre: str | None = Field(default=None, description="Cost centre tag")
	is_control_account: bool = Field(default=False, description="True = sub-ledger required")
	is_suspense: bool = Field(default=False, description="True = suspense/clearing account")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)

	@field_validator("currency")
	@classmethod
	def upper_currency(cls, v: str) -> str:
		return v.upper()

	@field_validator("code")
	@classmethod
	def strip_code(cls, v: str) -> str:
		return v.strip()


class JournalEntryLine(BaseModel):
	"""A single debit or credit line within a journal entry."""

	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	account_code: str = Field(..., description="GL account code", min_length=1, max_length=32)
	debit_amount: Decimal = Field(
		default=Decimal("0.0000"),
		description="Debit amount (4dp); exactly one of debit/credit must be non-zero",
		ge=Decimal("0"),
	)
	credit_amount: Decimal = Field(
		default=Decimal("0.0000"),
		description="Credit amount (4dp); exactly one of debit/credit must be non-zero",
		ge=Decimal("0"),
	)
	currency: str = Field(default="USD", min_length=3, max_length=3)
	narrative: str | None = Field(default=None, max_length=500)
	entity_id: str | None = Field(default=None, description="Sub-ledger entity (customer/vendor/employee)")
	cost_centre: str | None = Field(default=None)

	@model_validator(mode="after")
	def exactly_one_side(self) -> JournalEntryLine:
		dr = self.debit_amount or Decimal("0")
		cr = self.credit_amount or Decimal("0")
		if dr == 0 and cr == 0:
			raise ValueError("Each line must have a non-zero debit_amount OR credit_amount")
		if dr > 0 and cr > 0:
			raise ValueError("Each line must have debit_amount OR credit_amount, not both")
		return self

	@field_validator("currency")
	@classmethod
	def upper_currency(cls, v: str) -> str:
		return v.upper()


class JournalEntry(BaseModel):
	"""An immutable, balanced double-entry journal record."""

	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(...)
	description: str = Field(..., min_length=1, max_length=500)
	reference: str | None = Field(default=None, max_length=128)
	posting_date: date = Field(...)
	period_id: str = Field(...)
	journal_type: JournalType = Field(default=JournalType.STANDARD)
	status: JournalStatus = Field(default=JournalStatus.POSTED)
	lines: list[JournalEntryLine] = Field(default_factory=list, min_length=2)
	entry_hash: str = Field(default="", description="SHA-256 of canonical line data for tamper-evidence")
	created_by: str = Field(default="system")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	reverses_id: str | None = Field(default=None, description="ID of the journal this entry reverses")
	reversed_by_id: str | None = Field(default=None, description="ID of the reversal journal")
	batch_id: str | None = Field(default=None, description="Batch posting reference")
	source: str | None = Field(default=None, description="Originating module: AP, AR, PAYROLL, etc.")
	approved_by: str | None = Field(default=None)
	approved_at: datetime | None = Field(default=None)

	@model_validator(mode="after")
	def compute_hash(self) -> JournalEntry:
		if not self.entry_hash:
			self.entry_hash = _compute_entry_hash(self)
		return self


class AccountingPeriod(BaseModel):
	"""Fiscal accounting period — controls which dates accept journal postings."""

	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(...)
	year: int = Field(..., ge=1900, le=2200)
	month: int = Field(..., ge=1, le=12)
	status: PeriodStatus = Field(default=PeriodStatus.OPEN)
	open_date: datetime | None = Field(default=None, description="When the period was opened")
	close_date: datetime | None = Field(default=None, description="When the period was closed")
	locked_date: datetime | None = Field(default=None, description="When the period was locked")
	closed_by: str | None = Field(default=None)
	created_at: datetime = Field(default_factory=datetime.utcnow)

	@property
	def label(self) -> str:
		return f"{self.year:04d}-{self.month:02d}"

	def is_open(self) -> bool:
		return self.status == PeriodStatus.OPEN


class TrialBalanceRow(BaseModel):
	"""One row in a trial balance report."""

	model_config = _CFG

	account_code: str
	account_name: str
	account_type: GLAccountType
	debit_balance: Decimal = Field(
		default=Decimal("0.0000"),
		description="Accumulated debit balance (4dp)",
	)
	credit_balance: Decimal = Field(
		default=Decimal("0.0000"),
		description="Accumulated credit balance (4dp)",
	)
	currency: str = Field(default="USD")


class BalanceSheetItem(BaseModel):
	"""One line in a balance sheet."""

	model_config = _CFG

	account_code: str
	account_name: str
	balance: Decimal = Field(description="Signed balance — positive = normal direction (4dp)")
	account_type: GLAccountType
	currency: str = Field(default="USD")
	level: int = Field(default=1)


class PnLRow(BaseModel):
	"""One line in the profit & loss statement."""

	model_config = _CFG

	account_code: str
	account_name: str
	amount: Decimal = Field(description="Income positive, expense negative (4dp)")
	account_type: GLAccountType
	currency: str = Field(default="USD")


class SubLedgerEntry(BaseModel):
	"""Entity-level sub-ledger movement row."""

	model_config = _CFG

	journal_id: str
	posting_date: date
	account_code: str
	entity_id: str
	debit_amount: Decimal = Field(default=Decimal("0.0000"))
	credit_amount: Decimal = Field(default=Decimal("0.0000"))
	narrative: str | None = None
	reference: str | None = None
	running_balance: Decimal = Field(default=Decimal("0.0000"))


class AccountMovements(BaseModel):
	"""Debit and credit movement totals for an account within a period."""

	model_config = _CFG

	account_code: str
	account_name: str
	period_id: str
	opening_balance: Decimal = Field(default=Decimal("0.0000"))
	total_debits: Decimal = Field(default=Decimal("0.0000"))
	total_credits: Decimal = Field(default=Decimal("0.0000"))
	closing_balance: Decimal = Field(default=Decimal("0.0000"))
	currency: str = Field(default="USD")


class BatchEntryRequest(BaseModel):
	"""Request payload for idempotent batch journal posting."""

	model_config = _CFG

	batch_id: str = Field(..., description="Client-assigned idempotency key")
	entries: list[dict[str, Any]] = Field(..., min_length=1, description="List of journal entry dicts")


class FXRate(BaseModel):
	"""Foreign exchange rate for revaluation."""

	model_config = _CFG

	from_currency: str = Field(..., min_length=3, max_length=3)
	to_currency: str = Field(..., min_length=3, max_length=3)
	rate: Decimal = Field(..., description="1 from_currency = rate to_currency", gt=Decimal("0"))
	effective_date: date = Field(default_factory=date.today)


# ── Hash utility ─────────────────────────────────────────────────────────────

def _compute_entry_hash(entry: JournalEntry) -> str:
	"""Deterministic SHA-256 over the immutable fields of a journal entry."""
	payload = {
		"id": entry.id,
		"tenant_id": entry.tenant_id,
		"posting_date": str(entry.posting_date),
		"period_id": entry.period_id,
		"lines": [
			{
				"account_code": ln.account_code,
				"debit_amount": str(ln.debit_amount),
				"credit_amount": str(ln.credit_amount),
				"currency": ln.currency,
			}
			for ln in entry.lines
		],
	}
	raw = json.dumps(payload, sort_keys=True).encode()
	return hashlib.sha256(raw).hexdigest()
