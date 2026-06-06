"""
General Ledger — Pydantic v2 models.

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


# ---------------------------------------------------------------------------
# Configuration shared by all models
# ---------------------------------------------------------------------------
_CFG = ConfigDict(
	extra="forbid",
	validate_by_name=True,
	validate_by_alias=True,
	populate_by_name=True,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AccountType(str, Enum):
	ASSET = "asset"
	LIABILITY = "liability"
	EQUITY = "equity"
	REVENUE = "revenue"
	EXPENSE = "expense"
	CONTRA = "contra"


class NormalBalance(str, Enum):
	DEBIT = "debit"
	CREDIT = "credit"


class PeriodStatus(str, Enum):
	FUTURE = "future"
	OPEN = "open"
	SOFT_CLOSED = "soft_closed"
	CLOSED = "closed"
	LOCKED = "locked"


class JournalType(str, Enum):
	STANDARD = "standard"
	ADJUSTMENT = "adjustment"
	RECURRING = "recurring"
	REVERSAL = "reversal"
	INTERCOMPANY = "intercompany"
	ACCRUAL = "accrual"
	IMPORT = "import"
	MANUAL = "manual"


class JournalStatus(str, Enum):
	DRAFT = "draft"
	BALANCED = "balanced"
	PENDING_APPROVAL = "pending_approval"
	APPROVED = "approved"
	POSTED = "posted"
	REVERSED = "reversed"
	CANCELLED = "cancelled"


class ApprovalStatus(str, Enum):
	PENDING = "pending"
	AUTO_APPROVED = "auto_approved"
	APPROVED = "approved"
	REJECTED = "rejected"


class ReconciliationStatus(str, Enum):
	OPEN = "open"
	SUBMITTED = "submitted"
	APPROVED = "approved"
	REJECTED = "rejected"


class BudgetType(str, Enum):
	ORIGINAL = "original"
	REVISED = "revised"
	FORECAST = "forecast"


class ClosingType(str, Enum):
	MONTH_END = "month_end"
	QUARTER_END = "quarter_end"
	YEAR_END = "year_end"


# ---------------------------------------------------------------------------
# Base model — every entity inherits this
# ---------------------------------------------------------------------------

class GLBase(BaseModel):
	"""Common fields on every GL entity."""

	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"
	is_deleted: bool = False


# ---------------------------------------------------------------------------
# Chart of Accounts
# ---------------------------------------------------------------------------

class GLAccountCreate(BaseModel):
	"""Request model for creating a GL account."""

	model_config = _CFG

	tenant_id: str
	account_code: str = Field(min_length=1, max_length=20)
	account_name: str = Field(min_length=1, max_length=200)
	account_type: AccountType
	normal_balance: NormalBalance | None = None  # derived from type if None
	currency: str = Field(default="USD", min_length=3, max_length=3)
	allow_posting: bool = True
	parent_account_code: str | None = None
	description: str | None = None
	ifrs_mapping: str | None = None
	gaap_mapping: str | None = None
	tax_code: str | None = None
	cost_center_required: bool = False
	project_required: bool = False
	is_reconciliation_account: bool = False
	tags: list[str] = Field(default_factory=list)
	created_by: str = "system"

	@field_validator("account_code", mode="before")
	@classmethod
	def _normalise_code(cls, v: str) -> str:
		v = str(v).strip().upper()
		if not v.replace("-", "").replace("_", "").replace(".", "").isalnum():
			raise ValueError("account_code must be alphanumeric (hyphens, underscores, dots allowed)")
		return v

	@model_validator(mode="after")
	def _default_normal_balance(self) -> "GLAccountCreate":
		if self.normal_balance is None:
			self.normal_balance = (
				NormalBalance.DEBIT
				if self.account_type in (AccountType.ASSET, AccountType.EXPENSE)
				else NormalBalance.CREDIT
			)
		return self


class GLAccountUpdate(BaseModel):
	"""Partial update for a GL account."""

	model_config = _CFG

	account_name: str | None = None
	description: str | None = None
	allow_posting: bool | None = None
	is_reconciliation_account: bool | None = None
	cost_center_required: bool | None = None
	project_required: bool | None = None
	ifrs_mapping: str | None = None
	gaap_mapping: str | None = None
	tax_code: str | None = None
	tags: list[str] | None = None
	updated_by: str = "system"


class GLAccountResponse(GLBase):
	"""Full account response including computed fields."""

	account_code: str
	account_name: str
	account_type: AccountType
	normal_balance: NormalBalance
	currency: str = "USD"
	allow_posting: bool = True
	parent_account_id: str | None = None
	hierarchy_level: int = 0
	description: str | None = None
	ifrs_mapping: str | None = None
	gaap_mapping: str | None = None
	tax_code: str | None = None
	cost_center_required: bool = False
	project_required: bool = False
	is_reconciliation_account: bool = False
	tags: list[str] = Field(default_factory=list)
	status: str = "active"


# ---------------------------------------------------------------------------
# Accounting Period
# ---------------------------------------------------------------------------

class GLPeriodCreate(BaseModel):
	"""Create an accounting period."""

	model_config = _CFG

	tenant_id: str
	period_code: str = Field(min_length=4, max_length=20)
	fiscal_year: int = Field(ge=1900, le=2200)
	period_number: int = Field(ge=1, le=13)
	start_date: date
	end_date: date
	allows_adjustments: bool = False
	created_by: str = "system"

	@model_validator(mode="after")
	def _date_order(self) -> "GLPeriodCreate":
		if self.start_date > self.end_date:
			raise ValueError("start_date must be <= end_date")
		return self


class GLPeriodUpdate(BaseModel):
	model_config = _CFG

	status: PeriodStatus | None = None
	allows_adjustments: bool | None = None
	updated_by: str = "system"


class GLPeriodResponse(GLBase):
	period_code: str
	fiscal_year: int
	period_number: int
	start_date: date
	end_date: date
	status: PeriodStatus = PeriodStatus.FUTURE
	allows_adjustments: bool = False
	opened_by: str | None = None
	opened_at: datetime | None = None
	closed_by: str | None = None
	closed_at: datetime | None = None
	locked_by: str | None = None
	locked_at: datetime | None = None


# ---------------------------------------------------------------------------
# Journal Entry
# ---------------------------------------------------------------------------

class GLJournalLineCreate(BaseModel):
	"""Single line in a journal entry."""

	model_config = _CFG

	account_id: str
	debit: Decimal = Decimal("0")
	credit: Decimal = Decimal("0")
	description: str | None = None
	currency: str = "USD"
	exchange_rate: Decimal = Decimal("1")
	cost_center: str | None = None
	project: str | None = None
	entity: str | None = None
	tax_code: str | None = None
	tax_amount: Decimal = Decimal("0")
	segment: str | None = None

	@model_validator(mode="after")
	def _one_side_nonzero(self) -> "GLJournalLineCreate":
		if self.debit < 0 or self.credit < 0:
			raise ValueError("debit and credit must be non-negative")
		if self.debit > 0 and self.credit > 0:
			raise ValueError("a line cannot have both debit and credit")
		return self


class GLJournalLineResponse(BaseModel):
	model_config = _CFG

	line_number: int
	account_id: str
	account_code: str | None = None
	debit: Decimal = Decimal("0")
	credit: Decimal = Decimal("0")
	functional_debit: Decimal = Decimal("0")
	functional_credit: Decimal = Decimal("0")
	description: str | None = None
	currency: str = "USD"
	exchange_rate: Decimal = Decimal("1")
	cost_center: str | None = None
	project: str | None = None
	entity: str | None = None
	tax_code: str | None = None
	tax_amount: Decimal = Decimal("0")


class GLJournalEntryCreate(BaseModel):
	"""Create request for a journal entry."""

	model_config = _CFG

	tenant_id: str
	journal_date: date
	journal_type: JournalType = JournalType.STANDARD
	description: str = Field(min_length=1, max_length=500)
	reference: str | None = None
	lines: list[GLJournalLineCreate] = Field(min_length=2)
	posted_by: str = "system"
	attachments: list[str] = Field(default_factory=list)

	@model_validator(mode="after")
	def _balanced(self) -> "GLJournalEntryCreate":
		total_d = sum(ln.debit for ln in self.lines)
		total_c = sum(ln.credit for ln in self.lines)
		if total_d != total_c:
			raise ValueError(f"journal_not_balanced: debits={total_d} credits={total_c}")
		if total_d == 0:
			raise ValueError("journal_total_must_be_positive")
		return self


class GLJournalEntryUpdate(BaseModel):
	model_config = _CFG

	description: str | None = None
	reference: str | None = None
	attachments: list[str] | None = None
	updated_by: str = "system"


class GLJournalEntryResponse(GLBase):
	journal_number: str
	journal_date: date
	period_code: str | None = None
	journal_type: JournalType
	description: str
	reference: str | None = None
	status: JournalStatus = JournalStatus.DRAFT
	total_debit: Decimal = Decimal("0")
	total_credit: Decimal = Decimal("0")
	lines: list[GLJournalLineResponse] = Field(default_factory=list)
	attachments: list[str] = Field(default_factory=list)
	approval_status: ApprovalStatus | None = None
	posted_by: str | None = None
	posted_at: datetime | None = None
	reversed_at: datetime | None = None
	reversal_journal_id: str | None = None


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

class GLBudgetCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	budget_code: str
	fiscal_year: int
	budget_type: BudgetType = BudgetType.ORIGINAL
	account_code: str
	period_code: str
	amount: Decimal
	currency: str = "USD"
	created_by: str = "system"


class GLBudgetUpdate(BaseModel):
	model_config = _CFG

	amount: Decimal | None = None
	budget_type: BudgetType | None = None
	updated_by: str = "system"


class GLBudgetResponse(GLBase):
	budget_code: str
	fiscal_year: int
	budget_type: BudgetType
	account_code: str
	period_code: str
	amount: Decimal
	currency: str = "USD"


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

class GLReconciliationItem(BaseModel):
	model_config = _CFG

	description: str
	amount: Decimal
	item_type: str  # timing_difference | error | outstanding_cheque | deposit_in_transit


class GLReconciliationCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	account_code: str
	period_code: str
	created_by: str = "system"


class GLReconciliationSubmit(BaseModel):
	model_config = _CFG

	reconciled_by: str
	reconciling_items: list[GLReconciliationItem] = Field(default_factory=list)
	balance_per_statement: Decimal | None = None


class GLReconciliationResponse(GLBase):
	account_code: str
	period_code: str
	balance_per_gl: Decimal | None = None
	balance_per_statement: Decimal | None = None
	reconciling_items: list[GLReconciliationItem] = Field(default_factory=list)
	unreconciled_difference: Decimal | None = None
	status: ReconciliationStatus = ReconciliationStatus.OPEN
	reconciled_by: str | None = None
	reconciled_at: datetime | None = None
	approved_by: str | None = None
	approved_at: datetime | None = None


# ---------------------------------------------------------------------------
# Trial Balance row
# ---------------------------------------------------------------------------

class GLTrialBalanceRow(BaseModel):
	model_config = _CFG

	account_code: str
	account_name: str
	account_type: str
	opening_balance: Decimal
	period_debit: Decimal
	period_credit: Decimal
	closing_debit: Decimal
	closing_credit: Decimal


class GLTrialBalanceResponse(BaseModel):
	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	period_code: str
	rows: list[GLTrialBalanceRow]
	total_closing_debit: Decimal
	total_closing_credit: Decimal
	balanced: bool
	generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Closing Entry
# ---------------------------------------------------------------------------

class GLClosingEntryResponse(BaseModel):
	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	fiscal_year: int
	closing_type: ClosingType = ClosingType.YEAR_END
	retained_earnings_account: str
	net_to_retained_earnings: Decimal
	closing_journal_id: str | None = None
	status: str = "completed"
	closed_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Currency Rate
# ---------------------------------------------------------------------------

class GLCurrencyRateCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	from_currency: str = Field(min_length=3, max_length=3)
	to_currency: str = Field(min_length=3, max_length=3)
	rate_type: str = "spot"
	effective_date: date
	exchange_rate: Decimal = Field(gt=0)
	rate_source: str = "manual"
	created_by: str = "system"


class GLCurrencyRateResponse(GLBase):
	from_currency: str
	to_currency: str
	rate_type: str
	effective_date: date
	exchange_rate: Decimal
	rate_source: str


# ---------------------------------------------------------------------------
# Report request helpers
# ---------------------------------------------------------------------------

class GLReportRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	period_code: str
	comparative_period: str | None = None
	segment: str | None = None
	include_zero_balances: bool = False


class GLBudgetVsActualRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	period_code: str
	budget_version: str = "approved"


class GLYearEndRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	fiscal_year: int
	retained_earnings_account: str
	executed_by: str = "system"


class GLPriorYearAdjRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	account_code: str
	amount: Decimal
	adjustment_reason: str = Field(min_length=5)
	executed_by: str = "system"


class GLConsolidationRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	subsidiaries: list[str]
	group_adjustments: list[dict[str, Any]] = Field(default_factory=list)
	minority_interest: dict[str, Any] = Field(default_factory=dict)


class GLIntercompanyRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	counterpart_entity: str
	amount: Decimal = Field(gt=0)
	currency: str = "USD"
	account_mapping: dict[str, str]


# ---------------------------------------------------------------------------
# Pagination / list wrappers
# ---------------------------------------------------------------------------

class GLListResponse(BaseModel):
	model_config = _CFG

	items: list[Any]
	total: int
	page: int = 1
	page_size: int = 50


# ---------------------------------------------------------------------------
# CF-prefixed canonical aliases (capability-framework naming convention)
# ---------------------------------------------------------------------------

class CFGLJournalLine(GLJournalLineCreate):
	"""Canonical alias for a GL journal line used in the capability framework."""
	pass


class CFGLJournalEntry(GLJournalEntryCreate):
	"""Canonical alias for a GL journal entry used in the capability framework."""
	pass


class CFGLPosting(GLJournalEntryResponse):
	"""Canonical alias for a posted GL journal entry used in the capability framework."""
	pass


# Short aliases that tests and integration code reference by the unprefixed name.
GLJournalEntry = CFGLJournalEntry
GLJournalLine = CFGLJournalLine
GLPosting = CFGLPosting
