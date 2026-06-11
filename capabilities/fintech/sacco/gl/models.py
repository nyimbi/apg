"""Pydantic v2 models for SACCO General Ledger."""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


# ── Chart of Accounts ─────────────────────────────────────────────────────────

class AccountCategory(str, Enum):
	ASSET = "asset"
	LIABILITY = "liability"
	EQUITY = "equity"
	INCOME = "income"
	EXPENSE = "expense"


class NormalBalance(str, Enum):
	DEBIT = "debit"
	CREDIT = "credit"


class SACCOAccountCode(str, Enum):
	# ASSETS
	CASH                    = "1001"
	BANK                    = "1010"
	LOANS_FOSA              = "1100"
	LOANS_BOSA              = "1110"
	LOANS_NON_MEMBER        = "1120"
	PROVISION_LOANS         = "1125"  # contra-asset
	INVESTMENTS             = "1200"
	FIXED_ASSETS            = "1300"
	ACCUMULATED_DEPREC      = "1305"  # contra-asset
	OTHER_ASSETS            = "1400"

	# LIABILITIES
	DEPOSITS_FOSA           = "2100"
	DEPOSITS_BOSA           = "2110"
	EXTERNAL_BORROWINGS     = "2200"
	DIVIDENDS_PAYABLE       = "2300"
	OTHER_LIABILITIES       = "2400"

	# EQUITY
	INSTITUTIONAL_CAPITAL   = "3100"
	SHARE_CAPITAL           = "3200"
	RETAINED_SURPLUS        = "3300"
	RESERVES                = "3400"

	# INCOME
	INTEREST_INCOME_LOANS   = "4100"
	INTEREST_INCOME_INVEST  = "4200"
	FEE_INCOME              = "4300"
	PENALTY_INCOME          = "4350"
	OTHER_INCOME            = "4400"

	# EXPENSES
	INTEREST_EXPENSE        = "5100"
	LOAN_LOSS_PROVISIONS    = "5200"
	STAFF_COSTS             = "5300"
	ADMIN_EXPENSES          = "5400"
	DEPRECIATION            = "5500"
	OTHER_EXPENSES          = "5600"


# Canonical chart of accounts definition
STANDARD_COA: list[dict[str, Any]] = [
	# ASSETS
	{"code": "1001", "name": "Cash",                          "category": "asset",     "normal_balance": "debit",  "description": "Petty cash and vault"},
	{"code": "1010", "name": "Bank",                          "category": "asset",     "normal_balance": "debit",  "description": "Current and savings bank accounts"},
	{"code": "1100", "name": "Member Loans - FOSA",           "category": "asset",     "normal_balance": "debit",  "description": "Front Office Service Activity loans"},
	{"code": "1110", "name": "Member Loans - BOSA",           "category": "asset",     "normal_balance": "debit",  "description": "Back Office Service Activity loans"},
	{"code": "1120", "name": "Non-Member Loans",              "category": "asset",     "normal_balance": "debit",  "description": "Loans to non-members"},
	{"code": "1125", "name": "Provision for Loan Losses",     "category": "asset",     "normal_balance": "credit", "description": "Contra-asset: cumulative loan loss provisions"},
	{"code": "1200", "name": "Investment Securities",         "category": "asset",     "normal_balance": "debit",  "description": "Government bonds, T-bills, fixed deposits"},
	{"code": "1300", "name": "Fixed Assets",                  "category": "asset",     "normal_balance": "debit",  "description": "Property, plant and equipment at cost"},
	{"code": "1305", "name": "Accumulated Depreciation",      "category": "asset",     "normal_balance": "credit", "description": "Contra-asset: accumulated depreciation"},
	{"code": "1400", "name": "Other Assets",                  "category": "asset",     "normal_balance": "debit",  "description": "Prepayments, receivables, other assets"},
	# LIABILITIES
	{"code": "2100", "name": "Member Deposits - FOSA",        "category": "liability", "normal_balance": "credit", "description": "FOSA savings and current account deposits"},
	{"code": "2110", "name": "Member Deposits - BOSA",        "category": "liability", "normal_balance": "credit", "description": "BOSA share deposits and savings"},
	{"code": "2200", "name": "External Borrowings",           "category": "liability", "normal_balance": "credit", "description": "Bank loans, bonds payable"},
	{"code": "2300", "name": "Dividends Payable",             "category": "liability", "normal_balance": "credit", "description": "Declared but unpaid dividends"},
	{"code": "2400", "name": "Other Liabilities",             "category": "liability", "normal_balance": "credit", "description": "Accruals, deferred income, other payables"},
	# EQUITY
	{"code": "3100", "name": "Institutional Capital",         "category": "equity",    "normal_balance": "credit", "description": "Undistributable institutional capital (SASRA requirement)"},
	{"code": "3200", "name": "Share Capital",                 "category": "equity",    "normal_balance": "credit", "description": "Member share contributions"},
	{"code": "3300", "name": "Retained Surplus",              "category": "equity",    "normal_balance": "credit", "description": "Accumulated undistributed surplus"},
	{"code": "3400", "name": "Reserves",                      "category": "equity",    "normal_balance": "credit", "description": "Statutory and contingency reserves"},
	# INCOME
	{"code": "4100", "name": "Interest Income - Loans",       "category": "income",    "normal_balance": "credit", "description": "Interest earned on member and non-member loans"},
	{"code": "4200", "name": "Interest Income - Investments", "category": "income",    "normal_balance": "credit", "description": "Interest on bonds, T-bills, fixed deposits"},
	{"code": "4300", "name": "Fee Income",                    "category": "income",    "normal_balance": "credit", "description": "Processing fees, service charges"},
	{"code": "4350", "name": "Penalty Income",                "category": "income",    "normal_balance": "credit", "description": "Late payment penalties"},
	{"code": "4400", "name": "Other Income",                  "category": "income",    "normal_balance": "credit", "description": "Miscellaneous income"},
	# EXPENSES
	{"code": "5100", "name": "Interest Expense",              "category": "expense",   "normal_balance": "debit",  "description": "Interest paid on deposits and borrowings"},
	{"code": "5200", "name": "Loan Loss Provisions",          "category": "expense",   "normal_balance": "debit",  "description": "Charge for expected credit losses"},
	{"code": "5300", "name": "Staff Costs",                   "category": "expense",   "normal_balance": "debit",  "description": "Salaries, benefits, NSSF, NHIF"},
	{"code": "5400", "name": "Admin Expenses",                "category": "expense",   "normal_balance": "debit",  "description": "Rent, utilities, office expenses"},
	{"code": "5500", "name": "Depreciation",                  "category": "expense",   "normal_balance": "debit",  "description": "Depreciation charge for the period"},
	{"code": "5600", "name": "Other Expenses",                "category": "expense",   "normal_balance": "debit",  "description": "Miscellaneous expenses"},
]


# ── Core GL Models ─────────────────────────────────────────────────────────────

class GLAccount(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	code: str
	name: str
	category: AccountCategory
	normal_balance: NormalBalance
	description: str = ""
	is_active: bool = True
	balance: Decimal = Decimal("0")
	created_at: str = ""
	updated_at: str = ""


class JournalLine(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_code: str
	debit: Decimal = Decimal("0")
	credit: Decimal = Decimal("0")
	narrative: str = ""

	@field_validator("debit", "credit")
	@classmethod
	def non_negative(cls, v: Decimal) -> Decimal:
		if v < Decimal("0"):
			raise ValueError("debit/credit must be >= 0")
		return v


class JournalEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	reference: str
	transaction_type: str
	value_date: str
	posted_at: str
	posted_by: str
	narration: str = ""
	lines: list[JournalLine] = Field(default_factory=list)
	is_reversed: bool = False
	reversal_of: str | None = None
	total_debit: Decimal = Decimal("0")
	total_credit: Decimal = Decimal("0")
	period_key: str = ""  # YYYY-MM


class AccountingPeriod(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	year: int
	month: int
	status: str = "open"  # open | closed
	opened_at: str = ""
	closed_at: str | None = None
	closed_by: str | None = None


# ── Reporting Models ───────────────────────────────────────────────────────────

class TrialBalanceRow(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	code: str
	name: str
	category: str
	debit: Decimal = Decimal("0")
	credit: Decimal = Decimal("0")
	net: Decimal = Decimal("0")


class BalanceSheet(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	as_of_date: str
	tenant_id: str
	assets: dict[str, Decimal] = Field(default_factory=dict)
	liabilities: dict[str, Decimal] = Field(default_factory=dict)
	equity: dict[str, Decimal] = Field(default_factory=dict)
	total_assets: Decimal = Decimal("0")
	total_liabilities: Decimal = Decimal("0")
	total_equity: Decimal = Decimal("0")
	total_liabilities_equity: Decimal = Decimal("0")
	is_balanced: bool = True


class IncomeStatement(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	from_date: str
	to_date: str
	tenant_id: str
	income: dict[str, Decimal] = Field(default_factory=dict)
	expenses: dict[str, Decimal] = Field(default_factory=dict)
	total_income: Decimal = Decimal("0")
	total_expenses: Decimal = Decimal("0")
	surplus_deficit: Decimal = Decimal("0")


class ReconciliationResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id: str
	as_of_date: str
	reconciled: bool = True
	items: list[dict[str, Any]] = Field(default_factory=list)
	differences: list[dict[str, Any]] = Field(default_factory=list)
	gl_total_deposits: Decimal = Decimal("0")
	subsidiary_total_deposits: Decimal = Decimal("0")
	gl_total_loans: Decimal = Decimal("0")
	subsidiary_total_loans: Decimal = Decimal("0")


class GLSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id: str
	period: str
	total_assets: Decimal = Decimal("0")
	loan_book_gross: Decimal = Decimal("0")
	loan_book_net: Decimal = Decimal("0")
	deposit_base: Decimal = Decimal("0")
	share_capital: Decimal = Decimal("0")
	total_equity: Decimal = Decimal("0")
	capital_ratio_pct: Decimal = Decimal("0")   # equity / total_assets × 100
	npa_ratio_pct: Decimal = Decimal("0")       # provision / gross loans × 100
	total_income: Decimal = Decimal("0")
	total_expenses: Decimal = Decimal("0")
	surplus_deficit: Decimal = Decimal("0")
	journal_entry_count: int = 0
