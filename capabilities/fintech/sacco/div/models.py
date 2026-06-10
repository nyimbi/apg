"""Pydantic v2 models for SACCO Dividend & Distribution."""
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


# ── Financial Year ────────────────────────────────────────────────────────────

class FinancialYearCreateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	year_code: str  # e.g. "FY2025"
	start_date: str
	end_date: str
	description: str | None = None


class FinancialYearCloseModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	year_id: str
	total_income: Decimal
	total_expenses: Decimal
	statutory_reserve_pct: Decimal = Decimal("20")  # % of surplus to statutory reserve
	education_fund_pct: Decimal = Decimal("5")
	closed_by: str
	approved_by: str


# ── Dividend Declaration ──────────────────────────────────────────────────────

class DividendDeclarationModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	year_id: str
	dividend_rate_pct: Decimal   # % on share capital
	rebate_rate_pct: Decimal     # % on savings/deposits
	declared_by: str
	board_resolution_ref: str
	declaration_date: str
	payment_date: str


class DividendDeclarationUpdateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	dividend_rate_pct: Decimal | None = None
	rebate_rate_pct: Decimal | None = None
	payment_date: str | None = None


# ── Member Distribution ───────────────────────────────────────────────────────

class MemberDistributionModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	declaration_id: str
	member_id: str
	member_number: str | None = None
	share_capital: Decimal
	dividend_gross: Decimal
	savings_balance: Decimal
	rebate_gross: Decimal
	withholding_tax: Decimal
	net_payable: Decimal
	payment_method: str
	payment_reference: str | None = None
	status: str  # pending | paid | failed
	created_at: str


# ── Withholding Tax ───────────────────────────────────────────────────────────

class WithholdingTaxModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	declaration_id: str
	total_gross_dividends: Decimal
	total_gross_rebates: Decimal
	total_wht: Decimal
	wht_rate: Decimal
	kra_return_reference: str | None = None
	filed_by: str | None = None
	filed_at: str | None = None


# ── Surplus Allocation ────────────────────────────────────────────────────────

class SurplusAllocationModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	year_id: str
	gross_surplus: Decimal
	statutory_reserve: Decimal
	education_fund: Decimal
	dividend_pool: Decimal
	rebate_pool: Decimal
	retained_surplus: Decimal
	allocation_approved_by: str
	allocation_date: str


# ── Filter & Audit ────────────────────────────────────────────────────────────

class DividendFilterModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	year_id: str | None = None
	declaration_id: str | None = None
	member_id: str | None = None
	status: str | None = None
	from_date: str | None = None
	to_date: str | None = None


class DividendAuditModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	year_id: str | None = None
	declaration_id: str | None = None
	member_id: str | None = None
	amount: Decimal | None = None
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: str


class DividendListModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[dict[str, Any]] = Field(default_factory=list)
	total: int = 0
	page: int = 1
	page_size: int = 50
