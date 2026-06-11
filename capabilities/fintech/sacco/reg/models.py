"""Pydantic v2 models for SASRA Regulatory Reporting.

All monetary amounts are Decimal (KES). Ratios are Decimal percentages (e.g. 15.00 = 15%).
"""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


# ── Enumerations ──────────────────────────────────────────────────────────────

class ReturnType(str, Enum):
	QUARTERLY = "quarterly"
	ANNUAL = "annual"
	CAPITAL_ADEQUACY = "capital_adequacy"
	LIQUIDITY = "liquidity"
	LOAN_PORTFOLIO = "loan_portfolio"
	BOARD_REPORT = "board_report"


class FilingStatus(str, Enum):
	PENDING = "pending"
	SUBMITTED = "submitted"
	ACCEPTED = "accepted"
	REJECTED = "rejected"
	OVERDUE = "overdue"


class LoanClassificationBand(str, Enum):
	NORMAL = "normal"          # 0-30 DPD — 0% provision
	WATCH = "watch"            # 31-90 DPD — 1% provision
	SUBSTANDARD = "substandard"  # 91-180 DPD — 25% provision
	DOUBTFUL = "doubtful"      # 181-365 DPD — 50% provision
	LOSS = "loss"              # >365 DPD — 100% provision


class TrafficLight(str, Enum):
	GREEN = "green"    # compliant with margin
	AMBER = "amber"    # within 2% of minimum — warning
	RED = "red"        # breach


# ── SASRA Form 1: Balance Sheet ───────────────────────────────────────────────

class BalanceSheet(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	# Assets
	cash_and_bank: Decimal = Decimal("0")
	government_securities: Decimal = Decimal("0")
	other_liquid_assets: Decimal = Decimal("0")
	gross_loan_portfolio: Decimal = Decimal("0")
	loan_loss_provisions: Decimal = Decimal("0")
	net_loan_portfolio: Decimal = Decimal("0")
	other_investments: Decimal = Decimal("0")
	fixed_assets: Decimal = Decimal("0")
	other_assets: Decimal = Decimal("0")
	total_assets: Decimal = Decimal("0")

	# Liabilities
	member_deposits: Decimal = Decimal("0")
	external_borrowings: Decimal = Decimal("0")
	other_liabilities: Decimal = Decimal("0")
	total_liabilities: Decimal = Decimal("0")

	# Equity / Members' Funds
	share_capital: Decimal = Decimal("0")
	retained_earnings: Decimal = Decimal("0")
	statutory_reserve: Decimal = Decimal("0")
	other_reserves: Decimal = Decimal("0")
	total_equity: Decimal = Decimal("0")


# ── SASRA Form 2: Income Statement ───────────────────────────────────────────

class IncomeStatement(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	# Income
	interest_income_loans: Decimal = Decimal("0")
	interest_income_investments: Decimal = Decimal("0")
	fee_income: Decimal = Decimal("0")
	other_income: Decimal = Decimal("0")
	total_income: Decimal = Decimal("0")

	# Expenses
	interest_expense_deposits: Decimal = Decimal("0")
	interest_expense_borrowings: Decimal = Decimal("0")
	provision_for_loan_losses: Decimal = Decimal("0")
	staff_costs: Decimal = Decimal("0")
	administrative_expenses: Decimal = Decimal("0")
	other_expenses: Decimal = Decimal("0")
	total_expenses: Decimal = Decimal("0")

	net_surplus_deficit: Decimal = Decimal("0")


# ── SASRA Form 3: Capital Adequacy ───────────────────────────────────────────

class CapitalAdequacyResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	as_of_date: str
	# Core (Tier 1) capital
	core_capital: Decimal = Decimal("0")        # paid-up share capital + retained earnings
	# Secondary (Tier 2) capital
	secondary_capital: Decimal = Decimal("0")   # general provisions + statutory reserves
	institutional_capital: Decimal = Decimal("0")  # core + secondary

	# Risk-weighted assets
	total_assets: Decimal = Decimal("0")
	risk_weighted_assets: Decimal = Decimal("0")

	# Ratios
	capital_adequacy_ratio: Decimal = Decimal("0")      # institutional_capital / risk_weighted_assets
	core_capital_ratio: Decimal = Decimal("0")           # core_capital / total_assets
	minimum_required: Decimal = Decimal("10.00")         # SASRA: 10%
	compliant: bool = False
	shortfall: Decimal = Decimal("0")
	traffic_light: TrafficLight = TrafficLight.RED


# ── SASRA Form 4: Liquidity ───────────────────────────────────────────────────

class LiquidityResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	as_of_date: str
	cash_on_hand: Decimal = Decimal("0")
	bank_balances: Decimal = Decimal("0")
	government_securities: Decimal = Decimal("0")
	other_liquid_assets: Decimal = Decimal("0")
	total_liquid_assets: Decimal = Decimal("0")

	total_deposits: Decimal = Decimal("0")
	total_borrowings: Decimal = Decimal("0")
	total_deposits_and_borrowings: Decimal = Decimal("0")

	liquidity_ratio: Decimal = Decimal("0")       # liquid_assets / (deposits + borrowings)
	minimum_required: Decimal = Decimal("15.00")  # SASRA: 15%
	compliant: bool = False
	shortfall: Decimal = Decimal("0")
	traffic_light: TrafficLight = TrafficLight.RED


# ── SASRA Form 5: Loan Portfolio Quality ─────────────────────────────────────

class LoanBand(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	band: LoanClassificationBand
	dpd_range: str  # human-readable: "0-30", "31-90", etc.
	number_of_loans: int = 0
	outstanding_balance: Decimal = Decimal("0")
	provision_rate: Decimal = Decimal("0")
	required_provision: Decimal = Decimal("0")


class LoanClassification(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	as_of_date: str
	bands: list[LoanBand] = Field(default_factory=list)
	total_gross_portfolio: Decimal = Decimal("0")
	total_required_provisions: Decimal = Decimal("0")
	actual_provisions_held: Decimal = Decimal("0")
	provisioning_coverage: Decimal = Decimal("0")  # actual / required * 100
	npl_balance: Decimal = Decimal("0")             # substandard + doubtful + loss
	npl_ratio: Decimal = Decimal("0")               # npl / gross_portfolio * 100
	par30: Decimal = Decimal("0")                   # (watch+substandard+doubtful+loss) / gross
	par90: Decimal = Decimal("0")                   # (substandard+doubtful+loss) / gross


# ── Quarterly Return ──────────────────────────────────────────────────────────

class QuarterlyReturn(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	year: int
	quarter: int  # 1-4
	period_end: str  # ISO date of quarter-end
	generated_at: str

	form1_balance_sheet: BalanceSheet = Field(default_factory=BalanceSheet)
	form2_income_statement: IncomeStatement = Field(default_factory=IncomeStatement)
	form3_capital_adequacy: CapitalAdequacyResult | None = None
	form4_liquidity: LiquidityResult | None = None
	form5_loan_classification: LoanClassification | None = None

	# Aggregate compliance
	overall_compliant: bool = False
	violations: list[str] = Field(default_factory=list)
	warnings: list[str] = Field(default_factory=list)


# ── Annual Report ─────────────────────────────────────────────────────────────

class AnnualReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	year: int
	generated_at: str

	balance_sheet: BalanceSheet = Field(default_factory=BalanceSheet)
	income_statement: IncomeStatement = Field(default_factory=IncomeStatement)
	capital_adequacy: CapitalAdequacyResult | None = None
	liquidity: LiquidityResult | None = None
	loan_classification: LoanClassification | None = None

	# Key ratios for board pack
	key_ratios: dict[str, Any] = Field(default_factory=dict)
	quarterly_snapshots: list[dict[str, Any]] = Field(default_factory=list)


# ── Compliance Status ─────────────────────────────────────────────────────────

class RatioStatus(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	actual: Decimal
	minimum: Decimal | None = None
	maximum: Decimal | None = None
	compliant: bool
	traffic_light: TrafficLight
	description: str = ""


class ComplianceStatus(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id: str
	as_of_date: str
	overall_compliant: bool
	violations: list[str] = Field(default_factory=list)
	warnings: list[str] = Field(default_factory=list)
	ratios: list[RatioStatus] = Field(default_factory=list)
	checked_at: str


# ── Filing Record ─────────────────────────────────────────────────────────────

class FilingRecord(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	return_type: ReturnType
	period: str              # e.g. "2025-Q1", "2025-annual"
	filing_officer: str
	submitted_at: str
	filing_status: FilingStatus = FilingStatus.SUBMITTED
	reference_number: str = ""  # SASRA acknowledgement ref
	data_snapshot: dict[str, Any] = Field(default_factory=dict)
	notes: str = ""
	created_at: str


# ── Regulatory Calendar Entry ─────────────────────────────────────────────────

class RegulatoryDeadline(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	period: str
	return_type: ReturnType
	due_date: str
	description: str
	days_remaining: int
	overdue: bool
	filed: bool = False
	filing_id: str | None = None


# ── Dashboard ─────────────────────────────────────────────────────────────────

class ComplianceDashboard(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id: str
	as_of_date: str
	overall_status: TrafficLight
	ratios: list[RatioStatus] = Field(default_factory=list)
	pending_filings: list[RegulatoryDeadline] = Field(default_factory=list)
	last_filing: FilingRecord | None = None
	generated_at: str
