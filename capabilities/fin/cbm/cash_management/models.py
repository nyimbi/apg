"""
Cash Management — Pydantic v2 domain models.

Covers the full entity set: BankAccount, BankStatement, BankTransaction,
CashFlow, CashForecast, BankReconciliation, CashPosition, LiquidityPool,
IntercompanyLoan, FXPosition, HedgeInstrument, CashConcentration — plus
all supporting enums, base model, and report aggregation models.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_currency(v: str) -> str:
	v = v.strip().upper()
	if len(v) != 3 or not v.isalpha():
		raise ValueError(f"Invalid ISO 4217 currency code: {v!r}")
	return v


def _validate_positive(v: Decimal) -> Decimal:
	if v < Decimal("0"):
		raise ValueError("Amount must be non-negative")
	return v


def _validate_rate(v: Decimal) -> Decimal:
	"""Decimal rate — accepts any sign (negative rates exist in some markets)."""
	return v


CurrencyCode = Annotated[str, AfterValidator(_validate_currency)]
PositiveDecimal = Annotated[Decimal, AfterValidator(_validate_positive)]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AccountStatus(str, Enum):
	ACTIVE = "active"
	INACTIVE = "inactive"
	FROZEN = "frozen"
	RESTRICTED = "restricted"
	CLOSED = "closed"
	PENDING_CLOSURE = "pending_closure"


class AccountType(str, Enum):
	OPERATING = "operating"
	SAVINGS = "savings"
	MONEY_MARKET = "money_market"
	INVESTMENT = "investment"
	PETTY_CASH = "petty_cash"
	LOCKBOX = "lockbox"
	ESCROW = "escrow"
	OVERDRAFT = "overdraft"
	REVOLVING_CREDIT = "revolving_credit"
	CONCENTRATION = "concentration"
	NOTIONAL_POOL = "notional_pool"


class StatementFormat(str, Enum):
	MT940 = "MT940"
	MT942 = "MT942"
	CAMT053 = "camt.053"
	CAMT052 = "camt.052"
	MPESA = "mpesa"
	CSV = "csv"
	OFX = "ofx"
	BAI2 = "bai2"


class StatementStatus(str, Enum):
	PENDING = "pending"
	IMPORTED = "imported"
	PROCESSING = "processing"
	RECONCILED = "reconciled"
	FAILED = "failed"
	PARTIAL = "partial"


class TransactionStatus(str, Enum):
	PENDING = "pending"
	CLEARED = "cleared"
	REJECTED = "rejected"
	REVERSED = "reversed"
	IN_FLOAT = "in_float"
	SAME_DAY = "same_day"


class TransactionType(str, Enum):
	CREDIT = "credit"
	DEBIT = "debit"
	TRANSFER = "transfer"
	FEE = "fee"
	INTEREST = "interest"
	FX_CONVERSION = "fx_conversion"
	SWEEP = "sweep"
	MPESA = "mpesa"
	SWIFT = "swift"
	ACH = "ach"
	CHAPS = "chaps"
	RTGS = "rtgs"


class ReconciliationStatus(str, Enum):
	UNRECONCILED = "unreconciled"
	AUTO_MATCHED = "auto_matched"
	MANUALLY_MATCHED = "manually_matched"
	EXCEPTION = "exception"
	CLEARED = "cleared"


class ForecastScenario(str, Enum):
	BASE = "base"
	OPTIMISTIC = "optimistic"
	PESSIMISTIC = "pessimistic"
	STRESS = "stress"


class ForecastStatus(str, Enum):
	DRAFT = "draft"
	ACTIVE = "active"
	SUPERSEDED = "superseded"
	ARCHIVED = "archived"


class LoanStatus(str, Enum):
	PROPOSED = "proposed"
	ACTIVE = "active"
	SETTLED = "settled"
	OVERDUE = "overdue"
	CANCELLED = "cancelled"


class HedgeType(str, Enum):
	FORWARD = "forward"
	OPTION = "option"
	SWAP = "swap"
	NDF = "ndf"
	CROSS_CURRENCY_SWAP = "cross_currency_swap"


class HedgeStatus(str, Enum):
	PENDING = "pending"
	ACTIVE = "active"
	MATURED = "matured"
	CANCELLED = "cancelled"
	CLOSED = "closed"


class PoolingType(str, Enum):
	NOTIONAL = "notional"
	PHYSICAL = "physical"
	ZBA = "zero_balance"
	TARGET_BALANCE = "target_balance"


class ConcentrationMethod(str, Enum):
	ZERO_BALANCE = "zero_balance"
	TARGET_BALANCE = "target_balance"
	THRESHOLD = "threshold"
	MANUAL = "manual"


class FlowType(str, Enum):
	INFLOW = "inflow"
	OUTFLOW = "outflow"
	TRANSFER = "transfer"


class FlowStatus(str, Enum):
	FORECAST = "forecast"
	CONFIRMED = "confirmed"
	ACTUAL = "actual"
	CANCELLED = "cancelled"


class PositionStatus(str, Enum):
	DRAFT = "draft"
	CONFIRMED = "confirmed"
	REVIEWED = "reviewed"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class CbmBase(BaseModel):
	"""Multi-tenant base for all CBM entities."""

	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
	)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., min_length=1)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(..., min_length=1)
	is_deleted: bool = Field(default=False)


# ---------------------------------------------------------------------------
# BankAccount
# ---------------------------------------------------------------------------

class BankAccountCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	bank_code: str = Field(..., max_length=20)
	bank_name: str = Field(..., max_length=200)
	account_number: str = Field(..., min_length=4, max_length=50)
	account_name: str = Field(..., max_length=200)
	account_type: AccountType
	currency: CurrencyCode
	entity_id: str
	iban: str | None = None
	swift_bic: str | None = None
	routing_number: str | None = None
	overdraft_limit: Decimal = Field(default=Decimal("0"))
	minimum_balance: Decimal = Field(default=Decimal("0"))
	revolving_credit_limit: Decimal = Field(default=Decimal("0"))
	is_restricted: bool = False
	restriction_reason: str | None = None
	country_code: str = Field(default="US", max_length=2)
	branch_code: str | None = None
	open_banking_enabled: bool = False
	open_banking_provider: str | None = None


class BankAccountUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	account_name: str | None = None
	account_type: AccountType | None = None
	status: AccountStatus | None = None
	overdraft_limit: Decimal | None = None
	minimum_balance: Decimal | None = None
	revolving_credit_limit: Decimal | None = None
	is_restricted: bool | None = None
	restriction_reason: str | None = None
	open_banking_enabled: bool | None = None
	open_banking_provider: str | None = None


class BankAccount(CbmBase):
	"""Bank account master record."""

	bank_code: str
	bank_name: str
	account_number: str
	account_name: str
	account_type: AccountType
	currency: CurrencyCode
	entity_id: str
	status: AccountStatus = AccountStatus.ACTIVE
	iban: str | None = None
	swift_bic: str | None = None
	routing_number: str | None = None
	current_balance: Decimal = Decimal("0")
	available_balance: Decimal = Decimal("0")
	ledger_balance: Decimal = Decimal("0")
	overdraft_limit: Decimal = Decimal("0")
	minimum_balance: Decimal = Decimal("0")
	revolving_credit_limit: Decimal = Decimal("0")
	revolving_credit_utilised: Decimal = Decimal("0")
	is_restricted: bool = False
	restriction_reason: str | None = None
	country_code: str = "US"
	branch_code: str | None = None
	last_statement_date: date | None = None
	last_reconciled_date: date | None = None
	open_banking_enabled: bool = False
	open_banking_provider: str | None = None
	value_date_offset: int = 0  # same-day = 0, next-day = 1


# ---------------------------------------------------------------------------
# BankStatement
# ---------------------------------------------------------------------------

class BankStatementCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	account_id: str
	statement_date: date
	opening_balance: Decimal
	closing_balance: Decimal
	currency: CurrencyCode
	statement_format: StatementFormat = StatementFormat.MT940
	raw_content: str | None = None
	source_reference: str | None = None


class BankStatement(CbmBase):
	"""Imported bank statement header."""

	account_id: str
	statement_date: date
	opening_balance: Decimal
	closing_balance: Decimal
	currency: CurrencyCode
	statement_format: StatementFormat
	status: StatementStatus = StatementStatus.PENDING
	raw_content: str | None = None
	source_reference: str | None = None
	transaction_count: int = 0
	matched_count: int = 0
	unmatched_count: int = 0
	import_errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# BankTransaction
# ---------------------------------------------------------------------------

class BankTransactionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	statement_id: str
	account_id: str
	transaction_date: date
	value_date: date
	amount: Decimal  # positive = credit, negative = debit
	currency: CurrencyCode
	transaction_type: TransactionType
	reference: str | None = None
	counterparty_name: str | None = None
	counterparty_account: str | None = None
	description: str | None = None
	swift_gpi_uetr: str | None = None
	mpesa_reference: str | None = None
	bank_reference: str | None = None


class BankTransaction(CbmBase):
	"""Individual bank transaction line from a statement."""

	statement_id: str
	account_id: str
	transaction_date: date
	value_date: date
	amount: Decimal
	currency: CurrencyCode
	transaction_type: TransactionType
	status: TransactionStatus = TransactionStatus.PENDING
	reference: str | None = None
	counterparty_name: str | None = None
	counterparty_account: str | None = None
	description: str | None = None
	swift_gpi_uetr: str | None = None
	mpesa_reference: str | None = None
	bank_reference: str | None = None
	reconciliation_id: str | None = None
	gl_entry_id: str | None = None
	is_same_day_value: bool = False


# ---------------------------------------------------------------------------
# CashFlow
# ---------------------------------------------------------------------------

class CashFlowCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	account_id: str
	flow_type: FlowType
	amount: PositiveDecimal
	currency: CurrencyCode
	expected_date: date
	category: str = Field(..., max_length=100)
	subcategory: str | None = None
	counterparty: str | None = None
	source_module: str | None = None
	source_document_id: str | None = None
	is_recurring: bool = False
	recurrence_pattern: str | None = None
	notes: str | None = None


class CashFlow(CbmBase):
	"""Cash flow item — forecast, confirmed, or actual."""

	account_id: str
	flow_type: FlowType
	status: FlowStatus = FlowStatus.FORECAST
	amount: PositiveDecimal
	currency: CurrencyCode
	expected_date: date
	actual_date: date | None = None
	value_date: date | None = None
	category: str
	subcategory: str | None = None
	counterparty: str | None = None
	source_module: str | None = None
	source_document_id: str | None = None
	is_recurring: bool = False
	recurrence_pattern: str | None = None
	notes: str | None = None
	forecast_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# CashForecast
# ---------------------------------------------------------------------------

class CashForecastCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	entity_id: str
	currency: CurrencyCode
	horizon_days: int = Field(default=90, ge=1, le=365)
	scenario: ForecastScenario = ForecastScenario.BASE
	opening_balance: Decimal
	forecast_date: date


class CashForecastLineItem(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	forecast_date: date
	projected_inflows: Decimal
	projected_outflows: Decimal
	net_flow: Decimal
	closing_balance: Decimal
	confidence: float = Field(ge=0.0, le=1.0)


class CashForecast(CbmBase):
	"""Multi-day rolling cash forecast."""

	entity_id: str
	currency: CurrencyCode
	horizon_days: int
	scenario: ForecastScenario
	status: ForecastStatus = ForecastStatus.DRAFT
	forecast_date: date
	opening_balance: Decimal
	total_projected_inflows: Decimal = Decimal("0")
	total_projected_outflows: Decimal = Decimal("0")
	projected_closing_balance: Decimal = Decimal("0")
	overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
	line_items: list[CashForecastLineItem] = Field(default_factory=list)
	model_used: str = "statistical"
	shortfall_days: list[date] = Field(default_factory=list)
	peak_shortfall: Decimal = Decimal("0")
	reviewed_by: str | None = None
	reviewed_at: datetime | None = None


# ---------------------------------------------------------------------------
# BankReconciliation
# ---------------------------------------------------------------------------

class ReconciliationMatchCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	bank_transaction_id: str
	gl_entry_id: str | None = None
	cash_flow_id: str | None = None
	match_type: str = "auto"
	notes: str | None = None


class BankReconciliationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	account_id: str
	statement_id: str
	reconciliation_date: date
	bank_closing_balance: Decimal
	ledger_closing_balance: Decimal
	currency: CurrencyCode


class BankReconciliation(CbmBase):
	"""Bank reconciliation header."""

	account_id: str
	statement_id: str
	reconciliation_date: date
	status: ReconciliationStatus = ReconciliationStatus.UNRECONCILED
	bank_closing_balance: Decimal
	ledger_closing_balance: Decimal
	currency: CurrencyCode
	variance: Decimal = Decimal("0")
	matched_items: int = 0
	unmatched_items: int = 0
	outstanding_deposits: Decimal = Decimal("0")
	outstanding_payments: Decimal = Decimal("0")
	bank_errors: Decimal = Decimal("0")
	book_errors: Decimal = Decimal("0")
	reviewed_by: str | None = None
	reviewed_at: datetime | None = None
	approved_by: str | None = None
	approved_at: datetime | None = None


# ---------------------------------------------------------------------------
# CashPosition
# ---------------------------------------------------------------------------

class CashPositionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	account_id: str
	position_date: date
	available_balance: Decimal
	ledger_balance: Decimal
	value_dated_balance: Decimal | None = None
	float_amount: Decimal = Decimal("0")


class CashPosition(CbmBase):
	"""Point-in-time cash position for a single account."""

	account_id: str
	position_date: date
	status: PositionStatus = PositionStatus.DRAFT
	available_balance: Decimal
	ledger_balance: Decimal
	value_dated_balance: Decimal | None = None
	float_amount: Decimal = Decimal("0")
	projected_closing: Decimal | None = None
	days_cash_on_hand: int | None = None
	reviewed_by: str | None = None
	reviewed_at: datetime | None = None


# ---------------------------------------------------------------------------
# LiquidityPool
# ---------------------------------------------------------------------------

class LiquidityPoolCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	pool_name: str = Field(..., max_length=200)
	pooling_type: PoolingType
	header_account_id: str
	currency: CurrencyCode
	participating_account_ids: list[str] = Field(default_factory=list)
	target_balance: Decimal = Decimal("0")
	interest_rate_credit: Decimal = Decimal("0")
	interest_rate_debit: Decimal = Decimal("0")


class LiquidityPool(CbmBase):
	"""Cash pooling structure — notional or physical."""

	pool_name: str
	pooling_type: PoolingType
	header_account_id: str
	currency: CurrencyCode
	participating_account_ids: list[str] = Field(default_factory=list)
	notional_pool_balance: Decimal = Decimal("0")
	physical_pool_balance: Decimal = Decimal("0")
	target_balance: Decimal = Decimal("0")
	interest_rate_credit: Decimal = Decimal("0")
	interest_rate_debit: Decimal = Decimal("0")
	is_active: bool = True
	last_swept_at: datetime | None = None
	sweep_count: int = 0


# ---------------------------------------------------------------------------
# IntercompanyLoan
# ---------------------------------------------------------------------------

class IntercompanyLoanCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	lender_entity_id: str
	borrower_entity_id: str
	lender_account_id: str
	borrower_account_id: str
	principal: PositiveDecimal
	currency: CurrencyCode
	interest_rate: Decimal
	start_date: date
	maturity_date: date
	approved_by: str


class IntercompanyLoan(CbmBase):
	"""Intercompany lending / cash repatriation instrument."""

	lender_entity_id: str
	borrower_entity_id: str
	lender_account_id: str
	borrower_account_id: str
	principal: PositiveDecimal
	outstanding_balance: PositiveDecimal
	currency: CurrencyCode
	interest_rate: Decimal
	start_date: date
	maturity_date: date
	status: LoanStatus = LoanStatus.PROPOSED
	accrued_interest: Decimal = Decimal("0")
	total_interest_paid: Decimal = Decimal("0")
	approved_by: str
	settled_at: datetime | None = None
	settlement_account_id: str | None = None


# ---------------------------------------------------------------------------
# FXPosition
# ---------------------------------------------------------------------------

class FXPositionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	entity_id: str
	base_currency: CurrencyCode
	quote_currency: CurrencyCode
	position_date: date
	long_amount: Decimal = Decimal("0")
	short_amount: Decimal = Decimal("0")
	spot_rate: Decimal


class FXPosition(CbmBase):
	"""Foreign exchange exposure position."""

	entity_id: str
	base_currency: CurrencyCode
	quote_currency: CurrencyCode
	position_date: date
	long_amount: Decimal
	short_amount: Decimal
	net_position: Decimal = Decimal("0")
	spot_rate: Decimal
	base_equivalent: Decimal = Decimal("0")
	unrealised_pnl: Decimal = Decimal("0")
	hedged_amount: Decimal = Decimal("0")
	unhedged_amount: Decimal = Decimal("0")
	hedge_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# HedgeInstrument
# ---------------------------------------------------------------------------

class HedgeInstrumentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	entity_id: str
	fx_position_id: str
	hedge_type: HedgeType
	buy_currency: CurrencyCode
	sell_currency: CurrencyCode
	notional_amount: PositiveDecimal
	contracted_rate: Decimal
	trade_date: date
	maturity_date: date
	counterparty: str
	approved_by: str


class HedgeInstrument(CbmBase):
	"""FX hedge — forward, option, or swap."""

	entity_id: str
	fx_position_id: str
	hedge_type: HedgeType
	status: HedgeStatus = HedgeStatus.PENDING
	buy_currency: CurrencyCode
	sell_currency: CurrencyCode
	notional_amount: PositiveDecimal
	contracted_rate: Decimal
	spot_rate_at_trade: Decimal | None = None
	trade_date: date
	maturity_date: date
	counterparty: str
	approved_by: str
	current_fair_value: Decimal = Decimal("0")
	unrealised_pnl: Decimal = Decimal("0")
	premium_paid: Decimal = Decimal("0")
	settled_at: datetime | None = None
	settlement_rate: Decimal | None = None


# ---------------------------------------------------------------------------
# CashConcentration
# ---------------------------------------------------------------------------

class CashConcentrationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	pool_id: str
	concentration_method: ConcentrationMethod
	source_account_id: str
	target_account_id: str
	sweep_amount: PositiveDecimal
	currency: CurrencyCode
	sweep_date: date
	triggered_by: str = "manual"


class CashConcentration(CbmBase):
	"""Single cash sweep / concentration event."""

	pool_id: str
	concentration_method: ConcentrationMethod
	source_account_id: str
	target_account_id: str
	sweep_amount: PositiveDecimal
	currency: CurrencyCode
	sweep_date: date
	triggered_by: str = "manual"
	status: str = "pending"
	executed_at: datetime | None = None
	execution_reference: str | None = None
	notes: str | None = None


# ---------------------------------------------------------------------------
# Report / aggregation models
# ---------------------------------------------------------------------------

class DailyPositionSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	position_date: date
	entity_id: str
	currency: CurrencyCode
	total_cash: Decimal
	available_cash: Decimal
	restricted_cash: Decimal
	overdraft_used: Decimal
	net_fx_exposure: Decimal
	liquidity_coverage_ratio: float | None = None
	days_cash_on_hand: int | None = None


class ReconciliationSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_of_date: date
	total_accounts: int
	fully_reconciled: int
	with_exceptions: int
	total_variance: Decimal
	largest_variance_account: str | None = None
	largest_variance_amount: Decimal | None = None


class FXExposureSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_of_date: date
	base_currency: CurrencyCode
	total_long_exposure: Decimal
	total_short_exposure: Decimal
	net_exposure: Decimal
	total_hedged: Decimal
	total_unhedged: Decimal
	hedge_ratio: float
	unrealised_pnl: Decimal


class LiquidityStressResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	scenario: ForecastScenario
	as_of_date: date
	starting_balance: Decimal
	stressed_balance: Decimal
	minimum_balance: Decimal
	minimum_balance_date: date | None = None
	shortfall_amount: Decimal
	days_until_shortfall: int | None = None
	survival_days: int
	haircut_applied: float
	passed: bool


class CashPoolSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	pool_id: str
	pool_name: str
	pooling_type: PoolingType
	currency: CurrencyCode
	as_of_date: date
	notional_balance: Decimal
	physical_balance: Decimal
	participating_accounts: int
	sweep_count_today: int
	interest_savings_ytd: Decimal


class SwiftGpiStatus(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	uetr: str
	transaction_id: str
	status: str
	instructed_amount: Decimal
	instructed_currency: CurrencyCode
	credited_amount: Decimal | None = None
	credited_currency: CurrencyCode | None = None
	sender_bic: str
	receiver_bic: str
	last_update: datetime
	settlement_date: date | None = None
	charges_deducted: Decimal = Decimal("0")
	fx_rate_applied: Decimal | None = None
	tracker_events: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Alert / Optimization / Investment lightweight models
# (used by analytics/event modules that import from .models)
# ---------------------------------------------------------------------------

class AlertSeverity(str, Enum):
	INFO = "info"
	WARNING = "warning"
	CRITICAL = "critical"
	EMERGENCY = "emergency"


class AlertType(str, Enum):
	LOW_BALANCE = "low_balance"
	OVERDRAFT_RISK = "overdraft_risk"
	LARGE_TRANSACTION = "large_transaction"
	RECONCILIATION_VARIANCE = "reconciliation_variance"
	FORECAST_DEVIATION = "forecast_deviation"
	FX_EXPOSURE_BREACH = "fx_exposure_breach"
	COVENANT_BREACH = "covenant_breach"
	LIQUIDITY_DEFICIT = "liquidity_deficit"
	TRAPPED_CASH = "trapped_cash"
	PAYMENT_FAILURE = "payment_failure"


class InvestmentStatus(str, Enum):
	PENDING = "pending"
	ACTIVE = "active"
	MATURED = "matured"
	CANCELLED = "cancelled"
	REDEEMED = "redeemed"


class CashAlert(CbmBase):
	"""Operational alert on a cash account or position."""
	alert_type: AlertType
	severity: AlertSeverity
	account_id: str | None = None
	message: str
	threshold: Decimal | None = None
	actual_value: Decimal | None = None
	acknowledged: bool = False
	acknowledged_by: str | None = None
	acknowledged_at: datetime | None = None


class Investment(CbmBase):
	"""Short-term investment instrument."""
	account_id: str
	investment_type: str  # from SUPPORTED_INVESTMENT_TYPES
	counterparty: str
	principal: Decimal
	currency: CurrencyCode
	annual_rate: Decimal
	start_date: date
	maturity_date: date
	interest_accrued: Decimal = Decimal("0")
	status: InvestmentStatus = InvestmentStatus.PENDING
	reference: str | None = None
	notes: str | None = None


class OptimizationRule(CbmBase):
	"""Rule governing automated cash concentration / sweep logic."""
	rule_name: str
	account_id: str
	target_balance: Decimal
	min_balance: Decimal
	max_balance: Decimal
	sweep_to_pool_id: str | None = None
	trigger: str = "end_of_day"  # end_of_day | intraday | threshold
	is_active: bool = True
	priority: int = 100


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

# Some peripheral modules import "Bank" — map to BankAccount
Bank = BankAccount
# Some modules import "CashAccount" — map to BankAccount (same entity)
CashAccount = BankAccount
