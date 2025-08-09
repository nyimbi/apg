"""
APG Cash Management - Data Models

Enterprise-grade data models for APG Cash Management capability.
Implements CLAUDE.md standards with async Python, modern typing, and APG integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import asyncio
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict
from pydantic.types import EmailStr, PositiveFloat, PositiveInt
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from uuid_extensions import uuid7str


# =============================================================================
# Configuration and Validation
# =============================================================================

def _log_validation_error(field_name: str, value: Any, error: str) -> str:
	"""Log validation errors with consistent formatting."""
	return f"Validation failed for {field_name}: {value} - {error}"


def validate_positive_amount(value: Union[float, Decimal]) -> Decimal:
	"""Validate that monetary amounts are positive."""
	if value < 0:
		raise ValueError(_log_validation_error("amount", value, "cannot be negative"))
	return Decimal(str(value))


def validate_currency_code(value: str) -> str:
	"""Validate ISO 4217 currency codes."""
	if len(value) != 3 or not value.isupper() or not value.isalpha():
		raise ValueError(_log_validation_error("currency_code", value, "must be 3-letter ISO code"))
	return value


def validate_account_number(value: str) -> str:
	"""Validate bank account number format."""
	if not value or not value.strip():
		raise ValueError(_log_validation_error("account_number", value, "cannot be empty"))
	# Remove spaces and special characters for validation
	cleaned = "".join(c for c in value if c.isalnum())
	if len(cleaned) < 4:
		raise ValueError(_log_validation_error("account_number", value, "too short"))
	return value.strip()


def validate_swift_code(value: str) -> str:
	"""Validate SWIFT/BIC code format."""
	if len(value) not in [8, 11] or not value.isalnum():
		raise ValueError(_log_validation_error("swift_code", value, "must be 8 or 11 alphanumeric characters"))
	return value.upper()


def validate_iban(value: str) -> str:
	"""Validate IBAN format (basic validation)."""
	cleaned = "".join(c for c in value if c.isalnum()).upper()
	if len(cleaned) < 15 or len(cleaned) > 34:
		raise ValueError(_log_validation_error("iban", value, "invalid length"))
	return cleaned


def validate_percentage(value: float) -> float:
	"""Validate percentage values (0-100)."""
	if not 0 <= value <= 100:
		raise ValueError(_log_validation_error("percentage", value, "must be between 0 and 100"))
	return value


# Type aliases with validation
PositiveAmount = Annotated[Decimal, AfterValidator(validate_positive_amount)]
CurrencyCode = Annotated[str, AfterValidator(validate_currency_code)]
AccountNumber = Annotated[str, AfterValidator(validate_account_number)]
SwiftCode = Annotated[str, AfterValidator(validate_swift_code)]
IbanCode = Annotated[str, AfterValidator(validate_iban)]
Percentage = Annotated[float, AfterValidator(validate_percentage)]


# =============================================================================
# Enumerations
# =============================================================================

class CashAccountType(str, Enum):
	"""Cash account type enumeration."""
	CHECKING = "checking"
	SAVINGS = "savings"
	MONEY_MARKET = "money_market"
	CERTIFICATE_DEPOSIT = "certificate_deposit"
	INVESTMENT = "investment"
	CREDIT_LINE = "credit_line"
	PETTY_CASH = "petty_cash"
	ESCROW = "escrow"


class CashAccountStatus(str, Enum):
	"""Cash account status enumeration."""
	ACTIVE = "active"
	INACTIVE = "inactive"
	CLOSED = "closed"
	FROZEN = "frozen"
	RESTRICTED = "restricted"
	PENDING_CLOSURE = "pending_closure"


class BankStatus(str, Enum):
	"""Bank relationship status enumeration."""
	ACTIVE = "active"
	INACTIVE = "inactive"
	UNDER_REVIEW = "under_review"
	RESTRICTED = "restricted"
	TERMINATED = "terminated"


class InvestmentType(str, Enum):
	"""Investment instrument type enumeration."""
	MONEY_MARKET_FUND = "money_market_fund"
	TREASURY_BILL = "treasury_bill"
	CERTIFICATE_DEPOSIT = "certificate_deposit"
	COMMERCIAL_PAPER = "commercial_paper"
	REPURCHASE_AGREEMENT = "repurchase_agreement"
	TERM_DEPOSIT = "term_deposit"
	GOVERNMENT_BOND = "government_bond"
	CORPORATE_BOND = "corporate_bond"


class InvestmentStatus(str, Enum):
	"""Investment status enumeration."""
	PENDING = "pending"
	ACTIVE = "active"
	MATURED = "matured"
	REDEEMED = "redeemed"
	CANCELLED = "cancelled"
	DEFAULTED = "defaulted"


class RiskRating(str, Enum):
	"""Risk rating enumeration."""
	VERY_LOW = "very_low"
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	VERY_HIGH = "very_high"


class ForecastType(str, Enum):
	"""Cash forecast type enumeration."""
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	ROLLING = "rolling"
	SCENARIO = "scenario"


class ForecastScenario(str, Enum):
	"""Forecast scenario enumeration."""
	BASE_CASE = "base_case"
	OPTIMISTIC = "optimistic"
	PESSIMISTIC = "pessimistic"
	STRESS_TEST = "stress_test"
	CUSTOM = "custom"


class TransactionType(str, Enum):
	"""Cash transaction type enumeration."""
	DEPOSIT = "deposit"
	WITHDRAWAL = "withdrawal"
	TRANSFER = "transfer"
	INVESTMENT = "investment"
	REDEMPTION = "redemption"
	INTEREST_EARNED = "interest_earned"
	FEES = "fees"
	FX_CONVERSION = "fx_conversion"
	SWEEP = "sweep"


class AlertType(str, Enum):
	"""Cash alert type enumeration."""
	BALANCE_LOW = "balance_low"
	BALANCE_HIGH = "balance_high"
	FORECAST_SHORTFALL = "forecast_shortfall"
	INVESTMENT_MATURITY = "investment_maturity"
	RATE_CHANGE = "rate_change"
	RISK_THRESHOLD = "risk_threshold"
	BANK_CONNECTION = "bank_connection"
	COMPLIANCE_VIOLATION = "compliance_violation"


class OptimizationGoal(str, Enum):
	"""Cash optimization goal enumeration."""
	MAXIMIZE_YIELD = "maximize_yield"
	MINIMIZE_RISK = "minimize_risk"
	MAXIMIZE_LIQUIDITY = "maximize_liquidity"
	BALANCE_YIELD_RISK = "balance_yield_risk"
	MINIMIZE_FEES = "minimize_fees"


# =============================================================================
# Base Models
# =============================================================================

class APGBaseModel(BaseModel):
	"""Base model with APG multi-tenant patterns and common fields."""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		validate_default=True
	)
	
	# APG Integration Fields
	id: str = Field(default_factory=uuid7str, description="Unique identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Audit Fields (APG audit_compliance integration)
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: str = Field(..., description="User who created the record")
	updated_by: str = Field(..., description="User who last updated the record")
	
	# Versioning for optimistic locking
	version: int = Field(default=1, description="Record version for optimistic locking")
	
	# Soft delete support
	is_deleted: bool = Field(default=False, description="Soft delete flag")
	deleted_at: Optional[datetime] = Field(default=None, description="Deletion timestamp")
	deleted_by: Optional[str] = Field(default=None, description="User who deleted the record")

	def _log_model_validation(self, action: str) -> str:
		"""Log model validation actions."""
		return f"Model {self.__class__.__name__} {action}: {self.id}"

	@root_validator
	def validate_apg_integration(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate APG integration requirements."""
		assert values.get('tenant_id'), "tenant_id required for APG multi-tenancy"
		assert values.get('created_by'), "created_by required for audit compliance"
		return values


# =============================================================================
# Bank and Account Models
# =============================================================================

class BankContact(BaseModel):
	"""Bank contact information."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	contact_type: str = Field(..., description="primary, treasury, operations, etc.")
	name: str = Field(..., min_length=1, max_length=100)
	title: Optional[str] = Field(None, max_length=100)
	email: Optional[EmailStr] = Field(None, max_length=255)
	phone: Optional[str] = Field(None, max_length=50)
	mobile: Optional[str] = Field(None, max_length=50)
	is_primary: bool = Field(default=False)


class Bank(APGBaseModel):
	"""
	Bank master data model for cash management.
	
	Integrates with APG's vendor management for bank relationship tracking
	and provides comprehensive bank information for treasury operations.
	"""
	
	# Bank identification
	bank_code: str = Field(..., min_length=1, max_length=20, description="Internal bank code")
	bank_name: str = Field(..., min_length=1, max_length=200, description="Bank legal name")
	swift_code: SwiftCode = Field(..., description="SWIFT/BIC code")
	
	# Bank details
	country_code: str = Field(..., min_length=2, max_length=3, description="ISO country code")
	city: str = Field(..., min_length=1, max_length=100)
	address_line1: str = Field(..., min_length=1, max_length=255)
	address_line2: Optional[str] = Field(None, max_length=255)
	postal_code: str = Field(..., min_length=1, max_length=20)
	
	# Contact information
	contacts: List[BankContact] = Field(default_factory=list)
	
	# Relationship details
	status: BankStatus = Field(default=BankStatus.ACTIVE)
	relationship_manager: Optional[str] = Field(None, max_length=100)
	credit_rating: Optional[str] = Field(None, max_length=10)
	
	# API integration
	api_enabled: bool = Field(default=False)
	api_endpoint: Optional[str] = Field(None, max_length=500)
	api_credentials_encrypted: Optional[str] = Field(None, description="Encrypted API credentials")
	last_api_sync: Optional[datetime] = Field(None)
	
	# Fees and terms
	standard_fees: Dict[str, Decimal] = Field(default_factory=dict)
	fee_structure: Optional[str] = Field(None, description="Fee structure documentation")


class CashAccount(APGBaseModel):
	"""
	Cash account model for treasury management.
	
	Represents bank accounts, investment accounts, and other cash holdings
	with real-time balance tracking and automated reconciliation support.
	"""
	
	# Account identification
	account_number: AccountNumber = Field(..., description="Bank account number")
	account_name: str = Field(..., min_length=1, max_length=200)
	iban: Optional[IbanCode] = Field(None, description="International Bank Account Number")
	
	# Account details
	bank_id: str = Field(..., description="Reference to Bank")
	account_type: CashAccountType = Field(...)
	currency_code: CurrencyCode = Field(...)
	
	# Entity and organization
	entity_id: str = Field(..., description="Business entity owning the account")
	cost_center: Optional[str] = Field(None, max_length=50)
	department: Optional[str] = Field(None, max_length=100)
	purpose: Optional[str] = Field(None, max_length=500, description="Account purpose description")
	
	# Balance information
	current_balance: Decimal = Field(default=Decimal('0'), description="Current book balance")
	available_balance: Decimal = Field(default=Decimal('0'), description="Available for use")
	pending_credits: Decimal = Field(default=Decimal('0'))
	pending_debits: Decimal = Field(default=Decimal('0'))
	last_balance_update: Optional[datetime] = Field(None)
	
	# Account status and limits
	status: CashAccountStatus = Field(default=CashAccountStatus.ACTIVE)
	overdraft_limit: Optional[Decimal] = Field(None, description="Overdraft facility limit")
	minimum_balance: Optional[Decimal] = Field(None, description="Required minimum balance")
	maximum_balance: Optional[Decimal] = Field(None, description="Maximum balance limit")
	
	# Interest and fees
	interest_rate: Optional[Decimal] = Field(None, description="Current interest rate")
	fee_schedule: Dict[str, Decimal] = Field(default_factory=dict)
	
	# Automation settings
	auto_sweep_enabled: bool = Field(default=False)
	sweep_target_account: Optional[str] = Field(None, description="Target for cash sweeping")
	sweep_threshold: Optional[Decimal] = Field(None, description="Balance threshold for sweeping")
	
	# Reconciliation
	last_reconciled_date: Optional[date] = Field(None)
	reconciliation_status: str = Field(default="pending", description="pending/reconciled/variance")
	
	@property
	def effective_balance(self) -> Decimal:
		"""Calculate effective balance including pending transactions."""
		return self.current_balance + self.pending_credits - self.pending_debits
	
	@property
	def is_overdrawn(self) -> bool:
		"""Check if account is overdrawn."""
		return self.effective_balance < Decimal('0')


# =============================================================================
# Cash Position and Flow Models
# =============================================================================

class CashPosition(APGBaseModel):
	"""
	Real-time cash position model for treasury visibility.
	
	Aggregates cash balances across accounts with multi-currency support
	and provides real-time position tracking for treasury operations.
	"""
	
	# Position identification
	position_date: date = Field(..., description="Position as of date")
	entity_id: str = Field(..., description="Business entity")
	currency_code: CurrencyCode = Field(...)
	
	# Balance aggregation
	total_cash: Decimal = Field(default=Decimal('0'), description="Total cash across all accounts")
	available_cash: Decimal = Field(default=Decimal('0'), description="Available for operations")
	restricted_cash: Decimal = Field(default=Decimal('0'), description="Restricted or earmarked cash")
	invested_cash: Decimal = Field(default=Decimal('0'), description="Cash in investments")
	
	# Account breakdown
	checking_balance: Decimal = Field(default=Decimal('0'))
	savings_balance: Decimal = Field(default=Decimal('0'))
	money_market_balance: Decimal = Field(default=Decimal('0'))
	investment_balance: Decimal = Field(default=Decimal('0'))
	
	# Projected flows (next 30 days)
	projected_inflows: Decimal = Field(default=Decimal('0'))
	projected_outflows: Decimal = Field(default=Decimal('0'))
	net_projected_flow: Decimal = Field(default=Decimal('0'))
	
	# Key ratios and metrics
	liquidity_ratio: Optional[float] = Field(None, description="Liquidity coverage ratio")
	concentration_risk: Optional[float] = Field(None, description="Single bank concentration %")
	yield_rate: Optional[float] = Field(None, description="Weighted average yield %")
	
	# Risk indicators
	days_cash_on_hand: Optional[int] = Field(None, description="Days of operating cash")
	stress_test_coverage: Optional[float] = Field(None, description="Stress scenario coverage")
	
	@property
	def total_position(self) -> Decimal:
		"""Calculate total cash position including investments."""
		return self.total_cash + self.invested_cash
	
	@property
	def projected_end_balance(self) -> Decimal:
		"""Calculate projected balance after net flows."""
		return self.available_cash + self.net_projected_flow


class CashFlow(APGBaseModel):
	"""
	Cash flow transaction model for detailed flow tracking.
	
	Records individual cash movements with categorization and source tracking
	for comprehensive cash flow analysis and forecasting.
	"""
	
	# Transaction identification
	flow_date: date = Field(..., description="Cash flow date")
	transaction_id: Optional[str] = Field(None, description="Source transaction reference")
	description: str = Field(..., min_length=1, max_length=500)
	
	# Flow details
	account_id: str = Field(..., description="Cash account affected")
	transaction_type: TransactionType = Field(...)
	amount: Decimal = Field(..., description="Flow amount (positive=inflow, negative=outflow)")
	currency_code: CurrencyCode = Field(...)
	
	# Categorization
	category: str = Field(..., max_length=100, description="Cash flow category")
	subcategory: Optional[str] = Field(None, max_length=100)
	business_unit: Optional[str] = Field(None, max_length=100)
	cost_center: Optional[str] = Field(None, max_length=50)
	
	# Source integration
	source_module: Optional[str] = Field(None, max_length=50, description="AP, AR, GL, etc.")
	source_document: Optional[str] = Field(None, max_length=100)
	counterparty: Optional[str] = Field(None, max_length=200)
	
	# Forecasting attributes
	is_recurring: bool = Field(default=False)
	recurrence_pattern: Optional[str] = Field(None, description="Monthly, quarterly, etc.")
	forecast_confidence: Optional[float] = Field(None, description="Forecast confidence 0-1")
	
	# Timing
	planned_date: Optional[date] = Field(None, description="Originally planned date")
	actual_date: Optional[date] = Field(None, description="Actual execution date")
	value_date: Optional[date] = Field(None, description="Value date for interest")
	
	@property
	def is_inflow(self) -> bool:
		"""Check if this is a cash inflow."""
		return self.amount > Decimal('0')
	
	@property
	def is_outflow(self) -> bool:
		"""Check if this is a cash outflow."""
		return self.amount < Decimal('0')


# =============================================================================
# Forecasting Models
# =============================================================================

class CashForecast(APGBaseModel):
	"""
	AI-powered cash forecast model for predictive analytics.
	
	Provides machine learning-based cash flow forecasting with confidence
	intervals and scenario analysis for strategic treasury planning.
	"""
	
	# Forecast identification
	forecast_id: str = Field(default_factory=uuid7str, description="Unique forecast identifier")
	forecast_date: date = Field(..., description="Date forecast was generated")
	forecast_type: ForecastType = Field(...)
	scenario: ForecastScenario = Field(default=ForecastScenario.BASE_CASE)
	
	# Forecast scope
	entity_id: str = Field(..., description="Business entity")
	currency_code: CurrencyCode = Field(...)
	horizon_days: int = Field(..., ge=1, le=365, description="Forecast horizon in days")
	
	# Opening position
	opening_balance: Decimal = Field(..., description="Starting cash position")
	opening_date: date = Field(..., description="Forecast start date")
	
	# Forecast components
	projected_inflows: Decimal = Field(default=Decimal('0'))
	projected_outflows: Decimal = Field(default=Decimal('0'))
	net_flow: Decimal = Field(default=Decimal('0'))
	closing_balance: Decimal = Field(default=Decimal('0'))
	
	# Statistical measures
	confidence_level: float = Field(..., ge=0.0, le=1.0, description="Forecast confidence 0-1")
	confidence_interval_lower: Decimal = Field(..., description="Lower bound of confidence interval")
	confidence_interval_upper: Decimal = Field(..., description="Upper bound of confidence interval")
	standard_deviation: Optional[Decimal] = Field(None, description="Forecast volatility")
	
	# Model information
	model_used: str = Field(..., max_length=100, description="ML model used for forecast")
	model_version: str = Field(..., max_length=50)
	training_data_period: Optional[str] = Field(None, description="Training data timeframe")
	feature_importance: Dict[str, float] = Field(default_factory=dict)
	
	# Forecast accuracy (for backtesting)
	actual_outcome: Optional[Decimal] = Field(None, description="Actual result when available")
	forecast_error: Optional[Decimal] = Field(None, description="Forecast vs actual")
	accuracy_percentage: Optional[float] = Field(None, description="Accuracy as percentage")
	
	# Risk assessment
	shortfall_probability: Optional[float] = Field(None, description="Probability of cash shortfall")
	stress_test_result: Optional[Decimal] = Field(None, description="Stress scenario outcome")
	value_at_risk: Optional[Decimal] = Field(None, description="VaR at 95% confidence")
	
	@property
	def forecast_accuracy_score(self) -> Optional[float]:
		"""Calculate forecast accuracy score (0-100)."""
		if self.accuracy_percentage is not None:
			return max(0, 100 - abs(self.accuracy_percentage))
		return None


class ForecastAssumption(APGBaseModel):
	"""
	Forecast assumption model for scenario analysis.
	
	Captures assumptions used in cash forecasting for transparency
	and scenario modeling with sensitivity analysis.
	"""
	
	# Assumption identification
	forecast_id: str = Field(..., description="Associated forecast")
	assumption_name: str = Field(..., max_length=200)
	category: str = Field(..., max_length=100, description="Revenue, expenses, timing, etc.")
	
	# Assumption details
	base_value: Decimal = Field(..., description="Base case assumption value")
	optimistic_value: Optional[Decimal] = Field(None, description="Optimistic scenario value")
	pessimistic_value: Optional[Decimal] = Field(None, description="Pessimistic scenario value")
	
	# Statistical properties
	probability_distribution: Optional[str] = Field(None, description="normal, uniform, triangular")
	mean: Optional[Decimal] = Field(None)
	standard_deviation: Optional[Decimal] = Field(None)
	minimum_value: Optional[Decimal] = Field(None)
	maximum_value: Optional[Decimal] = Field(None)
	
	# Sensitivity analysis
	sensitivity_coefficient: Optional[float] = Field(None, description="Impact on forecast")
	correlation_factors: Dict[str, float] = Field(default_factory=dict)
	
	# Documentation
	description: Optional[str] = Field(None, max_length=1000)
	data_source: Optional[str] = Field(None, max_length=200)
	last_reviewed: Optional[date] = Field(None)
	confidence_level: Optional[float] = Field(None, ge=0.0, le=1.0)


# =============================================================================
# Investment Models
# =============================================================================

class Investment(APGBaseModel):
	"""
	Investment instrument model for cash investment management.
	
	Tracks short-term investments with automated optimization and
	performance monitoring for maximum yield with appropriate risk.
	"""
	
	# Investment identification
	investment_number: str = Field(..., max_length=50, description="Internal investment number")
	external_reference: Optional[str] = Field(None, max_length=100, description="External trade/deal number")
	
	# Investment details
	investment_type: InvestmentType = Field(...)
	issuer: str = Field(..., max_length=200, description="Investment issuer/counterparty")
	issuer_rating: Optional[str] = Field(None, max_length=10, description="Credit rating")
	
	# Financial terms
	principal_amount: PositiveAmount = Field(..., description="Investment principal")
	currency_code: CurrencyCode = Field(...)
	interest_rate: Decimal = Field(..., description="Annual interest rate as decimal")
	compounding_frequency: str = Field(default="daily", description="Interest compounding")
	
	# Dates and timing
	trade_date: date = Field(..., description="Investment execution date")
	value_date: date = Field(..., description="Value/settlement date")
	maturity_date: date = Field(..., description="Investment maturity date")
	early_redemption_date: Optional[date] = Field(None, description="Early redemption option")
	
	# Status and lifecycle
	status: InvestmentStatus = Field(default=InvestmentStatus.PENDING)
	booking_account_id: str = Field(..., description="Cash account for booking")
	
	# Performance tracking
	expected_return: Decimal = Field(default=Decimal('0'), description="Expected total return")
	accrued_interest: Decimal = Field(default=Decimal('0'), description="Interest accrued to date")
	current_value: Decimal = Field(default=Decimal('0'), description="Current market value")
	realized_return: Optional[Decimal] = Field(None, description="Actual return if matured")
	
	# Risk management
	risk_rating: RiskRating = Field(default=RiskRating.LOW)
	credit_limit_impact: Decimal = Field(default=Decimal('0'), description="Impact on credit limits")
	liquidity_rating: Optional[str] = Field(None, max_length=20)
	regulatory_treatment: Optional[str] = Field(None, max_length=100)
	
	# Optimization metadata
	optimization_score: Optional[float] = Field(None, description="AI optimization score 0-100")
	selection_reason: Optional[str] = Field(None, max_length=500, description="Why this investment was selected")
	alternative_considered: Optional[str] = Field(None, max_length=200)
	
	@property
	def days_to_maturity(self) -> int:
		"""Calculate days to maturity."""
		return (self.maturity_date - date.today()).days
	
	@property
	def annualized_yield(self) -> Decimal:
		"""Calculate annualized yield to maturity."""
		if self.days_to_maturity <= 0:
			return Decimal('0')
		days_held = (self.maturity_date - self.value_date).days
		if days_held <= 0:
			return Decimal('0')
		total_return = self.expected_return / self.principal_amount
		return total_return * Decimal('365') / Decimal(str(days_held))


class InvestmentOpportunity(APGBaseModel):
	"""
	Investment opportunity model for automated optimization.
	
	Represents available investment opportunities with AI scoring
	and automated selection based on optimization objectives.
	"""
	
	# Opportunity identification
	opportunity_id: str = Field(default_factory=uuid7str)
	source: str = Field(..., max_length=100, description="Money market, bank, platform, etc.")
	provider: str = Field(..., max_length=200, description="Investment provider")
	
	# Investment details
	investment_type: InvestmentType = Field(...)
	minimum_amount: PositiveAmount = Field(..., description="Minimum investment amount")
	maximum_amount: Optional[PositiveAmount] = Field(None, description="Maximum investment amount")
	currency_code: CurrencyCode = Field(...)
	
	# Terms and conditions
	interest_rate: Decimal = Field(..., description="Offered interest rate")
	term_days: int = Field(..., ge=1, description="Investment term in days")
	early_redemption_penalty: Optional[Decimal] = Field(None, description="Early redemption cost")
	fees: Decimal = Field(default=Decimal('0'), description="Investment fees")
	
	# Counterparty information
	counterparty_name: str = Field(..., max_length=200)
	counterparty_rating: Optional[str] = Field(None, max_length=10)
	risk_rating: RiskRating = Field(...)
	
	# Opportunity window
	available_from: datetime = Field(..., description="Opportunity available from")
	available_until: datetime = Field(..., description="Opportunity expires at")
	last_updated: datetime = Field(default_factory=datetime.utcnow)
	
	# AI scoring and optimization
	ai_score: float = Field(..., ge=0.0, le=100.0, description="AI optimization score")
	yield_score: float = Field(..., ge=0.0, le=100.0, description="Yield attractiveness")
	risk_score: float = Field(..., ge=0.0, le=100.0, description="Risk assessment")
	liquidity_score: float = Field(..., ge=0.0, le=100.0, description="Liquidity assessment")
	
	# Recommendation
	recommended_amount: Optional[PositiveAmount] = Field(None, description="AI recommended amount")
	recommendation_reason: Optional[str] = Field(None, max_length=500)
	fit_score: Optional[float] = Field(None, description="Fit with portfolio 0-100")
	
	@property
	def is_available(self) -> bool:
		"""Check if opportunity is currently available."""
		now = datetime.utcnow()
		return self.available_from <= now <= self.available_until
	
	@property
	def annualized_yield(self) -> Decimal:
		"""Calculate annualized yield."""
		return self.interest_rate * Decimal('365') / Decimal(str(self.term_days))


# =============================================================================
# Alert and Notification Models
# =============================================================================

class CashAlert(APGBaseModel):
	"""
	Cash alert model for automated monitoring and notifications.
	
	Provides intelligent alerting for cash management with configurable
	thresholds and automated escalation procedures.
	"""
	
	# Alert identification
	alert_type: AlertType = Field(...)
	severity: str = Field(..., description="low, medium, high, critical")
	title: str = Field(..., max_length=200)
	description: str = Field(..., max_length=1000)
	
	# Alert context
	entity_id: str = Field(..., description="Affected business entity")
	account_id: Optional[str] = Field(None, description="Affected cash account")
	currency_code: Optional[CurrencyCode] = Field(None)
	
	# Alert data
	current_value: Optional[Decimal] = Field(None, description="Current value triggering alert")
	threshold_value: Optional[Decimal] = Field(None, description="Configured threshold")
	variance_amount: Optional[Decimal] = Field(None, description="Amount of variance")
	variance_percentage: Optional[float] = Field(None, description="Percentage variance")
	
	# Alert timing
	triggered_at: datetime = Field(default_factory=datetime.utcnow)
	escalated_at: Optional[datetime] = Field(None)
	resolved_at: Optional[datetime] = Field(None)
	acknowledged_at: Optional[datetime] = Field(None)
	acknowledged_by: Optional[str] = Field(None)
	
	# Alert status
	status: str = Field(default="active", description="active, acknowledged, resolved, dismissed")
	resolution_notes: Optional[str] = Field(None, max_length=1000)
	auto_resolved: bool = Field(default=False)
	
	# Notification tracking
	notifications_sent: List[str] = Field(default_factory=list, description="Notification IDs sent")
	escalation_level: int = Field(default=1, description="Current escalation level")
	max_escalations: int = Field(default=3, description="Maximum escalation levels")
	
	# Related data
	related_forecast_id: Optional[str] = Field(None, description="Related forecast if applicable")
	related_investment_id: Optional[str] = Field(None, description="Related investment if applicable")
	
	@property
	def is_active(self) -> bool:
		"""Check if alert is still active."""
		return self.status == "active"
	
	@property
	def hours_since_triggered(self) -> float:
		"""Calculate hours since alert was triggered."""
		return (datetime.utcnow() - self.triggered_at).total_seconds() / 3600


class OptimizationRule(APGBaseModel):
	"""
	Cash optimization rule model for automated decision making.
	
	Defines rules and policies for automated cash optimization
	with AI-powered decision support and risk management.
	"""
	
	# Rule identification
	rule_name: str = Field(..., max_length=200)
	rule_code: str = Field(..., max_length=50, description="Unique rule code")
	category: str = Field(..., max_length=100, description="investment, sweeping, hedging, etc.")
	
	# Rule scope
	entity_ids: List[str] = Field(default_factory=list, description="Applicable entities")
	currency_codes: List[CurrencyCode] = Field(default_factory=list, description="Applicable currencies")
	account_types: List[CashAccountType] = Field(default_factory=list)
	
	# Rule logic
	optimization_goal: OptimizationGoal = Field(...)
	priority: int = Field(default=50, ge=1, le=100, description="Rule priority 1-100")
	
	# Conditions
	minimum_amount: Optional[PositiveAmount] = Field(None, description="Minimum amount to trigger")
	maximum_amount: Optional[PositiveAmount] = Field(None, description="Maximum amount to process")
	time_constraints: Optional[str] = Field(None, description="Time-based constraints")
	market_conditions: Optional[str] = Field(None, description="Market condition requirements")
	
	# Investment parameters
	maximum_maturity_days: Optional[int] = Field(None, description="Maximum investment term")
	minimum_yield_rate: Optional[Decimal] = Field(None, description="Minimum acceptable yield")
	maximum_risk_rating: Optional[RiskRating] = Field(None)
	diversification_limits: Dict[str, Decimal] = Field(default_factory=dict)
	
	# Risk controls
	single_counterparty_limit: Optional[Decimal] = Field(None, description="Maximum exposure per counterparty")
	concentration_limit: Optional[Percentage] = Field(None, description="Maximum concentration %")
	stress_test_threshold: Optional[Decimal] = Field(None, description="Stress test minimum")
	
	# Execution settings
	auto_execute: bool = Field(default=False, description="Auto-execute without approval")
	approval_required_above: Optional[PositiveAmount] = Field(None, description="Approval threshold")
	notification_recipients: List[str] = Field(default_factory=list)
	
	# Rule status
	is_active: bool = Field(default=True)
	last_executed: Optional[datetime] = Field(None)
	execution_count: int = Field(default=0)
	success_rate: Optional[float] = Field(None, description="Rule success rate 0-1")
	
	# AI enhancement
	ai_enhanced: bool = Field(default=True, description="Use AI for rule optimization")
	learning_enabled: bool = Field(default=True, description="Enable machine learning")
	model_version: Optional[str] = Field(None, description="AI model version used")


# =============================================================================
# Export All Models
# =============================================================================

__all__ = [
	# Base models
	'APGBaseModel',
	
	# Type aliases
	'PositiveAmount', 'CurrencyCode', 'AccountNumber', 'SwiftCode', 
	'IbanCode', 'Percentage',
	
	# Enumerations
	'CashAccountType', 'CashAccountStatus', 'BankStatus', 'InvestmentType',
	'InvestmentStatus', 'RiskRating', 'ForecastType', 'ForecastScenario',
	'TransactionType', 'AlertType', 'OptimizationGoal',
	
	# Bank and account models
	'BankContact', 'Bank', 'CashAccount',
	
	# Cash position and flow models
	'CashPosition', 'CashFlow',
	
	# Forecasting models
	'CashForecast', 'ForecastAssumption',
	
	# Investment models
	'Investment', 'InvestmentOpportunity',
	
	# Alert and optimization models
	'CashAlert', 'OptimizationRule'
]