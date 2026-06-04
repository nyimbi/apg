"""
APG Budgeting & Forecasting — Pydantic v2 Data Models

© 2025 Datacraft. Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated
from uuid6 import uuid7


# ---------------------------------------------------------------------------
# uuid7 shim
# ---------------------------------------------------------------------------

def uuid7str() -> str:
	"""Return a UUID7 string."""
	return str(uuid7())


# ---------------------------------------------------------------------------
# Annotated scalar validators
# ---------------------------------------------------------------------------

def _validate_non_negative(v: Decimal) -> Decimal:
	if v < Decimal("0"):
		raise ValueError("amount cannot be negative")
	return v


def _validate_currency_code(v: str) -> str:
	v = v.strip().upper()
	if len(v) != 3 or not v.isalpha():
		raise ValueError("currency_code must be a 3-letter ISO 4217 code")
	return v


def _validate_non_empty(v: str) -> str:
	v = v.strip()
	if not v:
		raise ValueError("value cannot be empty")
	return v


def _validate_percentage(v: float) -> float:
	if not (0.0 <= v <= 100.0):
		raise ValueError("percentage must be between 0 and 100")
	return v


def _validate_probability(v: float) -> float:
	if not (0.0 <= v <= 1.0):
		raise ValueError("probability must be between 0 and 1")
	return v


NonNegativeDecimal = Annotated[Decimal, AfterValidator(_validate_non_negative)]
CurrencyCode = Annotated[str, AfterValidator(_validate_currency_code)]
NonEmptyStr = Annotated[str, AfterValidator(_validate_non_empty)]
PercentageFloat = Annotated[float, AfterValidator(_validate_percentage)]
ProbabilityFloat = Annotated[float, AfterValidator(_validate_probability)]


# ---------------------------------------------------------------------------
# Status / type enumerations  (BF = Budgeting & Forecasting prefix)
# ---------------------------------------------------------------------------

class BFBudgetType(str, Enum):
	ANNUAL = "annual"
	QUARTERLY = "quarterly"
	MONTHLY = "monthly"
	ROLLING = "rolling"
	PROJECT = "project"
	CAPITAL = "capital"
	OPERATIONAL = "operational"
	ZERO_BASED = "zero_based"


class BFBudgetStatus(str, Enum):
	DRAFT = "draft"
	SUBMITTED = "submitted"
	UNDER_REVIEW = "under_review"
	APPROVED = "approved"
	ACTIVE = "active"
	LOCKED = "locked"
	CLOSED = "closed"
	CANCELLED = "cancelled"


class BFLineType(str, Enum):
	REVENUE = "revenue"
	EXPENSE = "expense"
	CAPITAL = "capital"
	TRANSFER = "transfer"
	ALLOCATION = "allocation"
	CONTINGENCY = "contingency"


class BFDistributionMethod(str, Enum):
	EQUAL = "equal"
	TOP_DOWN = "top_down"
	BOTTOM_UP = "bottom_up"
	ZERO_BASED = "zero_based"
	SEASONAL = "seasonal"
	WEIGHTED = "weighted"
	DRIVER_BASED = "driver_based"


class BFForecastType(str, Enum):
	REVENUE = "revenue"
	EXPENSE = "expense"
	CASH_FLOW = "cash_flow"
	DEMAND = "demand"
	INTEGRATED = "integrated"
	SCENARIO = "scenario"


class BFForecastMethod(str, Enum):
	STATISTICAL = "statistical"
	ML = "ml"
	HYBRID = "hybrid"
	JUDGMENTAL = "judgmental"
	ENSEMBLE = "ensemble"
	DRIVER_BASED = "driver_based"
	ROLLING = "rolling"
	AI = "ai"


class BFForecastStatus(str, Enum):
	DRAFT = "draft"
	GENERATING = "generating"
	COMPLETED = "completed"
	PUBLISHED = "published"
	ARCHIVED = "archived"
	FAILED = "failed"


class BFVarianceType(str, Enum):
	FAVORABLE = "favorable"
	UNFAVORABLE = "unfavorable"
	NEUTRAL = "neutral"


class BFSignificanceLevel(str, Enum):
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	MINIMAL = "minimal"


class BFScenarioType(str, Enum):
	BASE = "base"
	OPTIMISTIC = "optimistic"
	PESSIMISTIC = "pessimistic"
	STRESS = "stress"
	WHAT_IF = "what_if"
	MONTE_CARLO = "monte_carlo"


class BFApprovalStatus(str, Enum):
	PENDING = "pending"
	APPROVED = "approved"
	REJECTED = "rejected"
	REQUIRES_REVISION = "requires_revision"
	DELEGATED = "delegated"


class BFVersionStatus(str, Enum):
	WORKING = "working"
	BASELINE = "baseline"
	ARCHIVED = "archived"


class BFDriverType(str, Enum):
	VOLUME = "volume"
	PRICE = "price"
	HEADCOUNT = "headcount"
	EXCHANGE_RATE = "exchange_rate"
	INFLATION = "inflation"
	CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class BFBase(BaseModel):
	"""Common audit / tenancy fields for every BF entity."""

	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		validate_default=True,
	)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(...)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str = Field(...)
	updated_by: str = Field(...)
	is_deleted: bool = Field(default=False)
	deleted_at: datetime | None = Field(default=None)
	deleted_by: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

class BFBudget(BFBase):
	"""Master budget record."""

	name: NonEmptyStr = Field(..., max_length=200)
	description: str | None = Field(default=None, max_length=2000)
	budget_type: BFBudgetType = Field(default=BFBudgetType.ANNUAL)
	status: BFBudgetStatus = Field(default=BFBudgetStatus.DRAFT)
	fiscal_year: int = Field(...)
	period_start: date = Field(...)
	period_end: date = Field(...)
	currency_code: CurrencyCode = Field(default="USD")
	owner_id: str = Field(...)
	department_id: str | None = Field(default=None)
	cost_center_id: str | None = Field(default=None)
	template_id: str | None = Field(default=None)
	total_revenue: NonNegativeDecimal = Field(default=Decimal("0"))
	total_expense: NonNegativeDecimal = Field(default=Decimal("0"))
	net_amount: Decimal = Field(default=Decimal("0"))
	version: int = Field(default=1)
	locked_at: datetime | None = Field(default=None)
	approved_at: datetime | None = Field(default=None)
	approved_by: str | None = Field(default=None)
	submitted_by: str | None = Field(default=None)
	submitted_at: datetime | None = Field(default=None)
	notes: str | None = Field(default=None, max_length=5000)
	tags: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)

	@model_validator(mode="after")
	def _period_valid(self) -> "BFBudget":
		if self.period_end <= self.period_start:
			raise ValueError("period_end must be after period_start")
		return self


class BFBudgetCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: NonEmptyStr
	description: str | None = None
	budget_type: BFBudgetType = BFBudgetType.ANNUAL
	fiscal_year: int
	period_start: date
	period_end: date
	currency_code: CurrencyCode = "USD"
	owner_id: str
	department_id: str | None = None
	cost_center_id: str | None = None
	template_id: str | None = None
	notes: str | None = None
	tags: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


class BFBudgetUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: NonEmptyStr | None = None
	description: str | None = None
	notes: str | None = None
	tags: list[str] | None = None
	metadata: dict[str, Any] | None = None


class BFBudgetResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	name: str
	description: str | None
	budget_type: BFBudgetType
	status: BFBudgetStatus
	fiscal_year: int
	period_start: date
	period_end: date
	currency_code: str
	owner_id: str
	department_id: str | None
	cost_center_id: str | None
	total_revenue: Decimal
	total_expense: Decimal
	net_amount: Decimal
	version: int
	approved_by: str | None
	approved_at: datetime | None
	submitted_by: str | None
	submitted_at: datetime | None
	created_at: datetime
	updated_at: datetime
	created_by: str
	tags: list[str]


# ---------------------------------------------------------------------------
# BudgetVersion
# ---------------------------------------------------------------------------

class BFBudgetVersion(BFBase):
	"""Immutable snapshot of a budget at a point in time."""

	budget_id: str = Field(...)
	version_number: int = Field(...)
	version_label: NonEmptyStr = Field(...)
	status: BFVersionStatus = Field(default=BFVersionStatus.WORKING)
	snapshot_data: dict[str, Any] = Field(default_factory=dict)
	total_revenue: NonNegativeDecimal = Field(default=Decimal("0"))
	total_expense: NonNegativeDecimal = Field(default=Decimal("0"))
	net_amount: Decimal = Field(default=Decimal("0"))
	change_summary: str | None = Field(default=None)
	notes: str | None = Field(default=None)


class BFBudgetVersionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	budget_id: str
	version_label: NonEmptyStr
	change_summary: str | None = None
	notes: str | None = None


class BFBudgetVersionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id: str
	budget_id: str
	version_number: int
	version_label: str
	status: BFVersionStatus
	total_revenue: Decimal
	total_expense: Decimal
	net_amount: Decimal
	change_summary: str | None
	created_at: datetime
	created_by: str


# ---------------------------------------------------------------------------
# BudgetLine
# ---------------------------------------------------------------------------

class BFBudgetLine(BFBase):
	"""Individual budget line within a budget."""

	budget_id: str = Field(...)
	line_number: int = Field(...)
	description: NonEmptyStr = Field(..., max_length=500)
	line_type: BFLineType = Field(...)
	account_code: NonEmptyStr = Field(..., max_length=50)
	gl_account: str | None = Field(default=None, max_length=50)
	department_code: str | None = Field(default=None)
	cost_center_code: str | None = Field(default=None)
	project_code: str | None = Field(default=None)
	period_start: date = Field(...)
	period_end: date = Field(...)
	distribution_method: BFDistributionMethod = Field(default=BFDistributionMethod.EQUAL)
	budgeted_amount: NonNegativeDecimal = Field(...)
	committed_amount: NonNegativeDecimal = Field(default=Decimal("0"))
	actual_amount: NonNegativeDecimal = Field(default=Decimal("0"))
	variance_amount: Decimal = Field(default=Decimal("0"))
	variance_pct: Decimal = Field(default=Decimal("0"))
	# Monthly breakdown (12 slots, Jan–Dec)
	month_amounts: list[Decimal] = Field(default_factory=lambda: [Decimal("0")] * 12)
	notes: str | None = Field(default=None)
	tags: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)

	@field_validator("month_amounts")
	@classmethod
	def _twelve_months(cls, v: list[Decimal]) -> list[Decimal]:
		if len(v) != 12:
			raise ValueError("month_amounts must have exactly 12 elements")
		return v


class BFBudgetLineCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	budget_id: str
	description: NonEmptyStr
	line_type: BFLineType
	account_code: NonEmptyStr
	gl_account: str | None = None
	department_code: str | None = None
	cost_center_code: str | None = None
	project_code: str | None = None
	period_start: date
	period_end: date
	distribution_method: BFDistributionMethod = BFDistributionMethod.EQUAL
	budgeted_amount: NonNegativeDecimal
	month_amounts: list[Decimal] | None = None
	notes: str | None = None
	tags: list[str] = Field(default_factory=list)


class BFBudgetLineUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	description: NonEmptyStr | None = None
	budgeted_amount: NonNegativeDecimal | None = None
	month_amounts: list[Decimal] | None = None
	notes: str | None = None
	actual_amount: NonNegativeDecimal | None = None


class BFBudgetLineResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id: str
	budget_id: str
	line_number: int
	description: str
	line_type: BFLineType
	account_code: str
	gl_account: str | None
	department_code: str | None
	cost_center_code: str | None
	period_start: date
	period_end: date
	distribution_method: BFDistributionMethod
	budgeted_amount: Decimal
	committed_amount: Decimal
	actual_amount: Decimal
	variance_amount: Decimal
	variance_pct: Decimal
	month_amounts: list[Decimal]
	notes: str | None


# ---------------------------------------------------------------------------
# BudgetTemplate
# ---------------------------------------------------------------------------

class BFBudgetTemplate(BFBase):
	"""Reusable budget structure template."""

	name: NonEmptyStr = Field(..., max_length=200)
	description: str | None = Field(default=None)
	budget_type: BFBudgetType = Field(...)
	line_definitions: list[dict[str, Any]] = Field(default_factory=list)
	distribution_rules: dict[str, Any] = Field(default_factory=dict)
	is_active: bool = Field(default=True)
	industry: str | None = Field(default=None)
	tags: list[str] = Field(default_factory=list)
	usage_count: int = Field(default=0)


class BFBudgetTemplateCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	name: NonEmptyStr
	description: str | None = None
	budget_type: BFBudgetType
	line_definitions: list[dict[str, Any]] = Field(default_factory=list)
	distribution_rules: dict[str, Any] = Field(default_factory=dict)
	industry: str | None = None
	tags: list[str] = Field(default_factory=list)


class BFBudgetTemplateResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id: str
	name: str
	description: str | None
	budget_type: BFBudgetType
	is_active: bool
	industry: str | None
	line_count: int
	usage_count: int
	created_at: datetime
	created_by: str


# ---------------------------------------------------------------------------
# ForecastModel
# ---------------------------------------------------------------------------

class BFForecastModel(BFBase):
	"""Statistical / ML forecast model configuration."""

	name: NonEmptyStr = Field(..., max_length=200)
	description: str | None = Field(default=None)
	method: BFForecastMethod = Field(...)
	horizon_periods: int = Field(..., ge=1, le=120)
	lookback_periods: int = Field(default=24, ge=1)
	seasonality: bool = Field(default=True)
	trend: bool = Field(default=True)
	confidence_level: PercentageFloat = Field(default=95.0)
	hyperparameters: dict[str, Any] = Field(default_factory=dict)
	feature_columns: list[str] = Field(default_factory=list)
	is_active: bool = Field(default=True)
	last_trained_at: datetime | None = Field(default=None)
	model_metrics: dict[str, float] = Field(default_factory=dict)
	training_data_start: date | None = Field(default=None)
	training_data_end: date | None = Field(default=None)


class BFForecastModelCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	name: NonEmptyStr
	description: str | None = None
	method: BFForecastMethod
	horizon_periods: int = Field(ge=1, le=120)
	lookback_periods: int = Field(default=24, ge=1)
	seasonality: bool = True
	trend: bool = True
	confidence_level: PercentageFloat = 95.0
	hyperparameters: dict[str, Any] = Field(default_factory=dict)
	feature_columns: list[str] = Field(default_factory=list)


class BFForecastModelResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id: str
	name: str
	method: BFForecastMethod
	horizon_periods: int
	lookback_periods: int
	confidence_level: float
	is_active: bool
	last_trained_at: datetime | None
	model_metrics: dict[str, float]
	created_at: datetime


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

class BFForecast(BFBase):
	"""A generated forecast instance."""

	forecast_model_id: str | None = Field(default=None)
	budget_id: str | None = Field(default=None)
	name: NonEmptyStr = Field(..., max_length=200)
	forecast_type: BFForecastType = Field(...)
	status: BFForecastStatus = Field(default=BFForecastStatus.DRAFT)
	period_start: date = Field(...)
	period_end: date = Field(...)
	currency_code: CurrencyCode = Field(default="USD")
	total_forecasted: Decimal = Field(default=Decimal("0"))
	confidence_lower: Decimal | None = Field(default=None)
	confidence_upper: Decimal | None = Field(default=None)
	mape: float | None = Field(default=None)
	rmse: float | None = Field(default=None)
	error_message: str | None = Field(default=None)
	generated_at: datetime | None = Field(default=None)
	published_at: datetime | None = Field(default=None)
	metadata: dict[str, Any] = Field(default_factory=dict)


class BFForecastCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	forecast_model_id: str | None = None
	budget_id: str | None = None
	name: NonEmptyStr
	forecast_type: BFForecastType
	period_start: date
	period_end: date
	currency_code: CurrencyCode = "USD"
	metadata: dict[str, Any] = Field(default_factory=dict)


class BFForecastUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	status: BFForecastStatus | None = None
	error_message: str | None = None
	metadata: dict[str, Any] | None = None


class BFForecastResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id: str
	name: str
	forecast_type: BFForecastType
	status: BFForecastStatus
	period_start: date
	period_end: date
	total_forecasted: Decimal
	confidence_lower: Decimal | None
	confidence_upper: Decimal | None
	mape: float | None
	rmse: float | None
	generated_at: datetime | None
	created_at: datetime
	created_by: str


# ---------------------------------------------------------------------------
# ForecastLine
# ---------------------------------------------------------------------------

class BFForecastLine(BFBase):
	"""Single data point within a forecast."""

	forecast_id: str = Field(...)
	period_date: date = Field(...)
	account_code: str = Field(...)
	forecasted_value: Decimal = Field(...)
	lower_bound: Decimal | None = Field(default=None)
	upper_bound: Decimal | None = Field(default=None)
	actual_value: Decimal | None = Field(default=None)
	residual: Decimal | None = Field(default=None)
	is_outlier: bool = Field(default=False)
	driver_values: dict[str, float] = Field(default_factory=dict)


class BFForecastLineCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	forecast_id: str
	period_date: date
	account_code: str
	forecasted_value: Decimal
	lower_bound: Decimal | None = None
	upper_bound: Decimal | None = None
	driver_values: dict[str, float] = Field(default_factory=dict)


class BFForecastLineResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id: str
	forecast_id: str
	period_date: date
	account_code: str
	forecasted_value: Decimal
	lower_bound: Decimal | None
	upper_bound: Decimal | None
	actual_value: Decimal | None
	residual: Decimal | None
	is_outlier: bool
	driver_values: dict[str, float]


# ---------------------------------------------------------------------------
# VarianceReport
# ---------------------------------------------------------------------------

class BFVarianceReport(BFBase):
	"""Budget-vs-actual variance analysis report."""

	budget_id: str = Field(...)
	report_period_start: date = Field(...)
	report_period_end: date = Field(...)
	total_budget: Decimal = Field(default=Decimal("0"))
	total_actual: Decimal = Field(default=Decimal("0"))
	total_variance: Decimal = Field(default=Decimal("0"))
	variance_pct: Decimal = Field(default=Decimal("0"))
	variance_type: BFVarianceType = Field(default=BFVarianceType.NEUTRAL)
	significance: BFSignificanceLevel = Field(default=BFSignificanceLevel.LOW)
	line_variances: list[dict[str, Any]] = Field(default_factory=list)
	recommendations: list[str] = Field(default_factory=list)
	generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	reviewed_by: str | None = Field(default=None)


class BFVarianceReportResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id: str
	budget_id: str
	report_period_start: date
	report_period_end: date
	total_budget: Decimal
	total_actual: Decimal
	total_variance: Decimal
	variance_pct: Decimal
	variance_type: BFVarianceType
	significance: BFSignificanceLevel
	line_variances: list[dict[str, Any]]
	recommendations: list[str]
	generated_at: datetime
	reviewed_by: str | None


# ---------------------------------------------------------------------------
# BudgetApproval
# ---------------------------------------------------------------------------

class BFBudgetApproval(BFBase):
	"""Approval workflow record for a budget."""

	budget_id: str = Field(...)
	approver_id: str = Field(...)
	approver_name: str = Field(...)
	approver_role: str = Field(...)
	status: BFApprovalStatus = Field(default=BFApprovalStatus.PENDING)
	sequence: int = Field(default=1)
	required_by: datetime | None = Field(default=None)
	decided_at: datetime | None = Field(default=None)
	comments: str | None = Field(default=None)
	conditions: list[str] = Field(default_factory=list)
	delegated_to: str | None = Field(default=None)
	digital_signature: str | None = Field(default=None)


class BFBudgetApprovalCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	budget_id: str
	approver_id: str
	approver_name: str
	approver_role: str
	sequence: int = 1
	required_by: datetime | None = None
	conditions: list[str] = Field(default_factory=list)


class BFBudgetApprovalUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	status: BFApprovalStatus | None = None
	comments: str | None = None
	conditions: list[str] | None = None
	delegated_to: str | None = None
	digital_signature: str | None = None


class BFBudgetApprovalResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id: str
	budget_id: str
	approver_id: str
	approver_name: str
	approver_role: str
	status: BFApprovalStatus
	sequence: int
	required_by: datetime | None
	decided_at: datetime | None
	comments: str | None
	conditions: list[str]
	delegated_to: str | None


# ---------------------------------------------------------------------------
# ScenarioModel
# ---------------------------------------------------------------------------

class BFScenarioModel(BFBase):
	"""What-if / scenario definition."""

	name: NonEmptyStr = Field(..., max_length=200)
	description: str | None = Field(default=None)
	scenario_type: BFScenarioType = Field(...)
	base_budget_id: str | None = Field(default=None)
	base_forecast_id: str | None = Field(default=None)
	assumptions: dict[str, Any] = Field(default_factory=dict)
	adjustments: list[dict[str, Any]] = Field(default_factory=list)
	probability: ProbabilityFloat = Field(default=0.5)
	is_active: bool = Field(default=True)
	results: dict[str, Any] = Field(default_factory=dict)
	ran_at: datetime | None = Field(default=None)
	net_impact: Decimal | None = Field(default=None)
	net_impact_pct: Decimal | None = Field(default=None)


class BFScenarioCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	name: NonEmptyStr
	description: str | None = None
	scenario_type: BFScenarioType
	base_budget_id: str | None = None
	base_forecast_id: str | None = None
	assumptions: dict[str, Any] = Field(default_factory=dict)
	adjustments: list[dict[str, Any]] = Field(default_factory=list)
	probability: ProbabilityFloat = 0.5


class BFScenarioUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	name: NonEmptyStr | None = None
	description: str | None = None
	assumptions: dict[str, Any] | None = None
	adjustments: list[dict[str, Any]] | None = None
	probability: ProbabilityFloat | None = None


class BFScenarioResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id: str
	name: str
	scenario_type: BFScenarioType
	probability: float
	is_active: bool
	results: dict[str, Any]
	ran_at: datetime | None
	net_impact: Decimal | None
	net_impact_pct: Decimal | None
	created_at: datetime
	created_by: str


# ---------------------------------------------------------------------------
# DriverBasedAssumption
# ---------------------------------------------------------------------------

class BFDriverBasedAssumption(BFBase):
	"""A business driver that feeds into driver-based forecasting."""

	name: NonEmptyStr = Field(..., max_length=200)
	driver_type: BFDriverType = Field(...)
	value: Decimal = Field(...)
	unit: str | None = Field(default=None, max_length=50)
	period_start: date = Field(...)
	period_end: date = Field(...)
	growth_rate: Decimal | None = Field(default=None)
	seasonality_factors: list[float] = Field(default_factory=lambda: [1.0] * 12)
	source: str | None = Field(default=None)
	confidence: PercentageFloat = Field(default=80.0)
	scenario_id: str | None = Field(default=None)
	linked_accounts: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)

	@field_validator("seasonality_factors")
	@classmethod
	def _twelve_factors(cls, v: list[float]) -> list[float]:
		if len(v) != 12:
			raise ValueError("seasonality_factors must have 12 elements (one per month)")
		return v


class BFDriverAssumptionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")

	name: NonEmptyStr
	driver_type: BFDriverType
	value: Decimal
	unit: str | None = None
	period_start: date
	period_end: date
	growth_rate: Decimal | None = None
	seasonality_factors: list[float] = Field(default_factory=lambda: [1.0] * 12)
	source: str | None = None
	confidence: PercentageFloat = 80.0
	scenario_id: str | None = None
	linked_accounts: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


class BFDriverAssumptionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id: str
	name: str
	driver_type: BFDriverType
	value: Decimal
	unit: str | None
	period_start: date
	period_end: date
	growth_rate: Decimal | None
	confidence: float
	scenario_id: str | None
	created_at: datetime
	created_by: str


# ---------------------------------------------------------------------------
# Aggregation / report models
# ---------------------------------------------------------------------------

class BFBudgetSummary(BaseModel):
	"""Rollup summary across multiple budgets — e.g. for consolidation."""

	model_config = ConfigDict(extra="forbid")

	tenant_id: str
	period_start: date
	period_end: date
	budget_count: int
	total_revenue: Decimal
	total_expense: Decimal
	net_amount: Decimal
	approved_count: int
	draft_count: int
	currency_code: str = "USD"
	generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BFConsolidationResult(BaseModel):
	"""Output of budget_consolidation()."""

	model_config = ConfigDict(extra="forbid")

	tenant_id: str
	included_budget_ids: list[str]
	total_revenue: Decimal
	total_expense: Decimal
	net_amount: Decimal
	by_department: dict[str, Decimal] = Field(default_factory=dict)
	by_cost_center: dict[str, Decimal] = Field(default_factory=dict)
	currency_code: str
	generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BFRollingForecastResult(BaseModel):
	"""Output of rolling_forecast()."""

	model_config = ConfigDict(extra="forbid")

	forecast_id: str
	periods: int
	method: str
	projected_values: list[dict[str, Any]]  # [{period, value, lower, upper}]
	mape: float | None
	generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BFScenarioAnalysisResult(BaseModel):
	"""Output of scenario_analysis()."""

	model_config = ConfigDict(extra="forbid")

	base_net: Decimal
	scenarios: list[dict[str, Any]]
	expected_value: Decimal
	best_case: Decimal
	worst_case: Decimal
	generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BFSensitivityResult(BaseModel):
	"""Output of sensitivity_analysis()."""

	model_config = ConfigDict(extra="forbid")

	driver_name: str
	base_value: Decimal
	perturbations: list[dict[str, Any]]
	generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BFDashboardKPIs(BaseModel):
	"""KPIs for the dashboard screen."""

	model_config = ConfigDict(extra="forbid")

	tenant_id: str
	budget_count: int
	approved_budget_count: int
	draft_budget_count: int
	total_budget_amount: Decimal
	total_actual_amount: Decimal
	overall_variance_pct: Decimal
	forecast_count: int
	scenario_count: int
	pending_approvals: int
	material_variances: int
	generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
