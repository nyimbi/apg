"""
APG Budgeting & Forecasting - Data Models

Enterprise-grade data models for the APG Budgeting & Forecasting capability.
Implements CLAUDE.md standards with async Python, modern typing, and APG integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict
from pydantic import EmailStr, PositiveFloat, PositiveInt
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


def validate_fiscal_year(value: int) -> int:
	"""Validate fiscal year is reasonable."""
	current_year = datetime.now().year
	if value < (current_year - 10) or value > (current_year + 10):
		raise ValueError(_log_validation_error("fiscal_year", value, "must be within 10 years of current"))
	return value


def validate_non_empty_string(value: str) -> str:
	"""Validate non-empty strings."""
	if not value or not value.strip():
		raise ValueError(_log_validation_error("string", value, "cannot be empty"))
	return value.strip()


def validate_percentage(value: float) -> float:
	"""Validate percentage values (0-100)."""
	if value < 0 or value > 100:
		raise ValueError(_log_validation_error("percentage", value, "must be between 0 and 100"))
	return value


def validate_probability(value: float) -> float:
	"""Validate probability values (0-1)."""
	if value < 0 or value > 1:
		raise ValueError(_log_validation_error("probability", value, "must be between 0 and 1"))
	return value


# Type aliases with validation
PositiveAmount = Annotated[Decimal, AfterValidator(validate_positive_amount)]
CurrencyCode = Annotated[str, AfterValidator(validate_currency_code)]
FiscalYear = Annotated[int, AfterValidator(validate_fiscal_year)]
NonEmptyString = Annotated[str, AfterValidator(validate_non_empty_string)]
PercentageValue = Annotated[float, AfterValidator(validate_percentage)]
ProbabilityValue = Annotated[float, AfterValidator(validate_probability)]


# =============================================================================
# Enumerations
# =============================================================================

class BFBudgetType(str, Enum):
	"""Budget type enumeration."""
	ANNUAL = "annual"
	QUARTERLY = "quarterly"
	MONTHLY = "monthly"
	ROLLING = "rolling"
	PROJECT = "project"
	CAPITAL = "capital"
	OPERATIONAL = "operational"


class BFBudgetStatus(str, Enum):
	"""Budget status enumeration."""
	DRAFT = "draft"
	SUBMITTED = "submitted"
	UNDER_REVIEW = "under_review"
	APPROVED = "approved"
	ACTIVE = "active"
	LOCKED = "locked"
	CLOSED = "closed"
	CANCELLED = "cancelled"


class BFLineType(str, Enum):
	"""Budget line type enumeration."""
	REVENUE = "revenue"
	EXPENSE = "expense"
	CAPITAL = "capital"
	TRANSFER = "transfer"
	ALLOCATION = "allocation"
	CONTINGENCY = "contingency"


class BFForecastType(str, Enum):
	"""Forecast type enumeration."""
	REVENUE = "revenue"
	EXPENSE = "expense"
	CASH_FLOW = "cash_flow"
	DEMAND = "demand"
	INTEGRATED = "integrated"
	SCENARIO = "scenario"


class BFForecastMethod(str, Enum):
	"""Forecast method enumeration."""
	STATISTICAL = "statistical"
	ML = "ml"
	HYBRID = "hybrid"
	JUDGMENTAL = "judgmental"
	ENSEMBLE = "ensemble"


class BFForecastStatus(str, Enum):
	"""Forecast status enumeration."""
	DRAFT = "draft"
	GENERATING = "generating"
	COMPLETED = "completed"
	PUBLISHED = "published"
	ARCHIVED = "archived"
	FAILED = "failed"


class BFVarianceType(str, Enum):
	"""Variance type enumeration."""
	FAVORABLE = "favorable"
	UNFAVORABLE = "unfavorable"
	NEUTRAL = "neutral"


class BFSignificanceLevel(str, Enum):
	"""Significance level enumeration."""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	MINIMAL = "minimal"


class BFScenarioType(str, Enum):
	"""Scenario type enumeration."""
	BASE = "base"
	OPTIMISTIC = "optimistic"
	PESSIMISTIC = "pessimistic"
	STRESS = "stress"
	WHAT_IF = "what_if"
	MONTE_CARLO = "monte_carlo"


class BFApprovalStatus(str, Enum):
	"""Approval status enumeration."""
	PENDING = "pending"
	APPROVED = "approved"
	REJECTED = "rejected"
	REQUIRES_REVISION = "requires_revision"


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
# Budget Management Models
# =============================================================================

class BFBudgetLine(BaseModel):
	"""Budget line item with detailed allocations and APG integration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	line_number: PositiveInt = Field(...)
	line_description: NonEmptyString = Field(..., max_length=500)
	line_category: NonEmptyString = Field(..., max_length=100)
	line_type: BFLineType = Field(...)
	
	# Account mapping
	account_code: NonEmptyString = Field(..., max_length=50)
	account_category_id: Optional[str] = Field(None)
	gl_account: Optional[str] = Field(None, max_length=50)
	
	# Organizational allocation
	department_code: Optional[str] = Field(None, max_length=50)
	cost_center_code: Optional[str] = Field(None, max_length=50)
	project_code: Optional[str] = Field(None, max_length=50)
	activity_code: Optional[str] = Field(None, max_length=50)
	location_code: Optional[str] = Field(None, max_length=50)
	
	# Time period allocation
	period_start: date = Field(...)
	period_end: date = Field(...)
	allocation_method: str = Field(default="equal", max_length=50)  # equal, weighted, seasonal, custom
	
	# Budget amounts
	budgeted_amount: PositiveAmount = Field(...)
	forecasted_amount: Optional[PositiveAmount] = Field(None)
	committed_amount: PositiveAmount = Field(default=Decimal('0.00'))
	actual_amount: PositiveAmount = Field(default=Decimal('0.00'))
	variance_amount: Decimal = Field(default=Decimal('0.00'))
	variance_percent: Decimal = Field(default=Decimal('0.00'), decimal_places=4)
	
	# Monthly breakdown (for detailed planning)
	month_01_amount: PositiveAmount = Field(default=Decimal('0.00'))
	month_02_amount: PositiveAmount = Field(default=Decimal('0.00'))
	month_03_amount: PositiveAmount = Field(default=Decimal('0.00'))
	month_04_amount: PositiveAmount = Field(default=Decimal('0.00'))
	month_05_amount: PositiveAmount = Field(default=Decimal('0.00'))
	month_06_amount: PositiveAmount = Field(default=Decimal('0.00'))
	month_07_amount: PositiveAmount = Field(default=Decimal('0.00'))
	month_08_amount: PositiveAmount = Field(default=Decimal('0.00'))
	month_09_amount: PositiveAmount = Field(default=Decimal('0.00'))
	month_10_amount: PositiveAmount = Field(default=Decimal('0.00'))
	month_11_amount: PositiveAmount = Field(default=Decimal('0.00'))
	month_12_amount: PositiveAmount = Field(default=Decimal('0.00'))
	
	# Quarterly breakdown
	q1_amount: PositiveAmount = Field(default=Decimal('0.00'))
	q2_amount: PositiveAmount = Field(default=Decimal('0.00'))
	q3_amount: PositiveAmount = Field(default=Decimal('0.00'))
	q4_amount: PositiveAmount = Field(default=Decimal('0.00'))
	
	# Currency and exchange
	currency_code: CurrencyCode = Field(default="USD")
	exchange_rate: Decimal = Field(default=Decimal('1.000000'), gt=0, decimal_places=6)
	base_currency_amount: Optional[PositiveAmount] = Field(None)
	
	# Driver-based budgeting
	quantity_driver: Optional[str] = Field(None, max_length=100)
	unit_quantity: Optional[Decimal] = Field(None, decimal_places=4)
	unit_price: Optional[Decimal] = Field(None, decimal_places=4)
	price_escalation_percent: Decimal = Field(default=Decimal('0.00'), decimal_places=4)
	
	# Approval and workflow
	approval_status: BFApprovalStatus = Field(default=BFApprovalStatus.PENDING)
	approval_level: int = Field(default=0, ge=0)
	approved_by: Optional[str] = Field(None)
	approved_date: Optional[date] = Field(None)
	rejection_reason: Optional[str] = Field(None)
	
	# AI/ML insights
	ai_confidence_score: ProbabilityValue = Field(default=0.000)
	seasonality_factor: Decimal = Field(default=Decimal('1.0000'), decimal_places=4)
	trend_factor: Decimal = Field(default=Decimal('1.0000'), decimal_places=4)
	ai_adjustments: Dict[str, Any] = Field(default_factory=dict)
	
	# Comments and notes
	line_notes: Optional[str] = Field(None)
	business_justification: Optional[str] = Field(None)
	assumptions: Optional[str] = Field(None)

	@validator('period_end')
	def validate_period_end(cls, v: date, values: Dict[str, Any]) -> date:
		"""Validate period end is after period start."""
		period_start = values.get('period_start')
		if period_start and v < period_start:
			raise ValueError(_log_validation_error("period_end", v, "must be after period start"))
		return v

	@root_validator
	def validate_monthly_quarterly_totals(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate monthly amounts sum to budgeted amount."""
		budgeted = values.get('budgeted_amount', Decimal('0.00'))
		
		# Calculate monthly total
		monthly_fields = [f'month_{i:02d}_amount' for i in range(1, 13)]
		monthly_total = sum(values.get(field, Decimal('0.00')) for field in monthly_fields)
		
		# Allow small rounding differences
		if abs(monthly_total - budgeted) > Decimal('0.01') and monthly_total > 0:
			# Auto-adjust if close
			if abs(monthly_total - budgeted) < budgeted * Decimal('0.01'):  # Within 1%
				# Distribute difference across months
				difference = budgeted - monthly_total
				values['month_01_amount'] = values.get('month_01_amount', Decimal('0.00')) + difference
			else:
				raise ValueError(f"Monthly amounts ({monthly_total}) don't match budgeted amount ({budgeted})")
		
		# Calculate quarterly totals
		values['q1_amount'] = sum(values.get(f'month_{i:02d}_amount', Decimal('0.00')) for i in [1, 2, 3])
		values['q2_amount'] = sum(values.get(f'month_{i:02d}_amount', Decimal('0.00')) for i in [4, 5, 6])
		values['q3_amount'] = sum(values.get(f'month_{i:02d}_amount', Decimal('0.00')) for i in [7, 8, 9])
		values['q4_amount'] = sum(values.get(f'month_{i:02d}_amount', Decimal('0.00')) for i in [10, 11, 12])
		
		return values


class BFBudget(APGBaseModel):
	"""
	Comprehensive budget model for APG Budgeting & Forecasting.
	
	Integrates with:
	- document_management for budget documents
	- ai_orchestration for budget optimization
	- workflow_engine for approval workflows
	- real_time_collaboration for collaborative planning
	"""
	
	# Budget identification
	budget_name: NonEmptyString = Field(..., max_length=255)
	budget_code: Optional[str] = Field(None, max_length=50)
	budget_type: BFBudgetType = Field(...)
	fiscal_year: FiscalYear = Field(...)
	budget_period_start: date = Field(...)
	budget_period_end: date = Field(...)
	
	# Status and workflow
	status: BFBudgetStatus = Field(default=BFBudgetStatus.DRAFT)
	workflow_state: Optional[str] = Field(None, max_length=100)
	approval_level: int = Field(default=0, ge=0)
	requires_approval: bool = Field(default=True)
	
	# Financial configuration
	base_currency: CurrencyCode = Field(default="USD")
	budget_method: str = Field(default="zero_based", max_length=50)  # zero_based, incremental, activity_based
	planning_horizon_months: int = Field(default=12, ge=1, le=60)
	
	# Template and inheritance
	template_id: Optional[str] = Field(None)
	parent_budget_id: Optional[str] = Field(None)
	is_template: bool = Field(default=False)
	template_usage_count: int = Field(default=0, ge=0)
	
	# Organizational hierarchy
	department_code: Optional[str] = Field(None, max_length=50)
	cost_center_code: Optional[str] = Field(None, max_length=50)
	business_unit: Optional[str] = Field(None, max_length=100)
	region_code: Optional[str] = Field(None, max_length=50)
	
	# Performance tracking
	total_budget_amount: PositiveAmount = Field(default=Decimal('0.00'))
	total_committed_amount: PositiveAmount = Field(default=Decimal('0.00'))
	total_actual_amount: PositiveAmount = Field(default=Decimal('0.00'))
	variance_amount: Decimal = Field(default=Decimal('0.00'))
	variance_percent: Decimal = Field(default=Decimal('0.00'), decimal_places=4)
	
	# AI/ML insights
	ai_confidence_score: ProbabilityValue = Field(default=0.000)
	risk_assessment_score: ProbabilityValue = Field(default=0.000)
	ai_recommendations: List[str] = Field(default_factory=list)
	forecast_accuracy_score: Optional[ProbabilityValue] = Field(None)
	
	# Collaboration and communication
	collaboration_enabled: bool = Field(default=True)
	notification_settings: Dict[str, Any] = Field(default_factory=dict)
	last_activity_date: Optional[date] = Field(None)
	active_contributors: List[str] = Field(default_factory=list)
	
	# APG Integration fields
	document_folder_id: Optional[str] = Field(None, description="Document management folder")
	workflow_instance_id: Optional[str] = Field(None, description="Workflow engine instance")
	ai_job_id: Optional[str] = Field(None, description="AI orchestration job")

	@validator('budget_code')
	def validate_budget_code(cls, v: Optional[str]) -> Optional[str]:
		"""Validate budget code format."""
		if v and not v.replace('-', '').replace('_', '').isalnum():
			raise ValueError(_log_validation_error("budget_code", v, "must be alphanumeric"))
		return v.upper() if v else None

	@validator('budget_period_end')
	def validate_budget_period_end(cls, v: date, values: Dict[str, Any]) -> date:
		"""Validate budget period end is after start."""
		budget_period_start = values.get('budget_period_start')
		if budget_period_start and v <= budget_period_start:
			raise ValueError(_log_validation_error("budget_period_end", v, "must be after budget period start"))
		return v

	@root_validator
	def validate_budget_consistency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate budget data consistency."""
		# Validate template usage
		is_template = values.get('is_template', False)
		template_id = values.get('template_id')
		
		if is_template and template_id:
			raise ValueError("Budget cannot be both a template and use a template")
		
		# Validate parent budget relationship
		parent_budget_id = values.get('parent_budget_id')
		if parent_budget_id == values.get('id'):
			raise ValueError("Budget cannot be its own parent")
		
		# Validate fiscal year consistency
		fiscal_year = values.get('fiscal_year')
		budget_period_start = values.get('budget_period_start')
		
		if fiscal_year and budget_period_start:
			if budget_period_start.year != fiscal_year:
				# Allow fiscal year to start in previous calendar year
				if not (budget_period_start.year == fiscal_year - 1 and budget_period_start.month >= 7):
					raise ValueError("Budget period start must align with fiscal year")
		
		return values


# =============================================================================
# Forecast Management Models
# =============================================================================

class BFForecastDataPoint(BaseModel):
	"""Individual forecast data point with detailed breakdown."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	period_date: date = Field(...)
	period_type: str = Field(..., max_length=20)  # daily, weekly, monthly, quarterly, annual
	period_sequence: int = Field(..., ge=1)
	fiscal_year: Optional[int] = Field(None)
	fiscal_quarter: Optional[int] = Field(None, ge=1, le=4)
	fiscal_month: Optional[int] = Field(None, ge=1, le=12)
	
	# Forecast values
	forecasted_value: Decimal = Field(...)
	confidence_lower: Optional[Decimal] = Field(None)
	confidence_upper: Optional[Decimal] = Field(None)
	actual_value: Optional[Decimal] = Field(None)
	variance_amount: Optional[Decimal] = Field(None)
	variance_percent: Optional[Decimal] = Field(None, decimal_places=4)
	
	# Decomposition components
	trend_component: Optional[Decimal] = Field(None)
	seasonal_component: Optional[Decimal] = Field(None)
	cyclical_component: Optional[Decimal] = Field(None)
	irregular_component: Optional[Decimal] = Field(None)
	external_factor_impact: Optional[Decimal] = Field(None)
	
	# Model insights
	prediction_strength: Optional[ProbabilityValue] = Field(None)
	volatility_score: Optional[ProbabilityValue] = Field(None)
	anomaly_score: Optional[ProbabilityValue] = Field(None)
	feature_importance: Dict[str, float] = Field(default_factory=dict)
	
	# Business drivers
	volume_driver: Optional[Decimal] = Field(None, decimal_places=4)
	price_driver: Optional[Decimal] = Field(None, decimal_places=4)
	mix_driver: Optional[Decimal] = Field(None, decimal_places=4)
	external_drivers: Dict[str, Any] = Field(default_factory=dict)
	
	# Currency handling
	currency_code: CurrencyCode = Field(default="USD")
	exchange_rate: Decimal = Field(default=Decimal('1.000000'), gt=0, decimal_places=6)
	base_currency_value: Optional[Decimal] = Field(None)
	
	# Notes and assumptions
	period_notes: Optional[str] = Field(None)
	assumptions: Optional[str] = Field(None)
	risk_factors: Optional[str] = Field(None)

	@validator('confidence_upper')
	def validate_confidence_interval(cls, v: Optional[Decimal], values: Dict[str, Any]) -> Optional[Decimal]:
		"""Validate confidence interval consistency."""
		if v is not None:
			confidence_lower = values.get('confidence_lower')
			if confidence_lower is not None and v < confidence_lower:
				raise ValueError(_log_validation_error("confidence_upper", v, "must be >= confidence_lower"))
		return v


class BFForecast(APGBaseModel):
	"""
	Comprehensive forecast model for APG Budgeting & Forecasting.
	
	Integrates with:
	- time_series_analytics for advanced forecasting
	- ai_orchestration for ML model management
	- federated_learning for collaborative forecasting
	"""
	
	# Forecast identification
	forecast_name: NonEmptyString = Field(..., max_length=255)
	forecast_code: Optional[str] = Field(None, max_length=50)
	forecast_type: BFForecastType = Field(...)
	forecast_method: BFForecastMethod = Field(...)
	
	# Time horizon and frequency
	forecast_horizon_months: int = Field(..., ge=1, le=60)
	forecast_frequency: str = Field(default="monthly", max_length=20)  # daily, weekly, monthly, quarterly
	base_period_start: date = Field(...)
	base_period_end: date = Field(...)
	forecast_period_start: date = Field(...)
	forecast_period_end: date = Field(...)
	
	# Model and algorithm configuration
	algorithm_type: Optional[str] = Field(None, max_length=50)  # arima, exponential_smoothing, neural_network, ensemble
	model_version: Optional[str] = Field(None, max_length=50)
	model_parameters: Dict[str, Any] = Field(default_factory=dict)
	feature_selection: Dict[str, Any] = Field(default_factory=dict)
	
	# Data sources and inputs
	data_sources: List[str] = Field(default_factory=list)
	input_variables: List[str] = Field(default_factory=list)
	external_factors: List[str] = Field(default_factory=list)
	historical_months_used: int = Field(default=24, ge=3, le=120)
	
	# Accuracy and confidence
	accuracy_score: Optional[ProbabilityValue] = Field(None)
	confidence_level: ProbabilityValue = Field(default=0.950)
	confidence_interval_lower: Optional[Decimal] = Field(None)
	confidence_interval_upper: Optional[Decimal] = Field(None)
	mae_score: Optional[Decimal] = Field(None, decimal_places=4)  # Mean Absolute Error
	mape_score: Optional[Decimal] = Field(None, decimal_places=4)  # Mean Absolute Percentage Error
	rmse_score: Optional[Decimal] = Field(None, decimal_places=4)  # Root Mean Square Error
	
	# Scenario analysis
	scenario_type: BFScenarioType = Field(default=BFScenarioType.BASE)
	probability_weight: ProbabilityValue = Field(default=1.000)
	scenario_assumptions: Optional[str] = Field(None)
	sensitivity_analysis: Dict[str, Any] = Field(default_factory=dict)
	
	# Business context
	department_code: Optional[str] = Field(None, max_length=50)
	business_unit: Optional[str] = Field(None, max_length=100)
	product_category: Optional[str] = Field(None, max_length=100)
	market_segment: Optional[str] = Field(None, max_length=100)
	geographic_region: Optional[str] = Field(None, max_length=100)
	
	# Status and lifecycle
	status: BFForecastStatus = Field(default=BFForecastStatus.DRAFT)
	generation_status: Optional[str] = Field(None, max_length=50)  # pending, running, completed, failed
	last_generation_date: Optional[datetime] = Field(None)
	next_generation_date: Optional[datetime] = Field(None)
	auto_generation_enabled: bool = Field(default=False)
	
	# Performance tracking
	forecast_value: Optional[Decimal] = Field(None)
	actual_value: Optional[Decimal] = Field(None)
	variance_amount: Optional[Decimal] = Field(None)
	variance_percent: Optional[Decimal] = Field(None, decimal_places=4)
	accuracy_trend: Optional[str] = Field(None, max_length=20)  # improving, stable, declining
	
	# APG Integration fields
	ai_job_id: Optional[str] = Field(None, description="AI orchestration job ID")
	time_series_job_id: Optional[str] = Field(None, description="Time series analytics job")
	federated_learning_session_id: Optional[str] = Field(None, description="Federated learning session")
	
	# Approval and review
	reviewed_by: Optional[str] = Field(None)
	review_date: Optional[date] = Field(None)
	review_notes: Optional[str] = Field(None)
	approved_for_planning: bool = Field(default=False)

	@validator('forecast_code')
	def validate_forecast_code(cls, v: Optional[str]) -> Optional[str]:
		"""Validate forecast code format."""
		if v and not v.replace('-', '').replace('_', '').isalnum():
			raise ValueError(_log_validation_error("forecast_code", v, "must be alphanumeric"))
		return v.upper() if v else None

	@root_validator
	def validate_forecast_periods(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate forecast period relationships."""
		base_start = values.get('base_period_start')
		base_end = values.get('base_period_end')
		forecast_start = values.get('forecast_period_start')
		forecast_end = values.get('forecast_period_end')
		
		if all([base_start, base_end, forecast_start, forecast_end]):
			# Base period validation
			if base_end <= base_start:
				raise ValueError("Base period end must be after start")
			
			# Forecast period validation
			if forecast_end <= forecast_start:
				raise ValueError("Forecast period end must be after start")
			
			# Relationship validation
			if forecast_start < base_end:
				raise ValueError("Forecast period must start after base period ends")
		
		# Validate confidence interval
		conf_lower = values.get('confidence_interval_lower')
		conf_upper = values.get('confidence_interval_upper')
		
		if conf_lower is not None and conf_upper is not None:
			if conf_upper < conf_lower:
				raise ValueError("Confidence interval upper must be >= lower")
		
		return values


# =============================================================================
# Variance Analysis Models
# =============================================================================

class BFVarianceAnalysis(APGBaseModel):
	"""
	Comprehensive variance analysis for budget vs actual performance.
	
	Integrates with:
	- ai_orchestration for intelligent variance explanation
	- notification_engine for variance alerts
	- workflow_engine for investigation workflows
	"""
	
	# Analysis identification
	analysis_name: NonEmptyString = Field(..., max_length=255)
	analysis_type: str = Field(..., max_length=50)  # budget_vs_actual, forecast_vs_actual, period_comparison
	analysis_period_start: date = Field(...)
	analysis_period_end: date = Field(...)
	comparison_period_start: Optional[date] = Field(None)
	comparison_period_end: Optional[date] = Field(None)
	
	# Subject of analysis
	budget_id: Optional[str] = Field(None)
	forecast_id: Optional[str] = Field(None)
	department_code: Optional[str] = Field(None, max_length=50)
	account_category: Optional[str] = Field(None, max_length=100)
	analysis_scope: str = Field(default="detailed", max_length=50)  # summary, detailed, line_item
	
	# Variance calculations
	baseline_amount: Decimal = Field(...)
	actual_amount: Decimal = Field(...)
	variance_amount: Decimal = Field(...)
	variance_percent: Decimal = Field(..., decimal_places=4)
	absolute_variance: PositiveAmount = Field(...)
	
	# Variance classification
	variance_type: BFVarianceType = Field(...)
	significance_level: BFSignificanceLevel = Field(...)
	variance_threshold_exceeded: bool = Field(default=False)
	requires_investigation: bool = Field(default=False)
	
	# Root cause analysis
	primary_cause: Optional[str] = Field(None, max_length=100)
	contributing_factors: List[str] = Field(default_factory=list)
	root_cause_category: Optional[str] = Field(None, max_length=50)  # volume, price, mix, timing, operational, external
	impact_assessment: Optional[str] = Field(None)
	
	# AI-powered insights
	ai_explanation: Optional[str] = Field(None)
	ai_confidence_score: Optional[ProbabilityValue] = Field(None)
	anomaly_detected: bool = Field(default=False)
	pattern_analysis: Dict[str, Any] = Field(default_factory=dict)
	correlation_factors: Dict[str, Any] = Field(default_factory=dict)
	
	# Corrective actions
	recommended_actions: List[str] = Field(default_factory=list)
	action_priority: str = Field(default="medium", max_length=20)
	estimated_impact: Optional[Decimal] = Field(None)
	action_timeline: Optional[str] = Field(None, max_length=50)
	responsible_party: Optional[str] = Field(None, max_length=100)
	
	# Investigation tracking
	investigation_status: str = Field(default="pending", max_length=50)
	investigated_by: Optional[str] = Field(None)
	investigation_date: Optional[date] = Field(None)
	investigation_notes: Optional[str] = Field(None)
	resolution_status: str = Field(default="open", max_length=50)
	
	# Performance metrics
	analysis_accuracy: Optional[ProbabilityValue] = Field(None)
	prediction_quality: Optional[ProbabilityValue] = Field(None)
	time_to_detection_days: Optional[int] = Field(None, ge=0)
	resolution_time_days: Optional[int] = Field(None, ge=0)
	
	# APG Integration fields
	ai_analysis_job_id: Optional[str] = Field(None)
	notification_sent: bool = Field(default=False)
	workflow_triggered: bool = Field(default=False)

	@validator('analysis_period_end')
	def validate_analysis_period_end(cls, v: date, values: Dict[str, Any]) -> date:
		"""Validate analysis period end is after start."""
		analysis_period_start = values.get('analysis_period_start')
		if analysis_period_start and v < analysis_period_start:
			raise ValueError(_log_validation_error("analysis_period_end", v, "must be after analysis period start"))
		return v

	@root_validator
	def validate_variance_calculations(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate variance calculation consistency."""
		baseline = values.get('baseline_amount', Decimal('0'))
		actual = values.get('actual_amount', Decimal('0'))
		variance = values.get('variance_amount', Decimal('0'))
		variance_percent = values.get('variance_percent', Decimal('0'))
		
		# Calculate expected variance
		expected_variance = actual - baseline
		
		if abs(variance - expected_variance) > Decimal('0.01'):
			raise ValueError(f"Variance amount ({variance}) doesn't match calculation ({expected_variance})")
		
		# Calculate expected variance percentage
		if baseline != 0:
			expected_variance_percent = (variance / baseline) * 100
			if abs(variance_percent - expected_variance_percent) > Decimal('0.01'):
				raise ValueError(f"Variance percent ({variance_percent}) doesn't match calculation ({expected_variance_percent})")
		
		# Set absolute variance
		values['absolute_variance'] = abs(variance)
		
		return values


# =============================================================================
# Scenario Planning Models
# =============================================================================

class BFScenario(APGBaseModel):
	"""
	Comprehensive scenario planning model for what-if analysis.
	
	Integrates with:
	- ai_orchestration for Monte Carlo simulation
	- business_intelligence for scenario comparison
	- decision_support for strategic planning
	"""
	
	# Scenario identification
	scenario_name: NonEmptyString = Field(..., max_length=255)
	scenario_description: Optional[str] = Field(None)
	scenario_type: BFScenarioType = Field(...)
	scenario_category: Optional[str] = Field(None, max_length=50)  # market, operational, financial, strategic
	
	# Scenario parameters
	probability_weight: ProbabilityValue = Field(default=0.333)
	time_horizon_months: int = Field(..., ge=1, le=60)
	scenario_start_date: date = Field(...)
	scenario_end_date: date = Field(...)
	
	# Base scenario reference
	base_budget_id: Optional[str] = Field(None)
	base_forecast_id: Optional[str] = Field(None)
	comparison_baseline: str = Field(default="current_budget", max_length=50)
	
	# Scenario assumptions
	key_assumptions: List[str] = Field(default_factory=list)
	variable_changes: Dict[str, float] = Field(default_factory=dict)  # {"revenue_growth": 0.15, "cost_inflation": 0.08}
	external_factors: Dict[str, Any] = Field(default_factory=dict)
	market_conditions: Optional[str] = Field(None)
	
	# Financial impact
	total_revenue_impact: Decimal = Field(default=Decimal('0.00'))
	total_expense_impact: Decimal = Field(default=Decimal('0.00'))
	net_income_impact: Decimal = Field(default=Decimal('0.00'))
	cash_flow_impact: Decimal = Field(default=Decimal('0.00'))
	
	# Risk assessment
	risk_level: str = Field(default="medium", max_length=20)  # low, medium, high, extreme
	downside_risk: Optional[Decimal] = Field(None)
	upside_potential: Optional[Decimal] = Field(None)
	value_at_risk: Optional[Decimal] = Field(None)
	
	# Sensitivity analysis
	sensitivity_variables: List[str] = Field(default_factory=list)
	elasticity_factors: Dict[str, float] = Field(default_factory=dict)
	break_even_points: Dict[str, Any] = Field(default_factory=dict)
	
	# Monte Carlo simulation
	simulation_enabled: bool = Field(default=False)
	simulation_iterations: int = Field(default=1000, ge=100, le=100000)
	confidence_intervals: Dict[str, Any] = Field(default_factory=dict)
	distribution_parameters: Dict[str, Any] = Field(default_factory=dict)
	
	# Decision support
	strategic_implications: Optional[str] = Field(None)
	recommended_decisions: List[str] = Field(default_factory=list)
	contingency_plans: List[str] = Field(default_factory=list)
	monitoring_indicators: List[str] = Field(default_factory=list)
	
	# Modeling and calculation
	calculation_method: str = Field(default="analytical", max_length=50)  # analytical, simulation, hybrid
	model_complexity: str = Field(default="medium", max_length=20)
	last_calculation_date: Optional[datetime] = Field(None)
	calculation_duration_seconds: Optional[int] = Field(None, ge=0)
	
	# Collaboration and review
	scenario_owner: Optional[str] = Field(None)
	review_participants: List[str] = Field(default_factory=list)
	last_review_date: Optional[date] = Field(None)
	review_status: str = Field(default="draft", max_length=50)
	approval_required: bool = Field(default=False)
	approved_by: Optional[str] = Field(None)
	approval_date: Optional[date] = Field(None)
	
	# APG Integration fields
	simulation_job_id: Optional[str] = Field(None)
	ai_modeling_job_id: Optional[str] = Field(None)
	document_folder_id: Optional[str] = Field(None)

	@validator('scenario_end_date')
	def validate_scenario_end_date(cls, v: date, values: Dict[str, Any]) -> date:
		"""Validate scenario end date is after start date."""
		scenario_start_date = values.get('scenario_start_date')
		if scenario_start_date and v <= scenario_start_date:
			raise ValueError(_log_validation_error("scenario_end_date", v, "must be after scenario start date"))
		return v

	@root_validator
	def validate_scenario_consistency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate scenario data consistency."""
		# Validate probability weights for multiple scenarios
		probability = values.get('probability_weight', 1.0)
		scenario_type = values.get('scenario_type')
		
		# Base scenarios should have higher probability
		if scenario_type == BFScenarioType.BASE and probability < 0.3:
			raise ValueError("Base scenario should have probability >= 0.3")
		
		# Validate simulation configuration
		simulation_enabled = values.get('simulation_enabled', False)
		if simulation_enabled:
			if not values.get('distribution_parameters'):
				raise ValueError("Distribution parameters required for simulation")
		
		return values


# =============================================================================
# Model Registration and Export
# =============================================================================

# Export all models for use by other modules
__all__ = [
	# Base Models
	'APGBaseModel',
	
	# Budget Models
	'BFBudget', 'BFBudgetLine',
	
	# Forecast Models
	'BFForecast', 'BFForecastDataPoint',
	
	# Analysis Models
	'BFVarianceAnalysis', 'BFScenario',
	
	# Enumerations
	'BFBudgetType', 'BFBudgetStatus', 'BFLineType',
	'BFForecastType', 'BFForecastMethod', 'BFForecastStatus',
	'BFVarianceType', 'BFSignificanceLevel', 'BFScenarioType',
	'BFApprovalStatus',
	
	# Type Aliases
	'PositiveAmount', 'CurrencyCode', 'FiscalYear', 'NonEmptyString',
	'PercentageValue', 'ProbabilityValue'
]


def _log_model_summary() -> str:
	"""Log summary of registered models."""
	model_count = len([name for name in __all__ if name.startswith('BF') and not name.endswith('Type') and not name.endswith('Status')])
	return f"APG Budgeting & Forecasting models loaded: {model_count} models, {len(__all__)} total exports"
