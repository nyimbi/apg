"""
APG Accounts Receivable - AI Cash Flow Forecasting
Predictive cash flow forecasting using APG time_series_analytics capability

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator, AfterValidator
from uuid_extensions import uuid7str

from apg.core.base import APGServiceBase
from apg.core.models import APGBaseModel
from apg.time_series_analytics import TimeSeriesAnalyticsService
from apg.ai_orchestration import AIOrchestrationService
from apg.notification_engine import NotificationService
from apg.audit_compliance import AuditLogger

from .models import (
	ARCustomer, ARInvoice, ARPayment, ARCollectionActivity,
	ARInvoiceStatus, ARPaymentStatus, ARCustomerStatus
)


# =============================================================================
# Enums and Constants
# =============================================================================

class ForecastPeriodType(str, Enum):
	"""Cash flow forecast period types."""
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"


class CashFlowComponent(str, Enum):
	"""Cash flow components for detailed analysis."""
	INVOICE_COLLECTIONS = "invoice_collections"
	PAYMENT_RECEIPTS = "payment_receipts"
	OVERDUE_COLLECTIONS = "overdue_collections"
	NEW_INVOICES = "new_invoices"
	DISPUTES_IMPACT = "disputes_impact"
	SEASONAL_ADJUSTMENTS = "seasonal_adjustments"


class ForecastConfidenceLevel(str, Enum):
	"""Forecast confidence levels."""
	HIGH = "high"			# >90% confidence
	MEDIUM = "medium"		# 70-90% confidence
	LOW = "low"				# 50-70% confidence
	INSUFFICIENT = "insufficient"  # <50% confidence


class ScenarioType(str, Enum):
	"""Cash flow forecast scenarios."""
	OPTIMISTIC = "optimistic"	# Best case scenario
	REALISTIC = "realistic"		# Most likely scenario
	PESSIMISTIC = "pessimistic"	# Worst case scenario
	CUSTOM = "custom"			# User-defined parameters


# =============================================================================
# Data Models
# =============================================================================

class CashFlowForecastInput(APGBaseModel):
	"""Input parameters for cash flow forecasting."""
	
	tenant_id: str = Field(..., description="APG tenant identifier")
	forecast_start_date: date = Field(..., description="Forecast start date")
	forecast_end_date: date = Field(..., description="Forecast end date")
	forecast_period: ForecastPeriodType = Field(default=ForecastPeriodType.DAILY)
	
	# Customer and invoice filters
	customer_ids: Optional[List[str]] = Field(default=None, description="Specific customers to include")
	include_overdue_only: bool = Field(default=False, description="Focus on overdue amounts only")
	min_invoice_amount: Optional[Decimal] = Field(default=None, description="Minimum invoice amount filter")
	max_invoice_amount: Optional[Decimal] = Field(default=None, description="Maximum invoice amount filter")
	
	# Scenario parameters
	scenario_type: ScenarioType = Field(default=ScenarioType.REALISTIC)
	collection_rate_adjustment: Optional[float] = Field(default=None, description="Custom collection rate (0.0-1.0)")
	payment_delay_adjustment: Optional[int] = Field(default=None, description="Payment delay adjustment in days")
	
	# Analysis options
	include_seasonal_trends: bool = Field(default=True, description="Include seasonal analysis")
	include_external_factors: bool = Field(default=True, description="Include economic indicators")
	confidence_level: float = Field(default=0.95, description="Statistical confidence level")
	
	@validator('forecast_end_date')
	def validate_forecast_dates(cls, v, values):
		"""Validate forecast date range."""
		if 'forecast_start_date' in values and v <= values['forecast_start_date']:
			raise ValueError("Forecast end date must be after start date")
		
		# Limit forecast horizon to 1 year
		if 'forecast_start_date' in values:
			max_date = values['forecast_start_date'] + timedelta(days=365)
			if v > max_date:
				raise ValueError("Forecast horizon cannot exceed 365 days")
		
		return v
	
	@validator('collection_rate_adjustment')
	def validate_collection_rate(cls, v):
		"""Validate collection rate adjustment."""
		if v is not None and (v < 0.0 or v > 1.0):
			raise ValueError("Collection rate adjustment must be between 0.0 and 1.0")
		return v


class CashFlowDataPoint(APGBaseModel):
	"""Individual cash flow forecast data point."""
	
	forecast_date: date = Field(..., description="Forecast date")
	forecast_period: ForecastPeriodType = Field(..., description="Period type")
	
	# Cash flow components
	expected_collections: Decimal = Field(default=Decimal('0.00'), description="Expected collections amount")
	invoice_receipts: Decimal = Field(default=Decimal('0.00'), description="New invoice receipts")
	overdue_collections: Decimal = Field(default=Decimal('0.00'), description="Overdue collections")
	total_cash_flow: Decimal = Field(default=Decimal('0.00'), description="Total cash flow")
	
	# Statistical measures
	confidence_interval_lower: Decimal = Field(default=Decimal('0.00'))
	confidence_interval_upper: Decimal = Field(default=Decimal('0.00'))
	standard_deviation: Decimal = Field(default=Decimal('0.00'))
	
	# Contributing factors
	customer_count: int = Field(default=0, description="Number of contributing customers")
	invoice_count: int = Field(default=0, description="Number of contributing invoices")
	average_payment_days: float = Field(default=0.0, description="Average payment days")
	
	# Scenario adjustments
	seasonal_adjustment: Decimal = Field(default=Decimal('0.00'))
	economic_adjustment: Decimal = Field(default=Decimal('0.00'))
	collection_efficiency_factor: float = Field(default=1.0)


class CashFlowForecastSummary(APGBaseModel):
	"""Summary of cash flow forecast results."""
	
	forecast_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="APG tenant identifier")
	forecast_period_start: date = Field(..., description="Forecast start date")
	forecast_period_end: date = Field(..., description="Forecast end date")
	
	# Summary metrics
	total_forecasted_collections: Decimal = Field(default=Decimal('0.00'))
	average_daily_collections: Decimal = Field(default=Decimal('0.00'))
	peak_collection_day: date = Field(...)
	peak_collection_amount: Decimal = Field(default=Decimal('0.00'))
	
	# Accuracy and confidence
	overall_confidence_level: ForecastConfidenceLevel = Field(...)
	model_accuracy_score: float = Field(..., description="Historical accuracy (0.0-1.0)")
	confidence_score: float = Field(..., description="Forecast confidence (0.0-1.0)")
	
	# Risk analysis
	forecast_volatility: float = Field(..., description="Coefficient of variation")
	downside_risk_amount: Decimal = Field(default=Decimal('0.00'), description="Potential loss amount")
	upside_potential_amount: Decimal = Field(default=Decimal('0.00'), description="Potential gain amount")
	
	# Model details
	model_version: str = Field(default="ar_cashflow_forecast_v1.0")
	data_points_used: int = Field(default=0)
	forecast_generated_at: datetime = Field(default_factory=datetime.utcnow)
	next_update_due: datetime = Field(...)


class CashFlowScenarioComparison(APGBaseModel):
	"""Comparison of different cash flow scenarios."""
	
	comparison_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Scenario forecasts
	optimistic_forecast: List[CashFlowDataPoint] = Field(default_factory=list)
	realistic_forecast: List[CashFlowDataPoint] = Field(default_factory=list)
	pessimistic_forecast: List[CashFlowDataPoint] = Field(default_factory=list)
	
	# Scenario summaries
	optimistic_total: Decimal = Field(default=Decimal('0.00'))
	realistic_total: Decimal = Field(default=Decimal('0.00'))
	pessimistic_total: Decimal = Field(default=Decimal('0.00'))
	
	# Risk metrics
	scenario_spread: Decimal = Field(default=Decimal('0.00'), description="Difference between best and worst case")
	risk_adjusted_forecast: Decimal = Field(default=Decimal('0.00'), description="Risk-weighted average")
	probability_weights: Dict[str, float] = Field(default_factory=dict)
	
	# Recommendations
	recommended_scenario: ScenarioType = Field(...)
	risk_mitigation_actions: List[str] = Field(default_factory=list)
	liquidity_recommendations: List[str] = Field(default_factory=list)


class CashFlowForecastingConfig(APGBaseModel):
	"""Configuration for cash flow forecasting system."""
	
	# Time series analytics integration
	time_series_endpoint: str = Field(..., description="APG time series analytics endpoint")
	forecasting_model_name: str = Field(default="ar_cashflow_predictor_v1")
	model_version: str = Field(default="1.0.0")
	
	# Forecasting parameters
	default_forecast_horizon_days: int = Field(default=90)
	max_forecast_horizon_days: int = Field(default=365)
	min_historical_data_points: int = Field(default=30)
	accuracy_threshold: float = Field(default=0.90, description="Target accuracy for 30-day forecasts")
	
	# Model tuning
	seasonal_decomposition_periods: List[int] = Field(default_factory=lambda: [7, 30, 365])  # Weekly, monthly, yearly
	trend_detection_sensitivity: float = Field(default=0.05)
	outlier_detection_threshold: float = Field(default=2.5)  # Standard deviations
	
	# Scenario modeling
	default_scenario_adjustments: Dict[str, Dict[str, float]] = Field(
		default_factory=lambda: {
			"optimistic": {"collection_rate": 0.95, "payment_delay": -3},
			"realistic": {"collection_rate": 0.85, "payment_delay": 0},
			"pessimistic": {"collection_rate": 0.70, "payment_delay": 7}
		}
	)
	
	# Performance targets
	forecast_accuracy_targets: Dict[str, float] = Field(
		default_factory=lambda: {
			"30_day": 0.90,
			"60_day": 0.85,
			"90_day": 0.80
		}
	)
	
	# Update frequencies
	forecast_update_intervals: Dict[str, int] = Field(
		default_factory=lambda: {
			"high_priority": 1,  # Daily
			"normal": 7,  # Weekly
			"low_priority": 30  # Monthly
		}
	)


# =============================================================================
# APG Cash Flow Forecasting Service
# =============================================================================

class APGCashFlowForecastingService(APGServiceBase):
	"""APG-integrated cash flow forecasting service using time series analytics."""
	
	def __init__(self, tenant_id: str, user_id: str, config: CashFlowForecastingConfig):
		super().__init__(tenant_id, user_id)
		self.config = config
		self.time_series_service = None
		self.ai_orchestration = None
		self.notification_service = None
		self.audit_logger = None
	
	async def _initialize_services(self):
		"""Initialize APG service dependencies."""
		if not self.time_series_service:
			self.time_series_service = await TimeSeriesAnalyticsService.create(
				tenant_id=self.tenant_id,
				endpoint=self.config.time_series_endpoint
			)
		
		if not self.ai_orchestration:
			self.ai_orchestration = await AIOrchestrationService.create(
				tenant_id=self.tenant_id
			)
		
		if not self.notification_service:
			self.notification_service = await NotificationService.create(
				tenant_id=self.tenant_id
			)
		
		if not self.audit_logger:
			self.audit_logger = AuditLogger(
				tenant_id=self.tenant_id,
				service_name="ar_cashflow_forecasting"
			)
	
	async def generate_cash_flow_forecast(
		self,
		forecast_input: CashFlowForecastInput
	) -> Tuple[List[CashFlowDataPoint], CashFlowForecastSummary]:
		"""Generate AI-powered cash flow forecast."""
		
		await self._initialize_services()
		
		# Validate permissions
		await self._validate_permissions("cash_flow_forecast")
		
		try:
			# 1. Extract historical cash flow data
			historical_data = await self._extract_historical_cash_flow_data(forecast_input)
			
			# 2. Prepare time series data for APG analytics
			time_series_data = await self._prepare_time_series_data(historical_data, forecast_input)
			
			# 3. Generate forecast using APG time series analytics
			forecast_results = await self._generate_time_series_forecast(time_series_data, forecast_input)
			
			# 4. Apply scenario adjustments
			adjusted_forecast = await self._apply_scenario_adjustments(forecast_results, forecast_input)
			
			# 5. Calculate confidence intervals and statistics
			final_forecast = await self._calculate_forecast_statistics(adjusted_forecast, forecast_input)
			
			# 6. Generate forecast summary
			summary = await self._generate_forecast_summary(final_forecast, forecast_input)
			
			# 7. Log forecast generation
			await self._log_forecast_generation(forecast_input, summary)
			
			return final_forecast, summary
			
		except Exception as e:
			await self.audit_logger.log_error(
				action="generate_cash_flow_forecast",
				error=str(e),
				context={"forecast_input": forecast_input.dict()}
			)
			raise
	
	async def generate_scenario_comparison(
		self,
		base_forecast_input: CashFlowForecastInput
	) -> CashFlowScenarioComparison:
		"""Generate comparison of optimistic, realistic, and pessimistic scenarios."""
		
		await self._initialize_services()
		
		scenarios = {}
		
		# Generate forecasts for all scenarios
		for scenario in [ScenarioType.OPTIMISTIC, ScenarioType.REALISTIC, ScenarioType.PESSIMISTIC]:
			scenario_input = base_forecast_input.copy(deep=True)
			scenario_input.scenario_type = scenario
			
			# Apply scenario-specific adjustments
			if scenario == ScenarioType.OPTIMISTIC:
				scenario_input.collection_rate_adjustment = self.config.default_scenario_adjustments["optimistic"]["collection_rate"]
				scenario_input.payment_delay_adjustment = int(self.config.default_scenario_adjustments["optimistic"]["payment_delay"])
			elif scenario == ScenarioType.PESSIMISTIC:
				scenario_input.collection_rate_adjustment = self.config.default_scenario_adjustments["pessimistic"]["collection_rate"]
				scenario_input.payment_delay_adjustment = int(self.config.default_scenario_adjustments["pessimistic"]["payment_delay"])
			
			forecast, _ = await self.generate_cash_flow_forecast(scenario_input)
			scenarios[scenario.value] = forecast
		
		# Create scenario comparison
		comparison = CashFlowScenarioComparison(
			tenant_id=self.tenant_id,
			optimistic_forecast=scenarios["optimistic"],
			realistic_forecast=scenarios["realistic"],
			pessimistic_forecast=scenarios["pessimistic"]
		)
		
		# Calculate totals and metrics
		comparison.optimistic_total = sum(dp.total_cash_flow for dp in scenarios["optimistic"])
		comparison.realistic_total = sum(dp.total_cash_flow for dp in scenarios["realistic"])
		comparison.pessimistic_total = sum(dp.total_cash_flow for dp in scenarios["pessimistic"])
		
		comparison.scenario_spread = comparison.optimistic_total - comparison.pessimistic_total
		comparison.risk_adjusted_forecast = (
			comparison.optimistic_total * 0.25 +
			comparison.realistic_total * 0.50 +
			comparison.pessimistic_total * 0.25
		)
		
		comparison.probability_weights = {
			"optimistic": 0.25,
			"realistic": 0.50,
			"pessimistic": 0.25
		}
		
		# Determine recommended scenario
		comparison.recommended_scenario = ScenarioType.REALISTIC
		
		# Generate recommendations
		comparison.risk_mitigation_actions = await self._generate_risk_mitigation_actions(comparison)
		comparison.liquidity_recommendations = await self._generate_liquidity_recommendations(comparison)
		
		return comparison
	
	async def monitor_forecast_accuracy(
		self,
		forecast_id: str,
		actual_cash_flows: List[CashFlowDataPoint]
	) -> Dict[str, Any]:
		"""Monitor and analyze forecast accuracy against actual results."""
		
		await self._initialize_services()
		
		# Retrieve original forecast
		original_forecast = await self._retrieve_forecast_by_id(forecast_id)
		
		if not original_forecast:
			raise ValueError(f"Forecast {forecast_id} not found")
		
		# Calculate accuracy metrics
		accuracy_metrics = await self._calculate_accuracy_metrics(original_forecast, actual_cash_flows)
		
		# Update model performance tracking
		await self._update_model_performance_tracking(accuracy_metrics)
		
		# Generate accuracy report
		accuracy_report = {
			"forecast_id": forecast_id,
			"measurement_period": {
				"start_date": min(dp.forecast_date for dp in actual_cash_flows),
				"end_date": max(dp.forecast_date for dp in actual_cash_flows)
			},
			"accuracy_metrics": accuracy_metrics,
			"performance_vs_targets": await self._compare_to_accuracy_targets(accuracy_metrics),
			"recommendations": await self._generate_accuracy_improvement_recommendations(accuracy_metrics)
		}
		
		# Log accuracy monitoring
		await self.audit_logger.log_action(
			action="monitor_forecast_accuracy",
			details=accuracy_report
		)
		
		return accuracy_report
	
	async def _extract_historical_cash_flow_data(
		self,
		forecast_input: CashFlowForecastInput
	) -> List[Dict[str, Any]]:
		"""Extract historical cash flow data for forecasting."""
		
		# Calculate lookback period (minimum 3x forecast horizon)
		forecast_days = (forecast_input.forecast_end_date - forecast_input.forecast_start_date).days
		lookback_days = max(forecast_days * 3, self.config.min_historical_data_points)
		
		lookback_date = forecast_input.forecast_start_date - timedelta(days=lookback_days)
		
		# Query historical data
		# This would typically involve database queries to get:
		# - Payment history
		# - Invoice collection patterns
		# - Customer payment behaviors
		# - Seasonal trends
		
		historical_data = []
		
		# Simulate historical data extraction
		current_date = lookback_date
		while current_date < forecast_input.forecast_start_date:
			daily_data = {
				"date": current_date,
				"collections": float(Decimal('5000.00') + Decimal(str(current_date.weekday() * 1000))),  # Simulate weekly patterns
				"invoices_paid": 15 + (current_date.weekday() * 3),
				"average_payment_days": 28.5 + (current_date.weekday() * 2),
				"customer_count": 50,
				"seasonal_factor": 1.0 + 0.1 * (current_date.month / 12)  # Seasonal variation
			}
			historical_data.append(daily_data)
			current_date += timedelta(days=1)
		
		return historical_data
	
	async def _prepare_time_series_data(
		self,
		historical_data: List[Dict[str, Any]],
		forecast_input: CashFlowForecastInput
	) -> Dict[str, Any]:
		"""Prepare data for APG time series analytics."""
		
		# Convert to time series format expected by APG
		time_series_data = {
			"data_points": [],
			"metadata": {
				"frequency": "daily",
				"target_variable": "collections",
				"features": ["invoices_paid", "average_payment_days", "customer_count", "seasonal_factor"],
				"forecast_horizon": (forecast_input.forecast_end_date - forecast_input.forecast_start_date).days,
				"seasonality_periods": self.config.seasonal_decomposition_periods
			}
		}
		
		for data_point in historical_data:
			time_series_point = {
				"timestamp": data_point["date"].isoformat(),
				"target": data_point["collections"],
				"features": {
					"invoices_paid": data_point["invoices_paid"],
					"average_payment_days": data_point["average_payment_days"],
					"customer_count": data_point["customer_count"],
					"seasonal_factor": data_point["seasonal_factor"]
				}
			}
			time_series_data["data_points"].append(time_series_point)
		
		return time_series_data
	
	async def _generate_time_series_forecast(
		self,
		time_series_data: Dict[str, Any],
		forecast_input: CashFlowForecastInput
	) -> List[Dict[str, Any]]:
		"""Generate forecast using APG time series analytics."""
		
		# Call APG time series analytics service
		forecast_request = {
			"model_name": self.config.forecasting_model_name,
			"model_version": self.config.model_version,
			"data": time_series_data,
			"forecast_config": {
				"confidence_level": forecast_input.confidence_level,
				"include_trend": True,
				"include_seasonality": forecast_input.include_seasonal_trends,
				"include_external_factors": forecast_input.include_external_factors
			}
		}
		
		forecast_results = await self.time_series_service.generate_forecast(forecast_request)
		
		return forecast_results["predictions"]
	
	async def _apply_scenario_adjustments(
		self,
		forecast_results: List[Dict[str, Any]],
		forecast_input: CashFlowForecastInput
	) -> List[Dict[str, Any]]:
		"""Apply scenario-specific adjustments to forecast."""
		
		adjusted_results = []
		
		for prediction in forecast_results:
			adjusted_prediction = prediction.copy()
			
			# Apply collection rate adjustment
			if forecast_input.collection_rate_adjustment:
				adjusted_prediction["predicted_value"] *= forecast_input.collection_rate_adjustment
			
			# Apply payment delay adjustment
			if forecast_input.payment_delay_adjustment:
				# Shift timing of collections based on payment delay
				original_date = datetime.fromisoformat(prediction["timestamp"]).date()
				adjusted_date = original_date + timedelta(days=forecast_input.payment_delay_adjustment)
				adjusted_prediction["timestamp"] = adjusted_date.isoformat()
			
			adjusted_results.append(adjusted_prediction)
		
		return adjusted_results
	
	async def _calculate_forecast_statistics(
		self,
		forecast_results: List[Dict[str, Any]],
		forecast_input: CashFlowForecastInput
	) -> List[CashFlowDataPoint]:
		"""Calculate statistical measures for forecast data points."""
		
		forecast_points = []
		
		for result in forecast_results:
			forecast_date = datetime.fromisoformat(result["timestamp"]).date()
			
			# Skip if outside forecast range
			if forecast_date < forecast_input.forecast_start_date or forecast_date > forecast_input.forecast_end_date:
				continue
			
			data_point = CashFlowDataPoint(
				forecast_date=forecast_date,
				forecast_period=forecast_input.forecast_period,
				expected_collections=Decimal(str(result["predicted_value"])),
				total_cash_flow=Decimal(str(result["predicted_value"])),
				confidence_interval_lower=Decimal(str(result.get("confidence_lower", result["predicted_value"] * 0.8))),
				confidence_interval_upper=Decimal(str(result.get("confidence_upper", result["predicted_value"] * 1.2))),
				standard_deviation=Decimal(str(result.get("prediction_std", result["predicted_value"] * 0.1))),
				customer_count=result.get("customer_count", 50),
				invoice_count=result.get("invoice_count", 15),
				average_payment_days=result.get("average_payment_days", 30.0),
				seasonal_adjustment=Decimal(str(result.get("seasonal_component", 0.0))),
				economic_adjustment=Decimal(str(result.get("economic_component", 0.0))),
				collection_efficiency_factor=result.get("efficiency_factor", 1.0)
			)
			
			forecast_points.append(data_point)
		
		return forecast_points
	
	async def _generate_forecast_summary(
		self,
		forecast_points: List[CashFlowDataPoint],
		forecast_input: CashFlowForecastInput
	) -> CashFlowForecastSummary:
		"""Generate comprehensive forecast summary."""
		
		if not forecast_points:
			raise ValueError("No forecast data points generated")
		
		# Calculate summary metrics
		total_forecasted = sum(dp.total_cash_flow for dp in forecast_points)
		average_daily = total_forecasted / len(forecast_points)
		
		# Find peak collection day
		peak_day = max(forecast_points, key=lambda dp: dp.total_cash_flow)
		
		# Calculate confidence level
		avg_confidence = sum(
			(dp.confidence_interval_upper - dp.confidence_interval_lower) / (2 * dp.total_cash_flow)
			for dp in forecast_points if dp.total_cash_flow > 0
		) / len([dp for dp in forecast_points if dp.total_cash_flow > 0])
		
		confidence_level = ForecastConfidenceLevel.HIGH
		if avg_confidence > 0.3:
			confidence_level = ForecastConfidenceLevel.LOW
		elif avg_confidence > 0.2:
			confidence_level = ForecastConfidenceLevel.MEDIUM
		elif avg_confidence > 0.1:
			confidence_level = ForecastConfidenceLevel.HIGH
		
		# Calculate volatility
		values = [float(dp.total_cash_flow) for dp in forecast_points]
		mean_value = sum(values) / len(values)
		variance = sum((v - mean_value) ** 2 for v in values) / len(values)
		volatility = (variance ** 0.5) / mean_value if mean_value > 0 else 0
		
		summary = CashFlowForecastSummary(
			tenant_id=self.tenant_id,
			forecast_period_start=forecast_input.forecast_start_date,
			forecast_period_end=forecast_input.forecast_end_date,
			total_forecasted_collections=total_forecasted,
			average_daily_collections=average_daily,
			peak_collection_day=peak_day.forecast_date,
			peak_collection_amount=peak_day.total_cash_flow,
			overall_confidence_level=confidence_level,
			model_accuracy_score=0.90,  # Would be retrieved from historical performance
			confidence_score=max(0.0, 1.0 - avg_confidence),
			forecast_volatility=volatility,
			downside_risk_amount=min(dp.confidence_interval_lower for dp in forecast_points),
			upside_potential_amount=max(dp.confidence_interval_upper for dp in forecast_points),
			data_points_used=len(forecast_points),
			next_update_due=datetime.utcnow() + timedelta(days=self.config.forecast_update_intervals["normal"])
		)
		
		return summary
	
	async def _log_forecast_generation(
		self,
		forecast_input: CashFlowForecastInput,
		summary: CashFlowForecastSummary
	):
		"""Log forecast generation for audit purposes."""
		
		await self.audit_logger.log_action(
			action="generate_cash_flow_forecast",
			details={
				"forecast_id": summary.forecast_id,
				"forecast_period": f"{forecast_input.forecast_start_date} to {forecast_input.forecast_end_date}",
				"scenario_type": forecast_input.scenario_type,
				"total_forecasted": float(summary.total_forecasted_collections),
				"confidence_level": summary.overall_confidence_level,
				"model_accuracy": summary.model_accuracy_score
			}
		)
	
	async def _generate_risk_mitigation_actions(
		self,
		comparison: CashFlowScenarioComparison
	) -> List[str]:
		"""Generate risk mitigation recommendations."""
		
		actions = []
		
		# Analyze scenario spread
		spread_percentage = float(comparison.scenario_spread / comparison.realistic_total) if comparison.realistic_total > 0 else 0
		
		if spread_percentage > 0.3:  # High variability
			actions.append("Implement dynamic collection strategies to reduce forecast uncertainty")
			actions.append("Increase frequency of customer payment behavior monitoring")
		
		if comparison.pessimistic_total < comparison.realistic_total * 0.7:
			actions.append("Establish credit line or cash reserves for downside protection")
			actions.append("Consider factoring or invoice financing for critical periods")
		
		actions.append("Monitor early warning indicators for scenario shifts")
		actions.append("Implement weekly forecast updates during volatile periods")
		
		return actions
	
	async def _generate_liquidity_recommendations(
		self,
		comparison: CashFlowScenarioComparison
	) -> List[str]:
		"""Generate liquidity management recommendations."""
		
		recommendations = []
		
		# Calculate liquidity needs
		min_expected = min(
			sum(dp.total_cash_flow for dp in comparison.pessimistic_forecast[:30])  # First 30 days
		)
		
		if min_expected < Decimal('50000.00'):  # Threshold for liquidity concern
			recommendations.append("Maintain additional cash reserves for low-collection periods")
			recommendations.append("Consider accelerating collection efforts for large overdue accounts")
		
		recommendations.append("Optimize payment terms for new customers based on forecast scenarios")
		recommendations.append("Implement cash flow-based credit limit adjustments")
		
		return recommendations
	
	async def _calculate_accuracy_metrics(
		self,
		forecasted: List[CashFlowDataPoint],
		actual: List[CashFlowDataPoint]
	) -> Dict[str, float]:
		"""Calculate forecast accuracy metrics."""
		
		# Align forecasted and actual data points by date
		forecast_dict = {dp.forecast_date: dp for dp in forecasted}
		actual_dict = {dp.forecast_date: dp for dp in actual}
		
		aligned_pairs = []
		for date in forecast_dict:
			if date in actual_dict:
				aligned_pairs.append((
					float(forecast_dict[date].total_cash_flow),
					float(actual_dict[date].total_cash_flow)
				))
		
		if not aligned_pairs:
			return {"error": "No matching dates between forecast and actual data"}
		
		# Calculate metrics
		forecasted_values = [pair[0] for pair in aligned_pairs]
		actual_values = [pair[1] for pair in aligned_pairs]
		
		# Mean Absolute Error (MAE)
		mae = sum(abs(f - a) for f, a in aligned_pairs) / len(aligned_pairs)
		
		# Mean Absolute Percentage Error (MAPE)
		mape = sum(abs((f - a) / a) if a != 0 else 0 for f, a in aligned_pairs) / len(aligned_pairs)
		
		# Root Mean Squared Error (RMSE)
		rmse = (sum((f - a) ** 2 for f, a in aligned_pairs) / len(aligned_pairs)) ** 0.5
		
		# Accuracy (1 - MAPE)
		accuracy = max(0.0, 1.0 - mape)
		
		return {
			"accuracy": accuracy,
			"mae": mae,
			"mape": mape,
			"rmse": rmse,
			"data_points": len(aligned_pairs)
		}
	
	async def _compare_to_accuracy_targets(
		self,
		accuracy_metrics: Dict[str, float]
	) -> Dict[str, Any]:
		"""Compare accuracy metrics to performance targets."""
		
		targets = self.config.forecast_accuracy_targets
		comparison = {}
		
		current_accuracy = accuracy_metrics.get("accuracy", 0.0)
		
		for period, target in targets.items():
			comparison[period] = {
				"target": target,
				"actual": current_accuracy,
				"meets_target": current_accuracy >= target,
				"gap": target - current_accuracy if current_accuracy < target else 0.0
			}
		
		return comparison
	
	async def _generate_accuracy_improvement_recommendations(
		self,
		accuracy_metrics: Dict[str, float]
	) -> List[str]:
		"""Generate recommendations for improving forecast accuracy."""
		
		recommendations = []
		
		accuracy = accuracy_metrics.get("accuracy", 0.0)
		mape = accuracy_metrics.get("mape", 1.0)
		
		if accuracy < 0.85:
			recommendations.append("Increase historical data collection period for better pattern recognition")
			recommendations.append("Implement customer-specific payment behavior models")
		
		if mape > 0.20:
			recommendations.append("Enhance feature engineering with external economic indicators")
			recommendations.append("Implement ensemble forecasting with multiple model approaches")
		
		recommendations.append("Increase forecast update frequency during volatile periods")
		recommendations.append("Implement real-time model retraining based on recent payment patterns")
		
		return recommendations
	
	# Additional helper methods would be implemented here...
	async def _retrieve_forecast_by_id(self, forecast_id: str) -> Optional[List[CashFlowDataPoint]]:
		"""Retrieve forecast by ID (placeholder implementation)."""
		# This would query the database to retrieve stored forecast
		return None
	
	async def _update_model_performance_tracking(self, accuracy_metrics: Dict[str, float]):
		"""Update model performance tracking (placeholder implementation)."""
		# This would update performance metrics in the database
		pass


# =============================================================================
# Factory Functions
# =============================================================================

async def create_cashflow_forecasting_service(
	tenant_id: str,
	user_id: str,
	config: Optional[CashFlowForecastingConfig] = None
) -> APGCashFlowForecastingService:
	"""Create and initialize cash flow forecasting service."""
	
	if config is None:
		config = CashFlowForecastingConfig(
			time_series_endpoint="https://time-series.apg.company.com/v1"
		)
	
	service = APGCashFlowForecastingService(tenant_id, user_id, config)
	await service._initialize_services()
	
	return service


def create_default_forecast_config() -> CashFlowForecastingConfig:
	"""Create default cash flow forecasting configuration."""
	
	return CashFlowForecastingConfig(
		time_series_endpoint="https://time-series.apg.company.com/v1",
		forecasting_model_name="ar_cashflow_predictor_v1",
		model_version="1.0.0",
		accuracy_threshold=0.90
	)