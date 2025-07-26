"""
Sales Forecasting Models

Database models for sales forecasting including historical data analysis,
trend prediction, seasonal adjustments, and forecast accuracy tracking.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ....auth_rbac.models import BaseMixin, AuditMixin, Model


class SOFForecastModel(Model, AuditMixin, BaseMixin):
	"""
	Forecasting model definitions and configurations.
	
	Manages different forecasting algorithms, parameters,
	and model performance tracking.
	"""
	__tablename__ = 'so_f_forecast_model'
	
	# Identity
	model_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Model Information
	model_name = Column(String(100), nullable=False)
	model_type = Column(String(20), nullable=False)  # LINEAR_REGRESSION, SEASONAL, MOVING_AVERAGE, ARIMA
	description = Column(Text, nullable=True)
	
	# Model Parameters
	parameters = Column(Text, nullable=True)  # JSON configuration
	
	# Performance Metrics
	accuracy_percentage = Column(DECIMAL(5, 2), default=0.00)
	mean_absolute_error = Column(DECIMAL(15, 2), default=0.00)
	last_trained_date = Column(DateTime, nullable=True)
	training_data_points = Column(Integer, default=0)
	
	# Application Scope
	forecast_horizon_days = Column(Integer, default=90)
	minimum_history_days = Column(Integer, default=365)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_default = Column(Boolean, default=False)
	auto_retrain = Column(Boolean, default=True)
	retrain_frequency_days = Column(Integer, default=30)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'model_name', name='uq_sof_model_name_tenant'),
	)
	
	# Relationships
	forecasts = relationship("SOFForecast", back_populates="model")
	
	def __repr__(self):
		return f"<SOFForecastModel {self.model_name} - {self.accuracy_percentage}%>"


class SOFForecast(Model, AuditMixin, BaseMixin):
	"""
	Sales forecast records with predictions and actuals.
	
	Stores forecast predictions and tracks actual results
	for accuracy measurement and model improvement.
	"""
	__tablename__ = 'so_f_forecast'
	
	# Identity
	forecast_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Forecast Information
	forecast_name = Column(String(200), nullable=False)
	model_id = Column(String(36), ForeignKey('so_f_forecast_model.model_id'), nullable=False, index=True)
	
	# Forecast Period
	forecast_date = Column(Date, nullable=False, index=True)  # Date forecast was created
	period_start = Column(Date, nullable=False, index=True)
	period_end = Column(Date, nullable=False, index=True)
	period_type = Column(String(10), default='MONTHLY')  # DAILY, WEEKLY, MONTHLY, QUARTERLY
	
	# Forecast Scope
	scope_type = Column(String(20), default='TOTAL')  # TOTAL, CUSTOMER, PRODUCT, TERRITORY
	scope_id = Column(String(36), nullable=True, index=True)  # ID of scoped entity
	scope_name = Column(String(200), nullable=True)
	
	# Forecast Values
	predicted_quantity = Column(DECIMAL(15, 4), default=0.0000)
	predicted_revenue = Column(DECIMAL(15, 2), default=0.00)
	predicted_orders = Column(Integer, default=0)
	
	# Confidence Intervals
	confidence_level = Column(DECIMAL(5, 2), default=95.00)
	lower_bound_revenue = Column(DECIMAL(15, 2), default=0.00)
	upper_bound_revenue = Column(DECIMAL(15, 2), default=0.00)
	
	# Actual Results (filled as period progresses)
	actual_quantity = Column(DECIMAL(15, 4), default=0.0000)
	actual_revenue = Column(DECIMAL(15, 2), default=0.00)
	actual_orders = Column(Integer, default=0)
	
	# Accuracy Metrics
	quantity_accuracy = Column(DECIMAL(5, 2), nullable=True)
	revenue_accuracy = Column(DECIMAL(5, 2), nullable=True)
	orders_accuracy = Column(DECIMAL(5, 2), nullable=True)
	
	# Adjustments
	manual_adjustment_revenue = Column(DECIMAL(15, 2), default=0.00)
	adjustment_reason = Column(String(200), nullable=True)
	adjusted_by = Column(String(36), nullable=True)
	
	# Status
	status = Column(String(20), default='ACTIVE')  # ACTIVE, COMPLETED, ARCHIVED
	is_published = Column(Boolean, default=False)
	published_date = Column(DateTime, nullable=True)
	
	# Currency
	currency_code = Column(String(3), default='USD')
	
	# Notes
	notes = Column(Text, nullable=True)
	assumptions = Column(Text, nullable=True)
	
	# Relationships
	model = relationship("SOFForecastModel", back_populates="forecasts")
	
	def __repr__(self):
		return f"<SOFForecast {self.forecast_name} - ${self.predicted_revenue}>"
	
	def calculate_accuracy(self):
		"""Calculate forecast accuracy metrics"""
		if self.actual_revenue > 0:
			revenue_error = abs(self.predicted_revenue - self.actual_revenue)
			self.revenue_accuracy = max(0, 100 - (revenue_error / self.actual_revenue * 100))
		
		if self.actual_quantity > 0:
			quantity_error = abs(self.predicted_quantity - self.actual_quantity)
			self.quantity_accuracy = max(0, 100 - (quantity_error / self.actual_quantity * 100))
		
		if self.actual_orders > 0:
			orders_error = abs(self.predicted_orders - self.actual_orders)
			self.orders_accuracy = max(0, 100 - (orders_error / self.actual_orders * 100))


class SOFHistoricalData(Model, AuditMixin, BaseMixin):
	"""
	Historical sales data for forecasting model training.
	
	Aggregated historical sales data used for training
	forecasting models and trend analysis.
	"""
	__tablename__ = 'so_f_historical_data'
	
	# Identity
	data_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Time Period
	data_date = Column(Date, nullable=False, index=True)
	period_type = Column(String(10), nullable=False)  # DAILY, WEEKLY, MONTHLY
	
	# Scope
	scope_type = Column(String(20), default='TOTAL')  # TOTAL, CUSTOMER, PRODUCT, TERRITORY
	scope_id = Column(String(36), nullable=True, index=True)
	scope_name = Column(String(200), nullable=True)
	
	# Sales Metrics
	total_orders = Column(Integer, default=0)
	total_quantity = Column(DECIMAL(15, 4), default=0.0000)
	total_revenue = Column(DECIMAL(15, 2), default=0.00)
	average_order_value = Column(DECIMAL(15, 2), default=0.00)
	
	# Customer Metrics
	new_customers = Column(Integer, default=0)
	returning_customers = Column(Integer, default=0)
	total_customers = Column(Integer, default=0)
	
	# Seasonal Factors
	is_weekend = Column(Boolean, default=False)
	is_holiday = Column(Boolean, default=False)
	season = Column(String(10), nullable=True)  # SPRING, SUMMER, FALL, WINTER
	month_number = Column(Integer, nullable=True)
	week_number = Column(Integer, nullable=True)
	day_of_week = Column(Integer, nullable=True)
	
	# External Factors
	weather_impact = Column(DECIMAL(5, 2), nullable=True)  # Weather impact score
	economic_indicator = Column(DECIMAL(10, 4), nullable=True)
	marketing_spend = Column(DECIMAL(15, 2), default=0.00)
	promotional_activity = Column(Boolean, default=False)
	
	# Data Quality
	data_completeness = Column(DECIMAL(5, 2), default=100.00)
	has_anomalies = Column(Boolean, default=False)
	anomaly_description = Column(String(200), nullable=True)
	
	# Currency
	currency_code = Column(String(3), default='USD')
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'data_date', 'period_type', 'scope_type', 'scope_id', 
						name='uq_sof_historical_data'),
	)
	
	def __repr__(self):
		return f"<SOFHistoricalData {self.data_date} - ${self.total_revenue}>"


class SOFSeasonalPattern(Model, AuditMixin, BaseMixin):
	"""
	Seasonal patterns and adjustments for forecasting.
	
	Manages seasonal factors and patterns identified
	from historical data analysis.
	"""
	__tablename__ = 'so_f_seasonal_pattern'
	
	# Identity
	pattern_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Pattern Information
	pattern_name = Column(String(100), nullable=False)
	pattern_type = Column(String(20), nullable=False)  # MONTHLY, WEEKLY, DAILY, HOLIDAY
	description = Column(Text, nullable=True)
	
	# Scope
	scope_type = Column(String(20), default='TOTAL')
	scope_id = Column(String(36), nullable=True)
	scope_name = Column(String(200), nullable=True)
	
	# Pattern Data (JSON array of seasonal factors)
	pattern_data = Column(Text, nullable=False)
	
	# Statistical Measures
	confidence_level = Column(DECIMAL(5, 2), default=95.00)
	statistical_significance = Column(DECIMAL(5, 2), default=0.00)
	
	# Discovery Information
	discovered_date = Column(Date, nullable=False)
	years_of_data = Column(Integer, default=0)
	data_points_analyzed = Column(Integer, default=0)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	auto_update = Column(Boolean, default=True)
	last_updated = Column(DateTime, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'pattern_name', 'scope_type', 'scope_id', 
						name='uq_sof_pattern_scope'),
	)
	
	def __repr__(self):
		return f"<SOFSeasonalPattern {self.pattern_name} - {self.pattern_type}>"
	
	def get_pattern_data(self) -> List[Dict[str, Any]]:
		"""Get seasonal pattern data"""
		return json.loads(self.pattern_data)
	
	def set_pattern_data(self, data: List[Dict[str, Any]]):
		"""Set seasonal pattern data"""
		self.pattern_data = json.dumps(data)
	
	def get_seasonal_factor(self, period_key: str) -> float:
		"""Get seasonal factor for specific period"""
		pattern_data = self.get_pattern_data()
		for item in pattern_data:
			if item.get('period') == period_key:
				return float(item.get('factor', 1.0))
		return 1.0