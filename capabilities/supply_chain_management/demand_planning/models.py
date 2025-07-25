"""
Demand Planning Data Models

Comprehensive models for demand forecasting, historical tracking, and accuracy analysis.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List
from uuid_extensions import uuid7str
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, Numeric, Date, JSON, ForeignKey, Index, CheckConstraint
from sqlalchemy.orm import relationship, Mapped
from pydantic import BaseModel, Field, validator, ConfigDict
from flask_appbuilder import Model

class SCDPForecast(Model):
	"""Demand forecast for products/SKUs"""
	__tablename__ = 'sc_dp_forecast'
	
	# Primary fields
	id: Mapped[str] = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: Mapped[str] = Column(String(50), nullable=False, index=True)
	
	# Forecast identification
	forecast_id: Mapped[str] = Column(String(100), nullable=False, index=True)
	product_sku: Mapped[str] = Column(String(100), nullable=False, index=True)
	location_code: Mapped[str] = Column(String(50), nullable=True, index=True)
	forecast_date: Mapped[date] = Column(Date, nullable=False, index=True)
	
	# Forecast details
	forecast_model_id: Mapped[str] = Column(String(50), ForeignKey('sc_dp_forecast_model.id'), nullable=False)
	forecast_quantity: Mapped[Decimal] = Column(Numeric(15, 4), nullable=False)
	forecast_value: Mapped[Decimal] = Column(Numeric(15, 2), nullable=True)
	confidence_level: Mapped[Decimal] = Column(Numeric(5, 4), nullable=True)  # 0-1
	
	# Forecast period
	period_start: Mapped[date] = Column(Date, nullable=False)
	period_end: Mapped[date] = Column(Date, nullable=False)
	period_type: Mapped[str] = Column(String(20), nullable=False, default='daily')  # daily, weekly, monthly
	
	# Statistical measures
	forecast_error: Mapped[Decimal] = Column(Numeric(15, 4), nullable=True)
	forecast_bias: Mapped[Decimal] = Column(Numeric(15, 4), nullable=True)
	seasonal_factor: Mapped[Decimal] = Column(Numeric(8, 4), nullable=True)
	trend_factor: Mapped[Decimal] = Column(Numeric(8, 4), nullable=True)
	
	# Status and metadata
	status: Mapped[str] = Column(String(20), nullable=False, default='draft')  # draft, approved, published, archived
	version: Mapped[int] = Column(Integer, nullable=False, default=1)
	is_baseline: Mapped[bool] = Column(Boolean, nullable=False, default=False)
	
	# Audit fields
	created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow)
	created_by: Mapped[str] = Column(String(100), nullable=False)
	updated_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	updated_by: Mapped[str] = Column(String(100), nullable=False)
	
	# Relationships
	forecast_model = relationship("SCDPForecastModel", back_populates="forecasts")
	
	# Indexes and constraints
	__table_args__ = (
		Index('ix_sc_dp_forecast_tenant_product_date', 'tenant_id', 'product_sku', 'forecast_date'),
		Index('ix_sc_dp_forecast_location_period', 'location_code', 'period_start', 'period_end'),
		CheckConstraint('confidence_level >= 0 AND confidence_level <= 1'),
		CheckConstraint('period_start <= period_end'),
		CheckConstraint('forecast_quantity >= 0'),
	)

class SCDPForecastModel(Model):
	"""Forecast models and algorithms configuration"""
	__tablename__ = 'sc_dp_forecast_model'
	
	# Primary fields
	id: Mapped[str] = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: Mapped[str] = Column(String(50), nullable=False, index=True)
	
	# Model identification
	model_name: Mapped[str] = Column(String(100), nullable=False)
	model_code: Mapped[str] = Column(String(50), nullable=False, index=True)
	model_type: Mapped[str] = Column(String(50), nullable=False)  # arima, exponential_smoothing, linear_regression, ml_ensemble
	
	# Model configuration
	algorithm: Mapped[str] = Column(String(100), nullable=False)
	parameters: Mapped[dict] = Column(JSON, nullable=True)
	hyperparameters: Mapped[dict] = Column(JSON, nullable=True)
	
	# Performance metrics
	accuracy_mape: Mapped[Decimal] = Column(Numeric(5, 4), nullable=True)  # Mean Absolute Percentage Error
	accuracy_rmse: Mapped[Decimal] = Column(Numeric(15, 4), nullable=True)  # Root Mean Square Error
	accuracy_mae: Mapped[Decimal] = Column(Numeric(15, 4), nullable=True)   # Mean Absolute Error
	
	# Training details
	training_data_start: Mapped[date] = Column(Date, nullable=True)
	training_data_end: Mapped[date] = Column(Date, nullable=True)
	training_samples: Mapped[int] = Column(Integer, nullable=True)
	last_trained: Mapped[datetime] = Column(DateTime, nullable=True)
	
	# Model lifecycle
	status: Mapped[str] = Column(String(20), nullable=False, default='development')  # development, trained, deployed, archived
	is_active: Mapped[bool] = Column(Boolean, nullable=False, default=True)
	auto_retrain: Mapped[bool] = Column(Boolean, nullable=False, default=False)
	retrain_frequency_days: Mapped[int] = Column(Integer, nullable=True, default=30)
	
	# Audit fields
	created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow)
	created_by: Mapped[str] = Column(String(100), nullable=False)
	updated_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	updated_by: Mapped[str] = Column(String(100), nullable=False)
	
	# Relationships
	forecasts = relationship("SCDPForecast", back_populates="forecast_model")
	
	# Indexes and constraints
	__table_args__ = (
		Index('ix_sc_dp_forecast_model_tenant_code', 'tenant_id', 'model_code'),
		Index('ix_sc_dp_forecast_model_type_status', 'model_type', 'status'),
	)

class SCDPDemandHistory(Model):
	"""Historical demand data for forecasting"""
	__tablename__ = 'sc_dp_demand_history'
	
	# Primary fields
	id: Mapped[str] = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: Mapped[str] = Column(String(50), nullable=False, index=True)
	
	# Product and location
	product_sku: Mapped[str] = Column(String(100), nullable=False, index=True)
	location_code: Mapped[str] = Column(String(50), nullable=True, index=True)
	demand_date: Mapped[date] = Column(Date, nullable=False, index=True)
	
	# Demand metrics
	actual_demand: Mapped[Decimal] = Column(Numeric(15, 4), nullable=False)
	fulfilled_demand: Mapped[Decimal] = Column(Numeric(15, 4), nullable=True)
	lost_sales: Mapped[Decimal] = Column(Numeric(15, 4), nullable=True, default=0)
	demand_value: Mapped[Decimal] = Column(Numeric(15, 2), nullable=True)
	
	# Context information
	day_of_week: Mapped[int] = Column(Integer, nullable=False)  # 1-7
	week_of_year: Mapped[int] = Column(Integer, nullable=False)  # 1-53
	month: Mapped[int] = Column(Integer, nullable=False)        # 1-12
	quarter: Mapped[int] = Column(Integer, nullable=False)      # 1-4
	
	# External factors
	promotion_active: Mapped[bool] = Column(Boolean, nullable=False, default=False)
	stockout_occurred: Mapped[bool] = Column(Boolean, nullable=False, default=False)
	weather_impact: Mapped[str] = Column(String(20), nullable=True)  # none, positive, negative
	holiday_impact: Mapped[str] = Column(String(20), nullable=True)  # none, positive, negative
	
	# Data quality
	data_source: Mapped[str] = Column(String(50), nullable=False)  # sales, pos, web, manual
	data_quality_score: Mapped[Decimal] = Column(Numeric(3, 2), nullable=True, default=1.0)  # 0-1
	is_outlier: Mapped[bool] = Column(Boolean, nullable=False, default=False)
	outlier_reason: Mapped[str] = Column(String(200), nullable=True)
	
	# Audit fields
	created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow)
	created_by: Mapped[str] = Column(String(100), nullable=False)
	updated_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	updated_by: Mapped[str] = Column(String(100), nullable=False)
	
	# Indexes and constraints
	__table_args__ = (
		Index('ix_sc_dp_demand_history_tenant_product_date', 'tenant_id', 'product_sku', 'demand_date'),
		Index('ix_sc_dp_demand_history_location_month', 'location_code', 'month', 'demand_date'),
		CheckConstraint('actual_demand >= 0'),
		CheckConstraint('data_quality_score >= 0 AND data_quality_score <= 1'),
		CheckConstraint('day_of_week >= 1 AND day_of_week <= 7'),
		CheckConstraint('month >= 1 AND month <= 12'),
	)

class SCDPSeasonalPattern(Model):
	"""Seasonal patterns and factors for demand forecasting"""
	__tablename__ = 'sc_dp_seasonal_pattern'
	
	# Primary fields
	id: Mapped[str] = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: Mapped[str] = Column(String(50), nullable=False, index=True)
	
	# Pattern identification
	pattern_name: Mapped[str] = Column(String(100), nullable=False)
	product_sku: Mapped[str] = Column(String(100), nullable=False, index=True)
	location_code: Mapped[str] = Column(String(50), nullable=True, index=True)
	
	# Seasonal details
	season_type: Mapped[str] = Column(String(20), nullable=False)  # weekly, monthly, quarterly, yearly, holiday
	season_period: Mapped[str] = Column(String(50), nullable=False)  # Q1, January, Week1, Christmas, etc.
	seasonal_factor: Mapped[Decimal] = Column(Numeric(8, 4), nullable=False)
	
	# Statistical measures
	confidence_interval: Mapped[Decimal] = Column(Numeric(5, 4), nullable=True)
	sample_size: Mapped[int] = Column(Integer, nullable=True)
	last_calculated: Mapped[datetime] = Column(DateTime, nullable=True)
	
	# Pattern validity
	valid_from: Mapped[date] = Column(Date, nullable=False)
	valid_to: Mapped[date] = Column(Date, nullable=True)
	is_active: Mapped[bool] = Column(Boolean, nullable=False, default=True)
	
	# Audit fields
	created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow)
	created_by: Mapped[str] = Column(String(100), nullable=False)
	updated_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	updated_by: Mapped[str] = Column(String(100), nullable=False)
	
	# Indexes and constraints
	__table_args__ = (
		Index('ix_sc_dp_seasonal_pattern_tenant_product', 'tenant_id', 'product_sku'),
		Index('ix_sc_dp_seasonal_pattern_type_period', 'season_type', 'season_period'),
	)

class SCDPForecastAccuracy(Model):
	"""Forecast accuracy tracking and metrics"""
	__tablename__ = 'sc_dp_forecast_accuracy'
	
	# Primary fields
	id: Mapped[str] = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: Mapped[str] = Column(String(50), nullable=False, index=True)
	
	# Accuracy measurement
	forecast_id: Mapped[str] = Column(String(50), ForeignKey('sc_dp_forecast.id'), nullable=False)
	product_sku: Mapped[str] = Column(String(100), nullable=False, index=True)
	measurement_date: Mapped[date] = Column(Date, nullable=False, index=True)
	
	# Forecast vs Actual
	forecasted_quantity: Mapped[Decimal] = Column(Numeric(15, 4), nullable=False)
	actual_quantity: Mapped[Decimal] = Column(Numeric(15, 4), nullable=False)
	absolute_error: Mapped[Decimal] = Column(Numeric(15, 4), nullable=False)
	percentage_error: Mapped[Decimal] = Column(Numeric(10, 6), nullable=False)
	
	# Accuracy metrics
	mape: Mapped[Decimal] = Column(Numeric(10, 6), nullable=True)  # Mean Absolute Percentage Error
	bias: Mapped[Decimal] = Column(Numeric(15, 4), nullable=True)
	
	# Classification
	accuracy_category: Mapped[str] = Column(String(20), nullable=False)  # excellent, good, fair, poor
	error_type: Mapped[str] = Column(String(20), nullable=False)         # under_forecast, over_forecast, accurate
	
	# Context
	forecast_horizon_days: Mapped[int] = Column(Integer, nullable=False)
	model_used: Mapped[str] = Column(String(100), nullable=True)
	
	# Audit fields
	created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow)
	created_by: Mapped[str] = Column(String(100), nullable=False)
	
	# Relationships
	forecast = relationship("SCDPForecast")
	
	# Indexes and constraints
	__table_args__ = (
		Index('ix_sc_dp_forecast_accuracy_tenant_product_date', 'tenant_id', 'product_sku', 'measurement_date'),
		Index('ix_sc_dp_forecast_accuracy_category', 'accuracy_category'),
		CheckConstraint('forecasted_quantity >= 0'),
		CheckConstraint('actual_quantity >= 0'),
		CheckConstraint('absolute_error >= 0'),
	)

# Pydantic models for API
class ForecastCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	forecast_id: str = Field(..., min_length=1, max_length=100)
	product_sku: str = Field(..., min_length=1, max_length=100)
	location_code: str | None = Field(None, max_length=50)
	forecast_model_id: str = Field(..., min_length=1)
	forecast_quantity: Decimal = Field(..., ge=0)
	forecast_value: Decimal | None = Field(None, ge=0)
	confidence_level: Decimal | None = Field(None, ge=0, le=1)
	period_start: date = Field(...)
	period_end: date = Field(...)
	period_type: str = Field(default='daily')
	
	@validator('period_end')
	def validate_period_end(cls, v, values):
		if 'period_start' in values and v < values['period_start']:
			raise ValueError('period_end must be after period_start')
		return v

class DemandHistoryCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	product_sku: str = Field(..., min_length=1, max_length=100)
	location_code: str | None = Field(None, max_length=50)
	demand_date: date = Field(...)
	actual_demand: Decimal = Field(..., ge=0)
	fulfilled_demand: Decimal | None = Field(None, ge=0)
	lost_sales: Decimal = Field(default=Decimal('0'), ge=0)
	demand_value: Decimal | None = Field(None, ge=0)
	data_source: str = Field(..., min_length=1, max_length=50)
	promotion_active: bool = Field(default=False)
	stockout_occurred: bool = Field(default=False)

class ForecastModelCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	model_name: str = Field(..., min_length=1, max_length=100)
	model_code: str = Field(..., min_length=1, max_length=50)
	model_type: str = Field(..., min_length=1, max_length=50)
	algorithm: str = Field(..., min_length=1, max_length=100)
	parameters: dict | None = Field(None)
	hyperparameters: dict | None = Field(None)
	auto_retrain: bool = Field(default=False)
	retrain_frequency_days: int | None = Field(None, ge=1, le=365)