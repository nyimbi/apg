"""
Production Planning Models

Database models for production planning functionality including master production
schedules, production orders, demand forecasts, and resource planning.
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlalchemy import Column, String, Integer, DateTime, Date, Numeric, Text, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from uuid_extensions import uuid7str

from ...core_financials.general_ledger.models import BaseModel as SQLBaseModel

class ProductionOrderStatus(str, Enum):
	"""Production order status enumeration"""
	PLANNED = "planned"
	RELEASED = "released"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	CANCELLED = "cancelled"
	ON_HOLD = "on_hold"

class SchedulingPriority(str, Enum):
	"""Scheduling priority levels"""
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	URGENT = "urgent"
	CRITICAL = "critical"

class PlanningHorizon(str, Enum):
	"""Planning time horizon types"""
	SHORT_TERM = "short_term"  # 1-4 weeks
	MEDIUM_TERM = "medium_term"  # 1-6 months
	LONG_TERM = "long_term"  # 6+ months

# SQLAlchemy Models

class MFPMasterProductionSchedule(SQLBaseModel):
	"""Master Production Schedule (MPS) for high-level production planning"""
	__tablename__ = 'mfp_master_production_schedules'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Schedule identification
	schedule_name = Column(String(200), nullable=False)
	planning_period = Column(String(50), nullable=False)  # e.g., "2024-Q1", "2024-W15"
	planning_horizon = Column(String(20), nullable=False)  # short_term, medium_term, long_term
	
	# Planning details
	product_id = Column(String(36), nullable=False, index=True)
	facility_id = Column(String(36), nullable=False, index=True)
	production_line_id = Column(String(36), index=True)
	
	# Schedule quantities and dates
	planned_quantity = Column(Numeric(15, 4), nullable=False)
	planned_start_date = Column(Date, nullable=False)
	planned_end_date = Column(Date, nullable=False)
	
	# Demand and capacity
	forecast_demand = Column(Numeric(15, 4))
	available_capacity = Column(Numeric(15, 4))
	capacity_utilization_pct = Column(Numeric(5, 2))
	
	# Status and priority
	status = Column(String(20), nullable=False, default="active")
	priority = Column(String(20), nullable=False, default="normal")
	
	# Planning parameters
	safety_stock_days = Column(Integer, default=0)
	lead_time_days = Column(Integer, default=0)
	batch_size_min = Column(Numeric(15, 4))
	batch_size_max = Column(Numeric(15, 4))
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	production_orders = relationship("MFPProductionOrder", back_populates="master_schedule")
	
	__table_args__ = (
		Index('idx_mfp_mps_tenant_period', 'tenant_id', 'planning_period'),
		Index('idx_mfp_mps_product_facility', 'product_id', 'facility_id'),
		Index('idx_mfp_mps_dates', 'planned_start_date', 'planned_end_date'),
	)

class MFPProductionOrder(SQLBaseModel):
	"""Production orders for detailed manufacturing execution"""
	__tablename__ = 'mfp_production_orders'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Order identification
	order_number = Column(String(50), nullable=False, unique=True)
	order_type = Column(String(30), nullable=False)  # standard, rush, rework, etc.
	master_schedule_id = Column(String(36), ForeignKey('mfp_master_production_schedules.id'), index=True)
	
	# Product and location details
	product_id = Column(String(36), nullable=False, index=True)
	product_sku = Column(String(100), nullable=False)
	product_name = Column(String(200), nullable=False)
	facility_id = Column(String(36), nullable=False, index=True)
	production_line_id = Column(String(36), index=True)
	work_center_id = Column(String(36), index=True)
	
	# Quantities
	ordered_quantity = Column(Numeric(15, 4), nullable=False)
	produced_quantity = Column(Numeric(15, 4), default=0)
	scrap_quantity = Column(Numeric(15, 4), default=0)
	rework_quantity = Column(Numeric(15, 4), default=0)
	
	# Scheduling
	scheduled_start_date = Column(DateTime, nullable=False)
	scheduled_end_date = Column(DateTime, nullable=False)
	actual_start_date = Column(DateTime)
	actual_end_date = Column(DateTime)
	
	# Status and priority
	status = Column(String(20), nullable=False, default="planned")
	priority = Column(String(20), nullable=False, default="normal")
	
	# Resource requirements
	estimated_labor_hours = Column(Numeric(10, 2))
	actual_labor_hours = Column(Numeric(10, 2))
	estimated_machine_hours = Column(Numeric(10, 2))
	actual_machine_hours = Column(Numeric(10, 2))
	
	# Cost tracking
	estimated_material_cost = Column(Numeric(15, 2))
	actual_material_cost = Column(Numeric(15, 2))
	estimated_labor_cost = Column(Numeric(15, 2))
	actual_labor_cost = Column(Numeric(15, 2))
	estimated_overhead_cost = Column(Numeric(15, 2))
	actual_overhead_cost = Column(Numeric(15, 2))
	
	# BOM and routing
	bom_id = Column(String(36), index=True)
	routing_id = Column(String(36), index=True)
	
	# Notes and instructions
	production_notes = Column(Text)
	special_instructions = Column(Text)
	quality_requirements = Column(Text)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	master_schedule = relationship("MFPMasterProductionSchedule", back_populates="production_orders")
	
	__table_args__ = (
		Index('idx_mfp_po_tenant_status', 'tenant_id', 'status'),
		Index('idx_mfp_po_product_facility', 'product_id', 'facility_id'),
		Index('idx_mfp_po_schedule_dates', 'scheduled_start_date', 'scheduled_end_date'),
		Index('idx_mfp_po_order_number', 'order_number'),
	)

class MFPDemandForecast(SQLBaseModel):
	"""Demand forecasting for production planning"""
	__tablename__ = 'mfp_demand_forecasts'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Forecast identification
	forecast_name = Column(String(200), nullable=False)
	forecast_period = Column(String(50), nullable=False)  # e.g., "2024-01", "2024-W15"
	forecast_type = Column(String(30), nullable=False)  # sales, production, material
	
	# Product and location
	product_id = Column(String(36), nullable=False, index=True)
	facility_id = Column(String(36), index=True)
	customer_id = Column(String(36), index=True)
	
	# Forecast quantities
	forecast_quantity = Column(Numeric(15, 4), nullable=False)
	forecast_value = Column(Numeric(15, 2))
	actual_quantity = Column(Numeric(15, 4))
	actual_value = Column(Numeric(15, 2))
	
	# Forecast accuracy
	forecast_error = Column(Numeric(15, 4))
	forecast_accuracy_pct = Column(Numeric(5, 2))
	
	# Time periods
	period_start_date = Column(Date, nullable=False)
	period_end_date = Column(Date, nullable=False)
	
	# Forecasting method and parameters
	forecast_method = Column(String(50))  # moving_average, exponential_smoothing, etc.
	confidence_level = Column(Numeric(5, 2))
	seasonality_factor = Column(Numeric(10, 4))
	trend_factor = Column(Numeric(10, 4))
	
	# Status
	status = Column(String(20), nullable=False, default="active")
	is_approved = Column(Boolean, default=False)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('idx_mfp_df_tenant_period', 'tenant_id', 'forecast_period'),
		Index('idx_mfp_df_product_facility', 'product_id', 'facility_id'),
		Index('idx_mfp_df_period_dates', 'period_start_date', 'period_end_date'),
	)

class MFPResourceCapacity(SQLBaseModel):
	"""Resource capacity planning and tracking"""
	__tablename__ = 'mfp_resource_capacities'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Resource identification
	resource_type = Column(String(30), nullable=False)  # machine, labor, facility, tool
	resource_id = Column(String(36), nullable=False, index=True)
	resource_name = Column(String(200), nullable=False)
	facility_id = Column(String(36), nullable=False, index=True)
	work_center_id = Column(String(36), index=True)
	
	# Capacity details
	planning_period = Column(String(50), nullable=False)
	capacity_unit = Column(String(20), nullable=False)  # hours, pieces, kg, etc.
	available_capacity = Column(Numeric(15, 4), nullable=False)
	planned_capacity = Column(Numeric(15, 4), default=0)
	actual_capacity = Column(Numeric(15, 4), default=0)
	
	# Utilization metrics
	capacity_utilization_pct = Column(Numeric(5, 2))
	efficiency_pct = Column(Numeric(5, 2))
	availability_pct = Column(Numeric(5, 2))
	
	# Time periods
	period_start_date = Column(Date, nullable=False)
	period_end_date = Column(Date, nullable=False)
	
	# Shift and calendar information
	shifts_per_day = Column(Integer, default=1)
	hours_per_shift = Column(Numeric(4, 2), default=8.0)
	working_days_per_week = Column(Integer, default=5)
	
	# Constraints and limitations
	max_capacity = Column(Numeric(15, 4))
	min_capacity = Column(Numeric(15, 4))
	setup_time_hours = Column(Numeric(8, 2))
	maintenance_time_hours = Column(Numeric(8, 2))
	
	# Status
	status = Column(String(20), nullable=False, default="active")
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('idx_mfp_rc_tenant_period', 'tenant_id', 'planning_period'),
		Index('idx_mfp_rc_resource_facility', 'resource_id', 'facility_id'),
		Index('idx_mfp_rc_period_dates', 'period_start_date', 'period_end_date'),
	)

# Pydantic Models for API

class MasterProductionScheduleCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	schedule_name: str = Field(..., min_length=1, max_length=200)
	planning_period: str = Field(..., min_length=1, max_length=50)
	planning_horizon: PlanningHorizon
	product_id: str = Field(..., min_length=36, max_length=36)
	facility_id: str = Field(..., min_length=36, max_length=36)
	production_line_id: str | None = None
	planned_quantity: Decimal = Field(..., gt=0)
	planned_start_date: date
	planned_end_date: date
	forecast_demand: Decimal | None = None
	available_capacity: Decimal | None = None
	priority: SchedulingPriority = SchedulingPriority.NORMAL
	safety_stock_days: int = Field(default=0, ge=0)
	lead_time_days: int = Field(default=0, ge=0)
	batch_size_min: Decimal | None = None
	batch_size_max: Decimal | None = None

class ProductionOrderCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	order_number: str = Field(..., min_length=1, max_length=50)
	order_type: str = Field(..., min_length=1, max_length=30)
	master_schedule_id: str | None = None
	product_id: str = Field(..., min_length=36, max_length=36)
	product_sku: str = Field(..., min_length=1, max_length=100)
	product_name: str = Field(..., min_length=1, max_length=200)
	facility_id: str = Field(..., min_length=36, max_length=36)
	production_line_id: str | None = None
	work_center_id: str | None = None
	ordered_quantity: Decimal = Field(..., gt=0)
	scheduled_start_date: datetime
	scheduled_end_date: datetime
	priority: SchedulingPriority = SchedulingPriority.NORMAL
	estimated_labor_hours: Decimal | None = None
	estimated_machine_hours: Decimal | None = None
	bom_id: str | None = None
	routing_id: str | None = None
	production_notes: str | None = None
	special_instructions: str | None = None
	quality_requirements: str | None = None

class DemandForecastCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	forecast_name: str = Field(..., min_length=1, max_length=200)
	forecast_period: str = Field(..., min_length=1, max_length=50)
	forecast_type: str = Field(..., min_length=1, max_length=30)
	product_id: str = Field(..., min_length=36, max_length=36)
	facility_id: str | None = None  
	customer_id: str | None = None
	forecast_quantity: Decimal = Field(..., gt=0)
	forecast_value: Decimal | None = None
	period_start_date: date
	period_end_date: date
	forecast_method: str | None = None
	confidence_level: Decimal | None = Field(None, ge=0, le=100)
	seasonality_factor: Decimal | None = None
	trend_factor: Decimal | None = None

class ResourceCapacityCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	resource_type: str = Field(..., min_length=1, max_length=30)
	resource_id: str = Field(..., min_length=36, max_length=36)
	resource_name: str = Field(..., min_length=1, max_length=200)
	facility_id: str = Field(..., min_length=36, max_length=36)
	work_center_id: str | None = None
	planning_period: str = Field(..., min_length=1, max_length=50)
	capacity_unit: str = Field(..., min_length=1, max_length=20)
	available_capacity: Decimal = Field(..., gt=0)
	period_start_date: date
	period_end_date: date
	shifts_per_day: int = Field(default=1, ge=1, le=3)
	hours_per_shift: Decimal = Field(default=Decimal('8.0'), gt=0, le=24)
	working_days_per_week: int = Field(default=5, ge=1, le=7)
	max_capacity: Decimal | None = None
	min_capacity: Decimal | None = None
	setup_time_hours: Decimal | None = None
	maintenance_time_hours: Decimal | None = None

__all__ = [
	"ProductionOrderStatus", "SchedulingPriority", "PlanningHorizon",
	"MFPMasterProductionSchedule", "MFPProductionOrder", "MFPDemandForecast", "MFPResourceCapacity",
	"MasterProductionScheduleCreate", "ProductionOrderCreate", "DemandForecastCreate", "ResourceCapacityCreate"
]