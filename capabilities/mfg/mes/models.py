"""
Manufacturing Execution System (MES) Models

Database models for MES functionality including real-time production monitoring,
work order execution, resource management, and performance tracking.
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

class WorkOrderStatus(str, Enum):
	"""Work order execution status"""
	CREATED = "created"
	RELEASED = "released"
	STARTED = "started"
	IN_PROGRESS = "in_progress"
	PAUSED = "paused"
	COMPLETED = "completed"
	CLOSED = "closed"
	CANCELLED = "cancelled"

class ResourceStatus(str, Enum):
	"""Resource availability status"""
	AVAILABLE = "available"
	BUSY = "busy"
	MAINTENANCE = "maintenance"
	BREAKDOWN = "breakdown"
	OFFLINE = "offline"

class ProductionEventType(str, Enum):
	"""Production event types"""
	START = "start"
	PAUSE = "pause"
	RESUME = "resume"
	COMPLETE = "complete"
	STOP = "stop"
	BREAKDOWN = "breakdown"
	CHANGEOVER = "changeover"
	QUALITY_CHECK = "quality_check"

class PerformanceMetricType(str, Enum):
	"""Performance metric types"""
	OEE = "oee"  # Overall Equipment Effectiveness
	AVAILABILITY = "availability"
	PERFORMANCE = "performance"
	QUALITY = "quality"
	THROUGHPUT = "throughput"
	CYCLE_TIME = "cycle_time"
	DOWNTIME = "downtime"

class AlertSeverity(str, Enum):
	"""Alert severity levels"""
	INFO = "info"
	WARNING = "warning"
	ERROR = "error"
	CRITICAL = "critical"

# SQLAlchemy Models

class MFMWorkOrderExecution(SQLBaseModel):
	"""Work order execution tracking in MES"""
	__tablename__ = 'mfm_work_order_executions'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Work order reference
	work_order_id = Column(String(36), nullable=False, index=True)
	work_order_number = Column(String(50), nullable=False)
	
	# Product information
	product_id = Column(String(36), nullable=False, index=True)
	product_sku = Column(String(100), nullable=False)
	product_name = Column(String(200), nullable=False)
	
	# Quantities
	planned_quantity = Column(Numeric(15, 4), nullable=False)
	produced_quantity = Column(Numeric(15, 4), default=0)
	scrapped_quantity = Column(Numeric(15, 4), default=0)
	reworked_quantity = Column(Numeric(15, 4), default=0)
	
	# Location and resources
	facility_id = Column(String(36), nullable=False, index=True)
	production_line_id = Column(String(36), index=True)
	work_center_id = Column(String(36), index=True)
	
	# Timing
	planned_start_time = Column(DateTime)
	actual_start_time = Column(DateTime)
	planned_end_time = Column(DateTime)
	actual_end_time = Column(DateTime)
	
	# Status tracking
	status = Column(String(20), nullable=False, default="created")
	current_operation = Column(String(200))
	percent_complete = Column(Numeric(5, 2), default=0)
	
	# Personnel
	supervisor_id = Column(String(36))
	operators = Column(Text)  # JSON array of operator IDs
	
	# Performance metrics
	cycle_time_actual = Column(Numeric(10, 2))
	cycle_time_standard = Column(Numeric(10, 2))
	efficiency_percentage = Column(Numeric(5, 2))
	
	# Quality metrics
	first_pass_yield = Column(Numeric(5, 2))
	scrap_rate = Column(Numeric(5, 2))
	rework_rate = Column(Numeric(5, 2))
	
	# Real-time data
	current_speed = Column(Numeric(10, 2))
	target_speed = Column(Numeric(10, 2))
	temperature_readings = Column(Text)  # JSON array of temperature data
	pressure_readings = Column(Text)  # JSON array of pressure data
	
	# Material tracking
	materials_consumed = Column(Text)  # JSON with material consumption data
	material_waste = Column(Numeric(15, 4))
	
	# Downtime tracking
	total_downtime_minutes = Column(Integer, default=0)
	planned_downtime_minutes = Column(Integer, default=0)
	unplanned_downtime_minutes = Column(Integer, default=0)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	production_events = relationship("MFMProductionEvent", back_populates="work_order_execution")
	resource_assignments = relationship("MFMResourceAssignment", back_populates="work_order_execution")
	
	__table_args__ = (
		Index('idx_mfm_woe_tenant_status', 'tenant_id', 'status'),
		Index('idx_mfm_woe_work_order', 'work_order_id'),
		Index('idx_mfm_woe_product_facility', 'product_id', 'facility_id'),
		Index('idx_mfm_woe_timing', 'actual_start_time', 'actual_end_time'),
	)

class MFMProductionEvent(SQLBaseModel):
	"""Real-time production events and state changes"""
	__tablename__ = 'mfm_production_events'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Work order reference
	work_order_execution_id = Column(String(36), ForeignKey('mfm_work_order_executions.id'), nullable=False, index=True)
	
	# Event details
	event_type = Column(String(30), nullable=False)
	event_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
	event_duration_minutes = Column(Integer)
	
	# Location
	facility_id = Column(String(36), nullable=False, index=True)
	production_line_id = Column(String(36), index=True)
	work_center_id = Column(String(36), index=True)
	station_id = Column(String(36))
	
	# Personnel
	operator_id = Column(String(36))
	shift_supervisor_id = Column(String(36))
	
	# Event data
	previous_state = Column(String(50))
	new_state = Column(String(50))
	reason_code = Column(String(50))
	reason_description = Column(String(200))
	
	# Quantities affected
	quantity_processed = Column(Numeric(15, 4))
	quantity_good = Column(Numeric(15, 4))
	quantity_scrap = Column(Numeric(15, 4))
	
	# Process parameters at event time
	machine_speed = Column(Numeric(10, 2))
	temperature = Column(Numeric(8, 2))
	pressure = Column(Numeric(10, 4))
	quality_measurements = Column(Text)  # JSON object with measurements
	
	# Downtime details (if applicable)
	is_downtime_event = Column(Boolean, default=False)
	downtime_category = Column(String(50))  # planned, unplanned, changeover
	downtime_reason = Column(String(200))
	
	# Material information
	materials_consumed = Column(Text)  # JSON with materials used during event
	lot_numbers_processed = Column(String(500))
	
	# Comments and notes
	operator_comments = Column(Text)
	automatic_data_capture = Column(Text)  # JSON with automatically captured data
	
	# Audit trail
	recorded_by = Column(String(36))
	recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	
	# Relationships
	work_order_execution = relationship("MFMWorkOrderExecution", back_populates="production_events")
	
	__table_args__ = (
		Index('idx_mfm_pe_tenant_timestamp', 'tenant_id', 'event_timestamp'),
		Index('idx_mfm_pe_work_order', 'work_order_execution_id'),
		Index('idx_mfm_pe_type_facility', 'event_type', 'facility_id'),
		Index('idx_mfm_pe_downtime', 'is_downtime_event'),
	)

class MFMResourceStatus(SQLBaseModel):
	"""Real-time resource status tracking"""
	__tablename__ = 'mfm_resource_status'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Resource identification
	resource_id = Column(String(36), nullable=False, index=True)
	resource_type = Column(String(30), nullable=False)  # machine, operator, tool, equipment
	resource_name = Column(String(200), nullable=False)
	
	# Location
	facility_id = Column(String(36), nullable=False, index=True)
	production_line_id = Column(String(36), index=True)
	work_center_id = Column(String(36), index=True)
	
	# Current status
	current_status = Column(String(20), nullable=False)
	status_since = Column(DateTime, nullable=False)
	previous_status = Column(String(20))
	
	# Current assignment
	current_work_order_id = Column(String(36))
	current_operation = Column(String(200))
	assigned_operator = Column(String(36))
	
	# Performance metrics
	utilization_percentage = Column(Numeric(5, 2))
	efficiency_percentage = Column(Numeric(5, 2))
	availability_percentage = Column(Numeric(5, 2))
	
	# Counters (for machines)
	production_count_today = Column(Integer, default=0)
	cycle_count_total = Column(Integer, default=0)
	uptime_minutes_today = Column(Integer, default=0)
	downtime_minutes_today = Column(Integer, default=0)
	
	# Current process parameters (for machines)
	current_speed = Column(Numeric(10, 2))
	target_speed = Column(Numeric(10, 2))
	current_temperature = Column(Numeric(8, 2))
	current_pressure = Column(Numeric(10, 4))
	
	# Quality metrics
	quality_rate_today = Column(Numeric(5, 2))
	scrap_count_today = Column(Integer, default=0)
	rework_count_today = Column(Integer, default=0)
	
	# Maintenance information
	last_maintenance_date = Column(Date)
	next_maintenance_due = Column(Date)
	maintenance_status = Column(String(20))
	
	# Alerts and notifications
	active_alerts = Column(Text)  # JSON array of active alert IDs
	alert_count = Column(Integer, default=0)
	
	# Last update information
	last_data_update = Column(DateTime, nullable=False, default=datetime.utcnow)
	data_source = Column(String(50))  # manual, plc, scada, sensor
	
	# Audit trail
	updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('idx_mfm_rs_tenant_resource', 'tenant_id', 'resource_id'),
		Index('idx_mfm_rs_facility_status', 'facility_id', 'current_status'),
		Index('idx_mfm_rs_work_order', 'current_work_order_id'),
		Index('idx_mfm_rs_last_update', 'last_data_update'),
	)

class MFMResourceAssignment(SQLBaseModel):
	"""Resource assignments to work orders and operations"""
	__tablename__ = 'mfm_resource_assignments'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Work order reference
	work_order_execution_id = Column(String(36), ForeignKey('mfm_work_order_executions.id'), nullable=False, index=True)
	
	# Resource details
	resource_id = Column(String(36), nullable=False, index=True)
	resource_type = Column(String(30), nullable=False)
	resource_name = Column(String(200), nullable=False)
	
	# Assignment details
	operation_sequence = Column(Integer)
	operation_name = Column(String(200))
	
	# Timing
	assigned_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	planned_start_time = Column(DateTime)
	planned_end_time = Column(DateTime)
	actual_start_time = Column(DateTime)
	actual_end_time = Column(DateTime)
	
	# Assignment status
	assignment_status = Column(String(20), nullable=False, default="assigned")
	priority = Column(String(20), default="normal")
	
	# Setup requirements
	setup_time_minutes = Column(Integer)
	setup_required = Column(Boolean, default=False)
	setup_instructions = Column(Text)
	
	# Performance tracking
	planned_cycle_time = Column(Numeric(10, 2))
	actual_cycle_time = Column(Numeric(10, 2))
	pieces_produced = Column(Integer, default=0)
	
	# Personnel (for operators)
	operator_skill_level = Column(String(20))
	certification_required = Column(String(100))
	
	# Notes
	assignment_notes = Column(Text)
	completion_notes = Column(Text)
	
	# Audit trail
	assigned_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	work_order_execution = relationship("MFMWorkOrderExecution", back_populates="resource_assignments")
	
	__table_args__ = (
		Index('idx_mfm_ra_tenant_work_order', 'tenant_id', 'work_order_execution_id'),
		Index('idx_mfm_ra_resource', 'resource_id'),
		Index('idx_mfm_ra_timing', 'planned_start_time', 'planned_end_time'),
	)

class MFMPerformanceMetric(SQLBaseModel):
	"""Performance metrics and KPIs"""
	__tablename__ = 'mfm_performance_metrics'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Metric identification
	metric_type = Column(String(30), nullable=False)
	metric_name = Column(String(200), nullable=False)
	
	# Scope
	facility_id = Column(String(36), index=True)
	production_line_id = Column(String(36), index=True)
	work_center_id = Column(String(36), index=True)
	resource_id = Column(String(36), index=True)
	work_order_id = Column(String(36), index=True)
	
	# Time period
	measurement_date = Column(Date, nullable=False)
	measurement_hour = Column(Integer)  # 0-23 for hourly metrics
	measurement_timestamp = Column(DateTime, nullable=False)
	
	# Metric values
	metric_value = Column(Numeric(15, 4), nullable=False)
	target_value = Column(Numeric(15, 4))
	variance = Column(Numeric(15, 4))
	variance_percentage = Column(Numeric(8, 2))
	
	# Supporting data
	numerator = Column(Numeric(15, 4))
	denominator = Column(Numeric(15, 4))
	sample_size = Column(Integer)
	
	# Data quality
	data_quality_score = Column(Numeric(3, 2))  # 0-1 scale
	data_completeness_pct = Column(Numeric(5, 2))
	calculated_vs_actual = Column(String(20))  # calculated, actual, estimated
	
	# Context information
	shift_id = Column(String(36))
	product_id = Column(String(36))
	operator_id = Column(String(36))
	
	# Thresholds and alerts
	lower_control_limit = Column(Numeric(15, 4))
	upper_control_limit = Column(Numeric(15, 4))
	is_out_of_control = Column(Boolean, default=False)
	alert_generated = Column(Boolean, default=False)
	
	# Notes
	calculation_notes = Column(Text)
	
	# Audit trail
	calculated_by = Column(String(36))
	calculated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	
	__table_args__ = (
		Index('idx_mfm_pm_tenant_type', 'tenant_id', 'metric_type'),
		Index('idx_mfm_pm_facility_date', 'facility_id', 'measurement_date'),
		Index('idx_mfm_pm_timestamp', 'measurement_timestamp'),
		Index('idx_mfm_pm_out_of_control', 'is_out_of_control'),
	)

class MFMAlert(SQLBaseModel):
	"""System alerts and notifications"""
	__tablename__ = 'mfm_alerts'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Alert identification
	alert_code = Column(String(50), nullable=False)
	alert_title = Column(String(200), nullable=False)
	alert_description = Column(Text, nullable=False)
	
	# Alert classification
	alert_category = Column(String(50), nullable=False)  # production, quality, maintenance, safety
	severity = Column(String(20), nullable=False)
	priority = Column(String(20), default="medium")
	
	# Source information
	source_type = Column(String(30), nullable=False)  # manual, automatic, system
	source_system = Column(String(100))
	
	# Location
	facility_id = Column(String(36), index=True)
	production_line_id = Column(String(36), index=True)
	work_center_id = Column(String(36), index=True)
	resource_id = Column(String(36), index=True)
	
	# Work order context
	work_order_execution_id = Column(String(36), index=True)
	
	# Timing
	alert_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
	acknowledged_at = Column(DateTime)
	resolved_at = Column(DateTime)
	
	# Status
	alert_status = Column(String(20), nullable=False, default="active")
	is_acknowledged = Column(Boolean, default=False)
	is_resolved = Column(Boolean, default=False)
	
	# Personnel
	created_by = Column(String(36))
	acknowledged_by = Column(String(36))
	resolved_by = Column(String(36))
	assigned_to = Column(String(36))
	
	# Resolution
	resolution_action = Column(Text)
	resolution_notes = Column(Text)
	
	# Impact assessment
	production_impact = Column(String(20))  # none, low, medium, high
	quality_impact = Column(String(20))  # none, low, medium, high
	safety_impact = Column(String(20))  # none, low, medium, high
	
	# Related data
	metric_values = Column(Text)  # JSON with relevant metric data at time of alert
	process_parameters = Column(Text)  # JSON with process data
	
	# Escalation
	escalation_level = Column(Integer, default=0)
	escalated_at = Column(DateTime)
	escalated_to = Column(String(36))
	
	# Notifications sent
	notifications_sent = Column(Text)  # JSON array of notification records
	
	# Audit trail
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('idx_mfm_alert_tenant_status', 'tenant_id', 'alert_status'),
		Index('idx_mfm_alert_severity', 'severity'),
		Index('idx_mfm_alert_facility', 'facility_id'),
		Index('idx_mfm_alert_timestamp', 'alert_timestamp'),
		Index('idx_mfm_alert_unresolved', 'is_resolved', 'alert_timestamp'),
	)

# Pydantic Models for API

class WorkOrderExecutionCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	work_order_id: str = Field(..., min_length=36, max_length=36)
	work_order_number: str = Field(..., min_length=1, max_length=50)
	product_id: str = Field(..., min_length=36, max_length=36)
	product_sku: str = Field(..., min_length=1, max_length=100)
	product_name: str = Field(..., min_length=1, max_length=200)
	planned_quantity: Decimal = Field(..., gt=0)
	facility_id: str = Field(..., min_length=36, max_length=36)
	production_line_id: str | None = None
	work_center_id: str | None = None
	planned_start_time: datetime | None = None
	planned_end_time: datetime | None = None
	supervisor_id: str | None = None
	operators: str | None = None
	cycle_time_standard: Decimal | None = None
	target_speed: Decimal | None = None

class ProductionEventCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	event_type: ProductionEventType
	event_timestamp: datetime = Field(default_factory=datetime.utcnow)
	event_duration_minutes: int | None = None
	facility_id: str = Field(..., min_length=36, max_length=36)
	production_line_id: str | None = None
	work_center_id: str | None = None
	station_id: str | None = None
	operator_id: str | None = None
	shift_supervisor_id: str | None = None
	previous_state: str | None = None
	new_state: str | None = None
	reason_code: str | None = None
	reason_description: str | None = None
	quantity_processed: Decimal | None = None
	quantity_good: Decimal | None = None
	quantity_scrap: Decimal | None = None
	machine_speed: Decimal | None = None
	temperature: Decimal | None = None
	pressure: Decimal | None = None
	quality_measurements: str | None = None
	is_downtime_event: bool = False
	downtime_category: str | None = None
	downtime_reason: str | None = None
	materials_consumed: str | None = None
	lot_numbers_processed: str | None = None
	operator_comments: str | None = None
	automatic_data_capture: str | None = None

class AlertCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	alert_code: str = Field(..., min_length=1, max_length=50)
	alert_title: str = Field(..., min_length=1, max_length=200)
	alert_description: str = Field(..., min_length=1)
	alert_category: str = Field(..., min_length=1, max_length=50)
	severity: AlertSeverity
	priority: str = Field(default="medium", regex="^(low|medium|high|urgent|critical)$")
	source_type: str = Field(..., min_length=1, max_length=30)
	source_system: str | None = None
	facility_id: str | None = None
	production_line_id: str | None = None
	work_center_id: str | None = None
	resource_id: str | None = None
	work_order_execution_id: str | None = None
	alert_timestamp: datetime = Field(default_factory=datetime.utcnow)
	assigned_to: str | None = None
	production_impact: str | None = Field(None, regex="^(none|low|medium|high)$")
	quality_impact: str | None = Field(None, regex="^(none|low|medium|high)$")
	safety_impact: str | None = Field(None, regex="^(none|low|medium|high)$")
	metric_values: str | None = None
	process_parameters: str | None = None

__all__ = [
	"WorkOrderStatus", "ResourceStatus", "ProductionEventType", "PerformanceMetricType", "AlertSeverity",
	"MFMWorkOrderExecution", "MFMProductionEvent", "MFMResourceStatus", "MFMResourceAssignment",
	"MFMPerformanceMetric", "MFMAlert",
	"WorkOrderExecutionCreate", "ProductionEventCreate", "AlertCreate"
]