"""
Material Requirements Planning (MRP) Models

Database models for MRP functionality including MRP runs, material requirements,
planned orders, and inventory availability analysis.
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

class MRPRunStatus(str, Enum):
	"""MRP run status enumeration"""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"

class RequirementType(str, Enum):
	"""Material requirement type"""
	GROSS_REQUIREMENT = "gross_requirement"
	SCHEDULED_RECEIPT = "scheduled_receipt"
	PROJECTED_AVAILABLE = "projected_available"
	NET_REQUIREMENT = "net_requirement"
	PLANNED_ORDER = "planned_order"

class PlannedOrderType(str, Enum):
	"""Planned order type enumeration"""
	PURCHASE_ORDER = "purchase_order"
	WORK_ORDER = "work_order"
	TRANSFER_ORDER = "transfer_order"

class MRPExceptionType(str, Enum):
	"""MRP exception message types"""
	INSUFFICIENT_INVENTORY = "insufficient_inventory"
	LATE_DELIVERY = "late_delivery"
	CAPACITY_SHORTAGE = "capacity_shortage"
	BOM_MISSING = "bom_missing"
	VENDOR_ISSUE = "vendor_issue"
	LEAD_TIME_CHANGE = "lead_time_change"

# SQLAlchemy Models

class MFMRPRun(SQLBaseModel):
	"""MRP run execution tracking"""
	__tablename__ = 'mfm_mrp_runs'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Run identification
	run_name = Column(String(200), nullable=False)
	run_number = Column(String(50), nullable=False, unique=True)
	planning_horizon_days = Column(Integer, nullable=False, default=90)
	
	# Run scope
	facility_id = Column(String(36), index=True)
	product_filter = Column(String(200))  # Optional product filter criteria
	include_safety_stock = Column(Boolean, default=True)
	include_forecast = Column(Boolean, default=True)
	
	# Run parameters
	cutoff_date = Column(Date, nullable=False)
	regenerative_run = Column(Boolean, default=True)  # vs net change
	explosion_method = Column(String(30), default="low_level_coded")
	
	# Execution tracking
	status = Column(String(20), nullable=False, default="pending")
	start_time = Column(DateTime)
	end_time = Column(DateTime)
	duration_seconds = Column(Integer)
	
	# Results summary
	materials_processed = Column(Integer, default=0)
	planned_orders_created = Column(Integer, default=0)
	exceptions_generated = Column(Integer, default=0)
	
	# Messages and logs
	execution_log = Column(Text)
	error_message = Column(Text)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	material_requirements = relationship("MFMMaterialRequirement", back_populates="mrp_run")
	planned_orders = relationship("MFMPlannedOrder", back_populates="mrp_run")
	exceptions = relationship("MFMRPException", back_populates="mrp_run")
	
	__table_args__ = (
		Index('idx_mfm_mrp_tenant_status', 'tenant_id', 'status'),
		Index('idx_mfm_mrp_run_number', 'run_number'),
		Index('idx_mfm_mrp_cutoff_date', 'cutoff_date'),
	)

class MFMMaterialRequirement(SQLBaseModel):
	"""Material requirements calculation results"""
	__tablename__ = 'mfm_material_requirements'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# MRP run reference
	mrp_run_id = Column(String(36), ForeignKey('mfm_mrp_runs.id'), nullable=False, index=True)
	
	# Material identification
	material_id = Column(String(36), nullable=False, index=True)
	material_sku = Column(String(100), nullable=False)
	material_name = Column(String(200), nullable=False)
	facility_id = Column(String(36), nullable=False, index=True)
	
	# Time bucket
	requirement_date = Column(Date, nullable=False)
	time_bucket = Column(String(20), nullable=False)  # daily, weekly, monthly
	
	# Requirement details
	requirement_type = Column(String(30), nullable=False)
	gross_requirement = Column(Numeric(15, 4), default=0)
	scheduled_receipt = Column(Numeric(15, 4), default=0)
	projected_available = Column(Numeric(15, 4), default=0)
	net_requirement = Column(Numeric(15, 4), default=0)
	planned_order_quantity = Column(Numeric(15, 4), default=0)
	
	# Current inventory
	beginning_inventory = Column(Numeric(15, 4), default=0)
	safety_stock = Column(Numeric(15, 4), default=0)
	allocated_quantity = Column(Numeric(15, 4), default=0)
	available_to_promise = Column(Numeric(15, 4), default=0)
	
	# Planning parameters used
	lot_size_rule = Column(String(30))  # lot_for_lot, fixed_quantity, economic_order_quantity
	lot_size_quantity = Column(Numeric(15, 4))
	lead_time_days = Column(Integer, default=0)
	safety_lead_time_days = Column(Integer, default=0)
	
	# Source information
	source_requirement_type = Column(String(50))  # master_schedule, dependent_demand, forecast
	source_order_id = Column(String(36))
	parent_material_id = Column(String(36))
	
	# Audit trail
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	
	# Relationships
	mrp_run = relationship("MFMRPRun", back_populates="material_requirements")
	
	__table_args__ = (
		Index('idx_mfm_mr_tenant_run', 'tenant_id', 'mrp_run_id'),
		Index('idx_mfm_mr_material_facility', 'material_id', 'facility_id'),
		Index('idx_mfm_mr_requirement_date', 'requirement_date'),
	)

class MFMPlannedOrder(SQLBaseModel):
	"""Planned orders generated by MRP"""
	__tablename__ = 'mfm_planned_orders'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# MRP run reference
	mrp_run_id = Column(String(36), ForeignKey('mfm_mrp_runs.id'), nullable=False, index=True)
	
	# Order identification
	planned_order_number = Column(String(50), nullable=False, unique=True)
	order_type = Column(String(30), nullable=False)
	
	# Material details
	material_id = Column(String(36), nullable=False, index=True)
	material_sku = Column(String(100), nullable=False)
	material_name = Column(String(200), nullable=False)
	facility_id = Column(String(36), nullable=False, index=True)
	
	# Order quantities and dates
	planned_quantity = Column(Numeric(15, 4), nullable=False)
	planned_order_date = Column(Date, nullable=False)
	planned_receipt_date = Column(Date, nullable=False)
	
	# Source requirement
	source_requirement_id = Column(String(36), ForeignKey('mfm_material_requirements.id'))
	due_date = Column(Date, nullable=False)
	
	# Supplier/vendor information (for purchase orders)
	preferred_supplier_id = Column(String(36))
	supplier_part_number = Column(String(100))
	unit_cost = Column(Numeric(12, 4))
	extended_cost = Column(Numeric(15, 2))
	
	# Manufacturing information (for work orders)
	routing_id = Column(String(36))
	work_center_id = Column(String(36))
	estimated_hours = Column(Numeric(10, 2))
	
	# Planning parameters
	lot_size_used = Column(Numeric(15, 4))
	lead_time_used = Column(Integer)
	safety_lead_time_used = Column(Integer)
	
	# Conversion status
	is_converted = Column(Boolean, default=False)
	converted_order_id = Column(String(36))  # Reference to actual PO/WO
	converted_at = Column(DateTime)
	converted_by = Column(String(36))
	
	# Status and priority
	status = Column(String(20), nullable=False, default="active")
	priority = Column(String(20), default="normal")
	
	# Notes
	planning_notes = Column(Text)
	
	# Audit trail
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	mrp_run = relationship("MFMRPRun", back_populates="planned_orders")
	
	__table_args__ = (
		Index('idx_mfm_po_tenant_run', 'tenant_id', 'mrp_run_id'),
		Index('idx_mfm_po_material_facility', 'material_id', 'facility_id'),
		Index('idx_mfm_po_dates', 'planned_order_date', 'planned_receipt_date'),
		Index('idx_mfm_po_number', 'planned_order_number'),
	)

class MFMRPException(SQLBaseModel):
	"""MRP exception messages and alerts"""
	__tablename__ = 'mfm_mrp_exceptions'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# MRP run reference
	mrp_run_id = Column(String(36), ForeignKey('mfm_mrp_runs.id'), nullable=False, index=True)
	
	# Exception details
	exception_type = Column(String(30), nullable=False)
	severity = Column(String(20), nullable=False, default="medium")  # low, medium, high, critical
	exception_code = Column(String(20))
	
	# Affected entities
	material_id = Column(String(36), index=True)
	facility_id = Column(String(36), index=True)
	planned_order_id = Column(String(36))
	supplier_id = Column(String(36))
	
	# Exception message
	message_title = Column(String(200), nullable=False)
	message_description = Column(Text, nullable=False)
	
	# Timing information
	exception_date = Column(Date, nullable=False)
	required_date = Column(Date)
	available_date = Column(Date)
	shortage_quantity = Column(Numeric(15, 4))
	
	# Resolution tracking
	is_resolved = Column(Boolean, default=False)
	resolution_action = Column(String(200))
	resolved_by = Column(String(36))
	resolved_at = Column(DateTime)
	resolution_notes = Column(Text)
	
	# Audit trail
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	
	# Relationships
	mrp_run = relationship("MFMRPRun", back_populates="exceptions")
	
	__table_args__ = (
		Index('idx_mfm_ex_tenant_run', 'tenant_id', 'mrp_run_id'),
		Index('idx_mfm_ex_type_severity', 'exception_type', 'severity'),
		Index('idx_mfm_ex_material_facility', 'material_id', 'facility_id'),
		Index('idx_mfm_ex_resolved', 'is_resolved'),
	)

class MFMInventoryPosition(SQLBaseModel):
	"""Current inventory positions for MRP calculations"""
	__tablename__ = 'mfm_inventory_positions'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Material identification
	material_id = Column(String(36), nullable=False, index=True)
	material_sku = Column(String(100), nullable=False)
	facility_id = Column(String(36), nullable=False, index=True)
	location_id = Column(String(36))
	
	# Inventory quantities
	on_hand_quantity = Column(Numeric(15, 4), default=0)
	allocated_quantity = Column(Numeric(15, 4), default=0)
	available_quantity = Column(Numeric(15, 4), default=0)
	in_transit_quantity = Column(Numeric(15, 4), default=0)
	on_order_quantity = Column(Numeric(15, 4), default=0)
	
	# Planning parameters
	safety_stock_quantity = Column(Numeric(15, 4), default=0)
	reorder_point = Column(Numeric(15, 4), default=0)
	economic_order_quantity = Column(Numeric(15, 4))
	
	# Lead times
	purchase_lead_time_days = Column(Integer, default=0)
	manufacturing_lead_time_days = Column(Integer, default=0)
	safety_lead_time_days = Column(Integer, default=0)
	
	# Cost information
	standard_cost = Column(Numeric(12, 4))
	last_cost = Column(Numeric(12, 4))
	average_cost = Column(Numeric(12, 4))
	
	# Planning flags
	is_mrp_controlled = Column(Boolean, default=True)
	make_or_buy = Column(String(10), default="buy")  # make, buy
	abc_classification = Column(String(1))  # A, B, C
	
	# Last update information
	last_counted_date = Column(Date)
	last_transaction_date = Column(DateTime)
	
	# Audit trail
	updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('idx_mfm_ip_tenant_material', 'tenant_id', 'material_id'),
		Index('idx_mfm_ip_facility_material', 'facility_id', 'material_id'),
		Index('idx_mfm_ip_mrp_controlled', 'is_mrp_controlled'),
	)

# Pydantic Models for API

class MRPRunCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	run_name: str = Field(..., min_length=1, max_length=200)
	planning_horizon_days: int = Field(default=90, ge=1, le=365)
	facility_id: str | None = None
	product_filter: str | None = None
	include_safety_stock: bool = True
	include_forecast: bool = True
	cutoff_date: date
	regenerative_run: bool = True
	explosion_method: str = Field(default="low_level_coded", max_length=30)

class PlannedOrderCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	order_type: PlannedOrderType
	material_id: str = Field(..., min_length=36, max_length=36)
	material_sku: str = Field(..., min_length=1, max_length=100)
	material_name: str = Field(..., min_length=1, max_length=200)
	facility_id: str = Field(..., min_length=36, max_length=36)
	planned_quantity: Decimal = Field(..., gt=0)
	planned_order_date: date
	planned_receipt_date: date
	due_date: date
	preferred_supplier_id: str | None = None
	supplier_part_number: str | None = None
	unit_cost: Decimal | None = None
	routing_id: str | None = None
	work_center_id: str | None = None
	estimated_hours: Decimal | None = None
	priority: str = Field(default="normal", max_length=20)
	planning_notes: str | None = None

class MRPExceptionCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	exception_type: MRPExceptionType
	severity: str = Field(..., regex="^(low|medium|high|critical)$")
	exception_code: str | None = None
	material_id: str | None = None
	facility_id: str | None = None
	planned_order_id: str | None = None
	supplier_id: str | None = None
	message_title: str = Field(..., min_length=1, max_length=200)
	message_description: str = Field(..., min_length=1)
	exception_date: date
	required_date: date | None = None
	available_date: date | None = None
	shortage_quantity: Decimal | None = None

class InventoryPositionUpdate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	material_id: str = Field(..., min_length=36, max_length=36)
	material_sku: str = Field(..., min_length=1, max_length=100)
	facility_id: str = Field(..., min_length=36, max_length=36)
	location_id: str | None = None
	on_hand_quantity: Decimal = Field(..., ge=0)
	allocated_quantity: Decimal = Field(default=Decimal('0'), ge=0)
	in_transit_quantity: Decimal = Field(default=Decimal('0'), ge=0)
	on_order_quantity: Decimal = Field(default=Decimal('0'), ge=0)
	safety_stock_quantity: Decimal = Field(default=Decimal('0'), ge=0)
	reorder_point: Decimal | None = None
	economic_order_quantity: Decimal | None = None
	purchase_lead_time_days: int = Field(default=0, ge=0)
	manufacturing_lead_time_days: int = Field(default=0, ge=0)
	safety_lead_time_days: int = Field(default=0, ge=0)
	standard_cost: Decimal | None = None
	make_or_buy: str = Field(default="buy", regex="^(make|buy)$")
	abc_classification: str | None = Field(None, regex="^[ABC]$")
	is_mrp_controlled: bool = True

__all__ = [
	"MRPRunStatus", "RequirementType", "PlannedOrderType", "MRPExceptionType",
	"MFMRPRun", "MFMMaterialRequirement", "MFMPlannedOrder", "MFMRPException", "MFMInventoryPosition",
	"MRPRunCreate", "PlannedOrderCreate", "MRPExceptionCreate", "InventoryPositionUpdate"
]