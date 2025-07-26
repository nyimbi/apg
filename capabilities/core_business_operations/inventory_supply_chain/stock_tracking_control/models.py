"""
Stock Tracking & Control Models

Database models for inventory items, locations, stock levels, movements,
and real-time tracking functionality.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ....auth_rbac.models import BaseMixin, AuditMixin, Model


class IMSTCItemCategory(Model, AuditMixin, BaseMixin):
	"""
	Item category classification for inventory organization.
	
	Supports hierarchical categorization for better inventory organization
	and reporting across different business units.
	"""
	__tablename__ = 'im_stc_item_category'
	
	# Identity
	category_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Category Information
	category_code = Column(String(20), nullable=False, index=True)
	category_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Hierarchy
	parent_category_id = Column(String(36), ForeignKey('im_stc_item_category.category_id'), nullable=True, index=True)
	level = Column(Integer, default=0)
	path = Column(String(500), nullable=True)  # Full path for efficient queries
	
	# Properties
	is_active = Column(Boolean, default=True)
	sort_order = Column(Integer, default=0)
	
	# Business Rules
	requires_serial_tracking = Column(Boolean, default=False)
	requires_lot_tracking = Column(Boolean, default=False)
	requires_expiry_tracking = Column(Boolean, default=False)
	abc_classification = Column(String(1), nullable=True)  # A, B, C analysis
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'category_code', name='uq_category_code_tenant'),
		Index('idx_category_hierarchy', 'tenant_id', 'parent_category_id', 'level'),
	)
	
	# Relationships
	parent_category = relationship("IMSTCItemCategory", remote_side=[category_id])
	child_categories = relationship("IMSTCItemCategory")
	items = relationship("IMSTCItem", back_populates="category")
	
	def __repr__(self):
		return f"<IMSTCItemCategory {self.category_name}>"


class IMSTCUnitOfMeasure(Model, AuditMixin, BaseMixin):
	"""
	Units of measure for inventory items.
	
	Supports conversion between different units for the same item.
	"""
	__tablename__ = 'im_stc_unit_of_measure'
	
	# Identity
	uom_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# UOM Information
	uom_code = Column(String(10), nullable=False, index=True)
	uom_name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	
	# Classification
	uom_type = Column(String(20), nullable=False)  # Weight, Volume, Length, Count, etc.
	base_unit_id = Column(String(36), ForeignKey('im_stc_unit_of_measure.uom_id'), nullable=True, index=True)
	conversion_factor = Column(DECIMAL(15, 6), default=1.0)  # To base unit
	
	# Properties
	is_active = Column(Boolean, default=True)
	is_base_unit = Column(Boolean, default=False)
	decimal_places = Column(Integer, default=2)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'uom_code', name='uq_uom_code_tenant'),
	)
	
	# Relationships
	base_unit = relationship("IMSTCUnitOfMeasure", remote_side=[uom_id])
	conversion_units = relationship("IMSTCUnitOfMeasure")
	items = relationship("IMSTCItem", back_populates="primary_uom")
	stock_levels = relationship("IMSTCStockLevel", back_populates="uom")
	
	def __repr__(self):
		return f"<IMSTCUnitOfMeasure {self.uom_code}>"


class IMSTCWarehouse(Model, AuditMixin, BaseMixin):
	"""
	Physical warehouses and storage facilities.
	
	Represents the highest level of inventory location hierarchy.
	"""
	__tablename__ = 'im_stc_warehouse'
	
	# Identity
	warehouse_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Warehouse Information
	warehouse_code = Column(String(20), nullable=False, index=True)
	warehouse_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Location Information
	address_line1 = Column(String(200), nullable=True)
	address_line2 = Column(String(200), nullable=True)
	city = Column(String(100), nullable=True)
	state_province = Column(String(100), nullable=True)
	postal_code = Column(String(20), nullable=True)
	country_code = Column(String(3), nullable=True)
	
	# Geographic Coordinates
	latitude = Column(DECIMAL(10, 8), nullable=True)
	longitude = Column(DECIMAL(11, 8), nullable=True)
	
	# Properties
	is_active = Column(Boolean, default=True)
	is_primary = Column(Boolean, default=False)
	warehouse_type = Column(String(50), nullable=True)  # Distribution, Manufacturing, etc.
	
	# Configuration
	allows_negative_stock = Column(Boolean, default=False)
	auto_allocate_stock = Column(Boolean, default=True)
	temperature_controlled = Column(Boolean, default=False)
	min_temperature = Column(DECIMAL(5, 2), nullable=True)
	max_temperature = Column(DECIMAL(5, 2), nullable=True)
	
	# Contact Information
	manager_name = Column(String(200), nullable=True)
	phone_number = Column(String(50), nullable=True)
	email_address = Column(String(200), nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'warehouse_code', name='uq_warehouse_code_tenant'),
	)
	
	# Relationships
	locations = relationship("IMSTCLocation", back_populates="warehouse")
	stock_levels = relationship("IMSTCStockLevel", back_populates="warehouse")
	stock_movements = relationship("IMSTCStockMovement", back_populates="warehouse")
	
	def __repr__(self):
		return f"<IMSTCWarehouse {self.warehouse_name}>"


class IMSTCLocation(Model, AuditMixin, BaseMixin):
	"""
	Specific storage locations within warehouses.
	
	Supports hierarchical locations (Zone > Aisle > Shelf > Bin).
	"""
	__tablename__ = 'im_stc_location'
	
	# Identity
	location_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Location Information
	location_code = Column(String(50), nullable=False, index=True)
	location_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Hierarchy
	warehouse_id = Column(String(36), ForeignKey('im_stc_warehouse.warehouse_id'), nullable=False, index=True)
	parent_location_id = Column(String(36), ForeignKey('im_stc_location.location_id'), nullable=True, index=True)
	level = Column(Integer, default=0)  # 0=Zone, 1=Aisle, 2=Shelf, 3=Bin
	path = Column(String(500), nullable=True)
	
	# Physical Properties
	location_type = Column(String(50), nullable=True)  # Zone, Aisle, Shelf, Bin, etc.
	capacity_volume = Column(DECIMAL(15, 4), nullable=True)
	capacity_weight = Column(DECIMAL(15, 4), nullable=True)
	length = Column(DECIMAL(10, 4), nullable=True)
	width = Column(DECIMAL(10, 4), nullable=True)
	height = Column(DECIMAL(10, 4), nullable=True)
	
	# Properties
	is_active = Column(Boolean, default=True)
	is_pickable = Column(Boolean, default=True)
	is_receivable = Column(Boolean, default=True)
	is_quarantine = Column(Boolean, default=False)
	is_damaged_goods = Column(Boolean, default=False)
	
	# Environment Controls
	temperature_controlled = Column(Boolean, default=False)
	min_temperature = Column(DECIMAL(5, 2), nullable=True)
	max_temperature = Column(DECIMAL(5, 2), nullable=True)
	humidity_controlled = Column(Boolean, default=False)
	max_humidity = Column(DECIMAL(5, 2), nullable=True)
	
	# Picking Properties
	pick_sequence = Column(Integer, default=0)
	put_sequence = Column(Integer, default=0)
	abc_zone = Column(String(1), nullable=True)  # A, B, C for velocity-based slotting
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'warehouse_id', 'location_code', name='uq_location_code_warehouse'),
		Index('idx_location_hierarchy', 'tenant_id', 'warehouse_id', 'parent_location_id'),
		Index('idx_location_pick_sequence', 'warehouse_id', 'pick_sequence'),
	)
	
	# Relationships
	warehouse = relationship("IMSTCWarehouse", back_populates="locations")
	parent_location = relationship("IMSTCLocation", remote_side=[location_id])
	child_locations = relationship("IMSTCLocation")
	stock_levels = relationship("IMSTCStockLevel", back_populates="location")
	stock_movements = relationship("IMSTCStockMovement", back_populates="location")
	
	def __repr__(self):
		return f"<IMSTCLocation {self.location_code}>"


class IMSTCItem(Model, AuditMixin, BaseMixin):
	"""
	Master inventory items that can be stocked and tracked.
	
	Central repository for all item master data including product information,
	classification, and tracking requirements.
	"""
	__tablename__ = 'im_stc_item'
	
	# Identity
	item_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Item Information
	item_code = Column(String(50), nullable=False, index=True)
	item_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	short_description = Column(String(500), nullable=True)
	
	# Classification
	category_id = Column(String(36), ForeignKey('im_stc_item_category.category_id'), nullable=True, index=True)
	item_type = Column(String(50), nullable=False, default='Finished Goods')  # Raw Material, WIP, Finished Goods, etc.
	abc_classification = Column(String(1), nullable=True)  # A, B, C analysis
	
	# Physical Properties
	primary_uom_id = Column(String(36), ForeignKey('im_stc_unit_of_measure.uom_id'), nullable=False, index=True)
	weight = Column(DECIMAL(15, 4), nullable=True)
	weight_uom = Column(String(10), nullable=True)
	volume = Column(DECIMAL(15, 4), nullable=True)
	volume_uom = Column(String(10), nullable=True)
	length = Column(DECIMAL(10, 4), nullable=True)
	width = Column(DECIMAL(10, 4), nullable=True)
	height = Column(DECIMAL(10, 4), nullable=True)
	dimension_uom = Column(String(10), nullable=True)
	
	# Tracking Requirements
	requires_serial_tracking = Column(Boolean, default=False)
	requires_lot_tracking = Column(Boolean, default=False)
	requires_expiry_tracking = Column(Boolean, default=False)
	shelf_life_days = Column(Integer, nullable=True)
	
	# Stock Control Parameters
	default_warehouse_id = Column(String(36), ForeignKey('im_stc_warehouse.warehouse_id'), nullable=True, index=True)
	default_location_id = Column(String(36), ForeignKey('im_stc_location.location_id'), nullable=True, index=True)
	min_stock_level = Column(DECIMAL(15, 4), default=0)
	max_stock_level = Column(DECIMAL(15, 4), nullable=True)
	reorder_point = Column(DECIMAL(15, 4), default=0)
	reorder_quantity = Column(DECIMAL(15, 4), nullable=True)
	safety_stock = Column(DECIMAL(15, 4), default=0)
	
	# Costing Information
	standard_cost = Column(DECIMAL(15, 4), default=0)
	last_cost = Column(DECIMAL(15, 4), default=0)
	average_cost = Column(DECIMAL(15, 4), default=0)
	cost_method = Column(String(20), default='Average')  # FIFO, LIFO, Average, Standard
	
	# Properties
	is_active = Column(Boolean, default=True)
	is_sellable = Column(Boolean, default=True)
	is_purchasable = Column(Boolean, default=True)
	is_serialized = Column(Boolean, default=False)
	is_lot_controlled = Column(Boolean, default=False)
	is_perishable = Column(Boolean, default=False)
	is_hazardous = Column(Boolean, default=False)
	
	# Quality Control
	requires_inspection = Column(Boolean, default=False)
	inspection_type = Column(String(50), nullable=True)  # Random, Full, Statistical
	acceptable_quality_level = Column(DECIMAL(5, 2), nullable=True)  # AQL percentage
	
	# Regulatory
	regulatory_class = Column(String(50), nullable=True)
	controlled_substance = Column(Boolean, default=False)
	requires_certification = Column(Boolean, default=False)
	certification_type = Column(String(100), nullable=True)
	
	# Storage Requirements
	storage_temperature_min = Column(DECIMAL(5, 2), nullable=True)
	storage_temperature_max = Column(DECIMAL(5, 2), nullable=True)
	storage_humidity_max = Column(DECIMAL(5, 2), nullable=True)
	special_handling_instructions = Column(Text, nullable=True)
	
	# External References
	manufacturer_part_number = Column(String(100), nullable=True)
	supplier_part_number = Column(String(100), nullable=True)
	customer_part_number = Column(String(100), nullable=True)
	
	# Additional Attributes (JSON for flexibility)
	custom_attributes = Column(Text, nullable=True)  # JSON stored as text
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'item_code', name='uq_item_code_tenant'),
		Index('idx_item_category_type', 'tenant_id', 'category_id', 'item_type'),
		Index('idx_item_tracking', 'tenant_id', 'requires_serial_tracking', 'requires_lot_tracking'),
		Index('idx_item_abc', 'tenant_id', 'abc_classification'),
	)
	
	# Relationships
	category = relationship("IMSTCItemCategory", back_populates="items")
	primary_uom = relationship("IMSTCUnitOfMeasure", back_populates="items")
	default_warehouse = relationship("IMSTCWarehouse")
	default_location = relationship("IMSTCLocation")
	stock_levels = relationship("IMSTCStockLevel", back_populates="item")
	stock_movements = relationship("IMSTCStockMovement", back_populates="item")
	
	def __repr__(self):
		return f"<IMSTCItem {self.item_code}: {self.item_name}>"
	
	def get_custom_attributes(self) -> Dict[str, Any]:
		"""Get custom attributes as dictionary"""
		if self.custom_attributes:
			try:
				return json.loads(self.custom_attributes)
			except (json.JSONDecodeError, TypeError):
				return {}
		return {}
	
	def set_custom_attributes(self, attributes: Dict[str, Any]):
		"""Set custom attributes from dictionary"""
		self.custom_attributes = json.dumps(attributes) if attributes else None


class IMSTCStockLevel(Model, AuditMixin, BaseMixin):
	"""
	Current stock levels by item, location, and additional dimensions.
	
	Maintains real-time inventory balances with support for multiple
	inventory dimensions including lot, serial, status, etc.
	"""
	__tablename__ = 'im_stc_stock_level'
	
	# Identity
	stock_level_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Stock Dimensions
	item_id = Column(String(36), ForeignKey('im_stc_item.item_id'), nullable=False, index=True)
	warehouse_id = Column(String(36), ForeignKey('im_stc_warehouse.warehouse_id'), nullable=False, index=True)
	location_id = Column(String(36), ForeignKey('im_stc_location.location_id'), nullable=False, index=True)
	uom_id = Column(String(36), ForeignKey('im_stc_unit_of_measure.uom_id'), nullable=False, index=True)
	
	# Optional Dimensions
	lot_number = Column(String(100), nullable=True, index=True)
	serial_number = Column(String(100), nullable=True, index=True)
	batch_number = Column(String(100), nullable=True, index=True)
	expiry_date = Column(Date, nullable=True, index=True)
	
	# Stock Status
	stock_status = Column(String(50), default='Available', index=True)  # Available, Allocated, On Hold, Quarantine, Damaged
	quality_status = Column(String(50), default='Approved', index=True)  # Approved, Pending, Rejected, Quarantine
	
	# Quantities
	on_hand_quantity = Column(DECIMAL(15, 4), default=0, nullable=False)
	allocated_quantity = Column(DECIMAL(15, 4), default=0, nullable=False)
	available_quantity = Column(DECIMAL(15, 4), default=0, nullable=False)  # on_hand - allocated
	on_order_quantity = Column(DECIMAL(15, 4), default=0, nullable=False)
	reserved_quantity = Column(DECIMAL(15, 4), default=0, nullable=False)
	
	# Cost Information
	unit_cost = Column(DECIMAL(15, 4), default=0)
	total_cost = Column(DECIMAL(15, 4), default=0)
	last_cost_update = Column(DateTime, nullable=True)
	
	# Dates
	first_received_date = Column(DateTime, nullable=True)
	last_received_date = Column(DateTime, nullable=True)
	last_issued_date = Column(DateTime, nullable=True)
	last_counted_date = Column(DateTime, nullable=True)
	
	# Physical Properties
	container_number = Column(String(100), nullable=True)
	pallet_number = Column(String(100), nullable=True)
	
	# Additional Attributes
	custom_attributes = Column(Text, nullable=True)  # JSON for flexibility
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'item_id', 'warehouse_id', 'location_id', 
						'lot_number', 'serial_number', 'stock_status', name='uq_stock_level_dimensions'),
		Index('idx_stock_level_item_location', 'tenant_id', 'item_id', 'warehouse_id', 'location_id'),
		Index('idx_stock_level_lot_serial', 'tenant_id', 'lot_number', 'serial_number'),
		Index('idx_stock_level_expiry', 'tenant_id', 'expiry_date'),
		Index('idx_stock_level_status', 'tenant_id', 'stock_status', 'quality_status'),
	)
	
	# Relationships
	item = relationship("IMSTCItem", back_populates="stock_levels")
	warehouse = relationship("IMSTCWarehouse", back_populates="stock_levels")
	location = relationship("IMSTCLocation", back_populates="stock_levels")
	uom = relationship("IMSTCUnitOfMeasure", back_populates="stock_levels")
	
	def __repr__(self):
		return f"<IMSTCStockLevel {self.item.item_code if self.item else 'Unknown'} @ {self.location.location_code if self.location else 'Unknown'}: {self.on_hand_quantity}>"
	
	def get_custom_attributes(self) -> Dict[str, Any]:
		"""Get custom attributes as dictionary"""
		if self.custom_attributes:
			try:
				return json.loads(self.custom_attributes)
			except (json.JSONDecodeError, TypeError):
				return {}
		return {}
	
	def set_custom_attributes(self, attributes: Dict[str, Any]):
		"""Set custom attributes from dictionary"""
		self.custom_attributes = json.dumps(attributes) if attributes else None


class IMSTCStockMovement(Model, AuditMixin, BaseMixin):
	"""
	Detailed record of all inventory movements and transactions.
	
	Provides complete audit trail for inventory changes including
	receipts, issues, transfers, adjustments, and cycle counts.
	"""
	__tablename__ = 'im_stc_stock_movement'
	
	# Identity
	movement_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Movement Information
	movement_type = Column(String(50), nullable=False, index=True)  # Receipt, Issue, Transfer, Adjustment, Count
	movement_subtype = Column(String(50), nullable=True)  # Purchase Receipt, Sales Issue, etc.
	reference_number = Column(String(100), nullable=True, index=True)
	reference_type = Column(String(50), nullable=True)  # PO, SO, TO, etc.
	movement_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
	
	# Stock Dimensions
	item_id = Column(String(36), ForeignKey('im_stc_item.item_id'), nullable=False, index=True)
	warehouse_id = Column(String(36), ForeignKey('im_stc_warehouse.warehouse_id'), nullable=False, index=True)
	location_id = Column(String(36), ForeignKey('im_stc_location.location_id'), nullable=False, index=True)
	uom_id = Column(String(36), ForeignKey('im_stc_unit_of_measure.uom_id'), nullable=False, index=True)
	
	# Optional Dimensions
	lot_number = Column(String(100), nullable=True, index=True)
	serial_number = Column(String(100), nullable=True, index=True)
	batch_number = Column(String(100), nullable=True, index=True)
	expiry_date = Column(Date, nullable=True)
	
	# From/To for Transfers
	from_warehouse_id = Column(String(36), ForeignKey('im_stc_warehouse.warehouse_id'), nullable=True, index=True)
	from_location_id = Column(String(36), ForeignKey('im_stc_location.location_id'), nullable=True, index=True)
	to_warehouse_id = Column(String(36), ForeignKey('im_stc_warehouse.warehouse_id'), nullable=True, index=True)
	to_location_id = Column(String(36), ForeignKey('im_stc_location.location_id'), nullable=True, index=True)
	
	# Quantities and Costs
	quantity = Column(DECIMAL(15, 4), nullable=False)
	unit_cost = Column(DECIMAL(15, 4), default=0)
	total_cost = Column(DECIMAL(15, 4), default=0)
	
	# Stock Status
	stock_status = Column(String(50), default='Available')
	quality_status = Column(String(50), default='Approved')
	
	# Running Balances (for performance)
	running_balance = Column(DECIMAL(15, 4), nullable=True)
	running_value = Column(DECIMAL(15, 4), nullable=True)
	
	# Transaction Information
	transaction_source = Column(String(100), nullable=True)  # System that originated transaction
	user_id = Column(String(36), nullable=True, index=True)
	reason_code = Column(String(50), nullable=True)
	notes = Column(Text, nullable=True)
	
	# Status
	status = Column(String(50), default='Posted', index=True)  # Posted, Pending, Cancelled
	posted_date = Column(DateTime, nullable=True)
	posted_by = Column(String(36), nullable=True)
	
	# Additional Attributes
	custom_attributes = Column(Text, nullable=True)  # JSON for flexibility
	
	# Constraints
	__table_args__ = (
		Index('idx_movement_item_date', 'tenant_id', 'item_id', 'movement_date'),
		Index('idx_movement_location_date', 'tenant_id', 'warehouse_id', 'location_id', 'movement_date'),
		Index('idx_movement_reference', 'tenant_id', 'reference_number', 'reference_type'),
		Index('idx_movement_lot_serial', 'tenant_id', 'lot_number', 'serial_number'),
		Index('idx_movement_type_date', 'tenant_id', 'movement_type', 'movement_date'),
	)
	
	# Relationships
	item = relationship("IMSTCItem", back_populates="stock_movements")
	warehouse = relationship("IMSTCWarehouse", back_populates="stock_movements")
	location = relationship("IMSTCLocation", back_populates="stock_movements")
	uom = relationship("IMSTCUnitOfMeasure")
	
	# From/To relationships for transfers
	from_warehouse = relationship("IMSTCWarehouse", foreign_keys=[from_warehouse_id])
	from_location = relationship("IMSTCLocation", foreign_keys=[from_location_id])
	to_warehouse = relationship("IMSTCWarehouse", foreign_keys=[to_warehouse_id])
	to_location = relationship("IMSTCLocation", foreign_keys=[to_location_id])
	
	def __repr__(self):
		return f"<IMSTCStockMovement {self.movement_type}: {self.quantity} {self.item.item_code if self.item else 'Unknown'}>"
	
	def get_custom_attributes(self) -> Dict[str, Any]:
		"""Get custom attributes as dictionary"""
		if self.custom_attributes:
			try:
				return json.loads(self.custom_attributes)
			except (json.JSONDecodeError, TypeError):
				return {}
		return {}
	
	def set_custom_attributes(self, attributes: Dict[str, Any]):
		"""Set custom attributes from dictionary"""
		self.custom_attributes = json.dumps(attributes) if attributes else None


class IMSTCCycleCount(Model, AuditMixin, BaseMixin):
	"""
	Cycle count schedules and execution tracking.
	
	Manages scheduled and ad-hoc physical inventory counts
	for maintaining inventory accuracy.
	"""
	__tablename__ = 'im_stc_cycle_count'
	
	# Identity
	cycle_count_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Count Information
	count_number = Column(String(50), nullable=False, index=True)
	count_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Schedule Information
	count_type = Column(String(50), nullable=False)  # Scheduled, Ad-hoc, ABC, Fast/Slow
	frequency = Column(String(50), nullable=True)  # Daily, Weekly, Monthly, Quarterly, Annual
	scheduled_date = Column(Date, nullable=True, index=True)
	due_date = Column(Date, nullable=True, index=True)
	
	# Scope
	warehouse_id = Column(String(36), ForeignKey('im_stc_warehouse.warehouse_id'), nullable=True, index=True)
	location_id = Column(String(36), ForeignKey('im_stc_location.location_id'), nullable=True, index=True)
	category_id = Column(String(36), ForeignKey('im_stc_item_category.category_id'), nullable=True, index=True)
	abc_classification = Column(String(1), nullable=True)  # A, B, C
	
	# Execution
	status = Column(String(50), default='Scheduled', index=True)  # Scheduled, In Progress, Completed, Cancelled
	assigned_to = Column(String(36), nullable=True, index=True)
	start_date = Column(DateTime, nullable=True)
	completion_date = Column(DateTime, nullable=True)
	
	# Results
	items_counted = Column(Integer, default=0)
	items_with_variance = Column(Integer, default=0)
	total_variance_value = Column(DECIMAL(15, 2), default=0)
	accuracy_percentage = Column(DECIMAL(5, 2), nullable=True)
	
	# Configuration
	tolerance_percentage = Column(DECIMAL(5, 2), default=0)  # Acceptable variance
	blind_count = Column(Boolean, default=True)  # Hide system quantities
	allow_negative_adjustment = Column(Boolean, default=True)
	require_approval = Column(Boolean, default=False)
	
	# Additional Information
	instructions = Column(Text, nullable=True)
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'count_number', name='uq_count_number_tenant'),
		Index('idx_cycle_count_schedule', 'tenant_id', 'scheduled_date', 'status'),
		Index('idx_cycle_count_warehouse', 'tenant_id', 'warehouse_id', 'status'),
	)
	
	# Relationships
	warehouse = relationship("IMSTCWarehouse")
	location = relationship("IMSTCLocation")
	category = relationship("IMSTCItemCategory")
	count_lines = relationship("IMSTCCycleCountLine", back_populates="cycle_count")
	
	def __repr__(self):
		return f"<IMSTCCycleCount {self.count_number}: {self.status}>"


class IMSTCCycleCountLine(Model, AuditMixin, BaseMixin):
	"""
	Individual line items within a cycle count.
	
	Records the expected vs actual quantities for each item counted.
	"""
	__tablename__ = 'im_stc_cycle_count_line'
	
	# Identity
	count_line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Count Reference
	cycle_count_id = Column(String(36), ForeignKey('im_stc_cycle_count.cycle_count_id'), nullable=False, index=True)
	line_number = Column(Integer, nullable=False)
	
	# Item Information
	item_id = Column(String(36), ForeignKey('im_stc_item.item_id'), nullable=False, index=True)
	warehouse_id = Column(String(36), ForeignKey('im_stc_warehouse.warehouse_id'), nullable=False, index=True)
	location_id = Column(String(36), ForeignKey('im_stc_location.location_id'), nullable=False, index=True)
	uom_id = Column(String(36), ForeignKey('im_stc_unit_of_measure.uom_id'), nullable=False, index=True)
	
	# Optional Dimensions
	lot_number = Column(String(100), nullable=True)
	serial_number = Column(String(100), nullable=True)
	batch_number = Column(String(100), nullable=True)
	expiry_date = Column(Date, nullable=True)
	
	# Count Information
	system_quantity = Column(DECIMAL(15, 4), default=0)  # Expected quantity
	counted_quantity = Column(DECIMAL(15, 4), nullable=True)  # Actual count
	variance_quantity = Column(DECIMAL(15, 4), default=0)  # Difference
	unit_cost = Column(DECIMAL(15, 4), default=0)
	variance_value = Column(DECIMAL(15, 4), default=0)
	
	# Status
	status = Column(String(50), default='Pending', index=True)  # Pending, Counted, Adjusted, Approved
	count_date = Column(DateTime, nullable=True)
	counted_by = Column(String(36), nullable=True)
	approved_by = Column(String(36), nullable=True)
	approval_date = Column(DateTime, nullable=True)
	
	# Variance Analysis
	variance_percentage = Column(DECIMAL(5, 2), nullable=True)
	within_tolerance = Column(Boolean, nullable=True)
	requires_recount = Column(Boolean, default=False)
	recount_reason = Column(String(200), nullable=True)
	
	# Additional Information
	notes = Column(Text, nullable=True)
	adjustment_reason = Column(String(200), nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('cycle_count_id', 'line_number', name='uq_count_line_number'),
		Index('idx_count_line_item', 'tenant_id', 'item_id', 'warehouse_id', 'location_id'),
		Index('idx_count_line_status', 'tenant_id', 'status'),
	)
	
	# Relationships
	cycle_count = relationship("IMSTCCycleCount", back_populates="count_lines")
	item = relationship("IMSTCItem")
	warehouse = relationship("IMSTCWarehouse")
	location = relationship("IMSTCLocation")
	uom = relationship("IMSTCUnitOfMeasure")
	
	def __repr__(self):
		return f"<IMSTCCycleCountLine {self.cycle_count.count_number if self.cycle_count else 'Unknown'}-{self.line_number}: {self.item.item_code if self.item else 'Unknown'}>"


class IMSTCStockAlert(Model, AuditMixin, BaseMixin):
	"""
	Automated stock alerts and notifications.
	
	Configurable alerts for low stock, overstock, expiring items,
	and other inventory conditions requiring attention.
	"""
	__tablename__ = 'im_stc_stock_alert'
	
	# Identity
	alert_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Alert Information
	alert_type = Column(String(50), nullable=False, index=True)  # Low Stock, Overstock, Expiring, etc.
	alert_priority = Column(String(20), default='Medium', index=True)  # High, Medium, Low
	alert_title = Column(String(200), nullable=False)
	alert_message = Column(Text, nullable=False)
	
	# Alert Scope
	item_id = Column(String(36), ForeignKey('im_stc_item.item_id'), nullable=True, index=True)
	warehouse_id = Column(String(36), ForeignKey('im_stc_warehouse.warehouse_id'), nullable=True, index=True)
	location_id = Column(String(36), ForeignKey('im_stc_location.location_id'), nullable=True, index=True)
	category_id = Column(String(36), ForeignKey('im_stc_item_category.category_id'), nullable=True, index=True)
	
	# Alert Conditions
	trigger_condition = Column(String(200), nullable=True)  # Description of trigger
	threshold_value = Column(DECIMAL(15, 4), nullable=True)
	current_value = Column(DECIMAL(15, 4), nullable=True)
	
	# Status and Timing
	status = Column(String(50), default='Active', index=True)  # Active, Acknowledged, Resolved
	created_date = Column(DateTime, default=datetime.utcnow, index=True)
	acknowledged_date = Column(DateTime, nullable=True)
	acknowledged_by = Column(String(36), nullable=True)
	resolved_date = Column(DateTime, nullable=True)
	resolved_by = Column(String(36), nullable=True)
	
	# Configuration
	auto_resolve = Column(Boolean, default=False)  # Auto-resolve when condition clears
	escalation_level = Column(Integer, default=1)
	snooze_until = Column(DateTime, nullable=True)  # Temporarily suppress alert
	
	# Additional Information
	resolution_notes = Column(Text, nullable=True)
	additional_data = Column(Text, nullable=True)  # JSON for flexibility
	
	# Constraints
	__table_args__ = (
		Index('idx_alert_type_status', 'tenant_id', 'alert_type', 'status'),
		Index('idx_alert_item_status', 'tenant_id', 'item_id', 'status'),
		Index('idx_alert_created_priority', 'tenant_id', 'created_date', 'alert_priority'),
	)
	
	# Relationships
	item = relationship("IMSTCItem")
	warehouse = relationship("IMSTCWarehouse")
	location = relationship("IMSTCLocation")
	category = relationship("IMSTCItemCategory")
	
	def __repr__(self):
		return f"<IMSTCStockAlert {self.alert_type}: {self.alert_title}>"
	
	def get_additional_data(self) -> Dict[str, Any]:
		"""Get additional data as dictionary"""
		if self.additional_data:
			try:
				return json.loads(self.additional_data)
			except (json.JSONDecodeError, TypeError):
				return {}
		return {}
	
	def set_additional_data(self, data: Dict[str, Any]):
		"""Set additional data from dictionary"""
		self.additional_data = json.dumps(data) if data else None