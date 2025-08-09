"""
Replenishment & Reordering Models

Database models for automated replenishment, purchase order generation,
supplier management, and demand forecasting.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ....auth_rbac.models import BaseMixin, AuditMixin, Model


class IMRRSupplier(Model, AuditMixin, BaseMixin):
	"""
	Supplier master data for replenishment.
	
	Manages supplier information, terms, and performance metrics
	for automated procurement decisions.
	"""
	__tablename__ = 'im_rr_supplier'
	
	# Identity
	supplier_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Supplier Information
	supplier_code = Column(String(20), nullable=False, index=True)
	supplier_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	supplier_type = Column(String(50), default='Standard')  # Standard, Preferred, Backup
	
	# Contact Information
	contact_person = Column(String(200), nullable=True)
	email_address = Column(String(200), nullable=True)
	phone_number = Column(String(50), nullable=True)
	website_url = Column(String(200), nullable=True)
	
	# Address Information
	address_line1 = Column(String(200), nullable=True)
	address_line2 = Column(String(200), nullable=True)
	city = Column(String(100), nullable=True)
	state_province = Column(String(100), nullable=True)
	postal_code = Column(String(20), nullable=True)
	country_code = Column(String(3), nullable=True)
	
	# Business Terms
	payment_terms = Column(String(50), nullable=True)  # Net 30, 2/10 Net 30, etc.
	currency_code = Column(String(3), default='USD')
	minimum_order_amount = Column(DECIMAL(15, 2), default=0)
	lead_time_days = Column(Integer, default=7)
	
	# Performance Metrics
	on_time_delivery_rate = Column(DECIMAL(5, 2), default=0)  # Percentage
	quality_rating = Column(DECIMAL(3, 2), default=0)  # 1-5 scale
	price_competitiveness = Column(DECIMAL(3, 2), default=0)  # 1-5 scale
	last_performance_review = Column(Date, nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True)
	is_approved = Column(Boolean, default=False)
	approval_date = Column(Date, nullable=True)
	approved_by = Column(String(36), nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'supplier_code', name='uq_supplier_code_tenant'),
	)
	
	# Relationships
	supplier_items = relationship("IMRRSupplierItem", back_populates="supplier")
	purchase_orders = relationship("IMRRPurchaseOrder", back_populates="supplier")
	
	def __repr__(self):
		return f"<IMRRSupplier {self.supplier_code}: {self.supplier_name}>"


class IMRRSupplierItem(Model, AuditMixin, BaseMixin):
	"""
	Supplier-specific item information and pricing.
	
	Links inventory items to suppliers with pricing, lead times,
	and ordering parameters.
	"""
	__tablename__ = 'im_rr_supplier_item'
	
	# Identity
	supplier_item_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# References
	supplier_id = Column(String(36), ForeignKey('im_rr_supplier.supplier_id'), nullable=False, index=True)
	item_id = Column(String(36), nullable=False, index=True)  # Reference to IMSTCItem
	
	# Supplier Item Information
	supplier_item_code = Column(String(100), nullable=True)
	supplier_item_name = Column(String(200), nullable=True)
	manufacturer_part_number = Column(String(100), nullable=True)
	
	# Pricing
	unit_price = Column(DECIMAL(15, 4), nullable=False)
	currency_code = Column(String(3), default='USD')
	price_uom_id = Column(String(36), nullable=True)  # UOM for pricing
	price_effective_date = Column(Date, nullable=True)
	price_expiry_date = Column(Date, nullable=True)
	
	# Ordering Parameters
	minimum_order_quantity = Column(DECIMAL(15, 4), default=1)
	order_multiple = Column(DECIMAL(15, 4), default=1)  # Must order in multiples of this
	lead_time_days = Column(Integer, default=7)
	
	# Quality and Compliance
	quality_rating = Column(DECIMAL(3, 2), default=0)
	certification_required = Column(Boolean, default=False)
	certification_type = Column(String(100), nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True)
	is_preferred = Column(Boolean, default=False)
	priority_rank = Column(Integer, default=1)  # 1 = highest priority
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'supplier_id', 'item_id', name='uq_supplier_item'),
		Index('idx_supplier_item_lookup', 'tenant_id', 'item_id', 'is_active'),
	)
	
	# Relationships
	supplier = relationship("IMRRSupplier", back_populates="supplier_items")
	
	def __repr__(self):
		return f"<IMRRSupplierItem {self.supplier.supplier_code if self.supplier else 'Unknown'} - Item {self.item_id}>"


class IMRRReplenishmentRule(Model, AuditMixin, BaseMixin):
	"""
	Automated replenishment rules and parameters.
	
	Defines when and how much to reorder based on various
	replenishment strategies (min/max, economic order quantity, etc.).
	"""
	__tablename__ = 'im_rr_replenishment_rule'
	
	# Identity
	rule_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Rule Information
	rule_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	rule_type = Column(String(50), nullable=False, index=True)  # Min/Max, EOQ, JIT, Periodic
	
	# Scope
	item_id = Column(String(36), nullable=True, index=True)  # Specific item or null for category/all
	category_id = Column(String(36), nullable=True, index=True)  # Item category
	warehouse_id = Column(String(36), nullable=True, index=True)  # Specific warehouse
	abc_classification = Column(String(1), nullable=True)  # A, B, C items
	
	# Replenishment Parameters
	reorder_point = Column(DECIMAL(15, 4), nullable=True)
	reorder_quantity = Column(DECIMAL(15, 4), nullable=True)
	max_stock_level = Column(DECIMAL(15, 4), nullable=True)
	safety_stock = Column(DECIMAL(15, 4), default=0)
	
	# Economic Order Quantity Parameters
	annual_demand = Column(DECIMAL(15, 4), nullable=True)
	carrying_cost_rate = Column(DECIMAL(5, 4), default=0.25)  # 25% per year
	ordering_cost = Column(DECIMAL(10, 2), default=0)
	
	# Time-based Parameters
	review_cycle_days = Column(Integer, nullable=True)  # For periodic review
	lead_time_days = Column(Integer, default=7)
	demand_forecast_days = Column(Integer, default=30)
	
	# Demand Variability
	demand_variability = Column(DECIMAL(5, 4), default=0)  # Coefficient of variation
	service_level_target = Column(DECIMAL(5, 2), default=95.0)  # Target service level %
	
	# Automation Settings
	auto_generate_po = Column(Boolean, default=False)
	auto_approve_po = Column(Boolean, default=False)
	max_auto_po_amount = Column(DECIMAL(15, 2), nullable=True)
	preferred_supplier_id = Column(String(36), ForeignKey('im_rr_supplier.supplier_id'), nullable=True)
	
	# Status and Scheduling
	is_active = Column(Boolean, default=True)
	last_run_date = Column(DateTime, nullable=True)
	next_run_date = Column(DateTime, nullable=True)
	run_frequency_hours = Column(Integer, default=24)  # How often to check
	
	# Performance Tracking
	suggestions_generated = Column(Integer, default=0)
	pos_generated = Column(Integer, default=0)
	accuracy_rate = Column(DECIMAL(5, 2), nullable=True)
	
	# Constraints
	__table_args__ = (
		Index('idx_replenishment_rule_scope', 'tenant_id', 'item_id', 'warehouse_id'),
		Index('idx_replenishment_rule_schedule', 'tenant_id', 'next_run_date', 'is_active'),
	)
	
	# Relationships
	preferred_supplier = relationship("IMRRSupplier")
	replenishment_suggestions = relationship("IMRRReplenishmentSuggestion", back_populates="rule")
	
	def __repr__(self):
		return f"<IMRRReplenishmentRule {self.rule_name}: {self.rule_type}>"


class IMRRReplenishmentSuggestion(Model, AuditMixin, BaseMixin):
	"""
	System-generated replenishment suggestions.
	
	Records automated suggestions for stock replenishment
	that can be reviewed and converted to purchase orders.
	"""
	__tablename__ = 'im_rr_replenishment_suggestion'
	
	# Identity
	suggestion_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Suggestion Information
	suggestion_date = Column(DateTime, default=datetime.utcnow, index=True)
	rule_id = Column(String(36), ForeignKey('im_rr_replenishment_rule.rule_id'), nullable=False, index=True)
	item_id = Column(String(36), nullable=False, index=True)
	warehouse_id = Column(String(36), nullable=False, index=True)
	
	# Current State
	current_stock = Column(DECIMAL(15, 4), nullable=False)
	allocated_stock = Column(DECIMAL(15, 4), default=0)
	available_stock = Column(DECIMAL(15, 4), nullable=False)
	on_order_quantity = Column(DECIMAL(15, 4), default=0)
	
	# Demand Analysis
	average_daily_demand = Column(DECIMAL(15, 4), default=0)
	demand_variability = Column(DECIMAL(5, 4), default=0)
	forecast_demand = Column(DECIMAL(15, 4), default=0)
	forecast_period_days = Column(Integer, default=30)
	
	# Replenishment Calculation
	reorder_point = Column(DECIMAL(15, 4), nullable=False)
	suggested_quantity = Column(DECIMAL(15, 4), nullable=False)
	target_stock_level = Column(DECIMAL(15, 4), nullable=True)
	safety_stock_req = Column(DECIMAL(15, 4), default=0)
	
	# Supplier and Costing
	recommended_supplier_id = Column(String(36), ForeignKey('im_rr_supplier.supplier_id'), nullable=True)
	unit_cost = Column(DECIMAL(15, 4), nullable=True)
	total_cost = Column(DECIMAL(15, 2), nullable=True)
	lead_time_days = Column(Integer, default=7)
	
	# Priority and Urgency
	priority_level = Column(String(20), default='Medium', index=True)  # High, Medium, Low
	urgency_score = Column(DECIMAL(5, 2), default=0)  # Calculated urgency score
	stockout_risk = Column(DECIMAL(5, 2), default=0)  # Risk of stockout %
	
	# Status
	status = Column(String(50), default='Pending', index=True)  # Pending, Approved, Rejected, Converted
	reviewed_by = Column(String(36), nullable=True)
	review_date = Column(DateTime, nullable=True)
	review_notes = Column(Text, nullable=True)
	
	# Conversion to PO
	po_id = Column(String(36), nullable=True, index=True)  # If converted to PO
	conversion_date = Column(DateTime, nullable=True)
	
	# Additional Information
	reason_code = Column(String(50), nullable=True)  # Low Stock, Seasonal, etc.
	additional_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		Index('idx_suggestion_item_status', 'tenant_id', 'item_id', 'status'),
		Index('idx_suggestion_priority', 'tenant_id', 'priority_level', 'suggestion_date'),
		Index('idx_suggestion_warehouse', 'tenant_id', 'warehouse_id', 'status'),
	)
	
	# Relationships
	rule = relationship("IMRRReplenishmentRule", back_populates="replenishment_suggestions")
	recommended_supplier = relationship("IMRRSupplier")
	
	def __repr__(self):
		return f"<IMRRReplenishmentSuggestion Item {self.item_id}: {self.suggested_quantity} units>"


class IMRRPurchaseOrder(Model, AuditMixin, BaseMixin):
	"""
	Purchase orders generated from replenishment process.
	
	Manages purchase orders created manually or automatically
	from replenishment suggestions.
	"""
	__tablename__ = 'im_rr_purchase_order'
	
	# Identity
	po_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Purchase Order Information
	po_number = Column(String(50), nullable=False, index=True)
	po_date = Column(Date, default=date.today, index=True)
	supplier_id = Column(String(36), ForeignKey('im_rr_supplier.supplier_id'), nullable=False, index=True)
	
	# Delivery Information
	requested_delivery_date = Column(Date, nullable=True, index=True)
	promised_delivery_date = Column(Date, nullable=True)
	warehouse_id = Column(String(36), nullable=False, index=True)
	
	# Financial Information
	currency_code = Column(String(3), default='USD')
	subtotal_amount = Column(DECIMAL(15, 2), default=0)
	tax_amount = Column(DECIMAL(15, 2), default=0)
	shipping_amount = Column(DECIMAL(15, 2), default=0)
	total_amount = Column(DECIMAL(15, 2), default=0)
	
	# Status
	status = Column(String(50), default='Draft', index=True)  # Draft, Sent, Acknowledged, Shipped, Received, Cancelled
	approval_status = Column(String(50), default='Pending')  # Pending, Approved, Rejected
	approved_by = Column(String(36), nullable=True)
	approval_date = Column(DateTime, nullable=True)
	
	# Tracking
	sent_date = Column(DateTime, nullable=True)
	acknowledgment_date = Column(DateTime, nullable=True)
	shipping_date = Column(DateTime, nullable=True)
	received_date = Column(DateTime, nullable=True)
	
	# Source Information
	created_from = Column(String(50), nullable=True)  # Manual, Replenishment, Emergency
	source_suggestion_id = Column(String(36), nullable=True)  # If from replenishment suggestion
	
	# Additional Information
	terms_and_conditions = Column(Text, nullable=True)
	special_instructions = Column(Text, nullable=True)
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'po_number', name='uq_po_number_tenant'),
		Index('idx_po_supplier_status', 'tenant_id', 'supplier_id', 'status'),
		Index('idx_po_delivery_date', 'tenant_id', 'requested_delivery_date'),
	)
	
	# Relationships
	supplier = relationship("IMRRSupplier", back_populates="purchase_orders")
	po_lines = relationship("IMRRPurchaseOrderLine", back_populates="purchase_order")
	
	def __repr__(self):
		return f"<IMRRPurchaseOrder {self.po_number}: {self.supplier.supplier_name if self.supplier else 'Unknown'}>"


class IMRRPurchaseOrderLine(Model, AuditMixin, BaseMixin):
	"""
	Individual line items within purchase orders.
	
	Details of specific items being ordered including quantities,
	prices, and delivery requirements.
	"""
	__tablename__ = 'im_rr_purchase_order_line'
	
	# Identity
	po_line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Purchase Order Reference
	po_id = Column(String(36), ForeignKey('im_rr_purchase_order.po_id'), nullable=False, index=True)
	line_number = Column(Integer, nullable=False)
	
	# Item Information
	item_id = Column(String(36), nullable=False, index=True)
	supplier_item_code = Column(String(100), nullable=True)
	
	# Quantities
	ordered_quantity = Column(DECIMAL(15, 4), nullable=False)
	received_quantity = Column(DECIMAL(15, 4), default=0)
	cancelled_quantity = Column(DECIMAL(15, 4), default=0)
	
	# Pricing
	unit_price = Column(DECIMAL(15, 4), nullable=False)
	line_total = Column(DECIMAL(15, 2), nullable=False)
	
	# Delivery
	requested_delivery_date = Column(Date, nullable=True)
	promised_delivery_date = Column(Date, nullable=True)
	location_id = Column(String(36), nullable=True)  # Specific delivery location
	
	# Status
	line_status = Column(String(50), default='Open', index=True)  # Open, Partial, Closed, Cancelled
	
	# Quality Requirements
	inspection_required = Column(Boolean, default=False)
	quality_specification = Column(Text, nullable=True)
	
	# Additional Information
	notes = Column(Text, nullable=True)
	
	# Source Information
	source_suggestion_id = Column(String(36), nullable=True)  # Link to original suggestion
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('po_id', 'line_number', name='uq_po_line_number'),
		Index('idx_po_line_item', 'tenant_id', 'item_id'),
		Index('idx_po_line_status', 'tenant_id', 'line_status'),
	)
	
	# Relationships
	purchase_order = relationship("IMRRPurchaseOrder", back_populates="po_lines")
	
	def __repr__(self):
		return f"<IMRRPurchaseOrderLine {self.purchase_order.po_number if self.purchase_order else 'Unknown'}-{self.line_number}: {self.ordered_quantity} units>"


class IMRRDemandForecast(Model, AuditMixin, BaseMixin):
	"""
	Demand forecast data for replenishment planning.
	
	Stores historical demand patterns and future forecasts
	to support replenishment calculations.
	"""
	__tablename__ = 'im_rr_demand_forecast'
	
	# Identity
	forecast_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Forecast Information
	item_id = Column(String(36), nullable=False, index=True)
	warehouse_id = Column(String(36), nullable=True, index=True)
	forecast_date = Column(Date, nullable=False, index=True)
	forecast_period = Column(String(20), nullable=False)  # Daily, Weekly, Monthly
	
	# Historical Data
	historical_demand = Column(DECIMAL(15, 4), nullable=True)
	moving_average = Column(DECIMAL(15, 4), nullable=True)
	exponential_smoothing = Column(DECIMAL(15, 4), nullable=True)
	
	# Forecast Values
	forecast_quantity = Column(DECIMAL(15, 4), nullable=False)
	forecast_method = Column(String(50), nullable=False)  # Manual, Moving Average, Exponential, ML
	confidence_level = Column(DECIMAL(5, 2), nullable=True)  # Confidence in forecast %
	
	# Seasonal Adjustments
	seasonal_factor = Column(DECIMAL(5, 4), default=1.0)
	trend_factor = Column(DECIMAL(5, 4), default=1.0)
	
	# Forecast Accuracy
	actual_demand = Column(DECIMAL(15, 4), nullable=True)  # Filled in after period
	forecast_error = Column(DECIMAL(15, 4), nullable=True)  # Actual - Forecast
	absolute_error = Column(DECIMAL(15, 4), nullable=True)  # ABS(Actual - Forecast)
	
	# Status
	is_approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approval_date = Column(DateTime, nullable=True)
	
	# Additional Information
	notes = Column(Text, nullable=True)
	external_factors = Column(Text, nullable=True)  # Events affecting demand
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'item_id', 'warehouse_id', 'forecast_date', 'forecast_period', 
						name='uq_forecast_item_date'),
		Index('idx_forecast_item_period', 'tenant_id', 'item_id', 'forecast_period'),
		Index('idx_forecast_date', 'tenant_id', 'forecast_date'),
	)
	
	def __repr__(self):
		return f"<IMRRDemandForecast Item {self.item_id} {self.forecast_date}: {self.forecast_quantity}>"