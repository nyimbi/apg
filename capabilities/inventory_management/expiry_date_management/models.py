"""
Expiry Date Management Models

Database models for expiry tracking, FEFO (First Expired First Out) management,
waste tracking, and compliance monitoring.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ....auth_rbac.models import BaseMixin, AuditMixin, Model


class IMEDMExpiryPolicy(Model, AuditMixin, BaseMixin):
	"""
	Expiry management policies for different item categories.
	
	Defines rules for expiry tracking, alerting thresholds,
	and disposition actions for expired products.
	"""
	__tablename__ = 'im_edm_expiry_policy'
	
	# Identity
	policy_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Policy Information
	policy_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	policy_type = Column(String(50), default='Standard')  # Standard, Pharmaceutical, Food, etc.
	
	# Scope
	item_id = Column(String(36), nullable=True, index=True)  # Specific item or null for category
	category_id = Column(String(36), nullable=True, index=True)  # Item category
	item_type = Column(String(50), nullable=True)  # Item type filter
	
	# Alert Thresholds (days before expiry)
	critical_alert_days = Column(Integer, default=7)  # Critical alert threshold
	warning_alert_days = Column(Integer, default=30)  # Warning alert threshold
	notification_alert_days = Column(Integer, default=60)  # Notification alert threshold
	
	# Disposition Rules
	auto_quarantine_expired = Column(Boolean, default=True)
	auto_dispose_days_after_expiry = Column(Integer, nullable=True)  # Auto-dispose X days after expiry
	require_manager_approval = Column(Boolean, default=False)
	allow_expired_usage = Column(Boolean, default=False)  # For non-regulated items
	
	# FEFO (First Expired First Out) Settings
	enforce_fefo = Column(Boolean, default=True)
	fefo_grace_period_days = Column(Integer, default=0)  # Allow newer stock if within grace period
	
	# Regulatory Requirements
	regulatory_required = Column(Boolean, default=False)
	regulatory_authority = Column(String(100), nullable=True)  # FDA, EMA, etc.
	destruction_documentation_required = Column(Boolean, default=False)
	
	# Extension Policies
	allow_shelf_life_extension = Column(Boolean, default=False)
	max_extension_days = Column(Integer, nullable=True)
	extension_approval_required = Column(Boolean, default=True)
	
	# Status
	is_active = Column(Boolean, default=True)
	effective_date = Column(Date, default=date.today)
	expiration_date = Column(Date, nullable=True)
	
	# Constraints
	__table_args__ = (
		Index('idx_expiry_policy_scope', 'tenant_id', 'item_id', 'category_id'),
		Index('idx_expiry_policy_active', 'tenant_id', 'is_active', 'effective_date'),
	)
	
	# Relationships
	expiry_items = relationship("IMEDMExpiryItem", back_populates="expiry_policy")
	
	def __repr__(self):
		return f"<IMEDMExpiryPolicy {self.policy_name}>"


class IMEDMExpiryItem(Model, AuditMixin, BaseMixin):
	"""
	Individual inventory items with expiry tracking.
	
	Tracks specific lots/batches with expiry dates and
	manages their lifecycle through to disposition.
	"""
	__tablename__ = 'im_edm_expiry_item'
	
	# Identity
	expiry_item_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Item Information
	item_id = Column(String(36), nullable=False, index=True)
	batch_number = Column(String(100), nullable=True, index=True)
	lot_number = Column(String(100), nullable=True, index=True)
	serial_number = Column(String(100), nullable=True, index=True)
	
	# Location
	warehouse_id = Column(String(36), nullable=False, index=True)
	location_id = Column(String(36), nullable=False, index=True)
	
	# Expiry Information
	expiry_date = Column(Date, nullable=False, index=True)
	manufactured_date = Column(Date, nullable=True)
	best_before_date = Column(Date, nullable=True)
	use_by_date = Column(Date, nullable=True)
	shelf_life_days = Column(Integer, nullable=True)
	
	# Quantities
	original_quantity = Column(DECIMAL(15, 4), nullable=False)
	current_quantity = Column(DECIMAL(15, 4), nullable=False)
	allocated_quantity = Column(DECIMAL(15, 4), default=0)
	available_quantity = Column(DECIMAL(15, 4), nullable=False)
	
	# Status
	expiry_status = Column(String(50), default='Active', index=True)  # Active, Expiring, Expired, Disposed
	quality_status = Column(String(50), default='Good', index=True)  # Good, Quarantine, Expired, Disposed
	disposition_status = Column(String(50), default='None', index=True)  # None, Quarantine, Return, Dispose, Extend
	
	# Policy Reference
	expiry_policy_id = Column(String(36), ForeignKey('im_edm_expiry_policy.policy_id'), nullable=True, index=True)
	
	# Extension Information
	shelf_life_extended = Column(Boolean, default=False)
	extended_expiry_date = Column(Date, nullable=True)
	extension_reason = Column(Text, nullable=True)
	extension_approved_by = Column(String(36), nullable=True)
	extension_approval_date = Column(DateTime, nullable=True)
	
	# Alert Status
	alert_level = Column(String(20), nullable=True, index=True)  # None, Notification, Warning, Critical
	last_alert_date = Column(DateTime, nullable=True)
	alert_acknowledged = Column(Boolean, default=False)
	alert_acknowledged_by = Column(String(36), nullable=True)
	
	# Disposition Information
	disposition_date = Column(DateTime, nullable=True)
	disposition_method = Column(String(50), nullable=True)  # Return, Destroy, Donate, etc.
	disposition_notes = Column(Text, nullable=True)
	disposition_approved_by = Column(String(36), nullable=True)
	
	# Cost Information
	unit_cost = Column(DECIMAL(15, 4), default=0)
	total_cost_value = Column(DECIMAL(15, 2), default=0)
	waste_cost = Column(DECIMAL(15, 2), default=0)
	
	# Additional Information
	supplier_id = Column(String(36), nullable=True)
	purchase_order_number = Column(String(100), nullable=True)
	storage_conditions = Column(String(200), nullable=True)
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		Index('idx_expiry_item_location', 'tenant_id', 'item_id', 'warehouse_id', 'location_id'),
		Index('idx_expiry_item_date', 'tenant_id', 'expiry_date', 'expiry_status'),
		Index('idx_expiry_item_batch', 'tenant_id', 'batch_number', 'lot_number'),
		Index('idx_expiry_item_alert', 'tenant_id', 'alert_level', 'alert_acknowledged'),
	)
	
	# Relationships
	expiry_policy = relationship("IMEDMExpiryPolicy", back_populates="expiry_items")
	movements = relationship("IMEDMExpiryMovement", back_populates="expiry_item")
	dispositions = relationship("IMEDMDisposition", back_populates="expiry_item")
	
	def __repr__(self):
		return f"<IMEDMExpiryItem {self.item_id}-{self.batch_number}: {self.expiry_date}>"
	
	@property
	def days_to_expiry(self) -> int:
		"""Calculate days until expiry"""
		if self.extended_expiry_date:
			target_date = self.extended_expiry_date
		else:
			target_date = self.expiry_date
		
		return (target_date - date.today()).days
	
	@property
	def is_expired(self) -> bool:
		"""Check if item is expired"""
		return self.days_to_expiry < 0
	
	@property
	def is_expiring_soon(self, days_threshold: int = 30) -> bool:
		"""Check if item is expiring soon"""
		return 0 <= self.days_to_expiry <= days_threshold


class IMEDMExpiryMovement(Model, AuditMixin, BaseMixin):
	"""
	Movement history for expiry-tracked items.
	
	Records all movements with FEFO compliance tracking
	and expiry status changes.
	"""
	__tablename__ = 'im_edm_expiry_movement'
	
	# Identity
	movement_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Movement Information
	movement_date = Column(DateTime, default=datetime.utcnow, index=True)
	movement_type = Column(String(50), nullable=False, index=True)  # Issue, Transfer, Adjust, Dispose
	expiry_item_id = Column(String(36), ForeignKey('im_edm_expiry_item.expiry_item_id'), nullable=False, index=True)
	
	# Quantity
	quantity = Column(DECIMAL(15, 4), nullable=False)
	remaining_quantity = Column(DECIMAL(15, 4), nullable=False)
	
	# FEFO Compliance
	fefo_compliant = Column(Boolean, nullable=True)  # True if followed FEFO, False if not, None if N/A
	fefo_violation_reason = Column(String(200), nullable=True)
	override_reason = Column(String(200), nullable=True)
	override_approved_by = Column(String(36), nullable=True)
	
	# Location Information
	from_location_id = Column(String(36), nullable=True)
	to_location_id = Column(String(36), nullable=True)
	
	# Reference Information
	reference_number = Column(String(100), nullable=True, index=True)
	reference_type = Column(String(50), nullable=True)
	
	# Status at Movement
	expiry_status_before = Column(String(50), nullable=True)
	expiry_status_after = Column(String(50), nullable=True)
	days_to_expiry = Column(Integer, nullable=True)
	
	# User Information
	user_id = Column(String(36), nullable=True)
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		Index('idx_expiry_movement_item', 'tenant_id', 'expiry_item_id', 'movement_date'),
		Index('idx_expiry_movement_fefo', 'tenant_id', 'fefo_compliant', 'movement_date'),
		Index('idx_expiry_movement_ref', 'tenant_id', 'reference_number', 'reference_type'),
	)
	
	# Relationships
	expiry_item = relationship("IMEDMExpiryItem", back_populates="movements")
	
	def __repr__(self):
		return f"<IMEDMExpiryMovement {self.movement_type}: {self.quantity}>"


class IMEDMDisposition(Model, AuditMixin, BaseMixin):
	"""
	Disposition records for expired or expiring items.
	
	Tracks how expired items are handled including destruction,
	return to supplier, donation, or alternative disposition.
	"""
	__tablename__ = 'im_edm_disposition'
	
	# Identity
	disposition_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Disposition Information
	disposition_number = Column(String(100), nullable=False, index=True)
	disposition_date = Column(DateTime, default=datetime.utcnow, index=True)
	expiry_item_id = Column(String(36), ForeignKey('im_edm_expiry_item.expiry_item_id'), nullable=False, index=True)
	
	# Disposition Details
	disposition_type = Column(String(50), nullable=False, index=True)  # Destroy, Return, Donate, Rework, Extend
	disposition_reason = Column(String(200), nullable=False)
	quantity_disposed = Column(DECIMAL(15, 4), nullable=False)
	
	# Status
	status = Column(String(50), default='Planned', index=True)  # Planned, In Progress, Completed, Cancelled
	
	# Approval
	requires_approval = Column(Boolean, default=True)
	approved_by = Column(String(36), nullable=True)
	approval_date = Column(DateTime, nullable=True)
	approval_notes = Column(Text, nullable=True)
	
	# Execution Details
	executed_by = Column(String(36), nullable=True)
	execution_date = Column(DateTime, nullable=True)
	actual_quantity_disposed = Column(DECIMAL(15, 4), nullable=True)
	
	# Cost Impact
	disposal_cost = Column(DECIMAL(15, 2), default=0)
	recovery_value = Column(DECIMAL(15, 2), default=0)
	net_loss = Column(DECIMAL(15, 2), default=0)
	
	# External Parties
	disposal_vendor = Column(String(200), nullable=True)
	disposal_vendor_id = Column(String(36), nullable=True)
	return_supplier_id = Column(String(36), nullable=True)
	donation_recipient = Column(String(200), nullable=True)
	
	# Documentation
	disposal_certificate_number = Column(String(100), nullable=True)
	disposal_certificate_path = Column(String(500), nullable=True)
	photos_taken = Column(Boolean, default=False)
	witness_signature = Column(String(200), nullable=True)
	
	# Regulatory Compliance
	regulatory_notification_required = Column(Boolean, default=False)
	regulatory_notification_sent = Column(Boolean, default=False)
	regulatory_case_number = Column(String(100), nullable=True)
	
	# Environmental Impact
	environmental_impact_assessed = Column(Boolean, default=False)
	disposal_method_eco_friendly = Column(Boolean, nullable=True)
	carbon_footprint_kg = Column(DECIMAL(10, 2), nullable=True)
	
	# Additional Information
	notes = Column(Text, nullable=True)
	lessons_learned = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'disposition_number', name='uq_disposition_number_tenant'),
		Index('idx_disposition_item', 'tenant_id', 'expiry_item_id'),
		Index('idx_disposition_status', 'tenant_id', 'status', 'disposition_type'),
		Index('idx_disposition_approval', 'tenant_id', 'requires_approval', 'approved_by'),
	)
	
	# Relationships
	expiry_item = relationship("IMEDMExpiryItem", back_populates="dispositions")
	
	def __repr__(self):
		return f"<IMEDMDisposition {self.disposition_number}: {self.disposition_type}>"


class IMEDMExpiryAlert(Model, AuditMixin, BaseMixin):
	"""
	Automated expiry alerts and notifications.
	
	Manages alert generation, escalation, and tracking
	for expiring and expired inventory items.
	"""
	__tablename__ = 'im_edm_expiry_alert'
	
	# Identity
	alert_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Alert Information
	alert_date = Column(DateTime, default=datetime.utcnow, index=True)
	alert_type = Column(String(50), nullable=False, index=True)  # Expiring, Expired, FEFO_Violation
	alert_level = Column(String(20), nullable=False, index=True)  # Notification, Warning, Critical
	
	# Scope
	item_id = Column(String(36), nullable=False, index=True)
	expiry_item_id = Column(String(36), ForeignKey('im_edm_expiry_item.expiry_item_id'), nullable=True, index=True)
	warehouse_id = Column(String(36), nullable=True, index=True)
	
	# Alert Content
	alert_title = Column(String(200), nullable=False)
	alert_message = Column(Text, nullable=False)
	
	# Timing Information
	expiry_date = Column(Date, nullable=True)
	days_to_expiry = Column(Integer, nullable=True)
	quantity_affected = Column(DECIMAL(15, 4), nullable=True)
	value_at_risk = Column(DECIMAL(15, 2), nullable=True)
	
	# Status and Response
	status = Column(String(50), default='Active', index=True)  # Active, Acknowledged, Resolved, Snoozed
	acknowledged_by = Column(String(36), nullable=True)
	acknowledgment_date = Column(DateTime, nullable=True)
	resolution_date = Column(DateTime, nullable=True)
	resolution_action = Column(String(200), nullable=True)
	
	# Escalation
	escalation_level = Column(Integer, default=1)
	escalated_to = Column(String(36), nullable=True)
	escalation_date = Column(DateTime, nullable=True)
	max_escalation_level = Column(Integer, default=3)
	
	# Snooze Functionality
	snoozed_until = Column(DateTime, nullable=True)
	snooze_reason = Column(String(200), nullable=True)
	
	# Recurrence
	is_recurring = Column(Boolean, default=False)
	recurrence_frequency_days = Column(Integer, nullable=True)
	last_occurrence_date = Column(DateTime, nullable=True)
	next_occurrence_date = Column(DateTime, nullable=True)
	
	# Additional Information
	priority_score = Column(DECIMAL(5, 2), nullable=True)  # Calculated priority score
	business_impact = Column(String(100), nullable=True)  # High, Medium, Low
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		Index('idx_expiry_alert_item', 'tenant_id', 'item_id', 'alert_type'),
		Index('idx_expiry_alert_status', 'tenant_id', 'status', 'alert_level'),
		Index('idx_expiry_alert_escalation', 'tenant_id', 'escalation_level', 'escalated_to'),
		Index('idx_expiry_alert_expiry', 'tenant_id', 'expiry_date', 'days_to_expiry'),
	)
	
	# Relationships
	expiry_item = relationship("IMEDMExpiryItem")
	
	def __repr__(self):
		return f"<IMEDMExpiryAlert {self.alert_type}-{self.alert_level}: {self.alert_title}>"


class IMEDMWasteReport(Model, AuditMixin, BaseMixin):
	"""
	Waste reporting and analytics for expired items.
	
	Tracks waste metrics, costs, and trends for
	management reporting and process improvement.
	"""
	__tablename__ = 'im_edm_waste_report'
	
	# Identity
	report_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Report Information
	report_date = Column(Date, default=date.today, index=True)
	report_period_start = Column(Date, nullable=False, index=True)
	report_period_end = Column(Date, nullable=False, index=True)
	report_type = Column(String(50), default='Monthly')  # Daily, Weekly, Monthly, Quarterly, Annual
	
	# Scope
	warehouse_id = Column(String(36), nullable=True, index=True)
	category_id = Column(String(36), nullable=True, index=True)
	item_id = Column(String(36), nullable=True, index=True)
	
	# Waste Metrics
	total_items_expired = Column(Integer, default=0)
	total_quantity_wasted = Column(DECIMAL(15, 4), default=0)
	total_value_wasted = Column(DECIMAL(15, 2), default=0)
	disposal_costs = Column(DECIMAL(15, 2), default=0)
	total_waste_cost = Column(DECIMAL(15, 2), default=0)  # Value + disposal costs
	
	# Disposition Breakdown
	quantity_destroyed = Column(DECIMAL(15, 4), default=0)
	quantity_returned = Column(DECIMAL(15, 4), default=0)
	quantity_donated = Column(DECIMAL(15, 4), default=0)
	quantity_extended = Column(DECIMAL(15, 4), default=0)
	quantity_reworked = Column(DECIMAL(15, 4), default=0)
	
	# Prevention Metrics
	prevented_waste_quantity = Column(DECIMAL(15, 4), default=0)  # Through early sale, transfer, etc.
	prevented_waste_value = Column(DECIMAL(15, 2), default=0)
	waste_prevention_rate = Column(DECIMAL(5, 2), default=0)  # Percentage
	
	# Comparison Metrics
	previous_period_waste = Column(DECIMAL(15, 2), nullable=True)
	waste_trend_percentage = Column(DECIMAL(5, 2), nullable=True)  # % change from previous period
	waste_rate_percentage = Column(DECIMAL(5, 2), nullable=True)  # Waste / Total inventory value
	
	# Root Cause Analysis
	top_waste_reasons = Column(Text, nullable=True)  # JSON array of reasons
	top_waste_categories = Column(Text, nullable=True)  # JSON array of categories
	seasonal_factors = Column(Text, nullable=True)
	
	# Action Items
	improvement_opportunities = Column(Text, nullable=True)
	recommended_actions = Column(Text, nullable=True)
	cost_saving_potential = Column(DECIMAL(15, 2), nullable=True)
	
	# Report Status
	status = Column(String(50), default='Draft', index=True)  # Draft, Final, Published
	generated_by = Column(String(36), nullable=True)
	reviewed_by = Column(String(36), nullable=True)
	review_date = Column(DateTime, nullable=True)
	
	# Additional Information
	notes = Column(Text, nullable=True)
	report_path = Column(String(500), nullable=True)  # Path to detailed report file
	
	# Constraints
	__table_args__ = (
		Index('idx_waste_report_period', 'tenant_id', 'report_period_start', 'report_period_end'),
		Index('idx_waste_report_scope', 'tenant_id', 'warehouse_id', 'category_id'),
		Index('idx_waste_report_date', 'tenant_id', 'report_date', 'report_type'),
	)
	
	def __repr__(self):
		return f"<IMEDMWasteReport {self.report_date}: {self.report_type}>"
	
	def get_top_waste_reasons(self) -> List[Dict[str, Any]]:
		"""Get top waste reasons as list"""
		if self.top_waste_reasons:
			try:
				return json.loads(self.top_waste_reasons)
			except (json.JSONDecodeError, TypeError):
				return []
		return []
	
	def set_top_waste_reasons(self, reasons: List[Dict[str, Any]]):
		"""Set top waste reasons from list"""
		self.top_waste_reasons = json.dumps(reasons) if reasons else None