"""
Batch & Lot Tracking Models

Database models for batch/lot management, genealogy tracking,
quality control, and recall management.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ....auth_rbac.models import BaseMixin, AuditMixin, Model


class IMBLTBatch(Model, AuditMixin, BaseMixin):
	"""
	Master batch/lot records.
	
	Manages batch identity, lifecycle, and key attributes
	for full traceability and recall capability.
	"""
	__tablename__ = 'im_blt_batch'
	
	# Identity
	batch_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Batch Information
	batch_number = Column(String(100), nullable=False, index=True)
	lot_number = Column(String(100), nullable=True, index=True)  # Alternative identifier
	item_id = Column(String(36), nullable=False, index=True)
	
	# Production Information
	production_date = Column(Date, nullable=True, index=True)
	production_line = Column(String(100), nullable=True)
	production_shift = Column(String(50), nullable=True)
	production_order_number = Column(String(100), nullable=True)
	
	# Expiry and Shelf Life
	expiry_date = Column(Date, nullable=True, index=True)
	best_before_date = Column(Date, nullable=True)
	shelf_life_days = Column(Integer, nullable=True)
	
	# Quantities
	original_quantity = Column(DECIMAL(15, 4), nullable=False)
	current_quantity = Column(DECIMAL(15, 4), nullable=False)
	available_quantity = Column(DECIMAL(15, 4), nullable=False)
	quarantine_quantity = Column(DECIMAL(15, 4), default=0)
	
	# Status
	batch_status = Column(String(50), default='Active', index=True)  # Active, Exhausted, Expired, Recalled
	quality_status = Column(String(50), default='Approved', index=True)  # Approved, Quarantine, Rejected, Testing
	recall_status = Column(String(50), default='None', index=True)  # None, Investigation, Recall
	
	# Quality Information
	quality_grade = Column(String(20), nullable=True)  # A, B, C or Premium, Standard, etc.
	quality_notes = Column(Text, nullable=True)
	certificate_of_analysis = Column(String(200), nullable=True)  # Reference to COA document
	
	# Supplier Information
	supplier_id = Column(String(36), nullable=True, index=True)
	supplier_batch_number = Column(String(100), nullable=True)
	supplier_lot_number = Column(String(100), nullable=True)
	
	# Storage Information
	warehouse_id = Column(String(36), nullable=True, index=True)
	storage_conditions = Column(String(200), nullable=True)
	temperature_range = Column(String(50), nullable=True)  # e.g., "2-8°C"
	humidity_range = Column(String(50), nullable=True)
	
	# Regulatory
	regulatory_status = Column(String(50), default='Compliant')
	regulatory_notes = Column(Text, nullable=True)
	requires_recall_notification = Column(Boolean, default=False)
	
	# Genealogy
	parent_batch_ids = Column(Text, nullable=True)  # JSON array of parent batch IDs
	child_batch_ids = Column(Text, nullable=True)   # JSON array of child batch IDs
	
	# Additional Attributes
	custom_attributes = Column(Text, nullable=True)  # JSON for flexibility
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'batch_number', 'item_id', name='uq_batch_number_item'),
		Index('idx_batch_item_status', 'tenant_id', 'item_id', 'batch_status'),
		Index('idx_batch_expiry', 'tenant_id', 'expiry_date'),
		Index('idx_batch_quality', 'tenant_id', 'quality_status'),
	)
	
	# Relationships
	transactions = relationship("IMBLTBatchTransaction", back_populates="batch")
	quality_tests = relationship("IMBLTQualityTest", back_populates="batch")
	recall_events = relationship("IMBLTRecallEvent", back_populates="batch")
	
	def __repr__(self):
		return f"<IMBLTBatch {self.batch_number}: {self.batch_status}>"
	
	def get_parent_batch_ids(self) -> List[str]:
		"""Get parent batch IDs as list"""
		if self.parent_batch_ids:
			try:
				return json.loads(self.parent_batch_ids)
			except (json.JSONDecodeError, TypeError):
				return []
		return []
	
	def set_parent_batch_ids(self, batch_ids: List[str]):
		"""Set parent batch IDs from list"""
		self.parent_batch_ids = json.dumps(batch_ids) if batch_ids else None
	
	def get_child_batch_ids(self) -> List[str]:
		"""Get child batch IDs as list"""
		if self.child_batch_ids:
			try:
				return json.loads(self.child_batch_ids)
			except (json.JSONDecodeError, TypeError):
				return []
		return []
	
	def set_child_batch_ids(self, batch_ids: List[str]):
		"""Set child batch IDs from list"""
		self.child_batch_ids = json.dumps(batch_ids) if batch_ids else None


class IMBLTBatchTransaction(Model, AuditMixin, BaseMixin):
	"""
	Detailed transaction history for batch movements.
	
	Tracks all movements of batch quantities for complete
	audit trail and genealogy tracking.
	"""
	__tablename__ = 'im_blt_batch_transaction'
	
	# Identity
	transaction_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Transaction Information
	transaction_date = Column(DateTime, default=datetime.utcnow, index=True)
	transaction_type = Column(String(50), nullable=False, index=True)  # Receipt, Issue, Transfer, Adjustment, etc.
	batch_id = Column(String(36), ForeignKey('im_blt_batch.batch_id'), nullable=False, index=True)
	
	# Movement Details
	from_location_id = Column(String(36), nullable=True)
	to_location_id = Column(String(36), nullable=True)
	quantity = Column(DECIMAL(15, 4), nullable=False)
	remaining_quantity = Column(DECIMAL(15, 4), nullable=False)  # After transaction
	
	# Reference Information
	reference_number = Column(String(100), nullable=True, index=True)
	reference_type = Column(String(50), nullable=True)  # SO, PO, TO, etc.
	
	# Associated Information
	customer_id = Column(String(36), nullable=True)
	supplier_id = Column(String(36), nullable=True)
	
	# Quality Status at Transaction
	quality_status = Column(String(50), nullable=True)
	quality_notes = Column(Text, nullable=True)
	
	# User and System Information
	user_id = Column(String(36), nullable=True)
	system_generated = Column(Boolean, default=False)
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		Index('idx_batch_transaction_date', 'tenant_id', 'batch_id', 'transaction_date'),
		Index('idx_batch_transaction_type', 'tenant_id', 'transaction_type'),
		Index('idx_batch_transaction_ref', 'tenant_id', 'reference_number', 'reference_type'),
	)
	
	# Relationships
	batch = relationship("IMBLTBatch", back_populates="transactions")
	
	def __repr__(self):
		return f"<IMBLTBatchTransaction {self.transaction_type}: {self.quantity}>"


class IMBLTQualityTest(Model, AuditMixin, BaseMixin):
	"""
	Quality test results for batches/lots.
	
	Records quality control testing results and compliance
	status for regulatory and quality assurance purposes.
	"""
	__tablename__ = 'im_blt_quality_test'
	
	# Identity
	test_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Test Information
	test_date = Column(DateTime, default=datetime.utcnow, index=True)
	test_type = Column(String(100), nullable=False, index=True)
	test_method = Column(String(100), nullable=True)
	batch_id = Column(String(36), ForeignKey('im_blt_batch.batch_id'), nullable=False, index=True)
	
	# Test Parameters
	parameter_name = Column(String(100), nullable=False)
	target_value = Column(String(100), nullable=True)
	actual_value = Column(String(100), nullable=True)
	unit_of_measure = Column(String(50), nullable=True)
	tolerance_range = Column(String(100), nullable=True)  # e.g., "±5%", "2.0-3.0"
	
	# Test Result
	test_result = Column(String(50), nullable=False, index=True)  # Pass, Fail, Out of Spec, Pending
	pass_fail = Column(Boolean, nullable=True)
	specification_met = Column(Boolean, nullable=True)
	
	# Lab Information
	lab_name = Column(String(200), nullable=True)
	technician_name = Column(String(200), nullable=True)
	equipment_used = Column(String(200), nullable=True)
	calibration_date = Column(Date, nullable=True)
	
	# Documentation
	certificate_number = Column(String(100), nullable=True)
	test_report_path = Column(String(500), nullable=True)  # Path to test report document
	
	# Quality Impact
	critical_parameter = Column(Boolean, default=False)
	affects_release = Column(Boolean, default=True)
	regulatory_requirement = Column(Boolean, default=False)
	
	# Status and Review
	status = Column(String(50), default='Final', index=True)  # Draft, Final, Reviewed, Disputed
	reviewed_by = Column(String(36), nullable=True)
	review_date = Column(DateTime, nullable=True)
	review_notes = Column(Text, nullable=True)
	
	# Additional Information
	notes = Column(Text, nullable=True)
	environmental_conditions = Column(String(200), nullable=True)  # Temperature, humidity during test
	
	# Constraints
	__table_args__ = (
		Index('idx_quality_test_batch', 'tenant_id', 'batch_id', 'test_date'),
		Index('idx_quality_test_type', 'tenant_id', 'test_type', 'test_result'),
		Index('idx_quality_test_parameter', 'tenant_id', 'parameter_name'),
	)
	
	# Relationships
	batch = relationship("IMBLTBatch", back_populates="quality_tests")
	
	def __repr__(self):
		return f"<IMBLTQualityTest {self.test_type}-{self.parameter_name}: {self.test_result}>"


class IMBLTRecallEvent(Model, AuditMixin, BaseMixin):
	"""
	Recall events and affected batches.
	
	Manages product recall events, affected batches,
	and recall execution tracking.
	"""
	__tablename__ = 'im_blt_recall_event'
	
	# Identity
	recall_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Recall Information
	recall_number = Column(String(100), nullable=False, index=True)
	recall_title = Column(String(200), nullable=False)
	recall_reason = Column(Text, nullable=False)
	recall_type = Column(String(50), nullable=False, index=True)  # Voluntary, Mandatory, Precautionary
	severity_level = Column(String(50), nullable=False, index=True)  # Class I, II, III or Critical, High, Medium, Low
	
	# Timing
	initiated_date = Column(DateTime, default=datetime.utcnow, index=True)
	notification_date = Column(DateTime, nullable=True)
	completion_date = Column(DateTime, nullable=True)
	
	# Scope
	affected_item_ids = Column(Text, nullable=True)  # JSON array of item IDs
	affected_batch_id = Column(String(36), ForeignKey('im_blt_batch.batch_id'), nullable=True, index=True)
	geographical_scope = Column(String(200), nullable=True)  # Regional, National, International
	
	# Status
	status = Column(String(50), default='Initiated', index=True)  # Initiated, In Progress, Completed, Terminated
	
	# Regulatory
	regulatory_authority = Column(String(200), nullable=True)  # FDA, EMA, etc.
	regulatory_case_number = Column(String(100), nullable=True)
	health_hazard_evaluation = Column(Text, nullable=True)
	
	# Communication
	customer_notification_sent = Column(Boolean, default=False)
	public_notification_required = Column(Boolean, default=False)
	press_release_issued = Column(Boolean, default=False)
	
	# Responsible Parties
	initiated_by = Column(String(36), nullable=True)
	recall_coordinator = Column(String(36), nullable=True)
	quality_manager = Column(String(36), nullable=True)
	
	# Tracking
	total_quantity_affected = Column(DECIMAL(15, 4), nullable=True)
	quantity_recovered = Column(DECIMAL(15, 4), default=0)
	quantity_destroyed = Column(DECIMAL(15, 4), default=0)
	recovery_percentage = Column(DECIMAL(5, 2), nullable=True)
	
	# Root Cause
	root_cause_analysis = Column(Text, nullable=True)
	corrective_actions = Column(Text, nullable=True)
	preventive_actions = Column(Text, nullable=True)
	
	# Documentation
	recall_report_path = Column(String(500), nullable=True)
	effectiveness_check_report = Column(String(500), nullable=True)
	
	# Additional Information
	notes = Column(Text, nullable=True)
	lessons_learned = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'recall_number', name='uq_recall_number_tenant'),
		Index('idx_recall_batch', 'tenant_id', 'affected_batch_id'),
		Index('idx_recall_status', 'tenant_id', 'status', 'severity_level'),
		Index('idx_recall_date', 'tenant_id', 'initiated_date'),
	)
	
	# Relationships
	batch = relationship("IMBLTBatch", back_populates="recall_events")
	recall_actions = relationship("IMBLTRecallAction", back_populates="recall_event")
	
	def __repr__(self):
		return f"<IMBLTRecallEvent {self.recall_number}: {self.status}>"
	
	def get_affected_item_ids(self) -> List[str]:
		"""Get affected item IDs as list"""
		if self.affected_item_ids:
			try:
				return json.loads(self.affected_item_ids)
			except (json.JSONDecodeError, TypeError):
				return []
		return []
	
	def set_affected_item_ids(self, item_ids: List[str]):
		"""Set affected item IDs from list"""
		self.affected_item_ids = json.dumps(item_ids) if item_ids else None


class IMBLTRecallAction(Model, AuditMixin, BaseMixin):
	"""
	Specific actions taken during recall execution.
	
	Tracks individual actions, notifications, and recovery
	activities during a recall event.
	"""
	__tablename__ = 'im_blt_recall_action'
	
	# Identity
	action_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Action Information
	recall_id = Column(String(36), ForeignKey('im_blt_recall_event.recall_id'), nullable=False, index=True)
	action_type = Column(String(50), nullable=False, index=True)  # Notification, Recovery, Destruction, Investigation
	action_description = Column(Text, nullable=False)
	
	# Timing
	planned_date = Column(DateTime, nullable=True)
	actual_date = Column(DateTime, nullable=True, index=True)
	due_date = Column(DateTime, nullable=True)
	
	# Responsible Party
	assigned_to = Column(String(36), nullable=True)
	completed_by = Column(String(36), nullable=True)
	
	# Scope
	target_customers = Column(Text, nullable=True)  # JSON array of customer IDs
	target_locations = Column(Text, nullable=True)  # JSON array of location IDs
	affected_quantity = Column(DECIMAL(15, 4), nullable=True)
	
	# Status
	status = Column(String(50), default='Planned', index=True)  # Planned, In Progress, Completed, Cancelled
	completion_percentage = Column(DECIMAL(5, 2), default=0)
	
	# Communication
	notification_method = Column(String(100), nullable=True)  # Email, Phone, Letter, etc.
	notification_sent = Column(Boolean, default=False)
	response_received = Column(Boolean, default=False)
	response_date = Column(DateTime, nullable=True)
	
	# Results
	quantity_recovered = Column(DECIMAL(15, 4), default=0)
	quantity_destroyed = Column(DECIMAL(15, 4), default=0)
	effectiveness_rating = Column(String(20), nullable=True)  # Effective, Partially Effective, Ineffective
	
	# Documentation
	evidence_collected = Column(Text, nullable=True)
	photos_taken = Column(Boolean, default=False)
	documentation_path = Column(String(500), nullable=True)
	
	# Additional Information
	notes = Column(Text, nullable=True)
	issues_encountered = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		Index('idx_recall_action_recall', 'tenant_id', 'recall_id', 'action_type'),
		Index('idx_recall_action_status', 'tenant_id', 'status', 'due_date'),
		Index('idx_recall_action_assigned', 'tenant_id', 'assigned_to', 'status'),
	)
	
	# Relationships
	recall_event = relationship("IMBLTRecallEvent", back_populates="recall_actions")
	
	def __repr__(self):
		return f"<IMBLTRecallAction {self.action_type}: {self.status}>"


class IMBLTGenealogyTrace(Model, AuditMixin, BaseMixin):
	"""
	Genealogy tracing records for batch relationships.
	
	Maintains forward and backward traceability links
	between parent and child batches across production processes.
	"""
	__tablename__ = 'im_blt_genealogy_trace'
	
	# Identity
	trace_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Relationship Information
	parent_batch_id = Column(String(36), nullable=False, index=True)
	child_batch_id = Column(String(36), nullable=False, index=True)
	relationship_type = Column(String(50), nullable=False)  # Blend, Split, Transform, Rework
	
	# Process Information
	process_step = Column(String(100), nullable=True)
	process_date = Column(DateTime, nullable=True, index=True)
	work_order_number = Column(String(100), nullable=True)
	operation_number = Column(String(50), nullable=True)
	
	# Quantity Information
	parent_quantity_consumed = Column(DECIMAL(15, 4), nullable=True)
	child_quantity_produced = Column(DECIMAL(15, 4), nullable=True)
	yield_percentage = Column(DECIMAL(5, 2), nullable=True)
	
	# Quality Impact
	quality_inherited = Column(Boolean, default=True)
	quality_notes = Column(Text, nullable=True)
	
	# Additional Information
	notes = Column(Text, nullable=True)
	process_parameters = Column(Text, nullable=True)  # JSON for process conditions
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'parent_batch_id', 'child_batch_id', 'relationship_type', 
						name='uq_genealogy_relationship'),
		Index('idx_genealogy_parent', 'tenant_id', 'parent_batch_id'),
		Index('idx_genealogy_child', 'tenant_id', 'child_batch_id'),
		Index('idx_genealogy_process', 'tenant_id', 'process_date'),
	)
	
	def __repr__(self):
		return f"<IMBLTGenealogyTrace {self.relationship_type}: {self.parent_batch_id} -> {self.child_batch_id}>"