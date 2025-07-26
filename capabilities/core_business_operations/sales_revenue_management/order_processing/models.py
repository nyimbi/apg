"""
Order Processing Models

Database models for order workflow management including fulfillment,
picking, packing, shipping, and invoicing processes.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ....auth_rbac.models import BaseMixin, AuditMixin, Model


class SOPOrderStatus(Model, AuditMixin, BaseMixin):
	"""
	Order status configuration and workflow definitions.
	
	Manages configurable order statuses and workflow transitions
	for different order types and business processes.
	"""
	__tablename__ = 'so_op_order_status'
	
	# Identity
	status_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Status Information
	status_code = Column(String(20), nullable=False, index=True)
	status_name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	
	# Workflow Configuration
	order_type = Column(String(20), nullable=True)  # Applies to specific order types
	sequence_number = Column(Integer, default=0)
	is_initial_status = Column(Boolean, default=False)
	is_final_status = Column(Boolean, default=False)
	
	# Display Properties
	display_color = Column(String(7), default='#000000')  # Hex color
	icon_class = Column(String(50), nullable=True)
	
	# Business Rules
	requires_approval = Column(Boolean, default=False)
	blocks_further_processing = Column(Boolean, default=False)
	triggers_inventory_allocation = Column(Boolean, default=False)
	triggers_picking = Column(Boolean, default=False)
	triggers_shipping = Column(Boolean, default=False)
	triggers_invoicing = Column(Boolean, default=False)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_system_status = Column(Boolean, default=False)  # Cannot be deleted
	
	# Allowed Transitions (JSON array of status codes)
	allowed_transitions = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'status_code', name='uq_sop_status_code_tenant'),
	)
	
	def __repr__(self):
		return f"<SOPOrderStatus {self.status_code} - {self.status_name}>"
	
	def get_allowed_transitions(self) -> List[str]:
		"""Get list of allowed status transitions"""
		if self.allowed_transitions:
			return json.loads(self.allowed_transitions)
		return []
	
	def set_allowed_transitions(self, transitions: List[str]):
		"""Set allowed status transitions"""
		self.allowed_transitions = json.dumps(transitions)
	
	def can_transition_to(self, target_status: str) -> bool:
		"""Check if can transition to target status"""
		allowed = self.get_allowed_transitions()
		return target_status in allowed if allowed else True


class SOPFulfillmentTask(Model, AuditMixin, BaseMixin):
	"""
	Fulfillment tasks for order processing workflow.
	
	Manages individual tasks in the order fulfillment process
	including picking, packing, quality control, and shipping.
	"""
	__tablename__ = 'so_op_fulfillment_task'
	
	# Identity
	task_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Task Information
	task_type = Column(String(20), nullable=False, index=True)  # PICK, PACK, QC, SHIP, etc.
	task_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Order Reference
	order_id = Column(String(36), nullable=False, index=True)  # References SOESalesOrder
	order_number = Column(String(50), nullable=False, index=True)
	
	# Task Details
	priority = Column(String(10), default='NORMAL')  # LOW, NORMAL, HIGH, URGENT
	estimated_duration = Column(Integer, nullable=True)  # Minutes
	
	# Assignment
	assigned_to = Column(String(36), nullable=True, index=True)  # User ID
	assigned_date = Column(DateTime, nullable=True)
	team_id = Column(String(36), nullable=True)  # Work team
	workstation_id = Column(String(36), nullable=True)  # Workstation
	
	# Status and Progress
	status = Column(String(20), default='PENDING', index=True)  # PENDING, ASSIGNED, IN_PROGRESS, COMPLETED, CANCELLED
	progress_percentage = Column(Integer, default=0)
	
	# Timing
	scheduled_start = Column(DateTime, nullable=True)
	scheduled_end = Column(DateTime, nullable=True)
	actual_start = Column(DateTime, nullable=True)
	actual_end = Column(DateTime, nullable=True)
	
	# Dependencies
	depends_on_task_ids = Column(Text, nullable=True)  # JSON array of task IDs
	blocks_task_ids = Column(Text, nullable=True)  # JSON array of task IDs
	
	# Task Data
	task_data = Column(Text, nullable=True)  # JSON data specific to task type
	
	# Quality Control
	quality_check_required = Column(Boolean, default=False)
	quality_check_passed = Column(Boolean, nullable=True)
	quality_check_notes = Column(Text, nullable=True)
	
	# Documents and Attachments
	instructions = Column(Text, nullable=True)
	attachments = Column(Text, nullable=True)  # JSON array of file paths
	
	# Notes
	notes = Column(Text, nullable=True)
	completion_notes = Column(Text, nullable=True)
	
	# Relationships
	line_tasks = relationship("SOPLineTask", back_populates="fulfillment_task", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<SOPFulfillmentTask {self.task_type} - {self.order_number}>"
	
	def get_dependencies(self) -> List[str]:
		"""Get list of dependency task IDs"""
		if self.depends_on_task_ids:
			return json.loads(self.depends_on_task_ids)
		return []
	
	def set_dependencies(self, task_ids: List[str]):
		"""Set dependency task IDs"""
		self.depends_on_task_ids = json.dumps(task_ids)
	
	def get_blocked_tasks(self) -> List[str]:
		"""Get list of tasks blocked by this task"""
		if self.blocks_task_ids:
			return json.loads(self.blocks_task_ids)
		return []
	
	def set_blocked_tasks(self, task_ids: List[str]):
		"""Set blocked task IDs"""
		self.blocks_task_ids = json.dumps(task_ids)
	
	def get_task_data(self) -> Dict[str, Any]:
		"""Get task-specific data"""
		if self.task_data:
			return json.loads(self.task_data)
		return {}
	
	def set_task_data(self, data: Dict[str, Any]):
		"""Set task-specific data"""
		self.task_data = json.dumps(data)
	
	def start_task(self, user_id: str):
		"""Start the task"""
		if self.status != 'ASSIGNED':
			raise ValueError("Task must be assigned before starting")
		
		self.status = 'IN_PROGRESS'
		self.actual_start = datetime.utcnow()
		self.assigned_to = user_id
	
	def complete_task(self, user_id: str, completion_notes: str = None):
		"""Complete the task"""
		if self.status != 'IN_PROGRESS':
			raise ValueError("Task must be in progress to complete")
		
		self.status = 'COMPLETED'
		self.actual_end = datetime.utcnow()
		self.progress_percentage = 100
		self.completion_notes = completion_notes
	
	def cancel_task(self, user_id: str, reason: str):
		"""Cancel the task"""
		self.status = 'CANCELLED'
		self.completion_notes = f"Cancelled by {user_id}: {reason}"


class SOPLineTask(Model, AuditMixin, BaseMixin):
	"""
	Line-level fulfillment tasks for individual order items.
	
	Manages tasks specific to individual order lines including
	item picking, quantity verification, and line completion.
	"""
	__tablename__ = 'so_op_line_task'
	
	# Identity
	line_task_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	task_id = Column(String(36), ForeignKey('so_op_fulfillment_task.task_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Reference
	order_line_id = Column(String(36), nullable=False, index=True)  # References SOEOrderLine
	line_number = Column(Integer, nullable=False)
	
	# Item Information
	item_id = Column(String(36), nullable=True, index=True)
	item_code = Column(String(50), nullable=False, index=True)
	item_description = Column(String(200), nullable=True)
	
	# Quantities
	quantity_required = Column(DECIMAL(12, 4), default=0.0000)
	quantity_picked = Column(DECIMAL(12, 4), default=0.0000)
	quantity_packed = Column(DECIMAL(12, 4), default=0.0000)
	quantity_shipped = Column(DECIMAL(12, 4), default=0.0000)
	
	# Location Information
	warehouse_id = Column(String(36), nullable=True, index=True)
	location_id = Column(String(36), nullable=True)
	bin_location = Column(String(50), nullable=True)
	
	# Inventory Tracking
	lot_number = Column(String(50), nullable=True)
	serial_numbers = Column(Text, nullable=True)  # JSON array for multiple serials
	expiry_date = Column(Date, nullable=True)
	
	# Status
	line_status = Column(String(20), default='PENDING')  # PENDING, PICKED, PACKED, SHIPPED, CANCELLED
	
	# Verification
	picked_by = Column(String(36), nullable=True)
	picked_date = Column(DateTime, nullable=True)
	verified_by = Column(String(36), nullable=True)
	verified_date = Column(DateTime, nullable=True)
	
	# Issues and Exceptions
	has_issues = Column(Boolean, default=False)
	issue_type = Column(String(50), nullable=True)  # SHORT_PICK, DAMAGED, EXPIRED, etc.
	issue_description = Column(Text, nullable=True)
	resolution = Column(Text, nullable=True)
	
	# Special Handling
	special_instructions = Column(Text, nullable=True)
	requires_special_handling = Column(Boolean, default=False)
	hazardous_material = Column(Boolean, default=False)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	fulfillment_task = relationship("SOPFulfillmentTask", back_populates="line_tasks")
	
	def __repr__(self):
		return f"<SOPLineTask {self.line_number}: {self.item_code}>"
	
	def get_serial_numbers(self) -> List[str]:
		"""Get list of serial numbers"""
		if self.serial_numbers:
			return json.loads(self.serial_numbers)
		return []
	
	def set_serial_numbers(self, serials: List[str]):
		"""Set serial numbers"""
		self.serial_numbers = json.dumps(serials)
	
	def pick_quantity(self, quantity: Decimal, user_id: str, location: str = None):
		"""Record picked quantity"""
		self.quantity_picked += quantity
		self.picked_by = user_id
		self.picked_date = datetime.utcnow()
		
		if location:
			self.bin_location = location
		
		if self.quantity_picked >= self.quantity_required:
			self.line_status = 'PICKED'
	
	def pack_quantity(self, quantity: Decimal, user_id: str):
		"""Record packed quantity"""
		if quantity > self.quantity_picked:
			raise ValueError("Cannot pack more than picked quantity")
		
		self.quantity_packed += quantity
		
		if self.quantity_packed >= self.quantity_required:
			self.line_status = 'PACKED'
	
	def report_issue(self, issue_type: str, description: str, user_id: str):
		"""Report an issue with the line"""
		self.has_issues = True
		self.issue_type = issue_type
		self.issue_description = description
		self.notes = (self.notes or '') + f"\nIssue reported by {user_id}: {description}"


class SOPShipment(Model, AuditMixin, BaseMixin):
	"""
	Shipment records for order fulfillment.
	
	Manages shipment creation, tracking, and delivery confirmation
	for completed orders and partial shipments.
	"""
	__tablename__ = 'so_op_shipment'
	
	# Identity
	shipment_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Shipment Information
	shipment_number = Column(String(50), nullable=False, index=True)
	shipment_type = Column(String(20), default='STANDARD')  # STANDARD, PARTIAL, BACKORDER, DROP_SHIP
	
	# Order References
	order_ids = Column(Text, nullable=False)  # JSON array of order IDs (can ship multiple orders)
	
	# Customer Information
	customer_id = Column(String(36), nullable=False, index=True)
	customer_name = Column(String(200), nullable=False)
	
	# Shipping Address
	ship_to_name = Column(String(100), nullable=True)
	ship_to_address_line1 = Column(String(100), nullable=False)
	ship_to_address_line2 = Column(String(100), nullable=True)
	ship_to_city = Column(String(50), nullable=False)
	ship_to_state_province = Column(String(50), nullable=False)
	ship_to_postal_code = Column(String(20), nullable=False)
	ship_to_country = Column(String(50), nullable=False)
	
	# Shipping Details
	carrier = Column(String(50), nullable=False)
	service_level = Column(String(50), nullable=False)
	tracking_number = Column(String(100), nullable=True, index=True)
	
	# Dates
	ship_date = Column(Date, nullable=False, index=True)
	estimated_delivery_date = Column(Date, nullable=True)
	actual_delivery_date = Column(Date, nullable=True)
	
	# Status
	shipment_status = Column(String(20), default='CREATED', index=True)  # CREATED, SHIPPED, IN_TRANSIT, DELIVERED, RETURNED
	
	# Shipping Costs
	shipping_cost = Column(DECIMAL(15, 2), default=0.00)
	insurance_cost = Column(DECIMAL(15, 2), default=0.00)
	total_shipping_cost = Column(DECIMAL(15, 2), default=0.00)
	
	# Package Information
	total_packages = Column(Integer, default=1)
	total_weight = Column(DECIMAL(10, 2), default=0.00)
	total_volume = Column(DECIMAL(10, 2), default=0.00)
	
	# Documents
	shipping_label_path = Column(String(500), nullable=True)
	packing_slip_path = Column(String(500), nullable=True)
	bill_of_lading_path = Column(String(500), nullable=True)
	
	# Special Instructions
	shipping_instructions = Column(Text, nullable=True)
	delivery_instructions = Column(Text, nullable=True)
	
	# Integration
	carrier_reference = Column(String(100), nullable=True)
	manifest_id = Column(String(50), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'shipment_number', name='uq_sop_shipment_number_tenant'),
	)
	
	# Relationships
	packages = relationship("SOPShipmentPackage", back_populates="shipment", cascade="all, delete-orphan")
	tracking_events = relationship("SOPTrackingEvent", back_populates="shipment", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<SOPShipment {self.shipment_number} - {self.customer_name}>"
	
	def get_order_ids(self) -> List[str]:
		"""Get list of order IDs in shipment"""
		return json.loads(self.order_ids)
	
	def set_order_ids(self, order_ids: List[str]):
		"""Set order IDs for shipment"""
		self.order_ids = json.dumps(order_ids)
	
	def ship_shipment(self, user_id: str, tracking_number: str = None):
		"""Mark shipment as shipped"""
		if self.shipment_status != 'CREATED':
			raise ValueError("Only created shipments can be shipped")
		
		self.shipment_status = 'SHIPPED'
		self.ship_date = date.today()
		
		if tracking_number:
			self.tracking_number = tracking_number
	
	def deliver_shipment(self, delivery_date: date = None, user_id: str = None):
		"""Mark shipment as delivered"""
		self.shipment_status = 'DELIVERED'
		self.actual_delivery_date = delivery_date or date.today()


class SOPShipmentPackage(Model, AuditMixin, BaseMixin):
	"""
	Individual packages within a shipment.
	
	Manages package-level details including contents, dimensions,
	and tracking for multi-package shipments.
	"""
	__tablename__ = 'so_op_shipment_package'
	
	# Identity
	package_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	shipment_id = Column(String(36), ForeignKey('so_op_shipment.shipment_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Package Information
	package_number = Column(Integer, nullable=False)  # 1, 2, 3, etc.
	package_type = Column(String(20), default='BOX')  # BOX, ENVELOPE, TUBE, PALLET
	
	# Dimensions
	length = Column(DECIMAL(8, 2), nullable=True)  # inches or cm
	width = Column(DECIMAL(8, 2), nullable=True)
	height = Column(DECIMAL(8, 2), nullable=True)
	weight = Column(DECIMAL(10, 2), nullable=True)  # lbs or kg
	
	# Tracking
	package_tracking_number = Column(String(100), nullable=True)
	
	# Contents (JSON array of line items)
	contents = Column(Text, nullable=True)
	
	# Special Handling
	fragile = Column(Boolean, default=False)
	hazardous = Column(Boolean, default=False)
	temperature_controlled = Column(Boolean, default=False)
	
	# Package Value
	declared_value = Column(DECIMAL(15, 2), default=0.00)
	insurance_value = Column(DECIMAL(15, 2), default=0.00)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	shipment = relationship("SOPShipment", back_populates="packages")
	
	def __repr__(self):
		return f"<SOPShipmentPackage {self.package_number} - {self.shipment.shipment_number}>"
	
	def get_contents(self) -> List[Dict[str, Any]]:
		"""Get package contents"""
		if self.contents:
			return json.loads(self.contents)
		return []
	
	def set_contents(self, contents: List[Dict[str, Any]]):
		"""Set package contents"""
		self.contents = json.dumps(contents)
	
	def add_item(self, item_code: str, quantity: Decimal, description: str = None):
		"""Add item to package contents"""
		contents = self.get_contents()
		contents.append({
			'item_code': item_code,
			'quantity': float(quantity),
			'description': description
		})
		self.set_contents(contents)


class SOPTrackingEvent(Model, AuditMixin, BaseMixin):
	"""
	Shipment tracking events and status updates.
	
	Records tracking events from carriers and internal systems
	for shipment visibility and customer notifications.
	"""
	__tablename__ = 'so_op_tracking_event'
	
	# Identity
	event_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	shipment_id = Column(String(36), ForeignKey('so_op_shipment.shipment_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Event Information
	event_type = Column(String(20), nullable=False)  # PICKUP, IN_TRANSIT, OUT_FOR_DELIVERY, DELIVERED, EXCEPTION
	event_code = Column(String(10), nullable=True)  # Carrier-specific event code
	description = Column(String(500), nullable=False)
	
	# Location
	location_city = Column(String(50), nullable=True)
	location_state = Column(String(50), nullable=True)
	location_country = Column(String(50), nullable=True)
	location_facility = Column(String(100), nullable=True)
	
	# Timing
	event_date = Column(Date, nullable=False, index=True)
	event_time = Column(DateTime, nullable=False, index=True)
	
	# Source
	event_source = Column(String(20), default='CARRIER')  # CARRIER, INTERNAL, CUSTOMER
	carrier_reference = Column(String(100), nullable=True)
	
	# Exception Information
	is_exception = Column(Boolean, default=False)
	exception_type = Column(String(50), nullable=True)
	resolution_required = Column(Boolean, default=False)
	
	# Customer Notification
	customer_notified = Column(Boolean, default=False)
	notification_date = Column(DateTime, nullable=True)
	notification_method = Column(String(20), nullable=True)  # EMAIL, SMS, PHONE
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	shipment = relationship("SOPShipment", back_populates="tracking_events")
	
	def __repr__(self):
		return f"<SOPTrackingEvent {self.event_type} - {self.shipment.shipment_number}>"


class SOPOrderWorkflow(Model, AuditMixin, BaseMixin):
	"""
	Order workflow configuration and rules.
	
	Defines workflow templates and business rules for different
	order types and customer segments.
	"""
	__tablename__ = 'so_op_order_workflow'
	
	# Identity
	workflow_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Workflow Information
	workflow_name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	workflow_type = Column(String(20), default='STANDARD')  # STANDARD, EXPRESS, DROP_SHIP, etc.
	
	# Application Rules
	order_type = Column(String(20), nullable=True)  # Apply to specific order types
	customer_type = Column(String(50), nullable=True)  # Apply to customer types
	minimum_order_value = Column(DECIMAL(15, 2), nullable=True)
	maximum_order_value = Column(DECIMAL(15, 2), nullable=True)
	
	# Workflow Steps (JSON array)
	workflow_steps = Column(Text, nullable=False)
	
	# SLA Configuration
	total_sla_hours = Column(Integer, nullable=True)  # Total processing SLA
	step_slas = Column(Text, nullable=True)  # JSON object with step SLAs
	
	# Approval Requirements
	requires_approval = Column(Boolean, default=False)
	approval_rules = Column(Text, nullable=True)  # JSON configuration
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_default = Column(Boolean, default=False)
	priority = Column(Integer, default=0)  # Higher priority workflows override lower
	
	# Effective Dates
	effective_date = Column(Date, nullable=True)
	expiration_date = Column(Date, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'workflow_name', name='uq_sop_workflow_name_tenant'),
	)
	
	def __repr__(self):
		return f"<SOPOrderWorkflow {self.workflow_name}>"
	
	def get_workflow_steps(self) -> List[Dict[str, Any]]:
		"""Get workflow steps configuration"""
		return json.loads(self.workflow_steps)
	
	def set_workflow_steps(self, steps: List[Dict[str, Any]]):
		"""Set workflow steps configuration"""
		self.workflow_steps = json.dumps(steps)
	
	def get_step_slas(self) -> Dict[str, int]:
		"""Get SLA configuration by step"""
		if self.step_slas:
			return json.loads(self.step_slas)
		return {}
	
	def set_step_slas(self, slas: Dict[str, int]):
		"""Set SLA configuration by step"""
		self.step_slas = json.dumps(slas)
	
	def is_applicable(self, order_data: Dict[str, Any]) -> bool:
		"""Check if workflow applies to order"""
		if self.order_type and order_data.get('order_type') != self.order_type:
			return False
		
		if self.customer_type and order_data.get('customer_type') != self.customer_type:
			return False
		
		order_value = order_data.get('total_amount', 0)
		if self.minimum_order_value and order_value < self.minimum_order_value:
			return False
		
		if self.maximum_order_value and order_value > self.maximum_order_value:
			return False
		
		# Check effective dates
		check_date = date.today()
		if self.effective_date and check_date < self.effective_date:
			return False
		
		if self.expiration_date and check_date > self.expiration_date:
			return False
		
		return True