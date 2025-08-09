"""
Requisitioning Models

Database models for the Requisitioning sub-capability including requisitions,
approval workflows, line items, and related requisition management functionality.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class PPRRequisition(Model, AuditMixin, BaseMixin):
	"""
	Purchase requisitions for procurement requests.
	
	Manages employee purchase requests with approval workflows,
	budget checking, and conversion to purchase orders.
	"""
	__tablename__ = 'ppr_requisition'
	
	# Identity
	requisition_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Requisition Information
	requisition_number = Column(String(50), nullable=False, index=True)
	title = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	business_justification = Column(Text, nullable=True)
	
	# Requester Information
	requestor_id = Column(String(36), nullable=False, index=True)
	requestor_name = Column(String(100), nullable=False)
	requestor_email = Column(String(100), nullable=True)
	department = Column(String(50), nullable=True, index=True)
	cost_center = Column(String(20), nullable=True, index=True)
	
	# Request Details
	request_date = Column(Date, default=date.today, nullable=False, index=True)
	required_date = Column(Date, nullable=False, index=True)
	delivery_location = Column(String(200), nullable=True)
	
	# Status and Workflow
	status = Column(String(20), default='Draft', index=True)  # Draft, Submitted, Approved, Rejected, Cancelled, Converted
	workflow_status = Column(String(50), nullable=True)  # Custom workflow status
	priority = Column(String(20), default='Normal')  # Low, Normal, High, Urgent
	
	# Financial Information
	currency_code = Column(String(3), default='USD')
	subtotal_amount = Column(DECIMAL(15, 2), default=0.00)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	total_amount = Column(DECIMAL(15, 2), default=0.00, index=True)
	
	# Budget Information
	budget_year = Column(Integer, nullable=True)
	budget_period = Column(String(20), nullable=True)
	budget_account_id = Column(String(36), nullable=True)
	budget_checked = Column(Boolean, default=False)
	budget_available = Column(DECIMAL(15, 2), default=0.00)
	budget_encumbered = Column(Boolean, default=False)
	
	# Approval Workflow
	requires_approval = Column(Boolean, default=True)
	approval_level = Column(Integer, default=0)
	current_approver_id = Column(String(36), nullable=True)
	current_approver_name = Column(String(100), nullable=True)
	
	# Final Approval
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	approved_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Rejection Information
	rejected = Column(Boolean, default=False)
	rejected_by = Column(String(36), nullable=True)
	rejected_date = Column(DateTime, nullable=True)
	rejection_reason = Column(Text, nullable=True)
	
	# Processing Information
	submitted = Column(Boolean, default=False)
	submitted_date = Column(DateTime, nullable=True)
	processed_by = Column(String(36), nullable=True)
	processed_date = Column(DateTime, nullable=True)
	
	# Conversion Information
	converted_to_po = Column(Boolean, default=False)
	purchase_order_id = Column(String(36), nullable=True)
	conversion_date = Column(DateTime, nullable=True)
	conversion_notes = Column(Text, nullable=True)
	
	# Project/Activity Tracking
	project_id = Column(String(36), nullable=True)
	project_name = Column(String(100), nullable=True)
	activity_code = Column(String(20), nullable=True)
	
	# Special Handling
	rush_order = Column(Boolean, default=False)
	drop_ship = Column(Boolean, default=False)
	special_instructions = Column(Text, nullable=True)
	
	# Document Management
	attachment_count = Column(Integer, default=0)
	document_path = Column(String(500), nullable=True)
	
	# Notes and Comments
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'requisition_number', name='uq_requisition_number_tenant'),
	)
	
	# Relationships
	lines = relationship("PPRRequisitionLine", back_populates="requisition", cascade="all, delete-orphan")
	workflow_steps = relationship("PPRApprovalWorkflow", back_populates="requisition", cascade="all, delete-orphan")
	comments = relationship("PPRRequisitionComment", back_populates="requisition", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<PPRRequisition {self.requisition_number} - {self.status} - ${self.total_amount}>"
	
	def calculate_totals(self):
		"""Recalculate requisition totals from lines"""
		self.subtotal_amount = sum(line.line_amount for line in self.lines)
		self.tax_amount = sum(line.tax_amount for line in self.lines)
		self.total_amount = self.subtotal_amount + self.tax_amount
	
	def can_submit(self) -> bool:
		"""Check if requisition can be submitted for approval"""
		return (
			self.status == 'Draft' and
			self.total_amount > 0 and
			len(self.lines) > 0 and
			self.required_date >= date.today()
		)
	
	def can_approve(self, user_id: str) -> bool:
		"""Check if requisition can be approved by user"""
		return (
			self.status == 'Submitted' and
			not self.approved and
			not self.rejected and
			self.current_approver_id == user_id
		)
	
	def can_reject(self, user_id: str) -> bool:
		"""Check if requisition can be rejected by user"""
		return (
			self.status == 'Submitted' and
			not self.approved and
			not self.rejected and
			self.current_approver_id == user_id
		)
	
	def can_edit(self, user_id: str) -> bool:
		"""Check if requisition can be edited by user"""
		return (
			(self.status == 'Draft' and self.requestor_id == user_id) or
			self.status == 'Rejected'
		)
	
	def can_cancel(self, user_id: str) -> bool:
		"""Check if requisition can be cancelled"""
		return (
			self.status in ['Draft', 'Submitted'] and
			(self.requestor_id == user_id or self.current_approver_id == user_id) and
			not self.converted_to_po
		)
	
	def submit_requisition(self):
		"""Submit requisition for approval"""
		if not self.can_submit():
			raise ValueError("Requisition cannot be submitted")
		
		self.status = 'Submitted'
		self.submitted = True
		self.submitted_date = datetime.utcnow()
		
		# Initialize approval workflow
		self._initialize_approval_workflow()
	
	def approve_requisition(self, user_id: str, comments: str = None):
		"""Approve the requisition at current level"""
		if not self.can_approve(user_id):
			raise ValueError("Requisition cannot be approved by this user")
		
		# Record approval at current level
		current_workflow = self._get_current_workflow_step()
		if current_workflow:
			current_workflow.approve_step(user_id, comments)
		
		# Check if more approvals needed
		if self._needs_additional_approval():
			self._advance_approval_workflow()
		else:
			# Final approval
			self.approved = True
			self.approved_by = user_id
			self.approved_date = datetime.utcnow()
			self.approved_amount = self.total_amount
			self.status = 'Approved'
			self.current_approver_id = None
			self.current_approver_name = None
	
	def reject_requisition(self, user_id: str, reason: str):
		"""Reject the requisition"""
		if not self.can_reject(user_id):
			raise ValueError("Requisition cannot be rejected by this user")
		
		self.rejected = True
		self.rejected_by = user_id
		self.rejected_date = datetime.utcnow()
		self.rejection_reason = reason
		self.status = 'Rejected'
		self.current_approver_id = None
		self.current_approver_name = None
		
		# Record rejection in workflow
		current_workflow = self._get_current_workflow_step()
		if current_workflow:
			current_workflow.reject_step(user_id, reason)
	
	def convert_to_po(self, purchase_order_id: str, user_id: str, notes: str = None):
		"""Convert requisition to purchase order"""
		if self.status != 'Approved':
			raise ValueError("Only approved requisitions can be converted to PO")
		
		self.converted_to_po = True
		self.purchase_order_id = purchase_order_id
		self.conversion_date = datetime.utcnow()
		self.conversion_notes = notes
		self.status = 'Converted'
		self.processed_by = user_id
		self.processed_date = datetime.utcnow()
	
	def check_budget_availability(self) -> Dict[str, Any]:
		"""Check budget availability for requisition"""
		# TODO: Integrate with budgeting system
		# This is a placeholder implementation
		
		if not self.budget_account_id:
			return {
				'available': False,
				'reason': 'No budget account specified',
				'budget_available': 0,
				'amount_requested': self.total_amount
			}
		
		# Simulate budget check
		# In real implementation, this would query budget tables
		available_budget = Decimal('100000.00')  # Placeholder
		
		return {
			'available': available_budget >= self.total_amount,
			'reason': 'Sufficient budget' if available_budget >= self.total_amount else 'Insufficient budget',
			'budget_available': available_budget,
			'amount_requested': self.total_amount,
			'remaining_after': available_budget - self.total_amount
		}
	
	def _initialize_approval_workflow(self):
		"""Initialize approval workflow based on amount and rules"""
		from .service import RequisitioningService
		
		service = RequisitioningService(self.tenant_id)
		workflow_config = service.get_approval_workflow_config(
			amount=self.total_amount,
			department=self.department,
			requestor_id=self.requestor_id
		)
		
		# Create workflow steps
		for step_config in workflow_config['steps']:
			workflow_step = PPRApprovalWorkflow(
				requisition_id=self.requisition_id,
				tenant_id=self.tenant_id,
				step_order=step_config['order'],
				approver_id=step_config['approver_id'],
				approver_name=step_config['approver_name'],
				approver_role=step_config.get('role'),
				required=step_config.get('required', True),
				approval_limit=step_config.get('limit'),
				step_name=step_config.get('name', f"Level {step_config['order']} Approval")
			)
			self.workflow_steps.append(workflow_step)
		
		# Set first approver
		if workflow_config['steps']:
			first_step = workflow_config['steps'][0]
			self.current_approver_id = first_step['approver_id']
			self.current_approver_name = first_step['approver_name']
			self.approval_level = 1
	
	def _get_current_workflow_step(self) -> Optional['PPRApprovalWorkflow']:
		"""Get current workflow step"""
		return next(
			(step for step in self.workflow_steps 
			 if step.step_order == self.approval_level and not step.completed),
			None
		)
	
	def _needs_additional_approval(self) -> bool:
		"""Check if additional approval levels are needed"""
		return any(
			step for step in self.workflow_steps
			if step.step_order > self.approval_level and step.required
		)
	
	def _advance_approval_workflow(self):
		"""Advance to next approval level"""
		next_step = next(
			(step for step in self.workflow_steps
			 if step.step_order > self.approval_level and step.required),
			None
		)
		
		if next_step:
			self.approval_level = next_step.step_order
			self.current_approver_id = next_step.approver_id
			self.current_approver_name = next_step.approver_name
	
	def get_approval_history(self) -> List[Dict[str, Any]]:
		"""Get approval history for requisition"""
		return [
			{
				'step_order': step.step_order,
				'step_name': step.step_name,
				'approver_name': step.approver_name,
				'status': step.status,
				'approved_date': step.approved_date,
				'comments': step.comments,
				'required': step.required
			}
			for step in sorted(self.workflow_steps, key=lambda x: x.step_order)
		]


class PPRRequisitionLine(Model, AuditMixin, BaseMixin):
	"""
	Individual requisition line items.
	
	Contains detailed line-level information including items, quantities,
	amounts, and GL account coding.
	"""
	__tablename__ = 'ppr_requisition_line'
	
	# Identity
	line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	requisition_id = Column(String(36), ForeignKey('ppr_requisition.requisition_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	description = Column(Text, nullable=False)
	detailed_specification = Column(Text, nullable=True)
	
	# Item Information
	item_code = Column(String(50), nullable=True, index=True)
	item_description = Column(String(200), nullable=True)
	item_category = Column(String(50), nullable=True)
	manufacturer = Column(String(100), nullable=True)
	model_number = Column(String(50), nullable=True)
	part_number = Column(String(50), nullable=True)
	
	# Quantities and Units
	quantity_requested = Column(DECIMAL(12, 4), default=1.0000)
	unit_of_measure = Column(String(20), default='EA')
	unit_price = Column(DECIMAL(15, 4), default=0.0000)
	line_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Tax Information
	tax_code = Column(String(20), nullable=True)
	tax_rate = Column(DECIMAL(5, 2), default=0.00)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	is_tax_inclusive = Column(Boolean, default=False)
	
	# GL Account Coding
	gl_account_id = Column(String(36), nullable=False, index=True)
	cost_center = Column(String(20), nullable=True)
	department = Column(String(20), nullable=True)
	project_id = Column(String(36), nullable=True)
	activity_code = Column(String(20), nullable=True)
	
	# Delivery Information
	required_date = Column(Date, nullable=True)
	delivery_location = Column(String(200), nullable=True)
	special_instructions = Column(Text, nullable=True)
	
	# Vendor Information (if known)
	preferred_vendor_id = Column(String(36), nullable=True)
	preferred_vendor_name = Column(String(100), nullable=True)
	vendor_part_number = Column(String(50), nullable=True)
	
	# Asset Information (for capital items)
	is_asset = Column(Boolean, default=False)
	asset_category = Column(String(50), nullable=True)
	useful_life_years = Column(Integer, nullable=True)
	
	# Service Information (for services)
	is_service = Column(Boolean, default=False)
	service_period_start = Column(Date, nullable=True)
	service_period_end = Column(Date, nullable=True)
	
	# Additional Information
	warranty_required = Column(Boolean, default=False)
	warranty_period = Column(String(50), nullable=True)
	technical_specs = Column(Text, nullable=True)
	
	# Status
	line_status = Column(String(20), default='Active')  # Active, Cancelled
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	requisition = relationship("PPRRequisition", back_populates="lines")
	
	def __repr__(self):
		return f"<PPRRequisitionLine {self.line_number}: {self.description} - ${self.line_amount}>"
	
	def calculate_line_amount(self):
		"""Calculate line amount from quantity and unit price"""
		self.line_amount = self.quantity_requested * self.unit_price
		self.calculate_tax()
	
	def calculate_tax(self):
		"""Calculate tax amount for the line"""
		if self.tax_rate > 0:
			if self.is_tax_inclusive:
				# Extract tax from inclusive amount
				self.tax_amount = self.line_amount * (self.tax_rate / (100 + self.tax_rate))
			else:
				# Add tax to exclusive amount
				self.tax_amount = self.line_amount * (self.tax_rate / 100)
		else:
			self.tax_amount = Decimal('0.00')
	
	def get_total_amount(self) -> Decimal:
		"""Get total amount including tax (if not tax inclusive)"""
		if self.is_tax_inclusive:
			return self.line_amount
		else:
			return self.line_amount + self.tax_amount


class PPRApprovalWorkflow(Model, AuditMixin, BaseMixin):
	"""
	Approval workflow steps for requisitions.
	
	Tracks multi-level approval process with configurable workflow
	steps based on amount thresholds and organizational hierarchy.
	"""
	__tablename__ = 'ppr_approval_workflow'
	
	# Identity
	workflow_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	requisition_id = Column(String(36), ForeignKey('ppr_requisition.requisition_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Workflow Step Information
	step_order = Column(Integer, nullable=False, index=True)  # 1, 2, 3, etc.
	step_name = Column(String(100), nullable=False)
	step_description = Column(Text, nullable=True)
	
	# Approver Information
	approver_id = Column(String(36), nullable=False, index=True)
	approver_name = Column(String(100), nullable=False)
	approver_email = Column(String(100), nullable=True)
	approver_role = Column(String(50), nullable=True)
	
	# Approval Configuration
	required = Column(Boolean, default=True)
	approval_limit = Column(DECIMAL(15, 2), nullable=True)  # Max amount this approver can approve
	parallel_approval = Column(Boolean, default=False)  # Can be approved in parallel with other steps
	
	# Status and Timing
	status = Column(String(20), default='Pending', index=True)  # Pending, Approved, Rejected, Skipped
	assigned_date = Column(DateTime, default=datetime.utcnow)
	due_date = Column(DateTime, nullable=True)
	completed = Column(Boolean, default=False)
	
	# Approval Information
	approved = Column(Boolean, default=False)
	approved_date = Column(DateTime, nullable=True)
	rejected = Column(Boolean, default=False)
	rejected_date = Column(DateTime, nullable=True)
	
	# Comments and Notes
	comments = Column(Text, nullable=True)
	rejection_reason = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	
	# Delegation Information
	delegated_to_id = Column(String(36), nullable=True)
	delegated_to_name = Column(String(100), nullable=True)
	delegated_date = Column(DateTime, nullable=True)
	delegation_reason = Column(Text, nullable=True)
	
	# Escalation Information
	escalated = Column(Boolean, default=False)
	escalated_to_id = Column(String(36), nullable=True)
	escalated_date = Column(DateTime, nullable=True)
	escalation_reason = Column(Text, nullable=True)
	
	# Notification Information
	notification_sent = Column(Boolean, default=False)
	notification_sent_date = Column(DateTime, nullable=True)
	reminder_count = Column(Integer, default=0)
	last_reminder_date = Column(DateTime, nullable=True)
	
	# Relationships
	requisition = relationship("PPRRequisition", back_populates="workflow_steps")
	
	def __repr__(self):
		return f"<PPRApprovalWorkflow {self.step_name} - {self.approver_name} - {self.status}>"
	
	def can_approve(self, user_id: str) -> bool:
		"""Check if user can approve this step"""
		return (
			self.status == 'Pending' and
			not self.completed and
			(self.approver_id == user_id or self.delegated_to_id == user_id)
		)
	
	def approve_step(self, user_id: str, comments: str = None):
		"""Approve this workflow step"""
		if not self.can_approve(user_id):
			raise ValueError("User cannot approve this workflow step")
		
		self.approved = True
		self.approved_date = datetime.utcnow()
		self.status = 'Approved'
		self.completed = True
		self.comments = comments
	
	def reject_step(self, user_id: str, reason: str):
		"""Reject this workflow step"""
		if not self.can_approve(user_id):
			raise ValueError("User cannot reject this workflow step")
		
		self.rejected = True
		self.rejected_date = datetime.utcnow()
		self.status = 'Rejected'
		self.completed = True
		self.rejection_reason = reason
	
	def delegate_approval(self, user_id: str, delegate_to_id: str, delegate_to_name: str, reason: str):
		"""Delegate approval to another user"""
		if self.approver_id != user_id:
			raise ValueError("Only the assigned approver can delegate")
		
		self.delegated_to_id = delegate_to_id
		self.delegated_to_name = delegate_to_name
		self.delegated_date = datetime.utcnow()
		self.delegation_reason = reason
	
	def escalate_approval(self, escalate_to_id: str, reason: str):
		"""Escalate approval to higher level"""
		self.escalated = True
		self.escalated_to_id = escalate_to_id
		self.escalated_date = datetime.utcnow()
		self.escalation_reason = reason
	
	def is_overdue(self) -> bool:
		"""Check if approval is overdue"""
		if not self.due_date or self.completed:
			return False
		return datetime.utcnow() > self.due_date
	
	def get_days_pending(self) -> int:
		"""Get number of days step has been pending"""
		if self.completed:
			completion_date = self.approved_date or self.rejected_date
			return (completion_date - self.assigned_date).days
		else:
			return (datetime.utcnow() - self.assigned_date).days


class PPRRequisitionComment(Model, AuditMixin, BaseMixin):
	"""
	Comments and notes on requisitions.
	
	Tracks communication and collaboration on requisition processing
	with support for internal and external comments.
	"""
	__tablename__ = 'ppr_requisition_comment'
	
	# Identity
	comment_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	requisition_id = Column(String(36), ForeignKey('ppr_requisition.requisition_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Comment Information
	comment_text = Column(Text, nullable=False)
	comment_type = Column(String(20), default='General')  # General, Approval, Rejection, Question, Answer
	is_internal = Column(Boolean, default=True)  # Internal vs external comment
	is_system_generated = Column(Boolean, default=False)
	
	# Author Information
	author_id = Column(String(36), nullable=False, index=True)
	author_name = Column(String(100), nullable=False)
	author_role = Column(String(50), nullable=True)
	
	# Timing
	comment_date = Column(DateTime, default=datetime.utcnow, index=True)
	
	# Visibility and Permissions
	visible_to_requestor = Column(Boolean, default=True)
	visible_to_approvers = Column(Boolean, default=True)
	requires_response = Column(Boolean, default=False)
	
	# Response Information
	parent_comment_id = Column(String(36), ForeignKey('ppr_requisition_comment.comment_id'), nullable=True)
	has_responses = Column(Boolean, default=False)
	response_count = Column(Integer, default=0)
	
	# Status
	comment_status = Column(String(20), default='Active')  # Active, Deleted, Archived
	
	# Relationships
	requisition = relationship("PPRRequisition", back_populates="comments")
	parent_comment = relationship("PPRRequisitionComment", remote_side="PPRRequisitionComment.comment_id")
	
	def __repr__(self):
		return f"<PPRRequisitionComment {self.comment_type} by {self.author_name}>"
	
	def can_edit(self, user_id: str) -> bool:
		"""Check if user can edit this comment"""
		return (
			self.author_id == user_id and
			self.comment_status == 'Active' and
			not self.is_system_generated
		)
	
	def can_delete(self, user_id: str) -> bool:
		"""Check if user can delete this comment"""
		return (
			self.author_id == user_id and
			self.comment_status == 'Active' and
			not self.is_system_generated and
			not self.has_responses
		)
	
	def soft_delete(self):
		"""Soft delete the comment"""
		self.comment_status = 'Deleted'
		self.comment_text = '[Comment deleted]'