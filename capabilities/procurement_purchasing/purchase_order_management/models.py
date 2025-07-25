"""
Purchase Order Management Models

Database models for purchase orders, receipts, three-way matching, and change orders.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class PPOPurchaseOrder(Model, AuditMixin, BaseMixin):
	"""Purchase orders for procurement"""
	__tablename__ = 'ppo_purchase_order'
	
	# Identity
	po_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# PO Information
	po_number = Column(String(50), nullable=False, index=True)
	title = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Vendor Information
	vendor_id = Column(String(36), nullable=False, index=True)
	vendor_name = Column(String(100), nullable=False)
	vendor_contact = Column(String(100), nullable=True)
	
	# Buyer Information
	buyer_id = Column(String(36), nullable=False, index=True)
	buyer_name = Column(String(100), nullable=False)
	department = Column(String(50), nullable=True)
	
	# Dates
	po_date = Column(Date, default=date.today, nullable=False, index=True)
	required_date = Column(Date, nullable=False, index=True)
	promised_date = Column(Date, nullable=True)
	
	# Status
	status = Column(String(20), default='Draft', index=True)  # Draft, Approved, Open, Closed, Cancelled
	
	# Financial Information
	currency_code = Column(String(3), default='USD')
	subtotal_amount = Column(DECIMAL(15, 2), default=0.00)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	freight_amount = Column(DECIMAL(15, 2), default=0.00)
	total_amount = Column(DECIMAL(15, 2), default=0.00, index=True)
	
	# Delivery Information
	ship_to_location = Column(String(200), nullable=True)
	ship_to_address = Column(Text, nullable=True)
	delivery_terms = Column(String(50), nullable=True)  # FOB, DDP, etc.
	
	# Terms and Conditions
	payment_terms = Column(String(50), nullable=True)
	freight_terms = Column(String(50), nullable=True)
	
	# Approval Information
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Receipt Tracking
	received_amount = Column(DECIMAL(15, 2), default=0.00)
	invoiced_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Reference Information
	requisition_id = Column(String(36), nullable=True, index=True)
	contract_id = Column(String(36), nullable=True)
	
	# Special Instructions
	special_instructions = Column(Text, nullable=True)
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'po_number', name='uq_po_number_tenant'),
	)
	
	# Relationships
	lines = relationship("PPOPurchaseOrderLine", back_populates="purchase_order", cascade="all, delete-orphan")
	receipts = relationship("PPOReceipt", back_populates="purchase_order")
	change_orders = relationship("PPOChangeOrder", back_populates="purchase_order")
	
	def __repr__(self):
		return f"<PPOPurchaseOrder {self.po_number} - {self.status} - ${self.total_amount}>"
	
	def calculate_totals(self):
		"""Recalculate PO totals from lines"""
		self.subtotal_amount = sum(line.line_amount for line in self.lines)
		self.tax_amount = sum(line.tax_amount for line in self.lines)
		self.total_amount = self.subtotal_amount + self.tax_amount + self.freight_amount


class PPOPurchaseOrderLine(Model, AuditMixin, BaseMixin):
	"""Purchase order line items"""
	__tablename__ = 'ppo_purchase_order_line'
	
	# Identity
	line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	po_id = Column(String(36), ForeignKey('ppo_purchase_order.po_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	description = Column(Text, nullable=False)
	
	# Item Information
	item_code = Column(String(50), nullable=True)
	item_description = Column(String(200), nullable=True)
	
	# Quantities and Pricing
	quantity_ordered = Column(DECIMAL(12, 4), default=1.0000)
	quantity_received = Column(DECIMAL(12, 4), default=0.0000)
	quantity_invoiced = Column(DECIMAL(12, 4), default=0.0000)
	unit_of_measure = Column(String(20), default='EA')
	unit_price = Column(DECIMAL(15, 4), default=0.0000)
	line_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Tax Information
	tax_code = Column(String(20), nullable=True)
	tax_rate = Column(DECIMAL(5, 2), default=0.00)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# GL Account Coding
	gl_account_id = Column(String(36), nullable=False, index=True)
	cost_center = Column(String(20), nullable=True)
	project_id = Column(String(36), nullable=True)
	
	# Delivery Information
	required_date = Column(Date, nullable=True)
	promised_date = Column(Date, nullable=True)
	
	# Status
	line_status = Column(String(20), default='Open')  # Open, Received, Closed, Cancelled
	
	# Reference Information
	requisition_line_id = Column(String(36), nullable=True)
	
	# Relationships
	purchase_order = relationship("PPOPurchaseOrder", back_populates="lines")
	receipt_lines = relationship("PPOReceiptLine", back_populates="po_line")
	
	def __repr__(self):
		return f"<PPOPurchaseOrderLine {self.line_number}: {self.description} - ${self.line_amount}>"


class PPOReceipt(Model, AuditMixin, BaseMixin):
	"""Goods receipt records"""
	__tablename__ = 'ppo_receipt'
	
	# Identity
	receipt_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Receipt Information
	receipt_number = Column(String(50), nullable=False, index=True)
	po_id = Column(String(36), ForeignKey('ppo_purchase_order.po_id'), nullable=False, index=True)
	
	# Receipt Details
	receipt_date = Column(Date, default=date.today, nullable=False, index=True)
	received_by = Column(String(36), nullable=False)
	received_by_name = Column(String(100), nullable=False)
	
	# Status
	status = Column(String(20), default='Draft', index=True)  # Draft, Posted, Cancelled
	
	# Reference Information
	packing_slip_number = Column(String(50), nullable=True)
	carrier = Column(String(100), nullable=True)
	tracking_number = Column(String(50), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'receipt_number', name='uq_receipt_number_tenant'),
	)
	
	# Relationships
	purchase_order = relationship("PPOPurchaseOrder", back_populates="receipts")
	lines = relationship("PPOReceiptLine", back_populates="receipt", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<PPOReceipt {self.receipt_number} - {self.status}>"


class PPOReceiptLine(Model, AuditMixin, BaseMixin):
	"""Receipt line items"""
	__tablename__ = 'ppo_receipt_line'
	
	# Identity
	receipt_line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	receipt_id = Column(String(36), ForeignKey('ppo_receipt.receipt_id'), nullable=False, index=True)
	po_line_id = Column(String(36), ForeignKey('ppo_purchase_order_line.line_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Receipt Information
	line_number = Column(Integer, nullable=False)
	quantity_received = Column(DECIMAL(12, 4), default=0.0000)
	unit_of_measure = Column(String(20), default='EA')
	
	# Quality Information
	quality_status = Column(String(20), default='Accepted')  # Accepted, Rejected, On Hold
	quality_notes = Column(Text, nullable=True)
	
	# Location Information
	location_code = Column(String(20), nullable=True)
	bin_location = Column(String(20), nullable=True)
	
	# Relationships
	receipt = relationship("PPOReceipt", back_populates="lines")
	po_line = relationship("PPOPurchaseOrderLine", back_populates="receipt_lines")
	
	def __repr__(self):
		return f"<PPOReceiptLine {self.line_number}: {self.quantity_received}>"


class PPOThreeWayMatch(Model, AuditMixin, BaseMixin):
	"""Three-way matching records (PO, Receipt, Invoice)"""
	__tablename__ = 'ppo_three_way_match'
	
	# Identity
	match_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Document References
	po_id = Column(String(36), ForeignKey('ppo_purchase_order.po_id'), nullable=False, index=True)
	receipt_id = Column(String(36), ForeignKey('ppo_receipt.receipt_id'), nullable=True, index=True)
	invoice_id = Column(String(36), nullable=True, index=True)  # From AP module
	
	# Match Information
	match_date = Column(DateTime, default=datetime.utcnow, nullable=False)
	matched_by = Column(String(36), nullable=False)
	match_status = Column(String(20), default='Matched', index=True)  # Matched, Exception, Resolved
	
	# Exception Information
	has_exceptions = Column(Boolean, default=False)
	exception_type = Column(String(50), nullable=True)  # Price, Quantity, Tax, etc.
	exception_description = Column(Text, nullable=True)
	
	# Resolution Information
	resolved = Column(Boolean, default=False)
	resolved_by = Column(String(36), nullable=True)
	resolved_date = Column(DateTime, nullable=True)
	resolution_notes = Column(Text, nullable=True)
	
	# Relationships
	purchase_order = relationship("PPOPurchaseOrder")
	receipt = relationship("PPOReceipt")
	
	def __repr__(self):
		return f"<PPOThreeWayMatch {self.match_status} - PO:{self.po_id}>"


class PPOChangeOrder(Model, AuditMixin, BaseMixin):
	"""Purchase order change orders"""
	__tablename__ = 'ppo_change_order'
	
	# Identity
	change_order_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Change Order Information
	change_order_number = Column(String(50), nullable=False, index=True)
	po_id = Column(String(36), ForeignKey('ppo_purchase_order.po_id'), nullable=False, index=True)
	
	# Change Details
	change_type = Column(String(50), nullable=False)  # Price, Quantity, Date, Terms, etc.
	change_reason = Column(Text, nullable=False)
	change_description = Column(Text, nullable=False)
	
	# Financial Impact
	original_amount = Column(DECIMAL(15, 2), default=0.00)
	new_amount = Column(DECIMAL(15, 2), default=0.00)
	amount_difference = Column(DECIMAL(15, 2), default=0.00)
	
	# Approval Information
	requested_by = Column(String(36), nullable=False)
	requested_date = Column(DateTime, default=datetime.utcnow)
	
	# Status
	status = Column(String(20), default='Pending', index=True)  # Pending, Approved, Rejected, Implemented
	
	# Approval
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Implementation
	implemented = Column(Boolean, default=False)
	implemented_by = Column(String(36), nullable=True)
	implemented_date = Column(DateTime, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'change_order_number', name='uq_change_order_number_tenant'),
	)
	
	# Relationships
	purchase_order = relationship("PPOPurchaseOrder", back_populates="change_orders")
	
	def __repr__(self):
		return f"<PPOChangeOrder {self.change_order_number} - {self.status}>"