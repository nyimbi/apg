"""
Accounts Payable Models

Database models for the Accounts Payable sub-capability including vendors,
invoices, payments, expense reports, and related AP functionality.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class CFAPVendor(Model, AuditMixin, BaseMixin):
	"""
	Vendor master data for accounts payable.
	
	Stores vendor information including contact details, payment terms,
	tax information, and vendor configuration.
	"""
	__tablename__ = 'cf_ap_vendor'
	
	# Identity
	vendor_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Vendor Information
	vendor_number = Column(String(20), nullable=False, index=True)
	vendor_name = Column(String(200), nullable=False, index=True)
	vendor_type = Column(String(50), default='SUPPLIER')  # SUPPLIER, CONTRACTOR, etc.
	
	# Contact Information
	contact_name = Column(String(100), nullable=True)
	email = Column(String(100), nullable=True)
	phone = Column(String(50), nullable=True)
	fax = Column(String(50), nullable=True)
	website = Column(String(200), nullable=True)
	
	# Address Information
	address_line1 = Column(String(100), nullable=True)
	address_line2 = Column(String(100), nullable=True)
	city = Column(String(50), nullable=True)
	state_province = Column(String(50), nullable=True)
	postal_code = Column(String(20), nullable=True)
	country = Column(String(50), nullable=True)
	
	# Payment Information
	payment_terms_code = Column(String(20), default='NET_30')
	payment_method = Column(String(50), default='CHECK')  # CHECK, ACH, WIRE, CARD
	currency_code = Column(String(3), default='USD')
	
	# Banking Information
	bank_name = Column(String(100), nullable=True)
	bank_account_number = Column(String(50), nullable=True)
	bank_routing_number = Column(String(20), nullable=True)
	bank_swift_code = Column(String(20), nullable=True)
	
	# Tax Information
	tax_id = Column(String(50), nullable=True)  # SSN/EIN/VAT number
	tax_exempt = Column(Boolean, default=False)
	tax_code = Column(String(20), nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_employee = Column(Boolean, default=False)  # For expense reimbursements
	credit_limit = Column(DECIMAL(15, 2), default=0.00)
	require_po = Column(Boolean, default=False)  # Require PO for invoices
	hold_payment = Column(Boolean, default=False)  # Hold all payments
	
	# 1099 Information (US specific)
	is_1099_vendor = Column(Boolean, default=False)
	form_1099_type = Column(String(20), nullable=True)  # MISC, NEC, etc.
	
	# Balance Information
	current_balance = Column(DECIMAL(15, 2), default=0.00)
	ytd_purchases = Column(DECIMAL(15, 2), default=0.00)
	
	# Default GL Account
	default_expense_account_id = Column(String(36), nullable=True)
	
	# Notes and Internal Information
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)  # Not visible to vendor
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'vendor_number', name='uq_vendor_number_tenant'),
	)
	
	# Relationships
	invoices = relationship("CFAPInvoice", back_populates="vendor")
	payments = relationship("CFAPPayment", back_populates="vendor")
	expense_reports = relationship("CFAPExpenseReport", back_populates="vendor")
	purchase_orders = relationship("CFAPPurchaseOrder", back_populates="vendor")
	
	def __repr__(self):
		return f"<CFAPVendor {self.vendor_number} - {self.vendor_name}>"
	
	def get_outstanding_balance(self, as_of_date: Optional[date] = None) -> Decimal:
		"""Calculate outstanding balance as of a specific date"""
		if as_of_date is None:
			as_of_date = date.today()
		
		total_invoices = sum(
			inv.total_amount for inv in self.invoices 
			if inv.status == 'Posted' and inv.invoice_date <= as_of_date
		)
		
		total_payments = sum(
			pay.total_amount for pay in self.payments
			if pay.status == 'Posted' and pay.payment_date <= as_of_date
		)
		
		return total_invoices - total_payments
	
	def is_on_hold(self) -> bool:
		"""Check if vendor is on payment hold"""
		return self.hold_payment or not self.is_active


class CFAPTaxCode(Model, AuditMixin, BaseMixin):
	"""
	Tax codes for AP transactions.
	
	Manages tax rates, calculations, and GL account mappings
	for purchase transactions.
	"""
	__tablename__ = 'cf_ap_tax_code'
	
	# Identity
	tax_code_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Tax Code Information
	code = Column(String(20), nullable=False, index=True)
	name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	
	# Tax Rates
	tax_rate = Column(DECIMAL(5, 2), default=0.00)  # Percentage
	is_compound = Column(Boolean, default=False)  # Compound tax calculation
	
	# GL Integration
	gl_account_id = Column(String(36), nullable=True)  # Tax payable account
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_recoverable = Column(Boolean, default=False)  # Input tax recovery
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'code', name='uq_tax_code_tenant'),
	)
	
	# Relationships
	invoice_lines = relationship("CFAPInvoiceLine", back_populates="tax_code_rel")
	expense_lines = relationship("CFAPExpenseLine", back_populates="tax_code_rel")
	
	def __repr__(self):
		return f"<CFAPTaxCode {self.code} - {self.tax_rate}%>"
	
	def calculate_tax(self, base_amount: Decimal) -> Decimal:
		"""Calculate tax amount for a base amount"""
		return base_amount * (self.tax_rate / 100)


class CFAPInvoice(Model, AuditMixin, BaseMixin):
	"""
	Vendor invoices - header record for invoice processing.
	
	Manages invoice workflow, approval, posting, and payment tracking.
	"""
	__tablename__ = 'cf_ap_invoice'
	
	# Identity
	invoice_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Invoice Information
	invoice_number = Column(String(50), nullable=False, index=True)
	vendor_invoice_number = Column(String(50), nullable=False, index=True)  # Vendor's invoice #
	description = Column(Text, nullable=True)
	
	# Vendor
	vendor_id = Column(String(36), ForeignKey('cf_ap_vendor.vendor_id'), nullable=False, index=True)
	
	# Dates
	invoice_date = Column(Date, nullable=False, index=True)
	due_date = Column(Date, nullable=False, index=True)
	received_date = Column(Date, default=date.today, index=True)
	
	# Purchase Order Reference
	purchase_order_id = Column(String(36), ForeignKey('cf_ap_purchase_order.po_id'), nullable=True, index=True)
	
	# Status and Workflow
	status = Column(String(20), default='Draft', index=True)  # Draft, Pending, Approved, Posted, Paid, Cancelled
	workflow_status = Column(String(50), nullable=True)  # Custom workflow status
	
	# Amounts
	subtotal_amount = Column(DECIMAL(15, 2), default=0.00)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	discount_amount = Column(DECIMAL(15, 2), default=0.00)
	freight_amount = Column(DECIMAL(15, 2), default=0.00)
	misc_amount = Column(DECIMAL(15, 2), default=0.00)
	total_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Payment Information
	payment_terms_code = Column(String(20), nullable=True)
	discount_terms = Column(String(50), nullable=True)  # e.g., "2/10 Net 30"
	currency_code = Column(String(3), default='USD')
	exchange_rate = Column(DECIMAL(10, 6), default=1.000000)
	
	# Approval Workflow
	requires_approval = Column(Boolean, default=True)
	approval_level = Column(Integer, default=0)
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Posting Information
	posted = Column(Boolean, default=False)
	posted_by = Column(String(36), nullable=True)
	posted_date = Column(DateTime, nullable=True)
	gl_batch_id = Column(String(36), nullable=True)  # GL posting batch
	
	# Payment Status
	payment_status = Column(String(20), default='Unpaid')  # Unpaid, Partial, Paid
	paid_amount = Column(DECIMAL(15, 2), default=0.00)
	outstanding_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Processing Information
	processed_by = Column(String(36), nullable=True)
	processed_date = Column(DateTime, nullable=True)
	
	# Hold and Special Handling
	hold_flag = Column(Boolean, default=False)
	hold_reason = Column(String(200), nullable=True)
	recurring_flag = Column(Boolean, default=False)
	
	# Document Management
	document_path = Column(String(500), nullable=True)  # Path to scanned invoice
	document_count = Column(Integer, default=0)
	
	# Notes
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'invoice_number', name='uq_invoice_number_tenant'),
		UniqueConstraint('tenant_id', 'vendor_id', 'vendor_invoice_number', name='uq_vendor_invoice_tenant'),
	)
	
	# Relationships
	vendor = relationship("CFAPVendor", back_populates="invoices")
	purchase_order = relationship("CFAPPurchaseOrder", back_populates="invoices")
	lines = relationship("CFAPInvoiceLine", back_populates="invoice", cascade="all, delete-orphan")
	payment_lines = relationship("CFAPPaymentLine", back_populates="invoice")
	
	def __repr__(self):
		return f"<CFAPInvoice {self.invoice_number} - ${self.total_amount}>"
	
	def calculate_totals(self):
		"""Recalculate invoice totals from lines"""
		self.subtotal_amount = sum(line.line_amount for line in self.lines)
		self.tax_amount = sum(line.tax_amount for line in self.lines)
		self.total_amount = self.subtotal_amount + self.tax_amount + self.freight_amount + self.misc_amount - self.discount_amount
		self.outstanding_amount = self.total_amount - self.paid_amount
	
	def can_approve(self) -> bool:
		"""Check if invoice can be approved"""
		return (
			self.status in ['Draft', 'Pending'] and
			not self.approved and
			self.requires_approval and
			self.total_amount > 0
		)
	
	def can_post(self) -> bool:
		"""Check if invoice can be posted to GL"""
		return (
			self.status == 'Approved' and
			not self.posted and
			(not self.requires_approval or self.approved)
		)
	
	def can_pay(self) -> bool:
		"""Check if invoice can be paid"""
		return (
			self.status == 'Posted' and
			self.payment_status in ['Unpaid', 'Partial'] and
			self.outstanding_amount > 0 and
			not self.hold_flag and
			self.vendor.is_active and
			not self.vendor.hold_payment
		)
	
	def approve_invoice(self, user_id: str):
		"""Approve the invoice"""
		if not self.can_approve():
			raise ValueError("Invoice cannot be approved")
		
		self.approved = True
		self.approved_by = user_id
		self.approved_date = datetime.utcnow()
		self.status = 'Approved'
	
	def post_invoice(self, user_id: str):
		"""Post invoice to GL"""
		if not self.can_post():
			raise ValueError("Invoice cannot be posted")
		
		self.posted = True
		self.posted_by = user_id
		self.posted_date = datetime.utcnow()
		self.status = 'Posted'
		self.outstanding_amount = self.total_amount


class CFAPInvoiceLine(Model, AuditMixin, BaseMixin):
	"""
	Individual invoice line items.
	
	Contains detailed line-level information including GL accounts,
	quantities, amounts, and tax calculations.
	"""
	__tablename__ = 'cf_ap_invoice_line'
	
	# Identity
	line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	invoice_id = Column(String(36), ForeignKey('cf_ap_invoice.invoice_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	description = Column(Text, nullable=True)
	
	# Item Information (if applicable)
	item_code = Column(String(50), nullable=True)
	item_description = Column(String(200), nullable=True)
	
	# Quantities and Amounts
	quantity = Column(DECIMAL(12, 4), default=1.0000)
	unit_price = Column(DECIMAL(15, 4), default=0.0000)
	line_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# GL Account
	gl_account_id = Column(String(36), nullable=False, index=True)
	
	# Tax Information
	tax_code = Column(String(20), nullable=True)
	tax_rate = Column(DECIMAL(5, 2), default=0.00)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	is_tax_inclusive = Column(Boolean, default=False)
	
	# Dimensions
	cost_center = Column(String(20), nullable=True)
	department = Column(String(20), nullable=True)
	project = Column(String(20), nullable=True)
	
	# Purchase Order Reference
	po_line_id = Column(String(36), nullable=True)
	
	# Asset Information (for capital expenses)
	is_asset = Column(Boolean, default=False)
	asset_id = Column(String(36), nullable=True)
	
	# Relationships
	invoice = relationship("CFAPInvoice", back_populates="lines")
	tax_code_rel = relationship("CFAPTaxCode", foreign_keys=[tax_code], primaryjoin="CFAPInvoiceLine.tax_code == CFAPTaxCode.code")
	
	def __repr__(self):
		return f"<CFAPInvoiceLine {self.line_number}: ${self.line_amount}>"
	
	def calculate_tax(self):
		"""Calculate tax amount for the line"""
		if self.tax_code_rel:
			if self.is_tax_inclusive:
				# Extract tax from inclusive amount
				self.tax_amount = self.line_amount * (self.tax_rate / (100 + self.tax_rate))
			else:
				# Add tax to exclusive amount
				self.tax_amount = self.line_amount * (self.tax_rate / 100)
		else:
			self.tax_amount = 0.00


class CFAPPayment(Model, AuditMixin, BaseMixin):
	"""
	Payment records for vendor payments.
	
	Manages payment processing, check printing, electronic payments,
	and payment allocation to invoices.
	"""
	__tablename__ = 'cf_ap_payment'
	
	# Identity
	payment_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Payment Information
	payment_number = Column(String(50), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Vendor
	vendor_id = Column(String(36), ForeignKey('cf_ap_vendor.vendor_id'), nullable=False, index=True)
	
	# Payment Details
	payment_date = Column(Date, nullable=False, index=True)
	payment_method = Column(String(50), nullable=False)  # CHECK, ACH, WIRE, CARD, CASH
	
	# Check Information
	check_number = Column(String(20), nullable=True, index=True)
	bank_account_id = Column(String(36), nullable=True)
	
	# Status
	status = Column(String(20), default='Draft', index=True)  # Draft, Pending, Approved, Posted, Cleared, Voided
	
	# Amounts
	payment_amount = Column(DECIMAL(15, 2), default=0.00)
	discount_taken = Column(DECIMAL(15, 2), default=0.00)
	total_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Currency
	currency_code = Column(String(3), default='USD')
	exchange_rate = Column(DECIMAL(10, 6), default=1.000000)
	
	# Approval
	requires_approval = Column(Boolean, default=True)
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Posting Information
	posted = Column(Boolean, default=False)
	posted_by = Column(String(36), nullable=True)
	posted_date = Column(DateTime, nullable=True)
	gl_batch_id = Column(String(36), nullable=True)
	
	# Bank Reconciliation
	cleared = Column(Boolean, default=False)
	cleared_date = Column(Date, nullable=True)
	bank_statement_date = Column(Date, nullable=True)
	
	# Void Information
	voided = Column(Boolean, default=False)
	void_date = Column(DateTime, nullable=True)
	void_reason = Column(String(200), nullable=True)
	
	# Processing Information
	processed_by = Column(String(36), nullable=True)
	processed_date = Column(DateTime, nullable=True)
	
	# Document Management
	document_path = Column(String(500), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'payment_number', name='uq_payment_number_tenant'),
	)
	
	# Relationships
	vendor = relationship("CFAPVendor", back_populates="payments")
	payment_lines = relationship("CFAPPaymentLine", back_populates="payment", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<CFAPPayment {self.payment_number} - ${self.total_amount}>"
	
	def calculate_totals(self):
		"""Recalculate payment totals from lines"""
		self.payment_amount = sum(line.payment_amount for line in self.payment_lines)
		self.discount_taken = sum(line.discount_taken for line in self.payment_lines)
		self.total_amount = self.payment_amount + self.discount_taken
	
	def can_approve(self) -> bool:
		"""Check if payment can be approved"""
		return (
			self.status in ['Draft', 'Pending'] and
			not self.approved and
			self.requires_approval and
			self.total_amount > 0
		)
	
	def can_post(self) -> bool:
		"""Check if payment can be posted"""
		return (
			self.status == 'Approved' and
			not self.posted and
			(not self.requires_approval or self.approved)
		)
	
	def can_void(self) -> bool:
		"""Check if payment can be voided"""
		return (
			self.status == 'Posted' and
			not self.voided and
			not self.cleared
		)
	
	def approve_payment(self, user_id: str):
		"""Approve the payment"""
		if not self.can_approve():
			raise ValueError("Payment cannot be approved")
		
		self.approved = True
		self.approved_by = user_id
		self.approved_date = datetime.utcnow()
		self.status = 'Approved'
	
	def post_payment(self, user_id: str):
		"""Post payment to GL"""
		if not self.can_post():
			raise ValueError("Payment cannot be posted")
		
		self.posted = True
		self.posted_by = user_id
		self.posted_date = datetime.utcnow()
		self.status = 'Posted'
	
	def void_payment(self, user_id: str, reason: str):
		"""Void the payment"""
		if not self.can_void():
			raise ValueError("Payment cannot be voided")
		
		self.voided = True
		self.void_date = datetime.utcnow()
		self.void_reason = reason
		self.status = 'Voided'


class CFAPPaymentLine(Model, AuditMixin, BaseMixin):
	"""
	Payment allocation lines.
	
	Tracks how payments are allocated to specific invoices,
	including discount calculations and partial payments.
	"""
	__tablename__ = 'cf_ap_payment_line'
	
	# Identity
	payment_line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	payment_id = Column(String(36), ForeignKey('cf_ap_payment.payment_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	
	# Invoice Reference
	invoice_id = Column(String(36), ForeignKey('cf_ap_invoice.invoice_id'), nullable=False, index=True)
	
	# Payment Allocation
	invoice_amount = Column(DECIMAL(15, 2), default=0.00)  # Original invoice amount
	payment_amount = Column(DECIMAL(15, 2), default=0.00)  # Amount being paid
	discount_taken = Column(DECIMAL(15, 2), default=0.00)  # Early payment discount
	remaining_amount = Column(DECIMAL(15, 2), default=0.00)  # Remaining after payment
	
	# Discount Information
	discount_available = Column(DECIMAL(15, 2), default=0.00)
	discount_date = Column(Date, nullable=True)  # Last date for discount
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	payment = relationship("CFAPPayment", back_populates="payment_lines")
	invoice = relationship("CFAPInvoice", back_populates="payment_lines")
	
	def __repr__(self):
		return f"<CFAPPaymentLine {self.line_number}: ${self.payment_amount}>"
	
	def calculate_discount(self) -> Decimal:
		"""Calculate available early payment discount"""
		if not self.invoice.vendor.payment_terms_code:
			return Decimal('0.00')
		
		# Simplified discount calculation
		# In practice, this would reference payment terms configuration
		if self.payment.payment_date <= self.discount_date:
			# Example: 2/10 Net 30 = 2% discount if paid within 10 days
			return self.invoice_amount * Decimal('0.02')
		
		return Decimal('0.00')


class CFAPExpenseReport(Model, AuditMixin, BaseMixin):
	"""
	Employee expense reports.
	
	Manages employee expense submissions, approval workflow,
	and reimbursement processing.
	"""
	__tablename__ = 'cf_ap_expense_report'
	
	# Identity
	expense_report_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Expense Report Information
	report_number = Column(String(50), nullable=False, index=True)
	report_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Employee Information
	employee_id = Column(String(36), nullable=False, index=True)
	employee_name = Column(String(100), nullable=False)
	department = Column(String(50), nullable=True)
	
	# Vendor Reference (for reimbursement)
	vendor_id = Column(String(36), ForeignKey('cf_ap_vendor.vendor_id'), nullable=True, index=True)
	
	# Dates
	report_date = Column(Date, nullable=False, index=True)
	period_start = Column(Date, nullable=False)
	period_end = Column(Date, nullable=False)
	submitted_date = Column(Date, nullable=True)
	
	# Status and Workflow
	status = Column(String(20), default='Draft', index=True)  # Draft, Submitted, Approved, Paid, Rejected
	
	# Amounts
	total_amount = Column(DECIMAL(15, 2), default=0.00)
	reimbursable_amount = Column(DECIMAL(15, 2), default=0.00)
	non_reimbursable_amount = Column(DECIMAL(15, 2), default=0.00)
	paid_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Currency
	currency_code = Column(String(3), default='USD')
	
	# Approval Workflow
	requires_approval = Column(Boolean, default=True)
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Rejection Information
	rejected = Column(Boolean, default=False)
	rejected_by = Column(String(36), nullable=True)
	rejected_date = Column(DateTime, nullable=True)
	rejection_reason = Column(Text, nullable=True)
	
	# Payment Information
	paid = Column(Boolean, default=False)
	payment_id = Column(String(36), nullable=True)
	paid_date = Column(Date, nullable=True)
	
	# Document Management
	receipt_count = Column(Integer, default=0)
	document_path = Column(String(500), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	manager_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'report_number', name='uq_expense_report_tenant'),
	)
	
	# Relationships
	vendor = relationship("CFAPVendor", back_populates="expense_reports")
	lines = relationship("CFAPExpenseLine", back_populates="expense_report", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<CFAPExpenseReport {self.report_number} - ${self.total_amount}>"
	
	def calculate_totals(self):
		"""Recalculate expense report totals from lines"""
		self.total_amount = sum(line.amount for line in self.lines)
		self.reimbursable_amount = sum(line.amount for line in self.lines if line.is_reimbursable)
		self.non_reimbursable_amount = self.total_amount - self.reimbursable_amount
	
	def can_submit(self) -> bool:
		"""Check if expense report can be submitted"""
		return (
			self.status == 'Draft' and
			self.total_amount > 0 and
			len(self.lines) > 0
		)
	
	def can_approve(self) -> bool:
		"""Check if expense report can be approved"""
		return (
			self.status == 'Submitted' and
			not self.approved and
			not self.rejected
		)
	
	def submit_report(self):
		"""Submit expense report for approval"""
		if not self.can_submit():
			raise ValueError("Expense report cannot be submitted")
		
		self.status = 'Submitted'
		self.submitted_date = date.today()
	
	def approve_report(self, user_id: str):
		"""Approve the expense report"""
		if not self.can_approve():
			raise ValueError("Expense report cannot be approved")
		
		self.approved = True
		self.approved_by = user_id
		self.approved_date = datetime.utcnow()
		self.status = 'Approved'
	
	def reject_report(self, user_id: str, reason: str):
		"""Reject the expense report"""
		if not self.can_approve():
			raise ValueError("Expense report cannot be rejected")
		
		self.rejected = True
		self.rejected_by = user_id
		self.rejected_date = datetime.utcnow()
		self.rejection_reason = reason
		self.status = 'Rejected'


class CFAPExpenseLine(Model, AuditMixin, BaseMixin):
	"""
	Individual expense line items.
	
	Detailed expense information including categories, amounts,
	tax calculations, and receipt tracking.
	"""
	__tablename__ = 'cf_ap_expense_line'
	
	# Identity
	expense_line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	expense_report_id = Column(String(36), ForeignKey('cf_ap_expense_report.expense_report_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	description = Column(Text, nullable=False)
	
	# Expense Details
	expense_date = Column(Date, nullable=False, index=True)
	expense_category = Column(String(50), nullable=False)  # TRAVEL, MEALS, etc.
	
	# Merchant Information
	merchant_name = Column(String(100), nullable=True)
	location = Column(String(100), nullable=True)
	
	# Amount Information
	amount = Column(DECIMAL(15, 2), default=0.00)
	currency_code = Column(String(3), default='USD')
	exchange_rate = Column(DECIMAL(10, 6), default=1.000000)
	home_currency_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Tax Information
	tax_code = Column(String(20), nullable=True)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	is_tax_inclusive = Column(Boolean, default=True)
	
	# GL Account
	gl_account_id = Column(String(36), nullable=False, index=True)
	
	# Reimbursement
	is_reimbursable = Column(Boolean, default=True)
	reimbursement_rate = Column(DECIMAL(5, 2), default=100.00)  # Percentage
	
	# Mileage (for vehicle expenses)
	is_mileage = Column(Boolean, default=False)
	mileage_distance = Column(DECIMAL(8, 2), default=0.00)
	mileage_rate = Column(DECIMAL(6, 4), default=0.0000)
	
	# Receipt Information
	has_receipt = Column(Boolean, default=False)
	receipt_path = Column(String(500), nullable=True)
	receipt_required = Column(Boolean, default=True)
	
	# Allocation
	is_personal = Column(Boolean, default=False)
	business_percentage = Column(DECIMAL(5, 2), default=100.00)
	
	# Dimensions
	cost_center = Column(String(20), nullable=True)
	project = Column(String(20), nullable=True)
	client_id = Column(String(36), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	expense_report = relationship("CFAPExpenseReport", back_populates="lines")
	tax_code_rel = relationship("CFAPTaxCode", foreign_keys=[tax_code], primaryjoin="CFAPExpenseLine.tax_code == CFAPTaxCode.code")
	
	def __repr__(self):
		return f"<CFAPExpenseLine {self.line_number}: {self.expense_category} - ${self.amount}>"
	
	def calculate_reimbursable_amount(self) -> Decimal:
		"""Calculate reimbursable amount"""
		base_amount = self.amount
		
		# Apply business percentage
		business_amount = base_amount * (self.business_percentage / 100)
		
		# Apply reimbursement rate
		reimbursable = business_amount * (self.reimbursement_rate / 100)
		
		return reimbursable if self.is_reimbursable else Decimal('0.00')


class CFAPPurchaseOrder(Model, AuditMixin, BaseMixin):
	"""
	Purchase orders for three-way matching.
	
	Simplified PO model for AP integration and three-way matching
	(PO, Receipt, Invoice) functionality.
	"""
	__tablename__ = 'cf_ap_purchase_order'
	
	# Identity
	po_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# PO Information
	po_number = Column(String(50), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Vendor
	vendor_id = Column(String(36), ForeignKey('cf_ap_vendor.vendor_id'), nullable=False, index=True)
	
	# Dates
	po_date = Column(Date, nullable=False, index=True)
	required_date = Column(Date, nullable=True)
	
	# Status
	status = Column(String(20), default='Open', index=True)  # Open, Received, Closed, Cancelled
	
	# Amounts
	po_amount = Column(DECIMAL(15, 2), default=0.00)
	received_amount = Column(DECIMAL(15, 2), default=0.00)
	invoiced_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Currency
	currency_code = Column(String(3), default='USD')
	
	# Approval
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'po_number', name='uq_po_number_tenant'),
	)
	
	# Relationships
	vendor = relationship("CFAPVendor", back_populates="purchase_orders")
	invoices = relationship("CFAPInvoice", back_populates="purchase_order")
	
	def __repr__(self):
		return f"<CFAPPurchaseOrder {self.po_number} - ${self.po_amount}>"
	
	def can_receive(self) -> bool:
		"""Check if PO can receive goods"""
		return self.status == 'Open' and self.approved
	
	def is_fully_received(self) -> bool:
		"""Check if PO is fully received"""
		return abs(self.received_amount - self.po_amount) < 0.01
	
	def is_fully_invoiced(self) -> bool:
		"""Check if PO is fully invoiced"""
		return abs(self.invoiced_amount - self.po_amount) < 0.01


class CFAPAging(Model, AuditMixin, BaseMixin):
	"""
	AP aging snapshots for performance and reporting.
	
	Stores aging analysis data to avoid recalculating
	aging buckets for each report.
	"""
	__tablename__ = 'cf_ap_aging'
	
	# Identity
	aging_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Aging Information
	as_of_date = Column(Date, nullable=False, index=True)
	vendor_id = Column(String(36), ForeignKey('cf_ap_vendor.vendor_id'), nullable=False, index=True)
	
	# Aging Buckets (configurable)
	current_amount = Column(DECIMAL(15, 2), default=0.00)  # 0-30 days
	days_31_60 = Column(DECIMAL(15, 2), default=0.00)
	days_61_90 = Column(DECIMAL(15, 2), default=0.00)
	days_91_120 = Column(DECIMAL(15, 2), default=0.00)
	over_120_days = Column(DECIMAL(15, 2), default=0.00)
	
	# Totals
	total_outstanding = Column(DECIMAL(15, 2), default=0.00)
	
	# Generation Info
	generated_date = Column(DateTime, default=datetime.utcnow)
	generated_by = Column(String(36), nullable=False)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'as_of_date', 'vendor_id', name='uq_aging_date_vendor'),
	)
	
	# Relationships
	vendor = relationship("CFAPVendor")
	
	def __repr__(self):
		return f"<CFAPAging {self.vendor.vendor_name} as of {self.as_of_date}>"
	
	def get_aging_buckets(self) -> Dict[str, Decimal]:
		"""Get aging buckets as dictionary"""
		return {
			'current': self.current_amount,
			'31_60': self.days_31_60,
			'61_90': self.days_61_90,
			'91_120': self.days_91_120,
			'over_120': self.over_120_days,
			'total': self.total_outstanding
		}