"""
Accounts Receivable Models

Database models for the Accounts Receivable sub-capability including customers,
invoices, payments, credit memos, collections, and related AR functionality.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class CFARCustomer(Model, AuditMixin, BaseMixin):
	"""
	Customer master data for accounts receivable.
	
	Stores customer information including contact details, billing terms,
	tax information, and customer configuration.
	"""
	__tablename__ = 'cf_ar_customer'
	
	# Identity
	customer_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Customer Information
	customer_number = Column(String(20), nullable=False, index=True)
	customer_name = Column(String(200), nullable=False, index=True)
	customer_type = Column(String(50), default='RETAIL')  # RETAIL, WHOLESALE, CORPORATE, etc.
	
	# Contact Information
	contact_name = Column(String(100), nullable=True)
	email = Column(String(100), nullable=True)
	phone = Column(String(50), nullable=True)
	fax = Column(String(50), nullable=True)
	website = Column(String(200), nullable=True)
	
	# Billing Address
	billing_address_line1 = Column(String(100), nullable=True)
	billing_address_line2 = Column(String(100), nullable=True)
	billing_city = Column(String(50), nullable=True)
	billing_state_province = Column(String(50), nullable=True)
	billing_postal_code = Column(String(20), nullable=True)
	billing_country = Column(String(50), nullable=True)
	
	# Shipping Address
	shipping_address_line1 = Column(String(100), nullable=True)
	shipping_address_line2 = Column(String(100), nullable=True)
	shipping_city = Column(String(50), nullable=True)
	shipping_state_province = Column(String(50), nullable=True)
	shipping_postal_code = Column(String(20), nullable=True)
	shipping_country = Column(String(50), nullable=True)
	
	# Payment Information
	payment_terms_code = Column(String(20), default='NET_30')
	payment_method = Column(String(50), default='CHECK')  # CHECK, ACH, CREDIT_CARD, WIRE
	currency_code = Column(String(3), default='USD')
	
	# Credit Information
	credit_limit = Column(DECIMAL(15, 2), default=0.00)
	credit_hold = Column(Boolean, default=False)
	credit_rating = Column(String(10), nullable=True)  # AAA, AA, A, BBB, etc.
	
	# Tax Information
	tax_id = Column(String(50), nullable=True)  # Customer tax ID
	tax_exempt = Column(Boolean, default=False)
	tax_exempt_number = Column(String(50), nullable=True)
	default_tax_code = Column(String(20), nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	allow_backorders = Column(Boolean, default=True)
	require_po = Column(Boolean, default=False)  # Require customer PO
	print_statements = Column(Boolean, default=True)
	send_dunning_letters = Column(Boolean, default=True)
	
	# Sales Information
	sales_rep_id = Column(String(36), nullable=True)
	territory_id = Column(String(36), nullable=True)
	price_level = Column(String(20), default='STANDARD')
	
	# Balance Information
	current_balance = Column(DECIMAL(15, 2), default=0.00)
	ytd_sales = Column(DECIMAL(15, 2), default=0.00)
	last_payment_date = Column(Date, nullable=True)
	last_payment_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Default GL Accounts
	default_ar_account_id = Column(String(36), nullable=True)
	default_revenue_account_id = Column(String(36), nullable=True)
	
	# Notes and Internal Information
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)  # Not visible to customer
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'customer_number', name='uq_customer_number_tenant'),
	)
	
	# Relationships
	invoices = relationship("CFARInvoice", back_populates="customer")
	payments = relationship("CFARPayment", back_populates="customer")
	credit_memos = relationship("CFARCreditMemo", back_populates="customer")
	statements = relationship("CFARStatement", back_populates="customer")
	collections = relationship("CFARCollection", back_populates="customer")
	recurring_billings = relationship("CFARRecurringBilling", back_populates="customer")
	
	def __repr__(self):
		return f"<CFARCustomer {self.customer_number} - {self.customer_name}>"
	
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
		
		total_credits = sum(
			cm.total_amount for cm in self.credit_memos
			if cm.status == 'Posted' and cm.credit_date <= as_of_date
		)
		
		return total_invoices - total_payments - total_credits
	
	def is_over_credit_limit(self) -> bool:
		"""Check if customer is over credit limit"""
		if self.credit_limit <= 0:
			return False
		return self.current_balance > self.credit_limit
	
	def can_ship(self) -> bool:
		"""Check if customer can receive shipments"""
		return (
			self.is_active and
			not self.credit_hold and
			not self.is_over_credit_limit()
		)


class CFARTaxCode(Model, AuditMixin, BaseMixin):
	"""
	Tax codes for AR transactions.
	
	Manages sales tax rates, calculations, and GL account mappings
	for sales transactions.
	"""
	__tablename__ = 'cf_ar_tax_code'
	
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
	effective_date = Column(Date, nullable=True)
	expiration_date = Column(Date, nullable=True)
	
	# Geographic Information
	jurisdiction = Column(String(100), nullable=True)  # State, Province, etc.
	authority = Column(String(100), nullable=True)  # Tax authority name
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'code', name='uq_ar_tax_code_tenant'),
	)
	
	# Relationships
	invoice_lines = relationship("CFARInvoiceLine", back_populates="tax_code_rel")
	credit_memo_lines = relationship("CFARCreditMemoLine", back_populates="tax_code_rel")
	
	def __repr__(self):
		return f"<CFARTaxCode {self.code} - {self.tax_rate}%>"
	
	def calculate_tax(self, base_amount: Decimal) -> Decimal:
		"""Calculate tax amount for a base amount"""
		return base_amount * (self.tax_rate / 100)
	
	def is_valid_for_date(self, check_date: date) -> bool:
		"""Check if tax code is valid for a specific date"""
		if not self.is_active:
			return False
		
		if self.effective_date and check_date < self.effective_date:
			return False
		
		if self.expiration_date and check_date > self.expiration_date:
			return False
		
		return True


class CFARInvoice(Model, AuditMixin, BaseMixin):
	"""
	Customer invoices - header record for sales invoice processing.
	
	Manages invoice workflow, posting, and payment tracking.
	"""
	__tablename__ = 'cf_ar_invoice'
	
	# Identity
	invoice_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Invoice Information
	invoice_number = Column(String(50), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Customer
	customer_id = Column(String(36), ForeignKey('cf_ar_customer.customer_id'), nullable=False, index=True)
	
	# Dates
	invoice_date = Column(Date, nullable=False, index=True)
	due_date = Column(Date, nullable=False, index=True)
	shipped_date = Column(Date, nullable=True)
	
	# Sales Order Reference
	sales_order_id = Column(String(36), nullable=True, index=True)
	customer_po_number = Column(String(50), nullable=True)
	
	# Status and Workflow
	status = Column(String(20), default='Draft', index=True)  # Draft, Pending, Posted, Paid, Cancelled
	
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
	
	# Posting Information
	posted = Column(Boolean, default=False)
	posted_by = Column(String(36), nullable=True)
	posted_date = Column(DateTime, nullable=True)
	gl_batch_id = Column(String(36), nullable=True)  # GL posting batch
	
	# Payment Status
	payment_status = Column(String(20), default='Unpaid')  # Unpaid, Partial, Paid
	paid_amount = Column(DECIMAL(15, 2), default=0.00)
	outstanding_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Collection Information
	collection_status = Column(String(20), default='Current')  # Current, Past Due, Collections
	dunning_level = Column(Integer, default=0)
	last_dunning_date = Column(Date, nullable=True)
	
	# Sales Information
	sales_rep_id = Column(String(36), nullable=True)
	territory_id = Column(String(36), nullable=True)
	commission_rate = Column(DECIMAL(5, 2), default=0.00)
	commission_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Shipping Information
	ship_to_name = Column(String(100), nullable=True)
	ship_to_address_line1 = Column(String(100), nullable=True)
	ship_to_address_line2 = Column(String(100), nullable=True)
	ship_to_city = Column(String(50), nullable=True)
	ship_to_state_province = Column(String(50), nullable=True)
	ship_to_postal_code = Column(String(20), nullable=True)
	ship_to_country = Column(String(50), nullable=True)
	
	# Document Management
	document_path = Column(String(500), nullable=True)
	print_count = Column(Integer, default=0)
	email_count = Column(Integer, default=0)
	
	# Special Handling
	hold_flag = Column(Boolean, default=False)
	hold_reason = Column(String(200), nullable=True)
	recurring_flag = Column(Boolean, default=False)
	
	# Notes
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'invoice_number', name='uq_ar_invoice_number_tenant'),
	)
	
	# Relationships
	customer = relationship("CFARCustomer", back_populates="invoices")
	lines = relationship("CFARInvoiceLine", back_populates="invoice", cascade="all, delete-orphan")
	payment_lines = relationship("CFARPaymentLine", back_populates="invoice")
	
	def __repr__(self):
		return f"<CFARInvoice {self.invoice_number} - ${self.total_amount}>"
	
	def calculate_totals(self):
		"""Recalculate invoice totals from lines"""
		self.subtotal_amount = sum(line.line_amount for line in self.lines)
		self.tax_amount = sum(line.tax_amount for line in self.lines)
		self.total_amount = self.subtotal_amount + self.tax_amount + self.freight_amount + self.misc_amount - self.discount_amount
		self.outstanding_amount = self.total_amount - self.paid_amount
	
	def can_post(self) -> bool:
		"""Check if invoice can be posted to GL"""
		return (
			self.status == 'Draft' and
			not self.posted and
			self.total_amount > 0 and
			not self.hold_flag
		)
	
	def can_pay(self) -> bool:
		"""Check if invoice can receive payments"""
		return (
			self.status == 'Posted' and
			self.payment_status in ['Unpaid', 'Partial'] and
			self.outstanding_amount > 0
		)
	
	def is_past_due(self, as_of_date: Optional[date] = None) -> bool:
		"""Check if invoice is past due"""
		if as_of_date is None:
			as_of_date = date.today()
		return self.due_date < as_of_date and self.outstanding_amount > 0
	
	def days_past_due(self, as_of_date: Optional[date] = None) -> int:
		"""Calculate days past due"""
		if as_of_date is None:
			as_of_date = date.today()
		if not self.is_past_due(as_of_date):
			return 0
		return (as_of_date - self.due_date).days
	
	def post_invoice(self, user_id: str):
		"""Post invoice to GL"""
		if not self.can_post():
			raise ValueError("Invoice cannot be posted")
		
		self.posted = True
		self.posted_by = user_id
		self.posted_date = datetime.utcnow()
		self.status = 'Posted'
		self.outstanding_amount = self.total_amount


class CFARInvoiceLine(Model, AuditMixin, BaseMixin):
	"""
	Individual invoice line items.
	
	Contains detailed line-level information including items,
	quantities, amounts, and tax calculations.
	"""
	__tablename__ = 'cf_ar_invoice_line'
	
	# Identity
	line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	invoice_id = Column(String(36), ForeignKey('cf_ar_invoice.invoice_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	description = Column(Text, nullable=True)
	
	# Item Information
	item_code = Column(String(50), nullable=True)
	item_description = Column(String(200), nullable=True)
	item_type = Column(String(20), default='PRODUCT')  # PRODUCT, SERVICE, DISCOUNT
	
	# Quantities and Amounts
	quantity = Column(DECIMAL(12, 4), default=1.0000)
	unit_price = Column(DECIMAL(15, 4), default=0.0000)
	line_amount = Column(DECIMAL(15, 2), default=0.00)
	unit_cost = Column(DECIMAL(15, 4), default=0.0000)  # For cost tracking
	
	# GL Account
	gl_account_id = Column(String(36), nullable=False, index=True)
	
	# Tax Information
	tax_code = Column(String(20), nullable=True)
	tax_rate = Column(DECIMAL(5, 2), default=0.00)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	is_tax_inclusive = Column(Boolean, default=False)
	
	# Discount Information
	discount_percentage = Column(DECIMAL(5, 2), default=0.00)
	discount_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Commission Information
	commission_rate = Column(DECIMAL(5, 2), default=0.00)
	commission_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Sales Order Reference
	sales_order_line_id = Column(String(36), nullable=True)
	
	# Dimensions
	cost_center = Column(String(20), nullable=True)
	department = Column(String(20), nullable=True)
	project = Column(String(20), nullable=True)
	
	# Relationships
	invoice = relationship("CFARInvoice", back_populates="lines")
	tax_code_rel = relationship("CFARTaxCode", foreign_keys=[tax_code], primaryjoin="CFARInvoiceLine.tax_code == CFARTaxCode.code")
	
	def __repr__(self):
		return f"<CFARInvoiceLine {self.line_number}: ${self.line_amount}>"
	
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
	
	def calculate_commission(self):
		"""Calculate commission amount for the line"""
		if self.commission_rate > 0:
			self.commission_amount = self.line_amount * (self.commission_rate / 100)
		else:
			self.commission_amount = 0.00


class CFARPayment(Model, AuditMixin, BaseMixin):
	"""
	Customer payment records.
	
	Manages payment processing, cash application, and payment allocation
	to invoices and credit memos.
	"""
	__tablename__ = 'cf_ar_payment'
	
	# Identity
	payment_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Payment Information
	payment_number = Column(String(50), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Customer
	customer_id = Column(String(36), ForeignKey('cf_ar_customer.customer_id'), nullable=False, index=True)
	
	# Payment Details
	payment_date = Column(Date, nullable=False, index=True)
	payment_method = Column(String(50), nullable=False)  # CHECK, CASH, CREDIT_CARD, ACH, WIRE
	
	# Check/Reference Information
	check_number = Column(String(20), nullable=True, index=True)
	reference_number = Column(String(50), nullable=True)  # Credit card auth, ACH trace, etc.
	bank_account_id = Column(String(36), nullable=True)
	
	# Status
	status = Column(String(20), default='Draft', index=True)  # Draft, Posted, Cleared, Returned, Voided
	
	# Amounts
	payment_amount = Column(DECIMAL(15, 2), default=0.00)
	discount_taken = Column(DECIMAL(15, 2), default=0.00)
	unapplied_amount = Column(DECIMAL(15, 2), default=0.00)
	total_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Currency
	currency_code = Column(String(3), default='USD')
	exchange_rate = Column(DECIMAL(10, 6), default=1.000000)
	
	# Posting Information
	posted = Column(Boolean, default=False)
	posted_by = Column(String(36), nullable=True)
	posted_date = Column(DateTime, nullable=True)
	gl_batch_id = Column(String(36), nullable=True)
	
	# Bank Reconciliation
	cleared = Column(Boolean, default=False)
	cleared_date = Column(Date, nullable=True)
	bank_statement_date = Column(Date, nullable=True)
	
	# Return/NSF Information
	returned = Column(Boolean, default=False)
	return_date = Column(Date, nullable=True)
	return_reason = Column(String(200), nullable=True)
	nsf_fee = Column(DECIMAL(15, 2), default=0.00)
	
	# Lockbox Processing
	lockbox_batch_id = Column(String(36), nullable=True)
	lockbox_processed = Column(Boolean, default=False)
	
	# Document Management
	document_path = Column(String(500), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'payment_number', name='uq_ar_payment_number_tenant'),
	)
	
	# Relationships
	customer = relationship("CFARCustomer", back_populates="payments")
	payment_lines = relationship("CFARPaymentLine", back_populates="payment", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<CFARPayment {self.payment_number} - ${self.total_amount}>"
	
	def calculate_totals(self):
		"""Recalculate payment totals from lines"""
		applied_amount = sum(line.payment_amount for line in self.payment_lines)
		self.discount_taken = sum(line.discount_taken for line in self.payment_lines)
		self.unapplied_amount = self.payment_amount - applied_amount
		self.total_amount = self.payment_amount
	
	def can_post(self) -> bool:
		"""Check if payment can be posted"""
		return (
			self.status == 'Draft' and
			not self.posted and
			self.payment_amount > 0
		)
	
	def can_void(self) -> bool:
		"""Check if payment can be voided"""
		return (
			self.status == 'Posted' and
			not self.cleared and
			not self.returned
		)
	
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
		
		self.status = 'Voided'
		self.notes = (self.notes or '') + f"\nVoided by {user_id}: {reason}"


class CFARPaymentLine(Model, AuditMixin, BaseMixin):
	"""
	Payment allocation lines.
	
	Tracks how payments are allocated to specific invoices and credit memos,
	including discount calculations and partial payments.
	"""
	__tablename__ = 'cf_ar_payment_line'
	
	# Identity
	payment_line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	payment_id = Column(String(36), ForeignKey('cf_ar_payment.payment_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	
	# Invoice/Credit Memo Reference
	invoice_id = Column(String(36), ForeignKey('cf_ar_invoice.invoice_id'), nullable=True, index=True)
	credit_memo_id = Column(String(36), ForeignKey('cf_ar_credit_memo.credit_memo_id'), nullable=True, index=True)
	
	# Payment Allocation
	original_amount = Column(DECIMAL(15, 2), default=0.00)  # Original invoice/credit amount
	payment_amount = Column(DECIMAL(15, 2), default=0.00)  # Amount being paid
	discount_taken = Column(DECIMAL(15, 2), default=0.00)  # Early payment discount
	remaining_amount = Column(DECIMAL(15, 2), default=0.00)  # Remaining after payment
	
	# Discount Information
	discount_available = Column(DECIMAL(15, 2), default=0.00)
	discount_date = Column(Date, nullable=True)  # Last date for discount
	
	# Write-off Information
	writeoff_amount = Column(DECIMAL(15, 2), default=0.00)
	writeoff_reason = Column(String(200), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	payment = relationship("CFARPayment", back_populates="payment_lines")
	invoice = relationship("CFARInvoice", back_populates="payment_lines")
	credit_memo = relationship("CFARCreditMemo", back_populates="payment_lines")
	
	def __repr__(self):
		return f"<CFARPaymentLine {self.line_number}: ${self.payment_amount}>"
	
	def calculate_discount(self) -> Decimal:
		"""Calculate available early payment discount"""
		if not self.invoice or not self.invoice.customer.payment_terms_code:
			return Decimal('0.00')
		
		# Simplified discount calculation
		# In practice, this would reference payment terms configuration
		if self.payment.payment_date <= self.discount_date:
			# Example: 2/10 Net 30 = 2% discount if paid within 10 days
			return self.original_amount * Decimal('0.02')
		
		return Decimal('0.00')


class CFARCreditMemo(Model, AuditMixin, BaseMixin):
	"""
	Customer credit memos for returns, adjustments, and allowances.
	
	Manages credit memo processing, posting, and application to invoices.
	"""
	__tablename__ = 'cf_ar_credit_memo'
	
	# Identity
	credit_memo_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Credit Memo Information
	credit_memo_number = Column(String(50), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Customer
	customer_id = Column(String(36), ForeignKey('cf_ar_customer.customer_id'), nullable=False, index=True)
	
	# Dates
	credit_date = Column(Date, nullable=False, index=True)
	
	# Reference Information
	reference_invoice_id = Column(String(36), ForeignKey('cf_ar_invoice.invoice_id'), nullable=True, index=True)
	reason_code = Column(String(20), nullable=True)  # RETURN, ADJUSTMENT, ALLOWANCE, etc.
	
	# Status
	status = Column(String(20), default='Draft', index=True)  # Draft, Posted, Applied
	
	# Amounts
	subtotal_amount = Column(DECIMAL(15, 2), default=0.00)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	total_amount = Column(DECIMAL(15, 2), default=0.00)
	applied_amount = Column(DECIMAL(15, 2), default=0.00)
	unapplied_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Currency
	currency_code = Column(String(3), default='USD')
	exchange_rate = Column(DECIMAL(10, 6), default=1.000000)
	
	# Posting Information
	posted = Column(Boolean, default=False)
	posted_by = Column(String(36), nullable=True)
	posted_date = Column(DateTime, nullable=True)
	gl_batch_id = Column(String(36), nullable=True)
	
	# Return Information (if applicable)
	return_authorization = Column(String(50), nullable=True)
	received_date = Column(Date, nullable=True)
	
	# Document Management
	document_path = Column(String(500), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'credit_memo_number', name='uq_ar_credit_memo_number_tenant'),
	)
	
	# Relationships
	customer = relationship("CFARCustomer", back_populates="credit_memos")
	reference_invoice = relationship("CFARInvoice", foreign_keys=[reference_invoice_id])
	lines = relationship("CFARCreditMemoLine", back_populates="credit_memo", cascade="all, delete-orphan")
	payment_lines = relationship("CFARPaymentLine", back_populates="credit_memo")
	
	def __repr__(self):
		return f"<CFARCreditMemo {self.credit_memo_number} - ${self.total_amount}>"
	
	def calculate_totals(self):
		"""Recalculate credit memo totals from lines"""
		self.subtotal_amount = sum(line.line_amount for line in self.lines)
		self.tax_amount = sum(line.tax_amount for line in self.lines)
		self.total_amount = self.subtotal_amount + self.tax_amount
		self.unapplied_amount = self.total_amount - self.applied_amount
	
	def can_post(self) -> bool:
		"""Check if credit memo can be posted"""
		return (
			self.status == 'Draft' and
			not self.posted and
			self.total_amount > 0
		)
	
	def post_credit_memo(self, user_id: str):
		"""Post credit memo to GL"""
		if not self.can_post():
			raise ValueError("Credit memo cannot be posted")
		
		self.posted = True
		self.posted_by = user_id
		self.posted_date = datetime.utcnow()
		self.status = 'Posted'
		self.unapplied_amount = self.total_amount


class CFARCreditMemoLine(Model, AuditMixin, BaseMixin):
	"""
	Individual credit memo line items.
	
	Contains detailed line-level information for credit memos
	including quantities, amounts, and tax calculations.
	"""
	__tablename__ = 'cf_ar_credit_memo_line'
	
	# Identity
	line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	credit_memo_id = Column(String(36), ForeignKey('cf_ar_credit_memo.credit_memo_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	description = Column(Text, nullable=True)
	
	# Item Information
	item_code = Column(String(50), nullable=True)
	item_description = Column(String(200), nullable=True)
	
	# Reference to Original Invoice Line
	original_invoice_line_id = Column(String(36), nullable=True)
	
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
	
	# Relationships
	credit_memo = relationship("CFARCreditMemo", back_populates="lines")
	tax_code_rel = relationship("CFARTaxCode", foreign_keys=[tax_code], primaryjoin="CFARCreditMemoLine.tax_code == CFARTaxCode.code")
	
	def __repr__(self):
		return f"<CFARCreditMemoLine {self.line_number}: ${self.line_amount}>"
	
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


class CFARStatement(Model, AuditMixin, BaseMixin):
	"""
	Customer statements for periodic balance communication.
	
	Manages statement generation, delivery, and customer communication.
	"""
	__tablename__ = 'cf_ar_statement'
	
	# Identity
	statement_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Statement Information
	statement_number = Column(String(50), nullable=False, index=True)
	statement_date = Column(Date, nullable=False, index=True)
	statement_period_start = Column(Date, nullable=False)
	statement_period_end = Column(Date, nullable=False)
	
	# Customer
	customer_id = Column(String(36), ForeignKey('cf_ar_customer.customer_id'), nullable=False, index=True)
	
	# Statement Type
	statement_type = Column(String(20), default='MONTHLY')  # MONTHLY, QUARTERLY, ON_DEMAND
	
	# Balance Information
	beginning_balance = Column(DECIMAL(15, 2), default=0.00)
	charges = Column(DECIMAL(15, 2), default=0.00)
	payments = Column(DECIMAL(15, 2), default=0.00)
	adjustments = Column(DECIMAL(15, 2), default=0.00)
	ending_balance = Column(DECIMAL(15, 2), default=0.00)
	
	# Aging Information
	current_amount = Column(DECIMAL(15, 2), default=0.00)
	days_31_60 = Column(DECIMAL(15, 2), default=0.00)
	days_61_90 = Column(DECIMAL(15, 2), default=0.00)
	days_91_120 = Column(DECIMAL(15, 2), default=0.00)
	over_120_days = Column(DECIMAL(15, 2), default=0.00)
	
	# Status
	status = Column(String(20), default='Draft')  # Draft, Generated, Printed, Emailed, Delivered
	
	# Delivery Information
	delivery_method = Column(String(20), default='PRINT')  # PRINT, EMAIL, BOTH
	email_address = Column(String(100), nullable=True)
	printed_date = Column(DateTime, nullable=True)
	emailed_date = Column(DateTime, nullable=True)
	
	# Document Management
	document_path = Column(String(500), nullable=True)
	pdf_generated = Column(Boolean, default=False)
	
	# Template Information
	template_name = Column(String(100), nullable=True)
	include_remittance_slip = Column(Boolean, default=True)
	
	# Currency
	currency_code = Column(String(3), default='USD')
	
	# Notes
	message = Column(Text, nullable=True)  # Custom message to customer
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'statement_number', name='uq_ar_statement_number_tenant'),
	)
	
	# Relationships
	customer = relationship("CFARCustomer", back_populates="statements")
	
	def __repr__(self):
		return f"<CFARStatement {self.statement_number} - {self.customer.customer_name}>"
	
	def calculate_aging(self, as_of_date: Optional[date] = None):
		"""Calculate aging buckets for statement"""
		if as_of_date is None:
			as_of_date = self.statement_date
		
		# Reset aging amounts
		self.current_amount = 0.00
		self.days_31_60 = 0.00
		self.days_61_90 = 0.00
		self.days_91_120 = 0.00
		self.over_120_days = 0.00
		
		# Calculate aging for each outstanding invoice
		for invoice in self.customer.invoices:
			if invoice.status == 'Posted' and invoice.outstanding_amount > 0:
				days_past_due = invoice.days_past_due(as_of_date)
				
				if days_past_due <= 30:
					self.current_amount += invoice.outstanding_amount
				elif days_past_due <= 60:
					self.days_31_60 += invoice.outstanding_amount
				elif days_past_due <= 90:
					self.days_61_90 += invoice.outstanding_amount
				elif days_past_due <= 120:
					self.days_91_120 += invoice.outstanding_amount
				else:
					self.over_120_days += invoice.outstanding_amount
	
	def generate_statement(self, user_id: str):
		"""Generate the statement"""
		self.calculate_aging()
		self.ending_balance = (
			self.current_amount + self.days_31_60 + 
			self.days_61_90 + self.days_91_120 + self.over_120_days
		)
		self.status = 'Generated'


class CFARCollection(Model, AuditMixin, BaseMixin):
	"""
	Collection activities and dunning management.
	
	Tracks collection efforts, dunning letters, and follow-up activities
	for past due accounts.
	"""
	__tablename__ = 'cf_ar_collection'
	
	# Identity
	collection_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Customer
	customer_id = Column(String(36), ForeignKey('cf_ar_customer.customer_id'), nullable=False, index=True)
	
	# Collection Information
	collection_date = Column(Date, nullable=False, index=True)
	collection_type = Column(String(20), nullable=False)  # CALL, EMAIL, LETTER, VISIT
	collector_id = Column(String(36), nullable=False)  # User who performed collection
	
	# Dunning Information
	dunning_level = Column(Integer, default=1)
	days_past_due = Column(Integer, default=0)
	amount_past_due = Column(DECIMAL(15, 2), default=0.00)
	
	# Activity Details
	subject = Column(String(200), nullable=False)
	notes = Column(Text, nullable=True)
	outcome = Column(String(50), nullable=True)  # PROMISE_TO_PAY, DISPUTE, NO_CONTACT, etc.
	
	# Follow-up Information
	follow_up_date = Column(Date, nullable=True)
	follow_up_required = Column(Boolean, default=False)
	
	# Promise to Pay
	promised_amount = Column(DECIMAL(15, 2), default=0.00)
	promised_date = Column(Date, nullable=True)
	promise_kept = Column(Boolean, nullable=True)
	
	# Document Management
	document_path = Column(String(500), nullable=True)  # Dunning letter, email copy, etc.
	
	# Status
	status = Column(String(20), default='Open')  # Open, Closed, Follow_up_Required
	
	# Related Invoices (JSON field for flexibility)
	related_invoice_ids = Column(Text, nullable=True)  # JSON array of invoice IDs
	
	# Relationships
	customer = relationship("CFARCustomer", back_populates="collections")
	
	def __repr__(self):
		return f"<CFARCollection {self.collection_type} - {self.customer.customer_name}>"
	
	def get_related_invoices(self) -> List[str]:
		"""Get related invoice IDs"""
		if self.related_invoice_ids:
			return json.loads(self.related_invoice_ids)
		return []
	
	def set_related_invoices(self, invoice_ids: List[str]):
		"""Set related invoice IDs"""
		self.related_invoice_ids = json.dumps(invoice_ids)
	
	def mark_promise_kept(self, kept: bool):
		"""Mark promise to pay as kept or broken"""
		self.promise_kept = kept
		if kept:
			self.status = 'Closed'
		else:
			self.follow_up_required = True
			self.status = 'Follow_up_Required'


class CFARAging(Model, AuditMixin, BaseMixin):
	"""
	AR aging snapshots for performance and reporting.
	
	Stores aging analysis data to avoid recalculating
	aging buckets for each report.
	"""
	__tablename__ = 'cf_ar_aging'
	
	# Identity
	aging_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Aging Information
	as_of_date = Column(Date, nullable=False, index=True)
	customer_id = Column(String(36), ForeignKey('cf_ar_customer.customer_id'), nullable=False, index=True)
	
	# Aging Buckets (configurable)
	current_amount = Column(DECIMAL(15, 2), default=0.00)  # 0-30 days
	days_31_60 = Column(DECIMAL(15, 2), default=0.00)
	days_61_90 = Column(DECIMAL(15, 2), default=0.00)
	days_91_120 = Column(DECIMAL(15, 2), default=0.00)
	over_120_days = Column(DECIMAL(15, 2), default=0.00)
	
	# Totals
	total_outstanding = Column(DECIMAL(15, 2), default=0.00)
	
	# Collection Information
	collection_status = Column(String(20), default='Current')  # Current, Past_Due, Collections
	dunning_level = Column(Integer, default=0)
	last_collection_date = Column(Date, nullable=True)
	
	# Generation Info
	generated_date = Column(DateTime, default=datetime.utcnow)
	generated_by = Column(String(36), nullable=False)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'as_of_date', 'customer_id', name='uq_ar_aging_date_customer'),
	)
	
	# Relationships
	customer = relationship("CFARCustomer")
	
	def __repr__(self):
		return f"<CFARAging {self.customer.customer_name} as of {self.as_of_date}>"
	
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
	
	def calculate_aging_percentage(self) -> Dict[str, float]:
		"""Calculate aging percentages"""
		if self.total_outstanding == 0:
			return {bucket: 0.0 for bucket in ['current', '31_60', '61_90', '91_120', 'over_120']}
		
		buckets = self.get_aging_buckets()
		return {
			bucket: float(amount / self.total_outstanding * 100)
			for bucket, amount in buckets.items()
			if bucket != 'total'
		}


class CFARRecurringBilling(Model, AuditMixin, BaseMixin):
	"""
	Recurring billing setup for subscription customers.
	
	Manages automated invoice generation for recurring services,
	subscriptions, and maintenance contracts.
	"""
	__tablename__ = 'cf_ar_recurring_billing'
	
	# Identity
	recurring_billing_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Recurring Billing Information
	billing_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Customer
	customer_id = Column(String(36), ForeignKey('cf_ar_customer.customer_id'), nullable=False, index=True)
	
	# Billing Schedule
	frequency = Column(String(20), nullable=False)  # WEEKLY, MONTHLY, QUARTERLY, ANNUAL
	start_date = Column(Date, nullable=False)
	end_date = Column(Date, nullable=True)  # Null for indefinite
	next_billing_date = Column(Date, nullable=False, index=True)
	
	# Billing Details
	billing_amount = Column(DECIMAL(15, 2), nullable=False)
	tax_code = Column(String(20), nullable=True)
	payment_terms_code = Column(String(20), nullable=True)
	
	# GL Account
	gl_account_id = Column(String(36), nullable=False)
	
	# Invoice Template
	invoice_template = Column(String(100), nullable=True)
	invoice_description_template = Column(Text, nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True)
	is_paused = Column(Boolean, default=False)
	pause_start_date = Column(Date, nullable=True)
	pause_end_date = Column(Date, nullable=True)
	
	# Processing Information
	last_processed_date = Column(Date, nullable=True)
	invoices_generated = Column(Integer, default=0)
	
	# Auto-processing
	auto_process = Column(Boolean, default=True)
	advance_days = Column(Integer, default=0)  # Days in advance to generate
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	customer = relationship("CFARCustomer", back_populates="recurring_billings")
	
	def __repr__(self):
		return f"<CFARRecurringBilling {self.billing_name} - {self.customer.customer_name}>"
	
	def calculate_next_billing_date(self):
		"""Calculate the next billing date based on frequency"""
		from datetime import timedelta
		
		if not self.next_billing_date:
			return
		
		if self.frequency == 'WEEKLY':
			self.next_billing_date += timedelta(days=7)
		elif self.frequency == 'BIWEEKLY':
			self.next_billing_date += timedelta(days=14)
		elif self.frequency == 'MONTHLY':
			# Handle month-end scenarios
			next_month = self.next_billing_date.replace(day=1) + timedelta(days=32)
			self.next_billing_date = next_month.replace(day=min(self.next_billing_date.day, 
															   (next_month.replace(day=1) - timedelta(days=1)).day))
		elif self.frequency == 'QUARTERLY':
			# Add 3 months
			for _ in range(3):
				next_month = self.next_billing_date.replace(day=1) + timedelta(days=32)
				self.next_billing_date = next_month.replace(day=min(self.next_billing_date.day, 
																   (next_month.replace(day=1) - timedelta(days=1)).day))
		elif self.frequency == 'ANNUAL':
			# Add 1 year
			try:
				self.next_billing_date = self.next_billing_date.replace(year=self.next_billing_date.year + 1)
			except ValueError:  # Handle leap year edge case
				self.next_billing_date = self.next_billing_date.replace(year=self.next_billing_date.year + 1, day=28)
	
	def is_ready_for_billing(self, check_date: Optional[date] = None) -> bool:
		"""Check if ready for billing"""
		if check_date is None:
			check_date = date.today()
		
		if not self.is_active or self.is_paused:
			return False
		
		if self.end_date and check_date > self.end_date:
			return False
		
		# Check if we should bill in advance
		billing_date = self.next_billing_date - timedelta(days=self.advance_days)
		
		return check_date >= billing_date
	
	def pause_billing(self, start_date: date, end_date: Optional[date] = None):
		"""Pause billing for a period"""
		self.is_paused = True
		self.pause_start_date = start_date
		self.pause_end_date = end_date
	
	def resume_billing(self):
		"""Resume billing"""
		self.is_paused = False
		self.pause_start_date = None
		self.pause_end_date = None