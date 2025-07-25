"""
Order Entry Models

Database models for customer order entry including sales orders, order lines,
customer information, and order configuration.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ....auth_rbac.models import BaseMixin, AuditMixin, Model


class SOECustomer(Model, AuditMixin, BaseMixin):
	"""
	Customer master for sales orders.
	
	Stores customer information specific to order entry including
	preferences, credit information, and order history.
	"""
	__tablename__ = 'so_oe_customer'
	
	# Identity
	customer_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Customer Information
	customer_number = Column(String(20), nullable=False, index=True)
	customer_name = Column(String(200), nullable=False, index=True)
	customer_type = Column(String(50), default='RETAIL')  # RETAIL, WHOLESALE, CORPORATE
	
	# Contact Information
	contact_name = Column(String(100), nullable=True)
	email = Column(String(100), nullable=True, index=True)
	phone = Column(String(50), nullable=True)
	mobile = Column(String(50), nullable=True)
	fax = Column(String(50), nullable=True)
	website = Column(String(200), nullable=True)
	
	# Billing Address
	billing_address_line1 = Column(String(100), nullable=True)
	billing_address_line2 = Column(String(100), nullable=True)
	billing_city = Column(String(50), nullable=True)
	billing_state_province = Column(String(50), nullable=True)
	billing_postal_code = Column(String(20), nullable=True)
	billing_country = Column(String(50), nullable=True)
	
	# Default Shipping Address
	shipping_address_line1 = Column(String(100), nullable=True)
	shipping_address_line2 = Column(String(100), nullable=True)
	shipping_city = Column(String(50), nullable=True)
	shipping_state_province = Column(String(50), nullable=True)
	shipping_postal_code = Column(String(20), nullable=True)
	shipping_country = Column(String(50), nullable=True)
	
	# Order Preferences
	preferred_payment_method = Column(String(50), default='CREDIT_CARD')
	preferred_shipping_method = Column(String(50), default='STANDARD')
	payment_terms_code = Column(String(20), default='NET_30')
	price_level_id = Column(String(36), nullable=True)  # Link to pricing levels
	
	# Credit Information
	credit_limit = Column(DECIMAL(15, 2), default=0.00)
	credit_hold = Column(Boolean, default=False)
	credit_rating = Column(String(10), nullable=True)
	credit_check_required = Column(Boolean, default=False)
	
	# Tax Information
	tax_id = Column(String(50), nullable=True)
	tax_exempt = Column(Boolean, default=False)
	tax_exempt_number = Column(String(50), nullable=True)
	default_tax_code = Column(String(20), nullable=True)
	
	# Sales Information
	sales_rep_id = Column(String(36), nullable=True, index=True)
	territory_id = Column(String(36), nullable=True)
	customer_since = Column(Date, nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	allow_backorders = Column(Boolean, default=True)
	require_po_number = Column(Boolean, default=False)
	auto_approve_orders = Column(Boolean, default=True)
	order_approval_limit = Column(DECIMAL(15, 2), default=0.00)
	
	# Currency
	currency_code = Column(String(3), default='USD')
	
	# Balance Information
	current_balance = Column(DECIMAL(15, 2), default=0.00)
	ytd_orders = Column(DECIMAL(15, 2), default=0.00)
	last_order_date = Column(Date, nullable=True)
	total_orders = Column(Integer, default=0)
	
	# Integration
	ar_customer_id = Column(String(36), nullable=True, index=True)  # Link to AR customer
	
	# Notes
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'customer_number', name='uq_soe_customer_number_tenant'),
	)
	
	# Relationships
	sales_orders = relationship("SOESalesOrder", back_populates="customer")
	ship_to_addresses = relationship("SOEShipToAddress", back_populates="customer")
	
	def __repr__(self):
		return f"<SOECustomer {self.customer_number} - {self.customer_name}>"
	
	def can_place_order(self, order_amount: Decimal) -> Dict[str, Any]:
		"""Check if customer can place an order"""
		issues = []
		warnings = []
		
		if not self.is_active:
			issues.append("Customer is inactive")
		
		if self.credit_hold:
			issues.append("Customer is on credit hold")
		
		if self.credit_limit > 0 and (self.current_balance + order_amount) > self.credit_limit:
			issues.append(f"Order would exceed credit limit of ${self.credit_limit}")
		
		if self.credit_check_required and not self.credit_rating:
			warnings.append("Credit check required but no credit rating on file")
		
		return {
			'can_order': len(issues) == 0,
			'issues': issues,
			'warnings': warnings,
			'requires_approval': order_amount > self.order_approval_limit
		}


class SOEShipToAddress(Model, AuditMixin, BaseMixin):
	"""
	Customer shipping addresses for flexible delivery options.
	
	Supports multiple shipping addresses per customer with
	address validation and shipping preferences.
	"""
	__tablename__ = 'so_oe_ship_to_address'
	
	# Identity
	ship_to_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	customer_id = Column(String(36), ForeignKey('so_oe_customer.customer_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Address Information
	address_name = Column(String(100), nullable=False)  # e.g., "Main Warehouse", "Store #1"
	contact_name = Column(String(100), nullable=True)
	
	# Address Details
	address_line1 = Column(String(100), nullable=False)
	address_line2 = Column(String(100), nullable=True)
	city = Column(String(50), nullable=False)
	state_province = Column(String(50), nullable=False)
	postal_code = Column(String(20), nullable=False)
	country = Column(String(50), nullable=False)
	
	# Contact Information
	phone = Column(String(50), nullable=True)
	email = Column(String(100), nullable=True)
	
	# Shipping Preferences
	preferred_carrier = Column(String(50), nullable=True)
	preferred_service_level = Column(String(50), nullable=True)
	delivery_instructions = Column(Text, nullable=True)
	
	# Address Validation
	is_validated = Column(Boolean, default=False)
	validation_date = Column(DateTime, nullable=True)
	validation_service = Column(String(50), nullable=True)
	
	# Configuration
	is_default = Column(Boolean, default=False)
	is_active = Column(Boolean, default=True)
	requires_appointment = Column(Boolean, default=False)
	loading_dock_available = Column(Boolean, default=False)
	
	# Geographic Information
	latitude = Column(DECIMAL(10, 8), nullable=True)
	longitude = Column(DECIMAL(11, 8), nullable=True)
	timezone = Column(String(50), nullable=True)
	
	# Tax Information
	tax_jurisdiction = Column(String(100), nullable=True)
	
	# Relationships
	customer = relationship("SOECustomer", back_populates="ship_to_addresses")
	sales_orders = relationship("SOESalesOrder", back_populates="ship_to_address")
	
	def __repr__(self):
		return f"<SOEShipToAddress {self.address_name} - {self.customer.customer_name}>"


class SOESalesOrder(Model, AuditMixin, BaseMixin):
	"""
	Sales order header containing customer order information.
	
	Manages the complete order lifecycle from entry through fulfillment
	with pricing, shipping, and payment integration.
	"""
	__tablename__ = 'so_oe_sales_order'
	
	# Identity
	order_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Order Information
	order_number = Column(String(50), nullable=False, index=True)
	description = Column(Text, nullable=True)
	order_type = Column(String(20), default='STANDARD')  # STANDARD, RUSH, BACKORDER, DROP_SHIP
	
	# Customer Information
	customer_id = Column(String(36), ForeignKey('so_oe_customer.customer_id'), nullable=False, index=True)
	ship_to_id = Column(String(36), ForeignKey('so_oe_ship_to_address.ship_to_id'), nullable=True, index=True)
	
	# Dates
	order_date = Column(Date, nullable=False, index=True)
	requested_date = Column(Date, nullable=True, index=True)
	promised_date = Column(Date, nullable=True, index=True)
	shipped_date = Column(Date, nullable=True)
	
	# Reference Information
	customer_po_number = Column(String(50), nullable=True, index=True)
	quote_id = Column(String(36), nullable=True, index=True)  # Link to quotation
	project_id = Column(String(36), nullable=True)
	
	# Status and Workflow
	status = Column(String(20), default='DRAFT', index=True)  # DRAFT, SUBMITTED, APPROVED, PROCESSING, SHIPPED, INVOICED, CANCELLED
	hold_status = Column(String(20), nullable=True)  # CREDIT_HOLD, INVENTORY_HOLD, PRICING_HOLD
	hold_reason = Column(String(200), nullable=True)
	
	# Approval Workflow
	requires_approval = Column(Boolean, default=False)
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	approval_notes = Column(Text, nullable=True)
	
	# Amounts (calculated from lines)
	subtotal_amount = Column(DECIMAL(15, 2), default=0.00)
	discount_amount = Column(DECIMAL(15, 2), default=0.00)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	shipping_amount = Column(DECIMAL(15, 2), default=0.00)
	handling_amount = Column(DECIMAL(15, 2), default=0.00)
	total_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Pricing Information
	price_level_id = Column(String(36), nullable=True)
	currency_code = Column(String(3), default='USD')
	exchange_rate = Column(DECIMAL(10, 6), default=1.000000)
	
	# Payment Information
	payment_method = Column(String(50), nullable=True)
	payment_terms_code = Column(String(20), nullable=True)
	credit_card_last_four = Column(String(4), nullable=True)
	
	# Shipping Information
	shipping_method = Column(String(50), nullable=True)
	carrier = Column(String(50), nullable=True)
	service_level = Column(String(50), nullable=True)
	tracking_number = Column(String(100), nullable=True)
	freight_terms = Column(String(20), default='PREPAID')  # PREPAID, COLLECT, THIRD_PARTY
	
	# Tax Information
	tax_exempt = Column(Boolean, default=False)
	tax_exempt_number = Column(String(50), nullable=True)
	
	# Sales Information
	sales_rep_id = Column(String(36), nullable=True, index=True)
	territory_id = Column(String(36), nullable=True)
	source_code = Column(String(20), nullable=True)  # WEB, PHONE, EMAIL, etc.
	
	# Commission Information
	commission_rate = Column(DECIMAL(5, 2), default=0.00)
	commission_amount = Column(DECIMAL(15, 2), default=0.00)
	commission_paid = Column(Boolean, default=False)
	
	# Fulfillment Information
	warehouse_id = Column(String(36), nullable=True, index=True)
	pick_list_printed = Column(Boolean, default=False)
	pick_list_date = Column(DateTime, nullable=True)
	packed = Column(Boolean, default=False)
	packed_date = Column(DateTime, nullable=True)
	
	# Document Management
	documents_generated = Column(Integer, default=0)
	order_confirmation_sent = Column(Boolean, default=False)
	
	# Integration Status
	exported_to_wms = Column(Boolean, default=False)
	exported_to_ar = Column(Boolean, default=False)
	wms_order_id = Column(String(36), nullable=True)
	ar_invoice_id = Column(String(36), nullable=True)
	
	# Special Instructions
	picking_instructions = Column(Text, nullable=True)
	packing_instructions = Column(Text, nullable=True)
	shipping_instructions = Column(Text, nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'order_number', name='uq_soe_order_number_tenant'),
		Index('idx_soe_order_status_date', 'status', 'order_date'),
		Index('idx_soe_order_customer_date', 'customer_id', 'order_date'),
	)
	
	# Relationships
	customer = relationship("SOECustomer", back_populates="sales_orders")
	ship_to_address = relationship("SOEShipToAddress", back_populates="sales_orders")
	lines = relationship("SOEOrderLine", back_populates="sales_order", cascade="all, delete-orphan")
	charges = relationship("SOEOrderCharge", back_populates="sales_order", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<SOESalesOrder {self.order_number} - ${self.total_amount}>"
	
	def calculate_totals(self):
		"""Recalculate order totals from lines and charges"""
		# Calculate line totals
		self.subtotal_amount = sum(line.extended_amount for line in self.lines)
		total_line_discount = sum(line.discount_amount for line in self.lines)
		
		# Calculate tax from lines
		self.tax_amount = sum(line.tax_amount for line in self.lines)
		
		# Calculate charges (shipping, handling, etc.)
		charge_amounts = {
			charge.charge_type: charge.charge_amount 
			for charge in self.charges
		}
		self.shipping_amount = charge_amounts.get('SHIPPING', 0.00)
		self.handling_amount = charge_amounts.get('HANDLING', 0.00)
		
		# Calculate order-level discount
		order_discount = charge_amounts.get('DISCOUNT', 0.00)
		self.discount_amount = total_line_discount + order_discount
		
		# Calculate total
		self.total_amount = (
			self.subtotal_amount - self.discount_amount +
			self.tax_amount + self.shipping_amount + self.handling_amount
		)
	
	def can_approve(self) -> bool:
		"""Check if order can be approved"""
		return (
			self.status in ['DRAFT', 'SUBMITTED'] and
			self.requires_approval and
			not self.approved and
			self.total_amount > 0 and
			not self.hold_status
		)
	
	def can_process(self) -> bool:
		"""Check if order can be processed"""
		return (
			self.status in ['APPROVED', 'SUBMITTED'] and
			(not self.requires_approval or self.approved) and
			not self.hold_status and
			self.total_amount > 0
		)
	
	def can_cancel(self) -> bool:
		"""Check if order can be cancelled"""
		return self.status in ['DRAFT', 'SUBMITTED', 'APPROVED'] and not self.shipped_date
	
	def submit_order(self, user_id: str):
		"""Submit order for processing"""
		if self.status != 'DRAFT':
			raise ValueError("Only draft orders can be submitted")
		
		self.calculate_totals()
		
		# Check customer credit and order limits
		check_result = self.customer.can_place_order(self.total_amount)
		if not check_result['can_order']:
			raise ValueError(f"Cannot submit order: {', '.join(check_result['issues'])}")
		
		# Set approval requirement
		self.requires_approval = check_result['requires_approval']
		
		if self.requires_approval:
			self.status = 'SUBMITTED'
		else:
			self.status = 'APPROVED'
			self.approved = True
			self.approved_by = user_id
			self.approved_date = datetime.utcnow()
	
	def approve_order(self, user_id: str, notes: str = None):
		"""Approve the order"""
		if not self.can_approve():
			raise ValueError("Order cannot be approved")
		
		self.approved = True
		self.approved_by = user_id
		self.approved_date = datetime.utcnow()
		self.approval_notes = notes
		self.status = 'APPROVED'
	
	def cancel_order(self, user_id: str, reason: str):
		"""Cancel the order"""
		if not self.can_cancel():
			raise ValueError("Order cannot be cancelled")
		
		self.status = 'CANCELLED'
		self.notes = (self.notes or '') + f"\nCancelled by {user_id}: {reason}"


class SOEOrderLine(Model, AuditMixin, BaseMixin):
	"""
	Individual order line items with pricing, inventory, and fulfillment details.
	
	Contains detailed line-level information including items, quantities,
	pricing, discounts, and inventory allocation.
	"""
	__tablename__ = 'so_oe_order_line'
	
	# Identity
	line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	order_id = Column(String(36), ForeignKey('so_oe_sales_order.order_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	line_type = Column(String(20), default='PRODUCT')  # PRODUCT, SERVICE, DISCOUNT, CHARGE
	description = Column(Text, nullable=True)
	
	# Item Information
	item_id = Column(String(36), nullable=True, index=True)  # Link to inventory item
	item_code = Column(String(50), nullable=True, index=True)
	item_description = Column(String(200), nullable=True)
	item_type = Column(String(20), default='PRODUCT')  # PRODUCT, SERVICE, KIT, BUNDLE
	
	# Quantities
	quantity_ordered = Column(DECIMAL(12, 4), default=1.0000)
	quantity_allocated = Column(DECIMAL(12, 4), default=0.0000)
	quantity_shipped = Column(DECIMAL(12, 4), default=0.0000)
	quantity_backordered = Column(DECIMAL(12, 4), default=0.0000)
	
	# Units
	unit_of_measure = Column(String(10), default='EA')
	unit_conversion_factor = Column(DECIMAL(10, 4), default=1.0000)
	
	# Pricing
	unit_price = Column(DECIMAL(15, 4), default=0.0000)
	list_price = Column(DECIMAL(15, 4), default=0.0000)
	cost_price = Column(DECIMAL(15, 4), default=0.0000)  # For margin calculation
	extended_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Discounts
	discount_percentage = Column(DECIMAL(5, 2), default=0.00)
	discount_amount = Column(DECIMAL(15, 2), default=0.00)
	discount_reason_code = Column(String(20), nullable=True)
	
	# Tax Information
	tax_code = Column(String(20), nullable=True)
	tax_rate = Column(DECIMAL(5, 2), default=0.00)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	is_taxable = Column(Boolean, default=True)
	
	# Inventory Information
	warehouse_id = Column(String(36), nullable=True, index=True)
	location_id = Column(String(36), nullable=True)
	lot_number = Column(String(50), nullable=True)
	serial_number = Column(String(50), nullable=True)
	
	# Dates
	requested_date = Column(Date, nullable=True)
	promised_date = Column(Date, nullable=True)
	shipped_date = Column(Date, nullable=True)
	
	# Status
	line_status = Column(String(20), default='OPEN')  # OPEN, ALLOCATED, SHIPPED, CANCELLED, BACKORDERED
	
	# Commission Information
	commission_rate = Column(DECIMAL(5, 2), default=0.00)
	commission_amount = Column(DECIMAL(15, 2), default=0.00)
	commissionable = Column(Boolean, default=True)
	
	# Special Handling
	special_instructions = Column(Text, nullable=True)
	requires_special_handling = Column(Boolean, default=False)
	hazardous_material = Column(Boolean, default=False)
	
	# Vendor Information (for drop ship)
	vendor_id = Column(String(36), nullable=True)
	vendor_item_code = Column(String(50), nullable=True)
	drop_ship = Column(Boolean, default=False)
	
	# Kit/Bundle Information
	parent_line_id = Column(String(36), ForeignKey('so_oe_order_line.line_id'), nullable=True)
	kit_sequence = Column(Integer, nullable=True)
	
	# Integration
	inventory_allocated = Column(Boolean, default=False)
	allocation_id = Column(String(36), nullable=True)
	
	# Dimensions for reporting
	cost_center = Column(String(20), nullable=True)
	department = Column(String(20), nullable=True)
	project = Column(String(20), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	sales_order = relationship("SOESalesOrder", back_populates="lines")
	parent_line = relationship("SOEOrderLine", remote_side=[line_id])
	child_lines = relationship("SOEOrderLine", back_populates="parent_line")
	
	def __repr__(self):
		return f"<SOEOrderLine {self.line_number}: {self.item_code} x {self.quantity_ordered}>"
	
	def calculate_extended_amount(self):
		"""Calculate extended amount after discounts"""
		gross_amount = self.quantity_ordered * self.unit_price
		
		if self.discount_percentage > 0:
			self.discount_amount = gross_amount * (self.discount_percentage / 100)
		
		self.extended_amount = gross_amount - self.discount_amount
	
	def calculate_tax(self):
		"""Calculate tax amount for the line"""
		if self.is_taxable and self.tax_rate > 0:
			self.tax_amount = self.extended_amount * (self.tax_rate / 100)
		else:
			self.tax_amount = 0.00
	
	def calculate_commission(self):
		"""Calculate commission amount for the line"""
		if self.commissionable and self.commission_rate > 0:
			self.commission_amount = self.extended_amount * (self.commission_rate / 100)
		else:
			self.commission_amount = 0.00
	
	def can_allocate_inventory(self) -> bool:
		"""Check if inventory can be allocated"""
		return (
			self.item_type == 'PRODUCT' and
			not self.drop_ship and
			self.quantity_ordered > self.quantity_allocated and
			self.line_status == 'OPEN'
		)
	
	def get_quantity_available_to_ship(self) -> Decimal:
		"""Get quantity available to ship"""
		return min(self.quantity_allocated, self.quantity_ordered - self.quantity_shipped)


class SOEOrderCharge(Model, AuditMixin, BaseMixin):
	"""
	Order-level charges and fees including shipping, handling, and discounts.
	
	Manages additional charges that apply to the entire order
	rather than specific line items.
	"""
	__tablename__ = 'so_oe_order_charge'
	
	# Identity
	charge_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	order_id = Column(String(36), ForeignKey('so_oe_sales_order.order_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Charge Information
	charge_type = Column(String(20), nullable=False)  # SHIPPING, HANDLING, DISCOUNT, SURCHARGE
	charge_code = Column(String(20), nullable=True)
	description = Column(String(200), nullable=False)
	
	# Amount Information
	charge_amount = Column(DECIMAL(15, 2), default=0.00)
	calculation_method = Column(String(20), default='FIXED')  # FIXED, PERCENTAGE, WEIGHT_BASED, etc.
	calculation_base = Column(DECIMAL(15, 2), nullable=True)  # Base amount for percentage calculations
	
	# Tax Information
	is_taxable = Column(Boolean, default=False)
	tax_code = Column(String(20), nullable=True)
	tax_rate = Column(DECIMAL(5, 2), default=0.00)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# GL Account
	gl_account_id = Column(String(36), nullable=True)
	
	# Configuration
	is_automatic = Column(Boolean, default=False)  # Automatically calculated
	can_override = Column(Boolean, default=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	sales_order = relationship("SOESalesOrder", back_populates="charges")
	
	def __repr__(self):
		return f"<SOEOrderCharge {self.charge_type}: ${self.charge_amount}>"
	
	def calculate_tax(self):
		"""Calculate tax amount for the charge"""
		if self.is_taxable and self.tax_rate > 0:
			self.tax_amount = self.charge_amount * (self.tax_rate / 100)
		else:
			self.tax_amount = 0.00


class SOEPriceLevel(Model, AuditMixin, BaseMixin):
	"""
	Customer price levels for tiered pricing strategies.
	
	Manages different pricing tiers for customers based on
	volume, relationship, or customer type.
	"""
	__tablename__ = 'so_oe_price_level'
	
	# Identity
	price_level_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Price Level Information
	level_code = Column(String(20), nullable=False, index=True)
	level_name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	
	# Pricing Configuration
	discount_percentage = Column(DECIMAL(5, 2), default=0.00)
	markup_percentage = Column(DECIMAL(5, 2), default=0.00)
	price_calculation_method = Column(String(20), default='LIST_MINUS_DISCOUNT')  # LIST_MINUS_DISCOUNT, COST_PLUS_MARKUP
	
	# Qualification Criteria
	minimum_order_amount = Column(DECIMAL(15, 2), default=0.00)
	minimum_annual_volume = Column(DECIMAL(15, 2), default=0.00)
	customer_type = Column(String(50), nullable=True)  # Restrict to specific customer types
	
	# Effective Dates
	effective_date = Column(Date, nullable=True)
	expiration_date = Column(Date, nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_default = Column(Boolean, default=False)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'level_code', name='uq_soe_price_level_code_tenant'),
	)
	
	def __repr__(self):
		return f"<SOEPriceLevel {self.level_code} - {self.level_name}>"
	
	def is_valid_for_date(self, check_date: date) -> bool:
		"""Check if price level is valid for a specific date"""
		if not self.is_active:
			return False
		
		if self.effective_date and check_date < self.effective_date:
			return False
		
		if self.expiration_date and check_date > self.expiration_date:
			return False
		
		return True
	
	def calculate_price(self, list_price: Decimal, cost_price: Decimal = None) -> Decimal:
		"""Calculate price based on price level configuration"""
		if self.price_calculation_method == 'LIST_MINUS_DISCOUNT':
			return list_price * (1 - self.discount_percentage / 100)
		elif self.price_calculation_method == 'COST_PLUS_MARKUP' and cost_price:
			return cost_price * (1 + self.markup_percentage / 100)
		else:
			return list_price


class SOEOrderTemplate(Model, AuditMixin, BaseMixin):
	"""
	Order templates for frequently ordered item combinations.
	
	Allows customers to save and reorder common item combinations
	to streamline the ordering process.
	"""
	__tablename__ = 'so_oe_order_template'
	
	# Identity
	template_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Template Information
	template_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	template_type = Column(String(20), default='CUSTOMER')  # CUSTOMER, COMPANY, PUBLIC
	
	# Customer Information
	customer_id = Column(String(36), ForeignKey('so_oe_customer.customer_id'), nullable=True, index=True)
	
	# Template Configuration
	is_active = Column(Boolean, default=True)
	is_public = Column(Boolean, default=False)  # Available to all customers
	usage_count = Column(Integer, default=0)
	last_used_date = Column(DateTime, nullable=True)
	
	# Default Values
	default_ship_to_id = Column(String(36), nullable=True)
	default_requested_date_offset = Column(Integer, default=0)  # Days from order date
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	customer = relationship("SOECustomer")
	template_lines = relationship("SOEOrderTemplateLine", back_populates="template", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<SOEOrderTemplate {self.template_name}>"
	
	def create_order_from_template(self, customer_id: str, order_date: date = None) -> Dict[str, Any]:
		"""Create order data structure from template"""
		if order_date is None:
			order_date = date.today()
		
		requested_date = order_date + timedelta(days=self.default_requested_date_offset) if self.default_requested_date_offset else None
		
		order_data = {
			'customer_id': customer_id,
			'ship_to_id': self.default_ship_to_id,
			'order_date': order_date,
			'requested_date': requested_date,
			'description': f"Order from template: {self.template_name}",
			'lines': []
		}
		
		for template_line in self.template_lines:
			line_data = {
				'item_id': template_line.item_id,
				'item_code': template_line.item_code,
				'quantity_ordered': template_line.default_quantity,
				'description': template_line.description,
				'notes': template_line.notes
			}
			order_data['lines'].append(line_data)
		
		# Update usage statistics
		self.usage_count += 1
		self.last_used_date = datetime.utcnow()
		
		return order_data


class SOEOrderTemplateLine(Model, AuditMixin, BaseMixin):
	"""
	Template line items for order templates.
	
	Defines the items and default quantities for order templates.
	"""
	__tablename__ = 'so_oe_order_template_line'
	
	# Identity
	template_line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	template_id = Column(String(36), ForeignKey('so_oe_order_template.template_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	description = Column(Text, nullable=True)
	
	# Item Information
	item_id = Column(String(36), nullable=True, index=True)
	item_code = Column(String(50), nullable=False, index=True)
	item_description = Column(String(200), nullable=True)
	
	# Default Quantities
	default_quantity = Column(DECIMAL(12, 4), default=1.0000)
	minimum_quantity = Column(DECIMAL(12, 4), nullable=True)
	maximum_quantity = Column(DECIMAL(12, 4), nullable=True)
	
	# Configuration
	is_required = Column(Boolean, default=False)
	allow_quantity_change = Column(Boolean, default=True)
	allow_substitution = Column(Boolean, default=False)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	template = relationship("SOEOrderTemplate", back_populates="template_lines")
	
	def __repr__(self):
		return f"<SOEOrderTemplateLine {self.line_number}: {self.item_code}>"


class SOEOrderSequence(Model, AuditMixin, BaseMixin):
	"""
	Order number sequence management for different order types.
	
	Manages automatic order number generation with configurable
	prefixes, suffixes, and numbering patterns.
	"""
	__tablename__ = 'so_oe_order_sequence'
	
	# Identity
	sequence_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Sequence Information
	sequence_name = Column(String(100), nullable=False)
	order_type = Column(String(20), nullable=False, index=True)
	
	# Number Format
	prefix = Column(String(10), nullable=True)
	suffix = Column(String(10), nullable=True)
	number_length = Column(Integer, default=6)
	current_number = Column(Integer, default=1)
	increment_by = Column(Integer, default=1)
	
	# Date-based numbering
	reset_period = Column(String(10), nullable=True)  # DAILY, MONTHLY, YEARLY, NEVER
	last_reset_date = Column(Date, nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	zero_pad = Column(Boolean, default=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'order_type', name='uq_soe_sequence_type_tenant'),
	)
	
	def __repr__(self):
		return f"<SOEOrderSequence {self.sequence_name}>"
	
	def get_next_number(self) -> str:
		"""Generate the next order number in sequence"""
		# Check if we need to reset based on period
		today = date.today()
		needs_reset = False
		
		if self.reset_period == 'DAILY' and (not self.last_reset_date or self.last_reset_date < today):
			needs_reset = True
		elif self.reset_period == 'MONTHLY' and (not self.last_reset_date or 
			self.last_reset_date.replace(day=1) < today.replace(day=1)):
			needs_reset = True
		elif self.reset_period == 'YEARLY' and (not self.last_reset_date or 
			self.last_reset_date.year < today.year):
			needs_reset = True
		
		if needs_reset:
			self.current_number = 1
			self.last_reset_date = today
		
		# Generate number
		number_part = str(self.current_number)
		if self.zero_pad:
			number_part = number_part.zfill(self.number_length)
		
		# Build full number
		full_number = ''
		if self.prefix:
			full_number += self.prefix
		full_number += number_part
		if self.suffix:
			full_number += self.suffix
		
		# Increment for next use
		self.current_number += self.increment_by
		
		return full_number