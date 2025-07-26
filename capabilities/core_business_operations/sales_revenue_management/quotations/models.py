"""
Quotations Models

Database models for customer quotations, proposals, and quote-to-order
conversion with comprehensive pricing and terms management.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ....auth_rbac.models import BaseMixin, AuditMixin, Model


class SOQQuotation(Model, AuditMixin, BaseMixin):
	"""
	Customer quotation header with pricing and terms.
	
	Manages complete quotation lifecycle from creation through
	approval, customer response, and conversion to orders.
	"""
	__tablename__ = 'so_q_quotation'
	
	# Identity
	quotation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Quotation Information
	quote_number = Column(String(50), nullable=False, index=True)
	quote_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Customer Information
	customer_id = Column(String(36), nullable=False, index=True)
	customer_name = Column(String(200), nullable=False)
	contact_name = Column(String(100), nullable=True)
	contact_email = Column(String(100), nullable=True)
	contact_phone = Column(String(50), nullable=True)
	
	# Request Information
	rfq_number = Column(String(50), nullable=True)  # Request for Quote reference
	opportunity_id = Column(String(36), nullable=True)  # CRM opportunity link
	
	# Dates
	quote_date = Column(Date, nullable=False, index=True)
	valid_until_date = Column(Date, nullable=False, index=True)
	requested_delivery_date = Column(Date, nullable=True)
	
	# Status and Workflow
	status = Column(String(20), default='DRAFT', index=True)  # DRAFT, SUBMITTED, APPROVED, SENT, ACCEPTED, REJECTED, EXPIRED, CONVERTED
	
	# Approval Workflow
	requires_approval = Column(Boolean, default=False)
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Amounts
	subtotal_amount = Column(DECIMAL(15, 2), default=0.00)
	discount_amount = Column(DECIMAL(15, 2), default=0.00)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	shipping_amount = Column(DECIMAL(15, 2), default=0.00)
	total_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Pricing Information
	currency_code = Column(String(3), default='USD')
	exchange_rate = Column(DECIMAL(10, 6), default=1.000000)
	price_level_id = Column(String(36), nullable=True)
	
	# Terms and Conditions
	payment_terms = Column(String(200), nullable=True)
	delivery_terms = Column(String(200), nullable=True)
	warranty_terms = Column(Text, nullable=True)
	special_terms = Column(Text, nullable=True)
	
	# Sales Information
	sales_rep_id = Column(String(36), nullable=True, index=True)
	territory_id = Column(String(36), nullable=True)
	project_id = Column(String(36), nullable=True)
	
	# Customer Response
	customer_response = Column(String(20), nullable=True)  # ACCEPTED, REJECTED, NEGOTIATING
	customer_response_date = Column(Date, nullable=True)
	customer_comments = Column(Text, nullable=True)
	
	# Conversion Information
	converted_to_order = Column(Boolean, default=False)
	order_id = Column(String(36), nullable=True, index=True)
	conversion_date = Column(Date, nullable=True)
	
	# Revision Management
	revision_number = Column(Integer, default=1)
	parent_quote_id = Column(String(36), ForeignKey('so_q_quotation.quotation_id'), nullable=True)
	is_current_revision = Column(Boolean, default=True)
	
	# Document Management
	document_template = Column(String(100), nullable=True)
	document_generated = Column(Boolean, default=False)
	document_path = Column(String(500), nullable=True)
	sent_to_customer = Column(Boolean, default=False)
	sent_date = Column(DateTime, nullable=True)
	
	# Follow-up
	follow_up_date = Column(Date, nullable=True)
	follow_up_notes = Column(Text, nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'quote_number', name='uq_soq_quote_number_tenant'),
	)
	
	# Relationships
	lines = relationship("SOQQuotationLine", back_populates="quotation", cascade="all, delete-orphan")
	revisions = relationship("SOQQuotation", back_populates="parent_quote")
	parent_quote = relationship("SOQQuotation", remote_side=[quotation_id])
	
	def __repr__(self):
		return f"<SOQQuotation {self.quote_number} - ${self.total_amount}>"
	
	def calculate_totals(self):
		"""Recalculate quotation totals from lines"""
		self.subtotal_amount = sum(line.extended_amount for line in self.lines)
		self.discount_amount = sum(line.discount_amount for line in self.lines)
		self.tax_amount = sum(line.tax_amount for line in self.lines)
		self.total_amount = self.subtotal_amount - self.discount_amount + self.tax_amount + self.shipping_amount
	
	def is_valid(self) -> bool:
		"""Check if quotation is still valid"""
		return date.today() <= self.valid_until_date and self.status not in ['EXPIRED', 'CONVERTED', 'REJECTED']
	
	def expire_quote(self):
		"""Mark quotation as expired"""
		if self.status in ['SENT', 'SUBMITTED']:
			self.status = 'EXPIRED'
	
	def convert_to_order(self, order_id: str, user_id: str):
		"""Mark quotation as converted to order"""
		self.converted_to_order = True
		self.order_id = order_id
		self.conversion_date = date.today()
		self.status = 'CONVERTED'
		self.customer_response = 'ACCEPTED'
		self.customer_response_date = date.today()


class SOQQuotationLine(Model, AuditMixin, BaseMixin):
	"""
	Individual quotation line items with detailed pricing.
	
	Contains line-level pricing, discounts, and configuration
	for complex quotation scenarios.
	"""
	__tablename__ = 'so_q_quotation_line'
	
	# Identity
	line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	quotation_id = Column(String(36), ForeignKey('so_q_quotation.quotation_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	line_type = Column(String(20), default='PRODUCT')  # PRODUCT, SERVICE, DISCOUNT, TEXT
	description = Column(Text, nullable=True)
	
	# Item Information
	item_id = Column(String(36), nullable=True, index=True)
	item_code = Column(String(50), nullable=True, index=True)
	item_description = Column(String(200), nullable=True)
	item_type = Column(String(20), default='PRODUCT')
	
	# Quantities and Units
	quantity = Column(DECIMAL(12, 4), default=1.0000)
	unit_of_measure = Column(String(10), default='EA')
	
	# Pricing
	list_price = Column(DECIMAL(15, 4), default=0.0000)
	unit_price = Column(DECIMAL(15, 4), default=0.0000)
	extended_amount = Column(DECIMAL(15, 2), default=0.00)
	cost_price = Column(DECIMAL(15, 4), default=0.0000)
	margin_percentage = Column(DECIMAL(5, 2), default=0.00)
	
	# Discounts
	discount_percentage = Column(DECIMAL(5, 2), default=0.00)
	discount_amount = Column(DECIMAL(15, 2), default=0.00)
	discount_reason = Column(String(200), nullable=True)
	
	# Tax Information
	tax_code = Column(String(20), nullable=True)
	tax_rate = Column(DECIMAL(5, 2), default=0.00)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	is_taxable = Column(Boolean, default=True)
	
	# Delivery Information
	lead_time_days = Column(Integer, nullable=True)
	delivery_date = Column(Date, nullable=True)
	
	# Configuration Options
	configuration_data = Column(Text, nullable=True)  # JSON for configurable products
	
	# Alternative Options
	is_alternative = Column(Boolean, default=False)
	alternative_group = Column(String(20), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	technical_specifications = Column(Text, nullable=True)
	
	# Relationships
	quotation = relationship("SOQQuotation", back_populates="lines")
	
	def __repr__(self):
		return f"<SOQQuotationLine {self.line_number}: {self.item_code} x {self.quantity}>"
	
	def calculate_extended_amount(self):
		"""Calculate extended amount after discounts"""
		gross_amount = self.quantity * self.unit_price
		self.extended_amount = gross_amount - self.discount_amount
	
	def calculate_margin(self):
		"""Calculate margin percentage"""
		if self.cost_price > 0 and self.unit_price > 0:
			self.margin_percentage = ((self.unit_price - self.cost_price) / self.unit_price) * 100
		else:
			self.margin_percentage = 0


class SOQQuoteTemplate(Model, AuditMixin, BaseMixin):
	"""
	Quotation templates for standardized quotes.
	
	Manages reusable quotation templates for common
	products, services, and customer scenarios.
	"""
	__tablename__ = 'so_q_quote_template'
	
	# Identity
	template_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Template Information
	template_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	template_type = Column(String(20), default='STANDARD')  # STANDARD, PRODUCT_BUNDLE, SERVICE_PACKAGE
	
	# Application Criteria
	product_category = Column(String(50), nullable=True)
	customer_type = Column(String(50), nullable=True)
	industry_focus = Column(String(50), nullable=True)
	
	# Default Terms
	default_payment_terms = Column(String(200), nullable=True)
	default_delivery_terms = Column(String(200), nullable=True)
	default_validity_days = Column(Integer, default=30)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_public = Column(Boolean, default=False)
	usage_count = Column(Integer, default=0)
	
	# Document Template
	document_template = Column(String(100), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	template_lines = relationship("SOQQuoteTemplateLine", back_populates="template", cascade="all, delete-orphan")
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'template_name', name='uq_soq_template_name_tenant'),
	)
	
	def __repr__(self):
		return f"<SOQQuoteTemplate {self.template_name}>"


class SOQQuoteTemplateLine(Model, AuditMixin, BaseMixin):
	"""
	Template line items for quotation templates.
	
	Defines standard line items and pricing for
	quotation templates.
	"""
	__tablename__ = 'so_q_quote_template_line'
	
	# Identity
	template_line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	template_id = Column(String(36), ForeignKey('so_q_quote_template.template_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	line_type = Column(String(20), default='PRODUCT')
	description = Column(Text, nullable=True)
	
	# Item Information
	item_id = Column(String(36), nullable=True)
	item_code = Column(String(50), nullable=True)
	item_description = Column(String(200), nullable=True)
	
	# Default Values
	default_quantity = Column(DECIMAL(12, 4), default=1.0000)
	default_unit_price = Column(DECIMAL(15, 4), default=0.0000)
	default_discount_percentage = Column(DECIMAL(5, 2), default=0.00)
	
	# Configuration
	is_required = Column(Boolean, default=False)
	is_optional = Column(Boolean, default=False)
	allow_quantity_change = Column(Boolean, default=True)
	allow_price_change = Column(Boolean, default=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	template = relationship("SOQQuoteTemplate", back_populates="template_lines")
	
	def __repr__(self):
		return f"<SOQQuoteTemplateLine {self.line_number}: {self.item_code}>"