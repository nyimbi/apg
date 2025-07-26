"""
Vendor Management Models

Database models for vendor master data, contacts, performance, and qualifications.
"""

from datetime import datetime, date
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class PPVVendor(Model, AuditMixin, BaseMixin):
	"""Vendor master data"""
	__tablename__ = 'ppv_vendor'
	
	# Identity
	vendor_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Vendor Information
	vendor_number = Column(String(20), nullable=False, index=True)
	vendor_name = Column(String(200), nullable=False, index=True)
	vendor_type = Column(String(50), default='SUPPLIER')
	legal_name = Column(String(200), nullable=True)
	dba_name = Column(String(200), nullable=True)
	
	# Business Information
	tax_id = Column(String(50), nullable=True)
	duns_number = Column(String(20), nullable=True)
	website = Column(String(200), nullable=True)
	industry = Column(String(100), nullable=True)
	business_size = Column(String(50), nullable=True)  # Small, Medium, Large, Enterprise
	
	# Address Information
	address_line1 = Column(String(100), nullable=True)
	address_line2 = Column(String(100), nullable=True)
	city = Column(String(50), nullable=True)
	state_province = Column(String(50), nullable=True)
	postal_code = Column(String(20), nullable=True)
	country = Column(String(50), nullable=True)
	
	# Financial Information
	credit_rating = Column(String(20), nullable=True)
	credit_limit = Column(DECIMAL(15, 2), default=0.00)
	payment_terms = Column(String(50), nullable=True)
	currency_code = Column(String(3), default='USD')
	
	# Status and Configuration
	is_active = Column(Boolean, default=True)
	is_approved = Column(Boolean, default=False)
	is_preferred = Column(Boolean, default=False)
	is_minority_owned = Column(Boolean, default=False)
	is_woman_owned = Column(Boolean, default=False)
	is_veteran_owned = Column(Boolean, default=False)
	
	# Performance Metrics
	overall_rating = Column(DECIMAL(3, 2), default=0.00)  # 0.00 to 5.00
	quality_rating = Column(DECIMAL(3, 2), default=0.00)
	delivery_rating = Column(DECIMAL(3, 2), default=0.00)
	service_rating = Column(DECIMAL(3, 2), default=0.00)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'vendor_number', name='uq_vendor_number_tenant'),
	)
	
	# Relationships
	contacts = relationship("PPVVendorContact", back_populates="vendor", cascade="all, delete-orphan")
	performance_records = relationship("PPVVendorPerformance", back_populates="vendor")
	qualifications = relationship("PPVVendorQualification", back_populates="vendor")
	insurance_records = relationship("PPVVendorInsurance", back_populates="vendor")
	
	def __repr__(self):
		return f"<PPVVendor {self.vendor_number} - {self.vendor_name}>"


class PPVVendorContact(Model, AuditMixin, BaseMixin):
	"""Vendor contact information"""
	__tablename__ = 'ppv_vendor_contact'
	
	# Identity
	contact_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	vendor_id = Column(String(36), ForeignKey('ppv_vendor.vendor_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Contact Information
	contact_type = Column(String(50), default='Primary')  # Primary, Billing, Technical, etc.
	first_name = Column(String(50), nullable=False)
	last_name = Column(String(50), nullable=False)
	title = Column(String(100), nullable=True)
	department = Column(String(100), nullable=True)
	
	# Communication
	email = Column(String(100), nullable=True)
	phone = Column(String(50), nullable=True)
	mobile = Column(String(50), nullable=True)
	fax = Column(String(50), nullable=True)
	
	# Status
	is_primary = Column(Boolean, default=False)
	is_active = Column(Boolean, default=True)
	
	# Relationships
	vendor = relationship("PPVVendor", back_populates="contacts")
	
	def __repr__(self):
		return f"<PPVVendorContact {self.first_name} {self.last_name} - {self.contact_type}>"


class PPVVendorPerformance(Model, AuditMixin, BaseMixin):
	"""Vendor performance tracking"""
	__tablename__ = 'ppv_vendor_performance'
	
	# Identity
	performance_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	vendor_id = Column(String(36), ForeignKey('ppv_vendor.vendor_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Performance Period
	period_start = Column(Date, nullable=False)
	period_end = Column(Date, nullable=False)
	period_type = Column(String(20), default='Monthly')  # Monthly, Quarterly, Annual
	
	# Performance Metrics
	total_orders = Column(Integer, default=0)
	total_value = Column(DECIMAL(15, 2), default=0.00)
	on_time_deliveries = Column(Integer, default=0)
	late_deliveries = Column(Integer, default=0)
	quality_issues = Column(Integer, default=0)
	price_competitiveness = Column(DECIMAL(3, 2), default=0.00)
	
	# Calculated Ratings
	delivery_performance = Column(DECIMAL(5, 2), default=0.00)  # Percentage
	quality_performance = Column(DECIMAL(5, 2), default=0.00)  # Percentage
	overall_score = Column(DECIMAL(3, 2), default=0.00)  # 0.00 to 5.00
	
	# Comments
	performance_notes = Column(Text, nullable=True)
	
	# Relationships
	vendor = relationship("PPVVendor", back_populates="performance_records")
	
	def __repr__(self):
		return f"<PPVVendorPerformance {self.vendor.vendor_name} - {self.period_start} to {self.period_end}>"


class PPVVendorCategory(Model, AuditMixin, BaseMixin):
	"""Vendor categories and capabilities"""
	__tablename__ = 'ppv_vendor_category'
	
	# Identity
	category_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Category Information
	category_code = Column(String(20), nullable=False)
	category_name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	parent_category_id = Column(String(36), nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	requires_qualification = Column(Boolean, default=False)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'category_code', name='uq_category_code_tenant'),
	)
	
	def __repr__(self):
		return f"<PPVVendorCategory {self.category_code} - {self.category_name}>"


class PPVVendorQualification(Model, AuditMixin, BaseMixin):
	"""Vendor qualifications and certifications"""
	__tablename__ = 'ppv_vendor_qualification'
	
	# Identity
	qualification_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	vendor_id = Column(String(36), ForeignKey('ppv_vendor.vendor_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Qualification Information
	qualification_type = Column(String(50), nullable=False)  # Certification, License, etc.
	qualification_name = Column(String(200), nullable=False)
	issuing_authority = Column(String(200), nullable=True)
	certificate_number = Column(String(100), nullable=True)
	
	# Validity
	issue_date = Column(Date, nullable=True)
	expiration_date = Column(Date, nullable=True)
	is_active = Column(Boolean, default=True)
	
	# Document Management
	document_path = Column(String(500), nullable=True)
	
	# Relationships
	vendor = relationship("PPVVendor", back_populates="qualifications")
	
	def __repr__(self):
		return f"<PPVVendorQualification {self.qualification_name}>"


class PPVVendorInsurance(Model, AuditMixin, BaseMixin):
	"""Vendor insurance information"""
	__tablename__ = 'ppv_vendor_insurance'
	
	# Identity
	insurance_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	vendor_id = Column(String(36), ForeignKey('ppv_vendor.vendor_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Insurance Information
	insurance_type = Column(String(50), nullable=False)  # General Liability, Professional, etc.
	insurance_company = Column(String(200), nullable=False)
	policy_number = Column(String(100), nullable=False)
	
	# Coverage
	coverage_amount = Column(DECIMAL(15, 2), default=0.00)
	deductible_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Validity
	effective_date = Column(Date, nullable=False)
	expiration_date = Column(Date, nullable=False)
	is_active = Column(Boolean, default=True)
	
	# Document Management
	certificate_path = Column(String(500), nullable=True)
	
	# Relationships
	vendor = relationship("PPVVendor", back_populates="insurance_records")
	
	def __repr__(self):
		return f"<PPVVendorInsurance {self.insurance_type} - {self.policy_number}>"
