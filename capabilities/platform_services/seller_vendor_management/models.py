"""
Seller/Vendor Management Models

Database models for managing multi-vendor marketplace operations including
vendor onboarding, verification, performance tracking, and payouts.
"""

from sqlalchemy import Column, String, Text, Boolean, Integer, DateTime, Numeric, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum

Base = declarative_base()

class VendorStatus(str, Enum):
	PENDING = "pending"
	UNDER_REVIEW = "under_review"
	APPROVED = "approved"
	ACTIVE = "active"
	SUSPENDED = "suspended"
	REJECTED = "rejected"
	DEACTIVATED = "deactivated"

class VerificationStatus(str, Enum):
	NOT_VERIFIED = "not_verified"
	PENDING_VERIFICATION = "pending_verification"
	VERIFIED = "verified"
	FAILED_VERIFICATION = "failed_verification"

class PayoutStatus(str, Enum):
	PENDING = "pending"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"

class PerformanceRating(str, Enum):
	EXCELLENT = "excellent"
	GOOD = "good"
	AVERAGE = "average"
	POOR = "poor"
	CRITICAL = "critical"

# SQLAlchemy Models
class PSVendor(Base):
	__tablename__ = 'ps_vendors'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	
	# Basic Information
	business_name = Column(String(255), nullable=False)
	legal_name = Column(String(255))
	display_name = Column(String(255))
	slug = Column(String(255), nullable=False, unique=True)
	
	# Contact Information
	email = Column(String(255), nullable=False)
	phone = Column(String(50))
	website = Column(String(500))
	
	# Business Details
	business_type = Column(String(100))  # corporation, llc, sole_proprietorship, etc.
	tax_id = Column(String(100))
	registration_number = Column(String(100))
	industry = Column(String(100))
	
	# Address
	address_line1 = Column(String(255))
	address_line2 = Column(String(255))
	city = Column(String(100))
	state = Column(String(100))
	postal_code = Column(String(20))
	country = Column(String(3))  # ISO country code
	
	# Status and Verification
	status = Column(String(20), nullable=False, default=VendorStatus.PENDING.value)
	verification_status = Column(String(30), nullable=False, default=VerificationStatus.NOT_VERIFIED.value)
	is_featured = Column(Boolean, default=False)
	is_trusted = Column(Boolean, default=False)
	
	# Store Configuration
	store_name = Column(String(255))
	store_description = Column(Text)
	store_logo_url = Column(String(500))
	store_banner_url = Column(String(500))
	store_theme = Column(String(100))
	
	# Business Hours
	business_hours = Column(JSON)  # {"monday": {"open": "09:00", "close": "17:00"}, ...}
	timezone = Column(String(50), default='UTC')
	
	# Financial Information
	commission_rate = Column(Numeric(5, 2), default=15.00)  # Percentage
	payment_terms = Column(String(100), default='net_30')
	minimum_payout = Column(Numeric(10, 2), default=50.00)
	
	# Performance Metrics
	total_sales = Column(Numeric(15, 2), default=0.00)
	total_orders = Column(Integer, default=0)
	total_products = Column(Integer, default=0)
	average_rating = Column(Numeric(3, 2), default=0.00)
	rating_count = Column(Integer, default=0)
	performance_score = Column(Numeric(5, 2), default=0.00)
	
	# Settings
	auto_approve_products = Column(Boolean, default=False)
	allow_cod = Column(Boolean, default=True)
	shipping_policy = Column(Text)
	return_policy = Column(Text)
	
	# Onboarding
	onboarding_completed = Column(Boolean, default=False)
	onboarding_step = Column(String(50))
	welcome_email_sent = Column(Boolean, default=False)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	approved_at = Column(DateTime)
	last_active_at = Column(DateTime)
	created_by = Column(String(36))
	updated_by = Column(String(36))
	
	# Relationships
	profile = relationship("PSVendorProfile", back_populates="vendor", uselist=False, cascade="all, delete-orphan")
	verifications = relationship("PSVendorVerification", back_populates="vendor", cascade="all, delete-orphan")
	contracts = relationship("PSVendorContract", back_populates="vendor", cascade="all, delete-orphan")
	performance = relationship("PSVendorPerformance", back_populates="vendor", cascade="all, delete-orphan")
	payouts = relationship("PSVendorPayout", back_populates="vendor", cascade="all, delete-orphan")
	products = relationship("PSVendorProduct", back_populates="vendor", cascade="all, delete-orphan")
	stores = relationship("PSVendorStore", back_populates="vendor", cascade="all, delete-orphan")
	documents = relationship("PSVendorDocument", back_populates="vendor", cascade="all, delete-orphan")
	communications = relationship("PSVendorCommunication", back_populates="vendor", cascade="all, delete-orphan")
	
	# Indexes
	__table_args__ = (
		Index('idx_vendor_tenant_email', 'tenant_id', 'email', unique=True),
		Index('idx_vendor_slug', 'slug', unique=True),
		Index('idx_vendor_status', 'status'),
		Index('idx_vendor_verification', 'verification_status'),
		Index('idx_vendor_featured', 'is_featured'),
	)

class PSVendorProfile(Base):
	__tablename__ = 'ps_vendor_profiles'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	vendor_id = Column(String(36), ForeignKey('ps_vendors.id'), nullable=False)
	
	# Extended Profile Information
	about = Column(Text)
	specialties = Column(JSON)  # List of specialties
	certifications = Column(JSON)  # List of certifications
	awards = Column(JSON)  # List of awards
	
	# Social Media
	facebook_url = Column(String(500))
	twitter_url = Column(String(500))
	instagram_url = Column(String(500))
	linkedin_url = Column(String(500))
	
	# Additional Contact
	support_email = Column(String(255))
	support_phone = Column(String(50))
	
	# Marketing
	seo_title = Column(String(255))
	seo_description = Column(Text)
	seo_keywords = Column(Text)
	
	# Preferences
	notification_preferences = Column(JSON)
	privacy_settings = Column(JSON)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	# Relationships
	vendor = relationship("PSVendor", back_populates="profile")

class PSVendorVerification(Base):
	__tablename__ = 'ps_vendor_verifications'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	vendor_id = Column(String(36), ForeignKey('ps_vendors.id'), nullable=False)
	
	# Verification Information
	verification_type = Column(String(50), nullable=False)  # identity, business, tax, bank, etc.
	status = Column(String(30), nullable=False, default=VerificationStatus.PENDING_VERIFICATION.value)
	
	# Submitted Data
	submitted_data = Column(JSON)  # Verification specific data
	documents = Column(JSON)  # List of document URLs
	
	# Review Information
	reviewed_by = Column(String(36))
	reviewed_at = Column(DateTime)
	review_notes = Column(Text)
	rejection_reason = Column(Text)
	
	# Verification Details
	verification_method = Column(String(100))  # manual, automated, third_party
	external_reference = Column(String(255))  # Third-party verification ID
	confidence_score = Column(Numeric(5, 2))  # For automated verifications
	
	# Expiry (for time-limited verifications)
	expires_at = Column(DateTime)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	# Relationships
	vendor = relationship("PSVendor", back_populates="verifications")

class PSVendorContract(Base):
	__tablename__ = 'ps_vendor_contracts'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	vendor_id = Column(String(36), ForeignKey('ps_vendors.id'), nullable=False)
	
	# Contract Information
	contract_type = Column(String(50), nullable=False)  # standard, custom, enterprise
	title = Column(String(255), nullable=False)
	version = Column(String(20), default='1.0')
	
	# Content
	terms_text = Column(Text)
	terms_url = Column(String(500))  # Link to contract document
	
	# Agreement
	agreed_by_vendor = Column(Boolean, default=False)
	agreed_at = Column(DateTime)
	vendor_signature = Column(String(500))  # Digital signature
	vendor_ip_address = Column(String(45))
	
	# Platform Agreement
	agreed_by_platform = Column(Boolean, default=False)
	platform_agreed_at = Column(DateTime)
	platform_agreed_by = Column(String(36))
	
	# Status and Dates
	status = Column(String(20), default='draft')  # draft, active, expired, terminated
	effective_from = Column(DateTime)
	effective_until = Column(DateTime)
	
	# Terms
	commission_rate = Column(Numeric(5, 2))
	payment_terms = Column(String(100))
	termination_notice_days = Column(Integer, default=30)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36))
	
	# Relationships
	vendor = relationship("PSVendor", back_populates="contracts")

class PSVendorPerformance(Base):
	__tablename__ = 'ps_vendor_performance'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	vendor_id = Column(String(36), ForeignKey('ps_vendors.id'), nullable=False)
	
	# Performance Period
	period_start = Column(DateTime, nullable=False)
	period_end = Column(DateTime, nullable=False)
	period_type = Column(String(20), default='monthly')  # daily, weekly, monthly, quarterly
	
	# Sales Metrics
	total_sales = Column(Numeric(15, 2), default=0.00)
	total_orders = Column(Integer, default=0)
	average_order_value = Column(Numeric(10, 2), default=0.00)
	conversion_rate = Column(Numeric(5, 2), default=0.00)
	
	# Order Fulfillment
	orders_fulfilled = Column(Integer, default=0)
	orders_cancelled = Column(Integer, default=0)
	fulfillment_rate = Column(Numeric(5, 2), default=0.00)
	average_fulfillment_time = Column(Numeric(8, 2))  # Hours
	
	# Customer Satisfaction
	average_rating = Column(Numeric(3, 2), default=0.00)
	total_reviews = Column(Integer, default=0)
	positive_reviews = Column(Integer, default=0)
	negative_reviews = Column(Integer, default=0)
	
	# Return and Refund Metrics
	return_rate = Column(Numeric(5, 2), default=0.00)
	refund_rate = Column(Numeric(5, 2), default=0.00)
	dispute_rate = Column(Numeric(5, 2), default=0.00)
	
	# Product Metrics
	products_added = Column(Integer, default=0)
	products_updated = Column(Integer, default=0)
	out_of_stock_incidents = Column(Integer, default=0)
	
	# Communication Metrics
	response_time_hours = Column(Numeric(8, 2))  # Average response time
	messages_responded = Column(Integer, default=0)
	messages_total = Column(Integer, default=0)
	
	# Overall Performance
	performance_score = Column(Numeric(5, 2), default=0.00)  # Calculated score 0-100
	performance_rating = Column(String(20))  # excellent, good, average, poor, critical
	
	# Issues and Violations
	policy_violations = Column(Integer, default=0)
	warnings_issued = Column(Integer, default=0)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	calculated_at = Column(DateTime)
	
	# Relationships
	vendor = relationship("PSVendor", back_populates="performance")
	
	# Indexes
	__table_args__ = (
		Index('idx_performance_vendor_period', 'vendor_id', 'period_start', 'period_end'),
		Index('idx_performance_score', 'performance_score'),
	)

class PSVendorPayout(Base):
	__tablename__ = 'ps_vendor_payouts'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	vendor_id = Column(String(36), ForeignKey('ps_vendors.id'), nullable=False)
	
	# Payout Information
	payout_period_start = Column(DateTime, nullable=False)
	payout_period_end = Column(DateTime, nullable=False)
	
	# Financial Details
	gross_sales = Column(Numeric(15, 2), nullable=False)
	commission_amount = Column(Numeric(15, 2), nullable=False)
	platform_fees = Column(Numeric(15, 2), default=0.00)
	processing_fees = Column(Numeric(15, 2), default=0.00)
	adjustments = Column(Numeric(15, 2), default=0.00)
	net_amount = Column(Numeric(15, 2), nullable=False)
	currency = Column(String(3), default='USD')
	
	# Status and Processing
	status = Column(String(20), nullable=False, default=PayoutStatus.PENDING.value)
	payment_method = Column(String(50))  # bank_transfer, paypal, check, etc.
	payment_reference = Column(String(255))
	
	# Transaction Details
	orders_count = Column(Integer, default=0)
	returns_amount = Column(Numeric(15, 2), default=0.00)
	refunds_amount = Column(Numeric(15, 2), default=0.00)
	
	# Processing Information
	processed_by = Column(String(36))
	processed_at = Column(DateTime)
	scheduled_date = Column(DateTime)
	
	# Payment Details
	bank_account = Column(String(255))  # Encrypted/masked account info
	payment_notes = Column(Text)
	failure_reason = Column(Text)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	# Relationships
	vendor = relationship("PSVendor", back_populates="payouts")
	
	# Indexes
	__table_args__ = (
		Index('idx_payout_vendor_period', 'vendor_id', 'payout_period_start'),
		Index('idx_payout_status', 'status'),
		Index('idx_payout_scheduled', 'scheduled_date'),
	)

# Additional models would include:
# PSVendorProduct, PSVendorStore, PSVendorDocument, PSVendorCommunication

# Pydantic Models for API/Views
class VendorCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	business_name: str = Field(..., min_length=1, max_length=255)
	legal_name: str | None = Field(None, max_length=255)
	email: str = Field(..., min_length=1, max_length=255)
	phone: str | None = Field(None, max_length=50)
	business_type: str | None = Field(None, max_length=100)
	industry: str | None = Field(None, max_length=100)