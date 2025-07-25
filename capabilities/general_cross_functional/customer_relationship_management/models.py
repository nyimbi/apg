"""
Customer Relationship Management Data Models

Comprehensive models for customer management, sales pipeline, and service operations.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List
from uuid_extensions import uuid7str
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, Numeric, Date, JSON, ForeignKey, Index, CheckConstraint
from sqlalchemy.orm import relationship, Mapped
from pydantic import BaseModel, Field, validator, ConfigDict
from flask_appbuilder import Model

class GCCRCustomer(Model):
	"""Customer master record"""
	__tablename__ = 'gc_cr_customer'
	
	# Primary fields
	id: Mapped[str] = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: Mapped[str] = Column(String(50), nullable=False, index=True)
	
	# Customer identification
	customer_number: Mapped[str] = Column(String(50), nullable=False, unique=True, index=True)
	customer_name: Mapped[str] = Column(String(200), nullable=False, index=True)
	customer_type: Mapped[str] = Column(String(20), nullable=False, default='prospect')  # prospect, customer, partner
	
	# Company information
	company_name: Mapped[str] = Column(String(200), nullable=True)
	industry: Mapped[str] = Column(String(100), nullable=True)
	annual_revenue: Mapped[Decimal] = Column(Numeric(15, 2), nullable=True)
	employee_count: Mapped[int] = Column(Integer, nullable=True)
	
	# Contact information
	primary_email: Mapped[str] = Column(String(255), nullable=True, index=True)
	primary_phone: Mapped[str] = Column(String(50), nullable=True)
	website: Mapped[str] = Column(String(255), nullable=True)
	
	# Address information
	billing_address: Mapped[dict] = Column(JSON, nullable=True)
	shipping_address: Mapped[dict] = Column(JSON, nullable=True)
	
	# Customer classification
	customer_segment: Mapped[str] = Column(String(50), nullable=True, index=True)
	credit_rating: Mapped[str] = Column(String(20), nullable=True)
	payment_terms: Mapped[str] = Column(String(50), nullable=True, default='net_30')
	
	# Relationship management
	account_manager_id: Mapped[str] = Column(String(50), nullable=True, index=True)
	customer_since: Mapped[date] = Column(Date, nullable=True)
	last_activity_date: Mapped[date] = Column(Date, nullable=True)
	
	# Customer value metrics
	lifetime_value: Mapped[Decimal] = Column(Numeric(15, 2), nullable=True)
	total_revenue: Mapped[Decimal] = Column(Numeric(15, 2), nullable=True, default=0)
	total_orders: Mapped[int] = Column(Integer, nullable=False, default=0)
	
	# Status and preferences
	status: Mapped[str] = Column(String(20), nullable=False, default='active')  # active, inactive, suspended
	is_vip: Mapped[bool] = Column(Boolean, nullable=False, default=False)
	marketing_opt_in: Mapped[bool] = Column(Boolean, nullable=False, default=True)
	communication_preferences: Mapped[dict] = Column(JSON, nullable=True)
	
	# Audit fields
	created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow)
	created_by: Mapped[str] = Column(String(100), nullable=False)
	updated_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	updated_by: Mapped[str] = Column(String(100), nullable=False)
	
	# Relationships
	contacts = relationship("GCCRContact", back_populates="customer")
	opportunities = relationship("GCCROpportunity", back_populates="customer")
	activities = relationship("GCCRActivity", back_populates="customer")
	service_cases = relationship("GCCRServiceCase", back_populates="customer")
	
	# Indexes and constraints
	__table_args__ = (
		Index('ix_gc_cr_customer_tenant_number', 'tenant_id', 'customer_number'),
		Index('ix_gc_cr_customer_segment_status', 'customer_segment', 'status'),
		CheckConstraint('total_revenue >= 0'),
		CheckConstraint('total_orders >= 0'),
	)

class GCCRContact(Model):
	"""Customer contact persons"""
	__tablename__ = 'gc_cr_contact'
	
	# Primary fields
	id: Mapped[str] = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: Mapped[str] = Column(String(50), nullable=False, index=True)
	
	# Contact identification
	customer_id: Mapped[str] = Column(String(50), ForeignKey('gc_cr_customer.id'), nullable=False, index=True)
	contact_number: Mapped[str] = Column(String(50), nullable=False, index=True)
	
	# Personal information
	first_name: Mapped[str] = Column(String(100), nullable=False)
	last_name: Mapped[str] = Column(String(100), nullable=False)
	full_name: Mapped[str] = Column(String(200), nullable=False, index=True)  # computed
	title: Mapped[str] = Column(String(100), nullable=True)
	department: Mapped[str] = Column(String(100), nullable=True)
	
	# Contact information
	email: Mapped[str] = Column(String(255), nullable=True, index=True)
	phone: Mapped[str] = Column(String(50), nullable=True)
	mobile: Mapped[str] = Column(String(50), nullable=True)
	linkedin_url: Mapped[str] = Column(String(255), nullable=True)
	
	# Role and influence
	is_primary_contact: Mapped[bool] = Column(Boolean, nullable=False, default=False)
	is_decision_maker: Mapped[bool] = Column(Boolean, nullable=False, default=False)
	influence_level: Mapped[str] = Column(String(20), nullable=True)  # low, medium, high
	
	# Communication preferences
	preferred_contact_method: Mapped[str] = Column(String(20), nullable=True, default='email')
	time_zone: Mapped[str] = Column(String(50), nullable=True)
	language: Mapped[str] = Column(String(10), nullable=True, default='en')
	
	# Status
	status: Mapped[str] = Column(String(20), nullable=False, default='active')  # active, inactive
	
	# Audit fields
	created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow)
	created_by: Mapped[str] = Column(String(100), nullable=False)
	updated_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	updated_by: Mapped[str] = Column(String(100), nullable=False)
	
	# Relationships
	customer = relationship("GCCRCustomer", back_populates="contacts")
	activities = relationship("GCCRActivity", back_populates="contact")
	
	# Indexes and constraints
	__table_args__ = (
		Index('ix_gc_cr_contact_tenant_customer', 'tenant_id', 'customer_id'),
		Index('ix_gc_cr_contact_name', 'full_name'),
	)

class GCCRLead(Model):
	"""Sales leads and prospects"""
	__tablename__ = 'gc_cr_lead'
	
	# Primary fields
	id: Mapped[str] = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: Mapped[str] = Column(String(50), nullable=False, index=True)
	
	# Lead identification
	lead_number: Mapped[str] = Column(String(50), nullable=False, unique=True, index=True)
	lead_source: Mapped[str] = Column(String(100), nullable=False, index=True)
	
	# Contact information
	first_name: Mapped[str] = Column(String(100), nullable=False)
	last_name: Mapped[str] = Column(String(100), nullable=False)
	company_name: Mapped[str] = Column(String(200), nullable=True)
	title: Mapped[str] = Column(String(100), nullable=True)
	email: Mapped[str] = Column(String(255), nullable=True, index=True)
	phone: Mapped[str] = Column(String(50), nullable=True)
	
	# Lead qualification
	lead_status: Mapped[str] = Column(String(20), nullable=False, default='new', index=True)
	lead_score: Mapped[int] = Column(Integer, nullable=True, default=0)
	qualification_notes: Mapped[str] = Column(Text, nullable=True)
	
	# Interest and requirements
	product_interest: Mapped[str] = Column(String(200), nullable=True)
	budget_range: Mapped[str] = Column(String(50), nullable=True)
	timeline: Mapped[str] = Column(String(50), nullable=True)
	requirements: Mapped[str] = Column(Text, nullable=True)
	
	# Assignment and ownership
	assigned_to: Mapped[str] = Column(String(50), nullable=True, index=True)
	assigned_date: Mapped[datetime] = Column(DateTime, nullable=True)
	
	# Conversion tracking
	converted_to_opportunity: Mapped[bool] = Column(Boolean, nullable=False, default=False)
	opportunity_id: Mapped[str] = Column(String(50), nullable=True)
	converted_date: Mapped[datetime] = Column(DateTime, nullable=True)
	
	# Audit fields
	created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow)
	created_by: Mapped[str] = Column(String(100), nullable=False)
	updated_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	updated_by: Mapped[str] = Column(String(100), nullable=False)
	
	# Indexes and constraints
	__table_args__ = (
		Index('ix_gc_cr_lead_tenant_status', 'tenant_id', 'lead_status'),
		Index('ix_gc_cr_lead_source_assigned', 'lead_source', 'assigned_to'),
		CheckConstraint('lead_score >= 0 AND lead_score <= 100'),
	)

class GCCROpportunity(Model):
	"""Sales opportunities and pipeline"""
	__tablename__ = 'gc_cr_opportunity'
	
	# Primary fields
	id: Mapped[str] = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: Mapped[str] = Column(String(50), nullable=False, index=True)
	
	# Opportunity identification
	opportunity_number: Mapped[str] = Column(String(50), nullable=False, unique=True, index=True)
	opportunity_name: Mapped[str] = Column(String(200), nullable=False, index=True)
	customer_id: Mapped[str] = Column(String(50), ForeignKey('gc_cr_customer.id'), nullable=False, index=True)
	
	# Opportunity details
	description: Mapped[str] = Column(Text, nullable=True)
	opportunity_type: Mapped[str] = Column(String(50), nullable=True)  # new_business, upsell, renewal
	lead_source: Mapped[str] = Column(String(100), nullable=True)
	
	# Financial information
	estimated_value: Mapped[Decimal] = Column(Numeric(15, 2), nullable=False)
	probability: Mapped[int] = Column(Integer, nullable=False, default=50)  # 0-100%
	weighted_value: Mapped[Decimal] = Column(Numeric(15, 2), nullable=True)  # computed
	
	# Timeline
	created_date: Mapped[date] = Column(Date, nullable=False, default=date.today)
	expected_close_date: Mapped[date] = Column(Date, nullable=False, index=True)
	actual_close_date: Mapped[date] = Column(Date, nullable=True)
	
	# Sales process
	sales_stage: Mapped[str] = Column(String(50), nullable=False, default='prospect', index=True)
	next_step: Mapped[str] = Column(String(200), nullable=True)
	next_step_date: Mapped[date] = Column(Date, nullable=True)
	
	# Assignment
	owner_id: Mapped[str] = Column(String(50), nullable=False, index=True)
	sales_team: Mapped[str] = Column(String(100), nullable=True)
	
	# Competition
	competitors: Mapped[list] = Column(JSON, nullable=True)
	competitive_position: Mapped[str] = Column(String(20), nullable=True)  # strong, weak, unknown
	
	# Outcome
	status: Mapped[str] = Column(String(20), nullable=False, default='open', index=True)  # open, won, lost
	close_reason: Mapped[str] = Column(String(200), nullable=True)
	win_loss_reason: Mapped[str] = Column(Text, nullable=True)
	
	# Audit fields
	created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow)
	created_by: Mapped[str] = Column(String(100), nullable=False)
	updated_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	updated_by: Mapped[str] = Column(String(100), nullable=False)
	
	# Relationships
	customer = relationship("GCCRCustomer", back_populates="opportunities")
	activities = relationship("GCCRActivity", back_populates="opportunity")
	
	# Indexes and constraints
	__table_args__ = (
		Index('ix_gc_cr_opportunity_tenant_stage', 'tenant_id', 'sales_stage'),
		Index('ix_gc_cr_opportunity_owner_status', 'owner_id', 'status'),
		Index('ix_gc_cr_opportunity_close_date', 'expected_close_date'),
		CheckConstraint('probability >= 0 AND probability <= 100'),
		CheckConstraint('estimated_value >= 0'),
	)

class GCCRActivity(Model):
	"""Customer activities and interactions"""
	__tablename__ = 'gc_cr_activity'
	
	# Primary fields
	id: Mapped[str] = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: Mapped[str] = Column(String(50), nullable=False, index=True)
	
	# Activity identification
	activity_type: Mapped[str] = Column(String(50), nullable=False, index=True)  # call, email, meeting, task
	activity_date: Mapped[datetime] = Column(DateTime, nullable=False, index=True)
	subject: Mapped[str] = Column(String(200), nullable=False)
	description: Mapped[str] = Column(Text, nullable=True)
	
	# Related records
	customer_id: Mapped[str] = Column(String(50), ForeignKey('gc_cr_customer.id'), nullable=True, index=True)
	contact_id: Mapped[str] = Column(String(50), ForeignKey('gc_cr_contact.id'), nullable=True, index=True)
	opportunity_id: Mapped[str] = Column(String(50), ForeignKey('gc_cr_opportunity.id'), nullable=True, index=True)
	
	# Activity details
	direction: Mapped[str] = Column(String(20), nullable=True)  # inbound, outbound
	outcome: Mapped[str] = Column(String(100), nullable=True)
	duration_minutes: Mapped[int] = Column(Integer, nullable=True)
	
	# Assignment and status
	assigned_to: Mapped[str] = Column(String(50), nullable=False, index=True)
	status: Mapped[str] = Column(String(20), nullable=False, default='completed')  # planned, completed, cancelled
	priority: Mapped[str] = Column(String(20), nullable=False, default='medium')  # low, medium, high
	
	# Follow-up
	due_date: Mapped[datetime] = Column(DateTime, nullable=True, index=True)
	follow_up_required: Mapped[bool] = Column(Boolean, nullable=False, default=False)
	follow_up_date: Mapped[datetime] = Column(DateTime, nullable=True)
	
	# Audit fields
	created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow)
	created_by: Mapped[str] = Column(String(100), nullable=False)
	updated_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	updated_by: Mapped[str] = Column(String(100), nullable=False)
	
	# Relationships
	customer = relationship("GCCRCustomer", back_populates="activities")
	contact = relationship("GCCRContact", back_populates="activities")
	opportunity = relationship("GCCROpportunity", back_populates="activities")
	
	# Indexes and constraints
	__table_args__ = (
		Index('ix_gc_cr_activity_tenant_date', 'tenant_id', 'activity_date'),
		Index('ix_gc_cr_activity_assigned_status', 'assigned_to', 'status'),
		Index('ix_gc_cr_activity_customer_type', 'customer_id', 'activity_type'),
	)

class GCCRServiceCase(Model):
	"""Customer service cases and support tickets"""
	__tablename__ = 'gc_cr_service_case'
	
	# Primary fields
	id: Mapped[str] = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: Mapped[str] = Column(String(50), nullable=False, index=True)
	
	# Case identification
	case_number: Mapped[str] = Column(String(50), nullable=False, unique=True, index=True)
	customer_id: Mapped[str] = Column(String(50), ForeignKey('gc_cr_customer.id'), nullable=False, index=True)
	contact_id: Mapped[str] = Column(String(50), ForeignKey('gc_cr_contact.id'), nullable=True, index=True)
	
	# Case details
	subject: Mapped[str] = Column(String(200), nullable=False)
	description: Mapped[str] = Column(Text, nullable=True)
	case_type: Mapped[str] = Column(String(50), nullable=False, index=True)  # incident, request, question
	category: Mapped[str] = Column(String(100), nullable=True, index=True)
	subcategory: Mapped[str] = Column(String(100), nullable=True)
	
	# Priority and severity
	priority: Mapped[str] = Column(String(20), nullable=False, default='medium', index=True)  # low, medium, high, critical
	severity: Mapped[str] = Column(String(20), nullable=True)  # low, medium, high, critical
	impact: Mapped[str] = Column(String(20), nullable=True)    # low, medium, high
	
	# Status and assignment
	status: Mapped[str] = Column(String(20), nullable=False, default='new', index=True)  # new, open, pending, resolved, closed
	assigned_to: Mapped[str] = Column(String(50), nullable=True, index=True)
	assigned_team: Mapped[str] = Column(String(100), nullable=True, index=True)
	
	# Timeline
	created_date: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow)
	first_response_date: Mapped[datetime] = Column(DateTime, nullable=True)
	resolution_date: Mapped[datetime] = Column(DateTime, nullable=True)
	closed_date: Mapped[datetime] = Column(DateTime, nullable=True)
	
	# SLA tracking
	sla_response_due: Mapped[datetime] = Column(DateTime, nullable=True, index=True)
	sla_resolution_due: Mapped[datetime] = Column(DateTime, nullable=True, index=True)
	sla_breached: Mapped[bool] = Column(Boolean, nullable=False, default=False)
	
	# Resolution
	resolution: Mapped[str] = Column(Text, nullable=True)
	resolution_code: Mapped[str] = Column(String(50), nullable=True)
	
	# Customer satisfaction
	satisfaction_rating: Mapped[int] = Column(Integer, nullable=True)  # 1-5 scale
	satisfaction_comments: Mapped[str] = Column(Text, nullable=True)
	
	# Audit fields
	created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow)
	created_by: Mapped[str] = Column(String(100), nullable=False)
	updated_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	updated_by: Mapped[str] = Column(String(100), nullable=False)
	
	# Relationships
	customer = relationship("GCCRCustomer", back_populates="service_cases")
	contact = relationship("GCCRContact")
	
	# Indexes and constraints
	__table_args__ = (
		Index('ix_gc_cr_service_case_tenant_status', 'tenant_id', 'status'),
		Index('ix_gc_cr_service_case_priority_assigned', 'priority', 'assigned_to'),
		Index('ix_gc_cr_service_case_sla_due', 'sla_response_due', 'sla_resolution_due'),
		CheckConstraint('satisfaction_rating >= 1 AND satisfaction_rating <= 5'),
	)

# Pydantic models for API
class CustomerCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	customer_number: str = Field(..., min_length=1, max_length=50)
	customer_name: str = Field(..., min_length=1, max_length=200)
	customer_type: str = Field(default='prospect')
	company_name: str | None = Field(None, max_length=200)
	industry: str | None = Field(None, max_length=100)
	primary_email: str | None = Field(None, max_length=255)
	primary_phone: str | None = Field(None, max_length=50)
	customer_segment: str | None = Field(None, max_length=50)

class OpportunityCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	opportunity_name: str = Field(..., min_length=1, max_length=200)
	customer_id: str = Field(..., min_length=1)
	estimated_value: Decimal = Field(..., ge=0)
	probability: int = Field(default=50, ge=0, le=100)
	expected_close_date: date = Field(...)
	sales_stage: str = Field(default='prospect')
	owner_id: str = Field(..., min_length=1)

class ServiceCaseCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	customer_id: str = Field(..., min_length=1)
	contact_id: str | None = Field(None)
	subject: str = Field(..., min_length=1, max_length=200)
	description: str | None = Field(None)
	case_type: str = Field(..., min_length=1, max_length=50)
	priority: str = Field(default='medium')
	category: str | None = Field(None, max_length=100)