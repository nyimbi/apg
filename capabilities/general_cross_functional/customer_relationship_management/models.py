"""
Customer Relationship Management Models

Comprehensive SQLAlchemy models for CRM including customers, contacts, leads,
opportunities, sales pipeline, activities, campaigns, cases, and analytics.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

from sqlalchemy import Column, String, Text, Integer, DateTime, Date, Boolean, Numeric, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.declarative import declarative_base
from flask_appbuilder import Model
from flask_appbuilder.models.mixins import AuditMixin
from uuid_extensions import uuid7str
from pydantic import Field, ConfigDict, field_validator, AfterValidator
from typing_extensions import Annotated

Base = declarative_base()

# Enums for CRM

class LeadSource(Enum):
	WEBSITE = "website"
	SOCIAL_MEDIA = "social_media"
	EMAIL_CAMPAIGN = "email_campaign"
	TRADE_SHOW = "trade_show"
	REFERRAL = "referral"
	COLD_CALL = "cold_call"
	ADVERTISEMENT = "advertisement"
	PARTNER = "partner"
	OTHER = "other"

class LeadStatus(Enum):
	NEW = "new"
	CONTACTED = "contacted"
	QUALIFIED = "qualified"
	UNQUALIFIED = "unqualified"
	CONVERTED = "converted"
	LOST = "lost"

class LeadRating(Enum):
	HOT = "hot"
	WARM = "warm"
	COLD = "cold"

class OpportunityStage(Enum):
	PROSPECTING = "prospecting"
	QUALIFICATION = "qualification"
	NEEDS_ANALYSIS = "needs_analysis"
	PROPOSAL = "proposal"
	NEGOTIATION = "negotiation"
	CLOSED_WON = "closed_won"
	CLOSED_LOST = "closed_lost"

class ActivityType(Enum):
	CALL = "call"
	EMAIL = "email"
	MEETING = "meeting"
	TASK = "task"
	NOTE = "note"
	DEMO = "demo"
	PROPOSAL = "proposal"
	FOLLOW_UP = "follow_up"

class ActivityPriority(Enum):
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	URGENT = "urgent"

class CampaignType(Enum):
	EMAIL_MARKETING = "email_marketing"
	DIRECT_MAIL = "direct_mail"
	TELEMARKETING = "telemarketing"
	WEB_CAMPAIGN = "web_campaign"
	SOCIAL_MEDIA = "social_media"
	TRADE_SHOW = "trade_show"
	ADVERTISEMENT = "advertisement"
	WEBINAR = "webinar"
	EVENT = "event"

class CampaignStatus(Enum):
	PLANNING = "planning"
	ACTIVE = "active"
	COMPLETED = "completed"
	CANCELLED = "cancelled"
	PAUSED = "paused"

class CaseType(Enum):
	QUESTION = "question"
	PROBLEM = "problem"
	FEATURE_REQUEST = "feature_request"
	COMPLAINT = "complaint"
	BUG_REPORT = "bug_report"
	SERVICE_REQUEST = "service_request"

class CasePriority(Enum):
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	CRITICAL = "critical"

class CaseStatus(Enum):
	NEW = "new"
	IN_PROGRESS = "in_progress"
	PENDING = "pending"
	RESOLVED = "resolved"
	CLOSED = "closed"
	CANCELLED = "cancelled"

class CustomerType(Enum):
	INDIVIDUAL = "individual"
	SMALL_BUSINESS = "small_business"
	MEDIUM_BUSINESS = "medium_business"
	ENTERPRISE = "enterprise"
	GOVERNMENT = "government"
	NON_PROFIT = "non_profit"

class CustomerStatus(Enum):
	ACTIVE = "active"
	INACTIVE = "inactive"
	PROSPECT = "prospect"
	FORMER_CUSTOMER = "former_customer"

class QuoteStatus(Enum):
	DRAFT = "draft"
	SENT = "sent"
	REVIEWED = "reviewed"
	ACCEPTED = "accepted"
	REJECTED = "rejected"
	EXPIRED = "expired"

# Core CRM Models

class GCCRMAccount(Model, AuditMixin):
	"""CRM Account - represents companies/organizations"""
	
	__tablename__ = 'gc_crm_account'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	account_name: str = Column(String(200), nullable=False)
	account_number: str = Column(String(50), unique=True, nullable=False)
	account_type: str = Column(String(50), default=CustomerType.SMALL_BUSINESS.value)
	parent_account_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_account.id'))
	
	# Contact Information
	phone: Optional[str] = Column(String(30))
	fax: Optional[str] = Column(String(30))
	email: Optional[str] = Column(String(100))
	website: Optional[str] = Column(String(200))
	
	# Address Information
	billing_street: Optional[str] = Column(Text)
	billing_city: Optional[str] = Column(String(100))
	billing_state: Optional[str] = Column(String(50))
	billing_postal_code: Optional[str] = Column(String(20))
	billing_country: Optional[str] = Column(String(50))
	
	shipping_street: Optional[str] = Column(Text)
	shipping_city: Optional[str] = Column(String(100))
	shipping_state: Optional[str] = Column(String(50))
	shipping_postal_code: Optional[str] = Column(String(20))
	shipping_country: Optional[str] = Column(String(50))
	
	# Business Information
	industry: Optional[str] = Column(String(100))
	annual_revenue: Optional[Decimal] = Column(Numeric(15, 2))
	number_of_employees: Optional[int] = Column(Integer)
	description: Optional[str] = Column(Text)
	
	# Relationship Management
	account_owner_id: Optional[str] = Column(String(50))
	account_status: str = Column(String(30), default=CustomerStatus.PROSPECT.value)
	customer_since: Optional[date] = Column(Date)
	last_activity_date: Optional[datetime] = Column(DateTime)
	
	# Financial Information
	credit_limit: Optional[Decimal] = Column(Numeric(15, 2))
	payment_terms: Optional[str] = Column(String(50))
	preferred_currency: str = Column(String(10), default='USD')
	
	# Territory and Assignment
	territory_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_territory.id'))
	team_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_team.id'))
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_customer: bool = Column(Boolean, default=False)
	is_partner: bool = Column(Boolean, default=False)
	is_competitor: bool = Column(Boolean, default=False)
	
	# Relationships
	parent_account = relationship("GCCRMAccount", remote_side=[id])
	child_accounts = relationship("GCCRMAccount", back_populates="parent_account")
	contacts = relationship("GCCRMContact", back_populates="account")
	opportunities = relationship("GCCRMOpportunity", back_populates="account")
	cases = relationship("GCCRMCase", back_populates="account")
	territory = relationship("GCCRMTerritory", back_populates="accounts")
	team = relationship("GCCRMTeam", back_populates="accounts")
	
	def __repr__(self):
		return f"<Account {self.account_name}>"

class GCCRMCustomer(Model, AuditMixin):
	"""CRM Customer - represents individual customers/contacts"""
	
	__tablename__ = 'gc_crm_customer'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	customer_number: str = Column(String(50), unique=True, nullable=False)
	customer_type: str = Column(String(50), default=CustomerType.INDIVIDUAL.value)
	customer_status: str = Column(String(30), default=CustomerStatus.PROSPECT.value)
	
	# Personal Information
	salutation: Optional[str] = Column(String(20))
	first_name: str = Column(String(100), nullable=False)
	last_name: str = Column(String(100), nullable=False)
	middle_name: Optional[str] = Column(String(100))
	preferred_name: Optional[str] = Column(String(100))
	suffix: Optional[str] = Column(String(20))
	
	# Contact Information
	email: Optional[str] = Column(String(100))
	phone: Optional[str] = Column(String(30))
	mobile: Optional[str] = Column(String(30))
	fax: Optional[str] = Column(String(30))
	
	# Address Information
	street_address: Optional[str] = Column(Text)
	city: Optional[str] = Column(String(100))
	state: Optional[str] = Column(String(50))
	postal_code: Optional[str] = Column(String(20))
	country: Optional[str] = Column(String(50))
	
	# Business Information
	job_title: Optional[str] = Column(String(100))
	department: Optional[str] = Column(String(100))
	company: Optional[str] = Column(String(200))
	
	# Personal Details
	birth_date: Optional[date] = Column(Date)
	preferred_language: Optional[str] = Column(String(20))
	preferred_contact_method: Optional[str] = Column(String(30))
	
	# Relationship Management
	customer_owner_id: Optional[str] = Column(String(50))
	customer_since: Optional[date] = Column(Date)
	last_activity_date: Optional[datetime] = Column(DateTime)
	last_contact_date: Optional[datetime] = Column(DateTime)
	
	# Financial Information
	credit_limit: Optional[Decimal] = Column(Numeric(15, 2))
	lifetime_value: Optional[Decimal] = Column(Numeric(15, 2))
	total_purchases: Optional[Decimal] = Column(Numeric(15, 2))
	
	# Marketing Preferences
	email_opt_in: bool = Column(Boolean, default=True)
	sms_opt_in: bool = Column(Boolean, default=False)
	phone_opt_in: bool = Column(Boolean, default=True)
	
	# Territory and Assignment
	territory_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_territory.id'))
	team_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_team.id'))
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_vip: bool = Column(Boolean, default=False)
	is_decision_maker: bool = Column(Boolean, default=False)
	
	# Relationships
	opportunities = relationship("GCCRMOpportunity", back_populates="customer")
	cases = relationship("GCCRMCase", back_populates="customer")
	activities = relationship("GCCRMActivity", back_populates="customer")
	territory = relationship("GCCRMTerritory", back_populates="customers")
	team = relationship("GCCRMTeam", back_populates="customers")
	
	@property
	def full_name(self) -> str:
		"""Get full name of customer"""
		parts = [self.first_name, self.middle_name, self.last_name]
		return ' '.join(filter(None, parts))
	
	def __repr__(self):
		return f"<Customer {self.full_name}>"

class GCCRMContact(Model, AuditMixin):
	"""CRM Contact - represents contacts within accounts"""
	
	__tablename__ = 'gc_crm_contact'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	account_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_account.id'))
	
	# Basic Information
	contact_number: str = Column(String(50), unique=True, nullable=False)
	salutation: Optional[str] = Column(String(20))
	first_name: str = Column(String(100), nullable=False)
	last_name: str = Column(String(100), nullable=False)
	middle_name: Optional[str] = Column(String(100))
	suffix: Optional[str] = Column(String(20))
	
	# Contact Information
	email: Optional[str] = Column(String(100))
	phone: Optional[str] = Column(String(30))
	mobile: Optional[str] = Column(String(30))
	fax: Optional[str] = Column(String(30))
	assistant_phone: Optional[str] = Column(String(30))
	
	# Business Information
	job_title: Optional[str] = Column(String(100))
	department: Optional[str] = Column(String(100))
	reports_to_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_contact.id'))
	assistant_name: Optional[str] = Column(String(100))
	
	# Personal Details
	birth_date: Optional[date] = Column(Date)
	preferred_language: Optional[str] = Column(String(20))
	preferred_contact_method: Optional[str] = Column(String(30))
	
	# Relationship Management
	contact_owner_id: Optional[str] = Column(String(50))
	last_activity_date: Optional[datetime] = Column(DateTime)
	last_contact_date: Optional[datetime] = Column(DateTime)
	
	# Marketing Preferences
	email_opt_in: bool = Column(Boolean, default=True)
	sms_opt_in: bool = Column(Boolean, default=False)
	phone_opt_in: bool = Column(Boolean, default=True)
	do_not_call: bool = Column(Boolean, default=False)
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_primary_contact: bool = Column(Boolean, default=False)
	is_decision_maker: bool = Column(Boolean, default=False)
	is_influencer: bool = Column(Boolean, default=False)
	
	# Relationships
	account = relationship("GCCRMAccount", back_populates="contacts")
	reports_to = relationship("GCCRMContact", remote_side=[id])
	subordinates = relationship("GCCRMContact", back_populates="reports_to")
	opportunities = relationship("GCCRMOpportunity", back_populates="primary_contact")
	activities = relationship("GCCRMActivity", back_populates="contact")
	
	@property
	def full_name(self) -> str:
		"""Get full name of contact"""
		parts = [self.first_name, self.middle_name, self.last_name]
		return ' '.join(filter(None, parts))
	
	def __repr__(self):
		return f"<Contact {self.full_name}>"

class GCCRMLead(Model, AuditMixin):
	"""CRM Lead - represents potential customers"""
	
	__tablename__ = 'gc_crm_lead'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	lead_number: str = Column(String(50), unique=True, nullable=False)
	lead_source: str = Column(String(50), default=LeadSource.WEBSITE.value)
	lead_status: str = Column(String(30), default=LeadStatus.NEW.value)
	lead_rating: str = Column(String(20), default=LeadRating.WARM.value)
	
	# Personal Information
	salutation: Optional[str] = Column(String(20))
	first_name: str = Column(String(100), nullable=False)
	last_name: str = Column(String(100), nullable=False)
	company: Optional[str] = Column(String(200))
	job_title: Optional[str] = Column(String(100))
	
	# Contact Information
	email: Optional[str] = Column(String(100))
	phone: Optional[str] = Column(String(30))
	mobile: Optional[str] = Column(String(30))
	website: Optional[str] = Column(String(200))
	
	# Address Information
	street: Optional[str] = Column(Text)
	city: Optional[str] = Column(String(100))
	state: Optional[str] = Column(String(50))
	postal_code: Optional[str] = Column(String(20))
	country: Optional[str] = Column(String(50))
	
	# Business Information
	industry: Optional[str] = Column(String(100))
	annual_revenue: Optional[Decimal] = Column(Numeric(15, 2))
	number_of_employees: Optional[int] = Column(Integer)
	
	# Lead Qualification
	budget: Optional[Decimal] = Column(Numeric(15, 2))
	timeline: Optional[str] = Column(String(100))
	decision_maker: Optional[str] = Column(String(200))
	pain_points: Optional[str] = Column(Text)
	current_solution: Optional[str] = Column(Text)
	
	# Lead Scoring
	lead_score: Optional[int] = Column(Integer, default=0)
	demographic_score: Optional[int] = Column(Integer, default=0)
	behavioral_score: Optional[int] = Column(Integer, default=0)
	
	# Assignment and Tracking
	lead_owner_id: Optional[str] = Column(String(50))
	assigned_date: Optional[datetime] = Column(DateTime)
	last_activity_date: Optional[datetime] = Column(DateTime)
	last_contact_date: Optional[datetime] = Column(DateTime)
	
	# Conversion Information
	converted_date: Optional[datetime] = Column(DateTime)
	converted_account_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_account.id'))
	converted_contact_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_contact.id'))
	converted_opportunity_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_opportunity.id'))
	
	# Campaign Information
	campaign_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_campaign.id'))
	marketing_list_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_marketing_list.id'))
	
	# Territory and Assignment
	territory_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_territory.id'))
	team_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_team.id'))
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_qualified: bool = Column(Boolean, default=False)
	is_converted: bool = Column(Boolean, default=False)
	do_not_call: bool = Column(Boolean, default=False)
	
	# Marketing Preferences
	email_opt_in: bool = Column(Boolean, default=True)
	sms_opt_in: bool = Column(Boolean, default=False)
	
	# Relationships
	converted_account = relationship("GCCRMAccount")
	converted_contact = relationship("GCCRMContact")
	converted_opportunity = relationship("GCCRMOpportunity")
	campaign = relationship("GCCRMCampaign", back_populates="leads")
	marketing_list = relationship("GCCRMMarketingList", back_populates="leads")
	activities = relationship("GCCRMActivity", back_populates="lead")
	territory = relationship("GCCRMTerritory", back_populates="leads")
	team = relationship("GCCRMTeam", back_populates="leads")
	
	@property
	def full_name(self) -> str:
		"""Get full name of lead"""
		return f"{self.first_name} {self.last_name}"
	
	def __repr__(self):
		return f"<Lead {self.full_name}>"

class GCCRMOpportunity(Model, AuditMixin):
	"""CRM Opportunity - represents sales opportunities"""
	
	__tablename__ = 'gc_crm_opportunity'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	opportunity_name: str = Column(String(200), nullable=False)
	opportunity_number: str = Column(String(50), unique=True, nullable=False)
	description: Optional[str] = Column(Text)
	
	# Relationships
	account_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_account.id'))
	customer_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_customer.id'))
	primary_contact_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_contact.id'))
	lead_source_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_lead.id'))
	
	# Sales Information
	amount: Optional[Decimal] = Column(Numeric(15, 2))
	currency: str = Column(String(10), default='USD')
	stage: str = Column(String(50), default=OpportunityStage.PROSPECTING.value)
	probability: Optional[int] = Column(Integer, default=0)  # 0-100%
	
	# Timeline
	close_date: Optional[date] = Column(Date)
	created_date: date = Column(Date, default=date.today)
	last_stage_change_date: Optional[datetime] = Column(DateTime)
	
	# Assignment
	opportunity_owner_id: Optional[str] = Column(String(50))
	sales_rep_id: Optional[str] = Column(String(50))
	
	# Business Details
	type: Optional[str] = Column(String(50))  # New Business, Existing Business, etc.
	next_step: Optional[str] = Column(Text)
	loss_reason: Optional[str] = Column(Text)
	competitor: Optional[str] = Column(String(200))
	
	# Territory and Assignment
	territory_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_territory.id'))
	team_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_team.id'))
	
	# Campaign Information
	campaign_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_campaign.id'))
	
	# Tracking
	last_activity_date: Optional[datetime] = Column(DateTime)
	days_in_stage: Optional[int] = Column(Integer, default=0)
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_closed: bool = Column(Boolean, default=False)
	is_won: bool = Column(Boolean, default=False)
	is_private: bool = Column(Boolean, default=False)
	
	# Forecast Information
	forecast_category: Optional[str] = Column(String(50))  # Pipeline, Best Case, Commit, Closed
	weighted_amount: Optional[Decimal] = Column(Numeric(15, 2))
	
	# Relationships
	account = relationship("GCCRMAccount", back_populates="opportunities")
	customer = relationship("GCCRMCustomer", back_populates="opportunities")
	primary_contact = relationship("GCCRMContact", back_populates="opportunities")
	lead_source = relationship("GCCRMLead")
	quote_lines = relationship("GCCRMQuoteLine", back_populates="opportunity")
	activities = relationship("GCCRMActivity", back_populates="opportunity")
	territory = relationship("GCCRMTerritory", back_populates="opportunities")
	team = relationship("GCCRMTeam", back_populates="opportunities")
	campaign = relationship("GCCRMCampaign", back_populates="opportunities")
	
	def __repr__(self):
		return f"<Opportunity {self.opportunity_name}>"

class GCCRMSalesStage(Model, AuditMixin):
	"""CRM Sales Stage - defines sales pipeline stages"""
	
	__tablename__ = 'gc_crm_sales_stage'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	name: str = Column(String(100), nullable=False)
	description: Optional[str] = Column(Text)
	stage_order: int = Column(Integer, nullable=False)
	
	# Stage Configuration
	default_probability: int = Column(Integer, default=0)  # 0-100%
	is_closed: bool = Column(Boolean, default=False)
	is_won: bool = Column(Boolean, default=False)
	
	# Appearance
	color: Optional[str] = Column(String(20))
	icon: Optional[str] = Column(String(50))
	
	# Behavior
	requires_approval: bool = Column(Boolean, default=False)
	auto_advance_days: Optional[int] = Column(Integer)
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_default: bool = Column(Boolean, default=False)
	
	def __repr__(self):
		return f"<SalesStage {self.name}>"

class GCCRMActivity(Model, AuditMixin):
	"""CRM Activity - represents interactions and tasks"""
	
	__tablename__ = 'gc_crm_activity'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	subject: str = Column(String(200), nullable=False)
	activity_type: str = Column(String(50), default=ActivityType.TASK.value)
	activity_status: str = Column(String(30), default="scheduled")
	priority: str = Column(String(20), default=ActivityPriority.NORMAL.value)
	
	# Content
	description: Optional[str] = Column(Text)
	result: Optional[str] = Column(Text)
	notes: Optional[str] = Column(Text)
	
	# Relationships
	account_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_account.id'))
	contact_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_contact.id'))
	customer_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_customer.id'))
	lead_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_lead.id'))
	opportunity_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_opportunity.id'))
	case_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_case.id'))
	
	# Scheduling
	start_date: Optional[datetime] = Column(DateTime)
	end_date: Optional[datetime] = Column(DateTime)
	due_date: Optional[datetime] = Column(DateTime)
	completed_date: Optional[datetime] = Column(DateTime)
	
	# Assignment
	assigned_to_id: Optional[str] = Column(String(50))
	created_by_id: Optional[str] = Column(String(50))
	
	# Meeting/Call Details
	location: Optional[str] = Column(String(200))
	duration_minutes: Optional[int] = Column(Integer)
	attendees: Optional[str] = Column(Text)  # JSON string of attendee list
	
	# Email Details
	email_from: Optional[str] = Column(String(100))
	email_to: Optional[str] = Column(Text)  # JSON string of recipient list
	email_cc: Optional[str] = Column(Text)  # JSON string
	email_bcc: Optional[str] = Column(Text)  # JSON string
	
	# Flags
	is_completed: bool = Column(Boolean, default=False)
	is_private: bool = Column(Boolean, default=False)
	is_all_day: bool = Column(Boolean, default=False)
	
	# Relationships
	account = relationship("GCCRMAccount")
	contact = relationship("GCCRMContact", back_populates="activities")
	customer = relationship("GCCRMCustomer", back_populates="activities")
	lead = relationship("GCCRMLead", back_populates="activities")
	opportunity = relationship("GCCRMOpportunity", back_populates="activities")
	case = relationship("GCCRMCase", back_populates="activities")
	
	def __repr__(self):
		return f"<Activity {self.subject}>"

class GCCRMTask(Model, AuditMixin):
	"""CRM Task - represents action items and follow-ups"""
	
	__tablename__ = 'gc_crm_task'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	subject: str = Column(String(200), nullable=False)
	description: Optional[str] = Column(Text)
	task_status: str = Column(String(30), default="not_started")
	priority: str = Column(String(20), default=ActivityPriority.NORMAL.value)
	
	# Scheduling
	due_date: Optional[datetime] = Column(DateTime)
	start_date: Optional[datetime] = Column(DateTime)
	completed_date: Optional[datetime] = Column(DateTime)
	reminder_date: Optional[datetime] = Column(DateTime)
	
	# Assignment
	assigned_to_id: Optional[str] = Column(String(50))
	created_by_id: Optional[str] = Column(String(50))
	
	# Progress
	percent_complete: int = Column(Integer, default=0)
	actual_hours: Optional[Decimal] = Column(Numeric(8, 2))
	estimated_hours: Optional[Decimal] = Column(Numeric(8, 2))
	
	# Relationships
	related_to_type: Optional[str] = Column(String(50))  # account, contact, lead, opportunity, case
	related_to_id: Optional[str] = Column(String(50))
	
	# Flags
	is_completed: bool = Column(Boolean, default=False)
	is_private: bool = Column(Boolean, default=False)
	is_recurring: bool = Column(Boolean, default=False)
	
	# Recurrence Information
	recurrence_pattern: Optional[str] = Column(String(200))  # JSON string
	recurrence_end_date: Optional[date] = Column(Date)
	
	def __repr__(self):
		return f"<Task {self.subject}>"

class GCCRMAppointment(Model, AuditMixin):
	"""CRM Appointment - represents scheduled meetings and events"""
	
	__tablename__ = 'gc_crm_appointment'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	subject: str = Column(String(200), nullable=False)
	description: Optional[str] = Column(Text)
	appointment_type: str = Column(String(50), default="meeting")
	
	# Scheduling
	start_date: datetime = Column(DateTime, nullable=False)
	end_date: datetime = Column(DateTime, nullable=False)
	is_all_day: bool = Column(Boolean, default=False)
	
	# Location
	location: Optional[str] = Column(String(200))
	meeting_url: Optional[str] = Column(String(500))  # For virtual meetings
	
	# Assignment
	organizer_id: Optional[str] = Column(String(50))
	assigned_to_id: Optional[str] = Column(String(50))
	
	# Attendees
	attendees: Optional[str] = Column(Text)  # JSON string of attendee list
	required_attendees: Optional[str] = Column(Text)  # JSON string
	optional_attendees: Optional[str] = Column(Text)  # JSON string
	
	# Relationships
	related_to_type: Optional[str] = Column(String(50))  # account, contact, lead, opportunity
	related_to_id: Optional[str] = Column(String(50))
	
	# Status and Flags
	appointment_status: str = Column(String(30), default="scheduled")
	is_private: bool = Column(Boolean, default=False)
	is_recurring: bool = Column(Boolean, default=False)
	
	# Recurrence Information
	recurrence_pattern: Optional[str] = Column(String(200))  # JSON string
	recurrence_end_date: Optional[date] = Column(Date)
	
	# Reminder
	reminder_minutes: Optional[int] = Column(Integer, default=15)
	
	def __repr__(self):
		return f"<Appointment {self.subject}>"

class GCCRMCampaign(Model, AuditMixin):
	"""CRM Campaign - represents marketing campaigns"""
	
	__tablename__ = 'gc_crm_campaign'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	name: str = Column(String(200), nullable=False)
	campaign_number: str = Column(String(50), unique=True, nullable=False)
	description: Optional[str] = Column(Text)
	campaign_type: str = Column(String(50), default=CampaignType.EMAIL_MARKETING.value)
	
	# Campaign Details
	objective: Optional[str] = Column(Text)
	target_audience: Optional[str] = Column(Text)
	message: Optional[str] = Column(Text)
	
	# Timeline
	start_date: Optional[date] = Column(Date)
	end_date: Optional[date] = Column(Date)
	
	# Budget and Costs
	budgeted_cost: Optional[Decimal] = Column(Numeric(15, 2))
	actual_cost: Optional[Decimal] = Column(Numeric(15, 2))
	cost_per_lead: Optional[Decimal] = Column(Numeric(10, 2))
	
	# Status and Assignment
	status: str = Column(String(30), default=CampaignStatus.PLANNING.value)
	campaign_owner_id: Optional[str] = Column(String(50))
	
	# Metrics and Results
	expected_response_rate: Optional[Decimal] = Column(Numeric(5, 2))  # Percentage
	actual_response_rate: Optional[Decimal] = Column(Numeric(5, 2))  # Percentage
	target_size: Optional[int] = Column(Integer)
	
	# Statistics (calculated fields)
	total_leads: int = Column(Integer, default=0)
	total_opportunities: int = Column(Integer, default=0)
	total_customers: int = Column(Integer, default=0)
	total_revenue: Optional[Decimal] = Column(Numeric(15, 2), default=0)
	
	# Email Campaign Specific
	email_template_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_email_template.id'))
	from_email: Optional[str] = Column(String(100))
	reply_to_email: Optional[str] = Column(String(100))
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_archived: bool = Column(Boolean, default=False)
	
	# Relationships
	leads = relationship("GCCRMLead", back_populates="campaign")
	opportunities = relationship("GCCRMOpportunity", back_populates="campaign")
	campaign_members = relationship("GCCRMCampaignMember", back_populates="campaign")
	email_template = relationship("GCCRMEmailTemplate", back_populates="campaigns")
	
	def __repr__(self):
		return f"<Campaign {self.name}>"

class GCCRMCampaignMember(Model, AuditMixin):
	"""CRM Campaign Member - links contacts/leads to campaigns"""
	
	__tablename__ = 'gc_crm_campaign_member'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Relationships
	campaign_id: str = Column(String(50), ForeignKey('gc_crm_campaign.id'), nullable=False)
	contact_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_contact.id'))
	lead_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_lead.id'))
	
	# Member Status
	status: str = Column(String(30), default="sent")  # sent, responded, bounced, unsubscribed
	member_type: str = Column(String(30), default="target")  # target, exclude
	
	# Response Tracking
	first_responded_date: Optional[datetime] = Column(DateTime)
	last_activity_date: Optional[datetime] = Column(DateTime)
	response_count: int = Column(Integer, default=0)
	
	# Email Tracking
	email_sent_date: Optional[datetime] = Column(DateTime)
	email_opened_date: Optional[datetime] = Column(DateTime)
	email_clicked_date: Optional[datetime] = Column(DateTime)
	email_bounced_date: Optional[datetime] = Column(DateTime)
	unsubscribed_date: Optional[datetime] = Column(DateTime)
	
	# Flags
	has_responded: bool = Column(Boolean, default=False)
	do_not_call: bool = Column(Boolean, default=False)
	email_opt_out: bool = Column(Boolean, default=False)
	
	# Relationships
	campaign = relationship("GCCRMCampaign", back_populates="campaign_members")
	contact = relationship("GCCRMContact")
	lead = relationship("GCCRMLead")
	
	def __repr__(self):
		return f"<CampaignMember {self.campaign.name}>"

class GCCRMMarketingList(Model, AuditMixin):
	"""CRM Marketing List - represents lists of contacts/leads for campaigns"""
	
	__tablename__ = 'gc_crm_marketing_list'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	name: str = Column(String(200), nullable=False)
	description: Optional[str] = Column(Text)
	list_type: str = Column(String(30), default="static")  # static, dynamic
	
	# List Criteria (for dynamic lists)
	criteria: Optional[str] = Column(Text)  # JSON string of filter criteria
	
	# Statistics
	member_count: int = Column(Integer, default=0)
	last_updated_date: Optional[datetime] = Column(DateTime)
	
	# Assignment
	list_owner_id: Optional[str] = Column(String(50))
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_locked: bool = Column(Boolean, default=False)
	
	# Relationships
	leads = relationship("GCCRMLead", back_populates="marketing_list")
	
	def __repr__(self):
		return f"<MarketingList {self.name}>"

class GCCRMEmailTemplate(Model, AuditMixin):
	"""CRM Email Template - represents email templates for campaigns"""
	
	__tablename__ = 'gc_crm_email_template'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	name: str = Column(String(200), nullable=False)
	description: Optional[str] = Column(Text)
	template_type: str = Column(String(50), default="campaign")  # campaign, followup, welcome
	
	# Email Content
	subject: str = Column(String(300), nullable=False)
	body_html: Optional[str] = Column(Text)
	body_text: Optional[str] = Column(Text)
	
	# Email Settings
	from_name: Optional[str] = Column(String(100))
	from_email: Optional[str] = Column(String(100))
	reply_to_email: Optional[str] = Column(String(100))
	
	# Template Configuration
	merge_fields: Optional[str] = Column(Text)  # JSON string of available merge fields
	
	# Assignment
	template_owner_id: Optional[str] = Column(String(50))
	
	# Usage Statistics
	usage_count: int = Column(Integer, default=0)
	last_used_date: Optional[datetime] = Column(DateTime)
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_global: bool = Column(Boolean, default=False)
	
	# Relationships
	campaigns = relationship("GCCRMCampaign", back_populates="email_template")
	
	def __repr__(self):
		return f"<EmailTemplate {self.name}>"

class GCCRMCase(Model, AuditMixin):
	"""CRM Case - represents customer service cases"""
	
	__tablename__ = 'gc_crm_case'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	case_number: str = Column(String(50), unique=True, nullable=False)
	subject: str = Column(String(200), nullable=False)
	description: Optional[str] = Column(Text)
	case_type: str = Column(String(50), default=CaseType.QUESTION.value)
	
	# Relationships
	account_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_account.id'))
	contact_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_contact.id'))
	customer_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_customer.id'))
	parent_case_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_case.id'))
	
	# Case Management
	status: str = Column(String(30), default=CaseStatus.NEW.value)
	priority: str = Column(String(20), default=CasePriority.NORMAL.value)
	origin: Optional[str] = Column(String(50))  # phone, email, web, chat
	reason: Optional[str] = Column(String(100))
	
	# Assignment
	case_owner_id: Optional[str] = Column(String(50))
	assigned_to_id: Optional[str] = Column(String(50))
	escalated_to_id: Optional[str] = Column(String(50))
	
	# Timeline
	reported_date: Optional[datetime] = Column(DateTime)
	assigned_date: Optional[datetime] = Column(DateTime)
	first_response_date: Optional[datetime] = Column(DateTime)
	escalated_date: Optional[datetime] = Column(DateTime)
	resolved_date: Optional[datetime] = Column(DateTime)
	closed_date: Optional[datetime] = Column(DateTime)
	
	# SLA Management
	response_due_date: Optional[datetime] = Column(DateTime)
	resolution_due_date: Optional[datetime] = Column(DateTime)
	is_sla_breached: bool = Column(Boolean, default=False)
	
	# Resolution Information
	resolution: Optional[str] = Column(Text)
	resolution_code: Optional[str] = Column(String(50))
	root_cause: Optional[str] = Column(Text)
	
	# Customer Satisfaction
	satisfaction_rating: Optional[int] = Column(Integer)  # 1-5 scale
	satisfaction_comments: Optional[str] = Column(Text)
	
	# Escalation
	escalation_level: int = Column(Integer, default=0)
	escalation_reason: Optional[str] = Column(Text)
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_escalated: bool = Column(Boolean, default=False)
	is_closed: bool = Column(Boolean, default=False)
	is_published: bool = Column(Boolean, default=False)  # For knowledge base
	
	# Relationships
	account = relationship("GCCRMAccount", back_populates="cases")
	contact = relationship("GCCRMContact")
	customer = relationship("GCCRMCustomer", back_populates="cases")
	parent_case = relationship("GCCRMCase", remote_side=[id])
	child_cases = relationship("GCCRMCase", back_populates="parent_case")
	case_comments = relationship("GCCRMCaseComment", back_populates="case")
	activities = relationship("GCCRMActivity", back_populates="case")
	
	def __repr__(self):
		return f"<Case {self.case_number}: {self.subject}>"

class GCCRMCaseComment(Model, AuditMixin):
	"""CRM Case Comment - represents comments on cases"""
	
	__tablename__ = 'gc_crm_case_comment'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	case_id: str = Column(String(50), ForeignKey('gc_crm_case.id'), nullable=False)
	
	# Comment Information
	comment: str = Column(Text, nullable=False)
	comment_type: str = Column(String(30), default="internal")  # internal, external, solution
	
	# Author Information
	created_by_id: Optional[str] = Column(String(50))
	author_name: Optional[str] = Column(String(100))
	author_email: Optional[str] = Column(String(100))
	
	# Flags
	is_public: bool = Column(Boolean, default=False)
	is_solution: bool = Column(Boolean, default=False)
	
	# Relationships
	case = relationship("GCCRMCase", back_populates="case_comments")
	
	def __repr__(self):
		return f"<CaseComment {self.case.case_number}>"

class GCCRMProduct(Model, AuditMixin):
	"""CRM Product - represents products/services for quotes and opportunities"""
	
	__tablename__ = 'gc_crm_product'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	product_code: str = Column(String(50), unique=True, nullable=False)
	product_name: str = Column(String(200), nullable=False)
	description: Optional[str] = Column(Text)
	product_family: Optional[str] = Column(String(100))
	
	# Pricing
	list_price: Optional[Decimal] = Column(Numeric(15, 2))
	standard_price: Optional[Decimal] = Column(Numeric(15, 2))
	cost_price: Optional[Decimal] = Column(Numeric(15, 2))
	currency: str = Column(String(10), default='USD')
	
	# Product Details
	category: Optional[str] = Column(String(100))
	brand: Optional[str] = Column(String(100))
	manufacturer: Optional[str] = Column(String(100))
	model: Optional[str] = Column(String(100))
	sku: Optional[str] = Column(String(100))
	
	# Inventory
	quantity_on_hand: Optional[int] = Column(Integer, default=0)
	quantity_available: Optional[int] = Column(Integer, default=0)
	reorder_level: Optional[int] = Column(Integer)
	
	# Product Configuration
	is_active: bool = Column(Boolean, default=True)
	is_taxable: bool = Column(Boolean, default=True)
	tax_category: Optional[str] = Column(String(50))
	
	# Relationships
	quote_lines = relationship("GCCRMQuoteLine", back_populates="product")
	
	def __repr__(self):
		return f"<Product {self.product_name}>"

class GCCRMPriceList(Model, AuditMixin):
	"""CRM Price List - represents different pricing structures"""
	
	__tablename__ = 'gc_crm_price_list'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	name: str = Column(String(200), nullable=False)
	description: Optional[str] = Column(Text)
	currency: str = Column(String(10), default='USD')
	
	# Price List Configuration
	effective_date: Optional[date] = Column(Date)
	expiry_date: Optional[date] = Column(Date)
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_default: bool = Column(Boolean, default=False)
	
	def __repr__(self):
		return f"<PriceList {self.name}>"

class GCCRMQuote(Model, AuditMixin):
	"""CRM Quote - represents sales quotes/proposals"""
	
	__tablename__ = 'gc_crm_quote'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	quote_number: str = Column(String(50), unique=True, nullable=False)
	quote_name: str = Column(String(200), nullable=False)
	description: Optional[str] = Column(Text)
	
	# Relationships
	opportunity_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_opportunity.id'))
	account_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_account.id'))
	contact_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_contact.id'))
	price_list_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_price_list.id'))
	
	# Quote Details
	status: str = Column(String(30), default=QuoteStatus.DRAFT.value)
	
	# Pricing
	subtotal: Optional[Decimal] = Column(Numeric(15, 2), default=0)
	discount_amount: Optional[Decimal] = Column(Numeric(15, 2), default=0)
	discount_percent: Optional[Decimal] = Column(Numeric(5, 2), default=0)
	tax_amount: Optional[Decimal] = Column(Numeric(15, 2), default=0)
	shipping_amount: Optional[Decimal] = Column(Numeric(15, 2), default=0)
	total_amount: Optional[Decimal] = Column(Numeric(15, 2), default=0)
	currency: str = Column(String(10), default='USD')
	
	# Timeline
	quote_date: Optional[date] = Column(Date)
	expiry_date: Optional[date] = Column(Date)
	valid_until_date: Optional[date] = Column(Date)
	
	# Assignment
	quote_owner_id: Optional[str] = Column(String(50))
	prepared_by_id: Optional[str] = Column(String(50))
	
	# Billing and Shipping
	billing_address: Optional[str] = Column(Text)
	shipping_address: Optional[str] = Column(Text)
	
	# Terms and Conditions
	payment_terms: Optional[str] = Column(String(100))
	delivery_terms: Optional[str] = Column(String(100))
	terms_and_conditions: Optional[str] = Column(Text)
	
	# Flags
	is_primary: bool = Column(Boolean, default=False)
	is_synced: bool = Column(Boolean, default=False)
	
	# Relationships
	opportunity = relationship("GCCRMOpportunity", back_populates="quote_lines")
	account = relationship("GCCRMAccount")
	contact = relationship("GCCRMContact")
	price_list = relationship("GCCRMPriceList")
	quote_lines = relationship("GCCRMQuoteLine", back_populates="quote")
	
	def __repr__(self):
		return f"<Quote {self.quote_number}>"

class GCCRMQuoteLine(Model, AuditMixin):
	"""CRM Quote Line - represents line items in quotes"""
	
	__tablename__ = 'gc_crm_quote_line'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Relationships
	quote_id: str = Column(String(50), ForeignKey('gc_crm_quote.id'), nullable=False)
	opportunity_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_opportunity.id'))
	product_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_product.id'))
	
	# Line Item Details
	line_number: int = Column(Integer, nullable=False)
	description: Optional[str] = Column(Text)
	
	# Quantity and Pricing
	quantity: Decimal = Column(Numeric(12, 2), default=1)
	list_price: Optional[Decimal] = Column(Numeric(15, 2))
	unit_price: Optional[Decimal] = Column(Numeric(15, 2))
	sales_price: Optional[Decimal] = Column(Numeric(15, 2))
	discount_amount: Optional[Decimal] = Column(Numeric(15, 2), default=0)
	discount_percent: Optional[Decimal] = Column(Numeric(5, 2), default=0)
	total_price: Optional[Decimal] = Column(Numeric(15, 2))
	
	# Product Information
	product_code: Optional[str] = Column(String(50))
	product_name: Optional[str] = Column(String(200))
	
	# Service Information
	service_date: Optional[date] = Column(Date)
	
	# Flags
	is_optional: bool = Column(Boolean, default=False)
	
	# Relationships
	quote = relationship("GCCRMQuote", back_populates="quote_lines")
	opportunity = relationship("GCCRMOpportunity", back_populates="quote_lines")
	product = relationship("GCCRMProduct", back_populates="quote_lines")
	
	def __repr__(self):
		return f"<QuoteLine {self.quote.quote_number}: {self.product_name}>"

class GCCRMTerritory(Model, AuditMixin):
	"""CRM Territory - represents sales territories"""
	
	__tablename__ = 'gc_crm_territory'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	name: str = Column(String(200), nullable=False)
	territory_code: str = Column(String(50), unique=True, nullable=False)
	description: Optional[str] = Column(Text)
	
	# Territory Definition
	territory_type: str = Column(String(50), default="geographic")  # geographic, product, industry
	parent_territory_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_territory.id'))
	
	# Geographic Information
	countries: Optional[str] = Column(Text)  # JSON array
	states_provinces: Optional[str] = Column(Text)  # JSON array
	cities: Optional[str] = Column(Text)  # JSON array
	postal_codes: Optional[str] = Column(Text)  # JSON array
	
	# Assignment
	territory_manager_id: Optional[str] = Column(String(50))
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	
	# Relationships
	parent_territory = relationship("GCCRMTerritory", remote_side=[id])
	child_territories = relationship("GCCRMTerritory", back_populates="parent_territory")
	accounts = relationship("GCCRMAccount", back_populates="territory")
	customers = relationship("GCCRMCustomer", back_populates="territory")
	leads = relationship("GCCRMLead", back_populates="territory")
	opportunities = relationship("GCCRMOpportunity", back_populates="territory")
	
	def __repr__(self):
		return f"<Territory {self.name}>"

class GCCRMTeam(Model, AuditMixin):
	"""CRM Team - represents sales teams"""
	
	__tablename__ = 'gc_crm_team'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	name: str = Column(String(200), nullable=False)
	team_code: str = Column(String(50), unique=True, nullable=False)
	description: Optional[str] = Column(Text)
	
	# Team Configuration
	team_type: str = Column(String(50), default="sales")  # sales, support, marketing
	parent_team_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_team.id'))
	
	# Assignment
	team_leader_id: Optional[str] = Column(String(50))
	manager_id: Optional[str] = Column(String(50))
	
	# Team Members (stored as JSON for simplicity)
	team_members: Optional[str] = Column(Text)  # JSON array of user IDs
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	
	# Relationships
	parent_team = relationship("GCCRMTeam", remote_side=[id])
	child_teams = relationship("GCCRMTeam", back_populates="parent_team")
	accounts = relationship("GCCRMAccount", back_populates="team")
	customers = relationship("GCCRMCustomer", back_populates="team")
	leads = relationship("GCCRMLead", back_populates="team")
	opportunities = relationship("GCCRMOpportunity", back_populates="team")
	
	def __repr__(self):
		return f"<Team {self.name}>"

class GCCRMForecast(Model, AuditMixin):
	"""CRM Forecast - represents sales forecasts"""
	
	__tablename__ = 'gc_crm_forecast'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	name: str = Column(String(200), nullable=False)
	forecast_period: str = Column(String(50), nullable=False)  # Q1-2024, Jan-2024, etc.
	
	# Forecast Details
	forecast_type: str = Column(String(50), default="opportunity")  # opportunity, revenue, units
	territory_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_territory.id'))
	team_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_team.id'))
	
	# Period Information
	start_date: date = Column(Date, nullable=False)
	end_date: date = Column(Date, nullable=False)
	
	# Forecast Amounts
	committed_amount: Optional[Decimal] = Column(Numeric(15, 2), default=0)
	best_case_amount: Optional[Decimal] = Column(Numeric(15, 2), default=0)
	pipeline_amount: Optional[Decimal] = Column(Numeric(15, 2), default=0)
	
	# Assignment
	forecast_owner_id: Optional[str] = Column(String(50))
	submitted_by_id: Optional[str] = Column(String(50))
	
	# Status
	status: str = Column(String(30), default="draft")  # draft, submitted, approved
	submitted_date: Optional[datetime] = Column(DateTime)
	approved_date: Optional[datetime] = Column(DateTime)
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	
	# Relationships
	territory = relationship("GCCRMTerritory")
	team = relationship("GCCRMTeam")
	
	def __repr__(self):
		return f"<Forecast {self.name}>"

class GCCRMDashboardWidget(Model, AuditMixin):
	"""CRM Dashboard Widget - represents dashboard components"""
	
	__tablename__ = 'gc_crm_dashboard_widget'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	name: str = Column(String(200), nullable=False)
	widget_type: str = Column(String(50), nullable=False)  # chart, table, metric, list
	description: Optional[str] = Column(Text)
	
	# Widget Configuration
	data_source: str = Column(String(100), nullable=False)  # opportunities, leads, activities
	chart_type: Optional[str] = Column(String(50))  # bar, line, pie, donut
	filter_criteria: Optional[str] = Column(Text)  # JSON string
	
	# Layout
	position_x: int = Column(Integer, default=0)
	position_y: int = Column(Integer, default=0)
	width: int = Column(Integer, default=4)
	height: int = Column(Integer, default=3)
	
	# Assignment
	widget_owner_id: Optional[str] = Column(String(50))
	
	# Refresh Configuration
	auto_refresh_minutes: Optional[int] = Column(Integer, default=30)
	last_refreshed: Optional[datetime] = Column(DateTime)
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_public: bool = Column(Boolean, default=False)
	
	def __repr__(self):
		return f"<DashboardWidget {self.name}>"

class GCCRMReport(Model, AuditMixin):
	"""CRM Report - represents saved reports"""
	
	__tablename__ = 'gc_crm_report'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	name: str = Column(String(200), nullable=False)
	description: Optional[str] = Column(Text)
	report_type: str = Column(String(50), nullable=False)  # tabular, summary, matrix, chart
	
	# Report Configuration
	data_source: str = Column(String(100), nullable=False)
	columns: Optional[str] = Column(Text)  # JSON array of column configurations
	filters: Optional[str] = Column(Text)  # JSON string of filter criteria
	sorting: Optional[str] = Column(Text)  # JSON string of sort configuration
	grouping: Optional[str] = Column(Text)  # JSON string of grouping configuration
	
	# Assignment
	report_owner_id: Optional[str] = Column(String(50))
	
	# Usage Statistics
	run_count: int = Column(Integer, default=0)
	last_run_date: Optional[datetime] = Column(DateTime)
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_public: bool = Column(Boolean, default=False)
	is_scheduled: bool = Column(Boolean, default=False)
	
	# Scheduling Configuration
	schedule_frequency: Optional[str] = Column(String(50))  # daily, weekly, monthly
	schedule_time: Optional[str] = Column(String(10))  # HH:MM format
	email_recipients: Optional[str] = Column(Text)  # JSON array of email addresses
	
	def __repr__(self):
		return f"<Report {self.name}>"

class GCCRMIntegrationLog(Model, AuditMixin):
	"""CRM Integration Log - tracks external system integrations"""
	
	__tablename__ = 'gc_crm_integration_log'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Integration Information
	integration_name: str = Column(String(100), nullable=False)
	operation_type: str = Column(String(50), nullable=False)  # sync, import, export, update
	entity_type: str = Column(String(50), nullable=False)  # lead, contact, account, opportunity
	
	# Operation Details
	start_time: datetime = Column(DateTime, nullable=False)
	end_time: Optional[datetime] = Column(DateTime)
	status: str = Column(String(30), default="running")  # running, completed, failed
	
	# Statistics
	records_processed: int = Column(Integer, default=0)
	records_successful: int = Column(Integer, default=0)
	records_failed: int = Column(Integer, default=0)
	
	# Error Information
	error_message: Optional[str] = Column(Text)
	error_details: Optional[str] = Column(Text)  # JSON string with detailed error info
	
	# Request/Response Information
	request_payload: Optional[str] = Column(Text)  # JSON string
	response_data: Optional[str] = Column(Text)  # JSON string
	
	def __repr__(self):
		return f"<IntegrationLog {self.integration_name}: {self.operation_type}>"

class GCCRMAuditLog(Model, AuditMixin):
	"""CRM Audit Log - tracks all CRM activities for compliance"""
	
	__tablename__ = 'gc_crm_audit_log'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Event Information
	event_type: str = Column(String(50), nullable=False)  # create, update, delete, view, export
	entity_type: str = Column(String(50), nullable=False)  # lead, contact, account, etc.
	entity_id: Optional[str] = Column(String(50))
	
	# User Information
	user_id: Optional[str] = Column(String(50))
	user_name: Optional[str] = Column(String(100))
	user_ip: Optional[str] = Column(String(50))
	user_agent: Optional[str] = Column(Text)
	
	# Event Details
	event_description: str = Column(Text, nullable=False)
	before_values: Optional[str] = Column(Text)  # JSON string of old values
	after_values: Optional[str] = Column(Text)  # JSON string of new values
	
	# Context Information
	session_id: Optional[str] = Column(String(100))
	request_id: Optional[str] = Column(String(100))
	
	def __repr__(self):
		return f"<AuditLog {self.event_type}: {self.entity_type}>"

# Advanced CRM Models for Enterprise Features

class GCCRMLeadSource(Model, AuditMixin):
	"""CRM Lead Source - configurable lead sources with tracking"""
	
	__tablename__ = 'gc_crm_lead_source'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	name: str = Column(String(100), nullable=False)
	code: str = Column(String(20), nullable=False)
	description: Optional[str] = Column(Text)
	
	# Configuration
	is_active: bool = Column(Boolean, default=True)
	is_trackable: bool = Column(Boolean, default=True)
	cost_per_lead: Optional[Decimal] = Column(Numeric(10, 2))
	expected_conversion_rate: Optional[Decimal] = Column(Numeric(5, 2))
	
	# Analytics
	total_leads: int = Column(Integer, default=0)
	converted_leads: int = Column(Integer, default=0)
	actual_conversion_rate: Optional[Decimal] = Column(Numeric(5, 2))
	average_deal_size: Optional[Decimal] = Column(Numeric(15, 2))
	
	# ROI Tracking
	total_cost: Optional[Decimal] = Column(Numeric(15, 2), default=0)
	total_revenue: Optional[Decimal] = Column(Numeric(15, 2), default=0)
	roi_percentage: Optional[Decimal] = Column(Numeric(8, 2))
	
	def __repr__(self):
		return f"<LeadSource {self.name}>"

class GCCRMCustomerSegment(Model, AuditMixin):
	"""CRM Customer Segment - AI-driven customer segmentation"""
	
	__tablename__ = 'gc_crm_customer_segment'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	name: str = Column(String(100), nullable=False)
	description: Optional[str] = Column(Text)
	segment_type: str = Column(String(50), default="behavioral")  # behavioral, demographic, psychographic, geographic
	
	# Segment Criteria
	criteria_json: Optional[str] = Column(Text)  # JSON criteria for dynamic segments
	is_dynamic: bool = Column(Boolean, default=True)
	
	# AI/ML Configuration
	ml_model_id: Optional[str] = Column(String(50))
	clustering_algorithm: Optional[str] = Column(String(50))  # kmeans, dbscan, hierarchical
	feature_weights: Optional[str] = Column(Text)  # JSON feature importance weights
	
	# Segment Statistics
	customer_count: int = Column(Integer, default=0)
	average_ltv: Optional[Decimal] = Column(Numeric(15, 2))
	churn_risk_score: Optional[Decimal] = Column(Numeric(5, 2))
	engagement_score: Optional[Decimal] = Column(Numeric(5, 2))
	
	# Refresh Settings
	last_updated: Optional[datetime] = Column(DateTime)
	auto_refresh_enabled: bool = Column(Boolean, default=True)
	refresh_frequency_hours: int = Column(Integer, default=24)
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	
	def __repr__(self):
		return f"<CustomerSegment {self.name}>"

class GCCRMCustomerScore(Model, AuditMixin):
	"""CRM Customer Score - AI-generated customer scores and ratings"""
	
	__tablename__ = 'gc_crm_customer_score'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Relationships
	customer_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_customer.id'))
	account_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_account.id'))
	lead_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_lead.id'))
	
	# Score Types
	score_type: str = Column(String(50), nullable=False)  # lead_score, customer_health, churn_risk, ltv, engagement
	
	# Score Values
	score_value: Decimal = Column(Numeric(8, 2), nullable=False)
	score_percentile: Optional[int] = Column(Integer)  # 0-100 percentile rank
	confidence_level: Optional[Decimal] = Column(Numeric(5, 2))  # Model confidence 0-1
	
	# Score Components
	demographic_score: Optional[Decimal] = Column(Numeric(8, 2))
	behavioral_score: Optional[Decimal] = Column(Numeric(8, 2))
	engagement_score: Optional[Decimal] = Column(Numeric(8, 2))
	financial_score: Optional[Decimal] = Column(Numeric(8, 2))
	
	# Model Information
	model_version: Optional[str] = Column(String(20))
	model_algorithm: Optional[str] = Column(String(50))
	feature_importance: Optional[str] = Column(Text)  # JSON feature weights
	
	# Validity
	calculated_date: datetime = Column(DateTime, default=datetime.utcnow)
	expires_date: Optional[datetime] = Column(DateTime)
	is_current: bool = Column(Boolean, default=True)
	
	# Relationships
	customer = relationship("GCCRMCustomer")
	account = relationship("GCCRMAccount")
	lead = relationship("GCCRMLead")
	
	def __repr__(self):
		return f"<CustomerScore {self.score_type}: {self.score_value}>"

class GCCRMSocialProfile(Model, AuditMixin):
	"""CRM Social Profile - social media integration and insights"""
	
	__tablename__ = 'gc_crm_social_profile'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Relationships
	contact_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_contact.id'))
	customer_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_customer.id'))
	account_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_account.id'))
	
	# Social Platform
	platform: str = Column(String(50), nullable=False)  # linkedin, twitter, facebook, instagram
	profile_url: Optional[str] = Column(String(500))
	profile_username: Optional[str] = Column(String(100))
	profile_id: Optional[str] = Column(String(100))
	
	# Profile Information
	display_name: Optional[str] = Column(String(200))
	bio: Optional[str] = Column(Text)
	location: Optional[str] = Column(String(100))
	industry: Optional[str] = Column(String(100))
	company: Optional[str] = Column(String(200))
	job_title: Optional[str] = Column(String(100))
	
	# Metrics
	followers_count: Optional[int] = Column(Integer)
	following_count: Optional[int] = Column(Integer)
	posts_count: Optional[int] = Column(Integer)
	engagement_rate: Optional[Decimal] = Column(Numeric(5, 2))
	influence_score: Optional[Decimal] = Column(Numeric(8, 2))
	
	# Activity Tracking
	last_post_date: Optional[datetime] = Column(DateTime)
	last_activity_date: Optional[datetime] = Column(DateTime)
	activity_frequency: Optional[str] = Column(String(20))  # daily, weekly, monthly, sporadic
	
	# Data Sync
	last_synced: Optional[datetime] = Column(DateTime)
	sync_enabled: bool = Column(Boolean, default=True)
	data_source: Optional[str] = Column(String(50))  # api, manual, imported
	
	# Flags
	is_verified: bool = Column(Boolean, default=False)
	is_active: bool = Column(Boolean, default=True)
	
	# Relationships
	contact = relationship("GCCRMContact")
	customer = relationship("GCCRMCustomer")
	account = relationship("GCCRMAccount")
	
	def __repr__(self):
		return f"<SocialProfile {self.platform}: {self.profile_username}>"

class GCCRMCommunication(Model, AuditMixin):
	"""CRM Communication - track all customer communications"""
	
	__tablename__ = 'gc_crm_communication'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Relationships
	contact_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_contact.id'))
	customer_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_customer.id'))
	account_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_account.id'))
	lead_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_lead.id'))
	opportunity_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_opportunity.id'))
	case_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_case.id'))
	
	# Communication Details
	communication_type: str = Column(String(50), nullable=False)  # email, call, meeting, sms, chat, social
	direction: str = Column(String(20), nullable=False)  # inbound, outbound
	subject: Optional[str] = Column(String(500))
	content: Optional[str] = Column(Text)
	
	# Participants
	from_address: Optional[str] = Column(String(200))
	to_addresses: Optional[str] = Column(Text)  # JSON array
	cc_addresses: Optional[str] = Column(Text)  # JSON array
	bcc_addresses: Optional[str] = Column(Text)  # JSON array
	
	# Timing
	sent_date: Optional[datetime] = Column(DateTime)
	received_date: Optional[datetime] = Column(DateTime)
	duration_minutes: Optional[int] = Column(Integer)
	
	# Status and Tracking
	status: str = Column(String(30), default="sent")  # sent, delivered, opened, replied, bounced
	delivery_status: Optional[str] = Column(String(50))
	open_count: int = Column(Integer, default=0)
	click_count: int = Column(Integer, default=0)
	reply_count: int = Column(Integer, default=0)
	
	# AI Analysis
	sentiment_score: Optional[Decimal] = Column(Numeric(5, 2))  # -1 to 1
	sentiment_label: Optional[str] = Column(String(20))  # positive, negative, neutral
	intent_classification: Optional[str] = Column(String(100))
	topic_tags: Optional[str] = Column(Text)  # JSON array of extracted topics
	urgency_score: Optional[Decimal] = Column(Numeric(3, 2))  # 0-1
	
	# Attachments and Media
	has_attachments: bool = Column(Boolean, default=False)
	attachment_count: int = Column(Integer, default=0)
	attachment_info: Optional[str] = Column(Text)  # JSON metadata
	
	# Integration Data
	external_id: Optional[str] = Column(String(100))
	message_id: Optional[str] = Column(String(200))
	thread_id: Optional[str] = Column(String(200))
	source_system: Optional[str] = Column(String(50))
	
	# Flags
	is_internal: bool = Column(Boolean, default=False)
	is_automated: bool = Column(Boolean, default=False)
	requires_response: bool = Column(Boolean, default=False)
	
	# Relationships
	contact = relationship("GCCRMContact")
	customer = relationship("GCCRMCustomer")
	account = relationship("GCCRMAccount")
	lead = relationship("GCCRMLead")
	opportunity = relationship("GCCRMOpportunity")
	case = relationship("GCCRMCase")
	
	def __repr__(self):
		return f"<Communication {self.communication_type}: {self.subject}>"

class GCCRMWorkflowDefinition(Model, AuditMixin):
	"""CRM Workflow Definition - configurable business process workflows"""
	
	__tablename__ = 'gc_crm_workflow_definition'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	name: str = Column(String(200), nullable=False)
	description: Optional[str] = Column(Text)
	workflow_type: str = Column(String(50), nullable=False)  # lead_nurturing, sales_process, support_escalation
	
	# Workflow Configuration
	trigger_conditions: str = Column(Text, nullable=False)  # JSON trigger conditions
	workflow_steps: str = Column(Text, nullable=False)  # JSON workflow definition
	version: str = Column(String(20), default="1.0")
	
	# Execution Settings
	is_active: bool = Column(Boolean, default=True)
	auto_start: bool = Column(Boolean, default=True)
	max_concurrent_executions: int = Column(Integer, default=100)
	timeout_hours: Optional[int] = Column(Integer)
	
	# Assignment
	workflow_owner_id: Optional[str] = Column(String(50))
	approval_required: bool = Column(Boolean, default=False)
	
	# Statistics
	total_executions: int = Column(Integer, default=0)
	successful_executions: int = Column(Integer, default=0)
	failed_executions: int = Column(Integer, default=0)
	average_duration_minutes: Optional[Decimal] = Column(Numeric(10, 2))
	
	# Relationships
	workflow_executions = relationship("GCCRMWorkflowExecution", back_populates="workflow_definition")
	
	def __repr__(self):
		return f"<WorkflowDefinition {self.name}>"

class GCCRMWorkflowExecution(Model, AuditMixin):
	"""CRM Workflow Execution - individual workflow execution instances"""
	
	__tablename__ = 'gc_crm_workflow_execution'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Relationships
	workflow_definition_id: str = Column(String(50), ForeignKey('gc_crm_workflow_definition.id'), nullable=False)
	
	# Target Entity
	entity_type: str = Column(String(50), nullable=False)  # lead, opportunity, case, customer
	entity_id: str = Column(String(50), nullable=False)
	
	# Execution Details
	execution_status: str = Column(String(30), default="running")  # running, completed, failed, paused, cancelled
	current_step: Optional[str] = Column(String(100))
	step_number: int = Column(Integer, default=1)
	total_steps: int = Column(Integer, default=1)
	
	# Timing
	started_date: datetime = Column(DateTime, default=datetime.utcnow)
	completed_date: Optional[datetime] = Column(DateTime)
	paused_date: Optional[datetime] = Column(DateTime)
	
	# Results
	execution_result: Optional[str] = Column(Text)  # JSON execution results
	error_message: Optional[str] = Column(Text)
	error_details: Optional[str] = Column(Text)
	
	# Context Data
	context_data: Optional[str] = Column(Text)  # JSON context variables
	input_data: Optional[str] = Column(Text)  # JSON input parameters
	output_data: Optional[str] = Column(Text)  # JSON output results
	
	# Assignment
	assigned_to_id: Optional[str] = Column(String(50))
	
	# Relationships
	workflow_definition = relationship("GCCRMWorkflowDefinition", back_populates="workflow_executions")
	
	def __repr__(self):
		return f"<WorkflowExecution {self.workflow_definition.name}: {self.execution_status}>"

class GCCRMNotification(Model, AuditMixin):
	"""CRM Notification - system and user notifications"""
	
	__tablename__ = 'gc_crm_notification'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Recipient Information
	recipient_user_id: str = Column(String(50), nullable=False)
	recipient_email: Optional[str] = Column(String(100))
	recipient_phone: Optional[str] = Column(String(30))
	
	# Notification Content
	notification_type: str = Column(String(50), nullable=False)  # task_due, opportunity_update, lead_assignment
	title: str = Column(String(200), nullable=False)
	message: str = Column(Text, nullable=False)
	priority: str = Column(String(20), default="normal")  # low, normal, high, urgent
	
	# Delivery Channels
	delivery_channels: str = Column(String(100), default="in_app")  # JSON: in_app, email, sms, push
	
	# Status and Tracking
	status: str = Column(String(30), default="pending")  # pending, sent, delivered, read, failed
	sent_date: Optional[datetime] = Column(DateTime)
	delivered_date: Optional[datetime] = Column(DateTime)
	read_date: Optional[datetime] = Column(DateTime)
	
	# Related Entity
	related_entity_type: Optional[str] = Column(String(50))
	related_entity_id: Optional[str] = Column(String(50))
	
	# Action Information
	action_url: Optional[str] = Column(String(500))
	action_text: Optional[str] = Column(String(100))
	action_data: Optional[str] = Column(Text)  # JSON action parameters
	
	# Scheduling
	scheduled_date: Optional[datetime] = Column(DateTime)
	expires_date: Optional[datetime] = Column(DateTime)
	
	# Flags
	is_actionable: bool = Column(Boolean, default=False)
	is_dismissible: bool = Column(Boolean, default=True)
	auto_dismiss: bool = Column(Boolean, default=False)
	
	def __repr__(self):
		return f"<Notification {self.notification_type}: {self.title}>"

class GCCRMKnowledgeBase(Model, AuditMixin):
	"""CRM Knowledge Base - centralized knowledge management"""
	
	__tablename__ = 'gc_crm_knowledge_base'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Basic Information
	title: str = Column(String(300), nullable=False)
	content: str = Column(Text, nullable=False)
	summary: Optional[str] = Column(Text)
	article_type: str = Column(String(50), default="general")  # faq, howto, troubleshooting, product_info
	
	# Categorization
	category: Optional[str] = Column(String(100))
	subcategory: Optional[str] = Column(String(100))
	tags: Optional[str] = Column(Text)  # JSON array of tags
	keywords: Optional[str] = Column(Text)
	
	# Content Management
	content_format: str = Column(String(20), default="markdown")  # markdown, html, text
	language: str = Column(String(10), default="en")
	version: str = Column(String(20), default="1.0")
	
	# Publishing
	status: str = Column(String(30), default="draft")  # draft, review, published, archived
	published_date: Optional[datetime] = Column(DateTime)
	expiry_date: Optional[datetime] = Column(DateTime)
	
	# Authoring
	author_id: Optional[str] = Column(String(50))
	reviewer_id: Optional[str] = Column(String(50))
	last_reviewed_date: Optional[datetime] = Column(DateTime)
	
	# Usage Analytics
	view_count: int = Column(Integer, default=0)
	helpful_votes: int = Column(Integer, default=0)
	unhelpful_votes: int = Column(Integer, default=0)
	average_rating: Optional[Decimal] = Column(Numeric(3, 2))
	
	# Search and AI
	search_vector: Optional[str] = Column(Text)  # Vector embeddings for semantic search
	ai_generated: bool = Column(Boolean, default=False)
	ai_confidence: Optional[Decimal] = Column(Numeric(5, 2))
	
	# Related Content
	related_articles: Optional[str] = Column(Text)  # JSON array of related article IDs
	parent_article_id: Optional[str] = Column(String(50), ForeignKey('gc_crm_knowledge_base.id'))
	
	# Flags
	is_featured: bool = Column(Boolean, default=False)
	is_internal: bool = Column(Boolean, default=False)
	requires_authentication: bool = Column(Boolean, default=True)
	
	# Relationships
	parent_article = relationship("GCCRMKnowledgeBase", remote_side=[id])
	child_articles = relationship("GCCRMKnowledgeBase", back_populates="parent_article")
	
	def __repr__(self):
		return f"<KnowledgeBase {self.title}>"

class GCCRMCustomField(Model, AuditMixin):
	"""CRM Custom Field - configurable custom fields for entities"""
	
	__tablename__ = 'gc_crm_custom_field'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Field Definition
	field_name: str = Column(String(100), nullable=False)
	field_label: str = Column(String(200), nullable=False)
	field_type: str = Column(String(50), nullable=False)  # text, number, date, boolean, picklist, textarea, url, email
	entity_type: str = Column(String(50), nullable=False)  # lead, contact, account, opportunity, case
	
	# Field Configuration
	is_required: bool = Column(Boolean, default=False)
	is_unique: bool = Column(Boolean, default=False)
	default_value: Optional[str] = Column(Text)
	help_text: Optional[str] = Column(Text)
	
	# Validation Rules
	min_length: Optional[int] = Column(Integer)
	max_length: Optional[int] = Column(Integer)
	min_value: Optional[Decimal] = Column(Numeric(15, 2))
	max_value: Optional[Decimal] = Column(Numeric(15, 2))
	validation_pattern: Optional[str] = Column(String(500))  # Regex pattern
	
	# Picklist Options
	picklist_values: Optional[str] = Column(Text)  # JSON array for picklist fields
	allow_multi_select: bool = Column(Boolean, default=False)
	
	# Display Configuration
	display_order: int = Column(Integer, default=0)
	field_width: Optional[str] = Column(String(20))  # full, half, quarter
	section_name: Optional[str] = Column(String(100))
	
	# Access Control
	visible_to_roles: Optional[str] = Column(Text)  # JSON array of role IDs
	editable_by_roles: Optional[str] = Column(Text)  # JSON array of role IDs
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_searchable: bool = Column(Boolean, default=True)
	is_reportable: bool = Column(Boolean, default=True)
	
	def __repr__(self):
		return f"<CustomField {self.field_name} ({self.entity_type})>"

class GCCRMCustomFieldValue(Model, AuditMixin):
	"""CRM Custom Field Value - stores custom field values"""
	
	__tablename__ = 'gc_crm_custom_field_value'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Relationships
	custom_field_id: str = Column(String(50), ForeignKey('gc_crm_custom_field.id'), nullable=False)
	entity_id: str = Column(String(50), nullable=False)
	
	# Value Storage
	text_value: Optional[str] = Column(Text)
	number_value: Optional[Decimal] = Column(Numeric(15, 4))
	date_value: Optional[date] = Column(Date)
	datetime_value: Optional[datetime] = Column(DateTime)
	boolean_value: Optional[bool] = Column(Boolean)
	json_value: Optional[str] = Column(Text)  # For complex data types
	
	# Relationships
	custom_field = relationship("GCCRMCustomField")
	
	def __repr__(self):
		return f"<CustomFieldValue {self.custom_field.field_name}: {self.text_value or self.number_value}>"

class GCCRMDocumentAttachment(Model, AuditMixin):
	"""CRM Document Attachment - file attachments for CRM entities"""
	
	__tablename__ = 'gc_crm_document_attachment'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Related Entity
	entity_type: str = Column(String(50), nullable=False)  # lead, contact, account, opportunity, case
	entity_id: str = Column(String(50), nullable=False)
	
	# File Information
	file_name: str = Column(String(300), nullable=False)
	original_name: str = Column(String(300), nullable=False)
	file_path: str = Column(String(500), nullable=False)
	file_size_bytes: int = Column(Integer, nullable=False)
	mime_type: Optional[str] = Column(String(100))
	file_extension: Optional[str] = Column(String(10))
	
	# Document Properties
	document_type: str = Column(String(50), default="general")  # contract, proposal, invoice, presentation
	description: Optional[str] = Column(Text)
	version: str = Column(String(20), default="1.0")
	
	# Upload Information
	uploaded_by_id: Optional[str] = Column(String(50))
	upload_date: datetime = Column(DateTime, default=datetime.utcnow)
	upload_source: Optional[str] = Column(String(50))  # manual, email, api, integration
	
	# Access Control
	is_public: bool = Column(Boolean, default=False)
	access_level: str = Column(String(30), default="internal")  # public, internal, restricted, confidential
	allowed_user_ids: Optional[str] = Column(Text)  # JSON array
	
	# Content Analysis
	text_content: Optional[str] = Column(Text)  # Extracted text for search
	has_been_analyzed: bool = Column(Boolean, default=False)
	analysis_results: Optional[str] = Column(Text)  # JSON analysis results
	
	# Virus Scanning
	virus_scan_status: str = Column(String(30), default="pending")  # pending, clean, infected, error
	virus_scan_date: Optional[datetime] = Column(DateTime)
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_encrypted: bool = Column(Boolean, default=False)
	
	def __repr__(self):
		return f"<DocumentAttachment {self.file_name}>"

class GCCRMEventLog(Model, AuditMixin):
	"""CRM Event Log - comprehensive event tracking for analytics"""
	
	__tablename__ = 'gc_crm_event_log'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Event Information
	event_type: str = Column(String(50), nullable=False)  # page_view, button_click, form_submit, api_call
	event_category: str = Column(String(50), nullable=False)  # user_action, system_event, integration
	event_action: str = Column(String(100), nullable=False)
	event_label: Optional[str] = Column(String(200))
	
	# Context Information
	user_id: Optional[str] = Column(String(50))
	session_id: Optional[str] = Column(String(100))
	ip_address: Optional[str] = Column(String(50))
	user_agent: Optional[str] = Column(Text)
	
	# Related Entity
	entity_type: Optional[str] = Column(String(50))
	entity_id: Optional[str] = Column(String(50))
	
	# Event Data
	event_data: Optional[str] = Column(Text)  # JSON event payload
	event_value: Optional[Decimal] = Column(Numeric(15, 2))
	duration_ms: Optional[int] = Column(Integer)
	
	# Timing
	event_timestamp: datetime = Column(DateTime, default=datetime.utcnow, index=True)
	client_timestamp: Optional[datetime] = Column(DateTime)
	
	# Request Information
	request_id: Optional[str] = Column(String(100))
	request_method: Optional[str] = Column(String(10))
	request_url: Optional[str] = Column(String(500))
	response_status: Optional[int] = Column(Integer)
	
	# Geolocation
	country: Optional[str] = Column(String(50))
	region: Optional[str] = Column(String(50))
	city: Optional[str] = Column(String(100))
	timezone: Optional[str] = Column(String(50))
	
	# Device Information
	device_type: Optional[str] = Column(String(20))  # desktop, mobile, tablet
	browser_name: Optional[str] = Column(String(50))
	browser_version: Optional[str] = Column(String(20))
	os_name: Optional[str] = Column(String(50))
	os_version: Optional[str] = Column(String(20))
	
	def __repr__(self):
		return f"<EventLog {self.event_type}: {self.event_action}>"

class GCCRMSystemConfiguration(Model, AuditMixin):
	"""CRM System Configuration - tenant-specific configuration settings"""
	
	__tablename__ = 'gc_crm_system_configuration'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Configuration Key
	config_key: str = Column(String(100), nullable=False)
	config_category: str = Column(String(50), nullable=False)  # ui, workflow, integration, security, ai
	
	# Configuration Value
	config_value: Optional[str] = Column(Text)
	data_type: str = Column(String(20), default="string")  # string, number, boolean, json, encrypted
	
	# Metadata
	description: Optional[str] = Column(Text)
	default_value: Optional[str] = Column(Text)
	validation_rules: Optional[str] = Column(Text)  # JSON validation configuration
	
	# Access Control
	is_user_configurable: bool = Column(Boolean, default=True)
	requires_admin: bool = Column(Boolean, default=False)
	is_sensitive: bool = Column(Boolean, default=False)
	
	# Environment
	environment: str = Column(String(20), default="all")  # all, development, staging, production
	
	# Flags
	is_active: bool = Column(Boolean, default=True)
	is_encrypted: bool = Column(Boolean, default=False)
	
	def __repr__(self):
		return f"<SystemConfiguration {self.config_key}: {self.config_value}>"

class GCCRMWebhookEndpoint(Model, AuditMixin):
	"""CRM Webhook Endpoint - external webhook configurations"""
	
	__tablename__ = 'gc_crm_webhook_endpoint'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Endpoint Configuration
	name: str = Column(String(200), nullable=False)
	url: str = Column(String(1000), nullable=False)
	http_method: str = Column(String(10), default="POST")
	
	# Authentication
	auth_type: str = Column(String(30), default="none")  # none, basic, bearer, api_key, oauth
	auth_config: Optional[str] = Column(Text)  # JSON auth configuration (encrypted)
	
	# Event Configuration
	event_types: str = Column(Text, nullable=False)  # JSON array of event types to trigger
	event_filters: Optional[str] = Column(Text)  # JSON filter conditions
	
	# Delivery Configuration
	retry_count: int = Column(Integer, default=3)
	retry_delay_seconds: int = Column(Integer, default=30)
	timeout_seconds: int = Column(Integer, default=30)
	
	# Headers and Payload
	custom_headers: Optional[str] = Column(Text)  # JSON custom headers
	payload_template: Optional[str] = Column(Text)  # Custom payload template
	
	# Status and Monitoring
	is_active: bool = Column(Boolean, default=True)
	last_triggered: Optional[datetime] = Column(DateTime)
	success_count: int = Column(Integer, default=0)
	failure_count: int = Column(Integer, default=0)
	
	# Security
	secret_key: Optional[str] = Column(String(200))  # For webhook signature validation
	allowed_ips: Optional[str] = Column(Text)  # JSON array of allowed IP addresses
	
	def __repr__(self):
		return f"<WebhookEndpoint {self.name}: {self.url}>"

class GCCRMWebhookDelivery(Model, AuditMixin):
	"""CRM Webhook Delivery - webhook delivery attempts and results"""
	
	__tablename__ = 'gc_crm_webhook_delivery'
	
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Relationships
	webhook_endpoint_id: str = Column(String(50), ForeignKey('gc_crm_webhook_endpoint.id'), nullable=False)
	
	# Event Information
	event_type: str = Column(String(50), nullable=False)
	event_data: str = Column(Text, nullable=False)  # JSON event payload
	entity_type: Optional[str] = Column(String(50))
	entity_id: Optional[str] = Column(String(50))
	
	# Delivery Attempt
	attempt_number: int = Column(Integer, default=1)
	delivery_status: str = Column(String(30), default="pending")  # pending, success, failed, retrying
	
	# Request Details
	request_url: str = Column(String(1000), nullable=False)
	request_method: str = Column(String(10), nullable=False)
	request_headers: Optional[str] = Column(Text)  # JSON request headers
	request_payload: str = Column(Text, nullable=False)
	
	# Response Details
	response_status_code: Optional[int] = Column(Integer)
	response_headers: Optional[str] = Column(Text)  # JSON response headers
	response_body: Optional[str] = Column(Text)
	response_time_ms: Optional[int] = Column(Integer)
	
	# Timing
	scheduled_time: datetime = Column(DateTime, default=datetime.utcnow)
	attempted_time: Optional[datetime] = Column(DateTime)
	completed_time: Optional[datetime] = Column(DateTime)
	next_retry_time: Optional[datetime] = Column(DateTime)
	
	# Error Information
	error_message: Optional[str] = Column(Text)
	error_code: Optional[str] = Column(String(50))
	
	# Relationships
	webhook_endpoint = relationship("GCCRMWebhookEndpoint")
	
	def __repr__(self):
		return f"<WebhookDelivery {self.event_type}: {self.delivery_status}>"

# Database Indexes for Performance (Enhanced)

Index('idx_gc_crm_account_tenant_active', GCCRMAccount.tenant_id, GCCRMAccount.is_active)
Index('idx_gc_crm_account_owner', GCCRMAccount.account_owner_id)
Index('idx_gc_crm_account_status', GCCRMAccount.account_status)
Index('idx_gc_crm_account_industry', GCCRMAccount.industry)
Index('idx_gc_crm_account_revenue', GCCRMAccount.annual_revenue)

Index('idx_gc_crm_customer_tenant_active', GCCRMCustomer.tenant_id, GCCRMCustomer.is_active)
Index('idx_gc_crm_customer_owner', GCCRMCustomer.customer_owner_id)
Index('idx_gc_crm_customer_email', GCCRMCustomer.email)
Index('idx_gc_crm_customer_status', GCCRMCustomer.customer_status)
Index('idx_gc_crm_customer_since', GCCRMCustomer.customer_since)

Index('idx_gc_crm_contact_tenant_active', GCCRMContact.tenant_id, GCCRMContact.is_active)
Index('idx_gc_crm_contact_account', GCCRMContact.account_id)
Index('idx_gc_crm_contact_email', GCCRMContact.email)
Index('idx_gc_crm_contact_owner', GCCRMContact.contact_owner_id)

Index('idx_gc_crm_lead_tenant_active', GCCRMLead.tenant_id, GCCRMLead.is_active)
Index('idx_gc_crm_lead_status', GCCRMLead.lead_status)
Index('idx_gc_crm_lead_owner', GCCRMLead.lead_owner_id)
Index('idx_gc_crm_lead_email', GCCRMLead.email)
Index('idx_gc_crm_lead_source', GCCRMLead.lead_source)
Index('idx_gc_crm_lead_score', GCCRMLead.lead_score)
Index('idx_gc_crm_lead_created', GCCRMLead.created_on)

Index('idx_gc_crm_opportunity_tenant_active', GCCRMOpportunity.tenant_id, GCCRMOpportunity.is_active)
Index('idx_gc_crm_opportunity_stage', GCCRMOpportunity.stage)
Index('idx_gc_crm_opportunity_close_date', GCCRMOpportunity.close_date)
Index('idx_gc_crm_opportunity_owner', GCCRMOpportunity.opportunity_owner_id)
Index('idx_gc_crm_opportunity_amount', GCCRMOpportunity.amount)
Index('idx_gc_crm_opportunity_account', GCCRMOpportunity.account_id)

Index('idx_gc_crm_activity_tenant', GCCRMActivity.tenant_id)
Index('idx_gc_crm_activity_assigned', GCCRMActivity.assigned_to_id)
Index('idx_gc_crm_activity_due_date', GCCRMActivity.due_date)
Index('idx_gc_crm_activity_type', GCCRMActivity.activity_type)
Index('idx_gc_crm_activity_status', GCCRMActivity.activity_status)
Index('idx_gc_crm_activity_entity', GCCRMActivity.opportunity_id, GCCRMActivity.contact_id)

Index('idx_gc_crm_case_tenant_active', GCCRMCase.tenant_id, GCCRMCase.is_active)
Index('idx_gc_crm_case_status', GCCRMCase.status)
Index('idx_gc_crm_case_owner', GCCRMCase.case_owner_id)
Index('idx_gc_crm_case_customer', GCCRMCase.customer_id)
Index('idx_gc_crm_case_priority', GCCRMCase.priority)
Index('idx_gc_crm_case_created', GCCRMCase.created_on)

Index('idx_gc_crm_campaign_tenant_active', GCCRMCampaign.tenant_id, GCCRMCampaign.is_active)
Index('idx_gc_crm_campaign_status', GCCRMCampaign.status)
Index('idx_gc_crm_campaign_dates', GCCRMCampaign.start_date, GCCRMCampaign.end_date)
Index('idx_gc_crm_campaign_type', GCCRMCampaign.campaign_type)

# Enhanced indexes for new models
Index('idx_gc_crm_customer_score_entity', GCCRMCustomerScore.customer_id, GCCRMCustomerScore.score_type)
Index('idx_gc_crm_customer_score_value', GCCRMCustomerScore.score_value, GCCRMCustomerScore.calculated_date)
Index('idx_gc_crm_customer_score_current', GCCRMCustomerScore.is_current, GCCRMCustomerScore.score_type)

Index('idx_gc_crm_communication_entity', GCCRMCommunication.contact_id, GCCRMCommunication.communication_type)
Index('idx_gc_crm_communication_date', GCCRMCommunication.sent_date, GCCRMCommunication.received_date)
Index('idx_gc_crm_communication_sentiment', GCCRMCommunication.sentiment_score)

Index('idx_gc_crm_workflow_execution_status', GCCRMWorkflowExecution.execution_status, GCCRMWorkflowExecution.started_date)
Index('idx_gc_crm_workflow_execution_entity', GCCRMWorkflowExecution.entity_type, GCCRMWorkflowExecution.entity_id)

Index('idx_gc_crm_notification_recipient', GCCRMNotification.recipient_user_id, GCCRMNotification.status)
Index('idx_gc_crm_notification_type', GCCRMNotification.notification_type, GCCRMNotification.created_on)

Index('idx_gc_crm_event_log_timestamp', GCCRMEventLog.event_timestamp, GCCRMEventLog.event_type)
Index('idx_gc_crm_event_log_user', GCCRMEventLog.user_id, GCCRMEventLog.event_timestamp)
Index('idx_gc_crm_event_log_entity', GCCRMEventLog.entity_type, GCCRMEventLog.entity_id)

Index('idx_gc_crm_custom_field_value_entity', GCCRMCustomFieldValue.entity_id, GCCRMCustomFieldValue.custom_field_id)

Index('idx_gc_crm_document_attachment_entity', GCCRMDocumentAttachment.entity_type, GCCRMDocumentAttachment.entity_id)
Index('idx_gc_crm_document_attachment_type', GCCRMDocumentAttachment.document_type, GCCRMDocumentAttachment.upload_date)

Index('idx_gc_crm_webhook_delivery_status', GCCRMWebhookDelivery.delivery_status, GCCRMWebhookDelivery.scheduled_time)
Index('idx_gc_crm_webhook_delivery_endpoint', GCCRMWebhookDelivery.webhook_endpoint_id, GCCRMWebhookDelivery.attempted_time)

Index('idx_gc_crm_audit_tenant_date', GCCRMAuditLog.tenant_id, GCCRMAuditLog.created_on)
Index('idx_gc_crm_audit_entity', GCCRMAuditLog.entity_type, GCCRMAuditLog.entity_id)
Index('idx_gc_crm_audit_user', GCCRMAuditLog.user_id)