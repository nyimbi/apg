"""
APG Customer Relationship Management - Data Models

Comprehensive Pydantic models for the revolutionary CRM capability providing
10x superior data modeling compared to industry leaders.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, EmailStr, validator, root_validator
from pydantic.config import ConfigDict
from uuid_extensions import uuid7str


# ================================
# Core Configuration Models
# ================================

class CRMCapabilityConfig(BaseModel):
	"""CRM capability configuration model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Business Configuration
	default_lead_score_threshold: float = Field(default=70.0, ge=0.0, le=100.0)
	default_opportunity_probability: float = Field(default=50.0, ge=0.0, le=100.0)
	customer_health_score_enabled: bool = Field(default=True)
	ai_recommendations_enabled: bool = Field(default=True)
	predictive_analytics_enabled: bool = Field(default=True)
	
	# Integration Configuration
	email_integration_enabled: bool = Field(default=True)
	calendar_integration_enabled: bool = Field(default=True)
	social_media_monitoring_enabled: bool = Field(default=True)
	document_management_enabled: bool = Field(default=True)
	
	# Performance Configuration
	max_records_per_page: int = Field(default=100, ge=10, le=1000)
	cache_ttl_seconds: int = Field(default=300, ge=60, le=3600)
	background_job_timeout: int = Field(default=3600, ge=300, le=7200)


# ================================
# Enumeration Types
# ================================

class RecordStatus(str, Enum):
	"""General record status"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	ARCHIVED = "archived"
	DELETED = "deleted"


class ContactType(str, Enum):
	"""Contact type classification"""
	PROSPECT = "prospect"
	CUSTOMER = "customer"
	PARTNER = "partner"
	VENDOR = "vendor"
	EMPLOYEE = "employee"
	OTHER = "other"


class AccountType(str, Enum):
	"""Account type classification"""
	PROSPECT = "prospect"
	CUSTOMER = "customer"
	PARTNER = "partner"
	VENDOR = "vendor"
	COMPETITOR = "competitor"
	OTHER = "other"


class LeadSource(str, Enum):
	"""Lead source types"""
	WEBSITE = "website"
	EMAIL_CAMPAIGN = "email_campaign"
	SOCIAL_MEDIA = "social_media"
	REFERRAL = "referral"
	COLD_CALL = "cold_call"
	TRADE_SHOW = "trade_show"
	WEBINAR = "webinar"
	ADVERTISEMENT = "advertisement"
	PARTNER = "partner"
	OTHER = "other"


class LeadStatus(str, Enum):
	"""Lead status progression"""
	NEW = "new"
	CONTACTED = "contacted"
	QUALIFIED = "qualified"
	UNQUALIFIED = "unqualified"
	NURTURING = "nurturing"
	CONVERTED = "converted"
	LOST = "lost"


class OpportunityStage(str, Enum):
	"""Opportunity sales stages"""
	PROSPECTING = "prospecting"
	QUALIFICATION = "qualification"
	NEEDS_ANALYSIS = "needs_analysis"
	VALUE_PROPOSITION = "value_proposition"
	NEGOTIATION = "negotiation"
	CLOSED_WON = "closed_won"
	CLOSED_LOST = "closed_lost"


class ActivityType(str, Enum):
	"""Activity types"""
	CALL = "call"
	EMAIL = "email"
	MEETING = "meeting"
	TASK = "task"
	NOTE = "note"
	DEMO = "demo"
	PROPOSAL = "proposal"
	FOLLOW_UP = "follow_up"
	OTHER = "other"


class Priority(str, Enum):
	"""Priority levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	URGENT = "urgent"


# ================================
# Base Models
# ================================

class BaseAuditModel(BaseModel):
	"""Base model with audit fields"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique identifier")
	tenant_id: str = Field(..., description="Multi-tenant identifier")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: str = Field(..., description="User who created the record")
	updated_by: Optional[str] = Field(None, description="User who last updated the record")
	version: int = Field(default=1, ge=1, description="Record version for optimistic locking")
	status: RecordStatus = Field(default=RecordStatus.ACTIVE, description="Record status")


class Address(BaseModel):
	"""Address information model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	street_address: Optional[str] = Field(None, max_length=200)
	street_address_2: Optional[str] = Field(None, max_length=200)
	city: Optional[str] = Field(None, max_length=100)
	state_province: Optional[str] = Field(None, max_length=100)
	postal_code: Optional[str] = Field(None, max_length=20)
	country: Optional[str] = Field(None, max_length=100)
	address_type: Optional[str] = Field(None, max_length=50)
	is_primary: bool = Field(default=False)


class PhoneNumber(BaseModel):
	"""Phone number model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	number: str = Field(..., max_length=50)
	type: Optional[str] = Field(None, max_length=20)  # mobile, work, home, fax
	is_primary: bool = Field(default=False)
	country_code: Optional[str] = Field(None, max_length=5)


# ================================
# Contact Management Models
# ================================

class CRMContact(BaseAuditModel):
	"""Comprehensive contact model"""
	
	# Basic Information
	first_name: str = Field(..., min_length=1, max_length=100, description="Contact first name")
	last_name: str = Field(..., min_length=1, max_length=100, description="Contact last name")
	email: Optional[EmailStr] = Field(None, description="Primary email address")
	phone: Optional[str] = Field(None, max_length=50, description="Primary phone number")
	
	# Professional Information
	job_title: Optional[str] = Field(None, max_length=200, description="Job title")
	company: Optional[str] = Field(None, max_length=200, description="Company name")
	account_id: Optional[str] = Field(None, description="Associated account ID")
	
	# Classification
	contact_type: ContactType = Field(default=ContactType.PROSPECT, description="Contact type")
	lead_source: Optional[LeadSource] = Field(None, description="Original lead source")
	
	# AI and Analytics
	lead_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="AI-generated lead score")
	customer_health_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Customer health score")
	
	# Addresses and Phone Numbers
	addresses: List[Address] = Field(default_factory=list, description="Contact addresses")
	phone_numbers: List[PhoneNumber] = Field(default_factory=list, description="Phone numbers")
	
	# Metadata
	notes: Optional[str] = Field(None, description="General notes")
	tags: List[str] = Field(default_factory=list, description="Contact tags")
	
	@property
	def full_name(self) -> str:
		"""Generate full name from components"""
		return f"{self.first_name} {self.last_name}"


# ================================
# Account Management Models
# ================================

class CRMAccount(BaseAuditModel):
	"""Comprehensive account model"""
	
	# Basic Information
	account_name: str = Field(..., min_length=1, max_length=200, description="Account name")
	account_type: AccountType = Field(default=AccountType.PROSPECT, description="Account type")
	
	# Business Information
	industry: Optional[str] = Field(None, max_length=100, description="Industry")
	annual_revenue: Optional[Decimal] = Field(None, ge=0, description="Annual revenue")
	employee_count: Optional[int] = Field(None, ge=0, description="Employee count")
	
	# Contact Information
	website: Optional[str] = Field(None, max_length=500, description="Website URL")
	main_phone: Optional[str] = Field(None, max_length=50, description="Main phone number")
	addresses: List[Address] = Field(default_factory=list, description="Account addresses")
	
	# Hierarchy
	parent_account_id: Optional[str] = Field(None, description="Parent account ID")
	
	# Assignment
	account_owner_id: str = Field(..., description="Primary account owner user ID")
	
	# AI and Analytics
	account_health_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Account health score")
	
	# Metadata
	description: Optional[str] = Field(None, description="Account description")
	tags: List[str] = Field(default_factory=list, description="Account tags")


# ================================
# Lead Management Models
# ================================

class CRMLead(BaseAuditModel):
	"""Comprehensive lead model"""
	
	# Basic Information
	first_name: str = Field(..., min_length=1, max_length=100, description="Lead first name")
	last_name: str = Field(..., min_length=1, max_length=100, description="Lead last name")
	company: Optional[str] = Field(None, max_length=200, description="Company name")
	
	# Contact Information
	email: Optional[EmailStr] = Field(None, description="Email address")
	phone: Optional[str] = Field(None, max_length=50, description="Phone number")
	
	# Lead Classification
	lead_source: LeadSource = Field(..., description="Lead source")
	lead_status: LeadStatus = Field(default=LeadStatus.NEW, description="Lead status")
	lead_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Lead score")
	
	# Qualification Information
	budget: Optional[Decimal] = Field(None, ge=0, description="Budget amount")
	timeline: Optional[str] = Field(None, max_length=200, description="Purchase timeline")
	
	# Assignment
	owner_id: Optional[str] = Field(None, description="Lead owner user ID")
	
	# Conversion Information
	is_converted: bool = Field(default=False, description="Conversion status")
	converted_date: Optional[datetime] = Field(None, description="Conversion date")
	converted_contact_id: Optional[str] = Field(None, description="Converted contact ID")
	converted_account_id: Optional[str] = Field(None, description="Converted account ID")
	converted_opportunity_id: Optional[str] = Field(None, description="Converted opportunity ID")
	
	# Metadata
	description: Optional[str] = Field(None, description="Lead description")
	tags: List[str] = Field(default_factory=list, description="Lead tags")
	
	@property
	def full_name(self) -> str:
		"""Generate full name"""
		return f"{self.first_name} {self.last_name}"


# ================================
# Opportunity Management Models
# ================================

class CRMOpportunity(BaseAuditModel):
	"""Comprehensive opportunity model"""
	
	# Basic Information
	opportunity_name: str = Field(..., min_length=1, max_length=200, description="Opportunity name")
	description: Optional[str] = Field(None, description="Opportunity description")
	
	# Financial Information
	amount: Decimal = Field(..., ge=0, description="Opportunity amount")
	probability: float = Field(..., ge=0.0, le=100.0, description="Close probability percentage")
	expected_revenue: Optional[Decimal] = Field(None, ge=0, description="Expected revenue")
	
	# Timeline
	close_date: date = Field(..., description="Expected close date")
	
	# Stage and Status
	stage: OpportunityStage = Field(default=OpportunityStage.PROSPECTING, description="Sales stage")
	is_closed: bool = Field(default=False, description="Closed status")
	is_won: Optional[bool] = Field(None, description="Won/lost status")
	
	# Relationships
	account_id: str = Field(..., description="Associated account ID")
	primary_contact_id: Optional[str] = Field(None, description="Primary contact ID")
	
	# Assignment
	owner_id: str = Field(..., description="Opportunity owner user ID")
	
	# AI and Analytics
	win_probability_ai: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI-calculated win probability")
	
	# Metadata
	notes: Optional[str] = Field(None, description="Opportunity notes")
	tags: List[str] = Field(default_factory=list, description="Opportunity tags")
	
	@validator('expected_revenue', always=True)
	def calculate_expected_revenue(cls, v, values):
		"""Calculate expected revenue if not provided"""
		if v is None and 'amount' in values and 'probability' in values:
			return values['amount'] * (values['probability'] / 100)
		return v


# ================================
# Activity Models
# ================================

class CRMActivity(BaseAuditModel):
	"""Comprehensive activity model"""
	
	# Basic Information
	subject: str = Field(..., min_length=1, max_length=200, description="Activity subject")
	activity_type: ActivityType = Field(..., description="Activity type")
	description: Optional[str] = Field(None, description="Activity description")
	
	# Scheduling
	start_datetime: datetime = Field(..., description="Activity start date/time")
	end_datetime: Optional[datetime] = Field(None, description="Activity end date/time")
	
	# Status and Priority
	status: str = Field(default="scheduled", max_length=20, description="Activity status")
	priority: Priority = Field(default=Priority.MEDIUM, description="Activity priority")
	is_completed: bool = Field(default=False, description="Completion status")
	
	# Relationships
	related_to_type: str = Field(..., max_length=50, description="Related record type")
	related_to_id: str = Field(..., description="Related record ID")
	
	# Assignment
	assigned_to_id: str = Field(..., description="Assigned user ID")
	
	# Metadata
	notes: Optional[str] = Field(None, description="Activity notes")
	tags: List[str] = Field(default_factory=list, description="Activity tags")


# ================================
# Campaign Models
# ================================

class CRMCampaign(BaseAuditModel):
	"""Marketing campaign model"""
	
	# Basic Information
	campaign_name: str = Field(..., min_length=1, max_length=200, description="Campaign name")
	campaign_type: str = Field(..., max_length=50, description="Campaign type")
	description: Optional[str] = Field(None, description="Campaign description")
	
	# Timeline
	start_date: date = Field(..., description="Campaign start date")
	end_date: Optional[date] = Field(None, description="Campaign end date")
	
	# Budget and Performance
	budget: Optional[Decimal] = Field(None, ge=0, description="Campaign budget")
	actual_cost: Optional[Decimal] = Field(None, ge=0, description="Actual cost")
	expected_leads: Optional[int] = Field(None, ge=0, description="Expected leads")
	actual_leads: Optional[int] = Field(None, ge=0, description="Actual leads")
	
	# Status
	status: str = Field(default="planned", max_length=20, description="Campaign status")
	is_active: bool = Field(default=False, description="Active status")
	
	# Metadata
	tags: List[str] = Field(default_factory=list, description="Campaign tags")


# ================================
# Export Models
# ================================

__all__ = [
	# Configuration
	"CRMCapabilityConfig",
	
	# Enums
	"RecordStatus", "ContactType", "AccountType", "LeadSource", "LeadStatus",
	"OpportunityStage", "ActivityType", "Priority",
	
	# Base Models
	"BaseAuditModel", "Address", "PhoneNumber",
	
	# Core Business Models
	"CRMContact", "CRMAccount", "CRMLead", "CRMOpportunity", "CRMActivity", "CRMCampaign"
]