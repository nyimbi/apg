"""
APG Vendor Management - Pydantic Models
Comprehensive data models for AI-powered vendor lifecycle management

Author: Nyimbi Odero (nyimbi@gmail.com)
Copyright: © 2025 Datacraft (www.datacraft.co.ke)
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator, ConfigDict
from pydantic.types import EmailStr, HttpUrl
from uuid_extensions import uuid7str


# ============================================================================
# BASE CONFIGURATION
# ============================================================================

class VMBaseModel(BaseModel):
	"""Base model for all Vendor Management models"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		use_enum_values=True
	)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class VendorType(str, Enum):
	"""Vendor type enumeration"""
	SUPPLIER = "supplier"
	SERVICE_PROVIDER = "service_provider"  
	CONTRACTOR = "contractor"
	CONSULTANT = "consultant"
	PARTNER = "partner"
	DISTRIBUTOR = "distributor"
	MANUFACTURER = "manufacturer"
	RESELLER = "reseller"


class VendorStatus(str, Enum):
	"""Vendor status enumeration"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	PENDING = "pending"
	SUSPENDED = "suspended"
	TERMINATED = "terminated"
	UNDER_REVIEW = "under_review"


class VendorLifecycleStage(str, Enum):
	"""Vendor lifecycle stage enumeration"""
	PROSPECTIVE = "prospective"
	QUALIFIED = "qualified"
	APPROVED = "approved"
	ACTIVE = "active"
	MATURE = "mature"
	DECLINING = "declining"
	SUNSET = "sunset"


class VendorSizeClassification(str, Enum):
	"""Vendor size classification enumeration"""
	ENTERPRISE = "enterprise"
	LARGE = "large"
	MEDIUM = "medium"
	SMALL = "small"
	MICRO = "micro"
	STARTUP = "startup"


class StrategicImportance(str, Enum):
	"""Strategic importance enumeration"""
	CRITICAL = "critical"
	STRATEGIC = "strategic"
	IMPORTANT = "important"
	STANDARD = "standard"
	COMMODITY = "commodity"


class RiskSeverity(str, Enum):
	"""Risk severity enumeration"""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	NEGLIGIBLE = "negligible"


class RiskImpact(str, Enum):
	"""Risk impact enumeration"""
	CATASTROPHIC = "catastrophic"
	MAJOR = "major"
	MODERATE = "moderate"
	MINOR = "minor"
	NEGLIGIBLE = "negligible"


class MitigationStatus(str, Enum):
	"""Risk mitigation status enumeration"""
	IDENTIFIED = "identified"
	ASSESSED = "assessed"
	PLANNED = "planned"
	IN_PROGRESS = "in_progress"
	IMPLEMENTED = "implemented"
	MONITORED = "monitored"
	RESOLVED = "resolved"


class ContractStatus(str, Enum):
	"""Contract status enumeration"""
	DRAFT = "draft"
	ACTIVE = "active"
	EXPIRED = "expired"
	TERMINATED = "terminated"
	SUSPENDED = "suspended"
	RENEWED = "renewed"


class CommunicationType(str, Enum):
	"""Communication type enumeration"""
	EMAIL = "email"
	MEETING = "meeting"
	CALL = "call"
	MESSAGE = "message"
	DOCUMENT = "document"
	VIDEO_CONFERENCE = "video_conference"


class CommunicationDirection(str, Enum):
	"""Communication direction enumeration"""
	INBOUND = "inbound"
	OUTBOUND = "outbound"
	INTERNAL = "internal"


class PortalUserStatus(str, Enum):
	"""Portal user status enumeration"""
	PENDING_VERIFICATION = "pending_verification"
	ACTIVE = "active"
	INACTIVE = "inactive"
	SUSPENDED = "suspended"
	LOCKED = "locked"


class ComplianceStatus(str, Enum):
	"""Compliance status enumeration"""
	COMPLIANT = "compliant"
	NON_COMPLIANT = "non_compliant"
	PARTIALLY_COMPLIANT = "partially_compliant"
	UNDER_REVIEW = "under_review"
	EXEMPTED = "exempted"


# ============================================================================
# CORE VENDOR MODELS
# ============================================================================

class VMVendor(VMBaseModel):
	"""Core vendor model with AI-powered intelligence"""
	
	# Primary Keys & Identification
	id: str = Field(default_factory=uuid7str, description="Unique vendor identifier")
	tenant_id: UUID = Field(..., description="Multi-tenant isolation ID")
	vendor_code: str = Field(..., max_length=50, description="Unique vendor code")
	
	# Basic Information
	name: str = Field(..., max_length=200, description="Vendor name")
	legal_name: Optional[str] = Field(None, max_length=250, description="Legal entity name")
	display_name: Optional[str] = Field(None, max_length=200, description="Display name")
	
	# Classification
	vendor_type: VendorType = Field(default=VendorType.SUPPLIER, description="Vendor type")
	category: str = Field(..., max_length=100, description="Primary vendor category")
	subcategory: Optional[str] = Field(None, max_length=100, description="Vendor subcategory")
	industry: Optional[str] = Field(None, max_length=100, description="Industry sector")
	size_classification: VendorSizeClassification = Field(
		default=VendorSizeClassification.MEDIUM, 
		description="Size classification"
	)
	
	# Status & Lifecycle
	status: VendorStatus = Field(default=VendorStatus.ACTIVE, description="Current status")
	lifecycle_stage: VendorLifecycleStage = Field(
		default=VendorLifecycleStage.QUALIFIED, 
		description="Lifecycle stage"
	)
	onboarding_date: Optional[datetime] = Field(None, description="Onboarding date")
	activation_date: Optional[datetime] = Field(None, description="Activation date")
	deactivation_date: Optional[datetime] = Field(None, description="Deactivation date")
	
	# AI-Powered Intelligence Scores
	intelligence_score: Decimal = Field(
		default=Decimal("85.00"),
		ge=Decimal("0"),
		le=Decimal("100"),
		description="AI-calculated intelligence score"
	)
	performance_score: Decimal = Field(
		default=Decimal("85.00"),
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Performance score"
	)
	risk_score: Decimal = Field(
		default=Decimal("25.00"),
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Risk score"
	)
	relationship_score: Decimal = Field(
		default=Decimal("75.00"),
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Relationship score"
	)
	
	# Predictive Analytics (JSONB)
	predicted_performance: Dict[str, Any] = Field(
		default_factory=dict,
		description="AI predictions for future performance"
	)
	risk_predictions: Dict[str, Any] = Field(
		default_factory=dict,
		description="AI risk predictions and scenarios"
	)
	optimization_recommendations: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="AI-generated optimization recommendations"
	)
	ai_insights: Dict[str, Any] = Field(
		default_factory=dict,
		description="Additional AI insights"
	)
	
	# Contact Information
	primary_contact_id: Optional[UUID] = Field(None, description="Primary contact ID")
	email: Optional[EmailStr] = Field(None, description="Primary email")
	phone: Optional[str] = Field(None, max_length=50, description="Primary phone")
	website: Optional[HttpUrl] = Field(None, description="Website URL")
	
	# Address Information
	address_line1: Optional[str] = Field(None, max_length=255, description="Address line 1")
	address_line2: Optional[str] = Field(None, max_length=255, description="Address line 2")
	city: Optional[str] = Field(None, max_length=100, description="City")
	state_province: Optional[str] = Field(None, max_length=100, description="State/Province")
	postal_code: Optional[str] = Field(None, max_length=20, description="Postal code")
	country: Optional[str] = Field(None, max_length=100, description="Country")
	
	# Financial Information
	credit_rating: Optional[str] = Field(None, max_length=10, description="Credit rating")
	payment_terms: str = Field(default="Net 30", max_length=50, description="Payment terms")
	currency: str = Field(default="USD", max_length=3, description="Primary currency")
	tax_id: Optional[str] = Field(None, max_length=50, description="Tax identification")
	duns_number: Optional[str] = Field(None, max_length=15, description="DUNS number")
	
	# Operational Details (JSONB arrays)
	capabilities: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Vendor capabilities"
	)
	certifications: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Certifications and accreditations"
	)
	geographic_coverage: List[str] = Field(
		default_factory=list,
		description="Geographic coverage areas"
	)
	capacity_metrics: Dict[str, Any] = Field(
		default_factory=dict,
		description="Capacity and resource metrics"
	)
	
	# Strategic Information
	strategic_importance: StrategicImportance = Field(
		default=StrategicImportance.STANDARD,
		description="Strategic importance"
	)
	preferred_vendor: bool = Field(default=False, description="Preferred vendor flag")
	strategic_partner: bool = Field(default=False, description="Strategic partner flag")
	diversity_category: Optional[str] = Field(
		None, 
		max_length=100, 
		description="Diversity category"
	)
	
	# Multi-tenant & Sharing
	shared_vendor: bool = Field(default=False, description="Shared across tenants")
	sharing_tenants: List[UUID] = Field(
		default_factory=list,
		description="Tenants with access to shared vendor"
	)
	
	# APG Integration & Audit
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: UUID = Field(..., description="Creator user ID")
	updated_by: UUID = Field(..., description="Last updater user ID")
	version: int = Field(default=1, description="Version number")
	is_active: bool = Field(default=True, description="Active flag")
	
	@validator('vendor_code')
	def validate_vendor_code(cls, v):
		"""Validate vendor code format"""
		if not v or len(v.strip()) == 0:
			raise ValueError('Vendor code cannot be empty')
		if not v.replace('-', '').replace('_', '').isalnum():
			raise ValueError('Vendor code must be alphanumeric with optional hyphens/underscores')
		return v.upper()
	
	@validator('intelligence_score', 'performance_score', 'risk_score', 'relationship_score')
	def validate_scores(cls, v):
		"""Validate score range"""
		if v < 0 or v > 100:
			raise ValueError('Scores must be between 0 and 100')
		return v
	
	@root_validator
	def validate_dates(cls, values):
		"""Validate date relationships"""
		onboarding = values.get('onboarding_date')
		activation = values.get('activation_date')
		deactivation = values.get('deactivation_date')
		
		if onboarding and activation and activation < onboarding:
			raise ValueError('Activation date cannot be before onboarding date')
		
		if activation and deactivation and deactivation < activation:
			raise ValueError('Deactivation date cannot be before activation date')
		
		return values


class VMPerformance(VMBaseModel):
	"""Vendor performance tracking model"""
	
	# Primary Keys
	id: str = Field(default_factory=uuid7str, description="Unique performance record ID")
	tenant_id: UUID = Field(..., description="Multi-tenant isolation ID")
	vendor_id: str = Field(..., description="Associated vendor ID")
	
	# Performance Period
	measurement_period: str = Field(..., description="Measurement period")
	start_date: datetime = Field(..., description="Period start date")
	end_date: datetime = Field(..., description="Period end date")
	
	# Core Performance Metrics
	overall_score: Decimal = Field(
		...,
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Overall performance score"
	)
	quality_score: Decimal = Field(
		...,
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Quality performance score"
	)
	delivery_score: Decimal = Field(
		...,
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Delivery performance score"
	)
	cost_score: Decimal = Field(
		...,
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Cost performance score"
	)
	service_score: Decimal = Field(
		...,
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Service performance score"
	)
	innovation_score: Decimal = Field(
		default=Decimal("0"),
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Innovation score"
	)
	
	# Detailed Performance Metrics
	on_time_delivery_rate: Decimal = Field(
		default=Decimal("0"),
		ge=Decimal("0"),
		le=Decimal("100"),
		description="On-time delivery percentage"
	)
	quality_rejection_rate: Decimal = Field(
		default=Decimal("0"),
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Quality rejection percentage"
	)
	cost_variance: Decimal = Field(default=Decimal("0"), description="Cost variance")
	service_level_achievement: Decimal = Field(
		default=Decimal("0"),
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Service level achievement percentage"
	)
	
	# Volume & Financial Metrics
	order_volume: Decimal = Field(default=Decimal("0"), description="Total order volume")
	order_count: int = Field(default=0, description="Number of orders")
	total_spend: Decimal = Field(default=Decimal("0"), description="Total spend amount")
	average_order_value: Decimal = Field(default=Decimal("0"), description="Average order value")
	
	# AI Insights & Analytics
	performance_trends: Dict[str, Any] = Field(
		default_factory=dict,
		description="Performance trend analysis"
	)
	improvement_recommendations: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="AI improvement recommendations"
	)
	benchmark_comparison: Dict[str, Any] = Field(
		default_factory=dict,
		description="Benchmark comparison data"
	)
	
	# Risk Indicators
	risk_indicators: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Performance risk indicators"
	)
	risk_score: Decimal = Field(
		default=Decimal("0"),
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Performance risk score"
	)
	mitigation_actions: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Risk mitigation actions"
	)
	
	# Data Quality & Validation
	data_completeness: Decimal = Field(
		default=Decimal("100"),
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Data completeness percentage"
	)
	data_sources: List[str] = Field(
		default_factory=list,
		description="Data source identifiers"
	)
	calculation_method: str = Field(
		default="weighted_average",
		max_length=100,
		description="Score calculation method"
	)
	
	# APG Integration & Audit
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: UUID = Field(..., description="Creator user ID")
	updated_by: UUID = Field(..., description="Last updater user ID")
	
	@root_validator
	def validate_period_dates(cls, values):
		"""Validate performance period dates"""
		start_date = values.get('start_date')
		end_date = values.get('end_date')
		
		if start_date and end_date and end_date <= start_date:
			raise ValueError('End date must be after start date')
		
		return values


class VMRisk(VMBaseModel):
	"""Vendor risk management model"""
	
	# Primary Keys
	id: str = Field(default_factory=uuid7str, description="Unique risk ID")
	tenant_id: UUID = Field(..., description="Multi-tenant isolation ID")
	vendor_id: str = Field(..., description="Associated vendor ID")
	
	# Risk Classification
	risk_type: str = Field(..., max_length=50, description="Risk type")
	risk_category: str = Field(..., max_length=100, description="Risk category")
	severity: RiskSeverity = Field(default=RiskSeverity.MEDIUM, description="Risk severity")
	probability: Decimal = Field(
		...,
		ge=Decimal("0"),
		le=Decimal("1"),
		description="Risk probability (0-1)"
	)
	impact: RiskImpact = Field(default=RiskImpact.MODERATE, description="Risk impact level")
	
	# Risk Details
	title: str = Field(..., max_length=200, description="Risk title")
	description: str = Field(..., description="Risk description")
	root_cause: Optional[str] = Field(None, description="Root cause analysis")
	potential_impact: Optional[str] = Field(None, description="Potential impact description")
	
	# Risk Scoring
	overall_risk_score: Decimal = Field(
		...,
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Overall risk score"
	)
	financial_impact: Decimal = Field(default=Decimal("0"), description="Financial impact estimate")
	operational_impact: int = Field(
		default=5,
		ge=1,
		le=10,
		description="Operational impact (1-10 scale)"
	)
	reputational_impact: int = Field(
		default=5,
		ge=1,
		le=10,
		description="Reputational impact (1-10 scale)"
	)
	
	# AI Predictions
	predicted_likelihood: Optional[Decimal] = Field(
		None,
		ge=Decimal("0"),
		le=Decimal("1"),
		description="AI-predicted likelihood"
	)
	time_horizon: Optional[int] = Field(None, description="Prediction time horizon (days)")
	confidence_level: Optional[Decimal] = Field(
		None,
		ge=Decimal("0"),
		le=Decimal("1"),
		description="Prediction confidence level"
	)
	ai_risk_factors: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="AI-identified risk factors"
	)
	
	# Mitigation & Response
	mitigation_strategy: Optional[str] = Field(None, description="Mitigation strategy")
	mitigation_actions: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Mitigation actions"
	)
	mitigation_status: MitigationStatus = Field(
		default=MitigationStatus.IDENTIFIED,
		description="Mitigation status"
	)
	target_residual_risk: Optional[Decimal] = Field(
		None,
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Target residual risk score"
	)
	
	# Monitoring & Review
	monitoring_frequency: str = Field(
		default="monthly",
		max_length=50,
		description="Monitoring frequency"
	)
	last_assessment: datetime = Field(
		default_factory=datetime.utcnow,
		description="Last assessment date"
	)
	next_assessment: Optional[datetime] = Field(None, description="Next assessment date")
	assigned_to: Optional[UUID] = Field(None, description="Assigned user ID")
	
	# Status & Lifecycle
	status: str = Field(default="active", max_length=50, description="Risk status")
	identified_date: datetime = Field(
		default_factory=datetime.utcnow,
		description="Risk identification date"
	)
	resolved_date: Optional[datetime] = Field(None, description="Risk resolution date")
	
	# APG Integration & Audit
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: UUID = Field(..., description="Creator user ID")
	updated_by: UUID = Field(..., description="Last updater user ID")


class VMContract(VMBaseModel):
	"""Contract management integration model"""
	
	# Primary Keys
	id: str = Field(default_factory=uuid7str, description="Unique contract ID")
	tenant_id: UUID = Field(..., description="Multi-tenant isolation ID")
	vendor_id: str = Field(..., description="Associated vendor ID")
	
	# Contract Identification
	contract_number: str = Field(..., max_length=100, description="Contract number")
	contract_name: str = Field(..., max_length=200, description="Contract name")
	contract_type: str = Field(..., max_length=50, description="Contract type")
	
	# Contract Dates
	effective_date: datetime = Field(..., description="Contract effective date")
	expiration_date: datetime = Field(..., description="Contract expiration date")
	renewal_date: Optional[datetime] = Field(None, description="Contract renewal date")
	notice_period_days: int = Field(default=30, description="Notice period in days")
	
	# Financial Terms
	contract_value: Decimal = Field(..., description="Total contract value")
	currency: str = Field(default="USD", max_length=3, description="Contract currency")
	payment_terms: Optional[str] = Field(None, max_length=100, description="Payment terms")
	pricing_model: Optional[str] = Field(None, max_length=50, description="Pricing model")
	
	# Contract Status
	status: ContractStatus = Field(default=ContractStatus.ACTIVE, description="Contract status")
	auto_renewal: bool = Field(default=False, description="Auto-renewal flag")
	
	# Document Management
	document_id: Optional[UUID] = Field(None, description="Document management system ID")
	contract_terms: Dict[str, Any] = Field(
		default_factory=dict,
		description="AI-extracted contract terms"
	)
	key_clauses: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Key contract clauses"
	)
	
	# Performance & Compliance
	performance_requirements: Dict[str, Any] = Field(
		default_factory=dict,
		description="Performance requirements"
	)
	compliance_requirements: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Compliance requirements"
	)
	sla_requirements: Dict[str, Any] = Field(
		default_factory=dict,
		description="SLA requirements"
	)
	
	# AI Analysis
	risk_analysis: Dict[str, Any] = Field(
		default_factory=dict,
		description="AI risk analysis"
	)
	optimization_opportunities: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="AI optimization opportunities"
	)
	
	# APG Integration & Audit
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: UUID = Field(..., description="Creator user ID")
	updated_by: UUID = Field(..., description="Last updater user ID")
	
	@root_validator
	def validate_contract_dates(cls, values):
		"""Validate contract date relationships"""
		effective = values.get('effective_date')
		expiration = values.get('expiration_date')
		renewal = values.get('renewal_date')
		
		if effective and expiration and expiration <= effective:
			raise ValueError('Expiration date must be after effective date')
		
		if renewal and expiration and renewal < expiration:
			raise ValueError('Renewal date should not be before expiration date')
		
		return values


class VMCommunication(VMBaseModel):
	"""Vendor communication & collaboration model"""
	
	# Primary Keys
	id: str = Field(default_factory=uuid7str, description="Unique communication ID")
	tenant_id: UUID = Field(..., description="Multi-tenant isolation ID")
	vendor_id: str = Field(..., description="Associated vendor ID")
	
	# Communication Details
	communication_type: CommunicationType = Field(..., description="Communication type")
	subject: Optional[str] = Field(None, max_length=500, description="Communication subject")
	content: Optional[str] = Field(None, description="Communication content")
	communication_date: datetime = Field(..., description="Communication date")
	
	# Participants
	internal_participants: List[UUID] = Field(
		default_factory=list,
		description="Internal participant user IDs"
	)
	vendor_participants: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Vendor participant details"
	)
	
	# Communication Metadata
	direction: CommunicationDirection = Field(..., description="Communication direction")
	priority: str = Field(default="normal", max_length=20, description="Priority level")
	status: str = Field(default="sent", max_length=50, description="Communication status")
	
	# Related Records
	related_project_id: Optional[UUID] = Field(None, description="Related project ID")
	related_contract_id: Optional[UUID] = Field(None, description="Related contract ID")
	related_issue_id: Optional[UUID] = Field(None, description="Related issue ID")
	
	# Attachments & References
	attachments: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Communication attachments"
	)
	references: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Communication references"
	)
	
	# AI Analysis
	sentiment_score: Optional[Decimal] = Field(
		None,
		ge=Decimal("-1"),
		le=Decimal("1"),
		description="AI sentiment analysis score (-1 to 1)"
	)
	topic_categories: List[str] = Field(
		default_factory=list,
		description="AI-identified topic categories"
	)
	action_items: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="AI-extracted action items"
	)
	ai_summary: Optional[str] = Field(None, description="AI-generated summary")
	
	# APG Integration & Audit
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: UUID = Field(..., description="Creator user ID")
	updated_by: UUID = Field(..., description="Last updater user ID")


# ============================================================================
# VENDOR INTELLIGENCE & ANALYTICS MODELS
# ============================================================================

class VMIntelligence(VMBaseModel):
	"""Vendor intelligence model"""
	
	# Primary Keys
	id: str = Field(default_factory=uuid7str, description="Unique intelligence ID")
	tenant_id: UUID = Field(..., description="Multi-tenant isolation ID")
	vendor_id: str = Field(..., description="Associated vendor ID")
	
	# Intelligence Generation
	intelligence_date: datetime = Field(
		default_factory=datetime.utcnow,
		description="Intelligence generation date"
	)
	model_version: str = Field(..., max_length=50, description="AI model version")
	confidence_score: Decimal = Field(
		...,
		ge=Decimal("0"),
		le=Decimal("1"),
		description="Intelligence confidence score"
	)
	
	# Intelligence Insights
	behavior_patterns: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Vendor behavior patterns"
	)
	predictive_insights: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Predictive insights"
	)
	performance_forecasts: Dict[str, Any] = Field(
		default_factory=dict,
		description="Performance forecast data"
	)
	risk_assessments: Dict[str, Any] = Field(
		default_factory=dict,
		description="Risk assessment data"
	)
	
	# Market Intelligence
	market_position: Dict[str, Any] = Field(
		default_factory=dict,
		description="Market position analysis"
	)
	competitive_analysis: Dict[str, Any] = Field(
		default_factory=dict,
		description="Competitive analysis"
	)
	pricing_intelligence: Dict[str, Any] = Field(
		default_factory=dict,
		description="Pricing intelligence"
	)
	
	# Optimization Recommendations
	improvement_opportunities: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Improvement opportunities"
	)
	cost_optimization: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Cost optimization recommendations"
	)
	relationship_optimization: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Relationship optimization recommendations"
	)
	
	# Data Sources & Quality
	data_sources: List[str] = Field(
		default_factory=list,
		description="Data source identifiers"
	)
	data_quality_score: Decimal = Field(
		default=Decimal("1.0"),
		ge=Decimal("0"),
		le=Decimal("1"),
		description="Data quality score"
	)
	analysis_scope: Dict[str, Any] = Field(
		default_factory=dict,
		description="Analysis scope parameters"
	)
	
	# Intelligence Validity
	valid_from: datetime = Field(
		default_factory=datetime.utcnow,
		description="Intelligence valid from date"
	)
	valid_until: datetime = Field(..., description="Intelligence valid until date")
	
	# APG Integration & Audit
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	created_by: UUID = Field(..., description="Creator user ID")
	
	@root_validator
	def validate_validity_period(cls, values):
		"""Validate intelligence validity period"""
		valid_from = values.get('valid_from')
		valid_until = values.get('valid_until')
		
		if valid_from and valid_until and valid_until <= valid_from:
			raise ValueError('Valid until date must be after valid from date')
		
		# Set default valid_until if not provided (30 days from now)
		if not valid_until:
			values['valid_until'] = datetime.utcnow() + timedelta(days=30)
		
		return values


class VMBenchmark(VMBaseModel):
	"""Vendor benchmarking model"""
	
	# Primary Keys
	id: str = Field(default_factory=uuid7str, description="Unique benchmark ID")
	tenant_id: UUID = Field(..., description="Multi-tenant isolation ID")
	vendor_id: str = Field(..., description="Associated vendor ID")
	
	# Benchmark Configuration
	benchmark_type: str = Field(..., max_length=50, description="Benchmark type")
	benchmark_category: str = Field(..., max_length=100, description="Benchmark category")
	measurement_period: str = Field(..., max_length=50, description="Measurement period")
	
	# Benchmark Data
	vendor_value: Decimal = Field(..., description="Vendor metric value")
	benchmark_value: Decimal = Field(..., description="Benchmark reference value")
	percentile_rank: Optional[int] = Field(
		None,
		ge=1,
		le=100,
		description="Percentile rank (1-100)"
	)
	
	# Benchmark Context
	peer_group_size: Optional[int] = Field(None, description="Peer group size")
	data_points: Optional[int] = Field(None, description="Number of data points")
	measurement_unit: Optional[str] = Field(None, max_length=50, description="Measurement unit")
	
	# Performance Analysis
	performance_gap: Optional[Decimal] = Field(None, description="Performance gap")
	improvement_potential: Optional[Decimal] = Field(None, description="Improvement potential")
	recommendations: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Improvement recommendations"
	)
	
	# APG Integration & Audit
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: UUID = Field(..., description="Creator user ID")


# ============================================================================
# VENDOR PORTAL & EXTERNAL ACCESS MODELS
# ============================================================================

class VMPortalUser(VMBaseModel):
	"""Vendor portal user model"""
	
	# Primary Keys
	id: str = Field(default_factory=uuid7str, description="Unique portal user ID")
	tenant_id: UUID = Field(..., description="Multi-tenant isolation ID")
	vendor_id: str = Field(..., description="Associated vendor ID")
	
	# User Information
	email: EmailStr = Field(..., description="User email address")
	first_name: str = Field(..., max_length=100, description="First name")
	last_name: str = Field(..., max_length=100, description="Last name")
	job_title: Optional[str] = Field(None, max_length=150, description="Job title")
	phone: Optional[str] = Field(None, max_length=50, description="Phone number")
	
	# Authentication
	password_hash: Optional[str] = Field(None, max_length=255, description="Password hash")
	mfa_enabled: bool = Field(default=True, description="MFA enabled flag")
	mfa_secret: Optional[str] = Field(None, max_length=255, description="MFA secret")
	
	# Account Status
	status: PortalUserStatus = Field(
		default=PortalUserStatus.PENDING_VERIFICATION,
		description="User account status"
	)
	email_verified: bool = Field(default=False, description="Email verification status")
	last_login: Optional[datetime] = Field(None, description="Last login timestamp")
	failed_login_attempts: int = Field(default=0, description="Failed login attempt count")
	account_locked_until: Optional[datetime] = Field(None, description="Account lock expiration")
	
	# Permissions & Access
	role: str = Field(default="vendor_portal_user", max_length=50, description="User role")
	permissions: List[str] = Field(
		default_factory=list,
		description="User permissions"
	)
	access_restrictions: Dict[str, Any] = Field(
		default_factory=dict,
		description="Access restrictions"
	)
	
	# Security Profile
	allowed_ip_ranges: List[str] = Field(
		default_factory=list,
		description="Allowed IP address ranges"
	)
	require_device_registration: bool = Field(
		default=True,
		description="Device registration requirement"
	)
	session_timeout_minutes: int = Field(default=30, description="Session timeout in minutes")
	
	# Portal Preferences
	language: str = Field(default="en", max_length=10, description="Preferred language")
	timezone: str = Field(default="UTC", max_length=100, description="Preferred timezone")
	notification_preferences: Dict[str, Any] = Field(
		default_factory=dict,
		description="Notification preferences"
	)
	
	# APG Integration & Audit
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: UUID = Field(..., description="Creator user ID")
	updated_by: UUID = Field(..., description="Last updater user ID")
	
	@validator('email')
	def validate_email_uniqueness(cls, v):
		"""Validate email format (basic validation, uniqueness enforced at DB level)"""
		return v.lower()


class VMPortalSession(VMBaseModel):
	"""Vendor portal session model"""
	
	# Primary Keys
	id: str = Field(default_factory=uuid7str, description="Unique session ID")
	user_id: str = Field(..., description="Portal user ID")
	vendor_id: str = Field(..., description="Associated vendor ID")
	
	# Session Details
	session_token: str = Field(..., max_length=255, description="Session token")
	csrf_token: str = Field(..., max_length=255, description="CSRF token")
	device_fingerprint: Optional[str] = Field(None, max_length=255, description="Device fingerprint")
	
	# Session Timing
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
	expires_at: datetime = Field(..., description="Session expiration time")
	last_activity: datetime = Field(
		default_factory=datetime.utcnow,
		description="Last activity timestamp"
	)
	
	# Request Context
	ip_address: Optional[str] = Field(None, description="Client IP address")
	user_agent: Optional[str] = Field(None, description="Client user agent")
	
	# Security Context
	security_context: Dict[str, Any] = Field(
		default_factory=dict,
		description="Security context data"
	)
	
	@root_validator
	def validate_session_timing(cls, values):
		"""Validate session timing"""
		created_at = values.get('created_at')
		expires_at = values.get('expires_at')
		
		if created_at and expires_at and expires_at <= created_at:
			raise ValueError('Session expiration must be after creation time')
		
		# Set default expiration if not provided (30 minutes from creation)
		if not expires_at:
			values['expires_at'] = datetime.utcnow() + timedelta(minutes=30)
		
		return values


# ============================================================================
# AUDIT & COMPLIANCE MODELS
# ============================================================================

class VMAuditLog(VMBaseModel):
	"""Vendor audit log model"""
	
	# Primary Keys
	id: str = Field(default_factory=uuid7str, description="Unique audit log ID")
	tenant_id: UUID = Field(..., description="Multi-tenant isolation ID")
	
	# Event Details
	event_type: str = Field(..., max_length=100, description="Event type")
	event_category: str = Field(..., max_length=50, description="Event category")
	event_severity: str = Field(default="info", max_length=20, description="Event severity")
	
	# Resource Information
	resource_type: str = Field(..., max_length=50, description="Resource type")
	resource_id: str = Field(..., description="Resource ID")
	vendor_id: Optional[str] = Field(None, description="Associated vendor ID")
	
	# User & Session Context
	user_id: Optional[UUID] = Field(None, description="User ID")
	session_id: Optional[UUID] = Field(None, description="Session ID")
	user_type: str = Field(default="internal", max_length=50, description="User type")
	
	# Event Data
	event_data: Dict[str, Any] = Field(
		default_factory=dict,
		description="Event data payload"
	)
	old_values: Dict[str, Any] = Field(
		default_factory=dict,
		description="Previous values before change"
	)
	new_values: Dict[str, Any] = Field(
		default_factory=dict,
		description="New values after change"
	)
	
	# Request Context
	ip_address: Optional[str] = Field(None, description="Client IP address")
	user_agent: Optional[str] = Field(None, description="Client user agent")
	request_method: Optional[str] = Field(None, max_length=10, description="HTTP request method")
	request_path: Optional[str] = Field(None, description="Request path")
	
	# Compliance & Audit
	compliance_tags: List[str] = Field(
		default_factory=list,
		description="Compliance framework tags"
	)
	business_impact: Dict[str, Any] = Field(
		default_factory=dict,
		description="Business impact assessment"
	)
	
	# Timing
	event_timestamp: datetime = Field(
		default_factory=datetime.utcnow,
		description="Event timestamp"
	)
	
	# Data Retention (for GDPR compliance)
	retention_until: Optional[datetime] = Field(None, description="Data retention expiration")


class VMCompliance(VMBaseModel):
	"""Vendor compliance tracking model"""
	
	# Primary Keys
	id: str = Field(default_factory=uuid7str, description="Unique compliance record ID")
	tenant_id: UUID = Field(..., description="Multi-tenant isolation ID")
	vendor_id: str = Field(..., description="Associated vendor ID")
	
	# Compliance Framework
	framework: str = Field(..., max_length=100, description="Compliance framework")
	requirement: str = Field(..., max_length=200, description="Specific requirement")
	requirement_type: str = Field(..., max_length=50, description="Requirement type")
	
	# Compliance Status
	status: ComplianceStatus = Field(
		default=ComplianceStatus.COMPLIANT,
		description="Compliance status"
	)
	compliance_score: Decimal = Field(
		default=Decimal("100"),
		ge=Decimal("0"),
		le=Decimal("100"),
		description="Compliance score percentage"
	)
	
	# Evidence & Documentation
	evidence_documents: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Evidence documents"
	)
	compliance_notes: Optional[str] = Field(None, description="Compliance notes")
	
	# Review & Monitoring
	last_review_date: Optional[datetime] = Field(None, description="Last review date")
	next_review_date: datetime = Field(..., description="Next review date")
	review_frequency: str = Field(
		default="annual",
		max_length=50,
		description="Review frequency"
	)
	assigned_reviewer: Optional[UUID] = Field(None, description="Assigned reviewer user ID")
	
	# Violations & Issues
	violations: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Compliance violations"
	)
	remediation_actions: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Remediation actions"
	)
	
	# APG Integration & Audit
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: UUID = Field(..., description="Creator user ID")
	updated_by: UUID = Field(..., description="Last updater user ID")
	
	@root_validator
	def validate_review_dates(cls, values):
		"""Validate review date relationships"""
		last_review = values.get('last_review_date')
		next_review = values.get('next_review_date')
		
		if last_review and next_review and next_review <= last_review:
			raise ValueError('Next review date must be after last review date')
		
		return values


# ============================================================================
# RESPONSE & VIEW MODELS
# ============================================================================

class VendorListResponse(VMBaseModel):
	"""Vendor list response model"""
	
	vendors: List[VMVendor] = Field(..., description="List of vendors")
	total_count: int = Field(..., description="Total vendor count")
	page: int = Field(default=1, description="Current page number")
	page_size: int = Field(default=50, description="Page size")
	has_next: bool = Field(default=False, description="Has next page flag")


class VendorPerformanceSummary(VMBaseModel):
	"""Vendor performance summary model"""
	
	vendor_id: str = Field(..., description="Vendor ID")
	vendor_name: str = Field(..., description="Vendor name")
	current_performance_score: Decimal = Field(..., description="Current performance score")
	performance_trend: str = Field(..., description="Performance trend")
	performance_rating: str = Field(..., description="Performance rating")
	last_assessment_date: datetime = Field(..., description="Last assessment date")
	
	# Key metrics
	quality_score: Decimal = Field(..., description="Quality score")
	delivery_score: Decimal = Field(..., description="Delivery score")
	cost_score: Decimal = Field(..., description="Cost score")
	service_score: Decimal = Field(..., description="Service score")
	
	# Risk indicators
	active_risks: int = Field(default=0, description="Number of active risks")
	high_risks: int = Field(default=0, description="Number of high risks")
	risk_score: Decimal = Field(..., description="Overall risk score")


class VendorIntelligenceSummary(VMBaseModel):
	"""Vendor intelligence summary model"""
	
	vendor_id: str = Field(..., description="Vendor ID")
	intelligence_score: Decimal = Field(..., description="Intelligence score")
	confidence_level: Decimal = Field(..., description="Confidence level")
	last_generated: datetime = Field(..., description="Last intelligence generation date")
	
	# Key insights
	behavior_patterns: List[str] = Field(
		default_factory=list,
		description="Key behavior patterns"
	)
	predictive_insights: List[str] = Field(
		default_factory=list,
		description="Key predictive insights"
	)
	optimization_opportunities: List[str] = Field(
		default_factory=list,
		description="Key optimization opportunities"
	)


# ============================================================================
# AI-SPECIFIC MODELS
# ============================================================================

class VendorAIDecision(VMBaseModel):
	"""AI decision model for autonomous vendor management"""
	
	decision_id: str = Field(default_factory=uuid7str, description="Unique decision ID")
	vendor_id: str = Field(..., description="Associated vendor ID")
	decision_type: str = Field(..., description="Type of decision")
	
	# Decision Details
	recommendation: Dict[str, Any] = Field(..., description="AI recommendation")
	confidence_score: Decimal = Field(
		...,
		ge=Decimal("0"),
		le=Decimal("1"),
		description="Decision confidence score"
	)
	risk_assessment: Dict[str, Any] = Field(..., description="Risk assessment")
	autonomous_approved: bool = Field(..., description="Autonomous approval flag")
	
	# Decision Context
	reasoning: str = Field(..., description="AI reasoning for decision")
	expected_outcomes: List[Dict[str, Any]] = Field(..., description="Expected outcomes")
	implementation_plan: Dict[str, Any] = Field(..., description="Implementation plan")
	monitoring_requirements: List[str] = Field(..., description="Monitoring requirements")
	
	# Execution Details
	execution_result: Optional[Dict[str, Any]] = Field(None, description="Execution result")
	executed_at: Optional[datetime] = Field(None, description="Execution timestamp")
	
	# Audit
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class VendorOptimizationPlan(VMBaseModel):
	"""Vendor optimization plan model"""
	
	vendor_id: str = Field(..., description="Associated vendor ID")
	optimization_objectives: List[str] = Field(..., description="Optimization objectives")
	
	# Plan Details
	current_baseline: Dict[str, Any] = Field(..., description="Current baseline metrics")
	recommended_actions: List[Dict[str, Any]] = Field(..., description="Recommended actions")
	predicted_outcomes: Dict[str, Any] = Field(..., description="Predicted outcomes")
	implementation_plan: Dict[str, Any] = Field(..., description="Implementation plan")
	
	# Success Metrics
	success_metrics: List[Dict[str, Any]] = Field(..., description="Success metrics")
	monitoring_schedule: Dict[str, Any] = Field(..., description="Monitoring schedule")
	
	# Metadata
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	expires_at: datetime = Field(..., description="Plan expiration timestamp")
