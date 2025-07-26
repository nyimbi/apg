"""
Product Lifecycle Management (PLM) Capability - Pydantic Views

Flask-AppBuilder UI views and Pydantic v2 models for PLM capability
following APG standards and patterns.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic.types import Annotated
from pydantic import AfterValidator
from enum import Enum


# Custom validators following APG patterns

def validate_product_name(v: str) -> str:
	"""Validate product name follows APG naming conventions"""
	if not v or len(v.strip()) < 3:
		raise ValueError("Product name must be at least 3 characters")
	if len(v) > 200:
		raise ValueError("Product name must not exceed 200 characters")
	if not v.replace(' ', '').replace('-', '').replace('_', '').isalnum():
		raise ValueError("Product name must contain only alphanumeric characters, spaces, hyphens, and underscores")
	return v.strip()

def validate_tenant_id(v: str) -> str:
	"""Validate tenant ID follows APG multi-tenant patterns"""
	if not v or len(v) < 8:
		raise ValueError("Tenant ID must be at least 8 characters")
	if not v.startswith('tenant_'):
		raise ValueError("Tenant ID must start with 'tenant_' prefix")
	return v

def validate_user_id(v: str) -> str:
	"""Validate user ID follows APG authentication patterns"""
	if not v or len(v) < 8:
		raise ValueError("User ID must be at least 8 characters")
	return v

def validate_positive_decimal(v: float) -> float:
	"""Validate positive decimal values"""
	if v < 0:
		raise ValueError("Value must be positive")
	return v

def validate_cost_value(v: float) -> float:
	"""Validate cost values in PLM"""
	if v < 0:
		raise ValueError("Cost must be non-negative")
	if v > 999999999.99:
		raise ValueError("Cost exceeds maximum allowed value")
	return round(v, 2)


# Enums for type safety

class ProductTypeEnum(str, Enum):
	"""Product type enumeration"""
	MANUFACTURED = "manufactured"
	PURCHASED = "purchased"
	VIRTUAL = "virtual"
	SERVICE = "service"
	KIT = "kit"
	RAW_MATERIAL = "raw_material"
	SUBASSEMBLY = "subassembly"
	FINISHED_GOOD = "finished_good"

class LifecyclePhaseEnum(str, Enum):
	"""Product lifecycle phase enumeration"""
	CONCEPT = "concept"
	DESIGN = "design"
	PROTOTYPE = "prototype"
	DEVELOPMENT = "development"
	TESTING = "testing"
	PRODUCTION = "production"
	ACTIVE = "active"
	MATURE = "mature"
	DECLINING = "declining"
	OBSOLETE = "obsolete"
	DISCONTINUED = "discontinued"

class ChangeTypeEnum(str, Enum):
	"""Engineering change type enumeration"""
	DESIGN = "design"
	PROCESS = "process"
	DOCUMENTATION = "documentation"
	COST_REDUCTION = "cost_reduction"
	QUALITY_IMPROVEMENT = "quality_improvement"
	SAFETY = "safety"
	REGULATORY = "regulatory"
	URGENT = "urgent"

class ChangeStatusEnum(str, Enum):
	"""Engineering change status enumeration"""
	DRAFT = "draft"
	SUBMITTED = "submitted"
	UNDER_REVIEW = "under_review"
	APPROVED = "approved"
	REJECTED = "rejected"
	IMPLEMENTED = "implemented"
	CANCELLED = "cancelled"

class CollaborationSessionTypeEnum(str, Enum):
	"""Collaboration session type enumeration"""
	DESIGN_REVIEW = "design_review"
	CHANGE_REVIEW = "change_review"
	BRAINSTORMING = "brainstorming"
	PROBLEM_SOLVING = "problem_solving"
	TRAINING = "training"
	CUSTOMER_MEETING = "customer_meeting"
	SUPPLIER_MEETING = "supplier_meeting"

class CollaborationStatusEnum(str, Enum):
	"""Collaboration session status enumeration"""
	SCHEDULED = "scheduled"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	CANCELLED = "cancelled"
	POSTPONED = "postponed"


# APG Configuration for all models
apg_model_config = ConfigDict(
	extra='forbid',
	validate_by_name=True,
	validate_by_alias=True,
	str_strip_whitespace=True,
	validate_assignment=True
)


# Pydantic v2 View Models

class PLMProductView(BaseModel):
	"""
	Product Lifecycle Management Product View Model
	
	Comprehensive product definition with APG integration patterns
	"""
	model_config = apg_model_config
	
	# Core Identity
	product_id: str = Field(default_factory=uuid7str, description="Unique product identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant isolation identifier")
	created_by: Annotated[str, AfterValidator(validate_user_id)] = Field(..., description="User who created the product")
	
	# Product Definition
	product_name: Annotated[str, AfterValidator(validate_product_name)] = Field(..., description="Product name following APG naming conventions")
	product_number: str = Field(..., min_length=3, max_length=50, description="Unique product number/SKU")
	product_description: Optional[str] = Field(None, max_length=2000, description="Detailed product description")
	product_type: ProductTypeEnum = Field(..., description="Product classification type")
	
	# Lifecycle Management
	lifecycle_phase: LifecyclePhaseEnum = Field(default=LifecyclePhaseEnum.CONCEPT, description="Current lifecycle phase")
	revision: str = Field(default="A", min_length=1, max_length=10, description="Product revision level")
	status: str = Field(default="active", description="Product status")
	
	# Financial Information
	target_cost: Annotated[float, AfterValidator(validate_cost_value)] = Field(default=0.0, description="Target manufacturing cost")
	current_cost: Annotated[float, AfterValidator(validate_cost_value)] = Field(default=0.0, description="Current actual cost")
	unit_of_measure: str = Field(default="each", description="Primary unit of measure")
	
	# APG Integration Fields
	manufacturing_status: Optional[str] = Field(None, description="Manufacturing capability status")
	digital_twin_id: Optional[str] = Field(None, description="Associated digital twin identifier")
	compliance_records: List[str] = Field(default_factory=list, description="Associated compliance record IDs")
	document_references: List[str] = Field(default_factory=list, description="Associated document IDs")
	
	# Metadata
	custom_attributes: Dict[str, Any] = Field(default_factory=dict, description="Flexible custom attributes")
	tags: List[str] = Field(default_factory=list, description="Product categorization tags")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	updated_by: Optional[str] = Field(None, description="User who last updated the product")
	
	@field_validator('product_number')
	@classmethod
	def validate_product_number(cls, v: str) -> str:
		"""Validate product number uniqueness and format"""
		if not v or len(v.strip()) < 3:
			raise ValueError("Product number must be at least 3 characters")
		# Remove spaces and convert to uppercase for consistency
		return v.strip().upper()
	
	@field_validator('tags')
	@classmethod
	def validate_tags(cls, v: List[str]) -> List[str]:
		"""Validate product tags"""
		if len(v) > 20:
			raise ValueError("Maximum 20 tags allowed")
		return [tag.strip().lower() for tag in v if tag.strip()]


class PLMProductStructureView(BaseModel):
	"""
	Product Structure (BOM) View Model
	
	Defines hierarchical product structures with APG manufacturing integration
	"""
	model_config = apg_model_config
	
	# Core Identity
	structure_id: str = Field(default_factory=uuid7str, description="Unique structure record identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant isolation identifier")
	
	# Structure Definition
	parent_product_id: str = Field(..., description="Parent product identifier")
	child_product_id: str = Field(..., description="Child component identifier")
	quantity: Annotated[float, AfterValidator(validate_positive_decimal)] = Field(..., description="Required quantity")
	unit_of_measure: str = Field(default="each", description="Quantity unit of measure")
	
	# Assembly Information
	reference_designator: Optional[str] = Field(None, max_length=50, description="Component reference designator")
	sequence_number: int = Field(default=1, ge=1, description="Assembly sequence order")
	is_critical: bool = Field(default=False, description="Critical component flag")
	
	# Manufacturing Integration
	manufacturing_bom_sync: bool = Field(default=True, description="Sync with manufacturing BOM")
	procurement_status: Optional[str] = Field(None, description="Component procurement status")
	supplier_info: Dict[str, Any] = Field(default_factory=dict, description="Primary supplier information")
	
	# Lifecycle Management
	effective_from: datetime = Field(default_factory=datetime.utcnow, description="Structure effective date")
	effective_to: Optional[datetime] = Field(None, description="Structure obsolescence date")
	revision: str = Field(default="A", description="Structure revision level")
	
	# Metadata
	notes: Optional[str] = Field(None, max_length=1000, description="Additional notes")
	created_by: Annotated[str, AfterValidator(validate_user_id)] = Field(..., description="User who created the structure")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	
	@field_validator('quantity')
	@classmethod
	def validate_quantity_positive(cls, v: float) -> float:
		"""Ensure quantity is positive"""
		if v <= 0:
			raise ValueError("Quantity must be greater than zero")
		return v


class PLMEngineeringChangeView(BaseModel):
	"""
	Engineering Change Management View Model
	
	Comprehensive change management with APG audit compliance integration
	"""
	model_config = apg_model_config
	
	# Core Identity
	change_id: str = Field(default_factory=uuid7str, description="Unique change identifier")
	change_number: str = Field(..., description="Human-readable change number")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant isolation identifier")
	
	# Change Definition
	change_title: str = Field(..., min_length=5, max_length=200, description="Change title")
	change_description: str = Field(..., min_length=10, max_length=2000, description="Detailed change description")
	change_type: ChangeTypeEnum = Field(..., description="Type of engineering change")
	change_category: str = Field(..., description="Change categorization")
	
	# Affected Items
	affected_products: List[str] = Field(..., min_items=1, description="List of affected product IDs")
	affected_documents: List[str] = Field(default_factory=list, description="List of affected document IDs")
	
	# Business Impact
	reason_for_change: str = Field(..., min_length=10, max_length=1000, description="Justification for change")
	business_impact: str = Field(..., description="Expected business impact")
	cost_impact: Annotated[float, AfterValidator(validate_cost_value)] = Field(default=0.0, description="Estimated cost impact")
	schedule_impact_days: int = Field(default=0, description="Schedule impact in days")
	
	# Workflow Management
	status: ChangeStatusEnum = Field(default=ChangeStatusEnum.DRAFT, description="Current change status")
	priority: str = Field(default="medium", description="Change priority level")
	urgency: str = Field(default="normal", description="Change urgency level")
	
	# Approval Process
	approvers: List[str] = Field(default_factory=list, description="Required approver user IDs")
	approved_by: List[str] = Field(default_factory=list, description="Users who have approved")
	approval_comments: Dict[str, str] = Field(default_factory=dict, description="Approval comments by user")
	
	# Implementation
	planned_implementation_date: Optional[datetime] = Field(None, description="Planned implementation date")
	actual_implementation_date: Optional[datetime] = Field(None, description="Actual implementation date")
	implementation_notes: Optional[str] = Field(None, max_length=2000, description="Implementation notes")
	
	# APG Integration
	audit_trail_id: Optional[str] = Field(None, description="Associated audit trail ID")
	compliance_verification: bool = Field(default=False, description="Compliance verification status")
	
	# Metadata
	created_by: Annotated[str, AfterValidator(validate_user_id)] = Field(..., description="Change requestor")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_by: Optional[str] = Field(None, description="User who last updated")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	@field_validator('change_number')
	@classmethod
	def validate_change_number(cls, v: str) -> str:
		"""Validate change number format"""
		if not v or len(v.strip()) < 5:
			raise ValueError("Change number must be at least 5 characters")
		return v.strip().upper()


class PLMProductConfigurationView(BaseModel):
	"""
	Product Configuration Management View Model
	
	Manages product variants and configurations with financial integration
	"""
	model_config = apg_model_config
	
	# Core Identity
	configuration_id: str = Field(default_factory=uuid7str, description="Unique configuration identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant isolation identifier")
	
	# Configuration Definition
	base_product_id: str = Field(..., description="Base product identifier")
	configuration_name: str = Field(..., min_length=3, max_length=100, description="Configuration name")
	configuration_description: Optional[str] = Field(None, max_length=1000, description="Configuration description")
	
	# Variant Management
	variant_attributes: Dict[str, Any] = Field(default_factory=dict, description="Configuration-specific attributes")
	option_codes: List[str] = Field(default_factory=list, description="Option codes for this configuration")
	feature_list: List[str] = Field(default_factory=list, description="Included feature list")
	
	# Pricing Information
	base_price: Annotated[float, AfterValidator(validate_cost_value)] = Field(default=0.0, description="Base configuration price")
	option_price_delta: Annotated[float, AfterValidator(validate_cost_value)] = Field(default=0.0, description="Additional cost for options")
	total_price: Annotated[float, AfterValidator(validate_cost_value)] = Field(default=0.0, description="Total configuration price")
	cost_delta: Annotated[float, AfterValidator(validate_cost_value)] = Field(default=0.0, description="Cost difference from base")
	
	# Manufacturing Integration
	manufacturing_complexity: str = Field(default="standard", description="Manufacturing complexity rating")
	lead_time_days: int = Field(default=30, ge=0, description="Manufacturing lead time")
	manufacturing_instructions: Optional[str] = Field(None, max_length=2000, description="Special manufacturing instructions")
	
	# Availability Management
	available_from: datetime = Field(default_factory=datetime.utcnow, description="Configuration availability start")
	available_to: Optional[datetime] = Field(None, description="Configuration availability end")
	orderable: bool = Field(default=True, description="Configuration can be ordered")
	
	# APG Integration
	financial_sync_status: Optional[str] = Field(None, description="Financial system sync status")
	manufacturing_sync_status: Optional[str] = Field(None, description="Manufacturing system sync status")
	
	# Metadata
	created_by: Annotated[str, AfterValidator(validate_user_id)] = Field(..., description="User who created configuration")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_by: Optional[str] = Field(None, description="User who last updated")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class PLMCollaborationSessionView(BaseModel):
	"""
	Real-time Collaboration Session View Model
	
	Manages collaborative design sessions with APG real-time collaboration integration
	"""
	model_config = apg_model_config
	
	# Core Identity
	session_id: str = Field(default_factory=uuid7str, description="Unique session identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant isolation identifier")
	
	# Session Definition
	session_name: str = Field(..., min_length=3, max_length=200, description="Session name")
	description: Optional[str] = Field(None, max_length=1000, description="Session description")
	session_type: CollaborationSessionTypeEnum = Field(..., description="Type of collaboration session")
	
	# Participant Management
	host_user_id: Annotated[str, AfterValidator(validate_user_id)] = Field(..., description="Session host user")
	participants: List[str] = Field(default_factory=list, description="Current participant user IDs")
	invited_users: List[str] = Field(default_factory=list, description="Invited user IDs")
	max_participants: int = Field(default=50, ge=1, le=100, description="Maximum participant limit")
	
	# Session Scheduling
	scheduled_start: datetime = Field(..., description="Scheduled session start time")
	scheduled_end: datetime = Field(..., description="Scheduled session end time")
	actual_start: Optional[datetime] = Field(None, description="Actual session start time")
	actual_end: Optional[datetime] = Field(None, description="Actual session end time")
	
	# Session Context
	products_discussed: List[str] = Field(default_factory=list, description="Products being discussed")
	documents_shared: List[str] = Field(default_factory=list, description="Shared document IDs")
	changes_proposed: List[str] = Field(default_factory=list, description="Proposed change IDs")
	
	# Session Features
	recording_enabled: bool = Field(default=False, description="Session recording enabled")
	whiteboard_enabled: bool = Field(default=True, description="Whiteboard collaboration enabled")
	file_sharing_enabled: bool = Field(default=True, description="File sharing enabled")
	3d_viewing_enabled: bool = Field(default=False, description="3D model viewing enabled")
	
	# Session Status
	status: CollaborationStatusEnum = Field(default=CollaborationStatusEnum.SCHEDULED, description="Current session status")
	session_notes: Optional[str] = Field(None, max_length=5000, description="Session notes and outcomes")
	action_items: List[Dict[str, Any]] = Field(default_factory=list, description="Action items from session")
	
	# APG Integration
	collaboration_room_id: Optional[str] = Field(None, description="APG collaboration room identifier")
	notification_sent: bool = Field(default=False, description="Notifications sent to participants")
	
	# Metadata
	created_by: Annotated[str, AfterValidator(validate_user_id)] = Field(..., description="Session creator")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_by: Optional[str] = Field(None, description="User who last updated")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	@field_validator('scheduled_end')
	@classmethod
	def validate_end_after_start(cls, v: datetime, info) -> datetime:
		"""Validate session end is after start"""
		if 'scheduled_start' in info.data and v <= info.data['scheduled_start']:
			raise ValueError("Session end must be after start time")
		return v


class PLMComplianceRecordView(BaseModel):
	"""
	Compliance Record View Model
	
	Tracks regulatory compliance with APG audit integration
	"""
	model_config = apg_model_config
	
	# Core Identity
	compliance_id: str = Field(default_factory=uuid7str, description="Unique compliance record identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant isolation identifier")
	
	# Compliance Definition
	product_id: str = Field(..., description="Associated product identifier")
	regulation_name: str = Field(..., min_length=3, max_length=200, description="Regulation or standard name")
	regulation_version: str = Field(..., description="Regulation version")
	compliance_type: str = Field(..., description="Type of compliance requirement")
	
	# Compliance Status
	status: str = Field(default="pending", description="Compliance status")
	certification_date: Optional[datetime] = Field(None, description="Date of certification")
	expiration_date: Optional[datetime] = Field(None, description="Certification expiration date")
	next_review_date: Optional[datetime] = Field(None, description="Next compliance review date")
	
	# Documentation
	certification_body: Optional[str] = Field(None, description="Certifying body or authority")
	certificate_number: Optional[str] = Field(None, description="Certificate or approval number")
	supporting_documents: List[str] = Field(default_factory=list, description="Supporting document IDs")
	compliance_notes: Optional[str] = Field(None, max_length=2000, description="Compliance notes and details")
	
	# APG Integration
	audit_trail_id: Optional[str] = Field(None, description="Associated audit trail ID")
	
	# Metadata
	created_by: Annotated[str, AfterValidator(validate_user_id)] = Field(..., description="Compliance record creator")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_by: Optional[str] = Field(None, description="User who last updated")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class PLMManufacturingIntegrationView(BaseModel):
	"""
	Manufacturing Integration View Model
	
	Manages PLM integration with APG manufacturing capabilities
	"""
	model_config = apg_model_config
	
	# Core Identity
	integration_id: str = Field(default_factory=uuid7str, description="Unique integration record identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant isolation identifier")
	
	# Integration Definition
	product_id: str = Field(..., description="Associated product identifier")
	manufacturing_part_number: str = Field(..., description="Manufacturing system part number")
	manufacturing_status: str = Field(default="active", description="Manufacturing status")
	
	# Synchronization Status
	last_sync_timestamp: Optional[datetime] = Field(None, description="Last synchronization timestamp")
	sync_status: str = Field(default="pending", description="Current synchronization status")
	sync_error_message: Optional[str] = Field(None, description="Last synchronization error")
	
	# Manufacturing Data
	manufacturing_data: Dict[str, Any] = Field(default_factory=dict, description="Manufacturing system data")
	bom_sync_required: bool = Field(default=True, description="BOM synchronization required")
	cost_sync_required: bool = Field(default=True, description="Cost synchronization required")
	
	# Metadata
	created_by: Annotated[str, AfterValidator(validate_user_id)] = Field(..., description="Integration record creator")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_by: Optional[str] = Field(None, description="User who last updated")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class PLMDigitalTwinBindingView(BaseModel):
	"""
	Digital Twin Binding View Model
	
	Manages binding between PLM products and APG digital twin marketplace
	"""
	model_config = apg_model_config
	
	# Core Identity
	binding_id: str = Field(default_factory=uuid7str, description="Unique binding identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant isolation identifier")
	
	# Binding Definition
	product_id: str = Field(..., description="PLM product identifier")
	digital_twin_id: str = Field(..., description="Digital twin marketplace identifier")
	binding_type: str = Field(default="product_twin", description="Type of digital twin binding")
	
	# Synchronization Configuration
	auto_sync_enabled: bool = Field(default=True, description="Automatic synchronization enabled")
	sync_frequency: str = Field(default="real_time", description="Synchronization frequency")
	last_sync_timestamp: Optional[datetime] = Field(None, description="Last synchronization timestamp")
	
	# Twin Configuration
	twin_properties: Dict[str, Any] = Field(default_factory=dict, description="Digital twin properties")
	simulation_enabled: bool = Field(default=False, description="Simulation capabilities enabled")
	iot_integration: bool = Field(default=False, description="IoT data integration enabled")
	
	# Status
	binding_status: str = Field(default="active", description="Binding status")
	sync_status: str = Field(default="synced", description="Current sync status")
	
	# Metadata
	created_by: Annotated[str, AfterValidator(validate_user_id)] = Field(..., description="Binding creator")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_by: Optional[str] = Field(None, description="User who last updated")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


# Dashboard and Analytics View Models

class PLMDashboardMetricsView(BaseModel):
	"""
	PLM Dashboard Metrics View Model
	
	Comprehensive dashboard metrics for PLM performance monitoring
	"""
	model_config = apg_model_config
	
	# Core Identity
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant isolation identifier")
	metrics_date: datetime = Field(default_factory=datetime.utcnow, description="Metrics calculation date")
	
	# Product Metrics
	total_products: int = Field(default=0, ge=0, description="Total number of products")
	active_products: int = Field(default=0, ge=0, description="Number of active products")
	products_in_development: int = Field(default=0, ge=0, description="Products in development phase")
	obsolete_products: int = Field(default=0, ge=0, description="Obsolete products")
	
	# Change Management Metrics
	open_changes: int = Field(default=0, ge=0, description="Open engineering changes")
	pending_approvals: int = Field(default=0, ge=0, description="Changes pending approval")
	implemented_changes_month: int = Field(default=0, ge=0, description="Changes implemented this month")
	average_approval_time_days: float = Field(default=0.0, ge=0, description="Average approval time in days")
	
	# Collaboration Metrics
	active_collaboration_sessions: int = Field(default=0, ge=0, description="Currently active collaboration sessions")
	scheduled_sessions_week: int = Field(default=0, ge=0, description="Sessions scheduled this week")
	average_session_duration_minutes: float = Field(default=0.0, ge=0, description="Average session duration")
	collaboration_participants_count: int = Field(default=0, ge=0, description="Unique collaboration participants")
	
	# Compliance Metrics
	compliant_products: int = Field(default=0, ge=0, description="Fully compliant products")
	pending_compliance_reviews: int = Field(default=0, ge=0, description="Pending compliance reviews")
	expiring_certifications_30days: int = Field(default=0, ge=0, description="Certifications expiring in 30 days")
	compliance_percentage: float = Field(default=0.0, ge=0, le=100, description="Overall compliance percentage")
	
	# Integration Metrics
	manufacturing_sync_success_rate: float = Field(default=0.0, ge=0, le=100, description="Manufacturing sync success rate")
	digital_twin_binding_count: int = Field(default=0, ge=0, description="Active digital twin bindings")
	failed_integrations_24h: int = Field(default=0, ge=0, description="Failed integrations in last 24 hours")
	
	# Performance Metrics
	system_performance_score: float = Field(default=100.0, ge=0, le=100, description="Overall system performance score")
	user_satisfaction_score: float = Field(default=0.0, ge=0, le=10, description="User satisfaction score")
	
	# Cost Metrics
	total_product_value: Annotated[float, AfterValidator(validate_cost_value)] = Field(default=0.0, description="Total value of all products")
	cost_savings_month: Annotated[float, AfterValidator(validate_cost_value)] = Field(default=0.0, description="Cost savings this month")
	
	# Metadata
	calculated_by: str = Field(default="system", description="User or system that calculated metrics")
	calculation_duration_ms: int = Field(default=0, ge=0, description="Metrics calculation time in milliseconds")


# Search and Filter Models

class PLMProductSearchView(BaseModel):
	"""
	Product Search View Model
	
	Advanced search capabilities for PLM products
	"""
	model_config = apg_model_config
	
	# Search Parameters
	search_text: Optional[str] = Field(None, max_length=200, description="Text search across name and description")
	product_types: List[ProductTypeEnum] = Field(default_factory=list, description="Filter by product types")
	lifecycle_phases: List[LifecyclePhaseEnum] = Field(default_factory=list, description="Filter by lifecycle phases")
	tags: List[str] = Field(default_factory=list, description="Filter by tags")
	
	# Date Filters
	created_from: Optional[datetime] = Field(None, description="Created date range start")
	created_to: Optional[datetime] = Field(None, description="Created date range end")
	updated_from: Optional[datetime] = Field(None, description="Updated date range start")
	updated_to: Optional[datetime] = Field(None, description="Updated date range end")
	
	# Cost Filters
	cost_min: Optional[float] = Field(None, ge=0, description="Minimum cost filter")
	cost_max: Optional[float] = Field(None, ge=0, description="Maximum cost filter")
	
	# Status Filters
	statuses: List[str] = Field(default_factory=list, description="Filter by status values")
	created_by_users: List[str] = Field(default_factory=list, description="Filter by creator users")
	
	# Pagination
	page: int = Field(default=1, ge=1, description="Page number")
	page_size: int = Field(default=20, ge=1, le=100, description="Records per page")
	
	# Sorting
	sort_by: str = Field(default="created_at", description="Sort field")
	sort_order: str = Field(default="desc", description="Sort order (asc/desc)")


# Module exports
__all__ = [
	"PLMProductView",
	"PLMProductStructureView", 
	"PLMEngineeringChangeView",
	"PLMProductConfigurationView",
	"PLMCollaborationSessionView",
	"PLMComplianceRecordView",
	"PLMManufacturingIntegrationView",
	"PLMDigitalTwinBindingView",
	"PLMDashboardMetricsView",
	"PLMProductSearchView",
	"ProductTypeEnum",
	"LifecyclePhaseEnum",
	"ChangeTypeEnum",
	"ChangeStatusEnum",
	"CollaborationSessionTypeEnum",
	"CollaborationStatusEnum"
]