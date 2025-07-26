"""
APG Capability Registry - Flask-AppBuilder Views

Pydantic v2 models and Flask-AppBuilder views for the capability registry
web interface with APG UI framework integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, field_validator, AfterValidator
from typing_extensions import Annotated

# APG UI Framework Configuration
APG_UI_CONFIG = {
	"theme": "apg-enterprise",
	"primary_color": "#2563eb",
	"secondary_color": "#64748b", 
	"success_color": "#16a34a",
	"warning_color": "#ea580c",
	"danger_color": "#dc2626",
	"font_family": "Inter, system-ui, sans-serif",
	"responsive_breakpoints": {
		"mobile": "640px",
		"tablet": "768px",
		"desktop": "1024px",
		"wide": "1280px"
	}
}

# =============================================================================
# Pydantic v2 Models for UI Data Transfer
# =============================================================================

class BaseUIModel(BaseModel):
	"""Base model for all UI data transfer objects."""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)

# Capability Models
class CapabilityListView(BaseUIModel):
	"""Capability list view model for table display."""
	capability_id: str = Field(..., description="Unique capability identifier")
	capability_code: str = Field(..., description="Capability code")
	capability_name: str = Field(..., description="Display name")
	description: Optional[str] = Field(None, description="Brief description")
	version: str = Field(..., description="Current version")
	category: str = Field(..., description="Capability category")
	status: str = Field(..., description="Current status")
	quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Quality score")
	popularity_score: float = Field(0.0, ge=0.0, le=1.0, description="Popularity score")
	usage_count: int = Field(0, ge=0, description="Usage count")
	created_at: Optional[datetime] = Field(None, description="Creation timestamp")
	
	@field_validator('quality_score', 'popularity_score')
	@classmethod
	def validate_scores(cls, v: float) -> float:
		return max(0.0, min(1.0, v))

class CapabilityDetailView(BaseUIModel):
	"""Detailed capability view model."""
	capability_id: str = Field(..., description="Unique capability identifier")
	capability_code: str = Field(..., description="Capability code")
	capability_name: str = Field(..., description="Display name")
	description: Optional[str] = Field(None, description="Detailed description")
	long_description: Optional[str] = Field(None, description="Extended description")
	version: str = Field(..., description="Current version")
	category: str = Field(..., description="Capability category")
	subcategory: Optional[str] = Field(None, description="Subcategory")
	status: str = Field(..., description="Current status")
	
	# Feature flags
	multi_tenant: bool = Field(True, description="Multi-tenant support")
	audit_enabled: bool = Field(True, description="Audit logging enabled")
	security_integration: bool = Field(True, description="Security integration")
	performance_optimized: bool = Field(False, description="Performance optimized")
	ai_enhanced: bool = Field(False, description="AI enhanced")
	
	# Business metadata
	target_users: List[str] = Field(default_factory=list, description="Target user types")
	business_value: Optional[str] = Field(None, description="Business value proposition")
	use_cases: List[str] = Field(default_factory=list, description="Use cases")
	industry_focus: List[str] = Field(default_factory=list, description="Industry focus")
	
	# Technical metadata
	composition_keywords: List[str] = Field(default_factory=list, description="Composition keywords")
	provides_services: List[str] = Field(default_factory=list, description="Provided services")
	data_models: List[str] = Field(default_factory=list, description="Data models")
	api_endpoints: List[str] = Field(default_factory=list, description="API endpoints")
	
	# File paths
	file_path: Optional[str] = Field(None, description="File path")
	module_path: Optional[str] = Field(None, description="Module path")
	documentation_path: Optional[str] = Field(None, description="Documentation path")
	repository_url: Optional[str] = Field(None, description="Repository URL")
	
	# Metrics
	complexity_score: float = Field(1.0, ge=0.0, description="Complexity score")
	quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Quality score")
	popularity_score: float = Field(0.0, ge=0.0, le=1.0, description="Popularity score")
	usage_count: int = Field(0, ge=0, description="Usage count")
	
	# Timestamps
	created_at: Optional[datetime] = Field(None, description="Creation timestamp")
	updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class CapabilityCreateForm(BaseUIModel):
	"""Form model for creating new capabilities."""
	capability_code: str = Field(..., min_length=3, max_length=100, description="Capability code")
	capability_name: str = Field(..., min_length=5, max_length=255, description="Display name")
	description: str = Field(..., min_length=10, description="Description")
	category: str = Field(..., description="Category")
	subcategory: Optional[str] = Field(None, description="Subcategory")
	version: str = Field("1.0.0", description="Initial version")
	
	# Optional fields
	target_users: List[str] = Field(default_factory=list, description="Target users")
	business_value: Optional[str] = Field(None, description="Business value")
	use_cases: List[str] = Field(default_factory=list, description="Use cases")
	industry_focus: List[str] = Field(default_factory=list, description="Industry focus")
	composition_keywords: List[str] = Field(default_factory=list, description="Keywords")
	
	@field_validator('capability_code')
	@classmethod
	def validate_capability_code(cls, v: str) -> str:
		# Must be uppercase, underscores allowed
		if not v.replace('_', '').replace('-', '').isalnum():
			raise ValueError('Capability code must contain only letters, numbers, underscores, and hyphens')
		return v.upper()

# Composition Models
class CompositionListView(BaseUIModel):
	"""Composition list view model."""
	composition_id: str = Field(..., description="Composition ID")
	name: str = Field(..., description="Composition name")
	description: Optional[str] = Field(None, description="Description")
	composition_type: str = Field(..., description="Composition type")
	version: str = Field(..., description="Version")
	validation_status: str = Field(..., description="Validation status")
	validation_score: float = Field(0.0, ge=0.0, le=1.0, description="Validation score")
	estimated_complexity: float = Field(1.0, ge=0.0, description="Complexity")
	estimated_cost: float = Field(0.0, ge=0.0, description="Estimated cost")
	capability_count: int = Field(0, ge=0, description="Number of capabilities")
	is_template: bool = Field(False, description="Is template")
	is_public: bool = Field(False, description="Is public")
	created_at: Optional[datetime] = Field(None, description="Creation timestamp")

class CompositionDetailView(BaseUIModel):
	"""Detailed composition view model."""
	composition_id: str = Field(..., description="Composition ID")
	name: str = Field(..., description="Composition name")
	description: Optional[str] = Field(None, description="Description")
	composition_type: str = Field(..., description="Composition type")
	version: str = Field(..., description="Version")
	
	# Validation information
	validation_status: str = Field(..., description="Validation status")
	validation_score: float = Field(0.0, ge=0.0, le=1.0, description="Validation score")
	validation_results: Dict[str, Any] = Field(default_factory=dict, description="Validation results")
	validation_errors: List[Dict[str, Any]] = Field(default_factory=list, description="Validation errors")
	validation_warnings: List[Dict[str, Any]] = Field(default_factory=list, description="Validation warnings")
	
	# Configuration
	configuration: Dict[str, Any] = Field(default_factory=dict, description="Configuration")
	environment_settings: Dict[str, Any] = Field(default_factory=dict, description="Environment settings")
	deployment_config: Dict[str, Any] = Field(default_factory=dict, description="Deployment config")
	
	# Performance and analytics
	estimated_complexity: float = Field(1.0, ge=0.0, description="Estimated complexity")
	estimated_cost: float = Field(0.0, ge=0.0, description="Estimated cost")
	estimated_deployment_time: Optional[str] = Field(None, description="Deployment time estimate")
	performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
	
	# Business metadata
	business_requirements: List[str] = Field(default_factory=list, description="Business requirements")
	compliance_requirements: List[str] = Field(default_factory=list, description="Compliance requirements")
	target_users: List[str] = Field(default_factory=list, description="Target users")
	
	# Sharing
	is_template: bool = Field(False, description="Is template")
	is_public: bool = Field(False, description="Is public")
	shared_with_tenants: List[str] = Field(default_factory=list, description="Shared tenants")
	
	# Timestamps
	created_at: Optional[datetime] = Field(None, description="Creation timestamp")
	updated_at: Optional[datetime] = Field(None, description="Update timestamp")
	
	# Capabilities in composition
	capabilities: List[Dict[str, Any]] = Field(default_factory=list, description="Composition capabilities")

class CompositionCreateForm(BaseUIModel):
	"""Form model for creating compositions."""
	name: str = Field(..., min_length=3, max_length=255, description="Composition name")
	description: str = Field(..., min_length=10, description="Description")
	composition_type: str = Field("custom", description="Composition type")
	capability_ids: List[str] = Field(..., min_items=1, description="Capability IDs")
	
	# Optional configuration
	industry_template: Optional[str] = Field(None, description="Industry template")
	deployment_strategy: Optional[str] = Field(None, description="Deployment strategy")
	business_requirements: List[str] = Field(default_factory=list, description="Business requirements")
	compliance_requirements: List[str] = Field(default_factory=list, description="Compliance requirements")
	target_users: List[str] = Field(default_factory=list, description="Target users")
	
	# Sharing settings
	is_template: bool = Field(False, description="Save as template")
	is_public: bool = Field(False, description="Make public")

# Search and Filter Models
class CapabilitySearchForm(BaseUIModel):
	"""Search form model for capabilities."""
	query: Optional[str] = Field(None, max_length=500, description="Search query")
	category: Optional[str] = Field(None, description="Category filter")
	status: Optional[str] = Field(None, description="Status filter")
	composition_keywords: List[str] = Field(default_factory=list, description="Keyword filters")
	min_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum quality")
	industry_focus: Optional[str] = Field(None, description="Industry focus")
	
	# Pagination
	page: int = Field(1, ge=1, description="Page number")
	per_page: int = Field(25, ge=5, le=100, description="Items per page")
	sort_by: str = Field("capability_name", description="Sort field")
	sort_order: str = Field("asc", description="Sort order")

class CompositionSearchForm(BaseUIModel):
	"""Search form model for compositions."""
	query: Optional[str] = Field(None, max_length=500, description="Search query")
	composition_type: Optional[str] = Field(None, description="Type filter")
	validation_status: Optional[str] = Field(None, description="Validation status filter")
	is_template: Optional[bool] = Field(None, description="Template filter")
	is_public: Optional[bool] = Field(None, description="Public filter")
	min_validation_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum validation score")
	
	# Pagination
	page: int = Field(1, ge=1, description="Page number")
	per_page: int = Field(25, ge=5, le=100, description="Items per page")
	sort_by: str = Field("name", description="Sort field")
	sort_order: str = Field("asc", description="Sort order")

# Dashboard Models
class RegistryDashboardData(BaseUIModel):
	"""Dashboard data model for registry overview."""
	# Summary statistics
	total_capabilities: int = Field(0, ge=0, description="Total capabilities")
	active_capabilities: int = Field(0, ge=0, description="Active capabilities")
	total_compositions: int = Field(0, ge=0, description="Total compositions")
	total_versions: int = Field(0, ge=0, description="Total versions")
	
	# Health metrics
	registry_health_score: float = Field(1.0, ge=0.0, le=1.0, description="Overall health")
	avg_quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Average quality")
	avg_popularity_score: float = Field(0.0, ge=0.0, le=1.0, description="Average popularity")
	
	# Recent activity
	recent_capabilities: List[CapabilityListView] = Field(default_factory=list, description="Recent capabilities")
	recent_compositions: List[CompositionListView] = Field(default_factory=list, description="Recent compositions")
	
	# Category breakdown
	category_stats: List[Dict[str, Any]] = Field(default_factory=list, description="Category statistics")
	
	# Performance metrics
	avg_discovery_time_ms: float = Field(0.0, ge=0.0, description="Average discovery time")
	avg_composition_time_ms: float = Field(0.0, ge=0.0, description="Average composition time")
	
	# Marketplace integration
	marketplace_enabled: bool = Field(True, description="Marketplace integration enabled")
	published_capabilities: int = Field(0, ge=0, description="Published capabilities")
	pending_submissions: int = Field(0, ge=0, description="Pending submissions")
	
	# Timestamps
	last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
	last_sync: Optional[datetime] = Field(None, description="Last marketplace sync")

# Version Models
class VersionListView(BaseUIModel):
	"""Version list view model."""
	version_id: str = Field(..., description="Version ID")
	version_number: str = Field(..., description="Version number")
	major_version: int = Field(..., ge=0, description="Major version")
	minor_version: int = Field(..., ge=0, description="Minor version")
	patch_version: int = Field(..., ge=0, description="Patch version")
	prerelease: Optional[str] = Field(None, description="Prerelease identifier")
	release_date: Optional[datetime] = Field(None, description="Release date")
	status: str = Field("active", description="Version status")
	backward_compatible: bool = Field(True, description="Backward compatible")
	quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Quality score")
	test_coverage: float = Field(0.0, ge=0.0, le=1.0, description="Test coverage")
	security_audit_passed: bool = Field(False, description="Security audit passed")

class VersionDetailView(BaseUIModel):
	"""Detailed version view model."""
	version_id: str = Field(..., description="Version ID")
	capability_id: str = Field(..., description="Capability ID")
	version_number: str = Field(..., description="Version number")
	major_version: int = Field(..., ge=0, description="Major version")
	minor_version: int = Field(..., ge=0, description="Minor version")
	patch_version: int = Field(..., ge=0, description="Patch version")
	prerelease: Optional[str] = Field(None, description="Prerelease identifier")
	build_metadata: Optional[str] = Field(None, description="Build metadata")
	
	# Release information
	release_date: Optional[datetime] = Field(None, description="Release date")
	release_notes: Optional[str] = Field(None, description="Release notes")
	breaking_changes: List[str] = Field(default_factory=list, description="Breaking changes")
	deprecations: List[str] = Field(default_factory=list, description="Deprecations")
	new_features: List[str] = Field(default_factory=list, description="New features")
	
	# Compatibility
	compatible_versions: List[str] = Field(default_factory=list, description="Compatible versions")
	incompatible_versions: List[str] = Field(default_factory=list, description="Incompatible versions")
	backward_compatible: bool = Field(True, description="Backward compatible")
	forward_compatible: bool = Field(False, description="Forward compatible")
	
	# Quality metrics
	quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Quality score")
	test_coverage: float = Field(0.0, ge=0.0, le=1.0, description="Test coverage")
	documentation_score: float = Field(0.0, ge=0.0, le=1.0, description="Documentation score")
	security_audit_passed: bool = Field(False, description="Security audit passed")
	
	# Lifecycle
	status: str = Field("active", description="Version status")
	support_level: str = Field("full", description="Support level")
	end_of_life_date: Optional[datetime] = Field(None, description="End of life date")
	
	# API changes
	api_changes: Dict[str, Any] = Field(default_factory=dict, description="API changes")
	migration_path: Dict[str, Any] = Field(default_factory=dict, description="Migration path")
	upgrade_instructions: Optional[str] = Field(None, description="Upgrade instructions")

# Marketplace Models
class MarketplaceListingView(BaseUIModel):
	"""Marketplace listing view model."""
	listing_id: str = Field(..., description="Listing ID")
	title: str = Field(..., description="Listing title")
	description: str = Field(..., description="Short description")
	capability_code: str = Field(..., description="Capability code")
	version: str = Field(..., description="Current version")
	author_name: str = Field(..., description="Author name")
	author_organization: Optional[str] = Field(None, description="Author organization")
	license_type: str = Field(..., description="License type")
	pricing_model: str = Field("free", description="Pricing model")
	price: float = Field(0.0, ge=0.0, description="Price")
	quality_level: str = Field("stable", description="Quality level")
	download_count: int = Field(0, ge=0, description="Download count")
	rating: float = Field(0.0, ge=0.0, le=5.0, description="Average rating")
	rating_count: int = Field(0, ge=0, description="Number of ratings")
	tags: List[str] = Field(default_factory=list, description="Tags")
	categories: List[str] = Field(default_factory=list, description="Categories")
	marketplace_status: str = Field("published", description="Marketplace status")
	published_at: Optional[datetime] = Field(None, description="Publication date")

# Analytics Models
class CapabilityAnalyticsView(BaseUIModel):
	"""Capability analytics view model."""
	capability_id: str = Field(..., description="Capability ID")
	capability_name: str = Field(..., description="Capability name")
	
	# Usage metrics
	total_usage_count: int = Field(0, ge=0, description="Total usage")
	monthly_usage: int = Field(0, ge=0, description="Monthly usage")
	weekly_usage: int = Field(0, ge=0, description="Weekly usage")
	daily_usage: int = Field(0, ge=0, description="Daily usage")
	
	# Performance metrics
	avg_response_time_ms: float = Field(0.0, ge=0.0, description="Average response time")
	avg_memory_usage_mb: float = Field(0.0, ge=0.0, description="Average memory usage")
	avg_cpu_usage_pct: float = Field(0.0, ge=0.0, le=100.0, description="Average CPU usage")
	error_rate_pct: float = Field(0.0, ge=0.0, le=100.0, description="Error rate")
	
	# User metrics
	unique_users: int = Field(0, ge=0, description="Unique users")
	active_compositions: int = Field(0, ge=0, description="Active compositions")
	
	# Trends
	usage_trend: str = Field("stable", description="Usage trend")
	performance_trend: str = Field("stable", description="Performance trend")
	
	# Time series data for charts
	usage_history: List[Dict[str, Any]] = Field(default_factory=list, description="Usage history")
	performance_history: List[Dict[str, Any]] = Field(default_factory=list, description="Performance history")

# Response Models
class UIResponse(BaseUIModel):
	"""Standard UI response model."""
	success: bool = Field(..., description="Success status")
	message: str = Field(..., description="Response message")
	data: Optional[Dict[str, Any]] = Field(None, description="Response data")
	errors: List[str] = Field(default_factory=list, description="Error messages")
	warnings: List[str] = Field(default_factory=list, description="Warning messages")
	
	# UI-specific fields
	redirect_url: Optional[str] = Field(None, description="Redirect URL")
	refresh_page: bool = Field(False, description="Refresh page flag")
	show_notification: bool = Field(True, description="Show notification")
	notification_type: str = Field("info", description="Notification type")

# Form Validation Helpers
def validate_capability_code(code: str) -> str:
	"""Validate capability code format."""
	if not code:
		raise ValueError("Capability code is required")
	
	# Must be alphanumeric with underscores/hyphens
	cleaned = code.replace('_', '').replace('-', '')
	if not cleaned.isalnum():
		raise ValueError("Capability code must contain only letters, numbers, underscores, and hyphens")
	
	return code.upper()

def validate_version_number(version: str) -> str:
	"""Validate semantic version number."""
	import re
	
	pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
	if not re.match(pattern, version):
		raise ValueError("Version must follow semantic versioning (MAJOR.MINOR.PATCH)")
	
	return version

# Custom field validators using Annotated
ValidatedCapabilityCode = Annotated[str, AfterValidator(validate_capability_code)]
ValidatedVersionNumber = Annotated[str, AfterValidator(validate_version_number)]

# Export all view models
__all__ = [
	# Base models
	"BaseUIModel",
	"UIResponse",
	
	# Capability models
	"CapabilityListView",
	"CapabilityDetailView", 
	"CapabilityCreateForm",
	"CapabilityAnalyticsView",
	
	# Composition models
	"CompositionListView",
	"CompositionDetailView",
	"CompositionCreateForm",
	
	# Version models
	"VersionListView",
	"VersionDetailView",
	
	# Search models
	"CapabilitySearchForm",
	"CompositionSearchForm",
	
	# Dashboard models
	"RegistryDashboardData",
	
	# Marketplace models
	"MarketplaceListingView",
	
	# Validation helpers
	"ValidatedCapabilityCode",
	"ValidatedVersionNumber",
	"validate_capability_code",
	"validate_version_number",
	
	# UI configuration
	"APG_UI_CONFIG"
]