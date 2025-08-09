"""
Multi-Tenant Management (MTen) View Models

Company: Datacraft
Copyright: © 2025
Author: Nyimbi Odero

Pydantic v2 request/response models for Flask-AppBuilder integration
following CLAUDE.md standards: modern typing, strict validation.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Annotated
from decimal import Decimal

from pydantic import BaseModel, Field, ConfigDict, AfterValidator, HttpUrl

from .models import (
	TenantStatus, TenantTier, CloudProvider, 
	Tenant, TenantMetrics, OptimizationRecommendation,
	validate_tenant_name, validate_domain
)


class TenantCreateRequest(BaseModel):
	"""Request model for creating new tenants"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	name: Annotated[str, AfterValidator(validate_tenant_name)] = Field(
		..., description="Unique tenant name (slug)", min_length=2, max_length=64
	)
	display_name: str = Field(..., description="Human-readable tenant name", min_length=2, max_length=128)
	organization_name: str = Field(..., description="Organization/company name", min_length=2, max_length=256)
	contact_email: str = Field(..., description="Primary contact email address")
	primary_domain: Annotated[str, AfterValidator(validate_domain)] = Field(
		..., description="Primary domain for tenant"
	)
	
	tier: TenantTier = Field(default=TenantTier.FREE, description="Requested service tier")
	cloud_provider: CloudProvider = Field(default=CloudProvider.AWS, description="Preferred cloud provider")
	
	# Optional resource specifications
	cpu_cores: Optional[int] = Field(None, description="Requested CPU cores", ge=1, le=1000)
	memory_gb: Optional[int] = Field(None, description="Requested memory in GB", ge=1, le=10000)
	storage_gb: Optional[int] = Field(None, description="Requested storage in GB", ge=1, le=100000)
	
	# Optional configuration
	custom_domain: Optional[str] = Field(None, description="Custom domain (if different from primary)")
	backup_retention_days: Optional[int] = Field(None, description="Backup retention period", ge=1, le=3650)
	feature_flags: Optional[Dict[str, bool]] = Field(None, description="Feature toggle requests")
	webhook_endpoints: Optional[List[HttpUrl]] = Field(None, description="Event webhook URLs")
	
	# Metadata
	template_id: Optional[str] = Field(None, description="Template to use for provisioning")
	tags: Optional[List[str]] = Field(None, description="Tenant classification tags")
	metadata: Optional[Dict[str, Any]] = Field(None, description="Additional tenant metadata")


class TenantUpdateRequest(BaseModel):
	"""Request model for updating existing tenants"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	display_name: Optional[str] = Field(None, description="Human-readable tenant name", min_length=2)
	organization_name: Optional[str] = Field(None, description="Organization/company name", min_length=2)
	contact_email: Optional[str] = Field(None, description="Primary contact email address")
	
	status: Optional[TenantStatus] = Field(None, description="Tenant status update")
	tier: Optional[TenantTier] = Field(None, description="Service tier update")
	
	# Resource allocation updates
	cpu_cores: Optional[int] = Field(None, description="CPU cores update", ge=1, le=1000)
	memory_gb: Optional[int] = Field(None, description="Memory in GB update", ge=1, le=10000)
	storage_gb: Optional[int] = Field(None, description="Storage in GB update", ge=1, le=100000)
	
	# Configuration updates
	custom_domain: Optional[str] = Field(None, description="Custom domain update")
	backup_retention_days: Optional[int] = Field(None, description="Backup retention update", ge=1, le=3650)
	feature_flags: Optional[Dict[str, bool]] = Field(None, description="Feature flags update")
	webhook_endpoints: Optional[List[HttpUrl]] = Field(None, description="Webhook endpoints update")
	
	# Metadata updates
	tags: Optional[List[str]] = Field(None, description="Tags update")
	metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata update")


class TenantResponse(BaseModel):
	"""Response model for tenant data"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Unique tenant identifier")
	name: str = Field(..., description="Tenant name (slug)")
	display_name: str = Field(..., description="Human-readable tenant name")
	organization_name: str = Field(..., description="Organization/company name")
	contact_email: str = Field(..., description="Primary contact email")
	primary_domain: str = Field(..., description="Primary domain")
	
	status: str = Field(..., description="Current tenant status")
	tier: str = Field(..., description="Service tier")
	cloud_provider: str = Field(..., description="Primary cloud provider")
	
	# Resource allocation
	cpu_cores: int = Field(..., description="Allocated CPU cores")
	memory_gb: int = Field(..., description="Allocated memory in GB")
	storage_gb: int = Field(..., description="Allocated storage in GB")
	bandwidth_mbps: int = Field(..., description="Allocated bandwidth in Mbps")
	database_connections: int = Field(..., description="Max database connections")
	api_rate_limit: int = Field(..., description="API requests per minute limit")
	
	# Configuration
	custom_domain: Optional[str] = Field(None, description="Custom domain")
	ssl_enabled: bool = Field(..., description="SSL/TLS encryption enabled")
	backup_retention_days: int = Field(..., description="Backup retention period")
	api_version: str = Field(..., description="Default API version")
	
	# Timestamps
	created_at: datetime = Field(..., description="Creation timestamp")
	created_by: str = Field(..., description="Creator user ID")
	updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
	updated_by: Optional[str] = Field(None, description="Last updater user ID")
	
	provisioning_started_at: Optional[datetime] = Field(None, description="Provisioning start time")
	provisioning_completed_at: Optional[datetime] = Field(None, description="Provisioning completion time")
	provisioning_duration_seconds: Optional[float] = Field(None, description="Provisioning duration")
	
	# Metadata
	tags: List[str] = Field(..., description="Tenant classification tags")
	feature_flags: Dict[str, bool] = Field(..., description="Enabled feature flags")
	
	@classmethod
	def from_tenant(cls, tenant: Tenant) -> 'TenantResponse':
		"""Create response model from tenant entity"""
		return cls(
			id=tenant.id,
			name=tenant.name,
			display_name=tenant.display_name,
			organization_name=tenant.organization_name,
			contact_email=str(tenant.contact_email),
			primary_domain=tenant.primary_domain,
			
			status=tenant.status.value,
			tier=tenant.tier.value,
			cloud_provider=tenant.cloud_provider.value,
			
			cpu_cores=tenant.resource_allocation.cpu_cores,
			memory_gb=tenant.resource_allocation.memory_gb,
			storage_gb=tenant.resource_allocation.storage_gb,
			bandwidth_mbps=tenant.resource_allocation.bandwidth_mbps,
			database_connections=tenant.resource_allocation.database_connections,
			api_rate_limit=tenant.resource_allocation.api_rate_limit,
			
			custom_domain=tenant.configuration.custom_domain,
			ssl_enabled=tenant.configuration.ssl_enabled,
			backup_retention_days=tenant.configuration.backup_retention_days,
			api_version=tenant.configuration.api_version,
			
			created_at=tenant.created_at,
			created_by=tenant.created_by,
			updated_at=tenant.updated_at,
			updated_by=tenant.updated_by,
			
			provisioning_started_at=tenant.provisioning_started_at,
			provisioning_completed_at=tenant.provisioning_completed_at,
			provisioning_duration_seconds=tenant.provisioning_duration_seconds(),
			
			tags=tenant.tags,
			feature_flags=tenant.configuration.feature_flags
		)


class TenantListResponse(BaseModel):
	"""Response model for paginated tenant lists"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	tenants: List[TenantResponse] = Field(..., description="List of tenant data")
	total_count: int = Field(..., description="Total number of matching tenants")
	page: int = Field(..., description="Current page number")
	page_size: int = Field(..., description="Number of items per page")
	total_pages: int = Field(..., description="Total number of pages")
	has_next: bool = Field(..., description="Whether there are more pages")
	has_previous: bool = Field(..., description="Whether there are previous pages")


class TenantMetricsResponse(BaseModel):
	"""Response model for tenant performance metrics"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	tenant_id: str = Field(..., description="Associated tenant ID")
	timestamp: datetime = Field(..., description="Metrics timestamp")
	
	cpu_usage_percent: float = Field(..., description="CPU utilization percentage")
	memory_usage_percent: float = Field(..., description="Memory utilization percentage") 
	storage_usage_gb: float = Field(..., description="Storage usage in GB")
	api_requests_per_minute: int = Field(..., description="API request rate")
	active_users: int = Field(..., description="Currently active users")
	data_transfer_gb: float = Field(..., description="Data transfer in GB")
	error_rate_percent: float = Field(..., description="Error rate percentage")
	response_time_ms: float = Field(..., description="Average response time in ms")
	
	is_healthy: bool = Field(..., description="Overall health status")
	performance_score: float = Field(..., description="Performance score (0-100)")
	
	@classmethod
	def from_metrics(cls, metrics: TenantMetrics) -> 'TenantMetricsResponse':
		"""Create response model from metrics entity"""
		return cls(
			tenant_id=metrics.tenant_id,
			timestamp=metrics.timestamp,
			cpu_usage_percent=metrics.cpu_usage_percent,
			memory_usage_percent=metrics.memory_usage_percent,
			storage_usage_gb=metrics.storage_usage_gb,
			api_requests_per_minute=metrics.api_requests_per_minute,
			active_users=metrics.active_users,
			data_transfer_gb=metrics.data_transfer_gb,
			error_rate_percent=metrics.error_rate_percent,
			response_time_ms=metrics.response_time_ms,
			is_healthy=metrics.is_healthy(),
			performance_score=metrics.performance_score()
		)


class OptimizationRecommendationResponse(BaseModel):
	"""Response model for AI-generated optimization recommendations"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Unique recommendation ID")
	tenant_id: str = Field(..., description="Target tenant ID")
	timestamp: datetime = Field(..., description="Generation timestamp")
	
	category: str = Field(..., description="Optimization category")
	title: str = Field(..., description="Recommendation title")
	description: str = Field(..., description="Detailed description")
	
	current_state: Dict[str, Any] = Field(..., description="Current configuration state")
	recommended_state: Dict[str, Any] = Field(..., description="Recommended configuration")
	
	estimated_savings_percent: float = Field(..., description="Estimated cost savings percentage")
	estimated_performance_improvement: float = Field(..., description="Performance improvement percentage")
	implementation_complexity: str = Field(..., description="Implementation complexity level")
	
	confidence_score: float = Field(..., description="AI confidence score")
	priority: str = Field(..., description="Recommendation priority")
	
	applied: bool = Field(..., description="Whether recommendation has been applied")
	applied_at: Optional[datetime] = Field(None, description="Application timestamp")
	applied_by: Optional[str] = Field(None, description="User who applied recommendation")
	
	@classmethod
	def from_recommendation(cls, rec: OptimizationRecommendation) -> 'OptimizationRecommendationResponse':
		"""Create response model from recommendation entity"""
		return cls(
			id=rec.id,
			tenant_id=rec.tenant_id,
			timestamp=rec.timestamp,
			category=rec.category,
			title=rec.title,
			description=rec.description,
			current_state=rec.current_state,
			recommended_state=rec.recommended_state,
			estimated_savings_percent=rec.estimated_savings_percent,
			estimated_performance_improvement=rec.estimated_performance_improvement,
			implementation_complexity=rec.implementation_complexity,
			confidence_score=rec.confidence_score,
			priority=rec.priority,
			applied=rec.applied,
			applied_at=rec.applied_at,
			applied_by=rec.applied_by
		)


class TenantQueryRequest(BaseModel):
	"""Request model for querying/filtering tenants"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	status: Optional[TenantStatus] = Field(None, description="Filter by tenant status")
	tier: Optional[TenantTier] = Field(None, description="Filter by service tier")
	cloud_provider: Optional[CloudProvider] = Field(None, description="Filter by cloud provider")
	
	name_contains: Optional[str] = Field(None, description="Filter by name containing text")
	organization_contains: Optional[str] = Field(None, description="Filter by organization containing text")
	email_contains: Optional[str] = Field(None, description="Filter by email containing text")
	
	created_after: Optional[datetime] = Field(None, description="Filter by creation date after")
	created_before: Optional[datetime] = Field(None, description="Filter by creation date before")
	
	tags: Optional[List[str]] = Field(None, description="Filter by tags (must have all specified)")
	has_custom_domain: Optional[bool] = Field(None, description="Filter by custom domain presence")
	
	page: int = Field(default=1, description="Page number", ge=1)
	page_size: int = Field(default=20, description="Items per page", ge=1, le=100)
	
	sort_by: str = Field(default="created_at", description="Sort field")
	sort_order: str = Field(default="desc", description="Sort order", pattern="^(asc|desc)$")


class TenantOperationResponse(BaseModel):
	"""Response model for tenant operations (create, update, delete)"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	success: bool = Field(..., description="Operation success status")
	message: str = Field(..., description="Operation result message")
	tenant: Optional[TenantResponse] = Field(None, description="Tenant data (if applicable)")
	operation_id: Optional[str] = Field(None, description="Async operation ID")
	estimated_completion_time: Optional[datetime] = Field(None, description="Expected completion time")


class TenantProvisioningStatusResponse(BaseModel):
	"""Response model for tenant provisioning status"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	tenant_id: str = Field(..., description="Tenant identifier")
	status: str = Field(..., description="Current provisioning status")
	progress_percent: int = Field(..., description="Provisioning progress percentage", ge=0, le=100)
	
	current_step: str = Field(..., description="Current provisioning step")
	completed_steps: List[str] = Field(..., description="List of completed steps")
	remaining_steps: List[str] = Field(..., description="List of remaining steps")
	
	started_at: datetime = Field(..., description="Provisioning start time")
	estimated_completion_at: Optional[datetime] = Field(None, description="Estimated completion time")
	completed_at: Optional[datetime] = Field(None, description="Actual completion time")
	
	resources_allocated: Dict[str, Any] = Field(..., description="Currently allocated resources")
	errors: List[str] = Field(default_factory=list, description="Any provisioning errors")
	
	sla_met: Optional[bool] = Field(None, description="Whether <60 second SLA was met")


class TenantAnalyticsResponse(BaseModel):
	"""Response model for tenant analytics and insights"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	tenant_id: str = Field(..., description="Tenant identifier")
	period_start: datetime = Field(..., description="Analytics period start")
	period_end: datetime = Field(..., description="Analytics period end")
	
	# Usage analytics
	total_api_calls: int = Field(..., description="Total API calls in period")
	unique_users: int = Field(..., description="Unique active users")
	data_transferred_gb: float = Field(..., description="Total data transferred in GB")
	storage_used_gb: float = Field(..., description="Peak storage usage in GB")
	
	# Performance analytics
	avg_response_time_ms: float = Field(..., description="Average response time")
	error_rate_percent: float = Field(..., description="Overall error rate")
	uptime_percent: float = Field(..., description="Service uptime percentage")
	
	# Cost analytics
	estimated_cost_usd: Decimal = Field(..., description="Estimated cost for period")
	cost_breakdown: Dict[str, Decimal] = Field(..., description="Cost breakdown by resource type")
	
	# Optimization opportunities
	optimization_score: float = Field(..., description="Optimization opportunity score (0-100)")
	potential_savings_percent: float = Field(..., description="Potential cost savings percentage")
	
	# Trends
	usage_trend: str = Field(..., description="Usage trend (increasing, stable, decreasing)")
	cost_trend: str = Field(..., description="Cost trend (increasing, stable, decreasing)")
	performance_trend: str = Field(..., description="Performance trend (improving, stable, degrading)")


class TenantTierUpgradeRequest(BaseModel):
	"""Request model for tenant tier upgrades"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	target_tier: TenantTier = Field(..., description="Target service tier")
	effective_date: Optional[datetime] = Field(None, description="When upgrade should take effect")
	reason: Optional[str] = Field(None, description="Reason for upgrade request")
	
	# Custom resource requests for CUSTOM tier
	custom_cpu_cores: Optional[int] = Field(None, description="Custom CPU allocation", ge=1)
	custom_memory_gb: Optional[int] = Field(None, description="Custom memory allocation", ge=1)
	custom_storage_gb: Optional[int] = Field(None, description="Custom storage allocation", ge=1)
	custom_features: Optional[List[str]] = Field(None, description="Custom feature requests")


class TenantSuspensionRequest(BaseModel):
	"""Request model for tenant suspension/reactivation"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	action: str = Field(..., description="Action to take", pattern="^(suspend|reactivate)$")
	reason: str = Field(..., description="Reason for suspension/reactivation")
	notify_users: bool = Field(default=True, description="Whether to notify tenant users")
	preserve_data: bool = Field(default=True, description="Whether to preserve tenant data")


class MultiTenantStatsResponse(BaseModel):
	"""Response model for multi-tenant system statistics"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	total_tenants: int = Field(..., description="Total number of tenants")
	active_tenants: int = Field(..., description="Number of active tenants")
	
	tenants_by_tier: Dict[str, int] = Field(..., description="Tenant count by tier")
	tenants_by_status: Dict[str, int] = Field(..., description="Tenant count by status")
	tenants_by_cloud_provider: Dict[str, int] = Field(..., description="Tenant count by cloud provider")
	
	total_provisioning_time_avg_seconds: float = Field(..., description="Average provisioning time")
	sla_compliance_percent: float = Field(..., description="Provisioning SLA compliance rate")
	
	system_resource_utilization: Dict[str, float] = Field(..., description="Overall resource utilization")
	total_monthly_cost_usd: Decimal = Field(..., description="Total system cost per month")
	
	recent_activity: List[Dict[str, Any]] = Field(..., description="Recent tenant activities")


class HealthCheckResponse(BaseModel):
	"""Response model for system health checks"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	status: str = Field(..., description="Overall health status", pattern="^(healthy|degraded|unhealthy)$")
	timestamp: datetime = Field(..., description="Health check timestamp")
	
	components: Dict[str, Dict[str, Any]] = Field(..., description="Component-specific health status")
	
	version: str = Field(..., description="MTen capability version")
	uptime_seconds: float = Field(..., description="System uptime in seconds")
	
	database_connected: bool = Field(..., description="Database connectivity status")
	cache_connected: bool = Field(..., description="Cache connectivity status")
	external_apis_reachable: bool = Field(..., description="External API connectivity")
	
	active_tenants_count: int = Field(..., description="Number of currently active tenants")
	provisioning_queue_length: int = Field(..., description="Number of tenants in provisioning queue")