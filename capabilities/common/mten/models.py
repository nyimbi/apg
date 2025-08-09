"""
Multi-Tenant Management (MTen) Core Data Models

Company: Datacraft
Copyright: © 2025
Author: Nyimbi Odero

Comprehensive Pydantic v2 models for enterprise-grade multi-tenant management
following CLAUDE.md standards: async throughout, modern typing, strict validation.
"""

from datetime import datetime, UTC
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Annotated
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, AfterValidator, HttpUrl
from uuid_extensions import uuid7str


def validate_tenant_name(name: str) -> str:
	"""Validate tenant name follows naming conventions"""
	if not name.replace('-', '').replace('_', '').isalnum():
		raise ValueError("Tenant name must be alphanumeric with hyphens/underscores only")
	if len(name) < 2 or len(name) > 64:
		raise ValueError("Tenant name must be 2-64 characters")
	if name.startswith('-') or name.endswith('-'):
		raise ValueError("Tenant name cannot start or end with hyphen")
	return name.lower()


def validate_positive_number(value: Union[int, float]) -> Union[int, float]:
	"""Validate number is positive"""
	if value <= 0:
		raise ValueError("Value must be positive")
	return value


def validate_domain(domain: str) -> str:
	"""Validate domain name format"""
	if not domain or len(domain) > 255:
		raise ValueError("Domain must be 1-255 characters")
	if not all(c.isalnum() or c in '.-' for c in domain):
		raise ValueError("Domain contains invalid characters")
	return domain.lower()


class TenantStatus(str, Enum):
	"""Tenant lifecycle status enumeration"""
	PROVISIONING = "provisioning"
	ACTIVE = "active"
	SUSPENDED = "suspended"
	DECOMMISSIONING = "decommissioning"
	ARCHIVED = "archived"


class TenantTier(str, Enum):
	"""Tenant service tier enumeration"""
	FREE = "free"
	PREMIUM = "premium"
	ENTERPRISE = "enterprise"
	CUSTOM = "custom"


class CloudProvider(str, Enum):
	"""Supported cloud provider enumeration"""
	AWS = "aws"
	AZURE = "azure"
	GCP = "gcp"
	HYBRID = "hybrid"
	ON_PREMISE = "on_premise"


class ResourceAllocation(BaseModel):
	"""Tenant resource allocation configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	cpu_cores: Annotated[int, AfterValidator(validate_positive_number)] = Field(
		..., description="Allocated CPU cores", ge=1, le=1000
	)
	memory_gb: Annotated[int, AfterValidator(validate_positive_number)] = Field(
		..., description="Allocated memory in GB", ge=1, le=10000
	)
	storage_gb: Annotated[int, AfterValidator(validate_positive_number)] = Field(
		..., description="Allocated storage in GB", ge=1, le=100000
	)
	bandwidth_mbps: Annotated[int, AfterValidator(validate_positive_number)] = Field(
		..., description="Network bandwidth in Mbps", ge=1, le=100000
	)
	database_connections: Annotated[int, AfterValidator(validate_positive_number)] = Field(
		default=10, description="Max database connections", ge=1, le=10000
	)
	api_rate_limit: int = Field(default=1000, description="API requests per minute", ge=1)
	
	def total_compute_units(self) -> int:
		"""Calculate total compute units for billing"""
		return (self.cpu_cores * 10) + (self.memory_gb * 2) + (self.storage_gb // 10)
	
	def is_within_limits(self, tier: TenantTier) -> bool:
		"""Check if allocation is within tier limits"""
		limits = {
			TenantTier.FREE: {"cpu": 2, "memory": 4, "storage": 20},
			TenantTier.PREMIUM: {"cpu": 16, "memory": 64, "storage": 1000},
			TenantTier.ENTERPRISE: {"cpu": 128, "memory": 512, "storage": 10000}
		}
		
		if tier == TenantTier.CUSTOM:
			return True
			
		limit = limits.get(tier, limits[TenantTier.FREE])
		return (
			self.cpu_cores <= limit["cpu"] and 
			self.memory_gb <= limit["memory"] and
			self.storage_gb <= limit["storage"]
		)


class TenantConfiguration(BaseModel):
	"""Tenant-specific configuration settings"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	custom_domain: Optional[Annotated[str, AfterValidator(validate_domain)]] = Field(
		None, description="Custom domain for tenant"
	)
	ssl_enabled: bool = Field(default=True, description="SSL/TLS encryption enabled")
	backup_retention_days: int = Field(default=30, description="Backup retention period", ge=1, le=3650)
	api_version: str = Field(default="v1", description="Default API version")
	feature_flags: Dict[str, bool] = Field(default_factory=dict, description="Feature toggles")
	custom_branding: Dict[str, str] = Field(default_factory=dict, description="UI branding config")
	webhook_endpoints: List[HttpUrl] = Field(default_factory=list, description="Event webhook URLs")
	integration_config: Dict[str, Any] = Field(default_factory=dict, description="Third-party integrations")


class TenantMetrics(BaseModel):
	"""Real-time tenant performance metrics"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	tenant_id: str = Field(..., description="Associated tenant ID")
	timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
	cpu_usage_percent: float = Field(..., description="CPU utilization percentage", ge=0, le=100)
	memory_usage_percent: float = Field(..., description="Memory utilization percentage", ge=0, le=100)
	storage_usage_gb: float = Field(..., description="Storage usage in GB", ge=0)
	api_requests_per_minute: int = Field(..., description="API request rate", ge=0)
	active_users: int = Field(..., description="Currently active users", ge=0)
	data_transfer_gb: float = Field(..., description="Data transfer in GB", ge=0)
	error_rate_percent: float = Field(default=0.0, description="Error rate percentage", ge=0, le=100)
	response_time_ms: float = Field(default=0.0, description="Average response time in ms", ge=0)
	
	def is_healthy(self) -> bool:
		"""Check if tenant metrics indicate healthy operation"""
		return (
			self.cpu_usage_percent < 80 and
			self.memory_usage_percent < 85 and 
			self.error_rate_percent < 5 and
			self.response_time_ms < 1000
		)
	
	def performance_score(self) -> float:
		"""Calculate overall performance score (0-100)"""
		cpu_score = max(0, 100 - self.cpu_usage_percent)
		memory_score = max(0, 100 - self.memory_usage_percent)
		error_score = max(0, 100 - (self.error_rate_percent * 10))
		response_score = max(0, 100 - (self.response_time_ms / 10))
		
		return (cpu_score + memory_score + error_score + response_score) / 4


class Tenant(BaseModel):
	"""Core tenant entity with comprehensive metadata"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique tenant identifier")
	name: Annotated[str, AfterValidator(validate_tenant_name)] = Field(
		..., description="Unique tenant name (slug)"
	)
	display_name: str = Field(..., description="Human-readable tenant name")
	organization_name: str = Field(..., description="Organization/company name")
	contact_email: str = Field(..., description="Primary contact email")
	primary_domain: Annotated[str, AfterValidator(validate_domain)] = Field(
		..., description="Primary domain for tenant"
	)
	
	status: TenantStatus = Field(default=TenantStatus.PROVISIONING, description="Current tenant status")
	tier: TenantTier = Field(default=TenantTier.FREE, description="Service tier")
	cloud_provider: CloudProvider = Field(default=CloudProvider.AWS, description="Primary cloud provider")
	
	resource_allocation: ResourceAllocation = Field(
		default_factory=lambda: ResourceAllocation(cpu_cores=1, memory_gb=2, storage_gb=10, bandwidth_mbps=100)
	)
	configuration: TenantConfiguration = Field(default_factory=TenantConfiguration)
	
	created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
	created_by: str = Field(..., description="User ID who created tenant")
	updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
	updated_by: Optional[str] = Field(None, description="User ID who last updated tenant")
	
	provisioning_started_at: Optional[datetime] = Field(None, description="Provisioning start time")
	provisioning_completed_at: Optional[datetime] = Field(None, description="Provisioning completion time")
	
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional tenant metadata")
	tags: List[str] = Field(default_factory=list, description="Tenant classification tags")
	
	def provisioning_duration_seconds(self) -> Optional[float]:
		"""Calculate provisioning duration in seconds"""
		if self.provisioning_started_at and self.provisioning_completed_at:
			return (self.provisioning_completed_at - self.provisioning_started_at).total_seconds()
		return None
	
	def is_provisioning_sla_met(self) -> bool:
		"""Check if provisioning met <60 second SLA"""
		duration = self.provisioning_duration_seconds()
		return duration is not None and duration < 60.0
	
	def update_timestamp(self, user_id: str) -> None:
		"""Update modification timestamp and user"""
		self.updated_at = datetime.now(UTC)
		self.updated_by = user_id


class TenantTemplate(BaseModel):
	"""Reusable tenant configuration template"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique template identifier")
	name: str = Field(..., description="Template name", min_length=2, max_length=128)
	description: str = Field(..., description="Template description")
	version: str = Field(default="1.0.0", description="Template version")
	
	default_tier: TenantTier = Field(default=TenantTier.FREE)
	default_cloud_provider: CloudProvider = Field(default=CloudProvider.AWS)
	default_resource_allocation: ResourceAllocation = Field(...)
	default_configuration: TenantConfiguration = Field(default_factory=TenantConfiguration)
	
	category: str = Field(..., description="Template category")
	tags: List[str] = Field(default_factory=list, description="Template classification tags")
	
	is_public: bool = Field(default=False, description="Available in public marketplace")
	created_by: str = Field(..., description="Template author user ID")
	created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
	
	usage_count: int = Field(default=0, description="Number of times used", ge=0)
	rating: float = Field(default=0.0, description="Average user rating", ge=0, le=5)
	
	parent_template_id: Optional[str] = Field(None, description="Parent template for inheritance")
	override_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration overrides")


class TenantAuditLog(BaseModel):
	"""Comprehensive tenant audit trail"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique audit log entry ID")
	tenant_id: str = Field(..., description="Associated tenant ID")
	timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
	
	action: str = Field(..., description="Action performed")
	actor_id: str = Field(..., description="User or system ID performing action")
	actor_type: str = Field(..., description="Actor type (user, system, api)")
	
	resource_type: str = Field(..., description="Type of resource affected")
	resource_id: str = Field(..., description="ID of affected resource")
	
	changes: Dict[str, Any] = Field(default_factory=dict, description="Before/after changes")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
	
	ip_address: Optional[str] = Field(None, description="Source IP address")
	user_agent: Optional[str] = Field(None, description="Client user agent")
	
	compliance_tags: Dict[str, str] = Field(default_factory=dict, description="Compliance framework tags")
	blockchain_hash: Optional[str] = Field(None, description="Blockchain verification hash")


class OptimizationRecommendation(BaseModel):
	"""AI-generated tenant optimization recommendation"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique recommendation ID")
	tenant_id: str = Field(..., description="Target tenant ID")
	timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
	
	category: str = Field(..., description="Optimization category")
	title: str = Field(..., description="Recommendation title")
	description: str = Field(..., description="Detailed description")
	
	current_state: Dict[str, Any] = Field(..., description="Current configuration state")
	recommended_state: Dict[str, Any] = Field(..., description="Recommended configuration")
	
	estimated_savings_percent: float = Field(..., description="Estimated cost savings %", ge=0, le=100)
	estimated_performance_improvement: float = Field(..., description="Performance improvement %", ge=0)
	implementation_complexity: str = Field(..., description="Implementation complexity level")
	
	confidence_score: float = Field(..., description="AI confidence score", ge=0, le=1)
	priority: str = Field(..., description="Recommendation priority", pattern="^(low|medium|high|critical)$")
	
	applied: bool = Field(default=False, description="Whether recommendation has been applied")
	applied_at: Optional[datetime] = Field(None, description="Application timestamp")
	applied_by: Optional[str] = Field(None, description="User who applied recommendation")


class DeploymentPlan(BaseModel):
	"""Multi-cloud deployment plan for tenant"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique deployment plan ID")
	tenant_id: str = Field(..., description="Target tenant ID")
	created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
	
	primary_region: str = Field(..., description="Primary deployment region")
	backup_regions: List[str] = Field(default_factory=list, description="Backup regions")
	
	cloud_deployments: List[Dict[str, Any]] = Field(
		default_factory=list, description="Per-provider deployment configs"
	)
	
	estimated_monthly_cost: Decimal = Field(..., description="Estimated monthly cost in USD")
	estimated_setup_time_minutes: int = Field(..., description="Setup time estimate in minutes")
	
	load_balancing_strategy: str = Field(default="round_robin", description="Load balancing strategy")
	failover_strategy: str = Field(default="active_passive", description="Failover strategy")
	
	approved: bool = Field(default=False, description="Deployment plan approved")
	approved_by: Optional[str] = Field(None, description="User who approved plan")
	approved_at: Optional[datetime] = Field(None, description="Approval timestamp")


# Tier configuration templates for resource limits
TIER_RESOURCE_TEMPLATES = {
	TenantTier.FREE: ResourceAllocation(
		cpu_cores=1,
		memory_gb=2,
		storage_gb=10,
		bandwidth_mbps=100,
		database_connections=5,
		api_rate_limit=500
	),
	TenantTier.PREMIUM: ResourceAllocation(
		cpu_cores=4,
		memory_gb=16,
		storage_gb=100,
		bandwidth_mbps=1000,
		database_connections=25,
		api_rate_limit=5000
	),
	TenantTier.ENTERPRISE: ResourceAllocation(
		cpu_cores=16,
		memory_gb=64,
		storage_gb=1000,
		bandwidth_mbps=10000,
		database_connections=100,
		api_rate_limit=25000
	)
}