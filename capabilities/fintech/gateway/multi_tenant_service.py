#!/usr/bin/env python3
"""
Multi-Tenant Architecture Service - APG Payment Gateway

Comprehensive multi-tenancy implementation with tenant isolation,
resource management, and scalable architecture patterns.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from uuid_extensions import uuid7str
from dataclasses import dataclass, field
import json
import logging

from pydantic import BaseModel, Field, ConfigDict, field_validator
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, JSON, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID

logger = logging.getLogger(__name__)

# Multi-tenant data models
class TenantStatus(str, Enum):
	"""Tenant status enumeration"""
	ACTIVE = "active"
	SUSPENDED = "suspended"
	PENDING_ACTIVATION = "pending_activation" 
	DEACTIVATED = "deactivated"
	ARCHIVED = "archived"

class TenantPlan(str, Enum):
	"""Tenant subscription plan types"""
	FREE = "free"
	BASIC = "basic"
	PROFESSIONAL = "professional"
	ENTERPRISE = "enterprise"
	CUSTOM = "custom"

class ResourceType(str, Enum):
	"""Resource types for tenant limits"""
	TRANSACTIONS_PER_MONTH = "transactions_per_month"
	API_CALLS_PER_DAY = "api_calls_per_day"
	STORAGE_GB = "storage_gb"
	CONCURRENT_CONNECTIONS = "concurrent_connections"
	WEBHOOK_ENDPOINTS = "webhook_endpoints"
	CUSTOM_PROCESSORS = "custom_processors"

@dataclass
class TenantResourceLimit:
	"""Resource limit configuration for tenants"""
	resource_type: ResourceType
	limit: int
	current_usage: int = 0
	reset_period: str = "monthly"  # daily, weekly, monthly, annually
	last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	
	@property
	def usage_percentage(self) -> float:
		"""Calculate usage percentage"""
		if self.limit == 0:
			return 0.0
		return min(100.0, (self.current_usage / self.limit) * 100)
	
	@property
	def is_exceeded(self) -> bool:
		"""Check if limit is exceeded"""
		return self.current_usage >= self.limit

class Tenant(BaseModel):
	"""
	Multi-tenant organization model with complete isolation and resource management
	"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Unique tenant ID")
	name: str = Field(..., min_length=1, max_length=100, description="Tenant organization name")
	slug: str = Field(..., min_length=1, max_length=50, description="URL-safe tenant identifier")
	
	# Business information
	business_type: str = Field(..., description="Type of business (ecommerce, saas, etc.)")
	industry: str | None = Field(None, description="Industry classification")
	country: str = Field(..., min_length=2, max_length=2, description="ISO country code")
	timezone: str = Field(default="UTC", description="Tenant timezone")
	
	# Subscription and billing
	plan: TenantPlan = Field(default=TenantPlan.FREE, description="Subscription plan")
	status: TenantStatus = Field(default=TenantStatus.PENDING_ACTIVATION, description="Tenant status")
	billing_email: str = Field(..., description="Billing contact email")
	
	# Technical configuration
	subdomain: str | None = Field(None, description="Custom subdomain")
	custom_domain: str | None = Field(None, description="Custom domain")
	api_version: str = Field(default="v1", description="API version preference")
	
	# Resource limits and usage
	resource_limits: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Resource limits by type")
	feature_flags: Dict[str, bool] = Field(default_factory=dict, description="Feature toggles")
	
	# Security settings
	require_mfa: bool = Field(default=False, description="Require multi-factor authentication")
	allowed_ip_ranges: List[str] = Field(default_factory=list, description="Allowed IP ranges (CIDR)")
	session_timeout_minutes: int = Field(default=480, ge=5, le=1440, description="Session timeout")
	
	# Compliance and data handling
	data_residency_region: str | None = Field(None, description="Required data residency region")
	pci_compliance_required: bool = Field(default=False, description="PCI DSS compliance required")
	gdpr_applicable: bool = Field(default=False, description="GDPR regulations apply")
	
	# Integration settings
	webhook_endpoints: List[str] = Field(default_factory=list, description="Configured webhook URLs")
	allowed_processors: List[str] = Field(default_factory=list, description="Allowed payment processors")
	default_currency: str = Field(default="USD", description="Default currency")
	
	# Metadata and customization
	branding: Dict[str, Any] = Field(default_factory=dict, description="Custom branding configuration")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	
	# Timestamps and audit
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	activated_at: datetime | None = Field(None, description="Tenant activation time")
	last_activity_at: datetime | None = Field(None, description="Last tenant activity")
	
	# Relationships
	parent_tenant_id: str | None = Field(None, description="Parent tenant for hierarchical structures")
	child_tenant_ids: List[str] = Field(default_factory=list, description="Child tenant IDs")
	
	@field_validator('slug')
	@classmethod
	def validate_slug(cls, v):
		"""Validate tenant slug format"""
		import re
		if not re.match(r'^[a-z0-9-]+$', v):
			raise ValueError('Slug must contain only lowercase letters, numbers, and hyphens')
		return v
	
	def get_resource_limit(self, resource_type: ResourceType) -> TenantResourceLimit | None:
		"""Get resource limit for specific type"""
		limit_data = self.resource_limits.get(resource_type.value)
		if limit_data:
			return TenantResourceLimit(
				resource_type=resource_type,
				limit=limit_data.get('limit', 0),
				current_usage=limit_data.get('current_usage', 0),
				reset_period=limit_data.get('reset_period', 'monthly'),
				last_reset=datetime.fromisoformat(limit_data.get('last_reset', datetime.now(timezone.utc).isoformat()))
			)
		return None
	
	def is_feature_enabled(self, feature_name: str) -> bool:
		"""Check if feature is enabled for tenant"""
		return self.feature_flags.get(feature_name, False)
	
	def _log_tenant_event(self, event: str, details: Dict[str, Any] = None):
		"""Log tenant events with structured logging"""
		logger.info(
			"tenant_event",
			tenant_id=self.id,
			tenant_name=self.name,
			event=event,
			details=details or {}
		)

class TenantIsolationService:
	"""
	Service for managing multi-tenant isolation and resource management
	"""
	
	def __init__(self, database_service=None):
		self._database_service = database_service
		self._tenant_cache: Dict[str, Tenant] = {}
		self._resource_monitors: Dict[str, Dict[ResourceType, TenantResourceLimit]] = {}
		self._isolation_rules: Dict[str, Any] = {}
		self._initialized = False
	
	async def initialize(self):
		"""Initialize multi-tenant service"""
		try:
			# Load default tenant configurations
			await self._load_default_configurations()
			
			# Initialize resource monitoring
			await self._initialize_resource_monitoring()
			
			# Setup isolation rules
			await self._setup_isolation_rules()
			
			self._initialized = True
			self._log_service_initialized()
			
		except Exception as e:
			logger.error("multi_tenant_initialization_failed", error=str(e))
			raise
	
	# Tenant Management
	
	async def create_tenant(self, tenant_data: Dict[str, Any]) -> Tenant:
		"""Create a new tenant with full configuration"""
		try:
			# Validate tenant data
			errors = self._validate_tenant_creation_data(tenant_data)
			if errors:
				raise ValueError(f"Invalid tenant data: {', '.join(errors)}")
			
			# Check if slug is available
			if await self._is_slug_taken(tenant_data['slug']):
				raise ValueError(f"Tenant slug '{tenant_data['slug']}' is already taken")
			
			# Create tenant with default settings
			tenant = Tenant(
				name=tenant_data['name'],
				slug=tenant_data['slug'],
				business_type=tenant_data['business_type'],
				industry=tenant_data.get('industry'),
				country=tenant_data['country'],
				timezone=tenant_data.get('timezone', 'UTC'),
				plan=TenantPlan(tenant_data.get('plan', 'free')),
				billing_email=tenant_data['billing_email'],
				subdomain=tenant_data.get('subdomain'),
				custom_domain=tenant_data.get('custom_domain')
			)
			
			# Set default resource limits based on plan
			tenant.resource_limits = self._get_default_resource_limits(tenant.plan)
			
			# Set default feature flags
			tenant.feature_flags = self._get_default_feature_flags(tenant.plan)
			
			# Set allowed processors
			tenant.allowed_processors = self._get_default_processors(tenant.plan)
			
			# Store tenant
			if self._database_service:
				await self._database_service.create_tenant(tenant)
			
			# Cache tenant
			self._tenant_cache[tenant.id] = tenant
			
			# Initialize resource monitoring
			await self._initialize_tenant_resource_monitoring(tenant.id)
			
			# Setup tenant-specific isolation
			await self._setup_tenant_isolation(tenant)
			
			tenant._log_tenant_event("tenant_created", {
				"plan": tenant.plan.value,
				"country": tenant.country,
				"business_type": tenant.business_type
			})
			
			return tenant
			
		except Exception as e:
			logger.error("tenant_creation_failed", error=str(e))
			raise
	
	async def get_tenant(self, tenant_id: str) -> Tenant | None:
		"""Get tenant by ID with caching"""
		try:
			# Check cache first
			if tenant_id in self._tenant_cache:
				return self._tenant_cache[tenant_id]
			
			# Load from database
			if self._database_service:
				tenant = await self._database_service.get_tenant(tenant_id)
				if tenant:
					self._tenant_cache[tenant_id] = tenant
					return tenant
			
			return None
			
		except Exception as e:
			logger.error("tenant_retrieval_failed", tenant_id=tenant_id, error=str(e))
			return None
	
	async def get_tenant_by_slug(self, slug: str) -> Tenant | None:
		"""Get tenant by slug"""
		try:
			# Check cache
			for tenant in self._tenant_cache.values():
				if tenant.slug == slug:
					return tenant
			
			# Load from database
			if self._database_service:
				tenant = await self._database_service.get_tenant_by_slug(slug)
				if tenant:
					self._tenant_cache[tenant.id] = tenant
					return tenant
			
			return None
			
		except Exception as e:
			logger.error("tenant_retrieval_by_slug_failed", slug=slug, error=str(e))
			return None
	
	async def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> None:
		"""Update tenant configuration"""
		try:
			tenant = await self.get_tenant(tenant_id)
			if not tenant:
				return
			
			# Validate updates
			errors = self._validate_tenant_updates(updates)
			if errors:
				raise ValueError(f"Invalid updates: {', '.join(errors)}")
			
			# Convert plan string to enum if needed
			processed_updates = updates.copy()
			if "plan" in processed_updates and isinstance(processed_updates["plan"], str):
				processed_updates["plan"] = TenantPlan(processed_updates["plan"])
			
			# Apply updates
			for key, value in processed_updates.items():
				if hasattr(tenant, key):
					setattr(tenant, key, value)
			
			tenant.updated_at = datetime.now(timezone.utc)
			
			# Update database (send original updates with string values)
			if self._database_service:
				db_updates = updates.copy()
				db_updates["updated_at"] = tenant.updated_at
				await self._database_service.update_tenant(tenant_id, db_updates)
			
			# Update cache
			self._tenant_cache[tenant_id] = tenant
			
			tenant._log_tenant_event("tenant_updated", {"updates": list(updates.keys())})
			
		except Exception as e:
			logger.error("tenant_update_failed", tenant_id=tenant_id, error=str(e))
			raise
	
	# Resource Management
	
	async def check_resource_limit(self, tenant_id: str, resource_type: ResourceType, requested_amount: int = 1) -> Dict[str, Any]:
		"""Check if tenant can use requested resources"""
		try:
			tenant = await self.get_tenant(tenant_id)
			if not tenant:
				return {"allowed": False, "error": "Tenant not found"}
			
			resource_limit = tenant.get_resource_limit(resource_type)
			if not resource_limit:
				# No limit configured - allow unlimited
				return {"allowed": True, "unlimited": True}
			
			# Check if request would exceed limit
			new_usage = resource_limit.current_usage + requested_amount
			if new_usage > resource_limit.limit:
				return {
					"allowed": False,
					"error": "Resource limit exceeded",
					"limit": resource_limit.limit,
					"current_usage": resource_limit.current_usage,
					"requested": requested_amount,
					"usage_percentage": resource_limit.usage_percentage
				}
			
			return {
				"allowed": True,
				"limit": resource_limit.limit,
				"current_usage": resource_limit.current_usage,
				"remaining": resource_limit.limit - new_usage,
				"usage_percentage": (new_usage / resource_limit.limit) * 100
			}
			
		except Exception as e:
			logger.error("resource_limit_check_failed", tenant_id=tenant_id, resource_type=resource_type.value, error=str(e))
			return {"allowed": False, "error": "Internal error"}
	
	async def consume_resource(self, tenant_id: str, resource_type: ResourceType, amount: int = 1) -> bool:
		"""Consume tenant resources and update usage"""
		try:
			# Check limit first
			check_result = await self.check_resource_limit(tenant_id, resource_type, amount)
			if not check_result["allowed"]:
				return False
			
			# Update resource usage
			await self._update_resource_usage(tenant_id, resource_type, amount)
			
			return True
			
		except Exception as e:
			logger.error("resource_consumption_failed", tenant_id=tenant_id, resource_type=resource_type.value, error=str(e))
			return False
	
	async def get_tenant_usage_report(self, tenant_id: str) -> Dict[str, Any]:
		"""Get comprehensive usage report for tenant"""
		try:
			tenant = await self.get_tenant(tenant_id)
			if not tenant:
				return {"error": "Tenant not found"}
			
			usage_report = {
				"tenant_id": tenant_id,
				"tenant_name": tenant.name,
				"plan": tenant.plan.value,
				"status": tenant.status.value,
				"report_generated_at": datetime.now(timezone.utc).isoformat(),
				"resources": {}
			}
			
			# Get usage for each resource type
			for resource_type in ResourceType:
				resource_limit = tenant.get_resource_limit(resource_type)
				if resource_limit:
					usage_report["resources"][resource_type.value] = {
						"limit": resource_limit.limit,
						"current_usage": resource_limit.current_usage,
						"usage_percentage": resource_limit.usage_percentage,
						"is_exceeded": resource_limit.is_exceeded,
						"reset_period": resource_limit.reset_period,
						"last_reset": resource_limit.last_reset.isoformat()
					}
			
			return usage_report
			
		except Exception as e:
			logger.error("usage_report_generation_failed", tenant_id=tenant_id, error=str(e))
			return {"error": "Internal error"}
	
	# Tenant Isolation
	
	async def get_tenant_database_connection(self, tenant_id: str) -> str:
		"""Get tenant-specific database connection string with proper isolation"""
		tenant = await self.get_tenant(tenant_id)
		if not tenant:
			raise ValueError("Tenant not found")
		
		# In production, this would return tenant-specific database
		# For now, return schema-isolated connection
		base_connection = "postgresql://user:pass@localhost/apg_payment_gateway"
		return f"{base_connection}?options=-csearch_path=tenant_{tenant.slug}"
	
	async def apply_row_level_security(self, tenant_id: str, query_context: Dict[str, Any]) -> Dict[str, Any]:
		"""Apply row-level security policies for tenant isolation"""
		tenant = await self.get_tenant(tenant_id)
		if not tenant:
			raise ValueError("Tenant not found")
		
		# Add tenant filter to all queries
		query_context["tenant_filters"] = {
			"tenant_id": tenant_id,
			"allowed_child_tenants": tenant.child_tenant_ids
		}
		
		# Apply additional security based on tenant settings
		if tenant.data_residency_region:
			query_context["data_residency_filter"] = tenant.data_residency_region
		
		return query_context
	
	# Internal Methods
	
	async def _load_default_configurations(self):
		"""Load default tenant configurations"""
		self._default_resource_limits = {
			TenantPlan.FREE: {
				ResourceType.TRANSACTIONS_PER_MONTH.value: {"limit": 100, "current_usage": 0, "reset_period": "monthly"},
				ResourceType.API_CALLS_PER_DAY.value: {"limit": 1000, "current_usage": 0, "reset_period": "daily"},
				ResourceType.STORAGE_GB.value: {"limit": 1, "current_usage": 0, "reset_period": "monthly"}
			},
			TenantPlan.BASIC: {
				ResourceType.TRANSACTIONS_PER_MONTH.value: {"limit": 1000, "current_usage": 0, "reset_period": "monthly"},
				ResourceType.API_CALLS_PER_DAY.value: {"limit": 10000, "current_usage": 0, "reset_period": "daily"},
				ResourceType.STORAGE_GB.value: {"limit": 10, "current_usage": 0, "reset_period": "monthly"},
				ResourceType.WEBHOOK_ENDPOINTS.value: {"limit": 5, "current_usage": 0, "reset_period": "monthly"}
			},
			TenantPlan.PROFESSIONAL: {
				ResourceType.TRANSACTIONS_PER_MONTH.value: {"limit": 10000, "current_usage": 0, "reset_period": "monthly"},
				ResourceType.API_CALLS_PER_DAY.value: {"limit": 100000, "current_usage": 0, "reset_period": "daily"},
				ResourceType.STORAGE_GB.value: {"limit": 100, "current_usage": 0, "reset_period": "monthly"},
				ResourceType.WEBHOOK_ENDPOINTS.value: {"limit": 20, "current_usage": 0, "reset_period": "monthly"},
				ResourceType.CUSTOM_PROCESSORS.value: {"limit": 3, "current_usage": 0, "reset_period": "monthly"}
			},
			TenantPlan.ENTERPRISE: {
				ResourceType.TRANSACTIONS_PER_MONTH.value: {"limit": 100000, "current_usage": 0, "reset_period": "monthly"},
				ResourceType.API_CALLS_PER_DAY.value: {"limit": 1000000, "current_usage": 0, "reset_period": "daily"},
				ResourceType.STORAGE_GB.value: {"limit": 1000, "current_usage": 0, "reset_period": "monthly"},
				ResourceType.WEBHOOK_ENDPOINTS.value: {"limit": 100, "current_usage": 0, "reset_period": "monthly"},
				ResourceType.CUSTOM_PROCESSORS.value: {"limit": 10, "current_usage": 0, "reset_period": "monthly"},
				ResourceType.CONCURRENT_CONNECTIONS.value: {"limit": 1000, "current_usage": 0, "reset_period": "monthly"}
			}
		}
		
		self._default_feature_flags = {
			TenantPlan.FREE: {
				"advanced_fraud_detection": False,
				"custom_webhooks": False,
				"api_rate_limiting": True,
				"basic_analytics": True
			},
			TenantPlan.BASIC: {
				"advanced_fraud_detection": True,
				"custom_webhooks": True,
				"api_rate_limiting": True,
				"basic_analytics": True,
				"subscription_billing": True
			},
			TenantPlan.PROFESSIONAL: {
				"advanced_fraud_detection": True,
				"custom_webhooks": True,
				"api_rate_limiting": True,
				"basic_analytics": True,
				"advanced_analytics": True,
				"subscription_billing": True,
				"custom_processors": True
			},
			TenantPlan.ENTERPRISE: {
				"advanced_fraud_detection": True,
				"custom_webhooks": True,
				"api_rate_limiting": True,
				"basic_analytics": True,
				"advanced_analytics": True,
				"subscription_billing": True,
				"custom_processors": True,
				"dedicated_infrastructure": True,
				"priority_support": True
			}
		}
		
		self._default_processors = {
			TenantPlan.FREE: ["stripe_test"],
			TenantPlan.BASIC: ["stripe", "paypal"],
			TenantPlan.PROFESSIONAL: ["stripe", "paypal", "adyen", "mpesa"],
			TenantPlan.ENTERPRISE: ["stripe", "paypal", "adyen", "mpesa", "custom"]
		}
	
	async def _initialize_resource_monitoring(self):
		"""Initialize resource monitoring system"""
		self._resource_monitors = {}
		logger.info("resource_monitoring_initialized")
	
	async def _setup_isolation_rules(self):
		"""Setup tenant isolation rules"""
		self._isolation_rules = {
			"database_isolation": "schema_per_tenant",
			"api_isolation": "api_key_tenant_scoping",
			"webhook_isolation": "tenant_specific_endpoints",
			"storage_isolation": "tenant_prefixed_keys"
		}
		logger.info("isolation_rules_configured", rules=list(self._isolation_rules.keys()))
	
	def _validate_tenant_creation_data(self, data: Dict[str, Any]) -> List[str]:
		"""Validate tenant creation data"""
		errors = []
		
		required_fields = ['name', 'slug', 'business_type', 'country', 'billing_email']
		for field in required_fields:
			if field not in data or not data[field]:
				errors.append(f"Missing required field: {field}")
		
		if 'slug' in data:
			import re
			if not re.match(r'^[a-z0-9-]+$', data['slug']):
				errors.append("Slug must contain only lowercase letters, numbers, and hyphens")
		
		if 'country' in data and len(data['country']) != 2:
			errors.append("Country must be 2-letter ISO code")
		
		if 'billing_email' in data:
			import re
			if not re.match(r'^[^@]+@[^@]+\.[^@]+$', data['billing_email']):
				errors.append("Invalid billing email format")
		
		return errors
	
	def _validate_tenant_updates(self, updates: Dict[str, Any]) -> List[str]:
		"""Validate tenant update data"""
		errors = []
		
		# Validate specific fields if present
		if 'slug' in updates:
			import re
			if not re.match(r'^[a-z0-9-]+$', updates['slug']):
				errors.append("Slug must contain only lowercase letters, numbers, and hyphens")
		
		if 'country' in updates and len(updates['country']) != 2:
			errors.append("Country must be 2-letter ISO code")
		
		return errors
	
	async def _is_slug_taken(self, slug: str) -> bool:
		"""Check if tenant slug is already taken"""
		existing_tenant = await self.get_tenant_by_slug(slug)
		return existing_tenant is not None
	
	def _get_default_resource_limits(self, plan: TenantPlan) -> Dict[str, Dict[str, Any]]:
		"""Get default resource limits for plan"""
		return self._default_resource_limits.get(plan, {}).copy()
	
	def _get_default_feature_flags(self, plan: TenantPlan) -> Dict[str, bool]:
		"""Get default feature flags for plan"""
		return self._default_feature_flags.get(plan, {}).copy()
	
	def _get_default_processors(self, plan: TenantPlan) -> List[str]:
		"""Get default allowed processors for plan"""
		return self._default_processors.get(plan, []).copy()
	
	async def _initialize_tenant_resource_monitoring(self, tenant_id: str):
		"""Initialize resource monitoring for tenant"""
		self._resource_monitors[tenant_id] = {}
		logger.info("tenant_resource_monitoring_initialized", tenant_id=tenant_id)
	
	async def _setup_tenant_isolation(self, tenant: Tenant):
		"""Setup isolation for new tenant"""
		# Create tenant-specific database schema
		if self._database_service:
			await self._database_service.create_tenant_schema(tenant.slug)
		
		logger.info("tenant_isolation_configured", tenant_id=tenant.id, slug=tenant.slug)
	
	async def _update_resource_usage(self, tenant_id: str, resource_type: ResourceType, amount: int):
		"""Update resource usage for tenant"""
		tenant = await self.get_tenant(tenant_id)
		if not tenant:
			return
		
		# Update resource usage in tenant model
		if resource_type.value in tenant.resource_limits:
			tenant.resource_limits[resource_type.value]["current_usage"] += amount
			
			# Update in database and cache
			await self.update_tenant(tenant_id, {
				"resource_limits": tenant.resource_limits,
				"last_activity_at": datetime.now(timezone.utc)
			})
	
	def _log_service_initialized(self):
		"""Log service initialization"""
		logger.info("multi_tenant_service_initialized", 
			isolation_rules=len(self._isolation_rules),
			default_plans=len(self._default_resource_limits)
		)

# Factory function
def create_multi_tenant_service(database_service=None) -> TenantIsolationService:
	"""Create and initialize multi-tenant service"""
	return TenantIsolationService(database_service)

# Test utility
async def test_multi_tenant_service():
	"""Test multi-tenant service functionality"""
	print("🏢 Testing Multi-Tenant Service")
	print("=" * 50)
	
	# Initialize service
	service = create_multi_tenant_service()
	await service.initialize()
	
	# Create test tenant
	tenant_data = {
		"name": "Test Company",
		"slug": "test-company",
		"business_type": "ecommerce",
		"industry": "retail",
		"country": "US",
		"timezone": "America/New_York",
		"plan": "professional",
		"billing_email": "billing@testcompany.com"
	}
	
	tenant = await service.create_tenant(tenant_data)
	print(f"✅ Created tenant: {tenant.name} ({tenant.id})")
	
	# Test resource limits
	resource_check = await service.check_resource_limit(
		tenant.id, 
		ResourceType.TRANSACTIONS_PER_MONTH, 
		50
	)
	print(f"📊 Resource check: {resource_check}")
	
	# Consume resources
	success = await service.consume_resource(
		tenant.id, 
		ResourceType.TRANSACTIONS_PER_MONTH, 
		25
	)
	print(f"💰 Resource consumption: {'success' if success else 'failed'}")
	
	# Get usage report
	usage_report = await service.get_tenant_usage_report(tenant.id)
	print(f"📈 Usage report generated: {len(usage_report.get('resources', {}))} resource types")
	
	print("🎉 Multi-tenant service test completed!")

if __name__ == "__main__":
	asyncio.run(test_multi_tenant_service())