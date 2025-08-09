"""
APG Integration API Management - Data Models

Comprehensive data models for API gateway, lifecycle management, security,
analytics, and developer portal with full SQLAlchemy and Pydantic integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from sqlalchemy import (
	Column, String, Text, Integer, Float, Boolean, DateTime, JSON,
	ForeignKey, Index, UniqueConstraint, CheckConstraint, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import JSONB, UUID

from pydantic import (
	BaseModel, Field, ConfigDict, validator, root_validator,
	AfterValidator, field_validator
)
from uuid_extensions import uuid7str

# =============================================================================
# Database Base and Configuration
# =============================================================================

Base = declarative_base()

# Common model configuration
model_config = ConfigDict(
	extra='forbid',
	validate_by_name=True,
	validate_by_alias=True,
	str_strip_whitespace=True,
	use_enum_values=True
)

# =============================================================================
# Enums and Constants
# =============================================================================

class APIStatus(str, Enum):
	"""API lifecycle status."""
	DRAFT = "draft"
	ACTIVE = "active"
	DEPRECATED = "deprecated"
	RETIRED = "retired"
	BLOCKED = "blocked"

class APIVersion(str, Enum):
	"""API versioning strategy."""
	MAJOR = "major"
	MINOR = "minor"
	PATCH = "patch"

class ProtocolType(str, Enum):
	"""Supported API protocols."""
	REST = "rest"
	GRAPHQL = "graphql"
	GRPC = "grpc"
	WEBSOCKET = "websocket"
	SOAP = "soap"

class AuthenticationType(str, Enum):
	"""Authentication methods."""
	NONE = "none"
	API_KEY = "api_key"
	OAUTH2 = "oauth2"
	JWT = "jwt"
	BASIC = "basic"
	MTLS = "mtls"
	CUSTOM = "custom"

class PolicyType(str, Enum):
	"""API policy types."""
	RATE_LIMITING = "rate_limiting"
	AUTHENTICATION = "authentication"
	AUTHORIZATION = "authorization"
	TRANSFORMATION = "transformation"
	CACHING = "caching"
	LOGGING = "logging"
	VALIDATION = "validation"
	CORS = "cors"

class DeploymentStrategy(str, Enum):
	"""API deployment strategies."""
	ROLLING = "rolling"
	BLUE_GREEN = "blue_green"
	CANARY = "canary"
	IMMEDIATE = "immediate"

class ConsumerStatus(str, Enum):
	"""API consumer status."""
	ACTIVE = "active"
	SUSPENDED = "suspended"
	PENDING = "pending"
	REJECTED = "rejected"

class MetricType(str, Enum):
	"""Analytics metric types."""
	COUNTER = "counter"
	GAUGE = "gauge"
	HISTOGRAM = "histogram"
	TIMER = "timer"

class LoadBalancingAlgorithm(str, Enum):
	"""Load balancing algorithms."""
	ROUND_ROBIN = "round_robin"
	WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
	LEAST_CONNECTIONS = "least_connections"
	IP_HASH = "ip_hash"
	RANDOM = "random"

# =============================================================================
# Core API Model
# =============================================================================

class AMAPI(Base):
	"""Core API registration and metadata model."""
	
	__tablename__ = "am_apis"
	
	# Primary identification
	api_id = Column(String(36), primary_key=True, default=lambda: f"api_{uuid7str()}")
	api_name = Column(String(200), nullable=False, index=True)
	api_title = Column(String(300), nullable=False)
	api_description = Column(Text, nullable=True)
	
	# Versioning
	version = Column(String(50), nullable=False, default="1.0.0")
	version_strategy = Column(String(20), nullable=False, default=APIVersion.MINOR.value)
	
	# Protocol and technical details
	protocol_type = Column(String(20), nullable=False, default=ProtocolType.REST.value)
	base_path = Column(String(500), nullable=False)
	upstream_url = Column(String(1000), nullable=False)
	
	# Status and lifecycle
	status = Column(String(20), nullable=False, default=APIStatus.DRAFT.value)
	is_public = Column(Boolean, nullable=False, default=False)
	
	# Documentation
	documentation_url = Column(String(1000), nullable=True)
	openapi_spec = Column(JSONB, nullable=True)
	graphql_schema = Column(Text, nullable=True)
	
	# Configuration
	timeout_ms = Column(Integer, nullable=False, default=30000)
	retry_attempts = Column(Integer, nullable=False, default=3)
	load_balancing_algorithm = Column(String(30), nullable=False, default=LoadBalancingAlgorithm.ROUND_ROBIN.value)
	
	# Authentication
	auth_type = Column(String(20), nullable=False, default=AuthenticationType.API_KEY.value)
	auth_config = Column(JSONB, nullable=False, default={})
	
	# Rate limiting defaults
	default_rate_limit = Column(Integer, nullable=True)
	default_quota_limit = Column(Integer, nullable=True)
	
	# Categorization and organization
	category = Column(String(100), nullable=True)
	tags = Column(JSONB, nullable=False, default=[])
	
	# Multi-tenancy
	tenant_id = Column(String(100), nullable=False, index=True)
	capability_id = Column(String(100), nullable=False, index=True)
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	created_by = Column(String(100), nullable=False)
	updated_by = Column(String(100), nullable=True)
	
	# Relationships
	endpoints = relationship("AMEndpoint", back_populates="api", cascade="all, delete-orphan")
	policies = relationship("AMPolicy", back_populates="api", cascade="all, delete-orphan")
	deployments = relationship("AMDeployment", back_populates="api")
	analytics = relationship("AMAnalytics", back_populates="api")
	
	# Indexes
	__table_args__ = (
		Index('idx_am_apis_tenant_capability', 'tenant_id', 'capability_id'),
		Index('idx_am_apis_name_version', 'api_name', 'version'),
		Index('idx_am_apis_status', 'status'),
		Index('idx_am_apis_public', 'is_public'),
		UniqueConstraint('api_name', 'version', 'tenant_id', name='uq_api_name_version_tenant'),
		CheckConstraint("timeout_ms > 0", name='check_timeout_positive'),
		CheckConstraint("retry_attempts >= 0", name='check_retry_non_negative')
	)
	
	@validates('api_name')
	def validate_api_name(self, key, value):
		if not value or len(value.strip()) == 0:
			raise ValueError("API name cannot be empty")
		# API name should be URL-safe
		if not all(c.isalnum() or c in '._-' for c in value):
			raise ValueError("API name can only contain alphanumeric characters, dots, underscores, and hyphens")
		return value.strip()
	
	@validates('base_path')
	def validate_base_path(self, key, value):
		if not value.startswith('/'):
			raise ValueError("Base path must start with '/'")
		return value
	
	def __repr__(self):
		return f"<AMAPI(id={self.api_id}, name={self.api_name}, version={self.version})>"

# =============================================================================
# API Endpoint Model
# =============================================================================

class AMEndpoint(Base):
	"""Individual API endpoint configuration."""
	
	__tablename__ = "am_endpoints"
	
	# Primary identification
	endpoint_id = Column(String(36), primary_key=True, default=lambda: f"ep_{uuid7str()}")
	api_id = Column(String(36), ForeignKey("am_apis.api_id"), nullable=False, index=True)
	
	# Endpoint details
	path = Column(String(500), nullable=False)
	method = Column(String(10), nullable=False)  # GET, POST, PUT, DELETE, etc.
	operation_id = Column(String(200), nullable=True)
	summary = Column(String(300), nullable=True)
	description = Column(Text, nullable=True)
	
	# Request/Response specifications
	request_schema = Column(JSONB, nullable=True)
	response_schema = Column(JSONB, nullable=True)
	parameters = Column(JSONB, nullable=False, default=[])
	
	# Security and policies
	auth_required = Column(Boolean, nullable=False, default=True)
	scopes_required = Column(JSONB, nullable=False, default=[])
	rate_limit_override = Column(Integer, nullable=True)
	
	# Caching
	cache_enabled = Column(Boolean, nullable=False, default=False)
	cache_ttl_seconds = Column(Integer, nullable=True)
	
	# Documentation
	deprecated = Column(Boolean, nullable=False, default=False)
	examples = Column(JSONB, nullable=False, default={})
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Relationships
	api = relationship("AMAPI", back_populates="endpoints")
	
	# Indexes
	__table_args__ = (
		Index('idx_am_endpoints_api_path', 'api_id', 'path', 'method'),
		Index('idx_am_endpoints_deprecated', 'deprecated'),
		UniqueConstraint('api_id', 'path', 'method', name='uq_endpoint_path_method'),
		CheckConstraint("method IN ('GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS')", 
						name='check_valid_http_method')
	)
	
	@validates('path')
	def validate_path(self, key, value):
		if not value.startswith('/'):
			raise ValueError("Endpoint path must start with '/'")
		return value
	
	def __repr__(self):
		return f"<AMEndpoint(id={self.endpoint_id}, method={self.method}, path={self.path})>"

# =============================================================================
# Policy Model
# =============================================================================

class AMPolicy(Base):
	"""API policies for security, rate limiting, transformation, etc."""
	
	__tablename__ = "am_policies"
	
	# Primary identification
	policy_id = Column(String(36), primary_key=True, default=lambda: f"pol_{uuid7str()}")
	api_id = Column(String(36), ForeignKey("am_apis.api_id"), nullable=False, index=True)
	
	# Policy details
	policy_name = Column(String(200), nullable=False)
	policy_type = Column(String(30), nullable=False, index=True)
	policy_description = Column(Text, nullable=True)
	
	# Configuration
	config = Column(JSONB, nullable=False, default={})
	
	# Execution order and conditions
	execution_order = Column(Integer, nullable=False, default=100)
	enabled = Column(Boolean, nullable=False, default=True)
	conditions = Column(JSONB, nullable=False, default={})
	
	# Scope
	applies_to_endpoints = Column(JSONB, nullable=False, default=[])  # Empty means all endpoints
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	created_by = Column(String(100), nullable=False)
	
	# Relationships
	api = relationship("AMAPI", back_populates="policies")
	
	# Indexes
	__table_args__ = (
		Index('idx_am_policies_api_type', 'api_id', 'policy_type'),
		Index('idx_am_policies_enabled', 'enabled'),
		Index('idx_am_policies_execution_order', 'execution_order'),
		CheckConstraint("execution_order >= 0", name='check_execution_order_non_negative')
	)
	
	@validates('policy_type')
	def validate_policy_type(self, key, value):
		if value not in [pt.value for pt in PolicyType]:
			raise ValueError(f"Invalid policy type: {value}")
		return value
	
	def __repr__(self):
		return f"<AMPolicy(id={self.policy_id}, name={self.policy_name}, type={self.policy_type})>"

# =============================================================================
# Consumer and API Key Model
# =============================================================================

class AMConsumer(Base):
	"""API consumers (developers/applications)."""
	
	__tablename__ = "am_consumers"
	
	# Primary identification
	consumer_id = Column(String(36), primary_key=True, default=lambda: f"con_{uuid7str()}")
	consumer_name = Column(String(200), nullable=False)
	
	# Organization details
	organization = Column(String(300), nullable=True)
	contact_email = Column(String(255), nullable=False)
	contact_name = Column(String(200), nullable=True)
	
	# Status and approval
	status = Column(String(20), nullable=False, default=ConsumerStatus.PENDING.value)
	approval_date = Column(DateTime(timezone=True), nullable=True)
	approved_by = Column(String(100), nullable=True)
	
	# Access control
	allowed_apis = Column(JSONB, nullable=False, default=[])  # Empty means all public APIs
	ip_whitelist = Column(JSONB, nullable=False, default=[])
	
	# Rate limiting
	global_rate_limit = Column(Integer, nullable=True)
	global_quota_limit = Column(Integer, nullable=True)
	
	# Developer portal access
	portal_access = Column(Boolean, nullable=False, default=True)
	
	# Multi-tenancy
	tenant_id = Column(String(100), nullable=False, index=True)
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	created_by = Column(String(100), nullable=False)
	
	# Relationships
	api_keys = relationship("AMAPIKey", back_populates="consumer", cascade="all, delete-orphan")
	subscriptions = relationship("AMSubscription", back_populates="consumer", cascade="all, delete-orphan")
	usage_records = relationship("AMUsageRecord", back_populates="consumer")
	
	# Indexes
	__table_args__ = (
		Index('idx_am_consumers_tenant', 'tenant_id'),
		Index('idx_am_consumers_status', 'status'),
		Index('idx_am_consumers_email', 'contact_email'),
		UniqueConstraint('consumer_name', 'tenant_id', name='uq_consumer_name_tenant')
	)
	
	def __repr__(self):
		return f"<AMConsumer(id={self.consumer_id}, name={self.consumer_name}, status={self.status})>"

class AMAPIKey(Base):
	"""API keys for authentication."""
	
	__tablename__ = "am_api_keys"
	
	# Primary identification
	key_id = Column(String(36), primary_key=True, default=lambda: f"key_{uuid7str()}")
	consumer_id = Column(String(36), ForeignKey("am_consumers.consumer_id"), nullable=False, index=True)
	
	# Key details
	key_name = Column(String(200), nullable=False)
	key_hash = Column(String(255), nullable=False, unique=True)  # Hashed API key
	key_prefix = Column(String(20), nullable=False)  # First few characters for identification
	
	# Scopes and permissions
	scopes = Column(JSONB, nullable=False, default=[])
	allowed_apis = Column(JSONB, nullable=False, default=[])  # Specific API access
	
	# Lifecycle
	active = Column(Boolean, nullable=False, default=True)
	expires_at = Column(DateTime(timezone=True), nullable=True)
	last_used_at = Column(DateTime(timezone=True), nullable=True)
	
	# Rate limiting overrides
	rate_limit_override = Column(Integer, nullable=True)
	quota_limit_override = Column(Integer, nullable=True)
	
	# Security
	ip_restrictions = Column(JSONB, nullable=False, default=[])
	referer_restrictions = Column(JSONB, nullable=False, default=[])
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	created_by = Column(String(100), nullable=False)
	
	# Relationships
	consumer = relationship("AMConsumer", back_populates="api_keys")
	
	# Indexes
	__table_args__ = (
		Index('idx_am_api_keys_consumer', 'consumer_id'),
		Index('idx_am_api_keys_prefix', 'key_prefix'),
		Index('idx_am_api_keys_active', 'active'),
		Index('idx_am_api_keys_expires', 'expires_at'),
		UniqueConstraint('consumer_id', 'key_name', name='uq_consumer_key_name')
	)
	
	def __repr__(self):
		return f"<AMAPIKey(id={self.key_id}, name={self.key_name}, prefix={self.key_prefix})>"

# =============================================================================
# Subscription Model
# =============================================================================

class AMSubscription(Base):
	"""Consumer subscriptions to specific APIs."""
	
	__tablename__ = "am_subscriptions"
	
	# Primary identification
	subscription_id = Column(String(36), primary_key=True, default=lambda: f"sub_{uuid7str()}")
	consumer_id = Column(String(36), ForeignKey("am_consumers.consumer_id"), nullable=False, index=True)
	api_id = Column(String(36), ForeignKey("am_apis.api_id"), nullable=False, index=True)
	
	# Subscription details
	subscription_name = Column(String(200), nullable=False)
	plan_name = Column(String(100), nullable=True)  # e.g., "Basic", "Premium"
	
	# Status and lifecycle
	status = Column(String(20), nullable=False, default="active")
	starts_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	expires_at = Column(DateTime(timezone=True), nullable=True)
	
	# Usage limits
	rate_limit = Column(Integer, nullable=True)
	quota_limit = Column(Integer, nullable=True)
	burst_limit = Column(Integer, nullable=True)
	
	# Billing
	billing_model = Column(String(20), nullable=True)  # "free", "usage", "subscription"
	price_per_request = Column(Float, nullable=True)
	monthly_fee = Column(Float, nullable=True)
	
	# Configuration
	configuration = Column(JSONB, nullable=False, default={})
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Relationships
	consumer = relationship("AMConsumer", back_populates="subscriptions")
	api = relationship("AMAPI")
	
	# Indexes
	__table_args__ = (
		Index('idx_am_subscriptions_consumer_api', 'consumer_id', 'api_id'),
		Index('idx_am_subscriptions_status', 'status'),
		Index('idx_am_subscriptions_expires', 'expires_at'),
		UniqueConstraint('consumer_id', 'api_id', name='uq_consumer_api_subscription')
	)
	
	def __repr__(self):
		return f"<AMSubscription(id={self.subscription_id}, consumer={self.consumer_id}, api={self.api_id})>"

# =============================================================================
# Deployment Model
# =============================================================================

class AMDeployment(Base):
	"""API deployment tracking."""
	
	__tablename__ = "am_deployments"
	
	# Primary identification
	deployment_id = Column(String(36), primary_key=True, default=lambda: f"dep_{uuid7str()}")
	api_id = Column(String(36), ForeignKey("am_apis.api_id"), nullable=False, index=True)
	
	# Deployment details
	deployment_name = Column(String(200), nullable=False)
	strategy = Column(String(20), nullable=False, default=DeploymentStrategy.ROLLING.value)
	environment = Column(String(50), nullable=False)  # dev, staging, production
	
	# Version information
	from_version = Column(String(50), nullable=True)
	to_version = Column(String(50), nullable=False)
	
	# Status and progress
	status = Column(String(20), nullable=False, default="pending")
	progress_percentage = Column(Integer, nullable=False, default=0)
	
	# Deployment configuration
	config = Column(JSONB, nullable=False, default={})
	
	# Traffic management
	traffic_percentage = Column(Integer, nullable=False, default=0)
	
	# Rollback information
	rollback_available = Column(Boolean, nullable=False, default=True)
	rollback_reason = Column(Text, nullable=True)
	
	# Timing
	started_at = Column(DateTime(timezone=True), nullable=True)
	completed_at = Column(DateTime(timezone=True), nullable=True)
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	created_by = Column(String(100), nullable=False)
	
	# Relationships
	api = relationship("AMAPI", back_populates="deployments")
	
	# Indexes
	__table_args__ = (
		Index('idx_am_deployments_api_env', 'api_id', 'environment'),
		Index('idx_am_deployments_status', 'status'),
		Index('idx_am_deployments_started', 'started_at'),
		CheckConstraint("progress_percentage >= 0 AND progress_percentage <= 100", 
						name='check_progress_percentage'),
		CheckConstraint("traffic_percentage >= 0 AND traffic_percentage <= 100", 
						name='check_traffic_percentage')
	)
	
	def __repr__(self):
		return f"<AMDeployment(id={self.deployment_id}, api={self.api_id}, status={self.status})>"

# =============================================================================
# Analytics Models
# =============================================================================

class AMAnalytics(Base):
	"""API usage analytics and metrics."""
	
	__tablename__ = "am_analytics"
	
	# Primary identification
	metric_id = Column(String(36), primary_key=True, default=lambda: f"met_{uuid7str()}")
	
	# Resource identification
	api_id = Column(String(36), ForeignKey("am_apis.api_id"), nullable=True, index=True)
	endpoint_id = Column(String(36), nullable=True, index=True)
	consumer_id = Column(String(36), nullable=True, index=True)
	
	# Metric details
	metric_name = Column(String(100), nullable=False, index=True)
	metric_type = Column(String(20), nullable=False)
	metric_value = Column(Float, nullable=False)
	metric_unit = Column(String(20), nullable=True)
	
	# Dimensions
	dimensions = Column(JSONB, nullable=False, default={})
	
	# Time information
	timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
	time_bucket = Column(DateTime(timezone=True), nullable=False, index=True)
	aggregation_period = Column(String(10), nullable=False, default="1m")
	
	# Multi-tenancy
	tenant_id = Column(String(100), nullable=False, index=True)
	
	# Relationships
	api = relationship("AMAPI", back_populates="analytics")
	
	# Indexes
	__table_args__ = (
		Index('idx_am_analytics_tenant_time', 'tenant_id', 'time_bucket'),
		Index('idx_am_analytics_metric_time', 'metric_name', 'time_bucket'),
		Index('idx_am_analytics_api_time', 'api_id', 'time_bucket'),
		Index('idx_am_analytics_consumer_time', 'consumer_id', 'time_bucket')
	)
	
	def __repr__(self):
		return f"<AMAnalytics(id={self.metric_id}, metric={self.metric_name}, value={self.metric_value})>"

class AMUsageRecord(Base):
	"""Detailed API usage records for billing and analytics."""
	
	__tablename__ = "am_usage_records"
	
	# Primary identification
	record_id = Column(String(36), primary_key=True, default=lambda: f"usg_{uuid7str()}")
	
	# Request identification
	request_id = Column(String(100), nullable=False, unique=True)
	consumer_id = Column(String(36), ForeignKey("am_consumers.consumer_id"), nullable=False, index=True)
	api_id = Column(String(36), nullable=False, index=True)
	endpoint_path = Column(String(500), nullable=False)
	method = Column(String(10), nullable=False)
	
	# Request details
	timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
	response_status = Column(Integer, nullable=False)
	response_time_ms = Column(Integer, nullable=False)
	request_size_bytes = Column(Integer, nullable=True)
	response_size_bytes = Column(Integer, nullable=True)
	
	# Source information
	client_ip = Column(String(45), nullable=True)
	user_agent = Column(String(500), nullable=True)
	referer = Column(String(1000), nullable=True)
	
	# Geographic information
	country_code = Column(String(2), nullable=True)
	region = Column(String(100), nullable=True)
	
	# Billing information
	billable = Column(Boolean, nullable=False, default=True)
	cost = Column(Float, nullable=True)
	
	# Error details
	error_code = Column(String(50), nullable=True)
	error_message = Column(String(500), nullable=True)
	
	# Multi-tenancy
	tenant_id = Column(String(100), nullable=False, index=True)
	
	# Relationships
	consumer = relationship("AMConsumer", back_populates="usage_records")
	
	# Indexes
	__table_args__ = (
		Index('idx_am_usage_tenant_time', 'tenant_id', 'timestamp'),
		Index('idx_am_usage_consumer_time', 'consumer_id', 'timestamp'),
		Index('idx_am_usage_api_time', 'api_id', 'timestamp'),
		Index('idx_am_usage_status', 'response_status'),
		Index('idx_am_usage_billable', 'billable'),
		CheckConstraint("response_status >= 100 AND response_status < 600", 
						name='check_valid_http_status'),
		CheckConstraint("response_time_ms >= 0", name='check_response_time_non_negative')
	)
	
	def __repr__(self):
		return f"<AMUsageRecord(id={self.record_id}, consumer={self.consumer_id}, status={self.response_status})>"

# =============================================================================
# Pydantic Models for API
# =============================================================================

class APIConfig(BaseModel):
	"""API configuration model for API registration."""
	model_config = model_config
	
	api_name: str = Field(..., min_length=1, max_length=200)
	api_title: str = Field(..., min_length=1, max_length=300)
	api_description: Optional[str] = Field(None, max_length=2000)
	version: str = Field(default="1.0.0", max_length=50)
	protocol_type: ProtocolType = Field(default=ProtocolType.REST)
	base_path: str = Field(..., min_length=1, max_length=500)
	upstream_url: str = Field(..., min_length=1, max_length=1000)
	is_public: bool = Field(default=False)
	documentation_url: Optional[str] = Field(None, max_length=1000)
	openapi_spec: Optional[Dict[str, Any]] = Field(None)
	timeout_ms: int = Field(default=30000, ge=1000, le=300000)
	retry_attempts: int = Field(default=3, ge=0, le=10)
	load_balancing_algorithm: LoadBalancingAlgorithm = Field(default=LoadBalancingAlgorithm.ROUND_ROBIN)
	auth_type: AuthenticationType = Field(default=AuthenticationType.API_KEY)
	auth_config: Dict[str, Any] = Field(default_factory=dict)
	default_rate_limit: Optional[int] = Field(None, ge=1)
	category: Optional[str] = Field(None, max_length=100)
	tags: List[str] = Field(default_factory=list)
	
	@field_validator('base_path')
	@classmethod
	def validate_base_path(cls, v):
		if not v.startswith('/'):
			raise ValueError('Base path must start with "/"')
		return v

class EndpointConfig(BaseModel):
	"""Endpoint configuration model."""
	model_config = model_config
	
	path: str = Field(..., min_length=1, max_length=500)
	method: str = Field(..., min_length=3, max_length=10)
	operation_id: Optional[str] = Field(None, max_length=200)
	summary: Optional[str] = Field(None, max_length=300)
	description: Optional[str] = Field(None, max_length=2000)
	request_schema: Optional[Dict[str, Any]] = Field(None)
	response_schema: Optional[Dict[str, Any]] = Field(None)
	parameters: List[Dict[str, Any]] = Field(default_factory=list)
	auth_required: bool = Field(default=True)
	scopes_required: List[str] = Field(default_factory=list)
	rate_limit_override: Optional[int] = Field(None, ge=1)
	cache_enabled: bool = Field(default=False)
	cache_ttl_seconds: Optional[int] = Field(None, ge=1)
	deprecated: bool = Field(default=False)
	examples: Dict[str, Any] = Field(default_factory=dict)
	
	@field_validator('method')
	@classmethod
	def validate_method(cls, v):
		valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
		if v.upper() not in valid_methods:
			raise ValueError(f'Method must be one of: {valid_methods}')
		return v.upper()
	
	@field_validator('path')
	@classmethod
	def validate_path(cls, v):
		if not v.startswith('/'):
			raise ValueError('Path must start with "/"')
		return v

class PolicyConfig(BaseModel):
	"""Policy configuration model."""
	model_config = model_config
	
	policy_name: str = Field(..., min_length=1, max_length=200)
	policy_type: PolicyType = Field(...)
	policy_description: Optional[str] = Field(None, max_length=2000)
	config: Dict[str, Any] = Field(...)
	execution_order: int = Field(default=100, ge=0, le=1000)
	enabled: bool = Field(default=True)
	conditions: Dict[str, Any] = Field(default_factory=dict)
	applies_to_endpoints: List[str] = Field(default_factory=list)

class ConsumerConfig(BaseModel):
	"""Consumer configuration model."""
	model_config = model_config
	
	consumer_name: str = Field(..., min_length=1, max_length=200)
	organization: Optional[str] = Field(None, max_length=300)
	contact_email: str = Field(..., min_length=5, max_length=255)
	contact_name: Optional[str] = Field(None, max_length=200)
	allowed_apis: List[str] = Field(default_factory=list)
	ip_whitelist: List[str] = Field(default_factory=list)
	global_rate_limit: Optional[int] = Field(None, ge=1)
	global_quota_limit: Optional[int] = Field(None, ge=1)
	portal_access: bool = Field(default=True)
	
	@field_validator('contact_email')
	@classmethod
	def validate_email(cls, v):
		import re
		if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
			raise ValueError('Invalid email address format')
		return v

class APIKeyConfig(BaseModel):
	"""API key configuration model."""
	model_config = model_config
	
	key_name: str = Field(..., min_length=1, max_length=200)
	scopes: List[str] = Field(default_factory=list)
	allowed_apis: List[str] = Field(default_factory=list)
	expires_at: Optional[datetime] = Field(None)
	rate_limit_override: Optional[int] = Field(None, ge=1)
	quota_limit_override: Optional[int] = Field(None, ge=1)
	ip_restrictions: List[str] = Field(default_factory=list)
	referer_restrictions: List[str] = Field(default_factory=list)

class SubscriptionConfig(BaseModel):
	"""Subscription configuration model."""
	model_config = model_config
	
	subscription_name: str = Field(..., min_length=1, max_length=200)
	api_id: str = Field(..., min_length=1, max_length=36)
	plan_name: Optional[str] = Field(None, max_length=100)
	expires_at: Optional[datetime] = Field(None)
	rate_limit: Optional[int] = Field(None, ge=1)
	quota_limit: Optional[int] = Field(None, ge=1)
	burst_limit: Optional[int] = Field(None, ge=1)
	billing_model: Optional[str] = Field(None, max_length=20)
	price_per_request: Optional[float] = Field(None, ge=0)
	monthly_fee: Optional[float] = Field(None, ge=0)
	configuration: Dict[str, Any] = Field(default_factory=dict)

# Export all models for external use
__all__ = [
	# SQLAlchemy models
	"Base",
	"AMAPI",
	"AMEndpoint",
	"AMPolicy",
	"AMConsumer",
	"AMAPIKey",
	"AMSubscription",
	"AMDeployment",
	"AMAnalytics",
	"AMUsageRecord",
	
	# Enums
	"APIStatus",
	"APIVersion",
	"ProtocolType",
	"AuthenticationType",
	"PolicyType",
	"DeploymentStrategy",
	"ConsumerStatus",
	"MetricType",
	"LoadBalancingAlgorithm",
	
	# Pydantic models
	"APIConfig",
	"EndpointConfig",
	"PolicyConfig",
	"ConsumerConfig",
	"APIKeyConfig",
	"SubscriptionConfig"
]