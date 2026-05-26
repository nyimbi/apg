#!/usr/bin/env python3
"""
Registry (regy) - APG Data Models
=================================

Comprehensive data models for API/Service Registry with intelligent service discovery,
health monitoring, and circuit breaking within the APG ecosystem.

Author: APG Platform Team
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal, Set
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, validator, field_validator
from pydantic.types import PositiveInt, NonNegativeFloat, PositiveFloat
from pydantic import AfterValidator
from typing_extensions import Annotated

# Use uuid7 for time-based UUIDs - APG standard
try:
	from uuid_extensions import uuid7str
except ImportError:
	import uuid
	def uuid7str() -> str:
		return str(uuid.uuid4())

# APG Model Configuration Standards - CLAUDE.md compliance
APG_MODEL_CONFIG = ConfigDict(
	extra='forbid',
	validate_by_name=True, 
	validate_by_alias=True,
	str_strip_whitespace=True,
	validate_default=True,
	use_enum_values=True
)

class ServiceStatus(str, Enum):
	"""Service operational status."""
	STARTING = "starting"
	HEALTHY = "healthy" 
	DEGRADED = "degraded"
	UNHEALTHY = "unhealthy"
	CRITICAL = "critical"
	MAINTENANCE = "maintenance"
	STOPPED = "stopped"
	UNKNOWN = "unknown"

class ServiceType(str, Enum):
	"""Types of registered services."""
	REST_API = "rest_api"
	GRAPHQL_API = "graphql_api"  
	GRPC_SERVICE = "grpc_service"
	WEBSOCKET_SERVICE = "websocket_service"
	WEB_SERVICE = "web_service"
	MESSAGE_QUEUE = "message_queue"
	DATABASE = "database"
	CACHE_SERVICE = "cache_service"
	AUTH_SERVICE = "auth_service"
	WEB_APPLICATION = "web_application"
	MICROSERVICE = "microservice"
	BATCH_SERVICE = "batch_service"
	AI_SERVICE = "ai_service"
	CUSTOM = "custom"

class HealthCheckType(str, Enum):
	"""Types of health check methods."""
	HTTP = "http"
	HTTPS = "https"
	TCP = "tcp"
	UDP = "udp" 
	GRPC = "grpc"
	DATABASE = "database"
	CUSTOM_SCRIPT = "custom_script"
	HEARTBEAT = "heartbeat"
	COMPOSITE = "composite"

class CircuitBreakerState(str, Enum):
	"""Circuit breaker operational states."""
	CLOSED = "closed"
	OPEN = "open" 
	HALF_OPEN = "half_open"
	FORCED_OPEN = "forced_open"
	DISABLED = "disabled"

class LoadBalanceStrategy(str, Enum):
	"""Load balancing strategies for service instances."""
	ROUND_ROBIN = "round_robin"
	WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
	LEAST_CONNECTIONS = "least_connections"
	WEIGHTED_RESPONSE_TIME = "weighted_response_time"
	CONSISTENT_HASH = "consistent_hash"
	RANDOM = "random"
	IP_HASH = "ip_hash"
	GEOGRAPHIC = "geographic"
	AI_OPTIMIZED = "ai_optimized"

class ProtocolType(str, Enum):
	"""Supported communication protocols."""
	HTTP = "http"
	HTTPS = "https"
	HTTP2 = "http2"
	HTTP3 = "http3"
	GRPC = "grpc"
	WEBSOCKET = "websocket"
	TCP = "tcp"
	UDP = "udp"
	MQTT = "mqtt"
	AMQP = "amqp"
	REDIS = "redis"

def validate_port_range(port: int) -> int:
	"""Validate port is in valid range."""
	assert 1 <= port <= 65535, f"Port must be between 1 and 65535, got {port}"
	return port

def validate_url_format(url: str) -> str:
	"""Validate URL format."""
	assert url.startswith(('http://', 'https://', 'grpc://', 'tcp://', 'udp://')), \
		f"Invalid URL format: {url}"
	return url

def validate_version_format(version: str) -> str:
	"""Validate semantic version format."""
	import re
	pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
	assert re.match(pattern, version), f"Invalid semantic version format: {version}"
	return version

ValidatedPort = Annotated[int, AfterValidator(validate_port_range)]
ValidatedURL = Annotated[str, AfterValidator(validate_url_format)]  
ValidatedVersion = Annotated[str, AfterValidator(validate_version_format)]

class ServiceEndpoint(BaseModel):
	"""Service endpoint configuration."""
	model_config = APG_MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Unique endpoint ID")
	path: str = Field(description="Endpoint path or route")
	protocol: ProtocolType = Field(default=ProtocolType.HTTP, description="Communication protocol")
	port: ValidatedPort = Field(description="Service port number")
	host: str = Field(description="Service host address")
	base_url: ValidatedURL = Field(description="Complete endpoint URL")
	
	# Health and Performance
	timeout_seconds: PositiveInt = Field(default=30, description="Request timeout")
	max_retries: NonNegativeFloat = Field(default=3, description="Maximum retry attempts")
	circuit_breaker_enabled: bool = Field(default=True, description="Enable circuit breaker")
	
	# Metadata
	description: Optional[str] = Field(None, description="Endpoint description")
	tags: List[str] = Field(default_factory=list, description="Endpoint tags")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
	
	# APG Integration
	tenant_id: str = Field(description="APG tenant identifier")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str = Field(description="User who created endpoint")

class HealthCheck(BaseModel):
	"""Service health check configuration."""
	model_config = APG_MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Unique health check ID")
	name: str = Field(description="Health check name")
	type: HealthCheckType = Field(description="Health check method")
	enabled: bool = Field(default=True, description="Health check enabled status")
	
	# Check Configuration
	url: Optional[ValidatedURL] = Field(None, description="Health check URL")
	interval_seconds: PositiveInt = Field(default=30, description="Check interval")
	timeout_seconds: PositiveInt = Field(default=10, description="Check timeout")
	healthy_threshold: PositiveInt = Field(default=2, description="Consecutive successes for healthy")
	unhealthy_threshold: PositiveInt = Field(default=3, description="Consecutive failures for unhealthy")
	
	# Advanced Configuration  
	expected_response_codes: List[int] = Field(default_factory=lambda: [200], description="Expected HTTP codes")
	expected_response_body: Optional[str] = Field(None, description="Expected response content")
	custom_headers: Dict[str, str] = Field(default_factory=dict, description="Custom HTTP headers")
	custom_script: Optional[str] = Field(None, description="Custom health check script")
	
	# ML-Powered Features
	adaptive_intervals: bool = Field(default=False, description="AI-powered adaptive check intervals")
	anomaly_detection: bool = Field(default=False, description="ML anomaly detection")
	predictive_analysis: bool = Field(default=False, description="Predictive failure analysis")
	
	# APG Integration
	tenant_id: str = Field(description="APG tenant identifier")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str = Field(description="User who created health check")

class CircuitBreakerConfig(BaseModel):
	"""Circuit breaker configuration with ML optimization."""
	model_config = APG_MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Unique circuit breaker ID")
	name: str = Field(description="Circuit breaker name")
	enabled: bool = Field(default=True, description="Circuit breaker enabled status")
	state: CircuitBreakerState = Field(default=CircuitBreakerState.CLOSED, description="Current state")
	
	# Threshold Configuration
	failure_threshold: PositiveInt = Field(default=5, description="Failure count to open circuit")
	success_threshold: PositiveInt = Field(default=3, description="Success count to close circuit")
	timeout_seconds: PositiveInt = Field(default=60, description="Timeout before half-open retry")
	
	# Advanced Configuration
	failure_rate_threshold: PositiveFloat = Field(default=50.0, le=100.0, description="Failure rate % threshold")
	minimum_request_threshold: PositiveInt = Field(default=10, description="Minimum requests before evaluation")
	rolling_window_seconds: PositiveInt = Field(default=60, description="Rolling window for statistics")
	
	# ML-Powered Features
	adaptive_thresholds: bool = Field(default=False, description="AI-optimized threshold management")
	pattern_recognition: bool = Field(default=False, description="Failure pattern recognition")
	intelligent_recovery: bool = Field(default=False, description="ML-powered recovery strategies")
	
	# Statistics
	total_requests: int = Field(default=0, description="Total request count")
	failed_requests: int = Field(default=0, description="Failed request count") 
	last_failure_time: Optional[datetime] = Field(None, description="Last failure timestamp")
	last_success_time: Optional[datetime] = Field(None, description="Last success timestamp")
	
	# APG Integration
	tenant_id: str = Field(description="APG tenant identifier")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str = Field(description="User who created circuit breaker")

class ServiceVersion(BaseModel):
	"""Service API version information.""" 
	model_config = APG_MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Unique version ID")
	version: ValidatedVersion = Field(description="Semantic version number")
	is_current: bool = Field(default=False, description="Current active version")
	is_deprecated: bool = Field(default=False, description="Deprecated version status")
	
	# Version Lifecycle
	release_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	deprecation_date: Optional[datetime] = Field(None, description="Version deprecation date")
	end_of_life_date: Optional[datetime] = Field(None, description="Version EOL date")
	
	# API Contract
	api_schema_url: Optional[ValidatedURL] = Field(None, description="API schema definition URL")
	breaking_changes: List[str] = Field(default_factory=list, description="Breaking changes list")
	migration_guide_url: Optional[ValidatedURL] = Field(None, description="Migration guide URL")
	
	# Compatibility
	backward_compatible: bool = Field(default=True, description="Backward compatibility status")
	supported_clients: List[str] = Field(default_factory=list, description="Supported client versions")
	minimum_client_version: Optional[ValidatedVersion] = Field(None, description="Minimum client version")
	
	# Usage Analytics
	usage_count: int = Field(default=0, description="Version usage count")
	last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
	
	# APG Integration
	tenant_id: str = Field(description="APG tenant identifier") 
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str = Field(description="User who created version")

class ServiceInstance(BaseModel):
	"""Individual service instance registration."""
	model_config = APG_MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Unique instance ID")
	service_id: str = Field(description="Parent service ID")
	instance_name: str = Field(description="Instance identifier")
	
	# Network Configuration
	host: str = Field(description="Instance host address")
	port: ValidatedPort = Field(description="Instance port")
	base_url: ValidatedURL = Field(description="Instance base URL")
	endpoints: List[ServiceEndpoint] = Field(default_factory=list, description="Instance endpoints")
	
	# Status and Health
	status: ServiceStatus = Field(default=ServiceStatus.STARTING, description="Instance status")
	health_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Health score (0-1)")
	last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	
	# Performance Metrics
	cpu_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="CPU usage %")
	memory_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="Memory usage %")
	response_time_ms: Optional[float] = Field(None, ge=0.0, description="Average response time")
	requests_per_second: Optional[float] = Field(None, ge=0.0, description="Request rate")
	
	# Load Balancing
	weight: int = Field(default=100, ge=0, le=1000, description="Load balancing weight")
	max_connections: Optional[int] = Field(None, ge=0, description="Maximum connections")
	current_connections: int = Field(default=0, ge=0, description="Current connections")
	
	# Health Monitoring
	health_checks: List[HealthCheck] = Field(default_factory=list, description="Health check configurations")
	circuit_breakers: List[CircuitBreakerConfig] = Field(default_factory=list, description="Circuit breakers")
	
	# Deployment Info
	container_id: Optional[str] = Field(None, description="Container/Pod ID")
	node_id: Optional[str] = Field(None, description="Node/Host ID")
	deployment_version: Optional[ValidatedVersion] = Field(None, description="Deployment version")
	environment: Optional[str] = Field(None, description="Deployment environment")
	
	# Metadata
	tags: List[str] = Field(default_factory=list, description="Instance tags")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
	
	# APG Integration
	tenant_id: str = Field(description="APG tenant identifier")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	registered_by: str = Field(description="Service that registered instance")

class ServiceRegistration(BaseModel):
	"""Main service registration with intelligent features."""
	model_config = APG_MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Unique service ID")
	name: str = Field(description="Service name")
	display_name: str = Field(description="Human-readable service name")
	description: Optional[str] = Field(None, description="Service description")
	service_type: ServiceType = Field(description="Type of service")
	
	# Service Configuration
	namespace: str = Field(default="default", description="Service namespace")
	environment: str = Field(description="Deployment environment")
	base_path: str = Field(default="/", description="Service base path")
	
	# Instances and Endpoints
	instances: List[ServiceInstance] = Field(default_factory=list, description="Service instances")
	versions: List[ServiceVersion] = Field(default_factory=list, description="API versions")
	current_version: Optional[ValidatedVersion] = Field(None, description="Current service version")
	
	# Service Discovery
	discovery_enabled: bool = Field(default=True, description="Enable service discovery")
	load_balance_strategy: LoadBalanceStrategy = Field(default=LoadBalanceStrategy.ROUND_ROBIN)
	sticky_sessions: bool = Field(default=False, description="Enable session affinity")
	
	# Health and Resilience
	health_check_enabled: bool = Field(default=True, description="Enable health monitoring")
	circuit_breaker_enabled: bool = Field(default=True, description="Enable circuit breaker")
	auto_scaling_enabled: bool = Field(default=False, description="Enable auto-scaling")
	
	# AI/ML Features
	predictive_scaling: bool = Field(default=False, description="AI-powered predictive scaling")
	intelligent_routing: bool = Field(default=False, description="ML-optimized routing")
	anomaly_detection: bool = Field(default=False, description="Service behavior anomaly detection")
	performance_optimization: bool = Field(default=False, description="AI performance optimization")
	
	# Security and Access
	authentication_required: bool = Field(default=False, description="Require authentication")
	authorization_policies: List[str] = Field(default_factory=list, description="Authorization policies")
	rate_limiting_enabled: bool = Field(default=False, description="Enable rate limiting")
	cors_enabled: bool = Field(default=False, description="Enable CORS")
	
	# Dependencies
	dependencies: List[str] = Field(default_factory=list, description="Service dependency IDs")
	dependents: List[str] = Field(default_factory=list, description="Dependent service IDs")
	
	# Metrics and Analytics
	total_requests: int = Field(default=0, description="Total request count")
	total_errors: int = Field(default=0, description="Total error count")
	average_response_time: float = Field(default=0.0, ge=0.0, description="Average response time")
	uptime_percentage: float = Field(default=100.0, ge=0.0, le=100.0, description="Service uptime %")
	
	# Lifecycle Management
	status: ServiceStatus = Field(default=ServiceStatus.STARTING, description="Service status")
	registration_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	ttl_seconds: Optional[int] = Field(None, ge=0, description="Time-to-live in seconds")
	
	# Metadata and Tags
	tags: List[str] = Field(default_factory=list, description="Service tags")
	labels: Dict[str, str] = Field(default_factory=dict, description="Service labels")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
	
	# APG Integration
	tenant_id: str = Field(description="APG tenant identifier")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str = Field(description="User who registered service")
	last_modified_by: str = Field(description="User who last modified service")

class ServiceDiscoveryQuery(BaseModel):
	"""Service discovery query with intelligent filtering."""
	model_config = APG_MODEL_CONFIG
	
	# Basic Query Parameters
	service_name: Optional[str] = Field(None, description="Service name filter")
	service_type: Optional[ServiceType] = Field(None, description="Service type filter")
	namespace: Optional[str] = Field(None, description="Namespace filter")
	environment: Optional[str] = Field(None, description="Environment filter")
	
	# Status and Health Filtering
	status: Optional[ServiceStatus] = Field(None, description="Service status filter")
	healthy_only: bool = Field(default=True, description="Return only healthy services")
	min_health_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum health score")
	
	# Performance Filtering
	max_response_time: Optional[float] = Field(None, ge=0.0, description="Maximum response time")
	min_availability: Optional[float] = Field(None, ge=0.0, le=100.0, description="Minimum availability %")
	
	# Geographic and Load Balancing
	preferred_regions: List[str] = Field(default_factory=list, description="Preferred geographic regions")
	load_balance_strategy: Optional[LoadBalanceStrategy] = Field(None, description="Load balancing preference")
	
	# Advanced Filtering
	tags: List[str] = Field(default_factory=list, description="Required tags")
	labels: Dict[str, str] = Field(default_factory=dict, description="Required labels")
	version_constraints: Optional[str] = Field(None, description="Version constraint expression")
	
	# AI/ML Query Features
	intelligent_ranking: bool = Field(default=False, description="AI-powered result ranking")
	predictive_filtering: bool = Field(default=False, description="Predictive availability filtering")
	similarity_search: bool = Field(default=False, description="Service similarity search")
	
	# Result Configuration
	limit: int = Field(default=50, ge=1, le=1000, description="Maximum results")
	offset: int = Field(default=0, ge=0, description="Result offset")
	include_instances: bool = Field(default=True, description="Include instance details")
	include_health: bool = Field(default=True, description="Include health information")
	include_metrics: bool = Field(default=False, description="Include performance metrics")
	
	# APG Integration
	tenant_id: str = Field(description="APG tenant identifier")

class ServiceDiscoveryResult(BaseModel):
	"""Service discovery result with intelligent ranking."""
	model_config = APG_MODEL_CONFIG
	
	# Result Metadata
	query_id: str = Field(default_factory=uuid7str, description="Unique query ID")
	total_count: int = Field(ge=0, description="Total matching services")
	returned_count: int = Field(ge=0, description="Services in this result")
	query_time_ms: float = Field(ge=0.0, description="Query execution time")
	
	# Service Results
	services: List[ServiceRegistration] = Field(default_factory=list, description="Matching services")
	
	# AI/ML Insights
	ranking_algorithm: Optional[str] = Field(None, description="Ranking algorithm used")
	confidence_scores: Dict[str, float] = Field(default_factory=dict, description="AI confidence scores")
	recommendations: List[str] = Field(default_factory=list, description="AI recommendations")
	
	# Performance Analytics
	average_response_time: float = Field(default=0.0, ge=0.0, description="Average response time")
	average_health_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Average health score")
	geographic_distribution: Dict[str, int] = Field(default_factory=dict, description="Geographic spread")
	
	# Caching Information
	cached_result: bool = Field(default=False, description="Result from cache")
	cache_ttl_seconds: Optional[int] = Field(None, description="Cache TTL remaining")
	
	# APG Integration
	tenant_id: str = Field(description="APG tenant identifier")
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ServiceHealthStatus(BaseModel):
	"""Comprehensive service health status."""
	model_config = APG_MODEL_CONFIG
	
	service_id: str = Field(description="Service identifier")
	instance_id: str = Field(description="Instance identifier") 
	
	# Health Assessment
	overall_status: ServiceStatus = Field(description="Overall health status")
	health_score: float = Field(ge=0.0, le=1.0, description="Composite health score")
	status_message: str = Field(description="Health status message")
	
	# Health Check Results
	http_health: Optional[bool] = Field(None, description="HTTP health check result")
	tcp_health: Optional[bool] = Field(None, description="TCP health check result")
	database_health: Optional[bool] = Field(None, description="Database health result")
	dependency_health: Optional[bool] = Field(None, description="Dependencies health result")
	
	# Performance Metrics
	response_time_ms: float = Field(default=0.0, ge=0.0, description="Current response time")
	cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="CPU usage percentage")
	memory_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="Memory usage percentage")
	active_connections: int = Field(default=0, ge=0, description="Active connection count")
	
	# Circuit Breaker Status
	circuit_breaker_state: CircuitBreakerState = Field(default=CircuitBreakerState.CLOSED, description="Circuit breaker state")
	failure_count: int = Field(default=0, ge=0, description="Recent failure count")
	last_failure_time: Optional[datetime] = Field(None, description="Last failure timestamp")
	
	# ML-Powered Insights
	anomaly_detected: bool = Field(default=False, description="Anomaly detection result")
	predicted_failure_probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Failure probability")
	recommended_actions: List[str] = Field(default_factory=list, description="AI recommendations")
	
	# Timestamps
	last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	next_check_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(seconds=30))
	
	# APG Integration
	tenant_id: str = Field(description="APG tenant identifier")

class ServiceEvent(BaseModel):
	"""Service registry events for audit and monitoring."""
	model_config = APG_MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Unique event ID")
	event_type: str = Field(description="Type of registry event")
	service_id: str = Field(description="Affected service ID")
	instance_id: Optional[str] = Field(None, description="Affected instance ID")
	
	# Event Details
	severity: Literal["info", "warning", "error", "critical"] = Field(description="Event severity")
	message: str = Field(description="Event description")
	details: Dict[str, Any] = Field(default_factory=dict, description="Event details")
	
	# State Changes
	previous_state: Optional[str] = Field(None, description="Previous state")
	new_state: Optional[str] = Field(None, description="New state")
	
	# Context Information
	triggered_by: str = Field(description="Event trigger source")
	correlation_id: Optional[str] = Field(None, description="Event correlation ID")
	tags: List[str] = Field(default_factory=list, description="Event tags")
	
	# Performance Impact
	performance_impact: Optional[str] = Field(None, description="Performance impact description")
	affected_users: int = Field(default=0, ge=0, description="Number of affected users")
	
	# Resolution
	resolved: bool = Field(default=False, description="Event resolution status")
	resolution_time: Optional[datetime] = Field(None, description="Resolution timestamp")
	resolution_notes: Optional[str] = Field(None, description="Resolution details")
	
	# APG Integration
	tenant_id: str = Field(description="APG tenant identifier")
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str = Field(description="Event creator")

class ServiceMetrics(BaseModel):
	"""Service performance and usage metrics."""
	model_config = APG_MODEL_CONFIG
	
	service_id: str = Field(description="Service identifier")
	instance_id: Optional[str] = Field(None, description="Instance identifier")
	metric_type: str = Field(description="Type of metric")
	
	# Time Series Data
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	time_window_seconds: int = Field(default=1, ge=1, description="Metric aggregation window")
	
	# Performance Metrics
	request_count: int = Field(default=0, ge=0, description="Request count")
	error_count: int = Field(default=0, ge=0, description="Error count")
	response_time_p50: float = Field(default=0.0, ge=0.0, description="50th percentile response time")
	response_time_p95: float = Field(default=0.0, ge=0.0, description="95th percentile response time")
	response_time_p99: float = Field(default=0.0, ge=0.0, description="99th percentile response time")
	
	# Resource Utilization
	cpu_usage_avg: float = Field(default=0.0, ge=0.0, le=100.0, description="Average CPU usage %")
	memory_usage_avg: float = Field(default=0.0, ge=0.0, le=100.0, description="Average memory usage %")
	disk_usage_avg: Optional[float] = Field(None, ge=0.0, le=100.0, description="Average disk usage %")
	network_bytes_in: int = Field(default=0, ge=0, description="Network bytes received")
	network_bytes_out: int = Field(default=0, ge=0, description="Network bytes sent")
	
	# Availability Metrics
	uptime_seconds: int = Field(default=0, ge=0, description="Uptime in seconds")
	downtime_seconds: int = Field(default=0, ge=0, description="Downtime in seconds")
	availability_percentage: float = Field(default=100.0, ge=0.0, le=100.0, description="Availability percentage")
	
	# Business Metrics
	active_users: int = Field(default=0, ge=0, description="Active user count")
	successful_transactions: int = Field(default=0, ge=0, description="Successful transaction count")
	revenue_impact: Optional[float] = Field(None, description="Revenue impact amount")
	
	# Custom Metrics
	custom_metrics: Dict[str, float] = Field(default_factory=dict, description="Custom metric values")
	
	# APG Integration
	tenant_id: str = Field(description="APG tenant identifier")

class _FieldMap(dict):
	"""Pydantic field map with attribute access for legacy REGY checks."""

	def __getattr__(self, name: str) -> Any:
		try:
			return self[name]
		except KeyError as exc:
			raise AttributeError(name) from exc

for _model in (
	ServiceEndpoint, HealthCheck, CircuitBreakerConfig, ServiceVersion,
	ServiceInstance, ServiceRegistration, ServiceDiscoveryQuery,
	ServiceDiscoveryResult, ServiceHealthStatus, ServiceEvent, ServiceMetrics
):
	_model.model_fields = _FieldMap(_model.model_fields)

# Export all models for easy importing
__all__ = [
	"ServiceStatus", "ServiceType", "HealthCheckType", "CircuitBreakerState", 
	"LoadBalanceStrategy", "ProtocolType", "ServiceEndpoint", "HealthCheck",
	"CircuitBreakerConfig", "ServiceVersion", "ServiceInstance", "ServiceRegistration",
	"ServiceDiscoveryQuery", "ServiceDiscoveryResult", "ServiceHealthStatus", 
	"ServiceEvent", "ServiceMetrics", "ValidatedPort", "ValidatedURL", "ValidatedVersion"
]
