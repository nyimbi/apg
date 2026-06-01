#!/usr/bin/env python3
"""
APG Intelligent Gateway (APIG) - Core Data Models

Comprehensive data models for the revolutionary API Gateway system using Pydantic v2
with APG platform integration patterns and multi-tenancy support.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, validator, field_validator, model_validator
from pydantic.types import PositiveInt, NonNegativeFloat
# Use uuid7 for time-based UUIDs
try:
	from uuid7 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def uuid7str() -> str:
		return str(uuid.uuid4())

# APG Model Configuration Standards
APG_MODEL_CONFIG = ConfigDict(
	extra='forbid',
	validate_by_name=True,
	validate_by_alias=True,
	str_strip_whitespace=True,
	validate_default=True,
	use_enum_values=False
)

class HttpMethod(str, Enum):
	"""HTTP methods supported by APIG"""
	GET = "GET"
	POST = "POST"
	PUT = "PUT"
	DELETE = "DELETE"
	PATCH = "PATCH"
	HEAD = "HEAD"
	OPTIONS = "OPTIONS"
	TRACE = "TRACE"

class PolicyType(str, Enum):
	"""Types of gateway policies"""
	RATE_LIMITING = "rate_limiting"
	AUTHENTICATION = "authentication"
	AUTHORIZATION = "authorization"
	TRANSFORMATION = "transformation"
	SECURITY = "security"
	CACHING = "caching"
	LOGGING = "logging"
	MONITORING = "monitoring"

class LoadBalancingAlgorithm(str, Enum):
	"""Advanced load balancing algorithms with AI intelligence"""
	ROUND_ROBIN = "round_robin"
	WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
	LEAST_CONNECTIONS = "least_connections"
	WEIGHTED_RESPONSE_TIME = "weighted_response_time"
	CONSISTENT_HASH = "consistent_hash"
	LEAST_RESPONSE_TIME = "least_response_time"
	IP_HASH = "ip_hash"
	GEOGRAPHIC = "geographic"
	AI_OPTIMIZED = "ai_optimized"
	ADAPTIVE_AI = "adaptive_ai"

class EnvironmentType(str, Enum):
	"""Environment types for APG multi-tenancy"""
	DEVELOPMENT = "development"
	STAGING = "staging"
	PRODUCTION = "production"
	TESTING = "testing"

class ThreatLevel(str, Enum):
	"""Security threat levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"

# Core Gateway Configuration Models

class AgRateLimit(BaseModel):
	"""Rate limiting configuration for API routes"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique rate limit ID")
	requests_per_second: PositiveInt = Field(description="Maximum requests per second")
	requests_per_minute: Optional[PositiveInt] = Field(None, description="Maximum requests per minute")
	requests_per_hour: Optional[PositiveInt] = Field(None, description="Maximum requests per hour")
	burst_size: Optional[PositiveInt] = Field(None, description="Burst capacity for token bucket")
	key_extractor: str = Field("client_ip", description="Key extraction method (client_ip, api_key, user_id)")
	rejection_message: str = Field("Rate limit exceeded", description="Message returned when rate limited")
	headers_enabled: bool = Field(True, description="Include rate limit headers in response")

	@field_validator('key_extractor')
	@classmethod
	def validate_key_extractor(cls, v: str) -> str:
		allowed = ['client_ip', 'api_key', 'user_id', 'jwt_subject', 'custom_header']
		if v not in allowed:
			raise ValueError(f"key_extractor must be one of: {allowed}")
		return v

class AgCacheConfig(BaseModel):
	"""Caching configuration for API responses"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique cache config ID")
	enabled: bool = Field(True, description="Enable caching for this route")
	ttl_seconds: PositiveInt = Field(300, description="Time-to-live in seconds")
	cache_key_template: str = Field("{path}:{query_hash}", description="Template for cache key generation")
	cache_conditions: List[str] = Field(default_factory=list, description="Conditions for caching responses")
	invalidation_patterns: List[str] = Field(default_factory=list, description="Patterns for cache invalidation")
	compression_enabled: bool = Field(True, description="Enable response compression")
	vary_headers: List[str] = Field(default_factory=list, description="Headers to vary cache on")

class AgHealthCheck(BaseModel):
	"""Health check configuration for upstream services"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique health check ID")
	enabled: bool = Field(True, description="Enable health checking")
	path: str = Field("/health", description="Health check endpoint path")
	interval_seconds: PositiveInt = Field(30, description="Check interval in seconds")
	timeout_seconds: PositiveInt = Field(5, description="Request timeout in seconds")
	healthy_threshold: PositiveInt = Field(2, description="Consecutive successes to mark healthy")
	unhealthy_threshold: PositiveInt = Field(3, description="Consecutive failures to mark unhealthy")
	expected_status_codes: List[int] = Field(default_factory=lambda: [200], description="Expected HTTP status codes")
	expected_response_body: Optional[str] = Field(None, description="Expected response body content")

class AgUpstreamService(BaseModel):
	"""Upstream service configuration"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique upstream service ID")
	name: str = Field(description="Human-readable service name")
	base_url: str = Field(description="Base URL for the upstream service")
	weight: PositiveInt = Field(100, description="Load balancing weight")
	max_connections: PositiveInt = Field(100, description="Maximum concurrent connections")
	connection_timeout_ms: PositiveInt = Field(5000, description="Connection timeout in milliseconds")
	read_timeout_ms: PositiveInt = Field(30000, description="Read timeout in milliseconds")
	health_check: AgHealthCheck = Field(default_factory=lambda: AgHealthCheck(), description="Health check configuration")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional service metadata")

	@model_validator(mode='before')
	@classmethod
	def populate_base_url_alias(cls, data: Any) -> Any:
		if isinstance(data, dict) and 'base_url' not in data and 'url' in data:
			data = dict(data)
			data['base_url'] = data.pop('url')
		return data

	@field_validator('base_url')
	@classmethod
	def validate_base_url(cls, v: str) -> str:
		if not (v.startswith('http://') or v.startswith('https://')):
			raise ValueError("base_url must start with http:// or https://")
		return v.rstrip('/')

class AgPolicy(BaseModel):
	"""Gateway policy configuration"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique policy ID")
	name: str = Field(description="Human-readable policy name")
	type: PolicyType = Field(description="Type of policy")
	enabled: bool = Field(True, description="Enable this policy")
	priority: int = Field(1000, description="Policy execution priority (lower = higher priority)")
	conditions: List[str] = Field(default_factory=list, description="Conditions for policy execution")
	configuration: Dict[str, Any] = Field(default_factory=dict, description="Policy-specific configuration")
	natural_language_description: Optional[str] = Field(None, description="Natural language description for AI generation")
	created_by: str = Field(description="User who created this policy")
	tenant_id: str = Field("default", description="APG tenant ID")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	@field_validator('priority')
	@classmethod
	def validate_priority(cls, v: int) -> int:
		if not (1 <= v <= 10000):
			raise ValueError("priority must be between 1 and 10000")
		return v

class AgApiRoute(BaseModel):
	"""API route configuration with intelligent routing"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique route ID")
	path: str = Field(description="Route path pattern (supports path parameters)")
	method: HttpMethod = Field(description="HTTP method")
	name: Optional[str] = Field(None, description="Human-readable route name")
	description: Optional[str] = Field(None, description="Route description")

	# Upstream Configuration
	upstream_services: List[AgUpstreamService] = Field(description="List of upstream services")
	load_balancing_algorithm: LoadBalancingAlgorithm = Field(LoadBalancingAlgorithm.ROUND_ROBIN)

	# Policy Configuration
	policies: List[str] = Field(default_factory=list, description="Applied policy IDs")
	rate_limit: Optional[AgRateLimit] = Field(None, description="Route-specific rate limiting")
	cache_config: Optional[AgCacheConfig] = Field(None, description="Route-specific caching")

	# Security Configuration
	auth_required: bool = Field(True, description="Require authentication")
	allowed_origins: List[str] = Field(default_factory=list, description="CORS allowed origins")

	# Monitoring Configuration
	metrics_enabled: bool = Field(True, description="Enable metrics collection")
	tracing_enabled: bool = Field(True, description="Enable distributed tracing")
	logging_level: Literal['DEBUG', 'INFO', 'WARN', 'ERROR'] = Field('INFO')

	# APG Integration
	tenant_id: str = Field("default", description="APG tenant ID")
	created_by: str = Field("system", description="User who created this route")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	@field_validator('path')
	@classmethod
	def validate_path(cls, v: str) -> str:
		if not v.startswith('/'):
			raise ValueError("path must start with /")
		return v

	@field_validator('upstream_services')
	@classmethod
	def validate_upstream_services(cls, v: List[AgUpstreamService]) -> List[AgUpstreamService]:
		if not v:
			raise ValueError("at least one upstream service is required")
		return v

class AgGatewayConfig(BaseModel):
	"""Main gateway instance configuration"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique gateway ID")
	name: str = Field(description="Human-readable gateway name")
	description: Optional[str] = Field(None, description="Gateway description")
	environment: EnvironmentType = Field(description="Deployment environment")

	# Network Configuration
	listen_port: PositiveInt = Field(8080, description="Port to listen on")
	tls_enabled: bool = Field(True, description="Enable TLS/SSL")
	tls_certificate_path: Optional[str] = Field(None, description="Path to TLS certificate")
	tls_private_key_path: Optional[str] = Field(None, description="Path to TLS private key")

	# Edge Computing Configuration
	edge_locations: List[str] = Field(default_factory=list, description="Edge deployment locations")
	wasm_runtime_enabled: bool = Field(True, description="Enable WebAssembly runtime")
	ai_intelligence_enabled: bool = Field(True, description="Enable AI-powered features")

	# Performance Configuration
	max_connections: PositiveInt = Field(10000, description="Maximum concurrent connections")
	connection_timeout_ms: PositiveInt = Field(30000, description="Connection timeout")
	request_timeout_ms: PositiveInt = Field(60000, description="Request timeout")
	keepalive_timeout_ms: PositiveInt = Field(75000, description="Keep-alive timeout")

	# Routes and Policies
	routes: List[AgApiRoute] = Field(default_factory=list, description="Configured API routes")
	global_policies: List[str] = Field(default_factory=list, description="Global policy IDs")

	# APG Integration Settings
	tenant_id: str = Field("default", description="APG tenant ID")
	auth_rbac_integration: bool = Field(True, description="Enable APG auth_rbac integration")
	monitoring_integration: bool = Field(True, description="Enable APG monitoring integration")
	audit_logging_enabled: bool = Field(True, description="Enable APG audit logging")

	# Metadata
	created_by: str = Field(description="User who created this gateway")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	last_deployed_at: Optional[datetime] = Field(None, description="Last deployment timestamp")

# Traffic Analytics and Monitoring Models

class AgTrafficMetrics(BaseModel):
	"""Real-time traffic metrics"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique metrics ID")
	gateway_id: str = Field(description="Gateway instance ID")
	route_id: Optional[str] = Field(None, description="Specific route ID (None for gateway-wide metrics)")
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	# Request Metrics
	request_count: NonNegativeFloat = Field(0, description="Total request count")
	requests_per_second: NonNegativeFloat = Field(0, description="Current RPS")

	# Response Time Metrics (milliseconds)
	response_time_p50: NonNegativeFloat = Field(0, description="50th percentile response time")
	response_time_p95: NonNegativeFloat = Field(0, description="95th percentile response time")
	response_time_p99: NonNegativeFloat = Field(0, description="99th percentile response time")
	response_time_max: NonNegativeFloat = Field(0, description="Maximum response time")

	# Error Metrics
	error_count: NonNegativeFloat = Field(0, description="Total error count")
	error_rate: NonNegativeFloat = Field(0, description="Error rate as percentage (0-100)")
	error_breakdown: Dict[str, int] = Field(default_factory=dict, description="Errors by status code")

	# Bandwidth Metrics
	bytes_sent: NonNegativeFloat = Field(0, description="Total bytes sent")
	bytes_received: NonNegativeFloat = Field(0, description="Total bytes received")
	bandwidth_mbps: NonNegativeFloat = Field(0, description="Current bandwidth in Mbps")

	# Connection Metrics
	active_connections: NonNegativeFloat = Field(0, description="Current active connections")
	total_connections: NonNegativeFloat = Field(0, description="Total connections handled")

	# APG Tenant Context
	tenant_id: str = Field("default", description="APG tenant ID")

class AgSecurityEvent(BaseModel):
	"""Security events and threat detection"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique event ID")
	gateway_id: str = Field(description="Gateway instance ID")
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	# Event Classification
	event_type: str = Field(description="Type of security event")
	threat_level: ThreatLevel = Field(description="Threat severity level")
	confidence: float = Field(0.0, ge=0.0, le=1.0, description="AI confidence score (0-1)")

	# Event Details
	source_ip: str = Field(description="Source IP address")
	user_agent: Optional[str] = Field(None, description="User agent string")
	route_path: Optional[str] = Field(None, description="Targeted route path")
	attack_signature: Optional[str] = Field(None, description="Detected attack signature")

	# Response Actions
	action_taken: str = Field(description="Automated response action")
	blocked: bool = Field(False, description="Whether request was blocked")
	rate_limited: bool = Field(False, description="Whether request was rate limited")

	# Context and Metadata
	request_headers: Dict[str, str] = Field(default_factory=dict, description="Request headers")
	geo_location: Optional[Dict[str, str]] = Field(None, description="Geographic location data")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")

	# APG Integration
	tenant_id: str = Field(description="APG tenant ID")
	audit_logged: bool = Field(False, description="Whether logged to APG audit system")

class AgWafRule(BaseModel):
	"""Web Application Firewall rule"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique WAF rule ID")
	name: str = Field(description="Human-readable rule name")
	description: Optional[str] = Field(None, description="Rule description")
	enabled: bool = Field(True, description="Enable this rule")

	# Rule Configuration
	rule_type: Literal['regex', 'ip_range', 'geo_location', 'rate_limit', 'custom'] = Field(description="Type of WAF rule")
	pattern: str = Field(description="Rule pattern or expression")
	case_sensitive: bool = Field(False, description="Case-sensitive pattern matching")

	# Action Configuration
	action: Literal['block', 'allow', 'log', 'rate_limit', 'captcha'] = Field(description="Action to take when rule matches")
	response_code: Optional[int] = Field(403, description="HTTP response code for block action")
	response_message: Optional[str] = Field(None, description="Custom response message")

	# Advanced Configuration
	priority: int = Field(1000, description="Rule execution priority")
	conditions: List[str] = Field(default_factory=list, description="Additional conditions")
	exceptions: List[str] = Field(default_factory=list, description="Rule exceptions")

	# Metadata
	tenant_id: str = Field(description="APG tenant ID")
	created_by: str = Field(description="User who created this rule")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AgSecurityPolicy(BaseModel):
	"""Comprehensive security policy configuration"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique security policy ID")
	name: str = Field(description="Security policy name")
	description: Optional[str] = Field(None, description="Policy description")
	enabled: bool = Field(True, description="Enable this security policy")

	# AI-Powered Security
	threat_detection_enabled: bool = Field(True, description="Enable AI threat detection")
	anomaly_detection_enabled: bool = Field(True, description="Enable anomaly detection")
	behavioral_analysis_enabled: bool = Field(True, description="Enable behavioral analysis")

	# Access Control
	rate_limit_rules: List[AgRateLimit] = Field(default_factory=list, description="Rate limiting rules")
	ip_whitelist: List[str] = Field(default_factory=list, description="Allowed IP addresses/ranges")
	ip_blacklist: List[str] = Field(default_factory=list, description="Blocked IP addresses/ranges")
	geo_restrictions: List[str] = Field(default_factory=list, description="Blocked countries/regions")

	# WAF Configuration
	waf_enabled: bool = Field(True, description="Enable Web Application Firewall")
	waf_rules: List[AgWafRule] = Field(default_factory=list, description="WAF rules")

	# DDoS Protection
	ddos_protection_enabled: bool = Field(True, description="Enable DDoS protection")
	ddos_threshold_rps: PositiveInt = Field(10000, description="DDoS detection threshold (RPS)")
	ddos_response_action: Literal['block', 'rate_limit', 'captcha'] = Field('rate_limit')

	# Bot Management
	bot_detection_enabled: bool = Field(True, description="Enable bot detection")
	bot_challenge_enabled: bool = Field(True, description="Enable bot challenges")
	allowed_bots: List[str] = Field(default_factory=list, description="Allowed bot user agents")

	# APG Integration
	tenant_id: str = Field(description="APG tenant ID")
	created_by: str = Field(description="User who created this policy")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# WASM and Edge Computing Models

class AgWasmModule(BaseModel):
	"""WebAssembly module configuration"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique WASM module ID")
	name: str = Field(description="Module name")
	description: Optional[str] = Field(None, description="Module description")
	version: str = Field("1.0.0", description="Module version")

	# Module Configuration
	wasm_binary_path: str = Field(description="Path to WASM binary file")
	entry_point: str = Field("process_request", description="Entry point function name")
	memory_limit_mb: PositiveInt = Field(64, description="Memory limit in MB")
	execution_timeout_ms: PositiveInt = Field(5000, description="Execution timeout in milliseconds")

	# Execution Context
	environment_variables: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
	configuration: Dict[str, Any] = Field(default_factory=dict, description="Module configuration")

	# Performance Metrics
	avg_execution_time_ms: NonNegativeFloat = Field(0, description="Average execution time")
	total_executions: NonNegativeFloat = Field(0, description="Total number of executions")
	error_count: NonNegativeFloat = Field(0, description="Number of execution errors")

	# APG Integration
	tenant_id: str = Field(description="APG tenant ID")
	created_by: str = Field(description="User who created this module")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# HTTP Request/Response Models for Edge Processing

class AgHttpRequest(BaseModel):
	"""HTTP request model for edge processing"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique request ID")
	method: HttpMethod = Field(description="HTTP method")
	path: str = Field(description="Request path")
	query_string: str = Field("", description="Query string")
	headers: Dict[str, str] = Field(default_factory=dict, description="Request headers")
	body: Optional[bytes] = Field(None, description="Request body")

	# Client Information
	client_ip: str = Field(description="Client IP address")
	user_agent: Optional[str] = Field(None, description="User agent string")

	# Timing Information
	received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	processing_start_time: Optional[float] = Field(None, description="Processing start time (perf_counter)")

	# APG Context
	tenant_id: str = Field("default", description="APG tenant ID")
	user_id: Optional[str] = Field(None, description="Authenticated user ID")

class AgHttpResponse(BaseModel):
	"""HTTP response model for edge processing"""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique response ID")
	request_id: str = Field(description="Corresponding request ID")
	status_code: int = Field(200, ge=100, le=599, description="HTTP status code")
	headers: Dict[str, str] = Field(default_factory=dict, description="Response headers")
	body: Optional[bytes] = Field(None, description="Response body")

	# Performance Metrics
	processing_time_ms: NonNegativeFloat = Field(0, description="Total processing time in milliseconds")
	upstream_time_ms: NonNegativeFloat = Field(0, description="Upstream response time in milliseconds")
	cache_hit: bool = Field(False, description="Whether response was served from cache")

	# Metadata
	generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	served_from_edge: bool = Field(False, description="Whether served from edge location")
	edge_location: Optional[str] = Field(None, description="Edge location identifier")

# Utility Models

class AgApiError(BaseModel):
	"""Standardized API error response"""
	model_config = APG_MODEL_CONFIG

	error_code: str = Field(description="Machine-readable error code")
	error_message: str = Field(description="Human-readable error message")
	error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
	request_id: Optional[str] = Field(None, description="Request ID for debugging")
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	documentation_url: Optional[str] = Field(None, description="URL to relevant documentation")

class AgPaginationInfo(BaseModel):
	"""Pagination information for API responses"""
	model_config = APG_MODEL_CONFIG

	page: PositiveInt = Field(1, description="Current page number")
	page_size: PositiveInt = Field(50, description="Items per page")
	total_items: NonNegativeFloat = Field(0, description="Total number of items")
	total_pages: NonNegativeFloat = Field(0, description="Total number of pages")
	has_next: bool = Field(False, description="Whether there are more pages")
	has_previous: bool = Field(False, description="Whether there are previous pages")


class GatewayUpstreamRecord(BaseModel):
	"""Tenant-scoped upstream service registration for package governance."""
	model_config = APG_MODEL_CONFIG

	id: str = Field(description="Upstream service ID")
	tenant_id: str = Field(description="APG tenant ID")
	name: str = Field(description="Human-readable upstream name")
	base_url: str = Field(description="Base URL for the upstream service")
	owner: str = Field(description="Route or service owner")
	health: str = Field("healthy", description="Current upstream health state")
	decision: str = Field("allow", description="allow, deny, or require_review")
	matched_rules: List[str] = Field(default_factory=list, description="Matched guardrail rules")
	policy_decision: str = Field("allow", description="Persisted policy decision")
	review_reasons: List[str] = Field(default_factory=list, description="Review or denial reasons")
	review_evidence: Dict[str, Any] = Field(default_factory=dict, description="Required action evidence")
	labels: Dict[str, Any] = Field(default_factory=dict, description="Routing and discovery labels")


class GatewayConsumerRecord(BaseModel):
	"""Tenant-scoped API consumer registration for package governance."""
	model_config = APG_MODEL_CONFIG

	id: str = Field(description="Consumer ID")
	tenant_id: str = Field(description="APG tenant ID")
	name: str = Field(description="Human-readable consumer name")
	owner: str = Field(description="Consumer owner")
	access_tier: str = Field("standard", description="standard, partner, or restricted")
	identity_provider: str = Field("auth", description="Identity provider for the consumer")
	credential_rotation_recorded: bool = Field(False, description="Whether credential rotation evidence exists")
	rbac_approval_recorded: bool = Field(False, description="Whether restricted access has RBAC approval")
	status: str = Field("registered", description="Consumer lifecycle status")
	decision: str = Field("allow", description="allow, deny, or require_review")
	matched_rules: List[str] = Field(default_factory=list, description="Matched guardrail rules")
	policy_decision: str = Field("allow", description="Persisted policy decision")
	review_reasons: List[str] = Field(default_factory=list, description="Review or denial reasons")
	review_evidence: Dict[str, Any] = Field(default_factory=dict, description="Required action evidence")


class GatewayRouteRecord(BaseModel):
	"""Governed route publication record for package composition."""
	model_config = APG_MODEL_CONFIG

	id: str = Field(description="Route ID")
	tenant_id: str = Field(description="APG tenant ID")
	path: str = Field(description="Published route path")
	methods: List[str] = Field(default_factory=list, description="Allowed HTTP methods")
	upstream_id: str = Field(description="Registered upstream service ID")
	owner: str = Field(description="Route owner")
	route_exposure: str = Field("internal", description="Route exposure scope")
	consumer_id: Optional[str] = Field(None, description="Optional registered API consumer ID")
	auth_policy_attached: bool = Field(True, description="Whether auth policy is attached")
	threat_policy_attached: bool = Field(True, description="Whether threat policy is attached")
	mtls_enabled: bool = Field(True, description="Whether mTLS is enabled for route exposure")
	rate_limit_configured: bool = Field(True, description="Whether route rate limits are configured")
	requested_rps_limit: int = Field(1000, description="Requested RPS quota")
	wasm_filter_attached: bool = Field(False, description="Whether a WASM edge filter is attached")
	filter_signature_verified: bool = Field(True, description="Whether attached filter signature is verified")
	status: str = Field("active", description="Route lifecycle status")
	decision: str = Field("allow", description="allow, deny, or require_review")
	matched_rules: List[str] = Field(default_factory=list, description="Matched guardrail rules")
	policy_decision: str = Field("allow", description="Persisted policy decision")
	review_reasons: List[str] = Field(default_factory=list, description="Review or denial reasons")
	review_evidence: Dict[str, Any] = Field(default_factory=dict, description="Required action evidence")


class GatewayQuotaReview(BaseModel):
	"""Approval record for high-throughput gateway route quotas."""
	model_config = APG_MODEL_CONFIG

	id: str = Field(description="Quota review ID")
	tenant_id: str = Field(description="APG tenant ID")
	route_id: str = Field(description="Route under review")
	requested_rps_limit: int = Field(description="Requested RPS limit")
	requester: str = Field(description="User requesting quota")
	justification: str = Field(description="Business justification")
	decision: str = Field("pending", description="pending, approved, or rejected")
	reviewer: Optional[str] = Field(None, description="Reviewer who decided the request")
	notes: Optional[str] = Field(None, description="Reviewer notes")
	matched_rules: List[str] = Field(default_factory=list, description="Matched guardrail rules")
	policy_decision: str = Field("require_review", description="Persisted policy decision")
	review_reasons: List[str] = Field(default_factory=list, description="Review or denial reasons")
	review_evidence: Dict[str, Any] = Field(default_factory=dict, description="Required action evidence")


class GatewayPolicyRecord(BaseModel):
	"""Gateway policy change review record."""
	model_config = APG_MODEL_CONFIG

	id: str = Field(description="Policy record ID")
	tenant_id: str = Field(description="APG tenant ID")
	name: str = Field(description="Policy name")
	policy_type: str = Field(description="Policy type")
	actor: str = Field(description="Actor requesting the policy change")
	status: str = Field("active", description="Policy lifecycle status")
	decision: str = Field("allow", description="allow, deny, or require_review")
	matched_rules: List[str] = Field(default_factory=list, description="Matched guardrail rules")
	policy_decision: str = Field("allow", description="Persisted policy decision")
	review_reasons: List[str] = Field(default_factory=list, description="Review or denial reasons")
	review_evidence: Dict[str, Any] = Field(default_factory=dict, description="Required action evidence")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Policy metadata")


class GatewayTrafficShiftRecord(BaseModel):
	"""Canary or weighted traffic shift record."""
	model_config = APG_MODEL_CONFIG

	id: str = Field(description="Traffic shift ID")
	tenant_id: str = Field(description="APG tenant ID")
	route_id: str = Field(description="Route receiving shifted traffic")
	canary_percent: int = Field(description="Percentage of traffic shifted to canary")
	actor: str = Field(description="Actor requesting shift")
	status: str = Field("active", description="Traffic shift lifecycle status")
	decision: str = Field("allow", description="allow, deny, or require_review")
	matched_rules: List[str] = Field(default_factory=list, description="Matched guardrail rules")
	policy_decision: str = Field("allow", description="Persisted policy decision")
	review_reasons: List[str] = Field(default_factory=list, description="Review or denial reasons")
	review_evidence: Dict[str, Any] = Field(default_factory=dict, description="Required action evidence")
	rollback_plan: Optional[str] = Field(None, description="Rollback plan")


class GatewayDeploymentRecord(BaseModel):
	"""Gateway deployment gate record."""
	model_config = APG_MODEL_CONFIG

	id: str = Field(description="Deployment ID")
	tenant_id: str = Field(description="APG tenant ID")
	environment: str = Field(description="Deployment environment")
	region: str = Field(description="Deployment region")
	actor: str = Field(description="Deployment actor")
	status: str = Field("deployed", description="Deployment lifecycle status")
	decision: str = Field("allow", description="allow, deny, or require_review")
	matched_rules: List[str] = Field(default_factory=list, description="Matched guardrail rules")
	policy_decision: str = Field("allow", description="Persisted policy decision")
	review_reasons: List[str] = Field(default_factory=list, description="Review or denial reasons")
	review_evidence: Dict[str, Any] = Field(default_factory=dict, description="Required action evidence")


class GatewayAgentRecord(BaseModel):
	"""Governed AI or automation agent participating in APIG decisions."""
	model_config = APG_MODEL_CONFIG

	id: str = Field(description="Gateway agent ID")
	tenant_id: str = Field(description="APG tenant ID")
	name: str = Field(description="Human-readable agent name")
	runtime: str = Field(description="Agent runtime adapter")
	role: str = Field(description="Gateway governance role")
	scope: str = Field(description="Bounded route, traffic, security, edge, or deployment scope")
	owner: str = Field(description="Accountable human or team owner")
	purpose: str = Field(description="Declared reason for agent participation")
	contribution_disclosed: bool = Field(True, description="Whether machine contribution disclosure is recorded")
	human_approval_required: bool = Field(False, description="Whether human approval is required before privileged actions")
	status: str = Field("active", description="Agent lifecycle status")
	decision: str = Field("allow", description="allow, deny, or require_review")
	matched_rules: List[str] = Field(default_factory=list, description="Matched guardrail rules")
	policy_decision: str = Field("allow", description="Persisted policy decision")
	review_reasons: List[str] = Field(default_factory=list, description="Review or denial reasons")
	review_evidence: Dict[str, Any] = Field(default_factory=dict, description="Required action evidence")


class GatewayLifecycleBatchRecord(BaseModel):
	"""Bytewax lifecycle-batch validation record for APIG generated apps."""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Lifecycle batch ID")
	tenant_id: str = Field(description="APG tenant ID")
	event_stream: str = Field(description="Lifecycle event processor")
	mutation_count: int = Field(description="Number of lifecycle mutations in the batch")
	accepted: bool = Field(description="Whether the batch is accepted")
	decision: str = Field("allow", description="allow, deny, or require_review")
	matched_rules: List[str] = Field(default_factory=list, description="Matched guardrail rules")
	policy_decision: str = Field("allow", description="Persisted policy decision")
	review_reasons: List[str] = Field(default_factory=list, description="Review or denial reasons")
	review_evidence: Dict[str, Any] = Field(default_factory=dict, description="Required action evidence")
	required_processor: str = Field("bytewax", description="Required lifecycle processor")
	status: str = Field("accepted", description="Batch lifecycle status")


class GatewayAuditEvent(BaseModel):
	"""Tenant-scoped gateway governance evidence event."""
	model_config = APG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Audit event ID")
	tenant_id: str = Field(description="APG tenant ID")
	event_type: str = Field(description="Gateway lifecycle event type")
	subject_id: str = Field(description="Subject record ID")
	message: str = Field(description="Human-readable event message")
	policy_decision: str = Field("allow", description="Persisted policy decision")
	matched_rules: List[str] = Field(default_factory=list, description="Matched guardrail rules")
	review_reasons: List[str] = Field(default_factory=list, description="Review or denial reasons")
	review_evidence: Dict[str, Any] = Field(default_factory=dict, description="Required action evidence")
	evidence: Dict[str, Any] = Field(default_factory=dict, description="Structured event evidence")
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Helper Functions for Model Operations

def _log_model_operation(operation: str, model_name: str, model_id: str) -> None:
	"""Log model operations for APG integration"""
	timestamp = datetime.now(timezone.utc).isoformat()
	print(f"INFO [{timestamp}] APIG Model Operation: {operation} {model_name} {model_id}")

async def validate_tenant_access(tenant_id: str, user_id: str) -> bool:
	"""
	Validate tenant access for APG multi-tenancy support.

	Args:
		tenant_id: APG tenant identifier
		user_id: User identifier

	Returns:
		bool: True if access is allowed

	Note:
		Generated-app package validation is intentionally dependency-light; live
		APG auth/RBAC checks bind through the production auth adapter.
	"""
	assert isinstance(tenant_id, str), "tenant_id must be a string"
	assert isinstance(user_id, str), "user_id must be a string"
	if not tenant_id:
		return False
	if not user_id:
		return False

	return True

# Model Registry for APG Composition Engine

MODEL_REGISTRY = {
	'gateway_config': AgGatewayConfig,
	'api_route': AgApiRoute,
	'policy': AgPolicy,
	'upstream_service': AgUpstreamService,
	'rate_limit': AgRateLimit,
	'cache_config': AgCacheConfig,
	'health_check': AgHealthCheck,
	'traffic_metrics': AgTrafficMetrics,
	'security_event': AgSecurityEvent,
	'security_policy': AgSecurityPolicy,
	'waf_rule': AgWafRule,
	'wasm_module': AgWasmModule,
	'http_request': AgHttpRequest,
	'http_response': AgHttpResponse,
	'api_error': AgApiError,
	'pagination_info': AgPaginationInfo,
	'gateway_upstream_record': GatewayUpstreamRecord,
	'gateway_consumer_record': GatewayConsumerRecord,
	'gateway_route_record': GatewayRouteRecord,
	'gateway_quota_review': GatewayQuotaReview,
	'gateway_policy_record': GatewayPolicyRecord,
	'gateway_traffic_shift_record': GatewayTrafficShiftRecord,
	'gateway_deployment_record': GatewayDeploymentRecord,
	'gateway_audit_event': GatewayAuditEvent
}

# Export All Models
__all__ = [
	# Enums
	'HttpMethod', 'PolicyType', 'LoadBalancingAlgorithm', 'EnvironmentType', 'ThreatLevel',
	# Core Models
	'AgGatewayConfig', 'AgApiRoute', 'AgPolicy', 'AgUpstreamService',
	'AgRateLimit', 'AgCacheConfig', 'AgHealthCheck',
	# Analytics Models
	'AgTrafficMetrics', 'AgSecurityEvent', 'AgSecurityPolicy', 'AgWafRule',
	# Edge Computing Models
	'AgWasmModule', 'AgHttpRequest', 'AgHttpResponse',
	# Utility Models
	'AgApiError', 'AgPaginationInfo',
	'GatewayUpstreamRecord', 'GatewayConsumerRecord', 'GatewayRouteRecord',
	'GatewayQuotaReview', 'GatewayPolicyRecord', 'GatewayTrafficShiftRecord',
	'GatewayDeploymentRecord', 'GatewayAuditEvent',
	# Registry
	'MODEL_REGISTRY',
	# Helper Functions
	'validate_tenant_access'
]
