"""
APG API Service Mesh - Data Models

Comprehensive SQLAlchemy models for service mesh components including service discovery,
traffic management, load balancing, monitoring, and security policies.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4

from sqlalchemy import (
	Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
	ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET

from pydantic import BaseModel, Field, ConfigDict, validator
from pydantic.types import Json
from uuid_extensions import uuid7str

# SQLAlchemy Base
Base = declarative_base()

# =============================================================================
# Enums
# =============================================================================

class ServiceStatus(str, Enum):
	"""Service status enumeration."""
	REGISTERING = "registering"
	HEALTHY = "healthy"
	UNHEALTHY = "unhealthy"
	DEGRADED = "degraded"
	MAINTENANCE = "maintenance"
	DEREGISTERING = "deregistering"
	FAILED = "failed"

class EndpointProtocol(str, Enum):
	"""Endpoint protocol enumeration."""
	HTTP = "http"
	HTTPS = "https"
	HTTP2 = "http2"
	GRPC = "grpc"
	TCP = "tcp"
	UDP = "udp"
	WEBSOCKET = "websocket"

class LoadBalancerAlgorithm(str, Enum):
	"""Load balancer algorithm enumeration."""
	ROUND_ROBIN = "round_robin"
	WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
	LEAST_CONNECTIONS = "least_connections"
	WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
	IP_HASH = "ip_hash"
	GEOGRAPHIC = "geographic"
	RANDOM = "random"
	CONSISTENT_HASH = "consistent_hash"

class HealthStatus(str, Enum):
	"""Health check status enumeration."""
	HEALTHY = "healthy"
	UNHEALTHY = "unhealthy"
	TIMEOUT = "timeout"
	CONNECTION_FAILED = "connection_failed"
	UNKNOWN = "unknown"

class PolicyType(str, Enum):
	"""Policy type enumeration."""
	TRAFFIC = "traffic"
	SECURITY = "security"
	RATE_LIMIT = "rate_limit"
	RETRY = "retry"
	TIMEOUT = "timeout"
	CIRCUIT_BREAKER = "circuit_breaker"
	AUTHENTICATION = "authentication"
	AUTHORIZATION = "authorization"

class RouteMatchType(str, Enum):
	"""Route matching type enumeration."""
	PREFIX = "prefix"
	EXACT = "exact"
	REGEX = "regex"
	HEADER = "header"
	QUERY = "query"

class AlertSeverity(str, Enum):
	"""Alert severity enumeration."""
	INFO = "info"
	WARNING = "warning"
	ERROR = "error"
	CRITICAL = "critical"

class TraceStatus(str, Enum):
	"""Trace status enumeration."""
	SUCCESS = "success"
	ERROR = "error"
	TIMEOUT = "timeout"
	CANCELLED = "cancelled"

# =============================================================================
# Core Service Models
# =============================================================================

class SMService(Base):
	"""Service registration and metadata model."""
	__tablename__ = "sm_services"
	
	# Primary fields
	service_id = Column(String(50), primary_key=True, default=lambda: f"svc_{uuid7str()}")
	service_name = Column(String(255), nullable=False)
	service_version = Column(String(50), nullable=False)
	namespace = Column(String(255), default="default")
	
	# Service metadata
	description = Column(Text)
	tags = Column(JSONB, default=list)
	metadata = Column(JSONB, default=dict)
	
	# Status and health
	status = Column(String(50), nullable=False, default=ServiceStatus.REGISTERING.value)
	health_status = Column(String(50), default=HealthStatus.UNKNOWN.value)
	last_health_check = Column(DateTime(timezone=True))
	
	# Configuration
	configuration = Column(JSONB, default=dict)
	environment = Column(String(100), default="production")
	
	# Relationships
	endpoints = relationship("SMEndpoint", back_populates="service", cascade="all, delete-orphan")
	routes = relationship("SMRoute", back_populates="service")
	health_checks = relationship("SMHealthCheck", back_populates="service")
	metrics = relationship("SMMetrics", back_populates="service")
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_by = Column(String(255), nullable=False)
	updated_by = Column(String(255))
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Constraints and indexes
	__table_args__ = (
		UniqueConstraint('service_name', 'service_version', 'namespace', 'tenant_id', 
						name='uq_service_name_version_namespace_tenant'),
		Index('ix_sm_services_status', 'status'),
		Index('ix_sm_services_health', 'health_status'),
		Index('ix_sm_services_tenant', 'tenant_id'),
		Index('ix_sm_services_environment', 'environment'),
	)

class SMEndpoint(Base):
	"""Service endpoint definitions model."""
	__tablename__ = "sm_endpoints"
	
	# Primary fields
	endpoint_id = Column(String(50), primary_key=True, default=lambda: f"ep_{uuid7str()}")
	service_id = Column(String(50), ForeignKey("sm_services.service_id"), nullable=False)
	
	# Endpoint details
	host = Column(String(255), nullable=False)
	port = Column(Integer, nullable=False)
	protocol = Column(String(50), nullable=False, default=EndpointProtocol.HTTP.value)
	path = Column(String(500), default="/")
	
	# Configuration
	weight = Column(Integer, default=100)
	enabled = Column(Boolean, default=True)
	metadata = Column(JSONB, default=dict)
	
	# Health check configuration
	health_check_path = Column(String(500), default="/health")
	health_check_interval = Column(Integer, default=30)  # seconds
	health_check_timeout = Column(Integer, default=5)   # seconds
	healthy_threshold = Column(Integer, default=2)
	unhealthy_threshold = Column(Integer, default=3)
	
	# TLS configuration
	tls_enabled = Column(Boolean, default=False)
	tls_verify = Column(Boolean, default=True)
	certificate_id = Column(String(50), ForeignKey("sm_certificates.certificate_id"))
	
	# Relationships
	service = relationship("SMService", back_populates="endpoints")
	certificate = relationship("SMCertificate")
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_by = Column(String(255), nullable=False)
	updated_by = Column(String(255))
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Constraints and indexes
	__table_args__ = (
		UniqueConstraint('service_id', 'host', 'port', 'path', 
						name='uq_endpoint_service_host_port_path'),
		Index('ix_sm_endpoints_service', 'service_id'),
		Index('ix_sm_endpoints_enabled', 'enabled'),
		Index('ix_sm_endpoints_tenant', 'tenant_id'),
		CheckConstraint('port > 0 AND port <= 65535', name='ck_endpoint_valid_port'),
	)

class SMRoute(Base):
	"""Routing rule configurations model."""
	__tablename__ = "sm_routes"
	
	# Primary fields
	route_id = Column(String(50), primary_key=True, default=lambda: f"rt_{uuid7str()}")
	route_name = Column(String(255), nullable=False)
	service_id = Column(String(50), ForeignKey("sm_services.service_id"))
	
	# Route matching
	match_type = Column(String(50), nullable=False, default=RouteMatchType.PREFIX.value)
	match_value = Column(String(1000), nullable=False)
	match_headers = Column(JSONB, default=dict)
	match_query = Column(JSONB, default=dict)
	
	# Route destination
	destination_services = Column(JSONB, nullable=False)  # List of service configs with weights
	backup_services = Column(JSONB, default=list)
	
	# Route policies
	timeout_ms = Column(Integer, default=30000)
	retry_attempts = Column(Integer, default=3)
	retry_timeout_ms = Column(Integer, default=1000)
	
	# Traffic management
	priority = Column(Integer, default=1000)
	enabled = Column(Boolean, default=True)
	
	# Request/Response transformation
	request_headers_add = Column(JSONB, default=dict)
	request_headers_remove = Column(JSONB, default=list)
	response_headers_add = Column(JSONB, default=dict)
	response_headers_remove = Column(JSONB, default=list)
	
	# Relationships
	service = relationship("SMService", back_populates="routes")
	policies = relationship("SMPolicy", back_populates="route")
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_by = Column(String(255), nullable=False)
	updated_by = Column(String(255))
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Constraints and indexes
	__table_args__ = (
		UniqueConstraint('route_name', 'tenant_id', name='uq_route_name_tenant'),
		Index('ix_sm_routes_enabled', 'enabled'),
		Index('ix_sm_routes_priority', 'priority'),
		Index('ix_sm_routes_match_type', 'match_type'),
		Index('ix_sm_routes_tenant', 'tenant_id'),
		CheckConstraint('priority > 0', name='ck_route_positive_priority'),
	)

class SMLoadBalancer(Base):
	"""Load balancing configurations model."""
	__tablename__ = "sm_load_balancers"
	
	# Primary fields
	load_balancer_id = Column(String(50), primary_key=True, default=lambda: f"lb_{uuid7str()}")
	load_balancer_name = Column(String(255), nullable=False)
	service_id = Column(String(50), ForeignKey("sm_services.service_id"))
	
	# Load balancing configuration
	algorithm = Column(String(50), nullable=False, default=LoadBalancerAlgorithm.ROUND_ROBIN.value)
	session_affinity = Column(Boolean, default=False)
	session_affinity_cookie = Column(String(100))
	
	# Health check configuration
	health_check_enabled = Column(Boolean, default=True)
	health_check_interval = Column(Integer, default=30)
	health_check_timeout = Column(Integer, default=5)
	healthy_threshold = Column(Integer, default=2)
	unhealthy_threshold = Column(Integer, default=3)
	
	# Circuit breaker configuration
	circuit_breaker_enabled = Column(Boolean, default=True)
	failure_threshold = Column(Integer, default=5)
	recovery_timeout = Column(Integer, default=30)
	half_open_requests = Column(Integer, default=3)
	
	# Connection pooling
	max_connections = Column(Integer, default=100)
	max_pending_requests = Column(Integer, default=50)
	max_requests_per_connection = Column(Integer, default=1000)
	connection_timeout_ms = Column(Integer, default=5000)
	
	# Configuration and metadata
	configuration = Column(JSONB, default=dict)
	enabled = Column(Boolean, default=True)
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_by = Column(String(255), nullable=False)
	updated_by = Column(String(255))
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Constraints and indexes
	__table_args__ = (
		UniqueConstraint('load_balancer_name', 'tenant_id', name='uq_lb_name_tenant'),
		Index('ix_sm_load_balancers_algorithm', 'algorithm'),
		Index('ix_sm_load_balancers_enabled', 'enabled'),
		Index('ix_sm_load_balancers_tenant', 'tenant_id'),
		CheckConstraint('failure_threshold > 0', name='ck_lb_positive_failure_threshold'),
		CheckConstraint('max_connections > 0', name='ck_lb_positive_max_connections'),
	)

class SMPolicy(Base):
	"""Traffic and security policies model."""
	__tablename__ = "sm_policies"
	
	# Primary fields
	policy_id = Column(String(50), primary_key=True, default=lambda: f"pol_{uuid7str()}")
	policy_name = Column(String(255), nullable=False)
	policy_type = Column(String(50), nullable=False)
	route_id = Column(String(50), ForeignKey("sm_routes.route_id"))
	
	# Policy configuration
	configuration = Column(JSONB, nullable=False)
	enabled = Column(Boolean, default=True)
	priority = Column(Integer, default=1000)
	
	# Rate limiting specific fields
	rate_limit_requests = Column(Integer)
	rate_limit_window_seconds = Column(Integer)
	rate_limit_burst = Column(Integer)
	
	# Authentication/Authorization
	auth_required = Column(Boolean, default=False)
	auth_config = Column(JSONB, default=dict)
	
	# Description and metadata
	description = Column(Text)
	metadata = Column(JSONB, default=dict)
	
	# Relationships
	route = relationship("SMRoute", back_populates="policies")
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_by = Column(String(255), nullable=False)
	updated_by = Column(String(255))
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Constraints and indexes
	__table_args__ = (
		UniqueConstraint('policy_name', 'tenant_id', name='uq_policy_name_tenant'),
		Index('ix_sm_policies_type', 'policy_type'),
		Index('ix_sm_policies_enabled', 'enabled'),
		Index('ix_sm_policies_priority', 'priority'),
		Index('ix_sm_policies_tenant', 'tenant_id'),
		CheckConstraint('priority > 0', name='ck_policy_positive_priority'),
	)

# =============================================================================
# Monitoring and Analytics Models
# =============================================================================

class SMMetrics(Base):
	"""Performance metrics collection model."""
	__tablename__ = "sm_metrics"
	
	# Primary fields
	metric_id = Column(String(50), primary_key=True, default=lambda: f"met_{uuid7str()}")
	service_id = Column(String(50), ForeignKey("sm_services.service_id"))
	
	# Metric identification
	metric_name = Column(String(255), nullable=False)
	metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
	labels = Column(JSONB, default=dict)
	
	# Metric values
	value = Column(Float, nullable=False)
	timestamp = Column(DateTime(timezone=True), nullable=False)
	
	# Request metrics
	request_count = Column(Integer, default=0)
	error_count = Column(Integer, default=0)
	response_time_ms = Column(Float)
	status_code = Column(Integer)
	
	# Additional metadata
	metadata = Column(JSONB, default=dict)
	
	# Relationships
	service = relationship("SMService", back_populates="metrics")
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	
	# Constraints and indexes
	__table_args__ = (
		Index('ix_sm_metrics_service_timestamp', 'service_id', 'timestamp'),
		Index('ix_sm_metrics_name_timestamp', 'metric_name', 'timestamp'),
		Index('ix_sm_metrics_type', 'metric_type'),
		Index('ix_sm_metrics_tenant', 'tenant_id'),
		Index('ix_sm_metrics_timestamp', 'timestamp'),
	)

class SMTrace(Base):
	"""Distributed tracing data model."""
	__tablename__ = "sm_traces"
	
	# Primary fields
	trace_id = Column(String(100), primary_key=True)
	span_id = Column(String(50), primary_key=True)
	parent_span_id = Column(String(50))
	
	# Service information
	service_name = Column(String(255), nullable=False)
	operation_name = Column(String(255), nullable=False)
	
	# Timing information
	start_time = Column(DateTime(timezone=True), nullable=False)
	end_time = Column(DateTime(timezone=True))
	duration_ms = Column(Float)
	
	# Trace status
	status = Column(String(50), default=TraceStatus.SUCCESS.value)
	error_message = Column(Text)
	
	# Request/Response details
	http_method = Column(String(10))
	http_url = Column(String(1000))
	http_status_code = Column(Integer)
	user_agent = Column(String(500))
	
	# Tags and metadata
	tags = Column(JSONB, default=dict)
	logs = Column(JSONB, default=list)
	
	# Multi-tenancy
	tenant_id = Column(String(50), nullable=False)
	
	# Constraints and indexes
	__table_args__ = (
		Index('ix_sm_traces_service_start', 'service_name', 'start_time'),
		Index('ix_sm_traces_operation', 'operation_name'),
		Index('ix_sm_traces_status', 'status'),
		Index('ix_sm_traces_tenant', 'tenant_id'),
		Index('ix_sm_traces_start_time', 'start_time'),
	)

class SMHealthCheck(Base):
	"""Service health monitoring model."""
	__tablename__ = "sm_health_checks"
	
	# Primary fields
	health_check_id = Column(String(50), primary_key=True, default=lambda: f"hc_{uuid7str()}")
	service_id = Column(String(50), ForeignKey("sm_services.service_id"), nullable=False)
	endpoint_id = Column(String(50), ForeignKey("sm_endpoints.endpoint_id"))
	
	# Health check details
	check_type = Column(String(50), default="http")
	check_url = Column(String(1000))
	check_interval = Column(Integer, default=30)
	check_timeout = Column(Integer, default=5)
	
	# Status and results
	status = Column(String(50), nullable=False, default=HealthStatus.UNKNOWN.value)
	response_time_ms = Column(Float)
	status_code = Column(Integer)
	response_body = Column(Text)
	error_message = Column(Text)
	
	# Consecutive status counts
	consecutive_successes = Column(Integer, default=0)
	consecutive_failures = Column(Integer, default=0)
	
	# Timestamps
	last_check_at = Column(DateTime(timezone=True))
	last_success_at = Column(DateTime(timezone=True))
	last_failure_at = Column(DateTime(timezone=True))
	
	# Relationships
	service = relationship("SMService", back_populates="health_checks")
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Constraints and indexes
	__table_args__ = (
		Index('ix_sm_health_checks_service', 'service_id'),
		Index('ix_sm_health_checks_status', 'status'),
		Index('ix_sm_health_checks_last_check', 'last_check_at'),
		Index('ix_sm_health_checks_tenant', 'tenant_id'),
	)

class SMAlert(Base):
	"""Alert definitions and history model."""
	__tablename__ = "sm_alerts"
	
	# Primary fields
	alert_id = Column(String(50), primary_key=True, default=lambda: f"alt_{uuid7str()}")
	alert_name = Column(String(255), nullable=False)
	
	# Alert configuration
	condition = Column(Text, nullable=False)
	threshold = Column(Float)
	severity = Column(String(50), nullable=False)
	enabled = Column(Boolean, default=True)
	
	# Alert state
	is_active = Column(Boolean, default=False)
	last_triggered_at = Column(DateTime(timezone=True))
	last_resolved_at = Column(DateTime(timezone=True))
	trigger_count = Column(Integer, default=0)
	
	# Notification configuration
	notification_channels = Column(JSONB, default=list)
	notification_template = Column(Text)
	
	# Description and metadata
	description = Column(Text)
	metadata = Column(JSONB, default=dict)
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_by = Column(String(255), nullable=False)
	updated_by = Column(String(255))
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Constraints and indexes
	__table_args__ = (
		UniqueConstraint('alert_name', 'tenant_id', name='uq_alert_name_tenant'),
		Index('ix_sm_alerts_severity', 'severity'),
		Index('ix_sm_alerts_enabled', 'enabled'),
		Index('ix_sm_alerts_active', 'is_active'),
		Index('ix_sm_alerts_tenant', 'tenant_id'),
	)

class SMTopology(Base):
	"""Service dependency mapping model."""
	__tablename__ = "sm_topology"
	
	# Primary fields
	topology_id = Column(String(50), primary_key=True, default=lambda: f"topo_{uuid7str()}")
	source_service_id = Column(String(50), ForeignKey("sm_services.service_id"), nullable=False)
	target_service_id = Column(String(50), ForeignKey("sm_services.service_id"), nullable=False)
	
	# Relationship details
	relationship_type = Column(String(50), default="dependency")
	weight = Column(Float, default=1.0)
	
	# Communication details
	protocol = Column(String(50))
	port = Column(Integer)
	endpoint_path = Column(String(500))
	
	# Performance metrics
	avg_response_time_ms = Column(Float)
	request_count = Column(Integer, default=0)
	error_count = Column(Integer, default=0)
	
	# Status and health
	status = Column(String(50), default="active")
	last_communication_at = Column(DateTime(timezone=True))
	
	# Metadata
	metadata = Column(JSONB, default=dict)
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Constraints and indexes
	__table_args__ = (
		UniqueConstraint('source_service_id', 'target_service_id', 'tenant_id', 
						name='uq_topology_source_target_tenant'),
		Index('ix_sm_topology_source', 'source_service_id'),
		Index('ix_sm_topology_target', 'target_service_id'),
		Index('ix_sm_topology_type', 'relationship_type'),
		Index('ix_sm_topology_tenant', 'tenant_id'),
	)

# =============================================================================
# Configuration Models
# =============================================================================

class SMConfiguration(Base):
	"""Service mesh configuration settings model."""
	__tablename__ = "sm_configurations"
	
	# Primary fields
	config_id = Column(String(50), primary_key=True, default=lambda: f"cfg_{uuid7str()}")
	config_name = Column(String(255), nullable=False)
	config_type = Column(String(50), nullable=False)
	
	# Configuration data
	configuration = Column(JSONB, nullable=False)
	schema_version = Column(String(20), default="1.0")
	
	# Status and validation
	enabled = Column(Boolean, default=True)
	validated = Column(Boolean, default=False)
	validation_errors = Column(JSONB, default=list)
	
	# Description and metadata
	description = Column(Text)
	metadata = Column(JSONB, default=dict)
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_by = Column(String(255), nullable=False)
	updated_by = Column(String(255))
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Constraints and indexes
	__table_args__ = (
		UniqueConstraint('config_name', 'tenant_id', name='uq_config_name_tenant'),
		Index('ix_sm_configurations_type', 'config_type'),
		Index('ix_sm_configurations_enabled', 'enabled'),
		Index('ix_sm_configurations_tenant', 'tenant_id'),
	)

class SMCertificate(Base):
	"""TLS certificate management model."""
	__tablename__ = "sm_certificates"
	
	# Primary fields
	certificate_id = Column(String(50), primary_key=True, default=lambda: f"cert_{uuid7str()}")
	certificate_name = Column(String(255), nullable=False)
	
	# Certificate details
	common_name = Column(String(255), nullable=False)
	subject_alt_names = Column(JSONB, default=list)
	issuer = Column(String(500))
	serial_number = Column(String(100))
	
	# Certificate content
	certificate_pem = Column(Text, nullable=False)
	private_key_pem = Column(Text)  # Encrypted or reference to secret store
	ca_certificate_pem = Column(Text)
	
	# Validity period
	not_before = Column(DateTime(timezone=True), nullable=False)
	not_after = Column(DateTime(timezone=True), nullable=False)
	
	# Status and management
	status = Column(String(50), default="active")
	auto_renew = Column(Boolean, default=True)
	renewal_days_before = Column(Integer, default=30)
	
	# Metadata
	metadata = Column(JSONB, default=dict)
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_by = Column(String(255), nullable=False)
	updated_by = Column(String(255))
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Constraints and indexes
	__table_args__ = (
		UniqueConstraint('certificate_name', 'tenant_id', name='uq_cert_name_tenant'),
		Index('ix_sm_certificates_common_name', 'common_name'),
		Index('ix_sm_certificates_not_after', 'not_after'),
		Index('ix_sm_certificates_status', 'status'),
		Index('ix_sm_certificates_tenant', 'tenant_id'),
	)

class SMSecurityPolicy(Base):
	"""Security rules and policies model."""
	__tablename__ = "sm_security_policies"
	
	# Primary fields
	security_policy_id = Column(String(50), primary_key=True, default=lambda: f"sp_{uuid7str()}")
	policy_name = Column(String(255), nullable=False)
	policy_type = Column(String(50), nullable=False)
	
	# Policy rules
	rules = Column(JSONB, nullable=False)
	enforcement_mode = Column(String(50), default="enforce")  # enforce, monitor, disabled
	
	# Access control
	allowed_sources = Column(JSONB, default=list)
	denied_sources = Column(JSONB, default=list)
	allowed_methods = Column(JSONB, default=list)
	allowed_paths = Column(JSONB, default=list)
	
	# Authentication requirements
	require_authentication = Column(Boolean, default=False)
	authentication_methods = Column(JSONB, default=list)
	
	# Status and priority
	enabled = Column(Boolean, default=True)
	priority = Column(Integer, default=1000)
	
	# Description and metadata
	description = Column(Text)
	metadata = Column(JSONB, default=dict)
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_by = Column(String(255), nullable=False)
	updated_by = Column(String(255))
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Constraints and indexes
	__table_args__ = (
		UniqueConstraint('policy_name', 'tenant_id', name='uq_security_policy_name_tenant'),
		Index('ix_sm_security_policies_type', 'policy_type'),
		Index('ix_sm_security_policies_enabled', 'enabled'),
		Index('ix_sm_security_policies_priority', 'priority'),
		Index('ix_sm_security_policies_tenant', 'tenant_id'),
		CheckConstraint('priority > 0', name='ck_security_policy_positive_priority'),
	)

class SMRateLimiter(Base):
	"""Rate limiting configurations model."""
	__tablename__ = "sm_rate_limiters"
	
	# Primary fields
	rate_limiter_id = Column(String(50), primary_key=True, default=lambda: f"rl_{uuid7str()}")
	limiter_name = Column(String(255), nullable=False)
	
	# Rate limiting configuration
	requests_per_second = Column(Integer, nullable=False)
	burst_size = Column(Integer, nullable=False)
	window_size_seconds = Column(Integer, default=60)
	
	# Limiting scope
	scope = Column(String(50), default="global")  # global, per_ip, per_user, per_service
	key_expression = Column(String(1000))  # Expression to generate limiting key
	
	# Response configuration
	rate_limit_response_code = Column(Integer, default=429)
	rate_limit_response_body = Column(Text)
	rate_limit_headers = Column(JSONB, default=dict)
	
	# Enforcement
	enabled = Column(Boolean, default=True)
	enforcement_mode = Column(String(50), default="enforce")  # enforce, monitor
	
	# Metadata and configuration
	metadata = Column(JSONB, default=dict)
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_by = Column(String(255), nullable=False)
	updated_by = Column(String(255))
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	# Constraints and indexes
	__table_args__ = (
		UniqueConstraint('limiter_name', 'tenant_id', name='uq_rate_limiter_name_tenant'),
		Index('ix_sm_rate_limiters_scope', 'scope'),
		Index('ix_sm_rate_limiters_enabled', 'enabled'),
		Index('ix_sm_rate_limiters_tenant', 'tenant_id'),
		CheckConstraint('requests_per_second > 0', name='ck_rl_positive_rps'),
		CheckConstraint('burst_size > 0', name='ck_rl_positive_burst'),
	)

# =============================================================================
# Revolutionary AI-Powered Models
# =============================================================================

class SMNaturalLanguagePolicy(Base):
	"""Natural language policy model for conversational mesh configuration."""
	__tablename__ = "sm_nl_policies"
	
	# Primary fields
	nl_policy_id = Column(String(50), primary_key=True, default=lambda: f"nlp_{uuid7str()}")
	policy_name = Column(String(255), nullable=False)
	
	# Natural language processing
	natural_language_intent = Column(Text, nullable=False)
	processed_intent = Column(JSONB, nullable=False)
	compiled_rules = Column(JSONB, nullable=False)
	confidence_score = Column(Float, nullable=False)
	
	# Affected services and resources
	affected_services = Column(JSONB, default=list)
	affected_routes = Column(JSONB, default=list)
	deployment_strategy = Column(String(100))
	
	# Status and validation
	status = Column(String(50), default="pending")  # pending, active, failed, reverted
	validation_results = Column(JSONB, default=dict)
	compliance_mappings = Column(JSONB, default=list)
	
	# APG AI integration
	ai_model_version = Column(String(50))
	processing_metadata = Column(JSONB, default=dict)
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_by = Column(String(255), nullable=False)
	updated_by = Column(String(255))
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	__table_args__ = (
		UniqueConstraint('policy_name', 'tenant_id', name='uq_nl_policy_name_tenant'),
		Index('ix_sm_nl_policies_status', 'status'),
		Index('ix_sm_nl_policies_confidence', 'confidence_score'),
		Index('ix_sm_nl_policies_tenant', 'tenant_id'),
	)

class SMIntelligentTopology(Base):
	"""AI-powered service topology with predictive insights."""
	__tablename__ = "sm_intelligent_topology"
	
	# Primary fields
	topology_id = Column(String(50), primary_key=True, default=lambda: f"itopo_{uuid7str()}")
	mesh_version = Column(String(50), nullable=False)
	
	# Topology data
	topology_snapshot = Column(JSONB, nullable=False)
	service_dependencies = Column(JSONB, nullable=False)
	traffic_patterns = Column(JSONB, default=dict)
	
	# AI predictions and insights
	failure_predictions = Column(JSONB, default=list)
	optimization_recommendations = Column(JSONB, default=list)
	scaling_predictions = Column(JSONB, default=list)
	performance_insights = Column(JSONB, default=dict)
	
	# Machine learning metadata
	ml_model_version = Column(String(50))
	prediction_confidence = Column(Float)
	learning_feedback = Column(JSONB, default=dict)
	
	# Real-time collaboration
	active_viewers = Column(JSONB, default=list)
	collaborative_annotations = Column(JSONB, default=list)
	
	# Multi-tenancy and versioning
	tenant_id = Column(String(50), nullable=False)
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	
	__table_args__ = (
		Index('ix_sm_intelligent_topology_version', 'mesh_version'),
		Index('ix_sm_intelligent_topology_confidence', 'prediction_confidence'),
		Index('ix_sm_intelligent_topology_tenant', 'tenant_id'),
	)

class SMAutonomousMeshDecision(Base):
	"""Autonomous mesh decisions and self-healing actions."""
	__tablename__ = "sm_autonomous_decisions"
	
	# Primary fields
	decision_id = Column(String(50), primary_key=True, default=lambda: f"auto_{uuid7str()}")
	decision_type = Column(String(100), nullable=False)
	
	# Decision context
	trigger_event = Column(JSONB, nullable=False)
	analyzed_data = Column(JSONB, nullable=False)
	decision_rationale = Column(Text, nullable=False)
	
	# Actions taken
	actions_executed = Column(JSONB, nullable=False)
	rollback_plan = Column(JSONB, nullable=False)
	execution_status = Column(String(50), default="pending")
	
	# Results and validation
	execution_results = Column(JSONB, default=dict)
	success_metrics = Column(JSONB, default=dict)
	rollback_triggered = Column(Boolean, default=False)
	
	# AI confidence and learning
	decision_confidence = Column(Float, nullable=False)
	feedback_score = Column(Float)  # Human feedback on decision quality
	learning_data = Column(JSONB, default=dict)
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	__table_args__ = (
		Index('ix_sm_autonomous_decisions_type', 'decision_type'),
		Index('ix_sm_autonomous_decisions_status', 'execution_status'),
		Index('ix_sm_autonomous_decisions_confidence', 'decision_confidence'),
		Index('ix_sm_autonomous_decisions_tenant', 'tenant_id'),
	)

class SMFederatedLearningInsight(Base):
	"""Federated learning insights shared across APG deployments."""
	__tablename__ = "sm_federated_insights"
	
	# Primary fields
	insight_id = Column(String(50), primary_key=True, default=lambda: f"fed_{uuid7str()}")
	insight_type = Column(String(100), nullable=False)
	
	# Global learning data
	global_pattern = Column(JSONB, nullable=False)
	local_adaptation = Column(JSONB, nullable=False)
	aggregated_metrics = Column(JSONB, nullable=False)
	
	# Performance impact
	optimization_impact = Column(JSONB, default=dict)
	deployment_clusters = Column(JSONB, default=list)
	adoption_rate = Column(Float, default=0.0)
	
	# Federated learning metadata
	model_version = Column(String(50), nullable=False)
	contribution_weight = Column(Float, default=1.0)
	privacy_preserved = Column(Boolean, default=True)
	
	# Multi-tenancy
	tenant_id = Column(String(50), nullable=False)
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	
	__table_args__ = (
		Index('ix_sm_federated_insights_type', 'insight_type'),
		Index('ix_sm_federated_insights_model', 'model_version'),
		Index('ix_sm_federated_insights_tenant', 'tenant_id'),
	)

class SMPredictiveAlert(Base):
	"""Predictive alerts with failure prevention."""
	__tablename__ = "sm_predictive_alerts"
	
	# Primary fields
	alert_id = Column(String(50), primary_key=True, default=lambda: f"pred_{uuid7str()}")
	prediction_type = Column(String(100), nullable=False)
	
	# Prediction details
	predicted_event = Column(JSONB, nullable=False)
	prediction_confidence = Column(Float, nullable=False)
	predicted_time_to_failure = Column(Integer)  # seconds
	impact_assessment = Column(JSONB, nullable=False)
	
	# Prevention actions
	suggested_actions = Column(JSONB, default=list)
	auto_remediation_enabled = Column(Boolean, default=False)
	remediation_executed = Column(JSONB, default=list)
	
	# Validation and feedback
	prediction_accuracy = Column(Float)
	actual_outcome = Column(JSONB)
	feedback_incorporated = Column(Boolean, default=False)
	
	# Status tracking
	status = Column(String(50), default="active")  # active, resolved, false_positive
	escalation_level = Column(Integer, default=1)
	
	# Multi-tenancy and audit
	tenant_id = Column(String(50), nullable=False)
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
	
	__table_args__ = (
		Index('ix_sm_predictive_alerts_type', 'prediction_type'),
		Index('ix_sm_predictive_alerts_confidence', 'prediction_confidence'),
		Index('ix_sm_predictive_alerts_status', 'status'),
		Index('ix_sm_predictive_alerts_tenant', 'tenant_id'),
	)

class SMCollaborativeSession(Base):
	"""Real-time collaborative troubleshooting sessions."""
	__tablename__ = "sm_collaborative_sessions"
	
	# Primary fields
	session_id = Column(String(50), primary_key=True, default=lambda: f"collab_{uuid7str()}")
	session_name = Column(String(255), nullable=False)
	
	# Session details
	problem_description = Column(Text, nullable=False)
	affected_services = Column(JSONB, nullable=False)
	session_type = Column(String(50), default="troubleshooting")
	
	# Participants
	active_participants = Column(JSONB, default=list)
	participant_roles = Column(JSONB, default=dict)
	session_leader = Column(String(255))
	
	# Collaboration data
	shared_annotations = Column(JSONB, default=list)
	investigation_timeline = Column(JSONB, default=list)
	findings = Column(JSONB, default=list)
	resolution_actions = Column(JSONB, default=list)
	
	# AI assistance
	ai_suggestions = Column(JSONB, default=list)
	root_cause_analysis = Column(JSONB)
	automated_diagnostics = Column(JSONB, default=dict)
	
	# Session status
	status = Column(String(50), default="active")  # active, resolved, paused, archived
	resolution_confidence = Column(Float)
	
	# Multi-tenancy and timing
	tenant_id = Column(String(50), nullable=False)
	started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	ended_at = Column(DateTime(timezone=True))
	
	__table_args__ = (
		Index('ix_sm_collaborative_sessions_status', 'status'),
		Index('ix_sm_collaborative_sessions_type', 'session_type'),
		Index('ix_sm_collaborative_sessions_tenant', 'tenant_id'),
	)

# =============================================================================
# Pydantic Models for API Validation
# =============================================================================

class ServiceConfig(BaseModel):
	"""Pydantic model for service configuration validation."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	service_name: str = Field(..., min_length=1, max_length=255)
	service_version: str = Field(..., min_length=1, max_length=50)
	namespace: str = Field("default", max_length=255)
	description: Optional[str] = None
	tags: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	configuration: Dict[str, Any] = Field(default_factory=dict)
	environment: str = Field("production", max_length=100)

class EndpointConfig(BaseModel):
	"""Pydantic model for endpoint configuration validation."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	host: str = Field(..., min_length=1, max_length=255)
	port: int = Field(..., gt=0, le=65535)
	protocol: EndpointProtocol = Field(default=EndpointProtocol.HTTP)
	path: str = Field("/", max_length=500)
	weight: int = Field(100, gt=0)
	enabled: bool = True
	health_check_path: str = Field("/health", max_length=500)
	health_check_interval: int = Field(30, gt=0)
	health_check_timeout: int = Field(5, gt=0)
	tls_enabled: bool = False

class RouteConfig(BaseModel):
	"""Pydantic model for route configuration validation."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	route_name: str = Field(..., min_length=1, max_length=255)
	match_type: RouteMatchType = Field(default=RouteMatchType.PREFIX)
	match_value: str = Field(..., min_length=1, max_length=1000)
	match_headers: Dict[str, str] = Field(default_factory=dict)
	match_query: Dict[str, str] = Field(default_factory=dict)
	destination_services: List[Dict[str, Any]] = Field(..., min_items=1)
	timeout_ms: int = Field(30000, gt=0)
	retry_attempts: int = Field(3, ge=0)
	priority: int = Field(1000, gt=0)
	enabled: bool = True

class LoadBalancerConfig(BaseModel):
	"""Pydantic model for load balancer configuration validation."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	load_balancer_name: str = Field(..., min_length=1, max_length=255)
	algorithm: LoadBalancerAlgorithm = Field(default=LoadBalancerAlgorithm.ROUND_ROBIN)
	session_affinity: bool = False
	health_check_enabled: bool = True
	health_check_interval: int = Field(30, gt=0)
	circuit_breaker_enabled: bool = True
	failure_threshold: int = Field(5, gt=0)
	max_connections: int = Field(100, gt=0)
	enabled: bool = True

class PolicyConfig(BaseModel):
	"""Pydantic model for policy configuration validation."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	policy_name: str = Field(..., min_length=1, max_length=255)
	policy_type: PolicyType
	configuration: Dict[str, Any] = Field(..., min_items=1)
	enabled: bool = True
	priority: int = Field(1000, gt=0)
	description: Optional[str] = None

# Revolutionary Pydantic Models
class NaturalLanguagePolicyRequest(BaseModel):
	"""Pydantic model for natural language policy requests."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	policy_name: str = Field(..., min_length=1, max_length=255)
	natural_language_intent: str = Field(..., min_length=10, max_length=5000)
	deployment_strategy: Optional[str] = Field(None, regex=r'^(canary|blue_green|rolling|immediate)$')
	compliance_requirements: List[str] = Field(default_factory=list)
	risk_tolerance: str = Field("medium", regex=r'^(low|medium|high)$')

class IntelligentTopologyRequest(BaseModel):
	"""Pydantic model for intelligent topology requests."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	mesh_version: str = Field(..., min_length=1, max_length=50)
	include_predictions: bool = True
	prediction_horizon_hours: int = Field(24, gt=0, le=168)  # Max 1 week
	collaboration_enabled: bool = False
	analysis_depth: str = Field("standard", regex=r'^(basic|standard|deep)$')

class CollaborativeSessionRequest(BaseModel):
	"""Pydantic model for collaborative troubleshooting sessions."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	session_name: str = Field(..., min_length=1, max_length=255)
	problem_description: str = Field(..., min_length=10, max_length=5000)
	affected_services: List[str] = Field(..., min_items=1)
	session_type: str = Field("troubleshooting", regex=r'^(troubleshooting|optimization|planning)$')
	invite_participants: List[str] = Field(default_factory=list)
	ai_assistance_level: str = Field("standard", regex=r'^(minimal|standard|advanced)$')

class AutoRemediationConfig(BaseModel):
	"""Pydantic model for autonomous remediation configuration."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	enabled: bool = True
	confidence_threshold: float = Field(0.8, ge=0.0, le=1.0)
	max_actions_per_hour: int = Field(10, gt=0, le=100)
	allowed_action_types: List[str] = Field(default_factory=list)
	require_approval_for: List[str] = Field(default_factory=list)
	rollback_timeout_minutes: int = Field(15, gt=0, le=1440)

class FederatedLearningConfig(BaseModel):
	"""Pydantic model for federated learning configuration."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	participation_enabled: bool = True
	privacy_level: str = Field("high", regex=r'^(standard|high|maximum)$')
	data_sharing_scope: str = Field("performance_only", regex=r'^(performance_only|topology_patterns|full_insights)$')
	contribution_weight: float = Field(1.0, ge=0.1, le=10.0)
	update_frequency_hours: int = Field(24, gt=0, le=168)

# Create all database tables
def create_tables(engine):
	"""Create all service mesh tables."""
	Base.metadata.create_all(engine)