#!/usr/bin/env python3
"""
Registry (regy) - Models Test Suite
===================================

Comprehensive tests for all Pydantic models with validation, 
edge cases, and APG integration patterns.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import pytest
from pydantic import ValidationError

from ..models import (
	ServiceRegistration, ServiceInstance, ServiceDiscoveryQuery, ServiceDiscoveryResult,
	ServiceHealthStatus, ServiceEvent, ServiceMetrics, HealthCheck, CircuitBreakerConfig,
	ServiceVersion, ServiceEndpoint, ServiceStatus, ServiceType, HealthCheckType, 
	CircuitBreakerState, LoadBalanceStrategy, ProtocolType, ValidatedPort, ValidatedURL, ValidatedVersion
)

from . import TEST_TENANT_ID, TEST_USER_ID, TEST_SERVICE_NAME

class TestServiceEndpoint:
	"""Test ServiceEndpoint model validation and behavior."""
	
	def test_valid_service_endpoint(self):
		"""Test creating a valid service endpoint."""
		endpoint = ServiceEndpoint(
			path="/api/v1/users",
			protocol=ProtocolType.HTTPS,
			port=443,
			host="api.example.com",
			base_url="https://api.example.com/api/v1/users",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		assert endpoint.path == "/api/v1/users"
		assert endpoint.protocol == ProtocolType.HTTPS
		assert endpoint.port == 443
		assert endpoint.host == "api.example.com"
		assert endpoint.timeout_seconds == 30  # default
		assert endpoint.circuit_breaker_enabled == True  # default
		assert endpoint.tenant_id == TEST_TENANT_ID
	
	def test_invalid_port_validation(self):
		"""Test port validation fails for invalid values."""
		with pytest.raises(ValidationError) as exc_info:
			ServiceEndpoint(
				path="/api/v1/test",
				port=70000,  # Invalid port
				host="api.example.com",
				base_url="https://api.example.com/api/v1/test",
				tenant_id=TEST_TENANT_ID,
				created_by=TEST_USER_ID
			)
		
		assert "Port must be between 1 and 65535" in str(exc_info.value)
	
	def test_invalid_url_validation(self):
		"""Test URL validation fails for invalid formats."""
		with pytest.raises(ValidationError) as exc_info:
			ServiceEndpoint(
				path="/api/v1/test",
				port=443,
				host="api.example.com",
				base_url="invalid-url",  # Invalid URL format
				tenant_id=TEST_TENANT_ID,
				created_by=TEST_USER_ID
			)
		
		assert "Invalid URL format" in str(exc_info.value)

class TestHealthCheck:
	"""Test HealthCheck model validation and behavior."""
	
	def test_valid_health_check(self):
		"""Test creating a valid health check."""
		health_check = HealthCheck(
			name="HTTP Health Check",
			type=HealthCheckType.HTTP,
			url="https://api.example.com/health",
			interval_seconds=30,
			timeout_seconds=10,
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		assert health_check.name == "HTTP Health Check"
		assert health_check.type == HealthCheckType.HTTP
		assert health_check.enabled == True  # default
		assert health_check.healthy_threshold == 2  # default
		assert health_check.unhealthy_threshold == 3  # default
		assert len(health_check.expected_response_codes) == 1
		assert health_check.expected_response_codes[0] == 200
	
	def test_health_check_with_ml_features(self):
		"""Test health check with ML-powered features enabled."""
		health_check = HealthCheck(
			name="ML Health Check",
			type=HealthCheckType.HTTP,
			url="https://api.example.com/health",
			adaptive_intervals=True,
			anomaly_detection=True,
			predictive_analysis=True,
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		assert health_check.adaptive_intervals == True
		assert health_check.anomaly_detection == True
		assert health_check.predictive_analysis == True
	
	def test_custom_health_check(self):
		"""Test custom health check configuration."""
		health_check = HealthCheck(
			name="Custom DB Check",
			type=HealthCheckType.CUSTOM_SCRIPT,
			custom_script="SELECT 1 FROM users LIMIT 1",
			interval_seconds=60,
			timeout_seconds=30,
			expected_response_codes=[200, 201, 202],
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		assert health_check.type == HealthCheckType.CUSTOM_SCRIPT
		assert health_check.custom_script == "SELECT 1 FROM users LIMIT 1"
		assert len(health_check.expected_response_codes) == 3

class TestCircuitBreakerConfig:
	"""Test CircuitBreakerConfig model validation and behavior."""
	
	def test_valid_circuit_breaker(self):
		"""Test creating a valid circuit breaker configuration."""
		cb_config = CircuitBreakerConfig(
			name="API Circuit Breaker",
			failure_threshold=5,
			success_threshold=3,
			timeout_seconds=60,
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		assert cb_config.name == "API Circuit Breaker"
		assert cb_config.enabled == True  # default
		assert cb_config.state == CircuitBreakerState.CLOSED  # default
		assert cb_config.failure_threshold == 5
		assert cb_config.success_threshold == 3
		assert cb_config.failure_rate_threshold == 50.0  # default
	
	def test_circuit_breaker_with_ml_features(self):
		"""Test circuit breaker with ML optimization features."""
		cb_config = CircuitBreakerConfig(
			name="ML Circuit Breaker",
			adaptive_thresholds=True,
			pattern_recognition=True,
			intelligent_recovery=True,
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		assert cb_config.adaptive_thresholds == True
		assert cb_config.pattern_recognition == True
		assert cb_config.intelligent_recovery == True
	
	def test_circuit_breaker_statistics(self):
		"""Test circuit breaker statistics tracking."""
		cb_config = CircuitBreakerConfig(
			name="Stats Circuit Breaker",
			total_requests=1000,
			failed_requests=50,
			last_failure_time=datetime.now(timezone.utc),
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		assert cb_config.total_requests == 1000
		assert cb_config.failed_requests == 50
		assert cb_config.last_failure_time is not None
		failure_rate = (cb_config.failed_requests / cb_config.total_requests) * 100
		assert failure_rate == 5.0  # 5% failure rate

class TestServiceVersion:
	"""Test ServiceVersion model validation and behavior."""
	
	def test_valid_service_version(self):
		"""Test creating a valid service version."""
		version = ServiceVersion(
			version="1.2.3",
			is_current=True,
			release_date=datetime.now(timezone.utc),
			api_schema_url="https://api.example.com/schema/v1.2.3",
			backward_compatible=True,
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		assert version.version == "1.2.3"
		assert version.is_current == True
		assert version.is_deprecated == False  # default
		assert version.backward_compatible == True
		assert version.usage_count == 0  # default
	
	def test_invalid_version_format(self):
		"""Test semantic version validation."""
		with pytest.raises(ValidationError) as exc_info:
			ServiceVersion(
				version="invalid-version",  # Invalid semantic version
				tenant_id=TEST_TENANT_ID,
				created_by=TEST_USER_ID
			)
		
		assert "Invalid semantic version format" in str(exc_info.value)
	
	def test_version_with_prerelease(self):
		"""Test semantic version with prerelease identifiers."""
		version = ServiceVersion(
			version="2.0.0-beta.1",
			is_current=False,
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		assert version.version == "2.0.0-beta.1"
	
	def test_deprecated_version(self):
		"""Test deprecated version configuration."""
		version = ServiceVersion(
			version="1.0.0",
			is_deprecated=True,
			deprecation_date=datetime.now(timezone.utc),
			end_of_life_date=datetime.now(timezone.utc) + timedelta(days=90),
			breaking_changes=["Removed deprecated API endpoint /v1/old"],
			migration_guide_url="https://docs.example.com/migration/v1-to-v2",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		assert version.is_deprecated == True
		assert version.deprecation_date is not None
		assert version.end_of_life_date is not None
		assert len(version.breaking_changes) == 1
		assert version.migration_guide_url is not None

class TestServiceInstance:
	"""Test ServiceInstance model validation and behavior."""
	
	def test_valid_service_instance(self):
		"""Test creating a valid service instance."""
		endpoint = ServiceEndpoint(
			path="/api/v1/health",
			port=8080,
			host="service-1.example.com",
			base_url="http://service-1.example.com:8080/api/v1/health",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		health_check = HealthCheck(
			name="Instance Health Check",
			type=HealthCheckType.HTTP,
			url="http://service-1.example.com:8080/health",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		instance = ServiceInstance(
			service_id="service-123",
			instance_name="service-instance-1",
			host="service-1.example.com",
			port=8080,
			base_url="http://service-1.example.com:8080",
			endpoints=[endpoint],
			health_checks=[health_check],
			weight=100,
			max_connections=500,
			tenant_id=TEST_TENANT_ID,
			registered_by="service-manager"
		)
		
		assert instance.service_id == "service-123"
		assert instance.instance_name == "service-instance-1"
		assert instance.status == ServiceStatus.STARTING  # default
		assert instance.health_score == 1.0  # default
		assert instance.weight == 100
		assert instance.current_connections == 0  # default
		assert len(instance.endpoints) == 1
		assert len(instance.health_checks) == 1
	
	def test_instance_with_performance_metrics(self):
		"""Test service instance with performance metrics."""
		instance = ServiceInstance(
			service_id="service-123",
			instance_name="performance-instance",
			host="perf.example.com",
			port=8080,
			base_url="http://perf.example.com:8080",
			cpu_usage_percent=45.5,
			memory_usage_percent=67.8,
			response_time_ms=120.5,
			requests_per_second=150.0,
			current_connections=25,
			tenant_id=TEST_TENANT_ID,
			registered_by="performance-monitor"
		)
		
		assert instance.cpu_usage_percent == 45.5
		assert instance.memory_usage_percent == 67.8
		assert instance.response_time_ms == 120.5
		assert instance.requests_per_second == 150.0
		assert instance.current_connections == 25
	
	def test_instance_with_deployment_info(self):
		"""Test service instance with deployment information."""
		instance = ServiceInstance(
			service_id="service-123",
			instance_name="deployed-instance",
			host="deploy.example.com",
			port=8080,
			base_url="http://deploy.example.com:8080",
			container_id="container-abc123",
			node_id="node-xyz789",
			deployment_version="2.1.0",
			environment="production",
			tags=["production", "high-availability", "monitored"],
			tenant_id=TEST_TENANT_ID,
			registered_by="deployment-system"
		)
		
		assert instance.container_id == "container-abc123"
		assert instance.node_id == "node-xyz789"
		assert instance.deployment_version == "2.1.0"
		assert instance.environment == "production"
		assert len(instance.tags) == 3
		assert "production" in instance.tags

class TestServiceRegistration:
	"""Test ServiceRegistration model validation and behavior."""
	
	def create_sample_service_instance(self) -> ServiceInstance:
		"""Create a sample service instance for testing."""
		return ServiceInstance(
			service_id="test-service-id",
			instance_name="test-instance",
			host="test.example.com",
			port=8080,
			base_url="http://test.example.com:8080",
			tenant_id=TEST_TENANT_ID,
			registered_by=TEST_USER_ID
		)
	
	def create_sample_service_version(self) -> ServiceVersion:
		"""Create a sample service version for testing."""
		return ServiceVersion(
			version="1.0.0",
			is_current=True,
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
	
	def test_valid_service_registration(self):
		"""Test creating a valid service registration."""
		instance = self.create_sample_service_instance()
		version = self.create_sample_service_version()
		
		service = ServiceRegistration(
			name=TEST_SERVICE_NAME,
			display_name="Test Service",
			description="A test service for unit testing",
			service_type=ServiceType.REST_API,
			namespace="testing",
			environment="test",
			instances=[instance],
			versions=[version],
			current_version="1.0.0",
			tags=["test", "api", "microservice"],
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		assert service.name == TEST_SERVICE_NAME
		assert service.display_name == "Test Service"
		assert service.service_type == ServiceType.REST_API
		assert service.namespace == "testing"
		assert service.environment == "test"
		assert service.discovery_enabled == True  # default
		assert service.load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN  # default
		assert len(service.instances) == 1
		assert len(service.versions) == 1
		assert len(service.tags) == 3
		assert service.tenant_id == TEST_TENANT_ID
	
	def test_service_with_ml_features(self):
		"""Test service registration with ML/AI features enabled."""
		service = ServiceRegistration(
			name="ml-enabled-service",
			display_name="ML Enabled Service",
			service_type=ServiceType.AI_SERVICE,
			namespace="ml",
			environment="production",
			predictive_scaling=True,
			intelligent_routing=True,
			anomaly_detection=True,
			performance_optimization=True,
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		assert service.service_type == ServiceType.AI_SERVICE
		assert service.predictive_scaling == True
		assert service.intelligent_routing == True
		assert service.anomaly_detection == True
		assert service.performance_optimization == True
	
	def test_service_with_security_features(self):
		"""Test service registration with security features."""
		service = ServiceRegistration(
			name="secure-service",
			display_name="Secure Service",
			service_type=ServiceType.REST_API,
			namespace="secure",
			environment="production",
			authentication_required=True,
			authorization_policies=["admin-only", "read-write"],
			rate_limiting_enabled=True,
			cors_enabled=True,
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		assert service.authentication_required == True
		assert len(service.authorization_policies) == 2
		assert service.rate_limiting_enabled == True
		assert service.cors_enabled == True
		assert "admin-only" in service.authorization_policies
	
	def test_service_with_dependencies(self):
		"""Test service registration with dependencies."""
		service = ServiceRegistration(
			name="dependent-service",
			display_name="Dependent Service",
			service_type=ServiceType.MICROSERVICE,
			namespace="app",
			environment="production",
			dependencies=["auth-service", "config-service", "database-service"],
			dependents=["frontend-service", "mobile-service"],
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		assert len(service.dependencies) == 3
		assert len(service.dependents) == 2
		assert "auth-service" in service.dependencies
		assert "frontend-service" in service.dependents
	
	def test_service_with_metrics(self):
		"""Test service registration with performance metrics."""
		service = ServiceRegistration(
			name="metrics-service",
			display_name="Service with Metrics",
			service_type=ServiceType.REST_API,
			namespace="monitoring",
			environment="production",
			total_requests=50000,
			total_errors=123,
			average_response_time=45.6,
			uptime_percentage=99.95,
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		assert service.total_requests == 50000
		assert service.total_errors == 123
		assert service.average_response_time == 45.6
		assert service.uptime_percentage == 99.95
		
		# Calculate error rate
		error_rate = (service.total_errors / service.total_requests) * 100
		assert error_rate == 0.246  # 0.246% error rate

class TestServiceDiscoveryQuery:
	"""Test ServiceDiscoveryQuery model validation and behavior."""
	
	def test_basic_discovery_query(self):
		"""Test creating a basic service discovery query."""
		query = ServiceDiscoveryQuery(
			service_name="test-service",
			service_type=ServiceType.REST_API,
			namespace="production",
			environment="prod",
			tenant_id=TEST_TENANT_ID
		)
		
		assert query.service_name == "test-service"
		assert query.service_type == ServiceType.REST_API
		assert query.namespace == "production"
		assert query.environment == "prod"
		assert query.healthy_only == True  # default
		assert query.min_health_score == 0.0  # default
		assert query.limit == 50  # default
		assert query.offset == 0  # default
	
	def test_advanced_discovery_query(self):
		"""Test advanced service discovery with ML features."""
		query = ServiceDiscoveryQuery(
			service_type=ServiceType.MICROSERVICE,
			healthy_only=True,
			min_health_score=0.8,
			intelligent_ranking=True,
			predictive_filtering=True,
			similarity_search=True,
			tags=["production", "high-availability"],
			labels={"tier": "critical", "team": "backend"},
			version_constraints=">=2.0.0",
			limit=100,
			include_instances=True,
			include_health=True,
			include_metrics=True,
			tenant_id=TEST_TENANT_ID
		)
		
		assert query.min_health_score == 0.8
		assert query.intelligent_ranking == True
		assert query.predictive_filtering == True
		assert query.similarity_search == True
		assert len(query.tags) == 2
		assert query.labels["tier"] == "critical"
		assert query.version_constraints == ">=2.0.0"
		assert query.limit == 100
		assert query.include_metrics == True
	
	def test_performance_filtering_query(self):
		"""Test discovery query with performance filtering."""
		query = ServiceDiscoveryQuery(
			max_response_time=100.0,
			min_availability=99.0,
			preferred_regions=["us-east-1", "us-west-2"],
			load_balance_strategy=LoadBalanceStrategy.LEAST_CONNECTIONS,
			tenant_id=TEST_TENANT_ID
		)
		
		assert query.max_response_time == 100.0
		assert query.min_availability == 99.0
		assert len(query.preferred_regions) == 2
		assert query.load_balance_strategy == LoadBalanceStrategy.LEAST_CONNECTIONS

class TestServiceDiscoveryResult:
	"""Test ServiceDiscoveryResult model validation and behavior."""
	
	def test_discovery_result_basic(self):
		"""Test creating a basic discovery result."""
		result = ServiceDiscoveryResult(
			total_count=150,
			returned_count=50,
			query_time_ms=25.5,
			services=[],
			tenant_id=TEST_TENANT_ID
		)
		
		assert result.total_count == 150
		assert result.returned_count == 50
		assert result.query_time_ms == 25.5
		assert len(result.services) == 0
		assert result.cached_result == False  # default
		assert result.tenant_id == TEST_TENANT_ID
	
	def test_discovery_result_with_ai_insights(self):
		"""Test discovery result with AI-powered insights."""
		result = ServiceDiscoveryResult(
			total_count=25,
			returned_count=25,
			query_time_ms=45.2,
			services=[],
			ranking_algorithm="ml_health_score_v2",
			confidence_scores={"health_prediction": 0.92, "load_prediction": 0.87},
			recommendations=["Consider scaling service-1", "Monitor service-2 closely"],
			average_response_time=67.8,
			average_health_score=0.94,
			geographic_distribution={"us-east-1": 15, "us-west-2": 10},
			tenant_id=TEST_TENANT_ID
		)
		
		assert result.ranking_algorithm == "ml_health_score_v2"
		assert len(result.confidence_scores) == 2
		assert len(result.recommendations) == 2
		assert result.average_response_time == 67.8
		assert result.average_health_score == 0.94
		assert sum(result.geographic_distribution.values()) == 25

class TestServiceHealthStatus:
	"""Test ServiceHealthStatus model validation and behavior."""
	
	def test_basic_health_status(self):
		"""Test creating a basic health status."""
		health = ServiceHealthStatus(
			service_id="service-123",
			instance_id="instance-456",
			overall_status=ServiceStatus.HEALTHY,
			health_score=0.95,
			status_message="Service is operating normally",
			response_time_ms=45.2,
			cpu_usage_percent=35.7,
			memory_usage_percent=52.3,
			active_connections=25,
			circuit_breaker_state=CircuitBreakerState.CLOSED,
			failure_count=0,
			tenant_id=TEST_TENANT_ID
		)
		
		assert health.service_id == "service-123"
		assert health.instance_id == "instance-456"
		assert health.overall_status == ServiceStatus.HEALTHY
		assert health.health_score == 0.95
		assert health.response_time_ms == 45.2
		assert health.circuit_breaker_state == CircuitBreakerState.CLOSED
		assert health.failure_count == 0
	
	def test_unhealthy_status_with_ml_insights(self):
		"""Test unhealthy status with ML-powered insights."""
		health = ServiceHealthStatus(
			service_id="service-789",
			instance_id="instance-abc",
			overall_status=ServiceStatus.CRITICAL,
			health_score=0.15,
			status_message="Service is experiencing critical issues",
			response_time_ms=2500.0,
			cpu_usage_percent=95.8,
			memory_usage_percent=98.2,
			active_connections=500,
			circuit_breaker_state=CircuitBreakerState.OPEN,
			failure_count=25,
			anomaly_detected=True,
			predicted_failure_probability=0.89,
			recommended_actions=["Restart service immediately", "Scale up resources", "Check database connections"],
			tenant_id=TEST_TENANT_ID
		)
		
		assert health.overall_status == ServiceStatus.CRITICAL
		assert health.health_score == 0.15
		assert health.circuit_breaker_state == CircuitBreakerState.OPEN
		assert health.anomaly_detected == True
		assert health.predicted_failure_probability == 0.89
		assert len(health.recommended_actions) == 3

class TestServiceEvent:
	"""Test ServiceEvent model validation and behavior."""
	
	def test_basic_service_event(self):
		"""Test creating a basic service event."""
		event = ServiceEvent(
			event_type="service_registered",
			service_id="service-123",
			severity="info",
			message="Service registered successfully",
			triggered_by="service-manager",
			tenant_id=TEST_TENANT_ID,
			created_by="system"
		)
		
		assert event.event_type == "service_registered"
		assert event.service_id == "service-123"
		assert event.severity == "info"
		assert event.message == "Service registered successfully"
		assert event.triggered_by == "service-manager"
		assert event.resolved == False  # default
		assert event.tenant_id == TEST_TENANT_ID
	
	def test_critical_event_with_details(self):
		"""Test critical event with detailed information."""
		event = ServiceEvent(
			event_type="service_failure",
			service_id="service-456",
			instance_id="instance-789",
			severity="critical",
			message="Service instance failed health checks",
			details={
				"failure_reason": "Connection timeout",
				"last_successful_check": "2025-01-01T10:00:00Z",
				"consecutive_failures": 5
			},
			previous_state="healthy",
			new_state="critical",
			triggered_by="health_monitor",
			correlation_id="correlation-abc123",
			tags=["health", "failure", "critical"],
			performance_impact="High - affecting 1000+ users",
			affected_users=1250,
			tenant_id=TEST_TENANT_ID,
			created_by="health_monitor"
		)
		
		assert event.severity == "critical"
		assert event.instance_id == "instance-789"
		assert event.details["consecutive_failures"] == 5
		assert event.previous_state == "healthy"
		assert event.new_state == "critical"
		assert event.correlation_id == "correlation-abc123"
		assert len(event.tags) == 3
		assert event.affected_users == 1250
	
	def test_resolved_event(self):
		"""Test event with resolution information."""
		resolution_time = datetime.now(timezone.utc)
		
		event = ServiceEvent(
			event_type="service_recovery",
			service_id="service-123",
			severity="info",
			message="Service recovered from failure",
			triggered_by="auto_healing",
			resolved=True,
			resolution_time=resolution_time,
			resolution_notes="Service automatically recovered after circuit breaker reset",
			tenant_id=TEST_TENANT_ID,
			created_by="auto_healing"
		)
		
		assert event.resolved == True
		assert event.resolution_time == resolution_time
		assert "automatically recovered" in event.resolution_notes

class TestServiceMetrics:
	"""Test ServiceMetrics model validation and behavior."""
	
	def test_basic_service_metrics(self):
		"""Test creating basic service metrics."""
		metrics = ServiceMetrics(
			service_id="service-123",
			metric_type="performance",
			request_count=1000,
			error_count=25,
			response_time_p50=45.2,
			response_time_p95=120.5,
			response_time_p99=250.8,
			cpu_usage_avg=55.7,
			memory_usage_avg=67.3,
			tenant_id=TEST_TENANT_ID
		)
		
		assert metrics.service_id == "service-123"
		assert metrics.metric_type == "performance"
		assert metrics.request_count == 1000
		assert metrics.error_count == 25
		assert metrics.response_time_p50 == 45.2
		assert metrics.cpu_usage_avg == 55.7
		assert metrics.time_window_seconds == 1  # default
	
	def test_detailed_metrics_with_business_data(self):
		"""Test metrics with business and custom data."""
		metrics = ServiceMetrics(
			service_id="business-service",
			instance_id="business-instance-1",
			metric_type="business",
			time_window_seconds=3600,  # 1 hour
			request_count=50000,
			error_count=150,
			response_time_p50=65.4,
			response_time_p95=180.2,
			response_time_p99=350.7,
			cpu_usage_avg=45.8,
			memory_usage_avg=62.1,
			disk_usage_avg=78.9,
			network_bytes_in=1048576,  # 1MB
			network_bytes_out=2097152,  # 2MB
			uptime_seconds=3540,  # 59 minutes
			downtime_seconds=60,  # 1 minute
			availability_percentage=98.33,
			active_users=1250,
			successful_transactions=49850,
			revenue_impact=12500.75,
			custom_metrics={
				"cache_hit_rate": 0.92,
				"database_queries": 25000,
				"external_api_calls": 5000
			},
			tenant_id=TEST_TENANT_ID
		)
		
		assert metrics.time_window_seconds == 3600
		assert metrics.availability_percentage == 98.33
		assert metrics.active_users == 1250
		assert metrics.successful_transactions == 49850
		assert metrics.revenue_impact == 12500.75
		assert len(metrics.custom_metrics) == 3
		assert metrics.custom_metrics["cache_hit_rate"] == 0.92
		
		# Calculate derived metrics
		error_rate = (metrics.error_count / metrics.request_count) * 100
		assert abs(error_rate - 0.3) < 0.001  # 0.3% error rate

class TestModelIntegration:
	"""Test model integration and complex scenarios."""
	
	def test_complete_service_with_all_components(self):
		"""Test creating a complete service with all components."""
		# Create endpoint
		endpoint = ServiceEndpoint(
			path="/api/v1/users",
			port=8080,
			host="users.example.com",
			base_url="http://users.example.com:8080/api/v1/users",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		# Create health check
		health_check = HealthCheck(
			name="User Service Health",
			type=HealthCheckType.HTTP,
			url="http://users.example.com:8080/health",
			adaptive_intervals=True,
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		# Create circuit breaker
		circuit_breaker = CircuitBreakerConfig(
			name="User Service Circuit Breaker",
			adaptive_thresholds=True,
			pattern_recognition=True,
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		# Create version
		version = ServiceVersion(
			version="2.1.0",
			is_current=True,
			backward_compatible=True,
			api_schema_url="http://users.example.com:8080/api/schema",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID
		)
		
		# Create instance
		instance = ServiceInstance(
			service_id="users-service",
			instance_name="users-primary",
			host="users.example.com",
			port=8080,
			base_url="http://users.example.com:8080",
			endpoints=[endpoint],
			health_checks=[health_check],
			circuit_breakers=[circuit_breaker],
			weight=100,
			deployment_version="2.1.0",
			environment="production",
			tags=["users", "api", "production"],
			tenant_id=TEST_TENANT_ID,
			registered_by="deployment-system"
		)
		
		# Create complete service
		service = ServiceRegistration(
			name="users-service",
			display_name="User Management Service",
			description="Comprehensive user management and authentication service",
			service_type=ServiceType.REST_API,
			namespace="core",
			environment="production",
			base_path="/api/v1",
			instances=[instance],
			versions=[version],
			current_version="2.1.0",
			discovery_enabled=True,
			load_balance_strategy=LoadBalanceStrategy.WEIGHTED_RESPONSE_TIME,
			health_check_enabled=True,
			circuit_breaker_enabled=True,
			predictive_scaling=True,
			intelligent_routing=True,
			anomaly_detection=True,
			authentication_required=True,
			authorization_policies=["user-read", "user-write"],
			rate_limiting_enabled=True,
			dependencies=["auth-service", "database-service"],
			tags=["users", "authentication", "core-service", "production"],
			labels={"tier": "critical", "team": "platform"},
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		# Validate the complete service
		assert service.name == "users-service"
		assert len(service.instances) == 1
		assert len(service.instances[0].endpoints) == 1
		assert len(service.instances[0].health_checks) == 1
		assert len(service.instances[0].circuit_breakers) == 1
		assert len(service.versions) == 1
		assert service.predictive_scaling == True
		assert service.intelligent_routing == True
		assert len(service.authorization_policies) == 2
		assert len(service.dependencies) == 2
		assert service.labels["tier"] == "critical"
	
	def test_tenant_isolation_validation(self):
		"""Test that all models enforce tenant isolation."""
		models_with_tenant_id = [
			ServiceEndpoint,
			HealthCheck, 
			CircuitBreakerConfig,
			ServiceVersion,
			ServiceInstance,
			ServiceRegistration,
			ServiceDiscoveryQuery,
			ServiceDiscoveryResult,
			ServiceHealthStatus,
			ServiceEvent,
			ServiceMetrics
		]
		
		for model_class in models_with_tenant_id:
			# Each model should have tenant_id field
			assert hasattr(model_class.model_fields, 'tenant_id'), f"{model_class.__name__} missing tenant_id field"
			
			# tenant_id should be required (no default unless specified)
			tenant_field = model_class.model_fields['tenant_id']
			if hasattr(tenant_field, 'is_required'):
				# Pydantic v2 style
				assert tenant_field.is_required() or tenant_field.default is not None, \
					f"{model_class.__name__} tenant_id should be required or have default"

# Run async tests with asyncio event loop
def test_async_model_methods():
	"""Test any async methods in models (if added in future)."""
	# Placeholder for future async model methods
	pass

if __name__ == "__main__":
	pytest.main([__file__, "-v"])