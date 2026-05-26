#!/usr/bin/env python3
"""
APG Intelligent Gateway (APIG) - Simple Model Tests

Basic model tests that can run independently without complex APG dependencies.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
import sys
import os
from datetime import datetime, timezone

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
	AgGatewayConfig, AgApiRoute, AgUpstreamService, AgRateLimit,
	AgTrafficMetrics, AgSecurityEvent, AgWasmModule, AgHttpRequest,
	EnvironmentType, HttpMethod, PolicyType, ThreatLevel, LoadBalancingAlgorithm,
	validate_tenant_access
)

class TestModelCreation:
	"""Test basic model creation and validation."""

	def test_gateway_config(self):
		"""Test gateway configuration creation."""
		config = AgGatewayConfig(
			name='test-gateway',
			environment=EnvironmentType.DEVELOPMENT,
			tenant_id='test-tenant',
			created_by='test-user'
		)

		assert config.name == 'test-gateway'
		assert config.environment == EnvironmentType.DEVELOPMENT
		assert config.tenant_id == 'test-tenant'
		assert config.created_by == 'test-user'
		assert config.listen_port == 8080  # Default
		assert config.tls_enabled is True  # Default
		assert len(config.id) > 0  # UUID generated

	def test_upstream_service(self):
		"""Test upstream service creation."""
		service = AgUpstreamService(
			name='api-service',
			base_url='https://api.example.com'
		)

		assert service.name == 'api-service'
		assert service.base_url == 'https://api.example.com'
		assert service.weight == 100  # Default
		assert service.max_connections == 100  # Default
		assert len(service.id) > 0

	def test_api_route(self):
		"""Test API route creation."""
		upstream = AgUpstreamService(
			name='test-api',
			base_url='https://api.test.com'
		)

		route = AgApiRoute(
			path='/api/v1/users',
			method=HttpMethod.GET,
			upstream_services=[upstream],
			tenant_id='test-tenant',
			created_by='test-user'
		)

		assert route.path == '/api/v1/users'
		assert route.method == HttpMethod.GET
		assert len(route.upstream_services) == 1
		assert route.upstream_services[0].name == 'test-api'
		assert route.auth_required is True  # Default
		assert len(route.id) > 0

	def test_rate_limit(self):
		"""Test rate limit creation."""
		rate_limit = AgRateLimit(
			requests_per_second=100,
			requests_per_minute=5000
		)

		assert rate_limit.requests_per_second == 100
		assert rate_limit.requests_per_minute == 5000
		assert rate_limit.key_extractor == 'client_ip'  # Default
		assert len(rate_limit.id) > 0

	def test_traffic_metrics(self):
		"""Test traffic metrics creation."""
		metrics = AgTrafficMetrics(
			gateway_id='gateway-123',
			tenant_id='test-tenant'
		)

		assert metrics.gateway_id == 'gateway-123'
		assert metrics.tenant_id == 'test-tenant'
		assert metrics.request_count == 0  # Default
		assert metrics.error_count == 0  # Default
		assert isinstance(metrics.timestamp, datetime)
		assert len(metrics.id) > 0

	def test_security_event(self):
		"""Test security event creation."""
		event = AgSecurityEvent(
			gateway_id='gateway-123',
			event_type='suspicious_activity',
			threat_level=ThreatLevel.MEDIUM,
			confidence=0.8,
			source_ip='192.168.1.100',
			action_taken='logged',
			tenant_id='test-tenant'
		)

		assert event.gateway_id == 'gateway-123'
		assert event.event_type == 'suspicious_activity'
		assert event.threat_level == ThreatLevel.MEDIUM
		assert event.confidence == 0.8
		assert event.source_ip == '192.168.1.100'
		assert event.blocked is False  # Default
		assert len(event.id) > 0

	def test_wasm_module(self):
		"""Test WASM module creation."""
		wasm = AgWasmModule(
			name='request-transformer',
			wasm_binary_path='/path/to/module.wasm',
			tenant_id='test-tenant',
			created_by='test-user'
		)

		assert wasm.name == 'request-transformer'
		assert wasm.wasm_binary_path == '/path/to/module.wasm'
		assert wasm.entry_point == 'process_request'  # Default
		assert wasm.memory_limit_mb == 64  # Default
		assert len(wasm.id) > 0

	def test_http_request(self):
		"""Test HTTP request creation."""
		request = AgHttpRequest(
			method=HttpMethod.POST,
			path='/api/v1/users',
			client_ip='192.168.1.50',
			tenant_id='test-tenant'
		)

		assert request.method == HttpMethod.POST
		assert request.path == '/api/v1/users'
		assert request.client_ip == '192.168.1.50'
		assert request.query_string == ''  # Default
		assert len(request.headers) == 0  # Default
		assert isinstance(request.received_at, datetime)
		assert len(request.id) > 0

class TestModelValidation:
	"""Test model field validation."""

	def test_upstream_url_validation(self):
		"""Test upstream service URL validation."""
		# Valid URLs should work
		valid_urls = [
			'http://api.example.com',
			'https://api.example.com',
			'https://api.example.com:8080'
		]

		for url in valid_urls:
			service = AgUpstreamService(
				name='test',
				base_url=url
			)
			assert url.rstrip('/') in service.base_url

		# Invalid URLs should raise errors
		with pytest.raises(ValueError):
			AgUpstreamService(
				name='test',
				base_url='ftp://invalid.com'
			)

	def test_route_path_validation(self):
		"""Test API route path validation."""
		upstream = AgUpstreamService(
			name='test',
			base_url='https://api.test.com'
		)

		# Valid paths
		valid_paths = ['/api/v1/users', '/health', '/']
		for path in valid_paths:
			route = AgApiRoute(
				path=path,
				method=HttpMethod.GET,
				upstream_services=[upstream],
				tenant_id='test-tenant',
				created_by='test-user'
			)
			assert route.path == path

		# Invalid paths should raise errors
		with pytest.raises(ValueError):
			AgApiRoute(
				path='invalid-path',  # Doesn't start with /
				method=HttpMethod.GET,
				upstream_services=[upstream],
				tenant_id='test-tenant',
				created_by='test-user'
			)

	def test_rate_limit_validation(self):
		"""Test rate limit validation."""
		# Valid extractor
		rate_limit = AgRateLimit(
			requests_per_second=100,
			key_extractor='api_key'
		)
		assert rate_limit.key_extractor == 'api_key'

		# Invalid extractor should raise error
		with pytest.raises(ValueError):
			AgRateLimit(
				requests_per_second=100,
				key_extractor='invalid_extractor'
			)

class TestAsyncFunctions:
	"""Test async model functions."""

	async def test_tenant_validation(self):
		"""Test tenant access validation."""
		# Should work with valid inputs
		result = await validate_tenant_access('test-tenant', 'test-user')
		assert result is True

		# Empty credentials should be denied without crashing callers
		assert await validate_tenant_access('', 'test-user') is False
		assert await validate_tenant_access('test-tenant', '') is False

# Simple test runner for direct execution
if __name__ == '__main__':
	# Run sync tests
	test_model = TestModelCreation()

	print('✓ Testing model creation...')
	test_model.test_gateway_config()
	test_model.test_upstream_service()
	test_model.test_api_route()
	test_model.test_rate_limit()
	test_model.test_traffic_metrics()
	test_model.test_security_event()
	test_model.test_wasm_module()
	test_model.test_http_request()

	print('✓ Testing model validation...')
	test_validation = TestModelValidation()
	test_validation.test_upstream_url_validation()
	test_validation.test_route_path_validation()
	test_validation.test_rate_limit_validation()

	print('✓ Testing async functions...')
	async def run_async_tests():
		test_async = TestAsyncFunctions()
		await test_async.test_tenant_validation()

	asyncio.run(run_async_tests())

	print('✓ All model tests passed successfully!')
