#!/usr/bin/env python3
"""
APG Intelligent Gateway (APIG) - Traffic Manager Tests

Comprehensive tests for advanced traffic management including intelligent
load balancing, circuit breakers, and adaptive rate limiting.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
import time
from datetime import datetime, timezone, timedelta
from typing import List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_manager import (
	TrafficManager, IntelligentLoadBalancer, CircuitBreakerManager, 
	AdaptiveRateLimiter, LoadBalancingDecision, ServiceHealth,
	CircuitBreakerConfig, HealthStatus, CircuitState, TrafficClass
)
from models import AgUpstreamService, AgHttpRequest, HttpMethod, LoadBalancingAlgorithm

class TestIntelligentLoadBalancer:
	"""Test intelligent load balancing functionality."""
	
	async def test_load_balancer_initialization(self):
		"""Test load balancer initialization."""
		lb = IntelligentLoadBalancer('test-tenant')
		
		assert lb.tenant_id == 'test-tenant'
		assert len(lb.service_health) == 0
		assert len(lb.algorithm_performance) == 6  # All supported algorithms
		
	async def test_round_robin_selection(self):
		"""Test round-robin load balancing."""
		lb = IntelligentLoadBalancer('test-tenant')
		
		services = [
			AgUpstreamService(name='service-1', base_url='http://service1:8080'),
			AgUpstreamService(name='service-2', base_url='http://service2:8080'),
			AgUpstreamService(name='service-3', base_url='http://service3:8080')
		]
		
		request = AgHttpRequest(
			method=HttpMethod.GET,
			path='/api/test',
			client_ip='192.168.1.100',
			tenant_id='test-tenant'
		)
		
		# Test multiple selections to verify round-robin behavior
		selections = []
		for _ in range(6):
			decision = await lb.select_upstream_service(
				services, request, LoadBalancingAlgorithm.ROUND_ROBIN
			)
			selections.append(decision.selected_service.name)
		
		# Should cycle through services in order
		expected = ['service-1', 'service-2', 'service-3', 'service-1', 'service-2', 'service-3']
		assert selections == expected
		
	async def test_weighted_response_time_selection(self):
		"""Test weighted response time load balancing."""
		lb = IntelligentLoadBalancer('test-tenant')
		
		services = [
			AgUpstreamService(name='fast-service', base_url='http://fast:8080', weight=100),
			AgUpstreamService(name='slow-service', base_url='http://slow:8080', weight=50)
		]
		
		# Update health with different response times
		await lb.update_service_health(services[0].id, ServiceHealth(
			service_id=services[0].id,
			status=HealthStatus.HEALTHY,
			response_time_ms=10.0,
			success_rate=0.99
		))
		
		await lb.update_service_health(services[1].id, ServiceHealth(
			service_id=services[1].id,
			status=HealthStatus.HEALTHY,
			response_time_ms=100.0,
			success_rate=0.95
		))
		
		request = AgHttpRequest(
			method=HttpMethod.GET,
			path='/api/test',
			client_ip='192.168.1.100',
			tenant_id='test-tenant'
		)
		
		# Fast service should be selected more often
		fast_selections = 0
		total_selections = 10
		
		for _ in range(total_selections):
			decision = await lb.select_upstream_service(
				services, request, LoadBalancingAlgorithm.WEIGHTED_RESPONSE_TIME
			)
			if decision.selected_service.name == 'fast-service':
				fast_selections += 1
		
		# Fast service should be selected majority of times
		assert fast_selections >= 7  # At least 70% selection rate
		
	async def test_least_connections_selection(self):
		"""Test least connections load balancing."""
		lb = IntelligentLoadBalancer('test-tenant')
		
		services = [
			AgUpstreamService(name='service-low', base_url='http://low:8080'),
			AgUpstreamService(name='service-high', base_url='http://high:8080')
		]
		
		# Update health with different connection counts
		await lb.update_service_health(services[0].id, ServiceHealth(
			service_id=services[0].id,
			status=HealthStatus.HEALTHY,
			active_connections=5
		))
		
		await lb.update_service_health(services[1].id, ServiceHealth(
			service_id=services[1].id,
			status=HealthStatus.HEALTHY,
			active_connections=20
		))
		
		request = AgHttpRequest(
			method=HttpMethod.GET,
			path='/api/test',
			client_ip='192.168.1.100',
			tenant_id='test-tenant'
		)
		
		decision = await lb.select_upstream_service(
			services, request, LoadBalancingAlgorithm.LEAST_CONNECTIONS
		)
		
		# Should select service with fewer connections
		assert decision.selected_service.name == 'service-low'
		
	async def test_adaptive_algorithm_selection(self):
		"""Test adaptive AI algorithm selection."""
		lb = IntelligentLoadBalancer('test-tenant')
		
		services = [
			AgUpstreamService(name='service-1', base_url='http://service1:8080'),
			AgUpstreamService(name='service-2', base_url='http://service2:8080')
		]
		
		request = AgHttpRequest(
			method=HttpMethod.GET,
			path='/api/v1/session/data',  # Should trigger consistent hash
			client_ip='192.168.1.100',
			tenant_id='test-tenant'
		)
		
		decision = await lb.select_upstream_service(
			services, request, LoadBalancingAlgorithm.ADAPTIVE_AI
		)
		
		assert decision.selected_service is not None
		assert decision.algorithm_used in [
			LoadBalancingAlgorithm.ROUND_ROBIN,
			LoadBalancingAlgorithm.WEIGHTED_RESPONSE_TIME,
			LoadBalancingAlgorithm.LEAST_CONNECTIONS,
			LoadBalancingAlgorithm.CONSISTENT_HASH
		]

class TestCircuitBreakerManager:
	"""Test circuit breaker functionality."""
	
	async def test_circuit_breaker_initialization(self):
		"""Test circuit breaker initialization."""
		cb = CircuitBreakerManager('test-tenant')
		
		assert cb.tenant_id == 'test-tenant'
		assert cb.default_config.failure_threshold == 5
		assert len(cb.circuit_states) == 0
		
	async def test_circuit_closed_state(self):
		"""Test circuit breaker in closed state."""
		cb = CircuitBreakerManager('test-tenant')
		
		service_id = 'test-service'
		
		# Initially should allow requests (closed state)
		allowed = await cb.should_allow_request(service_id)
		assert allowed is True
		
		# Record some successes
		for _ in range(3):
			await cb.record_success(service_id, 50.0)
		
		# Should still allow requests
		allowed = await cb.should_allow_request(service_id)
		assert allowed is True
		
	async def test_circuit_open_transition(self):
		"""Test circuit breaker opening on failures."""
		config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1)
		cb = CircuitBreakerManager('test-tenant', config)
		
		service_id = 'test-service'
		
		# Record failures to trigger circuit opening
		for _ in range(3):
			await cb.record_failure(service_id, 'timeout')
		
		# Circuit should now be open
		allowed = await cb.should_allow_request(service_id)
		assert allowed is False
		
		# Check circuit status
		status = await cb.get_circuit_status(service_id)
		assert status['state'] == 'open'
		assert status['failure_count'] == 3
		
	async def test_circuit_half_open_recovery(self):
		"""Test circuit breaker half-open recovery."""
		config = CircuitBreakerConfig(
			failure_threshold=2, 
			recovery_timeout=1,  # 1 second recovery
			success_threshold=2
		)
		cb = CircuitBreakerManager('test-tenant', config)
		
		service_id = 'test-service'
		
		# Trigger circuit opening
		for _ in range(2):
			await cb.record_failure(service_id, 'error')
		
		# Wait for recovery timeout
		await asyncio.sleep(1.1)
		
		# Should transition to half-open
		allowed = await cb.should_allow_request(service_id)
		assert allowed is True
		
		status = await cb.get_circuit_status(service_id)
		assert status['state'] == 'half_open'
		
		# Record successes to close circuit
		for _ in range(2):
			await cb.record_success(service_id, 30.0)
		
		# Circuit should now be closed
		status = await cb.get_circuit_status(service_id)
		assert status['state'] == 'closed'

class TestAdaptiveRateLimiter:
	"""Test adaptive rate limiting functionality."""
	
	async def test_rate_limiter_initialization(self):
		"""Test rate limiter initialization."""
		rl = AdaptiveRateLimiter('test-tenant')
		
		assert rl.tenant_id == 'test-tenant'
		assert len(rl.token_buckets) == 0
		assert rl.system_health_score == 1.0
		
	async def test_token_bucket_behavior(self):
		"""Test basic token bucket rate limiting."""
		rl = AdaptiveRateLimiter('test-tenant')
		
		key = 'test-user'
		limit = 10  # Reasonable limit for testing
		
		# Test that rate limiter creates bucket and tracks requests
		allowed, metadata = await rl.should_allow_request(key, limit)
		assert allowed is True
		assert 'tokens_remaining' in metadata
		assert 'adaptive_limit' in metadata
		assert 'base_limit' in metadata
		
		# Test that bucket exists after first request
		assert key in rl.token_buckets
		
		# Test eventual rate limiting by making many rapid requests
		blocked_found = False
		for i in range(50):  # Many requests to trigger limiting
			allowed, metadata = await rl.should_allow_request(key, limit)
			if not allowed:
				blocked_found = True
				assert 'retry_after' in metadata
				break
		
		# Should eventually hit rate limit
		assert blocked_found, "Rate limiter should eventually block requests"
		
	async def test_adaptive_limit_calculation(self):
		"""Test adaptive limit calculation based on system health."""
		rl = AdaptiveRateLimiter('test-tenant')
		
		key = 'test-user'
		base_limit = 10
		
		# Test with high system health (should increase limit)
		await rl.update_system_health(1.5)
		
		request = AgHttpRequest(
			method=HttpMethod.GET,
			path='/api/v1/critical',
			client_ip='192.168.1.100',
			tenant_id='test-tenant'
		)
		
		allowed, metadata = await rl.should_allow_request(key, base_limit, request)
		assert allowed is True
		assert metadata['adaptive_limit'] > base_limit
		
	async def test_user_behavior_analysis(self):
		"""Test user behavior analysis for adaptive limiting."""
		rl = AdaptiveRateLimiter('test-tenant')
		
		good_user = 'good-user'
		bad_user = 'bad-user'
		
		# Simulate good user behavior (low frequency)
		for _ in range(3):
			await rl.should_allow_request(good_user, 10)
			await asyncio.sleep(0.1)  # Space out requests
		
		# Simulate bad user behavior (high frequency)
		for _ in range(20):
			await rl.should_allow_request(bad_user, 10)
		
		# Good user should get better treatment
		good_allowed, good_meta = await rl.should_allow_request(good_user, 10)
		bad_allowed, bad_meta = await rl.should_allow_request(bad_user, 10)
		
		# Good user should have higher or equal limit
		assert good_meta.get('adaptive_limit', 0) >= bad_meta.get('adaptive_limit', 0)

class TestTrafficManager:
	"""Test comprehensive traffic management."""
	
	async def test_traffic_manager_initialization(self):
		"""Test traffic manager initialization."""
		tm = TrafficManager('test-tenant')
		
		assert tm.tenant_id == 'test-tenant'
		assert tm.load_balancer is not None
		assert tm.circuit_breaker is not None
		assert tm.rate_limiter is not None
		assert tm.request_count == 0
		
	async def test_comprehensive_request_processing(self):
		"""Test end-to-end request processing."""
		tm = TrafficManager('test-tenant')
		
		services = [
			AgUpstreamService(name='service-1', base_url='http://service1:8080'),
			AgUpstreamService(name='service-2', base_url='http://service2:8080')
		]
		
		request = AgHttpRequest(
			method=HttpMethod.GET,
			path='/api/v1/data',
			client_ip='192.168.1.100',
			tenant_id='test-tenant'
		)
		
		# Process request
		result = await tm.process_request(
			request=request,
			available_services=services,
			rate_limit_key='192.168.1.100',
			rate_limit_per_second=100
		)
		
		assert result['allowed'] is True
		assert 'selected_service' in result
		assert 'algorithm_used' in result
		assert 'processing_time_ms' in result
		assert result['selected_service'].name in ['service-1', 'service-2']
		
	async def test_rate_limiting_protection(self):
		"""Test rate limiting protection."""
		tm = TrafficManager('test-tenant')
		
		services = [
			AgUpstreamService(name='service-1', base_url='http://service1:8080')
		]
		
		request = AgHttpRequest(
			method=HttpMethod.GET,
			path='/api/test',
			client_ip='192.168.1.100',
			tenant_id='test-tenant'
		)
		
		rate_limit = 2  # Very low limit for testing
		
		# Should allow first requests
		for _ in range(2):
			result = await tm.process_request(
				request, services, '192.168.1.100', rate_limit
			)
			assert result['allowed'] is True
		
		# Should block subsequent request
		result = await tm.process_request(
			request, services, '192.168.1.100', rate_limit
		)
		assert result['allowed'] is False
		assert result['reason'] == 'rate_limited'
		
	async def test_circuit_breaker_protection(self):
		"""Test circuit breaker protection."""
		tm = TrafficManager('test-tenant')
		
		services = [
			AgUpstreamService(name='failing-service', base_url='http://failing:8080')
		]
		
		request = AgHttpRequest(
			method=HttpMethod.GET,
			path='/api/test',
			client_ip='192.168.1.100',
			tenant_id='test-tenant'
		)
		
		# Simulate failures to open circuit
		service_id = services[0].id
		for _ in range(5):  # Exceed failure threshold
			await tm.record_response(service_id, False, 1000.0, 'timeout')
		
		# Request should now be blocked by circuit breaker
		result = await tm.process_request(
			request, services, '192.168.1.100', 100
		)
		
		# Should be blocked due to open circuit (no backup services)
		assert result['allowed'] is False
		assert result['reason'] == 'circuit_open'
		
	async def test_backup_service_selection(self):
		"""Test backup service selection when primary circuit is open."""
		tm = TrafficManager('test-tenant')
		
		services = [
			AgUpstreamService(name='primary-service', base_url='http://primary:8080'),
			AgUpstreamService(name='backup-service', base_url='http://backup:8080')
		]
		
		request = AgHttpRequest(
			method=HttpMethod.GET,
			path='/api/test',
			client_ip='192.168.1.100',
			tenant_id='test-tenant'
		)
		
		# Open circuit for first service
		primary_id = services[0].id
		for _ in range(5):
			await tm.record_response(primary_id, False, 1000.0, 'error')
		
		# Request should use backup service
		result = await tm.process_request(
			request, services, '192.168.1.100', 100
		)
		
		assert result['allowed'] is True
		# Should either select backup initially or switch to backup due to circuit
		assert 'backup' in result['reasoning'] or result['selected_service'].name == 'backup-service'
		
	async def test_traffic_statistics(self):
		"""Test traffic statistics collection."""
		tm = TrafficManager('test-tenant')
		
		services = [
			AgUpstreamService(name='service-1', base_url='http://service1:8080')
		]
		
		request = AgHttpRequest(
			method=HttpMethod.GET,
			path='/api/test',
			client_ip='192.168.1.100',
			tenant_id='test-tenant'
		)
		
		# Process several requests
		for _ in range(5):
			await tm.process_request(request, services, '192.168.1.100', 100)
		
		# Record some responses
		service_id = services[0].id
		for i in range(5):
			await tm.record_response(service_id, True, 50.0 + i * 10)
		
		# Get statistics
		stats = await tm.get_traffic_stats()
		
		assert stats['total_requests'] == 5
		assert stats['successful_requests'] == 5
		assert stats['failed_requests'] == 0
		assert stats['success_rate'] == 1.0
		assert stats['average_response_time_ms'] > 0
		assert 'rate_limiting' in stats

# Simple test runner for direct execution
if __name__ == '__main__':
	async def run_tests():
		print('🧪 Running Traffic Manager Tests...')
		
		# Test Load Balancer
		print('\\n📊 Testing Intelligent Load Balancer...')
		lb_tests = TestIntelligentLoadBalancer()
		
		await lb_tests.test_load_balancer_initialization()
		print('  ✅ Load balancer initialization')
		
		await lb_tests.test_round_robin_selection()
		print('  ✅ Round-robin selection')
		
		await lb_tests.test_weighted_response_time_selection()
		print('  ✅ Weighted response time selection')
		
		await lb_tests.test_least_connections_selection()
		print('  ✅ Least connections selection')
		
		await lb_tests.test_adaptive_algorithm_selection()
		print('  ✅ Adaptive AI algorithm selection')
		
		# Test Circuit Breaker
		print('\\n⚡ Testing Circuit Breaker Manager...')
		cb_tests = TestCircuitBreakerManager()
		
		await cb_tests.test_circuit_breaker_initialization()
		print('  ✅ Circuit breaker initialization')
		
		await cb_tests.test_circuit_closed_state()
		print('  ✅ Circuit closed state behavior')
		
		await cb_tests.test_circuit_open_transition()
		print('  ✅ Circuit opening on failures')
		
		await cb_tests.test_circuit_half_open_recovery()
		print('  ✅ Circuit half-open recovery')
		
		# Test Rate Limiter
		print('\\n🚦 Testing Adaptive Rate Limiter...')
		rl_tests = TestAdaptiveRateLimiter()
		
		await rl_tests.test_rate_limiter_initialization()
		print('  ✅ Rate limiter initialization')
		
		await rl_tests.test_token_bucket_behavior()
		print('  ✅ Token bucket behavior')
		
		await rl_tests.test_adaptive_limit_calculation()
		print('  ✅ Adaptive limit calculation')
		
		await rl_tests.test_user_behavior_analysis()
		print('  ✅ User behavior analysis')
		
		# Test Traffic Manager
		print('\\n🚀 Testing Comprehensive Traffic Manager...')
		tm_tests = TestTrafficManager()
		
		await tm_tests.test_traffic_manager_initialization()
		print('  ✅ Traffic manager initialization')
		
		await tm_tests.test_comprehensive_request_processing()
		print('  ✅ End-to-end request processing')
		
		await tm_tests.test_rate_limiting_protection()
		print('  ✅ Rate limiting protection')
		
		await tm_tests.test_circuit_breaker_protection()
		print('  ✅ Circuit breaker protection')
		
		await tm_tests.test_backup_service_selection()
		print('  ✅ Backup service selection')
		
		await tm_tests.test_traffic_statistics()
		print('  ✅ Traffic statistics collection')
		
		print('\\n🎉 All Traffic Manager tests passed successfully!')
		print('💡 Revolutionary traffic management system is working perfectly!')
	
	# Run all tests
	asyncio.run(run_tests())