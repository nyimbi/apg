"""
APG Integration API Management - Gateway Tests

Unit and integration tests for the API gateway including routing,
load balancing, circuit breaking, and middleware functionality.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import pytest_asyncio
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import web, ClientSession, ClientTimeout, ClientError
from aiohttp.test_utils import make_mocked_coro

from ..gateway import (
	APIGateway, GatewayRouter, LoadBalancer, CircuitBreaker,
	UpstreamServer, RouteConfig, GatewayRequest, GatewayResponse,
	RateLimitMiddleware, AuthenticationMiddleware, PolicyEnforcementMiddleware,
	CachingMiddleware, LoadBalancingStrategy
)
from ..service import APILifecycleService, ConsumerManagementService

# =============================================================================
# Load Balancer Tests
# =============================================================================

@pytest.mark.unit
class TestLoadBalancer:
	"""Test load balancer functionality."""
	
	def test_round_robin_selection(self):
		"""Test round robin server selection."""
		load_balancer = LoadBalancer()
		
		servers = [
			UpstreamServer(url="http://server1.local"),
			UpstreamServer(url="http://server2.local"),
			UpstreamServer(url="http://server3.local")
		]
		
		# Test multiple selections
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		selected_servers = []
		for _ in range(6):
			server = loop.run_until_complete(
				load_balancer.select_server(servers, LoadBalancingStrategy.ROUND_ROBIN)
			)
			selected_servers.append(server.url)
		
		# Should cycle through servers
		expected = [
			"http://server1.local", "http://server2.local", "http://server3.local",
			"http://server1.local", "http://server2.local", "http://server3.local"
		]
		assert selected_servers == expected
		
		loop.close()
	
	def test_weighted_round_robin_selection(self):
		"""Test weighted round robin server selection."""
		load_balancer = LoadBalancer()
		
		servers = [
			UpstreamServer(url="http://server1.local", weight=3),
			UpstreamServer(url="http://server2.local", weight=1),
			UpstreamServer(url="http://server3.local", weight=2)
		]
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		selected_servers = []
		for _ in range(12):  # 2 full cycles (3+1+2)*2
			server = loop.run_until_complete(
				load_balancer.select_server(servers, LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN)
			)
			selected_servers.append(server.url)
		
		# Count selections
		server1_count = selected_servers.count("http://server1.local")
		server2_count = selected_servers.count("http://server2.local")
		server3_count = selected_servers.count("http://server3.local")
		
		# Should respect weights (6:2:4 ratio)
		assert server1_count == 6
		assert server2_count == 2
		assert server3_count == 4
		
		loop.close()
	
	def test_least_connections_selection(self):
		"""Test least connections server selection."""
		load_balancer = LoadBalancer()
		
		servers = [
			UpstreamServer(url="http://server1.local", current_connections=5),
			UpstreamServer(url="http://server2.local", current_connections=2),
			UpstreamServer(url="http://server3.local", current_connections=8)
		]
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		server = loop.run_until_complete(
			load_balancer.select_server(servers, LoadBalancingStrategy.LEAST_CONNECTIONS)
		)
		
		# Should select server2 with least connections
		assert server.url == "http://server2.local"
		
		loop.close()
	
	def test_ip_hash_selection(self):
		"""Test IP hash server selection for session affinity."""
		load_balancer = LoadBalancer()
		
		servers = [
			UpstreamServer(url="http://server1.local"),
			UpstreamServer(url="http://server2.local"),
			UpstreamServer(url="http://server3.local")
		]
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		# Same IP should always get same server
		client_ip = "192.168.1.100"
		
		selected_servers = []
		for _ in range(5):
			server = loop.run_until_complete(
				load_balancer.select_server(servers, LoadBalancingStrategy.IP_HASH, client_ip)
			)
			selected_servers.append(server.url)
		
		# All selections should be the same
		assert all(url == selected_servers[0] for url in selected_servers)
		
		loop.close()
	
	def test_no_healthy_servers(self):
		"""Test behavior when no servers are healthy."""
		load_balancer = LoadBalancer()
		
		servers = [
			UpstreamServer(url="http://server1.local", is_healthy=False),
			UpstreamServer(url="http://server2.local", is_healthy=False)
		]
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		server = loop.run_until_complete(
			load_balancer.select_server(servers, LoadBalancingStrategy.ROUND_ROBIN)
		)
		
		assert server is None
		
		loop.close()

# =============================================================================
# Circuit Breaker Tests
# =============================================================================

@pytest.mark.unit
class TestCircuitBreaker:
	"""Test circuit breaker functionality."""
	
	@pytest.mark.asyncio
	async def test_circuit_breaker_closed_state(self):
		"""Test circuit breaker in closed state."""
		circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=60)
		
		# Successful calls should work
		async def successful_function():
			return "success"
		
		result = await circuit_breaker.call(successful_function)
		assert result == "success"
		assert circuit_breaker.state == "CLOSED"
		assert circuit_breaker.failure_count == 0
	
	@pytest.mark.asyncio
	async def test_circuit_breaker_open_state(self):
		"""Test circuit breaker opening after failures."""
		circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=60)
		
		# Function that always fails
		async def failing_function():
			raise Exception("Test failure")
		
		# First failure
		with pytest.raises(Exception):
			await circuit_breaker.call(failing_function)
		assert circuit_breaker.state == "CLOSED"
		assert circuit_breaker.failure_count == 1
		
		# Second failure - should open circuit
		with pytest.raises(Exception):
			await circuit_breaker.call(failing_function)
		assert circuit_breaker.state == "OPEN"
		assert circuit_breaker.failure_count == 2
		
		# Third call should be blocked
		with pytest.raises(Exception, match="Circuit breaker is OPEN"):
			await circuit_breaker.call(failing_function)
	
	@pytest.mark.asyncio
	async def test_circuit_breaker_half_open_state(self):
		"""Test circuit breaker half-open state."""
		circuit_breaker = CircuitBreaker(failure_threshold=1, timeout=0.1)
		
		# Cause failure to open circuit
		async def failing_function():
			raise Exception("Test failure")
		
		with pytest.raises(Exception):
			await circuit_breaker.call(failing_function)
		assert circuit_breaker.state == "OPEN"
		
		# Wait for timeout
		await asyncio.sleep(0.2)
		
		# Next call should be in half-open state
		async def successful_function():
			return "success"
		
		result = await circuit_breaker.call(successful_function)
		assert result == "success"
		assert circuit_breaker.state == "CLOSED"
		assert circuit_breaker.failure_count == 0

# =============================================================================
# Middleware Tests
# =============================================================================

@pytest.mark.unit
class TestRateLimitMiddleware:
	"""Test rate limiting middleware."""
	
	@pytest_asyncio.fixture
	async def rate_limit_middleware(self, redis_client):
		"""Create rate limiting middleware."""
		return RateLimitMiddleware(redis_client)
	
	@pytest.mark.asyncio
	async def test_rate_limit_allowed(self, rate_limit_middleware, mock_gateway_request):
		"""Test request allowed under rate limit."""
		# Mock request and handler
		request = MagicMock()
		request.get.return_value = mock_gateway_request
		
		async def mock_handler(req):
			return web.Response(text="OK")
		
		# Should allow request
		response = await rate_limit_middleware(request, mock_handler)
		assert response.text == "OK"
	
	@pytest.mark.asyncio
	async def test_rate_limit_exceeded(self, rate_limit_middleware, mock_gateway_request, redis_client):
		"""Test request blocked when rate limit exceeded."""
		# Set up rate limit data in Redis
		key = f"rate_limit:{mock_gateway_request.client_ip}:anonymous"
		
		# Add many requests to exceed limit
		import time
		current_time = int(time.time())
		for i in range(1001):  # Exceed default limit of 1000
			await redis_client.zadd(key, {f"req_{i}": current_time})
		
		# Mock request and handler
		request = MagicMock()
		request.get.return_value = mock_gateway_request
		
		async def mock_handler(req):
			return web.Response(text="OK")
		
		# Should block request
		response = await rate_limit_middleware(request, mock_handler)
		assert response.status == 429

@pytest.mark.unit
class TestAuthenticationMiddleware:
	"""Test authentication middleware."""
	
	@pytest_asyncio.fixture
	async def auth_middleware(self):
		"""Create authentication middleware."""
		consumer_service = MagicMock()
		return AuthenticationMiddleware(consumer_service)
	
	@pytest.mark.asyncio
	async def test_auth_with_valid_api_key(self, auth_middleware, mock_gateway_request):
		"""Test authentication with valid API key."""
		# Add API key to request
		mock_gateway_request.headers["X-API-Key"] = "valid_api_key"
		
		# Mock successful validation
		auth_middleware.consumer_service.validate_api_key = AsyncMock(return_value="consumer_123")
		
		# Mock request and handler
		request = MagicMock()
		request.get.return_value = mock_gateway_request
		
		async def mock_handler(req):
			return web.Response(text="OK")
		
		# Should allow request
		response = await auth_middleware(request, mock_handler)
		assert response.text == "OK"
		assert mock_gateway_request.consumer_id == "consumer_123"
	
	@pytest.mark.asyncio
	async def test_auth_with_invalid_api_key(self, auth_middleware, mock_gateway_request):
		"""Test authentication with invalid API key."""
		# Add invalid API key to request
		mock_gateway_request.headers["X-API-Key"] = "invalid_api_key"
		
		# Mock failed validation
		auth_middleware.consumer_service.validate_api_key = AsyncMock(return_value=None)
		
		# Mock request and handler
		request = MagicMock()
		request.get.return_value = mock_gateway_request
		
		async def mock_handler(req):
			return web.Response(text="OK")
		
		# Should block request
		response = await auth_middleware(request, mock_handler)
		assert response.status == 401
	
	@pytest.mark.asyncio
	async def test_auth_health_check_bypass(self, auth_middleware, mock_gateway_request):
		"""Test that health checks bypass authentication."""
		# Set path to health check
		mock_gateway_request.path = "/health"
		
		# Mock request and handler
		request = MagicMock()
		request.get.return_value = mock_gateway_request
		
		async def mock_handler(req):
			return web.Response(text="OK")
		
		# Should allow request without authentication
		response = await auth_middleware(request, mock_handler)
		assert response.text == "OK"

@pytest.mark.unit
class TestCachingMiddleware:
	"""Test caching middleware."""
	
	@pytest_asyncio.fixture
	async def caching_middleware(self, redis_client):
		"""Create caching middleware."""
		return CachingMiddleware(redis_client)
	
	@pytest.mark.asyncio
	async def test_cache_miss(self, caching_middleware, mock_gateway_request):
		"""Test cache miss scenario."""
		# Mock GET request
		mock_gateway_request.method = "GET"
		
		# Mock request and handler
		request = MagicMock()
		request.get.return_value = mock_gateway_request
		
		async def mock_handler(req):
			return web.Response(text="Fresh response", status=200)
		
		# Should call handler and cache response
		response = await caching_middleware(request, mock_handler)
		assert response.text == "Fresh response"
		assert response.status == 200
	
	@pytest.mark.asyncio
	async def test_cache_hit(self, caching_middleware, mock_gateway_request, redis_client):
		"""Test cache hit scenario."""
		# Set up cached response
		cache_key = caching_middleware._generate_cache_key(mock_gateway_request)
		cached_data = {
			"status": 200,
			"headers": {"Content-Type": "application/json"},
			"body": "Cached response"
		}
		await redis_client.setex(cache_key, 300, json.dumps(cached_data))
		
		# Mock GET request
		mock_gateway_request.method = "GET"
		
		# Mock request and handler
		request = MagicMock()
		request.get.return_value = mock_gateway_request
		
		async def mock_handler(req):
			return web.Response(text="Fresh response")
		
		# Should return cached response without calling handler
		response = await caching_middleware(request, mock_handler)
		assert response.body == b"Cached response"
		assert response.status == 200

# =============================================================================
# Gateway Router Tests
# =============================================================================

@pytest.mark.unit
class TestGatewayRouter:
	"""Test gateway router functionality."""
	
	@pytest_asyncio.fixture
	async def gateway_router(self, redis_client):
		"""Create gateway router."""
		api_service = MagicMock()
		consumer_service = MagicMock()
		policy_service = MagicMock()
		analytics_service = MagicMock()
		
		router = GatewayRouter(
			api_service, consumer_service, policy_service, 
			analytics_service, redis_client
		)
		await router.initialize()
		return router
	
	@pytest.mark.asyncio
	async def test_route_not_found(self, gateway_router, mock_gateway_request):
		"""Test routing when no matching route is found."""
		# Mock no matching API
		gateway_router.api_service.get_apis_by_tenant = AsyncMock(return_value=[])
		
		response = await gateway_router.route_request(mock_gateway_request)
		
		assert response.status_code == 404
		assert "Route not found" in response.body.decode()
	
	@pytest.mark.asyncio
	async def test_successful_routing(self, gateway_router, mock_gateway_request, mock_upstream_server):
		"""Test successful request routing."""
		# Mock matching API
		mock_api = MagicMock()
		mock_api.api_id = "test_api"
		mock_api.status = "active"
		mock_api.base_path = "/test"
		mock_api.upstream_url = "http://localhost:8000"
		mock_api.load_balancing_algorithm = "round_robin"
		mock_api.timeout_ms = 30000
		mock_api.retry_attempts = 3
		
		gateway_router.api_service.get_apis_by_tenant = AsyncMock(return_value=[mock_api])
		
		# Mock successful upstream response
		with patch('aiohttp.ClientSession') as mock_session:
			mock_response = AsyncMock()
			mock_response.status = 200
			mock_response.headers = {"Content-Type": "application/json"}
			mock_response.read = AsyncMock(return_value=b'{"result": "success"}')
			
			mock_session.return_value.__aenter__.return_value.request = AsyncMock(return_value=mock_response)
			mock_session.return_value.__aenter__.return_value.__aenter__ = AsyncMock(return_value=mock_response)
			mock_session.return_value.__aenter__.return_value.__aexit__ = AsyncMock(return_value=None)
			
			response = await gateway_router.route_request(mock_gateway_request)
			
			assert response.status_code == 200
			assert b'{"result": "success"}' == response.body

# =============================================================================
# API Gateway Integration Tests
# =============================================================================

@pytest.mark.integration
class TestAPIGatewayIntegration:
	"""Test API gateway integration."""
	
	@pytest_asyncio.fixture
	async def api_gateway(self, test_config, redis_client):
		"""Create API gateway instance."""
		gateway = APIGateway(
			host=test_config.gateway.host,
			port=test_config.gateway.port,
			redis_url=f"redis://{test_config.redis.host}:{test_config.redis.port}/{test_config.redis.database}"
		)
		
		# Initialize but don't start server
		await gateway.initialize()
		
		yield gateway
		
		if gateway.router:
			await gateway.router.shutdown()
		if gateway.redis:
			await gateway.redis.close()
	
	@pytest.mark.asyncio
	async def test_health_check_endpoint(self, api_gateway):
		"""Test gateway health check endpoint."""
		from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
		from aiohttp import web
		
		# Create test app with health check
		app = web.Application()
		app.router.add_get('/health', api_gateway._health_check)
		
		# Create test client
		from aiohttp.test_utils import make_mocked_request
		
		request = make_mocked_request('GET', '/health')
		response = await api_gateway._health_check(request)
		
		assert response.status == 200
		
		# Parse response
		response_data = json.loads(response.text)
		assert response_data['status'] in ['healthy', 'degraded']
		assert 'timestamp' in response_data
		assert 'version' in response_data

# =============================================================================
# Gateway Performance Tests
# =============================================================================

@pytest.mark.performance
class TestGatewayPerformance:
	"""Test gateway performance characteristics."""
	
	@pytest.mark.asyncio
	async def test_concurrent_request_handling(self, gateway_router, mock_upstream_server):
		"""Test handling of concurrent requests."""
		# Mock API and successful responses
		mock_api = MagicMock()
		mock_api.api_id = "perf_api"
		mock_api.status = "active"
		mock_api.base_path = "/perf"
		mock_api.upstream_url = "http://localhost:8000"
		mock_api.load_balancing_algorithm = "round_robin"
		mock_api.timeout_ms = 30000
		mock_api.retry_attempts = 3
		
		gateway_router.api_service.get_apis_by_tenant = AsyncMock(return_value=[mock_api])
		
		with patch('aiohttp.ClientSession') as mock_session:
			mock_response = AsyncMock()
			mock_response.status = 200
			mock_response.headers = {"Content-Type": "application/json"}
			mock_response.read = AsyncMock(return_value=b'{"result": "success"}')
			
			mock_session.return_value.__aenter__.return_value.request = AsyncMock(return_value=mock_response)
			mock_session.return_value.__aenter__.return_value.__aenter__ = AsyncMock(return_value=mock_response)
			mock_session.return_value.__aenter__.return_value.__aexit__ = AsyncMock(return_value=None)
			
			# Create multiple concurrent requests
			import time
			from ..gateway import GatewayRequest
			
			requests = []
			for i in range(50):
				req = GatewayRequest(
					request_id=f"perf_req_{i}",
					method="GET",
					path="/perf/test",
					headers={},
					query_params={},
					body=b"",
					client_ip="127.0.0.1",
					user_agent="Performance Test",
					timestamp=datetime.now(timezone.utc),
					tenant_id="perf_tenant"
				)
				requests.append(req)
			
			# Execute requests concurrently
			start_time = time.time()
			
			tasks = [gateway_router.route_request(req) for req in requests]
			responses = await asyncio.gather(*tasks)
			
			end_time = time.time()
			duration = end_time - start_time
			
			# Verify all requests succeeded
			successful_responses = [r for r in responses if r.status_code == 200]
			assert len(successful_responses) == 50
			
			# Check performance
			throughput = len(responses) / duration
			assert throughput > 10  # At least 10 requests per second
			
			print(f"Processed {len(responses)} requests in {duration:.2f}s (throughput: {throughput:.1f} req/s)")
	
	@pytest.mark.asyncio
	async def test_load_balancer_performance(self):
		"""Test load balancer performance under load."""
		load_balancer = LoadBalancer()
		
		servers = [
			UpstreamServer(url=f"http://server{i}.local")
			for i in range(10)
		]
		
		import time
		
		start_time = time.time()
		
		# Perform many load balancing decisions
		tasks = []
		for _ in range(1000):
			task = load_balancer.select_server(servers, LoadBalancingStrategy.ROUND_ROBIN)
			tasks.append(task)
		
		selected_servers = await asyncio.gather(*tasks)
		
		end_time = time.time()
		duration = end_time - start_time
		
		# Verify all selections were made
		assert len(selected_servers) == 1000
		assert all(server is not None for server in selected_servers)
		
		# Check performance
		selections_per_second = len(selected_servers) / duration
		assert selections_per_second > 1000  # At least 1000 selections per second
		
		print(f"Load balancer: {selections_per_second:.0f} selections/second")

# =============================================================================
# Gateway Error Handling Tests
# =============================================================================

@pytest.mark.unit
class TestGatewayErrorHandling:
	"""Test gateway error handling scenarios."""
	
	@pytest.mark.asyncio
	async def test_upstream_timeout(self, gateway_router, mock_gateway_request):
		"""Test handling of upstream timeouts."""
		# Mock API
		mock_api = MagicMock()
		mock_api.api_id = "timeout_api"
		mock_api.status = "active"
		mock_api.base_path = "/test"
		mock_api.upstream_url = "http://slow-server.local"
		mock_api.timeout_ms = 1000  # 1 second timeout
		
		gateway_router.api_service.get_apis_by_tenant = AsyncMock(return_value=[mock_api])
		
		# Mock timeout
		with patch('aiohttp.ClientSession') as mock_session:
			mock_session.return_value.__aenter__.return_value.request = AsyncMock(
				side_effect=asyncio.TimeoutError()
			)
			
			response = await gateway_router.route_request(mock_gateway_request)
			
			assert response.status_code == 504
			assert "Gateway Timeout" in response.body.decode()
	
	@pytest.mark.asyncio
	async def test_upstream_connection_error(self, gateway_router, mock_gateway_request):
		"""Test handling of upstream connection errors."""
		# Mock API
		mock_api = MagicMock()
		mock_api.api_id = "error_api"
		mock_api.status = "active"
		mock_api.base_path = "/test"
		mock_api.upstream_url = "http://unreachable-server.local"
		
		gateway_router.api_service.get_apis_by_tenant = AsyncMock(return_value=[mock_api])
		
		# Mock connection error
		with patch('aiohttp.ClientSession') as mock_session:
			mock_session.return_value.__aenter__.return_value.request = AsyncMock(
				side_effect=ClientError("Connection failed")
			)
			
			response = await gateway_router.route_request(mock_gateway_request)
			
			assert response.status_code == 502
			assert "Bad Gateway" in response.body.decode()
	
	@pytest.mark.asyncio
	async def test_no_healthy_upstream_servers(self, gateway_router, mock_gateway_request):
		"""Test handling when no upstream servers are healthy."""
		# Mock API with unhealthy servers
		mock_api = MagicMock()
		mock_api.api_id = "unhealthy_api"
		mock_api.status = "active"
		mock_api.base_path = "/test"
		mock_api.upstream_url = "http://unhealthy-server.local"
		
		gateway_router.api_service.get_apis_by_tenant = AsyncMock(return_value=[mock_api])
		
		# Mock load balancer returning None (no healthy servers)
		gateway_router.load_balancer.select_server = AsyncMock(return_value=None)
		
		response = await gateway_router.route_request(mock_gateway_request)
		
		assert response.status_code == 503
		assert "No healthy upstream servers" in response.body.decode()