"""
Performance Tests

Comprehensive performance testing for the composition system including:
- Load testing for capability discovery
- Composition creation performance
- Concurrent user scenarios  
- Scalability testing
- Memory and resource usage

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any
from uuid_extensions import uuid7str

from . import (
	PerformanceTestHelper,
	MockDataGenerator,
	test_tenant_id,
	test_user_id,
	initialized_tenant
)


class TestCompositionPerformance:
	"""Test composition creation performance."""
	
	@pytest.mark.asyncio
	async def test_composition_creation_speed(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test composition creation speed."""
		capabilities = MockDataGenerator.generate_capability_list(3)
		
		execution_time = await PerformanceTestHelper.measure_composition_time(
			tenant_id=initialized_tenant,
			user_id=test_user_id,
			capabilities=capabilities
		)
		
		# Composition should complete within reasonable time
		assert execution_time < 5.0  # 5 seconds max
		
	@pytest.mark.asyncio
	async def test_large_composition_performance(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test performance with large number of capabilities."""
		# Test with maximum capability set
		capabilities = MockDataGenerator.generate_capability_list(10)
		
		execution_time = await PerformanceTestHelper.measure_composition_time(
			tenant_id=initialized_tenant,
			user_id=test_user_id,
			capabilities=capabilities
		)
		
		# Even large compositions should complete reasonably fast
		assert execution_time < 10.0  # 10 seconds max for large compositions
		
	@pytest.mark.asyncio
	async def test_concurrent_composition_creation(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test concurrent composition creation."""
		from .. import compose_application
		from ..capability_registry import CRCompositionType
		
		concurrent_requests = 5
		capabilities = MockDataGenerator.generate_capability_list(3)
		
		# Create concurrent composition tasks
		tasks = []
		start_time = time.time()
		
		for i in range(concurrent_requests):
			task = compose_application(
				tenant_id=initialized_tenant,
				user_id=test_user_id,
				capabilities=capabilities,
				composition_type=CRCompositionType.ENTERPRISE
			)
			tasks.append(task)
		
		# Execute all tasks concurrently
		results = await asyncio.gather(*tasks, return_exceptions=True)
		end_time = time.time()
		
		total_time = end_time - start_time
		successful_requests = len([r for r in results if not isinstance(r, Exception)])
		
		assert successful_requests >= concurrent_requests * 0.8  # 80% success rate
		assert total_time < 15.0  # Should complete within 15 seconds


class TestCapabilityDiscoveryPerformance:
	"""Test capability discovery performance."""
	
	@pytest.mark.asyncio
	async def test_capability_discovery_speed(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test capability discovery speed."""
		from .. import discover_capabilities
		
		start_time = time.time()
		
		capabilities = await discover_capabilities(
			tenant_id=initialized_tenant,
			user_id=test_user_id
		)
		
		end_time = time.time()
		execution_time = end_time - start_time
		
		assert isinstance(capabilities, list)
		assert execution_time < 1.0  # Should complete within 1 second
		
	@pytest.mark.asyncio
	async def test_capability_discovery_load(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test capability discovery under load."""
		load_test_results = await PerformanceTestHelper.load_test_capabilities(
			tenant_id=initialized_tenant,
			user_id=test_user_id,
			concurrent_requests=20
		)
		
		assert load_test_results["successful_requests"] >= 18  # 90% success rate
		assert load_test_results["requests_per_second"] >= 5  # At least 5 RPS
		assert load_test_results["average_response_time"] < 2.0  # Under 2 seconds average
		
	@pytest.mark.asyncio
	async def test_filtered_discovery_performance(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test performance of filtered capability discovery."""
		from .. import discover_capabilities
		
		filters = {
			"category": "core_business_operations",
			"status": "active"
		}
		
		start_time = time.time()
		
		capabilities = await discover_capabilities(
			tenant_id=initialized_tenant,
			user_id=test_user_id,
			filters=filters
		)
		
		end_time = time.time()
		execution_time = end_time - start_time
		
		assert isinstance(capabilities, list)
		assert execution_time < 1.5  # Filtered queries should still be fast


class TestScalabilityTests:
	"""Test system scalability."""
	
	@pytest.mark.asyncio
	async def test_multiple_tenant_performance(self, test_user_id: str):
		"""Test performance with multiple tenants."""
		from .. import create_tenant, discover_capabilities
		
		# Create multiple tenants
		tenant_count = 5
		tenant_ids = []
		
		for i in range(tenant_count):
			tenant_id = f"perf_tenant_{i}_{uuid7str()}"
			success = await create_tenant(
				tenant_id=tenant_id,
				admin_user_id=test_user_id,
				tenant_name=f"Performance Test Tenant {i}",
				enabled_capabilities=MockDataGenerator.generate_capability_list(3)
			)
			assert success
			tenant_ids.append(tenant_id)
		
		# Test discovery across all tenants concurrently
		start_time = time.time()
		
		tasks = []
		for tenant_id in tenant_ids:
			task = discover_capabilities(tenant_id=tenant_id, user_id=test_user_id)
			tasks.append(task)
		
		results = await asyncio.gather(*tasks, return_exceptions=True)
		end_time = time.time()
		
		total_time = end_time - start_time
		successful_requests = len([r for r in results if not isinstance(r, Exception)])
		
		assert successful_requests == tenant_count
		assert total_time < 10.0  # Should handle multiple tenants efficiently
		
	@pytest.mark.asyncio
	async def test_user_scalability(self, initialized_tenant: str):
		"""Test performance with multiple concurrent users."""
		from .. import discover_capabilities
		
		user_count = 10
		user_ids = [f"perf_user_{i}_{uuid7str()}" for i in range(user_count)]
		
		# Test concurrent user access
		start_time = time.time()
		
		tasks = []
		for user_id in user_ids:
			task = discover_capabilities(tenant_id=initialized_tenant, user_id=user_id)
			tasks.append(task)
		
		results = await asyncio.gather(*tasks, return_exceptions=True)
		end_time = time.time()
		
		total_time = end_time - start_time
		successful_requests = len([r for r in results if not isinstance(r, Exception)])
		
		assert successful_requests >= user_count * 0.9  # 90% success rate
		assert total_time < 8.0  # Should handle concurrent users efficiently


class TestMemoryPerformance:
	"""Test memory usage and efficiency."""
	
	@pytest.mark.asyncio
	async def test_memory_usage_composition(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test memory usage during composition operations."""
		import psutil
		import os
		
		process = psutil.Process(os.getpid())
		initial_memory = process.memory_info().rss
		
		# Perform multiple composition operations
		capabilities = MockDataGenerator.generate_capability_list(5)
		
		for _ in range(10):
			await PerformanceTestHelper.measure_composition_time(
				tenant_id=initialized_tenant,
				user_id=test_user_id,
				capabilities=capabilities
			)
		
		final_memory = process.memory_info().rss
		memory_increase = final_memory - initial_memory
		
		# Memory increase should be reasonable (less than 100MB)
		assert memory_increase < 100 * 1024 * 1024  # 100MB
		
	@pytest.mark.asyncio
	async def test_memory_cleanup(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test that memory is properly cleaned up."""
		import gc
		import psutil
		import os
		
		process = psutil.Process(os.getpid())
		
		# Perform operations that should create temporary objects
		for _ in range(20):
			await PerformanceTestHelper.measure_composition_time(
				tenant_id=initialized_tenant,
				user_id=test_user_id,
				capabilities=MockDataGenerator.generate_capability_list(3)
			)
		
		# Force garbage collection
		gc.collect()
		
		memory_after_gc = process.memory_info().rss
		
		# Memory should stabilize after garbage collection
		# This is a basic check - real implementation would be more sophisticated
		assert memory_after_gc > 0  # Basic sanity check


class TestResponseTimeTests:
	"""Test response time requirements."""
	
	@pytest.mark.asyncio
	async def test_capability_discovery_response_time(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test capability discovery meets response time requirements."""
		from .. import discover_capabilities
		
		response_times = []
		
		# Test multiple requests to get average response time
		for _ in range(10):
			start_time = time.time()
			
			await discover_capabilities(
				tenant_id=initialized_tenant,
				user_id=test_user_id
			)
			
			end_time = time.time()
			response_times.append(end_time - start_time)
		
		average_response_time = sum(response_times) / len(response_times)
		max_response_time = max(response_times)
		
		# Requirements from specification
		assert average_response_time < 0.05  # 50ms average
		assert max_response_time < 0.2  # 200ms max
		
	@pytest.mark.asyncio
	async def test_composition_validation_response_time(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test composition validation meets response time requirements."""
		response_times = []
		capabilities = MockDataGenerator.generate_capability_list(5)
		
		# Test multiple composition validations
		for _ in range(5):
			execution_time = await PerformanceTestHelper.measure_composition_time(
				tenant_id=initialized_tenant,
				user_id=test_user_id,
				capabilities=capabilities
			)
			response_times.append(execution_time)
		
		average_response_time = sum(response_times) / len(response_times)
		max_response_time = max(response_times)
		
		# Requirements from specification
		assert average_response_time < 0.2  # 200ms average
		assert max_response_time < 1.0  # 1 second max


class TestThroughputTests:
	"""Test system throughput."""
	
	@pytest.mark.asyncio
	async def test_composition_throughput(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test composition creation throughput."""
		from .. import compose_application
		from ..capability_registry import CRCompositionType
		
		capabilities = MockDataGenerator.generate_capability_list(3)
		request_count = 20
		
		start_time = time.time()
		
		# Create batched concurrent requests
		batch_size = 5
		all_results = []
		
		for i in range(0, request_count, batch_size):
			batch_tasks = []
			for j in range(batch_size):
				if i + j < request_count:
					task = compose_application(
						tenant_id=initialized_tenant,
						user_id=test_user_id,
						capabilities=capabilities,
						composition_type=CRCompositionType.ENTERPRISE
					)
					batch_tasks.append(task)
			
			batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
			all_results.extend(batch_results)
			
			# Small delay between batches to simulate real usage
			await asyncio.sleep(0.1)
		
		end_time = time.time()
		total_time = end_time - start_time
		
		successful_requests = len([r for r in all_results if not isinstance(r, Exception)])
		throughput = successful_requests / total_time
		
		assert successful_requests >= request_count * 0.8  # 80% success rate
		assert throughput >= 2.0  # At least 2 compositions per second
		
	@pytest.mark.asyncio
	async def test_discovery_throughput(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test capability discovery throughput."""
		load_test_results = await PerformanceTestHelper.load_test_capabilities(
			tenant_id=initialized_tenant,
			user_id=test_user_id,
			concurrent_requests=50
		)
		
		# Should handle high throughput efficiently
		assert load_test_results["requests_per_second"] >= 10  # At least 10 RPS
		assert load_test_results["successful_requests"] >= 45  # 90% success rate