#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Performance Testing
Performance benchmarks and load testing for MDM operations

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from uuid_extensions import uuid7str
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...service import MDMService
from ...models import MdEntityCreate, EntityType, EntityStatus


class TestMDMPerformanceBenchmarks:
	"""Performance benchmark tests for MDM operations"""
	
	@pytest.mark.performance
	async def test_entity_creation_performance(self, test_mdm_service: MDMService,
	                                         test_tenant_id: str, test_user_id: str,
	                                         performance_benchmarks: Dict[str, float]):
		"""Test entity creation performance benchmark"""
		entity_data = MdEntityCreate(
			tenant_id=test_tenant_id,
			entity_type=EntityType.PERSON,
			entity_name="Performance Test Entity",
			business_key=f"PERF-{uuid7str()[:8]}",
			source_system="performance_test",
			status=EntityStatus.ACTIVE,
			attributes={
				"first_name": "John",
				"last_name": "Doe",
				"email": "john.doe@example.com",
				"phone": "+1-555-123-4567",
				"address": {
					"street": "123 Main St",
					"city": "Anytown",
					"state": "NY",
					"zip": "12345"
				},
				"metadata": {
					"source_confidence": 0.95,
					"last_verified": datetime.utcnow().isoformat()
				}
			},
			tags=["performance", "test", "benchmark"],
			data_classification="internal"
		)
		
		context = test_mdm_service.create_operation_context(
			tenant_id=test_tenant_id,
			user_id=test_user_id,
			operation_type="create_entity",
			source_system="performance_test"
		)
		
		# Benchmark single entity creation
		start_time = time.perf_counter()
		result = await test_mdm_service.entity_service.create_entity(entity_data, context)
		end_time = time.perf_counter()
		
		creation_time_ms = (end_time - start_time) * 1000
		
		assert result["status"] == "success"
		assert creation_time_ms <= performance_benchmarks["entity_creation_max_ms"]
		
		print(f"Entity creation time: {creation_time_ms:.2f}ms (limit: {performance_benchmarks['entity_creation_max_ms']}ms)")
	
	@pytest.mark.performance
	async def test_entity_retrieval_performance(self, test_mdm_service: MDMService,
	                                          created_test_entity: Dict[str, Any],
	                                          test_tenant_id: str,
	                                          performance_benchmarks: Dict[str, float]):
		"""Test entity retrieval performance benchmark"""
		entity_id = created_test_entity["entity_id"]
		
		# Benchmark entity retrieval
		start_time = time.perf_counter()
		result = await test_mdm_service.entity_service.get_entity(entity_id, test_tenant_id)
		end_time = time.perf_counter()
		
		retrieval_time_ms = (end_time - start_time) * 1000
		
		assert result["status"] == "success"
		assert retrieval_time_ms <= performance_benchmarks["entity_retrieval_max_ms"]
		
		print(f"Entity retrieval time: {retrieval_time_ms:.2f}ms (limit: {performance_benchmarks['entity_retrieval_max_ms']}ms)")
	
	@pytest.mark.performance
	async def test_entity_search_performance(self, test_mdm_service: MDMService,
	                                       multiple_test_entities: List[Dict[str, Any]],
	                                       test_tenant_id: str):
		"""Test entity search performance with various criteria"""
		search_scenarios = [
			# Basic type search
			{"entity_type": EntityType.PERSON, "limit": 10, "offset": 0},
			
			# Name pattern search
			{"entity_name": "Person", "limit": 10, "offset": 0},
			
			# Complex multi-criteria search
			{
				"entity_type": EntityType.PERSON,
				"source_system": "test_system",
				"status": EntityStatus.ACTIVE,
				"min_quality_score": 75.0,
				"limit": 20,
				"offset": 0,
				"sort_by": "quality_score",
				"sort_order": "desc"
			}
		]
		
		for i, search_criteria in enumerate(search_scenarios):
			start_time = time.perf_counter()
			result = await test_mdm_service.entity_service.search_entities(
				test_tenant_id, search_criteria
			)
			end_time = time.perf_counter()
			
			search_time_ms = (end_time - start_time) * 1000
			
			assert result["status"] == "success"
			assert search_time_ms <= 500.0  # 500ms limit for searches
			
			print(f"Search scenario {i+1} time: {search_time_ms:.2f}ms")
	
	@pytest.mark.performance
	async def test_quality_assessment_performance(self, test_mdm_service: MDMService,
	                                            created_test_entity: Dict[str, Any],
	                                            test_tenant_id: str,
	                                            performance_benchmarks: Dict[str, float]):
		"""Test quality assessment performance benchmark"""
		entity_id = created_test_entity["entity_id"]
		entity_attributes = created_test_entity["attributes"]
		entity_type = created_test_entity["entity_type"]
		
		# Benchmark quality assessment
		start_time = time.perf_counter()
		result = await test_mdm_service.quality_service.assess_quality(
			entity_id, test_tenant_id, entity_attributes, entity_type
		)
		end_time = time.perf_counter()
		
		assessment_time_ms = (end_time - start_time) * 1000
		
		assert result["status"] == "success"
		assert assessment_time_ms <= performance_benchmarks["quality_assessment_max_ms"]
		
		print(f"Quality assessment time: {assessment_time_ms:.2f}ms (limit: {performance_benchmarks['quality_assessment_max_ms']}ms)")
	
	@pytest.mark.performance
	async def test_duplicate_detection_performance(self, test_mdm_service: MDMService,
	                                             created_test_entity: Dict[str, Any],
	                                             test_tenant_id: str,
	                                             performance_benchmarks: Dict[str, float]):
		"""Test duplicate detection performance benchmark"""
		entity_id = created_test_entity["entity_id"]
		entity_data = created_test_entity
		
		# Benchmark duplicate detection
		start_time = time.perf_counter()
		result = await test_mdm_service.matching_service.find_duplicates(
			entity_id, test_tenant_id, entity_data
		)
		end_time = time.perf_counter()
		
		detection_time_ms = (end_time - start_time) * 1000
		
		assert result["status"] == "success"
		assert detection_time_ms <= performance_benchmarks["duplicate_detection_max_ms"]
		
		print(f"Duplicate detection time: {detection_time_ms:.2f}ms (limit: {performance_benchmarks['duplicate_detection_max_ms']}ms)")
	
	@pytest.mark.performance
	async def test_batch_operation_performance(self, test_mdm_service: MDMService,
	                                         test_tenant_id: str, test_user_id: str,
	                                         performance_benchmarks: Dict[str, float]):
		"""Test batch operation performance benchmark"""
		batch_size = 100
		entities_data = []
		
		# Generate batch entities
		for i in range(batch_size):
			entity_data = MdEntityCreate(
				tenant_id=test_tenant_id,
				entity_type=EntityType.PRODUCT,
				entity_name=f"Batch Product {i:03d}",
				business_key=f"BATCH-{i:04d}",
				source_system="batch_performance_test",
				status=EntityStatus.ACTIVE,
				attributes={
					"sku": f"SKU-{i:04d}",
					"category": "Performance Test",
					"price": 99.99 + i,
					"batch_index": i
				},
				tags=["batch", "performance"],
				data_classification="public"
			)
			entities_data.append(entity_data)
		
		context = test_mdm_service.create_operation_context(
			tenant_id=test_tenant_id,
			user_id=test_user_id,
			operation_type="batch_create",
			source_system="batch_performance_test"
		)
		
		# Benchmark batch creation
		start_time = time.perf_counter()
		result = await test_mdm_service.entity_service.batch_create_entities(
			entities_data, context
		)
		end_time = time.perf_counter()
		
		batch_time_seconds = end_time - start_time
		processing_rate = batch_size / batch_time_seconds
		
		assert result["status"] == "success"
		assert result["successful"] == batch_size
		assert processing_rate >= performance_benchmarks["batch_operation_max_per_second"]
		
		print(f"Batch processing rate: {processing_rate:.1f} entities/second (minimum: {performance_benchmarks['batch_operation_max_per_second']})")
	
	@pytest.mark.performance
	async def test_concurrent_operations_performance(self, test_mdm_service: MDMService,
	                                               test_tenant_id: str, test_user_id: str):
		"""Test concurrent operations performance"""
		concurrent_tasks = 10
		
		async def create_entity_task(task_id: int):
			entity_data = MdEntityCreate(
				tenant_id=test_tenant_id,
				entity_type=EntityType.PERSON,
				entity_name=f"Concurrent Entity {task_id}",
				business_key=f"CONC-{task_id:03d}",
				source_system="concurrent_test",
				status=EntityStatus.ACTIVE,
				attributes={"task_id": task_id},
				data_classification="internal"
			)
			
			context = test_mdm_service.create_operation_context(
				tenant_id=test_tenant_id,
				user_id=f"{test_user_id}-{task_id}",
				operation_type="create_entity",
				source_system="concurrent_test"
			)
			
			start_time = time.perf_counter()
			result = await test_mdm_service.entity_service.create_entity(entity_data, context)
			end_time = time.perf_counter()
			
			return {
				"task_id": task_id,
				"result": result,
				"duration_ms": (end_time - start_time) * 1000
			}
		
		# Execute concurrent tasks
		start_time = time.perf_counter()
		tasks = [create_entity_task(i) for i in range(concurrent_tasks)]
		results = await asyncio.gather(*tasks)
		end_time = time.perf_counter()
		
		total_time_seconds = end_time - start_time
		concurrent_throughput = concurrent_tasks / total_time_seconds
		
		# Verify all tasks succeeded
		successful_tasks = [r for r in results if r["result"]["status"] == "success"]
		assert len(successful_tasks) == concurrent_tasks
		
		# Performance assertions
		max_task_duration = max(r["duration_ms"] for r in results)
		avg_task_duration = sum(r["duration_ms"] for r in results) / len(results)
		
		print(f"Concurrent throughput: {concurrent_throughput:.1f} operations/second")
		print(f"Max task duration: {max_task_duration:.2f}ms")
		print(f"Avg task duration: {avg_task_duration:.2f}ms")
		
		assert concurrent_throughput >= 5.0  # Minimum 5 ops/sec concurrent
		assert max_task_duration <= 2000.0   # Max 2 seconds per task


class TestMDMLoadTesting:
	"""Load testing for MDM operations"""
	
	@pytest.mark.slow
	@pytest.mark.performance
	async def test_sustained_load_entity_operations(self, test_mdm_service: MDMService,
	                                               test_tenant_id: str, test_user_id: str):
		"""Test sustained load with mixed entity operations"""
		duration_seconds = 30
		target_ops_per_second = 10
		
		operations_completed = 0
		errors_encountered = 0
		start_time = time.perf_counter()
		
		async def mixed_operation(op_id: int):
			nonlocal operations_completed, errors_encountered
			
			try:
				if op_id % 4 == 0:  # 25% creates
					entity_data = MdEntityCreate(
						tenant_id=test_tenant_id,
						entity_type=EntityType.PERSON,
						entity_name=f"Load Test Entity {op_id}",
						business_key=f"LOAD-{op_id:05d}",
						source_system="load_test",
						status=EntityStatus.ACTIVE,
						attributes={"load_test_id": op_id},
						data_classification="internal"
					)
					
					context = test_mdm_service.create_operation_context(
						tenant_id=test_tenant_id,
						user_id=test_user_id,
						operation_type="create_entity",
						source_system="load_test"
					)
					
					result = await test_mdm_service.entity_service.create_entity(entity_data, context)
					if result["status"] == "success":
						operations_completed += 1
					else:
						errors_encountered += 1
				
				elif op_id % 4 == 1:  # 25% searches
					search_criteria = {
						"entity_type": EntityType.PERSON,
						"source_system": "load_test",
						"limit": 10,
						"offset": 0
					}
					
					result = await test_mdm_service.entity_service.search_entities(
						test_tenant_id, search_criteria
					)
					if result["status"] == "success":
						operations_completed += 1
					else:
						errors_encountered += 1
				
				else:  # 50% health checks (lightweight operations)
					result = await test_mdm_service.health_check()
					if result["status"] == "healthy":
						operations_completed += 1
					else:
						errors_encountered += 1
			
			except Exception as e:
				errors_encountered += 1
				print(f"Load test operation {op_id} failed: {e}")
		
		# Generate load for specified duration
		operation_id = 0
		
		while time.perf_counter() - start_time < duration_seconds:
			# Create batch of concurrent operations
			batch_size = min(5, target_ops_per_second)
			tasks = []
			
			for _ in range(batch_size):
				tasks.append(mixed_operation(operation_id))
				operation_id += 1
			
			await asyncio.gather(*tasks, return_exceptions=True)
			
			# Control rate
			await asyncio.sleep(1.0 / target_ops_per_second)
		
		end_time = time.perf_counter()
		actual_duration = end_time - start_time
		actual_ops_per_second = operations_completed / actual_duration
		error_rate = errors_encountered / (operations_completed + errors_encountered) if (operations_completed + errors_encountered) > 0 else 0
		
		print(f"Load test results:")
		print(f"  Duration: {actual_duration:.1f}s")
		print(f"  Operations completed: {operations_completed}")
		print(f"  Errors encountered: {errors_encountered}")
		print(f"  Actual ops/sec: {actual_ops_per_second:.1f}")
		print(f"  Error rate: {error_rate:.1%}")
		
		# Performance assertions
		assert operations_completed > 0
		assert error_rate <= 0.05  # Max 5% error rate
		assert actual_ops_per_second >= target_ops_per_second * 0.8  # 80% of target
	
	@pytest.mark.slow
	@pytest.mark.performance
	async def test_memory_usage_under_load(self, test_mdm_service: MDMService,
	                                     test_tenant_id: str, test_user_id: str):
		"""Test memory usage patterns under load"""
		import psutil
		import gc
		
		# Get initial memory usage
		process = psutil.Process()
		initial_memory_mb = process.memory_info().rss / 1024 / 1024
		
		# Create entities to build up memory usage
		batch_size = 50
		num_batches = 10
		
		for batch_num in range(num_batches):
			entities_data = []
			
			for i in range(batch_size):
				entity_data = MdEntityCreate(
					tenant_id=test_tenant_id,
					entity_type=EntityType.PRODUCT,
					entity_name=f"Memory Test Product {batch_num}-{i:03d}",
					business_key=f"MEM-{batch_num:02d}-{i:03d}",
					source_system="memory_test",
					status=EntityStatus.ACTIVE,
					attributes={
						"batch": batch_num,
						"index": i,
						"large_text_field": "x" * 1000,  # 1KB text field
						"metadata": {
							"created_batch": batch_num,
							"test_data": list(range(100))  # Some structured data
						}
					},
					data_classification="internal"
				)
				entities_data.append(entity_data)
			
			context = test_mdm_service.create_operation_context(
				tenant_id=test_tenant_id,
				user_id=test_user_id,
				operation_type="batch_create",
				source_system="memory_test"
			)
			
			result = await test_mdm_service.entity_service.batch_create_entities(
				entities_data, context
			)
			assert result["status"] == "success"
			
			# Check memory after each batch
			current_memory_mb = process.memory_info().rss / 1024 / 1024
			memory_growth_mb = current_memory_mb - initial_memory_mb
			
			print(f"Batch {batch_num + 1}/{num_batches}: Memory usage: {current_memory_mb:.1f}MB (+{memory_growth_mb:.1f}MB)")
			
			# Force garbage collection
			gc.collect()
		
		# Final memory check
		final_memory_mb = process.memory_info().rss / 1024 / 1024
		total_memory_growth_mb = final_memory_mb - initial_memory_mb
		
		print(f"Memory usage summary:")
		print(f"  Initial: {initial_memory_mb:.1f}MB")
		print(f"  Final: {final_memory_mb:.1f}MB")
		print(f"  Growth: {total_memory_growth_mb:.1f}MB")
		
		# Memory growth should be reasonable (less than 500MB for this test)
		assert total_memory_growth_mb <= 500.0
		
		# Memory per entity should be reasonable
		total_entities = batch_size * num_batches
		memory_per_entity_kb = (total_memory_growth_mb * 1024) / total_entities
		print(f"  Memory per entity: {memory_per_entity_kb:.1f}KB")
		
		assert memory_per_entity_kb <= 50.0  # Max 50KB per entity


class TestMDMScalabilityLimits:
	"""Test scalability limits and boundaries"""
	
	@pytest.mark.slow
	@pytest.mark.performance
	async def test_large_batch_processing_limits(self, test_mdm_service: MDMService,
	                                           test_tenant_id: str, test_user_id: str):
		"""Test processing limits for large batches"""
		batch_sizes = [100, 500, 1000]
		
		for batch_size in batch_sizes:
			entities_data = []
			
			for i in range(batch_size):
				entity_data = MdEntityCreate(
					tenant_id=test_tenant_id,
					entity_type=EntityType.ASSET,
					entity_name=f"Scalability Test Asset {i:04d}",
					business_key=f"SCALE-{batch_size}-{i:04d}",
					source_system="scalability_test",
					status=EntityStatus.ACTIVE,
					attributes={
						"batch_size": batch_size,
						"index": i,
						"asset_type": "test_asset"
					},
					data_classification="public"
				)
				entities_data.append(entity_data)
			
			context = test_mdm_service.create_operation_context(
				tenant_id=test_tenant_id,
				user_id=test_user_id,
				operation_type="batch_create",
				source_system="scalability_test"
			)
			
			# Time the batch processing
			start_time = time.perf_counter()
			result = await test_mdm_service.entity_service.batch_create_entities(
				entities_data, context
			)
			end_time = time.perf_counter()
			
			processing_time_seconds = end_time - start_time
			processing_rate = batch_size / processing_time_seconds
			
			assert result["status"] == "success"
			assert result["successful"] == batch_size
			
			print(f"Batch size {batch_size}: {processing_time_seconds:.1f}s ({processing_rate:.1f} entities/sec)")
			
			# Scalability assertions
			assert processing_time_seconds <= 60.0  # Max 1 minute per batch
			assert processing_rate >= 10.0  # Minimum 10 entities/sec
	
	@pytest.mark.performance
	async def test_search_result_pagination_performance(self, test_mdm_service: MDMService,
	                                                   test_tenant_id: str):
		"""Test search performance with large result sets and pagination"""
		# This test assumes we have many entities created from previous tests
		page_sizes = [10, 50, 100, 500]
		
		for page_size in page_sizes:
			search_criteria = {
				"limit": page_size,
				"offset": 0,
				"sort_by": "created_at",
				"sort_order": "desc"
			}
			
			start_time = time.perf_counter()
			result = await test_mdm_service.entity_service.search_entities(
				test_tenant_id, search_criteria
			)
			end_time = time.perf_counter()
			
			search_time_ms = (end_time - start_time) * 1000
			
			assert result["status"] == "success"
			
			entities_returned = len(result["entities"])
			time_per_entity_ms = search_time_ms / entities_returned if entities_returned > 0 else 0
			
			print(f"Page size {page_size}: {search_time_ms:.1f}ms ({time_per_entity_ms:.2f}ms per entity)")
			
			# Performance should scale reasonably with page size
			assert search_time_ms <= 2000.0  # Max 2 seconds for any page size
			assert time_per_entity_ms <= 10.0  # Max 10ms per entity