"""
Performance Tests for AICR Capability
======================================

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>

Comprehensive performance tests for the AI Core Framework capability
covering throughput, latency, scalability, resource utilization, and
stress testing with detailed performance metrics and benchmarking.
"""

import pytest
import asyncio
import time
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from ..service import AICoreService
from ..monitoring import ai_monitoring_system, MetricsCollector
from ..ml_pipeline import ml_pipeline_framework
from ..models import (
	AICRModel,
	AICRInferenceRequest,
	AICRInferenceResponse,
	ModelType,
	InferenceStatus
)


@pytest.mark.performance
class TestInferencePerformance:
	"""Performance tests for inference operations."""

	@pytest.fixture
	async def performance_service(self):
		"""Create AI service optimized for performance testing."""
		service = AICoreService()

		with patch.object(service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(service, '_start_background_tasks', new_callable=AsyncMock):

			await service.initialize()

		# Setup high-performance mock engine
		mock_engine = Mock()
		mock_engine.deploy_model = AsyncMock(return_value={"success": True})
		mock_engine.undeploy_model = AsyncMock(return_value={"success": True})

		# Fast inference mock (10-100ms range)
		async def fast_inference(*args, **kwargs):
			await asyncio.sleep(0.01 + np.random.exponential(0.02))  # 10-50ms typical
			return {
				"predictions": {"class": "test", "confidence": 0.9},
				"processing_time_ms": 10 + np.random.exponential(20)
			}

		mock_engine.run_inference = fast_inference
		service.inference_engines["pytorch"] = mock_engine

		# Register and deploy test model
		model_data = {
			"name": "performance_test_model",
			"description": "Model for performance testing",
			"model_type": "classification",
			"framework": "pytorch"
		}

		model = await service.register_model(model_data)
		await service.deploy_model(model.model_id)

		return service, model

	@pytest.mark.asyncio
	async def test_single_inference_latency(self, performance_service):
		"""Test single inference request latency."""
		service, model = performance_service

		# Warm up
		for _ in range(5):
			request = AICRInferenceRequest(
				model_id=model.model_id,
				input_data={"data": [1, 2, 3]}
			)
			await service.run_inference(request)

		# Measure latency
		latencies = []
		for i in range(100):
			request = AICRInferenceRequest(
				model_id=model.model_id,
				input_data={"data": [i, i+1, i+2]}
			)

			start_time = time.perf_counter()
			response = await service.run_inference(request)
			end_time = time.perf_counter()

			latency_ms = (end_time - start_time) * 1000
			latencies.append(latency_ms)

			assert response.status == InferenceStatus.COMPLETED

		# Analyze latency statistics
		avg_latency = statistics.mean(latencies)
		p50_latency = statistics.median(latencies)
		p95_latency = np.percentile(latencies, 95)
		p99_latency = np.percentile(latencies, 99)

		print(f"\nInference Latency Statistics:")
		print(f"Average: {avg_latency:.2f}ms")
		print(f"P50: {p50_latency:.2f}ms")
		print(f"P95: {p95_latency:.2f}ms")
		print(f"P99: {p99_latency:.2f}ms")

		# Performance assertions
		assert avg_latency < 100, f"Average latency too high: {avg_latency:.2f}ms"
		assert p95_latency < 200, f"P95 latency too high: {p95_latency:.2f}ms"
		assert p99_latency < 300, f"P99 latency too high: {p99_latency:.2f}ms"

	@pytest.mark.asyncio
	async def test_concurrent_inference_throughput(self, performance_service):
		"""Test concurrent inference throughput."""
		service, model = performance_service

		# Test different concurrency levels
		concurrency_levels = [1, 5, 10, 20, 50]
		results = {}

		for concurrency in concurrency_levels:
			print(f"\nTesting concurrency level: {concurrency}")

			requests_per_worker = 20
			total_requests = concurrency * requests_per_worker

			async def worker(worker_id: int):
				worker_latencies = []
				for i in range(requests_per_worker):
					request = AICRInferenceRequest(
						model_id=model.model_id,
						input_data={"worker": worker_id, "request": i}
					)

					start_time = time.perf_counter()
					response = await service.run_inference(request)
					end_time = time.perf_counter()

					latency_ms = (end_time - start_time) * 1000
					worker_latencies.append(latency_ms)

					assert response.status == InferenceStatus.COMPLETED

				return worker_latencies

			# Start concurrent workers
			start_time = time.perf_counter()
			tasks = [worker(i) for i in range(concurrency)]
			worker_results = await asyncio.gather(*tasks)
			end_time = time.perf_counter()

			# Calculate throughput
			total_duration = end_time - start_time
			throughput = total_requests / total_duration

			# Aggregate latencies
			all_latencies = []
			for worker_latencies in worker_results:
				all_latencies.extend(worker_latencies)

			avg_latency = statistics.mean(all_latencies)
			p95_latency = np.percentile(all_latencies, 95)

			results[concurrency] = {
				"throughput_rps": throughput,
				"avg_latency_ms": avg_latency,
				"p95_latency_ms": p95_latency,
				"total_requests": total_requests,
				"duration_s": total_duration
			}

			print(f"Throughput: {throughput:.2f} RPS")
			print(f"Average latency: {avg_latency:.2f}ms")
			print(f"P95 latency: {p95_latency:.2f}ms")

		# Performance assertions
		assert results[1]["throughput_rps"] > 10, "Single-threaded throughput too low"
		assert results[10]["throughput_rps"] > results[1]["throughput_rps"], "No throughput improvement with concurrency"

		# Latency should not degrade significantly with moderate concurrency
		low_concurrency_latency = results[5]["avg_latency_ms"]
		high_concurrency_latency = results[20]["avg_latency_ms"]
		latency_increase = (high_concurrency_latency / low_concurrency_latency) - 1

		assert latency_increase < 2.0, f"Latency degradation too high: {latency_increase*100:.1f}%"

	@pytest.mark.asyncio
	async def test_batch_inference_performance(self, performance_service):
		"""Test batch inference performance."""
		service, model = performance_service

		# Setup batch inference mock
		async def batch_inference(model_id, batch_data, **kwargs):
			batch_size = len(batch_data)
			# Simulate batch processing efficiency
			processing_time = 50 + batch_size * 2  # Base time + per-item time
			await asyncio.sleep(processing_time / 1000)  # Convert to seconds

			return [
				{
					"predictions": {"class": f"class_{i}", "confidence": 0.9},
					"processing_time_ms": processing_time / batch_size
				}
				for i in range(batch_size)
			]

		service.inference_engines["pytorch"].run_batch_inference = batch_inference

		# Test different batch sizes
		batch_sizes = [1, 5, 10, 20, 50, 100]
		batch_results = {}

		for batch_size in batch_sizes:
			print(f"\nTesting batch size: {batch_size}")

			# Prepare batch data
			batch_data = [{"data": [i, i+1, i+2]} for i in range(batch_size)]

			# Measure batch inference time
			start_time = time.perf_counter()
			responses = await service.run_batch_inference(model.model_id, batch_data)
			end_time = time.perf_counter()

			total_time = (end_time - start_time) * 1000  # Convert to ms
			time_per_item = total_time / batch_size
			throughput = batch_size / (total_time / 1000)  # Items per second

			batch_results[batch_size] = {
				"total_time_ms": total_time,
				"time_per_item_ms": time_per_item,
				"throughput_ips": throughput
			}

			print(f"Total time: {total_time:.2f}ms")
			print(f"Time per item: {time_per_item:.2f}ms")
			print(f"Throughput: {throughput:.2f} items/sec")

			assert len(responses) == batch_size
			assert all(resp.status == InferenceStatus.COMPLETED for resp in responses)

		# Batch processing should be more efficient than individual requests
		single_time = batch_results[1]["time_per_item_ms"]
		batch_50_time = batch_results[50]["time_per_item_ms"]

		efficiency_gain = (single_time / batch_50_time) - 1
		assert efficiency_gain > 0.2, f"Batch processing not efficient enough: {efficiency_gain*100:.1f}% gain"

	@pytest.mark.asyncio
	@pytest.mark.slow
	async def test_sustained_load_performance(self, performance_service):
		"""Test performance under sustained load."""
		service, model = performance_service

		# Run sustained load for 60 seconds
		duration_seconds = 60
		target_rps = 20  # Target requests per second

		start_time = time.perf_counter()
		end_time = start_time + duration_seconds

		request_count = 0
		latencies = []
		error_count = 0

		async def sustained_worker():
			nonlocal request_count, error_count

			while time.perf_counter() < end_time:
				try:
					request = AICRInferenceRequest(
						model_id=model.model_id,
						input_data={"timestamp": time.time(), "count": request_count}
					)

					req_start = time.perf_counter()
					response = await service.run_inference(request)
					req_end = time.perf_counter()

					if response.status == InferenceStatus.COMPLETED:
						latency_ms = (req_end - req_start) * 1000
						latencies.append(latency_ms)
					else:
						error_count += 1

					request_count += 1

					# Rate limiting to target RPS
					await asyncio.sleep(1.0 / target_rps)

				except Exception:
					error_count += 1

		# Start multiple workers to achieve target RPS
		num_workers = min(10, max(1, target_rps // 5))
		workers = [sustained_worker() for _ in range(num_workers)]

		await asyncio.gather(*workers)

		actual_duration = time.perf_counter() - start_time
		actual_rps = request_count / actual_duration
		error_rate = error_count / request_count if request_count > 0 else 0

		print(f"\nSustained Load Results:")
		print(f"Duration: {actual_duration:.1f}s")
		print(f"Total requests: {request_count}")
		print(f"Actual RPS: {actual_rps:.2f}")
		print(f"Error rate: {error_rate*100:.2f}%")

		if latencies:
			avg_latency = statistics.mean(latencies)
			p95_latency = np.percentile(latencies, 95)
			p99_latency = np.percentile(latencies, 99)

			print(f"Average latency: {avg_latency:.2f}ms")
			print(f"P95 latency: {p95_latency:.2f}ms")
			print(f"P99 latency: {p99_latency:.2f}ms")

			# Performance assertions
			assert avg_latency < 150, f"Average latency too high under load: {avg_latency:.2f}ms"
			assert p99_latency < 500, f"P99 latency too high under load: {p99_latency:.2f}ms"

		assert error_rate < 0.01, f"Error rate too high: {error_rate*100:.2f}%"
		assert actual_rps > target_rps * 0.8, f"Failed to achieve target RPS: {actual_rps:.2f} < {target_rps * 0.8:.2f}"


@pytest.mark.performance
class TestModelManagementPerformance:
	"""Performance tests for model management operations."""

	@pytest.fixture
	async def model_service(self):
		"""Create service for model management performance testing."""
		service = AICoreService()

		with patch.object(service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(service, '_start_background_tasks', new_callable=AsyncMock):

			await service.initialize()

		return service

	@pytest.mark.asyncio
	async def test_model_registration_performance(self, model_service):
		"""Test model registration performance."""
		service = model_service

		# Test batch model registration
		num_models = 100
		registration_times = []

		print(f"\nRegistering {num_models} models...")

		for i in range(num_models):
			model_data = {
				"name": f"perf_model_{i:03d}",
				"description": f"Performance test model {i}",
				"model_type": "classification",
				"framework": "pytorch",
				"version": f"1.{i}.0"
			}

			start_time = time.perf_counter()
			model = await service.register_model(model_data)
			end_time = time.perf_counter()

			registration_time = (end_time - start_time) * 1000  # Convert to ms
			registration_times.append(registration_time)

			assert model.name == f"perf_model_{i:03d}"

			if (i + 1) % 25 == 0:
				print(f"Registered {i + 1} models...")

		# Analyze registration performance
		avg_time = statistics.mean(registration_times)
		p95_time = np.percentile(registration_times, 95)

		print(f"Average registration time: {avg_time:.2f}ms")
		print(f"P95 registration time: {p95_time:.2f}ms")
		print(f"Total models in service: {len(service.models)}")

		# Performance assertions
		assert avg_time < 50, f"Model registration too slow: {avg_time:.2f}ms"
		assert p95_time < 100, f"P95 registration time too high: {p95_time:.2f}ms"
		assert len(service.models) == num_models

	@pytest.mark.asyncio
	async def test_model_listing_performance(self, model_service):
		"""Test model listing performance with large numbers of models."""
		service = model_service

		# Register a large number of models
		num_models = 1000
		print(f"\nRegistering {num_models} models for listing test...")

		# Use different model types and frameworks for filtering tests
		model_types = ["classification", "regression", "clustering"]
		frameworks = ["pytorch", "tensorflow", "sklearn"]

		for i in range(num_models):
			model_data = {
				"name": f"list_test_model_{i:04d}",
				"description": f"Model {i} for listing performance test",
				"model_type": model_types[i % len(model_types)],
				"framework": frameworks[i % len(frameworks)],
				"version": "1.0.0"
			}

			await service.register_model(model_data)

			if (i + 1) % 200 == 0:
				print(f"Registered {i + 1} models...")

		# Test listing performance
		print(f"\nTesting model listing performance...")

		# Test 1: List all models
		start_time = time.perf_counter()
		all_models = await service.list_models()
		end_time = time.perf_counter()

		list_all_time = (end_time - start_time) * 1000
		print(f"List all {len(all_models)} models: {list_all_time:.2f}ms")

		# Test 2: List with type filter
		start_time = time.perf_counter()
		classification_models = await service.list_models(model_type="classification")
		end_time = time.perf_counter()

		list_filtered_time = (end_time - start_time) * 1000
		print(f"List {len(classification_models)} classification models: {list_filtered_time:.2f}ms")

		# Test 3: List with pagination
		start_time = time.perf_counter()
		paginated_models = await service.list_models(limit=50, offset=100)
		end_time = time.perf_counter()

		list_paginated_time = (end_time - start_time) * 1000
		print(f"List 50 models with pagination: {list_paginated_time:.2f}ms")

		# Performance assertions
		assert list_all_time < 500, f"Listing all models too slow: {list_all_time:.2f}ms"
		assert list_filtered_time < 200, f"Filtered listing too slow: {list_filtered_time:.2f}ms"
		assert list_paginated_time < 100, f"Paginated listing too slow: {list_paginated_time:.2f}ms"

		assert len(all_models) == num_models
		assert len(paginated_models) == 50

	@pytest.mark.asyncio
	async def test_concurrent_model_operations(self, model_service):
		"""Test concurrent model management operations."""
		service = model_service

		# Test concurrent registrations
		print(f"\nTesting concurrent model registrations...")

		num_concurrent = 20

		async def concurrent_registration(worker_id: int):
			worker_times = []
			for i in range(5):  # Each worker registers 5 models
				model_data = {
					"name": f"concurrent_model_w{worker_id}_m{i}",
					"description": f"Concurrent model worker {worker_id} model {i}",
					"model_type": "classification",
					"framework": "pytorch"
				}

				start_time = time.perf_counter()
				model = await service.register_model(model_data)
				end_time = time.perf_counter()

				registration_time = (end_time - start_time) * 1000
				worker_times.append(registration_time)

				assert model.name == f"concurrent_model_w{worker_id}_m{i}"

			return worker_times

		# Run concurrent registrations
		start_time = time.perf_counter()
		tasks = [concurrent_registration(i) for i in range(num_concurrent)]
		worker_results = await asyncio.gather(*tasks)
		end_time = time.perf_counter()

		total_duration = (end_time - start_time) * 1000
		total_models = num_concurrent * 5

		# Aggregate timing results
		all_times = []
		for worker_times in worker_results:
			all_times.extend(worker_times)

		avg_registration_time = statistics.mean(all_times)
		throughput = total_models / (total_duration / 1000)  # Models per second

		print(f"Concurrent registrations completed in {total_duration:.2f}ms")
		print(f"Average registration time: {avg_registration_time:.2f}ms")
		print(f"Registration throughput: {throughput:.2f} models/sec")
		print(f"Total models registered: {total_models}")

		# Verify all models were registered
		assert len(service.models) == total_models

		# Performance assertions
		assert avg_registration_time < 100, f"Concurrent registration too slow: {avg_registration_time:.2f}ms"
		assert throughput > 10, f"Registration throughput too low: {throughput:.2f} models/sec"


@pytest.mark.performance
class TestMonitoringPerformance:
	"""Performance tests for monitoring system."""

	@pytest.fixture
	async def monitoring_service(self):
		"""Create monitoring service for performance testing."""
		with patch.object(ai_monitoring_system.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_monitoring_system, '_create_default_dashboards', new_callable=AsyncMock), \
			 patch.object(ai_monitoring_system, '_start_monitoring_tasks', new_callable=AsyncMock):

			await ai_monitoring_system.initialize()

		return ai_monitoring_system

	@pytest.mark.asyncio
	async def test_metrics_collection_performance(self, monitoring_service):
		"""Test metrics collection performance under high load."""
		monitoring = monitoring_service

		# Test high-frequency metric collection
		num_metrics = 10000
		collection_times = []

		print(f"\nCollecting {num_metrics} metrics...")

		start_time = time.perf_counter()

		for i in range(num_metrics):
			metric_start = time.perf_counter()

			await monitoring.metrics_collector.collect_metric(
				metric_name=f"perf_metric_{i % 10}",  # 10 different metric names
				value=100.0 + np.random.normal(0, 10),
				source_component="performance_test",
				labels={"batch": str(i // 100), "index": str(i)}
			)

			metric_end = time.perf_counter()
			collection_times.append((metric_end - metric_start) * 1000)

			if (i + 1) % 1000 == 0:
				print(f"Collected {i + 1} metrics...")

		end_time = time.perf_counter()

		total_duration = (end_time - start_time) * 1000
		avg_collection_time = statistics.mean(collection_times)
		throughput = num_metrics / (total_duration / 1000)

		print(f"Total collection time: {total_duration:.2f}ms")
		print(f"Average collection time: {avg_collection_time:.4f}ms")
		print(f"Collection throughput: {throughput:.2f} metrics/sec")

		# Performance assertions
		assert avg_collection_time < 1.0, f"Metric collection too slow: {avg_collection_time:.4f}ms"
		assert throughput > 1000, f"Collection throughput too low: {throughput:.2f} metrics/sec"

	@pytest.mark.asyncio
	async def test_metrics_retrieval_performance(self, monitoring_service):
		"""Test metrics retrieval performance with large datasets."""
		monitoring = monitoring_service

		# First, populate with a large number of metrics
		num_metrics = 50000
		print(f"\nPopulating {num_metrics} metrics for retrieval test...")

		metric_names = ["cpu_usage", "memory_usage", "disk_io", "network_io", "request_rate"]

		for i in range(num_metrics):
			metric_name = metric_names[i % len(metric_names)]
			await monitoring.metrics_collector.collect_metric(
				metric_name=metric_name,
				value=50.0 + np.random.normal(0, 15),
				source_component="retrieval_test",
				labels={"server": f"server-{i % 10}", "region": f"region-{i % 3}"}
			)

			if (i + 1) % 10000 == 0:
				print(f"Populated {i + 1} metrics...")

		# Test different retrieval scenarios
		print(f"\nTesting metrics retrieval performance...")

		# Test 1: Retrieve all metrics
		start_time = time.perf_counter()
		all_metrics = await monitoring.metrics_collector.get_metrics()
		end_time = time.perf_counter()

		retrieve_all_time = (end_time - start_time) * 1000
		print(f"Retrieved {len(all_metrics)} metrics: {retrieve_all_time:.2f}ms")

		# Test 2: Retrieve filtered by metric name
		start_time = time.perf_counter()
		filtered_metrics = await monitoring.metrics_collector.get_metrics(
			metric_names=["cpu_usage", "memory_usage"]
		)
		end_time = time.perf_counter()

		retrieve_filtered_time = (end_time - start_time) * 1000
		print(f"Retrieved {len(filtered_metrics)} filtered metrics: {retrieve_filtered_time:.2f}ms")

		# Test 3: Retrieve with time range
		end_timestamp = datetime.utcnow()
		start_timestamp = end_timestamp - timedelta(hours=1)

		start_time = time.perf_counter()
		time_range_metrics = await monitoring.metrics_collector.get_metrics(
			time_range=(start_timestamp, end_timestamp)
		)
		end_time = time.perf_counter()

		retrieve_time_range_time = (end_time - start_time) * 1000
		print(f"Retrieved {len(time_range_metrics)} time-range metrics: {retrieve_time_range_time:.2f}ms")

		# Performance assertions
		assert retrieve_all_time < 1000, f"Retrieve all metrics too slow: {retrieve_all_time:.2f}ms"
		assert retrieve_filtered_time < 500, f"Filtered retrieval too slow: {retrieve_filtered_time:.2f}ms"
		assert retrieve_time_range_time < 800, f"Time range retrieval too slow: {retrieve_time_range_time:.2f}ms"

	@pytest.mark.asyncio
	async def test_concurrent_monitoring_operations(self, monitoring_service):
		"""Test concurrent monitoring operations."""
		monitoring = monitoring_service

		# Test concurrent metric collection and retrieval
		print(f"\nTesting concurrent monitoring operations...")

		num_collectors = 5
		num_retrievers = 3
		operations_per_worker = 100

		async def metric_collector(collector_id: int):
			collection_times = []
			for i in range(operations_per_worker):
				start_time = time.perf_counter()

				await monitoring.metrics_collector.collect_metric(
					metric_name=f"concurrent_metric_{collector_id}",
					value=i * 10.0,
					source_component=f"collector_{collector_id}",
					labels={"iteration": str(i)}
				)

				end_time = time.perf_counter()
				collection_times.append((end_time - start_time) * 1000)

			return collection_times

		async def metric_retriever(retriever_id: int):
			retrieval_times = []
			for i in range(operations_per_worker):
				start_time = time.perf_counter()

				metrics = await monitoring.metrics_collector.get_metrics(
					metric_names=[f"concurrent_metric_{retriever_id % num_collectors}"]
				)

				end_time = time.perf_counter()
				retrieval_times.append((end_time - start_time) * 1000)

			return retrieval_times, len(metrics)

		# Start concurrent operations
		start_time = time.perf_counter()

		collector_tasks = [metric_collector(i) for i in range(num_collectors)]
		retriever_tasks = [metric_retriever(i) for i in range(num_retrievers)]

		all_tasks = collector_tasks + retriever_tasks
		results = await asyncio.gather(*all_tasks)

		end_time = time.perf_counter()

		total_duration = (end_time - start_time) * 1000

		# Analyze results
		collection_results = results[:num_collectors]
		retrieval_results = results[num_collectors:]

		all_collection_times = []
		for collection_times in collection_results:
			all_collection_times.extend(collection_times)

		all_retrieval_times = []
		total_retrieved_metrics = 0
		for retrieval_times, metric_count in retrieval_results:
			all_retrieval_times.extend(retrieval_times)
			total_retrieved_metrics += metric_count

		avg_collection_time = statistics.mean(all_collection_times)
		avg_retrieval_time = statistics.mean(all_retrieval_times)

		total_operations = (num_collectors + num_retrievers) * operations_per_worker
		throughput = total_operations / (total_duration / 1000)

		print(f"Concurrent operations completed in {total_duration:.2f}ms")
		print(f"Average collection time: {avg_collection_time:.4f}ms")
		print(f"Average retrieval time: {avg_retrieval_time:.2f}ms")
		print(f"Total operations: {total_operations}")
		print(f"Operations throughput: {throughput:.2f} ops/sec")

		# Performance assertions
		assert avg_collection_time < 2.0, f"Concurrent collection too slow: {avg_collection_time:.4f}ms"
		assert avg_retrieval_time < 50, f"Concurrent retrieval too slow: {avg_retrieval_time:.2f}ms"
		assert throughput > 50, f"Concurrent throughput too low: {throughput:.2f} ops/sec"


@pytest.mark.performance
class TestResourceUtilization:
	"""Performance tests for resource utilization and memory management."""

	@pytest.mark.asyncio
	async def test_memory_usage_under_load(self):
		"""Test memory usage under sustained load."""
		print(f"\nTesting memory usage under load...")

		# Get initial memory usage
		process = psutil.Process()
		initial_memory = process.memory_info().rss / 1024 / 1024  # MB

		print(f"Initial memory usage: {initial_memory:.2f} MB")

		# Create AI service and run sustained operations
		service = AICoreService()

		with patch.object(service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(service, '_start_background_tasks', new_callable=AsyncMock):

			await service.initialize()

		# Setup mock engine
		mock_engine = Mock()
		mock_engine.deploy_model = AsyncMock(return_value={"success": True})
		mock_engine.run_inference = AsyncMock(return_value={
			"predictions": {"class": "test"}, "processing_time_ms": 50.0
		})
		service.inference_engines["pytorch"] = mock_engine

		memory_measurements = []

		# Phase 1: Register many models
		print("Phase 1: Registering models...")
		for i in range(500):
			model_data = {
				"name": f"memory_test_model_{i:03d}",
				"description": f"Memory test model {i}",
				"model_type": "classification",
				"framework": "pytorch"
			}
			await service.register_model(model_data)

			if (i + 1) % 100 == 0:
				current_memory = process.memory_info().rss / 1024 / 1024
				memory_measurements.append(("register", i + 1, current_memory))
				print(f"Registered {i + 1} models, Memory: {current_memory:.2f} MB")

		# Phase 2: Deploy and run inference
		print("Phase 2: Running inference operations...")
		models = list(service.models.values())[:50]  # Use first 50 models

		for model in models:
			await service.deploy_model(model.model_id)

		deployment_memory = process.memory_info().rss / 1024 / 1024
		memory_measurements.append(("deploy", len(models), deployment_memory))
		print(f"Deployed {len(models)} models, Memory: {deployment_memory:.2f} MB")

		# Run many inference requests
		for i in range(1000):
			model = models[i % len(models)]
			request = AICRInferenceRequest(
				model_id=model.model_id,
				input_data={"data": [i, i+1, i+2]}
			)
			await service.run_inference(request)

			if (i + 1) % 200 == 0:
				current_memory = process.memory_info().rss / 1024 / 1024
				memory_measurements.append(("inference", i + 1, current_memory))
				print(f"Completed {i + 1} inferences, Memory: {current_memory:.2f} MB")

		# Final memory measurement
		final_memory = process.memory_info().rss / 1024 / 1024
		memory_measurements.append(("final", 0, final_memory))

		print(f"Final memory usage: {final_memory:.2f} MB")
		print(f"Total memory increase: {final_memory - initial_memory:.2f} MB")

		# Analyze memory growth
		max_memory = max(measurement[2] for measurement in memory_measurements)
		memory_growth = final_memory - initial_memory
		memory_growth_rate = memory_growth / final_memory

		print(f"Peak memory usage: {max_memory:.2f} MB")
		print(f"Memory growth rate: {memory_growth_rate * 100:.1f}%")

		# Memory usage assertions
		assert memory_growth < 500, f"Memory growth too high: {memory_growth:.2f} MB"
		assert memory_growth_rate < 0.5, f"Memory growth rate too high: {memory_growth_rate * 100:.1f}%"
		assert max_memory < initial_memory + 600, f"Peak memory too high: {max_memory:.2f} MB"

	@pytest.mark.asyncio
	async def test_cpu_utilization_efficiency(self):
		"""Test CPU utilization efficiency."""
		print(f"\nTesting CPU utilization efficiency...")

		# Get initial CPU usage
		initial_cpu = psutil.cpu_percent(interval=1)
		print(f"Initial CPU usage: {initial_cpu:.1f}%")

		# Create service and run CPU-intensive operations
		service = AICoreService()

		with patch.object(service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(service, '_start_background_tasks', new_callable=AsyncMock):

			await service.initialize()

		# Setup mock engine with CPU simulation
		async def cpu_intensive_inference(*args, **kwargs):
			# Simulate CPU-intensive work
			start = time.perf_counter()
			while time.perf_counter() - start < 0.01:  # 10ms of work
				_ = sum(i * i for i in range(1000))

			return {
				"predictions": {"class": "test"},
				"processing_time_ms": 10.0
			}

		mock_engine = Mock()
		mock_engine.deploy_model = AsyncMock(return_value={"success": True})
		mock_engine.run_inference = cpu_intensive_inference
		service.inference_engines["pytorch"] = mock_engine

		# Register and deploy model
		model_data = {
			"name": "cpu_test_model",
			"description": "Model for CPU testing",
			"model_type": "classification",
			"framework": "pytorch"
		}

		model = await service.register_model(model_data)
		await service.deploy_model(model.model_id)

		# Run concurrent inference to test CPU utilization
		num_concurrent = 20
		requests_per_worker = 50

		cpu_measurements = []

		async def cpu_monitoring():
			for _ in range(20):  # Monitor for 20 seconds
				cpu_usage = psutil.cpu_percent(interval=1)
				cpu_measurements.append(cpu_usage)

		async def inference_worker():
			for i in range(requests_per_worker):
				request = AICRInferenceRequest(
					model_id=model.model_id,
					input_data={"data": [i, i+1, i+2]}
				)
				await service.run_inference(request)

		# Start CPU monitoring
		monitor_task = asyncio.create_task(cpu_monitoring())

		# Start inference workers
		start_time = time.perf_counter()
		inference_tasks = [inference_worker() for _ in range(num_concurrent)]
		await asyncio.gather(*inference_tasks)
		end_time = time.perf_counter()

		# Stop monitoring
		monitor_task.cancel()

		duration = end_time - start_time
		total_requests = num_concurrent * requests_per_worker
		throughput = total_requests / duration

		avg_cpu = statistics.mean(cpu_measurements) if cpu_measurements else 0
		max_cpu = max(cpu_measurements) if cpu_measurements else 0

		print(f"Test duration: {duration:.2f}s")
		print(f"Total requests: {total_requests}")
		print(f"Throughput: {throughput:.2f} req/sec")
		print(f"Average CPU usage: {avg_cpu:.1f}%")
		print(f"Peak CPU usage: {max_cpu:.1f}%")

		# CPU efficiency assertions
		cpu_efficiency = throughput / max(avg_cpu, 1)  # Requests per second per CPU percent

		print(f"CPU efficiency: {cpu_efficiency:.2f} req/sec per CPU%")

		assert avg_cpu < 80, f"Average CPU usage too high: {avg_cpu:.1f}%"
		assert cpu_efficiency > 0.5, f"CPU efficiency too low: {cpu_efficiency:.2f}"
		assert throughput > 50, f"Throughput too low under CPU load: {throughput:.2f} req/sec"


if __name__ == "__main__":
	# Run performance tests with detailed output
	pytest.main([__file__, "-v", "-s", "--tb=short"])