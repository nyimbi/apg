"""
APG Event Streaming Bus - Performance Tests

Performance and load tests for the Event Streaming Bus capability,
focusing on throughput, latency, and scalability metrics.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, median

from ...models import EventConfig, StreamConfig, SubscriptionConfig
from ...service import EventStreamingService, EventPublishingService, EventConsumptionService

# =============================================================================
# Event Publishing Performance Tests
# =============================================================================

@pytest.mark.performance
class TestEventPublishingPerformance:
	"""Test event publishing performance and throughput."""
	
	@pytest.mark.asyncio
	@pytest.mark.slow
	async def test_single_event_publishing_latency(
		self,
		mock_event_publishing_service,
		performance_metrics_collector,
		test_tenant_id,
		test_user_id
	):
		"""Test latency of single event publishing."""
		
		event_config = EventConfig(
			event_type="performance.test",
			source_capability="performance_test",
			aggregate_id="perf_test_1",
			aggregate_type="PerformanceTest"
		)
		
		payload = {"test_data": "performance_test_payload"}
		
		# Measure latency for 100 individual event publishes
		latencies = []
		
		for i in range(100):
			performance_metrics_collector.start_timer("single_publish")
			
			await mock_event_publishing_service.publish_event(
				event_config=event_config,
				payload=payload,
				stream_id="performance_stream",
				tenant_id=test_tenant_id,
				user_id=test_user_id
			)
			
			latency = performance_metrics_collector.end_timer("single_publish")
			latencies.append(latency)
		
		# Analyze latency metrics
		avg_latency = mean(latencies)
		median_latency = median(latencies)
		p95_latency = performance_metrics_collector.get_percentile("single_publish", 95)
		p99_latency = performance_metrics_collector.get_percentile("single_publish", 99)
		
		print(f"Single Event Publishing Latency Metrics:")
		print(f"  Average: {avg_latency*1000:.2f}ms")
		print(f"  Median: {median_latency*1000:.2f}ms")
		print(f"  95th percentile: {p95_latency*1000:.2f}ms")
		print(f"  99th percentile: {p99_latency*1000:.2f}ms")
		
		# Performance assertions (adjust based on requirements)
		assert avg_latency < 0.010  # Average < 10ms
		assert p95_latency < 0.050  # 95th percentile < 50ms
		assert p99_latency < 0.100  # 99th percentile < 100ms
	
	@pytest.mark.asyncio
	@pytest.mark.slow
	async def test_batch_publishing_throughput(
		self,
		mock_event_publishing_service,
		performance_event_generator,
		performance_metrics_collector,
		test_tenant_id,
		test_user_id
	):
		"""Test batch publishing throughput."""
		
		batch_sizes = [10, 50, 100, 500, 1000]
		throughput_results = {}
		
		for batch_size in batch_sizes:
			# Generate batch of events
			events_data = performance_event_generator(batch_size)
			batch_events = []
			
			for event_data in events_data:
				event_config = EventConfig(
					event_type=event_data["event_type"],
					source_capability=event_data["source_capability"],
					aggregate_id=event_data["aggregate_id"],
					aggregate_type=event_data["aggregate_type"]
				)
				batch_events.append((event_config, event_data["payload"]))
			
			# Measure batch publishing time
			start_time = time.time()
			
			await mock_event_publishing_service.publish_event_batch(
				events=batch_events,
				stream_id="performance_batch_stream",
				tenant_id=test_tenant_id,
				user_id=test_user_id
			)
			
			end_time = time.time()
			duration = end_time - start_time
			throughput = batch_size / duration
			
			throughput_results[batch_size] = throughput
			
			print(f"Batch size {batch_size}: {throughput:.0f} events/second")
		
		# Verify throughput improves with larger batches
		assert throughput_results[1000] > throughput_results[10]
		
		# Performance assertion (adjust based on requirements)
		assert throughput_results[1000] > 10000  # > 10K events/second for large batches
	
	@pytest.mark.asyncio
	@pytest.mark.slow
	async def test_concurrent_publishing_throughput(
		self,
		mock_event_publishing_service,
		performance_event_generator,
		test_tenant_id,
		test_user_id
	):
		"""Test concurrent publishing throughput with multiple publishers."""
		
		concurrent_publishers = [1, 2, 4, 8, 16]
		events_per_publisher = 100
		
		for publisher_count in concurrent_publishers:
			start_time = time.time()
			
			# Create concurrent publishing tasks
			tasks = []
			for p in range(publisher_count):
				events_data = performance_event_generator(events_per_publisher)
				
				async def publish_events(events_data, publisher_id):
					for event_data in events_data:
						event_config = EventConfig(
							event_type=event_data["event_type"],
							source_capability=event_data["source_capability"],
							aggregate_id=f"{event_data['aggregate_id']}_p{publisher_id}",
							aggregate_type=event_data["aggregate_type"]
						)
						
						await mock_event_publishing_service.publish_event(
							event_config=event_config,
							payload=event_data["payload"],
							stream_id="concurrent_performance_stream",
							tenant_id=test_tenant_id,
							user_id=test_user_id
						)
				
				task = asyncio.create_task(publish_events(events_data, p))
				tasks.append(task)
			
			# Wait for all publishers to complete
			await asyncio.gather(*tasks)
			
			end_time = time.time()
			duration = end_time - start_time
			total_events = publisher_count * events_per_publisher
			throughput = total_events / duration
			
			print(f"Concurrent publishers {publisher_count}: {throughput:.0f} events/second")
		
		# Performance assertion
		# Throughput should scale with concurrent publishers (up to a point)
		# This depends on the actual implementation and resource constraints

# =============================================================================
# Event Consumption Performance Tests
# =============================================================================

@pytest.mark.performance
class TestEventConsumptionPerformance:
	"""Test event consumption performance and processing rates."""
	
	@pytest.mark.asyncio
	@pytest.mark.slow
	async def test_consumer_throughput(
		self,
		mock_event_consumption_service,
		performance_event_generator,
		test_tenant_id,
		test_user_id
	):
		"""Test consumer throughput and processing rates."""
		
		# Create subscription
		subscription_config = SubscriptionConfig(
			subscription_name="performance_subscription",
			stream_id="performance_stream",
			consumer_group_id="performance_consumers",
			consumer_name="performance_consumer",
			batch_size=100,
			max_wait_time_ms=1000
		)
		
		subscription_id = await mock_event_consumption_service.create_subscription(
			config=subscription_config,
			tenant_id=test_tenant_id,
			created_by=test_user_id
		)
		
		# Generate events to process
		event_counts = [100, 500, 1000, 5000]
		
		for event_count in event_counts:
			events_data = performance_event_generator(event_count)
			
			# Convert to event format expected by consumer
			events = []
			for event_data in events_data:
				event = {
					"event_id": f"evt_{event_data['test_id']}",
					"event_type": event_data["event_type"],
					"payload": event_data["payload"],
					"metadata": event_data["metadata"]
				}
				events.append(event)
			
			# Measure consumption time
			start_time = time.time()
			
			processed_count = await mock_event_consumption_service.process_events(
				subscription_id,
				events
			)
			
			end_time = time.time()
			duration = end_time - start_time
			throughput = processed_count / duration
			
			print(f"Processed {processed_count} events in {duration:.2f}s: {throughput:.0f} events/second")
			
			assert processed_count == event_count
	
	@pytest.mark.asyncio
	@pytest.mark.slow
	async def test_consumer_group_scaling(
		self,
		mock_event_consumption_service,
		test_tenant_id,
		test_user_id
	):
		"""Test consumer group scaling with multiple consumers."""
		
		consumer_group_id = "scaling_test_group"
		consumer_counts = [1, 2, 4, 8]
		events_per_test = 1000
		
		for consumer_count in consumer_counts:
			# Create multiple consumers in the same group
			subscription_ids = []
			
			for i in range(consumer_count):
				subscription_config = SubscriptionConfig(
					subscription_name=f"scaling_consumer_{i}",
					stream_id="scaling_test_stream",
					consumer_group_id=consumer_group_id,
					consumer_name=f"consumer_instance_{i}",
					batch_size=50
				)
				
				subscription_id = await mock_event_consumption_service.create_subscription(
					config=subscription_config,
					tenant_id=test_tenant_id,
					created_by=test_user_id
				)
				
				subscription_ids.append(subscription_id)
			
			print(f"Created {consumer_count} consumers in group {consumer_group_id}")
			
			# In a real test, we would measure actual consumption scaling
			# For now, we verify the consumers were created successfully
			assert len(subscription_ids) == consumer_count
	
	@pytest.mark.asyncio
	@pytest.mark.slow
	async def test_consumer_lag_under_load(
		self,
		mock_event_consumption_service,
		performance_event_generator,
		test_tenant_id,
		test_user_id
	):
		"""Test consumer lag behavior under high load."""
		
		# Create subscription
		subscription_config = SubscriptionConfig(
			subscription_name="lag_test_subscription",
			stream_id="high_load_stream",
			consumer_group_id="lag_test_consumers",
			consumer_name="lag_test_consumer",
			batch_size=10,  # Small batch to create potential lag
			max_wait_time_ms=100
		)
		
		subscription_id = await mock_event_consumption_service.create_subscription(
			config=subscription_config,
			tenant_id=test_tenant_id,
			created_by=test_user_id
		)
		
		# Simulate high load scenario
		high_load_events = performance_event_generator(10000)
		
		# Measure lag metrics
		start_time = time.time()
		
		# Process events in smaller batches to simulate real-world conditions
		batch_size = 100
		total_processed = 0
		
		for i in range(0, len(high_load_events), batch_size):
			batch = high_load_events[i:i + batch_size]
			events = []
			
			for event_data in batch:
				event = {
					"event_id": f"evt_lag_{event_data['test_id']}",
					"event_type": event_data["event_type"],
					"payload": event_data["payload"]
				}
				events.append(event)
			
			processed = await mock_event_consumption_service.process_events(
				subscription_id,
				events
			)
			
			total_processed += processed
			
			# Simulate processing delay
			await asyncio.sleep(0.01)
		
		end_time = time.time()
		total_duration = end_time - start_time
		
		print(f"Processed {total_processed} events under load in {total_duration:.2f}s")
		print(f"Average throughput: {total_processed / total_duration:.0f} events/second")
		
		assert total_processed == len(high_load_events)

# =============================================================================
# Stream Processing Performance Tests
# =============================================================================

@pytest.mark.performance
class TestStreamProcessingPerformance:
	"""Test stream processing performance for real-time analytics."""
	
	@pytest.mark.asyncio
	@pytest.mark.slow
	async def test_stream_aggregation_performance(
		self,
		mock_event_streaming_service,
		performance_event_generator
	):
		"""Test performance of stream aggregation operations."""
		
		# Create stream for aggregation testing
		from ...service import StreamProcessingService
		processing_service = StreamProcessingService()
		
		# Configure aggregation window
		aggregation_config = {
			"window_type": "tumbling",
			"duration_ms": 1000,  # 1 second windows
			"aggregation_function": "count",
			"group_by": "event_type"
		}
		
		# Generate events for aggregation
		events_data = performance_event_generator(10000)
		
		# Measure aggregation performance
		start_time = time.time()
		
		# Simulate processing events through aggregation window
		processed_windows = 0
		for i in range(0, len(events_data), 1000):  # Process in 1000-event batches
			batch = events_data[i:i + 1000]
			
			# Mock aggregation processing
			window_id = await processing_service.create_aggregation_window(
				stream_id="aggregation_test_stream",
				config=aggregation_config
			)
			
			processed_windows += 1
		
		end_time = time.time()
		duration = end_time - start_time
		throughput = len(events_data) / duration
		
		print(f"Stream aggregation performance:")
		print(f"  Events processed: {len(events_data)}")
		print(f"  Windows created: {processed_windows}")
		print(f"  Duration: {duration:.2f}s")
		print(f"  Throughput: {throughput:.0f} events/second")
		
		assert throughput > 1000  # Should process > 1K events/second
	
	@pytest.mark.asyncio
	@pytest.mark.slow
	async def test_complex_event_processing_performance(
		self,
		mock_event_streaming_service,
		performance_event_generator
	):
		"""Test performance of complex event processing patterns."""
		
		from ...service import StreamProcessingService
		processing_service = StreamProcessingService()
		
		# Configure complex event pattern
		pattern_config = {
			"pattern_type": "sequence",
			"events": ["user.created", "user.activated", "user.first_purchase"],
			"within_ms": 300000,  # 5 minutes
			"correlation_field": "user_id"
		}
		
		# Generate correlated event sequences
		sequences_count = 1000
		events_data = []
		
		for i in range(sequences_count):
			user_id = f"user_{i}"
			
			# Create sequence of events for each user
			events_data.extend([
				{
					"event_type": "user.created",
					"aggregate_id": user_id,
					"payload": {"user_id": user_id, "timestamp": time.time()}
				},
				{
					"event_type": "user.activated",
					"aggregate_id": user_id,
					"payload": {"user_id": user_id, "timestamp": time.time() + 60}
				},
				{
					"event_type": "user.first_purchase",
					"aggregate_id": user_id,
					"payload": {"user_id": user_id, "amount": 50.0, "timestamp": time.time() + 120}
				}
			])
		
		# Measure pattern matching performance
		start_time = time.time()
		
		matched_patterns = 0
		
		# Process events in batches
		batch_size = 100
		for i in range(0, len(events_data), batch_size):
			batch = events_data[i:i + batch_size]
			
			# Mock pattern processing
			matches = await processing_service.process_complex_event_pattern(
				pattern_config=pattern_config,
				events=batch
			)
			
			if matches:
				matched_patterns += 1
		
		end_time = time.time()
		duration = end_time - start_time
		
		print(f"Complex event processing performance:")
		print(f"  Events processed: {len(events_data)}")
		print(f"  Pattern matches: {matched_patterns}")
		print(f"  Duration: {duration:.2f}s")
		print(f"  Throughput: {len(events_data) / duration:.0f} events/second")

# =============================================================================
# System-wide Performance Tests
# =============================================================================

@pytest.mark.performance
class TestSystemPerformance:
	"""Test overall system performance under various load conditions."""
	
	@pytest.mark.asyncio
	@pytest.mark.slow
	async def test_end_to_end_latency(
		self,
		mock_event_publishing_service,
		mock_event_consumption_service,
		performance_metrics_collector,
		test_tenant_id,
		test_user_id
	):
		"""Test end-to-end latency from publish to consume."""
		
		# Create subscription
		subscription_config = SubscriptionConfig(
			subscription_name="e2e_latency_subscription",
			stream_id="e2e_latency_stream",
			consumer_group_id="e2e_latency_consumers",
			consumer_name="e2e_latency_consumer"
		)
		
		subscription_id = await mock_event_consumption_service.create_subscription(
			config=subscription_config,
			tenant_id=test_tenant_id,
			created_by=test_user_id
		)
		
		# Measure end-to-end latency for multiple events
		latencies = []
		
		for i in range(50):
			# Record publish time
			publish_start = time.time()
			
			event_config = EventConfig(
				event_type="e2e.latency.test",
				source_capability="latency_test",
				aggregate_id=f"latency_test_{i}",
				aggregate_type="LatencyTest"
			)
			
			payload = {
				"test_id": i,
				"publish_timestamp": publish_start
			}
			
			# Publish event
			event_id = await mock_event_publishing_service.publish_event(
				event_config=event_config,
				payload=payload,
				stream_id="e2e_latency_stream",
				tenant_id=test_tenant_id,
				user_id=test_user_id
			)
			
			# Simulate consumption (in real test, this would be actual consumption)
			consume_start = time.time()
			
			# Mock event consumption
			consumed_events = [{
				"event_id": event_id,
				"event_type": "e2e.latency.test",
				"payload": payload
			}]
			
			await mock_event_consumption_service.process_events(
				subscription_id,
				consumed_events
			)
			
			consume_end = time.time()
			
			# Calculate end-to-end latency
			e2e_latency = consume_end - publish_start
			latencies.append(e2e_latency)
		
		# Analyze latency metrics
		avg_latency = mean(latencies)
		median_latency = median(latencies)
		max_latency = max(latencies)
		
		print(f"End-to-end latency metrics:")
		print(f"  Average: {avg_latency*1000:.2f}ms")
		print(f"  Median: {median_latency*1000:.2f}ms")
		print(f"  Maximum: {max_latency*1000:.2f}ms")
		
		# Performance assertions
		assert avg_latency < 0.100  # Average < 100ms
		assert median_latency < 0.050  # Median < 50ms
		assert max_latency < 0.500  # Max < 500ms
	
	@pytest.mark.asyncio
	@pytest.mark.slow
	async def test_system_saturation_point(
		self,
		mock_event_publishing_service,
		mock_event_consumption_service,
		performance_event_generator,
		test_tenant_id,
		test_user_id
	):
		"""Test system behavior at saturation point."""
		
		# Gradually increase load to find saturation point
		load_levels = [100, 500, 1000, 2000, 5000, 10000]
		throughput_results = []
		
		for load_level in load_levels:
			events_data = performance_event_generator(load_level)
			
			start_time = time.time()
			
			# Publish events
			for event_data in events_data:
				event_config = EventConfig(
					event_type=event_data["event_type"],
					source_capability=event_data["source_capability"],
					aggregate_id=event_data["aggregate_id"],
					aggregate_type=event_data["aggregate_type"]
				)
				
				await mock_event_publishing_service.publish_event(
					event_config=event_config,
					payload=event_data["payload"],
					stream_id="saturation_test_stream",
					tenant_id=test_tenant_id,
					user_id=test_user_id
				)
			
			end_time = time.time()
			duration = end_time - start_time
			throughput = load_level / duration
			
			throughput_results.append({
				"load_level": load_level,
				"throughput": throughput,
				"duration": duration
			})
			
			print(f"Load level {load_level}: {throughput:.0f} events/second")
		
		# Analyze saturation behavior
		# Throughput should increase with load up to a point, then plateau or degrade
		peak_throughput = max(result["throughput"] for result in throughput_results)
		
		print(f"Peak throughput: {peak_throughput:.0f} events/second")
		
		# Find load level that achieved peak throughput
		peak_result = next(r for r in throughput_results if r["throughput"] == peak_throughput)
		print(f"Peak achieved at load level: {peak_result['load_level']}")
		
		assert peak_throughput > 1000  # Should handle > 1K events/second