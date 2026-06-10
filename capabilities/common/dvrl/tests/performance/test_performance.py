#!/usr/bin/env python3
"""
Performance and Benchmarking Tests for DVRL Capability
Tests for performance characteristics, scalability, and resource usage

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import time
import pytest
import statistics
import psutil
import gc
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
from datetime import datetime, timezone

from capabilities.common.dvrl.service import DVRLService
from capabilities.common.dvrl.models import DataSource, DataSourceType, DataSourceStatus


class PerformanceProfiler:
	"""Helper class for performance profiling"""
	
	def __init__(self):
		self.start_time = None
		self.end_time = None
		self.memory_before = None
		self.memory_after = None
	
	def start(self):
		"""Start profiling"""
		gc.collect()  # Clean up before measurement
		process = psutil.Process()
		self.memory_before = process.memory_info().rss / 1024 / 1024  # MB
		self.start_time = time.perf_counter()
	
	def stop(self):
		"""Stop profiling and return metrics"""
		self.end_time = time.perf_counter()
		gc.collect()
		process = psutil.Process()
		self.memory_after = process.memory_info().rss / 1024 / 1024  # MB
		
		return {
			'duration_ms': (self.end_time - self.start_time) * 1000,
			'memory_used_mb': self.memory_after - self.memory_before,
			'memory_peak_mb': self.memory_after
		}


@pytest.fixture
async def performance_dvrl_service():
	"""Create DVRL service optimized for performance testing"""
	config = {
		'tenant_id': 'perf_test',
		'user_id': 'perf_user', 
		'enable_cache': True,
		'cache_ttl': 3600,
		'max_concurrent_queries': 50,  # Higher limit for performance testing
		'query_timeout': 30,
		'connection_pool_size': 20
	}
	
	service = DVRLService(config)
	await service.initialize()
	return service


@pytest.fixture
def sample_data_sources():
	"""Create multiple sample data sources for testing"""
	return [
		{
			'name': f'postgres_db_{i}',
			'type': 'postgresql',
			'host': 'localhost',
			'port': 5432 + i,
			'database': f'testdb_{i}',
			'user': 'testuser',
			'password': 'testpass'
		}
		for i in range(10)
	]


class TestQueryPerformance:
	"""Test query execution performance"""
	
	@patch('asyncpg.create_pool')
	async def test_single_query_performance(self, mock_create_pool, performance_dvrl_service):
		"""Test performance of single query execution"""
		# Mock fast connection pool
		mock_pool = AsyncMock()
		mock_connection = AsyncMock()
		mock_connection.fetch.return_value = [
			{'id': i, 'name': f'user_{i}', 'email': f'user_{i}@example.com'}
			for i in range(1000)  # 1000 rows
		]
		mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
		mock_create_pool.return_value = mock_pool
		
		# Register data source
		config = {
			'name': 'perf_test_db',
			'type': 'postgresql',
			'host': 'localhost',
			'port': 5432,
			'database': 'perfdb',
			'user': 'testuser',
			'password': 'testpass'
		}
		
		data_source = await performance_dvrl_service.register_data_source(config)
		
		# Performance test
		profiler = PerformanceProfiler()
		profiler.start()
		
		result = await performance_dvrl_service.execute_federated_query(
			"SELECT * FROM users LIMIT 1000",
			{},
			{}
		)
		
		metrics = profiler.stop()
		
		# Performance assertions
		assert metrics['duration_ms'] < 500  # Should complete in under 500ms
		assert metrics['memory_used_mb'] < 50  # Should not use excessive memory
		assert result is not None
	
	@patch('asyncpg.create_pool')
	async def test_concurrent_query_performance(self, mock_create_pool, performance_dvrl_service):
		"""Test performance under concurrent query load"""
		# Mock connection pool
		mock_pool = AsyncMock()
		mock_connection = AsyncMock()
		mock_connection.fetch.return_value = [{'count': 42}]
		mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
		mock_create_pool.return_value = mock_pool
		
		# Register data source
		config = {
			'name': 'concurrent_test_db',
			'type': 'postgresql',
			'host': 'localhost',
			'port': 5432,
			'database': 'concurrentdb',
			'user': 'testuser',
			'password': 'testpass'
		}
		
		await performance_dvrl_service.register_data_source(config)
		
		# Define query function
		async def execute_test_query(query_id: int):
			start_time = time.perf_counter()
			result = await performance_dvrl_service.execute_federated_query(
				f"SELECT COUNT(*) as count FROM table_{query_id % 5}",
				{},
				{}
			)
			end_time = time.perf_counter()
			return {
				'query_id': query_id,
				'duration_ms': (end_time - start_time) * 1000,
				'success': result is not None
			}
		
		# Execute 20 concurrent queries
		profiler = PerformanceProfiler()
		profiler.start()
		
		tasks = [execute_test_query(i) for i in range(20)]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		metrics = profiler.stop()
		
		# Analyze results
		successful_results = [r for r in results if isinstance(r, dict) and r['success']]
		durations = [r['duration_ms'] for r in successful_results]
		
		# Performance assertions
		assert len(successful_results) >= 18  # At least 90% success rate
		assert metrics['duration_ms'] < 2000  # Total time under 2 seconds
		assert statistics.mean(durations) < 300  # Average query time under 300ms
		assert max(durations) < 1000  # No query should take more than 1 second
	
	async def test_query_scaling_with_data_size(self, performance_dvrl_service):
		"""Test query performance scaling with data size"""
		data_sizes = [100, 1000, 10000, 50000]
		performance_results = []
		
		for size in data_sizes:
			with patch('asyncpg.create_pool') as mock_create_pool:
				# Mock data of varying sizes
				mock_pool = AsyncMock()
				mock_connection = AsyncMock()
				mock_connection.fetch.return_value = [
					{'id': i, 'data': f'data_{i}' * 10}  # Simulate larger records
					for i in range(size)
				]
				mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
				mock_create_pool.return_value = mock_pool
				
				# Register fresh data source for each test
				config = {
					'name': f'scaling_test_db_{size}',
					'type': 'postgresql', 
					'host': 'localhost',
					'port': 5432,
					'database': f'scalingdb_{size}',
					'user': 'testuser',
					'password': 'testpass'
				}
				
				await performance_dvrl_service.register_data_source(config)
				
				# Measure performance
				profiler = PerformanceProfiler()
				profiler.start()
				
				result = await performance_dvrl_service.execute_federated_query(
					f"SELECT * FROM large_table LIMIT {size}",
					{},
					{}
				)
				
				metrics = profiler.stop()
				
				performance_results.append({
					'data_size': size,
					'duration_ms': metrics['duration_ms'],
					'memory_mb': metrics['memory_used_mb'],
					'success': result is not None
				})
		
		# Analyze scaling characteristics
		assert all(r['success'] for r in performance_results)
		
		# Check that performance scales reasonably
		durations = [r['duration_ms'] for r in performance_results]
		data_sizes_actual = [r['data_size'] for r in performance_results]
		
		# Duration should not increase exponentially with data size
		# Allow for some performance degradation but not exponential
		ratio_10k_100 = durations[2] / durations[0]  # 10k vs 100 records
		assert ratio_10k_100 < 20  # Should not be more than 20x slower
	
	@patch('ollama.generate')
	async def test_nlp_query_performance(self, mock_generate, performance_dvrl_service):
		"""Test NLP query processing performance"""
		# Mock Ollama responses
		mock_generate.side_effect = [
			{'response': 'SELECT COUNT(*) FROM users WHERE active = true;'},
			{'response': 'This query counts active users'}
		]
		
		# Test multiple NL queries
		test_queries = [
			"How many active users do we have?",
			"What is the total revenue this month?",
			"Show me the top 10 products by sales",
			"Which customers have placed orders recently?",
			"What is the average order value?"
		]
		
		profiler = PerformanceProfiler()
		profiler.start()
		
		results = []
		for query in test_queries:
			query_start = time.perf_counter()
			
			result = await performance_dvrl_service.execute_natural_language_query(
				query,
				[],
				{}
			)
			
			query_end = time.perf_counter()
			
			results.append({
				'query': query,
				'duration_ms': (query_end - query_start) * 1000,
				'success': result is not None and 'generated_sql' in result
			})
		
		metrics = profiler.stop()
		
		# Performance assertions for NLP
		successful_results = [r for r in results if r['success']]
		durations = [r['duration_ms'] for r in successful_results]
		
		assert len(successful_results) >= 4  # At least 80% success rate
		assert statistics.mean(durations) < 1000  # Average NLP processing under 1 second
		assert metrics['duration_ms'] < 5000  # Total processing under 5 seconds


class TestConnectionPoolPerformance:
	"""Test database connection pool performance"""
	
	@patch('asyncpg.create_pool')
	async def test_connection_pool_efficiency(self, mock_create_pool, performance_dvrl_service):
		"""Test connection pool efficiency under load"""
		# Mock connection pool with limited connections
		mock_pool = AsyncMock()
		mock_connections = [AsyncMock() for _ in range(5)]  # 5 connections in pool
		
		connection_usage = []
		
		async def mock_acquire():
			"""Mock acquire that tracks usage"""
			if len(connection_usage) < 5:
				conn = mock_connections[len(connection_usage)]
				connection_usage.append(conn)
				return conn
			else:
				# Simulate waiting for available connection
				await asyncio.sleep(0.01)
				return mock_connections[0]  # Reuse first connection
		
		mock_pool.acquire.return_value.__aenter__ = mock_acquire
		mock_create_pool.return_value = mock_pool
		
		# Configure data source
		config = {
			'name': 'pool_test_db',
			'type': 'postgresql',
			'connection_string': 'postgresql://test:test@localhost/pooltest',
			'pool_size': 5
		}
		
		await performance_dvrl_service.register_data_source(config)
		
		# Test connection pool under concurrent load
		async def execute_with_connection(task_id: int):
			start_time = time.perf_counter()
			
			# Simulate database query
			connector = await performance_dvrl_service.connector_manager.get_connector(
				list(performance_dvrl_service.data_sources.keys())[0]
			)
			
			with patch.object(connector, 'execute_query', return_value={'data': [{'result': task_id}]}):
				result = await connector.execute_query("SELECT 1", {})
			
			end_time = time.perf_counter()
			return {
				'task_id': task_id,
				'duration_ms': (end_time - start_time) * 1000,
				'success': result is not None
			}
		
		# Execute 15 concurrent tasks (3x pool size)
		tasks = [execute_with_connection(i) for i in range(15)]
		results = await asyncio.gather(*tasks, return_exceptions=True)

		
		# Analyze connection pool performance
		successful_results = [r for r in results if r['success']]
		durations = [r['duration_ms'] for r in successful_results]
		
		assert len(successful_results) == 15  # All should succeed
		assert statistics.mean(durations) < 100  # Fast connection reuse
		assert max(durations) < 500  # No excessive waiting


class TestMemoryUsageAndLeaks:
	"""Test memory usage patterns and detect leaks"""
	
	async def test_memory_usage_during_operations(self, performance_dvrl_service):
		"""Test memory usage during typical operations"""
		process = psutil.Process()
		
		# Baseline memory
		gc.collect()
		baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
		
		# Perform operations and track memory
		memory_samples = [baseline_memory]
		
		with patch('asyncpg.create_pool') as mock_create_pool:
			mock_pool = AsyncMock()
			mock_connection = AsyncMock()
			mock_connection.fetch.return_value = [{'id': i} for i in range(1000)]
			mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
			mock_create_pool.return_value = mock_pool
			
			# Register multiple data sources
			for i in range(5):
				config = {
					'name': f'memory_test_db_{i}',
					'type': 'postgresql',
					'host': 'localhost',
					'port': 5432 + i,
					'database': f'memtest_{i}',
					'user': 'testuser',
					'password': 'testpass'
				}
				
				await performance_dvrl_service.register_data_source(config)
				
				# Sample memory after each registration
				gc.collect()
				current_memory = process.memory_info().rss / 1024 / 1024
				memory_samples.append(current_memory)
			
			# Execute queries and track memory
			for i in range(10):
				await performance_dvrl_service.execute_federated_query(
					"SELECT * FROM test_table",
					{},
					{}
				)
				
				if i % 3 == 0:  # Sample every 3rd query
					gc.collect()
					current_memory = process.memory_info().rss / 1024 / 1024
					memory_samples.append(current_memory)
		
		# Analyze memory usage
		peak_memory = max(memory_samples)
		final_memory = memory_samples[-1]
		memory_growth = final_memory - baseline_memory
		
		# Memory usage assertions
		assert peak_memory - baseline_memory < 200  # Should not use more than 200MB
		assert memory_growth < 100  # Should not have significant memory growth
		
		# Check for potential memory leaks (final memory should be reasonable)
		assert final_memory < baseline_memory + 50  # Allow some growth but not excessive
	
	async def test_garbage_collection_effectiveness(self, performance_dvrl_service):
		"""Test that garbage collection properly cleans up resources"""
		process = psutil.Process()
		
		# Force garbage collection and measure baseline
		gc.collect()
		baseline_memory = process.memory_info().rss / 1024 / 1024
		
		# Create and destroy many objects
		with patch('asyncpg.create_pool'):
			large_data_sets = []
			
			for i in range(10):
				# Create large data structures
				large_data = {
					'data_source_id': f'test_{i}',
					'large_result': [{'row': j, 'data': 'x' * 1000} for j in range(1000)],
					'metadata': {'created': datetime.now(), 'size': 1000}
				}
				large_data_sets.append(large_data)
			
			# Sample memory with large objects
			memory_with_objects = process.memory_info().rss / 1024 / 1024
			
			# Clear references
			large_data_sets.clear()
			del large_data_sets
			
			# Force garbage collection
			gc.collect()
			
			# Sample memory after cleanup
			memory_after_gc = process.memory_info().rss / 1024 / 1024
		
		# Verify garbage collection effectiveness
		memory_reclaimed = memory_with_objects - memory_after_gc
		
		# Should reclaim significant memory
		assert memory_reclaimed > 5  # Should reclaim at least 5MB
		assert memory_after_gc < baseline_memory + 20  # Should be close to baseline


class TestScalabilityLimits:
	"""Test system behavior at scale limits"""
	
	async def test_maximum_concurrent_connections(self, performance_dvrl_service, sample_data_sources):
		"""Test system behavior with maximum concurrent connections"""
		with patch('asyncpg.create_pool'), patch('aiomysql.create_pool'):
			# Register maximum number of data sources
			registered_sources = []
			
			for i, config in enumerate(sample_data_sources):
				try:
					source = await performance_dvrl_service.register_data_source(config)
					if source:
						registered_sources.append(source)
				except Exception as e:
					# Expected to hit limits at some point
					break
			
			# Test concurrent queries across all sources
			async def query_source(source_id: str):
				try:
					with patch.object(performance_dvrl_service, 'execute_federated_query') as mock_execute:
						mock_execute.return_value = Mock(
							status=Mock(value='completed'),
							rows_returned=10
						)
						
						result = await performance_dvrl_service.execute_federated_query(
							"SELECT COUNT(*) FROM test_table",
							{},
							{}
						)
						return {'source_id': source_id, 'success': True}
				except Exception:
					return {'source_id': source_id, 'success': False}
			
			# Execute queries against all registered sources
			query_tasks = [query_source(source.id) for source in registered_sources]
			query_results = await asyncio.gather(*query_tasks, return_exceptions=True)
			
			# Analyze scalability results
			successful_queries = sum(1 for r in query_results if isinstance(r, dict) and r.get('success'))
			success_rate = successful_queries / len(query_results) if query_results else 0
			
			# Should handle reasonable scale
			assert len(registered_sources) >= 5  # Should handle at least 5 data sources
			assert success_rate >= 0.8  # At least 80% success rate under load
	
	async def test_query_queue_behavior_under_load(self, performance_dvrl_service):
		"""Test query queue behavior when overloaded"""
		with patch('asyncpg.create_pool') as mock_create_pool:
			# Mock slow queries
			mock_pool = AsyncMock()
			mock_connection = AsyncMock()
			
			async def slow_fetch(*args, **kwargs):
				await asyncio.sleep(0.1)  # Simulate slow query
				return [{'result': 1}]
			
			mock_connection.fetch = slow_fetch
			mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
			mock_create_pool.return_value = mock_pool
			
			# Register data source
			config = {
				'name': 'queue_test_db',
				'type': 'postgresql',
				'host': 'localhost',
				'port': 5432,
				'database': 'queuetest',
				'user': 'testuser',
				'password': 'testpass'
			}
			
			await performance_dvrl_service.register_data_source(config)
			
			# Submit many queries simultaneously (more than max_concurrent_queries)
			num_queries = 100  # Submit more than the system can handle
			
			async def submit_query(query_id: int):
				start_time = time.perf_counter()
				try:
					result = await performance_dvrl_service.execute_federated_query(
						f"SELECT {query_id} as query_id",
						{},
						{}
					)
					end_time = time.perf_counter()
					return {
						'query_id': query_id,
						'duration_ms': (end_time - start_time) * 1000,
						'success': result is not None
					}
				except Exception as e:
					end_time = time.perf_counter()
					return {
						'query_id': query_id,
						'duration_ms': (end_time - start_time) * 1000,
						'success': False,
						'error': str(e)
					}
			
			# Execute all queries
			profiler = PerformanceProfiler()
			profiler.start()
			
			query_tasks = [submit_query(i) for i in range(num_queries)]
			results = await asyncio.gather(*query_tasks, return_exceptions=True)
			
			metrics = profiler.stop()
			
			# Analyze queue behavior
			successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
			durations = [r['duration_ms'] for r in successful_results]
			
			# System should handle overload gracefully
			assert len(successful_results) >= num_queries * 0.7  # At least 70% success
			assert metrics['duration_ms'] < 30000  # Should complete within 30 seconds
			
			if durations:
				# Queue should not cause excessive delays
				assert statistics.median(durations) < 5000  # Median response under 5 seconds


class TestResourceUtilization:
	"""Test CPU and I/O resource utilization"""
	
	async def test_cpu_utilization_during_queries(self, performance_dvrl_service):
		"""Test CPU utilization patterns during query processing"""
		process = psutil.Process()
		
		# Monitor CPU usage during query processing
		cpu_samples = []
		
		async def monitor_cpu():
			"""Background task to monitor CPU usage"""
			for _ in range(50):  # Monitor for 5 seconds
				cpu_percent = process.cpu_percent()
				cpu_samples.append(cpu_percent)
				await asyncio.sleep(0.1)
		
		with patch('asyncpg.create_pool') as mock_create_pool:
			# Mock CPU-intensive operations
			mock_pool = AsyncMock()
			mock_connection = AsyncMock()
			
			async def cpu_intensive_fetch(*args, **kwargs):
				# Simulate some CPU work
				total = 0
				for i in range(10000):
					total += i ** 2
				return [{'result': total % 1000}]
			
			mock_connection.fetch = cpu_intensive_fetch
			mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
			mock_create_pool.return_value = mock_pool
			
			# Register data source
			config = {
				'name': 'cpu_test_db',
				'type': 'postgresql',
				'host': 'localhost', 
				'port': 5432,
				'database': 'cputest',
				'user': 'testuser',
				'password': 'testpass'
			}
			
			await performance_dvrl_service.register_data_source(config)
			
			# Start CPU monitoring
			cpu_monitor_task = asyncio.create_task(monitor_cpu())
			
			# Execute multiple queries
			query_tasks = []
			for i in range(20):
				task = asyncio.create_task(
					performance_dvrl_service.execute_federated_query(
						f"SELECT * FROM cpu_intensive_table WHERE id = {i}",
						{},
						{}
					)
				)
				query_tasks.append(task)
			
			# Wait for queries to complete
			query_results = await asyncio.gather(*query_tasks, return_exceptions=True)
			
			# Stop CPU monitoring
			cpu_monitor_task.cancel()
			
			try:
				await cpu_monitor_task
			except asyncio.CancelledError:
				pass
		
		# Analyze CPU utilization
		if cpu_samples:
			avg_cpu = statistics.mean(cpu_samples)
			peak_cpu = max(cpu_samples)
			
			# CPU utilization should be reasonable
			assert avg_cpu < 90  # Average CPU should not be excessive
			assert peak_cpu < 100  # Should not max out CPU completely
			
			# Should show some CPU activity during processing
			assert avg_cpu > 1  # Should show measurable CPU usage


if __name__ == '__main__':
	pytest.main([__file__, '-v', '--tb=short'])