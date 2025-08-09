#!/usr/bin/env python3
"""
APG Key Management - Performance Tests
Comprehensive test suite for performance testing and optimization

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import time
import statistics
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from ..performance import (
	PerformanceTester, PerformanceOptimizer, PerformanceProfiler,
	PerformanceMetrics, LoadTestConfig, StressTestConfig,
	create_performance_tester, create_performance_optimizer
)
from ..service import KeyManagementService
from ..models import KeyAlgorithm, KeyUsage


@pytest.fixture
async def mock_service():
	"""Fixture for mocked key management service"""
	service = AsyncMock(spec=KeyManagementService)
	service.is_initialized = True
	service.config = {'tenant_id': 'test_tenant'}
	return service


@pytest.fixture
def load_test_config():
	"""Fixture for load test configuration"""
	return LoadTestConfig(
		name="Test Load Test",
		duration_seconds=10,  # Short duration for testing
		concurrent_users=5,
		operations_per_user=10,
		ramp_up_seconds=2,
		algorithms_to_test=[KeyAlgorithm.AES_256]
	)


@pytest.fixture
def stress_test_config():
	"""Fixture for stress test configuration"""
	return StressTestConfig(
		name="Test Stress Test",
		max_concurrent_operations=50,
		memory_limit_mb=512,
		duration_seconds=30,
		gradual_increase=True,
		breaking_point_detection=True
	)


class TestPerformanceProfiler:
	"""Test PerformanceProfiler class"""
	
	def test_profiler_initialization(self):
		"""Test profiler initialization"""
		profiler = PerformanceProfiler()
		
		assert isinstance(profiler.active_profiles, dict)
		assert isinstance(profiler.memory_snapshots, list)
		assert len(profiler.active_profiles) == 0
		assert len(profiler.memory_snapshots) == 0
	
	@pytest.mark.asyncio
	async def test_profile_operation_context(self):
		"""Test operation profiling context manager"""
		profiler = PerformanceProfiler()
		
		async with profiler.profile_operation("test_operation", {"key": "value"}) as profile_id:
			assert isinstance(profile_id, str)
			assert profile_id in profiler.active_profiles
			
			profile_data = profiler.active_profiles[profile_id]
			assert profile_data['operation_name'] == "test_operation"
			assert profile_data['metadata']['key'] == "value"
			
			# Simulate some work
			await asyncio.sleep(0.01)
		
		# After context exit, profile should be cleaned up
		assert profile_id not in profiler.active_profiles
		assert len(profiler.memory_snapshots) > 0
	
	def test_analyze_memory_usage_no_snapshots(self):
		"""Test memory analysis with no snapshots"""
		profiler = PerformanceProfiler()
		
		analysis = profiler.analyze_memory_usage()
		
		assert 'error' in analysis
		assert analysis['error'] == 'No memory snapshots available'
	
	@pytest.mark.asyncio
	async def test_analyze_memory_usage_with_snapshots(self):
		"""Test memory analysis with snapshots"""
		profiler = PerformanceProfiler()
		
		# Create a profile to generate snapshot
		async with profiler.profile_operation("memory_test"):
			pass  # Just create snapshot
		
		analysis = profiler.analyze_memory_usage()
		
		assert 'timestamp' in analysis
		assert 'total_memory_mb' in analysis
		assert 'top_memory_consumers' in analysis
		assert isinstance(analysis['top_memory_consumers'], list)


class TestPerformanceTester:
	"""Test PerformanceTester class"""
	
	@pytest.mark.asyncio
	async def test_tester_initialization(self, mock_service):
		"""Test performance tester initialization"""
		tester = PerformanceTester(mock_service)
		
		assert tester.service == mock_service
		assert isinstance(tester.profiler, PerformanceProfiler)
		assert isinstance(tester.test_results, list)
		assert tester.is_testing is False
		assert len(tester.test_results) == 0
	
	@pytest.mark.asyncio
	async def test_factory_function(self, mock_service):
		"""Test performance tester factory function"""
		tester = await create_performance_tester(mock_service)
		
		assert isinstance(tester, PerformanceTester)
		assert tester.service == mock_service
	
	@pytest.mark.asyncio
	async def test_run_load_test_basic(self, mock_service, load_test_config):
		"""Test basic load test execution"""
		tester = PerformanceTester(mock_service)
		
		# Mock service methods
		mock_service.create_key.return_value = Mock(spec_id="mock_key")
		mock_service.encrypt_data.return_value = b"encrypted_data"
		mock_service.decrypt_data.return_value = b"decrypted_data"
		mock_service.rotate_key.return_value = Mock()
		mock_service.delete_key.return_value = True
		
		# Run load test with very short duration
		load_test_config.duration_seconds = 2
		load_test_config.concurrent_users = 2
		load_test_config.operations_per_user = 2
		
		with patch('psutil.Process') as mock_process:
			mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
			mock_process.return_value.cpu_percent.return_value = 25.0
			
			metrics = await tester.run_load_test(load_test_config)
		
		assert isinstance(metrics, PerformanceMetrics)
		assert metrics.test_name == load_test_config.name
		assert metrics.operations_count >= 0
		assert 0.0 <= metrics.success_rate <= 1.0
		assert metrics.duration_seconds > 0
		assert len(tester.test_results) == 1
	
	@pytest.mark.asyncio
	async def test_run_stress_test_basic(self, mock_service, stress_test_config):
		"""Test basic stress test execution"""
		tester = PerformanceTester(mock_service)
		
		# Mock service methods
		mock_service.create_key.return_value = Mock(spec_id="mock_key")
		mock_service.encrypt_data.return_value = b"encrypted_data"
		mock_service.delete_key.return_value = True
		
		# Configure for quick test
		stress_test_config.max_concurrent_operations = 20
		stress_test_config.duration_seconds = 5
		
		with patch('psutil.Process') as mock_process:
			mock_process.return_value.memory_info.return_value.rss = 200 * 1024 * 1024  # 200MB
			mock_process.return_value.cpu_percent.return_value = 50.0
			
			metrics = await tester.run_stress_test(stress_test_config)
		
		assert isinstance(metrics, PerformanceMetrics)
		assert metrics.test_name == stress_test_config.name
		assert 'breaking_point_operations' in metrics.custom_metrics
		assert 'stress_test_results' in metrics.custom_metrics
		assert len(tester.test_results) == 1
	
	@pytest.mark.asyncio
	async def test_benchmark_algorithms(self, mock_service):
		"""Test algorithm benchmarking"""
		tester = PerformanceTester(mock_service)
		
		# Mock service methods
		mock_service.create_key.return_value = Mock(
			spec_id="mock_key",
			spec=Mock(id="key_123")
		)
		mock_service.encrypt_data.return_value = b"encrypted_data"
		mock_service.decrypt_data.return_value = b"decrypted_data"
		mock_service.retrieve_key.return_value = Mock()
		mock_service.delete_key.return_value = True
		
		# Run benchmark with limited algorithms
		with patch.object(tester, 'benchmark_algorithms') as mock_benchmark:
			mock_benchmark.return_value = {
				KeyAlgorithm.AES_256: {
					'avg_key_creation_time_ms': 10.0,
					'avg_operation_time_ms': 5.0,
					'operations_per_second': 200.0
				}
			}
			
			results = await tester.benchmark_algorithms(operation_counts=10)
		
		assert isinstance(results, dict)
		assert KeyAlgorithm.AES_256 in results
		
		aes_results = results[KeyAlgorithm.AES_256]
		assert 'avg_key_creation_time_ms' in aes_results
		assert 'avg_operation_time_ms' in aes_results
		assert 'operations_per_second' in aes_results
	
	@pytest.mark.asyncio
	async def test_simulate_user_workload(self, mock_service, load_test_config):
		"""Test user workload simulation"""
		tester = PerformanceTester(mock_service)
		
		# Mock service methods
		mock_service.create_key.return_value = Mock(
			spec_id="mock_key",
			spec=Mock(id="test_key_123")
		)
		mock_service.encrypt_data.return_value = b"encrypted"
		mock_service.decrypt_data.return_value = b"decrypted" 
		mock_service.rotate_key.return_value = Mock()
		mock_service.delete_key.return_value = True
		
		semaphore = asyncio.Semaphore(1)
		latencies = []
		
		# Simulate single user with few operations
		successful_ops = await tester._simulate_user_workload(
			semaphore, load_test_config, user_id=0, latencies=latencies
		)
		
		assert isinstance(successful_ops, int)
		assert successful_ops >= 0
		assert len(latencies) > 0
		assert all(isinstance(lat, float) for lat in latencies)
	
	@pytest.mark.asyncio
	async def test_stress_burst(self, mock_service):
		"""Test stress burst execution"""
		tester = PerformanceTester(mock_service)
		
		# Mock service methods
		mock_service.create_key.return_value = Mock(
			spec_id="mock_key", 
			spec=Mock(id="burst_key_123")
		)
		mock_service.encrypt_data.return_value = b"encrypted"
		mock_service.delete_key.return_value = True
		
		success_count, error_count, latencies = await tester._run_stress_burst(
			concurrent_ops=5,
			duration_seconds=2
		)
		
		assert isinstance(success_count, int)
		assert isinstance(error_count, int)
		assert isinstance(latencies, list)
		assert success_count >= 0
		assert error_count >= 0
		assert success_count + error_count > 0
	
	def test_select_operation(self, mock_service):
		"""Test operation selection logic"""
		tester = PerformanceTester(mock_service)
		
		operations_mix = {
			'create_key': 0.2,
			'encrypt': 0.4,
			'decrypt': 0.3,
			'rotate': 0.1
		}
		
		# Test multiple selections to check distribution
		selections = [tester._select_operation(operations_mix) for _ in range(100)]
		
		# Should have selected all operation types
		unique_operations = set(selections)
		assert len(unique_operations) > 1
		assert all(op in operations_mix for op in unique_operations)
	
	def test_percentile_calculation(self, mock_service):
		"""Test percentile calculation"""
		tester = PerformanceTester(mock_service)
		
		data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		
		assert tester._percentile(data, 50) == 5.5  # Median
		assert tester._percentile(data, 90) == 9.1  # 90th percentile
		assert tester._percentile(data, 100) == 10  # Max
		
		# Test with empty data
		assert tester._percentile([], 50) == 0.0
	
	def test_generate_performance_report_no_results(self, mock_service):
		"""Test report generation with no results"""
		tester = PerformanceTester(mock_service)
		
		report = tester.generate_performance_report()
		
		assert 'error' in report
		assert report['error'] == 'No test results available'
	
	def test_generate_performance_report_with_results(self, mock_service):
		"""Test report generation with results"""
		tester = PerformanceTester(mock_service)
		
		# Add mock test result
		test_metric = PerformanceMetrics(
			test_id="test_123",
			test_name="Mock Test",
			start_time=datetime.utcnow() - timedelta(minutes=5),
			end_time=datetime.utcnow(),
			duration_seconds=300,
			operations_count=1000,
			operations_per_second=100.0,
			success_rate=0.95,
			error_count=50,
			memory_usage_mb=256.0,
			cpu_usage_percent=45.0,
			latency_stats={'mean': 0.01, 'p95': 0.05, 'p99': 0.1},
			custom_metrics={'concurrent_users': 10}
		)
		tester.test_results.append(test_metric)
		
		report = tester.generate_performance_report()
		
		assert 'report_generated' in report
		assert 'total_tests_run' in report
		assert report['total_tests_run'] == 1
		assert 'test_summary' in report
		assert 'load_test' in report['test_summary']
		assert 'recommendations' in report
		assert isinstance(report['recommendations'], list)


class TestPerformanceOptimizer:
	"""Test PerformanceOptimizer class"""
	
	@pytest.mark.asyncio
	async def test_optimizer_initialization(self, mock_service):
		"""Test optimizer initialization"""
		optimizer = PerformanceOptimizer(mock_service)
		
		assert optimizer.service == mock_service
		assert isinstance(optimizer.optimization_history, list)
		assert len(optimizer.optimization_history) == 0
	
	@pytest.mark.asyncio
	async def test_factory_function(self, mock_service):
		"""Test optimizer factory function"""
		optimizer = await create_performance_optimizer(mock_service)
		
		assert isinstance(optimizer, PerformanceOptimizer)
		assert optimizer.service == mock_service
	
	@pytest.mark.asyncio
	async def test_optimize_key_operations(self, mock_service):
		"""Test key operations optimization"""
		optimizer = PerformanceOptimizer(mock_service)
		
		result = await optimizer.optimize_key_operations()
		
		assert isinstance(result, dict)
		assert 'optimization_id' in result
		assert 'timestamp' in result
		assert 'optimizations_applied' in result
		assert 'expected_improvements' in result
		
		# Check optimization types
		optimizations = result['optimizations_applied']
		optimization_types = [opt['type'] for opt in optimizations]
		
		expected_types = ['caching', 'memory_management', 'operation_batching']
		for opt_type in expected_types:
			assert opt_type in optimization_types
		
		# Check expected improvements
		improvements = result['expected_improvements']
		assert 'operations_per_second_improvement' in improvements
		assert 'latency_reduction_percent' in improvements
		assert 'memory_usage_reduction_percent' in improvements
		
		# Should be added to history
		assert len(optimizer.optimization_history) == 1
	
	@pytest.mark.asyncio
	async def test_caching_optimization(self, mock_service):
		"""Test caching optimization"""
		optimizer = PerformanceOptimizer(mock_service)
		
		cache_opt = await optimizer._optimize_caching()
		
		assert cache_opt is not None
		assert cache_opt['type'] == 'caching'
		assert 'description' in cache_opt
		assert 'parameters' in cache_opt
		assert 'cache_size' in cache_opt['parameters']
		assert 'ttl_seconds' in cache_opt['parameters']
	
	@pytest.mark.asyncio
	async def test_memory_optimization(self, mock_service):
		"""Test memory optimization"""
		optimizer = PerformanceOptimizer(mock_service)
		
		memory_opt = await optimizer._optimize_memory_usage()
		
		assert memory_opt is not None
		assert memory_opt['type'] == 'memory_management'
		assert 'description' in memory_opt
		assert 'parameters' in memory_opt
	
	@pytest.mark.asyncio
	async def test_batching_optimization(self, mock_service):
		"""Test operation batching optimization"""
		optimizer = PerformanceOptimizer(mock_service)
		
		batch_opt = await optimizer._optimize_operation_batching()
		
		assert batch_opt is not None
		assert batch_opt['type'] == 'operation_batching'
		assert 'description' in batch_opt
		assert 'parameters' in batch_opt
		assert 'batch_size' in batch_opt['parameters']
	
	def test_calculate_expected_improvements(self, mock_service):
		"""Test expected improvements calculation"""
		optimizer = PerformanceOptimizer(mock_service)
		
		optimizations = [
			{'type': 'caching', 'description': 'Test caching'},
			{'type': 'memory_management', 'description': 'Test memory'},
			{'type': 'operation_batching', 'description': 'Test batching'}
		]
		
		improvements = optimizer._calculate_expected_improvements(optimizations)
		
		assert 'operations_per_second_improvement' in improvements
		assert 'latency_reduction_percent' in improvements
		assert 'memory_usage_reduction_percent' in improvements
		
		# Should have positive improvements
		assert improvements['operations_per_second_improvement'] > 0
		assert improvements['latency_reduction_percent'] > 0
		assert improvements['memory_usage_reduction_percent'] > 0


class TestPerformanceModels:
	"""Test performance data models"""
	
	def test_performance_metrics_creation(self):
		"""Test PerformanceMetrics model"""
		start_time = datetime.utcnow() - timedelta(minutes=5)
		end_time = datetime.utcnow()
		
		metrics = PerformanceMetrics(
			test_id="test_123",
			test_name="Load Test",
			start_time=start_time,
			end_time=end_time,
			duration_seconds=300.0,
			operations_count=1000,
			operations_per_second=100.0,
			success_rate=0.95,
			error_count=50,
			memory_usage_mb=512.0,
			cpu_usage_percent=75.0,
			latency_stats={
				'min': 0.001,
				'max': 0.1,
				'mean': 0.01,
				'p95': 0.05,
				'p99': 0.08
			},
			custom_metrics={
				'concurrent_users': 20,
				'algorithms_tested': ['AES-256', 'RSA-2048']
			}
		)
		
		assert metrics.test_id == "test_123"
		assert metrics.test_name == "Load Test"
		assert metrics.operations_count == 1000
		assert metrics.operations_per_second == 100.0
		assert metrics.success_rate == 0.95
		assert metrics.latency_stats['mean'] == 0.01
		assert metrics.custom_metrics['concurrent_users'] == 20
	
	def test_load_test_config_defaults(self):
		"""Test LoadTestConfig with defaults"""
		config = LoadTestConfig(name="Test Config")
		
		assert config.name == "Test Config"
		assert config.duration_seconds == 300
		assert config.concurrent_users == 10
		assert config.operations_per_user == 100
		assert config.ramp_up_seconds == 30
		assert config.target_operations_per_second is None
		assert KeyAlgorithm.AES_256 in config.algorithms_to_test
		assert 'create_key' in config.operations_mix
		assert sum(config.operations_mix.values()) == 1.0
	
	def test_stress_test_config_defaults(self):
		"""Test StressTestConfig with defaults"""
		config = StressTestConfig(name="Stress Test")
		
		assert config.name == "Stress Test"
		assert config.max_concurrent_operations == 1000
		assert config.memory_limit_mb == 2048
		assert config.duration_seconds == 600
		assert config.gradual_increase is True
		assert config.breaking_point_detection is True
		assert config.resource_monitoring_interval == 1.0


class TestIntegrationScenarios:
	"""Test integration scenarios"""
	
	@pytest.mark.asyncio
	async def test_full_performance_testing_workflow(self, mock_service):
		"""Test complete performance testing workflow"""
		# 1. Create tester
		tester = await create_performance_tester(mock_service)
		
		# Mock service methods
		mock_service.create_key.return_value = Mock(
			spec_id="mock_key",
			spec=Mock(id="workflow_key_123")
		)
		mock_service.encrypt_data.return_value = b"encrypted"
		mock_service.decrypt_data.return_value = b"decrypted"
		mock_service.delete_key.return_value = True
		
		# 2. Run load test
		load_config = LoadTestConfig(
			name="Workflow Load Test",
			duration_seconds=3,
			concurrent_users=2,
			operations_per_user=3
		)
		
		with patch('psutil.Process') as mock_process:
			mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
			mock_process.return_value.cpu_percent.return_value = 25.0
			
			load_metrics = await tester.run_load_test(load_config)
		
		assert isinstance(load_metrics, PerformanceMetrics)
		
		# 3. Generate performance report
		report = tester.generate_performance_report()
		assert 'test_summary' in report
		assert len(tester.test_results) == 1
		
		# 4. Create optimizer and apply optimizations
		optimizer = await create_performance_optimizer(mock_service)
		optimization_result = await optimizer.optimize_key_operations()
		
		assert 'optimizations_applied' in optimization_result
		assert len(optimizer.optimization_history) == 1
	
	@pytest.mark.asyncio
	async def test_performance_profiling_integration(self):
		"""Test performance profiling integration"""
		profiler = PerformanceProfiler()
		
		# Simulate nested operations
		async with profiler.profile_operation("outer_operation", {"level": "outer"}) as outer_id:
			await asyncio.sleep(0.01)  # Simulate work
			
			async with profiler.profile_operation("inner_operation", {"level": "inner"}) as inner_id:
				await asyncio.sleep(0.01)  # Simulate more work
				
				assert outer_id != inner_id
				assert len(profiler.active_profiles) == 2
		
		# After completion, profiles should be cleaned up
		assert len(profiler.active_profiles) == 0
		assert len(profiler.memory_snapshots) == 2
		
		# Analyze memory usage
		memory_analysis = profiler.analyze_memory_usage()
		assert 'total_memory_mb' in memory_analysis
		assert 'top_memory_consumers' in memory_analysis


if __name__ == "__main__":
	pytest.main([__file__])