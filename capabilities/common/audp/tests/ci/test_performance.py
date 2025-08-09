"""
Audio Processing Performance Tests

Unit tests for performance optimization, caching, load balancing,
and scaling components.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from ...performance import (
	PerformanceMetrics, ResourceMonitor, CacheManager, LoadBalancer,
	PerformanceOptimizer, AutoScaler, performance_optimized,
	create_performance_optimizer
)


class TestPerformanceMetrics:
	"""Test PerformanceMetrics data structure"""
	
	def test_metrics_creation(self):
		"""Test creating performance metrics"""
		metrics = PerformanceMetrics(
			operation_type="transcription",
			tenant_id="test_tenant",
			start_time=time.time(),
			status="in_progress"
		)
		
		assert metrics.operation_type == "transcription"
		assert metrics.tenant_id == "test_tenant"
		assert metrics.status == "in_progress"
		assert metrics.end_time is None
		assert metrics.cache_hit is False
		assert metrics.resource_usage is None
	
	def test_metrics_completion(self):
		"""Test metrics completion"""
		start_time = time.time()
		metrics = PerformanceMetrics(
			operation_type="synthesis",
			tenant_id="test_tenant",
			start_time=start_time
		)
		
		# Simulate completion
		metrics.end_time = time.time()
		metrics.status = "completed"
		metrics.cache_hit = True
		metrics.resource_usage = {"cpu": 45.2, "memory": 60.1}
		
		assert metrics.end_time > start_time
		assert metrics.status == "completed"
		assert metrics.cache_hit is True
		assert "cpu" in metrics.resource_usage


class TestResourceMonitor:
	"""Test ResourceMonitor functionality"""
	
	@pytest.fixture
	def resource_monitor(self):
		"""Create resource monitor instance"""
		return ResourceMonitor()
	
	def test_monitor_initialization(self, resource_monitor):
		"""Test resource monitor initialization"""
		assert resource_monitor.cpu_threshold == 80.0
		assert resource_monitor.memory_threshold == 85.0
		assert resource_monitor.disk_threshold == 90.0
		assert resource_monitor.monitoring_interval == 10.0
		assert resource_monitor._monitoring_task is None
	
	async def test_start_stop_monitoring(self, resource_monitor):
		"""Test starting and stopping monitoring"""
		# Start monitoring
		await resource_monitor.start_monitoring()
		assert resource_monitor._monitoring_task is not None
		assert not resource_monitor._monitoring_task.done()
		
		# Stop monitoring
		await resource_monitor.stop_monitoring()
		assert resource_monitor._monitoring_task is None
	
	async def test_alert_generation(self, resource_monitor):
		"""Test alert generation"""
		# Generate test alert
		await resource_monitor._generate_alert("cpu", 95.0, 80.0)
		
		alerts = resource_monitor.get_recent_alerts(limit=5)
		assert len(alerts) == 1
		
		alert = alerts[0]
		assert alert['type'] == 'resource_alert'
		assert alert['resource'] == 'cpu'
		assert alert['current_value'] == 95.0
		assert alert['threshold'] == 80.0
		assert alert['severity'] in ['medium', 'high']
	
	def test_get_recent_alerts(self, resource_monitor):
		"""Test getting recent alerts"""
		# Initially no alerts
		alerts = resource_monitor.get_recent_alerts()
		assert len(alerts) == 0
		
		# Add some test alerts
		for i in range(5):
			alert = {
				'timestamp': datetime.utcnow(),
				'type': 'resource_alert',
				'resource': 'memory',
				'current_value': 90.0 + i,
				'threshold': 85.0,
				'severity': 'high'
			}
			resource_monitor._alerts.append(alert)
		
		# Get recent alerts
		alerts = resource_monitor.get_recent_alerts(limit=3)
		assert len(alerts) == 3


class TestCacheManager:
	"""Test CacheManager functionality"""
	
	@pytest.fixture
	def cache_manager(self):
		"""Create cache manager instance"""
		# Use mock Redis for testing
		with patch('redis.from_url') as mock_redis:
			mock_redis_instance = MagicMock()
			mock_redis.return_value = mock_redis_instance
			cache_manager = CacheManager("redis://localhost:6379/0")
			cache_manager.redis_client = mock_redis_instance
			return cache_manager
	
	async def test_local_cache_get_set(self, cache_manager):
		"""Test local cache get/set operations"""
		key = "test_key"
		value = "test_value"
		tenant_id = "test_tenant"
		
		# Initially cache miss
		result = await cache_manager.get(key, tenant_id)
		assert result is None
		
		# Set value
		await cache_manager.set(key, value, tenant_id)
		
		# Should now hit local cache
		result = await cache_manager.get(key, tenant_id)
		assert result == value
	
	async def test_redis_cache_fallback(self, cache_manager):
		"""Test Redis cache fallback"""
		key = "redis_test_key"
		value = "redis_test_value"
		tenant_id = "test_tenant"
		
		# Mock Redis get to return value
		cache_manager.redis_client.get.return_value = value
		
		# Should get from Redis
		result = await cache_manager.get(key, tenant_id)
		assert result == value
		
		# Redis get should have been called
		cache_manager.redis_client.get.assert_called_once()
	
	async def test_cache_invalidation(self, cache_manager):
		"""Test cache invalidation"""
		# Set up local cache entries
		await cache_manager._store_local("tenant_1:key1", "value1")
		await cache_manager._store_local("tenant_1:key2", "value2")
		await cache_manager._store_local("tenant_2:key1", "value3")
		
		# Mock Redis keys
		cache_manager.redis_client.keys.return_value = ["tenant_1:key1", "tenant_1:key2"]
		
		# Invalidate tenant_1 entries
		invalidated = await cache_manager.invalidate("key", "tenant_1")
		
		# Should have invalidated local entries
		assert invalidated >= 2
		
		# Redis delete should have been called
		cache_manager.redis_client.delete.assert_called_once()
	
	def test_cache_stats(self, cache_manager):
		"""Test cache statistics"""
		# Initialize stats
		cache_manager.cache_stats["transcription"]["hits"] = 10
		cache_manager.cache_stats["transcription"]["misses"] = 2
		cache_manager.cache_stats["synthesis"]["hits"] = 5
		cache_manager.cache_stats["synthesis"]["misses"] = 1
		
		stats = cache_manager.get_cache_stats()
		
		assert "transcription" in stats
		assert "synthesis" in stats
		assert stats["transcription"]["hits"] == 10
		assert stats["transcription"]["misses"] == 2


class TestLoadBalancer:
	"""Test LoadBalancer functionality"""
	
	@pytest.fixture
	def load_balancer(self):
		"""Create load balancer instance"""
		return LoadBalancer()
	
	def test_initialization(self, load_balancer):
		"""Test load balancer initialization"""
		assert "transcription" in load_balancer.worker_pools
		assert "synthesis" in load_balancer.worker_pools
		assert "analysis" in load_balancer.worker_pools
		assert "enhancement" in load_balancer.process_pools
		assert "voice_cloning" in load_balancer.process_pools
		
		# All pools should be healthy initially
		for pool_name in load_balancer.worker_health:
			assert load_balancer.worker_health[pool_name] is True
	
	async def test_submit_job_success(self, load_balancer):
		"""Test successful job submission"""
		def test_function(x, y):
			return x + y
		
		result = await load_balancer.submit_job("transcription", test_function, 5, 3)
		assert result == 8
		
		# Job count should be back to 0
		assert load_balancer.active_jobs["transcription"] == 0
	
	async def test_submit_job_invalid_operation(self, load_balancer):
		"""Test job submission with invalid operation type"""
		def test_function():
			return "test"
		
		with pytest.raises(ValueError, match="Unknown operation type"):
			await load_balancer.submit_job("invalid_operation", test_function)
	
	async def test_submit_job_unhealthy_worker(self, load_balancer):
		"""Test job submission to unhealthy worker"""
		# Mark worker as unhealthy
		load_balancer.worker_health["transcription"] = False
		
		def test_function():
			return "test"
		
		with pytest.raises(RuntimeError, match="Worker pool 'transcription' is unhealthy"):
			await load_balancer.submit_job("transcription", test_function)
	
	def test_get_load_stats(self, load_balancer):
		"""Test getting load statistics"""
		stats = load_balancer.get_load_stats()
		
		assert "active_jobs" in stats
		assert "worker_health" in stats
		assert "pool_sizes" in stats
		
		assert isinstance(stats["active_jobs"], dict)
		assert isinstance(stats["worker_health"], dict)
		assert isinstance(stats["pool_sizes"], dict)
	
	async def test_health_check(self, load_balancer):
		"""Test health check functionality"""
		health_status = await load_balancer.health_check()
		
		# Should have health status for all pools
		assert "transcription" in health_status
		assert "synthesis" in health_status
		assert "analysis" in health_status
		assert "enhancement" in health_status
		assert "voice_cloning" in health_status


class TestPerformanceOptimizer:
	"""Test PerformanceOptimizer functionality"""
	
	@pytest.fixture
	async def performance_optimizer(self):
		"""Create performance optimizer instance"""
		with patch('redis.from_url'):
			optimizer = PerformanceOptimizer("redis://localhost:6379/0")
			await optimizer.initialize()
			return optimizer
	
	async def test_initialization(self, performance_optimizer):
		"""Test performance optimizer initialization"""
		assert performance_optimizer.resource_monitor is not None
		assert performance_optimizer.cache_manager is not None
		assert performance_optimizer.load_balancer is not None
		assert isinstance(performance_optimizer.metrics, list)
	
	async def test_track_operation_success(self, performance_optimizer):
		"""Test operation tracking for successful operation"""
		async with performance_optimizer.track_operation("test_operation", "test_tenant") as metrics:
			await asyncio.sleep(0.1)  # Simulate work
			assert metrics.operation_type == "test_operation"
			assert metrics.tenant_id == "test_tenant"
			assert metrics.status == "in_progress"
		
		# After completion
		assert metrics.status == "completed"
		assert metrics.end_time is not None
		assert metrics.end_time > metrics.start_time
	
	async def test_track_operation_failure(self, performance_optimizer):
		"""Test operation tracking for failed operation"""
		try:
			async with performance_optimizer.track_operation("test_operation", "test_tenant") as metrics:
				raise ValueError("Test error")
		except ValueError:
			pass
		
		# Should mark as failed
		assert metrics.status == "failed"
		assert metrics.end_time is not None
	
	async def test_optimize_processing_with_cache_hit(self, performance_optimizer):
		"""Test processing optimization with cache hit"""
		# Mock cache hit
		performance_optimizer.cache_manager.get = AsyncMock(return_value="cached_result")
		
		def test_function():
			return "computed_result"
		
		result = await performance_optimizer.optimize_processing(
			"test_operation", test_function, tenant_id="test_tenant"
		)
		
		assert result == "cached_result"
		# Function should not have been called due to cache hit
		performance_optimizer.cache_manager.get.assert_called_once()
	
	async def test_optimize_processing_with_cache_miss(self, performance_optimizer):
		"""Test processing optimization with cache miss"""
		# Mock cache miss
		performance_optimizer.cache_manager.get = AsyncMock(return_value=None)
		performance_optimizer.cache_manager.set = AsyncMock()
		performance_optimizer.load_balancer.submit_job = AsyncMock(return_value="computed_result")
		
		def test_function():
			return "computed_result"
		
		result = await performance_optimizer.optimize_processing(
			"test_operation", test_function, tenant_id="test_tenant"
		)
		
		assert result == "computed_result"
		# Should have called load balancer and cached result
		performance_optimizer.load_balancer.submit_job.assert_called_once()
		performance_optimizer.cache_manager.set.assert_called_once()
	
	def test_generate_cache_key(self, performance_optimizer):
		"""Test cache key generation"""
		# Test with audio source
		key1 = performance_optimizer._generate_cache_key(
			"transcription", 
			(), 
			{"audio_source": {"file_path": "/tmp/test.wav"}, "language_code": "en-US"}
		)
		assert "transcription" in key1
		assert "/tmp/test.wav" in key1
		assert "language_code_en-US" in key1
		
		# Test with text content
		key2 = performance_optimizer._generate_cache_key(
			"synthesis",
			(),
			{"text_content": "Hello world", "voice_id": "voice_001"}
		)
		assert "synthesis" in key2
		assert "voice_id_voice_001" in key2
	
	async def test_get_performance_report(self, performance_optimizer):
		"""Test performance report generation"""
		# Add some test metrics
		for i in range(10):
			metrics = PerformanceMetrics(
				operation_type="test_operation",
				tenant_id="test_tenant",
				start_time=time.time() - 3600,  # 1 hour ago
				end_time=time.time() - 3500,    # 59 minutes ago
				status="completed" if i < 8 else "failed"
			)
			performance_optimizer.metrics.append(metrics)
		
		report = await performance_optimizer.get_performance_report(hours=2)
		
		assert "operations_stats" in report
		assert "total_metrics_collected" in report
		assert "resource_alerts" in report
		assert "cache_stats" in report
		assert "load_balancer_stats" in report
		
		# Check operation stats
		op_stats = report["operations_stats"]["test_operation"]
		assert op_stats["total_requests"] == 10
		assert op_stats["successful_requests"] == 8
		assert op_stats["failed_requests"] == 2


class TestAutoScaler:
	"""Test AutoScaler functionality"""
	
	@pytest.fixture
	def auto_scaler(self):
		"""Create auto scaler instance"""
		with patch('redis.from_url'):
			optimizer = PerformanceOptimizer("redis://localhost:6379/0")
			return AutoScaler(optimizer)
	
	async def test_evaluate_scaling_needs_stable(self, auto_scaler):
		"""Test scaling evaluation with stable load"""
		# Mock stable load stats
		auto_scaler.optimizer.load_balancer.get_load_stats = MagicMock(return_value={
			'active_jobs': {'transcription': 5, 'synthesis': 3},
			'worker_health': {'transcription': True, 'synthesis': True}
		})
		
		auto_scaler.optimizer.get_performance_report = AsyncMock(return_value={
			'operations_stats': {
				'transcription': {'avg_duration': 10.0},
				'synthesis': {'avg_duration': 8.0}
			}
		})
		
		recommendations = await auto_scaler.evaluate_scaling_needs()
		
		assert recommendations['current_status'] == 'stable'
		assert len(recommendations['scale_up']) == 0
		assert len(recommendations['scale_down']) == 0
	
	async def test_evaluate_scaling_needs_scale_up(self, auto_scaler):
		"""Test scaling evaluation requiring scale up"""
		# Mock high load stats
		auto_scaler.optimizer.load_balancer.get_load_stats = MagicMock(return_value={
			'active_jobs': {'transcription': 60, 'synthesis': 10},  # High transcription load
			'worker_health': {'transcription': True, 'synthesis': True}
		})
		
		auto_scaler.optimizer.get_performance_report = AsyncMock(return_value={
			'operations_stats': {
				'transcription': {'avg_duration': 35.0},  # High duration
				'synthesis': {'avg_duration': 8.0}
			}
		})
		
		recommendations = await auto_scaler.evaluate_scaling_needs()
		
		assert recommendations['current_status'] == 'scale_up_needed'
		assert len(recommendations['scale_up']) == 1
		
		scale_up_rec = recommendations['scale_up'][0]
		assert scale_up_rec['operation_type'] == 'transcription'
		assert scale_up_rec['reason'] == 'high_load'


class TestPerformanceDecorators:
	"""Test performance decorators"""
	
	def test_performance_optimized_decorator(self):
		"""Test performance optimized decorator"""
		@performance_optimized("test_operation")
		async def test_function(value: int) -> int:
			return value * 2
		
		# Function should be wrapped
		assert hasattr(test_function, '__wrapped__')
		assert callable(test_function)
	
	async def test_performance_optimized_execution(self):
		"""Test performance optimized decorator execution"""
		call_count = 0
		
		@performance_optimized("test_operation")
		async def test_function(value: int, tenant_id: str = "default") -> int:
			nonlocal call_count
			call_count += 1
			return value * 2
		
		# Mock the optimizer to avoid Redis dependency
		with patch('...performance.create_performance_optimizer') as mock_create:
			mock_optimizer = MagicMock()
			mock_optimizer.optimize_processing = AsyncMock(return_value=10)
			mock_optimizer.initialize = AsyncMock()
			mock_create.return_value = mock_optimizer
			
			result = await test_function(5, tenant_id="test_tenant")
			
			assert result == 10
			mock_optimizer.optimize_processing.assert_called_once()


class TestPerformanceIntegration:
	"""Test performance component integration"""
	
	async def test_full_optimization_pipeline(self):
		"""Test complete optimization pipeline"""
		with patch('redis.from_url'):
			optimizer = PerformanceOptimizer("redis://localhost:6379/0")
			
			# Mock dependencies
			optimizer.cache_manager.get = AsyncMock(return_value=None)
			optimizer.cache_manager.set = AsyncMock()
			optimizer.load_balancer.submit_job = AsyncMock(return_value="result")
			
			def test_operation():
				return "computed_result"
			
			# Track operation with optimization
			async with optimizer.track_operation("test_op", "tenant") as metrics:
				result = await optimizer.optimize_processing(
					"test_op", test_operation, tenant_id="tenant"
				)
			
			assert result == "computed_result"
			assert metrics.status == "completed"
			optimizer.cache_manager.get.assert_called_once()
			optimizer.cache_manager.set.assert_called_once()
			optimizer.load_balancer.submit_job.assert_called_once()
	
	def test_create_performance_optimizer_factory(self):
		"""Test performance optimizer factory function"""
		with patch('redis.from_url'):
			optimizer = create_performance_optimizer("redis://localhost:6379/0")
			
			assert isinstance(optimizer, PerformanceOptimizer)
			assert optimizer.cache_manager is not None
			assert optimizer.load_balancer is not None
			assert optimizer.resource_monitor is not None


class TestPerformanceEdgeCases:
	"""Test performance optimization edge cases"""
	
	async def test_cache_manager_redis_failure(self):
		"""Test cache manager behavior when Redis fails"""
		with patch('redis.from_url') as mock_redis:
			mock_redis_instance = MagicMock()
			mock_redis_instance.get.side_effect = Exception("Redis connection failed")
			mock_redis.return_value = mock_redis_instance
			
			cache_manager = CacheManager("redis://localhost:6379/0")
			
			# Should handle Redis failure gracefully
			result = await cache_manager.get("test_key", "test_tenant")
			assert result is None
	
	async def test_load_balancer_worker_failure(self):
		"""Test load balancer behavior when worker fails"""
		load_balancer = LoadBalancer()
		
		def failing_function():
			raise RuntimeError("Worker failed")
		
		with pytest.raises(RuntimeError, match="Worker failed"):
			await load_balancer.submit_job("transcription", failing_function)
		
		# Active job count should be reset to 0
		assert load_balancer.active_jobs["transcription"] == 0
	
	async def test_resource_monitor_alert_overflow(self):
		"""Test resource monitor alert buffer overflow"""
		monitor = ResourceMonitor()
		
		# Fill alert buffer beyond capacity
		for i in range(150):  # More than maxlen=100
			await monitor._generate_alert("cpu", 95.0, 80.0)
		
		alerts = monitor.get_recent_alerts(limit=200)
		assert len(alerts) <= 100  # Should not exceed maxlen