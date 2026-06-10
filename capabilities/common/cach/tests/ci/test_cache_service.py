#!/usr/bin/env python3
"""
APG Cache Management (CACH) - Cache Service Tests
Comprehensive tests for core cache service functionality

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
import json

from service import CacheService, CacheServiceConfig
from models import CacheEntry, SecurityLevel


class TestCacheService:
    """Test suite for CacheService core functionality"""
    
    async def test_service_initialization(self, cache_config):
        """Test cache service initialization"""
        service = CacheService(cache_config)
        
        # Service should not be running initially
        assert not service.running
        
        # Initialize the service
        await service.initialize()
        
        # Service should now be running
        assert service.running
        assert service._cache_store is not None
        assert service._entry_metadata is not None
        
        # Cleanup
        await service.shutdown()
        assert not service.running
    
    async def test_basic_cache_operations(self, cache_service):
        """Test basic cache get/set/delete operations"""
        key = "test:basic_ops"
        value = "test_value"
        namespace = "test"
        
        # Test SET operation
        success = await cache_service.set(key, value, namespace=namespace)
        assert success
        
        # Test GET operation
        retrieved_value = await cache_service.get(key, namespace=namespace)
        assert retrieved_value == value
        
        # Test key exists
        exists = await cache_service.exists(key, namespace=namespace)
        assert exists
        
        # Test DELETE operation
        deleted = await cache_service.delete(key, namespace=namespace)
        assert deleted
        
        # Verify key no longer exists
        retrieved_value = await cache_service.get(key, namespace=namespace)
        assert retrieved_value is None
        
        exists = await cache_service.exists(key, namespace=namespace)
        assert not exists
    
    async def test_ttl_expiration(self, cache_service):
        """Test TTL (Time To Live) functionality"""
        key = "test:ttl"
        value = "expiring_value"
        ttl = 1  # 1 second
        
        # Set with TTL
        success = await cache_service.set(key, value, ttl_seconds=ttl)
        assert success
        
        # Should be retrievable immediately
        retrieved = await cache_service.get(key)
        assert retrieved == value
        
        # Wait for expiration
        await asyncio.sleep(1.5)
        
        # Should be expired now
        retrieved = await cache_service.get(key)
        assert retrieved is None
    
    async def test_namespaces(self, cache_service):
        """Test namespace isolation"""
        key = "same_key"
        value1 = "namespace1_value"
        value2 = "namespace2_value"
        namespace1 = "ns1"
        namespace2 = "ns2"
        
        # Set same key in different namespaces
        await cache_service.set(key, value1, namespace=namespace1)
        await cache_service.set(key, value2, namespace=namespace2)
        
        # Values should be isolated by namespace
        retrieved1 = await cache_service.get(key, namespace=namespace1)
        retrieved2 = await cache_service.get(key, namespace=namespace2)
        
        assert retrieved1 == value1
        assert retrieved2 == value2
        assert retrieved1 != retrieved2
        
        # Delete from one namespace shouldn't affect the other
        await cache_service.delete(key, namespace=namespace1)
        
        retrieved1 = await cache_service.get(key, namespace=namespace1)
        retrieved2 = await cache_service.get(key, namespace=namespace2)
        
        assert retrieved1 is None
        assert retrieved2 == value2
    
    async def test_batch_operations(self, cache_service):
        """Test batch get/set operations"""
        keys = [f"batch:key:{i}" for i in range(10)]
        values = [f"batch_value_{i}" for i in range(10)]
        batch_data = dict(zip(keys, values))
        
        # Batch set
        results = await cache_service.set_batch(batch_data)
        assert len(results) == len(batch_data)
        assert all(results.values())  # All should succeed
        
        # Batch get
        retrieved = await cache_service.get_batch(keys)
        assert len(retrieved) == len(keys)
        
        for key in keys:
            assert retrieved[key] == batch_data[key]
        
        # Batch delete
        deleted_results = await cache_service.delete_batch(keys)
        assert len(deleted_results) == len(keys)
        assert all(deleted_results.values())  # All should succeed
        
        # Verify all deleted
        retrieved_after_delete = await cache_service.get_batch(keys)
        assert all(v is None for v in retrieved_after_delete.values())
    
    async def test_cache_statistics(self, cache_service):
        """Test cache statistics functionality"""
        # Populate cache with test data
        for i in range(50):
            await cache_service.set(f"stats:key:{i}", f"value_{i}")
        
        # Generate some hits and misses
        for i in range(25):
            await cache_service.get(f"stats:key:{i}")  # Hits
        
        for i in range(25, 50):
            await cache_service.get(f"stats:missing:{i}")  # Misses
        
        # Get statistics
        stats = await cache_service.get_stats()
        
        # Verify stats structure
        assert 'total_entries' in stats
        assert 'hit_rate' in stats
        assert 'miss_rate' in stats
        assert 'memory_usage_mb' in stats
        assert 'operations_per_second' in stats
        
        # Verify stats values
        assert stats['total_entries'] >= 50
        assert 0.0 <= stats['hit_rate'] <= 1.0
        assert 0.0 <= stats['miss_rate'] <= 1.0
        assert stats['hit_rate'] + stats['miss_rate'] == pytest.approx(1.0, abs=0.01)
    
    async def test_namespace_operations(self, cache_service):
        """Test namespace-specific operations"""
        namespace = "test_namespace"
        
        # Populate namespace with data
        for i in range(20):
            await cache_service.set(f"ns:key:{i}", f"value_{i}", namespace=namespace)
        
        # Get namespace size
        size = await cache_service.get_namespace_size(namespace)
        assert size == 20
        
        # List namespace keys
        keys = await cache_service.list_namespace_keys(namespace)
        assert len(keys) == 20
        assert all(key.startswith("ns:key:") for key in keys)
        
        # Clear namespace
        cleared_count = await cache_service.clear_namespace(namespace)
        assert cleared_count == 20
        
        # Verify namespace is empty
        size_after_clear = await cache_service.get_namespace_size(namespace)
        assert size_after_clear == 0
    
    async def test_pattern_matching(self, cache_service):
        """Test key pattern matching functionality"""
        # Set up test data with patterns
        await cache_service.set("user:123:profile", "profile_data")
        await cache_service.set("user:123:settings", "settings_data")
        await cache_service.set("user:456:profile", "profile_data_2")
        await cache_service.set("product:789:info", "product_info")
        
        # Test pattern matching
        user_keys = await cache_service.get_keys_by_pattern("user:*")
        assert len(user_keys) == 3
        assert all("user:" in key for key in user_keys)
        
        profile_keys = await cache_service.get_keys_by_pattern("*:profile")
        assert len(profile_keys) == 2
        assert all(key.endswith(":profile") for key in profile_keys)
        
        user_123_keys = await cache_service.get_keys_by_pattern("user:123:*")
        assert len(user_123_keys) == 2
        assert all(key.startswith("user:123:") for key in user_123_keys)
    
    async def test_memory_management(self, cache_service):
        """Test cache memory management and limits"""
        # Get initial stats
        initial_stats = await cache_service.get_stats()
        initial_entries = initial_stats['total_entries']
        
        # Fill cache to near capacity
        large_value = "x" * 1024  # 1KB value
        keys_added = []
        
        for i in range(100):
            key = f"memory:test:{i}"
            await cache_service.set(key, large_value)
            keys_added.append(key)
        
        # Check memory usage increased
        current_stats = await cache_service.get_stats()
        assert current_stats['total_entries'] > initial_entries
        assert current_stats['memory_usage_mb'] > initial_stats.get('memory_usage_mb', 0)
        
        # Verify all keys are accessible
        for key in keys_added[:50]:  # Check first 50
            value = await cache_service.get(key)
            assert value == large_value
    
    async def test_error_handling(self, cache_service):
        """Test error handling and edge cases"""
        # Test with invalid inputs
        with pytest.raises((ValueError, TypeError)):
            await cache_service.set("", "value")  # Empty key
        
        with pytest.raises((ValueError, TypeError)):
            await cache_service.set(None, "value")  # None key
        
        # Test with very large values
        very_large_value = "x" * (10 * 1024 * 1024)  # 10MB
        success = await cache_service.set("large_value", very_large_value)
        # Depending on configuration, this might succeed or fail
        if success:
            retrieved = await cache_service.get("large_value")
            assert retrieved == very_large_value
    
    async def test_concurrent_operations(self, cache_service):
        """Test concurrent cache operations"""
        async def set_operation(i):
            key = f"concurrent:set:{i}"
            value = f"value_{i}"
            return await cache_service.set(key, value)
        
        async def get_operation(i):
            key = f"concurrent:get:{i}"
            return await cache_service.get(key)
        
        # First, set some data
        set_tasks = [set_operation(i) for i in range(50)]
        set_results = await asyncio.gather(*set_tasks, return_exceptions=True)

        assert all(set_results)
        
        # Then, concurrently read the data
        get_tasks = [get_operation(i) for i in range(50)]
        get_results = await asyncio.gather(*get_tasks, return_exceptions=True)

        
        # Verify results
        for i, result in enumerate(get_results):
            expected_value = f"value_{i}"
            assert result == expected_value
    
    async def test_performance_monitoring(self, cache_service):
        """Test performance monitoring capabilities"""
        # Perform operations to generate metrics
        for i in range(100):
            await cache_service.set(f"perf:key:{i}", f"value_{i}")
            await cache_service.get(f"perf:key:{i}")
        
        # Get performance history
        history = await cache_service.get_performance_history()
        assert isinstance(history, list)
        
        if history:  # If history is available
            latest = history[-1]
            assert 'timestamp' in latest
            assert 'hit_rate' in latest
            assert 'operations_per_second' in latest
            assert 'average_latency_ms' in latest
    
    async def test_ai_integration(self, cache_service):
        """Test AI optimization integration"""
        # Generate access patterns
        for i in range(50):
            key = f"ai:pattern:{i % 10}"  # Create repeated access pattern
            await cache_service.set(key, f"value_{i}")
            await cache_service.get(key)
        
        # Get AI insights
        insights = await cache_service.get_ai_insights()
        assert isinstance(insights, dict)
        
        # Should contain optimization recommendations
        if 'optimization_opportunities' in insights:
            assert isinstance(insights['optimization_opportunities'], list)
        
        if 'performance_predictions' in insights:
            assert isinstance(insights['performance_predictions'], dict)
    
    async def test_security_features(self, cache_service):
        """Test security-related features"""
        sensitive_key = "user:sensitive:data"
        sensitive_value = "confidential_information"
        
        # Set sensitive data
        await cache_service.set(sensitive_key, sensitive_value)
        
        # Retrieve and verify
        retrieved = await cache_service.get(sensitive_key)
        assert retrieved == sensitive_value
        
        # Test secure deletion
        await cache_service.secure_delete(sensitive_key)
        
        # Verify secure deletion
        retrieved_after_delete = await cache_service.get(sensitive_key)
        assert retrieved_after_delete is None


class TestCacheServiceAdvanced:
    """Advanced test cases for cache service"""
    
    async def test_cache_warming(self, cache_service):
        """Test cache warming functionality"""
        # Set up initial data
        base_keys = [f"warm:key:{i}" for i in range(20)]
        for key in base_keys:
            await cache_service.set(key, f"value_for_{key}")
        
        # Simulate cache warming with predicted keys
        warming_keys = [f"warm:predicted:{i}" for i in range(10)]
        warming_data = {key: f"warmed_value_{i}" for i, key in enumerate(warming_keys)}
        
        warm_results = await cache_service.warm_cache(warming_data)
        assert len(warm_results) == len(warming_keys)
        assert all(warm_results.values())
        
        # Verify warmed data is accessible
        for key in warming_keys:
            value = await cache_service.get(key)
            assert value is not None
    
    async def test_cache_policies(self, cache_service, sample_policies):
        """Test cache policy management"""
        for policy in sample_policies.values():
            # Add policy to service
            await cache_service.add_policy(policy)
        
        # Test policy application
        # High-performance policy should apply to api:* keys
        api_key = "api:endpoint:users"
        await cache_service.set(api_key, "api_response_data")
        
        # Check that policy was applied
        entry_metadata = cache_service._entry_metadata.get(
            cache_service._build_cache_key(api_key, "default")
        )
        
        if entry_metadata:
            assert entry_metadata.tier_preference in ["L1", "L2"]  # High performance tiers
    
    async def test_multi_tier_operations(self, cache_service):
        """Test multi-tier cache operations"""
        # Test data that should go to different tiers
        hot_key = "hot:frequently_accessed"
        cold_key = "cold:rarely_accessed"
        
        # Set hot data (should prefer fast tiers)
        await cache_service.set(hot_key, "hot_data", tier_hint="L1")
        
        # Set cold data (should prefer storage tiers)
        await cache_service.set(cold_key, "cold_data", tier_hint="L3")
        
        # Both should be retrievable
        hot_value = await cache_service.get(hot_key)
        cold_value = await cache_service.get(cold_key)
        
        assert hot_value == "hot_data"
        assert cold_value == "cold_data"
    
    async def test_predictive_caching(self, cache_service):
        """Test predictive caching functionality"""
        # Create access pattern
        base_pattern = "user:123:"
        keys = [f"{base_pattern}{suffix}" for suffix in ["profile", "settings", "history", "preferences"]]
        
        # Access some keys to establish pattern
        for key in keys[:2]:
            await cache_service.set(key, f"data_for_{key}")
            await cache_service.get(key)
        
        # Trigger predictive caching
        predictions = await cache_service.get_prefetch_candidates(keys[:2])
        
        assert isinstance(predictions, list)
        # Should predict related keys in the same user namespace
        predicted_keys = [pred[0] for pred in predictions]
        related_keys = [key for key in predicted_keys if base_pattern in key]
        
        assert len(related_keys) > 0
    
    async def test_cache_analytics(self, cache_service):
        """Test cache analytics and reporting"""
        # Generate diverse access patterns
        patterns = {
            "sequential": [f"seq:{i:04d}" for i in range(50)],
            "random": [f"random:{i}" for i in [1, 5, 3, 8, 2, 9, 4, 7, 6]],
            "temporal": [f"hourly:{datetime.utcnow().hour:02d}:data:{i}" for i in range(20)]
        }
        
        for pattern_type, keys in patterns.items():
            for key in keys:
                await cache_service.set(key, f"{pattern_type}_data")
                await cache_service.get(key)
        
        # Get analytics
        analytics = await cache_service.get_cache_analytics()
        
        assert isinstance(analytics, dict)
        assert 'access_patterns' in analytics
        assert 'performance_metrics' in analytics
        assert 'optimization_suggestions' in analytics
    
    @pytest.mark.asyncio
    async def test_stress_operations(self, cache_service):
        """Stress test with high load"""
        operations_count = 1000
        concurrent_tasks = 50
        
        async def stress_worker(worker_id: int, operations_per_worker: int):
            results = {'success': 0, 'errors': 0}
            
            for i in range(operations_per_worker):
                try:
                    key = f"stress:{worker_id}:{i}"
                    value = f"stress_value_{worker_id}_{i}"
                    
                    # Mix of operations
                    if i % 3 == 0:
                        await cache_service.set(key, value)
                        results['success'] += 1
                    elif i % 3 == 1:
                        await cache_service.get(key)
                        results['success'] += 1
                    else:
                        await cache_service.delete(key)
                        results['success'] += 1
                        
                except Exception:
                    results['errors'] += 1
            
            return results
        
        # Create concurrent workers
        operations_per_worker = operations_count // concurrent_tasks
        tasks = [
            stress_worker(worker_id, operations_per_worker)
            for worker_id in range(concurrent_tasks)
        ]
        
        # Execute stress test
        results = await asyncio.gather(*tasks, return_exceptions=True)

        
        # Aggregate results
        total_success = sum(r['success'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        
        # Most operations should succeed
        success_rate = total_success / (total_success + total_errors)
        assert success_rate > 0.95  # 95% success rate minimum
        
        # Service should still be responsive after stress test
        test_key = "post_stress_test"
        test_value = "still_working"
        
        await cache_service.set(test_key, test_value)
        retrieved = await cache_service.get(test_key)
        assert retrieved == test_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])