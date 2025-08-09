#!/usr/bin/env python3
"""
APG Cache Management (CACH) - Test Configuration
Comprehensive test fixtures and configuration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, Generator, AsyncGenerator
import os
import sys

# Add the capability directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from service import CacheService, CacheServiceConfig
from models import CacheEntry, CacheCluster, CachePolicy, BackendType, SecurityLevel
from ai_optimization import OptimizationEngine
from predictive_engine import PredictiveEngine
from intelligent_warming import IntelligentWarmingEngine
from cache_hierarchy import MultiTierCacheHierarchy
from quantum_security import QuantumSecurityEngine
from zero_config_intelligence import ZeroConfigIntelligenceEngine


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def cache_config():
    """Create a test cache configuration."""
    return CacheServiceConfig(
        cache_size_mb=128,
        max_entries=5000,
        default_ttl_seconds=300,
        cleanup_interval_seconds=30,
        ai_optimization_enabled=True,
        predictive_caching_enabled=True,
        security_level=SecurityLevel.HIGH,
        monitoring_enabled=True,
        persist_data=False
    )


@pytest.fixture
async def cache_service(cache_config):
    """Create and initialize a cache service for testing."""
    service = CacheService(cache_config)
    await service.initialize()
    yield service
    await service.shutdown()


@pytest.fixture
async def optimization_engine():
    """Create an optimization engine for testing."""
    engine = OptimizationEngine({'max_optimization_cycles': 5})
    await engine.initialize()
    yield engine
    await engine.shutdown()


@pytest.fixture
async def predictive_engine():
    """Create a predictive engine for testing."""
    engine = PredictiveEngine({'confidence_threshold': 0.6})
    await engine.initialize()
    yield engine


@pytest.fixture
async def warming_engine():
    """Create an intelligent warming engine for testing."""
    engine = IntelligentWarmingEngine({'max_concurrent_warming': 5})
    await engine.initialize()
    yield engine
    await engine.shutdown()


@pytest.fixture
async def cache_hierarchy():
    """Create a cache hierarchy for testing."""
    hierarchy = MultiTierCacheHierarchy({'optimization_interval': 60})
    await hierarchy.initialize()
    yield hierarchy
    await hierarchy.shutdown()


@pytest.fixture
async def security_engine():
    """Create a quantum security engine for testing."""
    engine = QuantumSecurityEngine({'behavior_analysis_enabled': True})
    await engine.initialize()
    yield engine
    await engine.shutdown()


@pytest.fixture
async def zero_config_engine():
    """Create a zero-config intelligence engine for testing."""
    engine = ZeroConfigIntelligenceEngine({'discovery_interval': 30})
    await engine.initialize()
    yield engine


@pytest.fixture
def sample_cache_entries():
    """Create sample cache entries for testing."""
    entries = {}
    
    # Create various types of test entries
    for i in range(100):
        key = f"test:entry:{i}"
        entry = CacheEntry(
            key=key,
            value=f"test_value_{i}".encode(),
            size_bytes=len(f"test_value_{i}"),
            created_at=datetime.utcnow() - timedelta(minutes=i),
            last_accessed=datetime.utcnow() - timedelta(minutes=i//2),
            access_count=10 - (i // 10),
            ttl_seconds=300 + (i * 10),
            namespace="test"
        )
        entries[key] = entry
    
    # Add some high-frequency entries
    for i in range(10):
        key = f"hot:entry:{i}"
        entry = CacheEntry(
            key=key,
            value=f"hot_value_{i}".encode(),
            size_bytes=len(f"hot_value_{i}"),
            created_at=datetime.utcnow() - timedelta(hours=1),
            last_accessed=datetime.utcnow() - timedelta(minutes=1),
            access_count=100 + i,
            ttl_seconds=3600,
            namespace="hot"
        )
        entries[key] = entry
    
    return entries


@pytest.fixture
def sample_clusters():
    """Create sample cache clusters for testing."""
    clusters = {}
    
    # Redis cluster
    redis_cluster = CacheCluster(
        cluster_id="redis-test-cluster",
        name="Redis Test Cluster",
        description="Test Redis cluster",
        backend_type=BackendType.REDIS,
        nodes=["redis://localhost:6379"],
        max_memory_mb=512,
        ai_optimization_enabled=True
    )
    clusters[redis_cluster.cluster_id] = redis_cluster
    
    # Memory cluster
    memory_cluster = CacheCluster(
        cluster_id="memory-test-cluster",
        name="Memory Test Cluster",
        description="Test memory cluster",
        backend_type=BackendType.MEMORY,
        nodes=["memory://local"],
        max_memory_mb=256,
        ai_optimization_enabled=True
    )
    clusters[memory_cluster.cluster_id] = memory_cluster
    
    return clusters


@pytest.fixture
def sample_policies():
    """Create sample cache policies for testing."""
    policies = {}
    
    # High-performance policy
    perf_policy = CachePolicy(
        policy_id="high-performance",
        name="High Performance Policy",
        description="Optimized for low latency",
        key_patterns=["api:*", "user:*:session"],
        ttl_seconds=300,
        tier_preference="L1",
        eviction_strategy="LRU",
        compression_enabled=False,
        encryption_enabled=True,
        enabled=True
    )
    policies[perf_policy.policy_id] = perf_policy
    
    # Storage-optimized policy
    storage_policy = CachePolicy(
        policy_id="storage-optimized",
        name="Storage Optimized Policy", 
        description="Optimized for storage efficiency",
        key_patterns=["content:*", "static:*"],
        ttl_seconds=3600,
        tier_preference="L3",
        eviction_strategy="LFU",
        compression_enabled=True,
        encryption_enabled=True,
        enabled=True
    )
    policies[storage_policy.policy_id] = storage_policy
    
    return policies


@pytest.fixture
def performance_metrics():
    """Create sample performance metrics for testing."""
    return {
        'hit_rate': 0.87,
        'miss_rate': 0.13,
        'total_operations': 15000,
        'operations_per_second': 2500.0,
        'average_latency_ms': 2.3,
        'p95_latency_ms': 8.7,
        'p99_latency_ms': 15.2,
        'memory_utilization': 67.8,
        'cpu_utilization': 35.2,
        'error_rate': 0.002,
        'eviction_count': 120,
        'timestamp': datetime.utcnow()
    }


@pytest.fixture
def ai_insights():
    """Create sample AI insights for testing."""
    return {
        'optimization_opportunities': [
            {
                'type': 'tier_rebalancing',
                'description': 'L1 tier underutilized, consider rebalancing',
                'confidence': 0.85,
                'impact': 'medium'
            },
            {
                'type': 'prefetch_optimization',
                'description': 'Sequential access pattern detected for user:* keys',
                'confidence': 0.92,
                'impact': 'high'
            }
        ],
        'performance_predictions': {
            'next_hour_qps': 2800,
            'memory_pressure_risk': 0.3,
            'hit_rate_trend': 'increasing'
        },
        'recommendations': [
            {
                'action': 'increase_l1_size',
                'reason': 'High frequency access patterns detected',
                'priority': 'high'
            }
        ]
    }


@pytest.fixture
async def mock_data_populated_service(cache_service, sample_cache_entries):
    """Create a cache service populated with test data."""
    # Populate the service with sample data
    for key, entry in sample_cache_entries.items():
        await cache_service.set(
            key=entry.key,
            value=entry.value.decode(),
            ttl_seconds=entry.ttl_seconds,
            namespace=entry.namespace
        )
    
    yield cache_service


# Test data generators

def generate_access_pattern(pattern_type: str, count: int = 100):
    """Generate access patterns for testing."""
    patterns = []
    
    if pattern_type == "sequential":
        for i in range(count):
            patterns.append(f"seq:{i:04d}")
    elif pattern_type == "random":
        import random
        for i in range(count):
            patterns.append(f"random:{random.randint(1000, 9999)}")
    elif pattern_type == "temporal":
        for i in range(count):
            hour = i % 24
            patterns.append(f"hourly:{hour:02d}:data:{i}")
    else:
        for i in range(count):
            patterns.append(f"default:{i}")
    
    return patterns


def generate_performance_history(days: int = 7, samples_per_day: int = 24):
    """Generate performance history for testing."""
    history = []
    base_time = datetime.utcnow() - timedelta(days=days)
    
    for day in range(days):
        for hour in range(samples_per_day):
            timestamp = base_time + timedelta(days=day, hours=hour)
            
            # Simulate daily patterns
            business_hours = 9 <= hour <= 17
            hit_rate = 0.85 + (0.1 if business_hours else -0.05)
            qps = 2000 + (1000 if business_hours else -500)
            latency = 3.0 + (1.0 if business_hours else -0.5)
            
            history.append({
                'timestamp': timestamp.isoformat(),
                'hit_rate': min(max(hit_rate, 0.7), 0.95),
                'operations_per_second': max(qps, 500),
                'average_latency_ms': max(latency, 1.0),
                'memory_utilization': 60 + (hour % 24),
                'error_rate': 0.001 + (0.001 if business_hours else 0)
            })
    
    return history


# Async test helpers

async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
    """Wait for a condition to become true."""
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
            return True
        await asyncio.sleep(interval)
    
    return False


async def simulate_cache_load(cache_service, operations: int = 1000, pattern: str = "mixed"):
    """Simulate cache load for testing."""
    results = {'hits': 0, 'misses': 0, 'sets': 0, 'errors': 0}
    
    for i in range(operations):
        try:
            if pattern == "read_heavy":
                # 80% reads, 20% writes
                if i % 5 == 0:
                    await cache_service.set(f"load_test:{i}", f"value_{i}")
                    results['sets'] += 1
                else:
                    value = await cache_service.get(f"load_test:{i % 100}")
                    if value:
                        results['hits'] += 1
                    else:
                        results['misses'] += 1
            elif pattern == "write_heavy":
                # 20% reads, 80% writes
                if i % 5 != 0:
                    await cache_service.set(f"load_test:{i}", f"value_{i}")
                    results['sets'] += 1
                else:
                    value = await cache_service.get(f"load_test:{i % 100}")
                    if value:
                        results['hits'] += 1
                    else:
                        results['misses'] += 1
            else:  # mixed
                if i % 2 == 0:
                    await cache_service.set(f"load_test:{i}", f"value_{i}")
                    results['sets'] += 1
                else:
                    value = await cache_service.get(f"load_test:{i % 100}")
                    if value:
                        results['hits'] += 1
                    else:
                        results['misses'] += 1
        except Exception as e:
            results['errors'] += 1
    
    return results


# Performance testing utilities

class PerformanceTimer:
    """Context manager for measuring performance."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
    
    @property
    def duration_ms(self):
        return self.duration * 1000 if self.duration else None


# Cleanup utilities

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Cleanup operations if needed
    await asyncio.sleep(0.01)  # Small delay to allow cleanup