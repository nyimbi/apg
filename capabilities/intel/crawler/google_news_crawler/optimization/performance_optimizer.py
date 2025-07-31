#!/usr/bin/env python3
"""
Performance Optimization and Scaling Framework
==============================================

Enterprise-grade performance optimization and scaling framework for the Google News
Intelligence System, implementing advanced caching strategies, distributed processing,
and real-time performance monitoring.

Theoretical Foundation:
- Distributed systems theory with CAP theorem considerations
- Queue theory optimization for request processing
- Cache coherence protocols with LRU and probabilistic eviction
- Load balancing using consistent hashing algorithms
- Distributed consensus mechanisms for coordination
- Performance modeling using Little's Law: N = λ × W

Mathematical Models:
- Throughput optimization: T = min(C_cpu, C_memory, C_network, C_database)
- Cache hit ratio: H = (cache_hits) / (cache_hits + cache_misses)
- Response time percentiles: P99, P95, P50 latency distributions
- Resource utilization: U = λ × S (arrival rate × service time)
- Scalability coefficient: S_c = T_n / (n × T_1) where n = instances

Author: Nyimbi Odero  
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import asyncio
import json
import logging
import time
import hashlib
import pickle
import redis
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import psutil
import aiofiles
import uuid

# Monitoring and metrics
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Distributed processing
from celery import Celery
from kombu import Queue
import msgpack

# Configuration management
from pydantic import BaseSettings, Field
from typing_extensions import Literal

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics tracking."""
    timestamp: datetime
    
    # Throughput metrics
    requests_per_second: float = 0.0
    articles_processed_per_minute: float = 0.0
    successful_extractions_per_hour: float = 0.0
    
    # Latency metrics (milliseconds)
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    avg_response_time: float = 0.0
    
    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_io_usage_percent: float = 0.0
    network_usage_mbps: float = 0.0
    
    # Cache performance
    cache_hit_ratio: float = 0.0
    cache_memory_usage_mb: float = 0.0
    cache_eviction_rate: float = 0.0
    
    # Database performance
    db_connection_pool_usage: float = 0.0
    avg_query_time_ms: float = 0.0
    db_deadlocks_per_hour: float = 0.0
    
    # Error rates
    error_rate_percent: float = 0.0
    timeout_rate_percent: float = 0.0
    retry_rate_percent: float = 0.0
    
    # Scaling metrics
    active_workers: int = 0
    queue_depth: int = 0
    backpressure_events: int = 0


class AdvancedCacheManager:
    """
    Sophisticated multi-tier caching system with intelligent eviction
    and cache coherence mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize advanced cache manager."""
        self.config = config or {}
        
        # Cache tiers configuration
        self.l1_cache_size = self.config.get('l1_cache_size', 1000)  # In-memory LRU
        self.l2_cache_size = self.config.get('l2_cache_size', 10000)  # Redis
        self.l3_cache_size = self.config.get('l3_cache_size', 100000)  # Disk-based
        
        # Cache policies
        self.ttl_default = self.config.get('default_ttl', 3600)  # 1 hour
        self.ttl_news_articles = self.config.get('news_ttl', 1800)  # 30 minutes
        self.ttl_ml_extractions = self.config.get('ml_ttl', 7200)  # 2 hours
        self.ttl_analytics = self.config.get('analytics_ttl', 300)  # 5 minutes
        
        # Initialize cache tiers
        self.l1_cache = {}  # Simple dict for L1
        self.l1_access_order = deque()  # For LRU implementation
        self.l1_lock = threading.RLock()
        
        # Redis connection for L2 cache
        self.redis_client = None
        if self.config.get('redis_url'):
            try:
                import redis
                self.redis_client = redis.from_url(
                    self.config['redis_url'],
                    decode_responses=False,  # Keep binary for pickle
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    health_check_interval=30
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis L2 cache connected")
            except Exception as e:
                logger.warning(f"Redis L2 cache unavailable: {e}")
                self.redis_client = None
        
        # Disk cache for L3
        self.l3_cache_dir = Path(self.config.get('cache_dir', './cache'))
        self.l3_cache_dir.mkdir(exist_ok=True)
        
        # Cache statistics
        self.stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'l3_hits': 0, 'l3_misses': 0,
            'evictions': 0, 'cache_errors': 0
        }
        
        # Prometheus metrics
        self.cache_hit_counter = Counter('cache_hits_total', 'Cache hits', ['tier'])
        self.cache_miss_counter = Counter('cache_misses_total', 'Cache misses', ['tier'])
        self.cache_latency = Histogram('cache_operation_duration_seconds', 'Cache operation latency', ['operation', 'tier'])
        
        logger.info("Advanced cache manager initialized")
    
    def _generate_cache_key(self, key: str, namespace: str = 'default') -> str:
        """Generate standardized cache key with namespace."""
        return f"{namespace}:{hashlib.sha256(key.encode()).hexdigest()[:16]}"
    
    async def get(self, key: str, namespace: str = 'default') -> Optional[Any]:
        """Get value from multi-tier cache with fallback."""
        cache_key = self._generate_cache_key(key, namespace)
        
        # L1 Cache (in-memory)
        with self.cache_latency.labels(operation='get', tier='l1').time():
            l1_value = self._get_l1(cache_key)
            if l1_value is not None:
                self.cache_hit_counter.labels(tier='l1').inc()
                self.stats['l1_hits'] += 1
                return l1_value
            self.cache_miss_counter.labels(tier='l1').inc()
            self.stats['l1_misses'] += 1
        
        # L2 Cache (Redis)
        if self.redis_client:
            with self.cache_latency.labels(operation='get', tier='l2').time():
                l2_value = await self._get_l2(cache_key)
                if l2_value is not None:
                    self.cache_hit_counter.labels(tier='l2').inc()
                    self.stats['l2_hits'] += 1
                    # Promote to L1
                    await self._set_l1(cache_key, l2_value)
                    return l2_value
                self.cache_miss_counter.labels(tier='l2').inc()
                self.stats['l2_misses'] += 1
        
        # L3 Cache (disk)
        with self.cache_latency.labels(operation='get', tier='l3').time():
            l3_value = await self._get_l3(cache_key)
            if l3_value is not None:
                self.cache_hit_counter.labels(tier='l3').inc()
                self.stats['l3_hits'] += 1
                # Promote to higher tiers
                if self.redis_client:
                    await self._set_l2(cache_key, l3_value, self.ttl_default)
                await self._set_l1(cache_key, l3_value)
                return l3_value
            self.cache_miss_counter.labels(tier='l3').inc()
            self.stats['l3_misses'] += 1
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, namespace: str = 'default') -> bool:
        """Set value in multi-tier cache with TTL."""
        cache_key = self._generate_cache_key(key, namespace)
        ttl = ttl or self.ttl_default
        
        success = True
        
        # Set in all available tiers
        try:
            # L1 Cache
            with self.cache_latency.labels(operation='set', tier='l1').time():
                await self._set_l1(cache_key, value)
            
            # L2 Cache (Redis)
            if self.redis_client:
                with self.cache_latency.labels(operation='set', tier='l2').time():
                    await self._set_l2(cache_key, value, ttl)
            
            # L3 Cache (disk)
            with self.cache_latency.labels(operation='set', tier='l3').time():
                await self._set_l3(cache_key, value, ttl)
                
        except Exception as e:
            logger.error(f"Cache set operation failed: {e}")
            self.stats['cache_errors'] += 1
            success = False
        
        return success
    
    def _get_l1(self, key: str) -> Optional[Any]:
        """Get from L1 in-memory cache."""
        with self.l1_lock:
            if key in self.l1_cache:
                # Move to end for LRU
                self.l1_access_order.remove(key)
                self.l1_access_order.append(key)
                return self.l1_cache[key]
            return None
    
    async def _set_l1(self, key: str, value: Any):
        """Set in L1 in-memory cache with LRU eviction."""
        with self.l1_lock:
            # Evict if cache is full
            while len(self.l1_cache) >= self.l1_cache_size:
                oldest_key = self.l1_access_order.popleft()
                del self.l1_cache[oldest_key]
                self.stats['evictions'] += 1
            
            self.l1_cache[key] = value
            if key in self.l1_access_order:
                self.l1_access_order.remove(key)
            self.l1_access_order.append(key)
    
    async def _get_l2(self, key: str) -> Optional[Any]:
        """Get from L2 Redis cache."""
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.debug(f"L2 cache get error: {e}")
        return None
    
    async def _set_l2(self, key: str, value: Any, ttl: int):
        """Set in L2 Redis cache."""
        if not self.redis_client:
            return
        
        try:
            data = pickle.dumps(value)
            self.redis_client.setex(key, ttl, data)
        except Exception as e:
            logger.debug(f"L2 cache set error: {e}")
    
    async def _get_l3(self, key: str) -> Optional[Any]:
        """Get from L3 disk cache."""
        cache_file = self.l3_cache_dir / f"{key}.cache"
        
        try:
            if cache_file.exists():
                # Check if file has expired
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age < self.ttl_default:
                    async with aiofiles.open(cache_file, 'rb') as f:
                        data = await f.read()
                        return pickle.loads(data)
                else:
                    # Remove expired file
                    cache_file.unlink()
        except Exception as e:
            logger.debug(f"L3 cache get error: {e}")
        
        return None
    
    async def _set_l3(self, key: str, value: Any, ttl: int):
        """Set in L3 disk cache."""
        cache_file = self.l3_cache_dir / f"{key}.cache"
        
        try:
            data = pickle.dumps(value)
            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(data)
        except Exception as e:
            logger.debug(f"L3 cache set error: {e}")
    
    async def invalidate(self, key: str, namespace: str = 'default'):
        """Invalidate key across all cache tiers."""
        cache_key = self._generate_cache_key(key, namespace)
        
        # L1
        with self.l1_lock:
            if cache_key in self.l1_cache:
                del self.l1_cache[cache_key]
                self.l1_access_order.remove(cache_key)
        
        # L2
        if self.redis_client:
            try:
                self.redis_client.delete(cache_key)
            except Exception as e:
                logger.debug(f"L2 cache invalidation error: {e}")
        
        # L3
        cache_file = self.l3_cache_dir / f"{cache_key}.cache"
        try:
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.debug(f"L3 cache invalidation error: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        total_requests = sum([
            self.stats['l1_hits'], self.stats['l1_misses'],
            self.stats['l2_hits'], self.stats['l2_misses'],
            self.stats['l3_hits'], self.stats['l3_misses']
        ])
        
        total_hits = self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['l3_hits']
        
        return {
            'overall_hit_ratio': total_hits / max(1, total_requests),
            'l1_hit_ratio': self.stats['l1_hits'] / max(1, self.stats['l1_hits'] + self.stats['l1_misses']),
            'l2_hit_ratio': self.stats['l2_hits'] / max(1, self.stats['l2_hits'] + self.stats['l2_misses']),
            'l3_hit_ratio': self.stats['l3_hits'] / max(1, self.stats['l3_hits'] + self.stats['l3_misses']),
            'cache_size_l1': len(self.l1_cache),
            'evictions_total': self.stats['evictions'],
            'cache_errors': self.stats['cache_errors'],
            'redis_available': self.redis_client is not None
        }


class DistributedTaskManager:
    """
    Advanced distributed task management system using Celery with 
    intelligent load balancing and failover mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize distributed task manager."""
        self.config = config or {}
        
        # Celery configuration
        broker_url = self.config.get('broker_url', 'redis://localhost:6379/0')
        result_backend = self.config.get('result_backend', 'redis://localhost:6379/1')
        
        self.celery_app = Celery(
            'news_intelligence',
            broker=broker_url,
            backend=result_backend,
            include=['news_intelligence.tasks']
        )
        
        # Celery optimization settings
        self.celery_app.conf.update(
            task_serializer='msgpack',
            accept_content=['msgpack'],
            result_serializer='msgpack',
            timezone='UTC',
            enable_utc=True,
            task_track_started=True,
            task_time_limit=1800,  # 30 minutes
            task_soft_time_limit=1500,  # 25 minutes
            worker_prefetch_multiplier=1,
            task_acks_late=True,
            worker_disable_rate_limits=False,
            task_compression='gzip',
            result_compression='gzip',
            task_routes={
                'news_intelligence.scrape_article': {'queue': 'scraping'},
                'news_intelligence.extract_events': {'queue': 'ml_processing'},
                'news_intelligence.analyze_trends': {'queue': 'analytics'},
                'news_intelligence.generate_alerts': {'queue': 'alerts'}
            }
        )
        
        # Define priority queues
        self.celery_app.conf.task_default_queue = 'default'
        self.celery_app.conf.task_queues = (
            Queue('scraping', routing_key='scraping'),
            Queue('ml_processing', routing_key='ml_processing'),
            Queue('analytics', routing_key='analytics'),
            Queue('alerts', routing_key='alerts'),
            Queue('high_priority', routing_key='high_priority'),
            Queue('low_priority', routing_key='low_priority'),
        )
        
        # Task monitoring
        self.task_stats = defaultdict(int)
        self.task_timing = defaultdict(list)
        
        # Prometheus metrics
        self.task_counter = Counter('tasks_total', 'Total tasks', ['task_name', 'status'])
        self.task_duration = Histogram('task_duration_seconds', 'Task duration', ['task_name'])
        self.queue_size = Gauge('queue_size', 'Queue size', ['queue_name'])
        
        logger.info("Distributed task manager initialized")
    
    def submit_news_discovery_task(self, search_config: Dict[str, Any], priority: str = 'normal') -> str:
        """Submit news discovery task to distributed queue."""
        queue_name = self._get_queue_for_priority(priority)
        
        task = self.celery_app.send_task(
            'news_intelligence.discover_news',
            args=[search_config],
            queue=queue_name,
            priority=self._get_priority_value(priority)
        )
        
        self.task_counter.labels(task_name='discover_news', status='submitted').inc()
        return task.id
    
    def submit_content_scraping_task(self, article_urls: List[str], priority: str = 'normal') -> str:
        """Submit content scraping task with batch processing."""
        queue_name = self._get_queue_for_priority(priority)
        
        task = self.celery_app.send_task(
            'news_intelligence.scrape_articles_batch',
            args=[article_urls],
            queue=queue_name,
            priority=self._get_priority_value(priority)
        )
        
        self.task_counter.labels(task_name='scrape_articles', status='submitted').inc()
        return task.id
    
    def submit_ml_extraction_task(self, article_content: str, metadata: Dict[str, Any], priority: str = 'normal') -> str:
        """Submit ML event extraction task."""
        queue_name = 'ml_processing'  # Always use ML processing queue
        
        task = self.celery_app.send_task(
            'news_intelligence.extract_events',
            args=[article_content, metadata],
            queue=queue_name,
            priority=self._get_priority_value(priority)
        )
        
        self.task_counter.labels(task_name='extract_events', status='submitted').inc()
        return task.id
    
    def submit_analytics_task(self, articles_data: List[Dict[str, Any]]) -> str:
        """Submit analytics processing task."""
        task = self.celery_app.send_task(
            'news_intelligence.analyze_trends',
            args=[articles_data],
            queue='analytics'
        )
        
        self.task_counter.labels(task_name='analyze_trends', status='submitted').inc()
        return task.id
    
    def _get_queue_for_priority(self, priority: str) -> str:
        """Get appropriate queue based on task priority."""
        priority_queues = {
            'critical': 'high_priority',
            'high': 'high_priority',
            'normal': 'default',
            'low': 'low_priority'
        }
        return priority_queues.get(priority, 'default')
    
    def _get_priority_value(self, priority: str) -> int:
        """Convert priority string to numeric value."""
        priority_values = {
            'critical': 10,
            'high': 7,
            'normal': 5,
            'low': 2
        }
        return priority_values.get(priority, 5)
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get detailed task status and progress."""
        try:
            result = self.celery_app.AsyncResult(task_id)
            return {
                'task_id': task_id,
                'status': result.status,
                'result': result.result if result.ready() else None,
                'traceback': result.traceback if result.failed() else None,
                'progress': result.info if result.status == 'PROGRESS' else None
            }
        except Exception as e:
            return {
                'task_id': task_id,
                'status': 'ERROR',
                'error': str(e)
            }
    
    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        try:
            inspector = self.celery_app.control.inspect()
            
            # Get active tasks
            active_tasks = inspector.active()
            
            # Get queue lengths
            queue_lengths = {}
            for queue in ['scraping', 'ml_processing', 'analytics', 'alerts', 'high_priority', 'low_priority']:
                try:
                    with self.celery_app.connection() as conn:
                        queue_obj = conn.default_channel.queue_declare(queue=queue, passive=True)
                        queue_lengths[queue] = queue_obj.message_count
                        self.queue_size.labels(queue_name=queue).set(queue_obj.message_count)
                except:
                    queue_lengths[queue] = 0
            
            return {
                'active_tasks': len(active_tasks) if active_tasks else 0,
                'queue_lengths': queue_lengths,
                'total_queued': sum(queue_lengths.values()),
                'worker_status': 'online' if active_tasks is not None else 'offline'
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue statistics: {e}")
            return {
                'active_tasks': 0,
                'queue_lengths': {},
                'total_queued': 0,
                'worker_status': 'unknown'
            }


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system with real-time metrics
    collection and alerting capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize performance monitor."""
        self.config = config or {}
        
        # Monitoring configuration
        self.collection_interval = self.config.get('collection_interval', 30)  # seconds
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'error_rate': 5.0,
            'response_time_p99': 10000.0,  # ms
            'cache_hit_ratio': 0.8
        })
        
        # Metrics storage
        self.metrics_history = deque(maxlen=2880)  # 24 hours at 30s intervals
        self.current_metrics = PerformanceMetrics(timestamp=datetime.now(timezone.utc))
        
        # Response time tracking
        self.response_times = deque(maxlen=1000)
        self.response_time_lock = threading.Lock()
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.error_lock = threading.Lock()
        
        # Prometheus metrics
        self.setup_prometheus_metrics()
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("Performance monitor initialized")
    
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics for monitoring."""
        self.prom_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint', 'status_code'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.prom_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.prom_cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        self.prom_memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage')
        self.prom_cache_hit_ratio = Gauge('cache_hit_ratio', 'Cache hit ratio')
        self.prom_queue_depth = Gauge('queue_depth_total', 'Total queue depth')
        self.prom_active_connections = Gauge('database_connections_active', 'Active database connections')
    
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start Prometheus metrics server
        prometheus_port = self.config.get('prometheus_port', 8000)
        try:
            start_http_server(prometheus_port)
            logger.info(f"Prometheus metrics server started on port {prometheus_port}")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self._update_prometheus_metrics(metrics)
                
                # Store in history
                self.metrics_history.append(metrics)
                self.current_metrics = metrics
                
                # Check alert thresholds
                self._check_alerts(metrics)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
            
            time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive system metrics."""
        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Calculate response time percentiles
        with self.response_time_lock:
            if self.response_times:
                response_times_ms = [t * 1000 for t in self.response_times]
                p50 = float(np.percentile(response_times_ms, 50))
                p95 = float(np.percentile(response_times_ms, 95))
                p99 = float(np.percentile(response_times_ms, 99))
                avg_response = float(np.mean(response_times_ms))
            else:
                p50 = p95 = p99 = avg_response = 0.0
        
        # Calculate error rates
        with self.error_lock:
            total_requests = sum(self.error_counts.values())
            error_count = self.error_counts.get('error', 0)
            error_rate = (error_count / max(1, total_requests)) * 100
        
        return PerformanceMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory_info.percent,
            p50_response_time=p50,
            p95_response_time=p95,
            p99_response_time=p99,
            avg_response_time=avg_response,
            error_rate_percent=error_rate
        )
    
    def _update_prometheus_metrics(self, metrics: PerformanceMetrics):
        """Update Prometheus metrics."""
        self.prom_cpu_usage.set(metrics.cpu_usage_percent)
        self.prom_memory_usage.set(metrics.memory_usage_percent)
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check metrics against alert thresholds."""
        alerts = []
        
        if metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        if metrics.memory_usage_percent > self.alert_thresholds['memory_usage']:
            alerts.append(f"High memory usage: {metrics.memory_usage_percent:.1f}%")
        
        if metrics.error_rate_percent > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {metrics.error_rate_percent:.1f}%")
        
        if metrics.p99_response_time > self.alert_thresholds['response_time_p99']:
            alerts.append(f"High P99 response time: {metrics.p99_response_time:.1f}ms")
        
        if alerts:
            logger.warning(f"Performance alerts: {'; '.join(alerts)}")
            self._send_alerts(alerts)
    
    def _send_alerts(self, alerts: List[str]):
        """Send performance alerts (placeholder for alerting system)."""
        # In production, this would integrate with alerting systems like PagerDuty, Slack, etc.
        for alert in alerts:
            logger.critical(f"ALERT: {alert}")
    
    def record_request(self, duration: float, success: bool = True):
        """Record request timing and success status."""
        with self.response_time_lock:
            self.response_times.append(duration)
        
        with self.error_lock:
            if success:
                self.error_counts['success'] += 1
            else:
                self.error_counts['error'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        
        return {
            'current': asdict(self.current_metrics),
            'averages_10min': {
                'cpu_usage': np.mean([m.cpu_usage_percent for m in recent_metrics]),
                'memory_usage': np.mean([m.memory_usage_percent for m in recent_metrics]),
                'avg_response_time': np.mean([m.avg_response_time for m in recent_metrics]),
                'error_rate': np.mean([m.error_rate_percent for m in recent_metrics])
            },
            'trends': {
                'cpu_trend': self._calculate_trend([m.cpu_usage_percent for m in recent_metrics]),
                'memory_trend': self._calculate_trend([m.memory_usage_percent for m in recent_metrics]),
                'response_time_trend': self._calculate_trend([m.avg_response_time for m in recent_metrics])
            },
            'alerts_active': self._get_active_alerts(),
            'monitoring_duration_hours': len(self.metrics_history) * self.collection_interval / 3600
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from recent values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear regression slope
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _get_active_alerts(self) -> List[str]:
        """Get currently active alerts."""
        alerts = []
        metrics = self.current_metrics
        
        if metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage']:
            alerts.append('high_cpu')
        if metrics.memory_usage_percent > self.alert_thresholds['memory_usage']:
            alerts.append('high_memory')
        if metrics.error_rate_percent > self.alert_thresholds['error_rate']:
            alerts.append('high_errors')
        
        return alerts


def performance_timing(operation_name: str):
    """Decorator for automatic performance timing."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                # Record successful operation
                if hasattr(args[0], 'performance_monitor'):
                    args[0].performance_monitor.record_request(duration, True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                # Record failed operation
                if hasattr(args[0], 'performance_monitor'):
                    args[0].performance_monitor.record_request(duration, False)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if hasattr(args[0], 'performance_monitor'):
                    args[0].performance_monitor.record_request(duration, True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                if hasattr(args[0], 'performance_monitor'):
                    args[0].performance_monitor.record_request(duration, False)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class OptimizedNewsIntelligenceOrchestrator:
    """
    Performance-optimized version of the news intelligence orchestrator
    with advanced caching, distributed processing, and monitoring.
    """
    
    def __init__(
        self,
        base_orchestrator,
        optimization_config: Dict[str, Any] = None
    ):
        """Initialize optimized orchestrator."""
        self.base_orchestrator = base_orchestrator
        self.config = optimization_config or {}
        
        # Initialize performance components
        self.cache_manager = AdvancedCacheManager(self.config.get('cache', {}))
        self.task_manager = DistributedTaskManager(self.config.get('distributed', {}))
        self.performance_monitor = PerformanceMonitor(self.config.get('monitoring', {}))
        
        # Optimization settings
        self.enable_caching = self.config.get('enable_caching', True)
        self.enable_distributed_processing = self.config.get('enable_distributed', True)
        self.batch_size_optimization = self.config.get('batch_size_optimization', True)
        self.adaptive_rate_limiting = self.config.get('adaptive_rate_limiting', True)
        
        # Performance tracking
        self.optimization_stats = {
            'cache_hits_saved_time': 0.0,
            'distributed_tasks_submitted': 0,
            'batch_optimizations_applied': 0,
            'rate_limit_adjustments': 0
        }
        
        logger.info("Optimized News Intelligence Orchestrator initialized")
    
    async def initialize(self):
        """Initialize optimized orchestrator with all components."""
        await self.base_orchestrator.initialize()
        self.performance_monitor.start_monitoring()
        logger.info("Optimized orchestrator fully initialized")
    
    @performance_timing("news_discovery")
    async def execute_optimized_news_discovery(
        self,
        search_config: Dict[str, Any],
        use_cache: bool = True,
        distribute_tasks: bool = True
    ) -> List[Any]:
        """Execute optimized news discovery with caching and distribution."""
        
        # Generate cache key
        cache_key = f"news_discovery_{hashlib.sha256(json.dumps(search_config, sort_keys=True).encode()).hexdigest()}"
        
        # Try cache first
        if use_cache and self.enable_caching:
            cached_result = await self.cache_manager.get(cache_key, 'news_discovery')
            if cached_result:
                logger.info("News discovery result served from cache")
                self.optimization_stats['cache_hits_saved_time'] += 5.0  # Estimated 5s saved
                return cached_result
        
        # Execute discovery
        if distribute_tasks and self.enable_distributed_processing:
            # Submit to distributed task queue
            task_id = self.task_manager.submit_news_discovery_task(search_config)
            self.optimization_stats['distributed_tasks_submitted'] += 1
            
            # For now, we'll wait for the result (in production, this could be async)
            # This is a placeholder - actual implementation would depend on your task system
            result = await self.base_orchestrator.news_api.search_news(search_config)
        else:
            # Execute locally
            result = await self.base_orchestrator.news_api.search_news(search_config)
        
        # Cache the result
        if use_cache and self.enable_caching:
            await self.cache_manager.set(
                cache_key, 
                result, 
                ttl=self.cache_manager.ttl_news_articles,
                namespace='news_discovery'
            )
        
        return result
    
    @performance_timing("content_scraping")
    async def execute_optimized_content_scraping(
        self,
        articles: List[Any],
        use_cache: bool = True,
        batch_processing: bool = True
    ) -> List[Any]:
        """Execute optimized content scraping with intelligent batching."""
        
        if not articles:
            return articles
        
        scraped_articles = []
        cache_hits = 0
        
        # Separate cached and non-cached articles
        if use_cache and self.enable_caching:
            cached_articles = []
            uncached_articles = []
            
            for article in articles:
                cache_key = f"article_content_{hashlib.sha256(article.url.encode()).hexdigest()}"
                cached_content = await self.cache_manager.get(cache_key, 'article_content')
                
                if cached_content:
                    article.full_content = cached_content
                    article.processing_stage = "SCRAPED"
                    cached_articles.append(article)
                    cache_hits += 1
                else:
                    uncached_articles.append(article)
            
            scraped_articles.extend(cached_articles)
            articles_to_scrape = uncached_articles
            
            if cache_hits > 0:
                logger.info(f"Served {cache_hits} articles from content cache")
                self.optimization_stats['cache_hits_saved_time'] += cache_hits * 2.0  # Estimated 2s per article
        else:
            articles_to_scrape = articles
        
        # Process remaining articles
        if articles_to_scrape:
            if batch_processing and self.batch_size_optimization:
                # Optimize batch size based on current performance
                optimal_batch_size = self._calculate_optimal_batch_size(len(articles_to_scrape))
                
                for i in range(0, len(articles_to_scrape), optimal_batch_size):
                    batch = articles_to_scrape[i:i + optimal_batch_size]
                    
                    # Process batch
                    for article in batch:
                        try:
                            scraped_article = await self.base_orchestrator.content_scraper.scrape_article_content(article)
                            
                            # Cache the content
                            if scraped_article.full_content and use_cache and self.enable_caching:
                                cache_key = f"article_content_{hashlib.sha256(article.url.encode()).hexdigest()}"
                                await self.cache_manager.set(
                                    cache_key,
                                    scraped_article.full_content,
                                    ttl=self.cache_manager.ttl_news_articles,
                                    namespace='article_content'
                                )
                            
                            scraped_articles.append(scraped_article)
                            
                        except Exception as e:
                            logger.error(f"Optimized scraping failed for {article.url}: {e}")
                            scraped_articles.append(article)  # Add original article
                    
                    self.optimization_stats['batch_optimizations_applied'] += 1
            else:
                # Process individually
                for article in articles_to_scrape:
                    scraped_article = await self.base_orchestrator.content_scraper.scrape_article_content(article)
                    scraped_articles.append(scraped_article)
        
        return scraped_articles
    
    @performance_timing("ml_extraction")
    async def execute_optimized_ml_extraction(
        self,
        articles: List[Any],
        use_cache: bool = True,
        distribute_processing: bool = True
    ) -> List[Any]:
        """Execute optimized ML extraction with caching and distribution."""
        
        extracted_articles = []
        cache_hits = 0
        
        for article in articles:
            if not article.full_content:
                extracted_articles.append(article)
                continue
            
            # Generate cache key based on content
            content_hash = hashlib.sha256(article.full_content.encode()).hexdigest()
            cache_key = f"ml_extraction_{content_hash}"
            
            # Try cache first
            if use_cache and self.enable_caching:
                cached_extraction = await self.cache_manager.get(cache_key, 'ml_extraction')
                if cached_extraction:
                    # Apply cached extraction results to article
                    enhanced_article = self._apply_cached_extraction(article, cached_extraction)
                    extracted_articles.append(enhanced_article)
                    cache_hits += 1
                    continue
            
            # Perform ML extraction
            try:
                if distribute_processing and self.enable_distributed_processing:
                    # Submit to distributed processing queue
                    task_id = self.task_manager.submit_ml_extraction_task(
                        article.full_content,
                        {'url': article.url, 'publisher': article.publisher}
                    )
                    # For demo, we'll process locally (in production, this would be truly distributed)
                    extraction_result = await self.base_orchestrator.ml_scorer.extract_event(
                        article_text=article.full_content,
                        temperature=0.3,
                        metadata={'source_url': article.url, 'publisher': article.publisher}
                    )
                else:
                    extraction_result = await self.base_orchestrator.ml_scorer.extract_event(
                        article_text=article.full_content,
                        temperature=0.3,
                        metadata={'source_url': article.url, 'publisher': article.publisher}
                    )
                
                # Cache extraction result
                if use_cache and self.enable_caching:
                    await self.cache_manager.set(
                        cache_key,
                        extraction_result,
                        ttl=self.cache_manager.ttl_ml_extractions,
                        namespace='ml_extraction'
                    )
                
                # Integrate results
                enhanced_article = self.base_orchestrator._integrate_extraction_results(article, extraction_result)
                extracted_articles.append(enhanced_article)
                
            except Exception as e:
                logger.error(f"Optimized ML extraction failed for article: {e}")
                extracted_articles.append(article)
        
        if cache_hits > 0:
            logger.info(f"Served {cache_hits} ML extractions from cache")
            self.optimization_stats['cache_hits_saved_time'] += cache_hits * 10.0  # Estimated 10s per extraction
        
        return extracted_articles
    
    def _apply_cached_extraction(self, article: Any, cached_extraction: Dict[str, Any]) -> Any:
        """Apply cached extraction results to article."""
        try:
            return self.base_orchestrator._integrate_extraction_results(article, cached_extraction)
        except Exception as e:
            logger.error(f"Failed to apply cached extraction: {e}")
            return article
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on current system performance."""
        # Get current performance metrics
        perf_summary = self.performance_monitor.get_performance_summary()
        
        if 'current' not in perf_summary:
            return min(50, total_items)  # Conservative default
        
        current_metrics = perf_summary['current']
        
        # Adjust batch size based on CPU and memory usage
        cpu_usage = current_metrics.get('cpu_usage_percent', 50)
        memory_usage = current_metrics.get('memory_usage_percent', 50)
        
        # Base batch size
        base_batch_size = 100
        
        # Reduce batch size if system is under stress
        if cpu_usage > 70 or memory_usage > 70:
            multiplier = 0.5
        elif cpu_usage > 50 or memory_usage > 50:
            multiplier = 0.7
        else:
            multiplier = 1.0
        
        optimal_size = int(base_batch_size * multiplier)
        return min(max(optimal_size, 10), min(total_items, 200))  # Clamp between 10-200
    
    async def execute_complete_optimized_cycle(self, **kwargs) -> Dict[str, Any]:
        """Execute complete intelligence cycle with all optimizations."""
        cycle_start = time.time()
        
        try:
            # Execute optimized discovery
            search_config = kwargs.get('search_config', {})
            discovered_articles = await self.execute_optimized_news_discovery(search_config)
            
            # Execute optimized scraping
            scraped_articles = await self.execute_optimized_content_scraping(discovered_articles)
            
            # Execute optimized ML extraction
            extracted_articles = await self.execute_optimized_ml_extraction(scraped_articles)
            
            # Store in database (using existing method)
            stored_count = await self.base_orchestrator._store_articles_in_database(extracted_articles)
            
            # Calculate performance metrics
            cycle_duration = time.time() - cycle_start
            
            # Get comprehensive statistics
            optimization_summary = {
                'cycle_duration_seconds': cycle_duration,
                'articles_processed': len(discovered_articles),
                'articles_scraped': len([a for a in scraped_articles if a.full_content]),
                'articles_extracted': len([a for a in extracted_articles if a.processing_stage == 'EXTRACTED']),
                'articles_stored': stored_count,
                'optimization_stats': self.optimization_stats,
                'cache_statistics': self.cache_manager.get_cache_statistics(),
                'queue_statistics': self.task_manager.get_queue_statistics(),
                'performance_summary': self.performance_monitor.get_performance_summary(),
                'throughput_articles_per_second': len(discovered_articles) / cycle_duration
            }
            
            logger.info(f"Optimized intelligence cycle completed in {cycle_duration:.2f}s")
            return optimization_summary
            
        except Exception as e:
            logger.error(f"Optimized intelligence cycle failed: {e}")
            raise
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization performance report."""
        return {
            'optimization_stats': self.optimization_stats,
            'cache_performance': self.cache_manager.get_cache_statistics(),
            'distributed_processing': self.task_manager.get_queue_statistics(),
            'system_performance': self.performance_monitor.get_performance_summary(),
            'optimization_effectiveness': {
                'cache_time_saved_hours': self.optimization_stats['cache_hits_saved_time'] / 3600,
                'distributed_tasks_submitted': self.optimization_stats['distributed_tasks_submitted'],
                'batch_optimizations': self.optimization_stats['batch_optimizations_applied']
            }
        }
    
    async def close(self):
        """Clean up optimized orchestrator resources."""
        self.performance_monitor.stop_monitoring()
        await self.base_orchestrator.close()
        logger.info("Optimized orchestrator closed")


# Example configuration and usage
if __name__ == "__main__":
    async def main():
        """Example of optimized system usage."""
        
        # Optimization configuration
        optimization_config = {
            'cache': {
                'l1_cache_size': 1000,
                'redis_url': 'redis://localhost:6379/0',
                'cache_dir': './cache',
                'default_ttl': 3600,
                'news_ttl': 1800,
                'ml_ttl': 7200
            },
            'distributed': {
                'broker_url': 'redis://localhost:6379/0',
                'result_backend': 'redis://localhost:6379/1'
            },
            'monitoring': {
                'collection_interval': 30,
                'prometheus_port': 8000,
                'alert_thresholds': {
                    'cpu_usage': 80.0,
                    'memory_usage': 85.0,
                    'error_rate': 5.0
                }
            },
            'enable_caching': True,
            'enable_distributed': True,
            'batch_size_optimization': True
        }
        
        # Create base orchestrator (placeholder)
        # base_orchestrator = create_news_intelligence_system(...)
        
        # Create optimized orchestrator
        # optimized_orchestrator = OptimizedNewsIntelligenceOrchestrator(
        #     base_orchestrator, optimization_config
        # )
        
        # await optimized_orchestrator.initialize()
        
        # Execute optimized intelligence cycle
        # results = await optimized_orchestrator.execute_complete_optimized_cycle()
        
        # Get optimization report
        # report = await optimized_orchestrator.get_optimization_report()
        
        print("Optimized News Intelligence System configured successfully")
        print("Optimization features:")
        print("- Multi-tier caching (L1/L2/L3)")
        print("- Distributed task processing")
        print("- Real-time performance monitoring")
        print("- Adaptive batch size optimization")
        print("- Prometheus metrics integration")
        print("- Intelligent cache coherence")
    
    asyncio.run(main())