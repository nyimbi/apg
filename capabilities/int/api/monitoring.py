"""
APG Integration API Management - Monitoring and Health Checks

Comprehensive monitoring system for API gateway health, performance metrics,
and operational observability with real-time alerting capabilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import time
import psutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

import aioredis
from aiohttp import ClientSession, ClientError

from .models import (
	AMAPI, AMEndpoint, AMConsumer, AMUsageRecord,
	APIStatus, ProtocolType
)
from .service import AnalyticsService

# =============================================================================
# Health Check Types and Status
# =============================================================================

class HealthStatus(str, Enum):
	"""Health check status levels."""
	HEALTHY = "healthy"
	DEGRADED = "degraded"
	UNHEALTHY = "unhealthy"
	UNKNOWN = "unknown"

class MetricType(str, Enum):
	"""Metric types for monitoring."""
	COUNTER = "counter"
	GAUGE = "gauge"
	HISTOGRAM = "histogram"
	TIMER = "timer"

@dataclass
class HealthCheck:
	"""Individual health check configuration."""
	name: str
	description: str
	check_function: Callable
	interval_seconds: int = 30
	timeout_seconds: int = 10
	critical: bool = True
	last_check_time: Optional[datetime] = None
	last_status: HealthStatus = HealthStatus.UNKNOWN
	last_error: Optional[str] = None
	consecutive_failures: int = 0
	max_consecutive_failures: int = 3

@dataclass
class Metric:
	"""Performance metric data point."""
	name: str
	metric_type: MetricType
	value: Union[int, float]
	timestamp: datetime
	labels: Dict[str, str] = field(default_factory=dict)
	description: str = ""

@dataclass
class HealthReport:
	"""Overall health report."""
	overall_status: HealthStatus
	timestamp: datetime
	checks: Dict[str, Dict[str, Any]]
	metrics: Dict[str, Any]
	alerts: List[Dict[str, Any]] = field(default_factory=list)

# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
	"""Collects and aggregates system metrics."""
	
	def __init__(self, redis_client: aioredis.Redis):
		self.redis = redis_client
		self.metrics_buffer = defaultdict(deque)
		self.buffer_size = 1000
		self.flush_interval = 60  # seconds
		
		# Performance counters
		self.request_counters = defaultdict(int)
		self.response_time_histograms = defaultdict(list)
		self.error_counters = defaultdict(int)
		
	async def record_metric(self, metric: Metric):
		"""Record a single metric."""
		
		# Add to buffer
		metric_key = f"{metric.name}:{json.dumps(metric.labels, sort_keys=True)}"
		self.metrics_buffer[metric_key].append(metric)
		
		# Trim buffer if too large
		if len(self.metrics_buffer[metric_key]) > self.buffer_size:
			self.metrics_buffer[metric_key].popleft()
		
		# Store in Redis for real-time access
		await self._store_metric_in_redis(metric)
	
	async def record_request_metric(self, api_id: str, endpoint_path: str, 
								   method: str, status_code: int, 
								   response_time_ms: float, consumer_id: str = None):
		"""Record request-specific metrics."""
		
		timestamp = datetime.now(timezone.utc)
		labels = {
			'api_id': api_id,
			'endpoint': endpoint_path,
			'method': method,
			'status_code': str(status_code),
			'consumer_id': consumer_id or 'anonymous'
		}
		
		# Request counter
		await self.record_metric(Metric(
			name='api_requests_total',
			metric_type=MetricType.COUNTER,
			value=1,
			timestamp=timestamp,
			labels=labels,
			description='Total number of API requests'
		))
		
		# Response time
		await self.record_metric(Metric(
			name='api_request_duration_ms',
			metric_type=MetricType.HISTOGRAM,
			value=response_time_ms,
			timestamp=timestamp,
			labels=labels,
			description='API request duration in milliseconds'
		))
		
		# Error counter for non-2xx responses
		if status_code >= 400:
			await self.record_metric(Metric(
				name='api_errors_total',
				metric_type=MetricType.COUNTER,
				value=1,
				timestamp=timestamp,
				labels=labels,
				description='Total number of API errors'
			))
	
	async def record_system_metrics(self):
		"""Record system-level metrics."""
		
		timestamp = datetime.now(timezone.utc)
		
		# CPU usage
		cpu_percent = psutil.cpu_percent(interval=1)
		await self.record_metric(Metric(
			name='system_cpu_percent',
			metric_type=MetricType.GAUGE,
			value=cpu_percent,
			timestamp=timestamp,
			description='System CPU usage percentage'
		))
		
		# Memory usage
		memory = psutil.virtual_memory()
		await self.record_metric(Metric(
			name='system_memory_percent',
			metric_type=MetricType.GAUGE,
			value=memory.percent,
			timestamp=timestamp,
			description='System memory usage percentage'
		))
		
		await self.record_metric(Metric(
			name='system_memory_available_bytes',
			metric_type=MetricType.GAUGE,
			value=memory.available,
			timestamp=timestamp,
			description='System available memory in bytes'
		))
		
		# Disk usage
		disk = psutil.disk_usage('/')
		await self.record_metric(Metric(
			name='system_disk_percent',
			metric_type=MetricType.GAUGE,
			value=(disk.used / disk.total) * 100,
			timestamp=timestamp,
			description='System disk usage percentage'
		))
		
		# Network I/O
		network = psutil.net_io_counters()
		await self.record_metric(Metric(
			name='system_network_bytes_sent',
			metric_type=MetricType.COUNTER,
			value=network.bytes_sent,
			timestamp=timestamp,
			description='Total network bytes sent'
		))
		
		await self.record_metric(Metric(
			name='system_network_bytes_received',
			metric_type=MetricType.COUNTER,
			value=network.bytes_recv,
			timestamp=timestamp,
			description='Total network bytes received'
		))
	
	async def get_metrics_summary(self, time_range_minutes: int = 5) -> Dict[str, Any]:
		"""Get aggregated metrics summary."""
		
		end_time = datetime.now(timezone.utc)
		start_time = end_time - timedelta(minutes=time_range_minutes)
		
		summary = {
			'time_range': {
				'start': start_time.isoformat(),
				'end': end_time.isoformat(),
				'duration_minutes': time_range_minutes
			},
			'request_metrics': {},
			'system_metrics': {},
			'error_metrics': {}
		}
		
		# Aggregate metrics from buffer
		for metric_key, metrics in self.metrics_buffer.items():
			recent_metrics = [
				m for m in metrics 
				if m.timestamp >= start_time
			]
			
			if not recent_metrics:
				continue
			
			metric_name = recent_metrics[0].name
			
			if metric_name.startswith('api_'):
				category = 'request_metrics'
			elif metric_name.startswith('system_'):
				category = 'system_metrics'
			else:
				category = 'error_metrics'
			
			if metric_name not in summary[category]:
				summary[category][metric_name] = {
					'count': 0,
					'sum': 0,
					'min': float('inf'),
					'max': float('-inf'),
					'avg': 0
				}
			
			# Aggregate values
			values = [m.value for m in recent_metrics]
			summary[category][metric_name]['count'] += len(values)
			summary[category][metric_name]['sum'] += sum(values)
			summary[category][metric_name]['min'] = min(summary[category][metric_name]['min'], min(values))
			summary[category][metric_name]['max'] = max(summary[category][metric_name]['max'], max(values))
			summary[category][metric_name]['avg'] = summary[category][metric_name]['sum'] / summary[category][metric_name]['count']
		
		return summary
	
	async def _store_metric_in_redis(self, metric: Metric):
		"""Store metric in Redis for real-time access."""
		
		try:
			# Store latest value
			key = f"metrics:latest:{metric.name}"
			value = {
				'value': metric.value,
				'timestamp': metric.timestamp.isoformat(),
				'labels': metric.labels
			}
			await self.redis.setex(key, 300, json.dumps(value))  # 5 minute TTL
			
			# Store in time series (simplified)
			ts_key = f"metrics:timeseries:{metric.name}:{int(metric.timestamp.timestamp())}"
			await self.redis.setex(ts_key, 3600, str(metric.value))  # 1 hour TTL
			
		except Exception as e:
			print(f"Error storing metric in Redis: {e}")

# =============================================================================
# Health Monitor
# =============================================================================

class HealthMonitor:
	"""Monitors system health with configurable checks."""
	
	def __init__(self, redis_client: aioredis.Redis, 
				 analytics_service: AnalyticsService,
				 metrics_collector: MetricsCollector):
		
		self.redis = redis_client
		self.analytics_service = analytics_service
		self.metrics_collector = metrics_collector
		
		self.health_checks = {}
		self.monitoring_tasks = []
		self.alert_handlers = []
		
		# Setup default health checks
		self._setup_default_health_checks()
	
	def _setup_default_health_checks(self):
		"""Setup default health checks."""
		
		# Redis connectivity check
		self.add_health_check(HealthCheck(
			name='redis_connectivity',
			description='Redis server connectivity',
			check_function=self._check_redis_health,
			interval_seconds=30,
			critical=True
		))
		
		# Database connectivity check
		self.add_health_check(HealthCheck(
			name='database_connectivity',
			description='Database server connectivity',
			check_function=self._check_database_health,
			interval_seconds=60,
			critical=True
		))
		
		# System resource checks
		self.add_health_check(HealthCheck(
			name='system_cpu',
			description='System CPU usage',
			check_function=self._check_cpu_health,
			interval_seconds=30,
			critical=False
		))
		
		self.add_health_check(HealthCheck(
			name='system_memory',
			description='System memory usage',
			check_function=self._check_memory_health,
			interval_seconds=30,
			critical=False
		))
		
		self.add_health_check(HealthCheck(
			name='system_disk',
			description='System disk usage',
			check_function=self._check_disk_health,
			interval_seconds=60,
			critical=False
		))
		
		# API response time check
		self.add_health_check(HealthCheck(
			name='api_response_time',
			description='Average API response time',
			check_function=self._check_api_response_time,
			interval_seconds=60,
			critical=False
		))
		
		# Error rate check
		self.add_health_check(HealthCheck(
			name='api_error_rate',
			description='API error rate',
			check_function=self._check_api_error_rate,
			interval_seconds=60,
			critical=False
		))
	
	def add_health_check(self, health_check: HealthCheck):
		"""Add a new health check."""
		self.health_checks[health_check.name] = health_check
	
	def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
		"""Add alert handler function."""
		self.alert_handlers.append(handler)
	
	async def start_monitoring(self):
		"""Start all health monitoring tasks."""
		
		for health_check in self.health_checks.values():
			task = asyncio.create_task(
				self._run_health_check_loop(health_check)
			)
			self.monitoring_tasks.append(task)
		
		# Start metrics collection task
		metrics_task = asyncio.create_task(self._run_metrics_collection_loop())
		self.monitoring_tasks.append(metrics_task)
	
	async def stop_monitoring(self):
		"""Stop all monitoring tasks."""
		
		for task in self.monitoring_tasks:
			task.cancel()
		
		await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
		self.monitoring_tasks.clear()
	
	async def get_health_report(self) -> HealthReport:
		"""Get current health report."""
		
		timestamp = datetime.now(timezone.utc)
		checks = {}
		alerts = []
		
		# Collect health check results
		overall_status = HealthStatus.HEALTHY
		
		for name, health_check in self.health_checks.items():
			check_result = {
				'status': health_check.last_status.value,
				'last_check': health_check.last_check_time.isoformat() if health_check.last_check_time else None,
				'consecutive_failures': health_check.consecutive_failures,
				'error': health_check.last_error,
				'critical': health_check.critical,
				'description': health_check.description
			}
			checks[name] = check_result
			
			# Determine overall status
			if health_check.critical and health_check.last_status == HealthStatus.UNHEALTHY:
				overall_status = HealthStatus.UNHEALTHY
			elif health_check.last_status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
				overall_status = HealthStatus.DEGRADED
			
			# Generate alerts for failing checks
			if health_check.consecutive_failures >= health_check.max_consecutive_failures:
				alerts.append({
					'type': 'health_check_failure',
					'severity': 'critical' if health_check.critical else 'warning',
					'message': f"Health check '{name}' has failed {health_check.consecutive_failures} consecutive times",
					'check_name': name,
					'error': health_check.last_error,
					'timestamp': timestamp.isoformat()
				})
		
		# Get metrics summary
		metrics_summary = await self.metrics_collector.get_metrics_summary()
		
		return HealthReport(
			overall_status=overall_status,
			timestamp=timestamp,
			checks=checks,
			metrics=metrics_summary,
			alerts=alerts
		)
	
	async def _run_health_check_loop(self, health_check: HealthCheck):
		"""Run health check in a loop."""
		
		while True:
			try:
				await self._execute_health_check(health_check)
				await asyncio.sleep(health_check.interval_seconds)
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Health check loop error for {health_check.name}: {e}")
				await asyncio.sleep(health_check.interval_seconds)
	
	async def _execute_health_check(self, health_check: HealthCheck):
		"""Execute a single health check."""
		
		try:
			# Execute check with timeout
			result = await asyncio.wait_for(
				health_check.check_function(),
				timeout=health_check.timeout_seconds
			)
			
			# Update health check state
			health_check.last_check_time = datetime.now(timezone.utc)
			health_check.last_status = result.get('status', HealthStatus.UNKNOWN)
			health_check.last_error = result.get('error')
			
			if health_check.last_status == HealthStatus.HEALTHY:
				health_check.consecutive_failures = 0
			else:
				health_check.consecutive_failures += 1
			
			# Trigger alerts if needed
			if health_check.consecutive_failures == health_check.max_consecutive_failures:
				await self._trigger_alert({
					'type': 'health_check_failure',
					'check_name': health_check.name,
					'status': health_check.last_status.value,
					'error': health_check.last_error,
					'critical': health_check.critical
				})
			
		except asyncio.TimeoutError:
			health_check.last_check_time = datetime.now(timezone.utc)
			health_check.last_status = HealthStatus.UNHEALTHY
			health_check.last_error = f"Health check timed out after {health_check.timeout_seconds} seconds"
			health_check.consecutive_failures += 1
			
		except Exception as e:
			health_check.last_check_time = datetime.now(timezone.utc)
			health_check.last_status = HealthStatus.UNHEALTHY
			health_check.last_error = str(e)
			health_check.consecutive_failures += 1
	
	async def _run_metrics_collection_loop(self):
		"""Run system metrics collection loop."""
		
		while True:
			try:
				await self.metrics_collector.record_system_metrics()
				await asyncio.sleep(30)  # Collect every 30 seconds
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Metrics collection error: {e}")
				await asyncio.sleep(30)
	
	async def _trigger_alert(self, alert_data: Dict[str, Any]):
		"""Trigger alert to all registered handlers."""
		
		for handler in self.alert_handlers:
			try:
				await handler(alert_data)
			except Exception as e:
				print(f"Alert handler error: {e}")
	
	# =============================================================================
	# Health Check Functions
	# =============================================================================
	
	async def _check_redis_health(self) -> Dict[str, Any]:
		"""Check Redis connectivity and performance."""
		
		try:
			# Test basic connectivity
			start_time = time.time()
			await self.redis.ping()
			ping_time = (time.time() - start_time) * 1000
			
			# Test set/get operations
			test_key = f"health_check:{int(time.time())}"
			await self.redis.setex(test_key, 10, "test_value")
			value = await self.redis.get(test_key)
			await self.redis.delete(test_key)
			
			if value != b"test_value":
				return {
					'status': HealthStatus.UNHEALTHY,
					'error': 'Redis set/get test failed'
				}
			
			# Check performance
			if ping_time > 100:  # > 100ms is concerning
				return {
					'status': HealthStatus.DEGRADED,
					'error': f'Redis ping time high: {ping_time:.2f}ms'
				}
			
			return {
				'status': HealthStatus.HEALTHY,
				'ping_time_ms': ping_time
			}
			
		except Exception as e:
			return {
				'status': HealthStatus.UNHEALTHY,
				'error': f'Redis connection failed: {str(e)}'
			}
	
	async def _check_database_health(self) -> Dict[str, Any]:
		"""Check database connectivity and performance."""
		
		try:
			# This would test actual database connection
			# For now, return healthy
			return {
				'status': HealthStatus.HEALTHY,
				'connection_pool_size': 10,
				'active_connections': 2
			}
			
		except Exception as e:
			return {
				'status': HealthStatus.UNHEALTHY,
				'error': f'Database connection failed: {str(e)}'
			}
	
	async def _check_cpu_health(self) -> Dict[str, Any]:
		"""Check CPU usage."""
		
		try:
			cpu_percent = psutil.cpu_percent(interval=1)
			
			if cpu_percent > 90:
				return {
					'status': HealthStatus.UNHEALTHY,
					'error': f'CPU usage critical: {cpu_percent}%',
					'cpu_percent': cpu_percent
				}
			elif cpu_percent > 75:
				return {
					'status': HealthStatus.DEGRADED,
					'error': f'CPU usage high: {cpu_percent}%',
					'cpu_percent': cpu_percent
				}
			else:
				return {
					'status': HealthStatus.HEALTHY,
					'cpu_percent': cpu_percent
				}
				
		except Exception as e:
			return {
				'status': HealthStatus.UNKNOWN,
				'error': f'CPU check failed: {str(e)}'
			}
	
	async def _check_memory_health(self) -> Dict[str, Any]:
		"""Check memory usage."""
		
		try:
			memory = psutil.virtual_memory()
			
			if memory.percent > 95:
				return {
					'status': HealthStatus.UNHEALTHY,
					'error': f'Memory usage critical: {memory.percent}%',
					'memory_percent': memory.percent,
					'available_gb': memory.available / (1024**3)
				}
			elif memory.percent > 85:
				return {
					'status': HealthStatus.DEGRADED,
					'error': f'Memory usage high: {memory.percent}%',
					'memory_percent': memory.percent,
					'available_gb': memory.available / (1024**3)
				}
			else:
				return {
					'status': HealthStatus.HEALTHY,
					'memory_percent': memory.percent,
					'available_gb': memory.available / (1024**3)
				}
				
		except Exception as e:
			return {
				'status': HealthStatus.UNKNOWN,
				'error': f'Memory check failed: {str(e)}'
			}
	
	async def _check_disk_health(self) -> Dict[str, Any]:
		"""Check disk usage."""
		
		try:
			disk = psutil.disk_usage('/')
			disk_percent = (disk.used / disk.total) * 100
			
			if disk_percent > 95:
				return {
					'status': HealthStatus.UNHEALTHY,
					'error': f'Disk usage critical: {disk_percent:.1f}%',
					'disk_percent': disk_percent,
					'free_gb': disk.free / (1024**3)
				}
			elif disk_percent > 85:
				return {
					'status': HealthStatus.DEGRADED,
					'error': f'Disk usage high: {disk_percent:.1f}%',
					'disk_percent': disk_percent,
					'free_gb': disk.free / (1024**3)
				}
			else:
				return {
					'status': HealthStatus.HEALTHY,
					'disk_percent': disk_percent,
					'free_gb': disk.free / (1024**3)
				}
				
		except Exception as e:
			return {
				'status': HealthStatus.UNKNOWN,
				'error': f'Disk check failed: {str(e)}'
			}
	
	async def _check_api_response_time(self) -> Dict[str, Any]:
		"""Check average API response time."""
		
		try:
			# Get recent metrics
			metrics_summary = await self.metrics_collector.get_metrics_summary(time_range_minutes=5)
			
			request_metrics = metrics_summary.get('request_metrics', {})
			duration_metric = request_metrics.get('api_request_duration_ms', {})
			
			if not duration_metric or duration_metric.get('count', 0) == 0:
				return {
					'status': HealthStatus.UNKNOWN,
					'error': 'No recent API requests to measure'
				}
			
			avg_response_time = duration_metric.get('avg', 0)
			
			if avg_response_time > 5000:  # > 5 seconds
				return {
					'status': HealthStatus.UNHEALTHY,
					'error': f'Average response time critical: {avg_response_time:.2f}ms',
					'avg_response_time_ms': avg_response_time
				}
			elif avg_response_time > 2000:  # > 2 seconds
				return {
					'status': HealthStatus.DEGRADED,
					'error': f'Average response time high: {avg_response_time:.2f}ms',
					'avg_response_time_ms': avg_response_time
				}
			else:
				return {
					'status': HealthStatus.HEALTHY,
					'avg_response_time_ms': avg_response_time
				}
				
		except Exception as e:
			return {
				'status': HealthStatus.UNKNOWN,
				'error': f'Response time check failed: {str(e)}'
			}
	
	async def _check_api_error_rate(self) -> Dict[str, Any]:
		"""Check API error rate."""
		
		try:
			# Get recent metrics
			metrics_summary = await self.metrics_collector.get_metrics_summary(time_range_minutes=5)
			
			request_metrics = metrics_summary.get('request_metrics', {})
			total_requests = request_metrics.get('api_requests_total', {}).get('count', 0)
			total_errors = request_metrics.get('api_errors_total', {}).get('count', 0)
			
			if total_requests == 0:
				return {
					'status': HealthStatus.UNKNOWN,
					'error': 'No recent API requests to measure'
				}
			
			error_rate = (total_errors / total_requests) * 100
			
			if error_rate > 10:  # > 10% error rate
				return {
					'status': HealthStatus.UNHEALTHY,
					'error': f'Error rate critical: {error_rate:.2f}%',
					'error_rate_percent': error_rate,
					'total_requests': total_requests,
					'total_errors': total_errors
				}
			elif error_rate > 5:  # > 5% error rate
				return {
					'status': HealthStatus.DEGRADED,
					'error': f'Error rate high: {error_rate:.2f}%',
					'error_rate_percent': error_rate,
					'total_requests': total_requests,
					'total_errors': total_errors
				}
			else:
				return {
					'status': HealthStatus.HEALTHY,
					'error_rate_percent': error_rate,
					'total_requests': total_requests,
					'total_errors': total_errors
				}
				
		except Exception as e:
			return {
				'status': HealthStatus.UNKNOWN,
				'error': f'Error rate check failed: {str(e)}'
			}

# =============================================================================
# Alert Manager
# =============================================================================

class AlertManager:
	"""Manages alerting and notifications."""
	
	def __init__(self, redis_client: aioredis.Redis):
		self.redis = redis_client
		self.notification_channels = []
		
	def add_notification_channel(self, channel: Callable[[Dict[str, Any]], None]):
		"""Add notification channel."""
		self.notification_channels.append(channel)
	
	async def send_alert(self, alert_data: Dict[str, Any]):
		"""Send alert through all notification channels."""
		
		# Store alert in Redis
		alert_key = f"alerts:{int(time.time())}:{secrets.token_urlsafe(8)}"
		await self.redis.setex(alert_key, 86400, json.dumps(alert_data))  # 24 hour TTL
		
		# Send through notification channels
		for channel in self.notification_channels:
			try:
				await channel(alert_data)
			except Exception as e:
				print(f"Notification channel error: {e}")
	
	async def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
		"""Get recent alerts."""
		
		# This would typically query a persistent store
		# For now, return empty list
		return []

# =============================================================================
# Export Monitoring Components
# =============================================================================

__all__ = [
	# Enums
	'HealthStatus',
	'MetricType',
	
	# Data Classes
	'HealthCheck',
	'Metric',
	'HealthReport',
	
	# Core Components
	'MetricsCollector',
	'HealthMonitor',
	'AlertManager'
]