#!/usr/bin/env python3
"""
APG Key Management - Monitoring and Observability
Comprehensive monitoring, metrics collection, and observability implementation

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import time
import json
import logging
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
from opentelemetry import trace, metrics as otel_metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
import aiohttp
from uuid_extensions import uuid7str

from .service import KeyManagementService


@dataclass
class MetricConfig:
	"""Configuration for metrics collection"""
	enabled: bool = True
	collection_interval: int = 30
	retention_days: int = 30
	labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertConfig:
	"""Alert configuration"""
	name: str
	condition: str
	threshold: float
	duration: int
	severity: str
	channels: List[str] = field(default_factory=list)
	enabled: bool = True


class MetricsCollector:
	"""Advanced metrics collector for key management operations"""
	
	def __init__(self, service: KeyManagementService, config: MetricConfig = None):
		self.service = service
		self.config = config or MetricConfig()
		self.registry = CollectorRegistry()
		
		# Initialize Prometheus metrics
		self._init_prometheus_metrics()
		
		# Initialize OpenTelemetry
		self._init_opentelemetry()
		
		# Metrics storage
		self.metrics_history: List[Dict[str, Any]] = []
		self._collection_task: Optional[asyncio.Task] = None
		self._is_collecting = False
	
	def _init_prometheus_metrics(self):
		"""Initialize Prometheus metrics"""
		# Counter metrics
		self.operations_counter = Counter(
			'keym_operations_total',
			'Total number of key management operations',
			['operation', 'algorithm', 'status', 'tenant_id'],
			registry=self.registry
		)
		
		self.api_requests_counter = Counter(
			'keym_api_requests_total',
			'Total number of API requests',
			['method', 'endpoint', 'status_code'],
			registry=self.registry
		)
		
		self.errors_counter = Counter(
			'keym_errors_total',
			'Total number of errors',
			['error_type', 'component'],
			registry=self.registry
		)
		
		# Histogram metrics
		self.operation_duration = Histogram(
			'keym_operation_duration_seconds',
			'Duration of key management operations',
			['operation', 'algorithm'],
			buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
			registry=self.registry
		)
		
		self.api_request_duration = Histogram(
			'keym_api_request_duration_seconds',
			'Duration of API requests',
			['method', 'endpoint'],
			buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
			registry=self.registry
		)
		
		self.hsm_operation_duration = Histogram(
			'keym_hsm_operation_duration_seconds',
			'Duration of HSM operations',
			['hsm_id', 'operation'],
			buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
			registry=self.registry
		)
		
		# Gauge metrics
		self.active_keys_gauge = Gauge(
			'keym_active_keys_total',
			'Number of active keys',
			['tenant_id', 'algorithm'],
			registry=self.registry
		)
		
		self.hsm_sessions_gauge = Gauge(
			'keym_hsm_sessions_active',
			'Number of active HSM sessions',
			['hsm_id'],
			registry=self.registry
		)
		
		self.memory_usage_gauge = Gauge(
			'keym_memory_usage_bytes',
			'Memory usage in bytes',
			registry=self.registry
		)
		
		self.cpu_usage_gauge = Gauge(
			'keym_cpu_usage_percent',
			'CPU usage percentage',
			registry=self.registry
		)
		
		self.database_connections_gauge = Gauge(
			'keym_database_connections_active',
			'Number of active database connections',
			registry=self.registry
		)
		
		# Info metrics
		self.build_info = Info(
			'keym_build_info',
			'Build information',
			registry=self.registry
		)
		
		self.build_info.info({
			'version': '1.0.0',
			'build_date': datetime.utcnow().isoformat(),
			'python_version': '3.11',
			'commit_hash': 'latest'
		})
	
	def _init_opentelemetry(self):
		"""Initialize OpenTelemetry tracing and metrics"""
		# Tracing
		trace.set_tracer_provider(TracerProvider())
		tracer = trace.get_tracer(__name__)
		
		# Jaeger exporter
		jaeger_exporter = JaegerExporter(
			agent_host_name="localhost",
			agent_port=6831,
		)
		
		span_processor = BatchSpanProcessor(jaeger_exporter)
		trace.get_tracer_provider().add_span_processor(span_processor)
		
		self.tracer = tracer
		
		# Metrics
		reader = PrometheusMetricReader()
		otel_metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
		self.meter = otel_metrics.get_meter(__name__)
	
	async def start_collection(self):
		"""Start metrics collection"""
		if self._is_collecting:
			return
		
		self._is_collecting = True
		self._collection_task = asyncio.create_task(self._collection_loop())
		logging.info("Metrics collection started")
	
	async def stop_collection(self):
		"""Stop metrics collection"""
		if not self._is_collecting:
			return
		
		self._is_collecting = False
		if self._collection_task:
			self._collection_task.cancel()
			try:
				await self._collection_task
			except asyncio.CancelledError:
				pass
		
		logging.info("Metrics collection stopped")
	
	async def _collection_loop(self):
		"""Main metrics collection loop"""
		while self._is_collecting:
			try:
				await self._collect_system_metrics()
				await self._collect_application_metrics()
				await self._collect_hsm_metrics()
				await self._collect_database_metrics()
				
				await asyncio.sleep(self.config.collection_interval)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logging.error(f"Error in metrics collection: {e}")
				await asyncio.sleep(5)
	
	async def _collect_system_metrics(self):
		"""Collect system-level metrics"""
		# Memory usage
		process = psutil.Process()
		memory_info = process.memory_info()
		self.memory_usage_gauge.set(memory_info.rss)
		
		# CPU usage
		cpu_percent = process.cpu_percent()
		self.cpu_usage_gauge.set(cpu_percent)
		
		# Store in history
		system_metrics = {
			'timestamp': datetime.utcnow().isoformat(),
			'type': 'system',
			'memory_rss': memory_info.rss,
			'memory_vms': memory_info.vms,
			'cpu_percent': cpu_percent,
			'threads': process.num_threads(),
			'open_files': len(process.open_files())
		}
		
		self.metrics_history.append(system_metrics)
	
	async def _collect_application_metrics(self):
		"""Collect application-level metrics"""
		try:
			# Active keys by tenant and algorithm
			if hasattr(self.service, '_db_pool') and self.service._db_pool:
				async with self.service._db_pool.acquire() as conn:
					# Count active keys by tenant and algorithm
					key_counts = await conn.fetch("""
						SELECT tenant_id, algorithm, COUNT(*) as count
						FROM km_keys 
						WHERE status = 'active'
						GROUP BY tenant_id, algorithm
					""")
					
					for row in key_counts:
						self.active_keys_gauge.labels(
							tenant_id=row['tenant_id'],
							algorithm=row['algorithm']
						).set(row['count'])
					
					# Database connection count
					db_connections = await conn.fetchval("""
						SELECT COUNT(*) FROM pg_stat_activity 
						WHERE datname = current_database()
					""")
					
					self.database_connections_gauge.set(db_connections)
		
		except Exception as e:
			logging.error(f"Error collecting application metrics: {e}")
	
	async def _collect_hsm_metrics(self):
		"""Collect HSM-related metrics"""
		try:
			if hasattr(self.service, 'hsm_manager') and self.service.hsm_manager:
				hsm_status = await self.service.hsm_manager.get_all_hsm_status()
				
				for hsm_id, status in hsm_status.items():
					# Active sessions
					if 'active_sessions' in status:
						self.hsm_sessions_gauge.labels(hsm_id=hsm_id).set(
							status['active_sessions']
						)
					
					# Store detailed HSM metrics
					hsm_metrics = {
						'timestamp': datetime.utcnow().isoformat(),
						'type': 'hsm',
						'hsm_id': hsm_id,
						'status': status
					}
					
					self.metrics_history.append(hsm_metrics)
		
		except Exception as e:
			logging.error(f"Error collecting HSM metrics: {e}")
	
	async def _collect_database_metrics(self):
		"""Collect database-related metrics"""
		try:
			if hasattr(self.service, '_db_pool') and self.service._db_pool:
				async with self.service._db_pool.acquire() as conn:
					# Query performance statistics
					query_stats = await conn.fetch("""
						SELECT query, calls, total_time, mean_time, rows
						FROM pg_stat_statements 
						WHERE query LIKE '%km_%'
						ORDER BY total_time DESC
						LIMIT 10
					""")
					
					# Index usage statistics
					index_stats = await conn.fetch("""
						SELECT schemaname, tablename, indexname, 
							   idx_scan, idx_tup_read, idx_tup_fetch
						FROM pg_stat_user_indexes
						WHERE schemaname = 'public'
						ORDER BY idx_scan DESC
					""")
					
					# Store database metrics
					db_metrics = {
						'timestamp': datetime.utcnow().isoformat(),
						'type': 'database',
						'query_stats': [dict(row) for row in query_stats],
						'index_stats': [dict(row) for row in index_stats]
					}
					
					self.metrics_history.append(db_metrics)
		
		except Exception as e:
			logging.error(f"Error collecting database metrics: {e}")
	
	@asynccontextmanager
	async def track_operation(self, operation: str, **labels):
		"""Context manager to track operation metrics"""
		start_time = time.time()
		
		# Start span for tracing
		with self.tracer.start_as_current_span(f"keym_{operation}") as span:
			span.set_attributes(labels)
			
			try:
				yield
				
				# Record successful operation
				duration = time.time() - start_time
				self.operation_duration.labels(
					operation=operation,
					algorithm=labels.get('algorithm', 'unknown')
				).observe(duration)
				
				self.operations_counter.labels(
					operation=operation,
					algorithm=labels.get('algorithm', 'unknown'),
					status='success',
					tenant_id=labels.get('tenant_id', 'unknown')
				).inc()
				
				span.set_attribute("success", True)
				span.set_attribute("duration", duration)
				
			except Exception as e:
				# Record failed operation
				duration = time.time() - start_time
				self.operations_counter.labels(
					operation=operation,
					algorithm=labels.get('algorithm', 'unknown'),
					status='error',
					tenant_id=labels.get('tenant_id', 'unknown')
				).inc()
				
				self.errors_counter.labels(
					error_type=type(e).__name__,
					component='key_management'
				).inc()
				
				span.set_attribute("success", False)
				span.set_attribute("error", str(e))
				span.set_attribute("duration", duration)
				
				raise
	
	def track_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
		"""Track API request metrics"""
		self.api_requests_counter.labels(
			method=method,
			endpoint=endpoint,
			status_code=str(status_code)
		).inc()
		
		self.api_request_duration.labels(
			method=method,
			endpoint=endpoint
		).observe(duration)
	
	def track_hsm_operation(self, hsm_id: str, operation: str, duration: float):
		"""Track HSM operation metrics"""
		self.hsm_operation_duration.labels(
			hsm_id=hsm_id,
			operation=operation
		).observe(duration)
	
	def get_prometheus_metrics(self) -> str:
		"""Get Prometheus formatted metrics"""
		return generate_latest(self.registry).decode('utf-8')
	
	def get_metrics_summary(self, time_range_hours: int = 24) -> Dict[str, Any]:
		"""Get metrics summary for specified time range"""
		cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
		
		relevant_metrics = [
			m for m in self.metrics_history
			if datetime.fromisoformat(m['timestamp']) > cutoff_time
		]
		
		summary = {
			'time_range_hours': time_range_hours,
			'total_metrics_points': len(relevant_metrics),
			'system_metrics': len([m for m in relevant_metrics if m['type'] == 'system']),
			'hsm_metrics': len([m for m in relevant_metrics if m['type'] == 'hsm']),
			'database_metrics': len([m for m in relevant_metrics if m['type'] == 'database'])
		}
		
		return summary


class AlertManager:
	"""Alert management system"""
	
	def __init__(self, metrics_collector: MetricsCollector):
		self.metrics_collector = metrics_collector
		self.alert_configs: Dict[str, AlertConfig] = {}
		self.active_alerts: Dict[str, Dict[str, Any]] = {}
		self.alert_history: List[Dict[str, Any]] = []
		self._monitoring_task: Optional[asyncio.Task] = None
		self._is_monitoring = False
	
	def add_alert(self, config: AlertConfig):
		"""Add alert configuration"""
		self.alert_configs[config.name] = config
		logging.info(f"Alert configured: {config.name}")
	
	async def start_monitoring(self):
		"""Start alert monitoring"""
		if self._is_monitoring:
			return
		
		self._is_monitoring = True
		self._monitoring_task = asyncio.create_task(self._monitoring_loop())
		logging.info("Alert monitoring started")
	
	async def stop_monitoring(self):
		"""Stop alert monitoring"""
		if not self._is_monitoring:
			return
		
		self._is_monitoring = False
		if self._monitoring_task:
			self._monitoring_task.cancel()
			try:
				await self._monitoring_task
			except asyncio.CancelledError:
				pass
		
		logging.info("Alert monitoring stopped")
	
	async def _monitoring_loop(self):
		"""Main alert monitoring loop"""
		while self._is_monitoring:
			try:
				await self._check_alerts()
				await asyncio.sleep(30)  # Check every 30 seconds
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logging.error(f"Error in alert monitoring: {e}")
				await asyncio.sleep(10)
	
	async def _check_alerts(self):
		"""Check all configured alerts"""
		for alert_name, config in self.alert_configs.items():
			if not config.enabled:
				continue
			
			try:
				await self._evaluate_alert(alert_name, config)
			except Exception as e:
				logging.error(f"Error evaluating alert {alert_name}: {e}")
	
	async def _evaluate_alert(self, alert_name: str, config: AlertConfig):
		"""Evaluate a specific alert"""
		# Get current metric value
		current_value = await self._get_metric_value(config.condition)
		
		if current_value is None:
			return
		
		# Check if threshold is exceeded
		threshold_exceeded = self._check_threshold(current_value, config.threshold, config.condition)
		
		if threshold_exceeded:
			if alert_name not in self.active_alerts:
				# New alert
				alert_data = {
					'alert_id': uuid7str(),
					'name': alert_name,
					'condition': config.condition,
					'threshold': config.threshold,
					'current_value': current_value,
					'severity': config.severity,
					'started_at': datetime.utcnow(),
					'duration': 0
				}
				
				self.active_alerts[alert_name] = alert_data
				await self._send_alert(alert_data, 'triggered')
			else:
				# Update existing alert
				alert_data = self.active_alerts[alert_name]
				alert_data['current_value'] = current_value
				alert_data['duration'] = (datetime.utcnow() - alert_data['started_at']).total_seconds()
		else:
			if alert_name in self.active_alerts:
				# Alert resolved
				alert_data = self.active_alerts.pop(alert_name)
				alert_data['resolved_at'] = datetime.utcnow()
				alert_data['total_duration'] = (alert_data['resolved_at'] - alert_data['started_at']).total_seconds()
				
				self.alert_history.append(alert_data)
				await self._send_alert(alert_data, 'resolved')
	
	async def _get_metric_value(self, condition: str) -> Optional[float]:
		"""Get current value for a metric condition"""
		# Simple metric value extraction
		# In a real implementation, this would parse complex conditions
		
		if 'error_rate' in condition:
			# Calculate error rate from recent metrics
			recent_metrics = self.metrics_collector.metrics_history[-10:]
			if not recent_metrics:
				return 0.0
			
			total_ops = sum(1 for m in recent_metrics if m.get('type') == 'operation')
			error_ops = sum(1 for m in recent_metrics if m.get('type') == 'operation' and m.get('status') == 'error')
			
			return error_ops / total_ops if total_ops > 0 else 0.0
		
		elif 'memory_usage' in condition:
			# Get current memory usage percentage
			process = psutil.Process()
			memory_info = process.memory_info()
			return memory_info.rss / (1024 * 1024 * 1024)  # GB
		
		elif 'cpu_usage' in condition:
			# Get current CPU usage
			process = psutil.Process()
			return process.cpu_percent()
		
		elif 'hsm_availability' in condition:
			# Calculate HSM availability
			if hasattr(self.metrics_collector.service, 'hsm_manager'):
				try:
					hsm_status = await self.metrics_collector.service.hsm_manager.get_all_hsm_status()
					online_hsms = sum(1 for status in hsm_status.values() if status.get('status') == 'online')
					total_hsms = len(hsm_status)
					
					return online_hsms / total_hsms if total_hsms > 0 else 0.0
				except Exception:
					return 0.0
		
		return None
	
	def _check_threshold(self, current_value: float, threshold: float, condition: str) -> bool:
		"""Check if current value exceeds threshold"""
		if 'greater_than' in condition or 'error_rate' in condition or 'usage' in condition:
			return current_value > threshold
		elif 'less_than' in condition or 'availability' in condition:
			return current_value < threshold
		
		return False
	
	async def _send_alert(self, alert_data: Dict[str, Any], action: str):
		"""Send alert notification"""
		alert_config = self.alert_configs[alert_data['name']]
		
		message = self._format_alert_message(alert_data, action)
		
		for channel in alert_config.channels:
			try:
				await self._send_to_channel(channel, message, alert_data)
			except Exception as e:
				logging.error(f"Failed to send alert to {channel}: {e}")
	
	def _format_alert_message(self, alert_data: Dict[str, Any], action: str) -> str:
		"""Format alert message"""
		if action == 'triggered':
			return f"""
🚨 ALERT TRIGGERED: {alert_data['name']}

Severity: {alert_data['severity']}
Condition: {alert_data['condition']}
Threshold: {alert_data['threshold']}
Current Value: {alert_data['current_value']:.4f}
Started: {alert_data['started_at'].strftime('%Y-%m-%d %H:%M:%S')} UTC

Alert ID: {alert_data['alert_id']}
"""
		else:  # resolved
			return f"""
✅ ALERT RESOLVED: {alert_data['name']}

Duration: {alert_data.get('total_duration', 0):.0f} seconds
Resolved: {alert_data.get('resolved_at', datetime.utcnow()).strftime('%Y-%m-%d %H:%M:%S')} UTC

Alert ID: {alert_data['alert_id']}
"""
	
	async def _send_to_channel(self, channel: str, message: str, alert_data: Dict[str, Any]):
		"""Send alert to specific channel"""
		if channel == 'slack':
			await self._send_slack_alert(message)
		elif channel == 'email':
			await self._send_email_alert(message, alert_data)
		elif channel == 'pagerduty':
			await self._send_pagerduty_alert(alert_data)
		elif channel == 'webhook':
			await self._send_webhook_alert(alert_data)
	
	async def _send_slack_alert(self, message: str):
		"""Send alert to Slack"""
		webhook_url = os.getenv('SLACK_WEBHOOK_URL')
		if not webhook_url:
			return
		
		payload = {
			'text': message,
			'username': 'KEYM Monitor',
			'icon_emoji': ':warning:'
		}
		
		async with aiohttp.ClientSession() as session:
			await session.post(webhook_url, json=payload)
	
	async def _send_email_alert(self, message: str, alert_data: Dict[str, Any]):
		"""Send alert via email"""
		# Implementation would depend on email service
		logging.info(f"Email alert would be sent: {message}")
	
	async def _send_pagerduty_alert(self, alert_data: Dict[str, Any]):
		"""Send alert to PagerDuty"""
		service_key = os.getenv('PAGERDUTY_SERVICE_KEY')
		if not service_key:
			return
		
		payload = {
			'service_key': service_key,
			'event_type': 'trigger',
			'description': f"KEYM Alert: {alert_data['name']}",
			'incident_key': alert_data['alert_id'],
			'details': alert_data
		}
		
		async with aiohttp.ClientSession() as session:
			await session.post(
				'https://events.pagerduty.com/generic/2010-04-15/create_event.json',
				json=payload
			)
	
	async def _send_webhook_alert(self, alert_data: Dict[str, Any]):
		"""Send alert to webhook"""
		webhook_url = os.getenv('KEYM_ALERT_WEBHOOK_URL')
		if not webhook_url:
			return
		
		async with aiohttp.ClientSession() as session:
			await session.post(webhook_url, json=alert_data)


class HealthChecker:
	"""Comprehensive health checking system"""
	
	def __init__(self, service: KeyManagementService):
		self.service = service
		self.health_checks: Dict[str, Callable] = {}
		self.last_health_status: Dict[str, Dict[str, Any]] = {}
	
	def register_check(self, name: str, check_func: Callable):
		"""Register a health check function"""
		self.health_checks[name] = check_func
		logging.info(f"Health check registered: {name}")
	
	async def check_health(self, check_names: Optional[List[str]] = None) -> Dict[str, Any]:
		"""Perform health checks"""
		checks_to_run = check_names or list(self.health_checks.keys())
		
		results = {
			'timestamp': datetime.utcnow().isoformat(),
			'overall_status': 'healthy',
			'checks': {}
		}
		
		for check_name in checks_to_run:
			if check_name not in self.health_checks:
				continue
			
			try:
				check_result = await self._run_health_check(check_name)
				results['checks'][check_name] = check_result
				
				if not check_result['healthy']:
					results['overall_status'] = 'unhealthy'
			
			except Exception as e:
				results['checks'][check_name] = {
					'healthy': False,
					'error': str(e),
					'duration_ms': 0
				}
				results['overall_status'] = 'unhealthy'
		
		self.last_health_status = results
		return results
	
	async def _run_health_check(self, check_name: str) -> Dict[str, Any]:
		"""Run a single health check"""
		start_time = time.time()
		
		try:
			check_func = self.health_checks[check_name]
			result = await check_func()
			
			duration_ms = (time.time() - start_time) * 1000
			
			if isinstance(result, bool):
				return {
					'healthy': result,
					'duration_ms': duration_ms
				}
			elif isinstance(result, dict):
				result['duration_ms'] = duration_ms
				return result
			else:
				return {
					'healthy': True,
					'result': result,
					'duration_ms': duration_ms
				}
		
		except Exception as e:
			duration_ms = (time.time() - start_time) * 1000
			return {
				'healthy': False,
				'error': str(e),
				'duration_ms': duration_ms
			}


# Factory functions
async def create_monitoring_system(service: KeyManagementService) -> Dict[str, Any]:
	"""Create complete monitoring system"""
	# Metrics collector
	metrics_config = MetricConfig(
		enabled=True,
		collection_interval=30,
		retention_days=30
	)
	metrics_collector = MetricsCollector(service, metrics_config)
	
	# Alert manager with default alerts
	alert_manager = AlertManager(metrics_collector)
	
	# Default alerts
	default_alerts = [
		AlertConfig(
			name="high_error_rate",
			condition="error_rate > threshold",
			threshold=0.01,
			duration=300,
			severity="critical",
			channels=["slack", "email", "pagerduty"]
		),
		AlertConfig(
			name="high_memory_usage",
			condition="memory_usage > threshold",
			threshold=2.0,  # 2GB
			duration=300,
			severity="warning",
			channels=["slack"]
		),
		AlertConfig(
			name="hsm_availability_low",
			condition="hsm_availability < threshold",
			threshold=0.99,
			duration=60,
			severity="critical",
			channels=["slack", "pagerduty"]
		)
	]
	
	for alert_config in default_alerts:
		alert_manager.add_alert(alert_config)
	
	# Health checker with default checks
	health_checker = HealthChecker(service)
	
	# Register default health checks
	health_checker.register_check("database", service.check_database_health)
	health_checker.register_check("cache", service.check_cache_health)
	
	if hasattr(service, 'hsm_manager'):
		health_checker.register_check("hsm", service.hsm_manager.check_health)
	
	# Start monitoring
	await metrics_collector.start_collection()
	await alert_manager.start_monitoring()
	
	return {
		'metrics_collector': metrics_collector,
		'alert_manager': alert_manager,
		'health_checker': health_checker
	}


# Export main components
__all__ = [
	'MetricsCollector', 'AlertManager', 'HealthChecker',
	'MetricConfig', 'AlertConfig',
	'create_monitoring_system'
]