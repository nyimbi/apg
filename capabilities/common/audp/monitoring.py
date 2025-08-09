"""
Audio Processing Monitoring & Observability

Comprehensive monitoring, logging, alerting, and observability features
for production deployment and operational excellence.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import threading
import weakref

import psutil
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .models import ProcessingStatus, APTranscriptionJob, APVoiceSynthesisJob
from uuid_extensions import uuid7str


# Configure OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Custom metrics registry for detailed monitoring
MONITORING_REGISTRY = CollectorRegistry()

# Core performance metrics
PROCESSING_LATENCY = Histogram(
	'audio_processing_latency_seconds',
	'Processing latency for audio operations',
	['operation', 'provider', 'tenant_id'],
	registry=MONITORING_REGISTRY
)

ERROR_RATE = Counter(
	'audio_processing_errors_total',
	'Total number of processing errors',
	['operation', 'error_type', 'tenant_id'],
	registry=MONITORING_REGISTRY
)

THROUGHPUT = Counter(
	'audio_processing_throughput_total',
	'Total number of successful operations',
	['operation', 'tenant_id'],
	registry=MONITORING_REGISTRY
)

QUEUE_SIZE = Gauge(
	'audio_processing_queue_size',
	'Current queue size for processing operations',
	['operation', 'tenant_id'],
	registry=MONITORING_REGISTRY
)

# Resource utilization metrics
CPU_USAGE = Gauge(
	'audio_processing_cpu_usage_percent',
	'CPU usage percentage',
	registry=MONITORING_REGISTRY
)

MEMORY_USAGE = Gauge(
	'audio_processing_memory_usage_bytes',
	'Memory usage in bytes',
	registry=MONITORING_REGISTRY
)

GPU_USAGE = Gauge(
	'audio_processing_gpu_usage_percent',
	'GPU usage percentage',
	['device_id'],
	registry=MONITORING_REGISTRY
)

# Business metrics
MODEL_ACCURACY = Gauge(
	'audio_processing_model_accuracy',
	'Model accuracy scores',
	['model_type', 'model_version', 'tenant_id'],
	registry=MONITORING_REGISTRY
)

USER_SATISFACTION = Gauge(
	'audio_processing_user_satisfaction',
	'User satisfaction scores',
	['operation_type', 'tenant_id'],
	registry=MONITORING_REGISTRY
)


@dataclass
class MonitoringEvent:
	"""Structured monitoring event"""
	timestamp: datetime
	event_type: str
	operation: str
	tenant_id: str
	user_id: Optional[str] = None
	session_id: Optional[str] = None
	job_id: Optional[str] = None
	status: str = "info"
	duration_ms: Optional[float] = None
	metadata: Optional[Dict[str, Any]] = None
	error_details: Optional[Dict[str, Any]] = None


@dataclass
class AlertRule:
	"""Alert rule configuration"""
	rule_id: str
	name: str
	metric: str
	operator: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
	threshold: float
	duration_seconds: int = 300  # 5 minutes
	severity: str = "medium"  # 'low', 'medium', 'high', 'critical'
	notification_channels: List[str] = None
	enabled: bool = True


@dataclass
class Alert:
	"""Active alert"""
	alert_id: str
	rule_id: str
	title: str
	description: str
	severity: str
	status: str  # 'firing', 'resolved'
	triggered_at: datetime
	resolved_at: Optional[datetime] = None
	metadata: Optional[Dict[str, Any]] = None


class StructuredLogger:
	"""Enhanced structured logging for audio processing"""
	
	def __init__(self, name: str = "audio_processing"):
		self.logger = logging.getLogger(name)
		self._configure_logger()
		self.events: deque = deque(maxlen=10000)
		self._events_lock = threading.RLock()
	
	def _configure_logger(self) -> None:
		"""Configure structured logger with JSON formatting"""
		if not self.logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter(
				'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
			)
			handler.setFormatter(formatter)
			self.logger.addHandler(handler)
			self.logger.setLevel(logging.INFO)
	
	def log_event(self, event: MonitoringEvent) -> None:
		"""Log structured monitoring event"""
		event_dict = asdict(event)
		event_dict['timestamp'] = event.timestamp.isoformat()
		
		# Store event for analysis
		with self._events_lock:
			self.events.append(event)
		
		# Log based on status
		log_message = json.dumps(event_dict)
		
		if event.status == "error":
			self.logger.error(log_message)
		elif event.status == "warning":
			self.logger.warning(log_message)
		else:
			self.logger.info(log_message)
	
	def log_processing_start(self, operation: str, tenant_id: str, job_id: str, 
						   user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
		"""Log processing operation start"""
		event = MonitoringEvent(
			timestamp=datetime.utcnow(),
			event_type="processing_start",
			operation=operation,
			tenant_id=tenant_id,
			user_id=user_id,
			job_id=job_id,
			status="info",
			metadata=metadata
		)
		self.log_event(event)
		return event.job_id or job_id
	
	def log_processing_complete(self, operation: str, tenant_id: str, job_id: str,
							   duration_ms: float, status: str = "success",
							   metadata: Optional[Dict[str, Any]] = None) -> None:
		"""Log processing operation completion"""
		event = MonitoringEvent(
			timestamp=datetime.utcnow(),
			event_type="processing_complete",
			operation=operation,
			tenant_id=tenant_id,
			job_id=job_id,
			status=status,
			duration_ms=duration_ms,
			metadata=metadata
		)
		self.log_event(event)
	
	def log_error(self, operation: str, tenant_id: str, error: Exception,
				  job_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
		"""Log processing error"""
		error_details = {
			'error_type': type(error).__name__,
			'error_message': str(error),
			'error_traceback': getattr(error, '__traceback__', None)
		}
		
		event = MonitoringEvent(
			timestamp=datetime.utcnow(),
			event_type="processing_error",
			operation=operation,
			tenant_id=tenant_id,
			job_id=job_id,
			status="error",
			error_details=error_details,
			metadata=metadata
		)
		self.log_event(event)
	
	def get_recent_events(self, limit: int = 100, event_type: Optional[str] = None) -> List[MonitoringEvent]:
		"""Get recent monitoring events"""
		with self._events_lock:
			events = list(self.events)
		
		if event_type:
			events = [e for e in events if e.event_type == event_type]
		
		return events[-limit:]


class AlertManager:
	"""Alert management and notification system"""
	
	def __init__(self):
		self.alert_rules: Dict[str, AlertRule] = {}
		self.active_alerts: Dict[str, Alert] = {}
		self.alert_history: deque = deque(maxlen=1000)
		self.notification_handlers: Dict[str, Callable] = {}
		self._alerts_lock = threading.RLock()
		self._logger = StructuredLogger("alert_manager")
		
		# Default alert rules
		self._create_default_rules()
	
	def _create_default_rules(self) -> None:
		"""Create default alert rules"""
		default_rules = [
			AlertRule(
				rule_id="high_error_rate",
				name="High Error Rate",
				metric="error_rate",
				operator="gt",
				threshold=0.05,  # 5% error rate
				duration_seconds=300,
				severity="high",
				notification_channels=["email", "slack"]
			),
			AlertRule(
				rule_id="high_latency",
				name="High Processing Latency",
				metric="avg_latency",
				operator="gt",
				threshold=30.0,  # 30 seconds
				duration_seconds=180,
				severity="medium",
				notification_channels=["slack"]
			),
			AlertRule(
				rule_id="cpu_usage_high",
				name="High CPU Usage",
				metric="cpu_usage",
				operator="gt",
				threshold=85.0,  # 85% CPU
				duration_seconds=600,
				severity="medium",
				notification_channels=["email"]
			),
			AlertRule(
				rule_id="memory_usage_critical",
				name="Critical Memory Usage",
				metric="memory_usage",
				operator="gt",
				threshold=90.0,  # 90% memory
				duration_seconds=120,
				severity="critical",
				notification_channels=["email", "slack", "pagerduty"]
			),
			AlertRule(
				rule_id="queue_size_large",
				name="Large Processing Queue",
				metric="queue_size",
				operator="gt",
				threshold=100,
				duration_seconds=300,
				severity="medium",
				notification_channels=["slack"]
			)
		]
		
		for rule in default_rules:
			self.add_rule(rule)
	
	def add_rule(self, rule: AlertRule) -> None:
		"""Add alert rule"""
		with self._alerts_lock:
			self.alert_rules[rule.rule_id] = rule
		
		self._logger.log_event(MonitoringEvent(
			timestamp=datetime.utcnow(),
			event_type="alert_rule_added",
			operation="alert_management",
			tenant_id="system",
			status="info",
			metadata={"rule_id": rule.rule_id, "name": rule.name}
		))
	
	def remove_rule(self, rule_id: str) -> bool:
		"""Remove alert rule"""
		with self._alerts_lock:
			if rule_id in self.alert_rules:
				del self.alert_rules[rule_id]
				return True
		return False
	
	def register_notification_handler(self, channel: str, handler: Callable[[Alert], None]) -> None:
		"""Register notification handler for alert channel"""
		self.notification_handlers[channel] = handler
	
	async def check_metrics_and_alert(self, metrics: Dict[str, float], tenant_id: str = "system") -> List[Alert]:
		"""Check metrics against alert rules and fire alerts"""
		triggered_alerts = []
		
		with self._alerts_lock:
			for rule in self.alert_rules.values():
				if not rule.enabled:
					continue
				
				metric_value = metrics.get(rule.metric)
				if metric_value is None:
					continue
				
				# Evaluate rule condition
				should_alert = self._evaluate_condition(metric_value, rule.operator, rule.threshold)
				
				if should_alert:
					alert = await self._create_alert(rule, metric_value, tenant_id)
					triggered_alerts.append(alert)
				else:
					# Check if we should resolve existing alert
					await self._resolve_alert_if_exists(rule.rule_id)
		
		return triggered_alerts
	
	def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
		"""Evaluate alert condition"""
		if operator == "gt":
			return value > threshold
		elif operator == "gte":
			return value >= threshold
		elif operator == "lt":
			return value < threshold
		elif operator == "lte":
			return value <= threshold
		elif operator == "eq":
			return abs(value - threshold) < 0.001
		else:
			return False
	
	async def _create_alert(self, rule: AlertRule, metric_value: float, tenant_id: str) -> Alert:
		"""Create new alert"""
		alert = Alert(
			alert_id=uuid7str(),
			rule_id=rule.rule_id,
			title=rule.name,
			description=f"{rule.metric} is {metric_value} (threshold: {rule.threshold})",
			severity=rule.severity,
			status="firing",
			triggered_at=datetime.utcnow(),
			metadata={
				"metric_value": metric_value,
				"threshold": rule.threshold,
				"tenant_id": tenant_id
			}
		)
		
		with self._alerts_lock:
			self.active_alerts[alert.alert_id] = alert
			self.alert_history.append(alert)
		
		# Send notifications
		await self._send_notifications(alert, rule.notification_channels or [])
		
		self._logger.log_event(MonitoringEvent(
			timestamp=datetime.utcnow(),
			event_type="alert_triggered",
			operation="alert_management",
			tenant_id=tenant_id,
			status="warning",
			metadata={
				"alert_id": alert.alert_id,
				"rule_id": rule.rule_id,
				"severity": alert.severity
			}
		))
		
		return alert
	
	async def _resolve_alert_if_exists(self, rule_id: str) -> None:
		"""Resolve alert if it exists for the rule"""
		with self._alerts_lock:
			alerts_to_resolve = [
				alert for alert in self.active_alerts.values()
				if alert.rule_id == rule_id and alert.status == "firing"
			]
		
		for alert in alerts_to_resolve:
			await self.resolve_alert(alert.alert_id)
	
	async def resolve_alert(self, alert_id: str) -> bool:
		"""Resolve active alert"""
		with self._alerts_lock:
			alert = self.active_alerts.get(alert_id)
			if alert and alert.status == "firing":
				alert.status = "resolved"
				alert.resolved_at = datetime.utcnow()
				
				self._logger.log_event(MonitoringEvent(
					timestamp=datetime.utcnow(),
					event_type="alert_resolved",
					operation="alert_management",
					tenant_id=alert.metadata.get("tenant_id", "system"),
					status="info",
					metadata={
						"alert_id": alert_id,
						"rule_id": alert.rule_id
					}
				))
				
				return True
		
		return False
	
	async def _send_notifications(self, alert: Alert, channels: List[str]) -> None:
		"""Send alert notifications"""
		for channel in channels:
			handler = self.notification_handlers.get(channel)
			if handler:
				try:
					await asyncio.get_event_loop().run_in_executor(None, handler, alert)
				except Exception as e:
					self._logger.log_error("notification_send", "system", e, 
										  metadata={"channel": channel, "alert_id": alert.alert_id})
	
	def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
		"""Get currently active alerts"""
		with self._alerts_lock:
			alerts = [alert for alert in self.active_alerts.values() if alert.status == "firing"]
		
		if severity:
			alerts = [alert for alert in alerts if alert.severity == severity]
		
		return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)
	
	def get_alert_history(self, limit: int = 100) -> List[Alert]:
		"""Get alert history"""
		with self._alerts_lock:
			return list(self.alert_history)[-limit:]


class HealthChecker:
	"""Health checking for audio processing services"""
	
	def __init__(self):
		self.health_checks: Dict[str, Callable] = {}
		self.health_status: Dict[str, Dict[str, Any]] = {}
		self._health_lock = threading.RLock()
		self._logger = StructuredLogger("health_checker")
	
	def register_health_check(self, component: str, check_func: Callable[[], Dict[str, Any]]) -> None:
		"""Register health check for component"""
		self.health_checks[component] = check_func
	
	async def run_health_checks(self) -> Dict[str, Any]:
		"""Run all health checks"""
		overall_status = "healthy"
		component_statuses = {}
		
		for component, check_func in self.health_checks.items():
			try:
				status = await asyncio.get_event_loop().run_in_executor(None, check_func)
				component_statuses[component] = status
				
				if status.get("status") != "healthy":
					overall_status = "degraded"
			
			except Exception as e:
				component_statuses[component] = {
					"status": "unhealthy",
					"error": str(e),
					"checked_at": datetime.utcnow().isoformat()
				}
				overall_status = "unhealthy"
		
		health_report = {
			"overall_status": overall_status,
			"components": component_statuses,
			"checked_at": datetime.utcnow().isoformat()
		}
		
		with self._health_lock:
			self.health_status = health_report
		
		return health_report
	
	def get_health_status(self) -> Dict[str, Any]:
		"""Get current health status"""
		with self._health_lock:
			return self.health_status.copy()


class MonitoringDashboard:
	"""Monitoring dashboard for real-time metrics"""
	
	def __init__(self, logger: StructuredLogger, alert_manager: AlertManager, 
				 health_checker: HealthChecker):
		self.logger = logger
		self.alert_manager = alert_manager
		self.health_checker = health_checker
		self._metrics_cache: Dict[str, Any] = {}
		self._cache_timestamp = 0
		self._cache_ttl = 30  # 30 seconds
	
	async def get_dashboard_data(self) -> Dict[str, Any]:
		"""Get comprehensive dashboard data"""
		current_time = time.time()
		
		# Use cached data if fresh
		if current_time - self._cache_timestamp < self._cache_ttl:
			return self._metrics_cache
		
		# Gather fresh data
		dashboard_data = {
			"timestamp": datetime.utcnow().isoformat(),
			"system_health": await self.health_checker.run_health_checks(),
			"active_alerts": self.alert_manager.get_active_alerts(),
			"recent_events": self.logger.get_recent_events(limit=50),
			"performance_metrics": await self._get_performance_metrics(),
			"resource_utilization": self._get_resource_utilization(),
			"processing_stats": await self._get_processing_stats()
		}
		
		# Cache the data
		self._metrics_cache = dashboard_data
		self._cache_timestamp = current_time
		
		return dashboard_data
	
	async def _get_performance_metrics(self) -> Dict[str, Any]:
		"""Get performance metrics summary"""
		recent_events = self.logger.get_recent_events(limit=1000, event_type="processing_complete")
		
		if not recent_events:
			return {"avg_latency": 0, "throughput": 0, "error_rate": 0}
		
		# Calculate metrics
		latencies = [e.duration_ms for e in recent_events if e.duration_ms]
		errors = len([e for e in recent_events if e.status == "error"])
		
		return {
			"avg_latency": sum(latencies) / len(latencies) if latencies else 0,
			"max_latency": max(latencies) if latencies else 0,
			"throughput": len(recent_events),
			"error_rate": errors / len(recent_events) if recent_events else 0
		}
	
	def _get_resource_utilization(self) -> Dict[str, Any]:
		"""Get current resource utilization"""
		try:
			cpu_percent = psutil.cpu_percent(interval=1)
			memory = psutil.virtual_memory()
			disk = psutil.disk_usage('/')
			
			return {
				"cpu_usage": cpu_percent,
				"memory_usage": memory.percent,
				"memory_available_gb": memory.available / (1024**3),
				"disk_usage": (disk.used / disk.total) * 100,
				"disk_free_gb": disk.free / (1024**3)
			}
		except Exception as e:
			return {"error": str(e)}
	
	async def _get_processing_stats(self) -> Dict[str, Any]:
		"""Get processing operation statistics"""
		recent_events = self.logger.get_recent_events(limit=1000)
		
		stats_by_operation = defaultdict(lambda: {
			"total": 0, "successful": 0, "failed": 0, "avg_duration": 0
		})
		
		for event in recent_events:
			if event.event_type in ["processing_complete"]:
				op_stats = stats_by_operation[event.operation]
				op_stats["total"] += 1
				
				if event.status in ["success", "completed"]:
					op_stats["successful"] += 1
				else:
					op_stats["failed"] += 1
				
				if event.duration_ms:
					# Update running average
					current_avg = op_stats["avg_duration"]
					op_stats["avg_duration"] = (current_avg + event.duration_ms) / 2
		
		return dict(stats_by_operation)


# Factory functions for creating monitoring components
def create_structured_logger(name: str = "audio_processing") -> StructuredLogger:
	"""Create structured logger instance"""
	return StructuredLogger(name)


def create_alert_manager() -> AlertManager:
	"""Create alert manager instance"""
	return AlertManager()


def create_health_checker() -> HealthChecker:
	"""Create health checker instance"""
	return HealthChecker()


def create_monitoring_dashboard(logger: StructuredLogger, alert_manager: AlertManager, 
							   health_checker: HealthChecker) -> MonitoringDashboard:
	"""Create monitoring dashboard instance"""
	return MonitoringDashboard(logger, alert_manager, health_checker)


# Monitoring context manager for easy integration
@asynccontextmanager
async def monitoring_context(operation: str, tenant_id: str, logger: StructuredLogger, 
							job_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
	"""Context manager for automatic monitoring"""
	start_time = time.time()
	job_id = job_id or uuid7str()
	
	# Log operation start
	logger.log_processing_start(operation, tenant_id, job_id, metadata=metadata)
	
	try:
		yield job_id
		# Log successful completion
		duration_ms = (time.time() - start_time) * 1000
		logger.log_processing_complete(operation, tenant_id, job_id, duration_ms, "success", metadata)
		
		# Update metrics
		THROUGHPUT.labels(operation=operation, tenant_id=tenant_id).inc()
		PROCESSING_LATENCY.labels(operation=operation, provider="default", tenant_id=tenant_id).observe(duration_ms / 1000)
	
	except Exception as e:
		# Log error
		duration_ms = (time.time() - start_time) * 1000
		logger.log_error(operation, tenant_id, e, job_id, metadata)
		logger.log_processing_complete(operation, tenant_id, job_id, duration_ms, "error", metadata)
		
		# Update error metrics
		ERROR_RATE.labels(operation=operation, error_type=type(e).__name__, tenant_id=tenant_id).inc()
		
		raise


# Monitoring decorator for automatic instrumentation
def monitored_operation(operation_type: str, logger: Optional[StructuredLogger] = None):
	"""Decorator for automatic monitoring of operations"""
	def decorator(func):
		async def wrapper(*args, **kwargs):
			# Get logger instance
			monitoring_logger = logger or create_structured_logger()
			
			# Extract tenant_id from kwargs
			tenant_id = kwargs.get('tenant_id', 'default')
			
			# Generate job ID
			job_id = uuid7str()
			
			async with monitoring_context(operation_type, tenant_id, monitoring_logger, job_id):
				return await func(*args, **kwargs)
		
		return wrapper
	return decorator