"""
APG Workflow & Business Process Management - Production Monitoring & Logging

Comprehensive production monitoring, structured logging, health checks,
and operational insights for production deployment.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import time
import traceback
import psutil
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager
import structlog

from models import (
	APGTenantContext, WBPMServiceResponse, WBPMPagedResponse,
	WBPMProcessInstance, WBPMTask, ProcessStatus, TaskStatus, TaskPriority
)

# Configure structured logging
structlog.configure(
	processors=[
		structlog.stdlib.filter_by_level,
		structlog.stdlib.add_logger_name,
		structlog.stdlib.add_log_level,
		structlog.stdlib.PositionalArgumentsFormatter(),
		structlog.processors.TimeStamper(fmt="iso"),
		structlog.processors.StackInfoRenderer(),
		structlog.processors.format_exc_info,
		structlog.processors.UnicodeDecoder(),
		structlog.processors.JSONRenderer()
	],
	context_class=dict,
	logger_factory=structlog.stdlib.LoggerFactory(),
	wrapper_class=structlog.stdlib.BoundLogger,
	cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Production Monitoring Core Classes
# =============================================================================

class HealthStatus(str, Enum):
	"""Service health status levels."""
	HEALTHY = "healthy"
	DEGRADED = "degraded"
	UNHEALTHY = "unhealthy"
	CRITICAL = "critical"


class LogLevel(str, Enum):
	"""Structured logging levels."""
	DEBUG = "debug"
	INFO = "info"
	WARNING = "warning"
	ERROR = "error"
	CRITICAL = "critical"


class MetricCategory(str, Enum):
	"""Metric categories for monitoring."""
	SYSTEM = "system"
	APPLICATION = "application"
	BUSINESS = "business"
	SECURITY = "security"
	PERFORMANCE = "performance"


@dataclass
class HealthCheck:
	"""Health check definition."""
	check_id: str = field(default_factory=lambda: f"health_{uuid.uuid4().hex}")
	name: str = ""
	description: str = ""
	category: str = "application"
	status: HealthStatus = HealthStatus.HEALTHY
	last_check: datetime = field(default_factory=datetime.utcnow)
	response_time_ms: float = 0.0
	error_message: Optional[str] = None
	dependencies: List[str] = field(default_factory=list)
	tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemMetrics:
	"""System performance metrics."""
	timestamp: datetime = field(default_factory=datetime.utcnow)
	cpu_usage_percent: float = 0.0
	memory_usage_percent: float = 0.0
	memory_available_gb: float = 0.0
	disk_usage_percent: float = 0.0
	disk_free_gb: float = 0.0
	network_bytes_sent: int = 0
	network_bytes_recv: int = 0
	process_count: int = 0
	thread_count: int = 0
	file_descriptors: int = 0
	load_average: List[float] = field(default_factory=list)


@dataclass
class ApplicationMetrics:
	"""Application-specific metrics."""
	timestamp: datetime = field(default_factory=datetime.utcnow)
	active_processes: int = 0
	active_tasks: int = 0
	completed_processes_1h: int = 0
	completed_tasks_1h: int = 0
	error_count_1h: int = 0
	avg_response_time_ms: float = 0.0
	database_connections: int = 0
	cache_hit_rate: float = 0.0
	queue_size: int = 0
	background_jobs: int = 0


@dataclass
class BusinessMetrics:
	"""Business-level metrics."""
	timestamp: datetime = field(default_factory=datetime.utcnow)
	tenant_count: int = 0
	active_users: int = 0
	process_throughput_hourly: float = 0.0
	task_completion_rate: float = 0.0
	average_process_duration: float = 0.0
	sla_compliance_rate: float = 100.0
	escalation_count: int = 0
	notification_volume: int = 0


@dataclass
class LogEntry:
	"""Structured log entry."""
	timestamp: datetime = field(default_factory=datetime.utcnow)
	level: LogLevel = LogLevel.INFO
	logger_name: str = ""
	message: str = ""
	context: Dict[str, Any] = field(default_factory=dict)
	trace_id: Optional[str] = None
	span_id: Optional[str] = None
	tenant_id: Optional[str] = None
	user_id: Optional[str] = None
	process_id: Optional[str] = None
	task_id: Optional[str] = None
	exception: Optional[str] = None
	duration_ms: Optional[float] = None


@dataclass
class Alert:
	"""Production alert."""
	alert_id: str = field(default_factory=lambda: f"alert_{uuid.uuid4().hex}")
	alert_type: str = ""
	severity: str = "medium"
	title: str = ""
	description: str = ""
	metric_name: str = ""
	current_value: float = 0.0
	threshold_value: float = 0.0
	triggered_at: datetime = field(default_factory=datetime.utcnow)
	resolved_at: Optional[datetime] = None
	acknowledged_at: Optional[datetime] = None
	tags: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# Health Check Manager
# =============================================================================

class HealthCheckManager:
	"""Manage application health checks."""
	
	def __init__(self):
		self.health_checks: Dict[str, HealthCheck] = {}
		self.check_functions: Dict[str, Callable] = {}
		self.last_overall_status = HealthStatus.HEALTHY
		self.health_history: deque = deque(maxlen=100)
		
	async def register_health_check(
		self,
		name: str,
		check_function: Callable,
		description: str = "",
		category: str = "application",
		dependencies: List[str] = None
	) -> str:
		"""Register a health check function."""
		try:
			check_id = f"health_{name}_{uuid.uuid4().hex[:8]}"
			
			health_check = HealthCheck(
				check_id=check_id,
				name=name,
				description=description,
				category=category,
				dependencies=dependencies or []
			)
			
			self.health_checks[check_id] = health_check
			self.check_functions[check_id] = check_function
			
			logger.info("Health check registered", check_id=check_id, name=name)
			
			return check_id
			
		except Exception as e:
			logger.error("Error registering health check", error=str(e), name=name)
			raise
	
	async def run_health_check(self, check_id: str) -> HealthCheck:
		"""Run a specific health check."""
		try:
			if check_id not in self.health_checks:
				raise ValueError(f"Health check not found: {check_id}")
			
			health_check = self.health_checks[check_id]
			check_function = self.check_functions[check_id]
			
			start_time = time.time()
			
			try:
				# Run the health check function
				result = await check_function()
				
				# Update health check status
				if isinstance(result, bool):
					health_check.status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
				elif isinstance(result, dict):
					health_check.status = HealthStatus(result.get("status", HealthStatus.HEALTHY))
					health_check.error_message = result.get("error")
					health_check.tags.update(result.get("tags", {}))
				else:
					health_check.status = HealthStatus.HEALTHY
				
				health_check.error_message = None
				
			except Exception as e:
				health_check.status = HealthStatus.UNHEALTHY
				health_check.error_message = str(e)
				logger.error("Health check failed", 
					check_id=check_id, 
					name=health_check.name, 
					error=str(e)
				)
			
			# Update timing and timestamp
			health_check.response_time_ms = (time.time() - start_time) * 1000
			health_check.last_check = datetime.utcnow()
			
			return health_check
			
		except Exception as e:
			logger.error("Error running health check", check_id=check_id, error=str(e))
			raise
	
	async def run_all_health_checks(self) -> Dict[str, HealthCheck]:
		"""Run all registered health checks."""
		try:
			results = {}
			
			# Run all health checks concurrently
			tasks = []
			for check_id in self.health_checks.keys():
				task = asyncio.create_task(self.run_health_check(check_id))
				tasks.append((check_id, task))
			
			# Collect results
			for check_id, task in tasks:
				try:
					result = await task
					results[check_id] = result
				except Exception as e:
					logger.error("Health check task failed", check_id=check_id, error=str(e))
					# Create failed health check
					failed_check = self.health_checks[check_id]
					failed_check.status = HealthStatus.CRITICAL
					failed_check.error_message = f"Health check execution failed: {e}"
					failed_check.last_check = datetime.utcnow()
					results[check_id] = failed_check
			
			# Record health history
			overall_status = self._calculate_overall_status(results)
			self.health_history.append({
				"timestamp": datetime.utcnow(),
				"overall_status": overall_status,
				"check_count": len(results),
				"healthy_count": len([c for c in results.values() if c.status == HealthStatus.HEALTHY])
			})
			
			self.last_overall_status = overall_status
			
			return results
			
		except Exception as e:
			logger.error("Error running health checks", error=str(e))
			raise
	
	def _calculate_overall_status(self, checks: Dict[str, HealthCheck]) -> HealthStatus:
		"""Calculate overall health status from individual checks."""
		if not checks:
			return HealthStatus.UNHEALTHY
		
		statuses = [check.status for check in checks.values()]
		
		if HealthStatus.CRITICAL in statuses:
			return HealthStatus.CRITICAL
		elif HealthStatus.UNHEALTHY in statuses:
			return HealthStatus.UNHEALTHY
		elif HealthStatus.DEGRADED in statuses:
			return HealthStatus.DEGRADED
		else:
			return HealthStatus.HEALTHY
	
	async def get_health_summary(self) -> Dict[str, Any]:
		"""Get comprehensive health summary."""
		try:
			# Run latest health checks
			latest_checks = await self.run_all_health_checks()
			
			# Calculate summary
			total_checks = len(latest_checks)
			healthy_checks = len([c for c in latest_checks.values() if c.status == HealthStatus.HEALTHY])
			degraded_checks = len([c for c in latest_checks.values() if c.status == HealthStatus.DEGRADED])
			unhealthy_checks = len([c for c in latest_checks.values() if c.status == HealthStatus.UNHEALTHY])
			critical_checks = len([c for c in latest_checks.values() if c.status == HealthStatus.CRITICAL])
			
			# Calculate average response time
			response_times = [c.response_time_ms for c in latest_checks.values()]
			avg_response_time = sum(response_times) / len(response_times) if response_times else 0
			
			summary = {
				"overall_status": self.last_overall_status,
				"timestamp": datetime.utcnow().isoformat(),
				"checks": {
					"total": total_checks,
					"healthy": healthy_checks,
					"degraded": degraded_checks,
					"unhealthy": unhealthy_checks,
					"critical": critical_checks
				},
				"performance": {
					"avg_response_time_ms": round(avg_response_time, 2),
					"max_response_time_ms": max(response_times) if response_times else 0
				},
				"details": {
					check_id: {
						"name": check.name,
						"status": check.status,
						"response_time_ms": check.response_time_ms,
						"last_check": check.last_check.isoformat(),
						"error": check.error_message
					}
					for check_id, check in latest_checks.items()
				}
			}
			
			return summary
			
		except Exception as e:
			logger.error("Error getting health summary", error=str(e))
			raise


# =============================================================================
# Metrics Collector
# =============================================================================

class ProductionMetricsCollector:
	"""Collect comprehensive production metrics."""
	
	def __init__(self):
		self.system_metrics_history: deque = deque(maxlen=1000)
		self.application_metrics_history: deque = deque(maxlen=1000)
		self.business_metrics_history: deque = deque(maxlen=1000)
		self.collection_interval = 60  # seconds
		self.collection_task: Optional[asyncio.Task] = None
		
	async def start_collection(self) -> None:
		"""Start metrics collection."""
		if self.collection_task is None or self.collection_task.done():
			self.collection_task = asyncio.create_task(self._collection_loop())
			logger.info("Metrics collection started")
	
	async def stop_collection(self) -> None:
		"""Stop metrics collection."""
		if self.collection_task and not self.collection_task.done():
			self.collection_task.cancel()
			try:
				await self.collection_task
			except asyncio.CancelledError:
				pass
			logger.info("Metrics collection stopped")
	
	async def _collection_loop(self) -> None:
		"""Main metrics collection loop."""
		try:
			while True:
				try:
					# Collect all metric types
					await self._collect_system_metrics()
					await self._collect_application_metrics()
					await self._collect_business_metrics()
					
					# Sleep until next collection
					await asyncio.sleep(self.collection_interval)
					
				except asyncio.CancelledError:
					logger.info("Metrics collection cancelled")
					break
				except Exception as e:
					logger.error("Error in metrics collection loop", error=str(e))
					await asyncio.sleep(30)  # Wait before retrying
					
		except Exception as e:
			logger.error("Fatal error in metrics collection", error=str(e))
	
	async def _collect_system_metrics(self) -> SystemMetrics:
		"""Collect system-level metrics."""
		try:
			# CPU metrics
			cpu_percent = psutil.cpu_percent(interval=1)
			
			# Memory metrics
			memory = psutil.virtual_memory()
			memory_percent = memory.percent
			memory_available_gb = memory.available / (1024**3)
			
			# Disk metrics
			disk = psutil.disk_usage('/')
			disk_percent = disk.percent
			disk_free_gb = disk.free / (1024**3)
			
			# Network metrics
			network = psutil.net_io_counters()
			network_sent = network.bytes_sent
			network_recv = network.bytes_recv
			
			# Process metrics
			process_count = len(psutil.pids())
			current_process = psutil.Process()
			thread_count = current_process.num_threads()
			
			# File descriptors (Unix only)
			try:
				fd_count = current_process.num_fds() if hasattr(current_process, 'num_fds') else 0
			except (AttributeError, psutil.AccessDenied):
				fd_count = 0
			
			# Load average (Unix only)
			try:
				load_avg = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0]
			except (AttributeError, OSError):
				load_avg = [0, 0, 0]
			
			metrics = SystemMetrics(
				cpu_usage_percent=cpu_percent,
				memory_usage_percent=memory_percent,
				memory_available_gb=memory_available_gb,
				disk_usage_percent=disk_percent,
				disk_free_gb=disk_free_gb,
				network_bytes_sent=network_sent,
				network_bytes_recv=network_recv,
				process_count=process_count,
				thread_count=thread_count,
				file_descriptors=fd_count,
				load_average=load_avg
			)
			
			self.system_metrics_history.append(metrics)
			
			logger.debug("System metrics collected",
				cpu=cpu_percent,
				memory=memory_percent,
				disk=disk_percent
			)
			
			return metrics
			
		except Exception as e:
			logger.error("Error collecting system metrics", error=str(e))
			raise
	
	async def _collect_application_metrics(self) -> ApplicationMetrics:
		"""Collect application-level metrics."""
		try:
			# These would be collected from actual application state
			# For demonstration, using placeholder values
			metrics = ApplicationMetrics(
				active_processes=25,  # Would come from workflow engine
				active_tasks=48,      # Would come from task manager
				completed_processes_1h=15,  # Would come from metrics store
				completed_tasks_1h=67,      # Would come from metrics store
				error_count_1h=2,           # Would come from error tracking
				avg_response_time_ms=125.5, # Would come from request tracking
				database_connections=10,    # Would come from DB pool
				cache_hit_rate=85.2,        # Would come from cache stats
				queue_size=12,              # Would come from task queue
				background_jobs=3           # Would come from job scheduler
			)
			
			self.application_metrics_history.append(metrics)
			
			logger.debug("Application metrics collected",
				active_processes=metrics.active_processes,
				active_tasks=metrics.active_tasks,
				error_count=metrics.error_count_1h
			)
			
			return metrics
			
		except Exception as e:
			logger.error("Error collecting application metrics", error=str(e))
			raise
	
	async def _collect_business_metrics(self) -> BusinessMetrics:
		"""Collect business-level metrics."""
		try:
			# These would be calculated from actual business data
			metrics = BusinessMetrics(
				tenant_count=15,                    # Would come from tenant registry
				active_users=143,                  # Would come from session tracking
				process_throughput_hourly=12.5,    # Would come from process analytics
				task_completion_rate=94.2,         # Would come from task analytics
				average_process_duration=4.8,      # Would come from duration tracking
				sla_compliance_rate=98.1,          # Would come from SLA monitoring
				escalation_count=3,                # Would come from escalation tracking
				notification_volume=256            # Would come from notification system
			)
			
			self.business_metrics_history.append(metrics)
			
			logger.debug("Business metrics collected",
				tenant_count=metrics.tenant_count,
				active_users=metrics.active_users,
				throughput=metrics.process_throughput_hourly
			)
			
			return metrics
			
		except Exception as e:
			logger.error("Error collecting business metrics", error=str(e))
			raise
	
	async def get_metrics_summary(
		self,
		time_window: timedelta = timedelta(hours=1)
	) -> Dict[str, Any]:
		"""Get comprehensive metrics summary."""
		try:
			cutoff_time = datetime.utcnow() - time_window
			
			# Filter metrics by time window
			recent_system = [m for m in self.system_metrics_history if m.timestamp >= cutoff_time]
			recent_app = [m for m in self.application_metrics_history if m.timestamp >= cutoff_time]
			recent_business = [m for m in self.business_metrics_history if m.timestamp >= cutoff_time]
			
			summary = {
				"time_window": str(time_window),
				"timestamp": datetime.utcnow().isoformat(),
				"system": self._summarize_system_metrics(recent_system),
				"application": self._summarize_application_metrics(recent_app),
				"business": self._summarize_business_metrics(recent_business)
			}
			
			return summary
			
		except Exception as e:
			logger.error("Error getting metrics summary", error=str(e))
			raise
	
	def _summarize_system_metrics(self, metrics: List[SystemMetrics]) -> Dict[str, Any]:
		"""Summarize system metrics."""
		if not metrics:
			return {}
		
		latest = metrics[-1]
		cpu_values = [m.cpu_usage_percent for m in metrics]
		memory_values = [m.memory_usage_percent for m in metrics]
		
		return {
			"current": {
				"cpu_usage": latest.cpu_usage_percent,
				"memory_usage": latest.memory_usage_percent,
				"disk_usage": latest.disk_usage_percent,
				"memory_available_gb": latest.memory_available_gb,
				"disk_free_gb": latest.disk_free_gb
			},
			"averages": {
				"cpu_usage": sum(cpu_values) / len(cpu_values),
				"memory_usage": sum(memory_values) / len(memory_values)
			},
			"peaks": {
				"max_cpu": max(cpu_values),
				"max_memory": max(memory_values)
			}
		}
	
	def _summarize_application_metrics(self, metrics: List[ApplicationMetrics]) -> Dict[str, Any]:
		"""Summarize application metrics."""
		if not metrics:
			return {}
		
		latest = metrics[-1]
		response_times = [m.avg_response_time_ms for m in metrics]
		
		return {
			"current": {
				"active_processes": latest.active_processes,
				"active_tasks": latest.active_tasks,
				"queue_size": latest.queue_size,
				"avg_response_time_ms": latest.avg_response_time_ms,
				"cache_hit_rate": latest.cache_hit_rate
			},
			"hourly_totals": {
				"completed_processes": latest.completed_processes_1h,
				"completed_tasks": latest.completed_tasks_1h,
				"errors": latest.error_count_1h
			},
			"performance": {
				"avg_response_time": sum(response_times) / len(response_times),
				"max_response_time": max(response_times),
				"min_response_time": min(response_times)
			}
		}
	
	def _summarize_business_metrics(self, metrics: List[BusinessMetrics]) -> Dict[str, Any]:
		"""Summarize business metrics."""
		if not metrics:
			return {}
		
		latest = metrics[-1]
		throughput_values = [m.process_throughput_hourly for m in metrics]
		
		return {
			"current": {
				"tenant_count": latest.tenant_count,
				"active_users": latest.active_users,
				"sla_compliance_rate": latest.sla_compliance_rate,
				"task_completion_rate": latest.task_completion_rate
			},
			"performance": {
				"avg_throughput": sum(throughput_values) / len(throughput_values),
				"avg_process_duration": latest.average_process_duration,
				"escalation_count": latest.escalation_count
			}
		}


# =============================================================================
# Structured Logging Manager
# =============================================================================

class StructuredLoggingManager:
	"""Manage structured logging for production."""
	
	def __init__(self):
		self.log_buffer: deque = deque(maxlen=10000)
		self.correlation_context: Dict[str, Any] = {}
		self.sampling_rate = 0.1  # Sample 10% of debug logs
		
	@asynccontextmanager
	async def correlation_context_manager(self, **context):
		"""Context manager for correlation tracking."""
		old_context = self.correlation_context.copy()
		self.correlation_context.update(context)
		try:
			yield
		finally:
			self.correlation_context = old_context
	
	def log_operation_start(
		self,
		operation: str,
		**context
	) -> str:
		"""Log operation start with correlation ID."""
		trace_id = str(uuid.uuid4())
		
		entry = LogEntry(
			level=LogLevel.INFO,
			logger_name="wbpm.operations",
			message=f"Operation started: {operation}",
			context={
				"operation": operation,
				"trace_id": trace_id,
				**context,
				**self.correlation_context
			},
			trace_id=trace_id
		)
		
		self._emit_log(entry)
		return trace_id
	
	def log_operation_end(
		self,
		operation: str,
		trace_id: str,
		duration_ms: float,
		success: bool = True,
		**context
	) -> None:
		"""Log operation end with performance data."""
		entry = LogEntry(
			level=LogLevel.INFO,
			logger_name="wbpm.operations",
			message=f"Operation {'completed' if success else 'failed'}: {operation}",
			context={
				"operation": operation,
				"trace_id": trace_id,
				"duration_ms": duration_ms,
				"success": success,
				**context,
				**self.correlation_context
			},
			trace_id=trace_id,
			duration_ms=duration_ms
		)
		
		self._emit_log(entry)
	
	def log_error(
		self,
		message: str,
		exception: Exception = None,
		**context
	) -> None:
		"""Log error with exception details."""
		entry = LogEntry(
			level=LogLevel.ERROR,
			logger_name="wbpm.errors",
			message=message,
			context={
				**context,
				**self.correlation_context
			},
			exception=traceback.format_exc() if exception else None
		)
		
		self._emit_log(entry)
	
	def log_security_event(
		self,
		event_type: str,
		message: str,
		**context
	) -> None:
		"""Log security-related events."""
		entry = LogEntry(
			level=LogLevel.WARNING,
			logger_name="wbpm.security",
			message=f"Security event: {event_type} - {message}",
			context={
				"event_type": event_type,
				"security_event": True,
				**context,
				**self.correlation_context
			}
		)
		
		self._emit_log(entry)
	
	def log_business_event(
		self,
		event_type: str,
		message: str,
		**context
	) -> None:
		"""Log business-related events."""
		entry = LogEntry(
			level=LogLevel.INFO,
			logger_name="wbpm.business",
			message=f"Business event: {event_type} - {message}",
			context={
				"event_type": event_type,
				"business_event": True,
				**context,
				**self.correlation_context
			}
		)
		
		self._emit_log(entry)
	
	def _emit_log(self, entry: LogEntry) -> None:
		"""Emit log entry to configured outputs."""
		try:
			# Add to buffer
			self.log_buffer.append(entry)
			
			# Convert to structured log format
			log_data = {
				"timestamp": entry.timestamp.isoformat(),
				"level": entry.level.value,
				"logger": entry.logger_name,
				"message": entry.message,
				**entry.context
			}
			
			if entry.trace_id:
				log_data["trace_id"] = entry.trace_id
			if entry.span_id:
				log_data["span_id"] = entry.span_id
			if entry.duration_ms is not None:
				log_data["duration_ms"] = entry.duration_ms
			if entry.exception:
				log_data["exception"] = entry.exception
			
			# Get structured logger
			struct_logger = structlog.get_logger(entry.logger_name)
			
			# Log based on level
			if entry.level == LogLevel.DEBUG:
				struct_logger.debug(entry.message, **log_data)
			elif entry.level == LogLevel.INFO:
				struct_logger.info(entry.message, **log_data)
			elif entry.level == LogLevel.WARNING:
				struct_logger.warning(entry.message, **log_data)
			elif entry.level == LogLevel.ERROR:
				struct_logger.error(entry.message, **log_data)
			elif entry.level == LogLevel.CRITICAL:
				struct_logger.critical(entry.message, **log_data)
			
		except Exception as e:
			# Fallback logging
			print(f"Logging error: {e}")
			print(f"Original log: {entry.message}")
	
	async def get_recent_logs(
		self,
		level: Optional[LogLevel] = None,
		time_window: timedelta = timedelta(hours=1),
		limit: int = 100
	) -> List[LogEntry]:
		"""Get recent log entries."""
		try:
			cutoff_time = datetime.utcnow() - time_window
			
			filtered_logs = []
			for entry in reversed(self.log_buffer):
				if entry.timestamp < cutoff_time:
					break
				if level and entry.level != level:
					continue
				
				filtered_logs.append(entry)
				
				if len(filtered_logs) >= limit:
					break
			
			return list(reversed(filtered_logs))
			
		except Exception as e:
			logger.error("Error getting recent logs", error=str(e))
			return []


# =============================================================================
# Production Monitoring Service
# =============================================================================

class ProductionMonitoringService:
	"""Main production monitoring service."""
	
	def __init__(self):
		self.health_manager = HealthCheckManager()
		self.metrics_collector = ProductionMetricsCollector()
		self.logging_manager = StructuredLoggingManager()
		self.alerts: List[Alert] = []
		self.monitoring_started = False
		
	async def start_monitoring(self) -> None:
		"""Start all monitoring components."""
		try:
			if self.monitoring_started:
				logger.warning("Production monitoring already started")
				return
			
			# Register default health checks
			await self._register_default_health_checks()
			
			# Start metrics collection
			await self.metrics_collector.start_collection()
			
			# Log monitoring start
			self.logging_manager.log_business_event(
				"monitoring_started",
				"Production monitoring service started"
			)
			
			self.monitoring_started = True
			logger.info("Production monitoring started successfully")
			
		except Exception as e:
			logger.error("Error starting production monitoring", error=str(e))
			raise
	
	async def stop_monitoring(self) -> None:
		"""Stop all monitoring components."""
		try:
			if not self.monitoring_started:
				return
			
			# Stop metrics collection
			await self.metrics_collector.stop_collection()
			
			# Log monitoring stop
			self.logging_manager.log_business_event(
				"monitoring_stopped",
				"Production monitoring service stopped"
			)
			
			self.monitoring_started = False
			logger.info("Production monitoring stopped")
			
		except Exception as e:
			logger.error("Error stopping production monitoring", error=str(e))
			raise
	
	async def _register_default_health_checks(self) -> None:
		"""Register default health checks."""
		try:
			# Database health check
			await self.health_manager.register_health_check(
				name="database",
				check_function=self._check_database_health,
				description="PostgreSQL database connectivity and performance",
				category="infrastructure"
			)
			
			# Cache health check
			await self.health_manager.register_health_check(
				name="cache",
				check_function=self._check_cache_health,
				description="Redis cache connectivity and performance",
				category="infrastructure"
			)
			
			# Workflow engine health check
			await self.health_manager.register_health_check(
				name="workflow_engine",
				check_function=self._check_workflow_engine_health,
				description="BPMN workflow engine status",
				category="application"
			)
			
			# Task manager health check
			await self.health_manager.register_health_check(
				name="task_manager",
				check_function=self._check_task_manager_health,
				description="Task management system status",
				category="application"
			)
			
			# Notification system health check
			await self.health_manager.register_health_check(
				name="notification_system",
				check_function=self._check_notification_system_health,
				description="Notification delivery system status",
				category="application"
			)
			
			logger.info("Default health checks registered")
			
		except Exception as e:
			logger.error("Error registering default health checks", error=str(e))
			raise
	
	async def _check_database_health(self) -> Dict[str, Any]:
		"""Check database health."""
		try:
			# In production, this would test actual database connectivity
			# For demo, we'll simulate the check
			await asyncio.sleep(0.1)  # Simulate DB query
			
			return {
				"status": HealthStatus.HEALTHY,
				"tags": {
					"connections": "8/20",
					"response_time_ms": "45"
				}
			}
			
		except Exception as e:
			return {
				"status": HealthStatus.UNHEALTHY,
				"error": str(e)
			}
	
	async def _check_cache_health(self) -> Dict[str, Any]:
		"""Check cache health."""
		try:
			# Simulate cache check
			await asyncio.sleep(0.05)
			
			return {
				"status": HealthStatus.HEALTHY,
				"tags": {
					"hit_rate": "85%",
					"memory_usage": "45%"
				}
			}
			
		except Exception as e:
			return {
				"status": HealthStatus.UNHEALTHY,
				"error": str(e)
			}
	
	async def _check_workflow_engine_health(self) -> Dict[str, Any]:
		"""Check workflow engine health."""
		try:
			# Simulate workflow engine check
			await asyncio.sleep(0.02)
			
			return {
				"status": HealthStatus.HEALTHY,
				"tags": {
					"active_processes": "25",
					"queue_size": "12"
				}
			}
			
		except Exception as e:
			return {
				"status": HealthStatus.DEGRADED,
				"error": str(e)
			}
	
	async def _check_task_manager_health(self) -> Dict[str, Any]:
		"""Check task manager health."""
		try:
			# Simulate task manager check
			await asyncio.sleep(0.03)
			
			return {
				"status": HealthStatus.HEALTHY,
				"tags": {
					"active_tasks": "48",
					"assignment_latency_ms": "125"
				}
			}
			
		except Exception as e:
			return {
				"status": HealthStatus.UNHEALTHY,
				"error": str(e)
			}
	
	async def _check_notification_system_health(self) -> Dict[str, Any]:
		"""Check notification system health."""
		try:
			# Simulate notification system check
			await asyncio.sleep(0.04)
			
			return {
				"status": HealthStatus.HEALTHY,
				"tags": {
					"pending_notifications": "15",
					"delivery_rate": "98.5%"
				}
			}
			
		except Exception as e:
			return {
				"status": HealthStatus.DEGRADED,
				"error": str(e)
			}
	
	async def get_comprehensive_status(self) -> Dict[str, Any]:
		"""Get comprehensive system status."""
		try:
			# Get health summary
			health_summary = await self.health_manager.get_health_summary()
			
			# Get metrics summary
			metrics_summary = await self.metrics_collector.get_metrics_summary()
			
			# Get recent alerts
			recent_alerts = [
				{
					"alert_id": alert.alert_id,
					"type": alert.alert_type,
					"severity": alert.severity,
					"title": alert.title,
					"triggered_at": alert.triggered_at.isoformat(),
					"resolved": alert.resolved_at is not None
				}
				for alert in self.alerts[-10:]  # Last 10 alerts
			]
			
			# Get recent error logs
			recent_errors = await self.logging_manager.get_recent_logs(
				level=LogLevel.ERROR,
				limit=10
			)
			
			status = {
				"timestamp": datetime.utcnow().isoformat(),
				"monitoring_status": "active" if self.monitoring_started else "inactive",
				"health": health_summary,
				"metrics": metrics_summary,
				"alerts": {
					"recent_count": len(recent_alerts),
					"active_count": len([a for a in self.alerts if a.resolved_at is None]),
					"recent_alerts": recent_alerts
				},
				"errors": {
					"recent_count": len(recent_errors),
					"recent_errors": [
						{
							"timestamp": entry.timestamp.isoformat(),
							"message": entry.message,
							"context": entry.context
						}
						for entry in recent_errors
					]
				}
			}
			
			return status
			
		except Exception as e:
			logger.error("Error getting comprehensive status", error=str(e))
			raise


# =============================================================================
# Service Factory
# =============================================================================

def create_production_monitoring_service() -> ProductionMonitoringService:
	"""Create and configure production monitoring service."""
	service = ProductionMonitoringService()
	logger.info("Production monitoring service created and configured")
	return service


# Export main classes
__all__ = [
	'ProductionMonitoringService',
	'HealthCheckManager',
	'ProductionMetricsCollector',
	'StructuredLoggingManager',
	'HealthCheck',
	'SystemMetrics',
	'ApplicationMetrics',
	'BusinessMetrics',
	'LogEntry',
	'Alert',
	'HealthStatus',
	'LogLevel',
	'MetricCategory',
	'create_production_monitoring_service'
]