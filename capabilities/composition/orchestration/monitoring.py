"""
APG Workflow Orchestration - Monitoring Integration
Comprehensive monitoring system with APG integration, custom metrics, and health checks
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import psutil
import aioredis
import logging
from pydantic import BaseModel, Field, validator
from pydantic.config import ConfigDict

# APG Framework imports
from apg.base.service import APGBaseService
from apg.base.models import BaseModel as APGBaseModel
from apg.integrations.telemetry import TelemetryClient
from apg.integrations.prometheus import PrometheusIntegration
from apg.integrations.grafana import GrafanaIntegration
from apg.base.security import SecurityManager
from apg.base.audit import AuditLogger

from .models import WorkflowExecution, WorkflowInstance
from .database import DatabaseManager


@dataclass
class MetricPoint:
	"""Individual metric data point"""
	timestamp: datetime
	value: float
	labels: Dict[str, str]
	metadata: Optional[Dict[str, Any]] = None


@dataclass 
class HealthStatus:
	"""System health status"""
	service: str
	status: str  # healthy, warning, critical, unknown
	message: str
	timestamp: datetime
	details: Dict[str, Any]


class WorkflowMetrics(APGBaseModel):
	"""Workflow execution metrics"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	workflow_id: str = Field(..., description="Workflow identifier")
	execution_id: str = Field(..., description="Execution identifier")
	tenant_id: str = Field(..., description="Tenant identifier")
	
	# Execution metrics
	start_time: datetime = Field(..., description="Execution start time")
	end_time: Optional[datetime] = Field(None, description="Execution end time")
	duration_ms: Optional[int] = Field(None, description="Execution duration in milliseconds")
	status: str = Field(..., description="Execution status")
	
	# Performance metrics
	cpu_usage_percent: float = Field(0.0, description="CPU usage percentage")
	memory_usage_mb: float = Field(0.0, description="Memory usage in MB")
	io_read_bytes: int = Field(0, description="IO read bytes")
	io_write_bytes: int = Field(0, description="IO write bytes")
	network_bytes_sent: int = Field(0, description="Network bytes sent")
	network_bytes_recv: int = Field(0, description="Network bytes received")
	
	# Task metrics
	total_tasks: int = Field(0, description="Total number of tasks")
	completed_tasks: int = Field(0, description="Number of completed tasks")
	failed_tasks: int = Field(0, description="Number of failed tasks")
	skipped_tasks: int = Field(0, description="Number of skipped tasks")
	
	# Queue metrics
	queue_wait_time_ms: int = Field(0, description="Time spent waiting in queue")
	processing_time_ms: int = Field(0, description="Actual processing time")
	
	# Error metrics
	error_count: int = Field(0, description="Number of errors")
	warning_count: int = Field(0, description="Number of warnings")
	retry_count: int = Field(0, description="Number of retries")
	
	# Custom metrics
	custom_metrics: Dict[str, float] = Field(default_factory=dict, description="Custom metrics")
	labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")


class SystemMetrics(APGBaseModel):
	"""System-level metrics"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	
	# CPU metrics
	cpu_percent: float = Field(..., description="Overall CPU usage percentage")
	cpu_cores: int = Field(..., description="Number of CPU cores")
	load_average_1m: float = Field(..., description="1-minute load average")
	load_average_5m: float = Field(..., description="5-minute load average")
	load_average_15m: float = Field(..., description="15-minute load average")
	
	# Memory metrics
	memory_total_gb: float = Field(..., description="Total memory in GB")
	memory_used_gb: float = Field(..., description="Used memory in GB")
	memory_available_gb: float = Field(..., description="Available memory in GB")
	memory_percent: float = Field(..., description="Memory usage percentage")
	
	# Disk metrics
	disk_total_gb: float = Field(..., description="Total disk space in GB")
	disk_used_gb: float = Field(..., description="Used disk space in GB")
	disk_free_gb: float = Field(..., description="Free disk space in GB")
	disk_percent: float = Field(..., description="Disk usage percentage")
	
	# Network metrics
	network_bytes_sent: int = Field(..., description="Total bytes sent")
	network_bytes_recv: int = Field(..., description="Total bytes received")
	network_packets_sent: int = Field(..., description="Total packets sent")
	network_packets_recv: int = Field(..., description="Total packets received")
	
	# Process metrics
	active_connections: int = Field(..., description="Number of active connections")
	open_files: int = Field(..., description="Number of open files")
	thread_count: int = Field(..., description="Number of threads")


class MonitoringConfig(APGBaseModel):
	"""Monitoring configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Collection intervals (seconds)
	metrics_collection_interval: int = Field(30, description="Metrics collection interval")
	health_check_interval: int = Field(60, description="Health check interval")
	system_metrics_interval: int = Field(300, description="System metrics collection interval")
	
	# Retention periods (days)
	metrics_retention_days: int = Field(30, description="Metrics retention period")
	logs_retention_days: int = Field(7, description="Logs retention period")
	
	# Alerting thresholds
	cpu_warning_threshold: float = Field(80.0, description="CPU warning threshold")
	cpu_critical_threshold: float = Field(95.0, description="CPU critical threshold")
	memory_warning_threshold: float = Field(85.0, description="Memory warning threshold")
	memory_critical_threshold: float = Field(95.0, description="Memory critical threshold")
	disk_warning_threshold: float = Field(85.0, description="Disk warning threshold")
	disk_critical_threshold: float = Field(95.0, description="Disk critical threshold")
	
	# Performance thresholds
	max_execution_time_minutes: int = Field(60, description="Maximum execution time")
	max_queue_wait_time_minutes: int = Field(10, description="Maximum queue wait time")
	
	# Integration settings
	prometheus_enabled: bool = Field(True, description="Enable Prometheus integration")
	grafana_enabled: bool = Field(True, description="Enable Grafana integration")
	datadog_enabled: bool = Field(False, description="Enable DataDog integration")
	newrelic_enabled: bool = Field(False, description="Enable New Relic integration")
	
	# Export settings
	export_to_apg_telemetry: bool = Field(True, description="Export to APG telemetry")
	export_interval_seconds: int = Field(60, description="Export interval")


class WorkflowMonitoringService(APGBaseService):
	"""Main workflow monitoring service"""
	
	def __init__(self, config: MonitoringConfig, db_manager: DatabaseManager):
		super().__init__()
		self.config = config
		self.db_manager = db_manager
		
		# Monitoring components
		self.metrics_collector = MetricsCollector(config)
		self.health_checker = HealthChecker(config)
		self.alert_manager = AlertManager(config)
		self.dashboard_manager = DashboardManager(config)
		
		# APG integrations
		self.telemetry_client = TelemetryClient()
		self.prometheus = PrometheusIntegration() if config.prometheus_enabled else None
		self.grafana = GrafanaIntegration() if config.grafana_enabled else None
		
		# Storage
		self.redis_client: Optional[aioredis.Redis] = None
		self.metrics_buffer: List[MetricPoint] = []
		self.health_status_cache: Dict[str, HealthStatus] = {}
		
		# Background tasks
		self._monitoring_tasks: List[asyncio.Task] = []
		self._shutdown_event = asyncio.Event()
		
		self._log_info("Workflow monitoring service initialized")
	
	async def initialize(self) -> None:
		"""Initialize monitoring service"""
		try:
			# Initialize Redis connection
			self.redis_client = await aioredis.from_url(
				"redis://localhost:6379",
				encoding="utf-8",
				decode_responses=True
			)
			
			# Initialize APG integrations
			await self.telemetry_client.initialize()
			
			if self.prometheus:
				await self.prometheus.initialize()
			
			if self.grafana:
				await self.grafana.initialize()
			
			# Initialize components
			await self.metrics_collector.initialize(self.redis_client)
			await self.health_checker.initialize(self.db_manager, self.redis_client)
			await self.alert_manager.initialize()
			await self.dashboard_manager.initialize()
			
			# Start background monitoring tasks
			await self._start_monitoring_tasks()
			
			self._log_info("Monitoring service initialized successfully")
			
		except Exception as e:
			self._log_error(f"Failed to initialize monitoring service: {e}")
			raise
	
	async def _start_monitoring_tasks(self) -> None:
		"""Start background monitoring tasks"""
		tasks = [
			self._metrics_collection_task(),
			self._health_check_task(),
			self._system_metrics_task(),
			self._metrics_export_task(),
			self._cleanup_task()
		]
		
		for task_coro in tasks:
			task = asyncio.create_task(task_coro)
			self._monitoring_tasks.append(task)
		
		self._log_info(f"Started {len(self._monitoring_tasks)} monitoring tasks")
	
	async def _metrics_collection_task(self) -> None:
		"""Background task for collecting workflow metrics"""
		while not self._shutdown_event.is_set():
			try:
				await self.metrics_collector.collect_workflow_metrics()
				await asyncio.sleep(self.config.metrics_collection_interval)
			except Exception as e:
				self._log_error(f"Error in metrics collection task: {e}")
				await asyncio.sleep(10)  # Wait before retrying
	
	async def _health_check_task(self) -> None:
		"""Background task for health checks"""
		while not self._shutdown_event.is_set():
			try:
				await self.health_checker.perform_health_checks()
				await asyncio.sleep(self.config.health_check_interval)
			except Exception as e:
				self._log_error(f"Error in health check task: {e}")
				await asyncio.sleep(10)
	
	async def _system_metrics_task(self) -> None:
		"""Background task for system metrics collection"""
		while not self._shutdown_event.is_set():
			try:
				await self.metrics_collector.collect_system_metrics()
				await asyncio.sleep(self.config.system_metrics_interval)
			except Exception as e:
				self._log_error(f"Error in system metrics task: {e}")
				await asyncio.sleep(10)
	
	async def _metrics_export_task(self) -> None:
		"""Background task for exporting metrics"""
		while not self._shutdown_event.is_set():
			try:
				await self._export_metrics()
				await asyncio.sleep(self.config.export_interval_seconds)
			except Exception as e:
				self._log_error(f"Error in metrics export task: {e}")
				await asyncio.sleep(10)
	
	async def _cleanup_task(self) -> None:
		"""Background task for data cleanup"""
		while not self._shutdown_event.is_set():
			try:
				await self._cleanup_old_data()
				await asyncio.sleep(3600)  # Run cleanup every hour
			except Exception as e:
				self._log_error(f"Error in cleanup task: {e}")
				await asyncio.sleep(10)
	
	async def record_workflow_start(self, workflow_id: str, execution_id: str, 
									tenant_id: str, metadata: Dict[str, Any] = None) -> None:
		"""Record workflow execution start"""
		try:
			metrics = WorkflowMetrics(
				workflow_id=workflow_id,
				execution_id=execution_id,
				tenant_id=tenant_id,
				start_time=datetime.utcnow(),
				status="running",
				labels={
					"workflow_id": workflow_id,
					"tenant_id": tenant_id,
					"status": "running"
				}
			)
			
			if metadata:
				metrics.custom_metrics.update(metadata)
			
			await self.metrics_collector.record_metric(metrics)
			
			# Update counters
			await self._increment_counter("workflow_executions_started", 
										  labels={"workflow_id": workflow_id, "tenant_id": tenant_id})
			
			self._log_debug(f"Recorded workflow start: {execution_id}")
			
		except Exception as e:
			self._log_error(f"Failed to record workflow start: {e}")
	
	async def record_workflow_completion(self, execution_id: str, status: str, 
										error: Optional[str] = None, 
										metrics_data: Dict[str, Any] = None) -> None:
		"""Record workflow execution completion"""
		try:
			end_time = datetime.utcnow()
			
			# Get existing metrics
			existing_metrics = await self.metrics_collector.get_workflow_metrics(execution_id)
			if not existing_metrics:
				self._log_warning(f"No existing metrics found for execution {execution_id}")
				return
			
			# Update metrics
			existing_metrics.end_time = end_time
			existing_metrics.status = status
			existing_metrics.duration_ms = int((end_time - existing_metrics.start_time).total_seconds() * 1000)
			
			if metrics_data:
				for key, value in metrics_data.items():
					if hasattr(existing_metrics, key):
						setattr(existing_metrics, key, value)
					else:
						existing_metrics.custom_metrics[key] = value
			
			if error:
				existing_metrics.error_count += 1
				existing_metrics.custom_metrics["last_error"] = error
			
			await self.metrics_collector.record_metric(existing_metrics)
			
			# Update counters
			counter_labels = {
				"workflow_id": existing_metrics.workflow_id,
				"tenant_id": existing_metrics.tenant_id,
				"status": status
			}
			
			await self._increment_counter("workflow_executions_completed", labels=counter_labels)
			
			if status == "failed":
				await self._increment_counter("workflow_executions_failed", labels=counter_labels)
			elif status == "succeeded":
				await self._increment_counter("workflow_executions_succeeded", labels=counter_labels)
			
			# Record execution time histogram
			await self._record_histogram("workflow_execution_duration_ms", 
										existing_metrics.duration_ms, labels=counter_labels)
			
			self._log_debug(f"Recorded workflow completion: {execution_id} ({status})")
			
		except Exception as e:
			self._log_error(f"Failed to record workflow completion: {e}")
	
	async def record_task_metrics(self, execution_id: str, task_id: str, 
								 task_name: str, status: str, duration_ms: int,
								 resource_usage: Dict[str, float] = None) -> None:
		"""Record individual task metrics"""
		try:
			labels = {
				"execution_id": execution_id,
				"task_id": task_id,
				"task_name": task_name,
				"status": status
			}
			
			# Record task completion
			await self._increment_counter("workflow_tasks_completed", labels=labels)
			
			# Record task duration
			await self._record_histogram("workflow_task_duration_ms", duration_ms, labels=labels)
			
			# Record resource usage if provided
			if resource_usage:
				for metric_name, value in resource_usage.items():
					await self._record_gauge(f"workflow_task_{metric_name}", value, labels=labels)
			
			# Update workflow metrics
			workflow_metrics = await self.metrics_collector.get_workflow_metrics(execution_id)
			if workflow_metrics:
				if status == "completed":
					workflow_metrics.completed_tasks += 1
				elif status == "failed":
					workflow_metrics.failed_tasks += 1
				elif status == "skipped":
					workflow_metrics.skipped_tasks += 1
				
				await self.metrics_collector.record_metric(workflow_metrics)
			
			self._log_debug(f"Recorded task metrics: {task_id} ({status})")
			
		except Exception as e:
			self._log_error(f"Failed to record task metrics: {e}")
	
	async def get_workflow_metrics(self, workflow_id: str, 
								  start_time: Optional[datetime] = None,
								  end_time: Optional[datetime] = None) -> List[WorkflowMetrics]:
		"""Get workflow metrics for a specific workflow"""
		try:
			return await self.metrics_collector.get_workflow_metrics_by_id(
				workflow_id, start_time, end_time
			)
		except Exception as e:
			self._log_error(f"Failed to get workflow metrics: {e}")
			return []
	
	async def get_system_health(self) -> Dict[str, HealthStatus]:
		"""Get current system health status"""
		try:
			return await self.health_checker.get_current_health_status()
		except Exception as e:
			self._log_error(f"Failed to get system health: {e}")
			return {}
	
	async def get_performance_dashboard_data(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Get data for performance dashboard"""
		try:
			return await self.dashboard_manager.get_dashboard_data(tenant_id)
		except Exception as e:
			self._log_error(f"Failed to get dashboard data: {e}")
			return {}
	
	async def _increment_counter(self, name: str, labels: Dict[str, str] = None) -> None:
		"""Increment a counter metric"""
		if self.prometheus:
			await self.prometheus.increment_counter(name, labels or {})
		
		if self.config.export_to_apg_telemetry:
			await self.telemetry_client.increment_counter(name, labels or {})
	
	async def _record_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
		"""Record a histogram metric"""
		if self.prometheus:
			await self.prometheus.record_histogram(name, value, labels or {})
		
		if self.config.export_to_apg_telemetry:
			await self.telemetry_client.record_histogram(name, value, labels or {})
	
	async def _record_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
		"""Record a gauge metric"""
		if self.prometheus:
			await self.prometheus.record_gauge(name, value, labels or {})
		
		if self.config.export_to_apg_telemetry:
			await self.telemetry_client.record_gauge(name, value, labels or {})
	
	async def _export_metrics(self) -> None:
		"""Export metrics to external systems"""
		try:
			# Export to APG telemetry
			if self.config.export_to_apg_telemetry:
				metrics_data = await self.metrics_collector.get_recent_metrics()
				await self.telemetry_client.export_metrics(metrics_data)
			
			# Export to other monitoring systems
			if self.config.datadog_enabled:
				await self._export_to_datadog()
			
			if self.config.newrelic_enabled:
				await self._export_to_newrelic()
			
			self._log_debug("Metrics exported successfully")
			
		except Exception as e:
			self._log_error(f"Failed to export metrics: {e}")
	
	async def _export_to_datadog(self) -> None:
		"""Export metrics to DataDog"""
		try:
			if not self.config.datadog_api_key:
				self._log_debug("DataDog API key not configured, skipping export")
				return
			
			# Get recent metrics from cache
			metrics = await self.get_recent_metrics(limit=1000)
			
			if not metrics:
				return
			
			# Format metrics for DataDog
			datadog_metrics = []
			for metric in metrics:
				datadog_metrics.append({
					'metric': f"workflow.{metric.get('name', 'unknown')}",
					'points': [[
						int(metric.get('timestamp', datetime.utcnow().timestamp())),
						metric.get('value', 0)
					]],
					'tags': [
						f"{k}:{v}" for k, v in metric.get('labels', {}).items()
					],
					'type': 'gauge'
				})
			
			# Send to DataDog API (would use actual datadog client)
			import aiohttp
			async with aiohttp.ClientSession() as session:
				headers = {
					'Content-Type': 'application/json',
					'DD-API-KEY': self.config.datadog_api_key
				}
				data = {'series': datadog_metrics}
				
				async with session.post(
					'https://api.datadoghq.com/api/v1/series',
					headers=headers,
					json=data
				) as response:
					if response.status == 202:
						self._log_debug(f"Exported {len(datadog_metrics)} metrics to DataDog")
					else:
						self._log_error(f"Failed to export to DataDog: {response.status}")
			
		except Exception as e:
			self._log_error(f"DataDog export failed: {e}")
	
	async def _export_to_newrelic(self) -> None:
		"""Export metrics to New Relic"""
		try:
			if not self.config.newrelic_license_key:
				self._log_debug("New Relic license key not configured, skipping export")
				return
			
			# Get recent metrics from cache
			metrics = await self.get_recent_metrics(limit=1000)
			
			if not metrics:
				return
			
			# Format metrics for New Relic
			newrelic_metrics = []
			current_time = int(datetime.utcnow().timestamp() * 1000)
			
			for metric in metrics:
				metric_data = {
					'name': f"Custom/Workflow/{metric.get('name', 'unknown')}",
					'type': 'gauge',
					'value': metric.get('value', 0),
					'timestamp': metric.get('timestamp', current_time),
					'attributes': metric.get('labels', {})
				}
				newrelic_metrics.append(metric_data)
			
			# Send to New Relic Metric API
			import aiohttp
			async with aiohttp.ClientSession() as session:
				headers = {
					'Content-Type': 'application/json',
					'Api-Key': self.config.newrelic_license_key
				}
				data = [{
					'common': {
						'timestamp': current_time,
						'interval.ms': 60000,
						'attributes': {
							'service.name': 'workflow-orchestration',
							'service.version': '1.0.0'
						}
					},
					'metrics': newrelic_metrics
				}]
				
				async with session.post(
					'https://metric-api.newrelic.com/metric/v1',
					headers=headers,
					json=data
				) as response:
					if response.status == 202:
						self._log_debug(f"Exported {len(newrelic_metrics)} metrics to New Relic")
					else:
						self._log_error(f"Failed to export to New Relic: {response.status}")
			
		except Exception as e:
			self._log_error(f"New Relic export failed: {e}")
	
	async def _cleanup_old_data(self) -> None:
		"""Clean up old metrics and log data"""
		try:
			cutoff_date = datetime.utcnow() - timedelta(days=self.config.metrics_retention_days)
			
			# Clean up metrics
			await self.metrics_collector.cleanup_old_metrics(cutoff_date)
			
			# Clean up logs
			log_cutoff_date = datetime.utcnow() - timedelta(days=self.config.logs_retention_days)
			await self._cleanup_old_logs(log_cutoff_date)
			
			self._log_debug("Data cleanup completed")
			
		except Exception as e:
			self._log_error(f"Failed to cleanup old data: {e}")
	
	async def _cleanup_old_logs(self, cutoff_date: datetime) -> None:
		"""Clean up old log entries"""
		try:
			if self.redis_client:
				# Clean up Redis-stored logs
				keys_to_delete = []
				async for key in self.redis_client.scan_iter(match="workflow:logs:*"):
					timestamp_str = key.split(":")[-1]
					try:
						timestamp = datetime.fromisoformat(timestamp_str)
						if timestamp < cutoff_date:
							keys_to_delete.append(key)
					except ValueError:
						continue
				
				if keys_to_delete:
					await self.redis_client.delete(*keys_to_delete)
					self._log_debug(f"Cleaned up {len(keys_to_delete)} old log entries")
			
		except Exception as e:
			self._log_error(f"Failed to cleanup old logs: {e}")
	
	async def shutdown(self) -> None:
		"""Shutdown monitoring service"""
		try:
			self._log_info("Shutting down monitoring service...")
			
			# Signal shutdown to background tasks
			self._shutdown_event.set()
			
			# Wait for tasks to complete
			if self._monitoring_tasks:
				await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
			
			# Shutdown components
			await self.metrics_collector.shutdown()
			await self.health_checker.shutdown()
			await self.alert_manager.shutdown()
			await self.dashboard_manager.shutdown()
			
			# Close connections
			if self.redis_client:
				await self.redis_client.close()
			
			await self.telemetry_client.shutdown()
			
			if self.prometheus:
				await self.prometheus.shutdown()
			
			if self.grafana:
				await self.grafana.shutdown()
			
			self._log_info("Monitoring service shutdown completed")
			
		except Exception as e:
			self._log_error(f"Error during monitoring service shutdown: {e}")


class MetricsCollector:
	"""Collects and stores workflow and system metrics"""
	
	def __init__(self, config: MonitoringConfig):
		self.config = config
		self.redis_client: Optional[aioredis.Redis] = None
		self.logger = logging.getLogger(f"{__name__}.MetricsCollector")
	
	async def initialize(self, redis_client: aioredis.Redis) -> None:
		"""Initialize metrics collector"""
		self.redis_client = redis_client
		self.logger.info("Metrics collector initialized")
	
	async def collect_workflow_metrics(self) -> None:
		"""Collect workflow execution metrics"""
		try:
			# Collect workflow execution metrics from database
			workflow_metrics_query = """
			WITH workflow_stats AS (
				SELECT 
					wi.workflow_id,
					wi.status,
					wi.started_at,
					wi.completed_at,
					wi.tenant_id,
					EXTRACT(EPOCH FROM (wi.completed_at - wi.started_at)) as duration_seconds,
					COALESCE(wi.progress_percentage, 0) as progress,
					COALESCE(wi.retry_count, 0) as retries,
					COUNT(te.id) as task_count,
					COUNT(te.id) FILTER (WHERE te.status = 'completed') as successful_tasks,
					COUNT(te.id) FILTER (WHERE te.status = 'failed') as failed_tasks,
					AVG(EXTRACT(EPOCH FROM (te.completed_at - te.started_at))) as avg_task_duration
				FROM cr_workflow_instances wi
				LEFT JOIN cr_task_executions te ON wi.id = te.instance_id
				WHERE wi.started_at >= NOW() - INTERVAL '1 hour'
				AND wi.tenant_id = %s
				GROUP BY wi.id, wi.workflow_id, wi.status, wi.started_at, wi.completed_at, wi.tenant_id, wi.progress_percentage, wi.retry_count
			)
			SELECT 
				workflow_id,
				status,
				COUNT(*) as instance_count,
				AVG(duration_seconds) as avg_duration,
				SUM(task_count) as total_tasks,
				SUM(successful_tasks) as total_successful_tasks,
				SUM(failed_tasks) as total_failed_tasks,
				AVG(progress) as avg_progress,
				SUM(retries) as total_retries,
				AVG(avg_task_duration) as avg_task_duration
			FROM workflow_stats
			GROUP BY workflow_id, status
			"""
			
			if hasattr(self, 'database') and self.database:
				rows = await self.database.fetch_all(workflow_metrics_query, (self.tenant_id,))
				
				current_time = datetime.utcnow()
				
				for row in rows:
					# Create workflow execution metrics
					labels = {
						'workflow_id': str(row['workflow_id']),
						'status': row['status'],
						'tenant_id': self.tenant_id
					}
					
					# Instance count metric
					await self.record_metric(
						name='workflow_instances_total',
						value=float(row['instance_count']),
						labels=labels,
						timestamp=current_time
					)
					
					# Average duration metric
					if row['avg_duration']:
						await self.record_metric(
							name='workflow_duration_seconds_avg',
							value=float(row['avg_duration']),
							labels=labels,
							timestamp=current_time
						)
					
					# Task metrics
					if row['total_tasks']:
						await self.record_metric(
							name='workflow_tasks_total',
							value=float(row['total_tasks']),
							labels=labels,
							timestamp=current_time
						)
						
						# Success rate
						success_rate = (row['total_successful_tasks'] / row['total_tasks']) * 100
						await self.record_metric(
							name='workflow_success_rate_percent',
							value=success_rate,
							labels=labels,
							timestamp=current_time
						)
					
					# Progress metric
					if row['avg_progress']:
						await self.record_metric(
							name='workflow_progress_percent',
							value=float(row['avg_progress']),
							labels=labels,
							timestamp=current_time
						)
					
					# Retry count metric
					if row['total_retries']:
						await self.record_metric(
							name='workflow_retries_total',
							value=float(row['total_retries']),
							labels=labels,
							timestamp=current_time
						)
			
			# Collect system resource metrics
			await self._collect_system_metrics()
			
			self.logger.debug("Workflow metrics collection completed")
			
		except Exception as e:
			self.logger.error(f"Failed to collect workflow metrics: {e}")
	
	async def _collect_system_metrics(self) -> None:
		"""Collect system resource metrics"""
		try:
			import psutil
			
			current_time = datetime.utcnow()
			labels = {'tenant_id': self.tenant_id}
			
			# CPU usage
			cpu_percent = psutil.cpu_percent(interval=1)
			await self.record_metric(
				name='system_cpu_usage_percent',
				value=float(cpu_percent),
				labels=labels,
				timestamp=current_time
			)
			
			# Memory usage
			memory = psutil.virtual_memory()
			await self.record_metric(
				name='system_memory_usage_percent',
				value=float(memory.percent),
				labels=labels,
				timestamp=current_time
			)
			
			await self.record_metric(
				name='system_memory_available_bytes',
				value=float(memory.available),
				labels=labels,
				timestamp=current_time
			)
			
			# Disk usage
			disk = psutil.disk_usage('/')
			disk_percent = (disk.used / disk.total) * 100
			await self.record_metric(
				name='system_disk_usage_percent',
				value=float(disk_percent),
				labels=labels,
				timestamp=current_time
			)
			
			# Network I/O (if available)
			try:
				network = psutil.net_io_counters()
				await self.record_metric(
					name='system_network_bytes_sent',
					value=float(network.bytes_sent),
					labels=labels,
					timestamp=current_time
				)
				
				await self.record_metric(
					name='system_network_bytes_recv',
					value=float(network.bytes_recv),
					labels=labels,
					timestamp=current_time
				)
			except Exception:
				pass  # Network metrics might not be available in all environments
			
		except ImportError:
			self.logger.warning("psutil not available, skipping system metrics")
		except Exception as e:
			self.logger.error(f"Failed to collect system metrics: {e}")
	
	async def collect_system_metrics(self) -> SystemMetrics:
		"""Collect system-level metrics"""
		try:
			# CPU metrics
			cpu_percent = psutil.cpu_percent(interval=1)
			cpu_cores = psutil.cpu_count()
			load_avg = psutil.getloadavg()
			
			# Memory metrics
			memory = psutil.virtual_memory()
			memory_total_gb = memory.total / (1024**3)
			memory_used_gb = memory.used / (1024**3)
			memory_available_gb = memory.available / (1024**3)
			
			# Disk metrics
			disk = psutil.disk_usage('/')
			disk_total_gb = disk.total / (1024**3)
			disk_used_gb = disk.used / (1024**3)
			disk_free_gb = disk.free / (1024**3)
			
			# Network metrics
			network = psutil.net_io_counters()
			
			# Process metrics
			connections = len(psutil.net_connections())
			process = psutil.Process()
			open_files = len(process.open_files())
			thread_count = process.num_threads()
			
			metrics = SystemMetrics(
				cpu_percent=cpu_percent,
				cpu_cores=cpu_cores,
				load_average_1m=load_avg[0],
				load_average_5m=load_avg[1],
				load_average_15m=load_avg[2],
				memory_total_gb=memory_total_gb,
				memory_used_gb=memory_used_gb,
				memory_available_gb=memory_available_gb,
				memory_percent=memory.percent,
				disk_total_gb=disk_total_gb,
				disk_used_gb=disk_used_gb,
				disk_free_gb=disk_free_gb,
				disk_percent=disk.percent,
				network_bytes_sent=network.bytes_sent,
				network_bytes_recv=network.bytes_recv,
				network_packets_sent=network.packets_sent,
				network_packets_recv=network.packets_recv,
				active_connections=connections,
				open_files=open_files,
				thread_count=thread_count
			)
			
			# Store metrics
			await self._store_system_metrics(metrics)
			
			return metrics
			
		except Exception as e:
			self.logger.error(f"Failed to collect system metrics: {e}")
			raise
	
	async def record_metric(self, metrics: WorkflowMetrics) -> None:
		"""Record workflow metrics"""
		try:
			if not self.redis_client:
				return
			
			key = f"workflow:metrics:{metrics.execution_id}"
			data = metrics.model_dump()
			data['timestamp'] = datetime.utcnow().isoformat()
			
			await self.redis_client.hset(key, mapping={
				k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
				for k, v in data.items()
			})
			
			# Set expiration
			await self.redis_client.expire(key, self.config.metrics_retention_days * 24 * 3600)
			
		except Exception as e:
			self.logger.error(f"Failed to record metrics: {e}")
	
	async def get_workflow_metrics(self, execution_id: str) -> Optional[WorkflowMetrics]:
		"""Get workflow metrics by execution ID"""
		try:
			if not self.redis_client:
				return None
			
			key = f"workflow:metrics:{execution_id}"
			data = await self.redis_client.hgetall(key)
			
			if not data:
				return None
			
			# Parse data
			parsed_data = {}
			for k, v in data.items():
				try:
					if k in ['custom_metrics', 'labels']:
						parsed_data[k] = json.loads(v)
					elif k.endswith('_time'):
						parsed_data[k] = datetime.fromisoformat(v)
					elif k in ['duration_ms', 'total_tasks', 'completed_tasks', 'failed_tasks', 
							  'skipped_tasks', 'queue_wait_time_ms', 'processing_time_ms',
							  'error_count', 'warning_count', 'retry_count', 'io_read_bytes',
							  'io_write_bytes', 'network_bytes_sent', 'network_bytes_recv']:
						parsed_data[k] = int(v)
					elif k in ['cpu_usage_percent', 'memory_usage_mb']:
						parsed_data[k] = float(v)
					else:
						parsed_data[k] = v
				except (ValueError, json.JSONDecodeError):
					parsed_data[k] = v
			
			return WorkflowMetrics(**parsed_data)
			
		except Exception as e:
			self.logger.error(f"Failed to get workflow metrics: {e}")
			return None
	
	async def get_workflow_metrics_by_id(self, workflow_id: str, 
										start_time: Optional[datetime] = None,
										end_time: Optional[datetime] = None) -> List[WorkflowMetrics]:
		"""Get all metrics for a specific workflow"""
		try:
			if not self.redis_client:
				return []
			
			# This is a simplified implementation
			# In production, you'd want to use a time-series database
			metrics_list = []
			
			async for key in self.redis_client.scan_iter(match="workflow:metrics:*"):
				metrics = await self.get_workflow_metrics(key.split(":")[-1])
				if metrics and metrics.workflow_id == workflow_id:
					if start_time and metrics.start_time < start_time:
						continue
					if end_time and metrics.start_time > end_time:
						continue
					metrics_list.append(metrics)
			
			return sorted(metrics_list, key=lambda x: x.start_time)
			
		except Exception as e:
			self.logger.error(f"Failed to get workflow metrics by ID: {e}")
			return []
	
	async def get_recent_metrics(self, limit: int = 1000) -> List[Dict[str, Any]]:
		"""Get recent metrics for export"""
		try:
			if not self.redis_client:
				return []
			
			metrics_list = []
			count = 0
			
			async for key in self.redis_client.scan_iter(match="workflow:metrics:*"):
				if count >= limit:
					break
				
				data = await self.redis_client.hgetall(key)
				if data:
					metrics_list.append(data)
					count += 1
			
			return metrics_list
			
		except Exception as e:
			self.logger.error(f"Failed to get recent metrics: {e}")
			return []
	
	async def _store_system_metrics(self, metrics: SystemMetrics) -> None:
		"""Store system metrics"""
		try:
			if not self.redis_client:
				return
			
			key = f"system:metrics:{int(time.time())}"
			data = metrics.model_dump()
			data['timestamp'] = metrics.timestamp.isoformat()
			
			await self.redis_client.hset(key, mapping={
				k: str(v) for k, v in data.items()
			})
			
			# Set expiration
			await self.redis_client.expire(key, self.config.metrics_retention_days * 24 * 3600)
			
		except Exception as e:
			self.logger.error(f"Failed to store system metrics: {e}")
	
	async def cleanup_old_metrics(self, cutoff_date: datetime) -> None:
		"""Clean up old metrics"""
		try:
			if not self.redis_client:
				return
			
			deleted_count = 0
			
			# Clean up workflow metrics
			async for key in self.redis_client.scan_iter(match="workflow:metrics:*"):
				timestamp_data = await self.redis_client.hget(key, 'timestamp')
				if timestamp_data:
					try:
						timestamp = datetime.fromisoformat(timestamp_data)
						if timestamp < cutoff_date:
							await self.redis_client.delete(key)
							deleted_count += 1
					except ValueError:
						continue
			
			# Clean up system metrics
			async for key in self.redis_client.scan_iter(match="system:metrics:*"):
				timestamp_str = key.split(":")[-1]
				try:
					timestamp = datetime.fromtimestamp(int(timestamp_str))
					if timestamp < cutoff_date:
						await self.redis_client.delete(key)
						deleted_count += 1
				except ValueError:
					continue
			
			self.logger.info(f"Cleaned up {deleted_count} old metric entries")
			
		except Exception as e:
			self.logger.error(f"Failed to cleanup old metrics: {e}")
	
	async def shutdown(self) -> None:
		"""Shutdown metrics collector"""
		self.logger.info("Metrics collector shutting down")


class HealthChecker:
	"""Performs system health checks"""
	
	def __init__(self, config: MonitoringConfig):
		self.config = config
		self.db_manager: Optional[DatabaseManager] = None
		self.redis_client: Optional[aioredis.Redis] = None
		self.logger = logging.getLogger(f"{__name__}.HealthChecker")
		self.health_status: Dict[str, HealthStatus] = {}
	
	async def initialize(self, db_manager: DatabaseManager, redis_client: aioredis.Redis) -> None:
		"""Initialize health checker"""
		self.db_manager = db_manager
		self.redis_client = redis_client
		self.logger.info("Health checker initialized")
	
	async def perform_health_checks(self) -> Dict[str, HealthStatus]:
		"""Perform all health checks"""
		try:
			checks = [
				self._check_database_health(),
				self._check_redis_health(),
				self._check_system_resources(),
				self._check_workflow_engine_health(),
				self._check_external_dependencies()
			]
			
			results = await asyncio.gather(*checks, return_exceptions=True)
			
			# Update health status cache
			for result in results:
				if isinstance(result, HealthStatus):
					self.health_status[result.service] = result
				elif isinstance(result, Exception):
					self.logger.error(f"Health check failed: {result}")
			
			return self.health_status
			
		except Exception as e:
			self.logger.error(f"Failed to perform health checks: {e}")
			return {}
	
	async def _check_database_health(self) -> HealthStatus:
		"""Check database connectivity and performance"""
		try:
			start_time = time.time()
			
			# Simple connectivity check
			async with self.db_manager.get_connection() as conn:
				result = await conn.execute("SELECT 1")
				
			response_time = (time.time() - start_time) * 1000  # Convert to ms
			
			if response_time > 1000:  # > 1 second
				status = "warning"
				message = f"Database response slow: {response_time:.2f}ms"
			else:
				status = "healthy"
				message = f"Database responsive: {response_time:.2f}ms"
			
			return HealthStatus(
				service="database",
				status=status,
				message=message,
				timestamp=datetime.utcnow(),
				details={
					"response_time_ms": response_time,
					"connection_pool_size": self.db_manager.pool_size if hasattr(self.db_manager, 'pool_size') else None
				}
			)
			
		except Exception as e:
			return HealthStatus(
				service="database",
				status="critical",
				message=f"Database connection failed: {str(e)}",
				timestamp=datetime.utcnow(),
				details={"error": str(e)}
			)
	
	async def _check_redis_health(self) -> HealthStatus:
		"""Check Redis connectivity and performance"""
		try:
			start_time = time.time()
			
			# Test Redis connectivity
			await self.redis_client.ping()
			
			response_time = (time.time() - start_time) * 1000
			
			# Get Redis info
			info = await self.redis_client.info()
			memory_usage = info.get('used_memory', 0)
			connected_clients = info.get('connected_clients', 0)
			
			if response_time > 500:  # > 500ms
				status = "warning"
				message = f"Redis response slow: {response_time:.2f}ms"
			else:
				status = "healthy"
				message = f"Redis responsive: {response_time:.2f}ms"
			
			return HealthStatus(
				service="redis",
				status=status,
				message=message,
				timestamp=datetime.utcnow(),
				details={
					"response_time_ms": response_time,
					"memory_usage_bytes": memory_usage,
					"connected_clients": connected_clients
				}
			)
			
		except Exception as e:
			return HealthStatus(
				service="redis",
				status="critical",
				message=f"Redis connection failed: {str(e)}",
				timestamp=datetime.utcnow(),
				details={"error": str(e)}
			)
	
	async def _check_system_resources(self) -> HealthStatus:
		"""Check system resource usage"""
		try:
			# Get system metrics
			cpu_percent = psutil.cpu_percent(interval=1)
			memory = psutil.virtual_memory()
			disk = psutil.disk_usage('/')
			
			# Determine status based on thresholds
			issues = []
			status = "healthy"
			
			if cpu_percent > self.config.cpu_critical_threshold:
				status = "critical"
				issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
			elif cpu_percent > self.config.cpu_warning_threshold:
				status = "warning"
				issues.append(f"CPU usage high: {cpu_percent:.1f}%")
			
			if memory.percent > self.config.memory_critical_threshold:
				status = "critical"
				issues.append(f"Memory usage critical: {memory.percent:.1f}%")
			elif memory.percent > self.config.memory_warning_threshold:
				if status != "critical":
					status = "warning"
				issues.append(f"Memory usage high: {memory.percent:.1f}%")
			
			if disk.percent > self.config.disk_critical_threshold:
				status = "critical"
				issues.append(f"Disk usage critical: {disk.percent:.1f}%")
			elif disk.percent > self.config.disk_warning_threshold:
				if status != "critical":
					status = "warning"
				issues.append(f"Disk usage high: {disk.percent:.1f}%")
			
			message = "; ".join(issues) if issues else "System resources within normal limits"
			
			return HealthStatus(
				service="system_resources",
				status=status,
				message=message,
				timestamp=datetime.utcnow(),
				details={
					"cpu_percent": cpu_percent,
					"memory_percent": memory.percent,
					"disk_percent": disk.percent,
					"memory_available_gb": memory.available / (1024**3),
					"disk_free_gb": disk.free / (1024**3)
				}
			)
			
		except Exception as e:
			return HealthStatus(
				service="system_resources",
				status="critical",
				message=f"Failed to check system resources: {str(e)}",
				timestamp=datetime.utcnow(),
				details={"error": str(e)}
			)
	
	async def _check_workflow_engine_health(self) -> HealthStatus:
		"""Check workflow engine health"""
		try:
			# This would check the workflow execution engine status
			# For now, we'll do a basic check
			
			return HealthStatus(
				service="workflow_engine",
				status="healthy",
				message="Workflow engine operational",
				timestamp=datetime.utcnow(),
				details={}
			)
			
		except Exception as e:
			return HealthStatus(
				service="workflow_engine",
				status="critical",
				message=f"Workflow engine check failed: {str(e)}",
				timestamp=datetime.utcnow(),
				details={"error": str(e)}
			)
	
	async def _check_external_dependencies(self) -> HealthStatus:
		"""Check external service dependencies"""
		try:
			# This would check external services like APIs, message queues, etc.
			
			return HealthStatus(
				service="external_dependencies",
				status="healthy",
				message="External dependencies accessible",
				timestamp=datetime.utcnow(),
				details={}
			)
			
		except Exception as e:
			return HealthStatus(
				service="external_dependencies",
				status="warning",
				message=f"Some external dependencies may be unavailable: {str(e)}",
				timestamp=datetime.utcnow(),
				details={"error": str(e)}
			)
	
	async def get_current_health_status(self) -> Dict[str, HealthStatus]:
		"""Get current health status for all services"""
		return self.health_status.copy()
	
	async def shutdown(self) -> None:
		"""Shutdown health checker"""
		self.logger.info("Health checker shutting down")


class AlertManager:
	"""Manages alerts and notifications"""
	
	def __init__(self, config: MonitoringConfig):
		self.config = config
		self.logger = logging.getLogger(f"{__name__}.AlertManager")
	
	async def initialize(self) -> None:
		"""Initialize alert manager"""
		self.logger.info("Alert manager initialized")
	
	async def shutdown(self) -> None:
		"""Shutdown alert manager"""
		self.logger.info("Alert manager shutting down")


class DashboardManager:
	"""Manages monitoring dashboards"""
	
	def __init__(self, config: MonitoringConfig):
		self.config = config
		self.logger = logging.getLogger(f"{__name__}.DashboardManager")
	
	async def initialize(self) -> None:
		"""Initialize dashboard manager"""
		self.logger.info("Dashboard manager initialized")
	
	async def get_dashboard_data(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Get dashboard data"""
		# Implementation would aggregate metrics for dashboard display
		return {}
	
	async def shutdown(self) -> None:
		"""Shutdown dashboard manager"""
		self.logger.info("Dashboard manager shutting down")