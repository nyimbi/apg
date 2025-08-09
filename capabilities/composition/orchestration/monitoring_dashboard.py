#!/usr/bin/env python3
"""
APG Workflow Orchestration Monitoring & Analytics Dashboard

Real-time monitoring, performance charts, alerting, and visualizations for workflow orchestration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModel, ConfigDict, Field, validator
import statistics
import redis
from collections import defaultdict, deque

from apg.framework.base_service import APGBaseService
from apg.framework.database import APGDatabase
from apg.framework.audit_compliance import APGAuditLogger

from .config import get_config
from .models import WorkflowStatus, TaskStatus, WorkflowExecution


logger = logging.getLogger(__name__)


class MetricType(str, Enum):
	"""Types of metrics collected."""
	COUNTER = "counter"
	GAUGE = "gauge"
	HISTOGRAM = "histogram"
	TIMER = "timer"


class AlertLevel(str, Enum):
	"""Alert severity levels."""
	INFO = "info"
	WARNING = "warning"
	ERROR = "error"
	CRITICAL = "critical"


class ChartType(str, Enum):
	"""Chart visualization types."""
	LINE = "line"
	BAR = "bar"
	PIE = "pie"
	AREA = "area"
	SCATTER = "scatter"
	HEATMAP = "heatmap"
	GAUGE = "gauge"


@dataclass
class MetricValue:
	"""Individual metric value."""
	timestamp: datetime
	value: float
	labels: Dict[str, str] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
	"""Metric definition and storage."""
	name: str
	metric_type: MetricType
	description: str
	unit: str
	values: deque = field(default_factory=lambda: deque(maxlen=1000))
	labels: Dict[str, str] = field(default_factory=dict)
	retention_hours: int = 24
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Alert:
	"""Alert definition."""
	id: str
	name: str
	description: str
	metric_name: str
	condition: str
	threshold: float
	level: AlertLevel
	enabled: bool = True
	last_triggered: Optional[datetime] = None
	trigger_count: int = 0
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Dashboard:
	"""Dashboard configuration."""
	id: str
	name: str
	description: str
	charts: List[Dict[str, Any]] = field(default_factory=list)
	refresh_interval: int = 30  # seconds
	auto_refresh: bool = True
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


class MetricsCollector:
	"""Collects and stores workflow metrics."""
	
	def __init__(self):
		self.metrics: Dict[str, Metric] = {}
		self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
		self.collection_interval = 10  # seconds
		self._running = False
		self._collection_task: Optional[asyncio.Task] = None
	
	async def start(self):
		"""Start metrics collection."""
		self._running = True
		self._collection_task = asyncio.create_task(self._collection_loop())
		logger.info("Metrics collector started")
	
	async def stop(self):
		"""Stop metrics collection."""
		self._running = False
		if self._collection_task:
			self._collection_task.cancel()
			try:
				await self._collection_task
			except asyncio.CancelledError:
				pass
		logger.info("Metrics collector stopped")
	
	async def _collection_loop(self):
		"""Main collection loop."""
		while self._running:
			try:
				await self._collect_system_metrics()
				await self._collect_workflow_metrics()
				await self._collect_performance_metrics()
				await asyncio.sleep(self.collection_interval)
			except Exception as e:
				logger.error(f"Error in metrics collection: {e}")
				await asyncio.sleep(self.collection_interval)
	
	async def _collect_system_metrics(self):
		"""Collect system-level metrics."""
		import psutil
		
		# CPU metrics
		cpu_percent = psutil.cpu_percent(interval=1)
		await self.record_metric("system_cpu_percent", cpu_percent, MetricType.GAUGE, "percent")
		
		# Memory metrics
		memory = psutil.virtual_memory()
		await self.record_metric("system_memory_percent", memory.percent, MetricType.GAUGE, "percent")
		await self.record_metric("system_memory_used", memory.used / (1024**3), MetricType.GAUGE, "GB")
		
		# Disk metrics
		disk = psutil.disk_usage('/')
		await self.record_metric("system_disk_percent", disk.percent, MetricType.GAUGE, "percent")
		await self.record_metric("system_disk_used", disk.used / (1024**3), MetricType.GAUGE, "GB")
		
		# Network metrics
		network = psutil.net_io_counters()
		await self.record_metric("system_network_bytes_sent", network.bytes_sent, MetricType.COUNTER, "bytes")
		await self.record_metric("system_network_bytes_recv", network.bytes_recv, MetricType.COUNTER, "bytes")
	
	async def _collect_workflow_metrics(self):
		"""Collect workflow-specific metrics."""
		try:
			# Get workflow statistics from Redis
			workflow_stats = await self._get_workflow_stats()
			
			# Active workflows
			active_count = workflow_stats.get('active_workflows', 0)
			await self.record_metric("workflows_active", active_count, MetricType.GAUGE, "count")
			
			# Completed workflows (last hour)
			completed_count = workflow_stats.get('completed_last_hour', 0)
			await self.record_metric("workflows_completed_hour", completed_count, MetricType.GAUGE, "count")
			
			# Failed workflows (last hour)
			failed_count = workflow_stats.get('failed_last_hour', 0)
			await self.record_metric("workflows_failed_hour", failed_count, MetricType.GAUGE, "count")
			
			# Success rate
			total_recent = completed_count + failed_count
			success_rate = (completed_count / total_recent * 100) if total_recent > 0 else 100
			await self.record_metric("workflows_success_rate", success_rate, MetricType.GAUGE, "percent")
			
			# Average execution time
			avg_duration = workflow_stats.get('avg_execution_time', 0)
			await self.record_metric("workflows_avg_duration", avg_duration, MetricType.GAUGE, "seconds")
			
		except Exception as e:
			logger.error(f"Error collecting workflow metrics: {e}")
	
	async def _collect_performance_metrics(self):
		"""Collect performance metrics."""
		try:
			# Queue depths
			queue_stats = await self._get_queue_stats()
			await self.record_metric("queue_pending_tasks", queue_stats.get('pending', 0), MetricType.GAUGE, "count")
			await self.record_metric("queue_processing_tasks", queue_stats.get('processing', 0), MetricType.GAUGE, "count")
			
			# Database connection pool
			db_stats = await self._get_database_stats()
			await self.record_metric("db_connections_active", db_stats.get('active_connections', 0), MetricType.GAUGE, "count")
			await self.record_metric("db_connections_pool_size", db_stats.get('pool_size', 0), MetricType.GAUGE, "count")
			
			# API response times
			api_stats = await self._get_api_stats()
			await self.record_metric("api_response_time_avg", api_stats.get('avg_response_time', 0), MetricType.GAUGE, "ms")
			await self.record_metric("api_requests_per_minute", api_stats.get('requests_per_minute', 0), MetricType.GAUGE, "count")
			
		except Exception as e:
			logger.error(f"Error collecting performance metrics: {e}")
	
	async def record_metric(self, name: str, value: float, metric_type: MetricType, 
						   unit: str, labels: Dict[str, str] = None):
		"""Record a metric value."""
		if name not in self.metrics:
			self.metrics[name] = Metric(
				name=name,
				metric_type=metric_type,
				description=f"{name} metric",
				unit=unit
			)
		
		metric = self.metrics[name]
		metric_value = MetricValue(
			timestamp=datetime.utcnow(),
			value=value,
			labels=labels or {}
		)
		
		metric.values.append(metric_value)
		
		# Store in Redis for real-time access
		redis_key = f"metric:{name}"
		metric_data = {
			'timestamp': metric_value.timestamp.isoformat(),
			'value': value,
			'labels': labels or {}
		}
		
		# Store latest value
		await asyncio.get_event_loop().run_in_executor(
			None, self.redis_client.hset, redis_key, 'latest', json.dumps(metric_data)
		)
		
		# Store in time series (keep last 1000 points)
		await asyncio.get_event_loop().run_in_executor(
			None, self.redis_client.lpush, f"{redis_key}:series", json.dumps(metric_data)
		)
		await asyncio.get_event_loop().run_in_executor(
			None, self.redis_client.ltrim, f"{redis_key}:series", 0, 999
		)
	
	async def get_metric_values(self, name: str, start_time: datetime = None, 
							   end_time: datetime = None, limit: int = 100) -> List[MetricValue]:
		"""Get metric values within time range."""
		if name not in self.metrics:
			return []
		
		metric = self.metrics[name]
		values = list(metric.values)
		
		# Filter by time range
		if start_time:
			values = [v for v in values if v.timestamp >= start_time]
		if end_time:
			values = [v for v in values if v.timestamp <= end_time]
		
		# Apply limit
		values = values[-limit:] if limit else values
		
		return values
	
	async def get_latest_value(self, name: str) -> Optional[float]:
		"""Get latest metric value."""
		if name not in self.metrics or not self.metrics[name].values:
			return None
		return self.metrics[name].values[-1].value
	
	async def _get_workflow_stats(self) -> Dict[str, Any]:
		"""Get workflow statistics from database/cache."""
		# In a real implementation, this would query the database
		# For now, return simulated data
		import random
		return {
			'active_workflows': random.randint(5, 25),
			'completed_last_hour': random.randint(10, 50),
			'failed_last_hour': random.randint(0, 5),
			'avg_execution_time': random.uniform(30, 300)
		}
	
	async def _get_queue_stats(self) -> Dict[str, Any]:
		"""Get queue statistics."""
		import random
		return {
			'pending': random.randint(0, 20),
			'processing': random.randint(2, 10)
		}
	
	async def _get_database_stats(self) -> Dict[str, Any]:
		"""Get database statistics."""
		import random
		pool_size = 20
		active = random.randint(5, pool_size)
		return {
			'active_connections': active,
			'pool_size': pool_size,
			'utilization': (active / pool_size) * 100
		}
	
	async def _get_api_stats(self) -> Dict[str, Any]:
		"""Get API statistics."""
		import random
		return {
			'avg_response_time': random.uniform(50, 200),
			'requests_per_minute': random.randint(10, 100),
			'error_rate': random.uniform(0, 5)
		}


class AlertManager:
	"""Manages workflow monitoring alerts."""
	
	def __init__(self, metrics_collector: MetricsCollector):
		self.metrics_collector = metrics_collector
		self.alerts: Dict[str, Alert] = {}
		self.notification_handlers: List[Any] = []
		self._evaluation_interval = 30  # seconds
		self._running = False
		self._evaluation_task: Optional[asyncio.Task] = None
		
		# Initialize default alerts
		self._create_default_alerts()
	
	def _create_default_alerts(self):
		"""Create default system alerts."""
		default_alerts = [
			Alert(
				id="high_cpu_usage",
				name="High CPU Usage",
				description="CPU usage is above 90%",
				metric_name="system_cpu_percent",
				condition=">",
				threshold=90.0,
				level=AlertLevel.WARNING
			),
			Alert(
				id="high_memory_usage",
				name="High Memory Usage",
				description="Memory usage is above 85%",
				metric_name="system_memory_percent",
				condition=">",
				threshold=85.0,
				level=AlertLevel.WARNING
			),
			Alert(
				id="low_success_rate",
				name="Low Workflow Success Rate",
				description="Workflow success rate is below 95%",
				metric_name="workflows_success_rate",
				condition="<",
				threshold=95.0,
				level=AlertLevel.ERROR
			),
			Alert(
				id="high_failure_rate",
				name="High Workflow Failure Rate",
				description="More than 10 workflows failed in the last hour",
				metric_name="workflows_failed_hour",
				condition=">",
				threshold=10.0,
				level=AlertLevel.CRITICAL
			),
			Alert(
				id="slow_workflows",
				name="Slow Workflow Execution",
				description="Average workflow execution time is above 5 minutes",
				metric_name="workflows_avg_duration",
				condition=">",
				threshold=300.0,
				level=AlertLevel.WARNING
			),
			Alert(
				id="queue_backlog",
				name="Task Queue Backlog",
				description="More than 50 pending tasks in queue",
				metric_name="queue_pending_tasks",
				condition=">",
				threshold=50.0,
				level=AlertLevel.WARNING
			)
		]
		
		for alert in default_alerts:
			self.alerts[alert.id] = alert
	
	async def start(self):
		"""Start alert evaluation."""
		self._running = True
		self._evaluation_task = asyncio.create_task(self._evaluation_loop())
		logger.info("Alert manager started")
	
	async def stop(self):
		"""Stop alert evaluation."""
		self._running = False
		if self._evaluation_task:
			self._evaluation_task.cancel()
			try:
				await self._evaluation_task
			except asyncio.CancelledError:
				pass
		logger.info("Alert manager stopped")
	
	async def _evaluation_loop(self):
		"""Main alert evaluation loop."""
		while self._running:
			try:
				await self._evaluate_alerts()
				await asyncio.sleep(self._evaluation_interval)
			except Exception as e:
				logger.error(f"Error in alert evaluation: {e}")
				await asyncio.sleep(self._evaluation_interval)
	
	async def _evaluate_alerts(self):
		"""Evaluate all active alerts."""
		for alert in self.alerts.values():
			if not alert.enabled:
				continue
			
			try:
				current_value = await self.metrics_collector.get_latest_value(alert.metric_name)
				if current_value is None:
					continue
				
				triggered = self._evaluate_condition(current_value, alert.condition, alert.threshold)
				
				if triggered:
					await self._trigger_alert(alert, current_value)
			
			except Exception as e:
				logger.error(f"Error evaluating alert {alert.id}: {e}")
	
	def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
		"""Evaluate alert condition."""
		if condition == ">":
			return value > threshold
		elif condition == "<":
			return value < threshold
		elif condition == ">=":
			return value >= threshold
		elif condition == "<=":
			return value <= threshold
		elif condition == "==":
			return value == threshold
		elif condition == "!=":
			return value != threshold
		else:
			return False
	
	async def _trigger_alert(self, alert: Alert, current_value: float):
		"""Trigger an alert."""
		now = datetime.utcnow()
		
		# Check if alert was recently triggered (avoid spam)
		if alert.last_triggered:
			time_since_last = (now - alert.last_triggered).total_seconds()
			if time_since_last < 300:  # 5 minutes cooldown
				return
		
		alert.last_triggered = now
		alert.trigger_count += 1
		
		alert_data = {
			'alert_id': alert.id,
			'alert_name': alert.name,
			'description': alert.description,
			'level': alert.level.value,
			'metric_name': alert.metric_name,
			'current_value': current_value,
			'threshold': alert.threshold,
			'condition': alert.condition,
			'triggered_at': now.isoformat(),
			'trigger_count': alert.trigger_count
		}
		
		# Send notifications
		for handler in self.notification_handlers:
			try:
				await handler.send_alert(alert_data)
			except Exception as e:
				logger.error(f"Error sending alert notification: {e}")
		
		logger.warning(f"Alert triggered: {alert.name} - {alert.description} (value: {current_value})")
	
	def add_alert(self, alert: Alert):
		"""Add a new alert."""
		self.alerts[alert.id] = alert
	
	def remove_alert(self, alert_id: str):
		"""Remove an alert."""
		if alert_id in self.alerts:
			del self.alerts[alert_id]
	
	def get_alert(self, alert_id: str) -> Optional[Alert]:
		"""Get alert by ID."""
		return self.alerts.get(alert_id)
	
	def list_alerts(self) -> List[Alert]:
		"""Get all alerts."""
		return list(self.alerts.values())


class ChartGenerator:
	"""Generates chart configurations for the dashboard."""
	
	def __init__(self, metrics_collector: MetricsCollector):
		self.metrics_collector = metrics_collector
	
	async def generate_chart_data(self, chart_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate chart data based on configuration."""
		chart_type = chart_config.get('type', ChartType.LINE)
		metrics = chart_config.get('metrics', [])
		time_range = chart_config.get('time_range', '1h')
		
		# Parse time range
		start_time = self._parse_time_range(time_range)
		
		chart_data = {
			'type': chart_type.value,
			'title': chart_config.get('title', 'Chart'),
			'labels': [],
			'datasets': []
		}
		
		if chart_type == ChartType.LINE:
			chart_data = await self._generate_line_chart(metrics, start_time, chart_config)
		elif chart_type == ChartType.BAR:
			chart_data = await self._generate_bar_chart(metrics, start_time, chart_config)
		elif chart_type == ChartType.PIE:
			chart_data = await self._generate_pie_chart(metrics, chart_config)
		elif chart_type == ChartType.GAUGE:
			chart_data = await self._generate_gauge_chart(metrics, chart_config)
		
		return chart_data
	
	async def _generate_line_chart(self, metrics: List[str], start_time: datetime, 
								  config: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate line chart data."""
		chart_data = {
			'type': 'line',
			'title': config.get('title', 'Line Chart'),
			'labels': [],
			'datasets': []
		}
		
		# Get time points for x-axis
		now = datetime.utcnow()
		time_points = []
		current = start_time
		interval = timedelta(minutes=5)  # 5-minute intervals
		
		while current <= now:
			time_points.append(current)
			current += interval
		
		chart_data['labels'] = [t.strftime('%H:%M') for t in time_points]
		
		# Generate dataset for each metric
		colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
		
		for i, metric_name in enumerate(metrics):
			values = await self.metrics_collector.get_metric_values(
				metric_name, start_time, now, len(time_points)
			)
			
			# Interpolate values to match time points
			data_points = self._interpolate_values(values, time_points)
			
			dataset = {
				'label': metric_name.replace('_', ' ').title(),
				'data': data_points,
				'borderColor': colors[i % len(colors)],
				'backgroundColor': colors[i % len(colors)] + '20',  # Add transparency
				'fill': False,
				'tension': 0.4
			}
			
			chart_data['datasets'].append(dataset)
		
		return chart_data
	
	async def _generate_bar_chart(self, metrics: List[str], start_time: datetime,
								 config: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate bar chart data."""
		chart_data = {
			'type': 'bar',
			'title': config.get('title', 'Bar Chart'),
			'labels': [],
			'datasets': []
		}
		
		# Get latest values for each metric
		colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
		data_points = []
		labels = []
		
		for metric_name in metrics:
			latest_value = await self.metrics_collector.get_latest_value(metric_name)
			if latest_value is not None:
				data_points.append(latest_value)
				labels.append(metric_name.replace('_', ' ').title())
		
		chart_data['labels'] = labels
		chart_data['datasets'] = [{
			'label': 'Current Values',
			'data': data_points,
			'backgroundColor': colors[:len(data_points)],
			'borderColor': colors[:len(data_points)],
			'borderWidth': 1
		}]
		
		return chart_data
	
	async def _generate_pie_chart(self, metrics: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate pie chart data."""
		chart_data = {
			'type': 'pie',
			'title': config.get('title', 'Pie Chart'),
			'labels': [],
			'datasets': []
		}
		
		# Get latest values for each metric
		colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
		data_points = []
		labels = []
		
		for metric_name in metrics:
			latest_value = await self.metrics_collector.get_latest_value(metric_name)
			if latest_value is not None and latest_value > 0:
				data_points.append(latest_value)
				labels.append(metric_name.replace('_', ' ').title())
		
		chart_data['labels'] = labels
		chart_data['datasets'] = [{
			'data': data_points,
			'backgroundColor': colors[:len(data_points)],
			'borderColor': colors[:len(data_points)],
			'borderWidth': 1
		}]
		
		return chart_data
	
	async def _generate_gauge_chart(self, metrics: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate gauge chart data."""
		if not metrics:
			return {}
		
		metric_name = metrics[0]  # Gauge shows single metric
		latest_value = await self.metrics_collector.get_latest_value(metric_name)
		
		if latest_value is None:
			latest_value = 0
		
		max_value = config.get('max_value', 100)
		min_value = config.get('min_value', 0)
		
		chart_data = {
			'type': 'gauge',
			'title': config.get('title', metric_name.replace('_', ' ').title()),
			'value': latest_value,
			'min': min_value,
			'max': max_value,
			'unit': config.get('unit', ''),
			'thresholds': config.get('thresholds', [
				{'value': max_value * 0.7, 'color': '#FFCE56'},
				{'value': max_value * 0.9, 'color': '#FF6384'}
			])
		}
		
		return chart_data
	
	def _parse_time_range(self, time_range: str) -> datetime:
		"""Parse time range string to start datetime."""
		now = datetime.utcnow()
		
		if time_range == '1h':
			return now - timedelta(hours=1)
		elif time_range == '24h':
			return now - timedelta(hours=24)
		elif time_range == '7d':
			return now - timedelta(days=7)
		elif time_range == '30d':
			return now - timedelta(days=30)
		else:
			return now - timedelta(hours=1)  # Default to 1 hour
	
	def _interpolate_values(self, values: List[MetricValue], time_points: List[datetime]) -> List[float]:
		"""Interpolate metric values to match time points."""
		if not values:
			return [0] * len(time_points)
		
		# Create a simple interpolation
		data_points = []
		value_dict = {v.timestamp: v.value for v in values}
		
		for time_point in time_points:
			# Find closest value
			closest_value = 0
			min_diff = float('inf')
			
			for timestamp, value in value_dict.items():
				diff = abs((timestamp - time_point).total_seconds())
				if diff < min_diff:
					min_diff = diff
					closest_value = value
			
			data_points.append(closest_value)
		
		return data_points


class DashboardManager:
	"""Manages monitoring dashboards."""
	
	def __init__(self, metrics_collector: MetricsCollector, chart_generator: ChartGenerator):
		self.metrics_collector = metrics_collector
		self.chart_generator = chart_generator
		self.dashboards: Dict[str, Dashboard] = {}
		
		# Create default dashboards
		self._create_default_dashboards()
	
	def _create_default_dashboards(self):
		"""Create default monitoring dashboards."""
		
		# System Overview Dashboard
		system_dashboard = Dashboard(
			id="system_overview",
			name="System Overview",
			description="Overall system health and performance metrics",
			charts=[
				{
					'id': 'cpu_memory_chart',
					'type': ChartType.LINE,
					'title': 'CPU & Memory Usage',
					'metrics': ['system_cpu_percent', 'system_memory_percent'],
					'time_range': '1h',
					'size': 'large'
				},
				{
					'id': 'disk_usage_gauge',
					'type': ChartType.GAUGE,
					'title': 'Disk Usage',
					'metrics': ['system_disk_percent'],
					'max_value': 100,
					'unit': '%',
					'size': 'medium'
				},
				{
					'id': 'network_chart',
					'type': ChartType.LINE,
					'title': 'Network I/O',
					'metrics': ['system_network_bytes_sent', 'system_network_bytes_recv'],
					'time_range': '1h',
					'size': 'medium'
				}
			]
		)
		
		# Workflow Performance Dashboard
		workflow_dashboard = Dashboard(
			id="workflow_performance",
			name="Workflow Performance",
			description="Workflow execution metrics and performance analysis",
			charts=[
				{
					'id': 'workflow_success_rate_gauge',
					'type': ChartType.GAUGE,
					'title': 'Success Rate',
					'metrics': ['workflows_success_rate'],
					'max_value': 100,
					'unit': '%',
					'size': 'medium',
					'thresholds': [
						{'value': 90, 'color': '#FFCE56'},
						{'value': 95, 'color': '#FF6384'}
					]
				},
				{
					'id': 'workflow_counts_bar',
					'type': ChartType.BAR,
					'title': 'Workflow Activity (Last Hour)',
					'metrics': ['workflows_completed_hour', 'workflows_failed_hour'],
					'size': 'medium'
				},
				{
					'id': 'active_workflows_line',
					'type': ChartType.LINE,
					'title': 'Active Workflows',
					'metrics': ['workflows_active'],
					'time_range': '24h',
					'size': 'large'
				},
				{
					'id': 'execution_time_line',
					'type': ChartType.LINE,
					'title': 'Average Execution Time',
					'metrics': ['workflows_avg_duration'],
					'time_range': '24h',
					'size': 'large'
				},
				{
					'id': 'queue_status_bar',
					'type': ChartType.BAR,
					'title': 'Queue Status',
					'metrics': ['queue_pending_tasks', 'queue_processing_tasks'],
					'size': 'medium'
				}
			]
		)
		
		# Database & API Dashboard
		performance_dashboard = Dashboard(
			id="performance_metrics",
			name="Performance Metrics",
			description="Database and API performance monitoring",
			charts=[
				{
					'id': 'db_connections_gauge',
					'type': ChartType.GAUGE,
					'title': 'Database Connections',
					'metrics': ['db_connections_active'],
					'max_value': 20,
					'unit': '',
					'size': 'medium'
				},
				{
					'id': 'api_response_time_line',
					'type': ChartType.LINE,
					'title': 'API Response Time',
					'metrics': ['api_response_time_avg'],
					'time_range': '1h',
					'size': 'large'
				},
				{
					'id': 'api_requests_line',
					'type': ChartType.LINE,
					'title': 'API Requests per Minute',
					'metrics': ['api_requests_per_minute'],
					'time_range': '1h',
					'size': 'large'
				}
			]
		)
		
		self.dashboards['system_overview'] = system_dashboard
		self.dashboards['workflow_performance'] = workflow_dashboard
		self.dashboards['performance_metrics'] = performance_dashboard
	
	async def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
		"""Get complete dashboard data including chart data."""
		dashboard = self.dashboards.get(dashboard_id)
		if not dashboard:
			return None
		
		dashboard_data = {
			'id': dashboard.id,
			'name': dashboard.name,
			'description': dashboard.description,
			'refresh_interval': dashboard.refresh_interval,
			'auto_refresh': dashboard.auto_refresh,
			'charts': []
		}
		
		# Generate data for each chart
		for chart_config in dashboard.charts:
			try:
				chart_data = await self.chart_generator.generate_chart_data(chart_config)
				chart_data.update({
					'id': chart_config['id'],
					'size': chart_config.get('size', 'medium'),
					'position': chart_config.get('position', {})
				})
				dashboard_data['charts'].append(chart_data)
			except Exception as e:
				logger.error(f"Error generating chart data for {chart_config['id']}: {e}")
		
		return dashboard_data
	
	def list_dashboards(self) -> List[Dict[str, Any]]:
		"""List all available dashboards."""
		return [
			{
				'id': dashboard.id,
				'name': dashboard.name,
				'description': dashboard.description,
				'chart_count': len(dashboard.charts),
				'created_at': dashboard.created_at.isoformat(),
				'updated_at': dashboard.updated_at.isoformat()
			}
			for dashboard in self.dashboards.values()
		]
	
	def create_dashboard(self, dashboard: Dashboard):
		"""Create a new dashboard."""
		self.dashboards[dashboard.id] = dashboard
	
	def update_dashboard(self, dashboard_id: str, updates: Dict[str, Any]):
		"""Update an existing dashboard."""
		if dashboard_id in self.dashboards:
			dashboard = self.dashboards[dashboard_id]
			
			if 'name' in updates:
				dashboard.name = updates['name']
			if 'description' in updates:
				dashboard.description = updates['description']
			if 'charts' in updates:
				dashboard.charts = updates['charts']
			if 'refresh_interval' in updates:
				dashboard.refresh_interval = updates['refresh_interval']
			if 'auto_refresh' in updates:
				dashboard.auto_refresh = updates['auto_refresh']
			
			dashboard.updated_at = datetime.utcnow()
	
	def delete_dashboard(self, dashboard_id: str):
		"""Delete a dashboard."""
		if dashboard_id in self.dashboards:
			del self.dashboards[dashboard_id]


class MonitoringDashboardService(APGBaseService):
	"""Main monitoring and analytics dashboard service."""
	
	def __init__(self):
		super().__init__()
		self.metrics_collector = MetricsCollector()
		self.chart_generator = ChartGenerator(self.metrics_collector)
		self.dashboard_manager = DashboardManager(self.metrics_collector, self.chart_generator)
		self.alert_manager = AlertManager(self.metrics_collector)
		self.database = APGDatabase()
		self.audit = APGAuditLogger()
	
	async def start(self):
		"""Start monitoring dashboard service."""
		await super().start()
		await self.metrics_collector.start()
		await self.alert_manager.start()
		logger.info("Monitoring dashboard service started")
	
	async def stop(self):
		"""Stop monitoring dashboard service."""
		await self.metrics_collector.stop()
		await self.alert_manager.stop()
		await super().stop()
		logger.info("Monitoring dashboard service stopped")
	
	async def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
		"""Get dashboard data with real-time metrics."""
		try:
			return await self.dashboard_manager.get_dashboard_data(dashboard_id)
		except Exception as e:
			logger.error(f"Failed to get dashboard data: {e}")
			return None
	
	async def list_dashboards(self) -> List[Dict[str, Any]]:
		"""List all available dashboards."""
		try:
			return self.dashboard_manager.list_dashboards()
		except Exception as e:
			logger.error(f"Failed to list dashboards: {e}")
			return []
	
	async def get_realtime_metrics(self, metric_names: List[str]) -> Dict[str, Any]:
		"""Get real-time metric values."""
		try:
			metrics = {}
			for metric_name in metric_names:
				value = await self.metrics_collector.get_latest_value(metric_name)
				metrics[metric_name] = {
					'value': value,
					'timestamp': datetime.utcnow().isoformat()
				}
			return metrics
		except Exception as e:
			logger.error(f"Failed to get realtime metrics: {e}")
			return {}
	
	async def get_metric_history(self, metric_name: str, time_range: str = '1h') -> List[Dict[str, Any]]:
		"""Get historical metric data."""
		try:
			start_time = self._parse_time_range(time_range)
			values = await self.metrics_collector.get_metric_values(metric_name, start_time)
			
			return [
				{
					'timestamp': value.timestamp.isoformat(),
					'value': value.value,
					'labels': value.labels
				}
				for value in values
			]
		except Exception as e:
			logger.error(f"Failed to get metric history: {e}")
			return []
	
	async def get_alerts(self, active_only: bool = False) -> List[Dict[str, Any]]:
		"""Get alert configurations and status."""
		try:
			alerts = self.alert_manager.list_alerts()
			alert_data = []
			
			for alert in alerts:
				if active_only and not alert.enabled:
					continue
				
				alert_info = {
					'id': alert.id,
					'name': alert.name,
					'description': alert.description,
					'metric_name': alert.metric_name,
					'condition': alert.condition,
					'threshold': alert.threshold,
					'level': alert.level.value,
					'enabled': alert.enabled,
					'last_triggered': alert.last_triggered.isoformat() if alert.last_triggered else None,
					'trigger_count': alert.trigger_count,
					'created_at': alert.created_at.isoformat()
				}
				
				# Get current metric value
				current_value = await self.metrics_collector.get_latest_value(alert.metric_name)
				if current_value is not None:
					alert_info['current_value'] = current_value
					alert_info['is_triggered'] = self.alert_manager._evaluate_condition(
						current_value, alert.condition, alert.threshold
					)
				
				alert_data.append(alert_info)
			
			return alert_data
			
		except Exception as e:
			logger.error(f"Failed to get alerts: {e}")
			return []
	
	async def create_custom_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
		"""Create a custom dashboard."""
		try:
			dashboard_id = dashboard_config.get('id', uuid7str())
			dashboard = Dashboard(
				id=dashboard_id,
				name=dashboard_config['name'],
				description=dashboard_config.get('description', ''),
				charts=dashboard_config.get('charts', []),
				refresh_interval=dashboard_config.get('refresh_interval', 30),
				auto_refresh=dashboard_config.get('auto_refresh', True)
			)
			
			self.dashboard_manager.create_dashboard(dashboard)
			
			await self.audit.log_event({
				'event_type': 'dashboard_created',
				'dashboard_id': dashboard_id,
				'dashboard_name': dashboard.name
			})
			
			return dashboard_id
			
		except Exception as e:
			logger.error(f"Failed to create custom dashboard: {e}")
			raise
	
	async def get_system_health_summary(self) -> Dict[str, Any]:
		"""Get overall system health summary."""
		try:
			# Get key metrics
			cpu_usage = await self.metrics_collector.get_latest_value('system_cpu_percent') or 0
			memory_usage = await self.metrics_collector.get_latest_value('system_memory_percent') or 0
			disk_usage = await self.metrics_collector.get_latest_value('system_disk_percent') or 0
			success_rate = await self.metrics_collector.get_latest_value('workflows_success_rate') or 100
			active_workflows = await self.metrics_collector.get_latest_value('workflows_active') or 0
			
			# Determine overall health status
			health_score = 100
			issues = []
			
			if cpu_usage > 90:
				health_score -= 20
				issues.append(f"High CPU usage: {cpu_usage:.1f}%")
			elif cpu_usage > 80:
				health_score -= 10
				issues.append(f"Elevated CPU usage: {cpu_usage:.1f}%")
			
			if memory_usage > 85:
				health_score -= 20
				issues.append(f"High memory usage: {memory_usage:.1f}%")
			elif memory_usage > 75:
				health_score -= 10
				issues.append(f"Elevated memory usage: {memory_usage:.1f}%")
			
			if success_rate < 95:
				health_score -= 25
				issues.append(f"Low workflow success rate: {success_rate:.1f}%")
			elif success_rate < 98:
				health_score -= 10
				issues.append(f"Workflow success rate below optimal: {success_rate:.1f}%")
			
			# Determine health status
			if health_score >= 90:
				status = "excellent"
				status_color = "#4CAF50"
			elif health_score >= 75:
				status = "good"
				status_color = "#8BC34A"
			elif health_score >= 60:
				status = "fair"
				status_color = "#FFC107"
			elif health_score >= 40:
				status = "poor"
				status_color = "#FF9800"
			else:
				status = "critical"
				status_color = "#F44336"
			
			return {
				'health_score': health_score,
				'status': status,
				'status_color': status_color,
				'issues': issues,
				'metrics': {
					'cpu_usage': cpu_usage,
					'memory_usage': memory_usage,
					'disk_usage': disk_usage,
					'success_rate': success_rate,
					'active_workflows': int(active_workflows)
				},
				'timestamp': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Failed to get system health summary: {e}")
			return {
				'health_score': 0,
				'status': 'unknown',
				'status_color': '#9E9E9E',
				'issues': ['Unable to retrieve system health data'],
				'metrics': {},
				'timestamp': datetime.utcnow().isoformat()
			}
	
	def _parse_time_range(self, time_range: str) -> datetime:
		"""Parse time range string to start datetime."""
		now = datetime.utcnow()
		
		if time_range == '1h':
			return now - timedelta(hours=1)
		elif time_range == '24h':
			return now - timedelta(hours=24)
		elif time_range == '7d':
			return now - timedelta(days=7)
		elif time_range == '30d':
			return now - timedelta(days=30)
		else:
			return now - timedelta(hours=1)  # Default to 1 hour
	
	async def health_check(self) -> bool:
		"""Health check for monitoring service."""
		try:
			# Check if metrics are being collected
			cpu_value = await self.metrics_collector.get_latest_value('system_cpu_percent')
			return cpu_value is not None
		except Exception:
			return False


# Global monitoring dashboard service instance
monitoring_dashboard_service = MonitoringDashboardService()