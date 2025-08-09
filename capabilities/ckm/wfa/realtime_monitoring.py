"""
APG Workflow & Business Process Management - Real-time Monitoring & Dashboard

Real-time process monitoring, performance tracking, and intelligent alerting
with live dashboard capabilities and advanced analytics.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
import statistics
import time

from models import (
	APGTenantContext, WBPMServiceResponse, WBPMPagedResponse,
	WBPMProcessInstance, WBPMTask, ProcessStatus, TaskStatus, TaskPriority
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Monitoring Core Classes
# =============================================================================

class AlertSeverity(str, Enum):
	"""Alert severity levels."""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	INFO = "info"


class AlertType(str, Enum):
	"""Types of monitoring alerts."""
	PROCESS_DELAY = "process_delay"
	TASK_OVERDUE = "task_overdue"
	HIGH_ERROR_RATE = "high_error_rate"
	PERFORMANCE_DEGRADATION = "performance_degradation"
	CAPACITY_WARNING = "capacity_warning"
	SLA_BREACH = "sla_breach"
	SYSTEM_HEALTH = "system_health"
	BUSINESS_RULE_VIOLATION = "business_rule_violation"
	SECURITY_INCIDENT = "security_incident"


class MetricType(str, Enum):
	"""Types of metrics to track."""
	COUNTER = "counter"
	GAUGE = "gauge"
	HISTOGRAM = "histogram"
	TIMER = "timer"
	RATE = "rate"


class DashboardType(str, Enum):
	"""Dashboard types for different users."""
	EXECUTIVE = "executive"
	OPERATIONAL = "operational"
	TECHNICAL = "technical"
	CUSTOM = "custom"


@dataclass
class ProcessMetric:
	"""Individual process metric data point."""
	metric_id: str = field(default_factory=lambda: f"metric_{uuid.uuid4().hex}")
	process_id: str = ""
	metric_name: str = ""
	metric_type: MetricType = MetricType.GAUGE
	value: float = 0.0
	unit: str = ""
	timestamp: datetime = field(default_factory=datetime.utcnow)
	tags: Dict[str, str] = field(default_factory=dict)
	tenant_id: str = ""


@dataclass
class ProcessAlert:
	"""Process monitoring alert."""
	alert_id: str = field(default_factory=lambda: f"alert_{uuid.uuid4().hex}")
	alert_type: AlertType = AlertType.PROCESS_DELAY
	severity: AlertSeverity = AlertSeverity.MEDIUM
	title: str = ""
	description: str = ""
	process_id: Optional[str] = None
	task_id: Optional[str] = None
	metric_data: Dict[str, Any] = field(default_factory=dict)
	threshold_breached: Optional[float] = None
	current_value: Optional[float] = None
	created_at: datetime = field(default_factory=datetime.utcnow)
	acknowledged_at: Optional[datetime] = None
	acknowledged_by: Optional[str] = None
	resolved_at: Optional[datetime] = None
	resolved_by: Optional[str] = None
	tenant_id: str = ""


@dataclass
class PerformanceSnapshot:
	"""Point-in-time performance snapshot."""
	snapshot_id: str = field(default_factory=lambda: f"snapshot_{uuid.uuid4().hex}")
	timestamp: datetime = field(default_factory=datetime.utcnow)
	active_processes: int = 0
	active_tasks: int = 0
	completed_processes_1h: int = 0
	avg_process_duration: float = 0.0
	avg_task_duration: float = 0.0
	error_rate: float = 0.0
	throughput_per_hour: float = 0.0
	system_load: float = 0.0
	memory_usage: float = 0.0
	tenant_id: str = ""


@dataclass
class DashboardWidget:
	"""Dashboard widget configuration."""
	widget_id: str = field(default_factory=lambda: f"widget_{uuid.uuid4().hex}")
	widget_type: str = ""  # chart_line, chart_bar, metric_card, alert_list, etc.
	title: str = ""
	data_source: str = ""
	config: Dict[str, Any] = field(default_factory=dict)
	position: Dict[str, int] = field(default_factory=dict)  # x, y, width, height
	refresh_interval: int = 30  # seconds
	tenant_id: str = ""


@dataclass
class MonitoringThreshold:
	"""Configurable monitoring threshold."""
	threshold_id: str = field(default_factory=lambda: f"threshold_{uuid.uuid4().hex}")
	metric_name: str = ""
	alert_type: AlertType = AlertType.PERFORMANCE_DEGRADATION
	severity: AlertSeverity = AlertSeverity.MEDIUM
	operator: str = ">"  # >, <, >=, <=, ==, !=
	threshold_value: float = 0.0
	time_window: int = 300  # seconds
	min_occurrences: int = 1
	enabled: bool = True
	tenant_id: str = ""


# =============================================================================
# Real-time Metrics Collector
# =============================================================================

class MetricsCollector:
	"""Collect and aggregate real-time metrics."""
	
	def __init__(self, max_buffer_size: int = 10000):
		self.max_buffer_size = max_buffer_size
		self.metrics_buffer: deque = deque(maxlen=max_buffer_size)
		self.aggregated_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
		self.last_aggregation = datetime.utcnow()
		
	async def record_metric(
		self,
		metric: ProcessMetric,
		context: APGTenantContext
	) -> None:
		"""Record a new metric data point."""
		try:
			# Add tenant context
			metric.tenant_id = context.tenant_id
			
			# Add to buffer
			self.metrics_buffer.append(metric)
			
			# Update real-time aggregations
			await self._update_aggregations(metric)
			
			logger.debug(f"Metric recorded: {metric.metric_name} = {metric.value}")
			
		except Exception as e:
			logger.error(f"Error recording metric: {e}")
	
	async def _update_aggregations(self, metric: ProcessMetric) -> None:
		"""Update real-time metric aggregations."""
		key = f"{metric.process_id}:{metric.metric_name}"
		
		if key not in self.aggregated_metrics:
			self.aggregated_metrics[key] = {
				"count": 0,
				"sum": 0.0,
				"min": float('inf'),
				"max": float('-inf'),
				"values": deque(maxlen=100),  # Keep last 100 values
				"last_updated": datetime.utcnow()
			}
		
		agg = self.aggregated_metrics[key]
		agg["count"] += 1
		agg["sum"] += metric.value
		agg["min"] = min(agg["min"], metric.value)
		agg["max"] = max(agg["max"], metric.value)
		agg["values"].append(metric.value)
		agg["last_updated"] = datetime.utcnow()
		agg["avg"] = agg["sum"] / agg["count"]
		
		# Calculate standard deviation if we have enough values
		if len(agg["values"]) >= 2:
			agg["std_dev"] = statistics.stdev(agg["values"])
	
	async def get_metrics_summary(
		self,
		process_id: Optional[str] = None,
		metric_name: Optional[str] = None,
		time_window: Optional[timedelta] = None
	) -> Dict[str, Any]:
		"""Get aggregated metrics summary."""
		try:
			cutoff_time = datetime.utcnow() - (time_window or timedelta(hours=1))
			
			# Filter metrics
			filtered_metrics = []
			for metric in self.metrics_buffer:
				if metric.timestamp < cutoff_time:
					continue
				if process_id and metric.process_id != process_id:
					continue
				if metric_name and metric.metric_name != metric_name:
					continue
				filtered_metrics.append(metric)
			
			if not filtered_metrics:
				return {"total_metrics": 0, "aggregations": {}}
			
			# Calculate aggregations
			aggregations = defaultdict(lambda: {
				"count": 0,
				"sum": 0.0,
				"min": float('inf'),
				"max": float('-inf'),
				"values": []
			})
			
			for metric in filtered_metrics:
				key = metric.metric_name
				agg = aggregations[key]
				agg["count"] += 1
				agg["sum"] += metric.value
				agg["min"] = min(agg["min"], metric.value)
				agg["max"] = max(agg["max"], metric.value)
				agg["values"].append(metric.value)
			
			# Calculate derived metrics
			for key, agg in aggregations.items():
				agg["avg"] = agg["sum"] / agg["count"]
				if len(agg["values"]) >= 2:
					agg["std_dev"] = statistics.stdev(agg["values"])
				else:
					agg["std_dev"] = 0.0
				# Remove values array for response (too large)
				del agg["values"]
			
			return {
				"total_metrics": len(filtered_metrics),
				"time_window": str(time_window) if time_window else "1 hour",
				"aggregations": dict(aggregations)
			}
			
		except Exception as e:
			logger.error(f"Error getting metrics summary: {e}")
			return {"error": str(e)}


# =============================================================================
# Alert Management System
# =============================================================================

class AlertManager:
	"""Manage process monitoring alerts."""
	
	def __init__(self):
		self.active_alerts: Dict[str, ProcessAlert] = {}
		self.alert_history: List[ProcessAlert] = []
		self.thresholds: Dict[str, MonitoringThreshold] = {}
		self.notification_callbacks: List[Callable] = []
		
	async def add_threshold(
		self,
		threshold: MonitoringThreshold,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Add monitoring threshold."""
		try:
			threshold.tenant_id = context.tenant_id
			self.thresholds[threshold.threshold_id] = threshold
			
			logger.info(f"Monitoring threshold added: {threshold.metric_name}")
			
			return WBPMServiceResponse(
				success=True,
				message="Monitoring threshold added successfully",
				data={"threshold_id": threshold.threshold_id}
			)
			
		except Exception as e:
			logger.error(f"Error adding threshold: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to add threshold: {e}",
				errors=[str(e)]
			)
	
	async def check_thresholds(
		self,
		metric: ProcessMetric,
		context: APGTenantContext
	) -> List[ProcessAlert]:
		"""Check if metric breaches any thresholds."""
		alerts_triggered = []
		
		try:
			for threshold in self.thresholds.values():
				if not threshold.enabled:
					continue
				if threshold.tenant_id != context.tenant_id:
					continue
				if threshold.metric_name != metric.metric_name:
					continue
				
				# Check threshold condition
				breached = await self._evaluate_threshold(metric, threshold)
				
				if breached:
					alert = ProcessAlert(
						alert_type=threshold.alert_type,
						severity=threshold.severity,
						title=f"Threshold Breached: {threshold.metric_name}",
						description=f"Metric {threshold.metric_name} {threshold.operator} {threshold.threshold_value}",
						process_id=metric.process_id,
						metric_data={
							"metric_name": metric.metric_name,
							"threshold_value": threshold.threshold_value,
							"operator": threshold.operator
						},
						threshold_breached=threshold.threshold_value,
						current_value=metric.value,
						tenant_id=context.tenant_id
					)
					
					# Check if this is a new alert (not already active)
					alert_key = f"{metric.process_id}:{threshold.metric_name}:{threshold.alert_type}"
					if alert_key not in self.active_alerts:
						self.active_alerts[alert_key] = alert
						self.alert_history.append(alert)
						alerts_triggered.append(alert)
						
						# Trigger notifications
						await self._trigger_notifications(alert)
			
			return alerts_triggered
			
		except Exception as e:
			logger.error(f"Error checking thresholds: {e}")
			return []
	
	async def _evaluate_threshold(
		self,
		metric: ProcessMetric,
		threshold: MonitoringThreshold
	) -> bool:
		"""Evaluate if metric breaches threshold."""
		try:
			value = metric.value
			threshold_value = threshold.threshold_value
			operator = threshold.operator
			
			if operator == ">":
				return value > threshold_value
			elif operator == "<":
				return value < threshold_value
			elif operator == ">=":
				return value >= threshold_value
			elif operator == "<=":
				return value <= threshold_value
			elif operator == "==":
				return abs(value - threshold_value) < 0.001  # Float comparison
			elif operator == "!=":
				return abs(value - threshold_value) >= 0.001
			else:
				logger.warning(f"Unknown threshold operator: {operator}")
				return False
				
		except Exception as e:
			logger.error(f"Error evaluating threshold: {e}")
			return False
	
	async def _trigger_notifications(self, alert: ProcessAlert) -> None:
		"""Trigger alert notifications."""
		try:
			for callback in self.notification_callbacks:
				try:
					await callback(alert)
				except Exception as e:
					logger.error(f"Error in notification callback: {e}")
					
		except Exception as e:
			logger.error(f"Error triggering notifications: {e}")
	
	async def acknowledge_alert(
		self,
		alert_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Acknowledge an active alert."""
		try:
			# Find alert in active alerts
			alert_key = None
			for key, alert in self.active_alerts.items():
				if alert.alert_id == alert_id and alert.tenant_id == context.tenant_id:
					alert_key = key
					break
			
			if not alert_key:
				return WBPMServiceResponse(
					success=False,
					message="Alert not found or already resolved",
					errors=["Alert not found"]
				)
			
			# Acknowledge alert
			alert = self.active_alerts[alert_key]
			alert.acknowledged_at = datetime.utcnow()
			alert.acknowledged_by = context.user_id
			
			logger.info(f"Alert acknowledged: {alert_id} by {context.user_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Alert acknowledged successfully",
				data={"alert_id": alert_id}
			)
			
		except Exception as e:
			logger.error(f"Error acknowledging alert: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to acknowledge alert: {e}",
				errors=[str(e)]
			)
	
	async def resolve_alert(
		self,
		alert_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Resolve an active alert."""
		try:
			# Find and remove alert from active alerts
			alert_key = None
			for key, alert in self.active_alerts.items():
				if alert.alert_id == alert_id and alert.tenant_id == context.tenant_id:
					alert_key = key
					break
			
			if not alert_key:
				return WBPMServiceResponse(
					success=False,
					message="Alert not found",
					errors=["Alert not found"]
				)
			
			# Resolve alert
			alert = self.active_alerts.pop(alert_key)
			alert.resolved_at = datetime.utcnow()
			alert.resolved_by = context.user_id
			
			logger.info(f"Alert resolved: {alert_id} by {context.user_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Alert resolved successfully",
				data={"alert_id": alert_id}
			)
			
		except Exception as e:
			logger.error(f"Error resolving alert: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to resolve alert: {e}",
				errors=[str(e)]
			)
	
	async def get_active_alerts(
		self,
		context: APGTenantContext,
		severity_filter: Optional[AlertSeverity] = None
	) -> WBPMServiceResponse:
		"""Get active alerts for tenant."""
		try:
			# Filter alerts by tenant and severity
			filtered_alerts = [
				alert for alert in self.active_alerts.values()
				if alert.tenant_id == context.tenant_id and
				(not severity_filter or alert.severity == severity_filter)
			]
			
			# Sort by severity and creation time
			severity_order = {
				AlertSeverity.CRITICAL: 0,
				AlertSeverity.HIGH: 1,
				AlertSeverity.MEDIUM: 2,
				AlertSeverity.LOW: 3,
				AlertSeverity.INFO: 4
			}
			
			filtered_alerts.sort(
				key=lambda a: (severity_order.get(a.severity, 5), a.created_at),
				reverse=True
			)
			
			return WBPMServiceResponse(
				success=True,
				message="Active alerts retrieved successfully",
				data={
					"alerts": [alert.__dict__ for alert in filtered_alerts],
					"total_count": len(filtered_alerts)
				}
			)
			
		except Exception as e:
			logger.error(f"Error getting active alerts: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to get alerts: {e}",
				errors=[str(e)]
			)


# =============================================================================
# Performance Monitor
# =============================================================================

class PerformanceMonitor:
	"""Monitor system and process performance."""
	
	def __init__(self):
		self.performance_history: List[PerformanceSnapshot] = []
		self.process_timings: Dict[str, List[float]] = defaultdict(list)
		self.task_timings: Dict[str, List[float]] = defaultdict(list)
		
	async def record_process_completion(
		self,
		process_id: str,
		duration_seconds: float,
		context: APGTenantContext
	) -> None:
		"""Record process completion timing."""
		try:
			self.process_timings[context.tenant_id].append(duration_seconds)
			
			# Keep only recent timings (last 1000)
			if len(self.process_timings[context.tenant_id]) > 1000:
				self.process_timings[context.tenant_id] = self.process_timings[context.tenant_id][-500:]
			
			logger.debug(f"Process completion recorded: {process_id} in {duration_seconds}s")
			
		except Exception as e:
			logger.error(f"Error recording process completion: {e}")
	
	async def record_task_completion(
		self,
		task_id: str,
		duration_seconds: float,
		context: APGTenantContext
	) -> None:
		"""Record task completion timing."""
		try:
			self.task_timings[context.tenant_id].append(duration_seconds)
			
			# Keep only recent timings (last 1000)
			if len(self.task_timings[context.tenant_id]) > 1000:
				self.task_timings[context.tenant_id] = self.task_timings[context.tenant_id][-500:]
			
			logger.debug(f"Task completion recorded: {task_id} in {duration_seconds}s")
			
		except Exception as e:
			logger.error(f"Error recording task completion: {e}")
	
	async def create_performance_snapshot(
		self,
		active_processes: int,
		active_tasks: int,
		context: APGTenantContext
	) -> PerformanceSnapshot:
		"""Create current performance snapshot."""
		try:
			# Calculate averages
			process_timings = self.process_timings.get(context.tenant_id, [])
			task_timings = self.task_timings.get(context.tenant_id, [])
			
			avg_process_duration = statistics.mean(process_timings) if process_timings else 0.0
			avg_task_duration = statistics.mean(task_timings) if task_timings else 0.0
			
			# Calculate completed processes in last hour
			one_hour_ago = datetime.utcnow() - timedelta(hours=1)
			completed_1h = len([
				snapshot for snapshot in self.performance_history[-60:]  # Approximate
				if snapshot.timestamp >= one_hour_ago
			])
			
			# Calculate throughput
			throughput_per_hour = len(process_timings) if len(process_timings) <= 60 else 60
			
			# Create snapshot
			snapshot = PerformanceSnapshot(
				active_processes=active_processes,
				active_tasks=active_tasks,
				completed_processes_1h=completed_1h,
				avg_process_duration=avg_process_duration,
				avg_task_duration=avg_task_duration,
				error_rate=0.0,  # Would be calculated from error metrics
				throughput_per_hour=throughput_per_hour,
				system_load=0.0,  # Would be from system metrics
				memory_usage=0.0,  # Would be from system metrics
				tenant_id=context.tenant_id
			)
			
			# Store snapshot
			self.performance_history.append(snapshot)
			
			# Keep only recent snapshots (last 24 hours worth)
			if len(self.performance_history) > 1440:  # 24 hours * 60 minutes
				self.performance_history = self.performance_history[-720:]  # Keep 12 hours
			
			return snapshot
			
		except Exception as e:
			logger.error(f"Error creating performance snapshot: {e}")
			return PerformanceSnapshot(tenant_id=context.tenant_id)
	
	async def get_performance_trends(
		self,
		context: APGTenantContext,
		time_window: timedelta = timedelta(hours=4)
	) -> Dict[str, Any]:
		"""Get performance trends over time."""
		try:
			cutoff_time = datetime.utcnow() - time_window
			
			# Filter snapshots by time and tenant
			relevant_snapshots = [
				snapshot for snapshot in self.performance_history
				if snapshot.timestamp >= cutoff_time and snapshot.tenant_id == context.tenant_id
			]
			
			if not relevant_snapshots:
				return {"error": "No performance data available for the specified time window"}
			
			# Calculate trends
			timestamps = [s.timestamp.isoformat() for s in relevant_snapshots]
			active_processes = [s.active_processes for s in relevant_snapshots]
			active_tasks = [s.active_tasks for s in relevant_snapshots]
			avg_process_duration = [s.avg_process_duration for s in relevant_snapshots]
			throughput = [s.throughput_per_hour for s in relevant_snapshots]
			
			return {
				"time_window": str(time_window),
				"data_points": len(relevant_snapshots),
				"trends": {
					"timestamps": timestamps,
					"active_processes": active_processes,
					"active_tasks": active_tasks,
					"avg_process_duration": avg_process_duration,
					"throughput_per_hour": throughput
				},
				"summary": {
					"avg_active_processes": statistics.mean(active_processes),
					"avg_active_tasks": statistics.mean(active_tasks),
					"avg_duration": statistics.mean(avg_process_duration) if avg_process_duration else 0,
					"avg_throughput": statistics.mean(throughput)
				}
			}
			
		except Exception as e:
			logger.error(f"Error getting performance trends: {e}")
			return {"error": str(e)}


# =============================================================================
# Real-time Dashboard Manager
# =============================================================================

class RealtimeDashboard:
	"""Manage real-time monitoring dashboards."""
	
	def __init__(self):
		self.dashboard_configs: Dict[str, Dict[str, Any]] = {}
		self.widget_configs: Dict[str, DashboardWidget] = {}
		self.active_connections: Dict[str, Set[str]] = defaultdict(set)  # user_id -> dashboard_ids
		
	async def create_dashboard(
		self,
		dashboard_data: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Create new monitoring dashboard."""
		try:
			dashboard_id = f"dashboard_{uuid.uuid4().hex}"
			
			dashboard_config = {
				"dashboard_id": dashboard_id,
				"name": dashboard_data["name"],
				"description": dashboard_data.get("description", ""),
				"dashboard_type": DashboardType(dashboard_data.get("type", DashboardType.OPERATIONAL)),
				"widgets": [],
				"layout": dashboard_data.get("layout", {}),
				"refresh_interval": dashboard_data.get("refresh_interval", 30),
				"created_by": context.user_id,
				"created_at": datetime.utcnow().isoformat(),
				"tenant_id": context.tenant_id
			}
			
			self.dashboard_configs[dashboard_id] = dashboard_config
			
			logger.info(f"Dashboard created: {dashboard_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Dashboard created successfully",
				data={"dashboard_id": dashboard_id, "config": dashboard_config}
			)
			
		except Exception as e:
			logger.error(f"Error creating dashboard: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create dashboard: {e}",
				errors=[str(e)]
			)
	
	async def add_widget(
		self,
		dashboard_id: str,
		widget_data: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Add widget to dashboard."""
		try:
			if dashboard_id not in self.dashboard_configs:
				return WBPMServiceResponse(
					success=False,
					message="Dashboard not found",
					errors=["Dashboard not found"]
				)
			
			dashboard = self.dashboard_configs[dashboard_id]
			if dashboard["tenant_id"] != context.tenant_id:
				return WBPMServiceResponse(
					success=False,
					message="Access denied to dashboard",
					errors=["Access denied"]
				)
			
			# Create widget
			widget = DashboardWidget(
				widget_type=widget_data["widget_type"],
				title=widget_data["title"],
				data_source=widget_data["data_source"],
				config=widget_data.get("config", {}),
				position=widget_data.get("position", {}),
				refresh_interval=widget_data.get("refresh_interval", 30),
				tenant_id=context.tenant_id
			)
			
			# Store widget
			self.widget_configs[widget.widget_id] = widget
			dashboard["widgets"].append(widget.widget_id)
			
			logger.info(f"Widget added to dashboard {dashboard_id}: {widget.widget_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Widget added successfully",
				data={"widget_id": widget.widget_id}
			)
			
		except Exception as e:
			logger.error(f"Error adding widget: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to add widget: {e}",
				errors=[str(e)]
			)
	
	async def get_dashboard_data(
		self,
		dashboard_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Get dashboard configuration and data."""
		try:
			if dashboard_id not in self.dashboard_configs:
				return WBPMServiceResponse(
					success=False,
					message="Dashboard not found",
					errors=["Dashboard not found"]
				)
			
			dashboard = self.dashboard_configs[dashboard_id]
			if dashboard["tenant_id"] != context.tenant_id:
				return WBPMServiceResponse(
					success=False,
					message="Access denied to dashboard",
					errors=["Access denied"]
				)
			
			# Get widget configurations
			widgets = []
			for widget_id in dashboard["widgets"]:
				if widget_id in self.widget_configs:
					widget = self.widget_configs[widget_id]
					widgets.append({
						"widget_id": widget.widget_id,
						"widget_type": widget.widget_type,
						"title": widget.title,
						"data_source": widget.data_source,
						"config": widget.config,
						"position": widget.position,
						"refresh_interval": widget.refresh_interval
					})
			
			dashboard_data = {
				**dashboard,
				"widgets": widgets,
				"last_updated": datetime.utcnow().isoformat()
			}
			
			return WBPMServiceResponse(
				success=True,
				message="Dashboard data retrieved successfully",
				data=dashboard_data
			)
			
		except Exception as e:
			logger.error(f"Error getting dashboard data: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to get dashboard data: {e}",
				errors=[str(e)]
			)
	
	async def get_widget_data(
		self,
		widget_id: str,
		context: APGTenantContext,
		metrics_collector: MetricsCollector,
		performance_monitor: PerformanceMonitor
	) -> Dict[str, Any]:
		"""Get real-time data for widget."""
		try:
			if widget_id not in self.widget_configs:
				return {"error": "Widget not found"}
			
			widget = self.widget_configs[widget_id]
			if widget.tenant_id != context.tenant_id:
				return {"error": "Access denied"}
			
			# Get data based on widget data source
			if widget.data_source == "metrics_summary":
				data = await metrics_collector.get_metrics_summary()
			elif widget.data_source == "performance_trends":
				data = await performance_monitor.get_performance_trends(context)
			elif widget.data_source == "active_processes":
				# Would query actual process data
				data = {"active_count": 42, "pending_count": 7}
			elif widget.data_source == "task_queue":
				# Would query actual task data
				data = {"queue_size": 15, "processing": 8}
			else:
				data = {"error": f"Unknown data source: {widget.data_source}"}
			
			return {
				"widget_id": widget_id,
				"timestamp": datetime.utcnow().isoformat(),
				"data": data
			}
			
		except Exception as e:
			logger.error(f"Error getting widget data: {e}")
			return {"error": str(e)}


# =============================================================================
# Real-time Monitoring Service
# =============================================================================

class RealtimeMonitoringService:
	"""Main real-time monitoring service."""
	
	def __init__(self):
		self.metrics_collector = MetricsCollector()
		self.alert_manager = AlertManager()
		self.performance_monitor = PerformanceMonitor()
		self.dashboard = RealtimeDashboard()
		self.monitoring_tasks: Dict[str, asyncio.Task] = {}
		
	async def start_monitoring(self, context: APGTenantContext) -> WBPMServiceResponse:
		"""Start monitoring for tenant."""
		try:
			task_id = f"monitor_{context.tenant_id}"
			
			if task_id in self.monitoring_tasks:
				return WBPMServiceResponse(
					success=True,
					message="Monitoring already active",
					data={"status": "active"}
				)
			
			# Start monitoring task
			task = asyncio.create_task(self._monitoring_loop(context))
			self.monitoring_tasks[task_id] = task
			
			logger.info(f"Monitoring started for tenant: {context.tenant_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Monitoring started successfully",
				data={"status": "started", "task_id": task_id}
			)
			
		except Exception as e:
			logger.error(f"Error starting monitoring: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to start monitoring: {e}",
				errors=[str(e)]
			)
	
	async def _monitoring_loop(self, context: APGTenantContext) -> None:
		"""Main monitoring loop for a tenant."""
		try:
			logger.info(f"Monitoring loop started for tenant: {context.tenant_id}")
			
			while True:
				try:
					# Create performance snapshot
					# In production, these would be real counts from the database
					active_processes = 25  # Placeholder
					active_tasks = 18  # Placeholder
					
					snapshot = await self.performance_monitor.create_performance_snapshot(
						active_processes, active_tasks, context
					)
					
					# Record system metrics
					await self._record_system_metrics(snapshot, context)
					
					# Check for any performance issues
					await self._check_performance_alerts(snapshot, context)
					
					# Sleep for monitoring interval
					await asyncio.sleep(60)  # Check every minute
					
				except asyncio.CancelledError:
					logger.info(f"Monitoring cancelled for tenant: {context.tenant_id}")
					break
				except Exception as e:
					logger.error(f"Error in monitoring loop: {e}")
					await asyncio.sleep(60)  # Continue monitoring despite errors
					
		except Exception as e:
			logger.error(f"Fatal error in monitoring loop: {e}")
	
	async def _record_system_metrics(
		self,
		snapshot: PerformanceSnapshot,
		context: APGTenantContext
	) -> None:
		"""Record system performance metrics."""
		try:
			# Record various system metrics
			metrics = [
				ProcessMetric(
					process_id="system",
					metric_name="active_processes",
					metric_type=MetricType.GAUGE,
					value=float(snapshot.active_processes),
					unit="count",
					tenant_id=context.tenant_id
				),
				ProcessMetric(
					process_id="system",
					metric_name="active_tasks",
					metric_type=MetricType.GAUGE,
					value=float(snapshot.active_tasks),
					unit="count",
					tenant_id=context.tenant_id
				),
				ProcessMetric(
					process_id="system",
					metric_name="avg_process_duration",
					metric_type=MetricType.GAUGE,
					value=snapshot.avg_process_duration,
					unit="seconds",
					tenant_id=context.tenant_id
				),
				ProcessMetric(
					process_id="system",
					metric_name="throughput_per_hour",
					metric_type=MetricType.GAUGE,
					value=snapshot.throughput_per_hour,
					unit="processes/hour",
					tenant_id=context.tenant_id
				)
			]
			
			for metric in metrics:
				await self.metrics_collector.record_metric(metric, context)
				
				# Check thresholds for each metric
				alerts = await self.alert_manager.check_thresholds(metric, context)
				for alert in alerts:
					logger.warning(f"Alert triggered: {alert.title}")
					
		except Exception as e:
			logger.error(f"Error recording system metrics: {e}")
	
	async def _check_performance_alerts(
		self,
		snapshot: PerformanceSnapshot,
		context: APGTenantContext
	) -> None:
		"""Check for performance-related alerts."""
		try:
			# Check for high process load
			if snapshot.active_processes > 100:
				alert = ProcessAlert(
					alert_type=AlertType.CAPACITY_WARNING,
					severity=AlertSeverity.HIGH,
					title="High Process Load",
					description=f"Active processes ({snapshot.active_processes}) exceeds recommended capacity",
					metric_data={"active_processes": snapshot.active_processes},
					current_value=float(snapshot.active_processes),
					tenant_id=context.tenant_id
				)
				await self.alert_manager._trigger_notifications(alert)
			
			# Check for slow average process duration
			if snapshot.avg_process_duration > 3600:  # 1 hour
				alert = ProcessAlert(
					alert_type=AlertType.PERFORMANCE_DEGRADATION,
					severity=AlertSeverity.MEDIUM,
					title="Slow Process Performance",
					description=f"Average process duration ({snapshot.avg_process_duration:.1f}s) is unusually high",
					metric_data={"avg_duration": snapshot.avg_process_duration},
					current_value=snapshot.avg_process_duration,
					tenant_id=context.tenant_id
				)
				await self.alert_manager._trigger_notifications(alert)
				
		except Exception as e:
			logger.error(f"Error checking performance alerts: {e}")
	
	async def stop_monitoring(self, context: APGTenantContext) -> WBPMServiceResponse:
		"""Stop monitoring for tenant."""
		try:
			task_id = f"monitor_{context.tenant_id}"
			
			if task_id not in self.monitoring_tasks:
				return WBPMServiceResponse(
					success=True,
					message="Monitoring not active",
					data={"status": "inactive"}
				)
			
			# Cancel monitoring task
			task = self.monitoring_tasks.pop(task_id)
			task.cancel()
			
			try:
				await task
			except asyncio.CancelledError:
				pass
			
			logger.info(f"Monitoring stopped for tenant: {context.tenant_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Monitoring stopped successfully",
				data={"status": "stopped"}
			)
			
		except Exception as e:
			logger.error(f"Error stopping monitoring: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to stop monitoring: {e}",
				errors=[str(e)]
			)


# =============================================================================
# Service Factory
# =============================================================================

def create_realtime_monitoring_service() -> RealtimeMonitoringService:
	"""Create and configure real-time monitoring service."""
	service = RealtimeMonitoringService()
	logger.info("Real-time monitoring service created and configured")
	return service


# Export main classes
__all__ = [
	'RealtimeMonitoringService',
	'MetricsCollector',
	'AlertManager',
	'PerformanceMonitor',
	'RealtimeDashboard',
	'ProcessMetric',
	'ProcessAlert',
	'PerformanceSnapshot',
	'DashboardWidget',
	'MonitoringThreshold',
	'AlertSeverity',
	'AlertType',
	'MetricType',
	'DashboardType',
	'create_realtime_monitoring_service'
]