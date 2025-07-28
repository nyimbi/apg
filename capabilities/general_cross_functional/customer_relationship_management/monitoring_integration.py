"""
APG Customer Relationship Management - Monitoring Integration

Revolutionary monitoring and observability integration providing comprehensive
system health monitoring, performance metrics, business KPIs, and intelligent
alerting with APG monitoring ecosystem integration.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import defaultdict, deque

# APG Core imports (these would be actual APG framework imports)
from apg.core.monitoring import APGMonitoring, MetricCollector, AlertManager
from apg.core.events import EventBus, Event

# Local imports
from .service import CRMService


logger = logging.getLogger(__name__)


class MetricType(str, Enum):
	"""Types of metrics"""
	COUNTER = "counter"
	GAUGE = "gauge"
	HISTOGRAM = "histogram"
	SUMMARY = "summary"


class AlertSeverity(str, Enum):
	"""Alert severity levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


@dataclass
class Metric:
	"""Metric data structure"""
	name: str
	type: MetricType
	value: float
	timestamp: datetime
	labels: Dict[str, str]
	description: Optional[str] = None


@dataclass
class Alert:
	"""Alert data structure"""
	alert_id: str
	title: str
	description: str
	severity: AlertSeverity
	metric_name: str
	current_value: float
	threshold_value: float
	timestamp: datetime
	tenant_id: Optional[str] = None
	resolved: bool = False
	resolved_at: Optional[datetime] = None


class CRMMonitoring:
	"""
	CRM monitoring and observability manager integrating with APG monitoring
	"""
	
	def __init__(self, apg_monitoring: APGMonitoring):
		"""
		Initialize CRM monitoring
		
		Args:
			apg_monitoring: APG monitoring instance
		"""
		self.apg_monitoring = apg_monitoring
		self.metric_collector = MetricCollector()
		self.alert_manager = AlertManager()
		
		# Metrics storage
		self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
		self.metric_definitions: Dict[str, Dict[str, Any]] = {}
		
		# Alert configuration
		self.alert_rules: Dict[str, Dict[str, Any]] = {}
		self.active_alerts: Dict[str, Alert] = {}
		
		# Performance tracking
		self.request_metrics = deque(maxlen=10000)
		self.error_metrics = deque(maxlen=1000)
		
		# Business metrics cache
		self.business_metrics_cache: Dict[str, Any] = {}
		self.cache_expiry = timedelta(minutes=5)
		self.last_cache_update: Optional[datetime] = None
		
		# Monitoring state
		self._monitoring_running = False
		self._monitoring_task: Optional[asyncio.Task] = None
		self._initialized = False
		
		logger.info("📊 CRM Monitoring initialized")
	
	async def initialize(self):
		"""Initialize monitoring system"""
		try:
			logger.info("🔧 Initializing CRM monitoring...")
			
			# Setup metric definitions
			await self._setup_metric_definitions()
			
			# Configure alert rules
			await self._setup_alert_rules()
			
			# Start monitoring background tasks
			await self.start_monitoring()
			
			self._initialized = True
			logger.info("✅ CRM monitoring initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize CRM monitoring: {str(e)}", exc_info=True)
			raise
	
	async def _setup_metric_definitions(self):
		"""Setup CRM metric definitions"""
		self.metric_definitions = {
			# System metrics
			"crm_cpu_usage": {
				"type": MetricType.GAUGE,
				"description": "CPU usage percentage",
				"unit": "percent"
			},
			"crm_memory_usage": {
				"type": MetricType.GAUGE,
				"description": "Memory usage percentage", 
				"unit": "percent"
			},
			"crm_disk_usage": {
				"type": MetricType.GAUGE,
				"description": "Disk usage percentage",
				"unit": "percent"
			},
			
			# Application metrics
			"crm_request_count": {
				"type": MetricType.COUNTER,
				"description": "Total number of requests",
				"unit": "count"
			},
			"crm_request_duration": {
				"type": MetricType.HISTOGRAM,
				"description": "Request duration in milliseconds",
				"unit": "milliseconds"
			},
			"crm_error_count": {
				"type": MetricType.COUNTER,
				"description": "Total number of errors",
				"unit": "count"
			},
			"crm_active_sessions": {
				"type": MetricType.GAUGE,
				"description": "Number of active user sessions",
				"unit": "count"
			},
			
			# Database metrics
			"crm_db_connections": {
				"type": MetricType.GAUGE,
				"description": "Number of database connections",
				"unit": "count"
			},
			"crm_db_query_duration": {
				"type": MetricType.HISTOGRAM,
				"description": "Database query duration",
				"unit": "milliseconds"
			},
			"crm_db_query_count": {
				"type": MetricType.COUNTER,
				"description": "Number of database queries",
				"unit": "count"
			},
			
			# Business metrics
			"crm_contacts_created": {
				"type": MetricType.COUNTER,
				"description": "Number of contacts created",
				"unit": "count"
			},
			"crm_leads_created": {
				"type": MetricType.COUNTER,
				"description": "Number of leads created",
				"unit": "count"
			},
			"crm_leads_converted": {
				"type": MetricType.COUNTER,
				"description": "Number of leads converted",
				"unit": "count"
			},
			"crm_opportunities_won": {
				"type": MetricType.COUNTER,
				"description": "Number of opportunities won",
				"unit": "count"
			},
			"crm_revenue_generated": {
				"type": MetricType.COUNTER,
				"description": "Total revenue generated",
				"unit": "currency"
			},
			
			# Performance metrics
			"crm_api_response_time": {
				"type": MetricType.HISTOGRAM,
				"description": "API response time",
				"unit": "milliseconds"
			},
			"crm_throughput": {
				"type": MetricType.GAUGE,
				"description": "Requests per second",
				"unit": "requests_per_second"
			}
		}
		
		# Register metrics with APG monitoring
		for metric_name, definition in self.metric_definitions.items():
			await self.apg_monitoring.register_metric(
				name=metric_name,
				metric_type=definition["type"].value,
				description=definition["description"],
				unit=definition.get("unit", "")
			)
	
	async def _setup_alert_rules(self):
		"""Setup alert rules for CRM monitoring"""
		self.alert_rules = {
			"high_cpu_usage": {
				"metric": "crm_cpu_usage",
				"condition": ">",
				"threshold": 80.0,
				"severity": AlertSeverity.HIGH,
				"description": "CPU usage is above 80%"
			},
			"high_memory_usage": {
				"metric": "crm_memory_usage", 
				"condition": ">",
				"threshold": 85.0,
				"severity": AlertSeverity.HIGH,
				"description": "Memory usage is above 85%"
			},
			"high_disk_usage": {
				"metric": "crm_disk_usage",
				"condition": ">",
				"threshold": 90.0,
				"severity": AlertSeverity.CRITICAL,
				"description": "Disk usage is above 90%"
			},
			"high_error_rate": {
				"metric": "crm_error_rate",
				"condition": ">",
				"threshold": 5.0,
				"severity": AlertSeverity.MEDIUM,
				"description": "Error rate is above 5%"
			},
			"slow_response_time": {
				"metric": "crm_api_response_time",
				"condition": ">",
				"threshold": 2000.0,
				"severity": AlertSeverity.MEDIUM,
				"description": "API response time is above 2 seconds"
			},
			"database_connection_pool_exhausted": {
				"metric": "crm_db_connections",
				"condition": ">",
				"threshold": 45.0,  # Assuming max 50 connections
				"severity": AlertSeverity.HIGH,
				"description": "Database connection pool near exhaustion"
			}
		}
	
	async def start_monitoring(self):
		"""Start monitoring background tasks"""
		if self._monitoring_running:
			logger.warning("Monitoring already running")
			return
		
		self._monitoring_running = True
		self._monitoring_task = asyncio.create_task(self._monitoring_loop())
		logger.info("🔄 Monitoring background tasks started")
	
	async def stop_monitoring(self):
		"""Stop monitoring background tasks"""
		self._monitoring_running = False
		
		if self._monitoring_task:
			self._monitoring_task.cancel()
			try:
				await self._monitoring_task
			except asyncio.CancelledError:
				pass
		
		logger.info("🛑 Monitoring background tasks stopped")
	
	async def _monitoring_loop(self):
		"""Main monitoring loop"""
		while self._monitoring_running:
			try:
				# Collect system metrics
				await self.collect_system_metrics()
				
				# Collect application metrics
				await self.collect_application_metrics()
				
				# Evaluate alert rules
				await self.evaluate_alerts()
				
				# Update business metrics cache
				await self.update_business_metrics_cache()
				
				# Wait before next collection
				await asyncio.sleep(30)  # Collect every 30 seconds
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Monitoring loop error: {str(e)}", exc_info=True)
				await asyncio.sleep(30)
	
	async def collect_system_metrics(self):
		"""Collect system performance metrics"""
		try:
			# CPU usage
			cpu_percent = psutil.cpu_percent(interval=1)
			await self.record_metric("crm_cpu_usage", cpu_percent, MetricType.GAUGE)
			
			# Memory usage
			memory = psutil.virtual_memory()
			memory_percent = memory.percent
			await self.record_metric("crm_memory_usage", memory_percent, MetricType.GAUGE)
			
			# Disk usage
			disk = psutil.disk_usage('/')
			disk_percent = (disk.used / disk.total) * 100
			await self.record_metric("crm_disk_usage", disk_percent, MetricType.GAUGE)
			
		except Exception as e:
			logger.error(f"Failed to collect system metrics: {str(e)}")
	
	async def collect_application_metrics(self):
		"""Collect application-specific metrics"""
		try:
			# Request metrics
			current_time = time.time()
			recent_requests = [
				req for req in self.request_metrics 
				if current_time - req['timestamp'] <= 60  # Last minute
			]
			
			request_count = len(recent_requests)
			await self.record_metric("crm_request_count", request_count, MetricType.COUNTER)
			
			# Calculate throughput (requests per second)
			throughput = request_count / 60.0
			await self.record_metric("crm_throughput", throughput, MetricType.GAUGE)
			
			# Average response time
			if recent_requests:
				avg_response_time = sum(req['duration'] for req in recent_requests) / len(recent_requests)
				await self.record_metric("crm_api_response_time", avg_response_time, MetricType.HISTOGRAM)
			
			# Error metrics
			recent_errors = [
				err for err in self.error_metrics
				if current_time - err['timestamp'] <= 60  # Last minute
			]
			
			error_count = len(recent_errors)
			await self.record_metric("crm_error_count", error_count, MetricType.COUNTER)
			
			# Error rate
			error_rate = (error_count / max(request_count, 1)) * 100
			await self.record_metric("crm_error_rate", error_rate, MetricType.GAUGE)
			
		except Exception as e:
			logger.error(f"Failed to collect application metrics: {str(e)}")
	
	async def record_metric(
		self, 
		name: str, 
		value: float, 
		metric_type: MetricType,
		labels: Optional[Dict[str, str]] = None,
		tenant_id: Optional[str] = None
	):
		"""Record a metric value"""
		try:
			labels = labels or {}
			if tenant_id:
				labels["tenant_id"] = tenant_id
			
			metric = Metric(
				name=name,
				type=metric_type,
				value=value,
				timestamp=datetime.utcnow(),
				labels=labels,
				description=self.metric_definitions.get(name, {}).get("description")
			)
			
			# Store locally
			self.metrics[name].append(metric)
			
			# Send to APG monitoring
			await self.apg_monitoring.record_metric(
				name=name,
				value=value,
				labels=labels,
				timestamp=metric.timestamp
			)
			
		except Exception as e:
			logger.error(f"Failed to record metric {name}: {str(e)}")
	
	async def evaluate_alerts(self):
		"""Evaluate alert rules against current metrics"""
		try:
			for rule_name, rule in self.alert_rules.items():
				metric_name = rule["metric"]
				condition = rule["condition"]
				threshold = rule["threshold"]
				severity = AlertSeverity(rule["severity"])
				
				# Get latest metric value
				if metric_name in self.metrics and self.metrics[metric_name]:
					latest_metric = self.metrics[metric_name][-1]
					current_value = latest_metric.value
					
					# Evaluate condition
					alert_triggered = False
					if condition == ">" and current_value > threshold:
						alert_triggered = True
					elif condition == "<" and current_value < threshold:
						alert_triggered = True
					elif condition == "==" and current_value == threshold:
						alert_triggered = True
					
					# Handle alert
					if alert_triggered:
						await self._trigger_alert(rule_name, rule, current_value)
					else:
						await self._resolve_alert(rule_name)
			
		except Exception as e:
			logger.error(f"Failed to evaluate alerts: {str(e)}")
	
	async def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], current_value: float):
		"""Trigger an alert"""
		try:
			# Check if alert already exists
			if rule_name in self.active_alerts and not self.active_alerts[rule_name].resolved:
				return  # Alert already active
			
			alert = Alert(
				alert_id=f"crm_{rule_name}_{int(time.time())}",
				title=f"CRM Alert: {rule['description']}",
				description=f"Metric {rule['metric']} is {current_value}, threshold is {rule['threshold']}",
				severity=AlertSeverity(rule["severity"]),
				metric_name=rule["metric"],
				current_value=current_value,
				threshold_value=rule["threshold"],
				timestamp=datetime.utcnow()
			)
			
			# Store alert
			self.active_alerts[rule_name] = alert
			
			# Send to APG alert manager
			await self.alert_manager.create_alert(
				alert_id=alert.alert_id,
				title=alert.title,
				description=alert.description,
				severity=alert.severity.value,
				labels={
					"service": "customer_relationship_management",
					"metric": alert.metric_name,
					"severity": alert.severity.value
				}
			)
			
			logger.warning(f"🚨 Alert triggered: {alert.title}")
			
		except Exception as e:
			logger.error(f"Failed to trigger alert: {str(e)}")
	
	async def _resolve_alert(self, rule_name: str):
		"""Resolve an active alert"""
		try:
			if rule_name in self.active_alerts and not self.active_alerts[rule_name].resolved:
				alert = self.active_alerts[rule_name]
				alert.resolved = True
				alert.resolved_at = datetime.utcnow()
				
				# Send resolution to APG alert manager
				await self.alert_manager.resolve_alert(alert.alert_id)
				
				logger.info(f"✅ Alert resolved: {alert.title}")
			
		except Exception as e:
			logger.error(f"Failed to resolve alert: {str(e)}")
	
	async def update_business_metrics_cache(self):
		"""Update business metrics cache"""
		try:
			now = datetime.utcnow()
			
			# Check if cache needs update
			if (self.last_cache_update and 
				now - self.last_cache_update < self.cache_expiry):
				return
			
			# This would query the CRM service for business metrics
			# For now, use placeholder data
			self.business_metrics_cache = {
				"total_contacts": 10000,
				"total_leads": 5000,
				"total_opportunities": 2000,
				"conversion_rate": 25.5,
				"average_deal_size": 15000.0,
				"monthly_revenue": 500000.0,
				"active_campaigns": 15,
				"pipeline_value": 2500000.0
			}
			
			self.last_cache_update = now
			
		except Exception as e:
			logger.error(f"Failed to update business metrics cache: {str(e)}")
	
	def track_request(self, endpoint: str, method: str, duration: float, status_code: int):
		"""Track API request metrics"""
		request_data = {
			"timestamp": time.time(),
			"endpoint": endpoint,
			"method": method,
			"duration": duration,
			"status_code": status_code
		}
		
		self.request_metrics.append(request_data)
		
		# Track errors
		if status_code >= 400:
			error_data = {
				"timestamp": time.time(),
				"endpoint": endpoint,
				"method": method,
				"status_code": status_code
			}
			self.error_metrics.append(error_data)
	
	def get_metric_summary(self, metric_name: str, time_range: timedelta = None) -> Dict[str, Any]:
		"""Get summary statistics for a metric"""
		try:
			if metric_name not in self.metrics:
				return {"error": "Metric not found"}
			
			time_range = time_range or timedelta(hours=1)
			cutoff_time = datetime.utcnow() - time_range
			
			# Filter metrics by time range
			recent_metrics = [
				m for m in self.metrics[metric_name]
				if m.timestamp >= cutoff_time
			]
			
			if not recent_metrics:
				return {"error": "No data in time range"}
			
			values = [m.value for m in recent_metrics]
			
			return {
				"metric_name": metric_name,
				"count": len(values),
				"min": min(values),
				"max": max(values),
				"average": sum(values) / len(values),
				"latest": values[-1],
				"time_range": str(time_range)
			}
			
		except Exception as e:
			logger.error(f"Failed to get metric summary: {str(e)}")
			return {"error": str(e)}
	
	def get_health_metrics(self) -> Dict[str, Any]:
		"""Get health metrics for CRM system"""
		try:
			# Get latest system metrics
			latest_metrics = {}
			for metric_name in ["crm_cpu_usage", "crm_memory_usage", "crm_disk_usage"]:
				if metric_name in self.metrics and self.metrics[metric_name]:
					latest_metrics[metric_name] = self.metrics[metric_name][-1].value
			
			# Calculate health score
			health_score = 100.0
			
			if "crm_cpu_usage" in latest_metrics:
				if latest_metrics["crm_cpu_usage"] > 80:
					health_score -= 20
				elif latest_metrics["crm_cpu_usage"] > 60:
					health_score -= 10
			
			if "crm_memory_usage" in latest_metrics:
				if latest_metrics["crm_memory_usage"] > 85:
					health_score -= 25
				elif latest_metrics["crm_memory_usage"] > 70:
					health_score -= 10
			
			if "crm_disk_usage" in latest_metrics:
				if latest_metrics["crm_disk_usage"] > 90:
					health_score -= 30
				elif latest_metrics["crm_disk_usage"] > 80:
					health_score -= 15
			
			# Determine health status
			if health_score >= 90:
				health_status = "excellent"
			elif health_score >= 70:
				health_status = "good"
			elif health_score >= 50:
				health_status = "fair"
			else:
				health_status = "poor"
			
			return {
				"health_score": max(0, health_score),
				"health_status": health_status,
				"system_metrics": latest_metrics,
				"active_alerts": len([a for a in self.active_alerts.values() if not a.resolved]),
				"business_metrics": self.business_metrics_cache,
				"timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Failed to get health metrics: {str(e)}")
			return {
				"health_score": 0,
				"health_status": "unknown",
				"error": str(e)
			}
	
	def get_performance_report(self) -> Dict[str, Any]:
		"""Get comprehensive performance report"""
		try:
			current_time = time.time()
			
			# Request performance
			recent_requests = [
				req for req in self.request_metrics
				if current_time - req['timestamp'] <= 3600  # Last hour
			]
			
			request_performance = {
				"total_requests": len(recent_requests),
				"requests_per_minute": len(recent_requests) / 60.0,
				"average_response_time": (
					sum(req['duration'] for req in recent_requests) / len(recent_requests)
					if recent_requests else 0
				),
				"error_rate": (
					len([req for req in recent_requests if req['status_code'] >= 400]) / 
					max(len(recent_requests), 1) * 100
				)
			}
			
			# System performance
			system_performance = {}
			for metric_name in ["crm_cpu_usage", "crm_memory_usage", "crm_disk_usage"]:
				if metric_name in self.metrics and self.metrics[metric_name]:
					recent = [
						m.value for m in self.metrics[metric_name]
						if (datetime.utcnow() - m.timestamp).total_seconds() <= 3600
					]
					if recent:
						system_performance[metric_name] = {
							"current": recent[-1],
							"average": sum(recent) / len(recent),
							"max": max(recent)
						}
			
			return {
				"report_generated": datetime.utcnow().isoformat(),
				"time_range": "1 hour",
				"request_performance": request_performance,
				"system_performance": system_performance,
				"active_alerts": [
					{
						"title": alert.title,
						"severity": alert.severity.value,
						"timestamp": alert.timestamp.isoformat()
					}
					for alert in self.active_alerts.values()
					if not alert.resolved
				]
			}
			
		except Exception as e:
			logger.error(f"Failed to generate performance report: {str(e)}")
			return {"error": str(e)}
	
	async def health_check(self) -> Dict[str, Any]:
		"""Health check for monitoring system"""
		return {
			"status": "healthy" if self._initialized else "unhealthy",
			"monitoring_active": self._monitoring_running,
			"metrics_collected": len(self.metrics),
			"active_alerts": len([a for a in self.active_alerts.values() if not a.resolved]),
			"timestamp": datetime.utcnow().isoformat()
		}
	
	async def shutdown(self):
		"""Shutdown monitoring system"""
		try:
			logger.info("🛑 Shutting down CRM monitoring...")
			
			await self.stop_monitoring()
			
			# Clear caches
			self.metrics.clear()
			self.active_alerts.clear()
			self.business_metrics_cache.clear()
			
			self._initialized = False
			logger.info("✅ CRM monitoring shutdown completed")
			
		except Exception as e:
			logger.error(f"Error during monitoring shutdown: {str(e)}", exc_info=True)


# Monitoring middleware for FastAPI
class MonitoringMiddleware:
	"""Middleware to automatically track request metrics"""
	
	def __init__(self, monitoring: CRMMonitoring):
		self.monitoring = monitoring
	
	async def __call__(self, request, call_next):
		start_time = time.time()
		
		try:
			response = await call_next(request)
			duration = (time.time() - start_time) * 1000  # Convert to milliseconds
			
			# Track request metrics
			self.monitoring.track_request(
				endpoint=str(request.url.path),
				method=request.method,
				duration=duration,
				status_code=response.status_code
			)
			
			return response
			
		except Exception as e:
			duration = (time.time() - start_time) * 1000
			
			# Track error metrics
			self.monitoring.track_request(
				endpoint=str(request.url.path),
				method=request.method,
				duration=duration,
				status_code=500
			)
			
			raise


# Export classes
__all__ = [
	"CRMMonitoring",
	"MonitoringMiddleware",
	"Metric",
	"Alert",
	"MetricType", 
	"AlertSeverity"
]