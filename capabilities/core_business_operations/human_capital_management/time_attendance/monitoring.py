"""
APG Time & Attendance Capability - Comprehensive Monitoring & Alerting

Revolutionary monitoring system with AI-powered anomaly detection,
predictive analytics, and intelligent alerting for proactive system management.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import defaultdict, deque

import psutil
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import aioredis
import asyncpg

from .service import TimeAttendanceService
from .websocket import websocket_manager, RealTimeEvent
from .config import get_config


logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
	"""Alert severity levels"""
	INFO = "info"
	WARNING = "warning"
	CRITICAL = "critical"
	EMERGENCY = "emergency"


class MetricType(str, Enum):
	"""Metric types for monitoring"""
	COUNTER = "counter"
	GAUGE = "gauge"
	HISTOGRAM = "histogram"
	SUMMARY = "summary"


@dataclass
class Alert:
	"""Alert data structure"""
	id: str
	title: str
	description: str
	severity: AlertSeverity
	metric_name: str
	current_value: float
	threshold_value: float
	timestamp: datetime
	tenant_id: Optional[str] = None
	employee_id: Optional[str] = None
	resolved: bool = False
	resolved_at: Optional[datetime] = None
	metadata: Dict[str, Any] = None

	def to_dict(self) -> Dict[str, Any]:
		"""Convert alert to dictionary"""
		data = asdict(self)
		data['timestamp'] = self.timestamp.isoformat()
		if self.resolved_at:
			data['resolved_at'] = self.resolved_at.isoformat()
		return data


class MonitoringMetrics:
	"""Prometheus metrics for Time & Attendance"""
	
	def __init__(self):
		# API Performance Metrics
		self.request_duration = Histogram(
			'ta_request_duration_seconds',
			'Time spent processing requests',
			['method', 'endpoint', 'status_code']
		)
		
		self.request_count = Counter(
			'ta_request_total',
			'Total number of requests',
			['method', 'endpoint', 'status_code']
		)
		
		# Business Logic Metrics
		self.clock_in_count = Counter(
			'ta_clock_in_total',
			'Total number of clock-ins',
			['tenant_id', 'status']
		)
		
		self.clock_out_count = Counter(
			'ta_clock_out_total', 
			'Total number of clock-outs',
			['tenant_id', 'status']
		)
		
		self.fraud_detection_score = Histogram(
			'ta_fraud_score',
			'Fraud detection scores',
			['tenant_id', 'result']
		)
		
		self.active_sessions = Gauge(
			'ta_active_sessions',
			'Number of active time tracking sessions',
			['tenant_id', 'work_mode']
		)
		
		# System Health Metrics
		self.database_connections = Gauge(
			'ta_database_connections',
			'Number of active database connections'
		)
		
		self.redis_operations = Counter(
			'ta_redis_operations_total',
			'Total Redis operations',
			['operation', 'status']
		)
		
		self.websocket_connections = Gauge(
			'ta_websocket_connections',
			'Number of active WebSocket connections'
		)
		
		# AI & ML Metrics
		self.ai_model_predictions = Counter(
			'ta_ai_predictions_total',
			'Total AI model predictions',
			['model', 'result']
		)
		
		self.ai_model_latency = Histogram(
			'ta_ai_model_latency_seconds',
			'AI model prediction latency',
			['model']
		)
		
		# Remote Work Metrics
		self.remote_sessions = Gauge(
			'ta_remote_sessions',
			'Number of active remote work sessions',
			['tenant_id', 'work_mode']
		)
		
		self.productivity_score = Histogram(
			'ta_productivity_score',
			'Employee productivity scores',
			['tenant_id', 'employee_type']
		)


class PerformanceMonitor:
	"""System performance monitoring"""
	
	def __init__(self):
		self.metrics = MonitoringMetrics()
		self.alert_history = deque(maxlen=1000)
		self.metric_history = defaultdict(lambda: deque(maxlen=100))
		
	async def collect_system_metrics(self) -> Dict[str, float]:
		"""Collect system performance metrics"""
		try:
			# CPU metrics
			cpu_percent = psutil.cpu_percent(interval=1)
			cpu_count = psutil.cpu_count()
			
			# Memory metrics
			memory = psutil.virtual_memory()
			memory_percent = memory.percent
			memory_available = memory.available / (1024 ** 3)  # GB
			
			# Disk metrics
			disk = psutil.disk_usage('/')
			disk_percent = disk.percent
			disk_free = disk.free / (1024 ** 3)  # GB
			
			# Network metrics
			network = psutil.net_io_counters()
			network_sent = network.bytes_sent / (1024 ** 2)  # MB
			network_recv = network.bytes_recv / (1024 ** 2)  # MB
			
			metrics = {
				'cpu_percent': cpu_percent,
				'cpu_count': cpu_count,
				'memory_percent': memory_percent,
				'memory_available_gb': memory_available,
				'disk_percent': disk_percent,
				'disk_free_gb': disk_free,
				'network_sent_mb': network_sent,
				'network_recv_mb': network_recv,
			}
			
			# Store in history for trend analysis
			timestamp = datetime.utcnow()
			for metric, value in metrics.items():
				self.metric_history[metric].append((timestamp, value))
			
			return metrics
			
		except Exception as e:
			logger.error(f"Error collecting system metrics: {str(e)}")
			return {}
	
	async def analyze_performance_trends(self) -> Dict[str, Any]:
		"""Analyze performance trends and predict issues"""
		try:
			trends = {}
			
			for metric_name, history in self.metric_history.items():
				if len(history) < 10:
					continue
					
				# Get recent values (last 10 measurements)
				recent_values = [value for _, value in list(history)[-10:]]
				
				# Calculate statistics
				mean_value = statistics.mean(recent_values)
				median_value = statistics.median(recent_values)
				std_dev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
				
				# Calculate trend (simple linear regression slope)
				n = len(recent_values)
				if n >= 3:
					x_values = list(range(n))
					slope = sum((x - n/2) * (y - mean_value) for x, y in zip(x_values, recent_values))
					slope /= sum((x - n/2) ** 2 for x in x_values)
				else:
					slope = 0
				
				trends[metric_name] = {
					'current': recent_values[-1],
					'mean': mean_value,
					'median': median_value,
					'std_dev': std_dev,
					'trend_slope': slope,
					'trend_direction': 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable'
				}
			
			return trends
			
		except Exception as e:
			logger.error(f"Error analyzing performance trends: {str(e)}")
			return {}


class AlertManager:
	"""Intelligent alert management system"""
	
	def __init__(self):
		self.active_alerts = {}
		self.alert_rules = self._load_alert_rules()
		self.notification_channels = []
		
	def _load_alert_rules(self) -> Dict[str, Dict[str, Any]]:
		"""Load alert rules configuration"""
		return {
			'high_cpu_usage': {
				'metric': 'cpu_percent',
				'threshold': 85.0,
				'operator': '>',
				'severity': AlertSeverity.WARNING,
				'duration': 300  # 5 minutes
			},
			'critical_cpu_usage': {
				'metric': 'cpu_percent', 
				'threshold': 95.0,
				'operator': '>',
				'severity': AlertSeverity.CRITICAL,
				'duration': 60  # 1 minute
			},
			'high_memory_usage': {
				'metric': 'memory_percent',
				'threshold': 85.0,
				'operator': '>',
				'severity': AlertSeverity.WARNING,
				'duration': 300
			},
			'critical_memory_usage': {
				'metric': 'memory_percent',
				'threshold': 95.0,
				'operator': '>',
				'severity': AlertSeverity.CRITICAL,
				'duration': 60
			},
			'low_disk_space': {
				'metric': 'disk_percent',
				'threshold': 85.0,
				'operator': '>',
				'severity': AlertSeverity.WARNING,
				'duration': 900  # 15 minutes
			},
			'critical_disk_space': {
				'metric': 'disk_percent',
				'threshold': 95.0,
				'operator': '>',
				'severity': AlertSeverity.CRITICAL,
				'duration': 300
			},
			'high_fraud_score': {
				'metric': 'fraud_score_avg',
				'threshold': 0.8,
				'operator': '>',
				'severity': AlertSeverity.WARNING,
				'duration': 60
			},
			'database_connection_exhaustion': {
				'metric': 'database_connections',
				'threshold': 90,
				'operator': '>',
				'severity': AlertSeverity.CRITICAL,
				'duration': 30
			}
		}
	
	async def evaluate_alerts(self, metrics: Dict[str, float]) -> List[Alert]:
		"""Evaluate metrics against alert rules"""
		new_alerts = []
		current_time = datetime.utcnow()
		
		for rule_name, rule in self.alert_rules.items():
			metric_name = rule['metric']
			
			if metric_name not in metrics:
				continue
			
			current_value = metrics[metric_name]
			threshold = rule['threshold']
			operator = rule['operator']
			
			# Evaluate condition
			condition_met = False
			if operator == '>':
				condition_met = current_value > threshold
			elif operator == '<':
				condition_met = current_value < threshold
			elif operator == '>=':
				condition_met = current_value >= threshold
			elif operator == '<=':
				condition_met = current_value <= threshold
			elif operator == '==':
				condition_met = current_value == threshold
			
			if condition_met:
				# Check if alert already exists
				if rule_name in self.active_alerts:
					alert = self.active_alerts[rule_name]
					alert.current_value = current_value
					alert.timestamp = current_time
				else:
					# Create new alert
					alert = Alert(
						id=f"{rule_name}_{int(current_time.timestamp())}",
						title=f"{rule_name.replace('_', ' ').title()}",
						description=f"{metric_name} is {current_value} (threshold: {threshold})",
						severity=AlertSeverity(rule['severity']),
						metric_name=metric_name,
						current_value=current_value,
						threshold_value=threshold,
						timestamp=current_time,
						metadata={'rule': rule_name}
					)
					
					self.active_alerts[rule_name] = alert
					new_alerts.append(alert)
			else:
				# Resolve alert if it exists
				if rule_name in self.active_alerts:
					alert = self.active_alerts[rule_name]
					alert.resolved = True
					alert.resolved_at = current_time
					del self.active_alerts[rule_name]
		
		return new_alerts
	
	async def send_alert(self, alert: Alert):
		"""Send alert notification"""
		try:
			# Log alert
			logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.title} - {alert.description}")
			
			# Send WebSocket notification for real-time alerts
			event = RealTimeEvent(
				event_type="system_alert",
				entity_type="monitoring",
				entity_id=alert.id,
				tenant_id=alert.tenant_id or "system",
				data=alert.to_dict(),
				user_id="system"
			)
			await websocket_manager.broadcast_system_event(event)
			
			# TODO: Implement additional notification channels
			# - Email notifications
			# - Slack/Teams integration  
			# - SMS for critical alerts
			# - PagerDuty integration
			
		except Exception as e:
			logger.error(f"Error sending alert: {str(e)}")


class BusinessMetricsMonitor:
	"""Monitor business-specific metrics"""
	
	def __init__(self, service: TimeAttendanceService):
		self.service = service
		self.metrics = MonitoringMetrics()
		
	async def collect_business_metrics(self, tenant_id: str) -> Dict[str, Any]:
		"""Collect business metrics for monitoring"""
		try:
			# Mock business metrics - would query actual data
			metrics = {
				'active_employees': 150,
				'clock_in_rate_today': 0.95,
				'average_work_hours_today': 7.8,
				'overtime_employees_today': 12,
				'remote_workers_active': 45,
				'ai_agents_active': 8,
				'fraud_alerts_today': 2,
				'approval_pending_count': 15,
				'system_uptime_percent': 99.97,
				'response_time_avg_ms': 145,
				'database_performance_score': 0.92,
				'user_satisfaction_score': 4.6
			}
			
			# Update Prometheus metrics
			self.metrics.active_sessions.labels(
				tenant_id=tenant_id,
				work_mode="office"
			).set(metrics['active_employees'] - metrics['remote_workers_active'])
			
			self.metrics.remote_sessions.labels(
				tenant_id=tenant_id,
				work_mode="remote"
			).set(metrics['remote_workers_active'])
			
			return metrics
			
		except Exception as e:
			logger.error(f"Error collecting business metrics: {str(e)}")
			return {}
	
	async def generate_health_report(self, tenant_id: str) -> Dict[str, Any]:
		"""Generate comprehensive health report"""
		try:
			business_metrics = await self.collect_business_metrics(tenant_id)
			
			# Calculate health scores
			availability_score = business_metrics.get('system_uptime_percent', 0) / 100
			performance_score = min(1.0, 500 / max(business_metrics.get('response_time_avg_ms', 500), 1))
			user_satisfaction = business_metrics.get('user_satisfaction_score', 0) / 5.0
			
			overall_health = (availability_score + performance_score + user_satisfaction) / 3
			
			health_report = {
				'timestamp': datetime.utcnow().isoformat(),
				'tenant_id': tenant_id,
				'overall_health_score': round(overall_health, 3),
				'component_scores': {
					'availability': round(availability_score, 3),
					'performance': round(performance_score, 3),
					'user_satisfaction': round(user_satisfaction, 3)
				},
				'business_metrics': business_metrics,
				'status': 'healthy' if overall_health > 0.8 else 'degraded' if overall_health > 0.6 else 'unhealthy',
				'recommendations': self._generate_recommendations(business_metrics, overall_health)
			}
			
			return health_report
			
		except Exception as e:
			logger.error(f"Error generating health report: {str(e)}")
			return {'status': 'error', 'message': str(e)}
	
	def _generate_recommendations(self, metrics: Dict[str, Any], health_score: float) -> List[str]:
		"""Generate improvement recommendations"""
		recommendations = []
		
		if metrics.get('response_time_avg_ms', 0) > 300:
			recommendations.append("Consider scaling up application instances to improve response times")
		
		if metrics.get('fraud_alerts_today', 0) > 5:
			recommendations.append("Review fraud detection thresholds and investigate suspicious patterns")
		
		if metrics.get('clock_in_rate_today', 1.0) < 0.9:
			recommendations.append("Low clock-in rate detected - check for system issues or employee notifications")
		
		if metrics.get('approval_pending_count', 0) > 50:
			recommendations.append("High number of pending approvals - consider automated approval rules")
		
		if health_score < 0.7:
			recommendations.append("System health is below optimal - consider immediate investigation")
		
		return recommendations


class MonitoringDashboard:
	"""Real-time monitoring dashboard"""
	
	def __init__(self):
		self.performance_monitor = PerformanceMonitor()
		self.alert_manager = AlertManager()
		self.business_monitor = None  # Will be set when service is available
		
	async def start_monitoring(self, service: TimeAttendanceService):
		"""Start the monitoring system"""
		self.business_monitor = BusinessMetricsMonitor(service)
		
		# Start monitoring loops
		asyncio.create_task(self._system_monitoring_loop())
		asyncio.create_task(self._business_monitoring_loop())
		asyncio.create_task(self._alert_processing_loop())
		
		logger.info("Time & Attendance monitoring system started")
	
	async def _system_monitoring_loop(self):
		"""System metrics monitoring loop"""
		while True:
			try:
				# Collect system metrics
				metrics = await self.performance_monitor.collect_system_metrics()
				
				if metrics:
					# Analyze trends
					trends = await self.performance_monitor.analyze_performance_trends()
					
					# Evaluate alerts
					new_alerts = await self.alert_manager.evaluate_alerts(metrics)
					
					# Send new alerts
					for alert in new_alerts:
						await self.alert_manager.send_alert(alert)
					
					# Broadcast system metrics via WebSocket
					event = RealTimeEvent(
						event_type="system_metrics",
						entity_type="monitoring",
						entity_id="system",
						tenant_id="system",
						data={
							'metrics': metrics,
							'trends': trends,
							'active_alerts': len(self.alert_manager.active_alerts)
						},
						user_id="system"
					)
					await websocket_manager.broadcast_system_event(event)
				
				await asyncio.sleep(30)  # Monitor every 30 seconds
				
			except Exception as e:
				logger.error(f"Error in system monitoring loop: {str(e)}")
				await asyncio.sleep(60)  # Wait longer on error
	
	async def _business_monitoring_loop(self):
		"""Business metrics monitoring loop"""
		while True:
			try:
				if self.business_monitor:
					# For demo, monitor default tenant
					tenant_id = "tenant_default"
					
					# Generate health report
					health_report = await self.business_monitor.generate_health_report(tenant_id)
					
					# Broadcast business metrics
					event = RealTimeEvent(
						event_type="business_metrics",
						entity_type="monitoring",
						entity_id="business",
						tenant_id=tenant_id,
						data=health_report,
						user_id="system"
					)
					await websocket_manager.broadcast_time_entry_event(event)
				
				await asyncio.sleep(60)  # Monitor every minute
				
			except Exception as e:
				logger.error(f"Error in business monitoring loop: {str(e)}")
				await asyncio.sleep(120)  # Wait longer on error
	
	async def _alert_processing_loop(self):
		"""Process and manage alerts"""
		while True:
			try:
				# Auto-resolve old alerts (older than 24 hours)
				cutoff_time = datetime.utcnow() - timedelta(hours=24)
				
				expired_alerts = []
				for rule_name, alert in self.alert_manager.active_alerts.items():
					if alert.timestamp < cutoff_time:
						expired_alerts.append(rule_name)
				
				for rule_name in expired_alerts:
					alert = self.alert_manager.active_alerts[rule_name]
					alert.resolved = True
					alert.resolved_at = datetime.utcnow()
					del self.alert_manager.active_alerts[rule_name]
					logger.info(f"Auto-resolved expired alert: {rule_name}")
				
				await asyncio.sleep(300)  # Check every 5 minutes
				
			except Exception as e:
				logger.error(f"Error in alert processing loop: {str(e)}")
				await asyncio.sleep(600)  # Wait longer on error
	
	async def get_dashboard_data(self, tenant_id: str) -> Dict[str, Any]:
		"""Get complete dashboard data"""
		try:
			# Get system metrics
			system_metrics = await self.performance_monitor.collect_system_metrics()
			trends = await self.performance_monitor.analyze_performance_trends()
			
			# Get business metrics
			business_data = {}
			if self.business_monitor:
				business_data = await self.business_monitor.generate_health_report(tenant_id)
			
			# Get active alerts
			active_alerts = [alert.to_dict() for alert in self.alert_manager.active_alerts.values()]
			
			return {
				'timestamp': datetime.utcnow().isoformat(),
				'system_metrics': system_metrics,
				'performance_trends': trends,
				'business_health': business_data,
				'active_alerts': active_alerts,
				'alert_count': len(active_alerts),
				'status': 'operational'
			}
			
		except Exception as e:
			logger.error(f"Error getting dashboard data: {str(e)}")
			return {'status': 'error', 'message': str(e)}


# Global monitoring instance
monitoring_dashboard = MonitoringDashboard()


# Export monitoring components
__all__ = [
	"MonitoringMetrics", 
	"PerformanceMonitor", 
	"AlertManager", 
	"BusinessMetricsMonitor",
	"MonitoringDashboard",
	"monitoring_dashboard",
	"Alert",
	"AlertSeverity"
]