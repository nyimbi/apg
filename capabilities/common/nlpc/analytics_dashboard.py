"""
APG NLP Analytics and Reporting Dashboard

Enterprise analytics platform with comprehensive reporting, real-time monitoring,
performance tracking, and business intelligence for NLP operations.

Features:
- Real-time performance dashboards with interactive visualizations
- Comprehensive usage analytics and cost reporting
- Model performance monitoring and comparison
- Annotation quality analytics and team productivity
- Business intelligence with trend analysis
- Automated reporting and alerting
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
from uuid_extensions import uuid7str

from models import NLPTaskType, ModelProvider, ProcessingStatus

# Configure logging
logger = logging.getLogger(__name__)

class ReportType(str, Enum):
	"""Analytics report types"""
	USAGE_SUMMARY = "usage_summary"
	PERFORMANCE_ANALYSIS = "performance_analysis"
	MODEL_COMPARISON = "model_comparison"
	ANNOTATION_QUALITY = "annotation_quality"
	COST_ANALYSIS = "cost_analysis"
	PRODUCTIVITY_REPORT = "productivity_report"
	TREND_ANALYSIS = "trend_analysis"
	CUSTOM_DASHBOARD = "custom_dashboard"

class MetricType(str, Enum):
	"""Analytics metric types"""
	COUNTER = "counter"
	GAUGE = "gauge"
	HISTOGRAM = "histogram"
	RATE = "rate"
	PERCENTAGE = "percentage"

class AlertSeverity(str, Enum):
	"""Alert severity levels"""
	INFO = "info"
	WARNING = "warning"
	ERROR = "error"
	CRITICAL = "critical"

class AlertCondition(str, Enum):
	"""Alert condition types"""
	GREATER_THAN = "greater_than"
	LESS_THAN = "less_than"
	EQUALS = "equals"
	NOT_EQUALS = "not_equals"
	CONTAINS = "contains"
	THRESHOLD_EXCEEDED = "threshold_exceeded"

@dataclass
class MetricPoint:
	"""Individual metric data point"""
	timestamp: datetime
	value: float
	labels: Dict[str, str] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TimeSeries:
	"""Time series data for metrics"""
	metric_name: str
	metric_type: MetricType
	data_points: List[MetricPoint] = field(default_factory=list)
	aggregation_window: timedelta = field(default_factory=lambda: timedelta(minutes=5))
	
	def add_point(self, value: float, labels: Dict[str, str] = None, metadata: Dict[str, Any] = None):
		"""Add data point to time series"""
		point = MetricPoint(
			timestamp=datetime.utcnow(),
			value=value,
			labels=labels or {},
			metadata=metadata or {}
		)
		self.data_points.append(point)
		
		# Keep only recent data (last 24 hours by default)
		cutoff = datetime.utcnow() - timedelta(hours=24)
		self.data_points = [p for p in self.data_points if p.timestamp > cutoff]
	
	def get_recent_values(self, duration: timedelta = None) -> List[float]:
		"""Get recent values within duration"""
		if not duration:
			duration = timedelta(hours=1)
		
		cutoff = datetime.utcnow() - duration
		return [p.value for p in self.data_points if p.timestamp > cutoff]
	
	def calculate_rate(self, duration: timedelta = None) -> float:
		"""Calculate rate of change"""
		recent_values = self.get_recent_values(duration or timedelta(minutes=5))
		if len(recent_values) < 2:
			return 0.0
		
		return (recent_values[-1] - recent_values[0]) / len(recent_values)

@dataclass
class Alert:
	"""Analytics alert"""
	alert_id: str = field(default_factory=uuid7str)
	alert_name: str = ""
	severity: AlertSeverity = AlertSeverity.INFO
	message: str = ""
	metric_name: str = ""
	threshold_value: float = 0.0
	current_value: float = 0.0
	condition: str = ""  # gt, lt, eq, etc.
	
	created_at: datetime = field(default_factory=datetime.utcnow)
	acknowledged_at: Optional[datetime] = None
	resolved_at: Optional[datetime] = None
	acknowledged_by: Optional[str] = None
	
	metadata: Dict[str, Any] = field(default_factory=dict)
	
	@property
	def is_active(self) -> bool:
		"""Check if alert is active"""
		return self.resolved_at is None
	
	@property
	def duration_minutes(self) -> float:
		"""Calculate alert duration in minutes"""
		end_time = self.resolved_at or datetime.utcnow()
		return (end_time - self.created_at).total_seconds() / 60

class AnalyticsDashboard:
	"""Enterprise analytics and reporting dashboard"""
	
	def __init__(self, tenant_id: str, config: Dict[str, Any] = None):
		assert tenant_id, "Tenant ID is required for analytics dashboard"
		
		self.tenant_id = tenant_id
		self.config = config or {}
		
		# Metrics storage
		self.metrics: Dict[str, TimeSeries] = {}
		self.alerts: Dict[str, Alert] = {}
		self.alert_rules: Dict[str, Dict[str, Any]] = {}
		
		# Reporting
		self.report_cache: Dict[str, Dict[str, Any]] = {}
		self.custom_dashboards: Dict[str, Dict[str, Any]] = {}
		
		# Real-time streaming
		self.metric_subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
		self.dashboard_subscribers: List[asyncio.Queue] = []
		
		self._setup_default_metrics()
		self._setup_default_alerts()
		self._log_dashboard_initialized()
	
	def _setup_default_metrics(self) -> None:
		"""Setup default metrics tracking"""
		default_metrics = [
			("processing_requests_total", MetricType.COUNTER),
			("processing_latency_ms", MetricType.HISTOGRAM),
			("model_accuracy", MetricType.GAUGE),
			("annotation_rate", MetricType.RATE),
			("error_rate", MetricType.PERCENTAGE),
			("concurrent_sessions", MetricType.GAUGE),
			("training_cost_usd", MetricType.COUNTER),
			("storage_usage_gb", MetricType.GAUGE),
			("api_calls_per_minute", MetricType.RATE),
			("user_satisfaction_score", MetricType.GAUGE)
		]
		
		for metric_name, metric_type in default_metrics:
			self.metrics[metric_name] = TimeSeries(
				metric_name=metric_name,
				metric_type=metric_type
			)
	
	def _setup_default_alerts(self) -> None:
		"""Setup default alert rules"""
		self.alert_rules = {
			"high_error_rate": {
				"metric": "error_rate",
				"condition": "gt",
				"threshold": 5.0,  # 5% error rate
				"severity": AlertSeverity.ERROR,
				"description": "Error rate is above 5%"
			},
			"slow_processing": {
				"metric": "processing_latency_ms",
				"condition": "gt",
				"threshold": 1000.0,  # 1 second
				"severity": AlertSeverity.WARNING,
				"description": "Processing latency is above 1 second"
			},
			"low_model_accuracy": {
				"metric": "model_accuracy",
				"condition": "lt",
				"threshold": 0.8,  # 80% accuracy
				"severity": AlertSeverity.WARNING,
				"description": "Model accuracy has dropped below 80%"
			},
			"high_cost_burn": {
				"metric": "training_cost_usd",
				"condition": "rate_gt",
				"threshold": 100.0,  # $100/hour
				"severity": AlertSeverity.ERROR,
				"description": "Training cost burn rate exceeds $100/hour"
			},
			"storage_capacity": {
				"metric": "storage_usage_gb",
				"condition": "gt",
				"threshold": 900.0,  # 900GB of 1TB
				"severity": AlertSeverity.WARNING,
				"description": "Storage usage is above 90%"
			}
		}
	
	def _log_dashboard_initialized(self) -> None:
		"""Log dashboard initialization"""
		logger.info(f"Analytics dashboard initialized for tenant: {self.tenant_id}")
		logger.info(f"Tracking {len(self.metrics)} metrics with {len(self.alert_rules)} alert rules")
	
	def record_metric(self, metric_name: str, value: float, 
					  labels: Dict[str, str] = None, metadata: Dict[str, Any] = None) -> None:
		"""Record metric value"""
		if metric_name not in self.metrics:
			self.metrics[metric_name] = TimeSeries(
				metric_name=metric_name,
				metric_type=MetricType.GAUGE
			)
		
		self.metrics[metric_name].add_point(value, labels, metadata)
		
		# Check alert rules
		asyncio.create_task(self._check_alerts(metric_name, value))
		
		# Broadcast to subscribers
		asyncio.create_task(self._broadcast_metric_update(metric_name, value, labels))
	
	async def _check_alerts(self, metric_name: str, current_value: float) -> None:
		"""Check if any alert rules are triggered"""
		for rule_name, rule_config in self.alert_rules.items():
			if rule_config["metric"] != metric_name:
				continue
			
			condition = rule_config["condition"]
			threshold = rule_config["threshold"]
			triggered = False
			
			if condition == "gt" and current_value > threshold:
				triggered = True
			elif condition == "lt" and current_value < threshold:
				triggered = True
			elif condition == "eq" and abs(current_value - threshold) < 0.001:
				triggered = True
			elif condition == "rate_gt":
				rate = self.metrics[metric_name].calculate_rate()
				if rate > threshold:
					triggered = True
			
			if triggered:
				await self._create_alert(rule_name, rule_config, current_value)
	
	async def _create_alert(self, rule_name: str, rule_config: Dict[str, Any], current_value: float) -> None:
		"""Create new alert"""
		# Check if similar alert already exists
		existing_alerts = [
			alert for alert in self.alerts.values()
			if alert.alert_name == rule_name and alert.is_active
		]
		
		if existing_alerts:
			return  # Don't duplicate active alerts
		
		alert = Alert(
			alert_name=rule_name,
			severity=AlertSeverity(rule_config["severity"]),
			message=rule_config["description"],
			metric_name=rule_config["metric"],
			threshold_value=rule_config["threshold"],
			current_value=current_value,
			condition=rule_config["condition"],
			metadata={"rule_config": rule_config}
		)
		
		self.alerts[alert.alert_id] = alert
		
		# Broadcast alert
		await self._broadcast_alert(alert)
		
		self._log_alert_created(alert.alert_id, rule_name, alert.severity)
	
	def _log_alert_created(self, alert_id: str, rule_name: str, severity: AlertSeverity) -> None:
		"""Log alert creation"""
		logger.warning(f"Alert created: {rule_name} ({severity}) - {alert_id}")
	
	async def _broadcast_metric_update(self, metric_name: str, value: float, labels: Dict[str, str]) -> None:
		"""Broadcast metric update to subscribers"""
		update_message = {
			"type": "metric_update",
			"metric_name": metric_name,
			"value": value,
			"labels": labels or {},
			"timestamp": datetime.utcnow().isoformat()
		}
		
		# Send to metric-specific subscribers
		for queue in self.metric_subscribers[metric_name]:
			try:
				queue.put_nowait(update_message)
			except asyncio.QueueFull:
				pass  # Skip if queue is full
		
		# Send to dashboard subscribers
		for queue in self.dashboard_subscribers:
			try:
				queue.put_nowait(update_message)
			except asyncio.QueueFull:
				pass
	
	async def _broadcast_alert(self, alert: Alert) -> None:
		"""Broadcast alert to subscribers"""
		alert_message = {
			"type": "alert",
			"alert_id": alert.alert_id,
			"alert_name": alert.alert_name,
			"severity": alert.severity.value,
			"message": alert.message,
			"metric_name": alert.metric_name,
			"current_value": alert.current_value,
			"threshold_value": alert.threshold_value,
			"timestamp": alert.created_at.isoformat()
		}
		
		# Broadcast to all dashboard subscribers
		for queue in self.dashboard_subscribers:
			try:
				queue.put_nowait(alert_message)
			except asyncio.QueueFull:
				pass
	
	def get_real_time_metrics(self) -> Dict[str, Any]:
		"""Get current real-time metrics"""
		current_metrics = {}
		
		for metric_name, time_series in self.metrics.items():
			recent_values = time_series.get_recent_values(timedelta(minutes=5))
			
			if recent_values:
				current_value = recent_values[-1]
				
				if time_series.metric_type == MetricType.HISTOGRAM:
					current_metrics[metric_name] = {
						"current": current_value,
						"average": statistics.mean(recent_values),
						"p50": statistics.median(recent_values),
						"p95": statistics.quantiles(recent_values, n=20)[18] if len(recent_values) > 10 else current_value,
						"min": min(recent_values),
						"max": max(recent_values)
					}
				elif time_series.metric_type == MetricType.RATE:
					current_metrics[metric_name] = {
						"current_rate": time_series.calculate_rate(),
						"total": sum(recent_values)
					}
				else:
					current_metrics[metric_name] = {
						"current": current_value,
						"trend": "up" if len(recent_values) > 1 and current_value > recent_values[0] else "down"
					}
		
		# Active alerts summary
		active_alerts = [alert for alert in self.alerts.values() if alert.is_active]
		alert_summary = {
			"total_active": len(active_alerts),
			"by_severity": {
				severity.value: len([a for a in active_alerts if a.severity == severity])
				for severity in AlertSeverity
			}
		}
		
		return {
			"tenant_id": self.tenant_id,
			"timestamp": datetime.utcnow().isoformat(),
			"metrics": current_metrics,
			"alerts": alert_summary,
			"system_health": self._calculate_system_health()
		}
	
	def _calculate_system_health(self) -> Dict[str, Any]:
		"""Calculate overall system health score"""
		health_factors = {}
		
		# Error rate factor
		error_rate = self._get_recent_metric_value("error_rate")
		if error_rate is not None:
			health_factors["error_rate"] = max(0, 1.0 - (error_rate / 10.0))  # 0% error = 1.0, 10% error = 0.0
		
		# Latency factor
		latency = self._get_recent_metric_value("processing_latency_ms")
		if latency is not None:
			health_factors["latency"] = max(0, 1.0 - (latency / 2000.0))  # 0ms = 1.0, 2000ms = 0.0
		
		# Model accuracy factor
		accuracy = self._get_recent_metric_value("model_accuracy")
		if accuracy is not None:
			health_factors["accuracy"] = accuracy  # Direct accuracy score
		
		# Alert factor
		active_critical_alerts = len([a for a in self.alerts.values() 
									  if a.is_active and a.severity == AlertSeverity.CRITICAL])
		health_factors["alerts"] = max(0, 1.0 - (active_critical_alerts * 0.3))
		
		# Calculate overall health
		if health_factors:
			overall_health = statistics.mean(health_factors.values())
			health_status = "excellent" if overall_health >= 0.9 else \
						   "good" if overall_health >= 0.7 else \
						   "fair" if overall_health >= 0.5 else "poor"
		else:
			overall_health = 1.0
			health_status = "unknown"
		
		return {
			"overall_score": round(overall_health, 3),
			"status": health_status,
			"factors": {k: round(v, 3) for k, v in health_factors.items()},
			"active_alerts": len([a for a in self.alerts.values() if a.is_active])
		}
	
	def _get_recent_metric_value(self, metric_name: str) -> Optional[float]:
		"""Get most recent value for a metric"""
		if metric_name not in self.metrics:
			return None
		
		recent_values = self.metrics[metric_name].get_recent_values(timedelta(minutes=5))
		return recent_values[-1] if recent_values else None
	
	async def generate_report(self, report_type: ReportType, 
							  time_range: timedelta = None,
							  filters: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Generate comprehensive analytics report"""
		time_range = time_range or timedelta(days=7)
		filters = filters or {}
		
		report_data = {
			"report_type": report_type.value,
			"tenant_id": self.tenant_id,
			"generated_at": datetime.utcnow().isoformat(),
			"time_range": {
				"start": (datetime.utcnow() - time_range).isoformat(),
				"end": datetime.utcnow().isoformat(),
				"duration_hours": time_range.total_seconds() / 3600
			},
			"filters": filters
		}
		
		if report_type == ReportType.USAGE_SUMMARY:
			report_data["data"] = await self._generate_usage_summary(time_range, filters)
		elif report_type == ReportType.PERFORMANCE_ANALYSIS:
			report_data["data"] = await self._generate_performance_analysis(time_range, filters)
		elif report_type == ReportType.MODEL_COMPARISON:
			report_data["data"] = await self._generate_model_comparison(time_range, filters)
		elif report_type == ReportType.ANNOTATION_QUALITY:
			report_data["data"] = await self._generate_annotation_quality_report(time_range, filters)
		elif report_type == ReportType.COST_ANALYSIS:
			report_data["data"] = await self._generate_cost_analysis(time_range, filters)
		elif report_type == ReportType.PRODUCTIVITY_REPORT:
			report_data["data"] = await self._generate_productivity_report(time_range, filters)
		elif report_type == ReportType.TREND_ANALYSIS:
			report_data["data"] = await self._generate_trend_analysis(time_range, filters)
		else:
			report_data["data"] = {"error": f"Report type {report_type} not implemented"}
		
		# Cache report
		cache_key = f"{report_type.value}_{hash(str(filters))}"
		self.report_cache[cache_key] = report_data
		
		return report_data
	
	async def _generate_usage_summary(self, time_range: timedelta, filters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate usage summary report"""
		cutoff = datetime.utcnow() - time_range
		
		# Collect usage metrics
		total_requests = self._sum_metric_values("processing_requests_total", cutoff)
		total_api_calls = self._sum_metric_values("api_calls_per_minute", cutoff) * time_range.total_seconds() / 60
		average_latency = self._average_metric_values("processing_latency_ms", cutoff)
		
		# Task type distribution (mock)
		task_distribution = {
			"sentiment_analysis": 45.2,
			"named_entity_recognition": 25.8,
			"text_classification": 18.5,
			"language_detection": 10.5
		}
		
		# User activity (mock)
		unique_users = 150 + int(hash(str(cutoff)) % 50)
		active_sessions = self._average_metric_values("concurrent_sessions", cutoff)
		
		return {
			"summary": {
				"total_requests": int(total_requests),
				"total_api_calls": int(total_api_calls),
				"unique_users": unique_users,
				"average_latency_ms": round(average_latency, 2),
				"average_concurrent_sessions": round(active_sessions, 1)
			},
			"task_type_distribution": task_distribution,
			"usage_trends": {
				"daily_requests": self._get_daily_trend("processing_requests_total", time_range),
				"hourly_pattern": self._get_hourly_pattern("processing_requests_total", time_range)
			},
			"top_features": [
				{"feature": "Sentiment Analysis", "usage_percent": 45.2, "growth": 15.3},
				{"feature": "Entity Recognition", "usage_percent": 25.8, "growth": 8.7},
				{"feature": "Text Classification", "usage_percent": 18.5, "growth": -2.1},
				{"feature": "Language Detection", "usage_percent": 10.5, "growth": 12.8}
			]
		}
	
	async def _generate_performance_analysis(self, time_range: timedelta, filters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate performance analysis report"""
		cutoff = datetime.utcnow() - time_range
		
		# Performance metrics
		avg_latency = self._average_metric_values("processing_latency_ms", cutoff)
		error_rate = self._average_metric_values("error_rate", cutoff)
		accuracy = self._average_metric_values("model_accuracy", cutoff)
		
		# SLA compliance (mock)
		sla_targets = {
			"latency_p95_ms": 500,
			"availability_percent": 99.9,
			"error_rate_percent": 1.0
		}
		
		current_p95_latency = avg_latency * 1.5  # Approximate P95
		availability = 100 - error_rate
		
		sla_compliance = {
			"latency_p95": {
				"target": sla_targets["latency_p95_ms"],
				"actual": round(current_p95_latency, 2),
				"compliant": current_p95_latency <= sla_targets["latency_p95_ms"]
			},
			"availability": {
				"target": sla_targets["availability_percent"],
				"actual": round(availability, 3),
				"compliant": availability >= sla_targets["availability_percent"]
			},
			"error_rate": {
				"target": sla_targets["error_rate_percent"],
				"actual": round(error_rate, 3),
				"compliant": error_rate <= sla_targets["error_rate_percent"]
			}
		}
		
		return {
			"performance_summary": {
				"average_latency_ms": round(avg_latency, 2),
				"error_rate_percent": round(error_rate, 3),
				"model_accuracy": round(accuracy, 3),
				"p95_latency_ms": round(current_p95_latency, 2),
				"availability_percent": round(availability, 3)
			},
			"sla_compliance": sla_compliance,
			"performance_trends": {
				"latency_trend": self._get_daily_trend("processing_latency_ms", time_range),
				"error_rate_trend": self._get_daily_trend("error_rate", time_range),
				"accuracy_trend": self._get_daily_trend("model_accuracy", time_range)
			},
			"bottlenecks": [
				{"component": "Model Inference", "impact": "medium", "recommendation": "Consider model optimization"},
				{"component": "Data Preprocessing", "impact": "low", "recommendation": "Implement caching"},
				{"component": "Network Latency", "impact": "medium", "recommendation": "Deploy edge nodes"}
			]
		}
	
	async def _generate_model_comparison(self, time_range: timedelta, filters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate model comparison report"""
		# Mock model performance data
		models = [
			{
				"model_id": "bert-base-sentiment",
				"model_name": "BERT Base Sentiment",
				"provider": "transformers",
				"accuracy": 0.89,
				"latency_ms": 145,
				"throughput_rps": 25,
				"memory_mb": 512,
				"cost_per_1k": 0.08,
				"usage_percent": 45.2
			},
			{
				"model_id": "distilbert-sentiment", 
				"model_name": "DistilBERT Sentiment",
				"provider": "transformers",
				"accuracy": 0.85,
				"latency_ms": 89,
				"throughput_rps": 42,
				"memory_mb": 256,
				"cost_per_1k": 0.05,
				"usage_percent": 32.8
			},
			{
				"model_id": "roberta-large-sentiment",
				"model_name": "RoBERTa Large Sentiment", 
				"provider": "transformers",
				"accuracy": 0.92,
				"latency_ms": 289,
				"throughput_rps": 12,
				"memory_mb": 1024,
				"cost_per_1k": 0.15,
				"usage_percent": 22.0
			}
		]
		
		# Performance comparison matrix
		comparison_metrics = ["accuracy", "latency_ms", "throughput_rps", "memory_mb", "cost_per_1k"]
		
		best_performers = {}
		for metric in comparison_metrics:
			if metric in ["accuracy", "throughput_rps"]:
				# Higher is better
				best_performers[metric] = max(models, key=lambda m: m[metric])["model_name"]
			else:
				# Lower is better
				best_performers[metric] = min(models, key=lambda m: m[metric])["model_name"]
		
		return {
			"models": models,
			"comparison_matrix": {
				"metrics": comparison_metrics,
				"best_performers": best_performers
			},
			"recommendations": [
				{
					"model": "distilbert-sentiment",
					"use_case": "High-throughput, cost-sensitive applications",
					"reasoning": "Best balance of speed and accuracy"
				},
				{
					"model": "roberta-large-sentiment", 
					"use_case": "High-accuracy requirements",
					"reasoning": "Highest accuracy despite higher latency"
				},
				{
					"model": "bert-base-sentiment",
					"use_case": "General purpose applications",
					"reasoning": "Good balance across all metrics"
				}
			],
			"optimization_opportunities": [
				"Consider model quantization for RoBERTa to reduce memory usage",
				"Implement model ensembling for critical accuracy requirements",
				"Use DistilBERT for batch processing workloads"
			]
		}
	
	async def _generate_annotation_quality_report(self, time_range: timedelta, filters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate annotation quality report"""
		# Mock annotation quality data
		annotators = [
			{
				"annotator_id": "annotator_001",
				"name": "Alice Johnson",
				"role": "senior_annotator",
				"annotations_completed": 1250,
				"approval_rate": 94.5,
				"inter_annotator_agreement": 0.87,
				"average_time_per_annotation": 45.2,
				"quality_score": 0.91
			},
			{
				"annotator_id": "annotator_002", 
				"name": "Bob Smith",
				"role": "annotator",
				"annotations_completed": 980,
				"approval_rate": 89.2,
				"inter_annotator_agreement": 0.82,
				"average_time_per_annotation": 52.8,
				"quality_score": 0.86
			},
			{
				"annotator_id": "annotator_003",
				"name": "Carol Davis",
				"role": "annotator", 
				"annotations_completed": 1100,
				"approval_rate": 91.8,
				"inter_annotator_agreement": 0.85,
				"average_time_per_annotation": 48.1,
				"quality_score": 0.89
			}
		]
		
		# Project statistics
		projects = [
			{
				"project_id": "sentiment_project_001",
				"project_name": "Customer Feedback Sentiment",
				"total_annotations": 5500,
				"completed_annotations": 4200,
				"progress_percent": 76.4,
				"average_quality_score": 0.88,
				"consensus_rate": 0.84,
				"estimated_completion_days": 12
			},
			{
				"project_id": "ner_project_001",
				"project_name": "Legal Document NER",
				"total_annotations": 3200,
				"completed_annotations": 2850,
				"progress_percent": 89.1,
				"average_quality_score": 0.92,
				"consensus_rate": 0.89,
				"estimated_completion_days": 4
			}
		]
		
		return {
			"summary": {
				"total_annotators": len(annotators),
				"total_annotations_completed": sum(a["annotations_completed"] for a in annotators),
				"average_approval_rate": round(statistics.mean(a["approval_rate"] for a in annotators), 2),
				"average_inter_annotator_agreement": round(statistics.mean(a["inter_annotator_agreement"] for a in annotators), 3),
				"average_quality_score": round(statistics.mean(a["quality_score"] for a in annotators), 3)
			},
			"annotator_performance": annotators,
			"project_progress": projects,
			"quality_trends": {
				"approval_rate_trend": [89.5, 90.2, 91.1, 91.8, 92.1, 91.9, 92.3],
				"agreement_trend": [0.83, 0.84, 0.85, 0.86, 0.85, 0.87, 0.86],
				"productivity_trend": [42.1, 44.5, 46.2, 47.8, 48.1, 49.2, 48.7]
			},
			"quality_issues": [
				{
					"issue_type": "Low Inter-annotator Agreement",
					"affected_projects": ["sentiment_project_001"],
					"severity": "medium",
					"recommendation": "Provide additional training on edge cases"
				},
				{
					"issue_type": "Slow Annotation Speed",
					"affected_annotators": ["annotator_002"],
					"severity": "low", 
					"recommendation": "Consider workflow optimization"
				}
			]
		}
	
	async def _generate_cost_analysis(self, time_range: timedelta, filters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate cost analysis report"""
		# Mock cost data
		total_cost = self._sum_metric_values("training_cost_usd", datetime.utcnow() - time_range)
		
		cost_breakdown = {
			"compute_costs": {
				"training": 1250.30,
				"inference": 485.75,
				"preprocessing": 125.50
			},
			"storage_costs": {
				"model_storage": 45.20,
				"data_storage": 78.90,
				"backup_storage": 23.15
			},
			"api_costs": {
				"external_models": 234.80,
				"third_party_apis": 89.45
			}
		}
		
		total_computed = sum(
			sum(category.values()) for category in cost_breakdown.values()
		)
		
		# Cost efficiency metrics
		cost_per_request = total_computed / max(self._sum_metric_values("processing_requests_total", datetime.utcnow() - time_range), 1)
		cost_per_user = total_computed / 150  # Mock user count
		
		return {
			"summary": {
				"total_cost_usd": round(total_computed, 2),
				"cost_per_request": round(cost_per_request, 4),
				"cost_per_user": round(cost_per_user, 2),
				"budget_utilization_percent": 65.8,
				"projected_monthly_cost": round(total_computed * 30 / time_range.days, 2)
			},
			"cost_breakdown": cost_breakdown,
			"cost_trends": {
				"daily_costs": [45.2, 52.8, 48.1, 59.3, 67.2, 54.9, 62.1],
				"cost_categories": {
					"compute": 68.2,
					"storage": 18.5,
					"apis": 13.3
				}
			},
			"optimization_recommendations": [
				{
					"area": "Model Efficiency",
					"potential_savings": 280.50,
					"recommendation": "Implement model quantization and pruning"
				},
				{
					"area": "Storage Optimization",
					"potential_savings": 45.20,
					"recommendation": "Archive old training data and optimize retention policies"
				},
				{
					"area": "API Usage",
					"potential_savings": 78.90,
					"recommendation": "Cache frequently used API responses"
				}
			],
			"budget_alerts": [
				{
					"category": "training_costs",
					"current_spend": 1250.30,
					"budget": 1500.00,
					"utilization": 83.4,
					"status": "warning"
				}
			]
		}
	
	async def _generate_productivity_report(self, time_range: timedelta, filters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate team productivity report"""
		# Mock productivity data
		team_stats = {
			"total_team_members": 15,
			"active_members": 12,
			"total_tasks_completed": 1250,
			"average_task_completion_time": 3.2,  # hours
			"team_velocity": 85.3,  # tasks per week
			"quality_score": 0.89
		}
		
		individual_performance = [
			{
				"member_id": "user_001",
				"name": "Alice Johnson",
				"role": "Senior NLP Engineer",
				"tasks_completed": 145,
				"avg_completion_time": 2.8,
				"quality_score": 0.93,
				"productivity_score": 0.91
			},
			{
				"member_id": "user_002",
				"name": "Bob Smith", 
				"role": "Data Scientist",
				"tasks_completed": 120,
				"avg_completion_time": 3.5,
				"quality_score": 0.87,
				"productivity_score": 0.85
			},
			{
				"member_id": "user_003",
				"name": "Carol Davis",
				"role": "ML Engineer",
				"tasks_completed": 132,
				"avg_completion_time": 3.1,
				"quality_score": 0.91,
				"productivity_score": 0.89
			}
		]
		
		return {
			"team_summary": team_stats,
			"individual_performance": individual_performance,
			"productivity_trends": {
				"weekly_velocity": [78.2, 82.1, 85.3, 88.7, 85.3, 89.1, 87.4],
				"quality_trend": [0.86, 0.87, 0.89, 0.88, 0.89, 0.91, 0.89],
				"completion_time_trend": [3.8, 3.6, 3.4, 3.2, 3.1, 3.2, 3.2]
			},
			"team_insights": [
				{
					"insight": "Quality scores have improved 5.8% over the reporting period",
					"impact": "positive",
					"recommendation": "Continue current quality practices"
				},
				{
					"insight": "Task completion times have stabilized around 3.2 hours",
					"impact": "neutral", 
					"recommendation": "Look for automation opportunities"
				},
				{
					"insight": "Team velocity is trending upward",
					"impact": "positive",
					"recommendation": "Consider increasing sprint capacity"
				}
			]
		}
	
	async def _generate_trend_analysis(self, time_range: timedelta, filters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate trend analysis report"""
		# Mock trend data with growth calculations
		metrics_trends = {
			"usage_growth": {
				"metric": "Total API Calls",
				"current_period": 125000,
				"previous_period": 98000,
				"growth_rate": 27.6,
				"trend": "increasing",
				"forecast_next_period": 142000
			},
			"performance_improvement": {
				"metric": "Average Latency (ms)",
				"current_period": 145.2,
				"previous_period": 167.8,
				"improvement_rate": 13.5,
				"trend": "improving",
				"forecast_next_period": 138.0
			},
			"accuracy_trend": {
				"metric": "Model Accuracy",
				"current_period": 0.891,
				"previous_period": 0.873,
				"improvement_rate": 2.1,
				"trend": "improving",
				"forecast_next_period": 0.898
			},
			"cost_efficiency": {
				"metric": "Cost per Request (USD)",
				"current_period": 0.0048,
				"previous_period": 0.0052,
				"improvement_rate": 7.7,
				"trend": "improving",
				"forecast_next_period": 0.0045
			}
		}
		
		# Seasonal patterns
		seasonal_patterns = {
			"weekly_pattern": {
				"description": "Higher usage on weekdays, 40% drop on weekends",
				"peak_day": "Wednesday",
				"low_day": "Sunday"
			},
			"daily_pattern": {
				"description": "Peak usage between 9-11 AM and 2-4 PM",
				"peak_hour": "10 AM",
				"low_hour": "3 AM"
			},
			"monthly_pattern": {
				"description": "Consistent growth with seasonal dips in December",
				"growth_pattern": "linear_with_seasonal_variance"
			}
		}
		
		return {
			"summary": {
				"analysis_period": time_range.days,
				"trends_identified": len(metrics_trends),
				"overall_trajectory": "positive",
				"confidence_level": 0.87
			},
			"key_trends": metrics_trends,
			"seasonal_patterns": seasonal_patterns,
			"forecasts": {
				"short_term": {
					"horizon": "next_30_days",
					"confidence": 0.85,
					"predictions": {
						"api_calls": 142000,
						"avg_latency_ms": 138.0,
						"model_accuracy": 0.898,
						"estimated_cost": 2890.50
					}
				},
				"long_term": {
					"horizon": "next_quarter",
					"confidence": 0.72,
					"predictions": {
						"api_calls": 485000,
						"avg_latency_ms": 125.0,
						"model_accuracy": 0.915,
						"estimated_cost": 9250.75
					}
				}
			},
			"anomalies_detected": [
				{
					"date": "2024-02-15",
					"metric": "error_rate",
					"anomaly_type": "spike",
					"severity": "medium",
					"description": "Error rate increased by 300% for 4 hours"
				}
			],
			"recommendations": [
				"Continue current performance optimization efforts",
				"Prepare for projected 27% growth in API usage",
				"Monitor cost efficiency as scale increases",
				"Investigate error rate spike on 2024-02-15"
			]
		}
	
	def _sum_metric_values(self, metric_name: str, since: datetime) -> float:
		"""Sum metric values since given time"""
		if metric_name not in self.metrics:
			return 0.0
		
		values = [p.value for p in self.metrics[metric_name].data_points if p.timestamp >= since]
		return sum(values)
	
	def _average_metric_values(self, metric_name: str, since: datetime) -> float:
		"""Average metric values since given time"""
		if metric_name not in self.metrics:
			return 0.0
		
		values = [p.value for p in self.metrics[metric_name].data_points if p.timestamp >= since]
		return statistics.mean(values) if values else 0.0
	
	def _get_daily_trend(self, metric_name: str, time_range: timedelta) -> List[float]:
		"""Get daily trend data for metric"""
		# Mock daily trend data
		days = min(int(time_range.days), 30)
		return [50 + i * 2.5 + (hash(f"{metric_name}_{i}") % 20) for i in range(days)]
	
	def _get_hourly_pattern(self, metric_name: str, time_range: timedelta) -> List[float]:
		"""Get hourly pattern data for metric"""
		# Mock hourly pattern (24 hours)
		base_pattern = [20, 15, 10, 8, 12, 25, 45, 65, 85, 95, 100, 95, 
						90, 85, 95, 100, 90, 75, 60, 45, 35, 30, 25, 20]
		return [val + (hash(f"{metric_name}_{i}") % 10) for i, val in enumerate(base_pattern)]
	
	async def subscribe_to_metrics(self, metric_names: List[str] = None) -> asyncio.Queue:
		"""Subscribe to real-time metric updates"""
		subscriber_queue = asyncio.Queue(maxsize=1000)
		
		if metric_names:
			for metric_name in metric_names:
				self.metric_subscribers[metric_name].append(subscriber_queue)
		else:
			# Subscribe to all metrics
			self.dashboard_subscribers.append(subscriber_queue)
		
		return subscriber_queue
	
	async def unsubscribe_from_metrics(self, subscriber_queue: asyncio.Queue) -> None:
		"""Unsubscribe from metric updates"""
		# Remove from all subscription lists
		for queues in self.metric_subscribers.values():
			if subscriber_queue in queues:
				queues.remove(subscriber_queue)
		
		if subscriber_queue in self.dashboard_subscribers:
			self.dashboard_subscribers.remove(subscriber_queue)
	
	def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
		"""Acknowledge an alert"""
		if alert_id not in self.alerts:
			return False
		
		alert = self.alerts[alert_id]
		alert.acknowledged_at = datetime.utcnow()
		alert.acknowledged_by = user_id
		
		logger.info(f"Alert acknowledged: {alert_id} by {user_id}")
		return True
	
	def resolve_alert(self, alert_id: str, user_id: str) -> bool:
		"""Resolve an alert"""
		if alert_id not in self.alerts:
			return False
		
		alert = self.alerts[alert_id]
		alert.resolved_at = datetime.utcnow()
		
		logger.info(f"Alert resolved: {alert_id} by {user_id}")
		return True
	
	def get_active_alerts(self) -> List[Dict[str, Any]]:
		"""Get all active alerts"""
		active_alerts = [alert for alert in self.alerts.values() if alert.is_active]
		active_alerts.sort(key=lambda x: (x.severity.value, x.created_at), reverse=True)
		
		return [
			{
				"alert_id": alert.alert_id,
				"alert_name": alert.alert_name,
				"severity": alert.severity.value,
				"message": alert.message,
				"metric_name": alert.metric_name,
				"current_value": alert.current_value,
				"threshold_value": alert.threshold_value,
				"duration_minutes": alert.duration_minutes,
				"acknowledged": alert.acknowledged_at is not None,
				"acknowledged_by": alert.acknowledged_by,
				"created_at": alert.created_at.isoformat()
			}
			for alert in active_alerts
		]
	
	async def cleanup(self) -> None:
		"""Cleanup analytics dashboard resources"""
		# Clear all data
		self.metrics.clear()
		self.alerts.clear()
		self.alert_rules.clear()
		self.report_cache.clear()
		self.custom_dashboards.clear()
		
		# Clear subscribers
		self.metric_subscribers.clear()
		self.dashboard_subscribers.clear()
		
		logger.info(f"Analytics dashboard cleanup completed for tenant: {self.tenant_id}")

# Export main classes
__all__ = [
	"AnalyticsDashboard", "TimeSeries", "MetricPoint", "Alert",
	"ReportType", "MetricType", "AlertSeverity", "AlertCondition"
]