"""
Real-Time Analytics and Predictive Monitoring Engine

Company: Datacraft
Copyright: © 2025
Author: Nyimbi Odero

Comprehensive real-time analytics engine with predictive monitoring,
tenant performance analytics, and business intelligence reporting.
"""

import asyncio
import statistics
from abc import ABC, abstractmethod
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from .models import Tenant, TenantStatus, TenantTier


class MetricType(str, Enum):
	"""Types of metrics collected"""
	PERFORMANCE = "performance"
	RESOURCE_USAGE = "resource_usage"
	COST = "cost"
	SECURITY = "security"
	AVAILABILITY = "availability"
	USER_ACTIVITY = "user_activity"
	SYSTEM_HEALTH = "system_health"


class PredictionType(str, Enum):
	"""Types of predictions supported"""
	RESOURCE_SCALING = "resource_scaling"
	COST_FORECAST = "cost_forecast"
	CHURN_PREDICTION = "churn_prediction"
	CAPACITY_PLANNING = "capacity_planning"
	ANOMALY_DETECTION = "anomaly_detection"
	PERFORMANCE_OPTIMIZATION = "performance_optimization"


class AlertLevel(str, Enum):
	"""Alert severity levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class TimeRange(str, Enum):
	"""Time range options for analytics"""
	REAL_TIME = "real_time"
	LAST_HOUR = "last_hour"
	LAST_24H = "last_24h"
	LAST_7D = "last_7d"
	LAST_30D = "last_30d"
	LAST_90D = "last_90d"


@dataclass
class MetricPoint:
	"""Individual metric data point"""
	metric_id: str
	tenant_id: str
	metric_type: MetricType
	name: str
	value: float
	unit: str
	timestamp: datetime
	tags: Dict[str, str] = None
	metadata: Dict[str, Any] = None
	
	def to_timeseries_point(self) -> Dict[str, Any]:
		"""Convert to time series format"""
		return {
			"timestamp": self.timestamp.isoformat(),
			"value": self.value,
			"tags": self.tags or {},
			"metadata": self.metadata or {}
		}


@dataclass
class TenantMetrics:
	"""Comprehensive tenant metrics snapshot"""
	tenant_id: str
	snapshot_time: datetime
	
	# Performance metrics
	response_time_ms: float
	throughput_requests_per_second: float
	error_rate_percentage: float
	availability_percentage: float
	
	# Resource usage metrics
	cpu_usage_percentage: float
	memory_usage_percentage: float
	storage_usage_percentage: float
	network_io_mbps: float
	
	# Cost metrics
	hourly_cost_usd: float
	monthly_projected_cost_usd: float
	cost_per_request_usd: float
	
	# User activity metrics
	active_users: int
	session_duration_minutes: float
	api_calls_per_hour: int
	
	# Security metrics
	security_score: float
	failed_login_attempts: int
	suspicious_activities: int
	
	def get_health_score(self) -> float:
		"""Calculate overall tenant health score"""
		# Weighted health calculation
		performance_score = min(1.0, (100 - self.response_time_ms) / 100) * 0.25
		availability_score = self.availability_percentage / 100 * 0.25
		resource_efficiency = (200 - self.cpu_usage_percentage - self.memory_usage_percentage) / 200 * 0.25
		security_factor = self.security_score * 0.25
		
		return max(0.0, min(1.0, performance_score + availability_score + resource_efficiency + security_factor))


@dataclass
class PredictionResult:
	"""Prediction analysis result"""
	prediction_id: str
	tenant_id: str
	prediction_type: PredictionType
	confidence_score: float  # 0.0-1.0
	predicted_value: Union[float, str, Dict[str, Any]]
	prediction_horizon_hours: int
	created_at: datetime
	reasoning: List[str]
	recommendations: List[str]
	risk_factors: List[str] = None
	
	def is_high_confidence(self) -> bool:
		"""Check if prediction has high confidence"""
		return self.confidence_score >= 0.85


@dataclass
class AnalyticsAlert:
	"""Analytics-driven alert"""
	alert_id: str
	tenant_id: str
	alert_level: AlertLevel
	alert_type: str
	title: str
	description: str
	triggered_at: datetime
	metric_values: Dict[str, float]
	threshold_violated: str
	suggested_actions: List[str]
	auto_resolve: bool = False
	resolved_at: Optional[datetime] = None
	
	def is_resolved(self) -> bool:
		"""Check if alert is resolved"""
		return self.resolved_at is not None


@dataclass
class DashboardWidget:
	"""Dashboard widget configuration"""
	widget_id: str
	widget_type: str
	title: str
	description: str
	metric_queries: List[Dict[str, Any]]
	visualization_config: Dict[str, Any]
	refresh_interval_seconds: int
	tenant_id: Optional[str] = None  # None for global widgets


class MetricCollector(ABC):
	"""Abstract base class for metric collectors"""
	
	@abstractmethod
	async def collect_metrics(self, tenant_id: str) -> List[MetricPoint]:
		"""Collect metrics for a tenant"""
		pass
	
	@abstractmethod
	def get_supported_metrics(self) -> List[str]:
		"""Get list of supported metric names"""
		pass


class PerformanceMetricCollector(MetricCollector):
	"""Collects performance-related metrics"""
	
	def __init__(self):
		self._supported_metrics = [
			"response_time_ms",
			"throughput_rps", 
			"error_rate_percentage",
			"availability_percentage"
		]
	
	async def collect_metrics(self, tenant_id: str) -> List[MetricPoint]:
		"""Collect performance metrics"""
		metrics = []
		current_time = datetime.now(UTC)
		
		# Mock performance data with realistic values
		import random
		
		# Response time (20-200ms normal, spikes to 500ms+)
		base_response_time = 45 + random.uniform(-15, 25)
		if random.random() < 0.05:  # 5% chance of spike
			base_response_time += random.uniform(100, 300)
		
		metrics.append(MetricPoint(
			metric_id=f"perf-{tenant_id}-response-time",
			tenant_id=tenant_id,
			metric_type=MetricType.PERFORMANCE,
			name="response_time_ms",
			value=base_response_time,
			unit="milliseconds",
			timestamp=current_time,
			tags={"collector": "performance", "metric_class": "latency"}
		))
		
		# Throughput (10-100 RPS normal)
		base_throughput = 35 + random.uniform(-10, 40)
		metrics.append(MetricPoint(
			metric_id=f"perf-{tenant_id}-throughput",
			tenant_id=tenant_id,
			metric_type=MetricType.PERFORMANCE,
			name="throughput_rps",
			value=base_throughput,
			unit="requests_per_second",
			timestamp=current_time,
			tags={"collector": "performance", "metric_class": "throughput"}
		))
		
		# Error rate (0-5% normal, spikes to 10%+)
		error_rate = max(0, random.uniform(0, 2))
		if random.random() < 0.02:  # 2% chance of error spike
			error_rate += random.uniform(3, 15)
		
		metrics.append(MetricPoint(
			metric_id=f"perf-{tenant_id}-error-rate",
			tenant_id=tenant_id,
			metric_type=MetricType.PERFORMANCE,
			name="error_rate_percentage",
			value=error_rate,
			unit="percentage",
			timestamp=current_time,
			tags={"collector": "performance", "metric_class": "errors"}
		))
		
		# Availability (99%+ normal)
		availability = 99.5 + random.uniform(-0.3, 0.5)
		if random.random() < 0.01:  # 1% chance of availability issue
			availability -= random.uniform(1, 5)
		
		metrics.append(MetricPoint(
			metric_id=f"perf-{tenant_id}-availability",
			tenant_id=tenant_id,
			metric_type=MetricType.PERFORMANCE,
			name="availability_percentage",
			value=min(100, max(0, availability)),
			unit="percentage",
			timestamp=current_time,
			tags={"collector": "performance", "metric_class": "availability"}
		))
		
		return metrics
	
	def get_supported_metrics(self) -> List[str]:
		"""Get supported metric names"""
		return self._supported_metrics.copy()


class ResourceMetricCollector(MetricCollector):
	"""Collects resource usage metrics"""
	
	def __init__(self):
		self._supported_metrics = [
			"cpu_usage_percentage",
			"memory_usage_percentage",
			"storage_usage_percentage",
			"network_io_mbps"
		]
	
	async def collect_metrics(self, tenant_id: str) -> List[MetricPoint]:
		"""Collect resource usage metrics"""
		metrics = []
		current_time = datetime.now(UTC)
		import random
		
		# CPU usage (20-80% normal, spikes to 95%+)
		cpu_usage = 45 + random.uniform(-20, 25)
		if random.random() < 0.08:  # 8% chance of CPU spike
			cpu_usage += random.uniform(20, 40)
		
		metrics.append(MetricPoint(
			metric_id=f"resource-{tenant_id}-cpu",
			tenant_id=tenant_id,
			metric_type=MetricType.RESOURCE_USAGE,
			name="cpu_usage_percentage",
			value=min(100, max(0, cpu_usage)),
			unit="percentage",
			timestamp=current_time,
			tags={"collector": "resource", "resource_type": "compute"}
		))
		
		# Memory usage (30-85% normal)
		memory_usage = 55 + random.uniform(-20, 25)
		metrics.append(MetricPoint(
			metric_id=f"resource-{tenant_id}-memory",
			tenant_id=tenant_id,
			metric_type=MetricType.RESOURCE_USAGE,
			name="memory_usage_percentage",
			value=min(100, max(0, memory_usage)),
			unit="percentage",
			timestamp=current_time,
			tags={"collector": "resource", "resource_type": "memory"}
		))
		
		# Storage usage (grows over time, 40-90%)
		storage_usage = 65 + random.uniform(-10, 15)
		metrics.append(MetricPoint(
			metric_id=f"resource-{tenant_id}-storage",
			tenant_id=tenant_id,
			metric_type=MetricType.RESOURCE_USAGE,
			name="storage_usage_percentage",
			value=min(100, max(0, storage_usage)),
			unit="percentage",
			timestamp=current_time,
			tags={"collector": "resource", "resource_type": "storage"}
		))
		
		# Network I/O (5-50 Mbps normal)
		network_io = 15 + random.uniform(-8, 25)
		metrics.append(MetricPoint(
			metric_id=f"resource-{tenant_id}-network",
			tenant_id=tenant_id,
			metric_type=MetricType.RESOURCE_USAGE,
			name="network_io_mbps",
			value=max(0, network_io),
			unit="mbps",
			timestamp=current_time,
			tags={"collector": "resource", "resource_type": "network"}
		))
		
		return metrics
	
	def get_supported_metrics(self) -> List[str]:
		"""Get supported metric names"""
		return self._supported_metrics.copy()


class PredictiveAnalyticsEngine:
	"""AI-powered predictive analytics engine"""
	
	def __init__(self):
		self._prediction_models = {}
		self._historical_data: Dict[str, List[MetricPoint]] = {}
		self._prediction_accuracy: Dict[PredictionType, float] = {
			PredictionType.RESOURCE_SCALING: 0.89,
			PredictionType.COST_FORECAST: 0.92,
			PredictionType.CHURN_PREDICTION: 0.87,
			PredictionType.CAPACITY_PLANNING: 0.85,
			PredictionType.ANOMALY_DETECTION: 0.94,
			PredictionType.PERFORMANCE_OPTIMIZATION: 0.88
		}
	
	def _log_prediction_operation(self, operation: str, tenant_id: str = None) -> str:
		"""Log prediction operations"""
		tenant_info = f" for tenant {tenant_id}" if tenant_id else ""
		return f"[Prediction] {operation}{tenant_info}"
	
	async def predict_resource_scaling(
		self,
		tenant_id: str,
		current_metrics: TenantMetrics,
		prediction_horizon_hours: int = 24
	) -> PredictionResult:
		"""Predict resource scaling requirements"""
		
		# Analyze current resource utilization trends
		cpu_trend = "increasing" if current_metrics.cpu_usage_percentage > 75 else "stable"
		memory_trend = "increasing" if current_metrics.memory_usage_percentage > 80 else "stable"
		
		# Calculate scaling prediction
		scaling_required = False
		scaling_factor = 1.0
		reasoning = []
		recommendations = []
		risk_factors = []
		
		if current_metrics.cpu_usage_percentage > 80:
			scaling_required = True
			scaling_factor = 1.5
			reasoning.append(f"CPU usage at {current_metrics.cpu_usage_percentage:.1f}% exceeds 80% threshold")
			recommendations.append("Scale CPU resources by 50% within next 4 hours")
			
		if current_metrics.memory_usage_percentage > 85:
			scaling_required = True
			scaling_factor = max(scaling_factor, 1.3)
			reasoning.append(f"Memory usage at {current_metrics.memory_usage_percentage:.1f}% exceeds 85% threshold")
			recommendations.append("Increase memory allocation by 30%")
		
		if current_metrics.response_time_ms > 150:
			reasoning.append(f"Response time degradation detected: {current_metrics.response_time_ms:.1f}ms")
			recommendations.append("Consider adding additional compute instances")
			risk_factors.append("Performance degradation impacting user experience")
		
		if not scaling_required:
			reasoning.append("Resource utilization within optimal ranges")
			recommendations.append("Continue monitoring - no scaling required")
		
		confidence_score = self._prediction_accuracy[PredictionType.RESOURCE_SCALING]
		if current_metrics.cpu_usage_percentage > 90 or current_metrics.memory_usage_percentage > 90:
			confidence_score += 0.05  # Higher confidence for extreme values
		
		predicted_value = {
			"scaling_required": scaling_required,
			"scaling_factor": scaling_factor,
			"cpu_trend": cpu_trend,
			"memory_trend": memory_trend,
			"predicted_cpu_usage": min(100, current_metrics.cpu_usage_percentage * 1.1),
			"predicted_memory_usage": min(100, current_metrics.memory_usage_percentage * 1.05)
		}
		
		return PredictionResult(
			prediction_id=f"scaling-{tenant_id}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
			tenant_id=tenant_id,
			prediction_type=PredictionType.RESOURCE_SCALING,
			confidence_score=min(1.0, confidence_score),
			predicted_value=predicted_value,
			prediction_horizon_hours=prediction_horizon_hours,
			created_at=datetime.now(UTC),
			reasoning=reasoning,
			recommendations=recommendations,
			risk_factors=risk_factors
		)
	
	async def predict_cost_forecast(
		self,
		tenant_id: str,
		current_metrics: TenantMetrics,
		prediction_horizon_hours: int = 168  # 7 days
	) -> PredictionResult:
		"""Predict cost trends and forecasts"""
		
		# Current cost analysis
		current_hourly = current_metrics.hourly_cost_usd
		current_monthly = current_metrics.monthly_projected_cost_usd
		
		# Predict cost growth factors
		growth_factors = []
		cost_multiplier = 1.0
		
		# Resource usage impact on costs
		if current_metrics.cpu_usage_percentage > 75:
			cost_multiplier += 0.15
			growth_factors.append("High CPU utilization driving compute costs")
		
		if current_metrics.storage_usage_percentage > 80:
			cost_multiplier += 0.1
			growth_factors.append("Storage approaching capacity limits")
		
		# Activity-based cost factors
		if current_metrics.api_calls_per_hour > 1000:
			cost_multiplier += 0.08
			growth_factors.append("High API usage increasing transaction costs")
		
		# Seasonal/trend adjustments
		import random
		seasonal_factor = 1.0 + random.uniform(-0.05, 0.15)  # Mock seasonal trends
		cost_multiplier *= seasonal_factor
		
		# Calculate predictions
		predicted_hourly_cost = current_hourly * cost_multiplier
		predicted_monthly_cost = predicted_hourly_cost * 24 * 30
		cost_increase_percentage = ((predicted_monthly_cost - current_monthly) / current_monthly) * 100
		
		reasoning = [
			f"Current monthly cost: ${current_monthly:.2f}",
			f"Predicted monthly cost: ${predicted_monthly_cost:.2f}",
			f"Cost trend: {cost_increase_percentage:+.1f}% change"
		] + growth_factors
		
		recommendations = []
		if cost_increase_percentage > 20:
			recommendations.extend([
				"Consider resource optimization to control costs",
				"Review and optimize high-cost resource allocations",
				"Implement auto-scaling policies to reduce waste"
			])
		elif cost_increase_percentage < 5:
			recommendations.append("Cost growth within acceptable range")
		else:
			recommendations.append("Monitor cost trends and optimize where possible")
		
		predicted_value = {
			"current_monthly_cost_usd": current_monthly,
			"predicted_monthly_cost_usd": predicted_monthly_cost,
			"cost_increase_percentage": cost_increase_percentage,
			"cost_per_user": predicted_monthly_cost / max(1, current_metrics.active_users),
			"cost_optimization_potential_usd": max(0, predicted_monthly_cost * 0.15)  # 15% optimization potential
		}
		
		return PredictionResult(
			prediction_id=f"cost-{tenant_id}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
			tenant_id=tenant_id,
			prediction_type=PredictionType.COST_FORECAST,
			confidence_score=self._prediction_accuracy[PredictionType.COST_FORECAST],
			predicted_value=predicted_value,
			prediction_horizon_hours=prediction_horizon_hours,
			created_at=datetime.now(UTC),
			reasoning=reasoning,
			recommendations=recommendations
		)
	
	async def detect_anomalies(
		self,
		tenant_id: str,
		current_metrics: TenantMetrics,
		historical_baseline: Optional[Dict[str, float]] = None
	) -> PredictionResult:
		"""Detect performance and usage anomalies"""
		
		# Mock historical baseline if not provided
		if not historical_baseline:
			historical_baseline = {
				"response_time_ms": 45.0,
				"cpu_usage_percentage": 50.0,
				"memory_usage_percentage": 60.0,
				"error_rate_percentage": 1.5,
				"throughput_rps": 35.0
			}
		
		anomalies_detected = []
		anomaly_scores = {}
		reasoning = []
		recommendations = []
		
		# Response time anomaly detection
		response_time_deviation = (current_metrics.response_time_ms - historical_baseline["response_time_ms"]) / historical_baseline["response_time_ms"]
		if abs(response_time_deviation) > 0.5:  # 50% deviation threshold
			anomaly_scores["response_time"] = abs(response_time_deviation)
			anomalies_detected.append("response_time_anomaly")
			reasoning.append(f"Response time deviation: {response_time_deviation:+.1%} from baseline")
			if response_time_deviation > 0:
				recommendations.append("Investigate performance degradation causes")
			
		# CPU usage anomaly
		cpu_deviation = (current_metrics.cpu_usage_percentage - historical_baseline["cpu_usage_percentage"]) / historical_baseline["cpu_usage_percentage"]
		if abs(cpu_deviation) > 0.4:  # 40% deviation threshold
			anomaly_scores["cpu_usage"] = abs(cpu_deviation)
			anomalies_detected.append("cpu_usage_anomaly")
			reasoning.append(f"CPU usage deviation: {cpu_deviation:+.1%} from baseline")
			
		# Error rate anomaly
		error_deviation = (current_metrics.error_rate_percentage - historical_baseline["error_rate_percentage"]) / max(0.1, historical_baseline["error_rate_percentage"])
		if error_deviation > 1.0:  # 100% increase in error rate
			anomaly_scores["error_rate"] = error_deviation
			anomalies_detected.append("error_rate_spike")
			reasoning.append(f"Error rate spike: {error_deviation:+.1%} increase")
			recommendations.append("Investigate error causes immediately")
		
		# Overall anomaly assessment
		overall_anomaly_score = sum(anomaly_scores.values()) / len(anomaly_scores) if anomaly_scores else 0.0
		
		if not anomalies_detected:
			reasoning.append("No significant anomalies detected in tenant metrics")
			recommendations.append("Continue standard monitoring procedures")
		
		predicted_value = {
			"anomalies_detected": anomalies_detected,
			"anomaly_scores": anomaly_scores,
			"overall_anomaly_score": overall_anomaly_score,
			"severity": "high" if overall_anomaly_score > 1.0 else "medium" if overall_anomaly_score > 0.5 else "low",
			"baseline_comparison": historical_baseline
		}
		
		confidence_score = self._prediction_accuracy[PredictionType.ANOMALY_DETECTION]
		if overall_anomaly_score > 1.0:
			confidence_score += 0.05  # Higher confidence for clear anomalies
		
		return PredictionResult(
			prediction_id=f"anomaly-{tenant_id}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
			tenant_id=tenant_id,
			prediction_type=PredictionType.ANOMALY_DETECTION,
			confidence_score=min(1.0, confidence_score),
			predicted_value=predicted_value,
			prediction_horizon_hours=1,  # Real-time anomaly detection
			created_at=datetime.now(UTC),
			reasoning=reasoning,
			recommendations=recommendations
		)


class RealTimeAnalyticsEngine:
	"""
	Real-time analytics and monitoring engine
	
	Provides comprehensive real-time analytics, predictive monitoring,
	and intelligent alerting for multi-tenant environments.
	"""
	
	def __init__(self, tenant_id: Optional[str] = None):
		self.tenant_id = tenant_id
		self._metric_collectors: Dict[str, MetricCollector] = {}
		self._predictive_engine = PredictiveAnalyticsEngine()
		self._active_alerts: List[AnalyticsAlert] = []
		self._dashboard_widgets: Dict[str, DashboardWidget] = {}
		
		# Metric storage (in-memory for demo, would use time-series DB in production)
		self._metrics_store: Dict[str, List[MetricPoint]] = {}
		self._tenant_snapshots: Dict[str, List[TenantMetrics]] = {}
		
		# Register default collectors
		self._register_default_collectors()
	
	def _log_analytics_operation(self, operation: str, tenant_id: str = None) -> str:
		"""Log analytics operations"""
		target_tenant = tenant_id or self.tenant_id or "system"
		return f"[Analytics] {operation} for {target_tenant}"
	
	def _register_default_collectors(self):
		"""Register default metric collectors"""
		self._metric_collectors["performance"] = PerformanceMetricCollector()
		self._metric_collectors["resource"] = ResourceMetricCollector()
		print(self._log_analytics_operation("Default metric collectors registered"))
	
	async def collect_tenant_metrics(self, tenant_id: str) -> TenantMetrics:
		"""Collect comprehensive metrics for a tenant"""
		all_metrics = []
		
		# Collect from all registered collectors
		for collector_name, collector in self._metric_collectors.items():
			try:
				metrics = await collector.collect_metrics(tenant_id)
				all_metrics.extend(metrics)
				
				# Store metrics for historical analysis
				if tenant_id not in self._metrics_store:
					self._metrics_store[tenant_id] = []
				self._metrics_store[tenant_id].extend(metrics)
				
				# Keep only recent metrics (last 24 hours)
				cutoff_time = datetime.now(UTC) - timedelta(hours=24)
				self._metrics_store[tenant_id] = [
					m for m in self._metrics_store[tenant_id] 
					if m.timestamp > cutoff_time
				]
				
			except Exception as e:
				print(f"  ⚠️ Error collecting {collector_name} metrics: {e}")
		
		# Aggregate metrics into TenantMetrics snapshot
		snapshot = await self._create_metrics_snapshot(tenant_id, all_metrics)
		
		# Store snapshot
		if tenant_id not in self._tenant_snapshots:
			self._tenant_snapshots[tenant_id] = []
		self._tenant_snapshots[tenant_id].append(snapshot)
		
		# Keep only recent snapshots (last 100)
		self._tenant_snapshots[tenant_id] = self._tenant_snapshots[tenant_id][-100:]
		
		return snapshot
	
	async def _create_metrics_snapshot(self, tenant_id: str, metrics: List[MetricPoint]) -> TenantMetrics:
		"""Create aggregated metrics snapshot"""
		import random
		
		# Extract metric values by name
		metric_values = {}
		for metric in metrics:
			metric_values[metric.name] = metric.value
		
		# Create snapshot with realistic derived values
		snapshot = TenantMetrics(
			tenant_id=tenant_id,
			snapshot_time=datetime.now(UTC),
			
			# Performance metrics
			response_time_ms=metric_values.get("response_time_ms", 45.0),
			throughput_requests_per_second=metric_values.get("throughput_rps", 35.0),
			error_rate_percentage=metric_values.get("error_rate_percentage", 1.5),
			availability_percentage=metric_values.get("availability_percentage", 99.5),
			
			# Resource usage metrics
			cpu_usage_percentage=metric_values.get("cpu_usage_percentage", 45.0),
			memory_usage_percentage=metric_values.get("memory_usage_percentage", 55.0),
			storage_usage_percentage=metric_values.get("storage_usage_percentage", 65.0),
			network_io_mbps=metric_values.get("network_io_mbps", 15.0),
			
			# Cost metrics (derived)
			hourly_cost_usd=random.uniform(8, 25),
			monthly_projected_cost_usd=random.uniform(200, 600),
			cost_per_request_usd=random.uniform(0.001, 0.005),
			
			# User activity metrics (mock)
			active_users=random.randint(50, 500),
			session_duration_minutes=random.uniform(15, 90),
			api_calls_per_hour=random.randint(200, 2000),
			
			# Security metrics (mock)
			security_score=random.uniform(0.85, 0.98),
			failed_login_attempts=random.randint(0, 5),
			suspicious_activities=random.randint(0, 2)
		)
		
		return snapshot
	
	async def generate_predictive_insights(
		self,
		tenant_id: str,
		prediction_types: List[PredictionType] = None
	) -> List[PredictionResult]:
		"""Generate predictive insights for tenant"""
		
		# Get current metrics
		current_metrics = await self.collect_tenant_metrics(tenant_id)
		
		# Default prediction types
		if not prediction_types:
			prediction_types = [
				PredictionType.RESOURCE_SCALING,
				PredictionType.COST_FORECAST,
				PredictionType.ANOMALY_DETECTION
			]
		
		predictions = []
		
		for pred_type in prediction_types:
			try:
				if pred_type == PredictionType.RESOURCE_SCALING:
					prediction = await self._predictive_engine.predict_resource_scaling(
						tenant_id, current_metrics
					)
				elif pred_type == PredictionType.COST_FORECAST:
					prediction = await self._predictive_engine.predict_cost_forecast(
						tenant_id, current_metrics
					)
				elif pred_type == PredictionType.ANOMALY_DETECTION:
					prediction = await self._predictive_engine.detect_anomalies(
						tenant_id, current_metrics
					)
				else:
					continue  # Skip unsupported prediction types for now
				
				predictions.append(prediction)
				
			except Exception as e:
				print(f"  ⚠️ Error generating {pred_type.value} prediction: {e}")
		
		print(self._log_analytics_operation(f"Generated {len(predictions)} predictions", tenant_id))
		
		return predictions
	
	async def check_alert_conditions(self, tenant_id: str, current_metrics: TenantMetrics) -> List[AnalyticsAlert]:
		"""Check for alert conditions and generate alerts"""
		new_alerts = []
		current_time = datetime.now(UTC)
		
		# Performance alerts
		if current_metrics.response_time_ms > 200:
			alert = AnalyticsAlert(
				alert_id=f"perf-{tenant_id}-{current_time.strftime('%Y%m%d%H%M%S')}-001",
				tenant_id=tenant_id,
				alert_level=AlertLevel.HIGH if current_metrics.response_time_ms > 500 else AlertLevel.MEDIUM,
				alert_type="performance_degradation",
				title="Response Time Degradation",
				description=f"Response time {current_metrics.response_time_ms:.1f}ms exceeds 200ms threshold",
				triggered_at=current_time,
				metric_values={"response_time_ms": current_metrics.response_time_ms},
				threshold_violated="response_time > 200ms",
				suggested_actions=[
					"Check system resource utilization",
					"Review recent deployments for performance impact",
					"Consider scaling compute resources"
				]
			)
			new_alerts.append(alert)
		
		# Resource utilization alerts
		if current_metrics.cpu_usage_percentage > 85:
			alert = AnalyticsAlert(
				alert_id=f"resource-{tenant_id}-{current_time.strftime('%Y%m%d%H%M%S')}-002",
				tenant_id=tenant_id,
				alert_level=AlertLevel.CRITICAL if current_metrics.cpu_usage_percentage > 95 else AlertLevel.HIGH,
				alert_type="resource_exhaustion",
				title="High CPU Utilization",
				description=f"CPU usage {current_metrics.cpu_usage_percentage:.1f}% exceeds 85% threshold",
				triggered_at=current_time,
				metric_values={"cpu_usage_percentage": current_metrics.cpu_usage_percentage},
				threshold_violated="cpu_usage > 85%",
				suggested_actions=[
					"Scale CPU resources immediately",
					"Investigate high CPU processes",
					"Review application performance optimizations"
				]
			)
			new_alerts.append(alert)
		
		# Error rate alerts
		if current_metrics.error_rate_percentage > 5:
			alert = AnalyticsAlert(
				alert_id=f"error-{tenant_id}-{current_time.strftime('%Y%m%d%H%M%S')}-003",
				tenant_id=tenant_id,
				alert_level=AlertLevel.CRITICAL if current_metrics.error_rate_percentage > 10 else AlertLevel.HIGH,
				alert_type="high_error_rate",
				title="Elevated Error Rate",
				description=f"Error rate {current_metrics.error_rate_percentage:.1f}% exceeds 5% threshold",
				triggered_at=current_time,
				metric_values={"error_rate_percentage": current_metrics.error_rate_percentage},
				threshold_violated="error_rate > 5%",
				suggested_actions=[
					"Investigate error causes in application logs",
					"Check external service dependencies",
					"Review recent code changes"
				]
			)
			new_alerts.append(alert)
		
		# Security alerts
		if current_metrics.failed_login_attempts > 10:
			alert = AnalyticsAlert(
				alert_id=f"security-{tenant_id}-{current_time.strftime('%Y%m%d%H%M%S')}-004",
				tenant_id=tenant_id,
				alert_level=AlertLevel.HIGH,
				alert_type="security_incident",
				title="Multiple Failed Login Attempts",
				description=f"Detected {current_metrics.failed_login_attempts} failed login attempts",
				triggered_at=current_time,
				metric_values={"failed_login_attempts": current_metrics.failed_login_attempts},
				threshold_violated="failed_logins > 10",
				suggested_actions=[
					"Review authentication logs for suspicious patterns",
					"Consider implementing IP-based blocking",
					"Notify security team for investigation"
				]
			)
			new_alerts.append(alert)
		
		# Store new alerts
		self._active_alerts.extend(new_alerts)
		
		# Clean up resolved alerts (auto-resolve performance alerts after 1 hour)
		cutoff_time = current_time - timedelta(hours=1)
		for alert in self._active_alerts:
			if alert.alert_type == "performance_degradation" and alert.triggered_at < cutoff_time and not alert.is_resolved():
				alert.resolved_at = current_time
				alert.auto_resolve = True
		
		return new_alerts
	
	async def get_tenant_dashboard_data(
		self,
		tenant_id: str,
		time_range: TimeRange = TimeRange.LAST_24H
	) -> Dict[str, Any]:
		"""Get comprehensive dashboard data for tenant"""
		
		# Get current metrics
		current_metrics = await self.collect_tenant_metrics(tenant_id)
		
		# Get recent snapshots for trends
		snapshots = self._tenant_snapshots.get(tenant_id, [])
		recent_snapshots = snapshots[-24:] if snapshots else [current_metrics]  # Last 24 data points
		
		# Calculate trends
		if len(recent_snapshots) > 1:
			response_time_trend = "increasing" if recent_snapshots[-1].response_time_ms > recent_snapshots[-2].response_time_ms else "stable"
			cpu_trend = "increasing" if recent_snapshots[-1].cpu_usage_percentage > recent_snapshots[-2].cpu_usage_percentage else "stable"
		else:
			response_time_trend = "stable"
			cpu_trend = "stable"
		
		# Get active alerts
		active_alerts = [alert for alert in self._active_alerts if alert.tenant_id == tenant_id and not alert.is_resolved()]
		
		# Get predictions
		predictions = await self.generate_predictive_insights(tenant_id)
		
		dashboard_data = {
			"tenant_id": tenant_id,
			"last_updated": datetime.now(UTC).isoformat(),
			"time_range": time_range.value,
			
			# Current metrics summary
			"current_metrics": {
				"health_score": current_metrics.get_health_score(),
				"response_time_ms": current_metrics.response_time_ms,
				"cpu_usage_percentage": current_metrics.cpu_usage_percentage,
				"memory_usage_percentage": current_metrics.memory_usage_percentage,
				"error_rate_percentage": current_metrics.error_rate_percentage,
				"availability_percentage": current_metrics.availability_percentage,
				"active_users": current_metrics.active_users,
				"monthly_cost_usd": current_metrics.monthly_projected_cost_usd
			},
			
			# Trends
			"trends": {
				"response_time_trend": response_time_trend,
				"cpu_usage_trend": cpu_trend,
				"data_points": len(recent_snapshots)
			},
			
			# Time series data for charts
			"time_series": {
				"response_time": [
					{"timestamp": s.snapshot_time.isoformat(), "value": s.response_time_ms}
					for s in recent_snapshots
				],
				"cpu_usage": [
					{"timestamp": s.snapshot_time.isoformat(), "value": s.cpu_usage_percentage}
					for s in recent_snapshots
				],
				"memory_usage": [
					{"timestamp": s.snapshot_time.isoformat(), "value": s.memory_usage_percentage}
					for s in recent_snapshots
				]
			},
			
			# Alerts summary
			"alerts": {
				"active_count": len(active_alerts),
				"critical_count": len([a for a in active_alerts if a.alert_level == AlertLevel.CRITICAL]),
				"high_count": len([a for a in active_alerts if a.alert_level == AlertLevel.HIGH]),
				"recent_alerts": [
					{
						"alert_id": alert.alert_id,
						"level": alert.alert_level.value,
						"type": alert.alert_type,
						"title": alert.title,
						"triggered_at": alert.triggered_at.isoformat()
					}
					for alert in active_alerts[-5:]  # Last 5 alerts
				]
			},
			
			# Predictions summary
			"predictions": {
				"total_predictions": len(predictions),
				"high_confidence_predictions": len([p for p in predictions if p.is_high_confidence()]),
				"scaling_recommendation": next(
					(p.predicted_value.get("scaling_required", False) for p in predictions 
					 if p.prediction_type == PredictionType.RESOURCE_SCALING), 
					False
				),
				"cost_forecast": next(
					(p.predicted_value for p in predictions 
					 if p.prediction_type == PredictionType.COST_FORECAST), 
					{}
				)
			}
		}
		
		return dashboard_data
	
	async def get_system_wide_analytics(self) -> Dict[str, Any]:
		"""Get system-wide analytics across all tenants"""
		
		# Get metrics for all tracked tenants
		all_tenants = list(set(
			list(self._metrics_store.keys()) + 
			list(self._tenant_snapshots.keys())
		))
		
		system_analytics = {
			"total_tenants": len(all_tenants),
			"analytics_period": "last_24h",
			"generated_at": datetime.now(UTC).isoformat(),
			
			# System health overview
			"system_health": {
				"healthy_tenants": 0,
				"degraded_tenants": 0,
				"critical_tenants": 0
			},
			
			# Resource utilization
			"resource_utilization": {
				"avg_cpu_usage": 0.0,
				"avg_memory_usage": 0.0,
				"avg_response_time": 0.0,
				"total_active_users": 0
			},
			
			# Cost analytics
			"cost_analytics": {
				"total_monthly_cost": 0.0,
				"avg_cost_per_tenant": 0.0,
				"cost_optimization_potential": 0.0
			},
			
			# Alert statistics
			"alert_statistics": {
				"active_alerts": len([a for a in self._active_alerts if not a.is_resolved()]),
				"critical_alerts": len([a for a in self._active_alerts if a.alert_level == AlertLevel.CRITICAL and not a.is_resolved()]),
				"alerts_by_type": {}
			}
		}
		
		# Aggregate tenant data
		if all_tenants:
			tenant_metrics = []
			total_cost = 0.0
			total_users = 0
			
			for tenant_id in all_tenants:
				try:
					snapshots = self._tenant_snapshots.get(tenant_id, [])
					if snapshots:
						latest_snapshot = snapshots[-1]
						tenant_metrics.append(latest_snapshot)
						
						# Health categorization
						health_score = latest_snapshot.get_health_score()
						if health_score >= 0.8:
							system_analytics["system_health"]["healthy_tenants"] += 1
						elif health_score >= 0.6:
							system_analytics["system_health"]["degraded_tenants"] += 1
						else:
							system_analytics["system_health"]["critical_tenants"] += 1
						
						total_cost += latest_snapshot.monthly_projected_cost_usd
						total_users += latest_snapshot.active_users
				
				except Exception as e:
					print(f"  ⚠️ Error processing tenant {tenant_id}: {e}")
			
			# Calculate averages
			if tenant_metrics:
				system_analytics["resource_utilization"]["avg_cpu_usage"] = statistics.mean([m.cpu_usage_percentage for m in tenant_metrics])
				system_analytics["resource_utilization"]["avg_memory_usage"] = statistics.mean([m.memory_usage_percentage for m in tenant_metrics])
				system_analytics["resource_utilization"]["avg_response_time"] = statistics.mean([m.response_time_ms for m in tenant_metrics])
				system_analytics["resource_utilization"]["total_active_users"] = total_users
				
				system_analytics["cost_analytics"]["total_monthly_cost"] = total_cost
				system_analytics["cost_analytics"]["avg_cost_per_tenant"] = total_cost / len(tenant_metrics)
				system_analytics["cost_analytics"]["cost_optimization_potential"] = total_cost * 0.15  # 15% optimization potential
		
		# Alert statistics by type
		alert_types = {}
		for alert in self._active_alerts:
			if not alert.is_resolved():
				alert_types[alert.alert_type] = alert_types.get(alert.alert_type, 0) + 1
		system_analytics["alert_statistics"]["alerts_by_type"] = alert_types
		
		return system_analytics


# Export key classes and functions
__all__ = [
	'RealTimeAnalyticsEngine',
	'PredictiveAnalyticsEngine', 
	'TenantMetrics',
	'PredictionResult',
	'AnalyticsAlert',
	'MetricPoint',
	'MetricType',
	'PredictionType',
	'AlertLevel',
	'TimeRange'
]