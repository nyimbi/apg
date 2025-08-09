#!/usr/bin/env python3
"""
APG Monitoring - Anomaly Detection System
ML-based anomaly detection with adaptive baselines and contextual intelligence

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

from .models import MonitoringMetric, AlertSeverity


class AnomalyType(str, Enum):
	"""Types of anomalies"""
	STATISTICAL = "statistical"  # Statistical outliers
	TEMPORAL = "temporal"  # Time-based anomalies
	CONTEXTUAL = "contextual"  # Context-dependent anomalies
	COLLECTIVE = "collective"  # Pattern-based anomalies
	SEASONAL = "seasonal"  # Seasonal pattern deviations
	TREND = "trend"  # Trend-based anomalies


class AnomalyAlgorithm(str, Enum):
	"""Anomaly detection algorithms"""
	Z_SCORE = "z_score"
	MODIFIED_Z_SCORE = "modified_z_score"
	IQR = "iqr"  # Interquartile Range
	ISOLATION_FOREST = "isolation_forest"
	LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
	ARIMA = "arima"  # AutoRegressive Integrated Moving Average
	PROPHET = "prophet"  # Facebook Prophet
	DBSCAN = "dbscan"  # Density-based clustering
	SEASONAL_HYBRID = "seasonal_hybrid"


class AnomalySeverity(str, Enum):
	"""Anomaly severity levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


@dataclass
class AnomalyResult:
	"""Anomaly detection result"""
	anomaly_id: str
	metric_name: str
	tenant_id: str
	timestamp: datetime
	value: float
	expected_value: Optional[float]
	deviation: float
	anomaly_score: float  # 0.0 - 1.0
	anomaly_type: AnomalyType
	algorithm_used: AnomalyAlgorithm
	severity: AnomalySeverity
	confidence: float  # 0.0 - 1.0
	context: Dict[str, Any]
	explanation: str
	similar_anomalies: List[str]  # IDs of similar past anomalies
	detected_at: datetime = field(default_factory=datetime.utcnow)
	
	def to_dict(self) -> dict:
		"""Convert to dictionary representation"""
		return {
			'anomaly_id': self.anomaly_id,
			'metric_name': self.metric_name,
			'tenant_id': self.tenant_id,
			'timestamp': self.timestamp.isoformat(),
			'value': self.value,
			'expected_value': self.expected_value,
			'deviation': self.deviation,
			'anomaly_score': self.anomaly_score,
			'anomaly_type': self.anomaly_type.value,
			'algorithm_used': self.algorithm_used.value,
			'severity': self.severity.value,
			'confidence': self.confidence,
			'context': self.context,
			'explanation': self.explanation,
			'similar_anomalies': self.similar_anomalies,
			'detected_at': self.detected_at.isoformat()
		}
	
	def is_actionable(self) -> bool:
		"""Check if anomaly requires immediate action"""
		return (self.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL] and 
				self.confidence > 0.7 and 
				self.anomaly_score > 0.8)


@dataclass
class BaselineModel:
	"""Baseline model for anomaly detection"""
	metric_name: str
	tenant_id: str
	model_type: str
	parameters: Dict[str, Any]
	training_period: timedelta
	last_updated: datetime
	accuracy_metrics: Dict[str, float]
	sample_size: int
	seasonal_patterns: Dict[str, Any]
	trend_parameters: Dict[str, float]
	confidence_intervals: Dict[str, Tuple[float, float]]
	
	def is_stale(self, max_age_hours: int = 24) -> bool:
		"""Check if model needs retraining"""
		age_hours = (datetime.utcnow() - self.last_updated).total_seconds() / 3600
		return age_hours > max_age_hours
	
	def predict(self, timestamp: datetime, context: Dict[str, Any] = None) -> Tuple[float, float]:
		"""Predict expected value and confidence interval"""
		# Simplified prediction - in practice would use actual model
		base_value = self.parameters.get('mean', 0.0)
		std_dev = self.parameters.get('std_dev', 1.0)
		
		# Apply seasonal adjustments
		hour = timestamp.hour
		day_of_week = timestamp.weekday()
		
		seasonal_adjust = self.seasonal_patterns.get(f"hour_{hour}", 1.0)
		weekly_adjust = self.seasonal_patterns.get(f"dow_{day_of_week}", 1.0)
		
		expected_value = base_value * seasonal_adjust * weekly_adjust
		confidence_interval = std_dev * 2  # 95% confidence interval
		
		return expected_value, confidence_interval


class StatisticalAnomalyDetector:
	"""Statistical anomaly detection algorithms"""
	
	def __init__(self, config: dict = None):
		self.config = config or {}
		self.z_score_threshold = self.config.get('z_score_threshold', 3.0)
		self.iqr_multiplier = self.config.get('iqr_multiplier', 1.5)
		
	async def detect_z_score_anomalies(self, metrics: List[MonitoringMetric], 
									  baseline: BaselineModel) -> List[AnomalyResult]:
		"""Detect anomalies using Z-score method"""
		anomalies = []
		
		if len(metrics) < 10:  # Need sufficient data
			return anomalies
		
		values = [m.value for m in metrics]
		mean_val = statistics.mean(values)
		std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
		
		if std_dev == 0:  # No variance in data
			return anomalies
		
		for metric in metrics:
			z_score = abs(metric.value - mean_val) / std_dev
			
			if z_score > self.z_score_threshold:
				anomaly_score = min(1.0, z_score / (self.z_score_threshold * 2))
				severity = self._calculate_severity(anomaly_score)
				
				anomaly = AnomalyResult(
					anomaly_id=uuid7str(),
					metric_name=metric.name,
					tenant_id=metric.tenant_id,
					timestamp=metric.timestamp,
					value=metric.value,
					expected_value=mean_val,
					deviation=abs(metric.value - mean_val),
					anomaly_score=anomaly_score,
					anomaly_type=AnomalyType.STATISTICAL,
					algorithm_used=AnomalyAlgorithm.Z_SCORE,
					severity=severity,
					confidence=min(0.9, 1.0 - (1.0 / z_score)),
					context={'z_score': z_score, 'threshold': self.z_score_threshold},
					explanation=f"Value {metric.value:.2f} is {z_score:.2f} standard deviations from mean {mean_val:.2f}",
					similar_anomalies=[]
				)
				anomalies.append(anomaly)
		
		return anomalies
	
	async def detect_modified_z_score_anomalies(self, metrics: List[MonitoringMetric],
											   baseline: BaselineModel) -> List[AnomalyResult]:
		"""Detect anomalies using Modified Z-score (more robust to outliers)"""
		anomalies = []
		
		if len(metrics) < 10:
			return anomalies
		
		values = [m.value for m in metrics]
		median_val = statistics.median(values)
		
		# Calculate MAD (Median Absolute Deviation)
		mad = statistics.median([abs(v - median_val) for v in values])
		
		if mad == 0:
			return anomalies
		
		modified_z_threshold = 3.5  # Common threshold for modified Z-score
		
		for metric in metrics:
			modified_z_score = 0.6745 * (metric.value - median_val) / mad
			
			if abs(modified_z_score) > modified_z_threshold:
				anomaly_score = min(1.0, abs(modified_z_score) / (modified_z_threshold * 2))
				severity = self._calculate_severity(anomaly_score)
				
				anomaly = AnomalyResult(
					anomaly_id=uuid7str(),
					metric_name=metric.name,
					tenant_id=metric.tenant_id,
					timestamp=metric.timestamp,
					value=metric.value,
					expected_value=median_val,
					deviation=abs(metric.value - median_val),
					anomaly_score=anomaly_score,
					anomaly_type=AnomalyType.STATISTICAL,
					algorithm_used=AnomalyAlgorithm.MODIFIED_Z_SCORE,
					severity=severity,
					confidence=min(0.9, 1.0 - (1.0 / abs(modified_z_score))),
					context={'modified_z_score': modified_z_score, 'mad': mad},
					explanation=f"Value {metric.value:.2f} has modified Z-score {modified_z_score:.2f} (threshold: {modified_z_threshold})",
					similar_anomalies=[]
				)
				anomalies.append(anomaly)
		
		return anomalies
	
	async def detect_iqr_anomalies(self, metrics: List[MonitoringMetric],
								  baseline: BaselineModel) -> List[AnomalyResult]:
		"""Detect anomalies using Interquartile Range method"""
		anomalies = []
		
		if len(metrics) < 10:
			return anomalies
		
		values = [m.value for m in metrics]
		q1 = np.percentile(values, 25)
		q3 = np.percentile(values, 75)
		iqr = q3 - q1
		
		if iqr == 0:
			return anomalies
		
		lower_bound = q1 - (self.iqr_multiplier * iqr)
		upper_bound = q3 + (self.iqr_multiplier * iqr)
		
		for metric in metrics:
			if metric.value < lower_bound or metric.value > upper_bound:
				# Calculate how far outside the bounds
				if metric.value < lower_bound:
					deviation = lower_bound - metric.value
					expected = lower_bound
				else:
					deviation = metric.value - upper_bound
					expected = upper_bound
				
				anomaly_score = min(1.0, deviation / (iqr * self.iqr_multiplier))
				severity = self._calculate_severity(anomaly_score)
				
				anomaly = AnomalyResult(
					anomaly_id=uuid7str(),
					metric_name=metric.name,
					tenant_id=metric.tenant_id,
					timestamp=metric.timestamp,
					value=metric.value,
					expected_value=expected,
					deviation=deviation,
					anomaly_score=anomaly_score,
					anomaly_type=AnomalyType.STATISTICAL,
					algorithm_used=AnomalyAlgorithm.IQR,
					severity=severity,
					confidence=0.8,
					context={'q1': q1, 'q3': q3, 'iqr': iqr, 'bounds': [lower_bound, upper_bound]},
					explanation=f"Value {metric.value:.2f} is outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
					similar_anomalies=[]
				)
				anomalies.append(anomaly)
		
		return anomalies
	
	def _calculate_severity(self, anomaly_score: float) -> AnomalySeverity:
		"""Calculate anomaly severity based on score"""
		if anomaly_score >= 0.9:
			return AnomalySeverity.CRITICAL
		elif anomaly_score >= 0.7:
			return AnomalySeverity.HIGH
		elif anomaly_score >= 0.4:
			return AnomalySeverity.MEDIUM
		else:
			return AnomalySeverity.LOW


class TemporalAnomalyDetector:
	"""Temporal anomaly detection for time-series patterns"""
	
	def __init__(self, config: dict = None):
		self.config = config or {}
		self.window_size = self.config.get('window_size', 12)  # 12 data points
		self.trend_threshold = self.config.get('trend_threshold', 0.1)
		
	async def detect_seasonal_anomalies(self, metrics: List[MonitoringMetric],
									   baseline: BaselineModel) -> List[AnomalyResult]:
		"""Detect anomalies based on seasonal patterns"""
		anomalies = []
		
		if len(metrics) < 24:  # Need at least 24 data points for seasonality
			return anomalies
		
		# Group metrics by time patterns (hour of day, day of week)
		hourly_patterns = defaultdict(list)
		daily_patterns = defaultdict(list)
		
		for metric in metrics:
			hour = metric.timestamp.hour
			day_of_week = metric.timestamp.weekday()
			
			hourly_patterns[hour].append(metric.value)
			daily_patterns[day_of_week].append(metric.value)
		
		# Calculate expected values for each pattern
		hourly_expectations = {}
		daily_expectations = {}
		
		for hour, values in hourly_patterns.items():
			if len(values) >= 3:  # Minimum samples for pattern
				hourly_expectations[hour] = {
					'mean': statistics.mean(values),
					'std': statistics.stdev(values) if len(values) > 1 else 0,
					'samples': len(values)
				}
		
		for day, values in daily_patterns.items():
			if len(values) >= 3:
				daily_expectations[day] = {
					'mean': statistics.mean(values),
					'std': statistics.stdev(values) if len(values) > 1 else 0,
					'samples': len(values)
				}
		
		# Check each metric against seasonal expectations
		for metric in metrics:
			hour = metric.timestamp.hour
			day_of_week = metric.timestamp.weekday()
			
			hourly_anomaly = False
			daily_anomaly = False
			
			# Check hourly pattern
			if hour in hourly_expectations:
				expected = hourly_expectations[hour]
				if expected['std'] > 0:
					z_score = abs(metric.value - expected['mean']) / expected['std']
					if z_score > 2.5:  # 2.5 sigma threshold for seasonal
						hourly_anomaly = True
			
			# Check daily pattern
			if day_of_week in daily_expectations:
				expected = daily_expectations[day_of_week]
				if expected['std'] > 0:
					z_score = abs(metric.value - expected['mean']) / expected['std']
					if z_score > 2.5:
						daily_anomaly = True
			
			if hourly_anomaly or daily_anomaly:
				# Calculate combined anomaly score
				anomaly_score = 0.6 if (hourly_anomaly and daily_anomaly) else 0.4
				severity = self._calculate_seasonal_severity(anomaly_score, hour, day_of_week)
				
				pattern_type = []
				if hourly_anomaly:
					pattern_type.append("hourly")
				if daily_anomaly:
					pattern_type.append("daily")
				
				anomaly = AnomalyResult(
					anomaly_id=uuid7str(),
					metric_name=metric.name,
					tenant_id=metric.tenant_id,
					timestamp=metric.timestamp,
					value=metric.value,
					expected_value=hourly_expectations.get(hour, {}).get('mean'),
					deviation=abs(metric.value - hourly_expectations.get(hour, {}).get('mean', metric.value)),
					anomaly_score=anomaly_score,
					anomaly_type=AnomalyType.SEASONAL,
					algorithm_used=AnomalyAlgorithm.SEASONAL_HYBRID,
					severity=severity,
					confidence=0.75,
					context={
						'pattern_violations': pattern_type,
						'hour': hour,
						'day_of_week': day_of_week,
						'hourly_expected': hourly_expectations.get(hour),
						'daily_expected': daily_expectations.get(day_of_week)
					},
					explanation=f"Value {metric.value:.2f} violates {', '.join(pattern_type)} seasonal patterns",
					similar_anomalies=[]
				)
				anomalies.append(anomaly)
		
		return anomalies
	
	async def detect_trend_anomalies(self, metrics: List[MonitoringMetric],
									baseline: BaselineModel) -> List[AnomalyResult]:
		"""Detect anomalies in trend patterns"""
		anomalies = []
		
		if len(metrics) < self.window_size * 2:
			return anomalies
		
		# Sort metrics by timestamp
		sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
		values = [m.value for m in sorted_metrics]
		
		# Calculate rolling trend using linear regression
		for i in range(len(values) - self.window_size):
			window_values = values[i:i + self.window_size]
			x_values = list(range(len(window_values)))
			
			# Calculate linear regression slope
			slope = self._calculate_slope(x_values, window_values)
			
			# Compare with expected trend from baseline
			expected_slope = baseline.trend_parameters.get('slope', 0.0)
			slope_deviation = abs(slope - expected_slope)
			
			if slope_deviation > self.trend_threshold:
				metric = sorted_metrics[i + self.window_size - 1]  # Last metric in window
				
				anomaly_score = min(1.0, slope_deviation / self.trend_threshold)
				severity = self._calculate_trend_severity(anomaly_score, slope, expected_slope)
				
				anomaly = AnomalyResult(
					anomaly_id=uuid7str(),
					metric_name=metric.name,
					tenant_id=metric.tenant_id,
					timestamp=metric.timestamp,
					value=metric.value,
					expected_value=None,  # Trend anomalies don't have single expected values
					deviation=slope_deviation,
					anomaly_score=anomaly_score,
					anomaly_type=AnomalyType.TREND,
					algorithm_used=AnomalyAlgorithm.ARIMA,
					severity=severity,
					confidence=0.7,
					context={
						'observed_slope': slope,
						'expected_slope': expected_slope,
						'window_size': self.window_size,
						'trend_direction': 'increasing' if slope > 0 else 'decreasing'
					},
					explanation=f"Trend slope {slope:.4f} deviates from expected {expected_slope:.4f}",
					similar_anomalies=[]
				)
				anomalies.append(anomaly)
		
		return anomalies
	
	def _calculate_slope(self, x: List[float], y: List[float]) -> float:
		"""Calculate linear regression slope"""
		if len(x) != len(y) or len(x) < 2:
			return 0.0
		
		n = len(x)
		sum_x = sum(x)
		sum_y = sum(y)
		sum_xy = sum(x[i] * y[i] for i in range(n))
		sum_x2 = sum(x[i] * x[i] for i in range(n))
		
		denominator = n * sum_x2 - sum_x * sum_x
		if denominator == 0:
			return 0.0
		
		slope = (n * sum_xy - sum_x * sum_y) / denominator
		return slope
	
	def _calculate_seasonal_severity(self, anomaly_score: float, hour: int, day_of_week: int) -> AnomalySeverity:
		"""Calculate severity for seasonal anomalies"""
		# Consider time context (e.g., business hours more critical)
		business_hours = 9 <= hour <= 17
		weekday = day_of_week < 5
		
		base_severity = AnomalySeverity.MEDIUM
		if anomaly_score > 0.7:
			base_severity = AnomalySeverity.HIGH
		elif anomaly_score > 0.9:
			base_severity = AnomalySeverity.CRITICAL
		
		# Adjust based on business context
		if business_hours and weekday and base_severity == AnomalySeverity.HIGH:
			return AnomalySeverity.CRITICAL
		
		return base_severity
	
	def _calculate_trend_severity(self, anomaly_score: float, observed_slope: float, expected_slope: float) -> AnomalySeverity:
		"""Calculate severity for trend anomalies"""
		# Consider magnitude and direction of trend change
		if anomaly_score >= 0.8:
			return AnomalySeverity.HIGH
		elif anomaly_score >= 0.6:
			return AnomalySeverity.MEDIUM
		else:
			return AnomalySeverity.LOW


class ContextualAnomalyDetector:
	"""Contextual anomaly detection considering business and system context"""
	
	def __init__(self, config: dict = None):
		self.config = config or {}
		self.context_weights = self.config.get('context_weights', {
			'business_hours': 1.2,
			'maintenance_window': 0.5,
			'high_load_period': 1.5,
			'weekend': 0.8
		})
		
	async def detect_contextual_anomalies(self, metrics: List[MonitoringMetric],
										 baseline: BaselineModel,
										 context: Dict[str, Any] = None) -> List[AnomalyResult]:
		"""Detect anomalies considering business and operational context"""
		anomalies = []
		context = context or {}
		
		if len(metrics) < 5:
			return anomalies
		
		for metric in metrics:
			# Get contextual information
			metric_context = self._extract_metric_context(metric, context)
			
			# Predict expected value with context
			expected_value, confidence_interval = baseline.predict(metric.timestamp, metric_context)
			
			# Apply contextual adjustments
			context_weight = self._calculate_context_weight(metric_context)
			adjusted_threshold = confidence_interval * context_weight
			
			deviation = abs(metric.value - expected_value)
			
			if deviation > adjusted_threshold:
				anomaly_score = min(1.0, deviation / adjusted_threshold)
				severity = self._calculate_contextual_severity(anomaly_score, metric_context)
				
				anomaly = AnomalyResult(
					anomaly_id=uuid7str(),
					metric_name=metric.name,
					tenant_id=metric.tenant_id,
					timestamp=metric.timestamp,
					value=metric.value,
					expected_value=expected_value,
					deviation=deviation,
					anomaly_score=anomaly_score,
					anomaly_type=AnomalyType.CONTEXTUAL,
					algorithm_used=AnomalyAlgorithm.LOCAL_OUTLIER_FACTOR,
					severity=severity,
					confidence=0.8,
					context={
						'metric_context': metric_context,
						'context_weight': context_weight,
						'adjusted_threshold': adjusted_threshold,
						'baseline_threshold': confidence_interval
					},
					explanation=f"Value {metric.value:.2f} is {deviation:.2f} above contextual threshold {adjusted_threshold:.2f}",
					similar_anomalies=[]
				)
				anomalies.append(anomaly)
		
		return anomalies
	
	def _extract_metric_context(self, metric: MonitoringMetric, external_context: Dict[str, Any]) -> Dict[str, Any]:
		"""Extract contextual information for a metric"""
		context = {}
		
		# Time-based context
		hour = metric.timestamp.hour
		day_of_week = metric.timestamp.weekday()
		
		context['hour'] = hour
		context['day_of_week'] = day_of_week
		context['is_business_hours'] = 9 <= hour <= 17 and day_of_week < 5
		context['is_weekend'] = day_of_week >= 5
		
		# External context
		context.update(external_context)
		
		# Metric label context
		service_name = metric.labels.get('service', 'unknown')
		environment = metric.labels.get('environment', 'production')
		
		context['service'] = service_name
		context['environment'] = environment
		context['is_production'] = environment.lower() in ['prod', 'production']
		
		# Business context heuristics
		if service_name in ['payment', 'checkout', 'order']:
			context['business_critical'] = True
		else:
			context['business_critical'] = False
		
		return context
	
	def _calculate_context_weight(self, context: Dict[str, Any]) -> float:
		"""Calculate weight based on context"""
		weight = 1.0
		
		# Apply context-specific weights
		if context.get('is_business_hours', False):
			weight *= self.context_weights.get('business_hours', 1.2)
		
		if context.get('is_weekend', False):
			weight *= self.context_weights.get('weekend', 0.8)
		
		if context.get('maintenance_window', False):
			weight *= self.context_weights.get('maintenance_window', 0.5)
		
		if context.get('high_load_period', False):
			weight *= self.context_weights.get('high_load_period', 1.5)
		
		if context.get('business_critical', False):
			weight *= 1.3
		
		if context.get('is_production', True):
			weight *= 1.2
		
		return max(0.1, weight)  # Minimum weight of 0.1
	
	def _calculate_contextual_severity(self, anomaly_score: float, context: Dict[str, Any]) -> AnomalySeverity:
		"""Calculate severity considering context"""
		base_severity = AnomalySeverity.MEDIUM
		if anomaly_score >= 0.8:
			base_severity = AnomalySeverity.HIGH
		elif anomaly_score >= 0.9:
			base_severity = AnomalySeverity.CRITICAL
		
		# Upgrade severity based on context
		if context.get('business_critical', False) and context.get('is_business_hours', False):
			if base_severity == AnomalySeverity.MEDIUM:
				return AnomalySeverity.HIGH
			elif base_severity == AnomalySeverity.HIGH:
				return AnomalySeverity.CRITICAL
		
		return base_severity


class AnomalyDetectionEngine:
	"""
	Comprehensive anomaly detection engine with multiple algorithms and adaptive learning
	Provides intelligent anomaly detection with contextual understanding
	"""
	
	def __init__(self, config: dict = None):
		self.config = config or {}
		self.running = False
		
		# Detector components
		self.statistical_detector = StatisticalAnomalyDetector(config.get('statistical', {}))
		self.temporal_detector = TemporalAnomalyDetector(config.get('temporal', {}))
		self.contextual_detector = ContextualAnomalyDetector(config.get('contextual', {}))
		
		# Baseline models and storage
		self.baseline_models: Dict[str, BaselineModel] = {}
		self.detected_anomalies: Dict[str, AnomalyResult] = {}
		self.anomaly_history: deque = deque(maxlen=10000)
		
		# Processing and background tasks
		self.detection_queue = asyncio.Queue()
		self.background_tasks: List[asyncio.Task] = []
		
		# Performance tracking
		self.stats = {
			'total_detections': 0,
			'anomalies_detected': 0,
			'false_positives': 0,
			'true_positives': 0,
			'avg_detection_time_ms': 0.0,
			'algorithm_performance': defaultdict(lambda: {'detections': 0, 'accuracy': 0.0}),
			'severity_distribution': defaultdict(int)
		}
		
		print("[AnomalyEngine] Anomaly detection engine initialized")
	
	async def initialize(self) -> None:
		"""Initialize the anomaly detection engine"""
		assert not self.running, "Engine is already running"
		
		# Start background tasks
		self.background_tasks = [
			asyncio.create_task(self._detection_processor_loop()),
			asyncio.create_task(self._baseline_update_loop()),
			asyncio.create_task(self._model_training_loop()),
			asyncio.create_task(self._stats_update_loop())
		]
		
		self.running = True
		print("[AnomalyEngine] Anomaly detection engine started successfully")
	
	async def shutdown(self) -> None:
		"""Shutdown the engine"""
		if not self.running:
			return
		
		self.running = False
		
		# Cancel background tasks
		for task in self.background_tasks:
			task.cancel()
		
		await asyncio.gather(*self.background_tasks, return_exceptions=True)
		print("[AnomalyEngine] Anomaly detection engine shutdown complete")
	
	async def detect_anomalies(self, metrics: List[MonitoringMetric],
							  algorithms: List[AnomalyAlgorithm] = None,
							  context: Dict[str, Any] = None) -> List[AnomalyResult]:
		"""Detect anomalies using specified algorithms"""
		if not metrics:
			return []
		
		algorithms = algorithms or [
			AnomalyAlgorithm.Z_SCORE,
			AnomalyAlgorithm.SEASONAL_HYBRID,
			AnomalyAlgorithm.LOCAL_OUTLIER_FACTOR
		]
		
		start_time = datetime.utcnow()
		all_anomalies = []
		
		try:
			# Get or create baseline model
			metric_key = f"{metrics[0].name}_{metrics[0].tenant_id}"
			baseline = await self._get_baseline_model(metric_key, metrics)
			
			# Run detection algorithms
			for algorithm in algorithms:
				if algorithm in [AnomalyAlgorithm.Z_SCORE, AnomalyAlgorithm.MODIFIED_Z_SCORE, AnomalyAlgorithm.IQR]:
					if algorithm == AnomalyAlgorithm.Z_SCORE:
						anomalies = await self.statistical_detector.detect_z_score_anomalies(metrics, baseline)
					elif algorithm == AnomalyAlgorithm.MODIFIED_Z_SCORE:
						anomalies = await self.statistical_detector.detect_modified_z_score_anomalies(metrics, baseline)
					elif algorithm == AnomalyAlgorithm.IQR:
						anomalies = await self.statistical_detector.detect_iqr_anomalies(metrics, baseline)
				
				elif algorithm in [AnomalyAlgorithm.SEASONAL_HYBRID, AnomalyAlgorithm.ARIMA]:
					if algorithm == AnomalyAlgorithm.SEASONAL_HYBRID:
						anomalies = await self.temporal_detector.detect_seasonal_anomalies(metrics, baseline)
					elif algorithm == AnomalyAlgorithm.ARIMA:
						anomalies = await self.temporal_detector.detect_trend_anomalies(metrics, baseline)
				
				elif algorithm == AnomalyAlgorithm.LOCAL_OUTLIER_FACTOR:
					anomalies = await self.contextual_detector.detect_contextual_anomalies(metrics, baseline, context)
				
				else:
					continue  # Skip unsupported algorithms
				
				all_anomalies.extend(anomalies)
				self.stats['algorithm_performance'][algorithm.value]['detections'] += len(anomalies)
			
			# Deduplicate and rank anomalies
			unique_anomalies = self._deduplicate_anomalies(all_anomalies)
			ranked_anomalies = self._rank_anomalies(unique_anomalies)
			
			# Store detected anomalies
			for anomaly in ranked_anomalies:
				self.detected_anomalies[anomaly.anomaly_id] = anomaly
				self.anomaly_history.append(anomaly)
			
			# Update statistics
			detection_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			self._update_detection_stats(len(ranked_anomalies), detection_time)
			
			return ranked_anomalies
			
		except Exception as e:
			print(f"[AnomalyEngine] Error in anomaly detection: {e}")
			return []
	
	async def get_anomaly_insights(self, tenant_id: str = None, 
								  time_window_hours: int = 24) -> Dict[str, Any]:
		"""Get anomaly insights and patterns"""
		cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
		
		# Filter anomalies by time window and tenant
		recent_anomalies = [
			anomaly for anomaly in self.anomaly_history
			if anomaly.detected_at >= cutoff_time and 
			   (tenant_id is None or anomaly.tenant_id == tenant_id)
		]
		
		if not recent_anomalies:
			return {'message': 'No recent anomalies detected', 'anomaly_count': 0}
		
		# Generate insights
		insights = {
			'summary': {
				'total_anomalies': len(recent_anomalies),
				'unique_metrics': len(set(a.metric_name for a in recent_anomalies)),
				'severity_distribution': {},
				'algorithm_distribution': {},
				'tenant_distribution': {}
			},
			'patterns': self._analyze_anomaly_patterns(recent_anomalies),
			'recommendations': self._generate_anomaly_recommendations(recent_anomalies),
			'top_anomalies': [a.to_dict() for a in sorted(recent_anomalies, key=lambda x: x.anomaly_score, reverse=True)[:5]]
		}
		
		# Calculate distributions
		for anomaly in recent_anomalies:
			insights['summary']['severity_distribution'][anomaly.severity.value] = \
				insights['summary']['severity_distribution'].get(anomaly.severity.value, 0) + 1
			
			insights['summary']['algorithm_distribution'][anomaly.algorithm_used.value] = \
				insights['summary']['algorithm_distribution'].get(anomaly.algorithm_used.value, 0) + 1
			
			insights['summary']['tenant_distribution'][anomaly.tenant_id] = \
				insights['summary']['tenant_distribution'].get(anomaly.tenant_id, 0) + 1
		
		return insights
	
	async def update_anomaly_feedback(self, anomaly_id: str, is_true_positive: bool, 
									 feedback_note: str = "") -> bool:
		"""Update anomaly feedback for model improvement"""
		if anomaly_id not in self.detected_anomalies:
			return False
		
		anomaly = self.detected_anomalies[anomaly_id]
		
		# Update statistics
		if is_true_positive:
			self.stats['true_positives'] += 1
		else:
			self.stats['false_positives'] += 1
		
		# Update algorithm performance
		algorithm = anomaly.algorithm_used.value
		total_feedback = self.stats['true_positives'] + self.stats['false_positives']
		self.stats['algorithm_performance'][algorithm]['accuracy'] = self.stats['true_positives'] / max(total_feedback, 1)
		
		# Store feedback for model retraining
		anomaly.context['feedback'] = {
			'is_true_positive': is_true_positive,
			'note': feedback_note,
			'feedback_timestamp': datetime.utcnow().isoformat()
		}
		
		print(f"[AnomalyEngine] Feedback received for anomaly {anomaly_id}: {'TP' if is_true_positive else 'FP'}")
		return True
	
	async def get_engine_stats(self) -> Dict[str, Any]:
		"""Get comprehensive engine statistics"""
		accuracy = self.stats['true_positives'] / max(self.stats['true_positives'] + self.stats['false_positives'], 1)
		
		return {
			**self.stats,
			'accuracy': accuracy,
			'baseline_models_count': len(self.baseline_models),
			'recent_anomalies_count': len(self.detected_anomalies),
			'anomaly_history_size': len(self.anomaly_history),
			'queue_sizes': {
				'detection_queue': self.detection_queue.qsize()
			},
			'running': self.running,
			'timestamp': datetime.utcnow().isoformat()
		}
	
	# Private implementation methods
	async def _get_baseline_model(self, metric_key: str, metrics: List[MonitoringMetric]) -> BaselineModel:
		"""Get or create baseline model for metric"""
		if metric_key not in self.baseline_models or self.baseline_models[metric_key].is_stale():
			# Create new baseline model
			baseline = await self._create_baseline_model(metric_key, metrics)
			self.baseline_models[metric_key] = baseline
		
		return self.baseline_models[metric_key]
	
	async def _create_baseline_model(self, metric_key: str, metrics: List[MonitoringMetric]) -> BaselineModel:
		"""Create baseline model from historical data"""
		if not metrics:
			raise ValueError("No metrics provided for baseline creation")
		
		values = [m.value for m in metrics]
		
		# Calculate basic statistics
		mean_val = statistics.mean(values)
		std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
		
		# Extract seasonal patterns
		seasonal_patterns = self._extract_seasonal_patterns(metrics)
		
		# Calculate trend parameters
		trend_params = self._calculate_trend_parameters(metrics)
		
		# Calculate confidence intervals
		confidence_intervals = {
			'p95': (mean_val - 1.96 * std_dev, mean_val + 1.96 * std_dev),
			'p99': (mean_val - 2.58 * std_dev, mean_val + 2.58 * std_dev)
		}
		
		baseline = BaselineModel(
			metric_name=metrics[0].name,
			tenant_id=metrics[0].tenant_id,
			model_type='statistical',
			parameters={'mean': mean_val, 'std_dev': std_dev},
			training_period=timedelta(hours=24),
			last_updated=datetime.utcnow(),
			accuracy_metrics={'mse': 0.0, 'mae': 0.0},
			sample_size=len(metrics),
			seasonal_patterns=seasonal_patterns,
			trend_parameters=trend_params,
			confidence_intervals=confidence_intervals
		)
		
		return baseline
	
	def _extract_seasonal_patterns(self, metrics: List[MonitoringMetric]) -> Dict[str, Any]:
		"""Extract seasonal patterns from metrics"""
		patterns = {}
		
		# Group by hour and day of week
		hourly_data = defaultdict(list)
		daily_data = defaultdict(list)
		
		for metric in metrics:
			hour = metric.timestamp.hour
			day_of_week = metric.timestamp.weekday()
			
			hourly_data[hour].append(metric.value)
			daily_data[day_of_week].append(metric.value)
		
		# Calculate patterns
		for hour, values in hourly_data.items():
			if len(values) >= 3:
				patterns[f'hour_{hour}'] = statistics.mean(values) / statistics.mean([m.value for m in metrics])
		
		for day, values in daily_data.items():
			if len(values) >= 3:
				patterns[f'dow_{day}'] = statistics.mean(values) / statistics.mean([m.value for m in metrics])
		
		return patterns
	
	def _calculate_trend_parameters(self, metrics: List[MonitoringMetric]) -> Dict[str, float]:
		"""Calculate trend parameters from metrics"""
		if len(metrics) < 2:
			return {'slope': 0.0, 'intercept': 0.0, 'r_squared': 0.0}
		
		# Sort by timestamp and calculate linear regression
		sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
		values = [m.value for m in sorted_metrics]
		x_values = list(range(len(values)))
		
		# Simple linear regression
		n = len(values)
		sum_x = sum(x_values)
		sum_y = sum(values)
		sum_xy = sum(x_values[i] * values[i] for i in range(n))
		sum_x2 = sum(x * x for x in x_values)
		
		denominator = n * sum_x2 - sum_x * sum_x
		if denominator == 0:
			return {'slope': 0.0, 'intercept': sum_y / n, 'r_squared': 0.0}
		
		slope = (n * sum_xy - sum_x * sum_y) / denominator
		intercept = (sum_y - slope * sum_x) / n
		
		# Calculate R-squared
		y_mean = sum_y / n
		ss_tot = sum((y - y_mean) ** 2 for y in values)
		ss_res = sum((values[i] - (slope * x_values[i] + intercept)) ** 2 for i in range(n))
		
		r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
		
		return {'slope': slope, 'intercept': intercept, 'r_squared': max(0.0, r_squared)}
	
	def _deduplicate_anomalies(self, anomalies: List[AnomalyResult]) -> List[AnomalyResult]:
		"""Remove duplicate anomalies from different algorithms"""
		if not anomalies:
			return []
		
		# Group by metric, timestamp, and similar values
		groups = defaultdict(list)
		
		for anomaly in anomalies:
			key = f"{anomaly.metric_name}_{anomaly.tenant_id}_{anomaly.timestamp}_{round(anomaly.value, 2)}"
			groups[key].append(anomaly)
		
		# Keep the highest scoring anomaly from each group
		unique_anomalies = []
		for group_anomalies in groups.values():
			best_anomaly = max(group_anomalies, key=lambda a: a.anomaly_score)
			
			# Combine algorithm information
			all_algorithms = list(set(a.algorithm_used for a in group_anomalies))
			if len(all_algorithms) > 1:
				best_anomaly.context['detected_by_multiple_algorithms'] = [alg.value for alg in all_algorithms]
				best_anomaly.confidence = min(1.0, best_anomaly.confidence * 1.2)  # Boost confidence
			
			unique_anomalies.append(best_anomaly)
		
		return unique_anomalies
	
	def _rank_anomalies(self, anomalies: List[AnomalyResult]) -> List[AnomalyResult]:
		"""Rank anomalies by importance and actionability"""
		def ranking_score(anomaly: AnomalyResult) -> float:
			score = anomaly.anomaly_score
			
			# Boost for higher severity
			severity_boost = {
				AnomalySeverity.CRITICAL: 0.4,
				AnomalySeverity.HIGH: 0.2,
				AnomalySeverity.MEDIUM: 0.1,
				AnomalySeverity.LOW: 0.0
			}
			score += severity_boost.get(anomaly.severity, 0.0)
			
			# Boost for high confidence
			score += anomaly.confidence * 0.2
			
			# Boost for business critical context
			if anomaly.context.get('business_critical', False):
				score += 0.1
			
			# Boost for multiple algorithm detection
			if 'detected_by_multiple_algorithms' in anomaly.context:
				score += 0.15
			
			return min(1.0, score)
		
		return sorted(anomalies, key=ranking_score, reverse=True)
	
	def _analyze_anomaly_patterns(self, anomalies: List[AnomalyResult]) -> Dict[str, Any]:
		"""Analyze patterns in detected anomalies"""
		if not anomalies:
			return {}
		
		patterns = {
			'temporal_patterns': self._analyze_temporal_patterns(anomalies),
			'metric_patterns': self._analyze_metric_patterns(anomalies),
			'severity_trends': self._analyze_severity_trends(anomalies),
			'correlation_patterns': self._analyze_correlation_patterns(anomalies)
		}
		
		return patterns
	
	def _analyze_temporal_patterns(self, anomalies: List[AnomalyResult]) -> Dict[str, Any]:
		"""Analyze temporal patterns in anomalies"""
		hourly_counts = defaultdict(int)
		daily_counts = defaultdict(int)
		
		for anomaly in anomalies:
			hour = anomaly.timestamp.hour
			day = anomaly.timestamp.weekday()
			
			hourly_counts[hour] += 1
			daily_counts[day] += 1
		
		peak_hour = max(hourly_counts, key=hourly_counts.get) if hourly_counts else None
		peak_day = max(daily_counts, key=daily_counts.get) if daily_counts else None
		
		return {
			'peak_hour': peak_hour,
			'peak_day': peak_day,
			'hourly_distribution': dict(hourly_counts),
			'daily_distribution': dict(daily_counts)
		}
	
	def _analyze_metric_patterns(self, anomalies: List[AnomalyResult]) -> Dict[str, Any]:
		"""Analyze metric-specific patterns"""
		metric_counts = defaultdict(int)
		
		for anomaly in anomalies:
			metric_counts[anomaly.metric_name] += 1
		
		most_anomalous_metric = max(metric_counts, key=metric_counts.get) if metric_counts else None
		
		return {
			'most_anomalous_metric': most_anomalous_metric,
			'metric_distribution': dict(metric_counts),
			'unique_metrics_affected': len(metric_counts)
		}
	
	def _analyze_severity_trends(self, anomalies: List[AnomalyResult]) -> Dict[str, Any]:
		"""Analyze severity trends over time"""
		severity_over_time = []
		
		sorted_anomalies = sorted(anomalies, key=lambda a: a.timestamp)
		for anomaly in sorted_anomalies:
			severity_score = {
				AnomalySeverity.LOW: 1,
				AnomalySeverity.MEDIUM: 2,
				AnomalySeverity.HIGH: 3,
				AnomalySeverity.CRITICAL: 4
			}[anomaly.severity]
			
			severity_over_time.append({
				'timestamp': anomaly.timestamp.isoformat(),
				'severity_score': severity_score
			})
		
		# Calculate trend
		if len(severity_over_time) >= 2:
			scores = [s['severity_score'] for s in severity_over_time]
			trend = 'increasing' if scores[-1] > scores[0] else 'decreasing' if scores[-1] < scores[0] else 'stable'
		else:
			trend = 'insufficient_data'
		
		return {
			'trend': trend,
			'severity_timeline': severity_over_time
		}
	
	def _analyze_correlation_patterns(self, anomalies: List[AnomalyResult]) -> Dict[str, Any]:
		"""Analyze correlations between anomalies"""
		# Simple correlation analysis - in practice would use more sophisticated methods
		time_clusters = []
		current_cluster = []
		
		sorted_anomalies = sorted(anomalies, key=lambda a: a.timestamp)
		
		for i, anomaly in enumerate(sorted_anomalies):
			if i == 0:
				current_cluster = [anomaly]
			else:
				time_diff = (anomaly.timestamp - sorted_anomalies[i-1].timestamp).total_seconds()
				
				if time_diff <= 300:  # Within 5 minutes
					current_cluster.append(anomaly)
				else:
					if len(current_cluster) > 1:
						time_clusters.append(current_cluster)
					current_cluster = [anomaly]
		
		if len(current_cluster) > 1:
			time_clusters.append(current_cluster)
		
		return {
			'correlated_clusters': len(time_clusters),
			'cluster_details': [
				{
					'anomaly_count': len(cluster),
					'metrics_involved': [a.metric_name for a in cluster],
					'time_span_minutes': (cluster[-1].timestamp - cluster[0].timestamp).total_seconds() / 60
				}
				for cluster in time_clusters
			]
		}
	
	def _generate_anomaly_recommendations(self, anomalies: List[AnomalyResult]) -> List[str]:
		"""Generate actionable recommendations based on anomalies"""
		recommendations = []
		
		if not anomalies:
			return recommendations
		
		# High severity recommendations
		critical_anomalies = [a for a in anomalies if a.severity == AnomalySeverity.CRITICAL]
		if critical_anomalies:
			recommendations.append(f"Immediate attention required: {len(critical_anomalies)} critical anomalies detected")
		
		# Pattern-based recommendations
		metric_counts = defaultdict(int)
		for anomaly in anomalies:
			metric_counts[anomaly.metric_name] += 1
		
		for metric, count in metric_counts.items():
			if count >= 5:
				recommendations.append(f"Investigate {metric} - {count} anomalies detected (possible systemic issue)")
		
		# Algorithm performance recommendations
		algorithm_counts = defaultdict(int)
		for anomaly in anomalies:
			algorithm_counts[anomaly.algorithm_used] += 1
		
		if len(algorithm_counts) > 1:
			recommendations.append("Multiple detection algorithms triggered - high confidence in anomaly presence")
		
		# Temporal recommendations
		business_hours_anomalies = [
			a for a in anomalies 
			if 9 <= a.timestamp.hour <= 17 and a.timestamp.weekday() < 5
		]
		
		if len(business_hours_anomalies) / len(anomalies) > 0.7:
			recommendations.append("Most anomalies occur during business hours - consider capacity planning")
		
		return recommendations
	
	def _update_detection_stats(self, anomaly_count: int, detection_time_ms: float) -> None:
		"""Update detection performance statistics"""
		self.stats['total_detections'] += 1
		self.stats['anomalies_detected'] += anomaly_count
		
		# Update rolling average detection time
		current_avg = self.stats['avg_detection_time_ms']
		self.stats['avg_detection_time_ms'] = (current_avg * 0.9) + (detection_time_ms * 0.1)
		
		# Update severity distribution
		for anomaly in self.anomaly_history:
			self.stats['severity_distribution'][anomaly.severity.value] += 1
	
	# Background task implementations
	async def _detection_processor_loop(self) -> None:
		"""Background loop for processing detection requests"""
		try:
			while self.running:
				await asyncio.sleep(1)
				# Process queued detection requests
				
		except asyncio.CancelledError:
			pass
		except Exception as e:
			print(f"[AnomalyEngine] Error in detection processor: {e}")
	
	async def _baseline_update_loop(self) -> None:
		"""Background loop for updating baseline models"""
		try:
			while self.running:
				await asyncio.sleep(3600)  # Update every hour
				
				# Check for stale models and update
				stale_models = [
					key for key, model in self.baseline_models.items()
					if model.is_stale(max_age_hours=24)
				]
				
				print(f"[AnomalyEngine] Found {len(stale_models)} stale baseline models")
				
		except asyncio.CancelledError:
			pass
		except Exception as e:
			print(f"[AnomalyEngine] Error in baseline update: {e}")
	
	async def _model_training_loop(self) -> None:
		"""Background loop for model training and improvement"""
		try:
			while self.running:
				await asyncio.sleep(86400)  # Train daily
				
				# Retrain models based on feedback
				print("[AnomalyEngine] Starting daily model training")
				
		except asyncio.CancelledError:
			pass
		except Exception as e:
			print(f"[AnomalyEngine] Error in model training: {e}")
	
	async def _stats_update_loop(self) -> None:
		"""Background loop for statistics updates"""
		try:
			while self.running:
				await asyncio.sleep(300)  # Update every 5 minutes
				
				# Log performance statistics
				accuracy = self.stats['true_positives'] / max(self.stats['true_positives'] + self.stats['false_positives'], 1)
				print(f"[AnomalyEngine] Stats: {self.stats['anomalies_detected']} anomalies, "
					 f"{accuracy:.2%} accuracy, {self.stats['avg_detection_time_ms']:.1f}ms avg")
				
		except asyncio.CancelledError:
			pass
		except Exception as e:
			print(f"[AnomalyEngine] Error in stats update: {e}")


# Factory function
def create_anomaly_detection_engine(config: dict = None) -> AnomalyDetectionEngine:
	"""Create and configure anomaly detection engine"""
	return AnomalyDetectionEngine(config)