"""
APG Central Configuration - Advanced Analytics Engine

Real-time analytics, predictive insights, and intelligent reporting
for revolutionary configuration management platform.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
import statistics
from collections import defaultdict, deque

# Analytics libraries
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Time series analysis
try:
	import matplotlib.pyplot as plt
	import seaborn as sns
	PLOTTING_AVAILABLE = True
except ImportError:
	PLOTTING_AVAILABLE = False

# Advanced analytics
try:
	import networkx as nx
	NETWORKX_AVAILABLE = True
except ImportError:
	NETWORKX_AVAILABLE = False


class AnalyticsMetricType(Enum):
	"""Types of analytics metrics."""
	PERFORMANCE = "performance"
	USAGE = "usage"
	SECURITY = "security"
	COST = "cost"
	RELIABILITY = "reliability"
	COMPLIANCE = "compliance"
	USER_BEHAVIOR = "user_behavior"
	SYSTEM_HEALTH = "system_health"


class AggregationMethod(Enum):
	"""Data aggregation methods."""
	SUM = "sum"
	AVERAGE = "average"
	MEDIAN = "median"
	MIN = "min"
	MAX = "max"
	COUNT = "count"
	PERCENTILE_95 = "p95"
	PERCENTILE_99 = "p99"
	STANDARD_DEVIATION = "std"


class TimeGranularity(Enum):
	"""Time granularity for analytics."""
	MINUTE = "minute"
	HOUR = "hour"
	DAY = "day"
	WEEK = "week"
	MONTH = "month"


class TrendDirection(Enum):
	"""Trend direction indicators."""
	INCREASING = "increasing"
	DECREASING = "decreasing"
	STABLE = "stable"
	VOLATILE = "volatile"


@dataclass
class AnalyticsMetric:
	"""Analytics metric data point."""
	metric_type: AnalyticsMetricType
	name: str
	value: float
	timestamp: datetime
	tags: Dict[str, str]
	metadata: Dict[str, Any]


@dataclass
class TrendAnalysis:
	"""Trend analysis result."""
	metric_name: str
	direction: TrendDirection
	slope: float
	confidence: float
	correlation_coefficient: float
	projected_value: Optional[float]
	time_horizon: str
	significant_changes: List[Dict[str, Any]]


@dataclass
class AnomalyReport:
	"""Anomaly detection report."""
	metric_name: str
	anomalous_points: List[Tuple[datetime, float]]
	anomaly_type: str
	severity: str
	statistical_significance: float
	description: str
	recommended_actions: List[str]


@dataclass
class PerformanceReport:
	"""Performance analytics report."""
	report_id: str
	generated_at: datetime
	time_period: Tuple[datetime, datetime]
	summary_metrics: Dict[str, float]
	trend_analyses: List[TrendAnalysis]
	anomaly_reports: List[AnomalyReport]
	recommendations: List[str]
	configuration_insights: Dict[str, Any]


@dataclass
class UsagePattern:
	"""Configuration usage pattern."""
	pattern_id: str
	configuration_ids: List[str]
	usage_frequency: float
	peak_usage_times: List[str]
	user_segments: List[str]
	seasonal_trends: Dict[str, float]
	correlation_with_performance: float


class CentralConfigurationAnalytics:
	"""Advanced analytics engine for configuration management."""
	
	def __init__(self, redis_client=None, retention_days: int = 90):
		"""Initialize analytics engine."""
		self.redis_client = redis_client
		self.retention_days = retention_days
		
		# Time series data storage
		self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
		self.aggregated_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
		
		# Analytics models
		self.trend_models: Dict[str, Any] = {}
		self.anomaly_detectors: Dict[str, Any] = {}
		self.pattern_analyzers: Dict[str, Any] = {}
		
		# Real-time analytics state
		self.streaming_analytics: Dict[str, Any] = {}
		self.alert_thresholds: Dict[str, Dict[str, float]] = {}
		
		# Performance tracking
		self.performance_baselines: Dict[str, Dict[str, float]] = {}
		self.sla_targets: Dict[str, Dict[str, float]] = {}
		
		# Usage analytics
		self.usage_patterns: Dict[str, UsagePattern] = {}
		self.user_behavior_models: Dict[str, Any] = {}
		
		print("📊 Analytics engine initialized")
	
	# ==================== Metrics Collection ====================
	
	async def record_metric(
		self,
		metric_type: AnalyticsMetricType,
		name: str,
		value: float,
		timestamp: Optional[datetime] = None,
		tags: Optional[Dict[str, str]] = None,
		metadata: Optional[Dict[str, Any]] = None
	):
		"""Record a single analytics metric."""
		if timestamp is None:
			timestamp = datetime.now(timezone.utc)
		
		metric = AnalyticsMetric(
			metric_type=metric_type,
			name=name,
			value=value,
			timestamp=timestamp,
			tags=tags or {},
			metadata=metadata or {}
		)
		
		# Store in buffer for real-time processing
		metric_key = f"{metric_type.value}:{name}"
		self.metrics_buffer[metric_key].append(metric)
		
		# Store in Redis for persistence (if available)
		if self.redis_client:
			await self._store_metric_in_redis(metric)
		
		# Update real-time analytics
		await self._update_streaming_analytics(metric)
	
	async def _store_metric_in_redis(self, metric: AnalyticsMetric):
		"""Store metric in Redis with TTL."""
		try:
			metric_key = f"metrics:{metric.metric_type.value}:{metric.name}"
			timestamp_key = int(metric.timestamp.timestamp())
			
			# Store as time series
			await self.redis_client.zadd(
				metric_key,
				{json.dumps({
					'value': metric.value,
					'tags': metric.tags,
					'metadata': metric.metadata
				}): timestamp_key}
			)
			
			# Set TTL for data retention
			await self.redis_client.expire(metric_key, self.retention_days * 24 * 3600)
			
		except Exception as e:
			print(f"Failed to store metric in Redis: {e}")
	
	async def _update_streaming_analytics(self, metric: AnalyticsMetric):
		"""Update real-time streaming analytics."""
		metric_key = f"{metric.metric_type.value}:{metric.name}"
		
		if metric_key not in self.streaming_analytics:
			self.streaming_analytics[metric_key] = {
				'current_value': metric.value,
				'moving_average': metric.value,
				'min_value': metric.value,
				'max_value': metric.value,
				'value_count': 1,
				'last_updated': metric.timestamp
			}
		else:
			analytics = self.streaming_analytics[metric_key]
			
			# Update current value
			analytics['current_value'] = metric.value
			analytics['last_updated'] = metric.timestamp
			
			# Update moving average (exponential)
			alpha = 0.1  # Smoothing factor
			analytics['moving_average'] = (
				alpha * metric.value + (1 - alpha) * analytics['moving_average']
			)
			
			# Update min/max
			analytics['min_value'] = min(analytics['min_value'], metric.value)
			analytics['max_value'] = max(analytics['max_value'], metric.value)
			analytics['value_count'] += 1
		
		# Check for threshold alerts
		await self._check_alert_thresholds(metric)
	
	async def _check_alert_thresholds(self, metric: AnalyticsMetric):
		"""Check if metric exceeds alert thresholds."""
		metric_key = f"{metric.metric_type.value}:{metric.name}"
		
		if metric_key in self.alert_thresholds:
			thresholds = self.alert_thresholds[metric_key]
			
			if 'critical_high' in thresholds and metric.value > thresholds['critical_high']:
				await self._trigger_alert('critical', metric, 'high_threshold')
			elif 'warning_high' in thresholds and metric.value > thresholds['warning_high']:
				await self._trigger_alert('warning', metric, 'high_threshold')
			elif 'critical_low' in thresholds and metric.value < thresholds['critical_low']:
				await self._trigger_alert('critical', metric, 'low_threshold')
			elif 'warning_low' in thresholds and metric.value < thresholds['warning_low']:
				await self._trigger_alert('warning', metric, 'low_threshold')
	
	async def _trigger_alert(self, severity: str, metric: AnalyticsMetric, alert_type: str):
		"""Trigger an analytics alert."""
		alert = {
			'severity': severity,
			'metric_type': metric.metric_type.value,
			'metric_name': metric.name,
			'metric_value': metric.value,
			'alert_type': alert_type,
			'timestamp': metric.timestamp.isoformat(),
			'tags': metric.tags
		}
		
		print(f"🚨 Analytics Alert [{severity.upper()}]: {metric.name} = {metric.value}")
		
		# In production, this would send to alerting system
		# await self._send_to_alerting_system(alert)
	
	# ==================== Performance Analytics ====================
	
	async def analyze_performance_trends(
		self,
		start_time: datetime,
		end_time: datetime,
		metrics: Optional[List[str]] = None
	) -> List[TrendAnalysis]:
		"""Analyze performance trends over time period."""
		if metrics is None:
			metrics = ['response_time', 'throughput', 'error_rate', 'cpu_usage', 'memory_usage']
		
		trend_analyses = []
		
		for metric_name in metrics:
			# Get metric data
			data_points = await self._get_metric_data(
				AnalyticsMetricType.PERFORMANCE,
				metric_name,
				start_time,
				end_time
			)
			
			if len(data_points) < 10:  # Need minimum data points
				continue
			
			# Perform trend analysis
			trend_analysis = await self._analyze_trend(metric_name, data_points)
			trend_analyses.append(trend_analysis)
		
		return trend_analyses
	
	async def _get_metric_data(
		self,
		metric_type: AnalyticsMetricType,
		metric_name: str,
		start_time: datetime,
		end_time: datetime
	) -> List[Tuple[datetime, float]]:
		"""Get metric data from storage."""
		data_points = []
		
		# Try Redis first
		if self.redis_client:
			try:
				metric_key = f"metrics:{metric_type.value}:{metric_name}"
				start_ts = int(start_time.timestamp())
				end_ts = int(end_time.timestamp())
				
				raw_data = await self.redis_client.zrangebyscore(
					metric_key, start_ts, end_ts, withscores=True
				)
				
				for data_json, timestamp in raw_data:
					data = json.loads(data_json)
					dt = datetime.fromtimestamp(timestamp, timezone.utc)
					data_points.append((dt, data['value']))
				
			except Exception as e:
				print(f"Failed to get data from Redis: {e}")
		
		# Fallback to in-memory buffer
		if not data_points:
			buffer_key = f"{metric_type.value}:{metric_name}"
			if buffer_key in self.metrics_buffer:
				for metric in self.metrics_buffer[buffer_key]:
					if start_time <= metric.timestamp <= end_time:
						data_points.append((metric.timestamp, metric.value))
		
		# Sort by timestamp
		data_points.sort(key=lambda x: x[0])
		return data_points
	
	async def _analyze_trend(
		self,
		metric_name: str,
		data_points: List[Tuple[datetime, float]]
	) -> TrendAnalysis:
		"""Analyze trend for a specific metric."""
		if len(data_points) < 2:
			return TrendAnalysis(
				metric_name=metric_name,
				direction=TrendDirection.STABLE,
				slope=0.0,
				confidence=0.0,
				correlation_coefficient=0.0,
				projected_value=None,
				time_horizon="",
				significant_changes=[]
			)
		
		# Extract values and timestamps
		timestamps = [dp[0] for dp in data_points]
		values = [dp[1] for dp in data_points]
		
		# Convert timestamps to numeric for linear regression
		start_time = timestamps[0]
		x_values = [(ts - start_time).total_seconds() for ts in timestamps]
		
		# Perform linear regression
		correlation_coeff = np.corrcoef(x_values, values)[0, 1] if len(x_values) > 1 else 0.0
		
		if len(x_values) > 1:
			slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, values)
		else:
			slope, r_value, p_value = 0.0, 0.0, 1.0
		
		# Determine trend direction
		if abs(slope) < 0.001:  # Very small slope
			direction = TrendDirection.STABLE
		elif slope > 0:
			direction = TrendDirection.INCREASING
		else:
			direction = TrendDirection.DECREASING
		
		# Check for volatility
		if len(values) > 3:
			volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
			if volatility > 0.3:  # High volatility threshold
				direction = TrendDirection.VOLATILE
		
		# Calculate confidence
		confidence = min(abs(r_value), 1.0) if p_value < 0.05 else 0.0
		
		# Project future value (1 hour ahead)
		projected_value = None
		if confidence > 0.5 and len(x_values) > 0:
			future_x = x_values[-1] + 3600  # 1 hour in seconds
			projected_value = slope * future_x + intercept
		
		# Detect significant changes
		significant_changes = await self._detect_significant_changes(data_points)
		
		return TrendAnalysis(
			metric_name=metric_name,
			direction=direction,
			slope=slope,
			confidence=confidence,
			correlation_coefficient=correlation_coeff,
			projected_value=projected_value,
			time_horizon="1 hour",
			significant_changes=significant_changes
		)
	
	async def _detect_significant_changes(
		self,
		data_points: List[Tuple[datetime, float]]
	) -> List[Dict[str, Any]]:
		"""Detect significant changes in the data."""
		if len(data_points) < 10:
			return []
		
		values = [dp[1] for dp in data_points]
		timestamps = [dp[0] for dp in data_points]
		
		significant_changes = []
		
		# Moving window analysis
		window_size = min(10, len(values) // 3)
		
		for i in range(window_size, len(values) - window_size):
			# Compare before and after windows
			before_window = values[i-window_size:i]
			after_window = values[i:i+window_size]
			
			before_mean = np.mean(before_window)
			after_mean = np.mean(after_window)
			
			# Calculate change magnitude
			if before_mean != 0:
				change_percentage = ((after_mean - before_mean) / before_mean) * 100
			else:
				change_percentage = 0
			
			# Check for significant change (>20% change)
			if abs(change_percentage) > 20:
				significant_changes.append({
					'timestamp': timestamps[i].isoformat(),
					'change_percentage': change_percentage,
					'before_value': before_mean,
					'after_value': after_mean,
					'change_type': 'increase' if change_percentage > 0 else 'decrease'
				})
		
		return significant_changes
	
	# ==================== Anomaly Detection Analytics ====================
	
	async def detect_performance_anomalies(
		self,
		start_time: datetime,
		end_time: datetime,
		sensitivity: float = 0.05
	) -> List[AnomalyReport]:
		"""Detect performance anomalies using statistical methods."""
		performance_metrics = ['response_time', 'throughput', 'error_rate', 'cpu_usage']
		anomaly_reports = []
		
		for metric_name in performance_metrics:
			data_points = await self._get_metric_data(
				AnalyticsMetricType.PERFORMANCE,
				metric_name,
				start_time,
				end_time
			)
			
			if len(data_points) < 50:  # Need sufficient data
				continue
			
			# Detect anomalies
			anomalies = await self._detect_statistical_anomalies(
				metric_name, data_points, sensitivity
			)
			
			if anomalies:
				anomaly_reports.append(anomalies)
		
		return anomaly_reports
	
	async def _detect_statistical_anomalies(
		self,
		metric_name: str,
		data_points: List[Tuple[datetime, float]],
		sensitivity: float
	) -> Optional[AnomalyReport]:
		"""Detect anomalies using statistical methods."""
		values = [dp[1] for dp in data_points]
		timestamps = [dp[0] for dp in data_points]
		
		# Calculate statistical thresholds
		mean_value = np.mean(values)
		std_value = np.std(values)
		
		# Z-score threshold based on sensitivity
		z_threshold = stats.norm.ppf(1 - sensitivity/2)  # Two-tailed test
		
		upper_threshold = mean_value + z_threshold * std_value
		lower_threshold = mean_value - z_threshold * std_value
		
		# Find anomalous points
		anomalous_points = []
		for timestamp, value in data_points:
			if value > upper_threshold or value < lower_threshold:
				anomalous_points.append((timestamp, value))
		
		if not anomalous_points:
			return None
		
		# Determine anomaly type and severity
		anomaly_type = "outlier"
		severity = "medium"
		
		# Check for pattern anomalies
		if len(anomalous_points) > len(data_points) * 0.1:  # More than 10% anomalous
			anomaly_type = "pattern_shift"
			severity = "high"
		
		# Calculate statistical significance
		z_scores = [(value - mean_value) / std_value for _, value in anomalous_points]
		max_z_score = max(abs(z) for z in z_scores)
		statistical_significance = 1 - stats.norm.cdf(max_z_score)
		
		# Generate description
		description = await self._generate_anomaly_description(
			metric_name, anomalous_points, mean_value, std_value
		)
		
		# Generate recommendations
		recommendations = await self._generate_anomaly_recommendations(
			metric_name, anomaly_type, severity
		)
		
		return AnomalyReport(
			metric_name=metric_name,
			anomalous_points=anomalous_points,
			anomaly_type=anomaly_type,
			severity=severity,
			statistical_significance=statistical_significance,
			description=description,
			recommended_actions=recommendations
		)
	
	async def _generate_anomaly_description(
		self,
		metric_name: str,
		anomalous_points: List[Tuple[datetime, float]],
		mean_value: float,
		std_value: float
	) -> str:
		"""Generate human-readable anomaly description."""
		num_anomalies = len(anomalous_points)
		
		if num_anomalies == 1:
			timestamp, value = anomalous_points[0]
			deviation = abs(value - mean_value) / std_value
			return f"Single anomaly detected in {metric_name} at {timestamp.strftime('%Y-%m-%d %H:%M:%S')} " \
				   f"with value {value:.2f} ({deviation:.1f} standard deviations from mean {mean_value:.2f})"
		else:
			max_value = max(value for _, value in anomalous_points)
			min_value = min(value for _, value in anomalous_points)
			return f"{num_anomalies} anomalies detected in {metric_name} " \
				   f"with values ranging from {min_value:.2f} to {max_value:.2f} " \
				   f"(normal range: {mean_value - 2*std_value:.2f} to {mean_value + 2*std_value:.2f})"
	
	async def _generate_anomaly_recommendations(
		self,
		metric_name: str,
		anomaly_type: str,
		severity: str
	) -> List[str]:
		"""Generate recommendations for addressing anomalies."""
		recommendations = []
		
		if metric_name == 'response_time':
			recommendations.extend([
				"Investigate database query performance",
				"Check for network latency issues",
				"Review application code for bottlenecks",
				"Consider scaling resources if needed"
			])
		
		elif metric_name == 'throughput':
			recommendations.extend([
				"Analyze request patterns and traffic spikes",
				"Check system capacity and resource utilization",
				"Review load balancing configuration",
				"Consider auto-scaling policies"
			])
		
		elif metric_name == 'error_rate':
			recommendations.extend([
				"Examine application logs for error patterns",
				"Check for recent deployments or configuration changes",
				"Review error handling and retry mechanisms",
				"Validate external service dependencies"
			])
		
		elif metric_name == 'cpu_usage':
			recommendations.extend([
				"Identify CPU-intensive processes",
				"Review application efficiency and optimization",
				"Consider vertical or horizontal scaling",
				"Check for resource contention issues"
			])
		
		# Add severity-specific recommendations
		if severity == 'high':
			recommendations.insert(0, "Immediate attention required - consider emergency response procedures")
		elif severity == 'critical':
			recommendations.insert(0, "CRITICAL: Implement immediate remediation measures")
		
		return recommendations
	
	# ==================== Usage Analytics ====================
	
	async def analyze_configuration_usage_patterns(
		self,
		configuration_ids: List[str],
		start_time: datetime,
		end_time: datetime
	) -> List[UsagePattern]:
		"""Analyze usage patterns for configurations."""
		usage_patterns = []
		
		for config_id in configuration_ids:
			# Get usage data
			usage_data = await self._get_configuration_usage_data(
				config_id, start_time, end_time
			)
			
			if not usage_data:
				continue
			
			# Analyze pattern
			pattern = await self._analyze_usage_pattern(config_id, usage_data)
			usage_patterns.append(pattern)
		
		return usage_patterns
	
	async def _get_configuration_usage_data(
		self,
		config_id: str,
		start_time: datetime,
		end_time: datetime
	) -> List[Dict[str, Any]]:
		"""Get usage data for a specific configuration."""
		# This would typically come from access logs, API calls, etc.
		# For demonstration, we'll generate sample data
		
		usage_data = []
		current_time = start_time
		
		while current_time < end_time:
			# Simulate hourly usage data
			usage_count = max(0, int(np.random.poisson(10) + 
									np.sin((current_time.hour / 24) * 2 * np.pi) * 5))
			
			usage_data.append({
				'timestamp': current_time,
				'access_count': usage_count,
				'unique_users': max(1, usage_count // 3),
				'response_time': np.random.normal(100, 20)
			})
			
			current_time += timedelta(hours=1)
		
		return usage_data
	
	async def _analyze_usage_pattern(
		self,
		config_id: str,
		usage_data: List[Dict[str, Any]]
	) -> UsagePattern:
		"""Analyze usage pattern for a configuration."""
		if not usage_data:
			return UsagePattern(
				pattern_id=f"pattern_{config_id}",
				configuration_ids=[config_id],
				usage_frequency=0.0,
				peak_usage_times=[],
				user_segments=[],
				seasonal_trends={},
				correlation_with_performance=0.0
			)
		
		# Calculate usage frequency (accesses per hour)
		total_accesses = sum(data['access_count'] for data in usage_data)
		total_hours = len(usage_data)
		usage_frequency = total_accesses / total_hours if total_hours > 0 else 0.0
		
		# Identify peak usage times
		hourly_usage = defaultdict(list)
		for data in usage_data:
			hour = data['timestamp'].hour
			hourly_usage[hour].append(data['access_count'])
		
		# Calculate average usage per hour
		avg_hourly_usage = {
			hour: np.mean(counts) 
			for hour, counts in hourly_usage.items()
		}
		
		# Find peak hours (top 25%)
		sorted_hours = sorted(avg_hourly_usage.items(), key=lambda x: x[1], reverse=True)
		peak_count = max(1, len(sorted_hours) // 4)
		peak_hours = [f"{hour:02d}:00" for hour, _ in sorted_hours[:peak_count]]
		
		# Analyze seasonal trends (simplified)
		seasonal_trends = {
			'morning': np.mean([avg_hourly_usage.get(h, 0) for h in [6, 7, 8, 9]]),
			'afternoon': np.mean([avg_hourly_usage.get(h, 0) for h in [12, 13, 14, 15]]),
			'evening': np.mean([avg_hourly_usage.get(h, 0) for h in [18, 19, 20, 21]]),
			'night': np.mean([avg_hourly_usage.get(h, 0) for h in [22, 23, 0, 1]])
		}
		
		# Calculate correlation with performance
		access_counts = [data['access_count'] for data in usage_data]
		response_times = [data['response_time'] for data in usage_data]
		
		correlation_with_performance = 0.0
		if len(access_counts) > 1 and len(response_times) > 1:
			correlation_with_performance = np.corrcoef(access_counts, response_times)[0, 1]
			if np.isnan(correlation_with_performance):
				correlation_with_performance = 0.0
		
		return UsagePattern(
			pattern_id=f"pattern_{config_id}_{int(datetime.now().timestamp())}",
			configuration_ids=[config_id],
			usage_frequency=usage_frequency,
			peak_usage_times=peak_hours,
			user_segments=["web_users", "api_users"],  # Simplified
			seasonal_trends=seasonal_trends,
			correlation_with_performance=correlation_with_performance
		)
	
	# ==================== Reporting ====================
	
	async def generate_performance_report(
		self,
		start_time: datetime,
		end_time: datetime,
		include_predictions: bool = True
	) -> PerformanceReport:
		"""Generate comprehensive performance report."""
		report_id = f"perf_report_{int(datetime.now().timestamp())}"
		
		# Get summary metrics
		summary_metrics = await self._calculate_summary_metrics(start_time, end_time)
		
		# Analyze trends
		trend_analyses = await self.analyze_performance_trends(start_time, end_time)
		
		# Detect anomalies
		anomaly_reports = await self.detect_performance_anomalies(start_time, end_time)
		
		# Generate recommendations
		recommendations = await self._generate_performance_recommendations(
			summary_metrics, trend_analyses, anomaly_reports
		)
		
		# Configuration insights
		configuration_insights = await self._analyze_configuration_impact(
			start_time, end_time
		)
		
		return PerformanceReport(
			report_id=report_id,
			generated_at=datetime.now(timezone.utc),
			time_period=(start_time, end_time),
			summary_metrics=summary_metrics,
			trend_analyses=trend_analyses,
			anomaly_reports=anomaly_reports,
			recommendations=recommendations,
			configuration_insights=configuration_insights
		)
	
	async def _calculate_summary_metrics(
		self,
		start_time: datetime,
		end_time: datetime
	) -> Dict[str, float]:
		"""Calculate summary performance metrics."""
		metrics = {}
		
		# Key performance metrics
		performance_metrics = ['response_time', 'throughput', 'error_rate', 'cpu_usage', 'memory_usage']
		
		for metric_name in performance_metrics:
			data_points = await self._get_metric_data(
				AnalyticsMetricType.PERFORMANCE,
				metric_name,
				start_time,
				end_time
			)
			
			if data_points:
				values = [dp[1] for dp in data_points]
				metrics[f"{metric_name}_avg"] = np.mean(values)
				metrics[f"{metric_name}_p95"] = np.percentile(values, 95)
				metrics[f"{metric_name}_p99"] = np.percentile(values, 99)
				metrics[f"{metric_name}_min"] = np.min(values)
				metrics[f"{metric_name}_max"] = np.max(values)
		
		return metrics
	
	async def _generate_performance_recommendations(
		self,
		summary_metrics: Dict[str, float],
		trend_analyses: List[TrendAnalysis],
		anomaly_reports: List[AnomalyReport]
	) -> List[str]:
		"""Generate performance recommendations based on analysis."""
		recommendations = []
		
		# Response time recommendations
		if 'response_time_p95' in summary_metrics:
			p95_response_time = summary_metrics['response_time_p95']
			if p95_response_time > 500:  # > 500ms is concerning
				recommendations.append(
					f"High response time detected (P95: {p95_response_time:.1f}ms). "
					"Consider optimizing database queries and implementing caching."
				)
		
		# Error rate recommendations
		if 'error_rate_avg' in summary_metrics:
			avg_error_rate = summary_metrics['error_rate_avg']
			if avg_error_rate > 1.0:  # > 1% error rate
				recommendations.append(
					f"Elevated error rate detected ({avg_error_rate:.2f}%). "
					"Review application logs and implement better error handling."
				)
		
		# Resource utilization recommendations
		if 'cpu_usage_avg' in summary_metrics:
			avg_cpu = summary_metrics['cpu_usage_avg']
			if avg_cpu > 80:
				recommendations.append(
					f"High CPU utilization ({avg_cpu:.1f}%). "
					"Consider scaling resources or optimizing CPU-intensive operations."
				)
		
		# Trend-based recommendations
		for trend in trend_analyses:
			if trend.direction == TrendDirection.INCREASING and trend.confidence > 0.7:
				if 'response_time' in trend.metric_name:
					recommendations.append(
						f"Increasing trend detected in {trend.metric_name}. "
						"Proactive optimization recommended to prevent performance degradation."
					)
		
		# Anomaly-based recommendations
		for anomaly in anomaly_reports:
			if anomaly.severity in ['high', 'critical']:
				recommendations.append(
					f"Critical anomaly in {anomaly.metric_name}: {anomaly.description}"
				)
		
		return recommendations
	
	async def _analyze_configuration_impact(
		self,
		start_time: datetime,
		end_time: datetime
	) -> Dict[str, Any]:
		"""Analyze impact of configuration changes on performance."""
		# This would correlate configuration changes with performance metrics
		# For demonstration, we'll provide sample insights
		
		return {
			'configuration_changes_detected': 3,
			'performance_impact_score': 0.2,  # Low impact
			'most_impactful_changes': [
				{
					'configuration': 'database_connection_pool',
					'change_type': 'parameter_update',
					'impact_score': 0.15,
					'description': 'Increased connection pool size improved throughput by 15%'
				}
			],
			'optimization_opportunities': [
				'Enable query result caching',
				'Implement connection pooling for Redis',
				'Optimize batch processing configuration'
			]
		}
	
	# ==================== Real-time Analytics Dashboard ====================
	
	async def get_real_time_dashboard_data(self) -> Dict[str, Any]:
		"""Get real-time analytics data for dashboard."""
		dashboard_data = {
			'timestamp': datetime.now(timezone.utc).isoformat(),
			'streaming_metrics': {},
			'alerts': [],
			'top_configurations': [],
			'system_health': {}
		}
		
		# Current streaming metrics
		for metric_key, analytics in self.streaming_analytics.items():
			dashboard_data['streaming_metrics'][metric_key] = {
				'current_value': analytics['current_value'],
				'moving_average': analytics['moving_average'],
				'min_value': analytics['min_value'],
				'max_value': analytics['max_value'],
				'last_updated': analytics['last_updated'].isoformat()
			}
		
		# Recent alerts (would come from alert system)
		dashboard_data['alerts'] = [
			{
				'severity': 'warning',
				'message': 'Response time above threshold',
				'timestamp': (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
			}
		]
		
		# System health summary
		dashboard_data['system_health'] = {
			'overall_status': 'healthy',
			'performance_score': 0.85,
			'availability': 0.999,
			'active_configurations': 142,
			'total_requests_last_hour': 15420
		}
		
		return dashboard_data
	
	# ==================== Configuration Analytics ====================
	
	async def set_alert_thresholds(
		self,
		metric_type: AnalyticsMetricType,
		metric_name: str,
		thresholds: Dict[str, float]
	):
		"""Set alert thresholds for a metric."""
		metric_key = f"{metric_type.value}:{metric_name}"
		self.alert_thresholds[metric_key] = thresholds
		
		print(f"🚨 Alert thresholds set for {metric_key}: {thresholds}")
	
	async def get_analytics_summary(self) -> Dict[str, Any]:
		"""Get analytics engine summary."""
		return {
			'total_metrics_tracked': len(self.metrics_buffer),
			'streaming_analytics_active': len(self.streaming_analytics),
			'alert_thresholds_configured': len(self.alert_thresholds),
			'data_retention_days': self.retention_days,
			'last_updated': datetime.now(timezone.utc).isoformat()
		}
	
	async def close(self):
		"""Clean up analytics engine resources."""
		self.metrics_buffer.clear()
		self.streaming_analytics.clear()
		print("📊 Analytics engine closed")


# ==================== Factory Functions ====================

async def create_analytics_engine(
	redis_client=None,
	retention_days: int = 90
) -> CentralConfigurationAnalytics:
	"""Create and initialize analytics engine."""
	engine = CentralConfigurationAnalytics(redis_client, retention_days)
	
	# Set up default alert thresholds
	await engine.set_alert_thresholds(
		AnalyticsMetricType.PERFORMANCE,
		'response_time',
		{'warning_high': 200, 'critical_high': 500}
	)
	
	await engine.set_alert_thresholds(
		AnalyticsMetricType.PERFORMANCE,
		'error_rate',
		{'warning_high': 1.0, 'critical_high': 5.0}
	)
	
	await engine.set_alert_thresholds(
		AnalyticsMetricType.PERFORMANCE,
		'cpu_usage',
		{'warning_high': 80, 'critical_high': 95}
	)
	
	print("📊 Analytics engine initialized with default thresholds")
	return engine