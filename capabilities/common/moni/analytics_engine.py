#!/usr/bin/env python3
"""
APG Monitoring - Analytics Engine
Real-time analytics processing with statistical analysis and trend detection

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from uuid6 import uuid7
def uuid7str() -> str: return str(uuid7())

from .models import MonitoringMetric, MonitoringQuery, DataRetentionPolicy


class AnalysisType(str, Enum):
	"""Types of analytics analysis"""
	STATISTICAL = "statistical"
	TREND = "trend"
	CORRELATION = "correlation"
	PERFORMANCE = "performance"
	ANOMALY = "anomaly"
	FORECAST = "forecast"
	BASELINE = "baseline"


class AggregationFunction(str, Enum):
	"""Statistical aggregation functions"""
	MEAN = "mean"
	MEDIAN = "median"
	MODE = "mode"
	SUM = "sum"
	COUNT = "count"
	MIN = "min"
	MAX = "max"
	STDDEV = "stddev"
	VARIANCE = "variance"
	PERCENTILE = "percentile"


class TrendDirection(str, Enum):
	"""Trend direction indicators"""
	INCREASING = "increasing"
	DECREASING = "decreasing"
	STABLE = "stable"
	VOLATILE = "volatile"
	UNKNOWN = "unknown"


@dataclass
class StatisticalSummary:
	"""Statistical summary of metrics"""
	metric_name: str
	tenant_id: str
	time_window: str
	sample_count: int
	mean: float
	median: float
	mode: Optional[float]
	min_value: float
	max_value: float
	std_deviation: float
	variance: float
	percentiles: Dict[int, float]  # e.g., {50: median, 95: p95, 99: p99}
	outliers: List[float]
	data_quality_score: float
	analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
	
	def to_dict(self) -> dict:
		"""Convert to dictionary representation"""
		return {
			'metric_name': self.metric_name,
			'tenant_id': self.tenant_id,
			'time_window': self.time_window,
			'sample_count': self.sample_count,
			'statistics': {
				'mean': self.mean,
				'median': self.median,
				'mode': self.mode,
				'min': self.min_value,
				'max': self.max_value,
				'std_dev': self.std_deviation,
				'variance': self.variance
			},
			'percentiles': self.percentiles,
			'outliers': self.outliers,
			'data_quality_score': self.data_quality_score,
			'analysis_timestamp': self.analysis_timestamp.isoformat()
		}


@dataclass
class TrendAnalysis:
	"""Trend analysis results"""
	metric_name: str
	tenant_id: str
	time_window: str
	direction: TrendDirection
	slope: float
	r_squared: float  # Correlation coefficient squared
	confidence: float  # 0.0 - 1.0
	rate_of_change: float  # per time unit
	trend_strength: float  # 0.0 - 1.0
	change_points: List[datetime]  # Points where trend changes
	seasonal_component: bool
	forecast_next_period: Optional[float]
	analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
	
	def to_dict(self) -> dict:
		"""Convert to dictionary representation"""
		return {
			'metric_name': self.metric_name,
			'tenant_id': self.tenant_id,
			'time_window': self.time_window,
			'trend': {
				'direction': self.direction.value,
				'slope': self.slope,
				'r_squared': self.r_squared,
				'confidence': self.confidence,
				'rate_of_change': self.rate_of_change,
				'strength': self.trend_strength
			},
			'change_points': [cp.isoformat() for cp in self.change_points],
			'seasonal_component': self.seasonal_component,
			'forecast_next_period': self.forecast_next_period,
			'analysis_timestamp': self.analysis_timestamp.isoformat()
		}


@dataclass
class CorrelationResult:
	"""Correlation analysis between metrics"""
	metric1_name: str
	metric2_name: str
	tenant_id: str
	correlation_coefficient: float  # Pearson correlation
	correlation_strength: str  # weak, moderate, strong
	p_value: float  # Statistical significance
	sample_size: int
	time_window: str
	lag_correlation: Dict[int, float]  # Correlation with different time lags
	causality_direction: Optional[str]  # metric1 -> metric2 or vice versa
	analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
	
	def to_dict(self) -> dict:
		"""Convert to dictionary representation"""
		return {
			'metrics': [self.metric1_name, self.metric2_name],
			'tenant_id': self.tenant_id,
			'correlation': {
				'coefficient': self.correlation_coefficient,
				'strength': self.correlation_strength,
				'p_value': self.p_value,
				'sample_size': self.sample_size
			},
			'time_window': self.time_window,
			'lag_correlation': self.lag_correlation,
			'causality_direction': self.causality_direction,
			'analysis_timestamp': self.analysis_timestamp.isoformat()
		}


class StatisticalAnalyzer:
	"""Advanced statistical analysis for monitoring metrics"""
	
	def __init__(self, config: dict = None):
		self.config = config or {}
		self.outlier_threshold = self.config.get('outlier_threshold', 2.5)  # Standard deviations
		self.min_sample_size = self.config.get('min_sample_size', 10)
		
	async def analyze_metrics_statistical(self, metrics: List[MonitoringMetric], 
										 time_window: str = "1h") -> StatisticalSummary:
		"""Perform comprehensive statistical analysis on metrics"""
		
		if len(metrics) < self.min_sample_size:
			raise ValueError(f"Insufficient data points: {len(metrics)} < {self.min_sample_size}")
		
		# Extract values and basic info
		values = [m.value for m in metrics]
		metric_name = metrics[0].name if metrics else "unknown"
		tenant_id = metrics[0].tenant_id if metrics else "unknown"
		
		# Calculate basic statistics
		mean_val = statistics.mean(values)
		median_val = statistics.median(values)
		min_val = min(values)
		max_val = max(values)
		
		# Standard deviation and variance
		std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
		variance = statistics.variance(values) if len(values) > 1 else 0.0
		
		# Mode (most common value) - handle case where no mode exists
		try:
			mode_val = statistics.mode(values)
		except statistics.StatisticsError:
			mode_val = None
		
		# Percentiles
		percentiles = {}
		if len(values) >= 4:  # Need at least 4 points for meaningful percentiles
			percentiles = {
				25: np.percentile(values, 25),
				50: median_val,
				75: np.percentile(values, 75),
				90: np.percentile(values, 90),
				95: np.percentile(values, 95),
				99: np.percentile(values, 99)
			}
		
		# Identify outliers using z-score method
		outliers = []
		if std_dev > 0:
			z_scores = [(v - mean_val) / std_dev for v in values]
			outliers = [values[i] for i, z in enumerate(z_scores) if abs(z) > self.outlier_threshold]
		
		# Data quality score based on completeness and consistency
		data_quality_score = self._calculate_data_quality_score(metrics, values, outliers)
		
		return StatisticalSummary(
			metric_name=metric_name,
			tenant_id=tenant_id,
			time_window=time_window,
			sample_count=len(values),
			mean=mean_val,
			median=median_val,
			mode=mode_val,
			min_value=min_val,
			max_value=max_val,
			std_deviation=std_dev,
			variance=variance,
			percentiles=percentiles,
			outliers=outliers,
			data_quality_score=data_quality_score
		)
	
	async def analyze_trend(self, metrics: List[MonitoringMetric], 
						   time_window: str = "1h") -> TrendAnalysis:
		"""Analyze trend patterns in time-series data"""
		
		if len(metrics) < self.min_sample_size:
			raise ValueError(f"Insufficient data points for trend analysis: {len(metrics)}")
		
		# Sort metrics by timestamp
		sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
		values = [m.value for m in sorted_metrics]
		timestamps = [m.timestamp for m in sorted_metrics]
		
		# Convert timestamps to numeric values for regression
		time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
		
		# Linear regression for trend analysis
		slope, r_squared = self._calculate_linear_regression(time_numeric, values)
		
		# Determine trend direction and strength
		direction = self._determine_trend_direction(slope, r_squared)
		confidence = min(r_squared, 1.0)
		trend_strength = abs(slope) * r_squared  # Combine slope magnitude with fit quality
		
		# Rate of change (per hour)
		time_span_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
		rate_of_change = slope * 3600 if time_span_hours > 0 else 0  # Convert to per hour
		
		# Detect change points (simple implementation)
		change_points = self._detect_change_points(sorted_metrics)
		
		# Check for seasonal patterns (basic implementation)
		seasonal_component = self._detect_seasonality(values, timestamps)
		
		# Simple forecast for next period
		forecast_next_period = None
		if r_squared > 0.5:  # Only forecast if trend is reasonably strong
			next_time = time_numeric[-1] + (time_numeric[-1] - time_numeric[0]) / len(time_numeric)
			forecast_next_period = slope * next_time + (values[-1] - slope * time_numeric[-1])
		
		return TrendAnalysis(
			metric_name=sorted_metrics[0].name,
			tenant_id=sorted_metrics[0].tenant_id,
			time_window=time_window,
			direction=direction,
			slope=slope,
			r_squared=r_squared,
			confidence=confidence,
			rate_of_change=rate_of_change,
			trend_strength=trend_strength,
			change_points=change_points,
			seasonal_component=seasonal_component,
			forecast_next_period=forecast_next_period
		)
	
	async def analyze_correlation(self, metrics1: List[MonitoringMetric], 
								 metrics2: List[MonitoringMetric],
								 time_window: str = "1h") -> CorrelationResult:
		"""Analyze correlation between two metric series"""
		
		if len(metrics1) < self.min_sample_size or len(metrics2) < self.min_sample_size:
			raise ValueError("Insufficient data points for correlation analysis")
		
		# Align metrics by timestamp for proper correlation
		aligned_values1, aligned_values2 = self._align_metric_series(metrics1, metrics2)
		
		if len(aligned_values1) < self.min_sample_size:
			raise ValueError("Insufficient aligned data points for correlation")
		
		# Calculate Pearson correlation coefficient
		correlation_coef = np.corrcoef(aligned_values1, aligned_values2)[0, 1]
		
		# Determine correlation strength
		correlation_strength = self._classify_correlation_strength(abs(correlation_coef))
		
		# Calculate statistical significance (simplified p-value estimation)
		p_value = self._estimate_correlation_p_value(correlation_coef, len(aligned_values1))
		
		# Calculate lag correlations
		lag_correlation = self._calculate_lag_correlations(aligned_values1, aligned_values2)
		
		# Determine causality direction (simplified Granger causality test)
		causality_direction = self._estimate_causality_direction(
			aligned_values1, aligned_values2, lag_correlation
		)
		
		return CorrelationResult(
			metric1_name=metrics1[0].name,
			metric2_name=metrics2[0].name,
			tenant_id=metrics1[0].tenant_id,
			correlation_coefficient=correlation_coef,
			correlation_strength=correlation_strength,
			p_value=p_value,
			sample_size=len(aligned_values1),
			time_window=time_window,
			lag_correlation=lag_correlation,
			causality_direction=causality_direction
		)
	
	def _calculate_data_quality_score(self, metrics: List[MonitoringMetric], 
									 values: List[float], outliers: List[float]) -> float:
		"""Calculate data quality score based on various factors"""
		score = 1.0
		
		# Penalize for missing timestamps (gaps in data)
		if len(metrics) > 1:
			timestamps = [m.timestamp for m in sorted(metrics, key=lambda x: x.timestamp)]
			expected_intervals = len(timestamps) - 1
			if expected_intervals > 0:
				avg_interval = (timestamps[-1] - timestamps[0]) / expected_intervals
				actual_intervals = []
				for i in range(1, len(timestamps)):
					actual_intervals.append(timestamps[i] - timestamps[i-1])
				
				# Check for large gaps
				large_gaps = sum(1 for interval in actual_intervals if interval > avg_interval * 2)
				gap_penalty = large_gaps / len(actual_intervals)
				score -= gap_penalty * 0.2
		
		# Penalize for outliers
		if len(values) > 0:
			outlier_ratio = len(outliers) / len(values)
			score -= outlier_ratio * 0.3
		
		# Penalize for low quality scores in metrics
		quality_scores = [m.quality_score for m in metrics if hasattr(m, 'quality_score')]
		if quality_scores:
			avg_quality = sum(quality_scores) / len(quality_scores)
			score *= avg_quality
		
		return max(0.0, min(1.0, score))
	
	def _calculate_linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float]:
		"""Calculate linear regression slope and R-squared"""
		n = len(x)
		if n < 2:
			return 0.0, 0.0
		
		# Calculate means
		x_mean = sum(x) / n
		y_mean = sum(y) / n
		
		# Calculate slope and intercept
		numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
		denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
		
		if denominator == 0:
			return 0.0, 0.0
		
		slope = numerator / denominator
		intercept = y_mean - slope * x_mean
		
		# Calculate R-squared
		y_pred = [slope * x[i] + intercept for i in range(n)]
		ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
		ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
		
		r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
		
		return slope, max(0.0, min(1.0, r_squared))
	
	def _determine_trend_direction(self, slope: float, r_squared: float) -> TrendDirection:
		"""Determine trend direction based on slope and fit quality"""
		if r_squared < 0.1:  # Very weak correlation
			return TrendDirection.VOLATILE
		elif r_squared < 0.3:  # Weak correlation
			return TrendDirection.UNKNOWN
		
		# Strong enough correlation to determine direction
		if abs(slope) < 1e-6:  # Essentially flat
			return TrendDirection.STABLE
		elif slope > 0:
			return TrendDirection.INCREASING
		else:
			return TrendDirection.DECREASING
	
	def _detect_change_points(self, metrics: List[MonitoringMetric]) -> List[datetime]:
		"""Detect points where trend changes significantly"""
		if len(metrics) < 6:  # Need enough points to detect changes
			return []
		
		values = [m.value for m in metrics]
		timestamps = [m.timestamp for m in metrics]
		change_points = []
		
		# Simple change point detection using moving windows
		window_size = max(3, len(values) // 10)
		
		for i in range(window_size, len(values) - window_size):
			# Calculate slopes before and after this point
			before_values = values[i-window_size:i]
			after_values = values[i:i+window_size]
			
			before_x = list(range(len(before_values)))
			after_x = list(range(len(after_values)))
			
			slope_before, _ = self._calculate_linear_regression(before_x, before_values)
			slope_after, _ = self._calculate_linear_regression(after_x, after_values)
			
			# Significant change in slope indicates change point
			if abs(slope_before - slope_after) > abs(slope_before) * 0.5:
				change_points.append(timestamps[i])
		
		return change_points
	
	def _detect_seasonality(self, values: List[float], timestamps: List[datetime]) -> bool:
		"""Detect seasonal patterns in data (simplified implementation)"""
		if len(values) < 24:  # Need at least 24 points for daily seasonality
			return False
		
		# Check for periodic patterns using autocorrelation
		# This is a simplified implementation
		autocorrelations = []
		
		for lag in [12, 24, 168]:  # Check for 12h, 24h, weekly patterns
			if lag >= len(values):
				continue
			
			lagged_values = values[:-lag]
			current_values = values[lag:]
			
			if len(lagged_values) > 0 and len(current_values) > 0:
				correlation = np.corrcoef(lagged_values, current_values)[0, 1]
				if not np.isnan(correlation):
					autocorrelations.append(abs(correlation))
		
		# If any autocorrelation is strong, consider it seasonal
		return any(corr > 0.5 for corr in autocorrelations)
	
	def _align_metric_series(self, metrics1: List[MonitoringMetric], 
							metrics2: List[MonitoringMetric]) -> Tuple[List[float], List[float]]:
		"""Align two metric series by timestamp for correlation analysis"""
		# Create timestamp-indexed dictionaries
		dict1 = {m.timestamp: m.value for m in metrics1}
		dict2 = {m.timestamp: m.value for m in metrics2}
		
		# Find common timestamps
		common_timestamps = set(dict1.keys()) & set(dict2.keys())
		
		if len(common_timestamps) < self.min_sample_size:
			# If not enough exact matches, try time-based alignment with tolerance
			return self._align_with_tolerance(metrics1, metrics2, tolerance_seconds=60)
		
		# Extract aligned values
		aligned_timestamps = sorted(common_timestamps)
		values1 = [dict1[ts] for ts in aligned_timestamps]
		values2 = [dict2[ts] for ts in aligned_timestamps]
		
		return values1, values2
	
	def _align_with_tolerance(self, metrics1: List[MonitoringMetric], 
							 metrics2: List[MonitoringMetric], 
							 tolerance_seconds: int = 60) -> Tuple[List[float], List[float]]:
		"""Align metrics with timestamp tolerance"""
		aligned_values1, aligned_values2 = [], []
		
		sorted_metrics1 = sorted(metrics1, key=lambda m: m.timestamp)
		sorted_metrics2 = sorted(metrics2, key=lambda m: m.timestamp)
		
		tolerance = timedelta(seconds=tolerance_seconds)
		
		for m1 in sorted_metrics1:
			# Find closest metric in series 2
			closest_m2 = None
			min_diff = None
			
			for m2 in sorted_metrics2:
				diff = abs((m1.timestamp - m2.timestamp).total_seconds())
				if diff <= tolerance_seconds and (min_diff is None or diff < min_diff):
					min_diff = diff
					closest_m2 = m2
			
			if closest_m2:
				aligned_values1.append(m1.value)
				aligned_values2.append(closest_m2.value)
		
		return aligned_values1, aligned_values2
	
	def _classify_correlation_strength(self, correlation: float) -> str:
		"""Classify correlation strength"""
		if correlation >= 0.7:
			return "strong"
		elif correlation >= 0.3:
			return "moderate"
		else:
			return "weak"
	
	def _estimate_correlation_p_value(self, correlation: float, sample_size: int) -> float:
		"""Estimate p-value for correlation coefficient"""
		if sample_size < 3:
			return 1.0
		
		# Simplified t-test for correlation significance
		t_stat = correlation * math.sqrt((sample_size - 2) / (1 - correlation**2))
		
		# This is a very simplified p-value estimation
		# In practice, you would use proper statistical tables or libraries
		p_value = max(0.001, min(1.0, 2 * (1 - abs(t_stat) / 3)))
		
		return p_value
	
	def _calculate_lag_correlations(self, values1: List[float], 
								   values2: List[float], 
								   max_lag: int = 5) -> Dict[int, float]:
		"""Calculate correlations at different time lags"""
		lag_correlations = {}
		
		for lag in range(-max_lag, max_lag + 1):
			if lag == 0:
				corr = np.corrcoef(values1, values2)[0, 1]
			elif lag > 0:
				if len(values1) > lag and len(values2) > lag:
					corr = np.corrcoef(values1[:-lag], values2[lag:])[0, 1]
				else:
					continue
			else:  # lag < 0
				abs_lag = abs(lag)
				if len(values1) > abs_lag and len(values2) > abs_lag:
					corr = np.corrcoef(values1[abs_lag:], values2[:-abs_lag])[0, 1]
				else:
					continue
			
			if not np.isnan(corr):
				lag_correlations[lag] = corr
		
		return lag_correlations
	
	def _estimate_causality_direction(self, values1: List[float], values2: List[float], 
									 lag_correlations: Dict[int, float]) -> Optional[str]:
		"""Estimate causality direction using lag correlations"""
		if not lag_correlations:
			return None
		
		# Look for strongest correlation at different lags
		max_positive_lag = max(
			(lag for lag in lag_correlations.keys() if lag > 0), 
			key=lambda lag: abs(lag_correlations[lag]), 
			default=None
		)
		
		max_negative_lag = max(
			(lag for lag in lag_correlations.keys() if lag < 0), 
			key=lambda lag: abs(lag_correlations[lag]), 
			default=None
		)
		
		positive_strength = abs(lag_correlations.get(max_positive_lag, 0)) if max_positive_lag else 0
		negative_strength = abs(lag_correlations.get(max_negative_lag, 0)) if max_negative_lag else 0
		
		# If correlation is stronger at positive lag, metric1 leads metric2
		if positive_strength > negative_strength and positive_strength > 0.3:
			return "metric1_leads_metric2"
		elif negative_strength > positive_strength and negative_strength > 0.3:
			return "metric2_leads_metric1"
		else:
			return "no_clear_causality"


class AnalyticsEngine:
	"""
	Comprehensive analytics engine for monitoring data
	Provides real-time statistical analysis, trend detection, and performance insights
	"""
	
	def __init__(self, config: dict = None):
		self.config = config or {}
		self.running = False
		
		# Core components
		self.statistical_analyzer = StatisticalAnalyzer(config.get('statistical', {}))
		
		# Analysis cache and storage
		self.analysis_cache: Dict[str, Dict] = defaultdict(dict)
		self.cache_ttl_seconds = self.config.get('cache_ttl_seconds', 300)  # 5 minutes
		
		# Processing queues
		self.analysis_queue = asyncio.Queue()
		self.background_tasks: List[asyncio.Task] = []
		
		# Performance tracking
		self.stats = {
			'total_analyses': 0,
			'cached_results': 0,
			'failed_analyses': 0,
			'avg_analysis_time_ms': 0.0,
			'analysis_types_distribution': defaultdict(int),
			'last_analysis': None
		}
		
		print("[AnalyticsEngine] Analytics engine initialized")
	
	async def initialize(self) -> None:
		"""Initialize the analytics engine"""
		assert not self.running, "Analytics engine is already running"
		
		# Start background processors
		self.background_tasks = [
			asyncio.create_task(self._analysis_processor_loop()),
			asyncio.create_task(self._cache_cleanup_loop()),
			asyncio.create_task(self._stats_update_loop())
		]
		
		self.running = True
		print("[AnalyticsEngine] Analytics engine started successfully")
	
	async def shutdown(self) -> None:
		"""Shutdown the analytics engine"""
		if not self.running:
			return
		
		self.running = False
		
		# Cancel background tasks
		for task in self.background_tasks:
			task.cancel()
		
		await asyncio.gather(*self.background_tasks, return_exceptions=True)
		print("[AnalyticsEngine] Analytics engine shutdown complete")
	
	async def analyze_statistical(self, metrics: List[MonitoringMetric], 
								 time_window: str = "1h",
								 use_cache: bool = True) -> StatisticalSummary:
		"""Perform statistical analysis on metrics"""
		cache_key = f"statistical_{metrics[0].name}_{metrics[0].tenant_id}_{time_window}_{len(metrics)}"
		
		# Check cache first
		if use_cache:
			cached_result = self._get_cached_result(cache_key, AnalysisType.STATISTICAL)
			if cached_result:
				self.stats['cached_results'] += 1
				return StatisticalSummary(**cached_result)
		
		try:
			start_time = datetime.utcnow()
			
			# Perform analysis
			result = await self.statistical_analyzer.analyze_metrics_statistical(metrics, time_window)
			
			# Update statistics
			analysis_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			self._update_analysis_stats(AnalysisType.STATISTICAL, analysis_time)
			
			# Cache result
			if use_cache:
				self._cache_result(cache_key, AnalysisType.STATISTICAL, result.to_dict())
			
			return result
			
		except Exception as e:
			self.stats['failed_analyses'] += 1
			print(f"[AnalyticsEngine] Error in statistical analysis: {e}")
			raise
	
	async def analyze_trend(self, metrics: List[MonitoringMetric], 
						   time_window: str = "1h",
						   use_cache: bool = True) -> TrendAnalysis:
		"""Perform trend analysis on metrics"""
		cache_key = f"trend_{metrics[0].name}_{metrics[0].tenant_id}_{time_window}_{len(metrics)}"
		
		# Check cache first
		if use_cache:
			cached_result = self._get_cached_result(cache_key, AnalysisType.TREND)
			if cached_result:
				self.stats['cached_results'] += 1
				return TrendAnalysis(**cached_result)
		
		try:
			start_time = datetime.utcnow()
			
			# Perform analysis
			result = await self.statistical_analyzer.analyze_trend(metrics, time_window)
			
			# Update statistics
			analysis_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			self._update_analysis_stats(AnalysisType.TREND, analysis_time)
			
			# Cache result
			if use_cache:
				self._cache_result(cache_key, AnalysisType.TREND, result.to_dict())
			
			return result
			
		except Exception as e:
			self.stats['failed_analyses'] += 1
			print(f"[AnalyticsEngine] Error in trend analysis: {e}")
			raise
	
	async def analyze_correlation(self, metrics1: List[MonitoringMetric], 
								 metrics2: List[MonitoringMetric],
								 time_window: str = "1h",
								 use_cache: bool = True) -> CorrelationResult:
		"""Perform correlation analysis between metric series"""
		cache_key = f"correlation_{metrics1[0].name}_{metrics2[0].name}_{metrics1[0].tenant_id}_{time_window}"
		
		# Check cache first
		if use_cache:
			cached_result = self._get_cached_result(cache_key, AnalysisType.CORRELATION)
			if cached_result:
				self.stats['cached_results'] += 1
				return CorrelationResult(**cached_result)
		
		try:
			start_time = datetime.utcnow()
			
			# Perform analysis
			result = await self.statistical_analyzer.analyze_correlation(metrics1, metrics2, time_window)
			
			# Update statistics
			analysis_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			self._update_analysis_stats(AnalysisType.CORRELATION, analysis_time)
			
			# Cache result
			if use_cache:
				self._cache_result(cache_key, AnalysisType.CORRELATION, result.to_dict())
			
			return result
			
		except Exception as e:
			self.stats['failed_analyses'] += 1
			print(f"[AnalyticsEngine] Error in correlation analysis: {e}")
			raise
	
	async def analyze_performance_baseline(self, metrics: List[MonitoringMetric],
										  baseline_period_days: int = 7) -> Dict[str, Any]:
		"""Establish performance baseline from historical data"""
		try:
			start_time = datetime.utcnow()
			
			# Group metrics by time periods for baseline calculation
			daily_aggregates = self._aggregate_metrics_by_period(metrics, "daily")
			
			# Calculate baseline statistics
			baseline_stats = {}
			
			for day, day_metrics in daily_aggregates.items():
				day_values = [m.value for m in day_metrics]
				if day_values:
					baseline_stats[day] = {
						'mean': statistics.mean(day_values),
						'p50': np.percentile(day_values, 50),
						'p95': np.percentile(day_values, 95),
						'p99': np.percentile(day_values, 99),
						'sample_count': len(day_values)
					}
			
			# Calculate overall baseline
			all_means = [stats['mean'] for stats in baseline_stats.values()]
			all_p95s = [stats['p95'] for stats in baseline_stats.values()]
			
			baseline_result = {
				'metric_name': metrics[0].name if metrics else 'unknown',
				'tenant_id': metrics[0].tenant_id if metrics else 'unknown',
				'baseline_period_days': baseline_period_days,
				'overall_baseline': {
					'mean_baseline': statistics.mean(all_means) if all_means else 0,
					'p95_baseline': statistics.mean(all_p95s) if all_p95s else 0,
					'variability': statistics.stdev(all_means) if len(all_means) > 1 else 0,
					'stability_score': 1.0 / (1.0 + statistics.stdev(all_means)) if len(all_means) > 1 else 1.0
				},
				'daily_baselines': baseline_stats,
				'recommendations': self._generate_baseline_recommendations(baseline_stats),
				'analysis_timestamp': datetime.utcnow().isoformat()
			}
			
			# Update statistics
			analysis_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			self._update_analysis_stats(AnalysisType.BASELINE, analysis_time)
			
			return baseline_result
			
		except Exception as e:
			self.stats['failed_analyses'] += 1
			print(f"[AnalyticsEngine] Error in baseline analysis: {e}")
			raise
	
	async def get_analytics_dashboard(self, tenant_id: str, 
									dashboard_type: str = "operational") -> Dict[str, Any]:
		"""Generate analytics dashboard data"""
		try:
			# This would typically aggregate data from multiple sources
			# For now, return a structured dashboard format
			
			dashboard_data = {
				'dashboard_type': dashboard_type,
				'tenant_id': tenant_id,
				'generated_at': datetime.utcnow().isoformat(),
				'summary': {
					'total_analyses_performed': self.stats['total_analyses'],
					'cache_hit_rate': self.stats['cached_results'] / max(self.stats['total_analyses'], 1),
					'avg_analysis_time_ms': self.stats['avg_analysis_time_ms'],
					'analysis_distribution': dict(self.stats['analysis_types_distribution'])
				},
				'widgets': [
					{
						'type': 'metric_summary',
						'title': 'Analysis Performance',
						'data': {
							'total_analyses': self.stats['total_analyses'],
							'success_rate': 1.0 - (self.stats['failed_analyses'] / max(self.stats['total_analyses'], 1)),
							'avg_processing_time': self.stats['avg_analysis_time_ms']
						}
					},
					{
						'type': 'cache_efficiency',
						'title': 'Cache Performance',
						'data': {
							'hit_rate': self.stats['cached_results'] / max(self.stats['total_analyses'], 1),
							'cache_size': len(self.analysis_cache),
							'cache_utilization': 'optimal' if len(self.analysis_cache) < 1000 else 'high'
						}
					}
				],
				'insights': self._generate_analytics_insights(),
				'recommendations': self._generate_performance_recommendations()
			}
			
			return dashboard_data
			
		except Exception as e:
			print(f"[AnalyticsEngine] Error generating dashboard: {e}")
			return {'error': str(e), 'dashboard_type': dashboard_type, 'tenant_id': tenant_id}
	
	async def get_engine_stats(self) -> Dict[str, Any]:
		"""Get comprehensive engine statistics"""
		return {
			**self.stats,
			'cache_stats': {
				'total_entries': len(self.analysis_cache),
				'cache_types': {
					analysis_type: len(cache_data) 
					for analysis_type, cache_data in self.analysis_cache.items()
				}
			},
			'queue_sizes': {
				'analysis_queue': self.analysis_queue.qsize()
			},
			'running': self.running,
			'timestamp': datetime.utcnow().isoformat()
		}
	
	# Private implementation methods
	def _get_cached_result(self, cache_key: str, analysis_type: AnalysisType) -> Optional[Dict]:
		"""Get cached analysis result if not expired"""
		cache_data = self.analysis_cache[analysis_type.value].get(cache_key)
		
		if cache_data and cache_data['expires_at'] > datetime.utcnow():
			return cache_data['result']
		
		# Remove expired cache entry
		if cache_data:
			del self.analysis_cache[analysis_type.value][cache_key]
		
		return None
	
	def _cache_result(self, cache_key: str, analysis_type: AnalysisType, result: Dict) -> None:
		"""Cache analysis result with TTL"""
		expires_at = datetime.utcnow() + timedelta(seconds=self.cache_ttl_seconds)
		
		self.analysis_cache[analysis_type.value][cache_key] = {
			'result': result,
			'cached_at': datetime.utcnow(),
			'expires_at': expires_at
		}
	
	def _update_analysis_stats(self, analysis_type: AnalysisType, analysis_time_ms: float) -> None:
		"""Update analysis performance statistics"""
		self.stats['total_analyses'] += 1
		self.stats['analysis_types_distribution'][analysis_type.value] += 1
		self.stats['last_analysis'] = datetime.utcnow().isoformat()
		
		# Update rolling average analysis time
		current_avg = self.stats['avg_analysis_time_ms']
		self.stats['avg_analysis_time_ms'] = (current_avg * 0.9) + (analysis_time_ms * 0.1)
	
	def _aggregate_metrics_by_period(self, metrics: List[MonitoringMetric], 
									period: str) -> Dict[str, List[MonitoringMetric]]:
		"""Aggregate metrics by time period"""
		aggregates = defaultdict(list)
		
		for metric in metrics:
			if period == "daily":
				key = metric.timestamp.strftime("%Y-%m-%d")
			elif period == "hourly":
				key = metric.timestamp.strftime("%Y-%m-%d %H:00")
			else:
				key = "all"
			
			aggregates[key].append(metric)
		
		return dict(aggregates)
	
	def _generate_baseline_recommendations(self, baseline_stats: Dict) -> List[str]:
		"""Generate recommendations based on baseline analysis"""
		recommendations = []
		
		if not baseline_stats:
			recommendations.append("Insufficient data for baseline recommendations")
			return recommendations
		
		# Analyze variability
		daily_means = [stats['mean'] for stats in baseline_stats.values()]
		if len(daily_means) > 1:
			variability = statistics.stdev(daily_means) / statistics.mean(daily_means)
			
			if variability > 0.3:
				recommendations.append("High variability detected - consider investigating external factors")
			elif variability < 0.1:
				recommendations.append("Stable baseline detected - suitable for anomaly detection")
		
		# Analyze trends in baselines
		sorted_days = sorted(baseline_stats.keys())
		if len(sorted_days) >= 3:
			recent_means = [baseline_stats[day]['mean'] for day in sorted_days[-3:]]
			early_means = [baseline_stats[day]['mean'] for day in sorted_days[:3]]
			
			recent_avg = statistics.mean(recent_means)
			early_avg = statistics.mean(early_means)
			
			change_percent = ((recent_avg - early_avg) / early_avg) * 100 if early_avg > 0 else 0
			
			if abs(change_percent) > 20:
				if change_percent > 0:
					recommendations.append(f"Baseline trending upward (+{change_percent:.1f}%) - monitor for capacity issues")
				else:
					recommendations.append(f"Baseline trending downward ({change_percent:.1f}%) - validate measurement accuracy")
		
		return recommendations
	
	def _generate_analytics_insights(self) -> List[str]:
		"""Generate insights from analytics performance"""
		insights = []
		
		# Cache performance insights
		cache_hit_rate = self.stats['cached_results'] / max(self.stats['total_analyses'], 1)
		if cache_hit_rate > 0.8:
			insights.append("Excellent cache performance - most analyses served from cache")
		elif cache_hit_rate < 0.3:
			insights.append("Low cache hit rate - consider increasing cache TTL or analysis frequency")
		
		# Analysis type distribution insights
		if self.stats['analysis_types_distribution']:
			most_common = max(self.stats['analysis_types_distribution'], 
							 key=self.stats['analysis_types_distribution'].get)
			insights.append(f"Most requested analysis type: {most_common}")
		
		# Performance insights
		if self.stats['avg_analysis_time_ms'] > 1000:
			insights.append("Analysis performance could be improved - consider optimization")
		elif self.stats['avg_analysis_time_ms'] < 100:
			insights.append("Excellent analysis performance")
		
		return insights
	
	def _generate_performance_recommendations(self) -> List[str]:
		"""Generate performance optimization recommendations"""
		recommendations = []
		
		if self.stats['failed_analyses'] / max(self.stats['total_analyses'], 1) > 0.05:
			recommendations.append("High failure rate detected - review analysis parameters and data quality")
		
		if len(self.analysis_cache) > 1000:
			recommendations.append("Large cache size - consider reducing TTL or implementing cache eviction")
		
		if self.stats['avg_analysis_time_ms'] > 500:
			recommendations.append("Consider implementing analysis result pre-computation for frequently requested analyses")
		
		return recommendations
	
	async def _analysis_processor_loop(self) -> None:
		"""Background loop for processing analysis requests"""
		try:
			while self.running:
				# This would process queued analysis requests
				await asyncio.sleep(1)
				
		except asyncio.CancelledError:
			pass
		except Exception as e:
			print(f"[AnalyticsEngine] Error in analysis processor: {e}")
	
	async def _cache_cleanup_loop(self) -> None:
		"""Background loop for cache cleanup"""
		try:
			while self.running:
				await asyncio.sleep(300)  # Clean every 5 minutes
				
				# Clean expired cache entries
				current_time = datetime.utcnow()
				
				for analysis_type in self.analysis_cache:
					expired_keys = [
						key for key, data in self.analysis_cache[analysis_type].items()
						if data['expires_at'] <= current_time
					]
					
					for key in expired_keys:
						del self.analysis_cache[analysis_type][key]
				
		except asyncio.CancelledError:
			pass
		except Exception as e:
			print(f"[AnalyticsEngine] Error in cache cleanup: {e}")
	
	async def _stats_update_loop(self) -> None:
		"""Background loop for statistics updates"""
		try:
			while self.running:
				await asyncio.sleep(60)  # Update every minute
				
				# Log performance statistics
				print(f"[AnalyticsEngine] Stats: {self.stats['total_analyses']} analyses, "
					 f"{self.stats['avg_analysis_time_ms']:.1f}ms avg, "
					 f"{self.stats['cached_results']/(max(self.stats['total_analyses'], 1))*100:.1f}% cache hit rate")
				
		except asyncio.CancelledError:
			pass
		except Exception as e:
			print(f"[AnalyticsEngine] Error in stats update: {e}")


# Factory function
def create_analytics_engine(config: dict = None) -> AnalyticsEngine:
	"""Create and configure analytics engine"""
	return AnalyticsEngine(config)