"""
APG Workflow & Business Process Management - Real-time Process Performance Monitoring

Advanced performance monitoring with real-time metrics collection, anomaly detection,
and predictive analytics for workflow optimization.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
import statistics
import numpy as np
from scipy import stats

from models import (
	APGTenantContext, WBPMServiceResponse, WBPMPagedResponse,
	WBPMProcessInstance, WBPMTask, ProcessStatus, TaskStatus, TaskPriority
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Performance Monitoring Core Classes
# =============================================================================

class PerformanceMetric(str, Enum):
	"""Types of performance metrics."""
	PROCESS_DURATION = "process_duration"
	TASK_DURATION = "task_duration"
	QUEUE_TIME = "queue_time"
	PROCESSING_TIME = "processing_time"
	THROUGHPUT = "throughput"
	ERROR_RATE = "error_rate"
	RESOURCE_UTILIZATION = "resource_utilization"
	USER_ACTIVITY = "user_activity"
	SYSTEM_LOAD = "system_load"
	RESPONSE_TIME = "response_time"
	CONCURRENT_PROCESSES = "concurrent_processes"
	BOTTLENECK_SCORE = "bottleneck_score"


class AnomalyType(str, Enum):
	"""Types of performance anomalies."""
	OUTLIER = "outlier"
	TREND_CHANGE = "trend_change"
	THRESHOLD_BREACH = "threshold_breach"
	PATTERN_DEVIATION = "pattern_deviation"
	SEASONAL_ANOMALY = "seasonal_anomaly"
	CORRELATION_BREAK = "correlation_break"


class MonitoringLevel(str, Enum):
	"""Monitoring detail levels."""
	BASIC = "basic"
	DETAILED = "detailed"
	COMPREHENSIVE = "comprehensive"
	DEBUG = "debug"


@dataclass
class PerformanceDataPoint:
	"""Individual performance measurement."""
	data_id: str = field(default_factory=lambda: f"perf_{uuid.uuid4().hex}")
	metric_type: PerformanceMetric = PerformanceMetric.PROCESS_DURATION
	value: float = 0.0
	unit: str = "seconds"
	timestamp: datetime = field(default_factory=datetime.utcnow)
	process_id: Optional[str] = None
	task_id: Optional[str] = None
	user_id: Optional[str] = None
	tags: Dict[str, str] = field(default_factory=dict)
	context: Dict[str, Any] = field(default_factory=dict)
	tenant_id: str = ""


@dataclass
class PerformanceBaseline:
	"""Performance baseline for comparison."""
	baseline_id: str = field(default_factory=lambda: f"baseline_{uuid.uuid4().hex}")
	metric_type: PerformanceMetric = PerformanceMetric.PROCESS_DURATION
	baseline_value: float = 0.0
	standard_deviation: float = 0.0
	percentile_95: float = 0.0
	percentile_99: float = 0.0
	min_value: float = 0.0
	max_value: float = 0.0
	sample_size: int = 0
	calculation_period: timedelta = field(default_factory=lambda: timedelta(days=30))
	calculated_at: datetime = field(default_factory=datetime.utcnow)
	tags: Dict[str, str] = field(default_factory=dict)
	tenant_id: str = ""


@dataclass
class PerformanceAnomaly:
	"""Detected performance anomaly."""
	anomaly_id: str = field(default_factory=lambda: f"anomaly_{uuid.uuid4().hex}")
	anomaly_type: AnomalyType = AnomalyType.OUTLIER
	metric_type: PerformanceMetric = PerformanceMetric.PROCESS_DURATION
	detected_value: float = 0.0
	expected_value: float = 0.0
	deviation_score: float = 0.0  # Standard deviations from norm
	confidence: float = 0.0  # 0-1 confidence in anomaly
	detected_at: datetime = field(default_factory=datetime.utcnow)
	process_id: Optional[str] = None
	task_id: Optional[str] = None
	description: str = ""
	impact_assessment: str = ""
	recommended_actions: List[str] = field(default_factory=list)
	resolved_at: Optional[datetime] = None
	tenant_id: str = ""


@dataclass
class PerformanceTrend:
	"""Performance trend analysis."""
	trend_id: str = field(default_factory=lambda: f"trend_{uuid.uuid4().hex}")
	metric_type: PerformanceMetric = PerformanceMetric.PROCESS_DURATION
	trend_direction: str = "stable"  # improving, degrading, stable
	trend_magnitude: float = 0.0
	confidence: float = 0.0
	analysis_period: timedelta = field(default_factory=lambda: timedelta(days=7))
	data_points: int = 0
	r_squared: float = 0.0
	slope: float = 0.0
	calculated_at: datetime = field(default_factory=datetime.utcnow)
	tags: Dict[str, str] = field(default_factory=dict)
	tenant_id: str = ""


@dataclass
class BottleneckAnalysis:
	"""Process bottleneck analysis."""
	analysis_id: str = field(default_factory=lambda: f"bottleneck_{uuid.uuid4().hex}")
	process_id: str = ""
	bottleneck_activities: List[str] = field(default_factory=list)
	bottleneck_scores: Dict[str, float] = field(default_factory=dict)
	wait_times: Dict[str, float] = field(default_factory=dict)
	resource_constraints: List[str] = field(default_factory=list)
	impact_assessment: str = ""
	optimization_suggestions: List[str] = field(default_factory=list)
	analyzed_at: datetime = field(default_factory=datetime.utcnow)
	tenant_id: str = ""


@dataclass
class ResourceUtilization:
	"""Resource utilization metrics."""
	resource_id: str = field(default_factory=lambda: f"resource_{uuid.uuid4().hex}")
	resource_type: str = ""  # user, system, external_service
	resource_name: str = ""
	utilization_percentage: float = 0.0
	active_tasks: int = 0
	queue_length: int = 0
	avg_response_time: float = 0.0
	error_rate: float = 0.0
	availability: float = 100.0
	measured_at: datetime = field(default_factory=datetime.utcnow)
	tenant_id: str = ""


# =============================================================================
# Real-time Metrics Collector
# =============================================================================

class RealTimeMetricsCollector:
	"""Collect real-time performance metrics."""
	
	def __init__(self, buffer_size: int = 50000):
		self.buffer_size = buffer_size
		self.metrics_buffer: deque = deque(maxlen=buffer_size)
		self.active_processes: Dict[str, Dict[str, Any]] = {}
		self.active_tasks: Dict[str, Dict[str, Any]] = {}
		self.last_cleanup = datetime.utcnow()
		
	async def start_process_tracking(
		self,
		process_id: str,
		context: APGTenantContext
	) -> None:
		"""Start tracking a process."""
		try:
			self.active_processes[process_id] = {
				"start_time": datetime.utcnow(),
				"tenant_id": context.tenant_id,
				"user_id": context.user_id,
				"tasks": {},
				"metrics": defaultdict(list)
			}
			
			# Record process start metric
			await self.record_metric(
				PerformanceMetric.CONCURRENT_PROCESSES,
				len([p for p in self.active_processes.values() if p["tenant_id"] == context.tenant_id]),
				process_id=process_id,
				context=context
			)
			
			logger.debug(f"Started tracking process: {process_id}")
			
		except Exception as e:
			logger.error(f"Error starting process tracking: {e}")
	
	async def end_process_tracking(
		self,
		process_id: str,
		final_status: str,
		context: APGTenantContext
	) -> Optional[PerformanceDataPoint]:
		"""End process tracking and calculate duration."""
		try:
			if process_id not in self.active_processes:
				logger.warning(f"Process not being tracked: {process_id}")
				return None
			
			process_data = self.active_processes.pop(process_id)
			end_time = datetime.utcnow()
			start_time = process_data["start_time"]
			duration = (end_time - start_time).total_seconds()
			
			# Record process duration
			data_point = await self.record_metric(
				PerformanceMetric.PROCESS_DURATION,
				duration,
				process_id=process_id,
				context=context,
				additional_tags={"final_status": final_status}
			)
			
			# Record updated concurrent processes count
			await self.record_metric(
				PerformanceMetric.CONCURRENT_PROCESSES,
				len([p for p in self.active_processes.values() if p["tenant_id"] == context.tenant_id]),
				process_id=process_id,
				context=context
			)
			
			logger.debug(f"Ended tracking process {process_id}: {duration:.2f}s")
			
			return data_point
			
		except Exception as e:
			logger.error(f"Error ending process tracking: {e}")
			return None
	
	async def start_task_tracking(
		self,
		task_id: str,
		process_id: str,
		assignee: str,
		context: APGTenantContext
	) -> None:
		"""Start tracking a task."""
		try:
			self.active_tasks[task_id] = {
				"start_time": datetime.utcnow(),
				"process_id": process_id,
				"assignee": assignee,
				"tenant_id": context.tenant_id,
				"queue_time": 0.0,
				"processing_started": None
			}
			
			# Update process task count
			if process_id in self.active_processes:
				self.active_processes[process_id]["tasks"][task_id] = datetime.utcnow()
			
			logger.debug(f"Started tracking task: {task_id}")
			
		except Exception as e:
			logger.error(f"Error starting task tracking: {e}")
	
	async def mark_task_processing(
		self,
		task_id: str,
		context: APGTenantContext
	) -> None:
		"""Mark task as started processing (out of queue)."""
		try:
			if task_id not in self.active_tasks:
				logger.warning(f"Task not being tracked: {task_id}")
				return
			
			task_data = self.active_tasks[task_id]
			processing_start = datetime.utcnow()
			task_data["processing_started"] = processing_start
			
			# Calculate queue time
			queue_time = (processing_start - task_data["start_time"]).total_seconds()
			task_data["queue_time"] = queue_time
			
			# Record queue time metric
			await self.record_metric(
				PerformanceMetric.QUEUE_TIME,
				queue_time,
				task_id=task_id,
				process_id=task_data["process_id"],
				context=context,
				additional_tags={"assignee": task_data["assignee"]}
			)
			
			logger.debug(f"Task {task_id} started processing after {queue_time:.2f}s queue time")
			
		except Exception as e:
			logger.error(f"Error marking task processing: {e}")
	
	async def end_task_tracking(
		self,
		task_id: str,
		final_status: str,
		context: APGTenantContext
	) -> Optional[PerformanceDataPoint]:
		"""End task tracking and calculate metrics."""
		try:
			if task_id not in self.active_tasks:
				logger.warning(f"Task not being tracked: {task_id}")
				return None
			
			task_data = self.active_tasks.pop(task_id)
			end_time = datetime.utcnow()
			start_time = task_data["start_time"]
			total_duration = (end_time - start_time).total_seconds()
			
			# Calculate processing time
			if task_data["processing_started"]:
				processing_time = (end_time - task_data["processing_started"]).total_seconds()
			else:
				processing_time = total_duration
				task_data["queue_time"] = 0.0
			
			# Record task duration
			data_point = await self.record_metric(
				PerformanceMetric.TASK_DURATION,
				total_duration,
				task_id=task_id,
				process_id=task_data["process_id"],
				context=context,
				additional_tags={
					"final_status": final_status,
					"assignee": task_data["assignee"]
				}
			)
			
			# Record processing time
			await self.record_metric(
				PerformanceMetric.PROCESSING_TIME,
				processing_time,
				task_id=task_id,
				process_id=task_data["process_id"],
				context=context,
				additional_tags={
					"final_status": final_status,
					"assignee": task_data["assignee"]
				}
			)
			
			# Update process tracking
			if task_data["process_id"] in self.active_processes:
				process_tasks = self.active_processes[task_data["process_id"]]["tasks"]
				if task_id in process_tasks:
					del process_tasks[task_id]
			
			logger.debug(f"Ended tracking task {task_id}: {total_duration:.2f}s total, {processing_time:.2f}s processing")
			
			return data_point
			
		except Exception as e:
			logger.error(f"Error ending task tracking: {e}")
			return None
	
	async def record_metric(
		self,
		metric_type: PerformanceMetric,
		value: float,
		process_id: Optional[str] = None,
		task_id: Optional[str] = None,
		context: Optional[APGTenantContext] = None,
		additional_tags: Optional[Dict[str, str]] = None,
		unit: str = "seconds"
	) -> PerformanceDataPoint:
		"""Record a performance metric."""
		try:
			tags = additional_tags or {}
			if context:
				tags["tenant_id"] = context.tenant_id
				tags["user_id"] = context.user_id
			
			data_point = PerformanceDataPoint(
				metric_type=metric_type,
				value=value,
				unit=unit,
				process_id=process_id,
				task_id=task_id,
				user_id=context.user_id if context else None,
				tags=tags,
				tenant_id=context.tenant_id if context else ""
			)
			
			self.metrics_buffer.append(data_point)
			
			# Periodic cleanup
			if datetime.utcnow() - self.last_cleanup > timedelta(hours=1):
				await self._cleanup_old_tracking()
			
			return data_point
			
		except Exception as e:
			logger.error(f"Error recording metric: {e}")
			raise
	
	async def record_system_metric(
		self,
		metric_type: PerformanceMetric,
		value: float,
		context: APGTenantContext,
		unit: str = "count"
	) -> None:
		"""Record system-level metric."""
		try:
			await self.record_metric(
				metric_type=metric_type,
				value=value,
				context=context,
				unit=unit,
				additional_tags={"metric_source": "system"}
			)
			
		except Exception as e:
			logger.error(f"Error recording system metric: {e}")
	
	async def get_metrics(
		self,
		metric_type: Optional[PerformanceMetric] = None,
		time_window: Optional[timedelta] = None,
		process_id: Optional[str] = None,
		tenant_id: Optional[str] = None
	) -> List[PerformanceDataPoint]:
		"""Get metrics matching criteria."""
		try:
			cutoff_time = datetime.utcnow() - (time_window or timedelta(hours=1))
			
			filtered_metrics = []
			for metric in self.metrics_buffer:
				# Time filter
				if metric.timestamp < cutoff_time:
					continue
				
				# Type filter
				if metric_type and metric.metric_type != metric_type:
					continue
				
				# Process filter
				if process_id and metric.process_id != process_id:
					continue
				
				# Tenant filter
				if tenant_id and metric.tenant_id != tenant_id:
					continue
				
				filtered_metrics.append(metric)
			
			return sorted(filtered_metrics, key=lambda m: m.timestamp)
			
		except Exception as e:
			logger.error(f"Error getting metrics: {e}")
			return []
	
	async def calculate_throughput(
		self,
		time_window: timedelta,
		context: APGTenantContext
	) -> float:
		"""Calculate process throughput."""
		try:
			process_metrics = await self.get_metrics(
				metric_type=PerformanceMetric.PROCESS_DURATION,
				time_window=time_window,
				tenant_id=context.tenant_id
			)
			
			completed_count = len(process_metrics)
			hours = time_window.total_seconds() / 3600
			throughput = completed_count / hours if hours > 0 else 0.0
			
			# Record throughput metric
			await self.record_metric(
				PerformanceMetric.THROUGHPUT,
				throughput,
				context=context,
				unit="processes/hour"
			)
			
			return throughput
			
		except Exception as e:
			logger.error(f"Error calculating throughput: {e}")
			return 0.0
	
	async def _cleanup_old_tracking(self) -> None:
		"""Clean up old tracking data."""
		try:
			current_time = datetime.utcnow()
			cutoff_time = current_time - timedelta(hours=24)
			
			# Clean up abandoned process tracking
			abandoned_processes = []
			for process_id, data in self.active_processes.items():
				if data["start_time"] < cutoff_time:
					abandoned_processes.append(process_id)
			
			for process_id in abandoned_processes:
				del self.active_processes[process_id]
				logger.warning(f"Cleaned up abandoned process tracking: {process_id}")
			
			# Clean up abandoned task tracking
			abandoned_tasks = []
			for task_id, data in self.active_tasks.items():
				if data["start_time"] < cutoff_time:
					abandoned_tasks.append(task_id)
			
			for task_id in abandoned_tasks:
				del self.active_tasks[task_id]
				logger.warning(f"Cleaned up abandoned task tracking: {task_id}")
			
			self.last_cleanup = current_time
			
		except Exception as e:
			logger.error(f"Error cleaning up old tracking: {e}")


# =============================================================================
# Anomaly Detection Engine
# =============================================================================

class AnomalyDetectionEngine:
	"""Detect performance anomalies using statistical methods."""
	
	def __init__(self):
		self.baselines: Dict[str, PerformanceBaseline] = {}
		self.detected_anomalies: List[PerformanceAnomaly] = []
		self.detection_sensitivity = 2.5  # Standard deviations for outlier detection
		
	async def update_baseline(
		self,
		metric_type: PerformanceMetric,
		metrics: List[PerformanceDataPoint],
		context: APGTenantContext,
		tags: Optional[Dict[str, str]] = None
	) -> PerformanceBaseline:
		"""Update performance baseline from metrics."""
		try:
			if not metrics:
				raise ValueError("No metrics provided for baseline calculation")
			
			values = [m.value for m in metrics]
			
			baseline = PerformanceBaseline(
				metric_type=metric_type,
				baseline_value=statistics.mean(values),
				standard_deviation=statistics.stdev(values) if len(values) > 1 else 0.0,
				percentile_95=np.percentile(values, 95),
				percentile_99=np.percentile(values, 99),
				min_value=min(values),
				max_value=max(values),
				sample_size=len(values),
				tags=tags or {},
				tenant_id=context.tenant_id
			)
			
			baseline_key = f"{context.tenant_id}:{metric_type.value}"
			if tags:
				baseline_key += ":" + ":".join(f"{k}={v}" for k, v in sorted(tags.items()))
			
			self.baselines[baseline_key] = baseline
			
			logger.info(f"Updated baseline for {metric_type.value}: mean={baseline.baseline_value:.2f}, std={baseline.standard_deviation:.2f}")
			
			return baseline
			
		except Exception as e:
			logger.error(f"Error updating baseline: {e}")
			raise
	
	async def detect_anomalies(
		self,
		metrics: List[PerformanceDataPoint],
		context: APGTenantContext
	) -> List[PerformanceAnomaly]:
		"""Detect anomalies in metrics."""
		try:
			anomalies = []
			
			# Group metrics by type
			metrics_by_type = defaultdict(list)
			for metric in metrics:
				metrics_by_type[metric.metric_type].append(metric)
			
			# Detect anomalies for each metric type
			for metric_type, type_metrics in metrics_by_type.items():
				type_anomalies = await self._detect_type_anomalies(
					metric_type, type_metrics, context
				)
				anomalies.extend(type_anomalies)
			
			# Store detected anomalies
			self.detected_anomalies.extend(anomalies)
			
			# Keep only recent anomalies
			cutoff_time = datetime.utcnow() - timedelta(days=7)
			self.detected_anomalies = [
				a for a in self.detected_anomalies
				if a.detected_at >= cutoff_time
			]
			
			return anomalies
			
		except Exception as e:
			logger.error(f"Error detecting anomalies: {e}")
			return []
	
	async def _detect_type_anomalies(
		self,
		metric_type: PerformanceMetric,
		metrics: List[PerformanceDataPoint],
		context: APGTenantContext
	) -> List[PerformanceAnomaly]:
		"""Detect anomalies for specific metric type."""
		try:
			anomalies = []
			
			# Get baseline for this metric type
			baseline_key = f"{context.tenant_id}:{metric_type.value}"
			baseline = self.baselines.get(baseline_key)
			
			if not baseline or baseline.sample_size < 10:
				logger.debug(f"Insufficient baseline data for {metric_type.value}")
				return anomalies
			
			# Check each metric for anomalies
			for metric in metrics:
				metric_anomalies = await self._check_metric_anomalies(
					metric, baseline, context
				)
				anomalies.extend(metric_anomalies)
			
			# Detect trend anomalies
			if len(metrics) >= 10:
				trend_anomalies = await self._detect_trend_anomalies(
					metrics, baseline, context
				)
				anomalies.extend(trend_anomalies)
			
			return anomalies
			
		except Exception as e:
			logger.error(f"Error detecting type anomalies: {e}")
			return []
	
	async def _check_metric_anomalies(
		self,
		metric: PerformanceDataPoint,
		baseline: PerformanceBaseline,
		context: APGTenantContext
	) -> List[PerformanceAnomaly]:
		"""Check individual metric for anomalies."""
		try:
			anomalies = []
			
			if baseline.standard_deviation == 0:
				return anomalies
			
			# Calculate z-score
			z_score = abs(metric.value - baseline.baseline_value) / baseline.standard_deviation
			
			# Check for outlier
			if z_score > self.detection_sensitivity:
				anomaly = PerformanceAnomaly(
					anomaly_type=AnomalyType.OUTLIER,
					metric_type=metric.metric_type,
					detected_value=metric.value,
					expected_value=baseline.baseline_value,
					deviation_score=z_score,
					confidence=min(0.99, z_score / 5.0),  # Confidence based on z-score
					process_id=metric.process_id,
					task_id=metric.task_id,
					description=f"{metric.metric_type.value} value {metric.value:.2f} is {z_score:.1f} standard deviations from baseline {baseline.baseline_value:.2f}",
					impact_assessment=self._assess_impact(metric, z_score),
					recommended_actions=self._generate_recommendations(metric, z_score),
					tenant_id=context.tenant_id
				)
				anomalies.append(anomaly)
			
			# Check for threshold breaches
			if metric.value > baseline.percentile_99:
				anomaly = PerformanceAnomaly(
					anomaly_type=AnomalyType.THRESHOLD_BREACH,
					metric_type=metric.metric_type,
					detected_value=metric.value,
					expected_value=baseline.percentile_99,
					deviation_score=(metric.value - baseline.percentile_99) / baseline.baseline_value,
					confidence=0.95,
					process_id=metric.process_id,
					task_id=metric.task_id,
					description=f"{metric.metric_type.value} value {metric.value:.2f} exceeds 99th percentile threshold {baseline.percentile_99:.2f}",
					impact_assessment="High impact - performance significantly worse than normal",
					recommended_actions=["Investigate root cause", "Check system resources", "Review process definition"],
					tenant_id=context.tenant_id
				)
				anomalies.append(anomaly)
			
			return anomalies
			
		except Exception as e:
			logger.error(f"Error checking metric anomalies: {e}")
			return []
	
	async def _detect_trend_anomalies(
		self,
		metrics: List[PerformanceDataPoint],
		baseline: PerformanceBaseline,
		context: APGTenantContext
	) -> List[PerformanceAnomaly]:
		"""Detect trend-based anomalies."""
		try:
			anomalies = []
			
			# Sort metrics by timestamp
			sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
			values = [m.value for m in sorted_metrics]
			
			if len(values) < 10:
				return anomalies
			
			# Calculate trend using linear regression
			x = np.arange(len(values))
			slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
			
			# Detect significant trends
			r_squared = r_value ** 2
			if r_squared > 0.3 and p_value < 0.05:  # Significant trend
				trend_direction = "increasing" if slope > 0 else "decreasing"
				
				# Check if trend is concerning
				if slope > 0 and sorted_metrics[0].metric_type in [
					PerformanceMetric.PROCESS_DURATION,
					PerformanceMetric.TASK_DURATION,
					PerformanceMetric.ERROR_RATE
				]:
					# Increasing duration or error rate is bad
					anomaly = PerformanceAnomaly(
						anomaly_type=AnomalyType.TREND_CHANGE,
						metric_type=sorted_metrics[0].metric_type,
						detected_value=values[-1],
						expected_value=baseline.baseline_value,
						deviation_score=abs(slope) * len(values),
						confidence=r_squared,
						description=f"Concerning {trend_direction} trend detected in {sorted_metrics[0].metric_type.value}",
						impact_assessment=f"Performance is trending worse (slope: {slope:.4f})",
						recommended_actions=["Monitor closely", "Investigate trend causes", "Consider process optimization"],
						tenant_id=context.tenant_id
					)
					anomalies.append(anomaly)
			
			return anomalies
			
		except Exception as e:
			logger.error(f"Error detecting trend anomalies: {e}")
			return []
	
	def _assess_impact(self, metric: PerformanceDataPoint, z_score: float) -> str:
		"""Assess impact of anomaly."""
		if z_score > 5:
			return "Critical impact - immediate attention required"
		elif z_score > 3:
			return "High impact - significant performance degradation"
		elif z_score > 2:
			return "Medium impact - noticeable performance deviation"
		else:
			return "Low impact - minor performance variation"
	
	def _generate_recommendations(self, metric: PerformanceDataPoint, z_score: float) -> List[str]:
		"""Generate recommendations for anomaly."""
		recommendations = []
		
		if metric.metric_type == PerformanceMetric.PROCESS_DURATION:
			recommendations.extend([
				"Review process definition for inefficiencies",
				"Check for bottlenecks in process flow",
				"Verify system resource availability"
			])
		elif metric.metric_type == PerformanceMetric.TASK_DURATION:
			recommendations.extend([
				"Review task complexity and requirements",
				"Check assignee workload and availability",
				"Verify task instructions and resources"
			])
		elif metric.metric_type == PerformanceMetric.QUEUE_TIME:
			recommendations.extend([
				"Review task assignment algorithms",
				"Check resource capacity",
				"Consider load balancing improvements"
			])
		
		if z_score > 3:
			recommendations.append("Escalate to system administrators")
		
		return recommendations


# =============================================================================
# Bottleneck Analysis Engine
# =============================================================================

class BottleneckAnalysisEngine:
	"""Analyze process bottlenecks and resource constraints."""
	
	def __init__(self):
		self.analysis_cache: Dict[str, BottleneckAnalysis] = {}
		
	async def analyze_process_bottlenecks(
		self,
		process_id: str,
		metrics: List[PerformanceDataPoint],
		context: APGTenantContext
	) -> BottleneckAnalysis:
		"""Analyze bottlenecks for a specific process."""
		try:
			# Group metrics by activity/task
			task_metrics = defaultdict(list)
			for metric in metrics:
				if metric.task_id and metric.process_id == process_id:
					task_metrics[metric.task_id].append(metric)
			
			# Calculate bottleneck scores
			bottleneck_scores = {}
			wait_times = {}
			
			for task_id, task_data in task_metrics.items():
				# Calculate average duration and queue time
				durations = [m.value for m in task_data if m.metric_type == PerformanceMetric.TASK_DURATION]
				queue_times = [m.value for m in task_data if m.metric_type == PerformanceMetric.QUEUE_TIME]
				
				avg_duration = statistics.mean(durations) if durations else 0
				avg_queue_time = statistics.mean(queue_times) if queue_times else 0
				
				# Bottleneck score combines duration and queue time
				bottleneck_score = avg_duration + (avg_queue_time * 2)  # Weight queue time more
				bottleneck_scores[task_id] = bottleneck_score
				wait_times[task_id] = avg_queue_time
			
			# Identify top bottlenecks
			sorted_bottlenecks = sorted(
				bottleneck_scores.items(),
				key=lambda x: x[1],
				reverse=True
			)
			
			bottleneck_activities = [task_id for task_id, _ in sorted_bottlenecks[:5]]
			
			# Generate analysis
			analysis = BottleneckAnalysis(
				process_id=process_id,
				bottleneck_activities=bottleneck_activities,
				bottleneck_scores=bottleneck_scores,
				wait_times=wait_times,
				resource_constraints=await self._identify_resource_constraints(task_metrics),
				impact_assessment=self._assess_bottleneck_impact(bottleneck_scores),
				optimization_suggestions=self._generate_optimization_suggestions(bottleneck_activities, wait_times),
				tenant_id=context.tenant_id
			)
			
			# Cache analysis
			self.analysis_cache[process_id] = analysis
			
			logger.info(f"Bottleneck analysis completed for process {process_id}: {len(bottleneck_activities)} bottlenecks identified")
			
			return analysis
			
		except Exception as e:
			logger.error(f"Error analyzing process bottlenecks: {e}")
			raise
	
	async def _identify_resource_constraints(
		self,
		task_metrics: Dict[str, List[PerformanceDataPoint]]
	) -> List[str]:
		"""Identify resource constraints from task metrics."""
		try:
			constraints = []
			
			# Analyze assignee workload
			assignee_workload = defaultdict(int)
			for task_id, metrics in task_metrics.items():
				for metric in metrics:
					assignee = metric.tags.get("assignee")
					if assignee:
						assignee_workload[assignee] += 1
			
			# Identify overloaded assignees
			if assignee_workload:
				max_workload = max(assignee_workload.values())
				avg_workload = statistics.mean(assignee_workload.values())
				
				if max_workload > avg_workload * 2:
					constraints.append("Uneven workload distribution among assignees")
			
			# Analyze queue times for resource bottlenecks
			high_queue_tasks = []
			for task_id, metrics in task_metrics.items():
				queue_times = [m.value for m in metrics if m.metric_type == PerformanceMetric.QUEUE_TIME]
				if queue_times and statistics.mean(queue_times) > 300:  # 5 minutes
					high_queue_tasks.append(task_id)
			
			if len(high_queue_tasks) > len(task_metrics) * 0.3:
				constraints.append("Insufficient processing capacity")
			
			return constraints
			
		except Exception as e:
			logger.error(f"Error identifying resource constraints: {e}")
			return []
	
	def _assess_bottleneck_impact(self, bottleneck_scores: Dict[str, float]) -> str:
		"""Assess overall impact of bottlenecks."""
		if not bottleneck_scores:
			return "No significant bottlenecks identified"
		
		max_score = max(bottleneck_scores.values())
		avg_score = statistics.mean(bottleneck_scores.values())
		
		if max_score > avg_score * 3:
			return "Critical bottleneck significantly impacting process performance"
		elif max_score > avg_score * 2:
			return "Moderate bottleneck causing noticeable delays"
		else:
			return "Minor performance variations detected"
	
	def _generate_optimization_suggestions(
		self,
		bottleneck_activities: List[str],
		wait_times: Dict[str, float]
	) -> List[str]:
		"""Generate optimization suggestions."""
		suggestions = []
		
		if bottleneck_activities:
			suggestions.append(f"Focus optimization efforts on activities: {', '.join(bottleneck_activities[:3])}")
		
		high_wait_activities = [
			activity for activity, wait_time in wait_times.items()
			if wait_time > 300  # 5 minutes
		]
		
		if high_wait_activities:
			suggestions.extend([
				"Consider parallel processing for high-wait activities",
				"Review resource allocation and capacity",
				"Implement priority-based task routing"
			])
		
		if len(suggestions) == 0:
			suggestions.append("Process performance is within acceptable ranges")
		
		return suggestions


# =============================================================================
# Performance Monitoring Service
# =============================================================================

class PerformanceMonitoringService:
	"""Main performance monitoring service."""
	
	def __init__(self):
		self.metrics_collector = RealTimeMetricsCollector()
		self.anomaly_detector = AnomalyDetectionEngine()
		self.bottleneck_analyzer = BottleneckAnalysisEngine()
		self.monitoring_tasks: Dict[str, asyncio.Task] = {}
		
	async def start_monitoring(self, context: APGTenantContext) -> WBPMServiceResponse:
		"""Start performance monitoring for tenant."""
		try:
			task_id = f"perf_monitor_{context.tenant_id}"
			
			if task_id in self.monitoring_tasks:
				return WBPMServiceResponse(
					success=True,
					message="Performance monitoring already active",
					data={"status": "active"}
				)
			
			# Start monitoring task
			task = asyncio.create_task(self._monitoring_loop(context))
			self.monitoring_tasks[task_id] = task
			
			logger.info(f"Performance monitoring started for tenant: {context.tenant_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Performance monitoring started successfully",
				data={"status": "started"}
			)
			
		except Exception as e:
			logger.error(f"Error starting performance monitoring: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to start monitoring: {e}",
				errors=[str(e)]
			)
	
	async def _monitoring_loop(self, context: APGTenantContext) -> None:
		"""Main performance monitoring loop."""
		try:
			logger.info(f"Performance monitoring loop started for tenant: {context.tenant_id}")
			
			while True:
				try:
					# Collect system metrics
					await self._collect_system_metrics(context)
					
					# Update baselines (every hour)
					if datetime.utcnow().minute == 0:
						await self._update_baselines(context)
					
					# Detect anomalies (every 5 minutes)
					if datetime.utcnow().minute % 5 == 0:
						await self._detect_performance_anomalies(context)
					
					# Analyze bottlenecks (every 15 minutes)
					if datetime.utcnow().minute % 15 == 0:
						await self._analyze_bottlenecks(context)
					
					# Sleep for next cycle
					await asyncio.sleep(60)  # Check every minute
					
				except asyncio.CancelledError:
					logger.info(f"Performance monitoring cancelled for tenant: {context.tenant_id}")
					break
				except Exception as e:
					logger.error(f"Error in performance monitoring loop: {e}")
					await asyncio.sleep(60)
					
		except Exception as e:
			logger.error(f"Fatal error in performance monitoring loop: {e}")
	
	async def _collect_system_metrics(self, context: APGTenantContext) -> None:
		"""Collect system-level performance metrics."""
		try:
			# Record system load metrics
			active_processes = len([
				p for p in self.metrics_collector.active_processes.values()
				if p["tenant_id"] == context.tenant_id
			])
			
			active_tasks = len([
				t for t in self.metrics_collector.active_tasks.values()
				if t["tenant_id"] == context.tenant_id
			])
			
			await self.metrics_collector.record_system_metric(
				PerformanceMetric.CONCURRENT_PROCESSES,
				float(active_processes),
				context
			)
			
			await self.metrics_collector.record_system_metric(
				PerformanceMetric.CONCURRENT_PROCESSES,
				float(active_tasks),
				context,
				unit="tasks"
			)
			
			# Calculate and record throughput
			await self.metrics_collector.calculate_throughput(
				timedelta(hours=1), context
			)
			
		except Exception as e:
			logger.error(f"Error collecting system metrics: {e}")
	
	async def _update_baselines(self, context: APGTenantContext) -> None:
		"""Update performance baselines."""
		try:
			# Update baselines for different metric types
			metric_types = [
				PerformanceMetric.PROCESS_DURATION,
				PerformanceMetric.TASK_DURATION,
				PerformanceMetric.QUEUE_TIME,
				PerformanceMetric.PROCESSING_TIME
			]
			
			for metric_type in metric_types:
				metrics = await self.metrics_collector.get_metrics(
					metric_type=metric_type,
					time_window=timedelta(days=30),
					tenant_id=context.tenant_id
				)
				
				if len(metrics) >= 10:  # Need minimum data for baseline
					await self.anomaly_detector.update_baseline(
						metric_type, metrics, context
					)
					
		except Exception as e:
			logger.error(f"Error updating baselines: {e}")
	
	async def _detect_performance_anomalies(self, context: APGTenantContext) -> None:
		"""Detect performance anomalies."""
		try:
			# Get recent metrics for anomaly detection
			recent_metrics = await self.metrics_collector.get_metrics(
				time_window=timedelta(hours=1),
				tenant_id=context.tenant_id
			)
			
			if recent_metrics:
				anomalies = await self.anomaly_detector.detect_anomalies(
					recent_metrics, context
				)
				
				if anomalies:
					logger.warning(f"Detected {len(anomalies)} performance anomalies for tenant {context.tenant_id}")
					
		except Exception as e:
			logger.error(f"Error detecting performance anomalies: {e}")
	
	async def _analyze_bottlenecks(self, context: APGTenantContext) -> None:
		"""Analyze process bottlenecks."""
		try:
			# Get unique process IDs from recent metrics
			recent_metrics = await self.metrics_collector.get_metrics(
				time_window=timedelta(hours=4),
				tenant_id=context.tenant_id
			)
			
			process_ids = set(
				m.process_id for m in recent_metrics
				if m.process_id
			)
			
			# Analyze bottlenecks for each process
			for process_id in list(process_ids)[:5]:  # Limit to 5 processes
				process_metrics = [
					m for m in recent_metrics
					if m.process_id == process_id
				]
				
				if len(process_metrics) >= 5:  # Need minimum data
					analysis = await self.bottleneck_analyzer.analyze_process_bottlenecks(
						process_id, process_metrics, context
					)
					
					if analysis.bottleneck_activities:
						logger.info(f"Bottleneck analysis for process {process_id}: {len(analysis.bottleneck_activities)} bottlenecks")
						
		except Exception as e:
			logger.error(f"Error analyzing bottlenecks: {e}")
	
	async def get_performance_summary(
		self,
		context: APGTenantContext,
		time_window: timedelta = timedelta(hours=24)
	) -> WBPMServiceResponse:
		"""Get performance summary for tenant."""
		try:
			# Get metrics for time window
			metrics = await self.metrics_collector.get_metrics(
				time_window=time_window,
				tenant_id=context.tenant_id
			)
			
			# Calculate summary statistics
			summary = {
				"time_window": str(time_window),
				"total_metrics": len(metrics),
				"metrics_by_type": {},
				"active_processes": len([
					p for p in self.metrics_collector.active_processes.values()
					if p["tenant_id"] == context.tenant_id
				]),
				"active_tasks": len([
					t for t in self.metrics_collector.active_tasks.values()
					if t["tenant_id"] == context.tenant_id
				]),
				"recent_anomalies": len([
					a for a in self.anomaly_detector.detected_anomalies
					if a.tenant_id == context.tenant_id and
					a.detected_at >= datetime.utcnow() - time_window
				])
			}
			
			# Group metrics by type
			metrics_by_type = defaultdict(list)
			for metric in metrics:
				metrics_by_type[metric.metric_type].append(metric.value)
			
			# Calculate statistics for each type
			for metric_type, values in metrics_by_type.items():
				if values:
					summary["metrics_by_type"][metric_type.value] = {
						"count": len(values),
						"mean": statistics.mean(values),
						"median": statistics.median(values),
						"min": min(values),
						"max": max(values),
						"std_dev": statistics.stdev(values) if len(values) > 1 else 0
					}
			
			return WBPMServiceResponse(
				success=True,
				message="Performance summary retrieved successfully",
				data=summary
			)
			
		except Exception as e:
			logger.error(f"Error getting performance summary: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to get performance summary: {e}",
				errors=[str(e)]
			)


# =============================================================================
# Service Factory
# =============================================================================

def create_performance_monitoring_service() -> PerformanceMonitoringService:
	"""Create and configure performance monitoring service."""
	service = PerformanceMonitoringService()
	logger.info("Performance monitoring service created and configured")
	return service


# Export main classes
__all__ = [
	'PerformanceMonitoringService',
	'RealTimeMetricsCollector',
	'AnomalyDetectionEngine',
	'BottleneckAnalysisEngine',
	'PerformanceDataPoint',
	'PerformanceBaseline',
	'PerformanceAnomaly',
	'PerformanceTrend',
	'BottleneckAnalysis',
	'ResourceUtilization',
	'PerformanceMetric',
	'AnomalyType',
	'MonitoringLevel',
	'create_performance_monitoring_service'
]