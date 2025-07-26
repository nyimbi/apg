"""
APG Workflow & Business Process Management - Analytics Engine

Advanced process analytics with real-time monitoring, performance insights,
bottleneck detection, and predictive analytics for workflow optimization.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import uuid

from models import (
	APGTenantContext, WBPMServiceResponse, WBPMPagedResponse,
	WBPMProcessInstance, WBPMTask, TaskStatus, InstanceStatus, TaskPriority
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Analytics Core Classes
# =============================================================================

class MetricType(str, Enum):
	"""Types of process metrics."""
	VOLUME = "volume"
	DURATION = "duration"
	EFFICIENCY = "efficiency"
	QUALITY = "quality"
	COST = "cost"
	RESOURCE_UTILIZATION = "resource_utilization"
	SLA_COMPLIANCE = "sla_compliance"
	ERROR_RATE = "error_rate"


class TimeGranularity(str, Enum):
	"""Time granularity for analytics."""
	HOUR = "hour"
	DAY = "day"
	WEEK = "week"
	MONTH = "month"
	QUARTER = "quarter"
	YEAR = "year"


class AnalysisType(str, Enum):
	"""Types of process analysis."""
	DESCRIPTIVE = "descriptive"
	DIAGNOSTIC = "diagnostic"
	PREDICTIVE = "predictive"
	PRESCRIPTIVE = "prescriptive"


class BottleneckType(str, Enum):
	"""Types of process bottlenecks."""
	ACTIVITY_DELAY = "activity_delay"
	RESOURCE_CONTENTION = "resource_contention"
	APPROVAL_BACKLOG = "approval_backlog"
	GATEWAY_CONGESTION = "gateway_congestion"
	EXTERNAL_DEPENDENCY = "external_dependency"


@dataclass
class ProcessMetric:
	"""Individual process metric measurement."""
	metric_id: str = field(default_factory=lambda: f"metric_{uuid.uuid4().hex}")
	process_id: str = ""
	metric_type: MetricType = MetricType.DURATION
	metric_name: str = ""
	metric_value: float = 0.0
	metric_unit: str = ""
	timestamp: datetime = field(default_factory=datetime.utcnow)
	dimensions: Dict[str, Any] = field(default_factory=dict)
	tenant_id: str = ""


@dataclass
class ProcessKPI:
	"""Key Performance Indicator for processes."""
	kpi_id: str = field(default_factory=lambda: f"kpi_{uuid.uuid4().hex}")
	kpi_name: str = ""
	kpi_description: str = ""
	target_value: float = 0.0
	current_value: float = 0.0
	variance_percentage: float = 0.0
	trend_direction: str = "stable"  # improving, declining, stable
	calculation_formula: str = ""
	update_frequency: str = "daily"
	owner: str = ""
	tenant_id: str = ""


@dataclass
class ProcessBottleneck:
	"""Identified process bottleneck."""
	bottleneck_id: str = field(default_factory=lambda: f"bottleneck_{uuid.uuid4().hex}")
	process_id: str = ""
	bottleneck_type: BottleneckType = BottleneckType.ACTIVITY_DELAY
	element_id: str = ""
	element_name: str = ""
	severity_score: float = 0.0  # 0-100 scale
	impact_description: str = ""
	suggested_improvements: List[str] = field(default_factory=list)
	estimated_impact: Dict[str, float] = field(default_factory=dict)
	detected_at: datetime = field(default_factory=datetime.utcnow)
	is_resolved: bool = False
	tenant_id: str = ""


@dataclass
class ProcessInsight:
	"""AI-generated process insight."""
	insight_id: str = field(default_factory=lambda: f"insight_{uuid.uuid4().hex}")
	process_id: str = ""
	insight_type: str = ""
	title: str = ""
	description: str = ""
	confidence_score: float = 0.0  # 0-1 scale
	impact_assessment: str = ""
	recommended_actions: List[str] = field(default_factory=list)
	supporting_data: Dict[str, Any] = field(default_factory=dict)
	generated_at: datetime = field(default_factory=datetime.utcnow)
	tenant_id: str = ""


@dataclass
class AnalyticsReport:
	"""Comprehensive analytics report."""
	report_id: str = field(default_factory=lambda: f"report_{uuid.uuid4().hex}")
	report_name: str = ""
	process_ids: List[str] = field(default_factory=list)
	report_type: str = "performance"
	time_period: Dict[str, Any] = field(default_factory=dict)
	metrics: List[ProcessMetric] = field(default_factory=list)
	kpis: List[ProcessKPI] = field(default_factory=list)
	bottlenecks: List[ProcessBottleneck] = field(default_factory=list)
	insights: List[ProcessInsight] = field(default_factory=list)
	charts: List[Dict[str, Any]] = field(default_factory=list)
	generated_by: str = ""
	generated_at: datetime = field(default_factory=datetime.utcnow)
	tenant_id: str = ""


# =============================================================================
# Metrics Collection Engine
# =============================================================================

class MetricsCollector:
	"""Collect and process workflow metrics."""
	
	def __init__(self):
		self.metrics_buffer: List[ProcessMetric] = []
		self.metric_calculations = self._initialize_metric_calculations()
	
	async def collect_instance_metrics(
		self,
		instance: WBPMProcessInstance,
		context: APGTenantContext
	) -> List[ProcessMetric]:
		"""Collect metrics from process instance."""
		metrics = []
		
		try:
			# Duration metrics
			if instance.start_time and instance.end_time:
				duration_hours = (instance.end_time - instance.start_time).total_seconds() / 3600
				
				metrics.append(ProcessMetric(
					process_id=instance.process_id,
					metric_type=MetricType.DURATION,
					metric_name="instance_duration",
					metric_value=duration_hours,
					metric_unit="hours",
					dimensions={
						"instance_id": instance.id,
						"status": instance.instance_status,
						"priority": instance.priority
					},
					tenant_id=context.tenant_id
				))
			
			# Volume metrics
			metrics.append(ProcessMetric(
				process_id=instance.process_id,
				metric_type=MetricType.VOLUME,
				metric_name="instance_count",
				metric_value=1.0,
				metric_unit="count",
				dimensions={
					"status": instance.instance_status,
					"initiated_by": instance.initiated_by
				},
				tenant_id=context.tenant_id
			))
			
			# SLA compliance
			if instance.due_date:
				is_on_time = (instance.end_time or datetime.utcnow()) <= instance.due_date
				metrics.append(ProcessMetric(
					process_id=instance.process_id,
					metric_type=MetricType.SLA_COMPLIANCE,
					metric_name="sla_compliance",
					metric_value=1.0 if is_on_time else 0.0,
					metric_unit="boolean",
					dimensions={
						"instance_id": instance.id,
						"due_date": instance.due_date.isoformat()
					},
					tenant_id=context.tenant_id
				))
			
			logger.debug(f"Collected {len(metrics)} metrics for instance {instance.id}")
			return metrics
			
		except Exception as e:
			logger.error(f"Error collecting instance metrics: {e}")
			return []
	
	async def collect_task_metrics(
		self,
		task: WBPMTask,
		context: APGTenantContext
	) -> List[ProcessMetric]:
		"""Collect metrics from task."""
		metrics = []
		
		try:
			# Task duration
			if task.claim_time and task.completion_time:
				duration_minutes = (task.completion_time - task.claim_time).total_seconds() / 60
				
				metrics.append(ProcessMetric(
					process_id=task.process_instance_id,
					metric_type=MetricType.DURATION,
					metric_name="task_duration",
					metric_value=duration_minutes,
					metric_unit="minutes",
					dimensions={
						"task_id": task.id,
						"task_name": task.task_name,
						"assignee": task.assignee,
						"priority": task.priority
					},
					tenant_id=context.tenant_id
				))
			
			# Task waiting time
			if task.create_time and task.claim_time:
				waiting_minutes = (task.claim_time - task.create_time).total_seconds() / 60
				
				metrics.append(ProcessMetric(
					process_id=task.process_instance_id,
					metric_type=MetricType.EFFICIENCY,
					metric_name="task_waiting_time",
					metric_value=waiting_minutes,
					metric_unit="minutes",
					dimensions={
						"task_id": task.id,
						"task_name": task.task_name
					},
					tenant_id=context.tenant_id
				))
			
			# Task count by status
			metrics.append(ProcessMetric(
				process_id=task.process_instance_id,
				metric_type=MetricType.VOLUME,
				metric_name="task_count",
				metric_value=1.0,
				metric_unit="count",
				dimensions={
					"status": task.task_status,
					"assignee": task.assignee
				},
				tenant_id=context.tenant_id
			))
			
			return metrics
			
		except Exception as e:
			logger.error(f"Error collecting task metrics: {e}")
			return []
	
	async def store_metrics(self, metrics: List[ProcessMetric]) -> None:
		"""Store metrics in buffer."""
		self.metrics_buffer.extend(metrics)
		
		# Keep buffer size manageable
		if len(self.metrics_buffer) > 10000:
			self.metrics_buffer = self.metrics_buffer[-5000:]
	
	def _initialize_metric_calculations(self) -> Dict[str, Any]:
		"""Initialize metric calculation functions."""
		return {
			'average_duration': self._calculate_average_duration,
			'throughput': self._calculate_throughput,
			'cycle_time': self._calculate_cycle_time,
			'sla_compliance_rate': self._calculate_sla_compliance_rate
		}
	
	async def _calculate_average_duration(self, process_id: str, time_window: timedelta) -> float:
		"""Calculate average process duration."""
		cutoff_time = datetime.utcnow() - time_window
		duration_metrics = [
			metric for metric in self.metrics_buffer
			if (metric.process_id == process_id and
				metric.metric_name == "instance_duration" and
				metric.timestamp >= cutoff_time)
		]
		
		if duration_metrics:
			return statistics.mean([metric.metric_value for metric in duration_metrics])
		return 0.0
	
	async def _calculate_throughput(self, process_id: str, time_window: timedelta) -> float:
		"""Calculate process throughput (instances per hour)."""
		cutoff_time = datetime.utcnow() - time_window
		completed_metrics = [
			metric for metric in self.metrics_buffer
			if (metric.process_id == process_id and
				metric.metric_name == "instance_count" and
				metric.dimensions.get("status") == InstanceStatus.COMPLETED and
				metric.timestamp >= cutoff_time)
		]
		
		window_hours = time_window.total_seconds() / 3600
		return len(completed_metrics) / window_hours if window_hours > 0 else 0.0
	
	async def _calculate_cycle_time(self, process_id: str, time_window: timedelta) -> float:
		"""Calculate average cycle time."""
		# Same as average duration for simplicity
		return await self._calculate_average_duration(process_id, time_window)
	
	async def _calculate_sla_compliance_rate(self, process_id: str, time_window: timedelta) -> float:
		"""Calculate SLA compliance rate."""
		cutoff_time = datetime.utcnow() - time_window
		sla_metrics = [
			metric for metric in self.metrics_buffer
			if (metric.process_id == process_id and
				metric.metric_name == "sla_compliance" and
				metric.timestamp >= cutoff_time)
		]
		
		if sla_metrics:
			return statistics.mean([metric.metric_value for metric in sla_metrics])
		return 0.0


# =============================================================================
# Bottleneck Detection Engine
# =============================================================================

class BottleneckDetector:
	"""Detect and analyze process bottlenecks."""
	
	def __init__(self):
		self.detection_algorithms = self._initialize_detection_algorithms()
		self.severity_thresholds = {
			"critical": 80.0,
			"high": 60.0,
			"medium": 40.0,
			"low": 20.0
		}
	
	async def detect_bottlenecks(
		self,
		process_id: str,
		metrics: List[ProcessMetric],
		context: APGTenantContext
	) -> List[ProcessBottleneck]:
		"""Detect bottlenecks in process."""
		bottlenecks = []
		
		for algorithm_name, algorithm_func in self.detection_algorithms.items():
			try:
				detected = await algorithm_func(process_id, metrics, context)
				bottlenecks.extend(detected)
			except Exception as e:
				logger.error(f"Error in bottleneck detection algorithm {algorithm_name}: {e}")
		
		# Remove duplicates and sort by severity
		unique_bottlenecks = self._deduplicate_bottlenecks(bottlenecks)
		unique_bottlenecks.sort(key=lambda x: x.severity_score, reverse=True)
		
		return unique_bottlenecks
	
	async def _detect_activity_delays(
		self,
		process_id: str,
		metrics: List[ProcessMetric],
		context: APGTenantContext
	) -> List[ProcessBottleneck]:
		"""Detect activities with unusual delays."""
		bottlenecks = []
		
		# Group task duration metrics by task name
		task_durations = defaultdict(list)
		for metric in metrics:
			if (metric.process_id == process_id and
				metric.metric_name == "task_duration"):
				task_name = metric.dimensions.get("task_name", "unknown")
				task_durations[task_name].append(metric.metric_value)
		
		# Analyze each task type
		for task_name, durations in task_durations.items():
			if len(durations) < 3:  # Need minimum samples
				continue
			
			avg_duration = statistics.mean(durations)
			std_duration = statistics.stdev(durations) if len(durations) > 1 else 0
			
			# Detect outliers (values > mean + 2*std)
			threshold = avg_duration + 2 * std_duration
			outlier_count = sum(1 for d in durations if d > threshold)
			outlier_percentage = (outlier_count / len(durations)) * 100
			
			if outlier_percentage > 20:  # More than 20% are outliers
				severity = min(100.0, outlier_percentage * 2)
				
				bottlenecks.append(ProcessBottleneck(
					process_id=process_id,
					bottleneck_type=BottleneckType.ACTIVITY_DELAY,
					element_id=f"task_{task_name}",
					element_name=task_name,
					severity_score=severity,
					impact_description=f"Task '{task_name}' shows {outlier_percentage:.1f}% delay variance",
					suggested_improvements=[
						"Review task complexity and requirements",
						"Provide additional training to task performers",
						"Consider task automation or simplification"
					],
					estimated_impact={
						"time_reduction_percentage": min(30.0, outlier_percentage / 2),
						"efficiency_improvement": min(25.0, outlier_percentage / 3)
					},
					tenant_id=context.tenant_id
				))
		
		return bottlenecks
	
	async def _detect_resource_contention(
		self,
		process_id: str,
		metrics: List[ProcessMetric],
		context: APGTenantContext
	) -> List[ProcessBottleneck]:
		"""Detect resource contention bottlenecks."""
		bottlenecks = []
		
		# Analyze task waiting times by assignee
		assignee_wait_times = defaultdict(list)
		for metric in metrics:
			if (metric.process_id == process_id and
				metric.metric_name == "task_waiting_time"):
				assignee = metric.dimensions.get("assignee", "unassigned")
				assignee_wait_times[assignee].append(metric.metric_value)
		
		# Find assignees with high waiting times
		for assignee, wait_times in assignee_wait_times.items():
			if len(wait_times) < 2:
				continue
			
			avg_wait = statistics.mean(wait_times)
			if avg_wait > 60:  # More than 1 hour average wait
				severity = min(100.0, avg_wait / 2)  # Scale based on wait time
				
				bottlenecks.append(ProcessBottleneck(
					process_id=process_id,
					bottleneck_type=BottleneckType.RESOURCE_CONTENTION,
					element_id=f"user_{assignee}",
					element_name=f"User {assignee}",
					severity_score=severity,
					impact_description=f"High task waiting time ({avg_wait:.1f} minutes) for assignee",
					suggested_improvements=[
						"Balance workload distribution",
						"Add additional resources to high-demand areas",
						"Implement skill-based task routing"
					],
					estimated_impact={
						"wait_time_reduction_minutes": min(avg_wait * 0.6, 120),
						"throughput_improvement_percentage": min(25.0, severity / 4)
					},
					tenant_id=context.tenant_id
				))
		
		return bottlenecks
	
	async def _detect_gateway_congestion(
		self,
		process_id: str,
		metrics: List[ProcessMetric],
		context: APGTenantContext
	) -> List[ProcessBottleneck]:
		"""Detect gateway congestion bottlenecks."""
		bottlenecks = []
		
		# This would analyze flow patterns through gateways
		# For demonstration, we'll create a simple example
		
		# Count instances by priority to simulate gateway analysis
		priority_counts = Counter()
		for metric in metrics:
			if (metric.process_id == process_id and
				metric.metric_name == "instance_count"):
				priority = metric.dimensions.get("priority", "medium")
				priority_counts[priority] += 1
		
		total_instances = sum(priority_counts.values())
		if total_instances > 0:
			high_priority_percentage = (priority_counts.get("high", 0) + 
										priority_counts.get("critical", 0)) / total_instances * 100
			
			if high_priority_percentage > 40:  # More than 40% high priority
				bottlenecks.append(ProcessBottleneck(
					process_id=process_id,
					bottleneck_type=BottleneckType.GATEWAY_CONGESTION,
					element_id="priority_gateway",
					element_name="Priority Decision Point",
					severity_score=high_priority_percentage,
					impact_description=f"High priority instance ratio ({high_priority_percentage:.1f}%)",
					suggested_improvements=[
						"Review priority assignment criteria",
						"Implement dynamic priority adjustment",
						"Add express lanes for critical processes"
					],
					estimated_impact={
						"processing_time_reduction_percentage": min(20.0, high_priority_percentage / 3)
					},
					tenant_id=context.tenant_id
				))
		
		return bottlenecks
	
	def _deduplicate_bottlenecks(self, bottlenecks: List[ProcessBottleneck]) -> List[ProcessBottleneck]:
		"""Remove duplicate bottlenecks."""
		seen = set()
		unique = []
		
		for bottleneck in bottlenecks:
			key = (bottleneck.process_id, bottleneck.element_id, bottleneck.bottleneck_type)
			if key not in seen:
				seen.add(key)
				unique.append(bottleneck)
		
		return unique
	
	def _initialize_detection_algorithms(self) -> Dict[str, Any]:
		"""Initialize bottleneck detection algorithms."""
		return {
			'activity_delays': self._detect_activity_delays,
			'resource_contention': self._detect_resource_contention,
			'gateway_congestion': self._detect_gateway_congestion
		}


# =============================================================================
# AI Insights Generator
# =============================================================================

class InsightsGenerator:
	"""Generate AI-powered process insights."""
	
	def __init__(self):
		self.insight_templates = self._initialize_insight_templates()
	
	async def generate_insights(
		self,
		process_id: str,
		metrics: List[ProcessMetric],
		bottlenecks: List[ProcessBottleneck],
		context: APGTenantContext
	) -> List[ProcessInsight]:
		"""Generate process insights from metrics and bottlenecks."""
		insights = []
		
		# Performance insights
		insights.extend(await self._generate_performance_insights(process_id, metrics, context))
		
		# Efficiency insights
		insights.extend(await self._generate_efficiency_insights(process_id, metrics, context))
		
		# Bottleneck insights
		insights.extend(await self._generate_bottleneck_insights(process_id, bottlenecks, context))
		
		# Trend insights
		insights.extend(await self._generate_trend_insights(process_id, metrics, context))
		
		# Sort by confidence score
		insights.sort(key=lambda x: x.confidence_score, reverse=True)
		
		return insights[:10]  # Return top 10 insights
	
	async def _generate_performance_insights(
		self,
		process_id: str,
		metrics: List[ProcessMetric],
		context: APGTenantContext
	) -> List[ProcessInsight]:
		"""Generate performance-related insights."""
		insights = []
		
		# Analyze duration metrics
		duration_metrics = [
			m for m in metrics
			if m.process_id == process_id and m.metric_name == "instance_duration"
		]
		
		if len(duration_metrics) >= 5:
			durations = [m.metric_value for m in duration_metrics]
			avg_duration = statistics.mean(durations)
			
			if avg_duration > 24:  # More than 24 hours
				insights.append(ProcessInsight(
					process_id=process_id,
					insight_type="performance",
					title="Long Process Duration Detected",
					description=f"Average process duration is {avg_duration:.1f} hours, which may indicate inefficiencies.",
					confidence_score=0.8,
					impact_assessment="High - Long durations affect customer satisfaction and resource utilization",
					recommended_actions=[
						"Analyze process steps for optimization opportunities",
						"Consider parallel execution where possible",
						"Review approval workflows for delays"
					],
					supporting_data={
						"average_duration_hours": avg_duration,
						"sample_size": len(duration_metrics)
					},
					tenant_id=context.tenant_id
				))
		
		return insights
	
	async def _generate_efficiency_insights(
		self,
		process_id: str,
		metrics: List[ProcessMetric],
		context: APGTenantContext
	) -> List[ProcessInsight]:
		"""Generate efficiency-related insights."""
		insights = []
		
		# Analyze waiting time vs processing time
		wait_metrics = [
			m for m in metrics
			if m.process_id == process_id and m.metric_name == "task_waiting_time"
		]
		
		process_metrics = [
			m for m in metrics
			if m.process_id == process_id and m.metric_name == "task_duration"
		]
		
		if wait_metrics and process_metrics:
			avg_wait = statistics.mean([m.metric_value for m in wait_metrics])
			avg_process = statistics.mean([m.metric_value for m in process_metrics])
			
			if avg_wait > avg_process * 2:  # Waiting time > 2x processing time
				efficiency_ratio = avg_process / (avg_wait + avg_process) * 100
				
				insights.append(ProcessInsight(
					process_id=process_id,
					insight_type="efficiency",
					title="Low Process Efficiency Detected",
					description=f"Process efficiency is {efficiency_ratio:.1f}% due to high waiting times.",
					confidence_score=0.85,
					impact_assessment="Medium - Reducing wait times can significantly improve overall process speed",
					recommended_actions=[
						"Implement automated task assignment",
						"Balance workload across team members",
						"Review task prioritization logic"
					],
					supporting_data={
						"efficiency_percentage": efficiency_ratio,
						"average_wait_minutes": avg_wait,
						"average_process_minutes": avg_process
					},
					tenant_id=context.tenant_id
				))
		
		return insights
	
	async def _generate_bottleneck_insights(
		self,
		process_id: str,
		bottlenecks: List[ProcessBottleneck],
		context: APGTenantContext
	) -> List[ProcessInsight]:
		"""Generate insights from detected bottlenecks."""
		insights = []
		
		if not bottlenecks:
			insights.append(ProcessInsight(
				process_id=process_id,
				insight_type="bottleneck",
				title="No Critical Bottlenecks Detected",
				description="Process is performing well with no major bottlenecks identified.",
				confidence_score=0.7,
				impact_assessment="Positive - Process efficiency is good",
				recommended_actions=[
					"Monitor performance regularly",
					"Continue current practices",
					"Look for minor optimization opportunities"
				],
				supporting_data={"bottleneck_count": 0},
				tenant_id=context.tenant_id
			))
		else:
			# Focus on highest severity bottleneck
			top_bottleneck = max(bottlenecks, key=lambda x: x.severity_score)
			
			insights.append(ProcessInsight(
				process_id=process_id,
				insight_type="bottleneck",
				title=f"Critical Bottleneck: {top_bottleneck.element_name}",
				description=f"{top_bottleneck.impact_description} (Severity: {top_bottleneck.severity_score:.1f}/100)",
				confidence_score=0.9,
				impact_assessment="High - Addressing this bottleneck can significantly improve process performance",
				recommended_actions=top_bottleneck.suggested_improvements,
				supporting_data={
					"severity_score": top_bottleneck.severity_score,
					"bottleneck_type": top_bottleneck.bottleneck_type.value,
					"estimated_impact": top_bottleneck.estimated_impact
				},
				tenant_id=context.tenant_id
			))
		
		return insights
	
	async def _generate_trend_insights(
		self,
		process_id: str,
		metrics: List[ProcessMetric],
		context: APGTenantContext
	) -> List[ProcessInsight]:
		"""Generate trend-based insights."""
		insights = []
		
		# Analyze volume trends
		volume_metrics = [
			m for m in metrics
			if m.process_id == process_id and m.metric_name == "instance_count"
		]
		
		if len(volume_metrics) >= 7:  # Need at least a week of data
			# Group by day
			daily_counts = defaultdict(int)
			for metric in volume_metrics:
				day_key = metric.timestamp.date()
				daily_counts[day_key] += metric.metric_value
			
			if len(daily_counts) >= 3:
				counts = list(daily_counts.values())
				if len(counts) >= 3:
					# Simple trend detection
					recent_avg = statistics.mean(counts[-3:])
					earlier_avg = statistics.mean(counts[:-3]) if len(counts) > 3 else recent_avg
					
					if recent_avg > earlier_avg * 1.2:  # 20% increase
						insights.append(ProcessInsight(
							process_id=process_id,
							insight_type="trend",
							title="Increasing Process Volume Trend",
							description=f"Process volume has increased by {((recent_avg - earlier_avg) / earlier_avg * 100):.1f}% recently.",
							confidence_score=0.75,
							impact_assessment="Medium - Prepare for higher capacity requirements",
							recommended_actions=[
								"Scale up resources to handle increased volume",
								"Review process capacity planning",
								"Consider automation for routine tasks"
							],
							supporting_data={
								"recent_daily_average": recent_avg,
								"previous_daily_average": earlier_avg,
								"trend_percentage": (recent_avg - earlier_avg) / earlier_avg * 100
							},
							tenant_id=context.tenant_id
						))
		
		return insights
	
	def _initialize_insight_templates(self) -> Dict[str, Any]:
		"""Initialize insight templates."""
		return {
			"performance_degradation": {
				"title": "Performance Degradation Detected",
				"threshold": 0.8,
				"impact": "High"
			},
			"efficiency_opportunity": {
				"title": "Efficiency Improvement Opportunity",
				"threshold": 0.7,
				"impact": "Medium"
			},
			"volume_trend": {
				"title": "Volume Trend Analysis",
				"threshold": 0.6,
				"impact": "Variable"
			}
		}


# =============================================================================
# Analytics Engine
# =============================================================================

class ProcessAnalyticsEngine:
	"""Main analytics engine for workflow processes."""
	
	def __init__(self):
		self.metrics_collector = MetricsCollector()
		self.bottleneck_detector = BottleneckDetector()
		self.insights_generator = InsightsGenerator()
		self.cached_reports: Dict[str, AnalyticsReport] = {}
	
	async def generate_process_report(
		self,
		process_ids: List[str],
		time_period: Dict[str, Any],
		context: APGTenantContext,
		report_type: str = "comprehensive"
	) -> WBPMServiceResponse:
		"""Generate comprehensive process analytics report."""
		try:
			report = AnalyticsReport(
				report_name=f"Process Analytics Report - {datetime.utcnow().strftime('%Y-%m-%d')}",
				process_ids=process_ids,
				report_type=report_type,
				time_period=time_period,
				generated_by=context.user_id,
				tenant_id=context.tenant_id
			)
			
			# Collect metrics for all processes
			all_metrics = []
			for process_id in process_ids:
				process_metrics = await self._get_process_metrics(process_id, time_period, context)
				all_metrics.extend(process_metrics)
				report.metrics.extend(process_metrics)
			
			# Generate KPIs
			report.kpis = await self._calculate_process_kpis(process_ids, all_metrics, context)
			
			# Detect bottlenecks
			all_bottlenecks = []
			for process_id in process_ids:
				process_metrics = [m for m in all_metrics if m.process_id == process_id]
				bottlenecks = await self.bottleneck_detector.detect_bottlenecks(
					process_id, process_metrics, context
				)
				all_bottlenecks.extend(bottlenecks)
			report.bottlenecks = all_bottlenecks
			
			# Generate insights
			all_insights = []
			for process_id in process_ids:
				process_metrics = [m for m in all_metrics if m.process_id == process_id]
				process_bottlenecks = [b for b in all_bottlenecks if b.process_id == process_id]
				insights = await self.insights_generator.generate_insights(
					process_id, process_metrics, process_bottlenecks, context
				)
				all_insights.extend(insights)
			report.insights = all_insights
			
			# Generate charts
			report.charts = await self._generate_report_charts(report, context)
			
			# Cache report
			self.cached_reports[report.report_id] = report
			
			logger.info(f"Analytics report generated: {report.report_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Analytics report generated successfully",
				data={
					"report_id": report.report_id,
					"process_count": len(process_ids),
					"metrics_count": len(report.metrics),
					"kpis_count": len(report.kpis),
					"bottlenecks_count": len(report.bottlenecks),
					"insights_count": len(report.insights),
					"charts_count": len(report.charts)
				}
			)
			
		except Exception as e:
			logger.error(f"Error generating analytics report: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to generate analytics report: {e}",
				errors=[str(e)]
			)
	
	async def get_real_time_dashboard(
		self,
		process_ids: List[str],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Get real-time dashboard data."""
		try:
			dashboard_data = {}
			
			for process_id in process_ids:
				# Get recent metrics (last hour)
				time_window = timedelta(hours=1)
				recent_metrics = await self._get_recent_metrics(process_id, time_window, context)
				
				# Calculate real-time KPIs
				dashboard_data[process_id] = {
					"active_instances": len([
						m for m in recent_metrics
						if m.metric_name == "instance_count" and
						m.dimensions.get("status") in ["running", "suspended"]
					]),
					"completed_instances": len([
						m for m in recent_metrics
						if m.metric_name == "instance_count" and
						m.dimensions.get("status") == "completed"
					]),
					"average_duration": await self.metrics_collector._calculate_average_duration(
						process_id, time_window
					),
					"throughput": await self.metrics_collector._calculate_throughput(
						process_id, time_window
					),
					"sla_compliance": await self.metrics_collector._calculate_sla_compliance_rate(
						process_id, time_window
					)
				}
			
			return WBPMServiceResponse(
				success=True,
				message="Real-time dashboard data retrieved successfully",
				data={
					"timestamp": datetime.utcnow().isoformat(),
					"process_data": dashboard_data,
					"refresh_interval": 30  # seconds
				}
			)
			
		except Exception as e:
			logger.error(f"Error getting real-time dashboard: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to get dashboard data: {e}",
				errors=[str(e)]
			)
	
	async def _get_process_metrics(
		self,
		process_id: str,
		time_period: Dict[str, Any],
		context: APGTenantContext
	) -> List[ProcessMetric]:
		"""Get metrics for process within time period."""
		start_time = datetime.fromisoformat(time_period.get("start", "2025-01-01T00:00:00"))
		end_time = datetime.fromisoformat(time_period.get("end", datetime.utcnow().isoformat()))
		
		return [
			metric for metric in self.metrics_collector.metrics_buffer
			if (metric.process_id == process_id and
				start_time <= metric.timestamp <= end_time and
				metric.tenant_id == context.tenant_id)
		]
	
	async def _get_recent_metrics(
		self,
		process_id: str,
		time_window: timedelta,
		context: APGTenantContext
	) -> List[ProcessMetric]:
		"""Get recent metrics for process."""
		cutoff_time = datetime.utcnow() - time_window
		
		return [
			metric for metric in self.metrics_collector.metrics_buffer
			if (metric.process_id == process_id and
				metric.timestamp >= cutoff_time and
				metric.tenant_id == context.tenant_id)
		]
	
	async def _calculate_process_kpis(
		self,
		process_ids: List[str],
		metrics: List[ProcessMetric],
		context: APGTenantContext
	) -> List[ProcessKPI]:
		"""Calculate KPIs from metrics."""
		kpis = []
		
		for process_id in process_ids:
			process_metrics = [m for m in metrics if m.process_id == process_id]
			
			# Average Duration KPI
			duration_metrics = [m for m in process_metrics if m.metric_name == "instance_duration"]
			if duration_metrics:
				avg_duration = statistics.mean([m.metric_value for m in duration_metrics])
				kpis.append(ProcessKPI(
					kpi_name="Average Process Duration",
					kpi_description="Average time to complete process instances",
					target_value=8.0,  # Target 8 hours
					current_value=avg_duration,
					variance_percentage=((avg_duration - 8.0) / 8.0) * 100,
					trend_direction="stable",
					calculation_formula="AVERAGE(instance_duration)",
					owner=context.user_id,
					tenant_id=context.tenant_id
				))
			
			# SLA Compliance KPI
			sla_metrics = [m for m in process_metrics if m.metric_name == "sla_compliance"]
			if sla_metrics:
				compliance_rate = statistics.mean([m.metric_value for m in sla_metrics]) * 100
				kpis.append(ProcessKPI(
					kpi_name="SLA Compliance Rate",
					kpi_description="Percentage of instances completed within SLA",
					target_value=95.0,  # Target 95%
					current_value=compliance_rate,
					variance_percentage=((compliance_rate - 95.0) / 95.0) * 100,
					trend_direction="stable",
					calculation_formula="AVERAGE(sla_compliance) * 100",
					owner=context.user_id,
					tenant_id=context.tenant_id
				))
		
		return kpis
	
	async def _generate_report_charts(
		self,
		report: AnalyticsReport,
		context: APGTenantContext
	) -> List[Dict[str, Any]]:
		"""Generate chart data for report."""
		charts = []
		
		# Volume trend chart
		volume_chart = {
			"chart_id": f"volume_{uuid.uuid4().hex[:8]}",
			"chart_type": "line",
			"title": "Process Volume Trend",
			"data": self._prepare_volume_chart_data(report.metrics),
			"options": {
				"x_axis": "Date",
				"y_axis": "Instance Count",
				"time_granularity": "day"
			}
		}
		charts.append(volume_chart)
		
		# Duration distribution chart
		duration_chart = {
			"chart_id": f"duration_{uuid.uuid4().hex[:8]}",
			"chart_type": "histogram",
			"title": "Process Duration Distribution",
			"data": self._prepare_duration_chart_data(report.metrics),
			"options": {
				"x_axis": "Duration (Hours)",
				"y_axis": "Frequency",
				"bins": 20
			}
		}
		charts.append(duration_chart)
		
		# Bottleneck severity chart
		if report.bottlenecks:
			bottleneck_chart = {
				"chart_id": f"bottlenecks_{uuid.uuid4().hex[:8]}",
				"chart_type": "bar",
				"title": "Bottleneck Severity Analysis",
				"data": self._prepare_bottleneck_chart_data(report.bottlenecks),
				"options": {
					"x_axis": "Bottleneck",
					"y_axis": "Severity Score"
				}
			}
			charts.append(bottleneck_chart)
		
		return charts
	
	def _prepare_volume_chart_data(self, metrics: List[ProcessMetric]) -> List[Dict[str, Any]]:
		"""Prepare data for volume trend chart."""
		volume_metrics = [m for m in metrics if m.metric_name == "instance_count"]
		
		# Group by date
		daily_counts = defaultdict(int)
		for metric in volume_metrics:
			date_key = metric.timestamp.date().isoformat()
			daily_counts[date_key] += metric.metric_value
		
		return [
			{"date": date, "count": count}
			for date, count in sorted(daily_counts.items())
		]
	
	def _prepare_duration_chart_data(self, metrics: List[ProcessMetric]) -> List[Dict[str, Any]]:
		"""Prepare data for duration distribution chart."""
		duration_metrics = [m for m in metrics if m.metric_name == "instance_duration"]
		
		return [
			{"duration": metric.metric_value, "frequency": 1}
			for metric in duration_metrics
		]
	
	def _prepare_bottleneck_chart_data(self, bottlenecks: List[ProcessBottleneck]) -> List[Dict[str, Any]]:
		"""Prepare data for bottleneck chart."""
		return [
			{
				"bottleneck": bottleneck.element_name,
				"severity": bottleneck.severity_score,
				"type": bottleneck.bottleneck_type.value
			}
			for bottleneck in bottlenecks[:10]  # Top 10 bottlenecks
		]


# =============================================================================
# Service Factory
# =============================================================================

def create_analytics_engine() -> ProcessAnalyticsEngine:
	"""Create and configure analytics engine."""
	engine = ProcessAnalyticsEngine()
	logger.info("Process analytics engine created and configured")
	return engine


# Export main classes
__all__ = [
	'ProcessAnalyticsEngine',
	'MetricsCollector',
	'BottleneckDetector',
	'InsightsGenerator',
	'ProcessMetric',
	'ProcessKPI',
	'ProcessBottleneck',
	'ProcessInsight',
	'AnalyticsReport',
	'MetricType',
	'BottleneckType',
	'AnalysisType',
	'create_analytics_engine'
]