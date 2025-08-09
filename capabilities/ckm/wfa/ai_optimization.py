"""
APG Workflow & Business Process Management - AI-Powered Process Optimization

Advanced AI system for analyzing workflow performance and generating intelligent
optimization recommendations with machine learning insights.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import uuid
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

from models import (
	APGTenantContext, WBPMServiceResponse, WBPMPagedResponse,
	WBPMProcessInstance, WBPMTask, TaskStatus, InstanceStatus, TaskPriority
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# AI Optimization Core Classes
# =============================================================================

class OptimizationType(str, Enum):
	"""Types of process optimizations."""
	PERFORMANCE = "performance"
	EFFICIENCY = "efficiency"
	COST = "cost"
	QUALITY = "quality"
	RESOURCE_UTILIZATION = "resource_utilization"
	SLA_COMPLIANCE = "sla_compliance"
	AUTOMATION = "automation"
	COLLABORATION = "collaboration"


class RecommendationPriority(str, Enum):
	"""Priority levels for optimization recommendations."""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"


class ImpactLevel(str, Enum):
	"""Expected impact levels."""
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"


class ImplementationComplexity(str, Enum):
	"""Implementation complexity levels."""
	SIMPLE = "simple"
	MODERATE = "moderate"
	COMPLEX = "complex"
	ADVANCED = "advanced"


@dataclass
class ProcessMetrics:
	"""Comprehensive process performance metrics."""
	process_id: str = ""
	tenant_id: str = ""
	measurement_period: Dict[str, datetime] = field(default_factory=dict)
	
	# Volume metrics
	total_instances: int = 0
	completed_instances: int = 0
	failed_instances: int = 0
	cancelled_instances: int = 0
	
	# Duration metrics
	avg_duration_hours: float = 0.0
	median_duration_hours: float = 0.0
	min_duration_hours: float = 0.0
	max_duration_hours: float = 0.0
	duration_std_dev: float = 0.0
	
	# Task metrics
	total_tasks: int = 0
	avg_task_duration_minutes: float = 0.0
	avg_waiting_time_minutes: float = 0.0
	task_reassignment_rate: float = 0.0
	
	# Resource metrics
	unique_assignees: int = 0
	avg_workload_per_user: float = 0.0
	resource_utilization_rate: float = 0.0
	
	# Quality metrics
	sla_compliance_rate: float = 0.0
	error_rate: float = 0.0
	rework_rate: float = 0.0
	customer_satisfaction_score: float = 0.0
	
	# Cost metrics
	estimated_cost_per_instance: float = 0.0
	total_cost: float = 0.0
	
	# Bottleneck indicators
	bottleneck_activities: List[str] = field(default_factory=list)
	peak_usage_hours: List[int] = field(default_factory=list)


@dataclass
class OptimizationRecommendation:
	"""AI-generated optimization recommendation."""
	recommendation_id: str = field(default_factory=lambda: f"opt_{uuid.uuid4().hex}")
	process_id: str = ""
	tenant_id: str = ""
	optimization_type: OptimizationType = OptimizationType.PERFORMANCE
	title: str = ""
	description: str = ""
	rationale: str = ""
	
	# Prioritization
	priority: RecommendationPriority = RecommendationPriority.MEDIUM
	expected_impact: ImpactLevel = ImpactLevel.MEDIUM
	implementation_complexity: ImplementationComplexity = ImplementationComplexity.MODERATE
	confidence_score: float = 0.0  # 0-1 scale
	
	# Impact projections
	projected_improvements: Dict[str, float] = field(default_factory=dict)
	estimated_effort_hours: float = 0.0
	estimated_cost_savings: float = 0.0
	roi_percentage: float = 0.0
	
	# Implementation details
	implementation_steps: List[str] = field(default_factory=list)
	required_resources: List[str] = field(default_factory=list)
	prerequisites: List[str] = field(default_factory=list)
	risks: List[str] = field(default_factory=list)
	success_metrics: List[str] = field(default_factory=list)
	
	# AI insights
	supporting_data: Dict[str, Any] = field(default_factory=dict)
	similar_cases: List[str] = field(default_factory=list)
	
	# Lifecycle
	status: str = "active"  # active, implemented, rejected, obsolete
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary representation."""
		return {
			"recommendation_id": self.recommendation_id,
			"process_id": self.process_id,
			"tenant_id": self.tenant_id,
			"optimization_type": self.optimization_type.value,
			"title": self.title,
			"description": self.description,
			"rationale": self.rationale,
			"priority": self.priority.value,
			"expected_impact": self.expected_impact.value,
			"implementation_complexity": self.implementation_complexity.value,
			"confidence_score": self.confidence_score,
			"projected_improvements": self.projected_improvements,
			"estimated_effort_hours": self.estimated_effort_hours,
			"estimated_cost_savings": self.estimated_cost_savings,
			"roi_percentage": self.roi_percentage,
			"implementation_steps": self.implementation_steps,
			"required_resources": self.required_resources,
			"prerequisites": self.prerequisites,
			"risks": self.risks,
			"success_metrics": self.success_metrics,
			"supporting_data": self.supporting_data,
			"similar_cases": self.similar_cases,
			"status": self.status,
			"created_at": self.created_at.isoformat(),
			"updated_at": self.updated_at.isoformat()
		}


@dataclass
class ProcessPattern:
	"""Identified process pattern for optimization analysis."""
	pattern_id: str = field(default_factory=lambda: f"pattern_{uuid.uuid4().hex}")
	pattern_type: str = ""
	pattern_name: str = ""
	description: str = ""
	frequency: int = 0
	impact_score: float = 0.0
	optimization_potential: float = 0.0
	affected_processes: List[str] = field(default_factory=list)
	characteristics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Process Analytics Engine
# =============================================================================

class ProcessAnalyzer:
	"""Analyze process performance and identify optimization opportunities."""
	
	def __init__(self):
		self.historical_data: Dict[str, List[ProcessMetrics]] = defaultdict(list)
		self.pattern_cache: Dict[str, List[ProcessPattern]] = {}
	
	async def analyze_process_performance(
		self,
		process_id: str,
		instances: List[WBPMProcessInstance],
		tasks: List[WBPMTask],
		time_period: Dict[str, datetime],
		context: APGTenantContext
	) -> ProcessMetrics:
		"""Analyze comprehensive process performance metrics."""
		try:
			metrics = ProcessMetrics(
				process_id=process_id,
				tenant_id=context.tenant_id,
				measurement_period=time_period
			)
			
			# Analyze instances
			await self._analyze_instances(instances, metrics)
			
			# Analyze tasks
			await self._analyze_tasks(tasks, metrics)
			
			# Analyze resource utilization
			await self._analyze_resource_utilization(instances, tasks, metrics)
			
			# Identify bottlenecks
			await self._identify_bottlenecks(instances, tasks, metrics)
			
			# Store historical data
			self.historical_data[process_id].append(metrics)
			
			# Keep only recent history (last 100 measurements)
			if len(self.historical_data[process_id]) > 100:
				self.historical_data[process_id] = self.historical_data[process_id][-100:]
			
			logger.info(f"Process analysis completed for {process_id}: {metrics.total_instances} instances analyzed")
			
			return metrics
			
		except Exception as e:
			logger.error(f"Error analyzing process performance: {e}")
			raise
	
	async def _analyze_instances(self, instances: List[WBPMProcessInstance], metrics: ProcessMetrics) -> None:
		"""Analyze process instances."""
		metrics.total_instances = len(instances)
		
		# Count by status
		status_counts = Counter(instance.instance_status for instance in instances)
		metrics.completed_instances = status_counts.get(InstanceStatus.COMPLETED, 0)
		metrics.failed_instances = status_counts.get(InstanceStatus.FAILED, 0)
		metrics.cancelled_instances = status_counts.get(InstanceStatus.CANCELLED, 0)
		
		# Analyze durations for completed instances
		completed_instances = [
			instance for instance in instances
			if instance.instance_status == InstanceStatus.COMPLETED and instance.start_time and instance.end_time
		]
		
		if completed_instances:
			durations = [
				(instance.end_time - instance.start_time).total_seconds() / 3600
				for instance in completed_instances
			]
			
			metrics.avg_duration_hours = statistics.mean(durations)
			metrics.median_duration_hours = statistics.median(durations)
			metrics.min_duration_hours = min(durations)
			metrics.max_duration_hours = max(durations)
			metrics.duration_std_dev = statistics.stdev(durations) if len(durations) > 1 else 0.0
		
		# Calculate SLA compliance
		sla_compliant = 0
		total_with_sla = 0
		
		for instance in completed_instances:
			if instance.due_date:
				total_with_sla += 1
				if instance.end_time <= instance.due_date:
					sla_compliant += 1
		
		metrics.sla_compliance_rate = (sla_compliant / total_with_sla * 100) if total_with_sla > 0 else 0.0
		
		# Calculate error rate
		metrics.error_rate = (metrics.failed_instances / metrics.total_instances * 100) if metrics.total_instances > 0 else 0.0
	
	async def _analyze_tasks(self, tasks: List[WBPMTask], metrics: ProcessMetrics) -> None:
		"""Analyze process tasks."""
		metrics.total_tasks = len(tasks)
		
		# Analyze task durations
		completed_tasks = [
			task for task in tasks
			if task.task_status == TaskStatus.COMPLETED and task.claim_time and task.completion_time
		]
		
		if completed_tasks:
			task_durations = [
				(task.completion_time - task.claim_time).total_seconds() / 60
				for task in completed_tasks
			]
			metrics.avg_task_duration_minutes = statistics.mean(task_durations)
		
		# Analyze waiting times
		claimed_tasks = [
			task for task in tasks
			if task.claim_time and task.create_time
		]
		
		if claimed_tasks:
			waiting_times = [
				(task.claim_time - task.create_time).total_seconds() / 60
				for task in claimed_tasks
			]
			metrics.avg_waiting_time_minutes = statistics.mean(waiting_times)
		
		# Calculate reassignment rate
		reassigned_tasks = sum(1 for task in tasks if len(task.candidate_users) > 1)
		metrics.task_reassignment_rate = (reassigned_tasks / metrics.total_tasks * 100) if metrics.total_tasks > 0 else 0.0
	
	async def _analyze_resource_utilization(
		self,
		instances: List[WBPMProcessInstance],
		tasks: List[WBPMTask],
		metrics: ProcessMetrics
	) -> None:
		"""Analyze resource utilization."""
		# Count unique assignees
		assignees = set()
		for task in tasks:
			if task.assignee:
				assignees.add(task.assignee)
		
		metrics.unique_assignees = len(assignees)
		
		# Calculate average workload per user
		if assignees:
			workload_per_user = defaultdict(int)
			for task in tasks:
				if task.assignee:
					workload_per_user[task.assignee] += 1
			
			metrics.avg_workload_per_user = statistics.mean(workload_per_user.values())
		
		# Estimate resource utilization (simplified)
		total_task_hours = sum(
			(task.completion_time - task.claim_time).total_seconds() / 3600
			for task in tasks
			if task.task_status == TaskStatus.COMPLETED and task.claim_time and task.completion_time
		)
		
		if assignees and metrics.measurement_period:
			period_hours = (metrics.measurement_period["end"] - metrics.measurement_period["start"]).total_seconds() / 3600
			potential_hours = len(assignees) * period_hours * 0.8  # Assume 80% availability
			metrics.resource_utilization_rate = (total_task_hours / potential_hours * 100) if potential_hours > 0 else 0.0
	
	async def _identify_bottlenecks(
		self,
		instances: List[WBPMProcessInstance],
		tasks: List[WBPMTask],
		metrics: ProcessMetrics
	) -> None:
		"""Identify process bottlenecks."""
		# Analyze task durations by activity
		activity_durations = defaultdict(list)
		
		for task in tasks:
			if task.task_status == TaskStatus.COMPLETED and task.claim_time and task.completion_time:
				duration = (task.completion_time - task.claim_time).total_seconds() / 60
				activity_durations[task.task_name].append(duration)
		
		# Find activities with high average duration or high variance
		bottlenecks = []
		
		for activity, durations in activity_durations.items():
			if len(durations) >= 3:  # Need minimum samples
				avg_duration = statistics.mean(durations)
				std_dev = statistics.stdev(durations)
				
				# Consider bottleneck if duration is > 2 standard deviations above mean
				# or if variance is very high
				if avg_duration > 60 or std_dev > avg_duration * 0.5:  # 60 min threshold or high variance
					bottlenecks.append(activity)
		
		metrics.bottleneck_activities = bottlenecks
		
		# Analyze peak usage hours
		task_creation_hours = [
			task.create_time.hour for task in tasks
			if task.create_time
		]
		
		if task_creation_hours:
			hour_counts = Counter(task_creation_hours)
			# Get top 3 peak hours
			metrics.peak_usage_hours = [hour for hour, count in hour_counts.most_common(3)]
	
	async def identify_process_patterns(
		self,
		process_id: str,
		context: APGTenantContext
	) -> List[ProcessPattern]:
		"""Identify patterns in process execution."""
		patterns = []
		
		try:
			# Get historical metrics
			historical_metrics = self.historical_data.get(process_id, [])
			
			if len(historical_metrics) < 3:
				return patterns  # Need sufficient history
			
			# Pattern 1: Recurring bottlenecks
			bottleneck_pattern = await self._analyze_bottleneck_patterns(historical_metrics)
			if bottleneck_pattern:
				patterns.append(bottleneck_pattern)
			
			# Pattern 2: Performance degradation trends
			degradation_pattern = await self._analyze_degradation_patterns(historical_metrics)
			if degradation_pattern:
				patterns.append(degradation_pattern)
			
			# Pattern 3: Resource utilization patterns
			utilization_pattern = await self._analyze_utilization_patterns(historical_metrics)
			if utilization_pattern:
				patterns.append(utilization_pattern)
			
			# Pattern 4: Seasonal patterns
			seasonal_pattern = await self._analyze_seasonal_patterns(historical_metrics)
			if seasonal_pattern:
				patterns.append(seasonal_pattern)
			
			# Cache patterns
			self.pattern_cache[process_id] = patterns
			
			return patterns
			
		except Exception as e:
			logger.error(f"Error identifying process patterns: {e}")
			return []
	
	async def _analyze_bottleneck_patterns(self, metrics_history: List[ProcessMetrics]) -> Optional[ProcessPattern]:
		"""Analyze recurring bottleneck patterns."""
		# Count bottleneck frequencies
		bottleneck_counts = Counter()
		
		for metrics in metrics_history:
			for bottleneck in metrics.bottleneck_activities:
				bottleneck_counts[bottleneck] += 1
		
		# Find persistent bottlenecks (appearing in >50% of measurements)
		threshold = len(metrics_history) * 0.5
		persistent_bottlenecks = [
			activity for activity, count in bottleneck_counts.items()
			if count >= threshold
		]
		
		if persistent_bottlenecks:
			return ProcessPattern(
				pattern_type="recurring_bottleneck",
				pattern_name="Recurring Bottlenecks",
				description=f"Activities consistently showing bottleneck behavior: {', '.join(persistent_bottlenecks)}",
				frequency=len(persistent_bottlenecks),
				impact_score=0.8,
				optimization_potential=0.9,
				characteristics={
					"bottleneck_activities": persistent_bottlenecks,
					"occurrence_rate": max(bottleneck_counts.values()) / len(metrics_history)
				}
			)
		
		return None
	
	async def _analyze_degradation_patterns(self, metrics_history: List[ProcessMetrics]) -> Optional[ProcessPattern]:
		"""Analyze performance degradation trends."""
		if len(metrics_history) < 5:
			return None
		
		# Analyze duration trend
		recent_metrics = metrics_history[-5:]
		older_metrics = metrics_history[-10:-5] if len(metrics_history) >= 10 else metrics_history[:-5]
		
		if not older_metrics:
			return None
		
		recent_avg_duration = statistics.mean(m.avg_duration_hours for m in recent_metrics)
		older_avg_duration = statistics.mean(m.avg_duration_hours for m in older_metrics)
		
		# Check for degradation (>20% increase)
		if recent_avg_duration > older_avg_duration * 1.2:
			degradation_percentage = ((recent_avg_duration - older_avg_duration) / older_avg_duration) * 100
			
			return ProcessPattern(
				pattern_type="performance_degradation",
				pattern_name="Performance Degradation",
				description=f"Process duration has increased by {degradation_percentage:.1f}% in recent measurements",
				frequency=len(recent_metrics),
				impact_score=min(1.0, degradation_percentage / 50),  # Scale 0-1
				optimization_potential=0.8,
				characteristics={
					"degradation_percentage": degradation_percentage,
					"recent_avg_duration": recent_avg_duration,
					"previous_avg_duration": older_avg_duration
				}
			)
		
		return None
	
	async def _analyze_utilization_patterns(self, metrics_history: List[ProcessMetrics]) -> Optional[ProcessPattern]:
		"""Analyze resource utilization patterns."""
		utilization_rates = [m.resource_utilization_rate for m in metrics_history if m.resource_utilization_rate > 0]
		
		if not utilization_rates:
			return None
		
		avg_utilization = statistics.mean(utilization_rates)
		
		# Check for low utilization (<60%) or high utilization (>90%)
		if avg_utilization < 60:
			return ProcessPattern(
				pattern_type="low_utilization",
				pattern_name="Low Resource Utilization",
				description=f"Average resource utilization is {avg_utilization:.1f}%, indicating potential over-staffing",
				frequency=len(utilization_rates),
				impact_score=0.6,
				optimization_potential=0.7,
				characteristics={
					"avg_utilization_rate": avg_utilization,
					"underutilization_percentage": 60 - avg_utilization
				}
			)
		elif avg_utilization > 90:
			return ProcessPattern(
				pattern_type="high_utilization",
				pattern_name="High Resource Utilization",
				description=f"Average resource utilization is {avg_utilization:.1f}%, indicating potential capacity constraints",
				frequency=len(utilization_rates),
				impact_score=0.8,
				optimization_potential=0.8,
				characteristics={
					"avg_utilization_rate": avg_utilization,
					"overutilization_percentage": avg_utilization - 90
				}
			)
		
		return None
	
	async def _analyze_seasonal_patterns(self, metrics_history: List[ProcessMetrics]) -> Optional[ProcessPattern]:
		"""Analyze seasonal or cyclical patterns."""
		# Simple implementation - check for volume patterns
		volumes = [m.total_instances for m in metrics_history]
		
		if len(volumes) < 7:  # Need at least a week of data
			return None
		
		# Check for cyclical patterns (simplified)
		avg_volume = statistics.mean(volumes)
		high_volume_periods = sum(1 for v in volumes if v > avg_volume * 1.5)
		
		if high_volume_periods >= 2:
			return ProcessPattern(
				pattern_type="volume_spikes",
				pattern_name="Volume Spikes",
				description=f"Process experiences periodic volume spikes ({high_volume_periods} occurrences)",
				frequency=high_volume_periods,
				impact_score=0.6,
				optimization_potential=0.7,
				characteristics={
					"spike_frequency": high_volume_periods,
					"avg_volume": avg_volume,
					"max_volume": max(volumes)
				}
			)
		
		return None


# =============================================================================
# AI Recommendation Engine
# =============================================================================

class AIRecommendationEngine:
	"""Generate intelligent optimization recommendations using AI analysis."""
	
	def __init__(self):
		self.recommendation_templates = self._initialize_recommendation_templates()
		self.ml_models: Dict[str, Any] = {}
		self.recommendation_history: Dict[str, List[OptimizationRecommendation]] = defaultdict(list)
	
	async def generate_recommendations(
		self,
		process_id: str,
		metrics: ProcessMetrics,
		patterns: List[ProcessPattern],
		context: APGTenantContext
	) -> List[OptimizationRecommendation]:
		"""Generate comprehensive optimization recommendations."""
		recommendations = []
		
		try:
			# Performance optimization recommendations
			perf_recs = await self._generate_performance_recommendations(process_id, metrics, patterns, context)
			recommendations.extend(perf_recs)
			
			# Efficiency optimization recommendations  
			eff_recs = await self._generate_efficiency_recommendations(process_id, metrics, patterns, context)
			recommendations.extend(eff_recs)
			
			# Resource optimization recommendations
			resource_recs = await self._generate_resource_recommendations(process_id, metrics, patterns, context)
			recommendations.extend(resource_recs)
			
			# Quality optimization recommendations
			quality_recs = await self._generate_quality_recommendations(process_id, metrics, patterns, context)
			recommendations.extend(quality_recs)
			
			# Automation recommendations
			auto_recs = await self._generate_automation_recommendations(process_id, metrics, patterns, context)
			recommendations.extend(auto_recs)
			
			# Score and rank recommendations
			for rec in recommendations:
				rec.confidence_score = await self._calculate_confidence_score(rec, metrics, patterns)
				rec.roi_percentage = await self._estimate_roi(rec, metrics)
			
			# Sort by priority and confidence
			recommendations.sort(
				key=lambda r: (
					self._priority_weight(r.priority),
					r.confidence_score,
					self._impact_weight(r.expected_impact)
				),
				reverse=True
			)
			
			# Store recommendations
			self.recommendation_history[process_id].extend(recommendations)
			
			logger.info(f"Generated {len(recommendations)} optimization recommendations for process {process_id}")
			
			return recommendations[:10]  # Return top 10 recommendations
			
		except Exception as e:
			logger.error(f"Error generating recommendations: {e}")
			return []
	
	async def _generate_performance_recommendations(
		self,
		process_id: str,
		metrics: ProcessMetrics,
		patterns: List[ProcessPattern],
		context: APGTenantContext
	) -> List[OptimizationRecommendation]:
		"""Generate performance optimization recommendations."""
		recommendations = []
		
		# Recommendation 1: Address bottleneck activities
		if metrics.bottleneck_activities:
			rec = OptimizationRecommendation(
				process_id=process_id,
				tenant_id=context.tenant_id,
				optimization_type=OptimizationType.PERFORMANCE,
				title="Optimize Bottleneck Activities",
				description=f"Address performance bottlenecks in activities: {', '.join(metrics.bottleneck_activities)}",
				rationale="These activities consistently show longer execution times and high variance, causing process delays",
				priority=RecommendationPriority.HIGH,
				expected_impact=ImpactLevel.HIGH,
				implementation_complexity=ImplementationComplexity.MODERATE,
				projected_improvements={
					"duration_reduction_percentage": 25.0,
					"throughput_increase_percentage": 15.0
				},
				estimated_effort_hours=40.0,
				estimated_cost_savings=5000.0,
				implementation_steps=[
					"Analyze root causes of delays in bottleneck activities",
					"Implement process improvements or automation",
					"Redistribute workload or add resources",
					"Monitor performance improvements"
				],
				required_resources=["Process analyst", "Business stakeholders", "IT support"],
				success_metrics=["Reduced average activity duration", "Decreased process cycle time"],
				supporting_data={
					"bottleneck_activities": metrics.bottleneck_activities,
					"avg_duration_hours": metrics.avg_duration_hours
				}
			)
			recommendations.append(rec)
		
		# Recommendation 2: Address high duration variance
		if metrics.duration_std_dev > metrics.avg_duration_hours * 0.5:  # High variance
			rec = OptimizationRecommendation(
				process_id=process_id,
				tenant_id=context.tenant_id,
				optimization_type=OptimizationType.PERFORMANCE,
				title="Standardize Process Execution",
				description="High variance in process duration indicates inconsistent execution patterns",
				rationale=f"Standard deviation ({metrics.duration_std_dev:.1f}h) is {(metrics.duration_std_dev/metrics.avg_duration_hours*100):.1f}% of average duration",
				priority=RecommendationPriority.MEDIUM,
				expected_impact=ImpactLevel.MEDIUM,
				implementation_complexity=ImplementationComplexity.MODERATE,
				projected_improvements={
					"consistency_improvement_percentage": 30.0,
					"predictability_increase": 40.0
				},
				estimated_effort_hours=30.0,
				estimated_cost_savings=3000.0,
				implementation_steps=[
					"Document standard operating procedures",
					"Provide training to process participants",
					"Implement process governance and monitoring",
					"Regular process audits and improvements"
				],
				required_resources=["Process owner", "Training coordinator", "Quality assurance"],
				success_metrics=["Reduced duration variance", "Improved process predictability"],
				supporting_data={
					"duration_variance": metrics.duration_std_dev,
					"variance_percentage": (metrics.duration_std_dev / metrics.avg_duration_hours * 100) if metrics.avg_duration_hours > 0 else 0
				}
			)
			recommendations.append(rec)
		
		return recommendations
	
	async def _generate_efficiency_recommendations(
		self,
		process_id: str,
		metrics: ProcessMetrics,
		patterns: List[ProcessPattern],
		context: APGTenantContext
	) -> List[OptimizationRecommendation]:
		"""Generate efficiency optimization recommendations."""
		recommendations = []
		
		# Recommendation 1: Reduce task waiting time
		if metrics.avg_waiting_time_minutes > 60:  # More than 1 hour average wait
			rec = OptimizationRecommendation(
				process_id=process_id,
				tenant_id=context.tenant_id,
				optimization_type=OptimizationType.EFFICIENCY,
				title="Reduce Task Waiting Times",
				description=f"Tasks wait an average of {metrics.avg_waiting_time_minutes:.1f} minutes before being claimed",
				rationale="Long waiting times indicate inefficient task assignment or resource availability issues",
				priority=RecommendationPriority.HIGH,
				expected_impact=ImpactLevel.HIGH,
				implementation_complexity=ImplementationComplexity.SIMPLE,
				projected_improvements={
					"waiting_time_reduction_percentage": 50.0,
					"process_speed_increase_percentage": 20.0
				},
				estimated_effort_hours=20.0,
				estimated_cost_savings=2000.0,
				implementation_steps=[
					"Implement intelligent task routing",
					"Optimize resource allocation",
					"Add automated notifications for pending tasks",
					"Balance workload across team members"
				],
				required_resources=["Workflow administrator", "Team leads"],
				success_metrics=["Reduced average waiting time", "Faster task pickup"],
				supporting_data={
					"avg_waiting_time_minutes": metrics.avg_waiting_time_minutes
				}
			)
			recommendations.append(rec)
		
		# Recommendation 2: Optimize resource utilization
		if metrics.resource_utilization_rate < 60:  # Low utilization
			rec = OptimizationRecommendation(
				process_id=process_id,
				tenant_id=context.tenant_id,
				optimization_type=OptimizationType.RESOURCE_UTILIZATION,
				title="Optimize Resource Allocation",
				description=f"Resource utilization is {metrics.resource_utilization_rate:.1f}%, indicating potential over-staffing",
				rationale="Low resource utilization suggests inefficient allocation or capacity planning",
				priority=RecommendationPriority.MEDIUM,
				expected_impact=ImpactLevel.MEDIUM,
				implementation_complexity=ImplementationComplexity.COMPLEX,
				projected_improvements={
					"utilization_increase_percentage": 25.0,
					"cost_reduction_percentage": 15.0
				},
				estimated_effort_hours=50.0,
				estimated_cost_savings=8000.0,
				implementation_steps=[
					"Analyze current workload distribution",
					"Reallocate resources to high-demand areas",
					"Consider cross-training team members",
					"Implement dynamic resource assignment"
				],
				required_resources=["HR manager", "Process owner", "Finance team"],
				success_metrics=["Increased utilization rate", "Optimized staffing levels"],
				supporting_data={
					"current_utilization_rate": metrics.resource_utilization_rate,
					"optimization_potential": 60 - metrics.resource_utilization_rate
				}
			)
			recommendations.append(rec)
		
		return recommendations
	
	async def _generate_resource_recommendations(
		self,
		process_id: str,
		metrics: ProcessMetrics,
		patterns: List[ProcessPattern],
		context: APGTenantContext
	) -> List[OptimizationRecommendation]:
		"""Generate resource optimization recommendations."""
		recommendations = []
		
		# Recommendation 1: Address high task reassignment rate
		if metrics.task_reassignment_rate > 20:  # High reassignment rate
			rec = OptimizationRecommendation(
				process_id=process_id,
				tenant_id=context.tenant_id,
				optimization_type=OptimizationType.RESOURCE_UTILIZATION,
				title="Improve Task Assignment Accuracy",
				description=f"Task reassignment rate is {metrics.task_reassignment_rate:.1f}%, indicating suboptimal initial assignments",
				rationale="High reassignment rates cause delays and inefficiencies in task execution",
				priority=RecommendationPriority.MEDIUM,
				expected_impact=ImpactLevel.MEDIUM,
				implementation_complexity=ImplementationComplexity.MODERATE,
				projected_improvements={
					"reassignment_reduction_percentage": 40.0,
					"efficiency_increase_percentage": 15.0
				},
				estimated_effort_hours=25.0,
				estimated_cost_savings=1500.0,
				implementation_steps=[
					"Implement skill-based task routing",
					"Improve workload balancing algorithms",
					"Provide better task assignment training",
					"Add task complexity assessment"
				],
				required_resources=["Workflow administrator", "Team leads", "Training coordinator"],
				success_metrics=["Reduced reassignment rate", "Improved first-time assignment accuracy"],
				supporting_data={
					"current_reassignment_rate": metrics.task_reassignment_rate
				}
			)
			recommendations.append(rec)
		
		return recommendations
	
	async def _generate_quality_recommendations(
		self,
		process_id: str,
		metrics: ProcessMetrics,
		patterns: List[ProcessPattern],
		context: APGTenantContext
	) -> List[OptimizationRecommendation]:
		"""Generate quality optimization recommendations."""
		recommendations = []
		
		# Recommendation 1: Improve SLA compliance
		if metrics.sla_compliance_rate < 90:  # Below 90% compliance
			rec = OptimizationRecommendation(
				process_id=process_id,
				tenant_id=context.tenant_id,
				optimization_type=OptimizationType.SLA_COMPLIANCE,
				title="Improve SLA Compliance",
				description=f"SLA compliance rate is {metrics.sla_compliance_rate:.1f}%, below optimal levels",
				rationale="Poor SLA compliance affects customer satisfaction and business reputation",
				priority=RecommendationPriority.HIGH,
				expected_impact=ImpactLevel.HIGH,
				implementation_complexity=ImplementationComplexity.MODERATE,
				projected_improvements={
					"sla_compliance_increase_percentage": 15.0,
					"customer_satisfaction_improvement": 20.0
				},
				estimated_effort_hours=35.0,
				estimated_cost_savings=4000.0,
				implementation_steps=[
					"Analyze root causes of SLA violations",
					"Implement proactive monitoring and alerts",
					"Optimize process flow and resource allocation",
					"Establish escalation procedures for at-risk instances"
				],
				required_resources=["Process owner", "Quality assurance", "Customer service"],
				success_metrics=["Increased SLA compliance rate", "Reduced SLA violations"],
				supporting_data={
					"current_sla_compliance": metrics.sla_compliance_rate,
					"improvement_target": 95.0
				}
			)
			recommendations.append(rec)
		
		# Recommendation 2: Reduce error rate
		if metrics.error_rate > 5:  # Above 5% error rate
			rec = OptimizationRecommendation(
				process_id=process_id,
				tenant_id=context.tenant_id,
				optimization_type=OptimizationType.QUALITY,
				title="Reduce Process Error Rate",
				description=f"Process error rate is {metrics.error_rate:.1f}%, indicating quality issues",
				rationale="High error rates increase costs and reduce customer satisfaction",
				priority=RecommendationPriority.HIGH,
				expected_impact=ImpactLevel.HIGH,
				implementation_complexity=ImplementationComplexity.MODERATE,
				projected_improvements={
					"error_rate_reduction_percentage": 50.0,
					"quality_improvement_percentage": 30.0
				},
				estimated_effort_hours=40.0,
				estimated_cost_savings=6000.0,
				implementation_steps=[
					"Conduct root cause analysis of failures",
					"Implement quality checkpoints and validations",
					"Provide additional training to process participants",
					"Add automated error detection and prevention"
				],
				required_resources=["Quality assurance", "Process analyst", "Training coordinator"],
				success_metrics=["Reduced error rate", "Improved process quality"],
				supporting_data={
					"current_error_rate": metrics.error_rate,
					"failed_instances": metrics.failed_instances,
					"total_instances": metrics.total_instances
				}
			)
			recommendations.append(rec)
		
		return recommendations
	
	async def _generate_automation_recommendations(
		self,
		process_id: str,
		metrics: ProcessMetrics,
		patterns: List[ProcessPattern],
		context: APGTenantContext
	) -> List[OptimizationRecommendation]:
		"""Generate automation recommendations."""
		recommendations = []
		
		# Recommendation 1: Automate repetitive tasks
		if metrics.total_tasks > 100 and metrics.avg_task_duration_minutes < 30:  # Many short tasks
			rec = OptimizationRecommendation(
				process_id=process_id,
				tenant_id=context.tenant_id,
				optimization_type=OptimizationType.AUTOMATION,
				title="Automate Repetitive Tasks",
				description="Process contains many short-duration tasks that may be suitable for automation",
				rationale=f"With {metrics.total_tasks} tasks averaging {metrics.avg_task_duration_minutes:.1f} minutes, automation could provide significant efficiency gains",
				priority=RecommendationPriority.MEDIUM,
				expected_impact=ImpactLevel.HIGH,
				implementation_complexity=ImplementationComplexity.COMPLEX,
				projected_improvements={
					"automation_percentage": 40.0,
					"processing_time_reduction_percentage": 60.0,
					"cost_reduction_percentage": 30.0
				},
				estimated_effort_hours=80.0,
				estimated_cost_savings=15000.0,
				implementation_steps=[
					"Identify automation candidates through task analysis",
					"Design automation workflows and business rules",
					"Implement robotic process automation (RPA)",
					"Test and deploy automation solutions",
					"Monitor automation performance and optimize"
				],
				required_resources=["RPA developer", "Business analyst", "IT support", "Process owner"],
				prerequisites=["Stable process definition", "Clear business rules", "Technology infrastructure"],
				success_metrics=["Percentage of tasks automated", "Reduced manual effort", "Faster processing times"],
				supporting_data={
					"total_tasks": metrics.total_tasks,
					"avg_task_duration": metrics.avg_task_duration_minutes,
					"automation_potential": "high"
				}
			)
			recommendations.append(rec)
		
		return recommendations
	
	async def _calculate_confidence_score(
		self,
		recommendation: OptimizationRecommendation,
		metrics: ProcessMetrics,
		patterns: List[ProcessPattern]
	) -> float:
		"""Calculate confidence score for recommendation."""
		score = 0.5  # Base confidence
		
		# Increase confidence based on supporting data quality
		if recommendation.supporting_data:
			score += 0.2
		
		# Increase confidence based on clear problem indicators
		if recommendation.optimization_type == OptimizationType.PERFORMANCE:
			if metrics.avg_duration_hours > 8:  # Long duration
				score += 0.2
			if metrics.bottleneck_activities:  # Clear bottlenecks
				score += 0.2
		
		elif recommendation.optimization_type == OptimizationType.SLA_COMPLIANCE:
			if metrics.sla_compliance_rate < 90:  # Poor compliance
				score += 0.3
		
		elif recommendation.optimization_type == OptimizationType.QUALITY:
			if metrics.error_rate > 5:  # High error rate
				score += 0.3
		
		# Adjust based on supporting patterns
		relevant_patterns = [
			p for p in patterns
			if p.optimization_potential > 0.7
		]
		if relevant_patterns:
			score += 0.1
		
		return min(1.0, score)
	
	async def _estimate_roi(
		self,
		recommendation: OptimizationRecommendation,
		metrics: ProcessMetrics
	) -> float:
		"""Estimate return on investment for recommendation."""
		if recommendation.estimated_cost_savings <= 0 or recommendation.estimated_effort_hours <= 0:
			return 0.0
		
		# Assume hourly cost of $100 for implementation
		implementation_cost = recommendation.estimated_effort_hours * 100
		
		# Calculate ROI over 1 year
		annual_savings = recommendation.estimated_cost_savings * 12  # Monthly savings
		
		roi = ((annual_savings - implementation_cost) / implementation_cost) * 100
		
		return max(0.0, roi)
	
	def _priority_weight(self, priority: RecommendationPriority) -> int:
		"""Get numeric weight for priority."""
		weights = {
			RecommendationPriority.CRITICAL: 4,
			RecommendationPriority.HIGH: 3,
			RecommendationPriority.MEDIUM: 2,
			RecommendationPriority.LOW: 1
		}
		return weights.get(priority, 1)
	
	def _impact_weight(self, impact: ImpactLevel) -> int:
		"""Get numeric weight for impact."""
		weights = {
			ImpactLevel.HIGH: 3,
			ImpactLevel.MEDIUM: 2,
			ImpactLevel.LOW: 1
		}
		return weights.get(impact, 1)
	
	def _initialize_recommendation_templates(self) -> Dict[str, Dict[str, Any]]:
		"""Initialize recommendation templates."""
		return {
			"bottleneck_optimization": {
				"title": "Optimize Process Bottlenecks",
				"description": "Address activities causing process delays",
				"implementation_complexity": ImplementationComplexity.MODERATE,
				"expected_impact": ImpactLevel.HIGH
			},
			"automation": {
				"title": "Implement Process Automation",
				"description": "Automate repetitive manual tasks",
				"implementation_complexity": ImplementationComplexity.COMPLEX,
				"expected_impact": ImpactLevel.HIGH
			},
			"resource_optimization": {
				"title": "Optimize Resource Allocation",
				"description": "Improve resource utilization and distribution",
				"implementation_complexity": ImplementationComplexity.MODERATE,
				"expected_impact": ImpactLevel.MEDIUM
			}
		}


# =============================================================================
# AI Optimization Engine
# =============================================================================

class AIOptimizationEngine:
	"""Main AI-powered process optimization engine."""
	
	def __init__(self):
		self.process_analyzer = ProcessAnalyzer()
		self.recommendation_engine = AIRecommendationEngine()
		self.optimization_cache: Dict[str, Dict[str, Any]] = {}
	
	async def analyze_and_optimize(
		self,
		process_id: str,
		instances: List[WBPMProcessInstance],
		tasks: List[WBPMTask],
		time_period: Dict[str, datetime],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Perform comprehensive process analysis and generate optimization recommendations."""
		try:
			# Analyze process performance
			metrics = await self.process_analyzer.analyze_process_performance(
				process_id, instances, tasks, time_period, context
			)
			
			# Identify patterns
			patterns = await self.process_analyzer.identify_process_patterns(process_id, context)
			
			# Generate recommendations
			recommendations = await self.recommendation_engine.generate_recommendations(
				process_id, metrics, patterns, context
			)
			
			# Prepare optimization report
			optimization_report = {
				"analysis_timestamp": datetime.utcnow().isoformat(),
				"process_id": process_id,
				"analysis_period": time_period,
				"metrics": {
					"total_instances": metrics.total_instances,
					"completion_rate": (metrics.completed_instances / metrics.total_instances * 100) if metrics.total_instances > 0 else 0,
					"avg_duration_hours": metrics.avg_duration_hours,
					"sla_compliance_rate": metrics.sla_compliance_rate,
					"error_rate": metrics.error_rate,
					"resource_utilization_rate": metrics.resource_utilization_rate,
					"bottleneck_count": len(metrics.bottleneck_activities)
				},
				"patterns": [
					{
						"pattern_type": pattern.pattern_type,
						"pattern_name": pattern.pattern_name,
						"description": pattern.description,
						"optimization_potential": pattern.optimization_potential,
						"characteristics": pattern.characteristics
					}
					for pattern in patterns
				],
				"recommendations": [rec.to_dict() for rec in recommendations],
				"optimization_summary": {
					"total_recommendations": len(recommendations),
					"high_priority_recommendations": len([r for r in recommendations if r.priority == RecommendationPriority.HIGH]),
					"estimated_total_savings": sum(r.estimated_cost_savings for r in recommendations),
					"estimated_total_effort": sum(r.estimated_effort_hours for r in recommendations),
					"avg_confidence_score": statistics.mean([r.confidence_score for r in recommendations]) if recommendations else 0.0
				}
			}
			
			# Cache optimization report
			self.optimization_cache[process_id] = optimization_report
			
			logger.info(f"Process optimization completed for {process_id}: {len(recommendations)} recommendations generated")
			
			return WBPMServiceResponse(
				success=True,
				message="Process optimization analysis completed successfully",
				data=optimization_report
			)
			
		except Exception as e:
			logger.error(f"Error in process optimization analysis: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Process optimization analysis failed: {e}",
				errors=[str(e)]
			)
	
	async def get_optimization_summary(
		self,
		tenant_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Get optimization summary across all processes for tenant."""
		try:
			tenant_reports = [
				report for report in self.optimization_cache.values()
				if report.get("process_id") and report.get("process_id").startswith(tenant_id)
			]
			
			if not tenant_reports:
				return WBPMServiceResponse(
					success=True,
					message="No optimization data available",
					data={"summary": "No processes analyzed yet"}
				)
			
			# Aggregate statistics
			total_recommendations = sum(report["optimization_summary"]["total_recommendations"] for report in tenant_reports)
			total_estimated_savings = sum(report["optimization_summary"]["estimated_total_savings"] for report in tenant_reports)
			avg_confidence = statistics.mean([
				report["optimization_summary"]["avg_confidence_score"]
				for report in tenant_reports
				if report["optimization_summary"]["avg_confidence_score"] > 0
			]) if tenant_reports else 0.0
			
			# Top recommendation types
			all_recommendations = []
			for report in tenant_reports:
				all_recommendations.extend(report["recommendations"])
			
			recommendation_types = Counter(rec["optimization_type"] for rec in all_recommendations)
			
			summary = {
				"total_processes_analyzed": len(tenant_reports),
				"total_recommendations": total_recommendations,
				"total_estimated_savings": total_estimated_savings,
				"average_confidence_score": avg_confidence,
				"top_optimization_types": dict(recommendation_types.most_common(5)),
				"processes": [
					{
						"process_id": report["process_id"],
						"analysis_timestamp": report["analysis_timestamp"],
						"recommendations_count": report["optimization_summary"]["total_recommendations"],
						"estimated_savings": report["optimization_summary"]["estimated_total_savings"],
						"completion_rate": report["metrics"]["completion_rate"],
						"sla_compliance": report["metrics"]["sla_compliance_rate"]
					}
					for report in tenant_reports
				]
			}
			
			return WBPMServiceResponse(
				success=True,
				message="Optimization summary retrieved successfully",
				data=summary
			)
			
		except Exception as e:
			logger.error(f"Error getting optimization summary: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to get optimization summary: {e}",
				errors=[str(e)]
			)


# =============================================================================
# Service Factory
# =============================================================================

def create_ai_optimization_engine() -> AIOptimizationEngine:
	"""Create and configure AI optimization engine."""
	engine = AIOptimizationEngine()
	logger.info("AI optimization engine created and configured")
	return engine


# Export main classes
__all__ = [
	'AIOptimizationEngine',
	'ProcessAnalyzer',
	'AIRecommendationEngine',
	'ProcessMetrics',
	'OptimizationRecommendation',
	'ProcessPattern',
	'OptimizationType',
	'RecommendationPriority',
	'ImpactLevel',
	'ImplementationComplexity',
	'create_ai_optimization_engine'
]