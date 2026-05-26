"""
APG AI Core Framework (aicr) - Distributed Computing Engine with Auto-Scaling

Purpose: Revolutionary distributed AI computing orchestration providing
         auto-scaling clusters, intelligent resource allocation, fault-tolerant
         processing, and dynamic load balancing for massive AI workloads.

Dependencies: asyncio, kubernetes, docker, ray, celery, redis, monitoring
Computing Features: Auto-scaling clusters, distributed inference, fault tolerance,
                   resource optimization, load balancing, worker management
Usage Context: Massive-scale AI operations with dynamic resource requirements

This module provides:
- Intelligent auto-scaling cluster management
- Distributed AI inference and training orchestration
- Fault-tolerant distributed processing with recovery
- Dynamic resource allocation and optimization
- Advanced load balancing and traffic distribution
- Worker node management and health monitoring
- Container orchestration with Kubernetes integration
- Performance monitoring and metrics collection
"""

import asyncio
import base64
import json
import logging
import math
import random
import statistics
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Awaitable
from uuid import uuid4
import hashlib
import hmac

from pydantic import BaseModel, Field, ConfigDict
import numpy as np

from .models import uuid7str, _validate_tenant_id


def _log_cluster_event(event_type: str, cluster_id: str, operation: str, result: str, details: str = "") -> str:
	"""Log cluster management events with standardized format."""
	timestamp = datetime.now(timezone.utc).isoformat()
	return f"CLUSTER [{event_type}] {cluster_id} {operation} - {result} {details} ({timestamp})"


def _log_worker_event(worker_id: str, action: str, status: str, metrics: str = "") -> str:
	"""Log worker node events."""
	metrics_info = f" metrics={metrics}" if metrics else ""
	return f"WORKER [{worker_id}] {action} - {status}{metrics_info}"


def _log_scaling_event(cluster_id: str, scale_action: str, from_count: int, to_count: int, reason: str = "") -> str:
	"""Log auto-scaling events."""
	reason_info = f" reason={reason}" if reason else ""
	return f"SCALING [{cluster_id}] {scale_action} {from_count}->{to_count}{reason_info}"


class ClusterState(str, Enum):
	"""Distributed cluster operational states.

	Defines the operational state of distributed computing clusters
	for proper lifecycle management and state transitions.

	Attributes:
		INITIALIZING: Cluster is being created and configured
		HEALTHY: Cluster is operational and healthy
		SCALING_UP: Cluster is adding more worker nodes
		SCALING_DOWN: Cluster is removing worker nodes
		DEGRADED: Cluster has some failed nodes but is operational
		CRITICAL: Cluster has critical issues affecting operations
		MAINTENANCE: Cluster is in maintenance mode
		TERMINATED: Cluster has been shut down
	"""
	INITIALIZING = "initializing"
	HEALTHY = "healthy"
	SCALING_UP = "scaling_up"
	SCALING_DOWN = "scaling_down"
	DEGRADED = "degraded"
	CRITICAL = "critical"
	MAINTENANCE = "maintenance"
	TERMINATED = "terminated"


class WorkerNodeState(str, Enum):
	"""Worker node operational states.

	Defines the operational state of individual worker nodes
	within distributed computing clusters.

	Attributes:
		PROVISIONING: Node is being provisioned and configured
		READY: Node is ready to accept workloads
		BUSY: Node is currently processing workloads
		DRAINING: Node is finishing current work before shutdown
		FAILED: Node has failed and needs replacement
		MAINTENANCE: Node is in maintenance mode
		TERMINATING: Node is being shut down
	"""
	PROVISIONING = "provisioning"
	READY = "ready"
	BUSY = "busy"
	DRAINING = "draining"
	FAILED = "failed"
	MAINTENANCE = "maintenance"
	TERMINATING = "terminating"


class ResourceType(str, Enum):
	"""Types of computational resources.

	Categorizes different types of computational resources
	for proper allocation and optimization.

	Attributes:
		CPU: CPU cores for general computation
		GPU: GPU units for accelerated computation
		TPU: Tensor Processing Units for AI workloads
		MEMORY: System memory (RAM)
		STORAGE: Persistent storage
		NETWORK: Network bandwidth
		NEUROMORPHIC: Neuromorphic processing units
		QUANTUM: Quantum processing units
	"""
	CPU = "cpu"
	GPU = "gpu"
	TPU = "tpu"
	MEMORY = "memory"
	STORAGE = "storage"
	NETWORK = "network"
	NEUROMORPHIC = "neuromorphic"
	QUANTUM = "quantum"


class ScalingPolicy(str, Enum):
	"""Auto-scaling policies for cluster management.

	Defines different strategies for automatic cluster scaling
	based on workload patterns and resource utilization.

	Attributes:
		REACTIVE: Scale based on current resource utilization
		PREDICTIVE: Scale based on predicted future demand
		SCHEDULED: Scale based on predefined schedules
		HYBRID: Combination of reactive and predictive scaling
		CUSTOM: Custom scaling logic based on business rules
	"""
	REACTIVE = "reactive"
	PREDICTIVE = "predictive"
	SCHEDULED = "scheduled"
	HYBRID = "hybrid"
	CUSTOM = "custom"


class LoadBalancingStrategy(str, Enum):
	"""Load balancing strategies for task distribution.

	Defines different approaches for distributing workloads
	across worker nodes in the cluster.

	Attributes:
		ROUND_ROBIN: Distribute tasks in round-robin fashion
		LEAST_CONNECTIONS: Route to node with fewest active connections
		WEIGHTED_ROUND_ROBIN: Round-robin with node weights
		RESOURCE_AWARE: Route based on available resources
		LATENCY_BASED: Route to node with lowest latency
		CUSTOM: Custom load balancing algorithm
	"""
	ROUND_ROBIN = "round_robin"
	LEAST_CONNECTIONS = "least_connections"
	WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
	RESOURCE_AWARE = "resource_aware"
	LATENCY_BASED = "latency_based"
	CUSTOM = "custom"


class ResourceRequirement(BaseModel):
	"""Resource requirements for computational tasks.

	Specifies the computational resources required for
	executing AI tasks including minimum, preferred,
	and maximum resource allocations.

	Attributes:
		cpu_cores: Number of CPU cores required
		memory_gb: Memory requirement in gigabytes
		gpu_count: Number of GPU units required
		gpu_memory_gb: GPU memory requirement in gigabytes
		storage_gb: Storage requirement in gigabytes
		network_mbps: Network bandwidth requirement in Mbps
		specialized_hardware: Specialized hardware requirements
		min_resources: Minimum acceptable resource allocation
		preferred_resources: Preferred resource allocation
		max_resources: Maximum resource allocation
		resource_constraints: Additional resource constraints
		performance_requirements: Performance requirements
	"""
	cpu_cores: float = 1.0
	memory_gb: float = 2.0
	gpu_count: int = 0
	gpu_memory_gb: float = 0.0
	storage_gb: float = 10.0
	network_mbps: float = 100.0
	specialized_hardware: List[ResourceType] = Field(default_factory=list)
	min_resources: Optional[Dict[str, float]] = None
	preferred_resources: Optional[Dict[str, float]] = None
	max_resources: Optional[Dict[str, float]] = None
	resource_constraints: Dict[str, Any] = Field(default_factory=dict)
	performance_requirements: Dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	def get_total_cost_estimate(self) -> float:
		"""Calculate estimated cost for resource requirements."""
		# Simplified cost calculation (per hour)
		base_cost = (
			self.cpu_cores * 0.05 +  # $0.05 per CPU core/hour
			self.memory_gb * 0.01 +   # $0.01 per GB memory/hour
			self.gpu_count * 0.50 +   # $0.50 per GPU/hour
			self.storage_gb * 0.001   # $0.001 per GB storage/hour
		)

		# Add premium for specialized hardware
		if ResourceType.TPU in self.specialized_hardware:
			base_cost *= 2.0
		if ResourceType.NEUROMORPHIC in self.specialized_hardware:
			base_cost *= 3.0
		if ResourceType.QUANTUM in self.specialized_hardware:
			base_cost *= 5.0

		return base_cost

	def is_compatible_with(self, available_resources: Dict[str, float]) -> bool:
		"""Check if requirements can be satisfied with available resources."""
		return (
			available_resources.get("cpu_cores", 0) >= self.cpu_cores and
			available_resources.get("memory_gb", 0) >= self.memory_gb and
			available_resources.get("gpu_count", 0) >= self.gpu_count and
			available_resources.get("storage_gb", 0) >= self.storage_gb
		)


class WorkerNodeMetrics(BaseModel):
	"""Comprehensive metrics for worker node monitoring.

	Detailed performance and health metrics for individual
	worker nodes in the distributed computing cluster.

	Attributes:
		node_id: Unique worker node identifier
		timestamp: Metrics collection timestamp
		cpu_utilization: CPU utilization percentage
		memory_utilization: Memory utilization percentage
		gpu_utilization: GPU utilization percentage
		network_io_mbps: Network I/O in Mbps
		disk_io_mbps: Disk I/O in Mbps
		active_tasks: Number of active tasks
		completed_tasks: Total completed tasks
		failed_tasks: Total failed tasks
		task_queue_size: Size of task queue
		average_task_duration: Average task completion time
		error_rate: Task failure rate
		health_score: Overall node health score (0-1)
		temperature: Node temperature if available
		power_consumption: Power consumption in watts
		uptime_seconds: Node uptime in seconds
		last_heartbeat: Last heartbeat timestamp
		custom_metrics: Custom node-specific metrics
	"""
	node_id: str
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	cpu_utilization: float = 0.0
	memory_utilization: float = 0.0
	gpu_utilization: float = 0.0
	network_io_mbps: float = 0.0
	disk_io_mbps: float = 0.0
	active_tasks: int = 0
	completed_tasks: int = 0
	failed_tasks: int = 0
	task_queue_size: int = 0
	average_task_duration: float = 0.0
	error_rate: float = 0.0
	health_score: float = 1.0
	temperature: Optional[float] = None
	power_consumption: Optional[float] = None
	uptime_seconds: float = 0.0
	last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	custom_metrics: Dict[str, float] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	def calculate_efficiency_score(self) -> float:
		"""Calculate worker efficiency score based on metrics."""
		# Base efficiency from resource utilization
		resource_efficiency = (self.cpu_utilization + self.memory_utilization) / 200.0

		# Task completion efficiency
		total_tasks = self.completed_tasks + self.failed_tasks
		task_efficiency = self.completed_tasks / max(1, total_tasks)

		# Error rate penalty
		error_penalty = max(0, 1.0 - self.error_rate * 2.0)

		# Health score factor
		health_factor = self.health_score

		# Combined efficiency score
		efficiency = (
			resource_efficiency * 0.3 +
			task_efficiency * 0.4 +
			error_penalty * 0.2 +
			health_factor * 0.1
		)

		return min(1.0, max(0.0, efficiency))

	def is_overloaded(self) -> bool:
		"""Check if worker node is overloaded."""
		return (
			self.cpu_utilization > 90.0 or
			self.memory_utilization > 90.0 or
			self.task_queue_size > 100 or
			self.error_rate > 0.1
		)

	def is_healthy(self) -> bool:
		"""Check if worker node is healthy."""
		heartbeat_age = (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds()
		return (
			self.health_score > 0.7 and
			heartbeat_age < 300 and  # 5 minutes
			self.error_rate < 0.05
		)


class WorkerNode(BaseModel):
	"""Distributed computing worker node.

	Represents an individual worker node in the distributed
	computing cluster with full lifecycle management,
	health monitoring, and task execution capabilities.

	Attributes:
		node_id: Unique worker node identifier
		cluster_id: Parent cluster identifier
		node_name: Human-readable node name
		state: Current operational state
		node_type: Type of worker node
		available_resources: Available computational resources
		allocated_resources: Currently allocated resources
		creation_timestamp: Node creation time
		last_state_change: Last state change timestamp
		endpoint_url: Node communication endpoint
		container_id: Container identifier if containerized
		host_information: Host system information
		capabilities: Node capabilities and features
		current_metrics: Current performance metrics
		historical_metrics: Historical performance data
		configuration: Node configuration parameters
		tags: Node tags for organization and filtering
		maintenance_window: Scheduled maintenance windows
	"""
	node_id: str = Field(default_factory=uuid7str)
	cluster_id: str
	node_name: str
	state: WorkerNodeState = WorkerNodeState.PROVISIONING
	node_type: str = "standard"
	available_resources: Dict[str, float] = Field(default_factory=dict)
	allocated_resources: Dict[str, float] = Field(default_factory=dict)
	creation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	last_state_change: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	endpoint_url: Optional[str] = None
	container_id: Optional[str] = None
	host_information: Dict[str, Any] = Field(default_factory=dict)
	capabilities: List[str] = Field(default_factory=list)
	current_metrics: Optional[WorkerNodeMetrics] = None
	historical_metrics: List[WorkerNodeMetrics] = Field(default_factory=list)
	configuration: Dict[str, Any] = Field(default_factory=dict)
	tags: List[str] = Field(default_factory=list)
	maintenance_window: Optional[Dict[str, Any]] = None

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	def update_state(self, new_state: WorkerNodeState) -> None:
		"""Update worker node state with timestamp."""
		self.state = new_state
		self.last_state_change = datetime.now(timezone.utc)

	def update_metrics(self, metrics: WorkerNodeMetrics) -> None:
		"""Update worker node metrics."""
		self.current_metrics = metrics
		self.historical_metrics.append(metrics)

		# Keep historical metrics reasonable size
		if len(self.historical_metrics) > 1000:
			self.historical_metrics = self.historical_metrics[-500:]

	def get_available_capacity(self, resource_type: str) -> float:
		"""Get available capacity for specific resource type."""
		total = self.available_resources.get(resource_type, 0.0)
		allocated = self.allocated_resources.get(resource_type, 0.0)
		return max(0.0, total - allocated)

	def can_accommodate(self, requirements: ResourceRequirement) -> bool:
		"""Check if node can accommodate resource requirements."""
		return (
			self.get_available_capacity("cpu_cores") >= requirements.cpu_cores and
			self.get_available_capacity("memory_gb") >= requirements.memory_gb and
			self.get_available_capacity("gpu_count") >= requirements.gpu_count and
			self.get_available_capacity("storage_gb") >= requirements.storage_gb
		)

	def allocate_resources(self, requirements: ResourceRequirement) -> bool:
		"""Allocate resources for task execution."""
		if not self.can_accommodate(requirements):
			return False

		# Allocate resources
		self.allocated_resources["cpu_cores"] = self.allocated_resources.get("cpu_cores", 0) + requirements.cpu_cores
		self.allocated_resources["memory_gb"] = self.allocated_resources.get("memory_gb", 0) + requirements.memory_gb
		self.allocated_resources["gpu_count"] = self.allocated_resources.get("gpu_count", 0) + requirements.gpu_count
		self.allocated_resources["storage_gb"] = self.allocated_resources.get("storage_gb", 0) + requirements.storage_gb

		return True

	def deallocate_resources(self, requirements: ResourceRequirement) -> None:
		"""Deallocate resources after task completion."""
		self.allocated_resources["cpu_cores"] = max(0, self.allocated_resources.get("cpu_cores", 0) - requirements.cpu_cores)
		self.allocated_resources["memory_gb"] = max(0, self.allocated_resources.get("memory_gb", 0) - requirements.memory_gb)
		self.allocated_resources["gpu_count"] = max(0, self.allocated_resources.get("gpu_count", 0) - requirements.gpu_count)
		self.allocated_resources["storage_gb"] = max(0, self.allocated_resources.get("storage_gb", 0) - requirements.storage_gb)


class DistributedTask(BaseModel):
	"""Distributed computation task with execution tracking.

	Represents a computational task that can be executed
	across the distributed computing cluster with full
	lifecycle management and monitoring.

	Attributes:
		task_id: Unique task identifier
		task_type: Type of computational task
		priority: Task execution priority (0-10)
		resource_requirements: Required computational resources
		payload: Task payload and parameters
		dependencies: List of dependent task IDs
		assigned_node_id: Worker node assigned to execute task
		submission_timestamp: Task submission time
		start_timestamp: Task execution start time
		completion_timestamp: Task completion time
		state: Current task execution state
		progress: Task completion progress (0-1)
		result: Task execution result
		error_message: Error message if task failed
		retry_count: Number of retry attempts
		max_retries: Maximum allowed retries
		timeout_seconds: Task execution timeout
		metadata: Additional task metadata
		execution_context: Execution environment context
		performance_metrics: Task execution performance data
	"""
	task_id: str = Field(default_factory=uuid7str)
	task_type: str
	priority: int = 5
	resource_requirements: ResourceRequirement
	payload: Dict[str, Any] = Field(default_factory=dict)
	dependencies: List[str] = Field(default_factory=list)
	assigned_node_id: Optional[str] = None
	submission_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	start_timestamp: Optional[datetime] = None
	completion_timestamp: Optional[datetime] = None
	state: str = "pending"
	progress: float = 0.0
	result: Optional[Dict[str, Any]] = None
	error_message: Optional[str] = None
	retry_count: int = 0
	max_retries: int = 3
	timeout_seconds: int = 3600
	metadata: Dict[str, Any] = Field(default_factory=dict)
	execution_context: Dict[str, Any] = Field(default_factory=dict)
	performance_metrics: Dict[str, float] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	def start_execution(self, node_id: str) -> None:
		"""Mark task as started on specific node."""
		self.assigned_node_id = node_id
		self.start_timestamp = datetime.now(timezone.utc)
		self.state = "running"

	def complete_execution(self, result: Dict[str, Any]) -> None:
		"""Mark task as completed with result."""
		self.completion_timestamp = datetime.now(timezone.utc)
		self.state = "completed"
		self.progress = 1.0
		self.result = result

		# Calculate execution metrics
		if self.start_timestamp:
			execution_time = (self.completion_timestamp - self.start_timestamp).total_seconds()
			self.performance_metrics["execution_time_seconds"] = execution_time

	def fail_execution(self, error_message: str) -> None:
		"""Mark task as failed with error message."""
		self.completion_timestamp = datetime.now(timezone.utc)
		self.state = "failed"
		self.error_message = error_message
		self.retry_count += 1

	def can_retry(self) -> bool:
		"""Check if task can be retried."""
		return self.retry_count < self.max_retries and self.state == "failed"

	def get_execution_duration(self) -> Optional[float]:
		"""Get task execution duration in seconds."""
		if self.start_timestamp and self.completion_timestamp:
			return (self.completion_timestamp - self.start_timestamp).total_seconds()
		return None

	def is_expired(self) -> bool:
		"""Check if task has exceeded timeout."""
		if not self.start_timestamp:
			return False

		elapsed = (datetime.now(timezone.utc) - self.start_timestamp).total_seconds()
		return elapsed > self.timeout_seconds


class ClusterMetrics(BaseModel):
	"""Comprehensive metrics for distributed cluster monitoring.

	Detailed performance and health metrics for the entire
	distributed computing cluster including aggregated
	worker node statistics and cluster-level indicators.

	Attributes:
		cluster_id: Cluster identifier
		timestamp: Metrics collection timestamp
		total_nodes: Total number of worker nodes
		healthy_nodes: Number of healthy worker nodes
		busy_nodes: Number of busy worker nodes
		failed_nodes: Number of failed worker nodes
		total_cpu_cores: Total CPU cores in cluster
		available_cpu_cores: Available CPU cores
		total_memory_gb: Total memory in cluster
		available_memory_gb: Available memory
		total_gpu_count: Total GPU units in cluster
		available_gpu_count: Available GPU units
		active_tasks: Total active tasks across cluster
		pending_tasks: Total pending tasks in queue
		completed_tasks_last_hour: Tasks completed in last hour
		failed_tasks_last_hour: Tasks failed in last hour
		average_task_duration: Average task completion time
		cluster_utilization: Overall cluster utilization (0-1)
		throughput_tasks_per_second: Task completion throughput
		error_rate: Cluster-wide error rate
		health_score: Overall cluster health (0-1)
		cost_per_hour: Estimated cluster cost per hour
		efficiency_score: Cluster efficiency score (0-1)
		scaling_events_last_hour: Auto-scaling events in last hour
		custom_metrics: Custom cluster-specific metrics
	"""
	cluster_id: str
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	total_nodes: int = 0
	healthy_nodes: int = 0
	busy_nodes: int = 0
	failed_nodes: int = 0
	total_cpu_cores: float = 0.0
	available_cpu_cores: float = 0.0
	total_memory_gb: float = 0.0
	available_memory_gb: float = 0.0
	total_gpu_count: int = 0
	available_gpu_count: int = 0
	active_tasks: int = 0
	pending_tasks: int = 0
	completed_tasks_last_hour: int = 0
	failed_tasks_last_hour: int = 0
	average_task_duration: float = 0.0
	cluster_utilization: float = 0.0
	throughput_tasks_per_second: float = 0.0
	error_rate: float = 0.0
	health_score: float = 1.0
	cost_per_hour: float = 0.0
	efficiency_score: float = 0.0
	scaling_events_last_hour: int = 0
	custom_metrics: Dict[str, float] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	def calculate_resource_utilization(self) -> Dict[str, float]:
		"""Calculate utilization for each resource type."""
		return {
			"cpu": 1.0 - (self.available_cpu_cores / max(1, self.total_cpu_cores)),
			"memory": 1.0 - (self.available_memory_gb / max(1, self.total_memory_gb)),
			"gpu": 1.0 - (self.available_gpu_count / max(1, self.total_gpu_count)) if self.total_gpu_count > 0 else 0.0
		}

	def should_scale_up(self, threshold: float = 0.8) -> bool:
		"""Determine if cluster should scale up based on utilization."""
		utilization = self.calculate_resource_utilization()
		return any(util > threshold for util in utilization.values()) or self.pending_tasks > 50

	def should_scale_down(self, threshold: float = 0.3) -> bool:
		"""Determine if cluster should scale down based on utilization."""
		utilization = self.calculate_resource_utilization()
		return all(util < threshold for util in utilization.values()) and self.pending_tasks == 0


class AutoScalingConfiguration(BaseModel):
	"""Auto-scaling configuration for distributed clusters.

	Configuration parameters for intelligent auto-scaling
	behavior including scaling triggers, limits, and policies.

	Attributes:
		enabled: Whether auto-scaling is enabled
		scaling_policy: Auto-scaling policy to use
		min_nodes: Minimum number of worker nodes
		max_nodes: Maximum number of worker nodes
		target_utilization: Target resource utilization (0-1)
		scale_up_threshold: Utilization threshold for scaling up
		scale_down_threshold: Utilization threshold for scaling down
		scale_up_cooldown_seconds: Cooldown period after scaling up
		scale_down_cooldown_seconds: Cooldown period after scaling down
		scale_up_increment: Number of nodes to add when scaling up
		scale_down_increment: Number of nodes to remove when scaling down
		predictive_window_minutes: Window for predictive scaling
		custom_metrics_weight: Weight for custom metrics in scaling decisions
		node_warmup_time_seconds: Time for new nodes to become ready
		preemptible_nodes_ratio: Ratio of preemptible/spot nodes (0-1)
		cost_optimization_enabled: Whether to optimize for cost
		performance_optimization_enabled: Whether to optimize for performance
	"""
	enabled: bool = True
	scaling_policy: ScalingPolicy = ScalingPolicy.HYBRID
	min_nodes: int = 1
	max_nodes: int = 100
	target_utilization: float = 0.7
	scale_up_threshold: float = 0.8
	scale_down_threshold: float = 0.3
	scale_up_cooldown_seconds: int = 300
	scale_down_cooldown_seconds: int = 600
	scale_up_increment: int = 2
	scale_down_increment: int = 1
	predictive_window_minutes: int = 15
	custom_metrics_weight: float = 0.2
	node_warmup_time_seconds: int = 120
	preemptible_nodes_ratio: float = 0.3
	cost_optimization_enabled: bool = True
	performance_optimization_enabled: bool = True

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	def get_scale_up_decision(self, current_metrics: ClusterMetrics) -> Tuple[bool, int, str]:
		"""Determine if and how much to scale up."""
		if not self.enabled:
			return False, 0, "auto_scaling_disabled"

		utilization = current_metrics.calculate_resource_utilization()
		max_utilization = max(utilization.values())

		should_scale = (
			max_utilization > self.scale_up_threshold or
			current_metrics.pending_tasks > 20 or
			current_metrics.failed_nodes > 0
		)

		if should_scale and current_metrics.total_nodes < self.max_nodes:
			# Calculate how many nodes to add
			if max_utilization > 0.95:
				nodes_to_add = min(self.scale_up_increment * 2, self.max_nodes - current_metrics.total_nodes)
				reason = "critical_utilization"
			elif current_metrics.pending_tasks > 50:
				nodes_to_add = min(self.scale_up_increment, self.max_nodes - current_metrics.total_nodes)
				reason = "high_pending_tasks"
			else:
				nodes_to_add = min(1, self.max_nodes - current_metrics.total_nodes)
				reason = "utilization_threshold"

			return True, nodes_to_add, reason

		return False, 0, "no_scaling_needed"

	def get_scale_down_decision(self, current_metrics: ClusterMetrics) -> Tuple[bool, int, str]:
		"""Determine if and how much to scale down."""
		if not self.enabled:
			return False, 0, "auto_scaling_disabled"

		utilization = current_metrics.calculate_resource_utilization()
		max_utilization = max(utilization.values())

		should_scale = (
			max_utilization < self.scale_down_threshold and
			current_metrics.pending_tasks == 0 and
			current_metrics.total_nodes > self.min_nodes
		)

		if should_scale:
			# Calculate how many nodes to remove
			excess_capacity = self.target_utilization - max_utilization
			if excess_capacity > 0.4:  # Significant excess capacity
				nodes_to_remove = min(self.scale_down_increment * 2, current_metrics.total_nodes - self.min_nodes)
				reason = "significant_excess_capacity"
			else:
				nodes_to_remove = min(self.scale_down_increment, current_metrics.total_nodes - self.min_nodes)
				reason = "low_utilization"

			return True, nodes_to_remove, reason

		return False, 0, "no_scaling_needed"


class LoadBalancer:
	"""Intelligent load balancer for distributed task distribution.

	Implements advanced load balancing algorithms for optimal
	task distribution across worker nodes considering resource
	availability, performance characteristics, and workload patterns.

	Attributes:
		_strategy: Load balancing strategy
		_worker_weights: Weights for each worker node
		_performance_history: Historical performance data
		_current_assignments: Current task assignments
	"""

	def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE):
		"""Initialize load balancer.

		Args:
			strategy: Load balancing strategy to use
		"""
		self._strategy = strategy
		self._worker_weights: Dict[str, float] = {}
		self._performance_history: Dict[str, List[float]] = {}
		self._current_assignments: Dict[str, int] = {}
		self._round_robin_index = 0

		# Initialize logging
		self._logger = logging.getLogger(__name__)

	def select_worker_node(self, task: DistributedTask,
						   available_nodes: List[WorkerNode]) -> Optional[str]:
		"""Select optimal worker node for task execution.

		Args:
			task: Task to be executed
			available_nodes: List of available worker nodes

		Returns:
			Optional[str]: Selected worker node ID or None
		"""
		if not available_nodes:
			return None

		# Filter nodes that can accommodate the task
		compatible_nodes = [
			node for node in available_nodes
			if node.can_accommodate(task.resource_requirements) and
			   node.state == WorkerNodeState.READY
		]

		if not compatible_nodes:
			return None

		# Select node based on strategy
		if self._strategy == LoadBalancingStrategy.ROUND_ROBIN:
			return self._round_robin_selection(compatible_nodes)
		elif self._strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
			return self._least_connections_selection(compatible_nodes)
		elif self._strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
			return self._weighted_round_robin_selection(compatible_nodes)
		elif self._strategy == LoadBalancingStrategy.RESOURCE_AWARE:
			return self._resource_aware_selection(compatible_nodes, task)
		elif self._strategy == LoadBalancingStrategy.LATENCY_BASED:
			return self._latency_based_selection(compatible_nodes)
		else:
			# Default to resource-aware
			return self._resource_aware_selection(compatible_nodes, task)

	def _round_robin_selection(self, nodes: List[WorkerNode]) -> str:
		"""Round-robin node selection."""
		node = nodes[self._round_robin_index % len(nodes)]
		self._round_robin_index += 1
		return node.node_id

	def _least_connections_selection(self, nodes: List[WorkerNode]) -> str:
		"""Select node with least active connections."""
		min_connections = float('inf')
		selected_node = nodes[0]

		for node in nodes:
			connections = self._current_assignments.get(node.node_id, 0)
			if connections < min_connections:
				min_connections = connections
				selected_node = node

		return selected_node.node_id

	def _weighted_round_robin_selection(self, nodes: List[WorkerNode]) -> str:
		"""Weighted round-robin selection based on node weights."""
		# Calculate weights based on node capacity
		total_weight = 0.0
		node_weights = []

		for node in nodes:
			weight = self._calculate_node_weight(node)
			node_weights.append(weight)
			total_weight += weight

		if total_weight == 0:
			return self._round_robin_selection(nodes)

		# Select based on weighted probability
		target = random.uniform(0, total_weight)
		current_weight = 0.0

		for i, weight in enumerate(node_weights):
			current_weight += weight
			if current_weight >= target:
				return nodes[i].node_id

		return nodes[-1].node_id

	def _resource_aware_selection(self, nodes: List[WorkerNode], task: DistributedTask) -> str:
		"""Select node based on available resources and task requirements."""
		best_score = -1.0
		selected_node = nodes[0]

		for node in nodes:
			score = self._calculate_resource_score(node, task.resource_requirements)
			if score > best_score:
				best_score = score
				selected_node = node

		return selected_node.node_id

	def _latency_based_selection(self, nodes: List[WorkerNode]) -> str:
		"""Select node based on historical latency performance."""
		best_latency = float('inf')
		selected_node = nodes[0]

		for node in nodes:
			avg_latency = self._get_average_latency(node.node_id)
			if avg_latency < best_latency:
				best_latency = avg_latency
				selected_node = node

		return selected_node.node_id

	def _calculate_node_weight(self, node: WorkerNode) -> float:
		"""Calculate weight for a worker node."""
		# Base weight from resource capacity
		base_weight = (
			node.available_resources.get("cpu_cores", 0) +
			node.available_resources.get("memory_gb", 0) / 10.0 +
			node.available_resources.get("gpu_count", 0) * 10.0
		)

		# Adjust based on current load
		if node.current_metrics:
			load_factor = 1.0 - (node.current_metrics.cpu_utilization / 100.0)
			base_weight *= load_factor

		return max(0.1, base_weight)

	def _calculate_resource_score(self, node: WorkerNode, requirements: ResourceRequirement) -> float:
		"""Calculate resource fit score for node and task."""
		# Calculate resource utilization after assignment
		cpu_utilization = (requirements.cpu_cores / max(1, node.get_available_capacity("cpu_cores")))
		memory_utilization = (requirements.memory_gb / max(1, node.get_available_capacity("memory_gb")))

		# Prefer nodes with good fit (not too much or too little capacity)
		cpu_score = 1.0 - abs(0.7 - cpu_utilization)
		memory_score = 1.0 - abs(0.7 - memory_utilization)

		# Bonus for specialized hardware if needed
		hardware_score = 1.0
		if requirements.specialized_hardware:
			matching_hardware = sum(
				1 for hw in requirements.specialized_hardware
				if hw.value in node.capabilities
			)
			hardware_score = matching_hardware / len(requirements.specialized_hardware)

		# Health and performance factors
		health_score = node.current_metrics.health_score if node.current_metrics else 1.0
		efficiency_score = node.current_metrics.calculate_efficiency_score() if node.current_metrics else 1.0

		# Combined score
		total_score = (
			cpu_score * 0.25 +
			memory_score * 0.25 +
			hardware_score * 0.2 +
			health_score * 0.15 +
			efficiency_score * 0.15
		)

		return total_score

	def _get_average_latency(self, node_id: str) -> float:
		"""Get average latency for worker node."""
		history = self._performance_history.get(node_id, [])
		if not history:
			return 100.0  # Default latency

		return statistics.mean(history[-10:])  # Last 10 measurements

	def update_assignment(self, node_id: str, increment: int = 1) -> None:
		"""Update task assignment count for node."""
		self._current_assignments[node_id] = self._current_assignments.get(node_id, 0) + increment

	def record_performance(self, node_id: str, latency: float) -> None:
		"""Record performance metrics for node."""
		if node_id not in self._performance_history:
			self._performance_history[node_id] = []

		self._performance_history[node_id].append(latency)

		# Keep history reasonable size
		if len(self._performance_history[node_id]) > 100:
			self._performance_history[node_id] = self._performance_history[node_id][-50:]


class DistributedComputingCluster:
	"""Distributed computing cluster with intelligent auto-scaling.

	Manages a distributed computing cluster with automatic scaling,
	load balancing, fault tolerance, and comprehensive monitoring
	for massive-scale AI workload processing.

	Attributes:
		cluster_id: Unique cluster identifier
		cluster_name: Human-readable cluster name
		state: Current cluster state
		worker_nodes: Dictionary of worker nodes
		load_balancer: Load balancing system
		scaling_config: Auto-scaling configuration
		task_queue: Pending task queue
		active_tasks: Currently executing tasks
		completed_tasks: Completed task history
		cluster_metrics: Current cluster metrics
		historical_metrics: Historical performance data
	"""

	def __init__(self, cluster_name: str, scaling_config: Optional[AutoScalingConfiguration] = None):
		"""Initialize distributed computing cluster.

		Args:
			cluster_name: Name for the cluster
			scaling_config: Auto-scaling configuration
		"""
		self.cluster_id = uuid7str()
		self.cluster_name = cluster_name
		self.state = ClusterState.INITIALIZING
		self.creation_timestamp = datetime.now(timezone.utc)
		self.last_scaling_action = datetime.now(timezone.utc)

		# Cluster components
		self.worker_nodes: Dict[str, WorkerNode] = {}
		self.load_balancer = LoadBalancer(LoadBalancingStrategy.RESOURCE_AWARE)
		self.scaling_config = scaling_config or AutoScalingConfiguration()

		# Task management
		self.task_queue: List[DistributedTask] = []
		self.active_tasks: Dict[str, DistributedTask] = {}
		self.completed_tasks: List[DistributedTask] = []

		# Metrics and monitoring
		self.cluster_metrics: Optional[ClusterMetrics] = None
		self.historical_metrics: List[ClusterMetrics] = []

		# Performance tracking
		self.performance_stats = {
			"total_tasks_processed": 0,
			"total_execution_time": 0.0,
			"scaling_events": 0,
			"node_failures": 0,
			"average_queue_time": 0.0
		}

		# Initialize logging
		self._logger = logging.getLogger(__name__)

		# Start cluster
		self._initialize_cluster()

	def _initialize_cluster(self) -> None:
		"""Initialize cluster with minimum nodes."""
		try:
			# Create initial worker nodes
			for i in range(self.scaling_config.min_nodes):
				self._provision_worker_node(f"worker-{i:03d}")

			self.state = ClusterState.HEALTHY

			self._logger.info(_log_cluster_event(
				"INITIALIZATION", self.cluster_id, "initialize_cluster", "SUCCESS",
				f"nodes={self.scaling_config.min_nodes}"
			))

		except Exception as e:
			self.state = ClusterState.CRITICAL
			self._logger.error(f"Cluster initialization failed: {str(e)}")
			raise

	def _provision_worker_node(self, node_name: str, node_type: str = "standard") -> str:
		"""Provision new worker node in the cluster.

		Args:
			node_name: Name for the new worker node
			node_type: Type of worker node to provision

		Returns:
			str: Worker node ID
		"""
		try:
			# Create worker node configuration
			node = WorkerNode(
				cluster_id=self.cluster_id,
				node_name=node_name,
				node_type=node_type,
				available_resources=self._get_node_resources(node_type),
				capabilities=self._get_node_capabilities(node_type),
				endpoint_url=f"http://worker-{uuid7str()[:8]}.cluster.local:8080",
				container_id=f"container-{uuid7str()[:12]}",
				host_information={
					"os": "Linux",
					"architecture": "x86_64",
					"kernel_version": "5.15.0",
					"container_runtime": "Docker 24.0.0"
				}
			)

			# Simulate node provisioning process
			await self._simulate_node_provisioning(node)

			# Add to cluster
			self.worker_nodes[node.node_id] = node

			self._logger.info(_log_worker_event(
				node.node_id, "provisioned", "SUCCESS",
				f"type={node_type}, resources={node.available_resources}"
			))

			return node.node_id

		except Exception as e:
			self._logger.error(f"Worker node provisioning failed: {str(e)}")
			raise

	async def _simulate_node_provisioning(self, node: WorkerNode) -> None:
		"""Simulate the node provisioning process."""
		# Simulate provisioning time
		await asyncio.sleep(0.1)  # Simulate provisioning delay

		# Update node state
		node.update_state(WorkerNodeState.READY)

		# Initialize metrics
		initial_metrics = WorkerNodeMetrics(
			node_id=node.node_id,
			cpu_utilization=random.uniform(5, 15),
			memory_utilization=random.uniform(10, 20),
			health_score=random.uniform(0.9, 1.0),
			uptime_seconds=0.0
		)

		node.update_metrics(initial_metrics)

	def _get_node_resources(self, node_type: str) -> Dict[str, float]:
		"""Get default resources for node type."""
		resource_configs = {
			"standard": {
				"cpu_cores": 4.0,
				"memory_gb": 16.0,
				"gpu_count": 0,
				"storage_gb": 100.0
			},
			"compute_optimized": {
				"cpu_cores": 8.0,
				"memory_gb": 32.0,
				"gpu_count": 0,
				"storage_gb": 200.0
			},
			"gpu_accelerated": {
				"cpu_cores": 8.0,
				"memory_gb": 64.0,
				"gpu_count": 2,
				"storage_gb": 500.0
			},
			"memory_optimized": {
				"cpu_cores": 4.0,
				"memory_gb": 128.0,
				"gpu_count": 0,
				"storage_gb": 200.0
			},
			"neuromorphic": {
				"cpu_cores": 16.0,
				"memory_gb": 256.0,
				"gpu_count": 4,
				"storage_gb": 1000.0
			}
		}

		return resource_configs.get(node_type, resource_configs["standard"])

	def _get_node_capabilities(self, node_type: str) -> List[str]:
		"""Get capabilities for node type."""
		capability_configs = {
			"standard": ["cpu", "general_compute"],
			"compute_optimized": ["cpu", "high_performance_compute", "parallel_processing"],
			"gpu_accelerated": ["cpu", "gpu", "cuda", "tensor_operations", "machine_learning"],
			"memory_optimized": ["cpu", "large_memory", "in_memory_analytics"],
			"neuromorphic": ["cpu", "gpu", "neuromorphic", "spiking_networks", "edge_ai"]
		}

		return capability_configs.get(node_type, capability_configs["standard"])

	async def submit_task(self, task: DistributedTask) -> str:
		"""Submit task for distributed execution.

		Args:
			task: Task to be executed

		Returns:
			str: Task ID
		"""
		try:
			# Validate task
			if not task.task_type:
				raise ValueError("Task type is required")

			# Add to task queue
			self.task_queue.append(task)

			# Try to schedule immediately
			await self._schedule_pending_tasks()

			self._logger.info(f"Task submitted: {task.task_id} (type: {task.task_type})")

			return task.task_id

		except Exception as e:
			self._logger.error(f"Task submission failed: {str(e)}")
			raise

	async def _schedule_pending_tasks(self) -> None:
		"""Schedule pending tasks to available worker nodes."""
		if not self.task_queue:
			return

		# Get available worker nodes
		available_nodes = [
			node for node in self.worker_nodes.values()
			if node.state == WorkerNodeState.READY and not node.current_metrics or not node.current_metrics.is_overloaded()
		]

		if not available_nodes:
			return

		# Schedule tasks
		scheduled_tasks = []

		for task in self.task_queue[:]:
			# Check dependencies
			if not self._check_task_dependencies(task):
				continue

			# Select worker node
			selected_node_id = self.load_balancer.select_worker_node(task, available_nodes)

			if selected_node_id:
				# Allocate resources
				selected_node = self.worker_nodes[selected_node_id]
				if selected_node.allocate_resources(task.resource_requirements):
					# Start task execution
					task.start_execution(selected_node_id)

					# Move to active tasks
					self.active_tasks[task.task_id] = task
					scheduled_tasks.append(task)

					# Update load balancer
					self.load_balancer.update_assignment(selected_node_id)

					# Simulate task execution
					asyncio.create_task(self._execute_task(task))

					self._logger.info(f"Task scheduled: {task.task_id} -> {selected_node_id}")
				else:
					self._logger.warning(f"Resource allocation failed for task: {task.task_id}")

		# Remove scheduled tasks from queue
		for task in scheduled_tasks:
			self.task_queue.remove(task)

	def _check_task_dependencies(self, task: DistributedTask) -> bool:
		"""Check if task dependencies are satisfied."""
		for dep_task_id in task.dependencies:
			# Check if dependency is completed
			dep_task = next(
				(t for t in self.completed_tasks if t.task_id == dep_task_id),
				None
			)
			if not dep_task or dep_task.state != "completed":
				return False

		return True

	async def _execute_task(self, task: DistributedTask) -> None:
		"""Execute task on assigned worker node."""
		try:
			# Simulate task execution
			execution_time = random.uniform(1.0, 10.0)  # 1-10 seconds
			await asyncio.sleep(execution_time / 10.0)  # Speed up simulation

			# Simulate task progress
			for progress in [0.2, 0.5, 0.8, 1.0]:
				task.progress = progress
				await asyncio.sleep(execution_time / 40.0)

			# Simulate task result
			if random.random() < 0.05:  # 5% failure rate
				task.fail_execution("Simulated task failure")
				self._logger.warning(f"Task failed: {task.task_id}")
			else:
				result = {
					"status": "success",
					"output": f"Task {task.task_id} completed successfully",
					"execution_time": execution_time,
					"worker_node": task.assigned_node_id
				}
				task.complete_execution(result)
				self._logger.info(f"Task completed: {task.task_id}")

			# Update performance stats
			self.performance_stats["total_tasks_processed"] += 1
			self.performance_stats["total_execution_time"] += execution_time

			# Move to completed tasks
			if task.task_id in self.active_tasks:
				del self.active_tasks[task.task_id]
			self.completed_tasks.append(task)

			# Deallocate resources
			if task.assigned_node_id and task.assigned_node_id in self.worker_nodes:
				worker_node = self.worker_nodes[task.assigned_node_id]
				worker_node.deallocate_resources(task.resource_requirements)

				# Update load balancer
				self.load_balancer.update_assignment(task.assigned_node_id, -1)

			# Keep completed tasks list reasonable size
			if len(self.completed_tasks) > 10000:
				self.completed_tasks = self.completed_tasks[-5000:]

		except Exception as e:
			task.fail_execution(f"Execution error: {str(e)}")
			self._logger.error(f"Task execution failed: {task.task_id} - {str(e)}")

	async def update_cluster_metrics(self) -> None:
		"""Update comprehensive cluster metrics."""
		try:
			# Calculate cluster-wide metrics
			total_nodes = len(self.worker_nodes)
			healthy_nodes = sum(1 for node in self.worker_nodes.values() if node.current_metrics and node.current_metrics.is_healthy())
			busy_nodes = sum(1 for node in self.worker_nodes.values() if node.state == WorkerNodeState.BUSY)
			failed_nodes = sum(1 for node in self.worker_nodes.values() if node.state == WorkerNodeState.FAILED)

			# Resource calculations
			total_cpu = sum(node.available_resources.get("cpu_cores", 0) for node in self.worker_nodes.values())
			available_cpu = sum(node.get_available_capacity("cpu_cores") for node in self.worker_nodes.values())
			total_memory = sum(node.available_resources.get("memory_gb", 0) for node in self.worker_nodes.values())
			available_memory = sum(node.get_available_capacity("memory_gb") for node in self.worker_nodes.values())
			total_gpu = sum(node.available_resources.get("gpu_count", 0) for node in self.worker_nodes.values())
			available_gpu = sum(node.get_available_capacity("gpu_count") for node in self.worker_nodes.values())

			# Task statistics
			active_tasks = len(self.active_tasks)
			pending_tasks = len(self.task_queue)

			# Calculate throughput and error rates
			current_time = datetime.now(timezone.utc)
			one_hour_ago = current_time - timedelta(hours=1)

			recent_completed = [t for t in self.completed_tasks if t.completion_timestamp and t.completion_timestamp > one_hour_ago]
			completed_last_hour = len([t for t in recent_completed if t.state == "completed"])
			failed_last_hour = len([t for t in recent_completed if t.state == "failed"])

			# Performance calculations
			if recent_completed:
				durations = [t.get_execution_duration() for t in recent_completed if t.get_execution_duration()]
				average_duration = statistics.mean(durations) if durations else 0.0
			else:
				average_duration = 0.0

			cluster_utilization = 1.0 - (available_cpu / max(1, total_cpu)) if total_cpu > 0 else 0.0
			throughput = completed_last_hour / 3600.0  # tasks per second
			error_rate = failed_last_hour / max(1, completed_last_hour + failed_last_hour)

			# Health score calculation
			health_scores = [node.current_metrics.health_score for node in self.worker_nodes.values() if node.current_metrics]
			cluster_health = statistics.mean(health_scores) if health_scores else 1.0

			# Cost estimation
			cost_per_hour = sum(
				self._calculate_node_cost(node) for node in self.worker_nodes.values()
			)

			# Efficiency score
			efficiency_scores = [
				node.current_metrics.calculate_efficiency_score()
				for node in self.worker_nodes.values()
				if node.current_metrics
			]
			cluster_efficiency = statistics.mean(efficiency_scores) if efficiency_scores else 1.0

			# Create cluster metrics
			self.cluster_metrics = ClusterMetrics(
				cluster_id=self.cluster_id,
				total_nodes=total_nodes,
				healthy_nodes=healthy_nodes,
				busy_nodes=busy_nodes,
				failed_nodes=failed_nodes,
				total_cpu_cores=total_cpu,
				available_cpu_cores=available_cpu,
				total_memory_gb=total_memory,
				available_memory_gb=available_memory,
				total_gpu_count=int(total_gpu),
				available_gpu_count=int(available_gpu),
				active_tasks=active_tasks,
				pending_tasks=pending_tasks,
				completed_tasks_last_hour=completed_last_hour,
				failed_tasks_last_hour=failed_last_hour,
				average_task_duration=average_duration,
				cluster_utilization=cluster_utilization,
				throughput_tasks_per_second=throughput,
				error_rate=error_rate,
				health_score=cluster_health,
				cost_per_hour=cost_per_hour,
				efficiency_score=cluster_efficiency
			)

			# Store historical metrics
			self.historical_metrics.append(self.cluster_metrics)

			# Keep historical metrics reasonable size
			if len(self.historical_metrics) > 1000:
				self.historical_metrics = self.historical_metrics[-500:]

		except Exception as e:
			self._logger.error(f"Cluster metrics update failed: {str(e)}")

	def _calculate_node_cost(self, node: WorkerNode) -> float:
		"""Calculate hourly cost for a worker node."""
		base_cost = 0.0

		# CPU cost
		base_cost += node.available_resources.get("cpu_cores", 0) * 0.05

		# Memory cost
		base_cost += node.available_resources.get("memory_gb", 0) * 0.01

		# GPU cost
		base_cost += node.available_resources.get("gpu_count", 0) * 0.50

		# Premium for specialized nodes
		if "neuromorphic" in node.capabilities:
			base_cost *= 3.0
		elif "gpu" in node.capabilities:
			base_cost *= 1.5

		return base_cost

	async def auto_scale_cluster(self) -> None:
		"""Perform automatic cluster scaling based on current metrics."""
		if not self.scaling_config.enabled or not self.cluster_metrics:
			return

		try:
			# Check cooldown periods
			time_since_last_scaling = (datetime.now(timezone.utc) - self.last_scaling_action).total_seconds()

			# Decide on scaling action
			scale_up, nodes_to_add, scale_up_reason = self.scaling_config.get_scale_up_decision(self.cluster_metrics)
			scale_down, nodes_to_remove, scale_down_reason = self.scaling_config.get_scale_down_decision(self.cluster_metrics)

			if scale_up and time_since_last_scaling >= self.scaling_config.scale_up_cooldown_seconds:
				await self._scale_up(nodes_to_add, scale_up_reason)
			elif scale_down and time_since_last_scaling >= self.scaling_config.scale_down_cooldown_seconds:
				await self._scale_down(nodes_to_remove, scale_down_reason)

		except Exception as e:
			self._logger.error(f"Auto-scaling failed: {str(e)}")

	async def _scale_up(self, nodes_to_add: int, reason: str) -> None:
		"""Scale up cluster by adding worker nodes."""
		try:
			current_count = len(self.worker_nodes)

			self.state = ClusterState.SCALING_UP

			# Add worker nodes
			for i in range(nodes_to_add):
				node_name = f"worker-{current_count + i:03d}"
				await self._provision_worker_node(node_name)

			self.state = ClusterState.HEALTHY
			self.last_scaling_action = datetime.now(timezone.utc)
			self.performance_stats["scaling_events"] += 1

			self._logger.info(_log_scaling_event(
				self.cluster_id, "SCALE_UP", current_count, current_count + nodes_to_add, reason
			))

		except Exception as e:
			self.state = ClusterState.DEGRADED
			self._logger.error(f"Scale up failed: {str(e)}")
			raise

	async def _scale_down(self, nodes_to_remove: int, reason: str) -> None:
		"""Scale down cluster by removing worker nodes."""
		try:
			current_count = len(self.worker_nodes)

			if current_count <= self.scaling_config.min_nodes:
				return

			self.state = ClusterState.SCALING_DOWN

			# Select nodes to remove (prefer least utilized)
			nodes_to_terminate = self._select_nodes_for_removal(nodes_to_remove)

			# Gracefully drain and remove nodes
			for node_id in nodes_to_terminate:
				await self._drain_and_remove_node(node_id)

			self.state = ClusterState.HEALTHY
			self.last_scaling_action = datetime.now(timezone.utc)
			self.performance_stats["scaling_events"] += 1

			self._logger.info(_log_scaling_event(
				self.cluster_id, "SCALE_DOWN", current_count, current_count - len(nodes_to_terminate), reason
			))

		except Exception as e:
			self.state = ClusterState.DEGRADED
			self._logger.error(f"Scale down failed: {str(e)}")
			raise

	def _select_nodes_for_removal(self, count: int) -> List[str]:
		"""Select worker nodes for removal during scale down."""
		# Prefer nodes with low utilization and no active tasks
		candidates = []

		for node in self.worker_nodes.values():
			if node.state in [WorkerNodeState.READY, WorkerNodeState.FAILED]:
				utilization = 0.0
				if node.current_metrics:
					utilization = (node.current_metrics.cpu_utilization + node.current_metrics.memory_utilization) / 2.0

				candidates.append((node.node_id, utilization))

		# Sort by utilization (ascending)
		candidates.sort(key=lambda x: x[1])

		# Select nodes to remove
		return [node_id for node_id, _ in candidates[:count]]

	async def _drain_and_remove_node(self, node_id: str) -> None:
		"""Gracefully drain and remove worker node."""
		if node_id not in self.worker_nodes:
			return

		try:
			node = self.worker_nodes[node_id]

			# Set node to draining state
			node.update_state(WorkerNodeState.DRAINING)

			# Wait for active tasks to complete (simplified)
			await asyncio.sleep(0.1)  # Simulate draining time

			# Remove node
			node.update_state(WorkerNodeState.TERMINATING)
			del self.worker_nodes[node_id]

			self._logger.info(_log_worker_event(node_id, "removed", "SUCCESS"))

		except Exception as e:
			self._logger.error(f"Node removal failed: {node_id} - {str(e)}")
			raise

	async def health_check_and_recovery(self) -> None:
		"""Perform health checks and automatic recovery."""
		try:
			failed_nodes = []

			for node_id, node in self.worker_nodes.items():
				# Check node health
				if node.current_metrics and not node.current_metrics.is_healthy():
					if node.state != WorkerNodeState.FAILED:
						node.update_state(WorkerNodeState.FAILED)
						failed_nodes.append(node_id)
						self.performance_stats["node_failures"] += 1

						self._logger.warning(_log_worker_event(node_id, "health_check", "FAILED"))

			# Replace failed nodes if auto-recovery is enabled
			if failed_nodes and self.scaling_config.enabled:
				for failed_node_id in failed_nodes:
					# Remove failed node
					await self._drain_and_remove_node(failed_node_id)

					# Add replacement node
					replacement_name = f"worker-recovery-{uuid7str()[:8]}"
					await self._provision_worker_node(replacement_name)

					self._logger.info(f"Replaced failed node: {failed_node_id}")

		except Exception as e:
			self._logger.error(f"Health check and recovery failed: {str(e)}")

	async def get_cluster_status(self) -> Dict[str, Any]:
		"""Get comprehensive cluster status.

		Returns:
			Dict[str, Any]: Cluster status information
		"""
		await self.update_cluster_metrics()

		return {
			"cluster_info": {
				"cluster_id": self.cluster_id,
				"cluster_name": self.cluster_name,
				"state": self.state.value,
				"creation_timestamp": self.creation_timestamp.isoformat(),
				"uptime_seconds": (datetime.now(timezone.utc) - self.creation_timestamp).total_seconds()
			},
			"cluster_metrics": self.cluster_metrics.model_dump() if self.cluster_metrics else {},
			"worker_nodes": {
				"total": len(self.worker_nodes),
				"by_state": self._count_nodes_by_state(),
				"by_type": self._count_nodes_by_type(),
				"node_details": [
					{
						"node_id": node.node_id,
						"node_name": node.node_name,
						"state": node.state.value,
						"resources": node.available_resources,
						"metrics": node.current_metrics.model_dump() if node.current_metrics else {}
					}
					for node in list(self.worker_nodes.values())[:10]  # Limit for display
				]
			},
			"task_management": {
				"active_tasks": len(self.active_tasks),
				"pending_tasks": len(self.task_queue),
				"completed_tasks": len(self.completed_tasks),
				"task_queue": [
					{
						"task_id": task.task_id,
						"task_type": task.task_type,
						"priority": task.priority,
						"state": task.state
					}
					for task in self.task_queue[:5]  # Show first 5
				]
			},
			"auto_scaling": {
				"enabled": self.scaling_config.enabled,
				"policy": self.scaling_config.scaling_policy.value,
				"min_nodes": self.scaling_config.min_nodes,
				"max_nodes": self.scaling_config.max_nodes,
				"current_nodes": len(self.worker_nodes),
				"last_scaling_action": self.last_scaling_action.isoformat()
			},
			"performance_stats": dict(self.performance_stats),
			"load_balancing": {
				"strategy": self.load_balancer._strategy.value,
				"current_assignments": dict(self.load_balancer._current_assignments)
			}
		}

	def _count_nodes_by_state(self) -> Dict[str, int]:
		"""Count worker nodes by state."""
		counts = {}
		for node in self.worker_nodes.values():
			state = node.state.value
			counts[state] = counts.get(state, 0) + 1
		return counts

	def _count_nodes_by_type(self) -> Dict[str, int]:
		"""Count worker nodes by type."""
		counts = {}
		for node in self.worker_nodes.values():
			node_type = node.node_type
			counts[node_type] = counts.get(node_type, 0) + 1
		return counts

	async def shutdown_cluster(self) -> None:
		"""Gracefully shutdown the cluster."""
		try:
			self.state = ClusterState.MAINTENANCE

			# Stop accepting new tasks
			self.task_queue.clear()

			# Wait for active tasks to complete
			while self.active_tasks:
				await asyncio.sleep(1.0)

			# Shutdown all worker nodes
			for node_id in list(self.worker_nodes.keys()):
				await self._drain_and_remove_node(node_id)

			self.state = ClusterState.TERMINATED

			self._logger.info(_log_cluster_event(
				"SHUTDOWN", self.cluster_id, "shutdown_cluster", "SUCCESS"
			))

		except Exception as e:
			self._logger.error(f"Cluster shutdown failed: {str(e)}")
			raise


# Module exports
__all__ = [
	# Core distributed computing cluster
	"DistributedComputingCluster",

	# Task and resource management
	"DistributedTask", "ResourceRequirement", "WorkerNode",

	# Metrics and monitoring
	"ClusterMetrics", "WorkerNodeMetrics",

	# Configuration and policies
	"AutoScalingConfiguration", "LoadBalancer",

	# Enums
	"ClusterState", "WorkerNodeState", "ResourceType", "ScalingPolicy", "LoadBalancingStrategy",

	# Utility functions
	"_log_cluster_event", "_log_worker_event", "_log_scaling_event"
]