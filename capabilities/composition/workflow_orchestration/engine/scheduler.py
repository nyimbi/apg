"""
APG Workflow Orchestration Scheduler

Intelligent workflow scheduler with priority-based task queuing, multi-workflow
coordination, and dynamic resource allocation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import heapq
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict
from uuid_extensions import uuid7str

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func

from ..models import (
	Workflow, WorkflowInstance, TaskDefinition, TaskExecution,
	WorkflowStatus, TaskStatus, TaskType, Priority
)
from ..database import (
	CRWorkflow, CRWorkflowInstance, CRTaskExecution,
	DatabaseManager, create_repositories
)

logger = logging.getLogger(__name__)

class SchedulingStrategy(Enum):
	"""Task scheduling strategies."""
	FIFO = "fifo"  # First In, First Out
	PRIORITY = "priority"  # Priority-based scheduling
	SHORTEST_JOB_FIRST = "sjf"  # Shortest Job First
	ROUND_ROBIN = "round_robin"  # Round Robin scheduling
	FAIR_SHARE = "fair_share"  # Fair share scheduling
	DEADLINE_FIRST = "deadline_first"  # Earliest Deadline First
	ADAPTIVE = "adaptive"  # AI-driven adaptive scheduling

class ResourceType(Enum):
	"""Resource types for allocation."""
	CPU = "cpu"
	MEMORY = "memory"
	NETWORK = "network"
	STORAGE = "storage"
	CUSTOM = "custom"

@dataclass
class ScheduledTask:
	"""Task scheduled for execution."""
	task_id: str
	workflow_id: str
	instance_id: str
	task_definition: TaskDefinition
	priority: int
	scheduled_at: datetime
	estimated_duration: float  # seconds
	resource_requirements: Dict[ResourceType, float]
	dependencies: List[str]
	retry_count: int = 0
	max_retries: int = 3
	delay_until: Optional[datetime] = None
	tenant_id: str = ""
	user_id: str = ""
	
	def __lt__(self, other: 'ScheduledTask') -> bool:
		"""Compare tasks for priority scheduling."""
		# Higher priority number = higher priority (reverse of typical heap)
		if self.priority != other.priority:
			return self.priority > other.priority
		# If same priority, earlier scheduled time wins
		return self.scheduled_at < other.scheduled_at

@dataclass
class WorkerNode:
	"""Worker node for distributed execution."""
	node_id: str
	capacity: Dict[ResourceType, float]
	current_load: Dict[ResourceType, float] = field(default_factory=dict)
	active_tasks: Set[str] = field(default_factory=set)
	last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	is_available: bool = True
	metadata: Dict[str, Any] = field(default_factory=dict)
	
	def get_utilization(self, resource_type: ResourceType) -> float:
		"""Get current resource utilization percentage."""
		if resource_type not in self.capacity or self.capacity[resource_type] == 0:
			return 0.0
		current = self.current_load.get(resource_type, 0.0)
		return (current / self.capacity[resource_type]) * 100.0
	
	def can_handle_task(self, task: ScheduledTask) -> bool:
		"""Check if worker can handle the task."""
		if not self.is_available:
			return False
		
		for resource_type, required in task.resource_requirements.items():
			available = self.capacity.get(resource_type, 0.0) - self.current_load.get(resource_type, 0.0)
			if available < required:
				return False
		return True

class WorkflowScheduler:
	"""Intelligent workflow and task scheduler."""
	
	def __init__(
		self,
		database_manager: DatabaseManager,
		redis_client: redis.Redis,
		tenant_id: str,
		strategy: SchedulingStrategy = SchedulingStrategy.ADAPTIVE
	):
		self.database_manager = database_manager
		self.redis_client = redis_client
		self.tenant_id = tenant_id
		self.strategy = strategy
		
		# Task queues by priority
		self.task_queue: List[ScheduledTask] = []
		self.delayed_tasks: List[ScheduledTask] = []
		self.running_tasks: Dict[str, ScheduledTask] = {}
		
		# Worker nodes
		self.worker_nodes: Dict[str, WorkerNode] = {}
		
		# Scheduling statistics
		self.stats = {
			"tasks_scheduled": 0,
			"tasks_completed": 0,
			"tasks_failed": 0,
			"average_wait_time": 0.0,
			"average_execution_time": 0.0,
			"resource_utilization": defaultdict(float),
			"workflow_throughput": defaultdict(int),
			"scheduling_decisions": []
		}
		
		# Background tasks
		self._scheduler_task: Optional[asyncio.Task] = None
		self._resource_monitor_task: Optional[asyncio.Task] = None
		self._cleanup_task: Optional[asyncio.Task] = None
		
		# Event callbacks
		self.task_scheduled_callbacks: List[Callable] = []
		self.task_completed_callbacks: List[Callable] = []
		
		logger.info(f"Initialized WorkflowScheduler for tenant {tenant_id} with strategy {strategy.value}")
	
	async def start(self) -> None:
		"""Start the scheduler and background tasks."""
		self._scheduler_task = asyncio.create_task(self._scheduler_loop())
		self._resource_monitor_task = asyncio.create_task(self._resource_monitor_loop())
		self._cleanup_task = asyncio.create_task(self._cleanup_loop())
		
		# Initialize default worker if none exist
		if not self.worker_nodes:
			await self.register_worker_node({
				"node_id": f"default_{uuid7str()}",
				"capacity": {
					ResourceType.CPU: 4.0,
					ResourceType.MEMORY: 8.0,  # GB
					ResourceType.NETWORK: 1.0,  # Gbps
					ResourceType.STORAGE: 100.0  # GB
				}
			})
		
		logger.info("WorkflowScheduler started")
	
	async def stop(self) -> None:
		"""Stop the scheduler and cleanup resources."""
		if self._scheduler_task:
			self._scheduler_task.cancel()
		if self._resource_monitor_task:
			self._resource_monitor_task.cancel()
		if self._cleanup_task:
			self._cleanup_task.cancel()
		
		await asyncio.gather(
			self._scheduler_task, self._resource_monitor_task, self._cleanup_task,
			return_exceptions=True
		)
		
		logger.info("WorkflowScheduler stopped")
	
	async def schedule_task(
		self,
		workflow_id: str,
		instance_id: str,
		task_definition: TaskDefinition,
		user_id: str,
		priority: Optional[int] = None,
		estimated_duration: float = 60.0,
		resource_requirements: Optional[Dict[ResourceType, float]] = None
	) -> str:
		"""Schedule a task for execution."""
		
		# Calculate priority if not provided
		if priority is None:
			priority = self._calculate_task_priority(task_definition)
		
		# Default resource requirements
		if resource_requirements is None:
			resource_requirements = self._estimate_resource_requirements(task_definition)
		
		# Create scheduled task
		scheduled_task = ScheduledTask(
			task_id=task_definition.id,
			workflow_id=workflow_id,
			instance_id=instance_id,
			task_definition=task_definition,
			priority=priority,
			scheduled_at=datetime.now(timezone.utc),
			estimated_duration=estimated_duration,
			resource_requirements=resource_requirements,
			dependencies=task_definition.dependencies,
			tenant_id=self.tenant_id,
			user_id=user_id
		)
		
		# Add to appropriate queue
		if self._dependencies_met(scheduled_task):
			heapq.heappush(self.task_queue, scheduled_task)
		else:
			self.delayed_tasks.append(scheduled_task)
		
		# Update statistics
		self.stats["tasks_scheduled"] += 1
		
		# Notify callbacks
		await self._notify_task_scheduled(scheduled_task)
		
		# Cache in Redis for persistence
		await self._cache_scheduled_task(scheduled_task)
		
		logger.info(f"Scheduled task {task_definition.id} for workflow {workflow_id}")
		return scheduled_task.task_id
	
	async def cancel_task(self, task_id: str) -> bool:
		"""Cancel a scheduled or running task."""
		
		# Remove from task queue
		self.task_queue = [task for task in self.task_queue if task.task_id != task_id]
		heapq.heapify(self.task_queue)
		
		# Remove from delayed tasks
		self.delayed_tasks = [task for task in self.delayed_tasks if task.task_id != task_id]
		
		# Cancel running task
		if task_id in self.running_tasks:
			running_task = self.running_tasks[task_id]
			await self._cancel_running_task(running_task)
			del self.running_tasks[task_id]
		
		# Remove from Redis cache
		await self.redis_client.delete(f"scheduled_task:{self.tenant_id}:{task_id}")
		
		logger.info(f"Cancelled task {task_id}")
		return True
	
	async def register_worker_node(self, node_config: Dict[str, Any]) -> str:
		"""Register a new worker node."""
		
		node_id = node_config["node_id"]
		capacity = {ResourceType(k): v for k, v in node_config["capacity"].items()}
		
		worker_node = WorkerNode(
			node_id=node_id,
			capacity=capacity,
			current_load={resource_type: 0.0 for resource_type in capacity.keys()},
			metadata=node_config.get("metadata", {})
		)
		
		self.worker_nodes[node_id] = worker_node
		
		# Cache in Redis
		await self.redis_client.setex(
			f"worker_node:{self.tenant_id}:{node_id}",
			300,  # 5 minutes TTL
			json.dumps({
				"node_id": node_id,
				"capacity": {k.value: v for k, v in capacity.items()},
				"registered_at": datetime.now(timezone.utc).isoformat()
			})
		)
		
		logger.info(f"Registered worker node {node_id}")
		return node_id
	
	async def unregister_worker_node(self, node_id: str) -> bool:
		"""Unregister a worker node."""
		
		if node_id not in self.worker_nodes:
			return False
		
		# Cancel running tasks on this node
		node = self.worker_nodes[node_id]
		for task_id in list(node.active_tasks):
			if task_id in self.running_tasks:
				await self._cancel_running_task(self.running_tasks[task_id])
		
		# Remove from active nodes
		del self.worker_nodes[node_id]
		
		# Remove from Redis
		await self.redis_client.delete(f"worker_node:{self.tenant_id}:{node_id}")
		
		logger.info(f"Unregistered worker node {node_id}")
		return True
	
	async def get_scheduling_stats(self) -> Dict[str, Any]:
		"""Get current scheduling statistics."""
		
		# Calculate resource utilization
		total_utilization = defaultdict(float)
		node_count = len(self.worker_nodes)
		
		if node_count > 0:
			for node in self.worker_nodes.values():
				for resource_type in ResourceType:
					total_utilization[resource_type.value] += node.get_utilization(resource_type)
			
			# Average utilization across nodes
			for resource_type in total_utilization:
				total_utilization[resource_type] /= node_count
		
		return {
			"queue_size": len(self.task_queue),
			"delayed_tasks": len(self.delayed_tasks),
			"running_tasks": len(self.running_tasks),
			"worker_nodes": len(self.worker_nodes),
			"resource_utilization": dict(total_utilization),
			"tasks_scheduled": self.stats["tasks_scheduled"],
			"tasks_completed": self.stats["tasks_completed"],
			"tasks_failed": self.stats["tasks_failed"],
			"average_wait_time": self.stats["average_wait_time"],
			"average_execution_time": self.stats["average_execution_time"],
			"scheduling_strategy": self.strategy.value,
			"tenant_id": self.tenant_id
		}
	
	async def _scheduler_loop(self) -> None:
		"""Main scheduler loop."""
		
		while True:
			try:
				# Process delayed tasks that are now ready
				await self._process_delayed_tasks()
				
				# Schedule ready tasks to available workers
				await self._schedule_ready_tasks()
				
				# Update scheduling statistics
				await self._update_scheduling_stats()
				
				# Sleep based on queue size (adaptive polling)
				queue_size = len(self.task_queue) + len(self.delayed_tasks)
				sleep_time = max(0.1, min(5.0, 1.0 / max(1, queue_size)))
				await asyncio.sleep(sleep_time)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Scheduler loop error: {e}")
				await asyncio.sleep(1.0)
	
	async def _process_delayed_tasks(self) -> None:
		"""Process delayed tasks and move ready ones to main queue."""
		
		ready_tasks = []
		remaining_tasks = []
		
		for task in self.delayed_tasks:
			# Check if delay period has passed
			if task.delay_until and datetime.now(timezone.utc) < task.delay_until:
				remaining_tasks.append(task)
				continue
			
			# Check if dependencies are met
			if self._dependencies_met(task):
				ready_tasks.append(task)
			else:
				remaining_tasks.append(task)
		
		# Move ready tasks to main queue
		for task in ready_tasks:
			heapq.heappush(self.task_queue, task)
		
		# Update delayed tasks list
		self.delayed_tasks = remaining_tasks
	
	async def _schedule_ready_tasks(self) -> None:
		"""Schedule ready tasks to available workers."""
		
		scheduled_count = 0
		
		while self.task_queue and scheduled_count < 10:  # Limit batch size
			task = heapq.heappop(self.task_queue)
			
			# Find best worker for this task
			best_worker = self._find_best_worker(task)
			
			if best_worker:
				# Assign task to worker
				await self._assign_task_to_worker(task, best_worker)
				scheduled_count += 1
			else:
				# No suitable worker available, put task back
				heapq.heappush(self.task_queue, task)
				break
	
	def _find_best_worker(self, task: ScheduledTask) -> Optional[WorkerNode]:
		"""Find the best worker node for a task."""
		
		available_workers = [
			worker for worker in self.worker_nodes.values()
			if worker.can_handle_task(task)
		]
		
		if not available_workers:
			return None
		
		# Select worker based on strategy
		if self.strategy == SchedulingStrategy.PRIORITY:
			# Choose worker with lowest current utilization
			return min(available_workers, key=lambda w: sum(w.current_load.values()))
		
		elif self.strategy == SchedulingStrategy.FAIR_SHARE:
			# Choose worker with least number of active tasks
			return min(available_workers, key=lambda w: len(w.active_tasks))
		
		elif self.strategy == SchedulingStrategy.ADAPTIVE:
			# Use ML-based worker selection (simplified heuristic for now)
			scores = []
			for worker in available_workers:
				# Calculate composite score based on multiple factors
				utilization_score = 100 - sum(worker.get_utilization(rt) for rt in ResourceType) / len(ResourceType)
				task_count_score = max(0, 100 - len(worker.active_tasks) * 10)
				heartbeat_score = 100 if (datetime.now(timezone.utc) - worker.last_heartbeat).seconds < 30 else 50
				
				composite_score = (utilization_score * 0.4 + task_count_score * 0.4 + heartbeat_score * 0.2)
				scores.append((composite_score, worker))
			
			return max(scores, key=lambda x: x[0])[1] if scores else available_workers[0]
		
		else:
			# Default: first available worker
			return available_workers[0]
	
	async def _assign_task_to_worker(self, task: ScheduledTask, worker: WorkerNode) -> None:
		"""Assign a task to a worker node."""
		
		# Update worker load
		for resource_type, required in task.resource_requirements.items():
			worker.current_load[resource_type] = worker.current_load.get(resource_type, 0.0) + required
		
		# Add to worker's active tasks
		worker.active_tasks.add(task.task_id)
		
		# Add to running tasks
		self.running_tasks[task.task_id] = task
		
		# Execute the task (this would integrate with the WorkflowExecutor)
		asyncio.create_task(self._execute_task(task, worker))
		
		# Record scheduling decision
		self.stats["scheduling_decisions"].append({
			"task_id": task.task_id,
			"worker_id": worker.node_id,
			"scheduled_at": datetime.now(timezone.utc).isoformat(),
			"wait_time": (datetime.now(timezone.utc) - task.scheduled_at).total_seconds(),
			"strategy": self.strategy.value
		})
		
		logger.debug(f"Assigned task {task.task_id} to worker {worker.node_id}")
	
	async def _execute_task(self, task: ScheduledTask, worker: WorkerNode) -> None:
		"""Execute a task on a worker node."""
		
		start_time = datetime.now(timezone.utc)
		
		try:
			# Integrate with the actual WorkflowExecutor for real task execution
			from .executor import WorkflowExecutor
			
			# Get or create executor instance
			if not hasattr(self, '_executor'):
				self._executor = WorkflowExecutor()
			
			# Execute the task using the real executor
			execution_result = await self._executor.execute_task(
				task_id=task.task_id,
				workflow_instance_id=task.instance_id,
				input_data=task.input_data,
				context=task.execution_context
			)
			
			# Check execution result
			if execution_result.get('success', False):
				await self._task_completed(task, worker, success=True, result=execution_result)
			else:
				error_msg = execution_result.get('error', 'Task execution failed')
				await self._task_completed(task, worker, success=False, error=error_msg)
			
		except Exception as e:
			logger.error(f"Task execution failed {task.task_id}: {e}")
			await self._task_completed(task, worker, success=False, error=str(e))
	
	async def _task_completed(self, task: ScheduledTask, worker: WorkerNode, success: bool, error: str = None) -> None:
		"""Handle task completion."""
		
		# Update worker load
		for resource_type, required in task.resource_requirements.items():
			worker.current_load[resource_type] = max(0.0, worker.current_load.get(resource_type, 0.0) - required)
		
		# Remove from worker's active tasks
		worker.active_tasks.discard(task.task_id)
		
		# Remove from running tasks
		if task.task_id in self.running_tasks:
			del self.running_tasks[task.task_id]
		
		# Update statistics
		if success:
			self.stats["tasks_completed"] += 1
		else:
			self.stats["tasks_failed"] += 1
			
			# Retry logic
			if task.retry_count < task.max_retries:
				task.retry_count += 1
				task.delay_until = datetime.now(timezone.utc) + timedelta(seconds=2 ** task.retry_count)
				self.delayed_tasks.append(task)
				logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
		
		# Notify callbacks
		await self._notify_task_completed(task, success, error)
		
		# Clean up Redis cache
		await self.redis_client.delete(f"scheduled_task:{self.tenant_id}:{task.task_id}")
	
	async def _cancel_running_task(self, task: ScheduledTask) -> None:
		"""Cancel a running task."""
		# In a real implementation, this would send cancellation signals to workers
		logger.info(f"Cancelling running task {task.task_id}")
	
	def _dependencies_met(self, task: ScheduledTask) -> bool:
		"""Check if task dependencies are met."""
		
		if not task.dependencies:
			return True
		
		# Check if all dependency tasks are completed
		# This would integrate with the actual workflow state
		# For now, assume dependencies are met if not in running or delayed queues
		for dep_id in task.dependencies:
			if any(t.task_id == dep_id for t in self.task_queue + self.delayed_tasks + list(self.running_tasks.values())):
				return False
		
		return True
	
	def _calculate_task_priority(self, task_definition: TaskDefinition) -> int:
		"""Calculate task priority based on various factors."""
		
		base_priority = {
			Priority.CRITICAL: 100,
			Priority.HIGH: 80,
			Priority.MEDIUM: 60,
			Priority.LOW: 40
		}.get(task_definition.priority, 60)
		
		# Adjust based on task type
		type_adjustment = {
			TaskType.APPROVAL: 20,  # Higher priority for approvals
			TaskType.HUMAN: 10,     # Higher priority for human tasks
			TaskType.AUTOMATED: 0,
			TaskType.INTEGRATION: -10  # Lower priority for integrations
		}.get(task_definition.task_type, 0)
		
		# Adjust based on SLA
		sla_adjustment = 0
		if task_definition.sla_hours:
			if task_definition.sla_hours <= 1:
				sla_adjustment = 30
			elif task_definition.sla_hours <= 4:
				sla_adjustment = 20
			elif task_definition.sla_hours <= 24:
				sla_adjustment = 10
		
		return base_priority + type_adjustment + sla_adjustment
	
	def _estimate_resource_requirements(self, task_definition: TaskDefinition) -> Dict[ResourceType, float]:
		"""Estimate resource requirements for a task."""
		
		# Base requirements by task type
		base_requirements = {
			TaskType.AUTOMATED: {ResourceType.CPU: 0.5, ResourceType.MEMORY: 0.5},
			TaskType.HUMAN: {ResourceType.CPU: 0.1, ResourceType.MEMORY: 0.1},
			TaskType.APPROVAL: {ResourceType.CPU: 0.1, ResourceType.MEMORY: 0.1},
			TaskType.INTEGRATION: {ResourceType.CPU: 0.3, ResourceType.MEMORY: 0.3, ResourceType.NETWORK: 0.2}
		}.get(task_definition.task_type, {ResourceType.CPU: 0.5, ResourceType.MEMORY: 0.5})
		
		# Adjust based on task configuration
		if task_definition.configuration:
			if task_definition.configuration.get("high_memory", False):
				base_requirements[ResourceType.MEMORY] *= 2
			if task_definition.configuration.get("cpu_intensive", False):
				base_requirements[ResourceType.CPU] *= 2
			if task_definition.configuration.get("network_intensive", False):
				base_requirements[ResourceType.NETWORK] = base_requirements.get(ResourceType.NETWORK, 0.5) * 2
		
		return base_requirements
	
	async def _resource_monitor_loop(self) -> None:
		"""Monitor resource utilization and worker health."""
		
		while True:
			try:
				# Update worker heartbeats from Redis
				await self._update_worker_heartbeats()
				
				# Check for unhealthy workers
				await self._check_worker_health()
				
				# Update resource utilization metrics
				await self._update_resource_metrics()
				
				await asyncio.sleep(30)  # Check every 30 seconds
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Resource monitor error: {e}")
				await asyncio.sleep(5.0)
	
	async def _cleanup_loop(self) -> None:
		"""Cleanup old data and maintain system health."""
		
		while True:
			try:
				# Clean up old scheduling decisions
				cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
				self.stats["scheduling_decisions"] = [
					decision for decision in self.stats["scheduling_decisions"]
					if datetime.fromisoformat(decision["scheduled_at"].replace("Z", "+00:00")) > cutoff
				]
				
				# Clean up Redis cache
				await self._cleanup_redis_cache()
				
				await asyncio.sleep(3600)  # Cleanup every hour
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Cleanup loop error: {e}")
				await asyncio.sleep(300)
	
	async def _update_worker_heartbeats(self) -> None:
		"""Update worker heartbeats from Redis."""
		
		pattern = f"worker_node:{self.tenant_id}:*"
		keys = await self.redis_client.keys(pattern)
		
		for key in keys:
			try:
				data = await self.redis_client.get(key)
				if data:
					worker_data = json.loads(data)
					node_id = worker_data["node_id"]
					
					if node_id in self.worker_nodes:
						self.worker_nodes[node_id].last_heartbeat = datetime.now(timezone.utc)
			except Exception as e:
				logger.error(f"Error updating worker heartbeat: {e}")
	
	async def _check_worker_health(self) -> None:
		"""Check worker health and mark unhealthy workers as unavailable."""
		
		unhealthy_threshold = timedelta(minutes=5)
		current_time = datetime.now(timezone.utc)
		
		for node_id, worker in self.worker_nodes.items():
			if current_time - worker.last_heartbeat > unhealthy_threshold:
				if worker.is_available:
					worker.is_available = False
					logger.warning(f"Worker {node_id} marked as unhealthy")
					
					# Reschedule tasks from unhealthy worker
					for task_id in list(worker.active_tasks):
						if task_id in self.running_tasks:
							task = self.running_tasks[task_id]
							await self._reschedule_task(task, worker)
			elif not worker.is_available:
				worker.is_available = True
				logger.info(f"Worker {node_id} recovered and marked as healthy")
	
	async def _reschedule_task(self, task: ScheduledTask, failed_worker: WorkerNode) -> None:
		"""Reschedule a task from a failed worker."""
		
		# Remove from failed worker
		failed_worker.active_tasks.discard(task.task_id)
		for resource_type, required in task.resource_requirements.items():
			failed_worker.current_load[resource_type] = max(0.0, failed_worker.current_load.get(resource_type, 0.0) - required)
		
		# Remove from running tasks
		if task.task_id in self.running_tasks:
			del self.running_tasks[task.task_id]
		
		# Add back to queue with higher priority
		task.priority += 10
		task.retry_count += 1
		heapq.heappush(self.task_queue, task)
		
		logger.info(f"Rescheduled task {task.task_id} from failed worker")
	
	async def _update_resource_metrics(self) -> None:
		"""Update resource utilization metrics."""
		
		for resource_type in ResourceType:
			total_utilization = 0.0
			node_count = 0
			
			for worker in self.worker_nodes.values():
				if worker.is_available:
					total_utilization += worker.get_utilization(resource_type)
					node_count += 1
			
			if node_count > 0:
				self.stats["resource_utilization"][resource_type.value] = total_utilization / node_count
	
	async def _update_scheduling_stats(self) -> None:
		"""Update scheduling statistics."""
		
		# Calculate average wait time from recent decisions
		recent_decisions = self.stats["scheduling_decisions"][-100:]  # Last 100 decisions
		if recent_decisions:
			total_wait_time = sum(decision["wait_time"] for decision in recent_decisions)
			self.stats["average_wait_time"] = total_wait_time / len(recent_decisions)
	
	async def _cache_scheduled_task(self, task: ScheduledTask) -> None:
		"""Cache scheduled task in Redis."""
		
		task_data = {
			"task_id": task.task_id,
			"workflow_id": task.workflow_id,
			"instance_id": task.instance_id,
			"priority": task.priority,
			"scheduled_at": task.scheduled_at.isoformat(),
			"estimated_duration": task.estimated_duration,
			"resource_requirements": {k.value: v for k, v in task.resource_requirements.items()},
			"dependencies": task.dependencies,
			"retry_count": task.retry_count,
			"tenant_id": task.tenant_id,
			"user_id": task.user_id
		}
		
		await self.redis_client.setex(
			f"scheduled_task:{self.tenant_id}:{task.task_id}",
			3600,  # 1 hour TTL
			json.dumps(task_data)
		)
	
	async def _cleanup_redis_cache(self) -> None:
		"""Clean up old Redis cache entries."""
		
		# Clean up old scheduled tasks
		pattern = f"scheduled_task:{self.tenant_id}:*"
		keys = await self.redis_client.keys(pattern)
		
		for key in keys:
			ttl = await self.redis_client.ttl(key)
			if ttl == -1:  # No expiration set
				await self.redis_client.expire(key, 3600)
	
	async def _notify_task_scheduled(self, task: ScheduledTask) -> None:
		"""Notify callbacks about task scheduling."""
		
		for callback in self.task_scheduled_callbacks:
			try:
				if asyncio.iscoroutinefunction(callback):
					await callback(task)
				else:
					callback(task)
			except Exception as e:
				logger.error(f"Task scheduled callback error: {e}")
	
	async def _notify_task_completed(self, task: ScheduledTask, success: bool, error: str = None) -> None:
		"""Notify callbacks about task completion."""
		
		for callback in self.task_completed_callbacks:
			try:
				if asyncio.iscoroutinefunction(callback):
					await callback(task, success, error)
				else:
					callback(task, success, error)
			except Exception as e:
				logger.error(f"Task completed callback error: {e}")
	
	def add_task_scheduled_callback(self, callback: Callable) -> None:
		"""Add callback for task scheduled events."""
		self.task_scheduled_callbacks.append(callback)
	
	def add_task_completed_callback(self, callback: Callable) -> None:
		"""Add callback for task completed events."""
		self.task_completed_callbacks.append(callback)

# Export scheduler classes
__all__ = [
	"WorkflowScheduler",
	"ScheduledTask",
	"WorkerNode", 
	"SchedulingStrategy",
	"ResourceType"
]