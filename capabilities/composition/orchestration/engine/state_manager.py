"""
APG Workflow Orchestration State Management System

Persistent workflow state storage with transitions, validation, checkpoints,
recovery mechanisms, state history, and versioning.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict
from uuid_extensions import uuid7str

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func, text
from sqlalchemy.dialects.postgresql import insert

from ..models import (
	Workflow, WorkflowInstance, TaskDefinition, TaskExecution,
	WorkflowStatus, TaskStatus, TaskType, Priority
)
from ..database import (
	CRWorkflow, CRWorkflowInstance, CRTaskExecution,
	DatabaseManager, create_repositories
)

logger = logging.getLogger(__name__)

class StateTransitionType(Enum):
	"""Types of state transitions."""
	WORKFLOW_CREATED = "workflow_created"
	WORKFLOW_STARTED = "workflow_started"
	WORKFLOW_PAUSED = "workflow_paused"
	WORKFLOW_RESUMED = "workflow_resumed"
	WORKFLOW_COMPLETED = "workflow_completed"
	WORKFLOW_FAILED = "workflow_failed"
	WORKFLOW_CANCELLED = "workflow_cancelled"
	TASK_CREATED = "task_created"
	TASK_ASSIGNED = "task_assigned"
	TASK_STARTED = "task_started"
	TASK_COMPLETED = "task_completed"
	TASK_FAILED = "task_failed"
	TASK_RETRIED = "task_retried"
	TASK_TRANSFERRED = "task_transferred"
	TASK_ESCALATED = "task_escalated"
	CHECKPOINT_CREATED = "checkpoint_created"
	STATE_RESTORED = "state_restored"

class CheckpointType(Enum):
	"""Types of state checkpoints."""
	AUTOMATIC = "automatic"  # System-generated checkpoints
	MANUAL = "manual"       # User-triggered checkpoints
	MILESTONE = "milestone"  # Important workflow milestones
	RECOVERY = "recovery"   # Recovery points after failures

@dataclass
class StateTransition:
	"""Represents a state transition event."""
	transition_id: str
	transition_type: StateTransitionType
	entity_type: str  # 'workflow' or 'task'
	entity_id: str
	instance_id: str
	from_state: Optional[str]
	to_state: str
	triggered_by: str
	triggered_at: datetime
	metadata: Dict[str, Any] = field(default_factory=dict)
	tenant_id: str = ""

@dataclass 
class StateCheckpoint:
	"""Represents a state checkpoint for recovery."""
	checkpoint_id: str
	checkpoint_type: CheckpointType
	instance_id: str
	workflow_state: Dict[str, Any]
	task_states: Dict[str, Dict[str, Any]]
	variables: Dict[str, Any]
	created_at: datetime
	created_by: str
	description: str
	metadata: Dict[str, Any] = field(default_factory=dict)
	tenant_id: str = ""
	
	def get_size_bytes(self) -> int:
		"""Get checkpoint size in bytes."""
		return len(json.dumps(asdict(self), default=str).encode('utf-8'))

@dataclass
class StateSnapshot:
	"""Complete state snapshot for a workflow instance."""
	snapshot_id: str
	instance_id: str
	workflow_id: str
	workflow_version: str
	current_status: WorkflowStatus
	current_tasks: List[str]
	completed_tasks: List[str]
	failed_tasks: List[str]
	skipped_tasks: List[str]
	variables: Dict[str, Any]
	context: Dict[str, Any]
	task_executions: Dict[str, Dict[str, Any]]
	execution_history: List[StateTransition]
	checkpoints: List[str]  # List of checkpoint IDs
	created_at: datetime
	tenant_id: str = ""

class StateManager:
	"""Comprehensive workflow state management system."""
	
	def __init__(
		self,
		database_manager: DatabaseManager,
		redis_client: redis.Redis,
		tenant_id: str
	):
		self.database_manager = database_manager
		self.redis_client = redis_client
		self.tenant_id = tenant_id
		
		# State caches
		self.workflow_states: Dict[str, Dict[str, Any]] = {}
		self.task_states: Dict[str, Dict[str, Any]] = {}
		
		# State transition rules
		self.workflow_transitions = self._initialize_workflow_transitions()
		self.task_transitions = self._initialize_task_transitions()
		
		# Background tasks
		self._checkpoint_task: Optional[asyncio.Task] = None
		self._cleanup_task: Optional[asyncio.Task] = None
		self._sync_task: Optional[asyncio.Task] = None
		
		# Configuration
		self.checkpoint_interval_seconds = 300  # 5 minutes
		self.max_checkpoints_per_instance = 50
		self.state_retention_days = 30
		self.enable_automatic_checkpoints = True
		
		logger.info(f"Initialized StateManager for tenant {tenant_id}")
	
	async def start(self) -> None:
		"""Start the state manager and background tasks."""
		
		# Start background tasks
		self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())
		self._cleanup_task = asyncio.create_task(self._cleanup_loop())
		self._sync_task = asyncio.create_task(self._sync_loop())
		
		# Load existing states from database
		await self._load_active_states()
		
		logger.info("StateManager started")
	
	async def stop(self) -> None:
		"""Stop the state manager and cleanup resources."""
		
		# Cancel background tasks
		if self._checkpoint_task:
			self._checkpoint_task.cancel()
		if self._cleanup_task:
			self._cleanup_task.cancel()
		if self._sync_task:
			self._sync_task.cancel()
		
		await asyncio.gather(
			self._checkpoint_task, self._cleanup_task, self._sync_task,
			return_exceptions=True
		)
		
		# Flush state caches to database
		await self._flush_state_caches()
		
		logger.info("StateManager stopped")
	
	async def create_workflow_state(
		self,
		instance_id: str,
		workflow_id: str,
		workflow_version: str,
		initial_variables: Dict[str, Any],
		created_by: str
	) -> None:
		"""Create initial workflow state."""
		
		# Create initial state
		workflow_state = {
			"instance_id": instance_id,
			"workflow_id": workflow_id,
			"workflow_version": workflow_version,
			"status": WorkflowStatus.DRAFT.value,
			"current_tasks": [],
			"completed_tasks": [],
			"failed_tasks": [],
			"skipped_tasks": [],
			"variables": initial_variables.copy(),
			"context": {},
			"created_at": datetime.now(timezone.utc).isoformat(),
			"updated_at": datetime.now(timezone.utc).isoformat(),
			"created_by": created_by,
			"tenant_id": self.tenant_id
		}
		
		# Store in cache
		self.workflow_states[instance_id] = workflow_state
		
		# Store in Redis
		await self.redis_client.setex(
			f"workflow_state:{self.tenant_id}:{instance_id}",
			3600,  # 1 hour TTL
			json.dumps(workflow_state, default=str)
		)
		
		# Record state transition
		await self._record_state_transition(
			StateTransitionType.WORKFLOW_CREATED,
			"workflow",
			instance_id,
			instance_id,
			None,
			WorkflowStatus.DRAFT.value,
			created_by
		)
		
		logger.info(f"Created workflow state for instance {instance_id}")
	
	async def transition_workflow_state(
		self,
		instance_id: str,
		new_status: WorkflowStatus,
		triggered_by: str,
		metadata: Optional[Dict[str, Any]] = None
	) -> bool:
		"""Transition workflow to new state."""
		
		if instance_id not in self.workflow_states:
			await self._load_workflow_state(instance_id)
		
		if instance_id not in self.workflow_states:
			logger.error(f"Workflow state not found: {instance_id}")
			return False
		
		workflow_state = self.workflow_states[instance_id]
		current_status = WorkflowStatus(workflow_state["status"])
		
		# Validate transition
		if not self._is_valid_workflow_transition(current_status, new_status):
			logger.error(f"Invalid workflow transition: {current_status} -> {new_status}")
			return False
		
		# Update state
		old_status = workflow_state["status"]
		workflow_state["status"] = new_status.value
		workflow_state["updated_at"] = datetime.now(timezone.utc).isoformat()
		
		if metadata:
			workflow_state["context"].update(metadata)
		
		# Update Redis
		await self.redis_client.setex(
			f"workflow_state:{self.tenant_id}:{instance_id}",
			3600,
			json.dumps(workflow_state, default=str)
		)
		
		# Record state transition
		transition_type = self._get_workflow_transition_type(new_status)
		await self._record_state_transition(
			transition_type,
			"workflow",
			instance_id,
			instance_id,
			old_status,
			new_status.value,
			triggered_by,
			metadata
		)
		
		# Create checkpoint for important transitions
		if new_status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
			await self.create_checkpoint(instance_id, CheckpointType.MILESTONE, triggered_by, 
										f"Workflow {new_status.value}")
		
		logger.info(f"Transitioned workflow {instance_id}: {old_status} -> {new_status.value}")
		return True
	
	async def update_task_state(
		self,
		instance_id: str,
		task_id: str,
		new_status: TaskStatus,
		triggered_by: str,
		task_data: Optional[Dict[str, Any]] = None,
		metadata: Optional[Dict[str, Any]] = None
	) -> bool:
		"""Update task state and workflow accordingly."""
		
		# Load workflow state if not cached
		if instance_id not in self.workflow_states:
			await self._load_workflow_state(instance_id)
		
		if instance_id not in self.workflow_states:
			logger.error(f"Workflow state not found: {instance_id}")
			return False
		
		workflow_state = self.workflow_states[instance_id]
		task_key = f"{instance_id}:{task_id}"
		
		# Get or create task state
		if task_key not in self.task_states:
			self.task_states[task_key] = {
				"task_id": task_id,
				"instance_id": instance_id,
				"status": TaskStatus.PENDING.value,
				"created_at": datetime.now(timezone.utc).isoformat(),
				"updated_at": datetime.now(timezone.utc).isoformat(),
				"tenant_id": self.tenant_id
			}
		
		task_state = self.task_states[task_key]
		old_status = task_state["status"]
		
		# Validate transition
		if not self._is_valid_task_transition(TaskStatus(old_status), new_status):
			logger.error(f"Invalid task transition: {old_status} -> {new_status}")
			return False
		
		# Update task state
		task_state["status"] = new_status.value
		task_state["updated_at"] = datetime.now(timezone.utc).isoformat()
		
		if task_data:
			task_state.update(task_data)
		
		# Update workflow task lists
		self._update_workflow_task_lists(workflow_state, task_id, TaskStatus(old_status), new_status)
		
		# Update Redis
		await self.redis_client.setex(
			f"task_state:{self.tenant_id}:{task_key}",
			3600,
			json.dumps(task_state, default=str)
		)
		
		# Update workflow state
		workflow_state["updated_at"] = datetime.now(timezone.utc).isoformat()
		await self.redis_client.setex(
			f"workflow_state:{self.tenant_id}:{instance_id}",
			3600,
			json.dumps(workflow_state, default=str)
		)
		
		# Record state transition
		transition_type = self._get_task_transition_type(new_status)
		await self._record_state_transition(
			transition_type,
			"task",
			task_id,
			instance_id,
			old_status,
			new_status.value,
			triggered_by,
			metadata
		)
		
		logger.debug(f"Updated task state {task_id}: {old_status} -> {new_status.value}")
		return True
	
	async def create_checkpoint(
		self,
		instance_id: str,
		checkpoint_type: CheckpointType,
		created_by: str,
		description: str,
		metadata: Optional[Dict[str, Any]] = None
	) -> str:
		"""Create a state checkpoint for recovery."""
		
		# Load current state
		if instance_id not in self.workflow_states:
			await self._load_workflow_state(instance_id)
		
		if instance_id not in self.workflow_states:
			raise ValueError(f"Workflow state not found: {instance_id}")
		
		workflow_state = self.workflow_states[instance_id].copy()
		
		# Collect task states for this instance
		task_states = {}
		for task_key, task_state in self.task_states.items():
			if task_state["instance_id"] == instance_id:
				task_states[task_state["task_id"]] = task_state.copy()
		
		# Create checkpoint
		checkpoint = StateCheckpoint(
			checkpoint_id=uuid7str(),
			checkpoint_type=checkpoint_type,
			instance_id=instance_id,
			workflow_state=workflow_state,
			task_states=task_states,
			variables=workflow_state.get("variables", {}),
			created_at=datetime.now(timezone.utc),
			created_by=created_by,
			description=description,
			metadata=metadata or {},
			tenant_id=self.tenant_id
		)
		
		# Store checkpoint in database
		await self._store_checkpoint(checkpoint)
		
		# Store checkpoint reference in Redis
		await self.redis_client.lpush(
			f"checkpoints:{self.tenant_id}:{instance_id}",
			checkpoint.checkpoint_id
		)
		
		# Limit number of checkpoints
		await self.redis_client.ltrim(
			f"checkpoints:{self.tenant_id}:{instance_id}",
			0, self.max_checkpoints_per_instance - 1
		)
		
		# Record transition
		await self._record_state_transition(
			StateTransitionType.CHECKPOINT_CREATED,
			"workflow",
			instance_id,
			instance_id,
			workflow_state["status"],
			workflow_state["status"],
			created_by,
			{"checkpoint_id": checkpoint.checkpoint_id, "checkpoint_type": checkpoint_type.value}
		)
		
		logger.info(f"Created checkpoint {checkpoint.checkpoint_id} for instance {instance_id}")
		return checkpoint.checkpoint_id
	
	async def restore_from_checkpoint(
		self,
		instance_id: str,
		checkpoint_id: str,
		restored_by: str
	) -> bool:
		"""Restore workflow state from checkpoint."""
		
		# Load checkpoint
		checkpoint = await self._load_checkpoint(checkpoint_id)
		if not checkpoint:
			logger.error(f"Checkpoint not found: {checkpoint_id}")
			return False
		
		if checkpoint.instance_id != instance_id:
			logger.error(f"Checkpoint instance mismatch: {checkpoint.instance_id} != {instance_id}")
			return False
		
		# Restore workflow state
		self.workflow_states[instance_id] = checkpoint.workflow_state.copy()
		self.workflow_states[instance_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
		
		# Restore task states
		for task_id, task_state in checkpoint.task_states.items():
			task_key = f"{instance_id}:{task_id}"
			self.task_states[task_key] = task_state.copy()
			self.task_states[task_key]["updated_at"] = datetime.now(timezone.utc).isoformat()
		
		# Update Redis
		await self.redis_client.setex(
			f"workflow_state:{self.tenant_id}:{instance_id}",
			3600,
			json.dumps(self.workflow_states[instance_id], default=str)
		)
		
		for task_key, task_state in self.task_states.items():
			if task_state["instance_id"] == instance_id:
				await self.redis_client.setex(
					f"task_state:{self.tenant_id}:{task_key}",
					3600,
					json.dumps(task_state, default=str)
				)
		
		# Record transition
		await self._record_state_transition(
			StateTransitionType.STATE_RESTORED,
			"workflow",
			instance_id,
			instance_id,
			checkpoint.workflow_state["status"],
			checkpoint.workflow_state["status"],
			restored_by,
			{"checkpoint_id": checkpoint_id, "restored_from": checkpoint.created_at.isoformat()}
		)
		
		logger.info(f"Restored instance {instance_id} from checkpoint {checkpoint_id}")
		return True
	
	async def get_state_snapshot(self, instance_id: str) -> Optional[StateSnapshot]:
		"""Get complete state snapshot for an instance."""
		
		# Load workflow state
		if instance_id not in self.workflow_states:
			await self._load_workflow_state(instance_id)
		
		if instance_id not in self.workflow_states:
			return None
		
		workflow_state = self.workflow_states[instance_id]
		
		# Collect task executions
		task_executions = {}
		for task_key, task_state in self.task_states.items():
			if task_state["instance_id"] == instance_id:
				task_executions[task_state["task_id"]] = task_state
		
		# Get execution history
		execution_history = await self._get_execution_history(instance_id)
		
		# Get checkpoints
		checkpoint_ids = await self.redis_client.lrange(
			f"checkpoints:{self.tenant_id}:{instance_id}", 0, -1
		)
		
		# Create snapshot
		snapshot = StateSnapshot(
			snapshot_id=uuid7str(),
			instance_id=instance_id,
			workflow_id=workflow_state["workflow_id"],
			workflow_version=workflow_state["workflow_version"],
			current_status=WorkflowStatus(workflow_state["status"]),
			current_tasks=workflow_state.get("current_tasks", []),
			completed_tasks=workflow_state.get("completed_tasks", []),
			failed_tasks=workflow_state.get("failed_tasks", []),
			skipped_tasks=workflow_state.get("skipped_tasks", []),
			variables=workflow_state.get("variables", {}),
			context=workflow_state.get("context", {}),
			task_executions=task_executions,
			execution_history=execution_history,
			checkpoints=[cp.decode() if isinstance(cp, bytes) else cp for cp in checkpoint_ids],
			created_at=datetime.now(timezone.utc),
			tenant_id=self.tenant_id
		)
		
		return snapshot
	
	async def get_state_history(
		self,
		instance_id: str,
		limit: int = 100
	) -> List[StateTransition]:
		"""Get state transition history for an instance."""
		
		return await self._get_execution_history(instance_id, limit)
	
	async def validate_state_consistency(self, instance_id: str) -> Dict[str, Any]:
		"""Validate state consistency for an instance."""
		
		issues = []
		warnings = []
		
		# Load states
		if instance_id not in self.workflow_states:
			await self._load_workflow_state(instance_id)
		
		if instance_id not in self.workflow_states:
			issues.append("Workflow state not found")
			return {"valid": False, "issues": issues, "warnings": warnings}
		
		workflow_state = self.workflow_states[instance_id]
		
		# Check task state consistency
		current_tasks = set(workflow_state.get("current_tasks", []))
		completed_tasks = set(workflow_state.get("completed_tasks", []))
		failed_tasks = set(workflow_state.get("failed_tasks", []))
		skipped_tasks = set(workflow_state.get("skipped_tasks", []))
		
		# Check for overlapping task lists
		all_task_sets = [current_tasks, completed_tasks, failed_tasks, skipped_tasks]
		for i, set1 in enumerate(all_task_sets):
			for j, set2 in enumerate(all_task_sets[i+1:], i+1):
				overlap = set1 & set2
				if overlap:
					issues.append(f"Tasks appear in multiple states: {overlap}")
		
		# Check task states match workflow lists
		for task_key, task_state in self.task_states.items():
			if task_state["instance_id"] != instance_id:
				continue
			
			task_id = task_state["task_id"]
			task_status = TaskStatus(task_state["status"])
			
			if task_status == TaskStatus.IN_PROGRESS and task_id not in current_tasks:
				issues.append(f"Task {task_id} is in progress but not in current_tasks")
			elif task_status == TaskStatus.COMPLETED and task_id not in completed_tasks:
				issues.append(f"Task {task_id} is completed but not in completed_tasks")
			elif task_status == TaskStatus.FAILED and task_id not in failed_tasks:
				issues.append(f"Task {task_id} is failed but not in failed_tasks")
		
		# Check workflow status consistency
		workflow_status = WorkflowStatus(workflow_state["status"])
		if workflow_status == WorkflowStatus.COMPLETED and current_tasks:
			issues.append("Workflow is completed but has current tasks")
		elif workflow_status == WorkflowStatus.ACTIVE and not current_tasks and not completed_tasks:
			warnings.append("Active workflow has no current or completed tasks")
		
		return {
			"valid": len(issues) == 0,
			"issues": issues,
			"warnings": warnings,
			"checked_at": datetime.now(timezone.utc).isoformat()
		}
	
	def _initialize_workflow_transitions(self) -> Dict[WorkflowStatus, Set[WorkflowStatus]]:
		"""Initialize valid workflow state transitions."""
		
		return {
			WorkflowStatus.DRAFT: {WorkflowStatus.ACTIVE, WorkflowStatus.CANCELLED},
			WorkflowStatus.ACTIVE: {
				WorkflowStatus.PAUSED, WorkflowStatus.COMPLETED, 
				WorkflowStatus.FAILED, WorkflowStatus.CANCELLED
			},
			WorkflowStatus.PAUSED: {WorkflowStatus.ACTIVE, WorkflowStatus.CANCELLED},
			WorkflowStatus.COMPLETED: set(),  # Terminal state
			WorkflowStatus.FAILED: {WorkflowStatus.ACTIVE},  # Can retry
			WorkflowStatus.CANCELLED: set()  # Terminal state
		}
	
	def _initialize_task_transitions(self) -> Dict[TaskStatus, Set[TaskStatus]]:
		"""Initialize valid task state transitions."""
		
		return {
			TaskStatus.PENDING: {TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS, TaskStatus.SKIPPED},
			TaskStatus.ASSIGNED: {TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED},
			TaskStatus.IN_PROGRESS: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.PAUSED},
			TaskStatus.COMPLETED: set(),  # Terminal state
			TaskStatus.FAILED: {TaskStatus.RETRY, TaskStatus.SKIPPED},
			TaskStatus.RETRY: {TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS, TaskStatus.FAILED},
			TaskStatus.SKIPPED: set(),  # Terminal state
			TaskStatus.PAUSED: {TaskStatus.IN_PROGRESS, TaskStatus.FAILED, TaskStatus.CANCELLED},
			TaskStatus.CANCELLED: set()  # Terminal state
		}
	
	def _is_valid_workflow_transition(self, from_status: WorkflowStatus, to_status: WorkflowStatus) -> bool:
		"""Check if workflow state transition is valid.""" 
		
		return to_status in self.workflow_transitions.get(from_status, set())
	
	def _is_valid_task_transition(self, from_status: TaskStatus, to_status: TaskStatus) -> bool:
		"""Check if task state transition is valid."""
		
		return to_status in self.task_transitions.get(from_status, set())
	
	def _update_workflow_task_lists(
		self,
		workflow_state: Dict[str, Any],
		task_id: str,
		old_status: TaskStatus,
		new_status: TaskStatus
	) -> None:
		"""Update workflow task lists based on task status change."""
		
		# Remove task from old list
		if old_status == TaskStatus.IN_PROGRESS and task_id in workflow_state.get("current_tasks", []):
			workflow_state["current_tasks"].remove(task_id)
		elif old_status == TaskStatus.COMPLETED and task_id in workflow_state.get("completed_tasks", []):
			workflow_state["completed_tasks"].remove(task_id)
		elif old_status == TaskStatus.FAILED and task_id in workflow_state.get("failed_tasks", []):
			workflow_state["failed_tasks"].remove(task_id)
		elif old_status == TaskStatus.SKIPPED and task_id in workflow_state.get("skipped_tasks", []):
			workflow_state["skipped_tasks"].remove(task_id)
		
		# Add task to new list
		if new_status == TaskStatus.IN_PROGRESS:
			if "current_tasks" not in workflow_state:
				workflow_state["current_tasks"] = []
			if task_id not in workflow_state["current_tasks"]:
				workflow_state["current_tasks"].append(task_id)
		elif new_status == TaskStatus.COMPLETED:
			if "completed_tasks" not in workflow_state:
				workflow_state["completed_tasks"] = []
			if task_id not in workflow_state["completed_tasks"]:
				workflow_state["completed_tasks"].append(task_id)
		elif new_status == TaskStatus.FAILED:
			if "failed_tasks" not in workflow_state:
				workflow_state["failed_tasks"] = []
			if task_id not in workflow_state["failed_tasks"]:
				workflow_state["failed_tasks"].append(task_id)
		elif new_status == TaskStatus.SKIPPED:
			if "skipped_tasks" not in workflow_state:
				workflow_state["skipped_tasks"] = []
			if task_id not in workflow_state["skipped_tasks"]:
				workflow_state["skipped_tasks"].append(task_id)
	
	def _get_workflow_transition_type(self, status: WorkflowStatus) -> StateTransitionType:
		"""Get transition type for workflow status."""
		
		mapping = {
			WorkflowStatus.ACTIVE: StateTransitionType.WORKFLOW_STARTED,
			WorkflowStatus.PAUSED: StateTransitionType.WORKFLOW_PAUSED,
			WorkflowStatus.COMPLETED: StateTransitionType.WORKFLOW_COMPLETED,
			WorkflowStatus.FAILED: StateTransitionType.WORKFLOW_FAILED,
			WorkflowStatus.CANCELLED: StateTransitionType.WORKFLOW_CANCELLED
		}
		
		return mapping.get(status, StateTransitionType.WORKFLOW_STARTED)
	
	def _get_task_transition_type(self, status: TaskStatus) -> StateTransitionType:
		"""Get transition type for task status."""
		
		mapping = {
			TaskStatus.ASSIGNED: StateTransitionType.TASK_ASSIGNED,
			TaskStatus.IN_PROGRESS: StateTransitionType.TASK_STARTED,
			TaskStatus.COMPLETED: StateTransitionType.TASK_COMPLETED,
			TaskStatus.FAILED: StateTransitionType.TASK_FAILED,
			TaskStatus.RETRY: StateTransitionType.TASK_RETRIED
		}
		
		return mapping.get(status, StateTransitionType.TASK_CREATED)
	
	async def _record_state_transition(
		self,
		transition_type: StateTransitionType,
		entity_type: str,
		entity_id: str,
		instance_id: str,
		from_state: Optional[str],
		to_state: str,
		triggered_by: str,
		metadata: Optional[Dict[str, Any]] = None
	) -> None:
		"""Record a state transition."""
		
		transition = StateTransition(
			transition_id=uuid7str(),
			transition_type=transition_type,
			entity_type=entity_type,
			entity_id=entity_id,
			instance_id=instance_id,
			from_state=from_state,
			to_state=to_state,
			triggered_by=triggered_by,
			triggered_at=datetime.now(timezone.utc),
			metadata=metadata or {},
			tenant_id=self.tenant_id
		)
		
		# Store in Redis for recent access
		await self.redis_client.lpush(
			f"transitions:{self.tenant_id}:{instance_id}",
			json.dumps(asdict(transition), default=str)
		)
		
		# Keep only recent transitions in Redis
		await self.redis_client.ltrim(f"transitions:{self.tenant_id}:{instance_id}", 0, 999)
		
		# Store in database for persistence
		await self._store_transition(transition)
	
	async def _load_active_states(self) -> None:
		"""Load active workflow states from database."""
		
		try:
			async with self.database_manager.get_session() as session:
				# Load active workflow instances
				result = await session.execute(
					select(CRWorkflowInstance).where(
						and_(
							CRWorkflowInstance.tenant_id == self.tenant_id,
							CRWorkflowInstance.status.in_(['active', 'paused'])
						)
					)
				)
				
				for instance in result.scalars().all():
					workflow_state = {
						"instance_id": instance.instance_id,
						"workflow_id": instance.workflow_id,
						"workflow_version": "1.0.0",  # Default version
						"status": instance.status,
						"current_tasks": instance.current_tasks or [],
						"completed_tasks": instance.completed_tasks or [],
						"failed_tasks": instance.failed_tasks or [],
						"skipped_tasks": instance.skipped_tasks or [],
						"variables": instance.variables or {},
						"context": instance.context or {},
						"created_at": instance.started_at.isoformat(),
						"updated_at": datetime.now(timezone.utc).isoformat(),
						"created_by": instance.started_by,
						"tenant_id": self.tenant_id
					}
					
					self.workflow_states[instance.instance_id] = workflow_state
				
				logger.info(f"Loaded {len(self.workflow_states)} active workflow states")
		
		except Exception as e:
			logger.error(f"Error loading active states: {e}")
	
	async def _load_workflow_state(self, instance_id: str) -> bool:
		"""Load workflow state from Redis or database."""
		
		# Try Redis first
		try:
			data = await self.redis_client.get(f"workflow_state:{self.tenant_id}:{instance_id}")
			if data:
				self.workflow_states[instance_id] = json.loads(data)
				return True
		except Exception as e:
			logger.error(f"Error loading from Redis: {e}")
		
		# Try database
		try:
			async with self.database_manager.get_session() as session:
				result = await session.execute(
					select(CRWorkflowInstance).where(
						and_(
							CRWorkflowInstance.instance_id == instance_id,
							CRWorkflowInstance.tenant_id == self.tenant_id
						)
					)
				)
				
				instance = result.scalar_one_or_none()
				if instance:
					workflow_state = {
						"instance_id": instance.instance_id,
						"workflow_id": instance.workflow_id,
						"workflow_version": "1.0.0",
						"status": instance.status,
						"current_tasks": instance.current_tasks or [],
						"completed_tasks": instance.completed_tasks or [],
						"failed_tasks": instance.failed_tasks or [],
						"skipped_tasks": instance.skipped_tasks or [],
						"variables": instance.variables or {},
						"context": instance.context or {},
						"created_at": instance.started_at.isoformat(),
						"updated_at": datetime.now(timezone.utc).isoformat(),
						"created_by": instance.started_by,
						"tenant_id": self.tenant_id
					}
					
					self.workflow_states[instance_id] = workflow_state
					return True
		
		except Exception as e:
			logger.error(f"Error loading from database: {e}")
		
		return False
	
	async def _store_checkpoint(self, checkpoint: StateCheckpoint) -> None:
		"""Store checkpoint in database."""
		
		try:
			# Store as JSON in database (in production, might use dedicated checkpoint table)
			checkpoint_data = json.dumps(asdict(checkpoint), default=str)
			
			# Store in Redis with expiration
			await self.redis_client.setex(
				f"checkpoint:{self.tenant_id}:{checkpoint.checkpoint_id}",
				86400 * self.state_retention_days,  # TTL based on retention
				checkpoint_data
			)
			
		except Exception as e:
			logger.error(f"Error storing checkpoint: {e}")
	
	async def _load_checkpoint(self, checkpoint_id: str) -> Optional[StateCheckpoint]:
		"""Load checkpoint from storage."""
		
		try:
			data = await self.redis_client.get(f"checkpoint:{self.tenant_id}:{checkpoint_id}")
			if data:
				checkpoint_dict = json.loads(data)
				# Convert datetime strings back to datetime objects
				checkpoint_dict["created_at"] = datetime.fromisoformat(
					checkpoint_dict["created_at"].replace("Z", "+00:00")
				)
				checkpoint_dict["checkpoint_type"] = CheckpointType(checkpoint_dict["checkpoint_type"])
				
				return StateCheckpoint(**checkpoint_dict)
		
		except Exception as e:
			logger.error(f"Error loading checkpoint: {e}")
		
		return None
	
	async def _store_transition(self, transition: StateTransition) -> None:
		"""Store state transition in database."""
		
		try:
			# Store in Redis for persistence (in production, might use dedicated transitions table)
			transition_data = json.dumps(asdict(transition), default=str)
			
			await self.redis_client.setex(
				f"transition:{self.tenant_id}:{transition.transition_id}",
				86400 * self.state_retention_days,
				transition_data
			)
			
		except Exception as e:
			logger.error(f"Error storing transition: {e}")
	
	async def _get_execution_history(
		self,
		instance_id: str,
		limit: int = 100
	) -> List[StateTransition]:
		"""Get execution history for an instance."""
		
		try:
			# Get from Redis
			transitions_data = await self.redis_client.lrange(
				f"transitions:{self.tenant_id}:{instance_id}", 0, limit - 1
			)
			
			transitions = []
			for data in transitions_data:
				try:
					transition_dict = json.loads(data)
					# Convert datetime string back to datetime object
					transition_dict["triggered_at"] = datetime.fromisoformat(
						transition_dict["triggered_at"].replace("Z", "+00:00")
					)
					transition_dict["transition_type"] = StateTransitionType(transition_dict["transition_type"])
					
					transitions.append(StateTransition(**transition_dict))
				except Exception as e:
					logger.error(f"Error parsing transition: {e}")
					continue
			
			return transitions
		
		except Exception as e:
			logger.error(f"Error getting execution history: {e}")
			return []
	
	async def _checkpoint_loop(self) -> None:
		"""Background task for automatic checkpointing."""
		
		while True:
			try:
				if self.enable_automatic_checkpoints:
					# Create checkpoints for active workflows
					for instance_id, workflow_state in self.workflow_states.items():
						if workflow_state["status"] == WorkflowStatus.ACTIVE.value:
							# Check if enough time has passed since last checkpoint
							last_update = datetime.fromisoformat(
								workflow_state["updated_at"].replace("Z", "+00:00")
							)
							
							if (datetime.now(timezone.utc) - last_update).seconds > self.checkpoint_interval_seconds:
								await self.create_checkpoint(
									instance_id,
									CheckpointType.AUTOMATIC,
									"system",
									"Automatic checkpoint"
								)
				
				await asyncio.sleep(self.checkpoint_interval_seconds)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Checkpoint loop error: {e}")
				await asyncio.sleep(60)
	
	async def _cleanup_loop(self) -> None:
		"""Background task for cleaning up old data."""
		
		while True:
			try:
				cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.state_retention_days)
				
				# Clean up old checkpoints
				for instance_id in self.workflow_states.keys():
					checkpoint_ids = await self.redis_client.lrange(
						f"checkpoints:{self.tenant_id}:{instance_id}", 0, -1
					)
					
					for checkpoint_id in checkpoint_ids:
						checkpoint_id = checkpoint_id.decode() if isinstance(checkpoint_id, bytes) else checkpoint_id
						checkpoint = await self._load_checkpoint(checkpoint_id)
						
						if checkpoint and checkpoint.created_at < cutoff_date:
							await self.redis_client.delete(f"checkpoint:{self.tenant_id}:{checkpoint_id}")
							await self.redis_client.lrem(
								f"checkpoints:{self.tenant_id}:{instance_id}", 0, checkpoint_id
							)
				
				await asyncio.sleep(3600)  # Run cleanup every hour
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Cleanup loop error: {e}")
				await asyncio.sleep(300)
	
	async def _sync_loop(self) -> None:
		"""Background task for syncing state to database."""
		
		while True:
			try:
				# Periodically sync state caches to database
				await self._flush_state_caches()
				
				await asyncio.sleep(300)  # Sync every 5 minutes
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Sync loop error: {e}")
				await asyncio.sleep(60)
	
	async def _flush_state_caches(self) -> None:
		"""Flush state caches to database."""
		
		try:
			async with self.database_manager.get_session() as session:
				# Update workflow instances
				for instance_id, workflow_state in self.workflow_states.items():
					await session.execute(
						update(CRWorkflowInstance)
						.where(CRWorkflowInstance.instance_id == instance_id)
						.values(
							status=workflow_state["status"],
							current_tasks=workflow_state.get("current_tasks", []),
							completed_tasks=workflow_state.get("completed_tasks", []),
							failed_tasks=workflow_state.get("failed_tasks", []),
							variables=workflow_state.get("variables", {}),
							context=workflow_state.get("context", {})
						)
					)
				
				await session.commit()
				
		except Exception as e:
			logger.error(f"Error flushing state caches: {e}")

# Export state management classes
__all__ = [
	"StateManager",
	"StateTransition",
	"StateCheckpoint",
	"StateSnapshot",
	"StateTransitionType",
	"CheckpointType"
]