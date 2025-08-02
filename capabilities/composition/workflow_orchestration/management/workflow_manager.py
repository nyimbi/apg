"""
APG Workflow Orchestration Workflow Manager

Comprehensive workflow CRUD operations, lifecycle management, validation,
and business logic layer for workflow orchestration platform.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import json

from pydantic import BaseModel, Field, ConfigDict, validator
from uuid_extensions import uuid7str
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func

from ..models import (
	Workflow, WorkflowInstance, TaskDefinition, TaskExecution,
	WorkflowStatus, TaskStatus, Priority, TaskType
)
from ..database import (
	CRWorkflow, CRWorkflowInstance, CRTaskExecution,
	DatabaseManager, create_repositories
)

logger = logging.getLogger(__name__)

class WorkflowOperationType(Enum):
	"""Types of workflow operations."""
	CREATE = "create"
	READ = "read"
	UPDATE = "update"
	DELETE = "delete"
	EXECUTE = "execute"
	PAUSE = "pause"
	RESUME = "resume"
	STOP = "stop"
	CLONE = "clone"
	EXPORT = "export"
	IMPORT = "import"

class WorkflowValidationLevel(Enum):
	"""Levels of workflow validation."""
	BASIC = "basic"			# Basic syntax and structure
	STANDARD = "standard"	# Standard business rules
	STRICT = "strict"		# Strict validation with dependencies
	ENTERPRISE = "enterprise" # Full enterprise validation

class WorkflowSearchFilter(BaseModel):
	"""Filters for workflow search operations."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	name_pattern: Optional[str] = Field(default=None, description="Name pattern (supports wildcards)")
	status: Optional[List[WorkflowStatus]] = Field(default=None)
	tags: Optional[List[str]] = Field(default=None)
	created_after: Optional[datetime] = Field(default=None)
	created_before: Optional[datetime] = Field(default=None)
	updated_after: Optional[datetime] = Field(default=None)
	updated_before: Optional[datetime] = Field(default=None)
	author: Optional[str] = Field(default=None)
	priority: Optional[List[Priority]] = Field(default=None)
	category: Optional[str] = Field(default=None)
	limit: int = Field(default=100, ge=1, le=1000)
	offset: int = Field(default=0, ge=0)
	sort_by: str = Field(default="updated_at", regex="^(name|created_at|updated_at|status|priority)$")
	sort_order: str = Field(default="desc", regex="^(asc|desc)$")

class WorkflowValidationResult(BaseModel):
	"""Result of workflow validation."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	is_valid: bool
	errors: List[str] = Field(default_factory=list)
	warnings: List[str] = Field(default_factory=list)
	validation_level: WorkflowValidationLevel
	score: int = Field(default=0, ge=0, le=100)
	details: Dict[str, Any] = Field(default_factory=dict)
	validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class WorkflowStatistics(BaseModel):
	"""Workflow execution statistics."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	total_executions: int = 0
	successful_executions: int = 0
	failed_executions: int = 0
	average_duration_seconds: float = 0.0
	success_rate: float = 0.0
	last_execution: Optional[datetime] = Field(default=None)
	fastest_execution_seconds: Optional[float] = Field(default=None)
	slowest_execution_seconds: Optional[float] = Field(default=None)
	active_instances: int = 0
	total_tasks: int = 0
	completed_tasks: int = 0
	failed_tasks: int = 0

class WorkflowOperations:
	"""Core workflow operations interface."""
	
	def __init__(
		self,
		database_manager: DatabaseManager,
		redis_client: redis.Redis,
		tenant_id: str
	):
		self.database_manager = database_manager
		self.redis_client = redis_client
		self.tenant_id = tenant_id
		self.operation_history: List[Dict[str, Any]] = []
		
		logger.info(f"Initialized WorkflowOperations for tenant {tenant_id}")
	
	async def create_workflow(
		self,
		workflow_data: Dict[str, Any],
		user_id: str,
		validation_level: WorkflowValidationLevel = WorkflowValidationLevel.STANDARD
	) -> Workflow:
		"""Create a new workflow with validation."""
		
		try:
			# Create Pydantic model from data
			workflow_data.update({
				"tenant_id": self.tenant_id,
				"created_by": user_id,
				"updated_by": user_id
			})
			
			workflow = Workflow(**workflow_data)
			
			# Validate workflow
			validation_result = await self._validate_workflow(workflow, validation_level)
			if not validation_result.is_valid:
				raise ValueError(f"Workflow validation failed: {validation_result.errors}")
			
			# Save to database
			async with self.database_manager.get_session() as session:
				db_workflow = CRWorkflow(
					id=workflow.id,
					name=workflow.name,
					description=workflow.description,
					definition=workflow.model_dump(),
					status=workflow.status.value,
					priority=workflow.priority.value,
					tenant_id=workflow.tenant_id,
					created_by=workflow.created_by,
					updated_by=workflow.updated_by,
					created_at=workflow.created_at,
					updated_at=workflow.updated_at,
					tags=workflow.tags,
					configuration=workflow.configuration or {},
					metadata=workflow.metadata or {}
				)
				
				session.add(db_workflow)
				await session.commit()
				await session.refresh(db_workflow)
			
			# Cache workflow
			await self._cache_workflow(workflow)
			
			# Log operation
			await self._log_operation(WorkflowOperationType.CREATE, workflow.id, user_id, {
				"workflow_name": workflow.name,
				"validation_score": validation_result.score
			})
			
			logger.info(f"Created workflow: {workflow.name} ({workflow.id})")
			return workflow
			
		except Exception as e:
			logger.error(f"Failed to create workflow: {e}")
			raise
	
	async def get_workflow(
		self,
		workflow_id: str,
		user_id: str,
		include_instances: bool = False
	) -> Optional[Workflow]:
		"""Get workflow by ID."""
		
		try:
			# Try cache first
			cached_workflow = await self._get_cached_workflow(workflow_id)
			if cached_workflow:
				await self._log_operation(WorkflowOperationType.READ, workflow_id, user_id, {
					"source": "cache"
				})
				return cached_workflow
			
			# Query database
			async with self.database_manager.get_session() as session:
				result = await session.execute(
					select(CRWorkflow).where(
						and_(
							CRWorkflow.id == workflow_id,
							CRWorkflow.tenant_id == self.tenant_id,
							CRWorkflow.deleted_at.is_(None)
						)
					)
				)
				db_workflow = result.scalar_one_or_none()
				
				if not db_workflow:
					return None
				
				# Convert to Pydantic model
				workflow_data = db_workflow.definition
				workflow = Workflow(**workflow_data)
				
				# Cache workflow
				await self._cache_workflow(workflow)
				
				# Log operation
				await self._log_operation(WorkflowOperationType.READ, workflow_id, user_id, {
					"source": "database",
					"include_instances": include_instances
				})
				
				return workflow
				
		except Exception as e:
			logger.error(f"Failed to get workflow {workflow_id}: {e}")
			raise
	
	async def update_workflow(
		self,
		workflow_id: str,
		updates: Dict[str, Any],
		user_id: str,
		validation_level: WorkflowValidationLevel = WorkflowValidationLevel.STANDARD
	) -> Workflow:
		"""Update existing workflow."""
		
		try:
			# Get current workflow
			current_workflow = await self.get_workflow(workflow_id, user_id)
			if not current_workflow:
				raise ValueError(f"Workflow not found: {workflow_id}")
			
			# Create updated workflow
			workflow_data = current_workflow.model_dump()
			workflow_data.update(updates)
			workflow_data["updated_by"] = user_id
			workflow_data["updated_at"] = datetime.now(timezone.utc)
			
			updated_workflow = Workflow(**workflow_data)
			
			# Validate updated workflow
			validation_result = await self._validate_workflow(updated_workflow, validation_level)
			if not validation_result.is_valid:
				raise ValueError(f"Workflow validation failed: {validation_result.errors}")
			
			# Update in database
			async with self.database_manager.get_session() as session:
				await session.execute(
					update(CRWorkflow)
					.where(
						and_(
							CRWorkflow.id == workflow_id,
							CRWorkflow.tenant_id == self.tenant_id
						)
					)
					.values(
						name=updated_workflow.name,
						description=updated_workflow.description,
						definition=updated_workflow.model_dump(),
						status=updated_workflow.status.value,
						priority=updated_workflow.priority.value,
						updated_by=updated_workflow.updated_by,
						updated_at=updated_workflow.updated_at,
						tags=updated_workflow.tags,
						configuration=updated_workflow.configuration or {},
						metadata=updated_workflow.metadata or {}
					)
				)
				await session.commit()
			
			# Update cache
			await self._cache_workflow(updated_workflow)
			
			# Log operation
			await self._log_operation(WorkflowOperationType.UPDATE, workflow_id, user_id, {
				"updates": list(updates.keys()),
				"validation_score": validation_result.score
			})
			
			logger.info(f"Updated workflow: {workflow_id}")
			return updated_workflow
			
		except Exception as e:
			logger.error(f"Failed to update workflow {workflow_id}: {e}")
			raise
	
	async def delete_workflow(
		self,
		workflow_id: str,
		user_id: str,
		hard_delete: bool = False
	) -> bool:
		"""Delete workflow (soft delete by default)."""
		
		try:
			# Check if workflow exists
			workflow = await self.get_workflow(workflow_id, user_id)
			if not workflow:
				raise ValueError(f"Workflow not found: {workflow_id}")
			
			# Check for active instances
			active_instances = await self._count_active_instances(workflow_id)
			if active_instances > 0:
				raise ValueError(f"Cannot delete workflow with {active_instances} active instances")
			
			async with self.database_manager.get_session() as session:
				if hard_delete:
					# Hard delete - remove from database
					await session.execute(
						delete(CRWorkflow).where(
							and_(
								CRWorkflow.id == workflow_id,
								CRWorkflow.tenant_id == self.tenant_id
							)
						)
					)
				else:
					# Soft delete - mark as deleted
					await session.execute(
						update(CRWorkflow)
						.where(
							and_(
								CRWorkflow.id == workflow_id,
								CRWorkflow.tenant_id == self.tenant_id
							)
						)
						.values(
							deleted_at=datetime.now(timezone.utc),
							deleted_by=user_id
						)
					)
				
				await session.commit()
			
			# Remove from cache
			await self._remove_cached_workflow(workflow_id)
			
			# Log operation
			await self._log_operation(WorkflowOperationType.DELETE, workflow_id, user_id, {
				"hard_delete": hard_delete,
				"workflow_name": workflow.name
			})
			
			logger.info(f"Deleted workflow: {workflow_id} (hard={hard_delete})")
			return True
			
		except Exception as e:
			logger.error(f"Failed to delete workflow {workflow_id}: {e}")
			raise
	
	async def search_workflows(
		self,
		filters: WorkflowSearchFilter,
		user_id: str
	) -> List[Workflow]:
		"""Search workflows with filters."""
		
		try:
			async with self.database_manager.get_session() as session:
				query = select(CRWorkflow).where(
					and_(
						CRWorkflow.tenant_id == self.tenant_id,
						CRWorkflow.deleted_at.is_(None)
					)
				)
				
				# Apply filters
				if filters.name_pattern:
					pattern = filters.name_pattern.replace('*', '%')
					query = query.where(CRWorkflow.name.like(pattern))
				
				if filters.status:
					status_values = [s.value for s in filters.status]
					query = query.where(CRWorkflow.status.in_(status_values))
				
				if filters.tags:
					# PostgreSQL array overlap operator
					query = query.where(CRWorkflow.tags.op('&&')(filters.tags))
				
				if filters.created_after:
					query = query.where(CRWorkflow.created_at >= filters.created_after)
				
				if filters.created_before:
					query = query.where(CRWorkflow.created_at <= filters.created_before)
				
				if filters.updated_after:
					query = query.where(CRWorkflow.updated_at >= filters.updated_after)
				
				if filters.updated_before:
					query = query.where(CRWorkflow.updated_at <= filters.updated_before)
				
				if filters.author:
					query = query.where(CRWorkflow.created_by == filters.author)
				
				if filters.priority:
					priority_values = [p.value for p in filters.priority]
					query = query.where(CRWorkflow.priority.in_(priority_values))
				
				# Apply sorting
				sort_column = getattr(CRWorkflow, filters.sort_by)
				if filters.sort_order == "desc":
					query = query.order_by(sort_column.desc())
				else:
					query = query.order_by(sort_column.asc())
				
				# Apply pagination
				query = query.offset(filters.offset).limit(filters.limit)
				
				# Execute query
				result = await session.execute(query)
				db_workflows = result.scalars().all()
				
				# Convert to Pydantic models
				workflows = []
				for db_workflow in db_workflows:
					workflow = Workflow(**db_workflow.definition)
					workflows.append(workflow)
				
				# Log operation
				await self._log_operation(WorkflowOperationType.READ, "search", user_id, {
					"filters": filters.model_dump(exclude_none=True),
					"result_count": len(workflows)
				})
				
				return workflows
				
		except Exception as e:
			logger.error(f"Failed to search workflows: {e}")
			raise
	
	async def clone_workflow(
		self,
		workflow_id: str,
		new_name: str,
		user_id: str,
		modifications: Optional[Dict[str, Any]] = None
	) -> Workflow:
		"""Clone existing workflow with optional modifications."""
		
		try:
			# Get source workflow
			source_workflow = await self.get_workflow(workflow_id, user_id)
			if not source_workflow:
				raise ValueError(f"Source workflow not found: {workflow_id}")
			
			# Create clone data
			clone_data = source_workflow.model_dump()
			clone_data.update({
				"id": uuid7str(),
				"name": new_name,
				"created_by": user_id,
				"updated_by": user_id,
				"created_at": datetime.now(timezone.utc),
				"updated_at": datetime.now(timezone.utc),
				"status": WorkflowStatus.DRAFT
			})
			
			# Apply modifications
			if modifications:
				clone_data.update(modifications)
			
			# Create cloned workflow
			cloned_workflow = await self.create_workflow(clone_data, user_id)
			
			# Log operation
			await self._log_operation(WorkflowOperationType.CLONE, workflow_id, user_id, {
				"new_workflow_id": cloned_workflow.id,
				"new_name": new_name,
				"modifications": list(modifications.keys()) if modifications else []
			})
			
			logger.info(f"Cloned workflow: {workflow_id} -> {cloned_workflow.id}")
			return cloned_workflow
			
		except Exception as e:
			logger.error(f"Failed to clone workflow {workflow_id}: {e}")
			raise
	
	async def get_workflow_statistics(
		self,
		workflow_id: str,
		user_id: str,
		time_range_days: int = 30
	) -> WorkflowStatistics:
		"""Get workflow execution statistics."""
		
		try:
			cutoff_date = datetime.now(timezone.utc) - timedelta(days=time_range_days)
			
			async with self.database_manager.get_session() as session:
				# Get execution statistics
				result = await session.execute(
					select(
						func.count(CRWorkflowInstance.id).label('total_executions'),
						func.count().filter(CRWorkflowInstance.status == 'completed').label('successful_executions'),
						func.count().filter(CRWorkflowInstance.status == 'failed').label('failed_executions'),
						func.avg(
							func.extract('epoch', CRWorkflowInstance.completed_at - CRWorkflowInstance.started_at)
						).label('avg_duration'),
						func.min(
							func.extract('epoch', CRWorkflowInstance.completed_at - CRWorkflowInstance.started_at)
						).label('min_duration'),
						func.max(
							func.extract('epoch', CRWorkflowInstance.completed_at - CRWorkflowInstance.started_at)
						).label('max_duration'),
						func.max(CRWorkflowInstance.started_at).label('last_execution')
					).where(
						and_(
							CRWorkflowInstance.workflow_id == workflow_id,
							CRWorkflowInstance.tenant_id == self.tenant_id,
							CRWorkflowInstance.started_at >= cutoff_date
						)
					)
				)
				
				stats_row = result.first()
				
				# Get active instances count
				active_result = await session.execute(
					select(func.count(CRWorkflowInstance.id)).where(
						and_(
							CRWorkflowInstance.workflow_id == workflow_id,
							CRWorkflowInstance.tenant_id == self.tenant_id,
							CRWorkflowInstance.status.in_(['running', 'paused'])
						)
					)
				)
				active_instances = active_result.scalar() or 0
				
				# Get task statistics
				task_result = await session.execute(
					select(
						func.count(CRTaskExecution.id).label('total_tasks'),
						func.count().filter(CRTaskExecution.status == 'completed').label('completed_tasks'),
						func.count().filter(CRTaskExecution.status == 'failed').label('failed_tasks')
					).join(CRWorkflowInstance).where(
						and_(
							CRWorkflowInstance.workflow_id == workflow_id,
							CRWorkflowInstance.tenant_id == self.tenant_id,
							CRTaskExecution.started_at >= cutoff_date
						)
					)
				)
				
				task_stats = task_result.first()
				
				# Calculate statistics
				total_executions = stats_row.total_executions or 0
				successful_executions = stats_row.successful_executions or 0
				failed_executions = stats_row.failed_executions or 0
				
				success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0.0
				
				statistics = WorkflowStatistics(
					total_executions=total_executions,
					successful_executions=successful_executions,
					failed_executions=failed_executions,
					average_duration_seconds=float(stats_row.avg_duration or 0),
					success_rate=success_rate,
					last_execution=stats_row.last_execution,
					fastest_execution_seconds=float(stats_row.min_duration or 0) if stats_row.min_duration else None,
					slowest_execution_seconds=float(stats_row.max_duration or 0) if stats_row.max_duration else None,
					active_instances=active_instances,
					total_tasks=task_stats.total_tasks or 0,
					completed_tasks=task_stats.completed_tasks or 0,
					failed_tasks=task_stats.failed_tasks or 0
				)
				
				# Log operation
				await self._log_operation(WorkflowOperationType.READ, workflow_id, user_id, {
					"operation": "statistics",
					"time_range_days": time_range_days
				})
				
				return statistics
				
		except Exception as e:
			logger.error(f"Failed to get workflow statistics {workflow_id}: {e}")
			raise
	
	async def _validate_workflow(
		self,
		workflow: Workflow,
		validation_level: WorkflowValidationLevel
	) -> WorkflowValidationResult:
		"""Validate workflow based on specified level."""
		
		errors = []
		warnings = []
		score = 0
		details = {}
		
		try:
			# Basic validation
			if not workflow.name or len(workflow.name.strip()) == 0:
				errors.append("Workflow name is required")
			elif len(workflow.name) > 200:
				errors.append("Workflow name too long (max 200 characters)")
			else:
				score += 10
			
			if not workflow.tasks or len(workflow.tasks) == 0:
				errors.append("Workflow must have at least one task")
			else:
				score += 20
			
			# Task validation
			task_ids = set()
			for task in workflow.tasks:
				if task.id in task_ids:
					errors.append(f"Duplicate task ID: {task.id}")
				task_ids.add(task.id)
				
				# Validate task dependencies
				for dep_id in task.dependencies:
					if dep_id not in task_ids and dep_id not in [t.id for t in workflow.tasks]:
						warnings.append(f"Task {task.id} depends on non-existent task: {dep_id}")
			
			if len(task_ids) == len(workflow.tasks):
				score += 20
			
			# Standard validation
			if validation_level in [WorkflowValidationLevel.STANDARD, WorkflowValidationLevel.STRICT, WorkflowValidationLevel.ENTERPRISE]:
				# Check for circular dependencies
				if self._has_circular_dependencies(workflow.tasks):
					errors.append("Workflow contains circular dependencies")
				else:
					score += 20
				
				# Check for unreachable tasks
				unreachable_tasks = self._find_unreachable_tasks(workflow.tasks)
				if unreachable_tasks:
					warnings.append(f"Unreachable tasks found: {', '.join(unreachable_tasks)}")
				else:
					score += 15
			
			# Strict validation
			if validation_level in [WorkflowValidationLevel.STRICT, WorkflowValidationLevel.ENTERPRISE]:
				# Validate task configurations
				for task in workflow.tasks:
					if task.task_type == TaskType.INTEGRATION and not task.configuration:
						warnings.append(f"Integration task {task.id} missing configuration")
					
					if task.sla_hours and task.sla_hours <= 0:
						errors.append(f"Task {task.id} has invalid SLA: {task.sla_hours}")
				
				score += 10
			
			# Enterprise validation
			if validation_level == WorkflowValidationLevel.ENTERPRISE:
				# Security and compliance checks
				if not workflow.metadata or not workflow.metadata.get("compliance_level"):
					warnings.append("Missing compliance level metadata")
				
				# Performance analysis
				estimated_duration = sum(task.estimated_duration or 60 for task in workflow.tasks)
				if estimated_duration > 3600:  # More than 1 hour
					warnings.append("Workflow estimated duration exceeds 1 hour")
				
				score += 5
			
			# Calculate final score
			max_score = 100
			final_score = min(score, max_score)
			
			if errors:
				final_score = 0
			
			return WorkflowValidationResult(
				is_valid=len(errors) == 0,
				errors=errors,
				warnings=warnings,
				validation_level=validation_level,
				score=final_score,
				details={
					"task_count": len(workflow.tasks),
					"estimated_duration": estimated_duration if validation_level == WorkflowValidationLevel.ENTERPRISE else None,
					"dependency_graph_depth": self._calculate_dependency_depth(workflow.tasks)
				}
			)
			
		except Exception as e:
			logger.error(f"Workflow validation error: {e}")
			return WorkflowValidationResult(
				is_valid=False,
				errors=[f"Validation failed: {e}"],
				validation_level=validation_level,
				score=0
			)
	
	def _has_circular_dependencies(self, tasks: List[TaskDefinition]) -> bool:
		"""Check for circular dependencies in task graph."""
		
		# Create adjacency list
		graph = {task.id: task.dependencies for task in tasks}
		
		# Use DFS to detect cycles
		visited = set()
		rec_stack = set()
		
		def has_cycle(node: str) -> bool:
			if node in rec_stack:
				return True
			
			if node in visited:
				return False
			
			visited.add(node)
			rec_stack.add(node)
			
			for neighbor in graph.get(node, []):
				if has_cycle(neighbor):
					return True
			
			rec_stack.remove(node)
			return False
		
		for task_id in graph:
			if task_id not in visited:
				if has_cycle(task_id):
					return True
		
		return False
	
	def _find_unreachable_tasks(self, tasks: List[TaskDefinition]) -> List[str]:
		"""Find tasks that cannot be reached from entry points."""
		
		task_ids = {task.id for task in tasks}
		dependency_targets = set()
		
		for task in tasks:
			dependency_targets.update(task.dependencies)
		
		# Tasks without dependencies are entry points
		entry_points = {task.id for task in tasks if not task.dependencies}
		
		if not entry_points:
			# If no entry points, all tasks except those with external dependencies are unreachable
			return list(task_ids - dependency_targets)
		
		# BFS from entry points
		reachable = set(entry_points)
		queue = list(entry_points)
		
		# Create reverse dependency map
		dependents = {task_id: [] for task_id in task_ids}
		for task in tasks:
			for dep in task.dependencies:
				if dep in dependents:
					dependents[dep].append(task.id)
		
		while queue:
			current = queue.pop(0)
			for dependent in dependents[current]:
				if dependent not in reachable:
					reachable.add(dependent)
					queue.append(dependent)
		
		return list(task_ids - reachable)
	
	def _calculate_dependency_depth(self, tasks: List[TaskDefinition]) -> int:
		"""Calculate maximum dependency chain depth."""
		
		task_dict = {task.id: task for task in tasks}
		memo = {}
		
		def get_depth(task_id: str) -> int:
			if task_id in memo:
				return memo[task_id]
			
			if task_id not in task_dict:
				return 0
			
			task = task_dict[task_id]
			if not task.dependencies:
				memo[task_id] = 1
				return 1
			
			max_dep_depth = max(get_depth(dep_id) for dep_id in task.dependencies)
			depth = max_dep_depth + 1
			memo[task_id] = depth
			return depth
		
		return max(get_depth(task.id) for task in tasks) if tasks else 0
	
	async def _count_active_instances(self, workflow_id: str) -> int:
		"""Count active workflow instances."""
		
		async with self.database_manager.get_session() as session:
			result = await session.execute(
				select(func.count(CRWorkflowInstance.id)).where(
					and_(
						CRWorkflowInstance.workflow_id == workflow_id,
						CRWorkflowInstance.tenant_id == self.tenant_id,
						CRWorkflowInstance.status.in_(['running', 'paused'])
					)
				)
			)
			return result.scalar() or 0
	
	async def _cache_workflow(self, workflow: Workflow) -> None:
		"""Cache workflow in Redis."""
		
		try:
			cache_key = f"workflow:{self.tenant_id}:{workflow.id}"
			workflow_json = workflow.model_dump_json()
			
			await self.redis_client.setex(
				cache_key,
				3600,  # 1 hour TTL
				workflow_json
			)
		except Exception as e:
			logger.warning(f"Failed to cache workflow {workflow.id}: {e}")
	
	async def _get_cached_workflow(self, workflow_id: str) -> Optional[Workflow]:
		"""Get workflow from cache."""
		
		try:
			cache_key = f"workflow:{self.tenant_id}:{workflow_id}"
			cached_data = await self.redis_client.get(cache_key)
			
			if cached_data:
				workflow_data = json.loads(cached_data)
				return Workflow(**workflow_data)
			
			return None
		except Exception as e:
			logger.warning(f"Failed to get cached workflow {workflow_id}: {e}")
			return None
	
	async def _remove_cached_workflow(self, workflow_id: str) -> None:
		"""Remove workflow from cache."""
		
		try:
			cache_key = f"workflow:{self.tenant_id}:{workflow_id}"
			await self.redis_client.delete(cache_key)
		except Exception as e:
			logger.warning(f"Failed to remove cached workflow {workflow_id}: {e}")
	
	async def _log_operation(
		self,
		operation_type: WorkflowOperationType,
		workflow_id: str,
		user_id: str,
		details: Dict[str, Any]
	) -> None:
		"""Log workflow operation for audit purposes."""
		
		operation_log = {
			"operation": operation_type.value,
			"workflow_id": workflow_id,
			"user_id": user_id,
			"tenant_id": self.tenant_id,
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"details": details
		}
		
		# Add to in-memory history (limited size)
		self.operation_history.append(operation_log)
		if len(self.operation_history) > 1000:
			self.operation_history.pop(0)
		
		# Log to Redis for distributed access
		try:
			log_key = f"workflow_operations:{self.tenant_id}"
			await self.redis_client.lpush(log_key, json.dumps(operation_log))
			await self.redis_client.ltrim(log_key, 0, 9999)  # Keep last 10k operations
			await self.redis_client.expire(log_key, 604800)  # 7 days TTL
		except Exception as e:
			logger.warning(f"Failed to log operation to Redis: {e}")

class WorkflowManager:
	"""High-level workflow management service."""
	
	def __init__(
		self,
		database_manager: DatabaseManager,
		redis_client: redis.Redis,
		tenant_id: str
	):
		self.operations = WorkflowOperations(database_manager, redis_client, tenant_id)
		self.tenant_id = tenant_id
		
		logger.info(f"Initialized WorkflowManager for tenant {tenant_id}")
	
	async def create_workflow(
		self,
		workflow_data: Dict[str, Any],
		user_id: str,
		validation_level: WorkflowValidationLevel = WorkflowValidationLevel.STANDARD
	) -> Workflow:
		"""Create new workflow with comprehensive validation."""
		return await self.operations.create_workflow(workflow_data, user_id, validation_level)
	
	async def get_workflow(
		self,
		workflow_id: str,
		user_id: str,
		include_instances: bool = False
	) -> Optional[Workflow]:
		"""Get workflow by ID with optional instance data."""
		return await self.operations.get_workflow(workflow_id, user_id, include_instances)
	
	async def update_workflow(
		self,
		workflow_id: str,
		updates: Dict[str, Any],
		user_id: str,
		validation_level: WorkflowValidationLevel = WorkflowValidationLevel.STANDARD
	) -> Workflow:
		"""Update workflow with validation."""
		return await self.operations.update_workflow(workflow_id, updates, user_id, validation_level)
	
	async def delete_workflow(
		self,
		workflow_id: str,
		user_id: str,
		hard_delete: bool = False
	) -> bool:
		"""Delete workflow (soft or hard delete)."""
		return await self.operations.delete_workflow(workflow_id, user_id, hard_delete)
	
	async def search_workflows(
		self,
		filters: WorkflowSearchFilter,
		user_id: str
	) -> List[Workflow]:
		"""Search workflows with advanced filtering."""
		return await self.operations.search_workflows(filters, user_id)
	
	async def clone_workflow(
		self,
		workflow_id: str,
		new_name: str,
		user_id: str,
		modifications: Optional[Dict[str, Any]] = None
	) -> Workflow:
		"""Clone workflow with optional modifications."""
		return await self.operations.clone_workflow(workflow_id, new_name, user_id, modifications)
	
	async def get_workflow_statistics(
		self,
		workflow_id: str,
		user_id: str,
		time_range_days: int = 30
	) -> WorkflowStatistics:
		"""Get comprehensive workflow statistics."""
		return await self.operations.get_workflow_statistics(workflow_id, user_id, time_range_days)

# Export workflow management classes
__all__ = [
	"WorkflowManager",
	"WorkflowOperations", 
	"WorkflowOperationType",
	"WorkflowValidationLevel",
	"WorkflowSearchFilter",
	"WorkflowValidationResult",
	"WorkflowStatistics"
]