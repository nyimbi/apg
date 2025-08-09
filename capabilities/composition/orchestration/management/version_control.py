"""
APG Workflow Orchestration Version Control System

Comprehensive version control for workflows with branching, merging,
tagging, and collaborative development features.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone
from enum import Enum
import logging
import hashlib
import difflib

from pydantic import BaseModel, Field, ConfigDict, validator
from uuid_extensions import uuid7str
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, and_, or_, func, text

from ..models import Workflow, WorkflowStatus
from ..database import DatabaseManager

logger = logging.getLogger(__name__)

class VersionChangeType(Enum):
	"""Types of changes in workflow versions."""
	CREATED = "created"
	MODIFIED = "modified"
	DELETED = "deleted"
	RENAMED = "renamed"
	MOVED = "moved"

class VersionStatus(Enum):
	"""Status of workflow versions."""
	DRAFT = "draft"
	ACTIVE = "active"
	ARCHIVED = "archived"
	DEPRECATED = "deprecated"

class MergeStrategy(Enum):
	"""Strategies for merging workflow versions."""
	AUTO = "auto"			# Automatic merge where possible
	MANUAL = "manual"		# Manual conflict resolution required
	OURS = "ours"			# Keep our version in conflicts
	THEIRS = "theirs"		# Keep incoming version in conflicts
	UNION = "union"			# Combine both versions

class WorkflowVersion(BaseModel):
	"""Workflow version information."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	workflow_id: str = Field(..., description="Parent workflow ID")
	version_number: str = Field(..., description="Semantic version (e.g., 1.2.3)")
	branch_name: str = Field(default="main", description="Branch name")
	parent_version_id: Optional[str] = Field(default=None, description="Parent version ID")
	status: VersionStatus = Field(default=VersionStatus.DRAFT)
	title: str = Field(..., max_length=200, description="Version title")
	description: str = Field(default="", description="Version description")
	changelog: List[str] = Field(default_factory=list, description="List of changes")
	tags: List[str] = Field(default_factory=list, description="Version tags")
	checksum: str = Field(..., description="Content checksum for integrity")
	workflow_definition: Dict[str, Any] = Field(..., description="Workflow definition at this version")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Version metadata")
	created_by: str = Field(..., description="User who created this version")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	tenant_id: str = Field(..., description="APG tenant identifier")

class VersionComparison(BaseModel):
	"""Comparison between two workflow versions."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	from_version: str
	to_version: str
	changes: List[Dict[str, Any]] = Field(default_factory=list)
	added_tasks: List[str] = Field(default_factory=list)
	modified_tasks: List[str] = Field(default_factory=list)
	deleted_tasks: List[str] = Field(default_factory=list)
	configuration_changes: Dict[str, Any] = Field(default_factory=dict)
	compatibility_score: float = Field(default=100.0, ge=0.0, le=100.0)
	breaking_changes: bool = Field(default=False)
	summary: str = Field(default="")

class MergeConflict(BaseModel):
	"""Merge conflict information."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	path: str = Field(..., description="Path to conflicted element")
	conflict_type: str = Field(..., description="Type of conflict")
	base_value: Any = Field(description="Original value")
	ours_value: Any = Field(description="Our version value")
	theirs_value: Any = Field(description="Their version value")
	description: str = Field(default="", description="Conflict description")

class MergeResult(BaseModel):
	"""Result of a merge operation."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	success: bool
	merged_version_id: Optional[str] = Field(default=None)
	conflicts: List[MergeConflict] = Field(default_factory=list)
	auto_resolved: int = Field(default=0, description="Number of automatically resolved conflicts")
	manual_required: int = Field(default=0, description="Number of conflicts requiring manual resolution")
	summary: str = Field(default="")

class WorkflowVersionControl:
	"""Core version control operations for workflows."""
	
	def __init__(
		self,
		database_manager: DatabaseManager,
		redis_client: redis.Redis,
		tenant_id: str
	):
		self.database_manager = database_manager
		self.redis_client = redis_client
		self.tenant_id = tenant_id
		self.version_table = "cr_workflow_versions"
		
		logger.info(f"Initialized WorkflowVersionControl for tenant {tenant_id}")
	
	async def create_version(
		self,
		workflow: Workflow,
		version_number: str,
		title: str,
		description: str = "",
		branch_name: str = "main",
		parent_version_id: Optional[str] = None,
		user_id: str = ""
	) -> WorkflowVersion:
		"""Create a new version of a workflow."""
		
		try:
			# Calculate content checksum
			workflow_json = json.dumps(workflow.model_dump(), sort_keys=True)
			checksum = hashlib.sha256(workflow_json.encode()).hexdigest()
			
			# Create version object
			version = WorkflowVersion(
				workflow_id=workflow.id,
				version_number=version_number,
				branch_name=branch_name,
				parent_version_id=parent_version_id,
				title=title,
				description=description,
				checksum=checksum,
				workflow_definition=workflow.model_dump(),
				created_by=user_id,
				tenant_id=self.tenant_id
			)
			
			# Save to database
			async with self.database_manager.get_session() as session:
				await session.execute(
					text(f"""
					INSERT INTO {self.version_table} (
						id, workflow_id, version_number, branch_name, parent_version_id,
						status, title, description, changelog, tags, checksum,
						workflow_definition, metadata, created_by, created_at, tenant_id
					) VALUES (
						:id, :workflow_id, :version_number, :branch_name, :parent_version_id,
						:status, :title, :description, :changelog, :tags, :checksum,
						:workflow_definition, :metadata, :created_by, :created_at, :tenant_id
					)
					"""),
					{
						"id": version.id,
						"workflow_id": version.workflow_id,
						"version_number": version.version_number,
						"branch_name": version.branch_name,
						"parent_version_id": version.parent_version_id,
						"status": version.status.value,
						"title": version.title,
						"description": version.description,
						"changelog": json.dumps(version.changelog),
						"tags": json.dumps(version.tags),
						"checksum": version.checksum,
						"workflow_definition": json.dumps(version.workflow_definition),
						"metadata": json.dumps(version.metadata),
						"created_by": version.created_by,
						"created_at": version.created_at,
						"tenant_id": version.tenant_id
					}
				)
				await session.commit()
			
			# Cache version
			await self._cache_version(version)
			
			logger.info(f"Created version {version_number} for workflow {workflow.id}")
			return version
			
		except Exception as e:
			logger.error(f"Failed to create version: {e}")
			raise
	
	async def get_version(
		self,
		version_id: str
	) -> Optional[WorkflowVersion]:
		"""Get a specific workflow version."""
		
		try:
			# Try cache first
			cached_version = await self._get_cached_version(version_id)
			if cached_version:
				return cached_version
			
			# Query database
			async with self.database_manager.get_session() as session:
				result = await session.execute(
					text(f"""
					SELECT * FROM {self.version_table}
					WHERE id = :version_id AND tenant_id = :tenant_id
					"""),
					{"version_id": version_id, "tenant_id": self.tenant_id}
				)
				
				row = result.first()
				if not row:
					return None
				
				# Convert to WorkflowVersion
				version = WorkflowVersion(
					id=row.id,
					workflow_id=row.workflow_id,
					version_number=row.version_number,
					branch_name=row.branch_name,
					parent_version_id=row.parent_version_id,
					status=VersionStatus(row.status),
					title=row.title,
					description=row.description,
					changelog=json.loads(row.changelog) if row.changelog else [],
					tags=json.loads(row.tags) if row.tags else [],
					checksum=row.checksum,
					workflow_definition=json.loads(row.workflow_definition),
					metadata=json.loads(row.metadata) if row.metadata else {},
					created_by=row.created_by,
					created_at=row.created_at,
					tenant_id=row.tenant_id
				)
				
				# Cache version
				await self._cache_version(version)
				
				return version
				
		except Exception as e:
			logger.error(f"Failed to get version {version_id}: {e}")
			raise
	
	async def list_versions(
		self,
		workflow_id: str,
		branch_name: Optional[str] = None,
		status: Optional[VersionStatus] = None,
		limit: int = 50
	) -> List[WorkflowVersion]:
		"""List versions for a workflow."""
		
		try:
			query = f"""
			SELECT * FROM {self.version_table}
			WHERE workflow_id = :workflow_id AND tenant_id = :tenant_id
			"""
			params = {"workflow_id": workflow_id, "tenant_id": self.tenant_id}
			
			if branch_name:
				query += " AND branch_name = :branch_name"
				params["branch_name"] = branch_name
			
			if status:
				query += " AND status = :status"
				params["status"] = status.value
			
			query += " ORDER BY created_at DESC LIMIT :limit"
			params["limit"] = limit
			
			async with self.database_manager.get_session() as session:
				result = await session.execute(text(query), params)
				rows = result.fetchall()
				
				versions = []
				for row in rows:
					version = WorkflowVersion(
						id=row.id,
						workflow_id=row.workflow_id,
						version_number=row.version_number,
						branch_name=row.branch_name,
						parent_version_id=row.parent_version_id,
						status=VersionStatus(row.status),
						title=row.title,
						description=row.description,
						changelog=json.loads(row.changelog) if row.changelog else [],
						tags=json.loads(row.tags) if row.tags else [],
						checksum=row.checksum,
						workflow_definition=json.loads(row.workflow_definition),
						metadata=json.loads(row.metadata) if row.metadata else {},
						created_by=row.created_by,
						created_at=row.created_at,
						tenant_id=row.tenant_id
					)
					versions.append(version)
				
				return versions
				
		except Exception as e:
			logger.error(f"Failed to list versions for workflow {workflow_id}: {e}")
			raise
	
	async def compare_versions(
		self,
		from_version_id: str,
		to_version_id: str
	) -> VersionComparison:
		"""Compare two workflow versions."""
		
		try:
			# Get both versions
			from_version = await self.get_version(from_version_id)
			to_version = await self.get_version(to_version_id)
			
			if not from_version or not to_version:
				raise ValueError("One or both versions not found")
			
			# Extract workflow definitions
			from_workflow = Workflow(**from_version.workflow_definition)
			to_workflow = Workflow(**to_version.workflow_definition)
			
			# Compare tasks
			from_tasks = {task.id: task for task in from_workflow.tasks}
			to_tasks = {task.id: task for task in to_workflow.tasks}
			
			added_tasks = list(set(to_tasks.keys()) - set(from_tasks.keys()))
			deleted_tasks = list(set(from_tasks.keys()) - set(to_tasks.keys()))
			modified_tasks = []
			
			# Check for modified tasks
			for task_id in set(from_tasks.keys()) & set(to_tasks.keys()):
				from_task = from_tasks[task_id]
				to_task = to_tasks[task_id]
				
				if from_task.model_dump() != to_task.model_dump():
					modified_tasks.append(task_id)
			
			# Analyze configuration changes
			config_changes = {}
			if from_workflow.configuration != to_workflow.configuration:
				config_changes = {
					"from": from_workflow.configuration,
					"to": to_workflow.configuration
				}
			
			# Calculate compatibility score
			total_elements = len(from_tasks) + len(to_tasks)
			changed_elements = len(added_tasks) + len(deleted_tasks) + len(modified_tasks)
			compatibility_score = (1 - (changed_elements / max(total_elements, 1))) * 100
			
			# Detect breaking changes
			breaking_changes = (
				len(deleted_tasks) > 0 or
				len(modified_tasks) > len(from_tasks) * 0.5 or
				from_workflow.name != to_workflow.name
			)
			
			# Generate detailed changes
			changes = []
			
			for task_id in added_tasks:
				changes.append({
					"type": "task_added",
					"task_id": task_id,
					"task_name": to_tasks[task_id].name,
					"description": f"Added new task: {to_tasks[task_id].name}"
				})
			
			for task_id in deleted_tasks:
				changes.append({
					"type": "task_deleted",
					"task_id": task_id,
					"task_name": from_tasks[task_id].name,
					"description": f"Deleted task: {from_tasks[task_id].name}"
				})
			
			for task_id in modified_tasks:
				from_task = from_tasks[task_id]
				to_task = to_tasks[task_id]
				
				# Find specific differences
				task_changes = []
				if from_task.name != to_task.name:
					task_changes.append(f"name: '{from_task.name}' → '{to_task.name}'")
				if from_task.description != to_task.description:
					task_changes.append("description modified")
				if from_task.configuration != to_task.configuration:
					task_changes.append("configuration modified")
				
				changes.append({
					"type": "task_modified",
					"task_id": task_id,
					"task_name": to_task.name,
					"description": f"Modified task: {', '.join(task_changes)}"
				})
			
			# Generate summary
			summary_parts = []
			if added_tasks:
				summary_parts.append(f"{len(added_tasks)} task(s) added")
			if deleted_tasks:
				summary_parts.append(f"{len(deleted_tasks)} task(s) deleted")
			if modified_tasks:
				summary_parts.append(f"{len(modified_tasks)} task(s) modified")
			if config_changes:
				summary_parts.append("configuration changed")
			
			summary = ", ".join(summary_parts) if summary_parts else "No changes detected"
			
			return VersionComparison(
				from_version=from_version_id,
				to_version=to_version_id,
				changes=changes,
				added_tasks=added_tasks,
				modified_tasks=modified_tasks,
				deleted_tasks=deleted_tasks,
				configuration_changes=config_changes,
				compatibility_score=compatibility_score,
				breaking_changes=breaking_changes,
				summary=summary
			)
			
		except Exception as e:
			logger.error(f"Failed to compare versions: {e}")
			raise
	
	async def create_branch(
		self,
		workflow_id: str,
		branch_name: str,
		from_version_id: str,
		user_id: str,
		description: str = ""
	) -> WorkflowVersion:
		"""Create a new branch from an existing version."""
		
		try:
			# Get source version
			source_version = await self.get_version(from_version_id)
			if not source_version:
				raise ValueError(f"Source version not found: {from_version_id}")
			
			# Create new version on the branch
			branch_version = await self.create_version(
				workflow=Workflow(**source_version.workflow_definition),
				version_number="1.0.0",  # Start with 1.0.0 for new branch
				title=f"Branch {branch_name}",
				description=description,
				branch_name=branch_name,
				parent_version_id=from_version_id,
				user_id=user_id
			)
			
			logger.info(f"Created branch {branch_name} from version {from_version_id}")
			return branch_version
			
		except Exception as e:
			logger.error(f"Failed to create branch: {e}")
			raise
	
	async def merge_versions(
		self,
		base_version_id: str,
		merge_version_id: str,
		target_branch: str,
		strategy: MergeStrategy,
		user_id: str,
		merge_message: str = ""
	) -> MergeResult:
		"""Merge two workflow versions."""
		
		try:
			# Get versions to merge
			base_version = await self.get_version(base_version_id)
			merge_version = await self.get_version(merge_version_id)
			
			if not base_version or not merge_version:
				raise ValueError("One or both versions not found")
			
			base_workflow = Workflow(**base_version.workflow_definition)
			merge_workflow = Workflow(**merge_version.workflow_definition)
			
			# Detect conflicts
			conflicts = await self._detect_merge_conflicts(base_workflow, merge_workflow)
			
			if conflicts and strategy == MergeStrategy.MANUAL:
				return MergeResult(
					success=False,
					conflicts=conflicts,
					manual_required=len(conflicts),
					summary="Manual resolution required for conflicts"
				)
			
			# Resolve conflicts based on strategy
			merged_workflow, auto_resolved = await self._resolve_conflicts(
				base_workflow, merge_workflow, conflicts, strategy
			)
			
			remaining_conflicts = [c for c in conflicts if not self._is_conflict_resolved(c, strategy)]
			
			if remaining_conflicts:
				return MergeResult(
					success=False,
					conflicts=remaining_conflicts,
					auto_resolved=auto_resolved,
					manual_required=len(remaining_conflicts),
					summary=f"Automatic resolution failed for {len(remaining_conflicts)} conflicts"
				)
			
			# Create merged version
			next_version = await self._calculate_next_version(base_version.workflow_id, target_branch)
			
			merged_version = await self.create_version(
				workflow=merged_workflow,
				version_number=next_version,
				title=f"Merge {merge_version.branch_name} into {target_branch}",
				description=merge_message or f"Merged version {merge_version.version_number}",
				branch_name=target_branch,
				parent_version_id=base_version_id,
				user_id=user_id
			)
			
			return MergeResult(
				success=True,
				merged_version_id=merged_version.id,
				auto_resolved=auto_resolved,
				summary=f"Successfully merged {len(conflicts)} conflicts automatically"
			)
			
		except Exception as e:
			logger.error(f"Failed to merge versions: {e}")
			return MergeResult(
				success=False,
				summary=f"Merge failed: {str(e)}"
			)
	
	async def tag_version(
		self,
		version_id: str,
		tag_name: str,
		tag_message: str = "",
		user_id: str = ""
	) -> bool:
		"""Add a tag to a version."""
		
		try:
			version = await self.get_version(version_id)
			if not version:
				raise ValueError(f"Version not found: {version_id}")
			
			# Add tag to version
			if tag_name not in version.tags:
				version.tags.append(tag_name)
				
				# Update in database
				async with self.database_manager.get_session() as session:
					await session.execute(
						text(f"""
						UPDATE {self.version_table}
						SET tags = :tags
						WHERE id = :version_id AND tenant_id = :tenant_id
						"""),
						{
							"tags": json.dumps(version.tags),
							"version_id": version_id,
							"tenant_id": self.tenant_id
						}
					)
					await session.commit()
				
				# Update cache
				await self._cache_version(version)
				
				logger.info(f"Tagged version {version_id} with {tag_name}")
				return True
			
			return False
			
		except Exception as e:
			logger.error(f"Failed to tag version: {e}")
			raise
	
	async def restore_version(
		self,
		version_id: str,
		user_id: str
	) -> Workflow:
		"""Restore a workflow to a specific version."""
		
		try:
			version = await self.get_version(version_id)
			if not version:
				raise ValueError(f"Version not found: {version_id}")
			
			# Create new version based on restored content
			restored_workflow = Workflow(**version.workflow_definition)
			next_version = await self._calculate_next_version(version.workflow_id, "main")
			
			restore_version = await self.create_version(
				workflow=restored_workflow,
				version_number=next_version,
				title=f"Restore to version {version.version_number}",
				description=f"Restored workflow to version {version.version_number}",
				branch_name="main",
				user_id=user_id
			)
			
			logger.info(f"Restored workflow to version {version.version_number}")
			return restored_workflow
			
		except Exception as e:
			logger.error(f"Failed to restore version: {e}")
			raise
	
	async def _detect_merge_conflicts(
		self,
		base_workflow: Workflow,
		merge_workflow: Workflow
	) -> List[MergeConflict]:
		"""Detect conflicts between two workflows."""
		
		conflicts = []
		
		# Name conflicts
		if base_workflow.name != merge_workflow.name:
			conflicts.append(MergeConflict(
				path="name",
				conflict_type="value_mismatch",
				base_value=base_workflow.name,
				ours_value=base_workflow.name,
				theirs_value=merge_workflow.name,
				description="Workflow name differs between versions"
			))
		
		# Task conflicts
		base_tasks = {task.id: task for task in base_workflow.tasks}
		merge_tasks = {task.id: task for task in merge_workflow.tasks}
		
		# Check for task modifications
		for task_id in set(base_tasks.keys()) & set(merge_tasks.keys()):
			base_task = base_tasks[task_id]
			merge_task = merge_tasks[task_id]
			
			if base_task.model_dump() != merge_task.model_dump():
				conflicts.append(MergeConflict(
					path=f"tasks.{task_id}",
					conflict_type="task_modified",
					base_value=base_task.model_dump(),
					ours_value=base_task.model_dump(),
					theirs_value=merge_task.model_dump(),
					description=f"Task {task_id} was modified in both versions"
				))
		
		# Configuration conflicts
		if base_workflow.configuration != merge_workflow.configuration:
			conflicts.append(MergeConflict(
				path="configuration",
				conflict_type="configuration_mismatch",
				base_value=base_workflow.configuration,
				ours_value=base_workflow.configuration,
				theirs_value=merge_workflow.configuration,
				description="Workflow configuration differs between versions"
			))
		
		return conflicts
	
	async def _resolve_conflicts(
		self,
		base_workflow: Workflow,
		merge_workflow: Workflow,
		conflicts: List[MergeConflict],
		strategy: MergeStrategy
	) -> Tuple[Workflow, int]:
		"""Resolve merge conflicts based on strategy."""
		
		resolved_count = 0
		merged_data = base_workflow.model_dump()
		
		for conflict in conflicts:
			if strategy == MergeStrategy.OURS:
				# Keep base version
				resolved_count += 1
			
			elif strategy == MergeStrategy.THEIRS:
				# Use merge version
				if conflict.path == "name":
					merged_data["name"] = conflict.theirs_value
				elif conflict.path == "configuration":
					merged_data["configuration"] = conflict.theirs_value
				elif conflict.path.startswith("tasks."):
					task_id = conflict.path.split(".")[1]
					# Find and replace task
					for i, task in enumerate(merged_data["tasks"]):
						if task["id"] == task_id:
							merged_data["tasks"][i] = conflict.theirs_value
							break
				resolved_count += 1
			
			elif strategy == MergeStrategy.AUTO:
				# Apply automatic resolution logic
				if conflict.conflict_type == "value_mismatch":
					# Use theirs for simple value conflicts
					if conflict.path == "name":
						merged_data["name"] = conflict.theirs_value
					resolved_count += 1
				
				elif conflict.conflict_type == "task_modified":
					# For task modifications, prefer theirs if it adds new features
					theirs_task = conflict.theirs_value
					ours_task = conflict.ours_value
					
					# Simple heuristic: if theirs has more configuration, use it
					theirs_config_size = len(str(theirs_task.get("configuration", {})))
					ours_config_size = len(str(ours_task.get("configuration", {})))
					
					if theirs_config_size > ours_config_size:
						task_id = conflict.path.split(".")[1]
						for i, task in enumerate(merged_data["tasks"]):
							if task["id"] == task_id:
								merged_data["tasks"][i] = theirs_task
								break
					resolved_count += 1
			
			elif strategy == MergeStrategy.UNION:
				# Combine both versions where possible
				if conflict.path == "configuration":
					# Merge configurations
					base_config = conflict.ours_value or {}
					merge_config = conflict.theirs_value or {}
					merged_config = {**base_config, **merge_config}
					merged_data["configuration"] = merged_config
					resolved_count += 1
		
		return Workflow(**merged_data), resolved_count
	
	def _is_conflict_resolved(self, conflict: MergeConflict, strategy: MergeStrategy) -> bool:
		"""Check if a conflict can be resolved with the given strategy."""
		
		if strategy in [MergeStrategy.OURS, MergeStrategy.THEIRS]:
			return True
		
		if strategy == MergeStrategy.AUTO:
			return conflict.conflict_type in ["value_mismatch", "task_modified"]
		
		if strategy == MergeStrategy.UNION:
			return conflict.conflict_type in ["configuration_mismatch"]
		
		return False
	
	async def _calculate_next_version(self, workflow_id: str, branch_name: str) -> str:
		"""Calculate the next version number for a branch."""
		
		try:
			# Get latest version for branch
			versions = await self.list_versions(workflow_id, branch_name, limit=1)
			
			if not versions:
				return "1.0.0"
			
			latest_version = versions[0].version_number
			
			# Parse semantic version (major.minor.patch)
			parts = latest_version.split(".")
			if len(parts) != 3:
				return "1.0.0"
			
			major, minor, patch = map(int, parts)
			
			# Increment patch version
			patch += 1
			
			return f"{major}.{minor}.{patch}"
			
		except Exception as e:
			logger.warning(f"Failed to calculate next version: {e}")
			return "1.0.0"
	
	async def _cache_version(self, version: WorkflowVersion) -> None:
		"""Cache version in Redis."""
		
		try:
			cache_key = f"workflow_version:{self.tenant_id}:{version.id}"
			version_json = version.model_dump_json()
			
			await self.redis_client.setex(
				cache_key,
				3600,  # 1 hour TTL
				version_json
			)
		except Exception as e:
			logger.warning(f"Failed to cache version {version.id}: {e}")
	
	async def _get_cached_version(self, version_id: str) -> Optional[WorkflowVersion]:
		"""Get version from cache."""
		
		try:
			cache_key = f"workflow_version:{self.tenant_id}:{version_id}"
			cached_data = await self.redis_client.get(cache_key)
			
			if cached_data:
				version_data = json.loads(cached_data)
				return WorkflowVersion(**version_data)
			
			return None
		except Exception as e:
			logger.warning(f"Failed to get cached version {version_id}: {e}")
			return None

class VersionManager:
	"""High-level version management service."""
	
	def __init__(
		self,
		database_manager: DatabaseManager,
		redis_client: redis.Redis,
		tenant_id: str
	):
		self.version_control = WorkflowVersionControl(database_manager, redis_client, tenant_id)
		self.tenant_id = tenant_id
		
		logger.info(f"Initialized VersionManager for tenant {tenant_id}")
	
	async def create_version(
		self,
		workflow: Workflow,
		version_number: str,
		title: str,
		description: str = "",
		branch_name: str = "main",
		parent_version_id: Optional[str] = None,
		user_id: str = ""
	) -> WorkflowVersion:
		"""Create a new workflow version."""
		return await self.version_control.create_version(
			workflow, version_number, title, description, branch_name, parent_version_id, user_id
		)
	
	async def get_version(self, version_id: str) -> Optional[WorkflowVersion]:
		"""Get a specific workflow version."""
		return await self.version_control.get_version(version_id)
	
	async def list_versions(
		self,
		workflow_id: str,
		branch_name: Optional[str] = None,
		status: Optional[VersionStatus] = None,
		limit: int = 50
	) -> List[WorkflowVersion]:
		"""List versions for a workflow."""
		return await self.version_control.list_versions(workflow_id, branch_name, status, limit)
	
	async def compare_versions(
		self,
		from_version_id: str,
		to_version_id: str
	) -> VersionComparison:
		"""Compare two workflow versions."""
		return await self.version_control.compare_versions(from_version_id, to_version_id)
	
	async def create_branch(
		self,
		workflow_id: str,
		branch_name: str,
		from_version_id: str,
		user_id: str,
		description: str = ""
	) -> WorkflowVersion:
		"""Create a new branch from an existing version."""
		return await self.version_control.create_branch(workflow_id, branch_name, from_version_id, user_id, description)
	
	async def merge_versions(
		self,
		base_version_id: str,
		merge_version_id: str,
		target_branch: str,
		strategy: MergeStrategy,
		user_id: str,
		merge_message: str = ""
	) -> MergeResult:
		"""Merge two workflow versions."""
		return await self.version_control.merge_versions(
			base_version_id, merge_version_id, target_branch, strategy, user_id, merge_message
		)
	
	async def tag_version(
		self,
		version_id: str,
		tag_name: str,
		tag_message: str = "",
		user_id: str = ""
	) -> bool:
		"""Add a tag to a version."""
		return await self.version_control.tag_version(version_id, tag_name, tag_message, user_id)
	
	async def restore_version(
		self,
		version_id: str,
		user_id: str
	) -> Workflow:
		"""Restore a workflow to a specific version."""
		return await self.version_control.restore_version(version_id, user_id)

# Export version control classes
__all__ = [
	"WorkflowVersionControl",
	"VersionManager",
	"WorkflowVersion",
	"VersionComparison",
	"MergeConflict",
	"MergeResult",
	"VersionChangeType",
	"VersionStatus",
	"MergeStrategy"
]