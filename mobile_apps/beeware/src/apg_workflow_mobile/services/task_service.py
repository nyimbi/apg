"""
Task Service for APG Workflow Mobile

Handles task management operations.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

from ..models.task import Task, TaskStatus, TaskPriority, TaskType, TaskAssignment
from ..models.api_response import APIResponse, PaginationInfo
from ..services.api_service import APIService
from ..utils.exceptions import APIException, ValidationException
from ..utils.constants import URL_PATTERNS


class TaskService:
	"""Service for task management operations"""
	
	def __init__(self, api_service: APIService, app=None):
		self.api_service = api_service
		self.app = app
		self.logger = logging.getLogger(__name__)
		
		self.logger.info("Task Service initialized")
	
	async def get_tasks(self, params: Optional[Dict[str, Any]] = None) -> APIResponse:
		"""Get list of tasks with optional filtering and pagination"""
		try:
			self.logger.info("Fetching tasks")
			
			query_params = params or {}
			
			# Add default pagination if not provided
			if "page" not in query_params:
				query_params["page"] = 1
			if "limit" not in query_params:
				query_params["limit"] = 20
			
			response = await self.api_service.get(
				URL_PATTERNS["tasks"]["list"],
				params=query_params
			)
			
			if response.success and response.data:
				# Convert task data to Task objects
				tasks_data = response.data.get("tasks", [])
				tasks = []
				
				for task_data in tasks_data:
					try:
						task = Task(**task_data)
						tasks.append(task)
					except Exception as e:
						self.logger.warning(f"Failed to parse task {task_data.get('id', 'unknown')}: {e}")
				
				# Update response data
				response.data = {"tasks": tasks}
				
				# Cache tasks if app state available
				if self.app and hasattr(self.app, 'app_state'):
					for task in tasks:
						self.app.app_state.cache_task(task)
			
			self.logger.info(f"Fetched {len(tasks_data) if response.success else 0} tasks")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching tasks: {e}")
			raise APIException(f"Failed to fetch tasks: {e}")
	
	async def get_task_by_id(self, task_id: str) -> APIResponse:
		"""Get task by ID"""
		try:
			self.logger.info(f"Fetching task: {task_id}")
			
			# Check cache first
			if self.app and hasattr(self.app, 'app_state'):
				cached_task = self.app.app_state.get_cached_task(task_id)
				if cached_task:
					self.logger.info(f"Returning cached task: {task_id}")
					return APIResponse.success_response(data=cached_task)
			
			response = await self.api_service.get(
				URL_PATTERNS["tasks"]["detail"].format(task_id=task_id)
			)
			
			if response.success and response.data:
				try:
					task = Task(**response.data)
					response.data = task
					
					# Cache task
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_task(task)
					
				except Exception as e:
					self.logger.error(f"Failed to parse task data: {e}")
					raise ValidationException(f"Invalid task data: {e}")
			
			self.logger.info(f"Fetched task: {task_id}")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching task {task_id}: {e}")
			raise APIException(f"Failed to fetch task: {e}")
	
	async def get_my_tasks(self, params: Optional[Dict[str, Any]] = None) -> APIResponse:
		"""Get tasks assigned to current user"""
		try:
			self.logger.info("Fetching my tasks")
			
			query_params = params or {}
			
			# Add default pagination if not provided
			if "page" not in query_params:
				query_params["page"] = 1
			if "limit" not in query_params:
				query_params["limit"] = 20
			
			response = await self.api_service.get("/tasks/my", params=query_params)
			
			if response.success and response.data:
				# Convert task data to Task objects
				tasks_data = response.data.get("tasks", [])
				tasks = []
				
				for task_data in tasks_data:
					try:
						task = Task(**task_data)
						tasks.append(task)
					except Exception as e:
						self.logger.warning(f"Failed to parse task {task_data.get('id', 'unknown')}: {e}")
				
				# Update response data
				response.data = {"tasks": tasks}
				
				# Cache tasks
				if self.app and hasattr(self.app, 'app_state'):
					for task in tasks:
						self.app.app_state.cache_task(task)
			
			self.logger.info(f"Fetched {len(tasks_data) if response.success else 0} my tasks")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching my tasks: {e}")
			raise APIException(f"Failed to fetch my tasks: {e}")
	
	async def get_assigned_tasks(self, params: Optional[Dict[str, Any]] = None) -> APIResponse:
		"""Get tasks assigned by current user"""
		try:
			self.logger.info("Fetching assigned tasks")
			
			query_params = params or {}
			
			# Add default pagination if not provided
			if "page" not in query_params:
				query_params["page"] = 1
			if "limit" not in query_params:
				query_params["limit"] = 20
			
			response = await self.api_service.get("/tasks/assigned", params=query_params)
			
			if response.success and response.data:
				# Convert task data to Task objects
				tasks_data = response.data.get("tasks", [])
				tasks = []
				
				for task_data in tasks_data:
					try:
						task = Task(**task_data)
						tasks.append(task)
					except Exception as e:
						self.logger.warning(f"Failed to parse task {task_data.get('id', 'unknown')}: {e}")
				
				# Update response data
				response.data = {"tasks": tasks}
				
				# Cache tasks
				if self.app and hasattr(self.app, 'app_state'):
					for task in tasks:
						self.app.app_state.cache_task(task)
			
			self.logger.info(f"Fetched {len(tasks_data) if response.success else 0} assigned tasks")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching assigned tasks: {e}")
			raise APIException(f"Failed to fetch assigned tasks: {e}")
	
	async def create_task(self, task_data: Dict[str, Any]) -> APIResponse:
		"""Create new task"""
		try:
			self.logger.info("Creating new task")
			
			# Validate task data
			self._validate_task_data(task_data)
			
			response = await self.api_service.post(
				URL_PATTERNS["tasks"]["create"],
				data=task_data
			)
			
			if response.success and response.data:
				try:
					task = Task(**response.data)
					response.data = task
					
					# Cache task
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_task(task)
					
				except Exception as e:
					self.logger.error(f"Failed to parse created task data: {e}")
					raise ValidationException(f"Invalid task data: {e}")
			
			self.logger.info(f"Created task: {response.data.id if response.success else 'failed'}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error creating task: {e}")
			raise APIException(f"Failed to create task: {e}")
	
	async def update_task(self, task_id: str, updates: Dict[str, Any]) -> APIResponse:
		"""Update existing task"""
		try:
			self.logger.info(f"Updating task: {task_id}")
			
			# Validate update data
			self._validate_task_updates(updates)
			
			response = await self.api_service.put(
				URL_PATTERNS["tasks"]["update"].format(task_id=task_id),
				data=updates
			)
			
			if response.success and response.data:
				try:
					task = Task(**response.data)
					response.data = task
					
					# Update cache
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_task(task)
					
				except Exception as e:
					self.logger.error(f"Failed to parse updated task data: {e}")
					raise ValidationException(f"Invalid task data: {e}")
			
			self.logger.info(f"Updated task: {task_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error updating task {task_id}: {e}")
			raise APIException(f"Failed to update task: {e}")
	
	async def delete_task(self, task_id: str) -> APIResponse:
		"""Delete task"""
		try:
			self.logger.info(f"Deleting task: {task_id}")
			
			response = await self.api_service.delete(
				URL_PATTERNS["tasks"]["delete"].format(task_id=task_id)
			)
			
			if response.success:
				# Remove from cache
				if self.app and hasattr(self.app, 'app_state'):
					self.app.app_state._tasks_cache.pop(task_id, None)
			
			self.logger.info(f"Deleted task: {task_id}")
			return response
			
		except Exception as e:
			self.logger.error(f"Error deleting task {task_id}: {e}")
			raise APIException(f"Failed to delete task: {e}")
	
	async def assign_task(self, task_id: str, assignee_id: str, 
						  due_date: Optional[datetime] = None) -> APIResponse:
		"""Assign task to user"""
		try:
			self.logger.info(f"Assigning task {task_id} to {assignee_id}")
			
			assignment_data = {
				"assignee_id": assignee_id,
				"due_date": due_date.isoformat() if due_date else None
			}
			
			response = await self.api_service.post(
				URL_PATTERNS["tasks"]["assign"].format(task_id=task_id),
				data=assignment_data
			)
			
			if response.success and response.data:
				try:
					task = Task(**response.data)
					response.data = task
					
					# Update cache
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_task(task)
					
				except Exception as e:
					self.logger.error(f"Failed to parse task data: {e}")
					raise ValidationException(f"Invalid task data: {e}")
			
			self.logger.info(f"Assigned task: {task_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error assigning task {task_id}: {e}")
			raise APIException(f"Failed to assign task: {e}")
	
	async def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> APIResponse:
		"""Complete task"""
		try:
			self.logger.info(f"Completing task: {task_id}")
			
			completion_data = {
				"result": result or {},
				"completed_at": datetime.utcnow().isoformat()
			}
			
			response = await self.api_service.post(
				URL_PATTERNS["tasks"]["complete"].format(task_id=task_id),
				data=completion_data
			)
			
			if response.success and response.data:
				try:
					task = Task(**response.data)
					response.data = task
					
					# Update cache
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_task(task)
					
				except Exception as e:
					self.logger.error(f"Failed to parse task data: {e}")
					raise ValidationException(f"Invalid task data: {e}")
			
			self.logger.info(f"Completed task: {task_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error completing task {task_id}: {e}")
			raise APIException(f"Failed to complete task: {e}")
	
	async def start_task(self, task_id: str) -> APIResponse:
		"""Start task execution"""
		try:
			self.logger.info(f"Starting task: {task_id}")
			
			start_data = {
				"started_at": datetime.utcnow().isoformat()
			}
			
			response = await self.api_service.post(
				f"/tasks/{task_id}/start",
				data=start_data
			)
			
			if response.success and response.data:
				try:
					task = Task(**response.data)
					response.data = task
					
					# Update cache
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_task(task)
					
				except Exception as e:
					self.logger.error(f"Failed to parse task data: {e}")
					raise ValidationException(f"Invalid task data: {e}")
			
			self.logger.info(f"Started task: {task_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error starting task {task_id}: {e}")
			raise APIException(f"Failed to start task: {e}")
	
	async def pause_task(self, task_id: str) -> APIResponse:
		"""Pause task execution"""
		try:
			self.logger.info(f"Pausing task: {task_id}")
			
			response = await self.api_service.post(f"/tasks/{task_id}/pause")
			
			if response.success and response.data:
				try:
					task = Task(**response.data)
					response.data = task
					
					# Update cache
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_task(task)
					
				except Exception as e:
					self.logger.error(f"Failed to parse task data: {e}")
					raise ValidationException(f"Invalid task data: {e}")
			
			self.logger.info(f"Paused task: {task_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error pausing task {task_id}: {e}")
			raise APIException(f"Failed to pause task: {e}")
	
	async def resume_task(self, task_id: str) -> APIResponse:
		"""Resume task execution"""
		try:
			self.logger.info(f"Resuming task: {task_id}")
			
			response = await self.api_service.post(f"/tasks/{task_id}/resume")
			
			if response.success and response.data:
				try:
					task = Task(**response.data)
					response.data = task
					
					# Update cache
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_task(task)
					
				except Exception as e:
					self.logger.error(f"Failed to parse task data: {e}")
					raise ValidationException(f"Invalid task data: {e}")
			
			self.logger.info(f"Resumed task: {task_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error resuming task {task_id}: {e}")
			raise APIException(f"Failed to resume task: {e}")
	
	async def cancel_task(self, task_id: str) -> APIResponse:
		"""Cancel task execution"""
		try:
			self.logger.info(f"Cancelling task: {task_id}")
			
			response = await self.api_service.post(f"/tasks/{task_id}/cancel")
			
			if response.success and response.data:
				try:
					task = Task(**response.data)
					response.data = task
					
					# Update cache
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_task(task)
					
				except Exception as e:
					self.logger.error(f"Failed to parse task data: {e}")
					raise ValidationException(f"Invalid task data: {e}")
			
			self.logger.info(f"Cancelled task: {task_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error cancelling task {task_id}: {e}")
			raise APIException(f"Failed to cancel task: {e}")
	
	async def update_task_progress(self, task_id: str, progress: float) -> APIResponse:
		"""Update task progress"""
		try:
			self.logger.info(f"Updating task progress: {task_id} -> {progress}%")
			
			if not 0 <= progress <= 100:
				raise ValidationException("Progress must be between 0 and 100")
			
			progress_data = {
				"progress": progress,
				"updated_at": datetime.utcnow().isoformat()
			}
			
			response = await self.api_service.patch(
				f"/tasks/{task_id}/progress",
				data=progress_data
			)
			
			if response.success and response.data:
				try:
					task = Task(**response.data)
					response.data = task
					
					# Update cache
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_task(task)
					
				except Exception as e:
					self.logger.error(f"Failed to parse task data: {e}")
					raise ValidationException(f"Invalid task data: {e}")
			
			self.logger.info(f"Updated task progress: {task_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error updating task progress {task_id}: {e}")
			raise APIException(f"Failed to update task progress: {e}")
	
	async def add_comment(self, task_id: str, comment: str, is_internal: bool = False) -> APIResponse:
		"""Add comment to task"""
		try:
			self.logger.info(f"Adding comment to task: {task_id}")
			
			if not comment.strip():
				raise ValidationException("Comment cannot be empty")
			
			comment_data = {
				"content": comment.strip(),
				"is_internal": is_internal,
				"created_at": datetime.utcnow().isoformat()
			}
			
			response = await self.api_service.post(
				URL_PATTERNS["tasks"]["comments"].format(task_id=task_id),
				data=comment_data
			)
			
			self.logger.info(f"Added comment to task: {task_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error adding comment to task {task_id}: {e}")
			raise APIException(f"Failed to add comment: {e}")
	
	async def get_comments(self, task_id: str) -> APIResponse:
		"""Get task comments"""
		try:
			self.logger.info(f"Fetching comments for task: {task_id}")
			
			response = await self.api_service.get(
				URL_PATTERNS["tasks"]["comments"].format(task_id=task_id)
			)
			
			self.logger.info(f"Fetched comments for task: {task_id}")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching comments for task {task_id}: {e}")
			raise APIException(f"Failed to fetch comments: {e}")
	
	async def upload_attachment(self, task_id: str, file_path: Path, 
								description: Optional[str] = None) -> APIResponse:
		"""Upload file attachment to task"""
		try:
			self.logger.info(f"Uploading attachment to task: {task_id}")
			
			if not file_path.exists():
				raise ValidationException(f"File not found: {file_path}")
			
			# Use API service upload method
			response = await self.api_service.upload_file(
				URL_PATTERNS["tasks"]["attachments"].format(task_id=task_id),
				file_path,
				field_name="attachment",
				data={"description": description} if description else None
			)
			
			self.logger.info(f"Uploaded attachment to task: {task_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error uploading attachment to task {task_id}: {e}")
			raise APIException(f"Failed to upload attachment: {e}")
	
	async def get_attachments(self, task_id: str) -> APIResponse:
		"""Get task attachments"""
		try:
			self.logger.info(f"Fetching attachments for task: {task_id}")
			
			response = await self.api_service.get(
				URL_PATTERNS["tasks"]["attachments"].format(task_id=task_id)
			)
			
			self.logger.info(f"Fetched attachments for task: {task_id}")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching attachments for task {task_id}: {e}")
			raise APIException(f"Failed to fetch attachments: {e}")
	
	async def add_watcher(self, task_id: str, user_id: str) -> APIResponse:
		"""Add user as task watcher"""
		try:
			self.logger.info(f"Adding watcher to task: {task_id}")
			
			watcher_data = {"user_id": user_id}
			
			response = await self.api_service.post(
				f"/tasks/{task_id}/watchers",
				data=watcher_data
			)
			
			self.logger.info(f"Added watcher to task: {task_id}")
			return response
			
		except Exception as e:
			self.logger.error(f"Error adding watcher to task {task_id}: {e}")
			raise APIException(f"Failed to add watcher: {e}")
	
	async def remove_watcher(self, task_id: str, user_id: str) -> APIResponse:
		"""Remove user as task watcher"""
		try:
			self.logger.info(f"Removing watcher from task: {task_id}")
			
			response = await self.api_service.delete(f"/tasks/{task_id}/watchers/{user_id}")
			
			self.logger.info(f"Removed watcher from task: {task_id}")
			return response
			
		except Exception as e:
			self.logger.error(f"Error removing watcher from task {task_id}: {e}")
			raise APIException(f"Failed to remove watcher: {e}")
	
	def _validate_task_data(self, task_data: Dict[str, Any]):
		"""Validate task data"""
		required_fields = ["name", "tenant_id", "owner_id"]
		
		for field in required_fields:
			if field not in task_data:
				raise ValidationException(f"Missing required field: {field}")
		
		# Validate name
		name = task_data.get("name", "").strip()
		if not name or len(name) < 3:
			raise ValidationException("Task name must be at least 3 characters long")
		
		if len(name) > 200:
			raise ValidationException("Task name must be less than 200 characters")
		
		# Validate tenant_id and owner_id
		for field in ["tenant_id", "owner_id"]:
			value = task_data.get(field, "").strip()
			if not value:
				raise ValidationException(f"{field} is required")
		
		# Validate priority if provided
		priority = task_data.get("priority")
		if priority is not None:
			try:
				TaskPriority(priority)
			except ValueError:
				raise ValidationException(f"Invalid task priority: {priority}")
		
		# Validate task_type if provided
		task_type = task_data.get("task_type")
		if task_type is not None:
			try:
				TaskType(task_type)
			except ValueError:
				raise ValidationException(f"Invalid task type: {task_type}")
		
		# Validate status if provided
		status = task_data.get("status")
		if status is not None:
			try:
				TaskStatus(status)
			except ValueError:
				raise ValidationException(f"Invalid task status: {status}")
		
		# Validate due_date if provided
		if "due_date" in task_data and task_data["due_date"]:
			try:
				datetime.fromisoformat(task_data["due_date"])
			except ValueError:
				raise ValidationException("Invalid due_date format. Use ISO format.")
	
	def _validate_task_updates(self, updates: Dict[str, Any]):
		"""Validate task update data"""
		# Validate name if provided
		if "name" in updates:
			name = updates["name"].strip()
			if not name or len(name) < 3:
				raise ValidationException("Task name must be at least 3 characters long")
			
			if len(name) > 200:
				raise ValidationException("Task name must be less than 200 characters")
		
		# Validate priority if provided
		if "priority" in updates:
			try:
				TaskPriority(updates["priority"])
			except ValueError:
				raise ValidationException(f"Invalid task priority: {updates['priority']}")
		
		# Validate status if provided
		if "status" in updates:
			try:
				TaskStatus(updates["status"])
			except ValueError:
				raise ValidationException(f"Invalid task status: {updates['status']}")
		
		# Validate due_date if provided
		if "due_date" in updates and updates["due_date"]:
			try:
				datetime.fromisoformat(updates["due_date"])
			except ValueError:
				raise ValidationException("Invalid due_date format. Use ISO format.")
	
	async def search_tasks(self, query: str, filters: Optional[Dict[str, Any]] = None) -> APIResponse:
		"""Search tasks by name, description, or content"""
		try:
			self.logger.info(f"Searching tasks: {query}")
			
			params = {"search": query}
			if filters:
				params.update(filters)
			
			return await self.get_tasks(params)
			
		except Exception as e:
			self.logger.error(f"Error searching tasks: {e}")
			raise APIException(f"Failed to search tasks: {e}")
	
	async def get_overdue_tasks(self) -> APIResponse:
		"""Get overdue tasks"""
		try:
			self.logger.info("Fetching overdue tasks")
			
			response = await self.api_service.get("/tasks/overdue")
			
			if response.success and response.data:
				# Convert task data to Task objects
				tasks_data = response.data.get("tasks", [])
				tasks = []
				
				for task_data in tasks_data:
					try:
						task = Task(**task_data)
						tasks.append(task)
					except Exception as e:
						self.logger.warning(f"Failed to parse task {task_data.get('id', 'unknown')}: {e}")
				
				response.data = {"tasks": tasks}
			
			self.logger.info(f"Fetched overdue tasks")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching overdue tasks: {e}")
			raise APIException(f"Failed to fetch overdue tasks: {e}")
	
	async def get_task_statistics(self) -> APIResponse:
		"""Get task statistics"""
		try:
			self.logger.info("Fetching task statistics")
			
			response = await self.api_service.get("/tasks/statistics")
			
			self.logger.info("Fetched task statistics")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching task statistics: {e}")
			raise APIException(f"Failed to fetch task statistics: {e}")