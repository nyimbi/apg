"""
Workflow Service for APG Workflow Mobile

Handles workflow management operations.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..models.workflow import Workflow, WorkflowInstance, WorkflowStatus, TriggerType
from ..models.api_response import APIResponse, PaginationInfo
from ..services.api_service import APIService
from ..utils.exceptions import APIException, ValidationException
from ..utils.constants import URL_PATTERNS


class WorkflowService:
	"""Service for workflow management operations"""
	
	def __init__(self, api_service: APIService, app=None):
		self.api_service = api_service
		self.app = app
		self.logger = logging.getLogger(__name__)
		
		self.logger.info("Workflow Service initialized")
	
	async def get_workflows(self, params: Optional[Dict[str, Any]] = None) -> APIResponse:
		"""Get list of workflows with optional filtering and pagination"""
		try:
			self.logger.info("Fetching workflows")
			
			query_params = params or {}
			
			# Add default pagination if not provided
			if "page" not in query_params:
				query_params["page"] = 1
			if "limit" not in query_params:
				query_params["limit"] = 20
			
			response = await self.api_service.get(
				URL_PATTERNS["workflows"]["list"],
				params=query_params
			)
			
			if response.success and response.data:
				# Convert workflow data to Workflow objects
				workflows_data = response.data.get("workflows", [])
				workflows = []
				
				for workflow_data in workflows_data:
					try:
						workflow = Workflow(**workflow_data)
						workflows.append(workflow)
					except Exception as e:
						self.logger.warning(f"Failed to parse workflow {workflow_data.get('id', 'unknown')}: {e}")
				
				# Update response data
				response.data = {"workflows": workflows}
				
				# Cache workflows if app state available
				if self.app and hasattr(self.app, 'app_state'):
					for workflow in workflows:
						self.app.app_state.cache_workflow(workflow)
			
			self.logger.info(f"Fetched {len(workflows_data) if response.success else 0} workflows")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching workflows: {e}")
			raise APIException(f"Failed to fetch workflows: {e}")
	
	async def get_workflow_by_id(self, workflow_id: str) -> APIResponse:
		"""Get workflow by ID"""
		try:
			self.logger.info(f"Fetching workflow: {workflow_id}")
			
			# Check cache first
			if self.app and hasattr(self.app, 'app_state'):
				cached_workflow = self.app.app_state.get_cached_workflow(workflow_id)
				if cached_workflow:
					self.logger.info(f"Returning cached workflow: {workflow_id}")
					return APIResponse.success_response(data=cached_workflow)
			
			response = await self.api_service.get(
				URL_PATTERNS["workflows"]["detail"].format(workflow_id=workflow_id)
			)
			
			if response.success and response.data:
				try:
					workflow = Workflow(**response.data)
					response.data = workflow
					
					# Cache workflow
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_workflow(workflow)
					
				except Exception as e:
					self.logger.error(f"Failed to parse workflow data: {e}")
					raise ValidationException(f"Invalid workflow data: {e}")
			
			self.logger.info(f"Fetched workflow: {workflow_id}")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching workflow {workflow_id}: {e}")
			raise APIException(f"Failed to fetch workflow: {e}")
	
	async def create_workflow(self, workflow_data: Dict[str, Any]) -> APIResponse:
		"""Create new workflow"""
		try:
			self.logger.info("Creating new workflow")
			
			# Validate workflow data
			self._validate_workflow_data(workflow_data)
			
			response = await self.api_service.post(
				URL_PATTERNS["workflows"]["create"],
				data=workflow_data
			)
			
			if response.success and response.data:
				try:
					workflow = Workflow(**response.data)
					response.data = workflow
					
					# Cache workflow
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_workflow(workflow)
					
				except Exception as e:
					self.logger.error(f"Failed to parse created workflow data: {e}")
					raise ValidationException(f"Invalid workflow data: {e}")
			
			self.logger.info(f"Created workflow: {response.data.id if response.success else 'failed'}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error creating workflow: {e}")
			raise APIException(f"Failed to create workflow: {e}")
	
	async def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> APIResponse:
		"""Update existing workflow"""
		try:
			self.logger.info(f"Updating workflow: {workflow_id}")
			
			# Validate update data
			self._validate_workflow_updates(updates)
			
			response = await self.api_service.put(
				URL_PATTERNS["workflows"]["update"].format(workflow_id=workflow_id),
				data=updates
			)
			
			if response.success and response.data:
				try:
					workflow = Workflow(**response.data)
					response.data = workflow
					
					# Update cache
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_workflow(workflow)
					
				except Exception as e:
					self.logger.error(f"Failed to parse updated workflow data: {e}")
					raise ValidationException(f"Invalid workflow data: {e}")
			
			self.logger.info(f"Updated workflow: {workflow_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error updating workflow {workflow_id}: {e}")
			raise APIException(f"Failed to update workflow: {e}")
	
	async def delete_workflow(self, workflow_id: str) -> APIResponse:
		"""Delete workflow"""
		try:
			self.logger.info(f"Deleting workflow: {workflow_id}")
			
			response = await self.api_service.delete(
				URL_PATTERNS["workflows"]["delete"].format(workflow_id=workflow_id)
			)
			
			if response.success:
				# Remove from cache
				if self.app and hasattr(self.app, 'app_state'):
					self.app.app_state._workflows_cache.pop(workflow_id, None)
			
			self.logger.info(f"Deleted workflow: {workflow_id}")
			return response
			
		except Exception as e:
			self.logger.error(f"Error deleting workflow {workflow_id}: {e}")
			raise APIException(f"Failed to delete workflow: {e}")
	
	async def execute_workflow(self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None, 
							   priority: Optional[int] = None) -> APIResponse:
		"""Execute workflow"""
		try:
			self.logger.info(f"Executing workflow: {workflow_id}")
			
			execution_data = {
				"input_data": input_data or {},
				"priority": priority or 5,
				"triggered_by": "mobile_app"
			}
			
			response = await self.api_service.post(
				URL_PATTERNS["workflows"]["execute"].format(workflow_id=workflow_id),
				data=execution_data
			)
			
			if response.success and response.data:
				try:
					instance = WorkflowInstance(**response.data)
					response.data = instance
					
					# Cache instance
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state._workflow_instances_cache[instance.id] = instance
					
				except Exception as e:
					self.logger.error(f"Failed to parse workflow instance data: {e}")
					raise ValidationException(f"Invalid workflow instance data: {e}")
			
			self.logger.info(f"Executed workflow: {workflow_id}, instance: {response.data.id if response.success else 'failed'}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error executing workflow {workflow_id}: {e}")
			raise APIException(f"Failed to execute workflow: {e}")
	
	async def get_workflow_instances(self, params: Optional[Dict[str, Any]] = None) -> APIResponse:
		"""Get workflow instances"""
		try:
			self.logger.info("Fetching workflow instances")
			
			query_params = params or {}
			
			# Add default pagination if not provided
			if "page" not in query_params:
				query_params["page"] = 1
			if "limit" not in query_params:
				query_params["limit"] = 20
			
			# Determine URL based on whether workflow_id is provided
			if "workflow_id" in query_params:
				url = URL_PATTERNS["workflows"]["instances"].format(workflow_id=query_params["workflow_id"])
				query_params.pop("workflow_id")
			else:
				url = "/workflow-instances"  # Generic instances endpoint
			
			response = await self.api_service.get(url, params=query_params)
			
			if response.success and response.data:
				# Convert instance data to WorkflowInstance objects
				instances_data = response.data.get("instances", [])
				instances = []
				
				for instance_data in instances_data:
					try:
						instance = WorkflowInstance(**instance_data)
						instances.append(instance)
					except Exception as e:
						self.logger.warning(f"Failed to parse workflow instance {instance_data.get('id', 'unknown')}: {e}")
				
				# Update response data
				response.data = {"instances": instances}
				
				# Cache instances
				if self.app and hasattr(self.app, 'app_state'):
					for instance in instances:
						self.app.app_state._workflow_instances_cache[instance.id] = instance
			
			self.logger.info(f"Fetched {len(instances_data) if response.success else 0} workflow instances")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching workflow instances: {e}")
			raise APIException(f"Failed to fetch workflow instances: {e}")
	
	async def get_instance_by_id(self, instance_id: str) -> APIResponse:
		"""Get workflow instance by ID"""
		try:
			self.logger.info(f"Fetching workflow instance: {instance_id}")
			
			# Check cache first
			if self.app and hasattr(self.app, 'app_state'):
				cached_instance = self.app.app_state._workflow_instances_cache.get(instance_id)
				if cached_instance:
					self.logger.info(f"Returning cached workflow instance: {instance_id}")
					return APIResponse.success_response(data=cached_instance)
			
			response = await self.api_service.get(
				URL_PATTERNS["workflows"]["instance_detail"].format(instance_id=instance_id)
			)
			
			if response.success and response.data:
				try:
					instance = WorkflowInstance(**response.data)
					response.data = instance
					
					# Cache instance
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state._workflow_instances_cache[instance.id] = instance
					
				except Exception as e:
					self.logger.error(f"Failed to parse workflow instance data: {e}")
					raise ValidationException(f"Invalid workflow instance data: {e}")
			
			self.logger.info(f"Fetched workflow instance: {instance_id}")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching workflow instance {instance_id}: {e}")
			raise APIException(f"Failed to fetch workflow instance: {e}")
	
	async def cancel_instance(self, instance_id: str) -> APIResponse:
		"""Cancel workflow instance"""
		try:
			self.logger.info(f"Cancelling workflow instance: {instance_id}")
			
			response = await self.api_service.post(
				f"/workflow-instances/{instance_id}/cancel"
			)
			
			if response.success and response.data:
				try:
					instance = WorkflowInstance(**response.data)
					response.data = instance
					
					# Update cache
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state._workflow_instances_cache[instance.id] = instance
					
				except Exception as e:
					self.logger.error(f"Failed to parse workflow instance data: {e}")
					raise ValidationException(f"Invalid workflow instance data: {e}")
			
			self.logger.info(f"Cancelled workflow instance: {instance_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error cancelling workflow instance {instance_id}: {e}")
			raise APIException(f"Failed to cancel workflow instance: {e}")
	
	async def pause_instance(self, instance_id: str) -> APIResponse:
		"""Pause workflow instance"""
		try:
			self.logger.info(f"Pausing workflow instance: {instance_id}")
			
			response = await self.api_service.post(
				f"/workflow-instances/{instance_id}/pause"
			)
			
			if response.success and response.data:
				try:
					instance = WorkflowInstance(**response.data)
					response.data = instance
					
					# Update cache
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state._workflow_instances_cache[instance.id] = instance
					
				except Exception as e:
					self.logger.error(f"Failed to parse workflow instance data: {e}")
					raise ValidationException(f"Invalid workflow instance data: {e}")
			
			self.logger.info(f"Paused workflow instance: {instance_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error pausing workflow instance {instance_id}: {e}")
			raise APIException(f"Failed to pause workflow instance: {e}")
	
	async def resume_instance(self, instance_id: str) -> APIResponse:
		"""Resume workflow instance"""
		try:
			self.logger.info(f"Resuming workflow instance: {instance_id}")
			
			response = await self.api_service.post(
				f"/workflow-instances/{instance_id}/resume"
			)
			
			if response.success and response.data:
				try:
					instance = WorkflowInstance(**response.data)
					response.data = instance
					
					# Update cache
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state._workflow_instances_cache[instance.id] = instance
					
				except Exception as e:
					self.logger.error(f"Failed to parse workflow instance data: {e}")
					raise ValidationException(f"Invalid workflow instance data: {e}")
			
			self.logger.info(f"Resumed workflow instance: {instance_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error resuming workflow instance {instance_id}: {e}")
			raise APIException(f"Failed to resume workflow instance: {e}")
	
	async def duplicate_workflow(self, workflow_id: str, name: Optional[str] = None) -> APIResponse:
		"""Duplicate existing workflow"""
		try:
			self.logger.info(f"Duplicating workflow: {workflow_id}")
			
			duplicate_data = {}
			if name:
				duplicate_data["name"] = name
			
			response = await self.api_service.post(
				f"/workflows/{workflow_id}/duplicate",
				data=duplicate_data
			)
			
			if response.success and response.data:
				try:
					workflow = Workflow(**response.data)
					response.data = workflow
					
					# Cache workflow
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_workflow(workflow)
					
				except Exception as e:
					self.logger.error(f"Failed to parse duplicated workflow data: {e}")
					raise ValidationException(f"Invalid workflow data: {e}")
			
			self.logger.info(f"Duplicated workflow: {workflow_id} -> {response.data.id if response.success else 'failed'}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error duplicating workflow {workflow_id}: {e}")
			raise APIException(f"Failed to duplicate workflow: {e}")
	
	def _validate_workflow_data(self, workflow_data: Dict[str, Any]):
		"""Validate workflow data"""
		required_fields = ["name", "tenant_id"]
		
		for field in required_fields:
			if field not in workflow_data:
				raise ValidationException(f"Missing required field: {field}")
		
		# Validate name
		name = workflow_data.get("name", "").strip()
		if not name or len(name) < 3:
			raise ValidationException("Workflow name must be at least 3 characters long")
		
		if len(name) > 200:
			raise ValidationException("Workflow name must be less than 200 characters")
		
		# Validate tenant_id
		tenant_id = workflow_data.get("tenant_id", "").strip()
		if not tenant_id:
			raise ValidationException("Tenant ID is required")
		
		# Validate priority if provided
		priority = workflow_data.get("priority")
		if priority is not None:
			if not isinstance(priority, int) or priority < 1 or priority > 10:
				raise ValidationException("Priority must be an integer between 1 and 10")
		
		# Validate status if provided
		status = workflow_data.get("status")
		if status is not None:
			try:
				WorkflowStatus(status)
			except ValueError:
				raise ValidationException(f"Invalid workflow status: {status}")
	
	def _validate_workflow_updates(self, updates: Dict[str, Any]):
		"""Validate workflow update data"""
		# Validate name if provided
		if "name" in updates:
			name = updates["name"].strip()
			if not name or len(name) < 3:
				raise ValidationException("Workflow name must be at least 3 characters long")
			
			if len(name) > 200:
				raise ValidationException("Workflow name must be less than 200 characters")
		
		# Validate priority if provided
		if "priority" in updates:
			priority = updates["priority"]
			if not isinstance(priority, int) or priority < 1 or priority > 10:
				raise ValidationException("Priority must be an integer between 1 and 10")
		
		# Validate status if provided
		if "status" in updates:
			try:
				WorkflowStatus(updates["status"])
			except ValueError:
				raise ValidationException(f"Invalid workflow status: {updates['status']}")
	
	async def search_workflows(self, query: str, filters: Optional[Dict[str, Any]] = None) -> APIResponse:
		"""Search workflows by name, description, or tags"""
		try:
			self.logger.info(f"Searching workflows: {query}")
			
			params = {"search": query}
			if filters:
				params.update(filters)
			
			return await self.get_workflows(params)
			
		except Exception as e:
			self.logger.error(f"Error searching workflows: {e}")
			raise APIException(f"Failed to search workflows: {e}")
	
	async def get_workflow_templates(self) -> APIResponse:
		"""Get available workflow templates"""
		try:
			self.logger.info("Fetching workflow templates")
			
			response = await self.api_service.get("/workflow-templates")
			
			self.logger.info(f"Fetched workflow templates")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching workflow templates: {e}")
			raise APIException(f"Failed to fetch workflow templates: {e}")
	
	async def create_from_template(self, template_id: str, workflow_data: Dict[str, Any]) -> APIResponse:
		"""Create workflow from template"""
		try:
			self.logger.info(f"Creating workflow from template: {template_id}")
			
			# Validate workflow data
			self._validate_workflow_data(workflow_data)
			
			data = {
				"template_id": template_id,
				**workflow_data
			}
			
			response = await self.api_service.post("/workflow-templates/create", data=data)
			
			if response.success and response.data:
				try:
					workflow = Workflow(**response.data)
					response.data = workflow
					
					# Cache workflow
					if self.app and hasattr(self.app, 'app_state'):
						self.app.app_state.cache_workflow(workflow)
					
				except Exception as e:
					self.logger.error(f"Failed to parse workflow data: {e}")
					raise ValidationException(f"Invalid workflow data: {e}")
			
			self.logger.info(f"Created workflow from template: {template_id}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error creating workflow from template {template_id}: {e}")
			raise APIException(f"Failed to create workflow from template: {e}")