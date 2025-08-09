"""
APG Workflow & Business Process Management - API Documentation Generator

Comprehensive API documentation generation with interactive examples,
authentication details, and integration guides.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import inspect
import yaml

from models import (
	APGTenantContext, WBPMServiceResponse, WBPMPagedResponse,
	WBPMProcessInstance, WBPMTask, ProcessStatus, TaskStatus, TaskPriority
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# API Documentation Core Classes
# =============================================================================

class APIMethodType(str, Enum):
	"""HTTP method types."""
	GET = "GET"
	POST = "POST"
	PUT = "PUT"
	DELETE = "DELETE"
	PATCH = "PATCH"


class ParameterType(str, Enum):
	"""Parameter types."""
	PATH = "path"
	QUERY = "query"
	BODY = "body"
	HEADER = "header"
	FORM = "form"


class ResponseFormat(str, Enum):
	"""Response format types."""
	JSON = "application/json"
	XML = "application/xml"
	TEXT = "text/plain"
	BINARY = "application/octet-stream"


@dataclass
class APIParameter:
	"""API parameter definition."""
	name: str = ""
	parameter_type: ParameterType = ParameterType.QUERY
	data_type: str = "string"
	required: bool = False
	description: str = ""
	example: Any = None
	default_value: Any = None
	enum_values: List[str] = field(default_factory=list)
	validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIResponse:
	"""API response definition."""
	status_code: int = 200
	description: str = ""
	content_type: ResponseFormat = ResponseFormat.JSON
	schema: Dict[str, Any] = field(default_factory=dict)
	example: Dict[str, Any] = field(default_factory=dict)
	headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class APIEndpoint:
	"""API endpoint definition."""
	endpoint_id: str = field(default_factory=lambda: f"endpoint_{uuid.uuid4().hex}")
	path: str = ""
	method: APIMethodType = APIMethodType.GET
	summary: str = ""
	description: str = ""
	tags: List[str] = field(default_factory=list)
	parameters: List[APIParameter] = field(default_factory=list)
	request_body: Optional[Dict[str, Any]] = None
	responses: List[APIResponse] = field(default_factory=list)
	authentication_required: bool = True
	permissions_required: List[str] = field(default_factory=list)
	rate_limit: Optional[int] = None
	examples: List[Dict[str, Any]] = field(default_factory=list)
	deprecated: bool = False
	version: str = "1.0"


@dataclass
class APIDocumentationSection:
	"""Documentation section."""
	section_id: str = field(default_factory=lambda: f"section_{uuid.uuid4().hex}")
	title: str = ""
	content: str = ""
	subsections: List['APIDocumentationSection'] = field(default_factory=list)
	code_examples: List[Dict[str, str]] = field(default_factory=list)
	order: int = 0


@dataclass
class APISchema:
	"""API schema definition."""
	schema_id: str = field(default_factory=lambda: f"schema_{uuid.uuid4().hex}")
	name: str = ""
	description: str = ""
	type: str = "object"
	properties: Dict[str, Any] = field(default_factory=dict)
	required_fields: List[str] = field(default_factory=list)
	example: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# WBPM API Definitions
# =============================================================================

class WBPMAPIDefinitions:
	"""Define all WBPM API endpoints."""
	
	def __init__(self):
		self.endpoints: List[APIEndpoint] = []
		self.schemas: List[APISchema] = []
		self._define_core_schemas()
		self._define_process_endpoints()
		self._define_task_endpoints()
		self._define_analytics_endpoints()
		self._define_template_endpoints()
		self._define_monitoring_endpoints()
		self._define_notification_endpoints()
	
	def _define_core_schemas(self) -> None:
		"""Define core API schemas."""
		# APGTenantContext schema
		self.schemas.append(APISchema(
			name="APGTenantContext",
			description="APG tenant context for multi-tenant operations",
			properties={
				"tenant_id": {"type": "string", "description": "Unique tenant identifier"},
				"user_id": {"type": "string", "description": "Current user identifier"},
				"permissions": {"type": "array", "items": {"type": "string"}, "description": "User permissions"},
				"session_id": {"type": "string", "description": "Current session identifier"}
			},
			required_fields=["tenant_id", "user_id"],
			example={
				"tenant_id": "tenant_abc123",
				"user_id": "user_xyz789",
				"permissions": ["wbpm:process:read", "wbpm:task:write"],
				"session_id": "session_def456"
			}
		))
		
		# WBPMServiceResponse schema
		self.schemas.append(APISchema(
			name="WBPMServiceResponse",
			description="Standard WBPM service response format",
			properties={
				"success": {"type": "boolean", "description": "Operation success status"},
				"message": {"type": "string", "description": "Human-readable response message"},
				"data": {"type": "object", "description": "Response data payload"},
				"errors": {"type": "array", "items": {"type": "string"}, "description": "Error messages if any"},
				"metadata": {"type": "object", "description": "Additional response metadata"}
			},
			required_fields=["success", "message"],
			example={
				"success": True,
				"message": "Process created successfully",
				"data": {"process_id": "proc_123", "status": "active"},
				"errors": [],
				"metadata": {"execution_time_ms": 45}
			}
		))
		
		# Process Definition schema
		self.schemas.append(APISchema(
			name="ProcessDefinition",
			description="Workflow process definition",
			properties={
				"id": {"type": "string", "description": "Unique process definition identifier"},
				"process_key": {"type": "string", "description": "Process key for identification"},
				"process_name": {"type": "string", "description": "Human-readable process name"},
				"process_description": {"type": "string", "description": "Process description"},
				"bpmn_xml": {"type": "string", "description": "BPMN 2.0 XML definition"},
				"version": {"type": "integer", "description": "Process version number"},
				"category": {"type": "string", "description": "Process category"},
				"tags": {"type": "array", "items": {"type": "string"}, "description": "Process tags"},
				"created_at": {"type": "string", "format": "date-time", "description": "Creation timestamp"},
				"updated_at": {"type": "string", "format": "date-time", "description": "Last update timestamp"}
			},
			required_fields=["process_key", "process_name", "bpmn_xml"],
			example={
				"id": "proc_def_123",
				"process_key": "employee_onboarding",
				"process_name": "Employee Onboarding Process",
				"process_description": "Standard process for onboarding new employees",
				"bpmn_xml": "<?xml version=\"1.0\" encoding=\"UTF-8\"?>...",
				"version": 1,
				"category": "hr_process",
				"tags": ["onboarding", "hr", "employee"],
				"created_at": "2025-01-15T10:30:00Z",
				"updated_at": "2025-01-15T10:30:00Z"
			}
		))
	
	def _define_process_endpoints(self) -> None:
		"""Define process management endpoints."""
		# Create Process Definition
		self.endpoints.append(APIEndpoint(
			path="/api/v1/processes/definitions",
			method=APIMethodType.POST,
			summary="Create Process Definition",
			description="Create a new workflow process definition with BPMN 2.0 specification",
			tags=["Process Management"],
			parameters=[
				APIParameter(
					name="X-Tenant-ID",
					parameter_type=ParameterType.HEADER,
					data_type="string",
					required=True,
					description="Tenant identifier for multi-tenant operation"
				)
			],
			request_body={
				"content": {
					"application/json": {
						"schema": {"$ref": "#/components/schemas/ProcessDefinition"},
						"example": {
							"process_key": "invoice_approval",
							"process_name": "Invoice Approval Process",
							"process_description": "Automated invoice approval workflow",
							"bpmn_xml": "<?xml version=\"1.0\" encoding=\"UTF-8\"?>...",
							"category": "finance",
							"tags": ["finance", "approval", "invoice"]
						}
					}
				}
			},
			responses=[
				APIResponse(
					status_code=201,
					description="Process definition created successfully",
					example={
						"success": True,
						"message": "Process definition created successfully",
						"data": {
							"process_id": "proc_def_456",
							"process_key": "invoice_approval",
							"version": 1
						}
					}
				),
				APIResponse(
					status_code=400,
					description="Invalid request data",
					example={
						"success": False,
						"message": "Invalid BPMN XML format",
						"errors": ["BPMN validation failed: Missing start event"]
					}
				)
			],
			permissions_required=["wbpm:process:create"],
			examples=[
				{
					"title": "Basic Process Creation",
					"description": "Create a simple approval process",
					"request": {
						"process_key": "document_review",
						"process_name": "Document Review Process",
						"bpmn_xml": "<?xml version=\"1.0\"?>...",
						"tags": ["review", "document"]
					}
				}
			]
		))
		
		# Start Process Instance
		self.endpoints.append(APIEndpoint(
			path="/api/v1/processes/instances",
			method=APIMethodType.POST,
			summary="Start Process Instance",
			description="Start a new instance of a workflow process",
			tags=["Process Execution"],
			parameters=[
				APIParameter(
					name="process_key",
					parameter_type=ParameterType.QUERY,
					data_type="string",
					required=True,
					description="Process definition key to instantiate"
				),
				APIParameter(
					name="business_key",
					parameter_type=ParameterType.QUERY,
					data_type="string",
					required=False,
					description="Business key for the process instance"
				)
			],
			request_body={
				"content": {
					"application/json": {
						"schema": {
							"type": "object",
							"properties": {
								"variables": {"type": "object", "description": "Process variables"},
								"priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
								"assignee": {"type": "string", "description": "Initial assignee"}
							}
						},
						"example": {
							"variables": {
								"invoice_amount": 1500.00,
								"vendor_name": "Acme Corp",
								"requester": "john.doe@company.com"
							},
							"priority": "normal",
							"assignee": "finance.team@company.com"
						}
					}
				}
			},
			responses=[
				APIResponse(
					status_code=201,
					description="Process instance started successfully",
					example={
						"success": True,
						"message": "Process instance started successfully",
						"data": {
							"instance_id": "inst_789",
							"process_key": "invoice_approval",
							"status": "active",
							"started_at": "2025-01-15T11:00:00Z"
						}
					}
				)
			],
			permissions_required=["wbpm:process:execute"]
		))
		
		# Get Process Instance
		self.endpoints.append(APIEndpoint(
			path="/api/v1/processes/instances/{instance_id}",
			method=APIMethodType.GET,
			summary="Get Process Instance",
			description="Retrieve details of a specific process instance",
			tags=["Process Execution"],
			parameters=[
				APIParameter(
					name="instance_id",
					parameter_type=ParameterType.PATH,
					data_type="string",
					required=True,
					description="Process instance identifier"
				),
				APIParameter(
					name="include_tasks",
					parameter_type=ParameterType.QUERY,
					data_type="boolean",
					required=False,
					description="Include active tasks in response",
					default_value=False
				)
			],
			responses=[
				APIResponse(
					status_code=200,
					description="Process instance details retrieved successfully",
					example={
						"success": True,
						"message": "Process instance retrieved successfully",
						"data": {
							"instance_id": "inst_789",
							"process_key": "invoice_approval",
							"status": "active",
							"variables": {"invoice_amount": 1500.00},
							"started_at": "2025-01-15T11:00:00Z",
							"tasks": [
								{
									"task_id": "task_001",
									"task_name": "Review Invoice",
									"assignee": "reviewer@company.com",
									"status": "active"
								}
							]
						}
					}
				)
			],
			permissions_required=["wbpm:process:read"]
		))
	
	def _define_task_endpoints(self) -> None:
		"""Define task management endpoints."""
		# Get User Tasks
		self.endpoints.append(APIEndpoint(
			path="/api/v1/tasks/user/{user_id}",
			method=APIMethodType.GET,
			summary="Get User Tasks",
			description="Retrieve tasks assigned to a specific user",
			tags=["Task Management"],
			parameters=[
				APIParameter(
					name="user_id",
					parameter_type=ParameterType.PATH,
					data_type="string",
					required=True,
					description="User identifier"
				),
				APIParameter(
					name="status",
					parameter_type=ParameterType.QUERY,
					data_type="string",
					required=False,
					description="Filter by task status",
					enum_values=["active", "completed", "cancelled", "escalated"]
				),
				APIParameter(
					name="priority",
					parameter_type=ParameterType.QUERY,
					data_type="string",
					required=False,
					description="Filter by task priority",
					enum_values=["low", "normal", "high", "urgent"]
				),
				APIParameter(
					name="page",
					parameter_type=ParameterType.QUERY,
					data_type="integer",
					required=False,
					description="Page number for pagination",
					default_value=1
				),
				APIParameter(
					name="page_size",
					parameter_type=ParameterType.QUERY,
					data_type="integer",
					required=False,
					description="Number of items per page",
					default_value=20,
					validation_rules={"minimum": 1, "maximum": 100}
				)
			],
			responses=[
				APIResponse(
					status_code=200,
					description="User tasks retrieved successfully",
					example={
						"success": True,
						"message": "Tasks retrieved successfully",
						"data": {
							"tasks": [
								{
									"task_id": "task_001",
									"task_name": "Review Invoice",
									"process_name": "Invoice Approval",
									"priority": "normal",
									"status": "active",
									"due_date": "2025-01-20T17:00:00Z",
									"assigned_at": "2025-01-15T11:00:00Z"
								}
							],
							"pagination": {
								"page": 1,
								"page_size": 20,
								"total_count": 5,
								"total_pages": 1
							}
						}
					}
				)
			],
			permissions_required=["wbpm:task:read"]
		))
		
		# Complete Task
		self.endpoints.append(APIEndpoint(
			path="/api/v1/tasks/{task_id}/complete",
			method=APIMethodType.POST,
			summary="Complete Task",
			description="Mark a task as completed with optional output variables",
			tags=["Task Management"],
			parameters=[
				APIParameter(
					name="task_id",
					parameter_type=ParameterType.PATH,
					data_type="string",
					required=True,
					description="Task identifier"
				)
			],
			request_body={
				"content": {
					"application/json": {
						"schema": {
							"type": "object",
							"properties": {
								"variables": {"type": "object", "description": "Task output variables"},
								"comment": {"type": "string", "description": "Completion comment"}
							}
						},
						"example": {
							"variables": {
								"approval_status": "approved",
								"approval_amount": 1500.00,
								"comments": "Invoice approved for payment"
							},
							"comment": "Reviewed and approved invoice"
						}
					}
				}
			},
			responses=[
				APIResponse(
					status_code=200,
					description="Task completed successfully",
					example={
						"success": True,
						"message": "Task completed successfully",
						"data": {
							"task_id": "task_001",
							"completed_at": "2025-01-16T14:30:00Z",
							"next_tasks": ["task_002"]
						}
					}
				)
			],
			permissions_required=["wbpm:task:complete"]
		))
	
	def _define_analytics_endpoints(self) -> None:
		"""Define analytics and reporting endpoints."""
		# Get Process Analytics
		self.endpoints.append(APIEndpoint(
			path="/api/v1/analytics/processes",
			method=APIMethodType.GET,
			summary="Get Process Analytics",
			description="Retrieve comprehensive process performance analytics",
			tags=["Analytics"],
			parameters=[
				APIParameter(
					name="process_key",
					parameter_type=ParameterType.QUERY,
					data_type="string",
					required=False,
					description="Filter by specific process key"
				),
				APIParameter(
					name="time_range",
					parameter_type=ParameterType.QUERY,
					data_type="string",
					required=False,
					description="Time range for analytics",
					enum_values=["1d", "7d", "30d", "90d"],
					default_value="30d"
				),
				APIParameter(
					name="metrics",
					parameter_type=ParameterType.QUERY,
					data_type="string",
					required=False,
					description="Comma-separated list of metrics to include",
					example="duration,throughput,error_rate"
				)
			],
			responses=[
				APIResponse(
					status_code=200,
					description="Process analytics retrieved successfully",
					example={
						"success": True,
						"message": "Analytics retrieved successfully",
						"data": {
							"time_range": "30d",
							"total_instances": 245,
							"completed_instances": 230,
							"avg_duration_hours": 4.2,
							"throughput_per_day": 8.2,
							"error_rate": 0.02,
							"bottlenecks": [
								{
									"activity": "Manager Approval",
									"avg_wait_time": 2.1,
									"instances_affected": 45
								}
							],
							"trends": {
								"duration": "stable",
								"throughput": "improving",
								"quality": "stable"
							}
						}
					}
				)
			],
			permissions_required=["wbpm:analytics:read"]
		))
	
	def _define_template_endpoints(self) -> None:
		"""Define template management endpoints."""
		# Get Template Recommendations
		self.endpoints.append(APIEndpoint(
			path="/api/v1/templates/recommendations",
			method=APIMethodType.GET,
			summary="Get Template Recommendations",
			description="Get AI-powered template recommendations for user",
			tags=["Templates"],
			parameters=[
				APIParameter(
					name="category",
					parameter_type=ParameterType.QUERY,
					data_type="string",
					required=False,
					description="Filter by template category",
					enum_values=["business_process", "approval_workflow", "data_processing"]
				),
				APIParameter(
					name="limit",
					parameter_type=ParameterType.QUERY,
					data_type="integer",
					required=False,
					description="Maximum number of recommendations",
					default_value=10,
					validation_rules={"minimum": 1, "maximum": 50}
				)
			],
			responses=[
				APIResponse(
					status_code=200,
					description="Template recommendations retrieved successfully",
					example={
						"success": True,
						"message": "Recommendations generated successfully",
						"data": {
							"recommendations": [
								{
									"template_id": "tmpl_123",
									"template_name": "Employee Onboarding",
									"category": "hr_process",
									"recommendation_score": 0.85,
									"reason": "Matches your usage patterns and highly rated"
								}
							],
							"total_count": 5
						}
					}
				)
			],
			permissions_required=["wbpm:template:read"]
		))
	
	def _define_monitoring_endpoints(self) -> None:
		"""Define monitoring and alerting endpoints."""
		# Get Real-time Metrics
		self.endpoints.append(APIEndpoint(
			path="/api/v1/monitoring/metrics",
			method=APIMethodType.GET,
			summary="Get Real-time Metrics",
			description="Retrieve real-time system and process metrics",
			tags=["Monitoring"],
			parameters=[
				APIParameter(
					name="metric_type",
					parameter_type=ParameterType.QUERY,
					data_type="string",
					required=False,
					description="Type of metrics to retrieve",
					enum_values=["system", "process", "task", "user"]
				),
				APIParameter(
					name="time_window",
					parameter_type=ParameterType.QUERY,
					data_type="string",
					required=False,
					description="Time window for metrics",
					enum_values=["5m", "15m", "1h", "4h", "24h"],
					default_value="1h"
				)
			],
			responses=[
				APIResponse(
					status_code=200,
					description="Metrics retrieved successfully",
					example={
						"success": True,
						"message": "Metrics retrieved successfully",
						"data": {
							"time_window": "1h",
							"active_processes": 25,
							"active_tasks": 48,
							"avg_response_time": 1.2,
							"throughput": 15.5,
							"error_rate": 0.01,
							"alerts": [
								{
									"alert_id": "alert_001",
									"severity": "medium",
									"message": "High queue time detected"
								}
							]
						}
					}
				)
			],
			permissions_required=["wbpm:monitoring:read"]
		))
	
	def _define_notification_endpoints(self) -> None:
		"""Define notification management endpoints."""
		# Send Custom Notification
		self.endpoints.append(APIEndpoint(
			path="/api/v1/notifications/send",
			method=APIMethodType.POST,
			summary="Send Custom Notification",
			description="Send a custom notification through specified channels",
			tags=["Notifications"],
			request_body={
				"content": {
					"application/json": {
						"schema": {
							"type": "object",
							"properties": {
								"recipient_id": {"type": "string", "description": "Recipient user ID"},
								"channels": {"type": "array", "items": {"type": "string"}, "description": "Notification channels"},
								"priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
								"subject": {"type": "string", "description": "Notification subject"},
								"message": {"type": "string", "description": "Notification message"},
								"process_id": {"type": "string", "description": "Related process ID"}
							},
							"required": ["recipient_id", "channels", "subject", "message"]
						},
						"example": {
							"recipient_id": "user_123",
							"channels": ["email", "in_app"],
							"priority": "high",
							"subject": "Process Approval Required",
							"message": "Your approval is required for the invoice process",
							"process_id": "proc_456"
						}
					}
				}
			},
			responses=[
				APIResponse(
					status_code=202,
					description="Notification queued for delivery",
					example={
						"success": True,
						"message": "Notification queued successfully",
						"data": {
							"notification_id": "notif_789",
							"estimated_delivery": "2025-01-15T12:05:00Z"
						}
					}
				)
			],
			permissions_required=["wbpm:notification:send"]
		))


# =============================================================================
# Documentation Generator
# =============================================================================

class APIDocumentationGenerator:
	"""Generate comprehensive API documentation."""
	
	def __init__(self):
		self.api_definitions = WBPMAPIDefinitions()
		self.base_info = self._get_base_info()
		
	def _get_base_info(self) -> Dict[str, Any]:
		"""Get base API information."""
		return {
			"title": "APG Workflow & Business Process Management API",
			"description": "Comprehensive API for workflow automation, business process management, and intelligent task routing within the APG platform ecosystem.",
			"version": "1.0.0",
			"contact": {
				"name": "Datacraft API Support",
				"email": "nyimbi@gmail.com",
				"url": "https://www.datacraft.co.ke"
			},
			"license": {
				"name": "Proprietary",
				"url": "https://www.datacraft.co.ke/license"
			},
			"servers": [
				{
					"url": "https://api.datacraft.co.ke/wbpm",
					"description": "Production server"
				},
				{
					"url": "https://staging-api.datacraft.co.ke/wbpm",
					"description": "Staging server"
				},
				{
					"url": "http://localhost:8000/wbpm",
					"description": "Development server"
				}
			],
			"security_schemes": {
				"BearerAuth": {
					"type": "http",
					"scheme": "bearer",
					"bearer_format": "JWT",
					"description": "JWT token from APG authentication service"
				},
				"ApiKeyAuth": {
					"type": "apiKey",
					"in": "header",
					"name": "X-API-Key",
					"description": "API key for service-to-service authentication"
				}
			}
		}
	
	async def generate_openapi_spec(self) -> Dict[str, Any]:
		"""Generate OpenAPI 3.0 specification."""
		try:
			spec = {
				"openapi": "3.0.3",
				"info": self.base_info,
				"servers": self.base_info["servers"],
				"security": [
					{"BearerAuth": []},
					{"ApiKeyAuth": []}
				],
				"components": {
					"securitySchemes": self.base_info["security_schemes"],
					"schemas": {},
					"parameters": {},
					"responses": {}
				},
				"paths": {},
				"tags": [
					{"name": "Process Management", "description": "Process definition and lifecycle management"},
					{"name": "Process Execution", "description": "Process instance execution and monitoring"},
					{"name": "Task Management", "description": "Task assignment, completion, and routing"},
					{"name": "Analytics", "description": "Process analytics and performance insights"},
					{"name": "Templates", "description": "Process template management and recommendations"},
					{"name": "Monitoring", "description": "Real-time monitoring and alerting"},
					{"name": "Notifications", "description": "Notification management and delivery"}
				]
			}
			
			# Add schemas
			for schema in self.api_definitions.schemas:
				spec["components"]["schemas"][schema.name] = {
					"type": schema.type,
					"description": schema.description,
					"properties": schema.properties,
					"required": schema.required_fields,
					"example": schema.example
				}
			
			# Add endpoints
			for endpoint in self.api_definitions.endpoints:
				path = endpoint.path
				method = endpoint.method.value.lower()
				
				if path not in spec["paths"]:
					spec["paths"][path] = {}
				
				# Build endpoint specification
				endpoint_spec = {
					"summary": endpoint.summary,
					"description": endpoint.description,
					"tags": endpoint.tags,
					"security": [{"BearerAuth": []}] if endpoint.authentication_required else [],
					"parameters": [],
					"responses": {}
				}
				
				# Add parameters
				for param in endpoint.parameters:
					param_spec = {
						"name": param.name,
						"in": param.parameter_type.value,
						"required": param.required,
						"description": param.description,
						"schema": {"type": param.data_type}
					}
					
					if param.example is not None:
						param_spec["example"] = param.example
					if param.default_value is not None:
						param_spec["schema"]["default"] = param.default_value
					if param.enum_values:
						param_spec["schema"]["enum"] = param.enum_values
					
					endpoint_spec["parameters"].append(param_spec)
				
				# Add request body
				if endpoint.request_body:
					endpoint_spec["requestBody"] = endpoint.request_body
				
				# Add responses
				for response in endpoint.responses:
					endpoint_spec["responses"][str(response.status_code)] = {
						"description": response.description,
						"content": {
							response.content_type.value: {
								"schema": response.schema or {"$ref": "#/components/schemas/WBPMServiceResponse"},
								"example": response.example
							}
						}
					}
					
					if response.headers:
						endpoint_spec["responses"][str(response.status_code)]["headers"] = response.headers
				
				# Add to spec
				spec["paths"][path][method] = endpoint_spec
			
			logger.info("OpenAPI specification generated successfully")
			return spec
			
		except Exception as e:
			logger.error(f"Error generating OpenAPI spec: {e}")
			raise
	
	async def generate_markdown_documentation(self) -> str:
		"""Generate comprehensive Markdown documentation."""
		try:
			doc_sections = []
			
			# Title and overview
			doc_sections.append(f"""# {self.base_info['title']}

{self.base_info['description']}

**Version**: {self.base_info['version']}  
**Contact**: [{self.base_info['contact']['name']}]({self.base_info['contact']['url']}) - {self.base_info['contact']['email']}

## Overview

The APG Workflow & Business Process Management API provides comprehensive capabilities for:

- **Process Management**: Define, deploy, and manage BPMN 2.0 compliant workflows
- **Task Automation**: Intelligent task routing, assignment, and completion tracking
- **Real-time Analytics**: Process performance monitoring and optimization insights
- **Template Library**: AI-powered process template recommendations and management
- **Monitoring & Alerting**: Real-time system monitoring with configurable alerts
- **Multi-channel Notifications**: Flexible notification delivery across multiple channels

## Authentication

This API uses JWT-based authentication integrated with the APG platform:

```bash
# Using Bearer token
curl -H "Authorization: Bearer <jwt_token>" \\
     -H "X-Tenant-ID: <tenant_id>" \\
     https://api.datacraft.co.ke/wbmp/api/v1/processes
```

Required headers:
- `Authorization: Bearer <jwt_token>` - JWT token from APG auth service
- `X-Tenant-ID: <tenant_id>` - Tenant identifier for multi-tenant operations

## Rate Limiting

API requests are rate limited per tenant:
- **Standard**: 1000 requests per hour
- **Premium**: 5000 requests per hour
- **Enterprise**: Unlimited

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Request limit per hour
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Error Handling

All API responses follow the standard `WBPMServiceResponse` format:

```json
{
  "success": false,
  "message": "Validation failed",
  "data": null,
  "errors": [
    "Process key is required",
    "BPMN XML is invalid"
  ],
  "metadata": {
    "error_code": "VALIDATION_ERROR",
    "timestamp": "2025-01-15T12:00:00Z"
  }
}
```

Common HTTP status codes:
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error

""")
			
			# Group endpoints by tags
			endpoints_by_tag = {}
			for endpoint in self.api_definitions.endpoints:
				for tag in endpoint.tags:
					if tag not in endpoints_by_tag:
						endpoints_by_tag[tag] = []
					endpoints_by_tag[tag].append(endpoint)
			
			# Generate documentation for each tag
			for tag, endpoints in endpoints_by_tag.items():
				doc_sections.append(f"\n## {tag}\n")
				
				for endpoint in endpoints:
					doc_sections.append(self._generate_endpoint_documentation(endpoint))
			
			# Add code examples section
			doc_sections.append(self._generate_code_examples())
			
			# Add integration guide
			doc_sections.append(self._generate_integration_guide())
			
			return "\n".join(doc_sections)
			
		except Exception as e:
			logger.error(f"Error generating Markdown documentation: {e}")
			raise
	
	def _generate_endpoint_documentation(self, endpoint: APIEndpoint) -> str:
		"""Generate documentation for a single endpoint."""
		doc = []
		
		# Endpoint header
		doc.append(f"### {endpoint.method.value} {endpoint.path}")
		doc.append(f"\n{endpoint.description}\n")
		
		# Authentication
		if endpoint.authentication_required:
			doc.append("**Authentication**: Required")
			if endpoint.permissions_required:
				doc.append(f"**Permissions**: {', '.join(endpoint.permissions_required)}")
		else:
			doc.append("**Authentication**: Not required")
		
		# Parameters
		if endpoint.parameters:
			doc.append("\n**Parameters**:\n")
			for param in endpoint.parameters:
				required = " *(required)*" if param.required else " *(optional)*"
				doc.append(f"- `{param.name}` ({param.parameter_type.value}): {param.data_type}{required} - {param.description}")
				if param.example is not None:
					doc.append(f"  - Example: `{param.example}`")
				if param.default_value is not None:
					doc.append(f"  - Default: `{param.default_value}`")
		
		# Request body
		if endpoint.request_body:
			doc.append("\n**Request Body**:")
			doc.append("```json")
			if "application/json" in endpoint.request_body.get("content", {}):
				example = endpoint.request_body["content"]["application/json"].get("example", {})
				doc.append(json.dumps(example, indent=2))
			doc.append("```")
		
		# Responses
		if endpoint.responses:
			doc.append("\n**Responses**:\n")
			for response in endpoint.responses:
				doc.append(f"**{response.status_code}** - {response.description}")
				if response.example:
					doc.append("```json")
					doc.append(json.dumps(response.example, indent=2))
					doc.append("```")
		
		# Examples
		if endpoint.examples:
			doc.append("\n**Examples**:\n")
			for example in endpoint.examples:
				doc.append(f"**{example['title']}**")
				doc.append(f"{example['description']}")
				doc.append("```bash")
				doc.append(f"curl -X {endpoint.method.value} \\")
				doc.append(f"  -H \"Authorization: Bearer $JWT_TOKEN\" \\")
				doc.append(f"  -H \"X-Tenant-ID: $TENANT_ID\" \\")
				doc.append(f"  -H \"Content-Type: application/json\" \\")
				if example.get('request'):
					doc.append(f"  -d '{json.dumps(example['request'])}' \\")
				doc.append(f"  https://api.datacraft.co.ke/wbpm{endpoint.path}")
				doc.append("```")
		
		doc.append("\n---\n")
		return "\n".join(doc)
	
	def _generate_code_examples(self) -> str:
		"""Generate code examples section."""
		return """
## Code Examples

### Python SDK

```python
from apg_wbpm import WBPMClient

# Initialize client
client = WBPMClient(
    base_url="https://api.datacraft.co.ke/wbpm",
    jwt_token="your_jwt_token",
    tenant_id="your_tenant_id"
)

# Create process definition
process_def = await client.processes.create_definition({
    "process_key": "invoice_approval",
    "process_name": "Invoice Approval Process",
    "bpmn_xml": bpmn_content,
    "category": "finance"
})

# Start process instance
instance = await client.processes.start_instance(
    process_key="invoice_approval",
    variables={
        "invoice_amount": 1500.00,
        "vendor_name": "Acme Corp"
    }
)

# Get user tasks
tasks = await client.tasks.get_user_tasks(
    user_id="user_123",
    status="active"
)

# Complete task
result = await client.tasks.complete_task(
    task_id="task_456",
    variables={"approved": True, "comments": "Approved"}
)
```

### JavaScript SDK

```javascript
import { WBPMClient } from '@datacraft/apg-wbpm';

// Initialize client
const client = new WBPMClient({
  baseURL: 'https://api.datacraft.co.ke/wbpm',
  jwtToken: 'your_jwt_token',
  tenantId: 'your_tenant_id'
});

// Create process definition
const processDef = await client.processes.createDefinition({
  processKey: 'document_review',
  processName: 'Document Review Process',
  bpmnXml: bpmnContent,
  category: 'review'
});

// Start process instance
const instance = await client.processes.startInstance({
  processKey: 'document_review',
  variables: {
    documentId: 'doc_123',
    reviewerEmail: 'reviewer@company.com'
  }
});

// Get analytics
const analytics = await client.analytics.getProcessAnalytics({
  processKey: 'document_review',
  timeRange: '30d'
});
```

### cURL Examples

```bash
# Create process definition
curl -X POST \\
  -H "Authorization: Bearer $JWT_TOKEN" \\
  -H "X-Tenant-ID: $TENANT_ID" \\
  -H "Content-Type: application/json" \\
  -d '{
    "process_key": "expense_approval",
    "process_name": "Expense Approval Process",
    "bpmn_xml": "<?xml version=\"1.0\"?>...",
    "category": "finance"
  }' \\
  https://api.datacraft.co.ke/wbpm/api/v1/processes/definitions

# Get process analytics
curl -X GET \\
  -H "Authorization: Bearer $JWT_TOKEN" \\
  -H "X-Tenant-ID: $TENANT_ID" \\
  "https://api.datacraft.co.ke/wbpm/api/v1/analytics/processes?time_range=30d&metrics=duration,throughput"

# Send notification
curl -X POST \\
  -H "Authorization: Bearer $JWT_TOKEN" \\
  -H "X-Tenant-ID: $TENANT_ID" \\
  -H "Content-Type: application/json" \\
  -d '{
    "recipient_id": "user_123",
    "channels": ["email", "in_app"],
    "priority": "high",
    "subject": "Task Assignment",
    "message": "You have been assigned a new task"
  }' \\
  https://api.datacraft.co.ke/wbpm/api/v1/notifications/send
```
"""
	
	def _generate_integration_guide(self) -> str:
		"""Generate integration guide section."""
		return """
## Integration Guide

### APG Platform Integration

The WBPM API is designed to integrate seamlessly with other APG capabilities:

#### Authentication & Authorization
- Uses APG auth_rbac for unified authentication
- Inherits user permissions from APG permission system
- Supports multi-tenant isolation

#### Audit & Compliance
- All operations are logged through APG audit_compliance
- Maintains complete audit trails for regulatory compliance
- Supports data retention policies

#### Real-time Collaboration
- Integrates with APG real_time_collaboration for process collaboration
- Supports real-time process updates and notifications
- Enables collaborative process design

#### AI Orchestration
- Leverages APG ai_orchestration for intelligent automation
- Provides AI-powered task routing and assignment
- Supports predictive analytics and optimization

### Webhook Integration

Configure webhooks to receive real-time notifications:

```json
{
  "webhook_url": "https://your-app.com/webhooks/wbpm",
  "events": [
    "process.started",
    "process.completed", 
    "task.assigned",
    "task.completed",
    "alert.triggered"
  ],
  "secret": "your_webhook_secret"
}
```

Event payload example:
```json
{
  "event_type": "task.assigned",
  "timestamp": "2025-01-15T12:00:00Z",
  "tenant_id": "tenant_123",
  "data": {
    "task_id": "task_456",
    "process_id": "proc_789",
    "assignee": "user_123",
    "task_name": "Review Document",
    "due_date": "2025-01-20T17:00:00Z"
  },
  "signature": "sha256=hash_of_payload"
}
```

### Error Handling Best Practices

```python
import asyncio
from apg_wbpm import WBPMClient, WBPMError

async def handle_wbpm_operations():
    client = WBPMClient(...)
    
    try:
        # Attempt operation
        result = await client.processes.start_instance(...)
        
    except WBPMError as e:
        if e.status_code == 429:
            # Rate limited - implement exponential backoff
            await asyncio.sleep(e.retry_after)
            return await handle_wbpm_operations()
        elif e.status_code == 400:
            # Validation error - check request data
            logger.error(f"Validation failed: {e.errors}")
        elif e.status_code >= 500:
            # Server error - retry with backoff
            await asyncio.sleep(5)
            return await handle_wbpm_operations()
        else:
            # Other error - handle appropriately
            logger.error(f"WBPM error: {e.message}")
```

### Performance Optimization

1. **Use pagination** for large result sets
2. **Implement caching** for frequently accessed data
3. **Use batch operations** when available
4. **Monitor rate limits** and implement backoff
5. **Use webhooks** instead of polling for real-time updates

### Testing

Use the staging environment for testing:

```bash
export WBPM_BASE_URL="https://staging-api.datacraft.co.ke/wbpm"
export JWT_TOKEN="your_staging_token"
export TENANT_ID="your_staging_tenant"

# Run your tests
python test_wbpm_integration.py
```

For more detailed integration examples and SDKs, visit our [Developer Portal](https://developers.datacraft.co.ke/wbpm).
"""
	
	async def save_documentation(
		self,
		output_dir: Path,
		formats: List[str] = ["openapi", "markdown"]
	) -> Dict[str, str]:
		"""Save documentation in specified formats."""
		try:
			output_dir.mkdir(parents=True, exist_ok=True)
			saved_files = {}
			
			if "openapi" in formats:
				# Generate and save OpenAPI spec
				openapi_spec = await self.generate_openapi_spec()
				openapi_file = output_dir / "wbpm_api_openapi.yaml"
				
				with open(openapi_file, 'w') as f:
					yaml.dump(openapi_spec, f, default_flow_style=False, sort_keys=False)
				
				saved_files["openapi"] = str(openapi_file)
				logger.info(f"OpenAPI specification saved to: {openapi_file}")
			
			if "markdown" in formats:
				# Generate and save Markdown docs
				markdown_docs = await self.generate_markdown_documentation()
				markdown_file = output_dir / "wbpm_api_documentation.md"
				
				with open(markdown_file, 'w') as f:
					f.write(markdown_docs)
				
				saved_files["markdown"] = str(markdown_file)
				logger.info(f"Markdown documentation saved to: {markdown_file}")
			
			return saved_files
			
		except Exception as e:
			logger.error(f"Error saving documentation: {e}")
			raise


# =============================================================================
# Service Factory
# =============================================================================

async def generate_api_documentation(
	output_dir: str = "./docs",
	formats: List[str] = ["openapi", "markdown"]
) -> Dict[str, str]:
	"""Generate and save API documentation."""
	try:
		generator = APIDocumentationGenerator()
		output_path = Path(output_dir)
		
		saved_files = await generator.save_documentation(output_path, formats)
		
		logger.info("API documentation generation completed successfully")
		return saved_files
		
	except Exception as e:
		logger.error(f"Error generating API documentation: {e}")
		raise


# Export main classes
__all__ = [
	'APIDocumentationGenerator',
	'WBPMAPIDefinitions',
	'APIEndpoint',
	'APIParameter',
	'APIResponse',
	'APISchema',
	'generate_api_documentation'
]