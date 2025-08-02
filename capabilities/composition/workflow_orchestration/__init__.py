"""
APG Workflow Orchestration Capability

Enterprise-grade workflow automation and orchestration that integrates seamlessly 
with the APG platform ecosystem. Provides intelligent, adaptive workflow management
with native APG capability integration.

© 2025 Datacraft. All rights reserved.
Author: APG Development Team
"""

from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict
import asyncio
from datetime import datetime, timedelta

class WorkflowStatus(str, Enum):
	"""Workflow execution status."""
	DRAFT = "draft"
	ACTIVE = "active"
	PAUSED = "paused"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	SUSPENDED = "suspended"

class TaskStatus(str, Enum):
	"""Individual task status."""
	PENDING = "pending"
	ASSIGNED = "assigned"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"
	SKIPPED = "skipped"
	ESCALATED = "escalated"

class TaskType(str, Enum):
	"""Types of workflow tasks."""
	AUTOMATED = "automated"
	HUMAN = "human"
	APPROVAL = "approval"
	NOTIFICATION = "notification"
	INTEGRATION = "integration"
	CONDITIONAL = "conditional"
	PARALLEL = "parallel"
	SUBPROCESS = "subprocess"

class TriggerType(str, Enum):
	"""Workflow trigger types."""
	MANUAL = "manual"
	SCHEDULED = "scheduled"
	EVENT = "event"
	API = "api"
	WEBHOOK = "webhook"
	CONDITION = "condition"

@dataclass
class WorkflowContext:
	"""Workflow execution context."""
	tenant_id: str
	workflow_id: str
	instance_id: str
	user_id: Optional[str] = None
	variables: Dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)

class WorkflowTask(BaseModel):
	"""Individual workflow task definition."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str = ""
	task_type: TaskType
	assigned_to: Optional[str] = None  # user_id or role
	due_date: Optional[datetime] = None
	priority: int = 5  # 1-10 scale
	dependencies: List[str] = Field(default_factory=list)  # task IDs
	conditions: List[Dict[str, Any]] = Field(default_factory=list)
	actions: List[Dict[str, Any]] = Field(default_factory=list)
	sla_hours: Optional[int] = None
	escalation_rules: List[Dict[str, Any]] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkflowDefinition(BaseModel):
	"""Complete workflow definition."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str = ""
	version: str = "1.0.0"
	tenant_id: str
	created_by: str
	tasks: List[WorkflowTask]
	triggers: List[Dict[str, Any]] = Field(default_factory=list)
	variables: Dict[str, Any] = Field(default_factory=dict)
	tags: List[str] = Field(default_factory=list)
	category: str = "general"
	is_template: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)

class WorkflowInstance(BaseModel):
	"""Running workflow instance."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	workflow_id: str
	tenant_id: str
	started_by: str
	status: WorkflowStatus = WorkflowStatus.ACTIVE
	current_tasks: List[str] = Field(default_factory=list)  # task IDs currently executing
	completed_tasks: List[str] = Field(default_factory=list)
	failed_tasks: List[str] = Field(default_factory=list)
	context: Dict[str, Any] = Field(default_factory=dict)
	started_at: datetime = Field(default_factory=datetime.utcnow)
	completed_at: Optional[datetime] = None
	error_message: Optional[str] = None

class TaskExecution(BaseModel):
	"""Task execution record."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	instance_id: str
	task_id: str
	assigned_to: Optional[str] = None
	status: TaskStatus = TaskStatus.PENDING
	started_at: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	result: Dict[str, Any] = Field(default_factory=dict)
	error_message: Optional[str] = None
	attempts: int = 0
	max_attempts: int = 3

class WorkflowTaskHandler(ABC):
	"""Abstract base class for task handlers."""
	
	@property
	@abstractmethod
	def task_type(self) -> TaskType:
		"""Task type this handler supports."""
		pass
	
	@abstractmethod
	async def execute_task(
		self,
		task: WorkflowTask,
		execution: TaskExecution,
		context: WorkflowContext
	) -> Dict[str, Any]:
		"""Execute the task and return results."""
		pass

class AutomatedTaskHandler(WorkflowTaskHandler):
	"""Handler for automated tasks."""
	
	@property
	def task_type(self) -> TaskType:
		"""Task type this handler supports."""
		return TaskType.AUTOMATED
	
	async def execute_task(
		self,
		task: WorkflowTask,
		execution: TaskExecution,
		context: WorkflowContext
	) -> Dict[str, Any]:
		"""Execute automated task."""
		try:
			# Update execution status
			execution.status = TaskStatus.RUNNING
			execution.started_at = datetime.utcnow()
			
			# Execute based on task configuration
			task_config = task.configuration or {}
			task_type = task_config.get('automation_type', 'script')
			
			result = {}
			
			if task_type == 'script':
				# Execute custom script
				script_code = task_config.get('script', '')
				if script_code:
					result = await self._execute_script(script_code, context)
				else:
					result = {'error': 'No script provided for script task'}
					
			elif task_type == 'data_transformation':
				# Execute data transformation
				result = await self._execute_data_transformation(task_config, context)
				
			elif task_type == 'api_call':
				# Execute API call
				result = await self._execute_api_call(task_config, context)
				
			elif task_type == 'database_operation':
				# Execute database operation
				result = await self._execute_database_operation(task_config, context)
				
			else:
				result = {'error': f'Unknown automation type: {task_type}'}
			
			# Update execution status
			execution.completed_at = datetime.utcnow()
			execution.status = TaskStatus.COMPLETED if not result.get('error') else TaskStatus.FAILED
			execution.result = result
			
			return result
			
		except Exception as e:
			execution.status = TaskStatus.FAILED
			execution.completed_at = datetime.utcnow()
			execution.error_message = str(e)
			return {'error': str(e), 'task_id': task.id}
	
	async def _execute_script(self, script: str, context: WorkflowContext) -> Dict[str, Any]:
		"""Execute script safely."""
		try:
			import ast
			
			# Create safe execution environment
			safe_globals = {
				'__builtins__': {
					'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
					'list': list, 'dict': dict, 'tuple': tuple, 'abs': abs,
					'min': min, 'max': max, 'sum': sum, 'round': round
				},
				'context_data': context.variables,
				'input_data': context.input_data
			}
			
			# Parse and validate script
			tree = ast.parse(script)
			for node in ast.walk(tree):
				if isinstance(node, (ast.Import, ast.ImportFrom)):
					raise ValueError("Import statements not allowed")
			
			# Execute script
			local_vars = {}
			exec(compile(tree, '<script>', 'exec'), safe_globals, local_vars)
			
			return {
				'success': True,
				'result': local_vars.get('result', local_vars),
				'output_data': local_vars.get('output_data', {})
			}
			
		except Exception as e:
			return {'error': str(e), 'success': False}
	
	async def _execute_data_transformation(self, config: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
		"""Execute data transformation."""
		try:
			transformation_type = config.get('transformation_type', 'map')
			input_data = context.input_data
			
			if transformation_type == 'map':
				# Apply mapping transformation
				mapping = config.get('mapping', {})
				result = {}
				for key, source_key in mapping.items():
					if source_key in input_data:
						result[key] = input_data[source_key]
				return {'success': True, 'result': result}
				
			elif transformation_type == 'filter':
				# Apply filter transformation
				filter_expr = config.get('filter_expression', 'True')
				filtered_data = []
				if isinstance(input_data, list):
					for item in input_data:
						try:
							if eval(filter_expr, {'item': item}):
								filtered_data.append(item)
						except:
							pass
				return {'success': True, 'result': filtered_data}
				
			elif transformation_type == 'aggregate':
				# Apply aggregation
				agg_type = config.get('aggregation', 'count')
				if isinstance(input_data, list):
					if agg_type == 'count':
						result = len(input_data)
					elif agg_type == 'sum' and all(isinstance(x, (int, float)) for x in input_data):
						result = sum(input_data)
					elif agg_type == 'avg' and all(isinstance(x, (int, float)) for x in input_data):
						result = sum(input_data) / len(input_data) if input_data else 0
					else:
						result = len(input_data)
				return {'success': True, 'result': result}
			
			return {'error': f'Unknown transformation type: {transformation_type}'}
			
		except Exception as e:
			return {'error': str(e), 'success': False}
	
	async def _execute_api_call(self, config: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
		"""Execute API call."""
		try:
			import aiohttp
			
			url = config.get('url', '')
			method = config.get('method', 'GET').upper()
			headers = config.get('headers', {})
			data = config.get('data', {})
			timeout = config.get('timeout', 30)
			
			if not url:
				return {'error': 'No URL provided for API call'}
			
			# Replace variables in URL and data
			for key, value in context.variables.items():
				url = url.replace(f'{{{key}}}', str(value))
				if isinstance(data, dict):
					for data_key, data_value in data.items():
						if isinstance(data_value, str):
							data[data_key] = data_value.replace(f'{{{key}}}', str(value))
			
			# Make API call
			async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
				if method == 'GET':
					async with session.get(url, headers=headers, params=data) as response:
						result_data = await response.text()
						try:
							result_data = await response.json()
						except:
							pass
				else:
					async with session.request(method, url, headers=headers, json=data) as response:
						result_data = await response.text()
						try:
							result_data = await response.json()
						except:
							pass
				
				return {
					'success': True,
					'status_code': response.status,
					'result': result_data,
					'headers': dict(response.headers)
				}
			
		except Exception as e:
			return {'error': str(e), 'success': False}
	
	async def _execute_database_operation(self, config: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
		"""Execute database operation."""
		try:
			operation = config.get('operation', 'select').lower()
			table = config.get('table', '')
			conditions = config.get('conditions', {})
			data = config.get('data', {})
			
			if not table:
				return {'error': 'No table specified for database operation'}
			
			# Get database connection from context or create new one
			db_manager = context.services.get('database_manager')
			if not db_manager:
				return {'error': 'Database manager not available'}
			
			if operation == 'select':
				# Build SELECT query
				columns = config.get('columns', ['*'])
				query = f"SELECT {', '.join(columns)} FROM {table}"
				
				if conditions:
					where_clauses = []
					for key, value in conditions.items():
						where_clauses.append(f"{key} = %s")
					query += f" WHERE {' AND '.join(where_clauses)}"
				
				result = await db_manager.execute_query(query, list(conditions.values()))
				return {'success': True, 'result': result, 'row_count': len(result)}
				
			elif operation == 'insert':
				# Build INSERT query
				if not data:
					return {'error': 'No data provided for insert operation'}
				
				columns = list(data.keys())
				placeholders = [f"%s" for _ in columns]
				query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
				
				result = await db_manager.execute_query(query, list(data.values()))
				return {'success': True, 'result': 'Insert successful', 'affected_rows': result}
				
			elif operation == 'update':
				# Build UPDATE query
				if not data:
					return {'error': 'No data provided for update operation'}
				
				set_clauses = [f"{key} = %s" for key in data.keys()]
				query = f"UPDATE {table} SET {', '.join(set_clauses)}"
				
				query_params = list(data.values())
				if conditions:
					where_clauses = [f"{key} = %s" for key in conditions.keys()]
					query += f" WHERE {' AND '.join(where_clauses)}"
					query_params.extend(list(conditions.values()))
				
				result = await db_manager.execute_query(query, query_params)
				return {'success': True, 'result': 'Update successful', 'affected_rows': result}
				
			elif operation == 'delete':
				# Build DELETE query
				query = f"DELETE FROM {table}"
				query_params = []
				
				if conditions:
					where_clauses = [f"{key} = %s" for key in conditions.keys()]
					query += f" WHERE {' AND '.join(where_clauses)}"
					query_params = list(conditions.values())
				else:
					return {'error': 'DELETE without conditions not allowed for safety'}
				
				result = await db_manager.execute_query(query, query_params)
				return {'success': True, 'result': 'Delete successful', 'affected_rows': result}
			
			return {'error': f'Unknown database operation: {operation}'}
			
		except Exception as e:
			return {'error': str(e), 'success': False}

	@property
	def task_type(self) -> TaskType:
		return TaskType.AUTOMATED
	
	async def execute_task(
		self,
		task: WorkflowTask,
		execution: TaskExecution,
		context: WorkflowContext
	) -> Dict[str, Any]:
		"""Execute automated task."""
		# Simulate automated task execution
		await asyncio.sleep(0.1)  # Simulate processing time
		
		return {
			"success": True,
			"message": f"Automated task {task.name} completed",
			"data": {"processed_at": datetime.utcnow().isoformat()}
		}

class HumanTaskHandler(WorkflowTaskHandler):
	"""Handler for human tasks."""
	
	@property
	def task_type(self) -> TaskType:
		return TaskType.HUMAN
	
	async def execute_task(
		self,
		task: WorkflowTask,
		execution: TaskExecution,
		context: WorkflowContext
	) -> Dict[str, Any]:
		"""Execute human task (assign and wait)."""
		# Human tasks require external completion
		execution.status = TaskStatus.ASSIGNED
		
		return {
			"success": True,
			"message": f"Human task {task.name} assigned to {task.assigned_to}",
			"requires_completion": True
		}

class WorkflowEngine:
	"""Core workflow orchestration engine."""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.workflows: Dict[str, WorkflowDefinition] = {}
		self.instances: Dict[str, WorkflowInstance] = {}
		self.executions: Dict[str, List[TaskExecution]] = {}
		self.task_handlers: Dict[TaskType, WorkflowTaskHandler] = {
			TaskType.AUTOMATED: AutomatedTaskHandler(),
			TaskType.HUMAN: HumanTaskHandler(),
		}
		self.event_listeners: List[Callable] = []
	
	def register_task_handler(self, handler: WorkflowTaskHandler) -> bool:
		"""Register a custom task handler."""
		self.task_handlers[handler.task_type] = handler
		return True
	
	def create_workflow(self, definition: WorkflowDefinition) -> str:
		"""Create a new workflow definition."""
		definition.tenant_id = self.tenant_id
		self.workflows[definition.id] = definition
		return definition.id
	
	async def start_workflow(
		self,
		workflow_id: str,
		started_by: str,
		initial_context: Optional[Dict[str, Any]] = None
	) -> str:
		"""Start a new workflow instance."""
		workflow = self.workflows.get(workflow_id)
		if not workflow:
			raise ValueError(f"Workflow {workflow_id} not found")
		
		instance = WorkflowInstance(
			workflow_id=workflow_id,
			tenant_id=self.tenant_id,
			started_by=started_by,
			context=initial_context or {}
		)
		
		self.instances[instance.id] = instance
		self.executions[instance.id] = []
		
		# Start initial tasks (tasks with no dependencies)
		await self._start_ready_tasks(workflow, instance)
		
		return instance.id
	
	async def complete_task(
		self,
		instance_id: str,
		task_id: str,
		user_id: str,
		result: Dict[str, Any]
	) -> bool:
		"""Complete a human task with results."""
		instance = self.instances.get(instance_id)
		if not instance:
			return False
		
		workflow = self.workflows.get(instance.workflow_id)
		if not workflow:
			return False
		
		# Find the task execution
		execution = None
		for exec in self.executions[instance_id]:
			if exec.task_id == task_id and exec.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
				execution = exec
				break
		
		if not execution:
			return False
		
		# Complete the task
		execution.status = TaskStatus.COMPLETED
		execution.completed_at = datetime.utcnow()
		execution.result = result
		
		# Update instance
		if task_id in instance.current_tasks:
			instance.current_tasks.remove(task_id)
		instance.completed_tasks.append(task_id)
		
		# Check if workflow is complete
		await self._check_workflow_completion(workflow, instance)
		
		# Start next tasks
		if instance.status == WorkflowStatus.ACTIVE:
			await self._start_ready_tasks(workflow, instance)
		
		return True
	
	async def _start_ready_tasks(self, workflow: WorkflowDefinition, instance: WorkflowInstance):
		"""Start tasks that are ready to execute."""
		for task in workflow.tasks:
			if (task.id not in instance.completed_tasks and 
				task.id not in instance.failed_tasks and
				task.id not in instance.current_tasks):
				
				# Check if all dependencies are completed
				dependencies_met = all(
					dep_id in instance.completed_tasks 
					for dep_id in task.dependencies
				)
				
				if dependencies_met:
					await self._execute_task(workflow, instance, task)
	
	async def _execute_task(
		self,
		workflow: WorkflowDefinition,
		instance: WorkflowInstance,
		task: WorkflowTask
	):
		"""Execute a single task."""
		execution = TaskExecution(
			instance_id=instance.id,
			task_id=task.id,
			assigned_to=task.assigned_to
		)
		
		self.executions[instance.id].append(execution)
		instance.current_tasks.append(task.id)
		
		# Get task handler
		handler = self.task_handlers.get(task.task_type)
		if not handler:
			execution.status = TaskStatus.FAILED
			execution.error_message = f"No handler for task type {task.task_type}"
			return
		
		try:
			execution.status = TaskStatus.IN_PROGRESS
			execution.started_at = datetime.utcnow()
			
			context = WorkflowContext(
				tenant_id=self.tenant_id,
				workflow_id=workflow.id,
				instance_id=instance.id,
				variables=instance.context
			)
			
			result = await handler.execute_task(task, execution, context)
			
			if result.get("requires_completion"):
				# Human task - will be completed externally
				pass
			else:
				# Automated task - complete immediately
				execution.status = TaskStatus.COMPLETED
				execution.completed_at = datetime.utcnow()
				execution.result = result
				
				instance.current_tasks.remove(task.id)
				instance.completed_tasks.append(task.id)
		
		except Exception as e:
			execution.status = TaskStatus.FAILED
			execution.error_message = str(e)
			execution.completed_at = datetime.utcnow()
			
			instance.current_tasks.remove(task.id)
			instance.failed_tasks.append(task.id)
	
	async def _check_workflow_completion(
		self,
		workflow: WorkflowDefinition,
		instance: WorkflowInstance
	):
		"""Check if workflow is complete."""
		all_task_ids = {task.id for task in workflow.tasks}
		completed_and_failed = set(instance.completed_tasks + instance.failed_tasks)
		
		if all_task_ids <= completed_and_failed:
			if instance.failed_tasks:
				instance.status = WorkflowStatus.FAILED
			else:
				instance.status = WorkflowStatus.COMPLETED
			instance.completed_at = datetime.utcnow()
	
	def get_instance_status(self, instance_id: str) -> Optional[WorkflowInstance]:
		"""Get workflow instance status."""
		return self.instances.get(instance_id)
	
	def list_workflows(self) -> List[WorkflowDefinition]:
		"""List all workflow definitions."""
		return list(self.workflows.values())
	
	def list_instances(self, status: Optional[WorkflowStatus] = None) -> List[WorkflowInstance]:
		"""List workflow instances."""
		instances = list(self.instances.values())
		if status:
			instances = [i for i in instances if i.status == status]
		return instances

# Service factory
_workflow_engines: Dict[str, WorkflowEngine] = {}

def get_workflow_engine(tenant_id: str) -> WorkflowEngine:
	"""Get workflow engine for tenant."""
	if tenant_id not in _workflow_engines:
		_workflow_engines[tenant_id] = WorkflowEngine(tenant_id)
	return _workflow_engines[tenant_id]

# Workflow template examples
WORKFLOW_TEMPLATES = {
	"capability_approval": {
		"name": "Capability Approval Process",
		"description": "Standard approval workflow for new capability deployment",
		"category": "governance",
		"tasks": [
			{
				"name": "Technical Review",
				"task_type": "human",
				"assigned_to": "technical_reviewer",
				"sla_hours": 48
			},
			{
				"name": "Security Assessment", 
				"task_type": "automated",
				"dependencies": ["technical_review"]
			},
			{
				"name": "Manager Approval",
				"task_type": "approval",
				"assigned_to": "manager",
				"dependencies": ["security_assessment"],
				"sla_hours": 24
			}
		]
	},
	"incident_response": {
		"name": "Incident Response Workflow",
		"description": "Automated incident response and escalation",
		"category": "operations",
		"tasks": [
			{
				"name": "Incident Detection",
				"task_type": "automated"
			},
			{
				"name": "Initial Assessment",
				"task_type": "human",
				"assigned_to": "on_call_engineer",
				"dependencies": ["incident_detection"],
				"sla_hours": 1
			},
			{
				"name": "Escalate to Manager",
				"task_type": "notification",
				"conditions": [{"severity": "high"}],
				"dependencies": ["initial_assessment"]
			}
		]
	}
}

# APG Capability Metadata
CAPABILITY_INFO = {
	"name": "workflow_orchestration",
	"version": "1.0.0",
	"description": "Enterprise workflow orchestration and automation platform",
	"author": "APG Development Team",
	"category": "composition",
	"tags": ["workflow", "orchestration", "automation", "enterprise"],
	"apg_version": "1.0.0",
	"capabilities_required": [
		"auth_rbac",
		"audit_compliance", 
		"ai_orchestration",
		"federated_learning",
		"real_time_collaboration",
		"notification_engine",
		"document_management",
		"time_series_analytics"
	],
	"capabilities_optional": [
		"computer_vision",
		"nlp_processing",
		"visualization_3d"
	],
	"endpoints": {
		"api": "/api/v1/workflow",
		"ui": "/workflow",
		"websocket": "/ws/workflow"
	},
	"permissions": [
		"workflow.view",
		"workflow.create", 
		"workflow.edit",
		"workflow.delete",
		"workflow.execute",
		"workflow.admin"
	]
}

# APG Composition Registration
def register_with_apg_composition() -> Dict[str, Any]:
	"""Register this capability with APG's composition engine."""
	return {
		"capability_id": uuid7str(),
		"info": CAPABILITY_INFO,
		"service_endpoints": {
			"workflow_service": "workflow_orchestration.service.WorkflowService",
			"execution_engine": "workflow_orchestration.engine.executor.WorkflowExecutor",
			"designer_service": "workflow_orchestration.designer.canvas.WorkflowDesigner"
		},
		"database_models": [
			"workflow_orchestration.models.Workflow",
			"workflow_orchestration.models.WorkflowInstance", 
			"workflow_orchestration.models.Task",
			"workflow_orchestration.models.Trigger",
			"workflow_orchestration.models.Connector"
		],
		"ui_components": {
			"designer": "/workflow/designer",
			"dashboard": "/workflow/dashboard",
			"monitor": "/workflow/monitor"
		},
		"integration_hooks": {
			"pre_workflow_execute": "workflow_orchestration.hooks.pre_execute",
			"post_workflow_complete": "workflow_orchestration.hooks.post_complete",
			"workflow_failed": "workflow_orchestration.hooks.on_failure"
		}
	}

__all__ = [
	"CAPABILITY_INFO",
	"register_with_apg_composition",
	"WorkflowStatus",
	"TaskStatus",
	"TaskType", 
	"TriggerType",
	"WorkflowContext",
	"WorkflowTask",
	"WorkflowDefinition",
	"WorkflowInstance",
	"TaskExecution",
	"WorkflowTaskHandler",
	"AutomatedTaskHandler",
	"HumanTaskHandler",
	"WorkflowEngine",
	"get_workflow_engine",
	"WORKFLOW_TEMPLATES"
]