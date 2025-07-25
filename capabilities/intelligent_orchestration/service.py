#!/usr/bin/env python3
"""
Intelligent Twin Orchestration and Workflow Automation
=====================================================

Advanced orchestration system for coordinating multiple digital twins,
automating complex workflows, and enabling intelligent decision-making
across distributed twin networks with visual workflow designer.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import time
import networkx as nx
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intelligent_orchestration")

class WorkflowStatus(Enum):
	"""Workflow execution status"""
	DRAFT = "draft"
	ACTIVE = "active"
	RUNNING = "running"
	PAUSED = "paused"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"

class TaskType(Enum):
	"""Types of workflow tasks"""
	DATA_COLLECTION = "data_collection"
	DATA_PROCESSING = "data_processing"
	ANALYSIS = "analysis"
	SIMULATION = "simulation"
	NOTIFICATION = "notification"
	API_CALL = "api_call"
	CONDITION = "condition"
	LOOP = "loop"
	PARALLEL = "parallel"
	CUSTOM_FUNCTION = "custom_function"
	TWIN_OPERATION = "twin_operation"

class TriggerType(Enum):
	"""Workflow trigger types"""
	MANUAL = "manual"
	SCHEDULE = "schedule"
	EVENT = "event"
	THRESHOLD = "threshold"
	API = "api"
	CASCADE = "cascade"

class TaskStatus(Enum):
	"""Individual task status"""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	SKIPPED = "skipped"
	RETRYING = "retrying"

class OrchestrationStrategy(Enum):
	"""Twin orchestration strategies"""
	ROUND_ROBIN = "round_robin"
	LOAD_BALANCED = "load_balanced"
	PRIORITY_BASED = "priority_based"
	DEPENDENCY_AWARE = "dependency_aware"
	RESOURCE_OPTIMIZED = "resource_optimized"
	AI_DRIVEN = "ai_driven"

@dataclass
class WorkflowTask:
	"""Individual task in a workflow"""
	task_id: str
	name: str
	task_type: TaskType
	configuration: Dict[str, Any]
	dependencies: List[str]  # Task IDs this task depends on
	timeout_seconds: int
	retry_count: int
	retry_delay_seconds: int
	position: Dict[str, float]  # x, y coordinates for visual designer
	outputs: Dict[str, Any]  # Task output schema
	conditions: List[Dict[str, Any]]  # Conditional execution rules
	parallel_execution: bool
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'task_id': self.task_id,
			'name': self.name,
			'task_type': self.task_type.value,
			'configuration': self.configuration,
			'dependencies': self.dependencies,
			'timeout_seconds': self.timeout_seconds,
			'retry_count': self.retry_count,
			'retry_delay_seconds': self.retry_delay_seconds,
			'position': self.position,
			'outputs': self.outputs,
			'conditions': self.conditions,
			'parallel_execution': self.parallel_execution
		}

@dataclass
class WorkflowTrigger:
	"""Workflow execution trigger"""
	trigger_id: str
	trigger_type: TriggerType
	configuration: Dict[str, Any]
	is_active: bool
	last_triggered: Optional[datetime]
	trigger_count: int
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'trigger_id': self.trigger_id,
			'trigger_type': self.trigger_type.value,
			'configuration': self.configuration,
			'is_active': self.is_active,
			'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None,
			'trigger_count': self.trigger_count
		}

@dataclass
class OrchestrationWorkflow:
	"""Complete workflow definition"""
	workflow_id: str
	name: str
	description: str
	version: str
	status: WorkflowStatus
	tasks: Dict[str, WorkflowTask]
	triggers: List[WorkflowTrigger]
	global_variables: Dict[str, Any]
	metadata: Dict[str, Any]
	created_by: str
	created_at: datetime
	last_modified: datetime
	execution_count: int
	success_rate: float
	average_duration_seconds: float
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'workflow_id': self.workflow_id,
			'name': self.name,
			'description': self.description,
			'version': self.version,
			'status': self.status.value,
			'tasks': {k: v.to_dict() for k, v in self.tasks.items()},
			'triggers': [t.to_dict() for t in self.triggers],
			'global_variables': self.global_variables,
			'metadata': self.metadata,
			'created_by': self.created_by,
			'created_at': self.created_at.isoformat(),
			'last_modified': self.last_modified.isoformat(),
			'execution_count': self.execution_count,
			'success_rate': self.success_rate,
			'average_duration_seconds': self.average_duration_seconds
		}

@dataclass
class TaskExecution:
	"""Individual task execution instance"""
	execution_id: str
	task_id: str
	workflow_execution_id: str
	status: TaskStatus
	start_time: Optional[datetime]
	end_time: Optional[datetime]
	inputs: Dict[str, Any]
	outputs: Dict[str, Any]
	error_message: Optional[str]
	retry_attempt: int
	execution_logs: List[Dict[str, Any]]
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'execution_id': self.execution_id,
			'task_id': self.task_id,
			'workflow_execution_id': self.workflow_execution_id,
			'status': self.status.value,
			'start_time': self.start_time.isoformat() if self.start_time else None,
			'end_time': self.end_time.isoformat() if self.end_time else None,
			'inputs': self.inputs,
			'outputs': self.outputs,
			'error_message': self.error_message,
			'retry_attempt': self.retry_attempt,
			'execution_logs': self.execution_logs
		}

@dataclass
class WorkflowExecution:
	"""Complete workflow execution instance"""
	execution_id: str
	workflow_id: str
	status: WorkflowStatus
	start_time: datetime
	end_time: Optional[datetime]
	triggered_by: str
	trigger_data: Dict[str, Any]
	task_executions: Dict[str, TaskExecution]
	variables: Dict[str, Any]
	error_message: Optional[str]
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'execution_id': self.execution_id,
			'workflow_id': self.workflow_id,
			'status': self.status.value,
			'start_time': self.start_time.isoformat(),
			'end_time': self.end_time.isoformat() if self.end_time else None,
			'triggered_by': self.triggered_by,
			'trigger_data': self.trigger_data,
			'task_executions': {k: v.to_dict() for k, v in self.task_executions.items()},
			'variables': self.variables,
			'error_message': self.error_message
		}

class TaskExecutor:
	"""Executes individual workflow tasks"""
	
	def __init__(self):
		self.custom_functions: Dict[str, Callable] = {}
		self.twin_registry: Dict[str, Dict[str, Any]] = {}
	
	def register_custom_function(self, name: str, function: Callable):
		"""Register custom function for workflow tasks"""
		self.custom_functions[name] = function
		logger.info(f"Registered custom function: {name}")
	
	def register_twin(self, twin_id: str, twin_info: Dict[str, Any]):
		"""Register digital twin for orchestration"""
		self.twin_registry[twin_id] = twin_info
		logger.info(f"Registered twin: {twin_id}")
	
	async def execute_task(self, task: WorkflowTask, task_execution: TaskExecution,
						  workflow_variables: Dict[str, Any]) -> bool:
		"""Execute a single workflow task"""
		
		task_execution.status = TaskStatus.RUNNING
		task_execution.start_time = datetime.utcnow()
		
		try:
			if task.task_type == TaskType.DATA_COLLECTION:
				await self._execute_data_collection(task, task_execution, workflow_variables)
			
			elif task.task_type == TaskType.DATA_PROCESSING:
				await self._execute_data_processing(task, task_execution, workflow_variables)
			
			elif task.task_type == TaskType.ANALYSIS:
				await self._execute_analysis(task, task_execution, workflow_variables)
			
			elif task.task_type == TaskType.SIMULATION:
				await self._execute_simulation(task, task_execution, workflow_variables)
			
			elif task.task_type == TaskType.NOTIFICATION:
				await self._execute_notification(task, task_execution, workflow_variables)
			
			elif task.task_type == TaskType.API_CALL:
				await self._execute_api_call(task, task_execution, workflow_variables)
			
			elif task.task_type == TaskType.CONDITION:
				await self._execute_condition(task, task_execution, workflow_variables)
			
			elif task.task_type == TaskType.TWIN_OPERATION:
				await self._execute_twin_operation(task, task_execution, workflow_variables)
			
			elif task.task_type == TaskType.CUSTOM_FUNCTION:
				await self._execute_custom_function(task, task_execution, workflow_variables)
			
			else:
				raise ValueError(f"Unsupported task type: {task.task_type}")
			
			task_execution.status = TaskStatus.COMPLETED
			task_execution.end_time = datetime.utcnow()
			
			logger.info(f"Task {task.task_id} completed successfully")
			return True
			
		except Exception as e:
			task_execution.status = TaskStatus.FAILED
			task_execution.end_time = datetime.utcnow()
			task_execution.error_message = str(e)
			
			logger.error(f"Task {task.task_id} failed: {e}")
			return False
	
	async def _execute_data_collection(self, task: WorkflowTask, execution: TaskExecution,
									  variables: Dict[str, Any]):
		"""Execute data collection task"""
		config = task.configuration
		source = config.get('source', 'unknown')
		data_type = config.get('data_type', 'generic')
		
		# Simulate data collection
		await asyncio.sleep(1)
		
		collected_data = {
			'source': source,
			'data_type': data_type,
			'timestamp': datetime.utcnow().isoformat(),
			'values': [1.2, 3.4, 5.6, 7.8],  # Mock data
			'quality_score': 0.95
		}
		
		execution.outputs['collected_data'] = collected_data
		execution.execution_logs.append({
			'timestamp': datetime.utcnow().isoformat(),
			'level': 'INFO',
			'message': f'Collected data from {source}'
		})
	
	async def _execute_data_processing(self, task: WorkflowTask, execution: TaskExecution,
									  variables: Dict[str, Any]):
		"""Execute data processing task"""
		config = task.configuration
		operation = config.get('operation', 'filter')
		parameters = config.get('parameters', {})
		
		# Get input data from dependencies or variables
		input_data = execution.inputs.get('data', variables.get('data', []))
		
		# Simulate data processing
		await asyncio.sleep(1)
		
		if operation == 'filter':
			threshold = parameters.get('threshold', 5.0)
			processed_data = [x for x in input_data if x > threshold]
		elif operation == 'transform':
			scale_factor = parameters.get('scale_factor', 1.0)
			processed_data = [x * scale_factor for x in input_data]
		else:
			processed_data = input_data
		
		execution.outputs['processed_data'] = processed_data
		execution.execution_logs.append({
			'timestamp': datetime.utcnow().isoformat(),
			'level': 'INFO',
			'message': f'Processed data using {operation}'
		})
	
	async def _execute_analysis(self, task: WorkflowTask, execution: TaskExecution,
							   variables: Dict[str, Any]):
		"""Execute analysis task"""
		config = task.configuration
		analysis_type = config.get('analysis_type', 'statistical')
		
		# Get input data
		input_data = execution.inputs.get('data', [1, 2, 3, 4, 5])
		
		# Simulate analysis
		await asyncio.sleep(2)
		
		if analysis_type == 'statistical':
			results = {
				'mean': sum(input_data) / len(input_data),
				'min': min(input_data),
				'max': max(input_data),
				'count': len(input_data)
			}
		elif analysis_type == 'anomaly_detection':
			# Mock anomaly detection
			results = {
				'anomalies_detected': 2,
				'anomaly_score': 0.15,
				'anomalous_values': [input_data[0], input_data[-1]]
			}
		else:
			results = {'status': 'completed', 'data_points': len(input_data)}
		
		execution.outputs['analysis_results'] = results
		execution.execution_logs.append({
			'timestamp': datetime.utcnow().isoformat(),
			'level': 'INFO',
			'message': f'Completed {analysis_type} analysis'
		})
	
	async def _execute_simulation(self, task: WorkflowTask, execution: TaskExecution,
								 variables: Dict[str, Any]):
		"""Execute simulation task"""
		config = task.configuration
		simulation_type = config.get('simulation_type', 'monte_carlo')
		iterations = config.get('iterations', 1000)
		
		# Simulate long-running simulation
		await asyncio.sleep(3)
		
		simulation_results = {
			'simulation_type': simulation_type,
			'iterations': iterations,
			'mean_result': 42.5,
			'std_deviation': 5.2,
			'confidence_interval': [32.1, 52.9],
			'completion_time': datetime.utcnow().isoformat()
		}
		
		execution.outputs['simulation_results'] = simulation_results
		execution.execution_logs.append({
			'timestamp': datetime.utcnow().isoformat(),
			'level': 'INFO',
			'message': f'Completed {simulation_type} simulation with {iterations} iterations'
		})
	
	async def _execute_notification(self, task: WorkflowTask, execution: TaskExecution,
								   variables: Dict[str, Any]):
		"""Execute notification task"""
		config = task.configuration
		notification_type = config.get('type', 'email')
		recipients = config.get('recipients', [])
		message = config.get('message', 'Workflow notification')
		
		# Simulate notification sending
		await asyncio.sleep(0.5)
		
		execution.outputs['notification_sent'] = {
			'type': notification_type,
			'recipients': recipients,
			'message': message,
			'sent_at': datetime.utcnow().isoformat(),
			'status': 'sent'
		}
		
		execution.execution_logs.append({
			'timestamp': datetime.utcnow().isoformat(),
			'level': 'INFO',
			'message': f'Sent {notification_type} notification to {len(recipients)} recipients'
		})
	
	async def _execute_api_call(self, task: WorkflowTask, execution: TaskExecution,
							   variables: Dict[str, Any]):
		"""Execute API call task"""
		config = task.configuration
		url = config.get('url', 'https://api.example.com')
		method = config.get('method', 'GET')
		headers = config.get('headers', {})
		payload = config.get('payload', {})
		
		# Simulate API call
		await asyncio.sleep(1)
		
		# Mock response
		api_response = {
			'status_code': 200,
			'response_data': {
				'message': 'API call successful',
				'timestamp': datetime.utcnow().isoformat(),
				'data': {'result': 'success'}
			},
			'headers': {'content-type': 'application/json'},
			'duration_ms': 150
		}
		
		execution.outputs['api_response'] = api_response
		execution.execution_logs.append({
			'timestamp': datetime.utcnow().isoformat(),
			'level': 'INFO',
			'message': f'{method} request to {url} completed with status {api_response["status_code"]}'
		})
	
	async def _execute_condition(self, task: WorkflowTask, execution: TaskExecution,
								variables: Dict[str, Any]):
		"""Execute conditional logic task"""
		config = task.configuration
		condition_expression = config.get('condition', 'True')
		
		# Evaluate condition (simplified)
		# In a real implementation, this would use a proper expression evaluator
		try:
			# Simple condition evaluation
			if 'variables' in condition_expression:
				condition_result = len(variables) > 0
			else:
				condition_result = eval(condition_expression)
		except:
			condition_result = False
		
		execution.outputs['condition_result'] = condition_result
		execution.outputs['condition_met'] = condition_result
		
		execution.execution_logs.append({
			'timestamp': datetime.utcnow().isoformat(),
			'level': 'INFO',
			'message': f'Condition "{condition_expression}" evaluated to {condition_result}'
		})
	
	async def _execute_twin_operation(self, task: WorkflowTask, execution: TaskExecution,
									 variables: Dict[str, Any]):
		"""Execute digital twin operation"""
		config = task.configuration
		twin_id = config.get('twin_id')
		operation = config.get('operation', 'read_state')
		parameters = config.get('parameters', {})
		
		if twin_id not in self.twin_registry:
			raise ValueError(f"Twin {twin_id} not found in registry")
		
		twin_info = self.twin_registry[twin_id]
		
		# Simulate twin operation
		await asyncio.sleep(1)
		
		if operation == 'read_state':
			result = {
				'twin_id': twin_id,
				'current_state': twin_info.get('state', {}),
				'last_updated': datetime.utcnow().isoformat(),
				'status': 'active'
			}
		elif operation == 'update_state':
			new_state = parameters.get('new_state', {})
			twin_info['state'] = {**twin_info.get('state', {}), **new_state}
			result = {
				'twin_id': twin_id,
				'operation': 'update_state',
				'updated_at': datetime.utcnow().isoformat(),
				'success': True
			}
		elif operation == 'run_simulation':
			simulation_params = parameters.get('simulation_params', {})
			result = {
				'twin_id': twin_id,
				'simulation_id': f"sim_{uuid.uuid4().hex[:8]}",
				'simulation_params': simulation_params,
				'results': {'output': 'simulation completed'},
				'completed_at': datetime.utcnow().isoformat()
			}
		else:
			result = {'twin_id': twin_id, 'operation': operation, 'status': 'completed'}
		
		execution.outputs['twin_operation_result'] = result
		execution.execution_logs.append({
			'timestamp': datetime.utcnow().isoformat(),
			'level': 'INFO',
			'message': f'Executed {operation} on twin {twin_id}'
		})
	
	async def _execute_custom_function(self, task: WorkflowTask, execution: TaskExecution,
									  variables: Dict[str, Any]):
		"""Execute custom function task"""
		config = task.configuration
		function_name = config.get('function_name')
		function_args = config.get('args', {})
		
		if function_name not in self.custom_functions:
			raise ValueError(f"Custom function {function_name} not found")
		
		function = self.custom_functions[function_name]
		
		# Execute custom function
		try:
			if asyncio.iscoroutinefunction(function):
				result = await function(**function_args)
			else:
				result = function(**function_args)
		except Exception as e:
			raise ValueError(f"Custom function {function_name} failed: {e}")
		
		execution.outputs['function_result'] = result
		execution.execution_logs.append({
			'timestamp': datetime.utcnow().isoformat(),
			'level': 'INFO',
			'message': f'Executed custom function {function_name}'
		})

class WorkflowScheduler:
	"""Handles workflow scheduling and triggering"""
	
	def __init__(self, orchestrator):
		self.orchestrator = orchestrator
		self.scheduled_workflows: Dict[str, Dict[str, Any]] = {}
		self.is_running = False
		self.scheduler_thread = None
	
	def start_scheduler(self):
		"""Start the workflow scheduler"""
		self.is_running = True
		self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
		self.scheduler_thread.daemon = True
		self.scheduler_thread.start()
		logger.info("Workflow scheduler started")
	
	def stop_scheduler(self):
		"""Stop the workflow scheduler"""
		self.is_running = False
		if self.scheduler_thread:
			self.scheduler_thread.join()
		logger.info("Workflow scheduler stopped")
	
	def schedule_workflow(self, workflow_id: str, trigger: WorkflowTrigger):
		"""Schedule workflow with trigger"""
		if trigger.trigger_type == TriggerType.SCHEDULE:
			cron_expression = trigger.configuration.get('cron', '0 * * * *')  # Every hour
			next_run = self._calculate_next_run(cron_expression)
			
			self.scheduled_workflows[workflow_id] = {
				'trigger': trigger,
				'next_run': next_run,
				'cron': cron_expression
			}
			
			logger.info(f"Scheduled workflow {workflow_id} with cron {cron_expression}")
	
	def _scheduler_loop(self):
		"""Main scheduler loop"""
		while self.is_running:
			try:
				current_time = datetime.utcnow()
				
				for workflow_id, schedule_info in list(self.scheduled_workflows.items()):
					if current_time >= schedule_info['next_run']:
						# Trigger workflow
						asyncio.create_task(
							self.orchestrator.execute_workflow(
								workflow_id,
								triggered_by='scheduler',
								trigger_data={'scheduled_time': current_time.isoformat()}
							)
						)
						
						# Calculate next run
						schedule_info['next_run'] = self._calculate_next_run(
							schedule_info['cron']
						)
				
				time.sleep(60)  # Check every minute
				
			except Exception as e:
				logger.error(f"Error in scheduler loop: {e}")
				time.sleep(60)
	
	def _calculate_next_run(self, cron_expression: str) -> datetime:
		"""Calculate next run time from cron expression"""
		# Simplified cron calculation - in reality, use a proper cron library
		return datetime.utcnow() + timedelta(hours=1)

class IntelligentOrchestrator:
	"""Main intelligent orchestration engine"""
	
	def __init__(self):
		self.workflows: Dict[str, OrchestrationWorkflow] = {}
		self.active_executions: Dict[str, WorkflowExecution] = {}
		self.execution_history: List[WorkflowExecution] = []
		self.task_executor = TaskExecutor()
		self.scheduler = WorkflowScheduler(self)
		
		# AI-driven orchestration
		self.orchestration_strategy = OrchestrationStrategy.DEPENDENCY_AWARE
		self.twin_network = nx.DiGraph()  # Network graph of twins
		self.performance_metrics: Dict[str, Dict[str, float]] = {}
		
		logger.info("Intelligent Orchestrator initialized")
	
	def create_workflow(self, name: str, description: str, created_by: str) -> str:
		"""Create new orchestration workflow"""
		workflow_id = f"workflow_{uuid.uuid4().hex[:12]}"
		
		workflow = OrchestrationWorkflow(
			workflow_id=workflow_id,
			name=name,
			description=description,
			version="1.0",
			status=WorkflowStatus.DRAFT,
			tasks={},
			triggers=[],
			global_variables={},
			metadata={},
			created_by=created_by,
			created_at=datetime.utcnow(),
			last_modified=datetime.utcnow(),
			execution_count=0,
			success_rate=0.0,
			average_duration_seconds=0.0
		)
		
		self.workflows[workflow_id] = workflow
		logger.info(f"Created workflow {name} with ID {workflow_id}")
		return workflow_id
	
	def add_task_to_workflow(self, workflow_id: str, task: WorkflowTask):
		"""Add task to workflow"""
		if workflow_id not in self.workflows:
			raise ValueError(f"Workflow {workflow_id} not found")
		
		workflow = self.workflows[workflow_id]
		workflow.tasks[task.task_id] = task
		workflow.last_modified = datetime.utcnow()
		
		logger.info(f"Added task {task.name} to workflow {workflow_id}")
	
	def add_trigger_to_workflow(self, workflow_id: str, trigger: WorkflowTrigger):
		"""Add trigger to workflow"""
		if workflow_id not in self.workflows:
			raise ValueError(f"Workflow {workflow_id} not found")
		
		workflow = self.workflows[workflow_id]
		workflow.triggers.append(trigger)
		workflow.last_modified = datetime.utcnow()
		
		# Schedule if it's a scheduled trigger
		if trigger.trigger_type == TriggerType.SCHEDULE and trigger.is_active:
			self.scheduler.schedule_workflow(workflow_id, trigger)
		
		logger.info(f"Added trigger {trigger.trigger_type.value} to workflow {workflow_id}")
	
	def activate_workflow(self, workflow_id: str):
		"""Activate workflow for execution"""
		if workflow_id not in self.workflows:
			raise ValueError(f"Workflow {workflow_id} not found")
		
		workflow = self.workflows[workflow_id]
		
		# Validate workflow
		if not self._validate_workflow(workflow):
			raise ValueError("Workflow validation failed")
		
		workflow.status = WorkflowStatus.ACTIVE
		workflow.last_modified = datetime.utcnow()
		
		logger.info(f"Activated workflow {workflow_id}")
	
	def _validate_workflow(self, workflow: OrchestrationWorkflow) -> bool:
		"""Validate workflow structure"""
		if not workflow.tasks:
			logger.error("Workflow has no tasks")
			return False
		
		# Check for circular dependencies
		graph = nx.DiGraph()
		for task_id, task in workflow.tasks.items():
			graph.add_node(task_id)
			for dep in task.dependencies:
				if dep in workflow.tasks:
					graph.add_edge(dep, task_id)
		
		if not nx.is_directed_acyclic_graph(graph):
			logger.error("Workflow has circular dependencies")
			return False
		
		# Check that all dependencies exist
		for task_id, task in workflow.tasks.items():
			for dep in task.dependencies:
				if dep not in workflow.tasks:
					logger.error(f"Task {task_id} depends on non-existent task {dep}")
					return False
		
		return True
	
	async def execute_workflow(self, workflow_id: str, triggered_by: str = "manual",
							  trigger_data: Dict[str, Any] = None) -> str:
		"""Execute workflow"""
		if workflow_id not in self.workflows:
			raise ValueError(f"Workflow {workflow_id} not found")
		
		workflow = self.workflows[workflow_id]
		
		if workflow.status != WorkflowStatus.ACTIVE:
			raise ValueError(f"Workflow {workflow_id} is not active")
		
		execution_id = f"exec_{uuid.uuid4().hex[:12]}"
		
		# Create workflow execution
		execution = WorkflowExecution(
			execution_id=execution_id,
			workflow_id=workflow_id,
			status=WorkflowStatus.RUNNING,
			start_time=datetime.utcnow(),
			end_time=None,
			triggered_by=triggered_by,
			trigger_data=trigger_data or {},
			task_executions={},
			variables=workflow.global_variables.copy(),
			error_message=None
		)
		
		# Create task executions
		for task_id, task in workflow.tasks.items():
			task_execution = TaskExecution(
				execution_id=f"task_exec_{uuid.uuid4().hex[:8]}",
				task_id=task_id,
				workflow_execution_id=execution_id,
				status=TaskStatus.PENDING,
				start_time=None,
				end_time=None,
				inputs={},
				outputs={},
				error_message=None,
				retry_attempt=0,
				execution_logs=[]
			)
			execution.task_executions[task_id] = task_execution
		
		self.active_executions[execution_id] = execution
		
		# Execute workflow asynchronously
		asyncio.create_task(self._execute_workflow_tasks(execution))
		
		logger.info(f"Started execution {execution_id} for workflow {workflow_id}")
		return execution_id
	
	async def _execute_workflow_tasks(self, execution: WorkflowExecution):
		"""Execute all tasks in workflow"""
		workflow = self.workflows[execution.workflow_id]
		
		try:
			# Build execution graph
			graph = nx.DiGraph()
			for task_id, task in workflow.tasks.items():
				graph.add_node(task_id, task=task)
				for dep in task.dependencies:
					graph.add_edge(dep, task_id)
			
			# Execute tasks in topological order
			execution_order = list(nx.topological_sort(graph))
			
			for task_id in execution_order:
				task = workflow.tasks[task_id]
				task_execution = execution.task_executions[task_id]
				
				# Check if dependencies completed successfully
				dependencies_met = True
				for dep_id in task.dependencies:
					dep_execution = execution.task_executions[dep_id]
					if dep_execution.status != TaskStatus.COMPLETED:
						dependencies_met = False
						break
					
					# Pass outputs from dependencies as inputs
					task_execution.inputs.update(dep_execution.outputs)
				
				if not dependencies_met:
					task_execution.status = TaskStatus.SKIPPED
					continue
				
				# Execute task with retries
				success = False
				for retry in range(task.retry_count + 1):
					task_execution.retry_attempt = retry
					
					try:
						success = await asyncio.wait_for(
							self.task_executor.execute_task(task, task_execution, execution.variables),
							timeout=task.timeout_seconds
						)
						
						if success:
							# Update workflow variables with task outputs
							execution.variables.update(task_execution.outputs)
							break
						
					except asyncio.TimeoutError:
						task_execution.error_message = f"Task timed out after {task.timeout_seconds} seconds"
					except Exception as e:
						task_execution.error_message = str(e)
					
					if retry < task.retry_count:
						await asyncio.sleep(task.retry_delay_seconds)
				
				if not success:
					task_execution.status = TaskStatus.FAILED
					execution.status = WorkflowStatus.FAILED
					execution.error_message = f"Task {task_id} failed after {task.retry_count + 1} attempts"
					break
			
			# Update workflow execution status
			if execution.status == WorkflowStatus.RUNNING:
				all_completed = all(
					te.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
					for te in execution.task_executions.values()
				)
				
				if all_completed:
					execution.status = WorkflowStatus.COMPLETED
				else:
					execution.status = WorkflowStatus.FAILED
					execution.error_message = "Some tasks failed to complete"
			
			execution.end_time = datetime.utcnow()
			
			# Update workflow statistics
			workflow.execution_count += 1
			duration = (execution.end_time - execution.start_time).total_seconds()
			
			if workflow.execution_count == 1:
				workflow.average_duration_seconds = duration
				workflow.success_rate = 1.0 if execution.status == WorkflowStatus.COMPLETED else 0.0
			else:
				# Update running averages
				workflow.average_duration_seconds = (
					(workflow.average_duration_seconds * (workflow.execution_count - 1) + duration) /
					workflow.execution_count
				)
				
				success_count = len([e for e in self.execution_history 
								   if e.workflow_id == workflow.workflow_id and e.status == WorkflowStatus.COMPLETED])
				if execution.status == WorkflowStatus.COMPLETED:
					success_count += 1
				
				workflow.success_rate = success_count / workflow.execution_count
			
			# Move to history
			self.execution_history.append(execution)
			del self.active_executions[execution.execution_id]
			
			logger.info(f"Workflow execution {execution.execution_id} completed with status {execution.status.value}")
			
		except Exception as e:
			execution.status = WorkflowStatus.FAILED
			execution.end_time = datetime.utcnow()
			execution.error_message = str(e)
			
			self.execution_history.append(execution)
			del self.active_executions[execution.execution_id]
			
			logger.error(f"Workflow execution {execution.execution_id} failed: {e}")
	
	def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
		"""Get workflow execution status"""
		# Check active executions
		if execution_id in self.active_executions:
			return self.active_executions[execution_id].to_dict()
		
		# Check execution history
		for execution in self.execution_history:
			if execution.execution_id == execution_id:
				return execution.to_dict()
		
		return None
	
	def get_workflow_statistics(self, workflow_id: str) -> Dict[str, Any]:
		"""Get comprehensive workflow statistics"""
		if workflow_id not in self.workflows:
			raise ValueError(f"Workflow {workflow_id} not found")
		
		workflow = self.workflows[workflow_id]
		
		# Get execution history for this workflow
		executions = [e for e in self.execution_history if e.workflow_id == workflow_id]
		
		# Calculate detailed statistics
		total_executions = len(executions)
		successful_executions = len([e for e in executions if e.status == WorkflowStatus.COMPLETED])
		failed_executions = len([e for e in executions if e.status == WorkflowStatus.FAILED])
		
		if executions:
			avg_duration = sum([
				(e.end_time - e.start_time).total_seconds() 
				for e in executions if e.end_time
			]) / len([e for e in executions if e.end_time])
			
			# Task statistics
			task_stats = {}
			for execution in executions:
				for task_id, task_exec in execution.task_executions.items():
					if task_id not in task_stats:
						task_stats[task_id] = {
							'total_runs': 0,
							'successful_runs': 0,
							'failed_runs': 0,
							'avg_duration': 0.0
						}
					
					task_stats[task_id]['total_runs'] += 1
					if task_exec.status == TaskStatus.COMPLETED:
						task_stats[task_id]['successful_runs'] += 1
					elif task_exec.status == TaskStatus.FAILED:
						task_stats[task_id]['failed_runs'] += 1
					
					if task_exec.start_time and task_exec.end_time:
						duration = (task_exec.end_time - task_exec.start_time).total_seconds()
						task_stats[task_id]['avg_duration'] = (
							(task_stats[task_id]['avg_duration'] * (task_stats[task_id]['total_runs'] - 1) + duration) /
							task_stats[task_id]['total_runs']
						)
		else:
			avg_duration = 0.0
			task_stats = {}
		
		return {
			'workflow_id': workflow_id,
			'workflow_name': workflow.name,
			'status': workflow.status.value,
			'total_executions': total_executions,
			'successful_executions': successful_executions,
			'failed_executions': failed_executions,
			'success_rate': workflow.success_rate,
			'average_duration_seconds': avg_duration,
			'task_count': len(workflow.tasks),
			'trigger_count': len(workflow.triggers),
			'task_statistics': task_stats,
			'last_execution': executions[-1].to_dict() if executions else None
		}
	
	def generate_visual_workflow_definition(self, workflow_id: str) -> Dict[str, Any]:
		"""Generate visual workflow definition for UI"""
		if workflow_id not in self.workflows:
			raise ValueError(f"Workflow {workflow_id} not found")
		
		workflow = self.workflows[workflow_id]
		
		# Build visual representation
		nodes = []
		edges = []
		
		for task_id, task in workflow.tasks.items():
			nodes.append({
				'id': task_id,
				'label': task.name,
				'type': task.task_type.value,
				'position': task.position,
				'configuration': task.configuration,
				'status': 'draft'  # Would be updated based on execution
			})
			
			# Add edges for dependencies
			for dep_id in task.dependencies:
				edges.append({
					'source': dep_id,
					'target': task_id,
					'type': 'dependency'
				})
		
		return {
			'workflow_id': workflow_id,
			'name': workflow.name,
			'description': workflow.description,
			'nodes': nodes,
			'edges': edges,
			'triggers': [t.to_dict() for t in workflow.triggers],
			'variables': workflow.global_variables,
			'metadata': workflow.metadata
		}

# Test and example usage
async def test_intelligent_orchestration():
	"""Test the intelligent orchestration system"""
	
	# Initialize orchestrator
	orchestrator = IntelligentOrchestrator()
	orchestrator.scheduler.start_scheduler()
	
	print("Creating intelligent orchestration workflow...")
	
	# Create workflow
	workflow_id = orchestrator.create_workflow(
		name="Industrial Twin Monitoring Workflow",
		description="Automated monitoring and analysis of industrial digital twins",
		created_by="system_admin"
	)
	
	# Register some digital twins
	orchestrator.task_executor.register_twin("twin_001", {
		'name': 'Manufacturing Line A',
		'type': 'production_line',
		'state': {'temperature': 75, 'pressure': 12, 'status': 'operational'}
	})
	
	orchestrator.task_executor.register_twin("twin_002", {
		'name': 'Quality Control Station',
		'type': 'qc_station',
		'state': {'defect_rate': 0.02, 'throughput': 150, 'status': 'active'}
	})
	
	# Register custom function
	def calculate_efficiency(temperature, pressure, throughput):
		"""Custom efficiency calculation"""
		base_efficiency = 0.85
		temp_factor = max(0, 1 - abs(temperature - 70) / 100)
		pressure_factor = max(0, 1 - abs(pressure - 10) / 20)
		throughput_factor = min(1, throughput / 200)
		
		return base_efficiency * temp_factor * pressure_factor * throughput_factor
	
	orchestrator.task_executor.register_custom_function("calculate_efficiency", calculate_efficiency)
	
	# Create workflow tasks
	tasks = [
		WorkflowTask(
			task_id="collect_data",
			name="Collect Twin Data",
			task_type=TaskType.DATA_COLLECTION,
			configuration={
				'source': 'twin_sensors',
				'data_type': 'operational_metrics'
			},
			dependencies=[],
			timeout_seconds=30,
			retry_count=2,
			retry_delay_seconds=5,
			position={'x': 100, 'y': 100},
			outputs={'collected_data': 'dict'},
			conditions=[],
			parallel_execution=False
		),
		
		WorkflowTask(
			task_id="read_twin_001",
			name="Read Manufacturing Line State",
			task_type=TaskType.TWIN_OPERATION,
			configuration={
				'twin_id': 'twin_001',
				'operation': 'read_state'
			},
			dependencies=["collect_data"],
			timeout_seconds=20,
			retry_count=1,
			retry_delay_seconds=3,
			position={'x': 300, 'y': 100},
			outputs={'twin_state': 'dict'},
			conditions=[],
			parallel_execution=True
		),
		
		WorkflowTask(
			task_id="read_twin_002",
			name="Read QC Station State",
			task_type=TaskType.TWIN_OPERATION,
			configuration={
				'twin_id': 'twin_002',
				'operation': 'read_state'
			},
			dependencies=["collect_data"],
			timeout_seconds=20,
			retry_count=1,
			retry_delay_seconds=3,
			position={'x': 300, 'y': 250},
			outputs={'twin_state': 'dict'},
			conditions=[],
			parallel_execution=True
		),
		
		WorkflowTask(
			task_id="analyze_performance",
			name="Analyze Performance Metrics",
			task_type=TaskType.ANALYSIS,
			configuration={
				'analysis_type': 'statistical'
			},
			dependencies=["read_twin_001", "read_twin_002"],
			timeout_seconds=60,
			retry_count=1,
			retry_delay_seconds=5,
			position={'x': 500, 'y': 175},
			outputs={'analysis_results': 'dict'},
			conditions=[],
			parallel_execution=False
		),
		
		WorkflowTask(
			task_id="calculate_efficiency",
			name="Calculate System Efficiency",
			task_type=TaskType.CUSTOM_FUNCTION,
			configuration={
				'function_name': 'calculate_efficiency',
				'args': {
					'temperature': 75,
					'pressure': 12,
					'throughput': 150
				}
			},
			dependencies=["analyze_performance"],
			timeout_seconds=30,
			retry_count=1,
			retry_delay_seconds=3,
			position={'x': 700, 'y': 175},
			outputs={'efficiency_score': 'float'},
			conditions=[],
			parallel_execution=False
		),
		
		WorkflowTask(
			task_id="check_alerts",
			name="Check Alert Conditions",
			task_type=TaskType.CONDITION,
			configuration={
				'condition': 'efficiency_score < 0.7'
			},
			dependencies=["calculate_efficiency"],
			timeout_seconds=10,
			retry_count=0,
			retry_delay_seconds=0,
			position={'x': 900, 'y': 175},
			outputs={'alert_needed': 'bool'},
			conditions=[],
			parallel_execution=False
		),
		
		WorkflowTask(
			task_id="send_notification",
			name="Send Alert Notification",
			task_type=TaskType.NOTIFICATION,
			configuration={
				'type': 'email',
				'recipients': ['operations@company.com'],
				'message': 'System efficiency below threshold - immediate attention required'
			},
			dependencies=["check_alerts"],
			timeout_seconds=30,
			retry_count=2,
			retry_delay_seconds=5,
			position={'x': 1100, 'y': 175},
			outputs={'notification_sent': 'dict'},
			conditions=[{'field': 'alert_needed', 'operator': 'equals', 'value': True}],
			parallel_execution=False
		)
	]
	
	# Add tasks to workflow
	for task in tasks:
		orchestrator.add_task_to_workflow(workflow_id, task)
	
	# Add triggers
	manual_trigger = WorkflowTrigger(
		trigger_id=f"trigger_{uuid.uuid4().hex[:8]}",
		trigger_type=TriggerType.MANUAL,
		configuration={},
		is_active=True,
		last_triggered=None,
		trigger_count=0
	)
	
	scheduled_trigger = WorkflowTrigger(
		trigger_id=f"trigger_{uuid.uuid4().hex[:8]}",
		trigger_type=TriggerType.SCHEDULE,
		configuration={'cron': '0 */6 * * *'},  # Every 6 hours
		is_active=True,
		last_triggered=None,
		trigger_count=0
	)
	
	orchestrator.add_trigger_to_workflow(workflow_id, manual_trigger)
	orchestrator.add_trigger_to_workflow(workflow_id, scheduled_trigger)
	
	# Activate workflow
	orchestrator.activate_workflow(workflow_id)
	
	print(f"Created and activated workflow: {workflow_id}")
	
	# Execute workflow manually
	print("\nExecuting workflow manually...")
	execution_id = await orchestrator.execute_workflow(
		workflow_id,
		triggered_by="manual_test",
		trigger_data={'test_execution': True}
	)
	
	print(f"Started execution: {execution_id}")
	
	# Monitor execution
	for i in range(10):
		await asyncio.sleep(2)
		status = orchestrator.get_execution_status(execution_id)
		if status:
			print(f"Execution status: {status['status']}")
			
			if status['status'] in ['completed', 'failed']:
				print("\nTask execution details:")
				for task_id, task_exec in status['task_executions'].items():
					print(f"  {task_id}: {task_exec['status']}")
					if task_exec.get('execution_logs'):
						for log in task_exec['execution_logs']:
							print(f"    LOG: {log['message']}")
				break
		else:
			print("Execution not found")
			break
	
	# Get workflow statistics
	print("\nWorkflow statistics:")
	stats = orchestrator.get_workflow_statistics(workflow_id)
	print(f"  Total executions: {stats['total_executions']}")
	print(f"  Success rate: {stats['success_rate']:.1%}")
	print(f"  Average duration: {stats['average_duration_seconds']:.1f} seconds")
	print(f"  Task count: {stats['task_count']}")
	
	# Generate visual workflow definition
	print("\nGenerating visual workflow definition...")
	visual_def = orchestrator.generate_visual_workflow_definition(workflow_id)
	print(f"  Workflow: {visual_def['name']}")
	print(f"  Nodes: {len(visual_def['nodes'])}")
	print(f"  Edges: {len(visual_def['edges'])}")
	print(f"  Triggers: {len(visual_def['triggers'])}")
	
	# Stop scheduler
	orchestrator.scheduler.stop_scheduler()

if __name__ == "__main__":
	asyncio.run(test_intelligent_orchestration())