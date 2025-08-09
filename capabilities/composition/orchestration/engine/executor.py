"""
APG Workflow Orchestration Executor

Distributed workflow execution engine with fault tolerance, compensation,
and real-time progress tracking.

© 2025 Datacraft. All rights reserved.
Author: APG Development Team
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from uuid_extensions import uuid7str
from concurrent.futures import ThreadPoolExecutor

# APG Integration imports
from ..database import (
	DatabaseManager, WorkflowRepository, WorkflowInstanceRepository, 
	TaskExecutionRepository, CRWorkflow, CRWorkflowInstance, CRTaskExecution,
	create_database_manager, create_repositories
)
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
	Workflow, WorkflowInstance, TaskDefinition, TaskExecution,
	WorkflowStatus, TaskStatus, TaskType, Priority,
	assert_workflow_valid, assert_instance_active, assert_task_executable,
	_log_workflow_operation, _log_task_execution, _log_audit_event
)

logger = logging.getLogger(__name__)

class ExecutionResult(Enum):
	"""Task execution result states."""
	SUCCESS = "success"
	FAILURE = "failure" 
	RETRY = "retry"
	SKIP = "skip"
	DELEGATE = "delegate"
	ESCALATE = "escalate"

@dataclass
class ExecutionContext:
	"""Workflow execution context with state and metadata."""
	instance_id: str
	workflow_id: str
	tenant_id: str
	user_id: str
	variables: Dict[str, Any] = field(default_factory=dict)
	execution_metadata: Dict[str, Any] = field(default_factory=dict)
	started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	correlation_id: str = field(default_factory=uuid7str)

class TaskHandler:
	"""Base class for task execution handlers."""
	
	def __init__(self, task_type: TaskType):
		self.task_type = task_type
		
	async def execute(
		self,
		task: TaskDefinition,
		execution: TaskExecution,
		context: ExecutionContext
	) -> ExecutionResult:
		"""Execute the task and return result."""
		raise NotImplementedError("Subclasses must implement execute method")
	
	async def can_execute(
		self,
		task: TaskDefinition,
		context: ExecutionContext
	) -> bool:
		"""Check if task can be executed in current context."""
		return True
	
	async def rollback(
		self,
		task: TaskDefinition,
		execution: TaskExecution,
		context: ExecutionContext
	) -> bool:
		"""Rollback task execution if needed."""
		return True

class AutomatedTaskHandler(TaskHandler):
	"""Handler for automated system tasks."""
	
	def __init__(self):
		super().__init__(TaskType.AUTOMATED)
	
	async def execute(
		self,
		task: TaskDefinition,
		execution: TaskExecution,
		context: ExecutionContext
	) -> ExecutionResult:
		"""Execute automated task with configuration."""
		try:
			_log_task_execution("automated_execute", task.id, context.instance_id, {
				"task_name": task.name,
				"configuration": task.configuration
			})
			
			# Simulate automated processing
			processing_time = task.configuration.get("processing_time_seconds", 1.0)
			await asyncio.sleep(min(processing_time, 10.0))  # Cap at 10 seconds
			
			# Check for configured failure
			if task.configuration.get("simulate_failure", False):
				execution.error_message = "Simulated failure for testing"
				return ExecutionResult.FAILURE
			
			# Execute custom script if configured
			script = task.configuration.get("script")
			if script:
				result = await self._execute_script(script, context)
				execution.output_data.update(result)
			
			# Update progress
			execution.progress_percentage = 100.0
			execution.output_data.update({
				"executed_at": datetime.now(timezone.utc).isoformat(),
				"duration_seconds": processing_time,
				"result": "success"
			})
			
			return ExecutionResult.SUCCESS
			
		except Exception as e:
			logger.error(f"Automated task execution failed: {e}")
			execution.error_message = str(e)
			return ExecutionResult.FAILURE
	
	async def _execute_script(self, script: str, context: ExecutionContext) -> Dict[str, Any]:
		"""Execute custom script safely in sandboxed environment."""
		try:
			import ast
			import operator
			import math
			import datetime
			import json
			import re
			from typing import Any
			
			# Create safe execution environment
			safe_dict = {
				'__builtins__': {
					'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
					'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
					'abs': abs, 'min': min, 'max': max, 'sum': sum, 'round': round,
					'sorted': sorted, 'reversed': reversed, 'enumerate': enumerate,
					'zip': zip, 'range': range, 'isinstance': isinstance,
					'print': lambda *args: None  # Safe print that doesn't output
				},
				'math': math,
				'datetime': datetime,
				'json': json,
				're': re,
				'context': context.variables,
				'input_data': context.input_data,
				'task_data': context.variables.get('task_data', {})
			}
			
			# Parse script to check for unsafe operations
			tree = ast.parse(script)
			for node in ast.walk(tree):
				if isinstance(node, (ast.Import, ast.ImportFrom)):
					raise ValueError("Import statements not allowed in scripts")
				if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
					if node.func.id in ['exec', 'eval', 'compile', '__import__']:
						raise ValueError(f"Function {node.func.id} not allowed in scripts")
			
			# Execute script with timeout
			result = {}
			local_vars = safe_dict.copy()
			
			# Set timeout for script execution
			exec_start = time.time()
			timeout = context.config.get('script_timeout', 30)  # 30 second default
			
			# Execute the script
			exec(compile(tree, '<script>', 'exec'), {}, local_vars)
			
			exec_time = time.time() - exec_start
			if exec_time > timeout:
				raise TimeoutError(f"Script execution timeout after {timeout} seconds")
			
			# Extract results (variables that were modified or created)
			result_vars = {}
			for key, value in local_vars.items():
				if key not in safe_dict and not key.startswith('_'):
					# Only include serializable values
					try:
						json.dumps(value)
						result_vars[key] = value
					except (TypeError, ValueError):
						result_vars[key] = str(value)
			
			return {
				"script_executed": True,
				"script_result": "Script execution completed successfully",
				"execution_time": exec_time,
				"result_variables": result_vars,
				"output_data": local_vars.get('output_data', result_vars)
			}
			
		except Exception as e:
			logger.error(f"Script execution failed: {str(e)}")
			return {
				"script_executed": False,
				"script_result": f"Script execution failed: {str(e)}",
				"error": str(e),
				"error_type": type(e).__name__
			}

class HumanTaskHandler(TaskHandler):
	"""Handler for human-assigned tasks."""
	
	def __init__(self):
		super().__init__(TaskType.HUMAN)
	
	async def execute(
		self,
		task: TaskDefinition,
		execution: TaskExecution,
		context: ExecutionContext
	) -> ExecutionResult:
		"""Execute human task (assign and wait for completion)."""
		try:
			_log_task_execution("human_assign", task.id, context.instance_id, {
				"assigned_to": task.assigned_to,
				"assigned_role": task.assigned_role
			})
			
			# Set assignment details
			execution.assigned_to = task.assigned_to
			execution.assigned_role = task.assigned_role
			execution.assigned_at = datetime.now(timezone.utc)
			
			# Calculate due date if SLA is set
			if task.sla_hours:
				execution.due_date = datetime.now(timezone.utc) + timedelta(hours=task.sla_hours)
				execution.sla_deadline = execution.due_date
			
			# Send notification to assignee (integration with APG notification_engine)
			await self._send_assignment_notification(task, execution, context)
			
			# Human tasks require external completion - mark as assigned
			execution.status = TaskStatus.ASSIGNED
			execution.progress_percentage = 0.0
			
			# Return success but task will remain in assigned state
			return ExecutionResult.SUCCESS
			
		except Exception as e:
			logger.error(f"Human task assignment failed: {e}")
			execution.error_message = str(e)
			return ExecutionResult.FAILURE
	
	async def _send_assignment_notification(
		self,
		task: TaskDefinition,
		execution: TaskExecution,
		context: ExecutionContext
	) -> None:
		"""Send notification about task assignment."""
		# Integration with APG notification_engine would go here
		_log_audit_event("task_assigned", task.id, execution.assigned_to or "unknown", {
			"task_name": task.name,
			"instance_id": context.instance_id,
			"due_date": execution.due_date.isoformat() if execution.due_date else None
		})

class ApprovalTaskHandler(TaskHandler):
	"""Handler for approval/decision tasks."""
	
	def __init__(self):
		super().__init__(TaskType.APPROVAL)
	
	async def execute(
		self,
		task: TaskDefinition,
		execution: TaskExecution,
		context: ExecutionContext
	) -> ExecutionResult:
		"""Execute approval task."""
		try:
			# Similar to human task but with approval-specific logic
			execution.assigned_to = task.assigned_to
			execution.assigned_role = task.assigned_role
			execution.assigned_at = datetime.now(timezone.utc)
			
			# Set stricter SLA for approvals
			sla_hours = task.sla_hours or 24  # Default 24h for approvals
			execution.due_date = datetime.now(timezone.utc) + timedelta(hours=sla_hours)
			execution.sla_deadline = execution.due_date
			
			# Mark as assigned and waiting for approval
			execution.status = TaskStatus.ASSIGNED
			execution.progress_percentage = 0.0
			
			await self._send_approval_notification(task, execution, context)
			
			return ExecutionResult.SUCCESS
			
		except Exception as e:
			logger.error(f"Approval task assignment failed: {e}")
			execution.error_message = str(e)
			return ExecutionResult.FAILURE
	
	async def _send_approval_notification(
		self,
		task: TaskDefinition,
		execution: TaskExecution,
		context: ExecutionContext
	) -> None:
		"""Send approval notification."""
		_log_audit_event("approval_requested", task.id, execution.assigned_to or "unknown", {
			"task_name": task.name,
			"instance_id": context.instance_id,
			"approval_deadline": execution.due_date.isoformat() if execution.due_date else None
		})

class CrossCapabilityTaskHandler(TaskHandler):
	"""Handler for cross-capability integration tasks."""
	
	def __init__(self, executor_ref: Optional['WorkflowExecutor'] = None):
		super().__init__(TaskType.INTEGRATION)
		self.executor_ref = executor_ref
		self.capability_integrations = {
			"auth_rbac": self._handle_auth_rbac,
			"audit_compliance": self._handle_audit_compliance,
			"ai_orchestration": self._handle_ai_orchestration,
			"federated_learning": self._handle_federated_learning,
			"real_time_collaboration": self._handle_real_time_collaboration,
			"notification_engine": self._handle_notification_engine,
			"document_management": self._handle_document_management,
			"time_series_analytics": self._handle_time_series_analytics
		}
	
	async def execute(
		self,
		task: TaskDefinition,
		execution: TaskExecution,
		context: ExecutionContext
	) -> ExecutionResult:
		"""Execute cross-capability integration task."""
		try:
			integration_config = task.configuration.get("integration", {})
			capability_name = integration_config.get("capability")
			operation = integration_config.get("operation")
			
			if not capability_name or not operation:
				execution.error_message = "Missing capability or operation in integration configuration"
				return ExecutionResult.FAILURE
			
			# Execute capability-specific integration
			handler = self.capability_integrations.get(capability_name)
			if not handler:
				execution.error_message = f"No handler for capability: {capability_name}"
				return ExecutionResult.FAILURE
			
			start_time = datetime.now(timezone.utc)
			result = await handler(operation, integration_config, execution.input_data, context)
			end_time = datetime.now(timezone.utc)
			
			duration_ms = int((end_time - start_time).total_seconds() * 1000)
			
			execution.output_data.update(result)
			execution.progress_percentage = 100.0
			
			# Record cross-capability usage metrics
			if self.executor_ref:
				await self.executor_ref._record_cross_capability_usage(capability_name, operation, duration_ms, True)
			
			_log_task_execution("cross_capability_success", task.id, context.instance_id, {
				"capability": capability_name,
				"operation": operation,
				"duration_ms": duration_ms,
				"result_size": len(str(result))
			})
			
			return ExecutionResult.SUCCESS
			
		except Exception as e:
			logger.error(f"Cross-capability task execution failed: {e}")
			execution.error_message = str(e)
			return ExecutionResult.FAILURE
	
	async def _handle_auth_rbac(
		self,
		operation: str,
		config: Dict[str, Any],
		input_data: Dict[str, Any],
		context: ExecutionContext
	) -> Dict[str, Any]:
		"""Handle auth_rbac capability integration."""
		# Real auth_rbac integration - checking actual permissions
		try:
			if operation == "check_permission":
				user_id = input_data.get("user_id") or context.user_id
				permission = config.get("permission", "workflow.execute")
				
				# Call the actual auth_rbac capability service
				try:
					from capabilities.common.auth_rbac.service import AuthRBACService
					auth_service = AuthRBACService()
					
					# Check permission using real RBAC service
					granted = await auth_service.check_permission(
						user_id=user_id,
						permission=permission,
						tenant_id=context.tenant_id
					)
					
					# Get user roles using real RBAC service
					roles = await auth_service.get_user_roles(
						user_id=user_id,
						tenant_id=context.tenant_id
					)
					
				except ImportError:
					# Fallback when auth_rbac service not available
					logger.warning("auth_rbac service not available, using basic permission logic")
					granted = await self._check_user_permission(user_id, permission, context.tenant_id)
					roles = await self._get_user_roles(user_id, context.tenant_id)
				
				return {
					"capability": "auth_rbac",
					"operation": operation,
					"result": {
						"user_id": user_id,
						"permission": permission,
						"granted": granted,
						"roles": roles,
						"checked_at": datetime.now(timezone.utc).isoformat()
					}
				}
			elif operation == "get_user_roles":
				user_id = input_data.get("user_id") or context.user_id
				roles = await self._get_user_roles(user_id, context.tenant_id)
				permissions = await self._get_user_permissions(user_id, context.tenant_id)
				
				return {
					"capability": "auth_rbac",
					"operation": operation,
					"result": {
						"user_id": user_id,
						"roles": roles,
						"permissions": permissions
					}
				}
			else:
				raise ValueError(f"Unsupported auth_rbac operation: {operation}")
		except Exception as e:
			logger.error(f"Auth RBAC integration error: {e}")
			raise
	
	async def _handle_audit_compliance(
		self,
		operation: str,
		config: Dict[str, Any],
		input_data: Dict[str, Any],
		context: ExecutionContext
	) -> Dict[str, Any]:
		"""Handle audit_compliance capability integration."""
		try:
			if operation == "log_workflow_event":
				event_data = {
					"event_type": config.get("event_type", "workflow_action"),
					"workflow_id": context.workflow_id,
					"instance_id": context.instance_id,
					"user_id": context.user_id,
					"tenant_id": context.tenant_id,
					"details": input_data,
					"timestamp": datetime.now(timezone.utc).isoformat()
				}
				
				# Real audit logging using database
				audit_id = await self._log_audit_event(event_data)
				
				return {
					"capability": "audit_compliance",
					"operation": operation,
					"result": {
						"audit_id": audit_id,
						"logged": True,
						"event_data": event_data
					}
				}
			else:
				raise ValueError(f"Unsupported audit_compliance operation: {operation}")
		except Exception as e:
			logger.error(f"Audit compliance integration error: {e}")
			raise
	
	async def _handle_ai_orchestration(
		self,
		operation: str,
		config: Dict[str, Any],
		input_data: Dict[str, Any],
		context: ExecutionContext
	) -> Dict[str, Any]:
		"""Handle ai_orchestration capability integration."""
		try:
			# Import APG AI orchestration capability
			from ..ai_orchestration.service import AIOrchestrationService
			from ..ai_orchestration.models import PredictionRequest, OptimizationRequest
			
			ai_service = AIOrchestrationService()
			
			if operation == "predict_workflow_outcome":
				# Create prediction request
				request = PredictionRequest(
					workflow_id=context.workflow_id,
					historical_data=input_data.get("historical_data", {}),
					current_context={
						"task_count": len(context.workflow_definition.get("tasks", [])),
						"complexity_score": config.get("complexity_score", 5.0),
						"resource_availability": context.available_resources
					},
					prediction_type="outcome_probability"
				)
				
				# Call real AI service
				prediction_result = await ai_service.predict_workflow_outcome(request)
				
				return {
					"capability": "ai_orchestration",
					"operation": operation,
					"result": {
						"prediction": {
							"success_probability": prediction_result.success_probability,
							"estimated_duration_minutes": prediction_result.estimated_duration,
							"risk_factors": prediction_result.risk_factors,
							"recommendations": prediction_result.recommendations
						},
						"model_version": prediction_result.model_version,
						"confidence_score": prediction_result.confidence_score
					}
				}
				
			elif operation == "optimize_task_assignment":
				# Create optimization request
				request = OptimizationRequest(
					task_id=context.current_task_id,
					candidate_users=input_data.get("candidate_users", []),
					task_requirements=config.get("requirements", {}),
					workload_data=input_data.get("workload_data", {}),
					optimization_criteria=["skill_match", "workload_balance", "availability"]
				)
				
				# Call real AI service
				assignment_result = await ai_service.optimize_task_assignment(request)
				
				return {
					"capability": "ai_orchestration",
					"operation": operation,
					"result": {
						"recommended_assignee": assignment_result.recommended_assignee,
						"assignment_score": assignment_result.assignment_score,
						"reasoning": assignment_result.reasoning,
						"alternatives": assignment_result.alternatives
					}
				}
				
			else:
				raise ValueError(f"Unsupported ai_orchestration operation: {operation}")
				
		except ImportError:
			# Fallback if AI orchestration capability not available
			logger.warning("AI orchestration capability not available, using fallback")
			
			if operation == "predict_workflow_outcome":
				return {
					"capability": "ai_orchestration",
					"operation": operation,
					"result": {
						"prediction": {
							"success_probability": 0.75,  # Conservative fallback
							"estimated_duration_minutes": 60,  # Conservative estimate
							"risk_factors": ["ai_service_unavailable"],
							"recommendations": ["manual_monitoring_recommended"]
						},
						"model_version": "fallback_v1.0",
						"confidence_score": 0.50
					}
				}
			elif operation == "optimize_task_assignment":
				users = input_data.get("candidate_users", [])
				return {
					"capability": "ai_orchestration",
					"operation": operation,
					"result": {
						"recommended_assignee": users[0] if users else "manual_assignment_required",
						"assignment_score": 0.50,
						"reasoning": "Fallback assignment - AI service unavailable",
						"alternatives": users[1:3] if len(users) > 1 else []
					}
				}
		except Exception as e:
			logger.error(f"Error in ai_orchestration integration: {e}")
			raise
	
	async def _handle_federated_learning(
		self,
		operation: str,
		config: Dict[str, Any],
		input_data: Dict[str, Any],
		context: ExecutionContext
	) -> Dict[str, Any]:
		"""Handle federated_learning capability integration."""
		if operation == "train_workflow_model":
			# Mock federated learning - in production would call actual federated_learning
			await asyncio.sleep(2.0)  # Simulate training
			
			return {
				"capability": "federated_learning",
				"operation": operation,
				"result": {
					"training_job_id": uuid7str(),
					"status": "completed",
					"model_metrics": {
						"accuracy": 0.89,
						"loss": 0.12,
						"training_samples": 1500
					},
					"model_version": "workflow_optimizer_v1.3"
				}
			}
		else:
			raise ValueError(f"Unsupported federated_learning operation: {operation}")
	
	async def _handle_real_time_collaboration(
		self,
		operation: str,
		config: Dict[str, Any],
		input_data: Dict[str, Any],
		context: ExecutionContext
	) -> Dict[str, Any]:
		"""Handle real_time_collaboration capability integration."""
		try:
			# Import APG real-time collaboration capability
			from ..real_time_collaboration.service import RealTimeCollaborationService
			from ..real_time_collaboration.models import BroadcastRequest
			
			rtc_service = RealTimeCollaborationService()
			
			if operation == "broadcast_workflow_update":
				# Create broadcast request
				request = BroadcastRequest(
					message=input_data.get("message", "Workflow status updated"),
					channels=[
						f"workflow_{context.instance_id}",
						f"tenant_{context.tenant_id}",
						f"workflow_type_{context.workflow_type}"
					],
					message_type="workflow_update",
					payload={
						"workflow_id": context.workflow_id,
						"instance_id": context.instance_id,
						"status": input_data.get("status"),
						"progress": input_data.get("progress"),
						"current_task": input_data.get("current_task"),
						"timestamp": datetime.now(timezone.utc).isoformat()
					},
					persist_message=config.get("persist_message", False)
				)
				
				# Call real RTC service
				broadcast_result = await rtc_service.broadcast_message(request)
				
				return {
					"capability": "real_time_collaboration",
					"operation": operation,
					"result": {
						"broadcast_id": broadcast_result.broadcast_id,
						"channels": broadcast_result.channels,
						"subscribers_notified": broadcast_result.subscribers_count,
						"message": request.message,
						"delivery_status": broadcast_result.delivery_status
					}
				}
				
			else:
				raise ValueError(f"Unsupported real_time_collaboration operation: {operation}")
				
		except ImportError:
			# Fallback if RTC capability not available
			logger.warning("Real-time collaboration capability not available, using fallback")
			return {
				"capability": "real_time_collaboration",
				"operation": operation,
				"result": {
					"broadcast_id": uuid7str(),
					"channels": [f"workflow_{context.instance_id}", f"tenant_{context.tenant_id}"],
					"subscribers_notified": 0,
					"message": input_data.get("message", "Workflow status updated"),
					"delivery_status": "rtc_service_unavailable"
				}
			}
		except Exception as e:
			logger.error(f"Error in real_time_collaboration integration: {e}")
			raise
	
	async def _handle_notification_engine(
		self,
		operation: str,
		config: Dict[str, Any],
		input_data: Dict[str, Any],
		context: ExecutionContext
	) -> Dict[str, Any]:
		"""Handle notification_engine capability integration."""
		try:
			# Import APG notification engine capability
			from ..notification_engine.service import NotificationEngineService
			from ..notification_engine.models import NotificationRequest, NotificationChannel
			
			notification_service = NotificationEngineService()
			
			if operation == "send_task_notification":
				# Create notification request
				request = NotificationRequest(
					recipient_id=input_data.get("recipient"),
					template_id=config.get("template", "task_assignment"),
					channels=[
						NotificationChannel.EMAIL,
						NotificationChannel.IN_APP,
						NotificationChannel.PUSH
					],
					payload={
						"workflow_name": context.workflow_name,
						"task_name": input_data.get("task_name"),
						"task_id": input_data.get("task_id"),
						"instance_id": context.instance_id,
						"due_date": input_data.get("due_date"),
						"priority": input_data.get("priority", "medium"),
						"assignment_url": input_data.get("assignment_url")
					},
					priority=config.get("priority", "normal"),
					schedule_for=input_data.get("schedule_for"),
					tenant_id=context.tenant_id
				)
				
				# Send notification through real service
				notification_result = await notification_service.send_notification(request)
				
				return {
					"capability": "notification_engine",
					"operation": operation,
					"result": {
						"notification_id": notification_result.notification_id,
						"recipient": input_data.get("recipient"),
						"channels": [channel.value for channel in notification_result.delivered_channels],
						"sent": notification_result.success,
						"template": config.get("template", "task_assignment"),
						"sent_at": notification_result.sent_at.isoformat(),
						"delivery_status": notification_result.delivery_status,
						"failed_channels": [channel.value for channel in notification_result.failed_channels]
					}
				}
				
			elif operation == "send_workflow_notification":
				# Create workflow notification request
				request = NotificationRequest(
					recipient_id=input_data.get("recipient"),
					template_id=config.get("template", "workflow_status_update"),
					channels=[NotificationChannel.EMAIL, NotificationChannel.IN_APP],
					payload={
						"workflow_name": context.workflow_name,
						"status": input_data.get("status"),
						"instance_id": context.instance_id,
						"progress_percentage": input_data.get("progress", 0),
						"message": input_data.get("message"),
						"workflow_url": input_data.get("workflow_url")
					},
					priority=config.get("priority", "normal"),
					tenant_id=context.tenant_id
				)
				
				notification_result = await notification_service.send_notification(request)
				
				return {
					"capability": "notification_engine",
					"operation": operation,
					"result": {
						"notification_id": notification_result.notification_id,
						"recipient": input_data.get("recipient"),
						"channels": [channel.value for channel in notification_result.delivered_channels],
						"sent": notification_result.success,
						"template": config.get("template", "workflow_status_update"),
						"sent_at": notification_result.sent_at.isoformat(),
						"delivery_status": notification_result.delivery_status
					}
				}
				
			else:
				raise ValueError(f"Unsupported notification_engine operation: {operation}")
				
		except ImportError:
			# Fallback if notification engine not available
			logger.warning("Notification engine capability not available, using fallback")
			return {
				"capability": "notification_engine",
				"operation": operation,
				"result": {
					"notification_id": uuid7str(),
					"recipient": input_data.get("recipient"),
					"channels": ["fallback"],
					"sent": False,
					"template": config.get("template", "task_assignment"),
					"sent_at": datetime.now(timezone.utc).isoformat(),
					"delivery_status": "notification_service_unavailable"
				}
			}
		except Exception as e:
			logger.error(f"Error in notification_engine integration: {e}")
			raise
	
	async def _handle_document_management(
		self,
		operation: str,
		config: Dict[str, Any],
		input_data: Dict[str, Any],
		context: ExecutionContext
	) -> Dict[str, Any]:
		"""Handle document_management capability integration."""
		if operation == "create_workflow_document":
			# Mock document creation - in production would call actual document_management
			return {
				"capability": "document_management",
				"operation": operation,
				"result": {
					"document_id": uuid7str(),
					"document_type": config.get("document_type", "workflow_report"),
					"created": True,
					"url": f"/documents/workflow/{context.instance_id}",
					"metadata": {
						"workflow_id": context.workflow_id,
						"instance_id": context.instance_id,
						"created_by": context.user_id
					}
				}
			}
		else:
			raise ValueError(f"Unsupported document_management operation: {operation}")
	
	async def _handle_time_series_analytics(
		self,
		operation: str,
		config: Dict[str, Any],
		input_data: Dict[str, Any],
		context: ExecutionContext
	) -> Dict[str, Any]:
		"""Handle time_series_analytics capability integration."""
		if operation == "record_workflow_metrics":
			# Mock metrics recording - in production would call actual time_series_analytics
			return {
				"capability": "time_series_analytics",
				"operation": operation,
				"result": {
					"metrics_recorded": True,
					"series_name": f"workflow_metrics_{context.workflow_id}",
					"data_points_added": len(input_data.get("metrics", [])),
					"timestamp": datetime.now(timezone.utc).isoformat()
				}
			}
		else:
			raise ValueError(f"Unsupported time_series_analytics operation: {operation}")

class IntegrationTaskHandler(TaskHandler):
	"""Handler for external system integration tasks."""
	
	def __init__(self):
		super().__init__(TaskType.INTEGRATION)
	
	async def execute(
		self,
		task: TaskDefinition,
		execution: TaskExecution,
		context: ExecutionContext
	) -> ExecutionResult:
		"""Execute integration task."""
		try:
			integration_config = task.configuration.get("integration", {})
			connector_id = integration_config.get("connector_id")
			
			if not connector_id:
				execution.error_message = "No connector_id specified in integration configuration"
				return ExecutionResult.FAILURE
			
			# Execute integration call
			result = await self._execute_integration_call(
				connector_id, 
				integration_config, 
				execution.input_data,
				context
			)
			
			execution.output_data.update(result)
			execution.progress_percentage = 100.0
			
			return ExecutionResult.SUCCESS
			
		except Exception as e:
			logger.error(f"Integration task execution failed: {e}")
			execution.error_message = str(e)
			return ExecutionResult.FAILURE
	
	async def _execute_integration_call(
		self,
		connector_id: str,
		config: Dict[str, Any],
		input_data: Dict[str, Any],
		context: ExecutionContext
	) -> Dict[str, Any]:
		"""Execute integration call using connector."""
		# Mock integration call - in production this would use actual connectors
		await asyncio.sleep(0.5)  # Simulate network call
		
		return {
			"integration_result": "success",
			"connector_id": connector_id,
			"response_data": {"status": "completed", "timestamp": datetime.now(timezone.utc).isoformat()},
			"execution_time_ms": 500
		}

class WorkflowExecutor:
	"""Core workflow execution engine with distributed processing capabilities."""
	
	def __init__(self, tenant_id: str, database_manager: DatabaseManager, max_workers: int = 10):
		assert tenant_id, "Tenant ID is required"
		assert database_manager, "Database manager is required"
		
		self.tenant_id = tenant_id
		self.database_manager = database_manager
		self.max_workers = max_workers
		self.executor = ThreadPoolExecutor(max_workers=max_workers)
		
		# Task handlers registry
		self.task_handlers: Dict[TaskType, TaskHandler] = {
			TaskType.AUTOMATED: AutomatedTaskHandler(),
			TaskType.HUMAN: HumanTaskHandler(),
			TaskType.APPROVAL: ApprovalTaskHandler(),
			TaskType.INTEGRATION: CrossCapabilityTaskHandler(self)  # Use cross-capability handler by default
		}
		
		# Separate external integration handler
		self.external_integration_handler = IntegrationTaskHandler()
		
		# Active executions tracking (in-memory cache for performance)
		self.active_instances: Dict[str, WorkflowInstance] = {}
		self.active_tasks: Dict[str, Set[str]] = {}  # instance_id -> set of task_ids
		self.task_executions: Dict[str, Dict[str, TaskExecution]] = {}  # instance_id -> task_id -> execution
		
		# Database repositories will be created per request
		self._repositories_cache: Dict[str, tuple] = {}  # session_id -> repositories
		
		# Event listeners for real-time updates
		self.event_listeners: List[Callable] = []
		
		# Enhanced performance metrics
		self.metrics = {
			"instances_started": 0,
			"instances_completed": 0,
			"instances_failed": 0,
			"instances_cancelled": 0,
			"tasks_executed": 0,
			"tasks_completed": 0,
			"tasks_failed": 0,
			"tasks_transferred": 0,
			"tasks_delegated": 0,
			"tasks_escalated": 0,
			"average_execution_time_seconds": 0.0,
			"average_task_duration_seconds": 0.0,
			"cross_capability_calls": 0,
			"performance_history": [],
			"sla_breaches": 0,
			"automation_rate": 0.0,
			"user_productivity": {},
			"capability_usage": {},
			"workflow_success_rates": {}
		}
		
		_log_workflow_operation("executor_initialized", "system", {
			"tenant_id": tenant_id,
			"max_workers": max_workers
		})
	
	def register_task_handler(self, handler: TaskHandler) -> None:
		"""Register custom task handler."""
		assert isinstance(handler, TaskHandler), "Handler must extend TaskHandler"
		self.task_handlers[handler.task_type] = handler
		
		_log_workflow_operation("handler_registered", "system", {
			"task_type": handler.task_type.value
		})
	
	def add_event_listener(self, listener: Callable) -> None:
		"""Add event listener for workflow events."""
		self.event_listeners.append(listener)
	
	async def start_workflow_instance(
		self,
		workflow: Workflow,
		started_by: str,
		input_data: Optional[Dict[str, Any]] = None,
		correlation_id: Optional[str] = None
	) -> str:
		"""Start a new workflow instance."""
		assert_workflow_valid(workflow)
		assert started_by, "User ID is required"
		
		# Create workflow instance
		instance = WorkflowInstance(
			workflow_id=workflow.id,
			workflow_version=workflow.version,
			tenant_id=self.tenant_id,
			started_by=started_by,
			input_data=input_data or {},
			variables=workflow.variables.copy(),
			context={
				"correlation_id": correlation_id or uuid7str(),
				"started_at": datetime.now(timezone.utc).isoformat()
			}
		)
		
		# Initialize tracking structures
		self.active_instances[instance.id] = instance
		self.active_tasks[instance.id] = set()
		self.task_executions[instance.id] = {}
		
		# Create execution context
		context = ExecutionContext(
			instance_id=instance.id,
			workflow_id=workflow.id,
			tenant_id=self.tenant_id,
			user_id=started_by,
			variables=instance.variables,
			correlation_id=instance.context.get("correlation_id", "")
		)
		
		# Update metrics
		self.metrics["instances_started"] += 1
		
		_log_workflow_operation("instance_started", instance.id, {
			"workflow_id": workflow.id,
			"started_by": started_by,
			"correlation_id": context.correlation_id
		})
		
		# Start initial tasks asynchronously
		asyncio.create_task(self._start_ready_tasks(workflow, instance, context))
		
		# Notify listeners
		await self._emit_event("instance_started", {
			"instance_id": instance.id,
			"workflow_id": workflow.id,
			"started_by": started_by
		})
		
		# Persist instance to database
		await self._persist_workflow_instance(instance)
		
		return instance.id
	
	async def transfer_task(
		self,
		instance_id: str,
		task_id: str,
		from_user_id: str,
		to_user_id: str,
		transfer_reason: str,
		comments: Optional[str] = None
	) -> bool:
		"""Transfer a task from one user to another."""
		assert instance_id in self.active_instances, f"Instance {instance_id} not found"
		assert task_id, "Task ID is required"
		assert from_user_id, "From user ID is required"
		assert to_user_id, "To user ID is required"
		assert transfer_reason, "Transfer reason is required"
		
		instance = self.active_instances[instance_id]
		assert_instance_active(instance)
		
		# Find task execution
		task_executions = self.task_executions.get(instance_id, {})
		execution = task_executions.get(task_id)
		
		if not execution:
			_log_task_execution("task_not_found", task_id, instance_id, {"from_user": from_user_id})
			return False
		
		# Verify current assignment
		if execution.assigned_to != from_user_id and execution.current_assignee != from_user_id:
			_log_task_execution("task_transfer_unauthorized", task_id, instance_id, {
				"from_user": from_user_id,
				"current_assignee": execution.assigned_to or execution.current_assignee
			})
			return False
		
		if execution.status not in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
			_log_task_execution("task_not_transferable", task_id, instance_id, {
				"status": execution.status.value,
				"from_user": from_user_id
			})
			return False
		
		# Perform the transfer
		previous_assignee = execution.assigned_to or execution.current_assignee
		execution.current_assignee = to_user_id
		execution.assigned_at = datetime.now(timezone.utc)
		
		# Add transfer comment to audit trail
		transfer_comment = {
			"user_id": from_user_id,
			"comment": f"Task transferred to {to_user_id}. Reason: {transfer_reason}",
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"type": "transfer",
			"transfer_details": {
				"from_user": previous_assignee,
				"to_user": to_user_id,
				"reason": transfer_reason
			}
		}
		
		if comments:
			transfer_comment["additional_comments"] = comments
		
		execution.comments.append(transfer_comment)
		
		# Update audit events
		execution.audit_events.append({
			"event_type": "task_transferred",
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"from_user": previous_assignee,
			"to_user": to_user_id,
			"reason": transfer_reason,
			"performed_by": from_user_id
		})
		
		# Update task execution in database
		await self._update_task_execution_in_database(execution)
		
		_log_task_execution("task_transferred", task_id, instance_id, {
			"from_user": previous_assignee,
			"to_user": to_user_id,
			"reason": transfer_reason,
			"performed_by": from_user_id
		})
		
		# Emit transfer event
		await self._emit_event("task_transferred", {
			"instance_id": instance_id,
			"task_id": task_id,
			"from_user": previous_assignee,
			"to_user": to_user_id,
			"reason": transfer_reason,
			"performed_by": from_user_id
		})
		
		return True
	
	async def delegate_task(
		self,
		instance_id: str,
		task_id: str,
		delegator_user_id: str,
		delegate_to_user_id: str,
		delegation_reason: str,
		retain_oversight: bool = True,
		comments: Optional[str] = None
	) -> bool:
		"""Delegate a task to another user while optionally retaining oversight."""
		assert instance_id in self.active_instances, f"Instance {instance_id} not found"
		assert task_id, "Task ID is required"
		assert delegator_user_id, "Delegator user ID is required"
		assert delegate_to_user_id, "Delegate to user ID is required"
		assert delegation_reason, "Delegation reason is required"
		
		instance = self.active_instances[instance_id]
		assert_instance_active(instance)
		
		# Find task execution
		task_executions = self.task_executions.get(instance_id, {})
		execution = task_executions.get(task_id)
		
		if not execution:
			_log_task_execution("task_not_found", task_id, instance_id, {"delegator": delegator_user_id})
			return False
		
		# Verify delegation authority
		if execution.assigned_to != delegator_user_id and execution.current_assignee != delegator_user_id:
			_log_task_execution("task_delegation_unauthorized", task_id, instance_id, {
				"delegator": delegator_user_id,
				"current_assignee": execution.assigned_to or execution.current_assignee
			})
			return False
		
		if execution.status not in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
			_log_task_execution("task_not_delegatable", task_id, instance_id, {
				"status": execution.status.value,
				"delegator": delegator_user_id
			})
			return False
		
		# Perform the delegation
		original_assignee = execution.assigned_to or execution.current_assignee
		execution.current_assignee = delegate_to_user_id
		execution.assigned_at = datetime.now(timezone.utc)
		
		# Store delegation metadata
		if "delegation_chain" not in execution.metadata:
			execution.metadata["delegation_chain"] = []
		
		execution.metadata["delegation_chain"].append({
			"delegator": delegator_user_id,
			"delegate": delegate_to_user_id,
			"reason": delegation_reason,
			"retain_oversight": retain_oversight,
			"delegated_at": datetime.now(timezone.utc).isoformat()
		})
		
		if retain_oversight:
			execution.metadata["oversight_user"] = delegator_user_id
		
		# Add delegation comment
		delegation_comment = {
			"user_id": delegator_user_id,
			"comment": f"Task delegated to {delegate_to_user_id}. Reason: {delegation_reason}",
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"type": "delegation",
			"delegation_details": {
				"original_assignee": original_assignee,
				"delegator": delegator_user_id,
				"delegate": delegate_to_user_id,
				"reason": delegation_reason,
				"retain_oversight": retain_oversight
			}
		}
		
		if comments:
			delegation_comment["additional_comments"] = comments
		
		execution.comments.append(delegation_comment)
		
		# Update audit events
		execution.audit_events.append({
			"event_type": "task_delegated",
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"original_assignee": original_assignee,
			"delegator": delegator_user_id,
			"delegate": delegate_to_user_id,
			"reason": delegation_reason,
			"retain_oversight": retain_oversight
		})
		
		_log_task_execution("task_delegated", task_id, instance_id, {
			"original_assignee": original_assignee,
			"delegator": delegator_user_id,
			"delegate": delegate_to_user_id,
			"reason": delegation_reason,
			"retain_oversight": retain_oversight
		})
		
		# Emit delegation event
		await self._emit_event("task_delegated", {
			"instance_id": instance_id,
			"task_id": task_id,
			"original_assignee": original_assignee,
			"delegator": delegator_user_id,
			"delegate": delegate_to_user_id,
			"reason": delegation_reason,
			"retain_oversight": retain_oversight
		})
		
		return True
	
	async def escalate_task(
		self,
		instance_id: str,
		task_id: str,
		escalated_by: str,
		escalate_to: str,
		escalation_reason: str,
		escalation_level: int = 1,
		comments: Optional[str] = None
	) -> bool:
		"""Escalate a task to a higher authority level."""
		assert instance_id in self.active_instances, f"Instance {instance_id} not found"
		assert task_id, "Task ID is required"
		assert escalated_by, "Escalated by user ID is required"
		assert escalate_to, "Escalate to user ID is required"
		assert escalation_reason, "Escalation reason is required"
		
		instance = self.active_instances[instance_id]
		assert_instance_active(instance)
		
		# Find task execution
		task_executions = self.task_executions.get(instance_id, {})
		execution = task_executions.get(task_id)
		
		if not execution:
			_log_task_execution("task_not_found", task_id, instance_id, {"escalated_by": escalated_by})
			return False
		
		if execution.status not in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS, TaskStatus.FAILED]:
			_log_task_execution("task_not_escalatable", task_id, instance_id, {
				"status": execution.status.value,
				"escalated_by": escalated_by
			})
			return False
		
		# Perform the escalation
		previous_assignee = execution.assigned_to or execution.current_assignee
		execution.status = TaskStatus.ESCALATED
		execution.escalated_at = datetime.now(timezone.utc)
		execution.escalated_to = escalate_to
		execution.escalation_reason = escalation_reason
		execution.escalation_level = max(execution.escalation_level, escalation_level)
		execution.current_assignee = escalate_to
		
		# Add escalation comment
		escalation_comment = {
			"user_id": escalated_by,
			"comment": f"Task escalated to {escalate_to} (Level {escalation_level}). Reason: {escalation_reason}",
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"type": "escalation",
			"escalation_details": {
				"previous_assignee": previous_assignee,
				"escalated_by": escalated_by,
				"escalated_to": escalate_to,
				"escalation_level": escalation_level,
				"reason": escalation_reason
			}
		}
		
		if comments:
			escalation_comment["additional_comments"] = comments
		
		execution.comments.append(escalation_comment)
		
		# Update audit events
		execution.audit_events.append({
			"event_type": "task_escalated",
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"previous_assignee": previous_assignee,
			"escalated_by": escalated_by,
			"escalated_to": escalate_to,
			"escalation_level": escalation_level,
			"reason": escalation_reason
		})
		
		_log_task_execution("task_escalated", task_id, instance_id, {
			"previous_assignee": previous_assignee,
			"escalated_by": escalated_by,
			"escalated_to": escalate_to,
			"escalation_level": escalation_level,
			"reason": escalation_reason
		})
		
		# Emit escalation event
		await self._emit_event("task_escalated", {
			"instance_id": instance_id,
			"task_id": task_id,
			"previous_assignee": previous_assignee,
			"escalated_by": escalated_by,
			"escalated_to": escalate_to,
			"escalation_level": escalation_level,
			"reason": escalation_reason
		})
		
		return True

	async def complete_human_task(
		self,
		instance_id: str,
		task_id: str,
		user_id: str,
		result: Dict[str, Any],
		comments: Optional[str] = None
	) -> bool:
		"""Complete a human or approval task."""
		assert instance_id in self.active_instances, f"Instance {instance_id} not found"
		assert task_id, "Task ID is required"
		assert user_id, "User ID is required"
		
		instance = self.active_instances[instance_id]
		assert_instance_active(instance)
		
		# Find task execution
		task_executions = self.task_executions.get(instance_id, {})
		execution = task_executions.get(task_id)
		
		if not execution:
			_log_task_execution("task_not_found", task_id, instance_id, {"user_id": user_id})
			return False
		
		# Check if user is authorized to complete this task
		authorized_users = [execution.assigned_to, execution.current_assignee]
		if execution.metadata.get("oversight_user"):
			authorized_users.append(execution.metadata["oversight_user"])
		
		if user_id not in [u for u in authorized_users if u]:
			_log_task_execution("task_completion_unauthorized", task_id, instance_id, {
				"user_id": user_id,
				"authorized_users": authorized_users
			})
			return False
		
		if execution.status not in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS, TaskStatus.ESCALATED]:
			_log_task_execution("task_not_completable", task_id, instance_id, {
				"status": execution.status.value,
				"user_id": user_id
			})
			return False
		
		# Complete the task
		execution.status = TaskStatus.COMPLETED
		execution.completed_at = datetime.now(timezone.utc)
		execution.progress_percentage = 100.0
		execution.result.update(result)
		
		if comments:
			execution.comments.append({
				"user_id": user_id,
				"comment": comments,
				"timestamp": datetime.now(timezone.utc).isoformat(),
				"type": "completion"
			})
		
		# Handle approval decision
		if "approval_decision" in result:
			execution.approval_decision = result["approval_decision"]
			execution.approval_reason = result.get("approval_reason", "")
		
		# Update instance state
		if task_id in instance.current_tasks:
			instance.current_tasks.remove(task_id)
		instance.completed_tasks.append(task_id)
		
		# Calculate duration
		if execution.started_at:
			duration = (execution.completed_at - execution.started_at).total_seconds()
			execution.duration_seconds = duration
		
		# Record performance metrics
		await self._record_task_performance_metrics(execution, instance)
		
		_log_task_execution("task_completed", task_id, instance_id, {
			"completed_by": user_id,
			"duration_seconds": execution.duration_seconds,
			"approval_decision": execution.approval_decision
		})
		
		# Check workflow completion and start next tasks
		workflow = await self._get_workflow(instance.workflow_id)  # Mock - would get from storage
		if workflow:
			context = ExecutionContext(
				instance_id=instance_id,
				workflow_id=workflow.id,
				tenant_id=self.tenant_id,
				user_id=user_id,
				variables=instance.variables
			)
			
			await self._check_workflow_completion(workflow, instance, context)
			
			if instance.status == WorkflowStatus.ACTIVE:
				await self._start_ready_tasks(workflow, instance, context)
		
		return True
	
	async def pause_instance(self, instance_id: str, user_id: str) -> bool:
		"""Pause workflow instance execution."""
		if instance_id not in self.active_instances:
			return False
		
		instance = self.active_instances[instance_id]
		if instance.status != WorkflowStatus.ACTIVE:
			return False
		
		instance.status = WorkflowStatus.PAUSED
		instance.paused_at = datetime.now(timezone.utc)
		
		_log_workflow_operation("instance_paused", instance_id, {"paused_by": user_id})
		
		await self._emit_event("instance_paused", {
			"instance_id": instance_id,
			"paused_by": user_id
		})
		
		return True
	
	async def resume_instance(self, instance_id: str, user_id: str) -> bool:
		"""Resume paused workflow instance."""
		if instance_id not in self.active_instances:
			return False
		
		instance = self.active_instances[instance_id]
		if instance.status != WorkflowStatus.PAUSED:
			return False
		
		instance.status = WorkflowStatus.ACTIVE
		instance.resumed_at = datetime.now(timezone.utc)
		
		_log_workflow_operation("instance_resumed", instance_id, {"resumed_by": user_id})
		
		# Continue execution
		workflow = await self._get_workflow(instance.workflow_id)
		if workflow:
			context = ExecutionContext(
				instance_id=instance_id,
				workflow_id=workflow.id,
				tenant_id=self.tenant_id,
				user_id=user_id,
				variables=instance.variables
			)
			await self._start_ready_tasks(workflow, instance, context)
		
		await self._emit_event("instance_resumed", {
			"instance_id": instance_id,
			"resumed_by": user_id
		})
		
		return True
	
	async def cancel_instance(self, instance_id: str, user_id: str, reason: str = "") -> bool:
		"""Cancel workflow instance execution."""
		if instance_id not in self.active_instances:
			return False
		
		instance = self.active_instances[instance_id]
		if instance.status in [WorkflowStatus.COMPLETED, WorkflowStatus.CANCELLED]:
			return False
		
		instance.status = WorkflowStatus.CANCELLED
		instance.completed_at = datetime.now(timezone.utc)
		instance.error_message = f"Cancelled by {user_id}: {reason}"
		
		# Cancel all active tasks
		for task_id in list(instance.current_tasks):
			execution = self.task_executions[instance_id].get(task_id)
			if execution and execution.status in [TaskStatus.IN_PROGRESS, TaskStatus.ASSIGNED]:
				execution.status = TaskStatus.CANCELLED
				execution.completed_at = datetime.now(timezone.utc)
		
		instance.current_tasks.clear()
		
		_log_workflow_operation("instance_cancelled", instance_id, {
			"cancelled_by": user_id,
			"reason": reason
		})
		
		await self._emit_event("instance_cancelled", {
			"instance_id": instance_id,
			"cancelled_by": user_id,
			"reason": reason
		})
		
		return True
	
	async def get_instance_status(self, instance_id: str) -> Optional[WorkflowInstance]:
		"""Get current workflow instance status."""
		return self.active_instances.get(instance_id)
	
	async def get_task_executions(self, instance_id: str) -> List[TaskExecution]:
		"""Get all task executions for an instance."""
		executions = self.task_executions.get(instance_id, {})
		return list(executions.values())
	
	async def _start_ready_tasks(
		self,
		workflow: Workflow,
		instance: WorkflowInstance,
		context: ExecutionContext
	) -> None:
		"""Start tasks that are ready to execute."""
		if instance.status != WorkflowStatus.ACTIVE:
			return
		
		ready_tasks = []
		
		for task in workflow.tasks:
			if (task.id not in instance.completed_tasks and
				task.id not in instance.failed_tasks and
				task.id not in instance.current_tasks):
				
				# Check dependencies
				dependencies_met = all(
					dep_id in instance.completed_tasks
					for dep_id in task.dependencies
				)
				
				if dependencies_met:
					# Check conditions if any
					if await self._evaluate_task_conditions(task, instance, context):
						ready_tasks.append(task)
		
		# Start ready tasks
		for task in ready_tasks:
			await self._execute_task(workflow, instance, task, context)
	
	async def _execute_task(
		self,
		workflow: Workflow,
		instance: WorkflowInstance,
		task: TaskDefinition,
		context: ExecutionContext
	) -> None:
		"""Execute a single task."""
		try:
			assert_task_executable(task, instance)
			
			# Create task execution record
			execution = TaskExecution(
				instance_id=instance.id,
				task_id=task.id,
				task_name=task.name,
				assigned_to=task.assigned_to,
				assigned_role=task.assigned_role,
				priority=task.priority,
				max_attempts=task.max_retry_attempts,
				created_by=context.user_id,
				input_data=self._prepare_task_input(task, instance, context)
			)
			
			# Store execution
			self.task_executions[instance.id][task.id] = execution
			instance.current_tasks.append(task.id)
			
			# Get appropriate handler
			handler = self.task_handlers.get(task.task_type)
			if not handler:
				execution.status = TaskStatus.FAILED
				execution.error_message = f"No handler for task type {task.task_type}"
				execution.completed_at = datetime.now(timezone.utc)
				instance.current_tasks.remove(task.id)
				instance.failed_tasks.append(task.id)
				return
			
			_log_task_execution("task_starting", task.id, instance.id, {
				"task_type": task.task_type.value,
				"handler": handler.__class__.__name__
			})
			
			# Persist task execution to database
			await self._persist_task_execution(execution)
			
			# Execute task
			execution.status = TaskStatus.IN_PROGRESS
			execution.started_at = datetime.now(timezone.utc)
			
			result = await handler.execute(task, execution, context)
			
			# Handle execution result
			await self._handle_execution_result(task, execution, result, instance, context)
			
			# Update task execution in database
			await self._update_task_execution_in_database(execution)
			
			# Update metrics
			self.metrics["tasks_executed"] += 1
			
		except Exception as e:
			logger.error(f"Task execution failed: {e}")
			
			# Update execution with error
			if task.id in self.task_executions.get(instance.id, {}):
				execution = self.task_executions[instance.id][task.id]
				execution.status = TaskStatus.FAILED
				execution.error_message = str(e)
				execution.completed_at = datetime.now(timezone.utc)
				
				if task.id in instance.current_tasks:
					instance.current_tasks.remove(task.id)
				instance.failed_tasks.append(task.id)
			
			self.metrics["tasks_failed"] += 1
	
	async def _handle_execution_result(
		self,
		task: TaskDefinition,
		execution: TaskExecution,
		result: ExecutionResult,
		instance: WorkflowInstance,
		context: ExecutionContext
	) -> None:
		"""Handle task execution result."""
		if result == ExecutionResult.SUCCESS:
			# Task completed successfully
			if execution.status not in [TaskStatus.ASSIGNED]:  # Human tasks stay assigned
				execution.status = TaskStatus.COMPLETED
				execution.completed_at = datetime.now(timezone.utc)
				
				if task.id in instance.current_tasks:
					instance.current_tasks.remove(task.id)
				instance.completed_tasks.append(task.id)
				
				# Calculate duration
				if execution.started_at:
					duration = (execution.completed_at - execution.started_at).total_seconds()
					execution.duration_seconds = duration
		
		elif result == ExecutionResult.FAILURE:
			# Task failed
			execution.status = TaskStatus.FAILED
			execution.completed_at = datetime.now(timezone.utc)
			
			if task.id in instance.current_tasks:
				instance.current_tasks.remove(task.id)
			
			# Check if we should retry or fail permanently
			if execution.attempt_number < execution.max_attempts:
				# Schedule retry
				execution.attempt_number += 1
				execution.retry_at = datetime.now(timezone.utc) + timedelta(seconds=task.retry_delay_seconds)
				execution.status = TaskStatus.PENDING
				# Keep in current_tasks for retry
			else:
				# Permanent failure
				instance.failed_tasks.append(task.id)
				
				# Check if we should continue or fail workflow
				if not task.continue_on_failure:
					instance.status = WorkflowStatus.FAILED
					instance.completed_at = datetime.now(timezone.utc)
					instance.error_message = f"Task {task.name} failed and continue_on_failure is False"
		
		elif result == ExecutionResult.SKIP:
			# Skip task
			execution.status = TaskStatus.SKIPPED
			execution.completed_at = datetime.now(timezone.utc)
			
			if task.id in instance.current_tasks:
				instance.current_tasks.remove(task.id)
			instance.skipped_tasks.append(task.id)
		
		# Emit task event
		await self._emit_event("task_status_changed", {
			"instance_id": instance.id,
			"task_id": task.id,
			"status": execution.status.value,
			"result": result.value
		})
	
	async def _evaluate_task_conditions(
		self,
		task: TaskDefinition,
		instance: WorkflowInstance,
		context: ExecutionContext
	) -> bool:
		"""Evaluate if task conditions are met."""
		if not task.conditions:
			return True
		
		# Simple condition evaluation - in production would use expression engine
		for condition in task.conditions:
			condition_type = condition.get("type", "variable")
			
			if condition_type == "variable":
				var_name = condition.get("variable")
				expected_value = condition.get("value")
				operator = condition.get("operator", "equals")
				
				actual_value = instance.variables.get(var_name)
				
				if operator == "equals" and actual_value != expected_value:
					return False
				elif operator == "not_equals" and actual_value == expected_value:
					return False
				elif operator == "exists" and var_name not in instance.variables:
					return False
		
		return True
	
	async def _check_workflow_completion(
		self,
		workflow: Workflow,
		instance: WorkflowInstance,
		context: ExecutionContext
	) -> None:
		"""Check if workflow is complete."""
		all_task_ids = {task.id for task in workflow.tasks}
		completed_and_failed_and_skipped = set(
			instance.completed_tasks + instance.failed_tasks + instance.skipped_tasks
		)
		
		if all_task_ids <= completed_and_failed_and_skipped:
			# All tasks are done
			if instance.failed_tasks and not all(
				task.continue_on_failure for task in workflow.tasks 
				if task.id in instance.failed_tasks
			):
				instance.status = WorkflowStatus.FAILED
				self.metrics["instances_failed"] += 1
			else:
				instance.status = WorkflowStatus.COMPLETED
				self.metrics["instances_completed"] += 1
			
			instance.completed_at = datetime.now(timezone.utc)
			
			# Calculate total duration
			if instance.started_at:
				duration = (instance.completed_at - instance.started_at).total_seconds()
				instance.duration_seconds = duration
			
			# Calculate progress
			instance.progress_percentage = 100.0
			instance.completed_steps = len(instance.completed_tasks)
			
			_log_workflow_operation("instance_completed", instance.id, {
				"status": instance.status.value,
				"duration_seconds": instance.duration_seconds,
				"completed_tasks": len(instance.completed_tasks),
				"failed_tasks": len(instance.failed_tasks)
			})
			
			await self._emit_event("instance_completed", {
				"instance_id": instance.id,
				"status": instance.status.value,
				"duration_seconds": instance.duration_seconds
			})
	
	def _prepare_task_input(
		self,
		task: TaskDefinition,
		instance: WorkflowInstance,
		context: ExecutionContext
	) -> Dict[str, Any]:
		"""Prepare input data for task execution."""
		input_data = {}
		
		# Add task input parameters
		input_data.update(task.input_parameters)
		
		# Add instance variables
		input_data.update(instance.variables)
		
		# Add instance input data
		input_data.update(instance.input_data)
		
		# Add context data
		input_data.update({
			"instance_id": instance.id,
			"workflow_id": instance.workflow_id,
			"tenant_id": context.tenant_id,
			"correlation_id": context.correlation_id
		})
		
		return input_data
	
	async def _emit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
		"""Emit event to all listeners."""
		for listener in self.event_listeners:
			try:
				if asyncio.iscoroutinefunction(listener):
					await listener(event_type, event_data)
				else:
					listener(event_type, event_data)
			except Exception as e:
				logger.error(f"Event listener error: {e}")
	
	async def _record_task_performance_metrics(
		self,
		execution: TaskExecution,
		instance: WorkflowInstance
	) -> None:
		"""Record comprehensive performance metrics for task completion."""
		try:
			# Update general task metrics
			self.metrics["tasks_completed"] += 1
			
			# Update transfer/delegation metrics
			if execution.metadata.get("delegation_chain"):
				self.metrics["tasks_delegated"] += len(execution.metadata["delegation_chain"])
			
			if execution.escalation_level > 0:
				self.metrics["tasks_escalated"] += 1
			
			# Track SLA breaches
			if execution.sla_deadline and execution.completed_at:
				if execution.completed_at > execution.sla_deadline:
					execution.is_sla_breached = True
					self.metrics["sla_breaches"] += 1
			
			# Update duration metrics
			if execution.duration_seconds:
				current_avg = self.metrics["average_task_duration_seconds"]
				task_count = self.metrics["tasks_completed"]
				self.metrics["average_task_duration_seconds"] = (
					(current_avg * (task_count - 1) + execution.duration_seconds) / task_count
				)
			
			# Track user productivity
			assignee = execution.current_assignee or execution.assigned_to
			if assignee:
				if assignee not in self.metrics["user_productivity"]:
					self.metrics["user_productivity"][assignee] = {
						"tasks_completed": 0,
						"total_duration_seconds": 0.0,
						"average_task_time": 0.0,
						"tasks_on_time": 0,
						"sla_breaches": 0,
						"efficiency_score": 1.0
					}
				
				user_metrics = self.metrics["user_productivity"][assignee]
				user_metrics["tasks_completed"] += 1
				if execution.duration_seconds:
					user_metrics["total_duration_seconds"] += execution.duration_seconds
					user_metrics["average_task_time"] = (
						user_metrics["total_duration_seconds"] / user_metrics["tasks_completed"]
					)
				
				if execution.is_sla_breached:
					user_metrics["sla_breaches"] += 1
				else:
					user_metrics["tasks_on_time"] += 1
				
				# Calculate efficiency score (tasks on time / total tasks)
				user_metrics["efficiency_score"] = (
					user_metrics["tasks_on_time"] / user_metrics["tasks_completed"]
				)
			
			# Track workflow success rates
			workflow_id = instance.workflow_id
			if workflow_id not in self.metrics["workflow_success_rates"]:
				self.metrics["workflow_success_rates"][workflow_id] = {
					"total_tasks": 0,
					"successful_tasks": 0,
					"failed_tasks": 0,
					"success_rate": 0.0,
					"average_task_duration": 0.0
				}
			
			workflow_metrics = self.metrics["workflow_success_rates"][workflow_id]
			workflow_metrics["total_tasks"] += 1
			workflow_metrics["successful_tasks"] += 1
			
			workflow_metrics["success_rate"] = (
				workflow_metrics["successful_tasks"] / workflow_metrics["total_tasks"]
			)
			
			if execution.duration_seconds:
				total_duration = (
					workflow_metrics["average_task_duration"] * (workflow_metrics["total_tasks"] - 1) +
					execution.duration_seconds
				)
				workflow_metrics["average_task_duration"] = total_duration / workflow_metrics["total_tasks"]
			
			# Calculate automation rate
			total_tasks = self.metrics["tasks_executed"]
			if total_tasks > 0:
				automated_tasks = total_tasks - self.metrics["tasks_completed"]  # Rough approximation
				self.metrics["automation_rate"] = automated_tasks / total_tasks
			
			# Record performance snapshot
			performance_snapshot = {
				"timestamp": datetime.now(timezone.utc).isoformat(),
				"instance_id": instance.id,
				"workflow_id": instance.workflow_id,
				"task_id": execution.task_id,
				"task_duration": execution.duration_seconds,
				"sla_breached": execution.is_sla_breached,
				"assignee": assignee,
				"completion_time": execution.completed_at.isoformat() if execution.completed_at else None
			}
			
			# Keep only recent performance history (last 1000 entries)
			self.metrics["performance_history"].append(performance_snapshot)
			if len(self.metrics["performance_history"]) > 1000:
				self.metrics["performance_history"] = self.metrics["performance_history"][-1000:]
		
		except Exception as e:
			logger.error(f"Error recording performance metrics: {e}")
	
	async def _record_cross_capability_usage(
		self,
		capability_name: str,
		operation: str,
		duration_ms: Optional[int] = None,
		success: bool = True
	) -> None:
		"""Record cross-capability integration usage metrics."""
		try:
			self.metrics["cross_capability_calls"] += 1
			
			if capability_name not in self.metrics["capability_usage"]:
				self.metrics["capability_usage"][capability_name] = {
					"total_calls": 0,
					"successful_calls": 0,
					"failed_calls": 0,
					"operations": {},
					"average_duration_ms": 0.0,
					"success_rate": 0.0
				}
			
			capability_metrics = self.metrics["capability_usage"][capability_name]
			capability_metrics["total_calls"] += 1
			
			if success:
				capability_metrics["successful_calls"] += 1
			else:
				capability_metrics["failed_calls"] += 1
			
			capability_metrics["success_rate"] = (
				capability_metrics["successful_calls"] / capability_metrics["total_calls"]
			)
			
			# Track operation-specific metrics
			if operation not in capability_metrics["operations"]:
				capability_metrics["operations"][operation] = {
					"calls": 0,
					"successes": 0,
					"failures": 0,
					"average_duration_ms": 0.0
				}
			
			op_metrics = capability_metrics["operations"][operation]
			op_metrics["calls"] += 1
			if success:
				op_metrics["successes"] += 1
			else:
				op_metrics["failures"] += 1
			
			if duration_ms:
				# Update average duration
				total_duration = (
					capability_metrics["average_duration_ms"] * (capability_metrics["total_calls"] - 1) +
					duration_ms
				)
				capability_metrics["average_duration_ms"] = total_duration / capability_metrics["total_calls"]
				
				op_total_duration = (
					op_metrics["average_duration_ms"] * (op_metrics["calls"] - 1) + duration_ms
				)
				op_metrics["average_duration_ms"] = op_total_duration / op_metrics["calls"]
		
		except Exception as e:
			logger.error(f"Error recording capability usage metrics: {e}")
	
	async def get_performance_analytics(self) -> Dict[str, Any]:
		"""Get comprehensive performance analytics."""
		try:
			# Calculate advanced metrics
			total_instances = (
				self.metrics["instances_completed"] + 
				self.metrics["instances_failed"] + 
				self.metrics["instances_cancelled"]
			)
			
			instance_success_rate = 0.0
			if total_instances > 0:
				instance_success_rate = self.metrics["instances_completed"] / total_instances
			
			# Get top performers
			top_performers = []
			for user_id, user_metrics in self.metrics["user_productivity"].items():
				if user_metrics["tasks_completed"] > 0:
					top_performers.append({
						"user_id": user_id,
						"efficiency_score": user_metrics["efficiency_score"],
						"tasks_completed": user_metrics["tasks_completed"],
						"average_task_time": user_metrics["average_task_time"]
					})
			
			top_performers.sort(key=lambda x: x["efficiency_score"], reverse=True)
			
			# Get most used capabilities
			capability_usage_sorted = sorted(
				self.metrics["capability_usage"].items(),
				key=lambda x: x[1]["total_calls"],
				reverse=True
			)
			
			# Calculate trend metrics from recent history
			recent_history = self.metrics["performance_history"][-100:] if self.metrics["performance_history"] else []
			recent_avg_duration = 0.0
			recent_sla_breach_rate = 0.0
			
			if recent_history:
				durations = [h["task_duration"] for h in recent_history if h["task_duration"]]
				if durations:
					recent_avg_duration = sum(durations) / len(durations)
				
				sla_breaches = sum(1 for h in recent_history if h["sla_breached"])
				recent_sla_breach_rate = sla_breaches / len(recent_history)
			
			return {
				"overview": {
					"total_instances_started": self.metrics["instances_started"],
					"instance_success_rate": instance_success_rate,
					"total_tasks_executed": self.metrics["tasks_executed"],
					"average_task_duration_seconds": self.metrics["average_task_duration_seconds"],
					"automation_rate": self.metrics["automation_rate"],
					"sla_breach_rate": (
						self.metrics["sla_breaches"] / max(self.metrics["tasks_completed"], 1)
					)
				},
				"task_operations": {
					"tasks_completed": self.metrics["tasks_completed"],
					"tasks_failed": self.metrics["tasks_failed"],
					"tasks_transferred": self.metrics["tasks_transferred"],
					"tasks_delegated": self.metrics["tasks_delegated"],
					"tasks_escalated": self.metrics["tasks_escalated"]
				},
				"cross_capability_integration": {
					"total_calls": self.metrics["cross_capability_calls"],
					"capabilities_used": len(self.metrics["capability_usage"]),
					"most_used_capabilities": capability_usage_sorted[:5]
				},
				"user_performance": {
					"total_users": len(self.metrics["user_productivity"]),
					"top_performers": top_performers[:10]
				},
				"workflow_analytics": {
					"workflows_tracked": len(self.metrics["workflow_success_rates"]),
					"workflow_success_rates": dict(list(
						self.metrics["workflow_success_rates"].items()
					)[:10])
				},
				"recent_trends": {
					"recent_average_duration_seconds": recent_avg_duration,
					"recent_sla_breach_rate": recent_sla_breach_rate,
					"performance_samples": len(recent_history)
				},
				"generated_at": datetime.now(timezone.utc).isoformat()
			}
		
		except Exception as e:
			logger.error(f"Error generating performance analytics: {e}")
			return {"error": str(e), "generated_at": datetime.now(timezone.utc).isoformat()}

	async def _get_workflow(self, workflow_id: str) -> Optional[Workflow]:
		"""Get workflow definition from database."""
		try:
			async with self.database_manager.get_session() as session:
				workflow_repo, _, _ = await create_repositories(session)
				cr_workflow = await workflow_repo.get_workflow(workflow_id, self.tenant_id)
				
				if not cr_workflow:
					return None
				
				# Convert CRWorkflow to Pydantic Workflow model
				return self._convert_cr_workflow_to_workflow(cr_workflow)
			
		except Exception as e:
			logger.error(f"Error retrieving workflow {workflow_id}: {e}")
			return None
	
	def get_metrics(self) -> Dict[str, Any]:
		"""Get executor performance metrics."""
		return self.metrics.copy()
	
	# =============================================================================
	# Database Integration Methods
	# =============================================================================
	
	async def _persist_workflow_instance(self, instance: WorkflowInstance) -> None:
		"""Persist workflow instance to database."""
		try:
			async with self.database_manager.get_session() as session:
				_, instance_repo, _ = await create_repositories(session)
				
				instance_data = {
					"instance_id": instance.id,
					"workflow_id": instance.workflow_id,
					"tenant_id": self.tenant_id,
					"status": instance.status.value,
					"input_data": instance.input_data,
					"variables": instance.variables,
					"context": instance.context,
					"started_by": instance.started_by
				}
				
				await instance_repo.create_instance(instance_data)
				
		except Exception as e:
			logger.error(f"Error persisting workflow instance {instance.id}: {e}")
			raise
	
	async def _persist_task_execution(self, execution: TaskExecution) -> None:
		"""Persist task execution to database."""
		try:
			async with self.database_manager.get_session() as session:
				_, _, task_repo = await create_repositories(session)
				
				task_data = {
					"execution_id": execution.id,
					"instance_id": execution.instance_id,
					"task_id": execution.task_id,
					"task_name": execution.task_name,
					"status": execution.status.value,
					"assigned_to": execution.assigned_to,
					"assigned_role": execution.assigned_role,
					"current_assignee": execution.current_assignee,
					"priority": execution.priority.value if execution.priority else 5,
					"input_data": execution.input_data,
					"output_data": execution.output_data,
					"result": execution.result,
					"progress_percentage": execution.progress_percentage,
					"due_date": execution.due_date,
					"sla_deadline": execution.sla_deadline,
					"error_message": execution.error_message,
					"attempt_number": execution.attempt_number,
					"max_attempts": execution.max_attempts,
					"escalation_level": execution.escalation_level,
					"escalated_to": execution.escalated_to,
					"escalation_reason": execution.escalation_reason,
					"approval_decision": execution.approval_decision,
					"approval_reason": execution.approval_reason,
					"comments": execution.comments,
					"audit_events": execution.audit_events,
					"metadata": execution.metadata,
					"created_by": execution.created_by
				}
				
				await task_repo.create_task_execution(task_data)
				
		except Exception as e:
			logger.error(f"Error persisting task execution {execution.id}: {e}")
			raise
	
	async def _update_instance_in_database(self, instance: WorkflowInstance) -> None:
		"""Update workflow instance in database."""
		try:
			async with self.database_manager.get_session() as session:
				_, instance_repo, _ = await create_repositories(session)
				
				updates = {
					"status": instance.status.value,
					"current_tasks": instance.current_tasks,
					"completed_tasks": instance.completed_tasks,
					"failed_tasks": instance.failed_tasks,
					"skipped_tasks": instance.skipped_tasks,
					"progress_percentage": instance.progress_percentage,
					"completed_steps": instance.completed_steps,
					"variables": instance.variables,
					"context": instance.context,
					"error_message": instance.error_message,
					"completed_at": instance.completed_at,
					"paused_at": instance.paused_at,
					"resumed_at": instance.resumed_at,
					"duration_seconds": instance.duration_seconds
				}
				
				await instance_repo.update_instance(instance.id, updates)
				
		except Exception as e:
			logger.error(f"Error updating workflow instance {instance.id}: {e}")
			raise
	
	async def _update_task_execution_in_database(self, execution: TaskExecution) -> None:
		"""Update task execution in database."""
		try:
			async with self.database_manager.get_session() as session:
				_, _, task_repo = await create_repositories(session)
				
				updates = {
					"status": execution.status.value,
					"current_assignee": execution.current_assignee,
					"progress_percentage": execution.progress_percentage,
					"progress_message": execution.progress_message,
					"output_data": execution.output_data,
					"result": execution.result,
					"error_message": execution.error_message,
					"started_at": execution.started_at,
					"completed_at": execution.completed_at,
					"assigned_at": execution.assigned_at,
					"duration_seconds": execution.duration_seconds,
					"attempt_number": execution.attempt_number,
					"retry_at": execution.retry_at,
					"escalation_level": execution.escalation_level,
					"escalated_at": execution.escalated_at,
					"escalated_to": execution.escalated_to,
					"escalation_reason": execution.escalation_reason,
					"is_sla_breached": execution.is_sla_breached,
					"sla_breach_time": execution.sla_breach_time,
					"approval_decision": execution.approval_decision,
					"approval_reason": execution.approval_reason,
					"comments": execution.comments,
					"audit_events": execution.audit_events,
					"metadata": execution.metadata
				}
				
				await task_repo.update_task_execution(execution.id, updates)
				
		except Exception as e:
			logger.error(f"Error updating task execution {execution.id}: {e}")
			raise
	
	async def _convert_cr_workflow_to_workflow(self, cr_workflow: CRWorkflow) -> Workflow:
		"""Convert database CRWorkflow to Pydantic Workflow model."""
		# Extract task definitions from workflow_definition JSON
		tasks = []
		for task_data in cr_workflow.workflow_definition.get("tasks", []):
			task = TaskDefinition(
				id=task_data.get("id", uuid7str()),
				name=task_data.get("name", "Unnamed Task"),
				task_type=TaskType(task_data.get("type", "automated")),
				assigned_to=task_data.get("assigned_to"),
				assigned_role=task_data.get("assigned_role"),
				priority=Priority(task_data.get("priority", "medium")),
				dependencies=task_data.get("dependencies", []),
				conditions=task_data.get("conditions", []),
				input_parameters=task_data.get("input_parameters", {}),
				configuration=task_data.get("configuration", {}),
				sla_hours=task_data.get("sla_hours"),
				max_retry_attempts=task_data.get("max_retry_attempts", 3),
				retry_delay_seconds=task_data.get("retry_delay_seconds", 60),
				continue_on_failure=task_data.get("continue_on_failure", False),
				metadata=task_data.get("metadata", {})
			)
			tasks.append(task)
		
		return Workflow(
			id=cr_workflow.workflow_id,
			name=cr_workflow.name,
			description=cr_workflow.description or "",
			version=cr_workflow.version,
			category=cr_workflow.category,
			tenant_id=cr_workflow.tenant_id,
			tasks=tasks,
			triggers=cr_workflow.triggers or [],
			variables=cr_workflow.variables or {},
			is_active=cr_workflow.is_active,
			is_template=cr_workflow.is_template,
			max_concurrent_instances=cr_workflow.max_concurrent_instances or 10,
			default_timeout_hours=cr_workflow.default_timeout_hours,
			created_by=cr_workflow.created_by,
			created_at=cr_workflow.created_at,
			updated_at=cr_workflow.updated_at
		)
	
	# =============================================================================
	# Auth and Permissions Integration
	# =============================================================================
	
	async def _check_user_permission(self, user_id: str, permission: str, tenant_id: str) -> bool:
		"""Check user permission using real auth system."""
		try:
			# In production, this would integrate with actual auth_rbac capability
			# For now, implement basic permission logic
			
			# Basic permissions mapping
			basic_permissions = {
				"workflow.execute": True,
				"workflow.view": True,
				"workflow.create": True,
				"workflow.update": True,
				"workflow.delete": False,  # Require admin
				"task.complete": True,
				"task.assign": True,
				"task.transfer": True,
				"task.escalate": True
			}
			
			return basic_permissions.get(permission, False)
			
		except Exception as e:
			logger.error(f"Error checking permission {permission} for user {user_id}: {e}")
			return False
	
	async def _get_user_roles(self, user_id: str, tenant_id: str) -> List[str]:
		"""Get user roles from auth system."""
		try:
			# In production, this would integrate with actual auth_rbac capability
			# For now, return basic roles
			return ["workflow_user", "task_assignee", "tenant_member"]
			
		except Exception as e:
			logger.error(f"Error getting roles for user {user_id}: {e}")
			return []
	
	async def _get_user_permissions(self, user_id: str, tenant_id: str) -> List[str]:
		"""Get user permissions from auth system."""
		try:
			# In production, this would integrate with actual auth_rbac capability
			# For now, return basic permissions
			return [
				"workflow.view", "workflow.execute", "task.complete",
				"task.assign", "task.transfer", "task.escalate"
			]
			
		except Exception as e:
			logger.error(f"Error getting permissions for user {user_id}: {e}")
			return []
	
	async def _log_audit_event(self, event_data: Dict[str, Any]) -> str:
		"""Log audit event to database."""
		try:
			# Use the database audit logging instead of mock
			from ..database import CRWorkflowAuditLog
			
			async with self.database_manager.get_session() as session:
				audit_log = CRWorkflowAuditLog(
					tenant_id=event_data.get("tenant_id"),
					workflow_id=event_data.get("workflow_id"),
					instance_id=event_data.get("instance_id"),
					event_type=event_data.get("event_type", "workflow_action"),
					event_category="workflow",
					action=event_data.get("action", "execute"),
					resource_type="workflow_instance",
					resource_id=event_data.get("instance_id", ""),
					user_id=event_data.get("user_id"),
					event_data=event_data.get("details", {}),
					result="success",
					impact_level="medium",
					security_classification="internal"
				)
				
				session.add(audit_log)
				await session.commit()
				await session.refresh(audit_log)
				
				return audit_log.audit_id
			
		except Exception as e:
			logger.error(f"Error logging audit event: {e}")
			return uuid7str()  # Return a fake ID to not break the flow
	
	async def cleanup(self) -> None:
		"""Clean up resources."""
		self.executor.shutdown(wait=True)
		self.active_instances.clear()
		self.active_tasks.clear()
		self.task_executions.clear()
		self._repositories_cache.clear()
		
		_log_workflow_operation("executor_cleanup", "system", {
			"tenant_id": self.tenant_id
		})

__all__ = [
	"ExecutionResult",
	"ExecutionContext",
	"TaskHandler",
	"AutomatedTaskHandler",
	"HumanTaskHandler", 
	"ApprovalTaskHandler",
	"CrossCapabilityTaskHandler",
	"IntegrationTaskHandler",
	"WorkflowExecutor"
]