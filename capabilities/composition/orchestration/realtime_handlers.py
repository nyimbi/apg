#!/usr/bin/env python3
"""
APG Workflow Orchestration Real-time Event Handlers

Event handlers that integrate workflow execution with real-time API features.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from uuid_extensions import uuid7str

# Local imports
from .realtime_api import (
	RealTimeEvent, EventType, realtime_api_service
)
from .models import WorkflowStatus, TaskStatus
from .database import WorkflowDB, WorkflowInstanceDB, TaskExecutionDB
from .service import WorkflowOrchestrationService


logger = logging.getLogger(__name__)


class WorkflowEventHandler:
	"""Handles workflow-related real-time events."""
	
	def __init__(self, realtime_service, workflow_service: WorkflowOrchestrationService):
		self.realtime_service = realtime_service
		self.workflow_service = workflow_service
	
	async def on_workflow_created(self, workflow: WorkflowDB):
		"""Handle workflow creation event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.WORKFLOW_CREATED,
			source="workflow_service",
			resource_type="workflow",
			resource_id=workflow.id,
			tenant_id=workflow.tenant_id,
			user_id=workflow.created_by,
			data={
				'workflow_id': workflow.id,
				'workflow_name': workflow.name,
				'description': workflow.description,
				'version': workflow.version,
				'is_active': workflow.is_active,
				'created_at': workflow.created_at.isoformat() if workflow.created_at else None
			}
		)
		
		await self.realtime_service.publish_event(event)
		logger.info(f"Published workflow created event: {workflow.id}")
	
	async def on_workflow_updated(self, workflow: WorkflowDB, updated_fields: List[str]):
		"""Handle workflow update event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.WORKFLOW_UPDATED,
			source="workflow_service",
			resource_type="workflow",
			resource_id=workflow.id,
			tenant_id=workflow.tenant_id,
			user_id=workflow.created_by,
			data={
				'workflow_id': workflow.id,
				'workflow_name': workflow.name,
				'updated_fields': updated_fields,
				'version': workflow.version,
				'is_active': workflow.is_active,
				'updated_at': workflow.updated_at.isoformat() if workflow.updated_at else None
			}
		)
		
		await self.realtime_service.publish_event(event)
		logger.info(f"Published workflow updated event: {workflow.id}")
	
	async def on_workflow_deleted(self, workflow_id: str, tenant_id: str, user_id: str):
		"""Handle workflow deletion event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.WORKFLOW_DELETED,
			source="workflow_service",
			resource_type="workflow",
			resource_id=workflow_id,
			tenant_id=tenant_id,
			user_id=user_id,
			data={
				'workflow_id': workflow_id,
				'deleted_at': datetime.utcnow().isoformat()
			}
		)
		
		await self.realtime_service.publish_event(event)
		logger.info(f"Published workflow deleted event: {workflow_id}")
	
	async def on_workflow_executed(self, workflow: WorkflowDB, instance: WorkflowInstanceDB):
		"""Handle workflow execution event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.WORKFLOW_EXECUTED,
			source="workflow_service",
			resource_type="workflow",
			resource_id=workflow.id,
			tenant_id=workflow.tenant_id,
			user_id=instance.created_by,
			data={
				'workflow_id': workflow.id,
				'workflow_name': workflow.name,
				'instance_id': instance.id,
				'execution_context': instance.execution_context,
				'priority': instance.priority,
				'created_at': instance.created_at.isoformat() if instance.created_at else None
			}
		)
		
		await self.realtime_service.publish_event(event)
		logger.info(f"Published workflow executed event: {workflow.id} -> {instance.id}")


class WorkflowInstanceEventHandler:
	"""Handles workflow instance-related real-time events."""
	
	def __init__(self, realtime_service, workflow_service: WorkflowOrchestrationService):
		self.realtime_service = realtime_service
		self.workflow_service = workflow_service
	
	async def on_instance_created(self, instance: WorkflowInstanceDB):
		"""Handle workflow instance creation event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.INSTANCE_CREATED,
			source="workflow_service",
			resource_type="workflow_instance",
			resource_id=instance.id,
			tenant_id=instance.workflow.tenant_id,
			user_id=instance.created_by,
			data={
				'instance_id': instance.id,
				'workflow_id': instance.workflow_id,
				'workflow_name': instance.workflow.name if instance.workflow else None,
				'status': instance.status.value if instance.status else None,
				'priority': instance.priority,
				'execution_context': instance.execution_context,
				'created_at': instance.created_at.isoformat() if instance.created_at else None
			}
		)
		
		await self.realtime_service.publish_event(event)
		logger.info(f"Published instance created event: {instance.id}")
	
	async def on_instance_started(self, instance: WorkflowInstanceDB):
		"""Handle workflow instance start event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.INSTANCE_STARTED,
			source="workflow_service",
			resource_type="workflow_instance",
			resource_id=instance.id,
			tenant_id=instance.workflow.tenant_id,
			user_id=instance.created_by,
			data={
				'instance_id': instance.id,
				'workflow_id': instance.workflow_id,
				'workflow_name': instance.workflow.name if instance.workflow else None,
				'status': instance.status.value if instance.status else None,
				'started_at': instance.started_at.isoformat() if instance.started_at else None,
				'estimated_duration': self._estimate_duration(instance)
			}
		)
		
		await self.realtime_service.publish_event(event)
		logger.info(f"Published instance started event: {instance.id}")
	
	async def on_instance_completed(self, instance: WorkflowInstanceDB):
		"""Handle workflow instance completion event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.INSTANCE_COMPLETED,
			source="workflow_service",
			resource_type="workflow_instance",
			resource_id=instance.id,
			tenant_id=instance.workflow.tenant_id,
			user_id=instance.created_by,
			data={
				'instance_id': instance.id,
				'workflow_id': instance.workflow_id,
				'workflow_name': instance.workflow.name if instance.workflow else None,
				'status': instance.status.value if instance.status else None,
				'started_at': instance.started_at.isoformat() if instance.started_at else None,
				'completed_at': instance.completed_at.isoformat() if instance.completed_at else None,
				'duration_seconds': instance.duration_seconds,
				'result': instance.result
			}
		)
		
		await self.realtime_service.publish_event(event)
		logger.info(f"Published instance completed event: {instance.id}")
	
	async def on_instance_failed(self, instance: WorkflowInstanceDB):
		"""Handle workflow instance failure event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.INSTANCE_FAILED,
			source="workflow_service",
			resource_type="workflow_instance",
			resource_id=instance.id,
			tenant_id=instance.workflow.tenant_id,
			user_id=instance.created_by,
			data={
				'instance_id': instance.id,
				'workflow_id': instance.workflow_id,
				'workflow_name': instance.workflow.name if instance.workflow else None,
				'status': instance.status.value if instance.status else None,
				'started_at': instance.started_at.isoformat() if instance.started_at else None,
				'failed_at': instance.completed_at.isoformat() if instance.completed_at else None,
				'duration_seconds': instance.duration_seconds,
				'error_details': instance.error_details
			}
		)
		
		await self.realtime_service.publish_event(event)
		logger.info(f"Published instance failed event: {instance.id}")
	
	async def on_instance_cancelled(self, instance: WorkflowInstanceDB, cancelled_by: str):
		"""Handle workflow instance cancellation event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.INSTANCE_CANCELLED,
			source="workflow_service",
			resource_type="workflow_instance",
			resource_id=instance.id,
			tenant_id=instance.workflow.tenant_id,
			user_id=cancelled_by,
			data={
				'instance_id': instance.id,
				'workflow_id': instance.workflow_id,
				'workflow_name': instance.workflow.name if instance.workflow else None,
				'status': instance.status.value if instance.status else None,
				'cancelled_by': cancelled_by,
				'cancelled_at': datetime.utcnow().isoformat()
			}
		)
		
		await self.realtime_service.publish_event(event)
		logger.info(f"Published instance cancelled event: {instance.id}")
	
	async def on_instance_progress(self, instance: WorkflowInstanceDB, progress_data: Dict[str, Any]):
		"""Handle workflow instance progress update event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.INSTANCE_PROGRESS,
			source="workflow_service",
			resource_type="workflow_instance",
			resource_id=instance.id,
			tenant_id=instance.workflow.tenant_id,
			user_id=instance.created_by,
			data={
				'instance_id': instance.id,
				'workflow_id': instance.workflow_id,
				'progress': progress_data,
				'timestamp': datetime.utcnow().isoformat()
			},
			ttl_seconds=60  # Progress updates have shorter TTL
		)
		
		await self.realtime_service.publish_event(event)
	
	def _estimate_duration(self, instance: WorkflowInstanceDB) -> Optional[int]:
		"""Estimate workflow duration based on historical data."""
		try:
			# This would query historical executions and calculate average
			# For now, return a simple estimate
			return 300  # 5 minutes default estimate
		except Exception:
			return None


class TaskExecutionEventHandler:
	"""Handles task execution-related real-time events."""
	
	def __init__(self, realtime_service, workflow_service: WorkflowOrchestrationService):
		self.realtime_service = realtime_service
		self.workflow_service = workflow_service
	
	async def on_task_started(self, task_execution: TaskExecutionDB):
		"""Handle task execution start event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.TASK_STARTED,
			source="workflow_service",
			resource_type="task_execution",
			resource_id=task_execution.id,
			tenant_id=task_execution.workflow_instance.workflow.tenant_id,
			user_id=task_execution.workflow_instance.created_by,
			data={
				'task_execution_id': task_execution.id,
				'task_id': task_execution.task_id,
				'workflow_instance_id': task_execution.workflow_instance_id,
				'workflow_id': task_execution.workflow_instance.workflow_id,
				'status': task_execution.status.value if task_execution.status else None,
				'started_at': task_execution.started_at.isoformat() if task_execution.started_at else None,
				'input_data': task_execution.input_data
			}
		)
		
		await self.realtime_service.publish_event(event)
		logger.info(f"Published task started event: {task_execution.id}")
	
	async def on_task_completed(self, task_execution: TaskExecutionDB):
		"""Handle task execution completion event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.TASK_COMPLETED,
			source="workflow_service",
			resource_type="task_execution",
			resource_id=task_execution.id,
			tenant_id=task_execution.workflow_instance.workflow.tenant_id,
			user_id=task_execution.workflow_instance.created_by,
			data={
				'task_execution_id': task_execution.id,
				'task_id': task_execution.task_id,
				'workflow_instance_id': task_execution.workflow_instance_id,
				'workflow_id': task_execution.workflow_instance.workflow_id,
				'status': task_execution.status.value if task_execution.status else None,
				'started_at': task_execution.started_at.isoformat() if task_execution.started_at else None,
				'completed_at': task_execution.completed_at.isoformat() if task_execution.completed_at else None,
				'duration_seconds': task_execution.duration_seconds,
				'result': task_execution.result
			}
		)
		
		await self.realtime_service.publish_event(event)
		logger.info(f"Published task completed event: {task_execution.id}")
	
	async def on_task_failed(self, task_execution: TaskExecutionDB):
		"""Handle task execution failure event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.TASK_FAILED,
			source="workflow_service",
			resource_type="task_execution",
			resource_id=task_execution.id,
			tenant_id=task_execution.workflow_instance.workflow.tenant_id,
			user_id=task_execution.workflow_instance.created_by,
			data={
				'task_execution_id': task_execution.id,
				'task_id': task_execution.task_id,
				'workflow_instance_id': task_execution.workflow_instance_id,
				'workflow_id': task_execution.workflow_instance.workflow_id,
				'status': task_execution.status.value if task_execution.status else None,
				'started_at': task_execution.started_at.isoformat() if task_execution.started_at else None,
				'failed_at': task_execution.completed_at.isoformat() if task_execution.completed_at else None,
				'duration_seconds': task_execution.duration_seconds,
				'error_details': task_execution.error_details,
				'retry_count': task_execution.retry_count
			}
		)
		
		await self.realtime_service.publish_event(event)
		logger.info(f"Published task failed event: {task_execution.id}")
	
	async def on_task_retrying(self, task_execution: TaskExecutionDB):
		"""Handle task execution retry event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.TASK_RETRYING,
			source="workflow_service",
			resource_type="task_execution",
			resource_id=task_execution.id,
			tenant_id=task_execution.workflow_instance.workflow.tenant_id,
			user_id=task_execution.workflow_instance.created_by,
			data={
				'task_execution_id': task_execution.id,
				'task_id': task_execution.task_id,
				'workflow_instance_id': task_execution.workflow_instance_id,
				'workflow_id': task_execution.workflow_instance.workflow_id,
				'retry_count': task_execution.retry_count,
				'retry_at': datetime.utcnow().isoformat(),
				'previous_error': task_execution.error_details
			}
		)
		
		await self.realtime_service.publish_event(event)
		logger.info(f"Published task retrying event: {task_execution.id}")
	
	async def on_task_progress(self, task_execution: TaskExecutionDB, progress_data: Dict[str, Any]):
		"""Handle task execution progress update event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.TASK_PROGRESS,
			source="workflow_service",
			resource_type="task_execution",
			resource_id=task_execution.id,
			tenant_id=task_execution.workflow_instance.workflow.tenant_id,
			user_id=task_execution.workflow_instance.created_by,
			data={
				'task_execution_id': task_execution.id,
				'task_id': task_execution.task_id,
				'workflow_instance_id': task_execution.workflow_instance_id,
				'progress': progress_data,
				'timestamp': datetime.utcnow().isoformat()
			},
			ttl_seconds=60  # Progress updates have shorter TTL
		)
		
		await self.realtime_service.publish_event(event)


class CollaborationEventHandler:
	"""Handles collaboration-related real-time events."""
	
	def __init__(self, realtime_service):
		self.realtime_service = realtime_service
	
	async def on_component_added(self, workflow_id: str, tenant_id: str, user_id: str, 
								 component_data: Dict[str, Any]):
		"""Handle workflow designer component addition."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.COMPONENT_ADDED,
			source="workflow_designer",
			resource_type="workflow",
			resource_id=workflow_id,
			tenant_id=tenant_id,
			user_id=user_id,
			data={
				'workflow_id': workflow_id,
				'component': component_data,
				'timestamp': datetime.utcnow().isoformat()
			}
		)
		
		await self.realtime_service.publish_event(event)
	
	async def on_component_updated(self, workflow_id: str, tenant_id: str, user_id: str, 
								   component_data: Dict[str, Any]):
		"""Handle workflow designer component update."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.COMPONENT_UPDATED,
			source="workflow_designer",
			resource_type="workflow",
			resource_id=workflow_id,
			tenant_id=tenant_id,
			user_id=user_id,
			data={
				'workflow_id': workflow_id,
				'component': component_data,
				'timestamp': datetime.utcnow().isoformat()
			}
		)
		
		await self.realtime_service.publish_event(event)
	
	async def on_component_deleted(self, workflow_id: str, tenant_id: str, user_id: str, 
								   component_id: str):
		"""Handle workflow designer component deletion."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.COMPONENT_DELETED,
			source="workflow_designer",
			resource_type="workflow",
			resource_id=workflow_id,
			tenant_id=tenant_id,
			user_id=user_id,
			data={
				'workflow_id': workflow_id,
				'component_id': component_id,
				'timestamp': datetime.utcnow().isoformat()
			}
		)
		
		await self.realtime_service.publish_event(event)
	
	async def on_connection_created(self, workflow_id: str, tenant_id: str, user_id: str, 
									connection_data: Dict[str, Any]):
		"""Handle workflow designer connection creation."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.CONNECTION_CREATED,
			source="workflow_designer",
			resource_type="workflow",
			resource_id=workflow_id,
			tenant_id=tenant_id,
			user_id=user_id,
			data={
				'workflow_id': workflow_id,
				'connection': connection_data,
				'timestamp': datetime.utcnow().isoformat()
			}
		)
		
		await self.realtime_service.publish_event(event)
	
	async def on_connection_deleted(self, workflow_id: str, tenant_id: str, user_id: str, 
									connection_id: str):
		"""Handle workflow designer connection deletion."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.CONNECTION_DELETED,
			source="workflow_designer",
			resource_type="workflow",
			resource_id=workflow_id,
			tenant_id=tenant_id,
			user_id=user_id,
			data={
				'workflow_id': workflow_id,
				'connection_id': connection_id,
				'timestamp': datetime.utcnow().isoformat()
			}
		)
		
		await self.realtime_service.publish_event(event)
	
	async def on_canvas_updated(self, workflow_id: str, tenant_id: str, user_id: str, 
								canvas_data: Dict[str, Any]):
		"""Handle workflow designer canvas update."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.CANVAS_UPDATED,
			source="workflow_designer",
			resource_type="workflow",
			resource_id=workflow_id,
			tenant_id=tenant_id,
			user_id=user_id,
			data={
				'workflow_id': workflow_id,
				'canvas': canvas_data,
				'timestamp': datetime.utcnow().isoformat()
			},
			ttl_seconds=30  # Canvas updates have very short TTL
		)
		
		await self.realtime_service.publish_event(event)


class SystemEventHandler:
	"""Handles system-related real-time events."""
	
	def __init__(self, realtime_service):
		self.realtime_service = realtime_service
	
	async def on_system_status_change(self, status_data: Dict[str, Any]):
		"""Handle system status change event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.SYSTEM_STATUS,
			source="system_monitor",
			resource_type="system",
			resource_id="global",
			tenant_id="system",
			data=status_data
		)
		
		await self.realtime_service.publish_event(event)
	
	async def on_integration_status_change(self, integration_id: str, tenant_id: str, 
										   status_data: Dict[str, Any]):
		"""Handle integration status change event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.INTEGRATION_STATUS,
			source="integration_monitor",
			resource_type="integration",
			resource_id=integration_id,
			tenant_id=tenant_id,
			data=status_data
		)
		
		await self.realtime_service.publish_event(event)
	
	async def on_metrics_update(self, tenant_id: str, metrics_data: Dict[str, Any]):
		"""Handle metrics update event."""
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.METRICS_UPDATE,
			source="metrics_collector",
			resource_type="metrics",
			resource_id="dashboard",
			tenant_id=tenant_id,
			data=metrics_data,
			ttl_seconds=120  # Metrics updates have moderate TTL
		)
		
		await self.realtime_service.publish_event(event)


class RealTimeEventManager:
	"""Central manager for all real-time event handlers."""
	
	def __init__(self, workflow_service: WorkflowOrchestrationService):
		self.realtime_service = realtime_api_service
		self.workflow_service = workflow_service
		
		# Initialize handlers
		self.workflow_handler = WorkflowEventHandler(self.realtime_service, workflow_service)
		self.instance_handler = WorkflowInstanceEventHandler(self.realtime_service, workflow_service)
		self.task_handler = TaskExecutionEventHandler(self.realtime_service, workflow_service)
		self.collaboration_handler = CollaborationEventHandler(self.realtime_service)
		self.system_handler = SystemEventHandler(self.realtime_service)
		
		# Set up event listeners
		self._setup_event_listeners()
	
	def _setup_event_listeners(self):
		"""Set up event listeners for workflow service events."""
		try:
			# Set up event listeners for different workflow events
			event_mappings = {
				'workflow.created': self.handle_workflow_created,
				'workflow.started': self.handle_workflow_started,
				'workflow.completed': self.handle_workflow_completed,
				'workflow.failed': self.handle_workflow_failed,
				'workflow.paused': self.handle_workflow_paused,
				'workflow.resumed': self.handle_workflow_resumed,
				'workflow.cancelled': self.handle_workflow_cancelled,
				'task.started': self.handle_task_started,
				'task.completed': self.handle_task_completed,
				'task.failed': self.handle_task_failed,
				'task.retried': self.handle_task_retried,
				'metric.threshold_exceeded': self.handle_metric_threshold_exceeded,
				'system.health_check_failed': self.handle_system_health_check_failed,
				'user.connected': self.handle_user_connected,
				'user.disconnected': self.handle_user_disconnected
			}
			
			# Register event listeners with the workflow service
			for event_type, handler in event_mappings.items():
				# Register with internal event system
				self._register_internal_listener(event_type, handler)
				logger.debug(f"Registered event listener for {event_type}")
			
			# Set up periodic health check events
			self._setup_periodic_events()
			
			logger.info(f"Set up {len(event_mappings)} event listeners for real-time updates")
			
		except Exception as e:
			logger.error(f"Failed to set up event listeners: {e}")
			raise
	
	def _register_internal_listener(self, event_type: str, handler):
		"""Register an internal event listener."""
		try:
			# Store the mapping for internal event dispatch
			if not hasattr(self, '_event_handlers'):
				self._event_handlers = {}
			
			if event_type not in self._event_handlers:
				self._event_handlers[event_type] = []
			
			self._event_handlers[event_type].append(handler)
			
		except Exception as e:
			logger.error(f"Failed to register internal listener for {event_type}: {e}")
	
	def _setup_periodic_events(self):
		"""Set up periodic events for system monitoring."""
		try:
			import asyncio
			
			# Schedule periodic system health checks
			self._health_check_task = asyncio.create_task(self._periodic_health_check())
			
			# Schedule periodic metrics collection
			self._metrics_task = asyncio.create_task(self._periodic_metrics_collection())
			
			logger.info("Set up periodic event tasks")
			
		except Exception as e:
			logger.error(f"Failed to set up periodic events: {e}")
	
	async def _periodic_health_check(self):
		"""Perform periodic health checks and emit events."""
		while True:
			try:
				await asyncio.sleep(30)  # Check every 30 seconds
				
				# Perform system health check
				health_status = await self._check_system_health()
				
				if not health_status['healthy']:
					await self.emit_event('system.health_check_failed', health_status)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Error in periodic health check: {e}")
				await asyncio.sleep(60)  # Wait longer on error
	
	async def _periodic_metrics_collection(self):
		"""Collect and emit periodic metrics."""
		while True:
			try:
				await asyncio.sleep(60)  # Collect every minute
				
				# Collect system metrics
				metrics = await self._collect_system_metrics()
				
				# Check for threshold violations
				for metric_name, value in metrics.items():
					if await self._check_metric_threshold(metric_name, value):
						await self.emit_event('metric.threshold_exceeded', {
							'metric_name': metric_name,
							'value': value,
							'timestamp': datetime.utcnow().isoformat()
						})
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Error in periodic metrics collection: {e}")
				await asyncio.sleep(120)  # Wait longer on error
	
	async def _check_system_health(self) -> Dict[str, Any]:
		"""Check overall system health."""
		try:
			import psutil
			
			health_status = {
				'healthy': True,
				'timestamp': datetime.utcnow().isoformat(),
				'checks': {}
			}
			
			# Check CPU usage
			cpu_percent = psutil.cpu_percent(interval=1)
			health_status['checks']['cpu'] = {
				'value': cpu_percent,
				'healthy': cpu_percent < 90,
				'threshold': 90
			}
			
			# Check memory usage
			memory = psutil.virtual_memory()
			health_status['checks']['memory'] = {
				'value': memory.percent,
				'healthy': memory.percent < 85,
				'threshold': 85
			}
			
			# Check disk usage
			disk = psutil.disk_usage('/')
			disk_percent = (disk.used / disk.total) * 100
			health_status['checks']['disk'] = {
				'value': disk_percent,
				'healthy': disk_percent < 80,
				'threshold': 80
			}
			
			# Overall health status
			health_status['healthy'] = all(
				check['healthy'] for check in health_status['checks'].values()
			)
			
			return health_status
			
		except Exception as e:
			logger.error(f"Error checking system health: {e}")
			return {
				'healthy': False,
				'error': str(e),
				'timestamp': datetime.utcnow().isoformat()
			}
	
	async def _collect_system_metrics(self) -> Dict[str, float]:
		"""Collect current system metrics."""
		try:
			import psutil
			
			metrics = {}
			
			# CPU metrics
			metrics['cpu_percent'] = psutil.cpu_percent()
			metrics['cpu_count'] = psutil.cpu_count()
			
			# Memory metrics
			memory = psutil.virtual_memory()
			metrics['memory_percent'] = memory.percent
			metrics['memory_available_gb'] = memory.available / (1024**3)
			
			# Disk metrics
			disk = psutil.disk_usage('/')
			metrics['disk_percent'] = (disk.used / disk.total) * 100
			metrics['disk_free_gb'] = disk.free / (1024**3)
			
			# Network metrics
			net_io = psutil.net_io_counters()
			metrics['network_bytes_sent'] = net_io.bytes_sent
			metrics['network_bytes_recv'] = net_io.bytes_recv
			
			return metrics
			
		except Exception as e:
			logger.error(f"Error collecting system metrics: {e}")
			return {}
	
	async def _check_metric_threshold(self, metric_name: str, value: float) -> bool:
		"""Check if metric value exceeds threshold."""
		thresholds = {
			'cpu_percent': 85.0,
			'memory_percent': 80.0,
			'disk_percent': 75.0
		}
		
		threshold = thresholds.get(metric_name)
		if threshold is None:
			return False
		
		return value > threshold
	
	async def emit_event(self, event_type: str, data: Dict[str, Any]):
		"""Emit an event to registered handlers."""
		try:
			if hasattr(self, '_event_handlers') and event_type in self._event_handlers:
				for handler in self._event_handlers[event_type]:
					try:
						if asyncio.iscoroutinefunction(handler):
							await handler(data)
						else:
							handler(data)
					except Exception as e:
						logger.error(f"Error in event handler for {event_type}: {e}")
			
		except Exception as e:
			logger.error(f"Error emitting event {event_type}: {e}")
	
	async def start(self):
		"""Start the real-time event manager."""
		await self.realtime_service.start()
		logger.info("Real-time event manager started")
	
	async def stop(self):
		"""Stop the real-time event manager."""
		await self.realtime_service.stop()
		logger.info("Real-time event manager stopped")
	
	# Convenience methods for triggering events
	
	async def workflow_created(self, workflow: WorkflowDB):
		"""Trigger workflow created event."""
		await self.workflow_handler.on_workflow_created(workflow)
	
	async def workflow_updated(self, workflow: WorkflowDB, updated_fields: List[str]):
		"""Trigger workflow updated event."""
		await self.workflow_handler.on_workflow_updated(workflow, updated_fields)
	
	async def workflow_deleted(self, workflow_id: str, tenant_id: str, user_id: str):
		"""Trigger workflow deleted event."""
		await self.workflow_handler.on_workflow_deleted(workflow_id, tenant_id, user_id)
	
	async def workflow_executed(self, workflow: WorkflowDB, instance: WorkflowInstanceDB):
		"""Trigger workflow executed event."""
		await self.workflow_handler.on_workflow_executed(workflow, instance)
	
	async def instance_created(self, instance: WorkflowInstanceDB):
		"""Trigger instance created event."""
		await self.instance_handler.on_instance_created(instance)
	
	async def instance_started(self, instance: WorkflowInstanceDB):
		"""Trigger instance started event."""
		await self.instance_handler.on_instance_started(instance)
	
	async def instance_completed(self, instance: WorkflowInstanceDB):
		"""Trigger instance completed event."""
		await self.instance_handler.on_instance_completed(instance)
	
	async def instance_failed(self, instance: WorkflowInstanceDB):
		"""Trigger instance failed event."""
		await self.instance_handler.on_instance_failed(instance)
	
	async def instance_cancelled(self, instance: WorkflowInstanceDB, cancelled_by: str):
		"""Trigger instance cancelled event."""
		await self.instance_handler.on_instance_cancelled(instance, cancelled_by)
	
	async def instance_progress(self, instance: WorkflowInstanceDB, progress_data: Dict[str, Any]):
		"""Trigger instance progress event."""
		await self.instance_handler.on_instance_progress(instance, progress_data)
	
	async def task_started(self, task_execution: TaskExecutionDB):
		"""Trigger task started event."""
		await self.task_handler.on_task_started(task_execution)
	
	async def task_completed(self, task_execution: TaskExecutionDB):
		"""Trigger task completed event."""
		await self.task_handler.on_task_completed(task_execution)
	
	async def task_failed(self, task_execution: TaskExecutionDB):
		"""Trigger task failed event."""
		await self.task_handler.on_task_failed(task_execution)
	
	async def task_retrying(self, task_execution: TaskExecutionDB):
		"""Trigger task retrying event."""
		await self.task_handler.on_task_retrying(task_execution)
	
	async def task_progress(self, task_execution: TaskExecutionDB, progress_data: Dict[str, Any]):
		"""Trigger task progress event."""
		await self.task_handler.on_task_progress(task_execution, progress_data)
	
	# Collaboration events
	
	async def component_added(self, workflow_id: str, tenant_id: str, user_id: str, 
							  component_data: Dict[str, Any]):
		"""Trigger component added event."""
		await self.collaboration_handler.on_component_added(workflow_id, tenant_id, user_id, component_data)
	
	async def component_updated(self, workflow_id: str, tenant_id: str, user_id: str, 
								component_data: Dict[str, Any]):
		"""Trigger component updated event."""
		await self.collaboration_handler.on_component_updated(workflow_id, tenant_id, user_id, component_data)
	
	async def component_deleted(self, workflow_id: str, tenant_id: str, user_id: str, 
								component_id: str):
		"""Trigger component deleted event."""
		await self.collaboration_handler.on_component_deleted(workflow_id, tenant_id, user_id, component_id)
	
	async def connection_created(self, workflow_id: str, tenant_id: str, user_id: str, 
								 connection_data: Dict[str, Any]):
		"""Trigger connection created event."""
		await self.collaboration_handler.on_connection_created(workflow_id, tenant_id, user_id, connection_data)
	
	async def connection_deleted(self, workflow_id: str, tenant_id: str, user_id: str, 
								 connection_id: str):
		"""Trigger connection deleted event."""
		await self.collaboration_handler.on_connection_deleted(workflow_id, tenant_id, user_id, connection_id)
	
	async def canvas_updated(self, workflow_id: str, tenant_id: str, user_id: str, 
							 canvas_data: Dict[str, Any]):
		"""Trigger canvas updated event."""
		await self.collaboration_handler.on_canvas_updated(workflow_id, tenant_id, user_id, canvas_data)
	
	# System events
	
	async def system_status_change(self, status_data: Dict[str, Any]):
		"""Trigger system status change event."""
		await self.system_handler.on_system_status_change(status_data)
	
	async def integration_status_change(self, integration_id: str, tenant_id: str, 
										status_data: Dict[str, Any]):
		"""Trigger integration status change event."""
		await self.system_handler.on_integration_status_change(integration_id, tenant_id, status_data)
	
	async def metrics_update(self, tenant_id: str, metrics_data: Dict[str, Any]):
		"""Trigger metrics update event."""
		await self.system_handler.on_metrics_update(tenant_id, metrics_data)