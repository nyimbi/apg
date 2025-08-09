"""
APG Integration API Management - Platform Integration

APG platform integration layer providing event-driven capability orchestration,
cross-capability workflow management, and unified service mesh coordination.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

import aioredis
from pydantic import BaseModel, Field, validator

from .discovery import ServiceDiscovery, APGCapabilityInfo, ServiceHealth
from .service import APILifecycleService, ConsumerManagementService, AnalyticsService
from .monitoring import MetricsCollector, HealthMonitor

# =============================================================================
# Integration Models
# =============================================================================

class EventType(str, Enum):
	"""APG platform event types."""
	CAPABILITY_REGISTERED = "capability.registered"
	CAPABILITY_UNREGISTERED = "capability.unregistered"
	CAPABILITY_HEALTH_CHANGED = "capability.health_changed"
	API_REGISTERED = "api.registered"
	API_DEREGISTERED = "api.deregistered"
	API_STATUS_CHANGED = "api.status_changed"
	WORKFLOW_STARTED = "workflow.started"
	WORKFLOW_COMPLETED = "workflow.completed"
	WORKFLOW_FAILED = "workflow.failed"
	POLICY_APPLIED = "policy.applied"
	CONSUMER_REGISTERED = "consumer.registered"
	USAGE_THRESHOLD_EXCEEDED = "usage.threshold_exceeded"

class WorkflowStatus(str, Enum):
	"""Workflow execution status."""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	TIMEOUT = "timeout"

class IntegrationEventPriority(str, Enum):
	"""Event priority levels."""
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	CRITICAL = "critical"

@dataclass
class APGEvent:
	"""APG platform event."""
	event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
	event_type: EventType = EventType.CAPABILITY_REGISTERED
	source_capability: str = "integration_api_management"
	target_capabilities: List[str] = field(default_factory=list)
	payload: Dict[str, Any] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)
	priority: IntegrationEventPriority = IntegrationEventPriority.NORMAL
	timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	correlation_id: Optional[str] = None
	tenant_id: str = "default"

class WorkflowStep(BaseModel):
	"""Individual workflow step."""
	
	step_id: str = Field(..., description="Unique step identifier")
	step_name: str = Field(..., description="Human-readable step name")
	capability_id: str = Field(..., description="Target capability")
	action: str = Field(..., description="Action to execute")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
	timeout_seconds: int = Field(300, description="Step timeout")
	retry_attempts: int = Field(3, description="Number of retry attempts")
	on_success: Optional[str] = Field(None, description="Next step on success")
	on_failure: Optional[str] = Field(None, description="Next step on failure")
	conditions: Dict[str, Any] = Field(default_factory=dict, description="Execution conditions")

class CrossCapabilityWorkflow(BaseModel):
	"""Cross-capability workflow definition."""
	
	workflow_id: str = Field(..., description="Unique workflow identifier")
	workflow_name: str = Field(..., description="Human-readable workflow name")
	description: Optional[str] = Field(None, description="Workflow description")
	version: str = Field("1.0.0", description="Workflow version")
	
	# Triggers
	trigger_events: List[EventType] = Field(..., description="Events that trigger this workflow")
	trigger_conditions: Dict[str, Any] = Field(default_factory=dict, description="Trigger conditions")
	
	# Steps
	steps: List[WorkflowStep] = Field(..., description="Workflow steps")
	start_step: Optional[str] = Field(None, description="Initial step (defaults to first)")
	
	# Configuration
	timeout_seconds: int = Field(3600, description="Total workflow timeout")
	max_retries: int = Field(3, description="Maximum workflow retries")
	parallel_execution: bool = Field(False, description="Allow parallel step execution")
	
	# Status
	is_active: bool = Field(True, description="Workflow is active")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str = Field("system", description="Workflow creator")

@dataclass
class WorkflowExecution:
	"""Workflow execution instance."""
	execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
	workflow_id: str = ""
	status: WorkflowStatus = WorkflowStatus.PENDING
	trigger_event: Optional[APGEvent] = None
	current_step: Optional[str] = None
	completed_steps: List[str] = field(default_factory=list)
	failed_steps: List[str] = field(default_factory=list)
	step_results: Dict[str, Any] = field(default_factory=dict)
	error_message: Optional[str] = None
	started_at: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	tenant_id: str = "default"

class PolicyRule(BaseModel):
	"""APG platform policy rule."""
	
	rule_id: str = Field(..., description="Unique rule identifier")
	rule_name: str = Field(..., description="Human-readable rule name")
	rule_type: str = Field(..., description="Rule type (routing, security, etc.)")
	description: Optional[str] = Field(None, description="Rule description")
	
	# Conditions
	source_patterns: List[str] = Field(default_factory=list, description="Source capability patterns")
	target_patterns: List[str] = Field(default_factory=list, description="Target capability patterns")
	event_patterns: List[str] = Field(default_factory=list, description="Event type patterns")
	conditions: Dict[str, Any] = Field(default_factory=dict, description="Additional conditions")
	
	# Actions
	actions: List[Dict[str, Any]] = Field(..., description="Actions to execute")
	priority: int = Field(100, description="Rule priority (lower = higher priority)")
	
	# Status
	is_active: bool = Field(True, description="Rule is active")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# =============================================================================
# APG Platform Integration Manager
# =============================================================================

class APGIntegrationManager:
	"""Manages APG platform integration and orchestration."""
	
	def __init__(self, redis_client: aioredis.Redis,
				 service_discovery: ServiceDiscovery,
				 api_service: APILifecycleService,
				 consumer_service: ConsumerManagementService,
				 analytics_service: AnalyticsService,
				 metrics_collector: MetricsCollector,
				 health_monitor: HealthMonitor):
		
		self.redis = redis_client
		self.service_discovery = service_discovery
		self.api_service = api_service
		self.consumer_service = consumer_service
		self.analytics_service = analytics_service
		self.metrics_collector = metrics_collector
		self.health_monitor = health_monitor
		
		# Event management
		self.event_handlers = {}
		self.event_queue = asyncio.Queue()
		
		# Workflow management
		self.workflows = {}
		self.workflow_executions = {}
		self.workflow_engine_task = None
		
		# Policy management
		self.policy_rules = {}
		
		# Integration tasks
		self.integration_tasks = []
		
		# Callbacks
		self.event_callbacks = []
		self.workflow_callbacks = []
	
	async def initialize(self):
		"""Initialize the integration manager."""
		
		# Register service discovery callbacks
		self.service_discovery.add_service_added_callback(self._on_service_added)
		self.service_discovery.add_service_removed_callback(self._on_service_removed)
		self.service_discovery.add_service_health_changed_callback(self._on_service_health_changed)
		
		# Load existing workflows and policies
		await self._load_workflows_from_storage()
		await self._load_policies_from_storage()
		
		# Start integration tasks
		self.integration_tasks = [
			asyncio.create_task(self._event_processing_loop()),
			asyncio.create_task(self._workflow_engine_loop()),
			asyncio.create_task(self._policy_enforcement_loop()),
			asyncio.create_task(self._health_monitoring_loop())
		]
	
	async def shutdown(self):
		"""Shutdown the integration manager."""
		
		# Cancel all tasks
		for task in self.integration_tasks:
			task.cancel()
		
		await asyncio.gather(*self.integration_tasks, return_exceptions=True)
	
	# =============================================================================
	# Event Management
	# =============================================================================
	
	async def publish_event(self, event: APGEvent):
		"""Publish an APG platform event."""
		
		try:
			# Add to local queue
			await self.event_queue.put(event)
			
			# Store in Redis for other instances
			event_key = f"apg:events:{event.event_id}"
			await self.redis.setex(
				event_key,
				3600,  # 1 hour TTL
				json.dumps({
					'event_id': event.event_id,
					'event_type': event.event_type.value,
					'source_capability': event.source_capability,
					'target_capabilities': event.target_capabilities,
					'payload': event.payload,
					'metadata': event.metadata,
					'priority': event.priority.value,
					'timestamp': event.timestamp.isoformat(),
					'correlation_id': event.correlation_id,
					'tenant_id': event.tenant_id
				})
			)
			
			# Publish to Redis pub/sub for real-time distribution
			await self.redis.publish(
				f"apg:events:{event.tenant_id}",
				json.dumps({
					'event_id': event.event_id,
					'event_type': event.event_type.value,
					'source_capability': event.source_capability
				})
			)
			
		except Exception as e:
			print(f"Error publishing event {event.event_id}: {e}")
	
	def add_event_handler(self, event_type: EventType, handler: Callable[[APGEvent], None]):
		"""Add event handler for specific event type."""
		
		if event_type not in self.event_handlers:
			self.event_handlers[event_type] = []
		
		self.event_handlers[event_type].append(handler)
	
	async def _event_processing_loop(self):
		"""Main event processing loop."""
		
		while True:
			try:
				# Get event from queue (with timeout)
				event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
				
				# Process event
				await self._process_event(event)
				
			except asyncio.TimeoutError:
				continue
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Error in event processing loop: {e}")
	
	async def _process_event(self, event: APGEvent):
		"""Process a single event."""
		
		try:
			# Execute event handlers
			handlers = self.event_handlers.get(event.event_type, [])
			for handler in handlers:
				try:
					await handler(event)
				except Exception as e:
					print(f"Error in event handler for {event.event_type}: {e}")
			
			# Check for workflow triggers
			await self._check_workflow_triggers(event)
			
			# Apply policy rules
			await self._apply_policy_rules(event)
			
			# Record event metrics
			await self._record_event_metrics(event)
			
			# Trigger callbacks
			for callback in self.event_callbacks:
				try:
					await callback(event)
				except Exception as e:
					print(f"Error in event callback: {e}")
					
		except Exception as e:
			print(f"Error processing event {event.event_id}: {e}")
	
	# =============================================================================
	# Workflow Management
	# =============================================================================
	
	async def register_workflow(self, workflow: CrossCapabilityWorkflow) -> bool:
		"""Register a cross-capability workflow."""
		
		try:
			# Validate workflow
			if not await self._validate_workflow(workflow):
				return False
			
			# Store workflow
			self.workflows[workflow.workflow_id] = workflow
			
			# Persist to Redis
			workflow_key = f"apg:workflows:{workflow.workflow_id}"
			await self.redis.setex(
				workflow_key,
				86400,  # 24 hour TTL
				workflow.json()
			)
			
			return True
			
		except Exception as e:
			print(f"Error registering workflow {workflow.workflow_id}: {e}")
			return False
	
	async def unregister_workflow(self, workflow_id: str) -> bool:
		"""Unregister a workflow."""
		
		try:
			# Remove from local storage
			self.workflows.pop(workflow_id, None)
			
			# Remove from Redis
			workflow_key = f"apg:workflows:{workflow_id}"
			await self.redis.delete(workflow_key)
			
			return True
			
		except Exception as e:
			print(f"Error unregistering workflow {workflow_id}: {e}")
			return False
	
	async def execute_workflow(self, workflow_id: str, trigger_event: APGEvent) -> Optional[str]:
		"""Execute a workflow."""
		
		try:
			workflow = self.workflows.get(workflow_id)
			if not workflow or not workflow.is_active:
				return None
			
			# Create execution instance
			execution = WorkflowExecution(
				workflow_id=workflow_id,
				trigger_event=trigger_event,
				tenant_id=trigger_event.tenant_id
			)
			
			# Store execution
			self.workflow_executions[execution.execution_id] = execution
			
			# Start execution
			execution.status = WorkflowStatus.RUNNING
			execution.started_at = datetime.now(timezone.utc)
			
			# Queue for processing
			await self._queue_workflow_execution(execution)
			
			return execution.execution_id
			
		except Exception as e:
			print(f"Error executing workflow {workflow_id}: {e}")
			return None
	
	async def _workflow_engine_loop(self):
		"""Workflow execution engine loop."""
		
		while True:
			try:
				# Process pending workflow executions
				pending_executions = [
					exec for exec in self.workflow_executions.values()
					if exec.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]
				]
				
				for execution in pending_executions:
					await self._process_workflow_execution(execution)
				
				await asyncio.sleep(1)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Error in workflow engine loop: {e}")
				await asyncio.sleep(1)
	
	async def _process_workflow_execution(self, execution: WorkflowExecution):
		"""Process a workflow execution."""
		
		try:
			workflow = self.workflows.get(execution.workflow_id)
			if not workflow:
				execution.status = WorkflowStatus.FAILED
				execution.error_message = "Workflow not found"
				return
			
			# Check timeout
			if execution.started_at:
				elapsed = (datetime.now(timezone.utc) - execution.started_at).total_seconds()
				if elapsed > workflow.timeout_seconds:
					execution.status = WorkflowStatus.TIMEOUT
					execution.error_message = "Workflow timeout"
					return
			
			# Determine next step
			if not execution.current_step:
				execution.current_step = workflow.start_step or workflow.steps[0].step_id
			
			# Execute current step
			current_step = next((s for s in workflow.steps if s.step_id == execution.current_step), None)
			if not current_step:
				execution.status = WorkflowStatus.COMPLETED
				execution.completed_at = datetime.now(timezone.utc)
				return
			
			# Execute step
			step_result = await self._execute_workflow_step(execution, current_step)
			
			if step_result.get('success', False):
				# Step succeeded
				execution.completed_steps.append(current_step.step_id)
				execution.step_results[current_step.step_id] = step_result
				
				# Determine next step
				next_step_id = current_step.on_success
				if next_step_id:
					execution.current_step = next_step_id
				else:
					# No more steps, workflow completed
					execution.status = WorkflowStatus.COMPLETED
					execution.completed_at = datetime.now(timezone.utc)
			else:
				# Step failed
				execution.failed_steps.append(current_step.step_id)
				
				# Handle failure
				if current_step.on_failure:
					execution.current_step = current_step.on_failure
				else:
					execution.status = WorkflowStatus.FAILED
					execution.error_message = step_result.get('error', 'Step execution failed')
					execution.completed_at = datetime.now(timezone.utc)
			
		except Exception as e:
			execution.status = WorkflowStatus.FAILED
			execution.error_message = str(e)
			execution.completed_at = datetime.now(timezone.utc)
			print(f"Error processing workflow execution {execution.execution_id}: {e}")
	
	async def _execute_workflow_step(self, execution: WorkflowExecution, step: WorkflowStep) -> Dict[str, Any]:
		"""Execute a single workflow step."""
		
		try:
			# Resolve capability URL
			capability_url = await self.service_discovery.resolve_service(step.capability_id)
			if not capability_url:
				return {
					'success': False,
					'error': f'Capability {step.capability_id} not found'
				}
			
			# Build request payload
			payload = dict(step.parameters)
			
			# Substitute variables from trigger event and previous step results
			payload = self._substitute_workflow_variables(payload, execution)
			
			# Execute step action
			import aiohttp
			
			async with aiohttp.ClientSession() as session:
				action_url = f"{capability_url}/api/v1/{step.action}"
				
				async with session.post(
					action_url,
					json=payload,
					timeout=aiohttp.ClientTimeout(total=step.timeout_seconds)
				) as response:
					
					if response.status < 400:
						result_data = await response.json()
						return {
							'success': True,
							'data': result_data,
							'status_code': response.status
						}
					else:
						error_data = await response.text()
						return {
							'success': False,
							'error': f'HTTP {response.status}: {error_data}',
							'status_code': response.status
						}
		
		except asyncio.TimeoutError:
			return {
				'success': False,
				'error': f'Step timeout after {step.timeout_seconds} seconds'
			}
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}
	
	# =============================================================================
	# Policy Management
	# =============================================================================
	
	async def register_policy_rule(self, rule: PolicyRule) -> bool:
		"""Register a policy rule."""
		
		try:
			self.policy_rules[rule.rule_id] = rule
			
			# Persist to Redis
			rule_key = f"apg:policies:{rule.rule_id}"
			await self.redis.setex(rule_key, 86400, rule.json())
			
			return True
			
		except Exception as e:
			print(f"Error registering policy rule {rule.rule_id}: {e}")
			return False
	
	async def _policy_enforcement_loop(self):
		"""Policy enforcement background loop."""
		
		while True:
			try:
				# Periodic policy evaluation and enforcement
				await self._evaluate_policies()
				await asyncio.sleep(30)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Error in policy enforcement loop: {e}")
				await asyncio.sleep(30)
	
	# =============================================================================
	# Health Monitoring Integration
	# =============================================================================
	
	async def _health_monitoring_loop(self):
		"""Health monitoring integration loop."""
		
		while True:
			try:
				# Get overall platform health
				health_report = await self.health_monitor.get_health_report()
				
				# Publish health events if needed
				if health_report.overall_status != ServiceHealth.HEALTHY:
					await self.publish_event(APGEvent(
						event_type=EventType.CAPABILITY_HEALTH_CHANGED,
						payload={
							'status': health_report.overall_status.value,
							'alerts': health_report.alerts,
							'timestamp': health_report.timestamp.isoformat()
						},
						priority=IntegrationEventPriority.HIGH
					))
				
				await asyncio.sleep(60)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Error in health monitoring loop: {e}")
				await asyncio.sleep(60)
	
	# =============================================================================
	# Service Discovery Integration
	# =============================================================================
	
	async def _on_service_added(self, service_id: str, service_data: Dict[str, Any]):
		"""Handle service addition event."""
		
		await self.publish_event(APGEvent(
			event_type=EventType.CAPABILITY_REGISTERED,
			payload={
				'capability_id': service_id,
				'service_data': service_data
			}
		))
	
	async def _on_service_removed(self, service_id: str):
		"""Handle service removal event."""
		
		await self.publish_event(APGEvent(
			event_type=EventType.CAPABILITY_UNREGISTERED,
			payload={'capability_id': service_id}
		))
	
	async def _on_service_health_changed(self, service_id: str, health_status: ServiceHealth):
		"""Handle service health change event."""
		
		await self.publish_event(APGEvent(
			event_type=EventType.CAPABILITY_HEALTH_CHANGED,
			payload={
				'capability_id': service_id,
				'health_status': health_status.value
			},
			priority=IntegrationEventPriority.HIGH if health_status == ServiceHealth.UNHEALTHY else IntegrationEventPriority.NORMAL
		))
	
	# =============================================================================
	# Helper Methods
	# =============================================================================
	
	async def _load_workflows_from_storage(self):
		"""Load workflows from Redis storage."""
		
		try:
			pattern = "apg:workflows:*"
			async for key in self.redis.scan_iter(match=pattern):
				data = await self.redis.get(key)
				if data:
					try:
						workflow = CrossCapabilityWorkflow.parse_raw(data)
						self.workflows[workflow.workflow_id] = workflow
					except Exception as e:
						print(f"Error loading workflow from {key}: {e}")
		except Exception as e:
			print(f"Error loading workflows from storage: {e}")
	
	async def _load_policies_from_storage(self):
		"""Load policies from Redis storage."""
		
		try:
			pattern = "apg:policies:*"
			async for key in self.redis.scan_iter(match=pattern):
				data = await self.redis.get(key)
				if data:
					try:
						rule = PolicyRule.parse_raw(data)
						self.policy_rules[rule.rule_id] = rule
					except Exception as e:
						print(f"Error loading policy from {key}: {e}")
		except Exception as e:
			print(f"Error loading policies from storage: {e}")
	
	async def _validate_workflow(self, workflow: CrossCapabilityWorkflow) -> bool:
		"""Validate workflow definition."""
		
		try:
			# Check that all referenced capabilities exist
			for step in workflow.steps:
				capability = await self.service_discovery.get_capability(step.capability_id)
				if not capability:
					print(f"Workflow {workflow.workflow_id}: Capability {step.capability_id} not found")
					return False
			
			# Check step references
			step_ids = {step.step_id for step in workflow.steps}
			for step in workflow.steps:
				if step.on_success and step.on_success not in step_ids:
					print(f"Workflow {workflow.workflow_id}: Invalid on_success reference: {step.on_success}")
					return False
				if step.on_failure and step.on_failure not in step_ids:
					print(f"Workflow {workflow.workflow_id}: Invalid on_failure reference: {step.on_failure}")
					return False
			
			return True
			
		except Exception as e:
			print(f"Error validating workflow {workflow.workflow_id}: {e}")
			return False
	
	async def _check_workflow_triggers(self, event: APGEvent):
		"""Check if event triggers any workflows."""
		
		for workflow in self.workflows.values():
			if not workflow.is_active:
				continue
			
			if event.event_type in workflow.trigger_events:
				# Check trigger conditions
				if self._evaluate_trigger_conditions(event, workflow.trigger_conditions):
					await self.execute_workflow(workflow.workflow_id, event)
	
	def _evaluate_trigger_conditions(self, event: APGEvent, conditions: Dict[str, Any]) -> bool:
		"""Evaluate workflow trigger conditions."""
		
		# Simple condition evaluation (can be extended)
		for key, expected_value in conditions.items():
			if key in event.payload:
				if event.payload[key] != expected_value:
					return False
			elif key in event.metadata:
				if event.metadata[key] != expected_value:
					return False
		
		return True
	
	async def _apply_policy_rules(self, event: APGEvent):
		"""Apply policy rules to event."""
		
		# Sort rules by priority
		sorted_rules = sorted(
			[rule for rule in self.policy_rules.values() if rule.is_active],
			key=lambda r: r.priority
		)
		
		for rule in sorted_rules:
			if self._rule_matches_event(rule, event):
				await self._execute_policy_actions(rule, event)
	
	def _rule_matches_event(self, rule: PolicyRule, event: APGEvent) -> bool:
		"""Check if policy rule matches event."""
		
		# Check source patterns
		if rule.source_patterns:
			if not any(self._pattern_matches(pattern, event.source_capability) 
					  for pattern in rule.source_patterns):
				return False
		
		# Check event patterns
		if rule.event_patterns:
			if not any(self._pattern_matches(pattern, event.event_type.value) 
					  for pattern in rule.event_patterns):
				return False
		
		return True
	
	def _pattern_matches(self, pattern: str, value: str) -> bool:
		"""Check if pattern matches value (supports wildcards)."""
		
		import re
		
		# Convert shell-style wildcards to regex
		pattern = pattern.replace('*', '.*').replace('?', '.')
		return bool(re.match(f'^{pattern}$', value))
	
	async def _execute_policy_actions(self, rule: PolicyRule, event: APGEvent):
		"""Execute policy rule actions."""
		
		for action in rule.actions:
			action_type = action.get('type')
			
			if action_type == 'log':
				print(f"Policy {rule.rule_name}: {action.get('message', 'Action executed')}")
			elif action_type == 'block':
				# Block event processing
				return
			elif action_type == 'transform':
				# Transform event data
				transformations = action.get('transformations', {})
				for key, value in transformations.items():
					event.payload[key] = value
			elif action_type == 'route':
				# Route to specific capabilities
				targets = action.get('targets', [])
				event.target_capabilities.extend(targets)
	
	def _substitute_workflow_variables(self, data: Any, execution: WorkflowExecution) -> Any:
		"""Substitute workflow variables in data."""
		
		if isinstance(data, dict):
			return {k: self._substitute_workflow_variables(v, execution) for k, v in data.items()}
		elif isinstance(data, list):
			return [self._substitute_workflow_variables(item, execution) for item in data]
		elif isinstance(data, str) and data.startswith('${'):
			# Variable substitution
			var_path = data[2:-1]  # Remove ${ and }
			
			if var_path.startswith('trigger_event.'):
				# Access trigger event data
				path_parts = var_path[14:].split('.')
				value = execution.trigger_event
				for part in path_parts:
					if hasattr(value, part):
						value = getattr(value, part)
					elif isinstance(value, dict) and part in value:
						value = value[part]
					else:
						return data  # Return original if path not found
				return value
			elif var_path.startswith('step_results.'):
				# Access previous step results
				path_parts = var_path[13:].split('.')
				if len(path_parts) >= 2:
					step_id = path_parts[0]
					if step_id in execution.step_results:
						value = execution.step_results[step_id]
						for part in path_parts[1:]:
							if isinstance(value, dict) and part in value:
								value = value[part]
							else:
								return data
						return value
		
		return data
	
	async def _queue_workflow_execution(self, execution: WorkflowExecution):
		"""Queue workflow execution for processing."""
		
		# Store execution details in Redis for persistence
		execution_key = f"apg:executions:{execution.execution_id}"
		await self.redis.setex(
			execution_key,
			3600,  # 1 hour TTL
			json.dumps({
				'execution_id': execution.execution_id,
				'workflow_id': execution.workflow_id,
				'status': execution.status.value,
				'started_at': execution.started_at.isoformat() if execution.started_at else None,
				'tenant_id': execution.tenant_id
			})
		)
	
	async def _record_event_metrics(self, event: APGEvent):
		"""Record event metrics."""
		
		await self.metrics_collector.record_metric({
			'name': 'apg_events_total',
			'type': 'counter',
			'value': 1,
			'labels': {
				'event_type': event.event_type.value,
				'source_capability': event.source_capability,
				'priority': event.priority.value,
				'tenant_id': event.tenant_id
			}
		})
	
	async def _evaluate_policies(self):
		"""Periodic policy evaluation."""
		
		# This could include things like:
		# - Resource usage policies
		# - Security policy compliance checks
		# - Performance threshold monitoring
		pass

# =============================================================================
# Export Integration Components
# =============================================================================

__all__ = [
	# Enums
	'EventType',
	'WorkflowStatus',
	'IntegrationEventPriority',
	
	# Data Classes
	'APGEvent',
	'WorkflowExecution',
	
	# Models
	'WorkflowStep',
	'CrossCapabilityWorkflow',
	'PolicyRule',
	
	# Core Component
	'APGIntegrationManager'
]