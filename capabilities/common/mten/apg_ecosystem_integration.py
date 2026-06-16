#!/usr/bin/env python3
"""
APG Ecosystem Integration

Company: Datacraft
Copyright: © 2025
Author: Nyimbi Odero

Complete integration framework for Multi-Tenant Management (MTen) capability
within the APG ecosystem. Enables seamless cross-capability workflows,
composition orchestration, marketplace integration, and lifecycle management.

This module provides:
- Cross-capability workflow orchestration
- Capability composition and discovery
- Marketplace integration for capability lifecycle
- Event-driven capability communication
- Resource sharing and optimization across capabilities
- Unified configuration and monitoring
"""

import asyncio
import json
import time
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import uuid
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict
from pydantic import StrictStr, StrictInt, StrictFloat, StrictBool


class IntegrationType(str, Enum):
	"""Types of APG ecosystem integrations"""
	WORKFLOW_ORCHESTRATION = "workflow_orchestration"
	DATA_PIPELINE = "data_pipeline"
	SERVICE_MESH = "service_mesh"
	EVENT_STREAMING = "event_streaming"
	RESOURCE_SHARING = "resource_sharing"
	CAPABILITY_COMPOSITION = "capability_composition"


class WorkflowStatus(str, Enum):
	"""Status of workflow execution"""
	PENDING = "pending"
	RUNNING = "running"
	SUCCESS = "success"
	FAILED = "failed"
	CANCELLED = "cancelled"
	PAUSED = "paused"


class CapabilityState(str, Enum):
	"""State of capability within ecosystem"""
	DISCOVERING = "discovering"
	AVAILABLE = "available"
	ACTIVE = "active"
	BUSY = "busy"
	MAINTENANCE = "maintenance"
	DISABLED = "disabled"
	ERROR = "error"


class EventType(str, Enum):
	"""Types of ecosystem events"""
	CAPABILITY_REGISTERED = "capability_registered"
	CAPABILITY_UPDATED = "capability_updated"
	CAPABILITY_REMOVED = "capability_removed"
	WORKFLOW_STARTED = "workflow_started"
	WORKFLOW_COMPLETED = "workflow_completed"
	WORKFLOW_FAILED = "workflow_failed"
	RESOURCE_ALLOCATED = "resource_allocated"
	RESOURCE_RELEASED = "resource_released"
	TENANT_PROVISIONED = "tenant_provisioned"
	TENANT_UPDATED = "tenant_updated"
	TENANT_DELETED = "tenant_deleted"
	ALERT_TRIGGERED = "alert_triggered"
	PERFORMANCE_THRESHOLD_EXCEEDED = "performance_threshold_exceeded"


class ResourceType(str, Enum):
	"""Types of shared resources"""
	COMPUTE = "compute"
	STORAGE = "storage"
	NETWORK = "network"
	DATABASE = "database"
	CACHE = "cache"
	QUEUE = "queue"
	AI_MODEL = "ai_model"
	CREDENTIALS = "credentials"


# Pydantic Models

class EcosystemCapability(BaseModel):
	"""Represents a capability within the APG ecosystem"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	version: str
	namespace: str
	description: str
	category: str
	state: CapabilityState = CapabilityState.DISCOVERING
	health_endpoint: str
	api_endpoint: str
	supported_operations: List[str] = Field(default_factory=list)
	required_dependencies: List[str] = Field(default_factory=list)
	provided_interfaces: List[str] = Field(default_factory=list)
	resource_requirements: Dict[str, Any] = Field(default_factory=dict)
	configuration: Dict[str, Any] = Field(default_factory=dict)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
	last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkflowStep(BaseModel):
	"""Individual step in a cross-capability workflow"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	capability_id: str
	operation: str
	input_data: Dict[str, Any] = Field(default_factory=dict)
	output_mapping: Dict[str, str] = Field(default_factory=dict)
	dependencies: List[str] = Field(default_factory=list)
	timeout_seconds: int = 300
	retry_attempts: int = 3
	conditions: Dict[str, Any] = Field(default_factory=dict)
	metadata: Dict[str, Any] = Field(default_factory=dict)


class CrossCapabilityWorkflow(BaseModel):
	"""Definition and execution state of cross-capability workflow"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	version: str
	tenant_id: Optional[str] = None
	steps: List[WorkflowStep]
	status: WorkflowStatus = WorkflowStatus.PENDING
	current_step: Optional[str] = None
	context: Dict[str, Any] = Field(default_factory=dict)
	results: Dict[str, Any] = Field(default_factory=dict)
	errors: List[str] = Field(default_factory=list)
	started_at: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	created_by: str
	created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
	tags: List[str] = Field(default_factory=list)


class EcosystemEvent(BaseModel):
	"""Event within the APG ecosystem"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	id: str = Field(default_factory=uuid7str)
	type: EventType
	source_capability: str
	target_capabilities: List[str] = Field(default_factory=list)
	tenant_id: Optional[str] = None
	payload: Dict[str, Any] = Field(default_factory=dict)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
	correlation_id: Optional[str] = None
	parent_event_id: Optional[str] = None


class SharedResource(BaseModel):
	"""Shared resource within the ecosystem"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	resource_type: ResourceType
	provider_capability: str
	consumer_capabilities: List[str] = Field(default_factory=list)
	configuration: Dict[str, Any] = Field(default_factory=dict)
	allocation_policy: str = "fair_share"
	capacity: Dict[str, Any] = Field(default_factory=dict)
	current_usage: Dict[str, Any] = Field(default_factory=dict)
	health_status: str = "healthy"
	created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
	last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CapabilityComposition(BaseModel):
	"""Composition of multiple capabilities into a unified service"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	version: str
	component_capabilities: List[str]
	orchestration_config: Dict[str, Any] = Field(default_factory=dict)
	exposed_endpoints: List[str] = Field(default_factory=list)
	internal_workflows: List[str] = Field(default_factory=list)
	resource_allocation: Dict[str, Any] = Field(default_factory=dict)
	sla_requirements: Dict[str, Any] = Field(default_factory=dict)
	monitoring_config: Dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MarketplaceEntry(BaseModel):
	"""Marketplace entry for capability discovery and lifecycle management"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	id: str = Field(default_factory=uuid7str)
	capability_id: str
	name: str
	display_name: str
	description: str
	category: str
	tags: List[str] = Field(default_factory=list)
	version: str
	publisher: str
	license: str
	pricing_model: str = "free"
	installation_requirements: List[str] = Field(default_factory=list)
	compatibility: Dict[str, Any] = Field(default_factory=dict)
	documentation_url: Optional[str] = None
	support_url: Optional[str] = None
	screenshots: List[str] = Field(default_factory=list)
	downloads: int = 0
	rating: float = 0.0
	reviews: List[Dict[str, Any]] = Field(default_factory=list)
	published_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
	last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))


# Core Integration Classes

class EventBus:
	"""Enterprise-grade event bus for ecosystem communication"""
	
	def __init__(self):
		self.subscribers: Dict[EventType, List[Callable]] = {}
		self.event_history: List[EcosystemEvent] = []
		self.correlation_tracker: Dict[str, List[str]] = {}
		self.middleware: List[Callable] = []
		self.metrics = {
			'events_published': 0,
			'events_delivered': 0,
			'events_failed': 0,
			'active_subscriptions': 0
		}
	
	def subscribe(self, event_type: EventType, handler: Callable[[EcosystemEvent], Awaitable[None]]):
		"""Subscribe to ecosystem events"""
		if event_type not in self.subscribers:
			self.subscribers[event_type] = []
		self.subscribers[event_type].append(handler)
		self.metrics['active_subscriptions'] += 1
	
	def unsubscribe(self, event_type: EventType, handler: Callable):
		"""Unsubscribe from ecosystem events"""
		if event_type in self.subscribers and handler in self.subscribers[event_type]:
			self.subscribers[event_type].remove(handler)
			self.metrics['active_subscriptions'] -= 1
	
	async def publish(self, event: EcosystemEvent):
		"""Publish event to all subscribers"""
		try:
			# Apply middleware
			for middleware in self.middleware:
				event = await middleware(event)
			
			# Store event in history
			self.event_history.append(event)
			
			# Track correlation
			if event.correlation_id:
				if event.correlation_id not in self.correlation_tracker:
					self.correlation_tracker[event.correlation_id] = []
				self.correlation_tracker[event.correlation_id].append(event.id)
			
			# Deliver to subscribers
			if event.type in self.subscribers:
				delivery_tasks = []
				for handler in self.subscribers[event.type]:
					delivery_tasks.append(self._safe_deliver(handler, event))
				
				await asyncio.gather(*delivery_tasks, return_exceptions=True)
			
			self.metrics['events_published'] += 1
			
		except Exception as e:
			self.metrics['events_failed'] += 1
			raise Exception(f"Event publication failed: {e}")
	
	async def _safe_deliver(self, handler: Callable, event: EcosystemEvent):
		"""Safely deliver event to handler with error isolation"""
		try:
			await handler(event)
			self.metrics['events_delivered'] += 1
		except Exception as e:
			self.metrics['events_failed'] += 1
			# Log error but don't propagate to prevent cascade failures
			print(f"Event handler failed: {e}")


class CapabilityRegistry:
	"""Registry for discovering and managing capabilities within the ecosystem"""
	
	def __init__(self, event_bus: EventBus):
		self.capabilities: Dict[str, EcosystemCapability] = {}
		self.capability_dependencies: Dict[str, List[str]] = {}
		self.event_bus = event_bus
		self.health_check_interval = 30
		self.heartbeat_timeout = 60
		self.discovery_cache: Dict[str, List[str]] = {}
	
	async def register_capability(self, capability: EcosystemCapability) -> bool:
		"""Register a new capability in the ecosystem"""
		try:
			# Validate capability
			await self._validate_capability(capability)
			
			# Store capability
			self.capabilities[capability.id] = capability
			
			# Update dependencies
			self.capability_dependencies[capability.id] = capability.required_dependencies
			
			# Clear discovery cache
			self.discovery_cache.clear()
			
			# Publish registration event
			event = EcosystemEvent(
				type=EventType.CAPABILITY_REGISTERED,
				source_capability=capability.id,
				payload={
					'capability_name': capability.name,
					'namespace': capability.namespace,
					'version': capability.version,
					'operations': capability.supported_operations
				}
			)
			await self.event_bus.publish(event)
			
			return True
		except Exception as e:
			print(f"Failed to register capability {capability.name}: {e}")
			return False
	
	async def unregister_capability(self, capability_id: str) -> bool:
		"""Unregister a capability from the ecosystem"""
		try:
			if capability_id in self.capabilities:
				capability = self.capabilities[capability_id]
				del self.capabilities[capability_id]
				
				if capability_id in self.capability_dependencies:
					del self.capability_dependencies[capability_id]
				
				# Clear discovery cache
				self.discovery_cache.clear()
				
				# Publish removal event
				event = EcosystemEvent(
					type=EventType.CAPABILITY_REMOVED,
					source_capability=capability_id,
					payload={'capability_name': capability.name}
				)
				await self.event_bus.publish(event)
				
				return True
			return False
		except Exception as e:
			print(f"Failed to unregister capability {capability_id}: {e}")
			return False
	
	async def discover_capabilities(
		self, 
		category: Optional[str] = None,
		operation: Optional[str] = None,
		namespace: Optional[str] = None
	) -> List[EcosystemCapability]:
		"""Discover capabilities matching criteria"""
		cache_key = f"{category}:{operation}:{namespace}"
		
		if cache_key in self.discovery_cache:
			return [self.capabilities[cap_id] for cap_id in self.discovery_cache[cache_key]]
		
		matching_capabilities = []
		
		for capability in self.capabilities.values():
			if capability.state not in [CapabilityState.AVAILABLE, CapabilityState.ACTIVE]:
				continue
				
			if category and capability.category != category:
				continue
			
			if operation and operation not in capability.supported_operations:
				continue
			
			if namespace and capability.namespace != namespace:
				continue
			
			matching_capabilities.append(capability)
		
		# Cache results
		self.discovery_cache[cache_key] = [cap.id for cap in matching_capabilities]
		
		return matching_capabilities
	
	async def update_capability_state(self, capability_id: str, new_state: CapabilityState):
		"""Update capability state"""
		if capability_id in self.capabilities:
			old_state = self.capabilities[capability_id].state
			self.capabilities[capability_id].state = new_state
			self.capabilities[capability_id].last_heartbeat = datetime.now(UTC)
			
			# Publish state change event
			event = EcosystemEvent(
				type=EventType.CAPABILITY_UPDATED,
				source_capability=capability_id,
				payload={
					'old_state': old_state.value,
					'new_state': new_state.value
				}
			)
			await self.event_bus.publish(event)
			
			# Clear discovery cache if state affects availability
			if old_state != new_state:
				self.discovery_cache.clear()
	
	async def _validate_capability(self, capability: EcosystemCapability):
		"""Validate capability registration requirements"""
		# Check for required fields
		if not capability.name or not capability.version:
			raise ValueError("Capability name and version are required")
		
		# Check for unique name within namespace
		for existing_cap in self.capabilities.values():
			if (existing_cap.name == capability.name and 
				existing_cap.namespace == capability.namespace and
				existing_cap.id != capability.id):
				raise ValueError(f"Capability {capability.name} already exists in namespace {capability.namespace}")
		
		# Validate endpoints are accessible
		# In production, would perform actual health checks
		if not capability.health_endpoint or not capability.api_endpoint:
			raise ValueError("Health and API endpoints are required")


class WorkflowOrchestrator:
	"""Orchestrator for cross-capability workflows"""
	
	def __init__(self, capability_registry: CapabilityRegistry, event_bus: EventBus):
		self.capability_registry = capability_registry
		self.event_bus = event_bus
		self.active_workflows: Dict[str, CrossCapabilityWorkflow] = {}
		self.workflow_templates: Dict[str, CrossCapabilityWorkflow] = {}
		self.execution_history: List[Dict[str, Any]] = []
		self.step_executors: Dict[str, Callable] = {}
	
	def register_step_executor(self, operation: str, executor: Callable):
		"""Register executor for specific workflow step operations"""
		self.step_executors[operation] = executor
	
	async def create_workflow_template(self, workflow: CrossCapabilityWorkflow):
		"""Create a reusable workflow template"""
		self.workflow_templates[workflow.name] = workflow
	
	async def execute_workflow(self, workflow_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Execute a cross-capability workflow"""
		try:
			if workflow_id not in self.active_workflows:
				raise ValueError(f"Workflow {workflow_id} not found")
			
			workflow = self.active_workflows[workflow_id]
			
			# Initialize workflow execution
			workflow.status = WorkflowStatus.RUNNING
			workflow.started_at = datetime.now(UTC)
			if context:
				workflow.context.update(context)
			
			# Publish workflow start event
			event = EcosystemEvent(
				type=EventType.WORKFLOW_STARTED,
				source_capability="mten",
				tenant_id=workflow.tenant_id,
				payload={
					'workflow_id': workflow.id,
					'workflow_name': workflow.name,
					'step_count': len(workflow.steps)
				},
				correlation_id=workflow.id
			)
			await self.event_bus.publish(event)
			
			# Execute workflow steps
			execution_result = await self._execute_workflow_steps(workflow)
			
			# Update workflow status
			if execution_result['success']:
				workflow.status = WorkflowStatus.SUCCESS
				workflow.results.update(execution_result['results'])
				
				# Publish success event
				event = EcosystemEvent(
					type=EventType.WORKFLOW_COMPLETED,
					source_capability="mten",
					tenant_id=workflow.tenant_id,
					payload={
						'workflow_id': workflow.id,
						'execution_time': execution_result['execution_time'],
						'steps_completed': execution_result['steps_completed']
					},
					correlation_id=workflow.id
				)
				await self.event_bus.publish(event)
			else:
				workflow.status = WorkflowStatus.FAILED
				workflow.errors.extend(execution_result['errors'])
				
				# Publish failure event
				event = EcosystemEvent(
					type=EventType.WORKFLOW_FAILED,
					source_capability="mten",
					tenant_id=workflow.tenant_id,
					payload={
						'workflow_id': workflow.id,
						'errors': execution_result['errors'],
						'failed_step': execution_result.get('failed_step')
					},
					correlation_id=workflow.id
				)
				await self.event_bus.publish(event)
			
			workflow.completed_at = datetime.now(UTC)
			
			# Store in execution history
			self.execution_history.append({
				'workflow_id': workflow.id,
				'workflow_name': workflow.name,
				'status': workflow.status.value,
				'execution_time': execution_result['execution_time'],
				'started_at': workflow.started_at.isoformat(),
				'completed_at': workflow.completed_at.isoformat(),
				'tenant_id': workflow.tenant_id,
				'steps_completed': execution_result['steps_completed'],
				'errors': workflow.errors
			})
			
			return execution_result
			
		except Exception as e:
			# Handle workflow execution failure
			if workflow_id in self.active_workflows:
				workflow = self.active_workflows[workflow_id]
				workflow.status = WorkflowStatus.FAILED
				workflow.errors.append(str(e))
				workflow.completed_at = datetime.now(UTC)
			
			return {
				'success': False,
				'errors': [str(e)],
				'execution_time': 0,
				'steps_completed': 0
			}
	
	async def _execute_workflow_steps(self, workflow: CrossCapabilityWorkflow) -> Dict[str, Any]:
		"""Execute individual workflow steps with dependency resolution"""
		start_time = time.time()
		steps_completed = 0
		results = {}
		errors = []
		
		# Build dependency graph
		step_graph = self._build_step_dependency_graph(workflow.steps)
		
		# Execute steps in dependency order
		for step_id in step_graph:
			step = next((s for s in workflow.steps if s.id == step_id), None)
			if not step:
				continue
			
			workflow.current_step = step.id
			
			try:
				# Check step conditions
				if not self._evaluate_step_conditions(step, workflow.context, results):
					print(f"Step {step.name} conditions not met, skipping")
					continue
				
				# Prepare step input
				step_input = await self._prepare_step_input(step, workflow.context, results)
				
				# Execute step
				step_result = await self._execute_step(step, step_input)
				
				# Process step output
				results[step.id] = step_result
				
				# Apply output mapping
				self._apply_output_mapping(step, step_result, workflow.context)
				
				steps_completed += 1
				
			except Exception as e:
				error_msg = f"Step {step.name} failed: {str(e)}"
				errors.append(error_msg)
				print(error_msg)
				
				# Check if workflow should continue on error
				if step.metadata.get('continue_on_error', False):
					continue
				else:
					return {
						'success': False,
						'errors': errors,
						'execution_time': time.time() - start_time,
						'steps_completed': steps_completed,
						'failed_step': step.id,
						'results': results
					}
		
		execution_time = time.time() - start_time
		
		return {
			'success': len(errors) == 0,
			'errors': errors,
			'execution_time': execution_time,
			'steps_completed': steps_completed,
			'results': results
		}
	
	def _build_step_dependency_graph(self, steps: List[WorkflowStep]) -> List[str]:
		"""Build step execution order based on dependencies"""
		# Simple topological sort implementation
		step_map = {step.id: step for step in steps}
		visited = set()
		temp_visited = set()
		result = []
		
		def visit(step_id: str):
			if step_id in temp_visited:
				raise ValueError(f"Circular dependency detected involving step {step_id}")
			if step_id in visited:
				return
			
			temp_visited.add(step_id)
			
			step = step_map.get(step_id)
			if step:
				for dep in step.dependencies:
					if dep in step_map:
						visit(dep)
			
			temp_visited.remove(step_id)
			visited.add(step_id)
			result.append(step_id)
		
		for step in steps:
			if step.id not in visited:
				visit(step.id)
		
		return result
	
	def _evaluate_step_conditions(self, step: WorkflowStep, context: Dict[str, Any], results: Dict[str, Any]) -> bool:
		"""Evaluate whether step conditions are met"""
		if not step.conditions:
			return True
		
		# Simple condition evaluation - in production would use a proper expression engine
		for condition_key, condition_value in step.conditions.items():
			if condition_key.startswith("context."):
				context_key = condition_key[8:]
				if context.get(context_key) != condition_value:
					return False
			elif condition_key.startswith("results."):
				result_path = condition_key[8:].split(".")
				result_value = results
				for path_segment in result_path:
					if isinstance(result_value, dict) and path_segment in result_value:
						result_value = result_value[path_segment]
					else:
						return False
				if result_value != condition_value:
					return False
		
		return True
	
	async def _prepare_step_input(self, step: WorkflowStep, context: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
		"""Prepare input data for step execution"""
		step_input = step.input_data.copy()
		
		# Substitute context variables
		step_input = self._substitute_variables(step_input, context, "context")
		
		# Substitute results from previous steps
		step_input = self._substitute_variables(step_input, results, "results")
		
		return step_input
	
	def _substitute_variables(self, data: Any, source: Dict[str, Any], prefix: str) -> Any:
		"""Substitute variables in data structure"""
		if isinstance(data, dict):
			return {k: self._substitute_variables(v, source, prefix) for k, v in data.items()}
		elif isinstance(data, list):
			return [self._substitute_variables(item, source, prefix) for item in data]
		elif isinstance(data, str) and data.startswith(f"${{{prefix}."):
			# Extract variable path
			var_path = data[len(f"${{{prefix}."):-1].split(".")
			var_value = source
			for path_segment in var_path:
				if isinstance(var_value, dict) and path_segment in var_value:
					var_value = var_value[path_segment]
				else:
					return data  # Return original if path not found
			return var_value
		else:
			return data
	
	async def _execute_step(self, step: WorkflowStep, step_input: Dict[str, Any]) -> Any:
		"""Execute individual workflow step"""
		# Look for registered step executor
		if step.operation in self.step_executors:
			executor = self.step_executors[step.operation]
			return await executor(step, step_input)
		
		# Default execution - make API call to capability
		capability = await self._get_capability_for_step(step)
		if not capability:
			raise ValueError(f"Capability {step.capability_id} not available for step {step.name}")
		
		# Simulate API call - in production would make actual HTTP requests
		return await self._simulate_capability_call(capability, step.operation, step_input)
	
	async def _get_capability_for_step(self, step: WorkflowStep) -> Optional[EcosystemCapability]:
		"""Get capability instance for workflow step"""
		if step.capability_id in self.capability_registry.capabilities:
			capability = self.capability_registry.capabilities[step.capability_id]
			if capability.state in [CapabilityState.AVAILABLE, CapabilityState.ACTIVE]:
				return capability
		return None
	
	async def _simulate_capability_call(self, capability: EcosystemCapability, operation: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Simulate capability operation call - replace with actual API calls in production"""
		# Simulate processing time
		await asyncio.sleep(0.1)
		
		# Return mock result based on operation
		if operation == "create_tenant":
			return {
				'tenant_id': uuid7str(),
				'status': 'created',
				'endpoint': f"https://{input_data.get('subdomain', 'default')}.example.com"
			}
		elif operation == "configure_security":
			return {
				'security_profile_id': uuid7str(),
				'status': 'configured',
				'features_enabled': ['mfa', 'encryption', 'audit_logging']
			}
		elif operation == "provision_resources":
			return {
				'resource_allocation_id': uuid7str(),
				'status': 'provisioned',
				'resources': {
					'cpu': input_data.get('cpu_cores', 2),
					'memory': input_data.get('memory_gb', 4),
					'storage': input_data.get('storage_gb', 100)
				}
			}
		else:
			return {
				'operation': operation,
				'status': 'completed',
				'result': 'success'
			}
	
	def _apply_output_mapping(self, step: WorkflowStep, step_result: Any, context: Dict[str, Any]):
		"""Apply output mapping to update workflow context"""
		for output_key, context_key in step.output_mapping.items():
			if isinstance(step_result, dict) and output_key in step_result:
				# Set value in context using dot notation
				context_path = context_key.split(".")
				current_context = context
				
				for path_segment in context_path[:-1]:
					if path_segment not in current_context:
						current_context[path_segment] = {}
					current_context = current_context[path_segment]
				
				current_context[context_path[-1]] = step_result[output_key]
	
	async def start_workflow(self, template_name: str, context: Dict[str, Any] = None) -> str:
		"""Start workflow execution from template"""
		if template_name not in self.workflow_templates:
			raise ValueError(f"Workflow template {template_name} not found")
		
		template = self.workflow_templates[template_name]
		
		# Create workflow instance
		workflow_instance = CrossCapabilityWorkflow(
			name=template.name,
			description=template.description,
			version=template.version,
			tenant_id=context.get('tenant_id') if context else None,
			steps=template.steps.copy(),
			context=context or {},
			created_by=context.get('created_by', 'system') if context else 'system'
		)
		
		# Store active workflow
		self.active_workflows[workflow_instance.id] = workflow_instance
		
		# Start execution asynchronously
		asyncio.create_task(self.execute_workflow(workflow_instance.id))
		
		return workflow_instance.id


class ResourceManager:
	"""Manager for shared resources across capabilities"""
	
	def __init__(self, event_bus: EventBus):
		self.event_bus = event_bus
		self.shared_resources: Dict[str, SharedResource] = {}
		self.resource_allocations: Dict[str, Dict[str, Any]] = {}
		self.allocation_policies: Dict[str, Callable] = {}
		self.monitoring_active = False
	
	def register_allocation_policy(self, policy_name: str, policy_func: Callable):
		"""Register resource allocation policy"""
		self.allocation_policies[policy_name] = policy_func
	
	async def register_shared_resource(self, resource: SharedResource) -> bool:
		"""Register a shared resource"""
		try:
			# Validate resource
			await self._validate_shared_resource(resource)
			
			# Store resource
			self.shared_resources[resource.id] = resource
			
			# Initialize allocation tracking
			self.resource_allocations[resource.id] = {}
			
			# Publish resource registration event
			event = EcosystemEvent(
				type=EventType.RESOURCE_ALLOCATED,
				source_capability=resource.provider_capability,
				payload={
					'resource_id': resource.id,
					'resource_name': resource.name,
					'resource_type': resource.resource_type.value,
					'capacity': resource.capacity
				}
			)
			await self.event_bus.publish(event)
			
			return True
		except Exception as e:
			print(f"Failed to register shared resource {resource.name}: {e}")
			return False
	
	async def allocate_resource(
		self, 
		resource_id: str, 
		consumer_capability: str, 
		allocation_request: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Allocate shared resource to consuming capability"""
		if resource_id not in self.shared_resources:
			raise ValueError(f"Resource {resource_id} not found")
		
		resource = self.shared_resources[resource_id]
		
		# Check allocation policy
		policy_func = self.allocation_policies.get(resource.allocation_policy)
		if policy_func:
			allocation_decision = await policy_func(resource, consumer_capability, allocation_request)
			if not allocation_decision['approved']:
				raise ValueError(f"Resource allocation denied: {allocation_decision['reason']}")
		
		# Allocate resource
		allocation_id = uuid7str()
		self.resource_allocations[resource_id][allocation_id] = {
			'consumer_capability': consumer_capability,
			'allocation': allocation_request,
			'allocated_at': datetime.now(UTC).isoformat(),
			'status': 'active'
		}
		
		# Update resource usage
		await self._update_resource_usage(resource_id)
		
		# Publish allocation event
		event = EcosystemEvent(
			type=EventType.RESOURCE_ALLOCATED,
			source_capability=consumer_capability,
			payload={
				'resource_id': resource_id,
				'allocation_id': allocation_id,
				'allocation': allocation_request
			}
		)
		await self.event_bus.publish(event)
		
		return {
			'allocation_id': allocation_id,
			'resource_id': resource_id,
			'status': 'allocated',
			'allocation': allocation_request
		}
	
	async def release_resource(self, resource_id: str, allocation_id: str) -> bool:
		"""Release allocated resource"""
		try:
			if resource_id in self.resource_allocations and allocation_id in self.resource_allocations[resource_id]:
				allocation = self.resource_allocations[resource_id][allocation_id]
				allocation['status'] = 'released'
				allocation['released_at'] = datetime.now(UTC).isoformat()
				
				# Update resource usage
				await self._update_resource_usage(resource_id)
				
				# Publish release event
				event = EcosystemEvent(
					type=EventType.RESOURCE_RELEASED,
					source_capability=allocation['consumer_capability'],
					payload={
						'resource_id': resource_id,
						'allocation_id': allocation_id
					}
				)
				await self.event_bus.publish(event)
				
				return True
			return False
		except Exception as e:
			print(f"Failed to release resource allocation {allocation_id}: {e}")
			return False
	
	async def _validate_shared_resource(self, resource: SharedResource):
		"""Validate shared resource registration"""
		if not resource.name or not resource.provider_capability:
			raise ValueError("Resource name and provider capability are required")
		
		# Check for resource name uniqueness
		for existing_resource in self.shared_resources.values():
			if existing_resource.name == resource.name and existing_resource.id != resource.id:
				raise ValueError(f"Resource name {resource.name} already exists")
	
	async def _update_resource_usage(self, resource_id: str):
		"""Update current resource usage based on active allocations"""
		if resource_id not in self.shared_resources:
			return
		
		resource = self.shared_resources[resource_id]
		allocations = self.resource_allocations.get(resource_id, {})
		
		# Calculate current usage
		total_usage = {}
		active_allocations = [alloc for alloc in allocations.values() if alloc['status'] == 'active']
		
		for allocation in active_allocations:
			for key, value in allocation['allocation'].items():
				if key in total_usage:
					total_usage[key] += value
				else:
					total_usage[key] = value
		
		resource.current_usage = total_usage
		resource.last_updated = datetime.now(UTC)


class CapabilityMarketplace:
	"""Marketplace for capability discovery and lifecycle management"""
	
	def __init__(self, event_bus: EventBus):
		self.event_bus = event_bus
		self.marketplace_entries: Dict[str, MarketplaceEntry] = {}
		self.categories: Dict[str, List[str]] = {}
		self.search_index: Dict[str, List[str]] = {}
		self.installation_cache: Dict[str, Dict[str, Any]] = {}
	
	async def publish_capability(self, entry: MarketplaceEntry) -> bool:
		"""Publish capability to marketplace"""
		try:
			# Validate entry
			await self._validate_marketplace_entry(entry)
			
			# Store entry
			self.marketplace_entries[entry.id] = entry
			
			# Update categories
			if entry.category not in self.categories:
				self.categories[entry.category] = []
			if entry.id not in self.categories[entry.category]:
				self.categories[entry.category].append(entry.id)
			
			# Update search index
			self._update_search_index(entry)
			
			return True
		except Exception as e:
			print(f"Failed to publish capability {entry.name}: {e}")
			return False
	
	async def search_capabilities(
		self, 
		query: str = None, 
		category: str = None, 
		tags: List[str] = None
	) -> List[MarketplaceEntry]:
		"""Search marketplace for capabilities"""
		matching_entries = []
		
		for entry in self.marketplace_entries.values():
			match = True
			
			# Category filter
			if category and entry.category != category:
				match = False
			
			# Tags filter
			if tags:
				if not any(tag in entry.tags for tag in tags):
					match = False
			
			# Query search
			if query:
				query_lower = query.lower()
				search_text = f"{entry.name} {entry.display_name} {entry.description}".lower()
				if query_lower not in search_text and not any(tag.lower() == query_lower for tag in entry.tags):
					match = False
			
			if match:
				matching_entries.append(entry)
		
		# Sort by rating and downloads
		matching_entries.sort(key=lambda e: (e.rating, e.downloads), reverse=True)
		
		return matching_entries
	
	async def install_capability(self, capability_id: str, installation_config: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Install capability from marketplace"""
		if capability_id not in self.marketplace_entries:
			raise ValueError(f"Capability {capability_id} not found in marketplace")
		
		entry = self.marketplace_entries[capability_id]
		
		# Check installation requirements
		installation_checks = await self._check_installation_requirements(entry)
		if not installation_checks['can_install']:
			return {
				'success': False,
				'error': 'Installation requirements not met',
				'missing_requirements': installation_checks['missing_requirements']
			}
		
		# Simulate installation process
		installation_id = uuid7str()
		installation_result = await self._simulate_capability_installation(entry, installation_config or {})
		
		# Update download count
		entry.downloads += 1
		
		# Cache installation result
		self.installation_cache[installation_id] = {
			'capability_id': capability_id,
			'status': 'installed' if installation_result['success'] else 'failed',
			'installed_at': datetime.now(UTC).isoformat(),
			'config': installation_config
		}
		
		return {
			'installation_id': installation_id,
			'success': installation_result['success'],
			'capability_id': capability_id,
			'installed_version': entry.version
		}
	
	def _update_search_index(self, entry: MarketplaceEntry):
		"""Update search index for capability discovery"""
		search_terms = [
			entry.name.lower(),
			entry.display_name.lower(),
			entry.category.lower(),
			*[tag.lower() for tag in entry.tags],
			*entry.description.lower().split()
		]
		
		for term in search_terms:
			if term not in self.search_index:
				self.search_index[term] = []
			if entry.id not in self.search_index[term]:
				self.search_index[term].append(entry.id)
	
	async def _validate_marketplace_entry(self, entry: MarketplaceEntry):
		"""Validate marketplace entry"""
		if not entry.name or not entry.version or not entry.publisher:
			raise ValueError("Name, version, and publisher are required")
		
		# Check for duplicate entries
		for existing_entry in self.marketplace_entries.values():
			if (existing_entry.name == entry.name and 
				existing_entry.publisher == entry.publisher and
				existing_entry.id != entry.id):
				raise ValueError(f"Capability {entry.name} from {entry.publisher} already exists")
	
	async def _check_installation_requirements(self, entry: MarketplaceEntry) -> Dict[str, Any]:
		"""Check if installation requirements are met"""
		missing_requirements = []
		
		# Simulate requirement checking - in production would check actual dependencies
		for requirement in entry.installation_requirements:
			if requirement.startswith("python>="):
				# Would check Python version
				continue
			elif requirement.startswith("memory>="):
				# Would check available memory
				continue
			else:
				# Assume requirement is met for simulation
				continue
		
		return {
			'can_install': len(missing_requirements) == 0,
			'missing_requirements': missing_requirements
		}
	
	async def _simulate_capability_installation(self, entry: MarketplaceEntry, config: Dict[str, Any]) -> Dict[str, Any]:
		"""Simulate capability installation process"""
		# Simulate installation time
		await asyncio.sleep(1)
		
		# Simulate success/failure based on entry complexity
		success_rate = 0.95 if len(entry.installation_requirements) < 5 else 0.85
		
		import random
		success = random.random() < success_rate
		
		return {
			'success': success,
			'installation_time': 1.0,
			'installed_components': ['core', 'api', 'ui'] if success else [],
			'error': None if success else 'Simulated installation failure'
		}


# Main Integration Manager

class APGEcosystemIntegrationManager:
	"""Main manager for APG ecosystem integration"""
	
	def __init__(self):
		self.event_bus = EventBus()
		self.capability_registry = CapabilityRegistry(self.event_bus)
		self.workflow_orchestrator = WorkflowOrchestrator(self.capability_registry, self.event_bus)
		self.resource_manager = ResourceManager(self.event_bus)
		self.marketplace = CapabilityMarketplace(self.event_bus)
		self.compositions: Dict[str, CapabilityComposition] = {}
		self.integration_metrics = {
			'active_capabilities': 0,
			'active_workflows': 0,
			'shared_resources': 0,
			'marketplace_entries': 0,
			'events_processed': 0
		}
		self.started = False
		
		# Register default step executors
		self._register_default_step_executors()
		
		# Register default resource policies
		self._register_default_resource_policies()
		
		# Setup event handlers
		self._setup_event_handlers()
	
	async def start(self):
		"""Start the integration manager"""
		if self.started:
			return
		
		print("🚀 Starting APG Ecosystem Integration Manager...")
		
		# Register MTen capability
		await self._register_mten_capability()
		
		# Create default workflow templates
		await self._create_default_workflows()
		
		# Setup default shared resources
		await self._setup_default_shared_resources()
		
		# Populate marketplace with sample capabilities
		await self._populate_sample_marketplace()
		
		self.started = True
		print("✅ APG Ecosystem Integration Manager started successfully")
	
	async def stop(self):
		"""Stop the integration manager"""
		if not self.started:
			return
		
		print("🛑 Stopping APG Ecosystem Integration Manager...")
		
		# Cancel active workflows
		for workflow_id in list(self.workflow_orchestrator.active_workflows.keys()):
			await self.cancel_workflow(workflow_id)
		
		# Unregister capabilities
		for capability_id in list(self.capability_registry.capabilities.keys()):
			await self.capability_registry.unregister_capability(capability_id)
		
		self.started = False
		print("✅ APG Ecosystem Integration Manager stopped")
	
	async def get_integration_status(self) -> Dict[str, Any]:
		"""Get current integration status"""
		self.integration_metrics.update({
			'active_capabilities': len([c for c in self.capability_registry.capabilities.values() 
									   if c.state in [CapabilityState.AVAILABLE, CapabilityState.ACTIVE]]),
			'active_workflows': len([w for w in self.workflow_orchestrator.active_workflows.values() 
								   if w.status == WorkflowStatus.RUNNING]),
			'shared_resources': len(self.resource_manager.shared_resources),
			'marketplace_entries': len(self.marketplace.marketplace_entries),
			'events_processed': self.event_bus.metrics['events_delivered']
		})
		
		return {
			'status': 'running' if self.started else 'stopped',
			'metrics': self.integration_metrics,
			'capabilities': list(self.capability_registry.capabilities.keys()),
			'active_workflows': list(self.workflow_orchestrator.active_workflows.keys()),
			'resource_categories': list(set(r.resource_type.value for r in self.resource_manager.shared_resources.values())),
			'marketplace_categories': list(self.marketplace.categories.keys())
		}
	
	async def create_capability_composition(self, composition: CapabilityComposition) -> bool:
		"""Create a new capability composition"""
		try:
			# Validate composition
			await self._validate_capability_composition(composition)
			
			# Store composition
			self.compositions[composition.id] = composition
			
			return True
		except Exception as e:
			print(f"Failed to create capability composition {composition.name}: {e}")
			return False
	
	async def execute_tenant_provisioning_workflow(self, tenant_config: Dict[str, Any]) -> str:
		"""Execute comprehensive tenant provisioning workflow"""
		context = {
			'tenant_id': uuid7str(),
			'tenant_name': tenant_config.get('name', 'Default Tenant'),
			'subdomain': tenant_config.get('subdomain'),
			'tier': tenant_config.get('tier', 'basic'),
			'features': tenant_config.get('features', []),
			'resources': tenant_config.get('resources', {}),
			'security_profile': tenant_config.get('security_profile', 'standard'),
			'created_by': tenant_config.get('created_by', 'system')
		}
		
		workflow_id = await self.workflow_orchestrator.start_workflow(
			'comprehensive_tenant_provisioning',
			context
		)
		
		return workflow_id
	
	async def cancel_workflow(self, workflow_id: str) -> bool:
		"""Cancel active workflow"""
		if workflow_id in self.workflow_orchestrator.active_workflows:
			workflow = self.workflow_orchestrator.active_workflows[workflow_id]
			workflow.status = WorkflowStatus.CANCELLED
			workflow.completed_at = datetime.now(UTC)
			return True
		return False
	
	def _register_default_step_executors(self):
		"""Register default step executors for common operations"""
		
		async def mten_create_tenant_executor(step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
			"""MTen tenant creation step executor"""
			return {
				'tenant_id': input_data.get('tenant_id', uuid7str()),
				'status': 'created',
				'endpoint': f"https://{input_data.get('subdomain', 'default')}.mten.datacraft.co.ke",
				'database_name': f"tenant_{input_data.get('tenant_id', '').replace('-', '_')}",
				'created_at': datetime.now(UTC).isoformat()
			}
		
		async def auth_setup_executor(step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
			"""Authentication setup step executor"""
			return {
				'auth_provider_id': uuid7str(),
				'status': 'configured',
				'authentication_methods': ['password', 'oauth', 'saml'],
				'mfa_enabled': input_data.get('enable_mfa', True),
				'session_timeout': input_data.get('session_timeout', 3600)
			}
		
		async def security_configure_executor(step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
			"""Security configuration step executor"""
			return {
				'security_profile_id': uuid7str(),
				'status': 'configured',
				'encryption_enabled': True,
				'audit_logging_enabled': True,
				'compliance_profile': input_data.get('compliance_profile', 'standard'),
				'firewall_rules': ['default_allow_https', 'default_deny_all']
			}
		
		self.workflow_orchestrator.register_step_executor('mten_create_tenant', mten_create_tenant_executor)
		self.workflow_orchestrator.register_step_executor('auth_setup', auth_setup_executor)
		self.workflow_orchestrator.register_step_executor('security_configure', security_configure_executor)
	
	def _register_default_resource_policies(self):
		"""Register default resource allocation policies"""
		
		async def fair_share_policy(resource: SharedResource, consumer: str, request: Dict[str, Any]) -> Dict[str, Any]:
			"""Fair share resource allocation policy"""
			# Simple policy - allow if within capacity limits
			current_usage = resource.current_usage
			requested = request
			capacity = resource.capacity
			
			for key, requested_amount in requested.items():
				if key in capacity:
					current_amount = current_usage.get(key, 0)
					if current_amount + requested_amount > capacity[key]:
						return {
							'approved': False,
							'reason': f'Insufficient {key} capacity'
						}
			
			return {'approved': True, 'reason': 'Within capacity limits'}
		
		self.resource_manager.register_allocation_policy('fair_share', fair_share_policy)
	
	def _setup_event_handlers(self):
		"""Setup event handlers for ecosystem events"""
		
		async def handle_capability_registered(event: EcosystemEvent):
			"""Handle capability registration events"""
			print(f"📢 Capability registered: {event.payload.get('capability_name')} v{event.payload.get('version')}")
		
		async def handle_workflow_events(event: EcosystemEvent):
			"""Handle workflow lifecycle events"""
			if event.type == EventType.WORKFLOW_STARTED:
				print(f"🚀 Workflow started: {event.payload.get('workflow_name')} ({event.payload.get('workflow_id')})")
			elif event.type == EventType.WORKFLOW_COMPLETED:
				print(f"✅ Workflow completed: {event.payload.get('workflow_id')} in {event.payload.get('execution_time'):.2f}s")
			elif event.type == EventType.WORKFLOW_FAILED:
				print(f"❌ Workflow failed: {event.payload.get('workflow_id')} - {event.payload.get('errors')}")
		
		async def handle_tenant_events(event: EcosystemEvent):
			"""Handle tenant lifecycle events"""
			if event.type == EventType.TENANT_PROVISIONED:
				print(f"🏠 Tenant provisioned: {event.payload.get('tenant_name')} ({event.payload.get('tenant_id')})")
		
		# Subscribe to events
		self.event_bus.subscribe(EventType.CAPABILITY_REGISTERED, handle_capability_registered)
		self.event_bus.subscribe(EventType.WORKFLOW_STARTED, handle_workflow_events)
		self.event_bus.subscribe(EventType.WORKFLOW_COMPLETED, handle_workflow_events)
		self.event_bus.subscribe(EventType.WORKFLOW_FAILED, handle_workflow_events)
		self.event_bus.subscribe(EventType.TENANT_PROVISIONED, handle_tenant_events)
	
	async def _register_mten_capability(self):
		"""Register MTen as a capability in the ecosystem"""
		mten_capability = EcosystemCapability(
			name="multi-tenant-management",
			version="1.0.0",
			namespace="apg.common",
			description="Comprehensive multi-tenant management and orchestration capability",
			category="infrastructure",
			health_endpoint="/api/v1/health",
			api_endpoint="/api/v1/mten",
			supported_operations=[
				"create_tenant",
				"update_tenant", 
				"delete_tenant",
				"get_tenant_analytics",
				"optimize_tenant_performance",
				"backup_tenant_data",
				"restore_tenant_data",
				"scale_tenant_resources"
			],
			provided_interfaces=[
				"tenant_management_api",
				"analytics_api",
				"optimization_api"
			],
			resource_requirements={
				"cpu_cores": 2,
				"memory_gb": 4,
				"storage_gb": 100
			},
			configuration={
				"max_tenants": 1000,
				"backup_retention_days": 30,
				"analytics_retention_days": 90
			}
		)
		
		await self.capability_registry.register_capability(mten_capability)
	
	async def _create_default_workflows(self):
		"""Create default workflow templates"""
		
		# Comprehensive tenant provisioning workflow
		tenant_provisioning_workflow = CrossCapabilityWorkflow(
			name="comprehensive_tenant_provisioning",
			description="End-to-end tenant provisioning with security, analytics, and monitoring",
			version="1.0.0",
			steps=[
				WorkflowStep(
					name="Create Tenant Infrastructure",
					capability_id="multi-tenant-management",
					operation="mten_create_tenant",
					input_data={
						"tenant_id": "${context.tenant_id}",
						"tenant_name": "${context.tenant_name}",
						"subdomain": "${context.subdomain}",
						"tier": "${context.tier}"
					},
					output_mapping={
						"tenant_id": "tenant.id",
						"endpoint": "tenant.endpoint",
						"database_name": "tenant.database"
					}
				),
				WorkflowStep(
					name="Setup Authentication",
					capability_id="auth_rbac",
					operation="auth_setup",
					input_data={
						"tenant_id": "${context.tenant_id}",
						"enable_mfa": True,
						"session_timeout": 3600
					},
					dependencies=["Create Tenant Infrastructure"],
					output_mapping={
						"auth_provider_id": "tenant.auth_provider_id"
					}
				),
				WorkflowStep(
					name="Configure Security",
					capability_id="blockchain_security",
					operation="security_configure",
					input_data={
						"tenant_id": "${context.tenant_id}",
						"security_profile": "${context.security_profile}",
						"compliance_profile": "enterprise"
					},
					dependencies=["Setup Authentication"],
					output_mapping={
						"security_profile_id": "tenant.security_profile_id"
					}
				),
				WorkflowStep(
					name="Provision Cloud Resources", 
					capability_id="multi-tenant-management",
					operation="provision_resources",
					input_data={
						"tenant_id": "${context.tenant_id}",
						"resources": "${context.resources}",
						"tier": "${context.tier}"
					},
					dependencies=["Configure Security"],
					output_mapping={
						"resource_allocation_id": "tenant.resource_allocation_id"
					}
				),
				WorkflowStep(
					name="Setup Monitoring",
					capability_id="ai_orchestration",
					operation="setup_monitoring",
					input_data={
						"tenant_id": "${context.tenant_id}",
						"monitoring_level": "comprehensive"
					},
					dependencies=["Provision Cloud Resources"],
					output_mapping={
						"monitoring_config_id": "tenant.monitoring_config_id"
					}
				),
				WorkflowStep(
					name="Initialize Analytics",
					capability_id="multi-tenant-management",
					operation="initialize_analytics",
					input_data={
						"tenant_id": "${context.tenant_id}",
						"analytics_features": "${context.features}"
					},
					dependencies=["Setup Monitoring"],
					output_mapping={
						"analytics_dashboard_url": "tenant.analytics_url"
					}
				)
			],
			created_by="system"
		)
		
		await self.workflow_orchestrator.create_workflow_template(tenant_provisioning_workflow)
	
	async def _setup_default_shared_resources(self):
		"""Setup default shared resources"""
		
		# Database connection pool
		db_pool_resource = SharedResource(
			name="postgresql_connection_pool",
			resource_type=ResourceType.DATABASE,
			provider_capability="multi-tenant-management",
			configuration={
				"host": "localhost",
				"port": 5432,
				"pool_size": 100,
				"database": "mten_shared"
			},
			capacity={
				"connections": 100,
				"queries_per_second": 1000
			},
			allocation_policy="fair_share"
		)
		
		# Redis cache
		cache_resource = SharedResource(
			name="redis_cache_cluster",
			resource_type=ResourceType.CACHE,
			provider_capability="multi-tenant-management",
			configuration={
				"host": "localhost",
				"port": 6379,
				"cluster_size": 3
			},
			capacity={
				"memory_mb": 1024,
				"operations_per_second": 10000
			},
			allocation_policy="fair_share"
		)
		
		# Message queue
		queue_resource = SharedResource(
			name="event_message_queue",
			resource_type=ResourceType.QUEUE,
			provider_capability="event_streaming_bus",
			configuration={
				"runtime": "bytewax",
				"workers": 12,
				"replication_factor": 3
			},
			capacity={
				"messages_per_second": 5000,
				"storage_gb": 50
			},
			allocation_policy="fair_share"
		)
		
		await self.resource_manager.register_shared_resource(db_pool_resource)
		await self.resource_manager.register_shared_resource(cache_resource)
		await self.resource_manager.register_shared_resource(queue_resource)
	
	async def _populate_sample_marketplace(self):
		"""Populate marketplace with sample capabilities"""
		
		sample_capabilities = [
			MarketplaceEntry(
				capability_id="auth_rbac_enhanced",
				name="enhanced-auth-rbac",
				display_name="Enhanced Authentication & RBAC",
				description="Advanced authentication with role-based access control, multi-factor authentication, and compliance features",
				category="security",
				tags=["authentication", "authorization", "security", "compliance"],
				version="2.1.0",
				publisher="Datacraft",
				license="MIT",
				pricing_model="enterprise",
				installation_requirements=["python>=3.9", "memory>=2GB"],
				compatibility={"apg_version": ">=1.0.0"},
				rating=4.8,
				downloads=1250
			),
			MarketplaceEntry(
				capability_id="ai_ml_platform",
				name="ai-ml-platform",
				display_name="AI/ML Platform Integration",
				description="Comprehensive AI/ML platform with model management, training pipelines, and inference APIs",
				category="ai_ml",
				tags=["ai", "machine-learning", "models", "training", "inference"],
				version="1.5.2",
				publisher="Datacraft",
				license="Apache-2.0",
				pricing_model="usage_based",
				installation_requirements=["python>=3.8", "gpu_memory>=4GB", "storage>=10GB"],
				compatibility={"apg_version": ">=1.0.0", "cuda": ">=11.0"},
				rating=4.6,
				downloads=890
			),
			MarketplaceEntry(
				capability_id="advanced_analytics",
				name="advanced-analytics",
				display_name="Advanced Analytics Engine",
				description="Real-time analytics with predictive modeling, anomaly detection, and interactive dashboards",
				category="analytics",
				tags=["analytics", "real-time", "predictive", "dashboard", "reporting"],
				version="3.0.1",
				publisher="Datacraft",
				license="Commercial",
				pricing_model="subscription",
				installation_requirements=["python>=3.9", "memory>=4GB"],
				compatibility={"apg_version": ">=1.0.0"},
				rating=4.9,
				downloads=2100
			)
		]
		
		for capability in sample_capabilities:
			await self.marketplace.publish_capability(capability)
	
	async def _validate_capability_composition(self, composition: CapabilityComposition):
		"""Validate capability composition"""
		# Check that component capabilities exist
		for capability_id in composition.component_capabilities:
			if capability_id not in self.capability_registry.capabilities:
				raise ValueError(f"Component capability {capability_id} not found")
		
		# Validate composition name uniqueness
		for existing_comp in self.compositions.values():
			if existing_comp.name == composition.name and existing_comp.id != composition.id:
				raise ValueError(f"Composition name {composition.name} already exists")


# Validation Functions

async def validate_ecosystem_integration() -> bool:
	"""Validate APG ecosystem integration functionality"""
	print("🔍 Validating APG Ecosystem Integration...")
	
	try:
		# Initialize integration manager
		manager = APGEcosystemIntegrationManager()
		await manager.start()
		
		# Test capability registration
		test_capability = EcosystemCapability(
			name="test-capability",
			version="1.0.0", 
			namespace="test",
			description="Test capability for validation",
			category="test",
			health_endpoint="/health",
			api_endpoint="/api",
			supported_operations=["test_operation"]
		)
		
		registration_success = await manager.capability_registry.register_capability(test_capability)
		if not registration_success:
			print("❌ Capability registration failed")
			return False
		
		# Test capability discovery
		discovered = await manager.capability_registry.discover_capabilities(category="test")
		if len(discovered) == 0:
			print("❌ Capability discovery failed")
			return False
		
		# Test workflow execution
		workflow_id = await manager.execute_tenant_provisioning_workflow({
			'name': 'Test Tenant',
			'subdomain': 'test-tenant',
			'tier': 'basic',
			'created_by': 'validation_test'
		})
		
		if not workflow_id:
			print("❌ Workflow execution failed")
			return False
		
		# Wait for workflow to complete
		await asyncio.sleep(2)
		
		# Test resource allocation
		test_resource = SharedResource(
			name="test-resource",
			resource_type=ResourceType.COMPUTE,
			provider_capability="test-capability",
			capacity={"cpu_cores": 4, "memory_gb": 8},
			allocation_policy="fair_share"
		)
		
		resource_success = await manager.resource_manager.register_shared_resource(test_resource)
		if not resource_success:
			print("❌ Resource registration failed")
			return False
		
		# Test marketplace functionality
		capabilities = await manager.marketplace.search_capabilities(category="security")
		if len(capabilities) == 0:
			print("❌ Marketplace search failed")
			return False
		
		# Get integration status
		status = await manager.get_integration_status()
		if status['status'] != 'running':
			print("❌ Integration status check failed")
			return False
		
		await manager.stop()
		
		print("✅ APG Ecosystem Integration validation passed")
		return True
		
	except Exception as e:
		print(f"❌ APG Ecosystem Integration validation failed: {e}")
		return False


if __name__ == "__main__":
	# Run validation
	success = asyncio.run(validate_ecosystem_integration())
	exit(0 if success else 1)
