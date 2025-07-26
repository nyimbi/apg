"""
APG Event Streaming Bus - APG Platform Integration

Integration layer for connecting the Event Streaming Bus with the broader APG
platform ecosystem, including capability registry, service mesh, and
cross-capability event orchestration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from uuid import uuid4

from .models import (
	ESEvent, ESStream, ESSubscription, EventType, EventStatus,
	EventConfig, StreamConfig, SubscriptionConfig
)
from .service import (
	EventStreamingService, EventPublishingService, EventConsumptionService
)

# =============================================================================
# APG Integration Models
# =============================================================================

@dataclass
class APGCapabilityInfo:
	"""Information about an APG capability."""
	capability_id: str
	capability_name: str
	capability_type: str
	version: str
	endpoints: Dict[str, str]
	event_patterns: List[str]
	dependencies: List[str]
	status: str = "active"
	last_heartbeat: Optional[datetime] = None

@dataclass
class EventRoutingRule:
	"""Rule for routing events between capabilities."""
	rule_id: str
	source_pattern: str
	target_capabilities: List[str]
	event_type_patterns: List[str]
	transformation_config: Optional[Dict[str, Any]] = None
	conditions: Optional[Dict[str, Any]] = None
	priority: int = 100
	is_active: bool = True

@dataclass
class CrossCapabilityWorkflow:
	"""Definition of a cross-capability workflow."""
	workflow_id: str
	workflow_name: str
	trigger_events: List[str]
	steps: List[Dict[str, Any]]
	compensation_steps: List[Dict[str, Any]] = field(default_factory=list)
	timeout_seconds: int = 300
	max_retries: int = 3
	is_active: bool = True

@dataclass
class EventCompositionPattern:
	"""Pattern for composing complex business events."""
	pattern_id: str
	pattern_name: str
	event_inputs: List[str]
	composition_logic: Dict[str, Any]
	output_event_type: str
	window_duration_ms: int = 30000
	min_events: int = 1
	max_events: int = 100

# =============================================================================
# APG Event Streaming Integration Service
# =============================================================================

class APGEventStreamingIntegration:
	"""
	Main integration service connecting Event Streaming Bus with APG platform.
	"""
	
	def __init__(
		self,
		event_streaming_service: EventStreamingService,
		publishing_service: EventPublishingService,
		consumption_service: EventConsumptionService,
		capability_registry_url: str = "http://capability-registry:8080",
		service_mesh_url: str = "http://service-mesh:8080"
	):
		self.event_streaming_service = event_streaming_service
		self.publishing_service = publishing_service
		self.consumption_service = consumption_service
		self.capability_registry_url = capability_registry_url
		self.service_mesh_url = service_mesh_url
		
		# Internal state
		self.registered_capabilities: Dict[str, APGCapabilityInfo] = {}
		self.routing_rules: Dict[str, EventRoutingRule] = {}
		self.active_workflows: Dict[str, CrossCapabilityWorkflow] = {}
		self.composition_patterns: Dict[str, EventCompositionPattern] = {}
		self.event_handlers: Dict[str, List[Callable]] = {}
		
		# Event correlation tracking
		self.workflow_instances: Dict[str, Dict[str, Any]] = {}
		self.pattern_buffers: Dict[str, List[Dict[str, Any]]] = {}
		
		# Background tasks
		self._background_tasks: List[asyncio.Task] = []
		self._shutdown_event = asyncio.Event()
	
	async def initialize(self):
		"""Initialize the APG integration layer."""
		await self._register_with_capability_registry()
		await self._discover_existing_capabilities()
		await self._setup_core_event_streams()
		await self._start_background_tasks()
	
	async def shutdown(self):
		"""Shutdown the integration layer gracefully."""
		self._shutdown_event.set()
		
		# Cancel background tasks
		for task in self._background_tasks:
			task.cancel()
		
		# Wait for tasks to complete
		if self._background_tasks:
			await asyncio.gather(*self._background_tasks, return_exceptions=True)
	
	# =========================================================================
	# Capability Registration and Discovery
	# =========================================================================
	
	async def register_capability(self, capability_info: APGCapabilityInfo) -> bool:
		"""Register a new capability with the Event Streaming Bus."""
		try:
			# Store capability info
			self.registered_capabilities[capability_info.capability_id] = capability_info
			
			# Create dedicated streams for the capability
			await self._create_capability_streams(capability_info)
			
			# Set up default subscriptions
			await self._create_capability_subscriptions(capability_info)
			
			# Publish capability registration event
			await self._publish_capability_event(
				"capability.registered",
				{
					"capability_id": capability_info.capability_id,
					"capability_name": capability_info.capability_name,
					"capability_type": capability_info.capability_type,
					"version": capability_info.version,
					"event_patterns": capability_info.event_patterns
				}
			)
			
			return True
			
		except Exception as e:
			print(f"Error registering capability {capability_info.capability_id}: {e}")
			return False
	
	async def unregister_capability(self, capability_id: str) -> bool:
		"""Unregister a capability from the Event Streaming Bus."""
		try:
			if capability_id not in self.registered_capabilities:
				return False
			
			capability_info = self.registered_capabilities[capability_id]
			
			# Clean up streams and subscriptions
			await self._cleanup_capability_resources(capability_id)
			
			# Publish capability unregistration event
			await self._publish_capability_event(
				"capability.unregistered",
				{
					"capability_id": capability_id,
					"capability_name": capability_info.capability_name
				}
			)
			
			# Remove from registry
			del self.registered_capabilities[capability_id]
			
			return True
			
		except Exception as e:
			print(f"Error unregistering capability {capability_id}: {e}")
			return False
	
	async def discover_capabilities(self) -> List[APGCapabilityInfo]:
		"""Discover all registered capabilities."""
		return list(self.registered_capabilities.values())
	
	async def get_capability_health(self, capability_id: str) -> Dict[str, Any]:
		"""Get health status for a specific capability."""
		if capability_id not in self.registered_capabilities:
			return {"status": "not_found"}
		
		capability = self.registered_capabilities[capability_id]
		
		# Check recent heartbeat
		if capability.last_heartbeat:
			time_since_heartbeat = datetime.utcnow() - capability.last_heartbeat
			is_healthy = time_since_heartbeat < timedelta(minutes=5)
		else:
			is_healthy = False
		
		# Get event metrics for the capability
		stream_metrics = await self._get_capability_stream_metrics(capability_id)
		
		return {
			"status": "healthy" if is_healthy else "unhealthy",
			"last_heartbeat": capability.last_heartbeat,
			"stream_metrics": stream_metrics,
			"capability_info": capability
		}
	
	# =========================================================================
	# Event Routing and Orchestration
	# =========================================================================
	
	async def add_routing_rule(self, rule: EventRoutingRule) -> bool:
		"""Add an event routing rule."""
		try:
			self.routing_rules[rule.rule_id] = rule
			
			# Publish routing rule creation event
			await self._publish_platform_event(
				"routing.rule.created",
				{
					"rule_id": rule.rule_id,
					"source_pattern": rule.source_pattern,
					"target_capabilities": rule.target_capabilities,
					"event_type_patterns": rule.event_type_patterns
				}
			)
			
			return True
			
		except Exception as e:
			print(f"Error adding routing rule {rule.rule_id}: {e}")
			return False
	
	async def remove_routing_rule(self, rule_id: str) -> bool:
		"""Remove an event routing rule."""
		if rule_id in self.routing_rules:
			del self.routing_rules[rule_id]
			
			await self._publish_platform_event(
				"routing.rule.removed",
				{"rule_id": rule_id}
			)
			
			return True
		return False
	
	async def route_event(self, event: ESEvent) -> List[str]:
		"""Route an event to target capabilities based on routing rules."""
		routed_to = []
		
		try:
			for rule in self.routing_rules.values():
				if not rule.is_active:
					continue
				
				# Check if event matches routing rule
				if await self._event_matches_rule(event, rule):
					# Route to target capabilities
					for target_capability in rule.target_capabilities:
						if target_capability in self.registered_capabilities:
							await self._route_event_to_capability(event, target_capability, rule)
							routed_to.append(target_capability)
			
			return routed_to
			
		except Exception as e:
			print(f"Error routing event {event.event_id}: {e}")
			return []
	
	# =========================================================================
	# Cross-Capability Workflows
	# =========================================================================
	
	async def register_workflow(self, workflow: CrossCapabilityWorkflow) -> bool:
		"""Register a cross-capability workflow."""
		try:
			self.active_workflows[workflow.workflow_id] = workflow
			
			# Set up subscriptions for trigger events
			await self._setup_workflow_subscriptions(workflow)
			
			await self._publish_platform_event(
				"workflow.registered",
				{
					"workflow_id": workflow.workflow_id,
					"workflow_name": workflow.workflow_name,
					"trigger_events": workflow.trigger_events
				}
			)
			
			return True
			
		except Exception as e:
			print(f"Error registering workflow {workflow.workflow_id}: {e}")
			return False
	
	async def trigger_workflow(self, workflow_id: str, trigger_event: ESEvent) -> str:
		"""Trigger a cross-capability workflow execution."""
		if workflow_id not in self.active_workflows:
			raise ValueError(f"Workflow {workflow_id} not found")
		
		workflow = self.active_workflows[workflow_id]
		instance_id = f"{workflow_id}_{uuid4().hex[:8]}"
		
		# Create workflow instance
		workflow_instance = {
			"instance_id": instance_id,
			"workflow_id": workflow_id,
			"trigger_event": trigger_event,
			"started_at": datetime.utcnow(),
			"current_step": 0,
			"status": "running",
			"context": {},
			"step_results": []
		}
		
		self.workflow_instances[instance_id] = workflow_instance
		
		# Start workflow execution
		asyncio.create_task(self._execute_workflow(instance_id))
		
		await self._publish_platform_event(
			"workflow.triggered",
			{
				"workflow_id": workflow_id,
				"instance_id": instance_id,
				"trigger_event_id": trigger_event.event_id
			}
		)
		
		return instance_id
	
	async def get_workflow_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
		"""Get status of a workflow instance."""
		return self.workflow_instances.get(instance_id)
	
	# =========================================================================
	# Event Composition Patterns
	# =========================================================================
	
	async def register_composition_pattern(self, pattern: EventCompositionPattern) -> bool:
		"""Register an event composition pattern."""
		try:
			self.composition_patterns[pattern.pattern_id] = pattern
			
			# Initialize pattern buffer
			self.pattern_buffers[pattern.pattern_id] = []
			
			# Set up subscriptions for input events
			await self._setup_pattern_subscriptions(pattern)
			
			await self._publish_platform_event(
				"pattern.registered",
				{
					"pattern_id": pattern.pattern_id,
					"pattern_name": pattern.pattern_name,
					"event_inputs": pattern.event_inputs
				}
			)
			
			return True
			
		except Exception as e:
			print(f"Error registering composition pattern {pattern.pattern_id}: {e}")
			return False
	
	async def process_pattern_event(self, pattern_id: str, event: ESEvent) -> Optional[str]:
		"""Process an event for a composition pattern."""
		if pattern_id not in self.composition_patterns:
			return None
		
		pattern = self.composition_patterns[pattern_id]
		
		# Add event to pattern buffer
		self.pattern_buffers[pattern_id].append({
			"event": event,
			"timestamp": datetime.utcnow()
		})
		
		# Clean up old events outside the window
		cutoff_time = datetime.utcnow() - timedelta(milliseconds=pattern.window_duration_ms)
		self.pattern_buffers[pattern_id] = [
			item for item in self.pattern_buffers[pattern_id]
			if item["timestamp"] > cutoff_time
		]
		
		# Check if pattern is complete
		if await self._is_pattern_complete(pattern_id):
			return await self._compose_pattern_event(pattern_id)
		
		return None
	
	# =========================================================================
	# Event Handlers and Hooks
	# =========================================================================
	
	def add_event_handler(self, event_type: str, handler: Callable[[ESEvent], None]):
		"""Add a custom event handler."""
		if event_type not in self.event_handlers:
			self.event_handlers[event_type] = []
		self.event_handlers[event_type].append(handler)
	
	def remove_event_handler(self, event_type: str, handler: Callable[[ESEvent], None]):
		"""Remove a custom event handler."""
		if event_type in self.event_handlers:
			try:
				self.event_handlers[event_type].remove(handler)
			except ValueError:
				pass
	
	async def handle_event(self, event: ESEvent):
		"""Process an event through registered handlers."""
		# Call specific handlers for this event type
		if event.event_type in self.event_handlers:
			for handler in self.event_handlers[event.event_type]:
				try:
					await handler(event)
				except Exception as e:
					print(f"Error in event handler for {event.event_type}: {e}")
		
		# Call wildcard handlers
		if "*" in self.event_handlers:
			for handler in self.event_handlers["*"]:
				try:
					await handler(event)
				except Exception as e:
					print(f"Error in wildcard event handler: {e}")
		
		# Process routing rules
		await self.route_event(event)
		
		# Process composition patterns
		for pattern_id in self.composition_patterns:
			if await self._event_matches_pattern(event, pattern_id):
				await self.process_pattern_event(pattern_id, event)
		
		# Check for workflow triggers
		for workflow in self.active_workflows.values():
			if event.event_type in workflow.trigger_events:
				await self.trigger_workflow(workflow.workflow_id, event)
	
	# =========================================================================
	# Multi-tenant Event Isolation
	# =========================================================================
	
	async def ensure_tenant_isolation(self, event: ESEvent, requesting_tenant: str) -> bool:
		"""Ensure proper tenant isolation for event access."""
		return event.tenant_id == requesting_tenant
	
	async def get_tenant_streams(self, tenant_id: str) -> List[str]:
		"""Get all streams accessible to a tenant."""
		# In production, this would query the database with tenant filters
		tenant_streams = []
		for capability in self.registered_capabilities.values():
			if self._capability_accessible_to_tenant(capability, tenant_id):
				stream_id = f"{capability.capability_id}_events"
				tenant_streams.append(stream_id)
		
		return tenant_streams
	
	# =========================================================================
	# Private Helper Methods
	# =========================================================================
	
	async def _register_with_capability_registry(self):
		"""Register Event Streaming Bus with the capability registry."""
		registration_data = {
			"capability_id": "event_streaming_bus",
			"capability_name": "Event Streaming Bus",
			"capability_type": "composition_orchestration",
			"version": "1.0.0",
			"endpoints": {
				"api": "/api/v1",
				"websocket": "/ws",
				"health": "/health"
			},
			"provides": [
				"event_streaming",
				"event_sourcing",
				"real_time_messaging",
				"stream_processing"
			],
			"requires": [
				"capability_registry",
				"api_service_mesh"
			]
		}
		
		# In production, make HTTP request to capability registry
		print(f"Registering with capability registry: {registration_data}")
	
	async def _discover_existing_capabilities(self):
		"""Discover existing capabilities from the registry."""
		# In production, fetch from capability registry API
		pass
	
	async def _setup_core_event_streams(self):
		"""Set up core platform event streams."""
		core_streams = [
			{
				"stream_name": "apg.platform.events",
				"topic_name": "apg-platform-events",
				"description": "Core platform events and notifications",
				"event_category": EventType.SYSTEM_EVENT.value
			},
			{
				"stream_name": "apg.capability.events",
				"topic_name": "apg-capability-events", 
				"description": "Capability lifecycle and management events",
				"event_category": EventType.SYSTEM_EVENT.value
			},
			{
				"stream_name": "apg.integration.events",
				"topic_name": "apg-integration-events",
				"description": "Cross-capability integration events",
				"event_category": EventType.INTEGRATION_EVENT.value
			}
		]
		
		for stream_config in core_streams:
			try:
				config = StreamConfig(**stream_config, source_capability="event_streaming_bus")
				await self.event_streaming_service.create_stream(
					config=config,
					tenant_id="platform",
					created_by="system"
				)
			except Exception as e:
				print(f"Error creating core stream {stream_config['stream_name']}: {e}")
	
	async def _start_background_tasks(self):
		"""Start background tasks for integration management."""
		tasks = [
			self._heartbeat_monitor(),
			self._workflow_timeout_monitor(),
			self._pattern_window_cleanup(),
			self._capability_health_checker()
		]
		
		for task_coro in tasks:
			task = asyncio.create_task(task_coro)
			self._background_tasks.append(task)
	
	async def _heartbeat_monitor(self):
		"""Monitor capability heartbeats."""
		while not self._shutdown_event.is_set():
			try:
				# Check heartbeats and update capability status
				for capability_id, capability in self.registered_capabilities.items():
					if capability.last_heartbeat:
						time_since_heartbeat = datetime.utcnow() - capability.last_heartbeat
						if time_since_heartbeat > timedelta(minutes=5):
							capability.status = "unhealthy"
							await self._publish_capability_event(
								"capability.unhealthy",
								{"capability_id": capability_id}
							)
				
				await asyncio.sleep(60)  # Check every minute
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Error in heartbeat monitor: {e}")
				await asyncio.sleep(60)
	
	async def _workflow_timeout_monitor(self):
		"""Monitor workflow timeouts."""
		while not self._shutdown_event.is_set():
			try:
				current_time = datetime.utcnow()
				timed_out_instances = []
				
				for instance_id, instance in self.workflow_instances.items():
					if instance["status"] == "running":
						workflow_id = instance["workflow_id"]
						workflow = self.active_workflows[workflow_id]
						
						elapsed = current_time - instance["started_at"]
						if elapsed.total_seconds() > workflow.timeout_seconds:
							timed_out_instances.append(instance_id)
				
				# Handle timed out workflows
				for instance_id in timed_out_instances:
					await self._timeout_workflow(instance_id)
				
				await asyncio.sleep(30)  # Check every 30 seconds
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Error in workflow timeout monitor: {e}")
				await asyncio.sleep(30)
	
	async def _pattern_window_cleanup(self):
		"""Clean up expired events from pattern buffers."""
		while not self._shutdown_event.is_set():
			try:
				current_time = datetime.utcnow()
				
				for pattern_id, pattern in self.composition_patterns.items():
					cutoff_time = current_time - timedelta(milliseconds=pattern.window_duration_ms)
					
					# Clean up old events
					self.pattern_buffers[pattern_id] = [
						item for item in self.pattern_buffers[pattern_id]
						if item["timestamp"] > cutoff_time
					]
				
				await asyncio.sleep(60)  # Clean up every minute
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Error in pattern window cleanup: {e}")
				await asyncio.sleep(60)
	
	async def _capability_health_checker(self):
		"""Periodically check capability health."""
		while not self._shutdown_event.is_set():
			try:
				for capability_id in list(self.registered_capabilities.keys()):
					health = await self.get_capability_health(capability_id)
					
					if health["status"] == "unhealthy":
						await self._publish_capability_event(
							"capability.health.degraded",
							{
								"capability_id": capability_id,
								"health_status": health
							}
						)
				
				await asyncio.sleep(300)  # Check every 5 minutes
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Error in capability health checker: {e}")
				await asyncio.sleep(300)
	
	async def _publish_capability_event(self, event_type: str, payload: Dict[str, Any]):
		"""Publish a capability-related event."""
		config = EventConfig(
			event_type=event_type,
			source_capability="event_streaming_bus",
			aggregate_id="capability_registry",
			aggregate_type="CapabilityRegistry"
		)
		
		await self.publishing_service.publish_event(
			event_config=config,
			payload=payload,
			stream_id="apg.capability.events",
			tenant_id="platform",
			user_id="system"
		)
	
	async def _publish_platform_event(self, event_type: str, payload: Dict[str, Any]):
		"""Publish a platform-level event."""
		config = EventConfig(
			event_type=event_type,
			source_capability="event_streaming_bus",
			aggregate_id="platform",
			aggregate_type="Platform"
		)
		
		await self.publishing_service.publish_event(
			event_config=config,
			payload=payload,
			stream_id="apg.platform.events",
			tenant_id="platform",
			user_id="system"
		)
	
	async def _create_capability_streams(self, capability_info: APGCapabilityInfo):
		"""Create streams for a new capability."""
		streams_to_create = [
			{
				"stream_name": f"{capability_info.capability_id}.events",
				"topic_name": f"apg-{capability_info.capability_id}-events",
				"description": f"Events from {capability_info.capability_name}",
				"source_capability": capability_info.capability_id,
				"event_category": EventType.DOMAIN_EVENT.value
			},
			{
				"stream_name": f"{capability_info.capability_id}.notifications",
				"topic_name": f"apg-{capability_info.capability_id}-notifications",
				"description": f"Notifications from {capability_info.capability_name}",
				"source_capability": capability_info.capability_id,
				"event_category": EventType.NOTIFICATION_EVENT.value
			}
		]
		
		for stream_config in streams_to_create:
			try:
				config = StreamConfig(**stream_config)
				await self.event_streaming_service.create_stream(
					config=config,
					tenant_id="platform",
					created_by="system"
				)
			except Exception as e:
				print(f"Error creating stream for capability {capability_info.capability_id}: {e}")
	
	async def _create_capability_subscriptions(self, capability_info: APGCapabilityInfo):
		"""Create default subscriptions for a capability."""
		# Create subscription for capability's own events
		subscription_config = SubscriptionConfig(
			subscription_name=f"{capability_info.capability_id}_self_subscription",
			subscription_description=f"Default subscription for {capability_info.capability_name}",
			stream_id=f"{capability_info.capability_id}.events",
			consumer_group_id=f"{capability_info.capability_id}_consumers",
			consumer_name=f"{capability_info.capability_id}_default_consumer",
			event_type_patterns=capability_info.event_patterns
		)
		
		try:
			await self.consumption_service.create_subscription(
				config=subscription_config,
				tenant_id="platform",
				created_by="system"
			)
		except Exception as e:
			print(f"Error creating subscription for capability {capability_info.capability_id}: {e}")
	
	def _capability_accessible_to_tenant(self, capability: APGCapabilityInfo, tenant_id: str) -> bool:
		"""Check if a capability is accessible to a tenant."""
		# In production, implement proper tenant access control
		return True  # For now, all capabilities are accessible to all tenants
	
	async def _event_matches_rule(self, event: ESEvent, rule: EventRoutingRule) -> bool:
		"""Check if an event matches a routing rule."""
		# Check source pattern
		if not self._matches_pattern(event.source_capability, rule.source_pattern):
			return False
		
		# Check event type patterns
		event_type_match = False
		for pattern in rule.event_type_patterns:
			if self._matches_pattern(event.event_type, pattern):
				event_type_match = True
				break
		
		if not event_type_match:
			return False
		
		# Check additional conditions
		if rule.conditions:
			return await self._evaluate_conditions(event, rule.conditions)
		
		return True
	
	def _matches_pattern(self, value: str, pattern: str) -> bool:
		"""Check if a value matches a pattern (supports wildcards)."""
		import re
		
		# Convert wildcard pattern to regex
		regex_pattern = pattern.replace("*", ".*").replace("?", ".")
		return bool(re.match(f"^{regex_pattern}$", value))
	
	async def _evaluate_conditions(self, event: ESEvent, conditions: Dict[str, Any]) -> bool:
		"""Evaluate routing rule conditions."""
		# Implement condition evaluation logic
		# This could include payload field checks, metadata conditions, etc.
		return True  # Simplified for now

# Export main integration class
__all__ = [
	'APGEventStreamingIntegration',
	'APGCapabilityInfo',
	'EventRoutingRule',
	'CrossCapabilityWorkflow',
	'EventCompositionPattern'
]