"""
APG Multi-Factor Authentication (MFA) - Capability Composition

Advanced capability composition engine for orchestrating MFA with other APG
capabilities, providing seamless integration and intelligent workflow automation.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict

from .integration import APGIntegrationRouter
from .service import MFAService


def _log_composition_operation(operation: str, details: str = "") -> str:
	"""Log composition operations for debugging"""
	return f"[MFA Composition] {operation}: {details}"


class CompositionEventType(str, Enum):
	"""APG composition event types"""
	CAPABILITY_REGISTERED = "capability_registered"
	CAPABILITY_ACTIVATED = "capability_activated"
	CAPABILITY_DEACTIVATED = "capability_deactivated"
	WORKFLOW_STARTED = "workflow_started"
	WORKFLOW_COMPLETED = "workflow_completed"
	INTEGRATION_ESTABLISHED = "integration_established"
	EVENT_PUBLISHED = "event_published"
	EVENT_RECEIVED = "event_received"


class APGEvent(BaseModel):
	"""APG platform event model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	event_type: str
	source_capability: str
	target_capability: Optional[str] = None
	
	# Event data
	data: Dict[str, Any] = {}
	metadata: Dict[str, Any] = {}
	
	# Event routing
	broadcast: bool = False
	priority: str = "normal"  # low, normal, high, urgent
	
	# Timestamps
	created_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: Optional[datetime] = None


class CapabilityWorkflow(BaseModel):
	"""Multi-capability workflow model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	
	# Workflow definition
	trigger_events: List[str] = []
	involved_capabilities: List[str] = []
	workflow_steps: List[Dict[str, Any]] = []
	
	# Workflow state
	is_active: bool = True
	execution_count: int = 0
	success_count: int = 0
	failure_count: int = 0
	
	# Metadata
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class MFACompositionEngine:
	"""
	Advanced composition engine for orchestrating MFA capability with other
	APG capabilities, enabling intelligent workflow automation and seamless integration.
	"""
	
	def __init__(self, 
				 mfa_service: MFAService,
				 integration_router: APGIntegrationRouter):
		"""Initialize MFA composition engine"""
		self.mfa_service = mfa_service
		self.integration = integration_router
		self.logger = logging.getLogger(__name__)
		
		# Event handlers registry
		self.event_handlers: Dict[str, List[Callable]] = {}
		
		# Workflow registry
		self.workflows: Dict[str, CapabilityWorkflow] = {}
		
		# Capability integrations
		self.capability_integrations: Dict[str, Dict[str, Any]] = {}
		
		# Event publishing queue
		self.event_queue: List[APGEvent] = []
		
		# Initialize default integrations
		self._initialize_default_integrations()
		self._register_default_workflows()
		self._register_event_handlers()
	
	async def register_capability_integration(self, 
											  capability_id: str,
											  integration_config: Dict[str, Any]) -> bool:
		"""
		Register integration with another APG capability.
		
		Args:
			capability_id: ID of capability to integrate with
			integration_config: Integration configuration
		
		Returns:
			True if integration successful
		"""
		try:
			self.logger.info(_log_composition_operation(
				"register_integration", f"capability={capability_id}"
			))
			
			# Validate capability exists
			capability_status = await self.integration.call_capability(
				capability_id, "health_check", {}
			)
			
			if not capability_status.get("success", False):
				self.logger.warning(f"Capability {capability_id} not available for integration")
				return False
			
			# Store integration configuration
			self.capability_integrations[capability_id] = {
				"config": integration_config,
				"status": "active",
				"registered_at": datetime.utcnow().isoformat(),
				"last_interaction": None
			}
			
			# Establish bi-directional event flow
			await self._establish_event_flow(capability_id, integration_config)
			
			# Publish integration event
			await self.publish_event(APGEvent(
				event_type="mfa.integration.established",
				source_capability="mfa",
				data={"integrated_capability": capability_id},
				broadcast=True
			))
			
			self.logger.info(_log_composition_operation(
				"integration_registered", f"capability={capability_id}"
			))
			
			return True
			
		except Exception as e:
			self.logger.error(f"Integration registration error: {str(e)}", exc_info=True)
			return False
	
	async def execute_composed_workflow(self, 
									   workflow_id: str,
									   trigger_event: APGEvent,
									   context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Execute a multi-capability workflow triggered by an event.
		
		Args:
			workflow_id: Workflow to execute
			trigger_event: Event that triggered the workflow
			context: Execution context
		
		Returns:
			Workflow execution result
		"""
		try:
			workflow = self.workflows.get(workflow_id)
			if not workflow:
				return {"success": False, "error": "Workflow not found"}
			
			self.logger.info(_log_composition_operation(
				"execute_workflow", f"workflow={workflow_id}, trigger={trigger_event.event_type}"
			))
			
			# Initialize workflow execution context
			execution_context = {
				"workflow_id": workflow_id,
				"trigger_event": trigger_event.dict(),
				"execution_id": uuid7str(),
				"started_at": datetime.utcnow().isoformat(),
				"context": context,
				"step_results": []
			}
			
			# Execute workflow steps
			for step_index, step in enumerate(workflow.workflow_steps):
				step_result = await self._execute_workflow_step(
					step, execution_context, step_index
				)
				
				execution_context["step_results"].append(step_result)
				
				if not step_result.get("success", False):
					# Handle step failure
					failure_result = await self._handle_workflow_failure(
						workflow, execution_context, step_index, step_result
					)
					return failure_result
			
			# Workflow completed successfully
			workflow.execution_count += 1
			workflow.success_count += 1
			workflow.updated_at = datetime.utcnow()
			
			# Publish completion event
			await self.publish_event(APGEvent(
				event_type="mfa.workflow.completed",
				source_capability="mfa",
				data={
					"workflow_id": workflow_id,
					"execution_context": execution_context
				}
			))
			
			return {
				"success": True,
				"workflow_id": workflow_id,
				"execution_context": execution_context
			}
			
		except Exception as e:
			self.logger.error(f"Workflow execution error: {str(e)}", exc_info=True)
			return {"success": False, "error": str(e)}
	
	async def handle_capability_event(self, event: APGEvent) -> None:
		"""
		Handle incoming event from another APG capability.
		
		Args:
			event: Incoming event to handle
		"""
		try:
			self.logger.debug(_log_composition_operation(
				"handle_event", f"type={event.event_type}, source={event.source_capability}"
			))
			
			# Update integration last interaction
			if event.source_capability in self.capability_integrations:
				self.capability_integrations[event.source_capability]["last_interaction"] = datetime.utcnow().isoformat()
			
			# Find registered handlers for this event type
			handlers = self.event_handlers.get(event.event_type, [])
			
			# Execute handlers
			for handler in handlers:
				try:
					await handler(event)
				except Exception as e:
					self.logger.error(f"Event handler error: {str(e)}", exc_info=True)
			
			# Check for workflow triggers
			await self._check_workflow_triggers(event)
			
		except Exception as e:
			self.logger.error(f"Event handling error: {str(e)}", exc_info=True)
	
	async def publish_event(self, event: APGEvent) -> bool:
		"""
		Publish event to APG platform event bus.
		
		Args:
			event: Event to publish
		
		Returns:
			True if event published successfully
		"""
		try:
			# Add to event queue
			self.event_queue.append(event)
			
			# Process event queue
			await self._process_event_queue()
			
			return True
			
		except Exception as e:
			self.logger.error(f"Event publishing error: {str(e)}", exc_info=True)
			return False
	
	def register_event_handler(self, event_type: str, handler: Callable) -> None:
		"""Register event handler for specific event type"""
		if event_type not in self.event_handlers:
			self.event_handlers[event_type] = []
		
		self.event_handlers[event_type].append(handler)
		self.logger.info(_log_composition_operation(
			"register_handler", f"event_type={event_type}"
		))
	
	def register_workflow(self, workflow: CapabilityWorkflow) -> bool:
		"""Register multi-capability workflow"""
		try:
			self.workflows[workflow.id] = workflow
			self.logger.info(_log_composition_operation(
				"register_workflow", f"workflow={workflow.name}"
			))
			return True
			
		except Exception as e:
			self.logger.error(f"Workflow registration error: {str(e)}", exc_info=True)
			return False
	
	async def get_composition_status(self) -> Dict[str, Any]:
		"""Get comprehensive composition status"""
		return {
			"active_integrations": len(self.capability_integrations),
			"registered_workflows": len(self.workflows),
			"event_handlers": {
				event_type: len(handlers) 
				for event_type, handlers in self.event_handlers.items()
			},
			"event_queue_size": len(self.event_queue),
			"integrations": {
				capability_id: {
					"status": integration["status"],
					"last_interaction": integration["last_interaction"]
				}
				for capability_id, integration in self.capability_integrations.items()
			},
			"workflows": {
				workflow_id: {
					"name": workflow.name,
					"is_active": workflow.is_active,
					"execution_count": workflow.execution_count,
					"success_rate": (workflow.success_count / workflow.execution_count * 100) if workflow.execution_count > 0 else 0
				}
				for workflow_id, workflow in self.workflows.items()
			}
		}
	
	# Private helper methods
	
	def _initialize_default_integrations(self):
		"""Initialize default capability integrations"""
		default_integrations = {
			"auth_rbac": {
				"events": ["auth.user.login", "auth.user.logout", "auth.permission.changed"],
				"provides": ["user_authentication", "permission_validation"],
				"requires": ["mfa_verification"]
			},
			"audit_compliance": {
				"events": ["audit.security_event", "compliance.policy_violation"],
				"provides": ["audit_logging", "compliance_monitoring"],
				"requires": ["mfa_events"]
			},
			"notification": {
				"events": ["notification.template_registered"],
				"provides": ["notification_delivery"],
				"requires": ["mfa_notifications"]
			},
			"ai_orchestration": {
				"events": ["ai.model_updated", "ai.analysis_complete"],
				"provides": ["risk_analysis", "behavioral_analysis"],
				"requires": ["mfa_data"]
			}
		}
		
		for capability_id, config in default_integrations.items():
			self.capability_integrations[capability_id] = {
				"config": config,
				"status": "pending",
				"registered_at": datetime.utcnow().isoformat(),
				"last_interaction": None
			}
	
	def _register_default_workflows(self):
		"""Register default multi-capability workflows"""
		# Secure Login Workflow
		secure_login_workflow = CapabilityWorkflow(
			name="Secure User Login",
			description="Orchestrates secure user authentication with MFA and audit logging",
			trigger_events=["auth.user.login_attempt"],
			involved_capabilities=["mfa", "auth_rbac", "audit_compliance", "notification"],
			workflow_steps=[
				{
					"step": "risk_assessment",
					"capability": "mfa", 
					"action": "assess_login_risk",
					"required": True
				},
				{
					"step": "mfa_verification",
					"capability": "mfa",
					"action": "verify_authentication", 
					"required": True
				},
				{
					"step": "permission_check",
					"capability": "auth_rbac",
					"action": "validate_permissions",
					"required": True
				},
				{
					"step": "audit_log",
					"capability": "audit_compliance",
					"action": "log_authentication_event",
					"required": False
				},
				{
					"step": "notification",
					"capability": "notification", 
					"action": "send_login_notification",
					"required": False
				}
			]
		)
		
		self.workflows[secure_login_workflow.id] = secure_login_workflow
		
		# Security Incident Response Workflow
		incident_response_workflow = CapabilityWorkflow(
			name="Security Incident Response",
			description="Automated response to security incidents with MFA lockdown",
			trigger_events=["security.threat_detected", "mfa.suspicious_activity"],
			involved_capabilities=["mfa", "auth_rbac", "audit_compliance", "notification"],
			workflow_steps=[
				{
					"step": "threat_analysis",
					"capability": "ai_orchestration",
					"action": "analyze_threat_level",
					"required": True
				},
				{
					"step": "user_lockdown",
					"capability": "mfa",
					"action": "lockdown_user_account",
					"required": True
				},
				{
					"step": "revoke_sessions",
					"capability": "auth_rbac", 
					"action": "revoke_active_sessions",
					"required": True
				},
				{
					"step": "incident_audit",
					"capability": "audit_compliance",
					"action": "log_security_incident",
					"required": True
				},
				{
					"step": "alert_administrators",
					"capability": "notification",
					"action": "send_security_alert",
					"required": True
				}
			]
		)
		
		self.workflows[incident_response_workflow.id] = incident_response_workflow
	
	def _register_event_handlers(self):
		"""Register event handlers for APG platform events"""
		
		# Handle authentication events
		async def handle_auth_event(event: APGEvent):
			if event.event_type == "auth.user.login":
				# Trigger MFA verification workflow
				await self.execute_composed_workflow(
					"secure_login_workflow",
					event,
					{"user_id": event.data.get("user_id")}
				)
		
		# Handle security events
		async def handle_security_event(event: APGEvent):
			if event.event_type in ["security.threat_detected", "mfa.suspicious_activity"]:
				# Trigger incident response workflow
				await self.execute_composed_workflow(
					"incident_response_workflow", 
					event,
					{"threat_level": event.data.get("threat_level", "medium")}
				)
		
		# Handle AI analysis completion
		async def handle_ai_analysis(event: APGEvent):
			if event.event_type == "ai.analysis_complete":
				# Update risk scores based on AI analysis
				analysis_data = event.data
				if analysis_data.get("analysis_type") == "risk_assessment":
					# Update user risk profile
					pass
		
		# Register handlers
		self.register_event_handler("auth.user.login", handle_auth_event)
		self.register_event_handler("security.threat_detected", handle_security_event)
		self.register_event_handler("mfa.suspicious_activity", handle_security_event)
		self.register_event_handler("ai.analysis_complete", handle_ai_analysis)
	
	async def _execute_workflow_step(self, 
									 step: Dict[str, Any],
									 execution_context: Dict[str, Any],
									 step_index: int) -> Dict[str, Any]:
		"""Execute a single workflow step"""
		try:
			capability = step.get("capability")
			action = step.get("action")
			required = step.get("required", False)
			
			# Call capability action
			result = await self.integration.call_capability(
				capability, action, execution_context
			)
			
			return {
				"step_index": step_index,
				"step_name": step.get("step"),
				"capability": capability,
				"action": action,
				"success": result.get("success", False),
				"result": result,
				"required": required,
				"executed_at": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			return {
				"step_index": step_index,
				"success": False,
				"error": str(e),
				"executed_at": datetime.utcnow().isoformat()
			}
	
	async def _establish_event_flow(self, capability_id: str, config: Dict[str, Any]):
		"""Establish bi-directional event flow with capability"""
		# Subscribe to capability events
		subscribed_events = config.get("events", [])
		for event_type in subscribed_events:
			await self.integration.subscribe_to_events(capability_id, event_type)
		
		# Register MFA events with capability
		mfa_events = [
			"mfa.authentication.success",
			"mfa.authentication.failure", 
			"mfa.method.enrolled",
			"mfa.security.alert"
		]
		
		for event_type in mfa_events:
			await self.integration.register_event_type(capability_id, event_type)
	
	async def _process_event_queue(self):
		"""Process queued events for publishing"""
		while self.event_queue:
			event = self.event_queue.pop(0)
			
			try:
				# Publish to APG event bus
				if event.broadcast:
					# Broadcast to all capabilities
					await self.integration.broadcast_event(event)
				elif event.target_capability:
					# Send to specific capability
					await self.integration.send_event(event.target_capability, event)
				else:
					# Default broadcast
					await self.integration.broadcast_event(event)
					
			except Exception as e:
				self.logger.error(f"Event publishing error: {str(e)}", exc_info=True)
	
	async def _check_workflow_triggers(self, event: APGEvent):
		"""Check if event triggers any registered workflows"""
		for workflow_id, workflow in self.workflows.items():
			if workflow.is_active and event.event_type in workflow.trigger_events:
				# Execute workflow asynchronously
				asyncio.create_task(self.execute_composed_workflow(
					workflow_id, event, {"triggered_by": event.event_type}
				))
	
	async def _handle_workflow_failure(self, 
									   workflow: CapabilityWorkflow,
									   execution_context: Dict[str, Any],
									   failed_step: int,
									   step_result: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle workflow step failure"""
		workflow.execution_count += 1
		workflow.failure_count += 1
		
		# Publish failure event
		await self.publish_event(APGEvent(
			event_type="mfa.workflow.failed",
			source_capability="mfa",
			data={
				"workflow_id": workflow.id,
				"failed_step": failed_step,
				"execution_context": execution_context,
				"failure_reason": step_result.get("error")
			}
		))
		
		return {
			"success": False,
			"workflow_id": workflow.id,
			"failed_step": failed_step,
			"error": step_result.get("error"),
			"execution_context": execution_context
		}


__all__ = [
	'MFACompositionEngine',
	'APGEvent',
	'CapabilityWorkflow',
	'CompositionEventType'
]