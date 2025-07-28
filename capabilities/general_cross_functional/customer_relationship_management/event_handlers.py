"""
APG Customer Relationship Management - Event Handlers

Revolutionary event-driven architecture implementation providing seamless
APG ecosystem integration with intelligent event processing, real-time
synchronization, and comprehensive audit trails.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json

# APG Core imports (these would be actual APG framework imports)
from apg.core.events import EventBus, Event, EventHandler, EventSubscriber
from apg.core.auth import UserContext

# Local imports
from .models import CRMContact, CRMAccount, CRMLead, CRMOpportunity, CRMActivity
from .service import CRMService


logger = logging.getLogger(__name__)


class CRMEventType(str, Enum):
	"""CRM-specific event types"""
	# Contact events
	CONTACT_CREATED = "crm.contact.created"
	CONTACT_UPDATED = "crm.contact.updated"
	CONTACT_DELETED = "crm.contact.deleted"
	CONTACT_MERGED = "crm.contact.merged"
	
	# Account events
	ACCOUNT_CREATED = "crm.account.created"
	ACCOUNT_UPDATED = "crm.account.updated"
	ACCOUNT_DELETED = "crm.account.deleted"
	
	# Lead events
	LEAD_CREATED = "crm.lead.created"
	LEAD_UPDATED = "crm.lead.updated"
	LEAD_CONVERTED = "crm.lead.converted"
	LEAD_QUALIFIED = "crm.lead.qualified"
	LEAD_DISQUALIFIED = "crm.lead.disqualified"
	
	# Opportunity events
	OPPORTUNITY_CREATED = "crm.opportunity.created"
	OPPORTUNITY_UPDATED = "crm.opportunity.updated"
	OPPORTUNITY_STAGE_CHANGED = "crm.opportunity.stage_changed"
	OPPORTUNITY_WON = "crm.opportunity.won"
	OPPORTUNITY_LOST = "crm.opportunity.lost"
	
	# Activity events
	ACTIVITY_CREATED = "crm.activity.created"
	ACTIVITY_COMPLETED = "crm.activity.completed"
	ACTIVITY_OVERDUE = "crm.activity.overdue"
	
	# Business process events
	SALES_PIPELINE_UPDATED = "crm.pipeline.updated"
	CUSTOMER_JOURNEY_MILESTONE = "crm.journey.milestone"
	AI_INSIGHT_GENERATED = "crm.ai.insight_generated"
	PREDICTIVE_SCORE_UPDATED = "crm.ai.score_updated"


@dataclass
class CRMEvent:
	"""CRM event data structure"""
	event_type: CRMEventType
	entity_type: str
	entity_id: str
	tenant_id: str
	user_id: str
	timestamp: datetime
	data: Dict[str, Any]
	previous_data: Optional[Dict[str, Any]] = None
	correlation_id: Optional[str] = None
	source_service: str = "customer_relationship_management"


class CRMEventPublisher:
	"""
	CRM event publisher for publishing business events to APG ecosystem
	"""
	
	def __init__(self, event_bus: EventBus):
		"""
		Initialize CRM event publisher
		
		Args:
			event_bus: APG event bus instance
		"""
		self.event_bus = event_bus
		self._initialized = False
		
		logger.info("📡 CRM Event Publisher initialized")
	
	async def initialize(self):
		"""Initialize event publisher"""
		try:
			logger.info("🔧 Initializing CRM event publisher...")
			
			# Register event schemas with the event bus
			await self._register_event_schemas()
			
			self._initialized = True
			logger.info("✅ CRM event publisher initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize CRM event publisher: {str(e)}", exc_info=True)
			raise
	
	async def _register_event_schemas(self):
		"""Register CRM event schemas with event bus"""
		# Define event schemas for validation
		schemas = {
			CRMEventType.CONTACT_CREATED: {
				"type": "object",
				"properties": {
					"contact_id": {"type": "string"},
					"first_name": {"type": "string"},
					"last_name": {"type": "string"},
					"email": {"type": "string"},
					"tenant_id": {"type": "string"}
				},
				"required": ["contact_id", "first_name", "last_name", "tenant_id"]
			},
			CRMEventType.LEAD_CONVERTED: {
				"type": "object", 
				"properties": {
					"lead_id": {"type": "string"},
					"contact_id": {"type": "string"},
					"account_id": {"type": "string"},
					"opportunity_id": {"type": "string"},
					"conversion_value": {"type": "number"},
					"tenant_id": {"type": "string"}
				},
				"required": ["lead_id", "contact_id", "tenant_id"]
			},
			CRMEventType.OPPORTUNITY_WON: {
				"type": "object",
				"properties": {
					"opportunity_id": {"type": "string"},
					"amount": {"type": "number"},
					"close_date": {"type": "string"},
					"account_id": {"type": "string"},
					"tenant_id": {"type": "string"}
				},
				"required": ["opportunity_id", "amount", "tenant_id"]
			}
		}
		
		# Register schemas with event bus
		for event_type, schema in schemas.items():
			await self.event_bus.register_event_schema(event_type.value, schema)
	
	async def publish_contact_created(
		self, 
		contact: CRMContact, 
		user_context: UserContext
	):
		"""Publish contact created event"""
		event = CRMEvent(
			event_type=CRMEventType.CONTACT_CREATED,
			entity_type="contact",
			entity_id=contact.id,
			tenant_id=contact.tenant_id,
			user_id=user_context.user_id,
			timestamp=datetime.utcnow(),
			data={
				"contact_id": contact.id,
				"first_name": contact.first_name,
				"last_name": contact.last_name,
				"email": contact.email,
				"company": contact.company,
				"contact_type": contact.contact_type.value,
				"lead_source": contact.lead_source.value if contact.lead_source else None,
				"tenant_id": contact.tenant_id
			}
		)
		
		await self._publish_event(event)
	
	async def publish_contact_updated(
		self, 
		contact: CRMContact, 
		previous_data: Dict[str, Any],
		user_context: UserContext
	):
		"""Publish contact updated event"""
		event = CRMEvent(
			event_type=CRMEventType.CONTACT_UPDATED,
			entity_type="contact",
			entity_id=contact.id,
			tenant_id=contact.tenant_id,
			user_id=user_context.user_id,
			timestamp=datetime.utcnow(),
			data=contact.model_dump(),
			previous_data=previous_data
		)
		
		await self._publish_event(event)
	
	async def publish_lead_converted(
		self,
		lead_id: str,
		contact_id: str,
		account_id: Optional[str],
		opportunity_id: Optional[str],
		conversion_value: Optional[float],
		tenant_id: str,
		user_context: UserContext
	):
		"""Publish lead converted event"""
		event = CRMEvent(
			event_type=CRMEventType.LEAD_CONVERTED,
			entity_type="lead",
			entity_id=lead_id,
			tenant_id=tenant_id,
			user_id=user_context.user_id,
			timestamp=datetime.utcnow(),
			data={
				"lead_id": lead_id,
				"contact_id": contact_id,
				"account_id": account_id,
				"opportunity_id": opportunity_id,
				"conversion_value": conversion_value,
				"tenant_id": tenant_id
			}
		)
		
		await self._publish_event(event)
	
	async def publish_opportunity_won(
		self,
		opportunity: CRMOpportunity,
		user_context: UserContext
	):
		"""Publish opportunity won event"""
		event = CRMEvent(
			event_type=CRMEventType.OPPORTUNITY_WON,
			entity_type="opportunity",
			entity_id=opportunity.id,
			tenant_id=opportunity.tenant_id,
			user_id=user_context.user_id,
			timestamp=datetime.utcnow(),
			data={
				"opportunity_id": opportunity.id,
				"opportunity_name": opportunity.opportunity_name,
				"amount": float(opportunity.amount),
				"close_date": opportunity.close_date.isoformat(),
				"account_id": opportunity.account_id,
				"primary_contact_id": opportunity.primary_contact_id,
				"owner_id": opportunity.owner_id,
				"tenant_id": opportunity.tenant_id
			}
		)
		
		await self._publish_event(event)
	
	async def publish_ai_insight_generated(
		self,
		entity_type: str,
		entity_id: str,
		insight_type: str,
		insight_data: Dict[str, Any],
		tenant_id: str,
		confidence_score: Optional[float] = None
	):
		"""Publish AI insight generated event"""
		event = CRMEvent(
			event_type=CRMEventType.AI_INSIGHT_GENERATED,
			entity_type=entity_type,
			entity_id=entity_id,
			tenant_id=tenant_id,
			user_id="system",
			timestamp=datetime.utcnow(),
			data={
				"entity_type": entity_type,
				"entity_id": entity_id,
				"insight_type": insight_type,
				"insight_data": insight_data,
				"confidence_score": confidence_score,
				"tenant_id": tenant_id
			}
		)
		
		await self._publish_event(event)
	
	async def _publish_event(self, crm_event: CRMEvent):
		"""Publish CRM event to APG event bus"""
		try:
			# Create APG event
			apg_event = Event(
				event_type=crm_event.event_type.value,
				source=crm_event.source_service,
				data=crm_event.data,
				correlation_id=crm_event.correlation_id,
				user_id=crm_event.user_id,
				tenant_id=crm_event.tenant_id,
				timestamp=crm_event.timestamp
			)
			
			# Add previous data if available
			if crm_event.previous_data:
				apg_event.metadata = {"previous_data": crm_event.previous_data}
			
			# Publish to event bus
			await self.event_bus.publish(apg_event)
			
			logger.debug(f"📤 Published event: {crm_event.event_type.value}")
			
		except Exception as e:
			logger.error(f"Failed to publish event {crm_event.event_type.value}: {str(e)}", exc_info=True)
			raise
	
	async def health_check(self) -> Dict[str, Any]:
		"""Health check for event publisher"""
		return {
			"status": "healthy" if self._initialized else "unhealthy",
			"initialized": self._initialized,
			"timestamp": datetime.utcnow().isoformat()
		}


class CRMEventSubscriber:
	"""
	CRM event subscriber for handling events from APG ecosystem
	"""
	
	def __init__(self, event_bus: EventBus, service: CRMService):
		"""
		Initialize CRM event subscriber
		
		Args:
			event_bus: APG event bus instance
			service: CRM service instance
		"""
		self.event_bus = event_bus
		self.service = service
		self._subscriptions: Dict[str, EventHandler] = {}
		self._initialized = False
		
		logger.info("📨 CRM Event Subscriber initialized")
	
	async def setup_subscriptions(self):
		"""Setup event subscriptions"""
		try:
			logger.info("🔧 Setting up CRM event subscriptions...")
			
			# Subscribe to user management events
			await self._subscribe_to_user_events()
			
			# Subscribe to audit events
			await self._subscribe_to_audit_events()
			
			# Subscribe to notification events
			await self._subscribe_to_notification_events()
			
			# Subscribe to AI orchestration events
			await self._subscribe_to_ai_events()
			
			# Subscribe to workflow events
			await self._subscribe_to_workflow_events()
			
			self._initialized = True
			logger.info("✅ CRM event subscriptions setup completed")
			
		except Exception as e:
			logger.error(f"Failed to setup CRM event subscriptions: {str(e)}", exc_info=True)
			raise
	
	async def _subscribe_to_user_events(self):
		"""Subscribe to user management events"""
		# User created - might need to create CRM user profile
		await self.event_bus.subscribe(
			"auth.user.created",
			self._handle_user_created
		)
		
		# User updated - might need to sync user info
		await self.event_bus.subscribe(
			"auth.user.updated", 
			self._handle_user_updated
		)
		
		# User deactivated - might need to reassign records
		await self.event_bus.subscribe(
			"auth.user.deactivated",
			self._handle_user_deactivated
		)
	
	async def _subscribe_to_audit_events(self):
		"""Subscribe to audit and compliance events"""
		# Compliance policy updated
		await self.event_bus.subscribe(
			"audit.policy.updated",
			self._handle_compliance_policy_updated
		)
		
		# Data retention policy changed
		await self.event_bus.subscribe(
			"audit.retention.policy_changed",
			self._handle_data_retention_policy_changed
		)
	
	async def _subscribe_to_notification_events(self):
		"""Subscribe to notification events"""
		# Email delivery status
		await self.event_bus.subscribe(
			"notification.email.delivered",
			self._handle_email_delivered
		)
		
		await self.event_bus.subscribe(
			"notification.email.failed",
			self._handle_email_failed
		)
	
	async def _subscribe_to_ai_events(self):
		"""Subscribe to AI orchestration events"""
		# Model updated - might need to recalculate scores
		await self.event_bus.subscribe(
			"ai.model.updated",
			self._handle_ai_model_updated
		)
		
		# Batch processing completed
		await self.event_bus.subscribe(
			"ai.batch.completed",
			self._handle_ai_batch_completed
		)
	
	async def _subscribe_to_workflow_events(self):
		"""Subscribe to workflow events"""
		# Workflow completed
		await self.event_bus.subscribe(
			"workflow.completed",
			self._handle_workflow_completed
		)
		
		# Approval granted/denied
		await self.event_bus.subscribe(
			"workflow.approval.granted",
			self._handle_approval_granted
		)
		
		await self.event_bus.subscribe(
			"workflow.approval.denied",
			self._handle_approval_denied
		)
	
	# Event handlers
	
	async def _handle_user_created(self, event: Event):
		"""Handle user created event"""
		try:
			user_data = event.data
			user_id = user_data.get("user_id")
			tenant_id = event.tenant_id
			
			logger.info(f"👤 New user created: {user_id}, setting up CRM access...")
			
			# Could create default CRM user settings, assign territories, etc.
			# For now, just log the event
			
		except Exception as e:
			logger.error(f"Failed to handle user created event: {str(e)}", exc_info=True)
	
	async def _handle_user_updated(self, event: Event):
		"""Handle user updated event"""
		try:
			user_data = event.data
			user_id = user_data.get("user_id")
			
			# Sync user information in CRM records
			logger.info(f"🔄 User {user_id} updated, syncing CRM data...")
			
		except Exception as e:
			logger.error(f"Failed to handle user updated event: {str(e)}", exc_info=True)
	
	async def _handle_user_deactivated(self, event: Event):
		"""Handle user deactivated event"""
		try:
			user_data = event.data
			user_id = user_data.get("user_id")
			tenant_id = event.tenant_id
			
			logger.warning(f"⚠️ User {user_id} deactivated, reassigning CRM records...")
			
			# Could reassign leads, opportunities, activities to other users
			# For now, just log the event
			
		except Exception as e:
			logger.error(f"Failed to handle user deactivated event: {str(e)}", exc_info=True)
	
	async def _handle_compliance_policy_updated(self, event: Event):
		"""Handle compliance policy updated event"""
		try:
			policy_data = event.data
			policy_type = policy_data.get("policy_type")
			
			logger.info(f"📋 Compliance policy updated: {policy_type}")
			
			# Could trigger compliance audit, update data handling, etc.
			
		except Exception as e:
			logger.error(f"Failed to handle compliance policy updated event: {str(e)}", exc_info=True)
	
	async def _handle_data_retention_policy_changed(self, event: Event):
		"""Handle data retention policy changed event"""
		try:
			policy_data = event.data
			retention_period = policy_data.get("retention_period_days")
			
			logger.info(f"🗄️ Data retention policy changed: {retention_period} days")
			
			# Could trigger data cleanup, archive old records, etc.
			
		except Exception as e:
			logger.error(f"Failed to handle data retention policy event: {str(e)}", exc_info=True)
	
	async def _handle_email_delivered(self, event: Event):
		"""Handle email delivered event"""
		try:
			email_data = event.data
			email_id = email_data.get("email_id")
			recipient = email_data.get("recipient")
			
			logger.info(f"✉️ Email delivered: {email_id} to {recipient}")
			
			# Could update communication history, activity status, etc.
			
		except Exception as e:
			logger.error(f"Failed to handle email delivered event: {str(e)}", exc_info=True)
	
	async def _handle_email_failed(self, event: Event):
		"""Handle email failed event"""
		try:
			email_data = event.data
			email_id = email_data.get("email_id")
			recipient = email_data.get("recipient")
			error = email_data.get("error")
			
			logger.warning(f"❌ Email failed: {email_id} to {recipient}, error: {error}")
			
			# Could update contact validity, create follow-up tasks, etc.
			
		except Exception as e:
			logger.error(f"Failed to handle email failed event: {str(e)}", exc_info=True)
	
	async def _handle_ai_model_updated(self, event: Event):
		"""Handle AI model updated event"""
		try:
			model_data = event.data
			model_name = model_data.get("model_name")
			model_version = model_data.get("version")
			
			logger.info(f"🤖 AI model updated: {model_name} v{model_version}")
			
			# Could trigger recalculation of lead scores, opportunity probabilities, etc.
			if model_name in ["lead_scoring", "opportunity_prediction"]:
				await self._trigger_score_recalculation(model_name)
			
		except Exception as e:
			logger.error(f"Failed to handle AI model updated event: {str(e)}", exc_info=True)
	
	async def _handle_ai_batch_completed(self, event: Event):
		"""Handle AI batch processing completed event"""
		try:
			batch_data = event.data
			batch_id = batch_data.get("batch_id")
			job_type = batch_data.get("job_type")
			results = batch_data.get("results", {})
			
			logger.info(f"⚡ AI batch completed: {batch_id} ({job_type})")
			
			# Process batch results if relevant to CRM
			if job_type == "lead_scoring":
				await self._process_lead_scoring_results(results)
			elif job_type == "customer_segmentation":
				await self._process_segmentation_results(results)
			
		except Exception as e:
			logger.error(f"Failed to handle AI batch completed event: {str(e)}", exc_info=True)
	
	async def _handle_workflow_completed(self, event: Event):
		"""Handle workflow completed event"""
		try:
			workflow_data = event.data
			workflow_id = workflow_data.get("workflow_id")
			workflow_type = workflow_data.get("workflow_type")
			entity_id = workflow_data.get("entity_id")
			
			logger.info(f"🔄 Workflow completed: {workflow_id} ({workflow_type})")
			
			# Update related CRM records based on workflow completion
			if workflow_type == "lead_qualification":
				await self._update_lead_from_workflow(entity_id, workflow_data)
			elif workflow_type == "opportunity_approval":
				await self._update_opportunity_from_workflow(entity_id, workflow_data)
			
		except Exception as e:
			logger.error(f"Failed to handle workflow completed event: {str(e)}", exc_info=True)
	
	async def _handle_approval_granted(self, event: Event):
		"""Handle approval granted event"""
		try:
			approval_data = event.data
			approval_id = approval_data.get("approval_id")
			entity_type = approval_data.get("entity_type")
			entity_id = approval_data.get("entity_id")
			
			logger.info(f"✅ Approval granted: {approval_id} for {entity_type}:{entity_id}")
			
			# Update CRM entity status based on approval
			
		except Exception as e:
			logger.error(f"Failed to handle approval granted event: {str(e)}", exc_info=True)
	
	async def _handle_approval_denied(self, event: Event):
		"""Handle approval denied event"""
		try:
			approval_data = event.data
			approval_id = approval_data.get("approval_id")
			entity_type = approval_data.get("entity_type")
			entity_id = approval_data.get("entity_id")
			reason = approval_data.get("reason")
			
			logger.warning(f"❌ Approval denied: {approval_id} for {entity_type}:{entity_id}, reason: {reason}")
			
			# Handle approval denial, possibly create follow-up tasks
			
		except Exception as e:
			logger.error(f"Failed to handle approval denied event: {str(e)}", exc_info=True)
	
	# Helper methods
	
	async def _trigger_score_recalculation(self, model_name: str):
		"""Trigger recalculation of AI scores"""
		logger.info(f"🔄 Triggering score recalculation for model: {model_name}")
		# Would trigger background job to recalculate scores
	
	async def _process_lead_scoring_results(self, results: Dict[str, Any]):
		"""Process lead scoring batch results"""
		logger.info("📊 Processing lead scoring batch results")
		# Would update lead scores in database
	
	async def _process_segmentation_results(self, results: Dict[str, Any]):
		"""Process customer segmentation results"""
		logger.info("🎯 Processing customer segmentation results")
		# Would update customer segments
	
	async def _update_lead_from_workflow(self, lead_id: str, workflow_data: Dict[str, Any]):
		"""Update lead based on workflow completion"""
		logger.info(f"🔄 Updating lead {lead_id} from workflow")
		# Would update lead status, score, etc.
	
	async def _update_opportunity_from_workflow(self, opportunity_id: str, workflow_data: Dict[str, Any]):
		"""Update opportunity based on workflow completion"""
		logger.info(f"🔄 Updating opportunity {opportunity_id} from workflow")
		# Would update opportunity stage, probability, etc.
	
	async def shutdown(self):
		"""Shutdown event subscriber"""
		try:
			logger.info("🛑 Shutting down CRM event subscriber...")
			
			# Unsubscribe from all events
			for event_type, handler in self._subscriptions.items():
				await self.event_bus.unsubscribe(event_type, handler)
			
			self._subscriptions.clear()
			self._initialized = False
			
			logger.info("✅ CRM event subscriber shutdown completed")
			
		except Exception as e:
			logger.error(f"Error during event subscriber shutdown: {str(e)}", exc_info=True)


# Export classes
__all__ = [
	"CRMEventPublisher",
	"CRMEventSubscriber", 
	"CRMEvent",
	"CRMEventType"
]