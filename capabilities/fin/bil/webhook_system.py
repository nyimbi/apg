"""
APG Billing Webhook System

Real-time webhook system for payment events, invoice updates, subscription changes,
and other billing-related events with reliable delivery and retry mechanisms.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from uuid_extensions import uuid7str

import aiohttp
from cryptography.fernet import Fernet


class WebhookEventType(Enum):
	"""Webhook event types"""
	PAYMENT_SUCCEEDED = "payment.succeeded"
	PAYMENT_FAILED = "payment.failed"
	PAYMENT_REFUNDED = "payment.refunded"
	INVOICE_CREATED = "invoice.created"
	INVOICE_PAID = "invoice.paid"
	INVOICE_OVERDUE = "invoice.overdue"
	SUBSCRIPTION_CREATED = "subscription.created"
	SUBSCRIPTION_UPDATED = "subscription.updated"
	SUBSCRIPTION_CANCELLED = "subscription.cancelled"
	CUSTOMER_CREATED = "customer.created"
	CUSTOMER_UPDATED = "customer.updated"
	USAGE_SUBMITTED = "usage.submitted"
	DISPUTE_CREATED = "dispute.created"


class WebhookDeliveryStatus(Enum):
	"""Webhook delivery status"""
	PENDING = "pending"
	DELIVERED = "delivered"
	FAILED = "failed"
	RETRYING = "retrying"
	EXPIRED = "expired"


class WebhookEndpoint:
	"""Webhook endpoint configuration"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.url = data['url']
		self.secret = data.get('secret', self._generate_secret())
		self.enabled = data.get('enabled', True)
		self.events = set(data.get('events', []))
		self.created_at = datetime.fromisoformat(data.get('created_at', datetime.utcnow().isoformat()))
		self.updated_at = datetime.utcnow()
		
		# Delivery settings
		self.timeout_seconds = data.get('timeout_seconds', 30)
		self.max_retries = data.get('max_retries', 3)
		self.retry_delay_seconds = data.get('retry_delay_seconds', 60)
		
		# Custom headers
		self.headers = data.get('headers', {})
		
		# Filtering
		self.filters = data.get('filters', {})
	
	def _generate_secret(self) -> str:
		"""Generate webhook secret for signature verification"""
		return f"whsec_{uuid7str()}"
	
	def should_receive_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
		"""Check if endpoint should receive this event"""
		if not self.enabled:
			return False
		
		if event_type not in self.events and '*' not in self.events:
			return False
		
		# Apply filters
		for filter_key, filter_value in self.filters.items():
			if filter_key in event_data and event_data[filter_key] != filter_value:
				return False
		
		return True
	
	def generate_signature(self, payload: str, timestamp: str) -> str:
		"""Generate webhook signature"""
		signed_payload = f"{timestamp}.{payload}"
		signature = hmac.new(
			self.secret.encode('utf-8'),
			signed_payload.encode('utf-8'),
			hashlib.sha256
		).hexdigest()
		return f"t={timestamp},v1={signature}"


class WebhookEvent:
	"""Webhook event data"""
	
	def __init__(self, event_type: str, data: Dict[str, Any], tenant_id: str = None):
		self.id = uuid7str()
		self.type = event_type
		self.data = data
		self.tenant_id = tenant_id
		self.created_at = datetime.utcnow()
		self.api_version = "2025-01-01"
		
		# Delivery tracking
		self.deliveries: List[Dict[str, Any]] = []
	
	def to_payload(self) -> Dict[str, Any]:
		"""Convert to webhook payload"""
		return {
			'id': self.id,
			'type': self.type,
			'created': int(self.created_at.timestamp()),
			'data': self.data,
			'tenant_id': self.tenant_id,
			'api_version': self.api_version
		}


class WebhookDelivery:
	"""Webhook delivery attempt"""
	
	def __init__(self, event: WebhookEvent, endpoint: WebhookEndpoint):
		self.id = uuid7str()
		self.event_id = event.id
		self.endpoint_id = endpoint.id
		self.endpoint_url = endpoint.url
		self.status = WebhookDeliveryStatus.PENDING
		self.attempt_count = 0
		self.max_attempts = endpoint.max_retries + 1
		self.created_at = datetime.utcnow()
		self.next_attempt_at = datetime.utcnow()
		self.completed_at = None
		self.response_code = None
		self.response_body = None
		self.error_message = None


class WebhookSystem:
	"""Real-time webhook delivery system"""
	
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.WebhookSystem")
		self.endpoints: Dict[str, WebhookEndpoint] = {}
		self.events: Dict[str, WebhookEvent] = {}
		self.deliveries: Dict[str, WebhookDelivery] = {}
		self.delivery_queue: asyncio.Queue = asyncio.Queue()
		
		# Event handlers
		self.event_handlers: Dict[str, List[Callable]] = {}
		
		# System settings
		self.max_concurrent_deliveries = 10
		self.delivery_timeout = 30
		self.signature_required = True
		
		# Start background workers
		asyncio.create_task(self._start_delivery_workers())
		asyncio.create_task(self._start_retry_processor())
	
	async def _start_delivery_workers(self) -> None:
		"""Start webhook delivery workers"""
		workers = []
		for i in range(self.max_concurrent_deliveries):
			worker = asyncio.create_task(self._delivery_worker(f"worker-{i}"))
			workers.append(worker)
		
		self.logger.info(f"Started {len(workers)} webhook delivery workers")
	
	async def _delivery_worker(self, worker_name: str) -> None:
		"""Webhook delivery worker"""
		while True:
			try:
				delivery = await self.delivery_queue.get()
				await self._attempt_delivery(delivery)
				self.delivery_queue.task_done()
			except Exception as e:
				self.logger.error(f"Delivery worker {worker_name} error: {e}")
				await asyncio.sleep(1)
	
	async def _start_retry_processor(self) -> None:
		"""Start retry processor for failed deliveries"""
		while True:
			try:
				await self._process_retries()
				await asyncio.sleep(60)  # Check every minute
			except Exception as e:
				self.logger.error(f"Retry processor error: {e}")
				await asyncio.sleep(60)
	
	def register_endpoint(self, endpoint_data: Dict[str, Any]) -> WebhookEndpoint:
		"""Register a new webhook endpoint"""
		endpoint = WebhookEndpoint(endpoint_data)
		self.endpoints[endpoint.id] = endpoint
		self.logger.info(f"Registered webhook endpoint: {endpoint.url}")
		return endpoint
	
	def update_endpoint(self, endpoint_id: str, updates: Dict[str, Any]) -> Optional[WebhookEndpoint]:
		"""Update webhook endpoint"""
		endpoint = self.endpoints.get(endpoint_id)
		if not endpoint:
			return None
		
		# Update fields
		for key, value in updates.items():
			if hasattr(endpoint, key):
				setattr(endpoint, key, value)
		
		endpoint.updated_at = datetime.utcnow()
		self.logger.info(f"Updated webhook endpoint: {endpoint.url}")
		return endpoint
	
	def delete_endpoint(self, endpoint_id: str) -> bool:
		"""Delete webhook endpoint"""
		if endpoint_id in self.endpoints:
			endpoint = self.endpoints[endpoint_id]
			del self.endpoints[endpoint_id]
			self.logger.info(f"Deleted webhook endpoint: {endpoint.url}")
			return True
		return False
	
	async def emit_event(self, event_type: str, data: Dict[str, Any], tenant_id: str = None) -> WebhookEvent:
		"""Emit a webhook event"""
		try:
			# Create event
			event = WebhookEvent(event_type, data, tenant_id)
			self.events[event.id] = event
			
			# Find endpoints that should receive this event
			matching_endpoints = []
			for endpoint in self.endpoints.values():
				if endpoint.should_receive_event(event_type, data):
					matching_endpoints.append(endpoint)
			
			# Create deliveries
			for endpoint in matching_endpoints:
				delivery = WebhookDelivery(event, endpoint)
				self.deliveries[delivery.id] = delivery
				await self.delivery_queue.put(delivery)
			
			# Call local event handlers
			await self._call_event_handlers(event_type, event)
			
			self.logger.info(f"Emitted webhook event {event_type} to {len(matching_endpoints)} endpoints")
			return event
			
		except Exception as e:
			self.logger.error(f"Failed to emit webhook event {event_type}: {e}")
			raise
	
	async def _attempt_delivery(self, delivery: WebhookDelivery) -> None:
		"""Attempt webhook delivery"""
		try:
			delivery.attempt_count += 1
			delivery.status = WebhookDeliveryStatus.RETRYING if delivery.attempt_count > 1 else WebhookDeliveryStatus.PENDING
			
			# Get event and endpoint
			event = self.events.get(delivery.event_id)
			endpoint = self.endpoints.get(delivery.endpoint_id)
			
			if not event or not endpoint:
				delivery.status = WebhookDeliveryStatus.FAILED
				delivery.error_message = "Event or endpoint not found"
				delivery.completed_at = datetime.utcnow()
				return
			
			# Prepare payload
			payload = json.dumps(event.to_payload())
			timestamp = str(int(datetime.utcnow().timestamp()))
			
			# Prepare headers
			headers = {
				'Content-Type': 'application/json',
				'User-Agent': 'APG-Billing-Webhooks/1.0',
				'X-APG-Event-Type': event.type,
				'X-APG-Event-ID': event.id,
				'X-APG-Delivery-ID': delivery.id,
				'X-APG-Timestamp': timestamp,
				**endpoint.headers
			}
			
			# Add signature if required
			if self.signature_required:
				signature = endpoint.generate_signature(payload, timestamp)
				headers['X-APG-Signature'] = signature
			
			# Make HTTP request
			async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=endpoint.timeout_seconds)) as session:
				async with session.post(endpoint.url, data=payload, headers=headers) as response:
					delivery.response_code = response.status
					delivery.response_body = await response.text()
					
					if 200 <= response.status < 300:
						delivery.status = WebhookDeliveryStatus.DELIVERED
						delivery.completed_at = datetime.utcnow()
						self.logger.info(f"Webhook delivered successfully: {delivery.id}")
					else:
						raise aiohttp.ClientResponseError(
							request_info=response.request_info,
							history=response.history,
							status=response.status,
							message=f"HTTP {response.status}"
						)
		
		except Exception as e:
			delivery.error_message = str(e)
			
			if delivery.attempt_count >= delivery.max_attempts:
				delivery.status = WebhookDeliveryStatus.FAILED
				delivery.completed_at = datetime.utcnow()
				self.logger.error(f"Webhook delivery failed permanently: {delivery.id} - {e}")
			else:
				delivery.status = WebhookDeliveryStatus.RETRYING
				# Calculate next retry time with exponential backoff
				delay = endpoint.retry_delay_seconds * (2 ** (delivery.attempt_count - 1))
				delivery.next_attempt_at = datetime.utcnow() + timedelta(seconds=delay)
				self.logger.warning(f"Webhook delivery failed, will retry: {delivery.id} - {e}")
	
	async def _process_retries(self) -> None:
		"""Process webhook delivery retries"""
		now = datetime.utcnow()
		retry_deliveries = [
			delivery for delivery in self.deliveries.values()
			if delivery.status == WebhookDeliveryStatus.RETRYING and delivery.next_attempt_at <= now
		]
		
		for delivery in retry_deliveries:
			await self.delivery_queue.put(delivery)
		
		if retry_deliveries:
			self.logger.info(f"Queued {len(retry_deliveries)} webhook deliveries for retry")
	
	def register_event_handler(self, event_type: str, handler: Callable) -> None:
		"""Register local event handler"""
		if event_type not in self.event_handlers:
			self.event_handlers[event_type] = []
		self.event_handlers[event_type].append(handler)
		self.logger.info(f"Registered event handler for {event_type}")
	
	async def _call_event_handlers(self, event_type: str, event: WebhookEvent) -> None:
		"""Call local event handlers"""
		handlers = self.event_handlers.get(event_type, [])
		for handler in handlers:
			try:
				if asyncio.iscoroutinefunction(handler):
					await handler(event)
				else:
					handler(event)
			except Exception as e:
				self.logger.error(f"Event handler failed for {event_type}: {e}")
	
	async def get_event_deliveries(self, event_id: str) -> List[Dict[str, Any]]:
		"""Get delivery attempts for an event"""
		deliveries = [
			{
				'id': delivery.id,
				'endpoint_url': delivery.endpoint_url,
				'status': delivery.status.value,
				'attempt_count': delivery.attempt_count,
				'response_code': delivery.response_code,
				'error_message': delivery.error_message,
				'created_at': delivery.created_at.isoformat(),
				'completed_at': delivery.completed_at.isoformat() if delivery.completed_at else None
			}
			for delivery in self.deliveries.values()
			if delivery.event_id == event_id
		]
		
		return sorted(deliveries, key=lambda d: d['created_at'])
	
	async def retry_delivery(self, delivery_id: str) -> bool:
		"""Manually retry a webhook delivery"""
		delivery = self.deliveries.get(delivery_id)
		if not delivery:
			return False
		
		if delivery.status in [WebhookDeliveryStatus.DELIVERED, WebhookDeliveryStatus.EXPIRED]:
			return False
		
		# Reset for retry
		delivery.status = WebhookDeliveryStatus.PENDING
		delivery.next_attempt_at = datetime.utcnow()
		delivery.error_message = None
		
		await self.delivery_queue.put(delivery)
		self.logger.info(f"Manually retrying webhook delivery: {delivery_id}")
		return True
	
	async def get_webhook_statistics(self, tenant_id: str = None) -> Dict[str, Any]:
		"""Get webhook delivery statistics"""
		deliveries = list(self.deliveries.values())
		
		if tenant_id:
			# Filter by tenant
			tenant_events = {
				event_id: event for event_id, event in self.events.items()
				if event.tenant_id == tenant_id
			}
			deliveries = [
				delivery for delivery in deliveries
				if delivery.event_id in tenant_events
			]
		
		total_deliveries = len(deliveries)
		successful_deliveries = len([d for d in deliveries if d.status == WebhookDeliveryStatus.DELIVERED])
		failed_deliveries = len([d for d in deliveries if d.status == WebhookDeliveryStatus.FAILED])
		pending_deliveries = len([d for d in deliveries if d.status in [WebhookDeliveryStatus.PENDING, WebhookDeliveryStatus.RETRYING]])
		
		return {
			'total_deliveries': total_deliveries,
			'successful_deliveries': successful_deliveries,
			'failed_deliveries': failed_deliveries,
			'pending_deliveries': pending_deliveries,
			'success_rate': (successful_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0,
			'registered_endpoints': len(self.endpoints),
			'active_endpoints': len([e for e in self.endpoints.values() if e.enabled])
		}


# Convenience functions for common billing events
async def emit_payment_succeeded(webhook_system: WebhookSystem, payment_data: Dict[str, Any], tenant_id: str = None):
	"""Emit payment succeeded event"""
	await webhook_system.emit_event(WebhookEventType.PAYMENT_SUCCEEDED.value, payment_data, tenant_id)

async def emit_payment_failed(webhook_system: WebhookSystem, payment_data: Dict[str, Any], tenant_id: str = None):
	"""Emit payment failed event"""
	await webhook_system.emit_event(WebhookEventType.PAYMENT_FAILED.value, payment_data, tenant_id)

async def emit_invoice_paid(webhook_system: WebhookSystem, invoice_data: Dict[str, Any], tenant_id: str = None):
	"""Emit invoice paid event"""
	await webhook_system.emit_event(WebhookEventType.INVOICE_PAID.value, invoice_data, tenant_id)

async def emit_subscription_created(webhook_system: WebhookSystem, subscription_data: Dict[str, Any], tenant_id: str = None):
	"""Emit subscription created event"""
	await webhook_system.emit_event(WebhookEventType.SUBSCRIPTION_CREATED.value, subscription_data, tenant_id)


# Global webhook system
_webhook_system_instance: Optional[WebhookSystem] = None

def get_webhook_system() -> WebhookSystem:
	"""Get global webhook system instance"""
	global _webhook_system_instance
	if _webhook_system_instance is None:
		_webhook_system_instance = WebhookSystem()
	return _webhook_system_instance


__all__ = [
	'WebhookSystem',
	'WebhookEndpoint',
	'WebhookEvent',
	'WebhookEventType',
	'WebhookDeliveryStatus',
	'get_webhook_system',
	'emit_payment_succeeded',
	'emit_payment_failed',
	'emit_invoice_paid',
	'emit_subscription_created'
]