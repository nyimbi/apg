"""
APG Customer Relationship Management - Webhook Management System

This module provides comprehensive webhook management capabilities including
event subscription, delivery tracking, retry logic, transformation,
authentication, and monitoring for real-time integrations.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
import logging
import hmac
import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from uuid import uuid4
import aiohttp
from aiohttp import ClientTimeout
import jinja2

from pydantic import BaseModel, Field, validator, HttpUrl
from uuid_extensions import uuid7str

from .views import CRMResponse, CRMError


logger = logging.getLogger(__name__)


class WebhookEndpoint(BaseModel):
	"""Webhook endpoint configuration"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	webhook_name: str
	description: Optional[str] = None
	endpoint_url: HttpUrl
	http_method: str = "POST"
	headers: Dict[str, str] = Field(default_factory=dict)
	authentication: Dict[str, Any] = Field(default_factory=dict)
	event_types: List[str] = Field(default_factory=list)
	filters: Dict[str, Any] = Field(default_factory=dict)
	transformation_template: Optional[str] = None
	retry_config: Dict[str, Any] = Field(default_factory=lambda: {
		"max_retries": 3,
		"retry_delay_seconds": 60,
		"backoff_multiplier": 2.0,
		"max_delay_seconds": 3600
	})
	timeout_seconds: int = 30
	content_type: str = "application/json"
	secret_key: Optional[str] = None
	signature_header: str = "X-Webhook-Signature"
	verification_enabled: bool = True
	rate_limit: Optional[Dict[str, int]] = None
	tags: List[str] = Field(default_factory=list)
	is_active: bool = True
	failure_count: int = 0
	last_success_at: Optional[datetime] = None
	last_failure_at: Optional[datetime] = None
	last_failure_reason: Optional[str] = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


class WebhookEvent(BaseModel):
	"""Webhook event definition"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	event_category: str  # 'contact', 'lead', 'opportunity', 'account', 'activity'
	event_action: str  # 'created', 'updated', 'deleted', 'status_changed'
	entity_id: str
	entity_type: str
	entity_data: Dict[str, Any] = Field(default_factory=dict)
	previous_data: Optional[Dict[str, Any]] = None
	change_summary: Optional[Dict[str, Any]] = None
	metadata: Dict[str, Any] = Field(default_factory=dict)
	correlation_id: Optional[str] = None
	user_id: Optional[str] = None
	user_agent: Optional[str] = None
	ip_address: Optional[str] = None
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	processed: bool = False


class WebhookDelivery(BaseModel):
	"""Webhook delivery attempt tracking"""
	id: str = Field(default_factory=uuid7str)
	webhook_id: str
	event_id: str
	tenant_id: str
	delivery_url: str
	http_method: str
	headers: Dict[str, str] = Field(default_factory=dict)
	payload: Dict[str, Any] = Field(default_factory=dict)
	response_status: Optional[int] = None
	response_headers: Dict[str, str] = Field(default_factory=dict)
	response_body: Optional[str] = None
	delivery_time_ms: Optional[float] = None
	attempt_number: int = 1
	success: bool = False
	error_message: Optional[str] = None
	retry_after: Optional[int] = None
	scheduled_at: datetime = Field(default_factory=datetime.utcnow)
	delivered_at: Optional[datetime] = None
	next_retry_at: Optional[datetime] = None


class WebhookSubscription(BaseModel):
	"""Webhook subscription management"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	subscription_name: str
	webhook_endpoint_id: str
	event_filters: Dict[str, Any] = Field(default_factory=dict)
	field_filters: List[str] = Field(default_factory=list)  # Fields to include/exclude
	batch_config: Optional[Dict[str, Any]] = None
	rate_limit_config: Optional[Dict[str, Any]] = None
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


class WebhookManager:
	"""Comprehensive webhook management system"""
	
	def __init__(self, db_pool, config: Optional[Dict[str, Any]] = None):
		self.db_pool = db_pool
		self.config = config or {}
		self.session = None
		self.delivery_queue = asyncio.Queue()
		self.retry_queue = asyncio.Queue() 
		self.workers_running = False
		self.jinja_env = jinja2.Environment(loader=jinja2.DictLoader({}))
		self.event_handlers = {}
		self.delivery_stats = {
			'total_deliveries': 0,
			'successful_deliveries': 0,
			'failed_deliveries': 0,
			'retry_deliveries': 0
		}

	async def initialize(self) -> None:
		"""Initialize the webhook manager"""
		try:
			# Initialize HTTP session with connection pooling
			connector = aiohttp.TCPConnector(
				limit=100,  # Total connection pool size
				limit_per_host=30,  # Per-host connection limit
				ttl_dns_cache=300,  # DNS cache TTL
				use_dns_cache=True
			)
			
			timeout = ClientTimeout(total=30, connect=10)
			self.session = aiohttp.ClientSession(
				connector=connector,
				timeout=timeout,
				headers={'User-Agent': 'APG-CRM-Webhook/1.0'}
			)
			
			# Start delivery workers
			await self._start_delivery_workers()
			
			# Load active webhooks
			await self._load_active_webhooks()
			
			logger.info("✅ Webhook manager initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize webhook manager: {str(e)}")
			raise CRMError(f"Webhook manager initialization failed: {str(e)}")

	async def create_webhook_endpoint(
		self,
		tenant_id: str,
		webhook_name: str,
		endpoint_url: str,
		event_types: List[str],
		created_by: str,
		description: Optional[str] = None,
		http_method: str = "POST",
		headers: Optional[Dict[str, str]] = None,
		authentication: Optional[Dict[str, Any]] = None,
		retry_config: Optional[Dict[str, Any]] = None,
		timeout_seconds: int = 30,
		secret_key: Optional[str] = None
	) -> WebhookEndpoint:
		"""Create a new webhook endpoint"""
		try:
			webhook = WebhookEndpoint(
				tenant_id=tenant_id,
				webhook_name=webhook_name,
				description=description,
				endpoint_url=endpoint_url,
				http_method=http_method.upper(),
				headers=headers or {},
				authentication=authentication or {},
				event_types=event_types,
				retry_config=retry_config or {
					"max_retries": 3,
					"retry_delay_seconds": 60,
					"backoff_multiplier": 2.0,
					"max_delay_seconds": 3600
				},
				timeout_seconds=timeout_seconds,
				secret_key=secret_key,
				created_by=created_by
			)

			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_api_webhooks (
						id, tenant_id, webhook_name, description, endpoint_url,
						http_method, headers, authentication, event_types, filters,
						transformation_template, retry_config, timeout_seconds,
						is_active, created_at, created_by
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
				""",
				webhook.id, webhook.tenant_id, webhook.webhook_name, webhook.description,
				str(webhook.endpoint_url), webhook.http_method, json.dumps(webhook.headers),
				json.dumps(webhook.authentication), json.dumps(webhook.event_types),
				json.dumps(webhook.filters), webhook.transformation_template,
				json.dumps(webhook.retry_config), webhook.timeout_seconds,
				webhook.is_active, webhook.created_at, webhook.created_by)

			logger.info(f"Created webhook endpoint: {webhook.webhook_name} for tenant {tenant_id}")
			return webhook

		except Exception as e:
			logger.error(f"Failed to create webhook endpoint: {str(e)}")
			raise CRMError(f"Failed to create webhook endpoint: {str(e)}")

	async def emit_event(
		self,
		tenant_id: str,
		event_type: str,
		event_category: str,
		event_action: str,
		entity_id: str,
		entity_type: str,
		entity_data: Dict[str, Any],
		user_id: Optional[str] = None,
		previous_data: Optional[Dict[str, Any]] = None,
		metadata: Optional[Dict[str, Any]] = None
	) -> WebhookEvent:
		"""Emit a webhook event for processing"""
		try:
			# Create event
			event = WebhookEvent(
				tenant_id=tenant_id,
				event_type=event_type,
				event_category=event_category,
				event_action=event_action,
				entity_id=entity_id,
				entity_type=entity_type,
				entity_data=entity_data,
				previous_data=previous_data,
				metadata=metadata or {},
				user_id=user_id,
				correlation_id=uuid7str()
			)

			# Store event
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_webhook_events (
						id, tenant_id, event_type, event_category, event_action,
						entity_id, entity_type, entity_data, previous_data,
						change_summary, metadata, correlation_id, user_id,
						timestamp, processed
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
				""",
				event.id, event.tenant_id, event.event_type, event.event_category,
				event.event_action, event.entity_id, event.entity_type,
				json.dumps(event.entity_data), json.dumps(event.previous_data),
				json.dumps(event.change_summary), json.dumps(event.metadata),
				event.correlation_id, event.user_id, event.timestamp, event.processed)

			# Queue for delivery
			await self._queue_event_for_delivery(event)

			logger.debug(f"Emitted event: {event_type} for entity {entity_id}")
			return event

		except Exception as e:
			logger.error(f"Failed to emit webhook event: {str(e)}")
			raise CRMError(f"Failed to emit webhook event: {str(e)}")

	async def deliver_webhook(
		self,
		webhook_endpoint: WebhookEndpoint,
		event: WebhookEvent,
		attempt_number: int = 1
	) -> WebhookDelivery:
		"""Deliver webhook to endpoint"""
		try:
			delivery = WebhookDelivery(
				webhook_id=webhook_endpoint.id,
				event_id=event.id,
				tenant_id=event.tenant_id,
				delivery_url=str(webhook_endpoint.endpoint_url),
				http_method=webhook_endpoint.http_method,
				attempt_number=attempt_number
			)

			# Prepare payload
			payload = await self._prepare_webhook_payload(webhook_endpoint, event)
			delivery.payload = payload

			# Prepare headers
			headers = dict(webhook_endpoint.headers)
			headers.update({
				'Content-Type': webhook_endpoint.content_type,
				'X-Webhook-Event': event.event_type,
				'X-Webhook-ID': delivery.id,
				'X-Webhook-Timestamp': str(int(event.timestamp.timestamp())),
				'X-Webhook-Tenant': event.tenant_id
			})

			# Add signature if secret key provided
			if webhook_endpoint.secret_key and webhook_endpoint.verification_enabled:
				signature = await self._generate_signature(
					json.dumps(payload, sort_keys=True),
					webhook_endpoint.secret_key
				)
				headers[webhook_endpoint.signature_header] = signature

			delivery.headers = headers

			# Make HTTP request
			start_time = time.time()
			
			try:
				async with self.session.request(
					method=webhook_endpoint.http_method,
					url=str(webhook_endpoint.endpoint_url),
					json=payload,
					headers=headers,
					timeout=ClientTimeout(total=webhook_endpoint.timeout_seconds)
				) as response:
					
					delivery_time = (time.time() - start_time) * 1000
					delivery.delivery_time_ms = delivery_time
					delivery.response_status = response.status
					delivery.response_headers = dict(response.headers)
					delivery.delivered_at = datetime.utcnow()

					# Read response body
					try:
						response_text = await response.text()
						delivery.response_body = response_text[:10000]  # Limit response size
					except Exception:
						delivery.response_body = "Unable to read response body"

					# Check if delivery was successful
					if 200 <= response.status < 300:
						delivery.success = True
						await self._update_webhook_success(webhook_endpoint.id)
						self.delivery_stats['successful_deliveries'] += 1
					else:
						delivery.success = False
						delivery.error_message = f"HTTP {response.status}: {response.reason}"
						await self._update_webhook_failure(webhook_endpoint.id, delivery.error_message)
						self.delivery_stats['failed_deliveries'] += 1

			except asyncio.TimeoutError:
				delivery.error_message = "Request timeout"
				delivery.success = False
				await self._update_webhook_failure(webhook_endpoint.id, delivery.error_message)
				self.delivery_stats['failed_deliveries'] += 1

			except Exception as e:
				delivery.error_message = f"Connection error: {str(e)}"
				delivery.success = False
				await self._update_webhook_failure(webhook_endpoint.id, delivery.error_message)
				self.delivery_stats['failed_deliveries'] += 1

			# Save delivery record
			await self._save_delivery_record(delivery)

			# Schedule retry if failed and retries remaining
			if not delivery.success and attempt_number < webhook_endpoint.retry_config.get('max_retries', 3):
				await self._schedule_retry(webhook_endpoint, event, delivery, attempt_number + 1)

			self.delivery_stats['total_deliveries'] += 1
			
			logger.debug(f"Webhook delivered to {webhook_endpoint.webhook_name}: Success={delivery.success}")
			return delivery

		except Exception as e:
			logger.error(f"Failed to deliver webhook: {str(e)}")
			raise CRMError(f"Failed to deliver webhook: {str(e)}")

	async def get_webhook_endpoints(
		self,
		tenant_id: str,
		is_active: Optional[bool] = None,
		event_type: Optional[str] = None
	) -> List[Dict[str, Any]]:
		"""Get webhook endpoints for tenant"""
		try:
			async with self.db_pool.acquire() as conn:
				query = """
					SELECT * FROM crm_api_webhooks 
					WHERE tenant_id = $1
				"""
				params = [tenant_id]
				
				if is_active is not None:
					query += " AND is_active = $2"
					params.append(is_active)
					
				if event_type:
					query += f" AND event_types @> $${len(params) + 1}"
					params.append(json.dumps([event_type]))
					
				query += " ORDER BY created_at DESC"
				
				rows = await conn.fetch(query, *params)
				return [dict(row) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get webhook endpoints: {str(e)}")
			raise CRMError(f"Failed to get webhook endpoints: {str(e)}")

	async def get_delivery_history(
		self,
		tenant_id: str,
		webhook_id: Optional[str] = None,
		event_type: Optional[str] = None,
		success: Optional[bool] = None,
		limit: int = 100
	) -> List[Dict[str, Any]]:
		"""Get webhook delivery history"""
		try:
			async with self.db_pool.acquire() as conn:
				query = """
					SELECT wd.*, we.event_type, we.event_category, we.event_action,
						   wh.webhook_name
					FROM crm_webhook_deliveries wd
					LEFT JOIN crm_webhook_events we ON wd.event_id = we.id
					LEFT JOIN crm_api_webhooks wh ON wd.webhook_id = wh.id
					WHERE wd.tenant_id = $1
				"""
				params = [tenant_id]
				
				if webhook_id:
					query += " AND wd.webhook_id = $2"
					params.append(webhook_id)
					
				if event_type:
					query += f" AND we.event_type = $${len(params) + 1}"
					params.append(event_type)
					
				if success is not None:
					query += f" AND wd.success = $${len(params) + 1}"
					params.append(success)
					
				query += f" ORDER BY wd.scheduled_at DESC LIMIT $${len(params) + 1}"
				params.append(limit)
				
				rows = await conn.fetch(query, *params)
				return [dict(row) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get delivery history: {str(e)}")
			raise CRMError(f"Failed to get delivery history: {str(e)}")

	async def get_webhook_metrics(
		self,
		tenant_id: str,
		start_date: datetime,
		end_date: datetime
	) -> Dict[str, Any]:
		"""Get webhook delivery metrics"""
		try:
			async with self.db_pool.acquire() as conn:
				# Get delivery metrics
				metrics_row = await conn.fetchrow("""
					SELECT 
						COUNT(*) as total_deliveries,
						SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_deliveries,
						SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed_deliveries,
						AVG(delivery_time_ms) as avg_delivery_time_ms,
						COUNT(DISTINCT webhook_id) as active_webhooks,
						COUNT(DISTINCT event_id) as unique_events
					FROM crm_webhook_deliveries 
					WHERE tenant_id = $1 AND scheduled_at BETWEEN $2 AND $3
				""", tenant_id, start_date, end_date)

				# Get top webhooks by delivery count
				top_webhooks = await conn.fetch("""
					SELECT wh.webhook_name, COUNT(*) as delivery_count,
						   AVG(wd.delivery_time_ms) as avg_delivery_time,
						   SUM(CASE WHEN wd.success THEN 1 ELSE 0 END)::float / COUNT(*) as success_rate
					FROM crm_webhook_deliveries wd
					JOIN crm_api_webhooks wh ON wd.webhook_id = wh.id
					WHERE wd.tenant_id = $1 AND wd.scheduled_at BETWEEN $2 AND $3
					GROUP BY wh.id, wh.webhook_name
					ORDER BY delivery_count DESC
					LIMIT 10
				""", tenant_id, start_date, end_date)

				# Get event type breakdown
				event_breakdown = await conn.fetch("""
					SELECT we.event_type, COUNT(*) as event_count,
						   SUM(CASE WHEN wd.success THEN 1 ELSE 0 END) as successful_count
					FROM crm_webhook_deliveries wd
					JOIN crm_webhook_events we ON wd.event_id = we.id
					WHERE wd.tenant_id = $1 AND wd.scheduled_at BETWEEN $2 AND $3
					GROUP BY we.event_type
					ORDER BY event_count DESC
				""", tenant_id, start_date, end_date)

				return {
					'total_deliveries': metrics_row['total_deliveries'] or 0,
					'successful_deliveries': metrics_row['successful_deliveries'] or 0,
					'failed_deliveries': metrics_row['failed_deliveries'] or 0,
					'success_rate': (metrics_row['successful_deliveries'] or 0) / max(metrics_row['total_deliveries'] or 1, 1),
					'avg_delivery_time_ms': float(metrics_row['avg_delivery_time_ms'] or 0),
					'active_webhooks': metrics_row['active_webhooks'] or 0,
					'unique_events': metrics_row['unique_events'] or 0,
					'top_webhooks': [dict(row) for row in top_webhooks],
					'event_breakdown': [dict(row) for row in event_breakdown],
					'period': {
						'start_date': start_date.isoformat(),
						'end_date': end_date.isoformat()
					}
				}

		except Exception as e:
			logger.error(f"Failed to get webhook metrics: {str(e)}")
			raise CRMError(f"Failed to get webhook metrics: {str(e)}")

	async def test_webhook_endpoint(
		self,
		tenant_id: str,
		webhook_id: str
	) -> Dict[str, Any]:
		"""Test webhook endpoint connectivity"""
		try:
			# Get webhook configuration
			async with self.db_pool.acquire() as conn:
				webhook_row = await conn.fetchrow("""
					SELECT * FROM crm_api_webhooks 
					WHERE id = $1 AND tenant_id = $2
				""", webhook_id, tenant_id)

			if not webhook_row:
				raise CRMError("Webhook not found")

			webhook_config = dict(webhook_row)
			
			# Create test event
			test_event = WebhookEvent(
				tenant_id=tenant_id,
				event_type="webhook.test",
				event_category="system",
				event_action="test",
				entity_id="test_entity",
				entity_type="test",
				entity_data={"message": "This is a test webhook delivery"},
				metadata={"test": True, "timestamp": datetime.utcnow().isoformat()}
			)

			# Create webhook endpoint object
			webhook_endpoint = WebhookEndpoint(**webhook_config)

			# Attempt delivery
			delivery = await self.deliver_webhook(webhook_endpoint, test_event)

			return {
				"webhook_id": webhook_id,
				"test_successful": delivery.success,
				"response_status": delivery.response_status,
				"delivery_time_ms": delivery.delivery_time_ms,
				"error_message": delivery.error_message,
				"response_body": delivery.response_body[:500] if delivery.response_body else None
			}

		except Exception as e:
			logger.error(f"Failed to test webhook endpoint: {str(e)}")
			raise CRMError(f"Failed to test webhook endpoint: {str(e)}")

	# Helper methods

	async def _start_delivery_workers(self) -> None:
		"""Start background workers for webhook delivery"""
		self.workers_running = True
		
		# Start delivery worker
		asyncio.create_task(self._delivery_worker())
		
		# Start retry worker
		asyncio.create_task(self._retry_worker())
		
		logger.info("Started webhook delivery workers")

	async def _delivery_worker(self) -> None:
		"""Background worker for processing webhook deliveries"""
		while self.workers_running:
			try:
				# Get next delivery task
				delivery_task = await asyncio.wait_for(
					self.delivery_queue.get(), 
					timeout=1.0
				)
				
				webhook_endpoint, event = delivery_task
				
				# Deliver webhook
				await self.deliver_webhook(webhook_endpoint, event)
				
			except asyncio.TimeoutError:
				continue
			except Exception as e:
				logger.error(f"Delivery worker error: {str(e)}")
				await asyncio.sleep(1)

	async def _retry_worker(self) -> None:
		"""Background worker for processing webhook retries"""
		while self.workers_running:
			try:
				# Check for retries
				await self._process_pending_retries()
				
				# Wait before next check
				await asyncio.sleep(30)
				
			except Exception as e:
				logger.error(f"Retry worker error: {str(e)}")
				await asyncio.sleep(30)

	async def _load_active_webhooks(self) -> None:
		"""Load active webhook configurations"""
		try:
			async with self.db_pool.acquire() as conn:
				rows = await conn.fetch("""
					SELECT * FROM crm_api_webhooks 
					WHERE is_active = true
				""")
				
			logger.info(f"Loaded {len(rows)} active webhook endpoints")
			
		except Exception as e:
			logger.error(f"Failed to load active webhooks: {str(e)}")

	async def _queue_event_for_delivery(self, event: WebhookEvent) -> None:
		"""Queue event for webhook delivery"""
		try:
			# Find matching webhook endpoints
			async with self.db_pool.acquire() as conn:
				webhook_rows = await conn.fetch("""
					SELECT * FROM crm_api_webhooks 
					WHERE tenant_id = $1 AND is_active = true 
					AND (event_types @> $2 OR event_types @> $3)
				""", event.tenant_id, 
				json.dumps([event.event_type]), 
				json.dumps([f"{event.event_category}.*"]))

			# Queue delivery for each matching webhook
			for webhook_row in webhook_rows:
				webhook_endpoint = WebhookEndpoint(**dict(webhook_row))
				
				# Check if event matches filters
				if await self._event_matches_filters(event, webhook_endpoint.filters):
					await self.delivery_queue.put((webhook_endpoint, event))

		except Exception as e:
			logger.error(f"Failed to queue event for delivery: {str(e)}")

	async def _event_matches_filters(
		self, 
		event: WebhookEvent, 
		filters: Dict[str, Any]
	) -> bool:
		"""Check if event matches webhook filters"""
		if not filters:
			return True
			
		# Entity type filter
		if 'entity_types' in filters:
			if event.entity_type not in filters['entity_types']:
				return False
		
		# Field value filters
		if 'field_conditions' in filters:
			for condition in filters['field_conditions']:
				field_path = condition['field']
				operator = condition['operator']
				expected_value = condition['value']
				
				# Get field value from event data
				actual_value = self._get_nested_value(event.entity_data, field_path)
				
				# Apply condition
				if not self._evaluate_condition(actual_value, operator, expected_value):
					return False
		
		return True

	async def _prepare_webhook_payload(
		self, 
		webhook_endpoint: WebhookEndpoint, 
		event: WebhookEvent
	) -> Dict[str, Any]:
		"""Prepare webhook payload with transformations"""
		base_payload = {
			'event': {
				'id': event.id,
				'type': event.event_type,
				'category': event.event_category,
				'action': event.event_action,
				'timestamp': event.timestamp.isoformat(),
				'correlation_id': event.correlation_id
			},
			'data': {
				'entity_id': event.entity_id,
				'entity_type': event.entity_type,
				'current': event.entity_data,
				'previous': event.previous_data,
				'changes': event.change_summary
			},
			'metadata': event.metadata,
			'tenant_id': event.tenant_id
		}

		# Apply transformation template if provided
		if webhook_endpoint.transformation_template:
			try:
				template = self.jinja_env.from_string(webhook_endpoint.transformation_template)
				transformed_payload = template.render(
					event=event.model_dump(),
					payload=base_payload
				)
				return json.loads(transformed_payload)
			except Exception as e:
				logger.warning(f"Transformation template failed: {str(e)}")
				return base_payload

		return base_payload

	async def _generate_signature(self, payload: str, secret_key: str) -> str:
		"""Generate HMAC signature for webhook verification"""
		signature = hmac.new(
			secret_key.encode('utf-8'),
			payload.encode('utf-8'),
			hashlib.sha256
		).hexdigest()
		return f"sha256={signature}"

	async def _save_delivery_record(self, delivery: WebhookDelivery) -> None:
		"""Save webhook delivery record to database"""
		try:
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_webhook_deliveries (
						id, webhook_id, event_id, tenant_id, delivery_url,
						http_method, headers, payload, response_status,
						response_headers, response_body, delivery_time_ms,
						attempt_number, success, error_message, scheduled_at,
						delivered_at, next_retry_at
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
				""",
				delivery.id, delivery.webhook_id, delivery.event_id, delivery.tenant_id,
				delivery.delivery_url, delivery.http_method, json.dumps(delivery.headers),
				json.dumps(delivery.payload), delivery.response_status,
				json.dumps(delivery.response_headers), delivery.response_body,
				delivery.delivery_time_ms, delivery.attempt_number, delivery.success,
				delivery.error_message, delivery.scheduled_at, delivery.delivered_at,
				delivery.next_retry_at)

		except Exception as e:
			logger.error(f"Failed to save delivery record: {str(e)}")

	async def _schedule_retry(
		self, 
		webhook_endpoint: WebhookEndpoint, 
		event: WebhookEvent, 
		failed_delivery: WebhookDelivery,
		next_attempt: int
	) -> None:
		"""Schedule webhook delivery retry"""
		try:
			retry_config = webhook_endpoint.retry_config
			base_delay = retry_config.get('retry_delay_seconds', 60)
			backoff_multiplier = retry_config.get('backoff_multiplier', 2.0)
			max_delay = retry_config.get('max_delay_seconds', 3600)
			
			# Calculate delay with exponential backoff
			delay_seconds = min(
				base_delay * (backoff_multiplier ** (next_attempt - 2)),
				max_delay
			)
			
			next_retry_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
			
			# Update failed delivery record with retry schedule
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					UPDATE crm_webhook_deliveries 
					SET next_retry_at = $1
					WHERE id = $2
				""", next_retry_at, failed_delivery.id)

			self.delivery_stats['retry_deliveries'] += 1
			logger.debug(f"Scheduled retry for webhook {webhook_endpoint.webhook_name} in {delay_seconds} seconds")

		except Exception as e:
			logger.error(f"Failed to schedule retry: {str(e)}")

	async def _process_pending_retries(self) -> None:
		"""Process webhooks scheduled for retry"""
		try:
			async with self.db_pool.acquire() as conn:
				retry_rows = await conn.fetch("""
					SELECT wd.*, wh.*, we.*
					FROM crm_webhook_deliveries wd
					JOIN crm_api_webhooks wh ON wd.webhook_id = wh.id
					JOIN crm_webhook_events we ON wd.event_id = we.id
					WHERE wd.next_retry_at IS NOT NULL 
					AND wd.next_retry_at <= NOW()
					AND NOT wd.success
					AND wd.attempt_number < (wh.retry_config->>'max_retries')::int
					LIMIT 100
				""")

			for row in retry_rows:
				try:
					# Create webhook endpoint and event objects
					webhook_data = {k: v for k, v in row.items() if k in WebhookEndpoint.__fields__}
					event_data = {k: v for k, v in row.items() if k in WebhookEvent.__fields__}
					
					webhook_endpoint = WebhookEndpoint(**webhook_data)
					event = WebhookEvent(**event_data)
					
					# Attempt delivery
					await self.deliver_webhook(webhook_endpoint, event, row['attempt_number'] + 1)
					
					# Clear retry schedule
					await conn.execute("""
						UPDATE crm_webhook_deliveries 
						SET next_retry_at = NULL
						WHERE id = $1
					""", row['id'])

				except Exception as e:
					logger.error(f"Failed to process retry for delivery {row['id']}: {str(e)}")

		except Exception as e:
			logger.error(f"Failed to process pending retries: {str(e)}")

	async def _update_webhook_success(self, webhook_id: str) -> None:
		"""Update webhook success metrics"""
		try:
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					UPDATE crm_api_webhooks 
					SET last_success_at = NOW(), failure_count = 0
					WHERE id = $1
				""", webhook_id)
		except Exception as e:
			logger.error(f"Failed to update webhook success: {str(e)}")

	async def _update_webhook_failure(self, webhook_id: str, error_message: str) -> None:
		"""Update webhook failure metrics"""
		try:
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					UPDATE crm_api_webhooks 
					SET last_failure_at = NOW(), 
						failure_count = failure_count + 1,
						last_failure_reason = $2
					WHERE id = $1
				""", webhook_id, error_message)
		except Exception as e:
			logger.error(f"Failed to update webhook failure: {str(e)}")

	def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
		"""Get nested value from dictionary using dot notation"""
		keys = field_path.split('.')
		value = data
		
		for key in keys:
			if isinstance(value, dict) and key in value:
				value = value[key]
			else:
				return None
		
		return value

	def _evaluate_condition(self, actual_value: Any, operator: str, expected_value: Any) -> bool:
		"""Evaluate field condition"""
		if operator == 'equals':
			return actual_value == expected_value
		elif operator == 'not_equals':
			return actual_value != expected_value
		elif operator == 'contains':
			return expected_value in str(actual_value) if actual_value else False
		elif operator == 'in':
			return actual_value in expected_value if isinstance(expected_value, list) else False
		elif operator == 'greater_than':
			return float(actual_value) > float(expected_value) if actual_value else False
		elif operator == 'less_than':
			return float(actual_value) < float(expected_value) if actual_value else False
		else:
			return True

	async def shutdown(self) -> None:
		"""Shutdown webhook manager"""
		self.workers_running = False
		
		if self.session:
			await self.session.close()
		
		logger.info("Webhook manager shut down successfully")