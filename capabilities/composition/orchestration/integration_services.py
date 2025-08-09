#!/usr/bin/env python3
"""
APG Workflow Orchestration Integration Services

Comprehensive integration services for APG capabilities, external systems,
webhooks, authentication, and audit functionality.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum

import aiohttp
import asyncpg
from pydantic import BaseModel, ValidationError, Field, ConfigDict
from pydantic.types import UUID4
from sqlalchemy.ext.asyncio import AsyncSession
from uuid_extensions import uuid7str

# APG Framework imports
from apg.framework.auth_rbac import APGAuth, APGRole, APGPermission
from apg.framework.audit_compliance import APGAuditLogger, AuditEvent
from apg.framework.base_service import APGBaseService
from apg.framework.database import APGDatabase
from apg.framework.messaging import APGEventBus, APGMessage
from apg.framework.security import APGSecurity, APGEncryption

# Local imports
from .models import *
from .database import WorkflowDB, WorkflowInstanceDB, TaskExecutionDB
from .apg_integration import APGIntegration


logger = logging.getLogger(__name__)


class IntegrationType(str, Enum):
	"""Types of integrations supported."""
	
	APG_CAPABILITY = "apg_capability"
	WEBHOOK = "webhook"
	REST_API = "rest_api"
	GRAPHQL = "graphql"
	DATABASE = "database"
	MESSAGE_QUEUE = "message_queue"
	FILE_SYSTEM = "file_system"
	CLOUD_SERVICE = "cloud_service"


class AuthenticationType(str, Enum):
	"""Types of authentication methods."""
	
	NONE = "none"
	API_KEY = "api_key"
	BEARER_TOKEN = "bearer_token"
	BASIC_AUTH = "basic_auth"
	OAUTH2 = "oauth2"
	JWT = "jwt"
	CERTIFICATE = "certificate"
	CUSTOM = "custom"


class WebhookStatus(str, Enum):
	"""Webhook delivery status."""
	
	PENDING = "pending"
	DELIVERED = "delivered"
	FAILED = "failed"
	RETRYING = "retrying"
	EXPIRED = "expired"


@dataclass
class IntegrationConfig:
	"""Configuration for an integration."""
	
	id: str
	name: str
	type: IntegrationType
	endpoint: str
	auth_type: AuthenticationType
	auth_config: Dict[str, Any]
	headers: Dict[str, str]
	timeout_seconds: int = 30
	retry_attempts: int = 3
	retry_delay_seconds: int = 5
	tenant_id: str = ""
	is_active: bool = True
	metadata: Dict[str, Any] = None


class IntegrationRequest(BaseModel):
	"""Request model for integration calls."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	integration_id: str
	method: str = "POST"
	endpoint: Optional[str] = None
	payload: Dict[str, Any]
	headers: Optional[Dict[str, str]] = None
	timeout_seconds: Optional[int] = None
	tenant_id: str
	workflow_instance_id: Optional[str] = None
	task_execution_id: Optional[str] = None
	created_at: datetime = Field(default_factory=datetime.utcnow)


class IntegrationResponse(BaseModel):
	"""Response model for integration calls."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	request_id: str
	status_code: int
	response_data: Dict[str, Any]
	response_headers: Dict[str, str]
	duration_ms: int
	success: bool
	error_message: Optional[str] = None
	timestamp: datetime = Field(default_factory=datetime.utcnow)


class WebhookConfig(BaseModel):
	"""Configuration for webhook subscriptions."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	url: str
	secret: Optional[str] = None
	events: List[str]
	tenant_id: str
	is_active: bool = True
	retry_attempts: int = 3
	retry_delay_seconds: int = 5
	timeout_seconds: int = 30
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class WebhookDelivery(BaseModel):
	"""Webhook delivery tracking."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	webhook_id: str
	event_type: str
	payload: Dict[str, Any]
	status: WebhookStatus = WebhookStatus.PENDING
	attempts: int = 0
	last_attempt_at: Optional[datetime] = None
	response_code: Optional[int] = None
	response_body: Optional[str] = None
	error_message: Optional[str] = None
	expires_at: datetime
	created_at: datetime = Field(default_factory=datetime.utcnow)


class APGCapabilityIntegration(BaseModel):
	"""Integration with APG capabilities."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	capability_name: str
	endpoint: str
	method: str = "POST"
	input_mapping: Dict[str, str]
	output_mapping: Dict[str, str]
	auth_required: bool = True
	tenant_isolation: bool = True
	audit_enabled: bool = True


class IntegrationService(APGBaseService):
	"""Service for managing integrations with external systems and APG capabilities."""
	
	def __init__(self):
		super().__init__()
		self.auth = APGAuth()
		self.audit = APGAuditLogger()
		self.security = APGSecurity()
		self.encryption = APGEncryption()
		self.event_bus = APGEventBus()
		self.database = APGDatabase()
		
		self.integrations: Dict[str, IntegrationConfig] = {}
		self.webhooks: Dict[str, WebhookConfig] = {}
		self.apg_capabilities: Dict[str, APGCapabilityIntegration] = {}
		
		# HTTP session for external calls
		self.http_session: Optional[aiohttp.ClientSession] = None
		
		# Initialize APG capability integrations
		self._initialize_apg_integrations()
	
	async def start(self):
		"""Start the integration service."""
		await super().start()
		
		# Initialize HTTP session
		self.http_session = aiohttp.ClientSession(
			timeout=aiohttp.ClientTimeout(total=60),
			connector=aiohttp.TCPConnector(limit=100, limit_per_host=10)
		)
		
		# Load configurations from database
		await self._load_configurations()
		
		# Start webhook delivery worker
		asyncio.create_task(self._webhook_delivery_worker())
		
		logger.info("Integration service started")
	
	async def stop(self):
		"""Stop the integration service."""
		if self.http_session:
			await self.http_session.close()
		
		await super().stop()
		logger.info("Integration service stopped")
	
	# Integration Management
	
	async def register_integration(self, config: IntegrationConfig) -> str:
		"""Register a new integration."""
		try:
			# Validate configuration
			await self._validate_integration_config(config)
			
			# Encrypt sensitive data
			config.auth_config = await self._encrypt_auth_config(config.auth_config)
			
			# Store in database
			async with self.database.get_session() as session:
				integration_data = {
					'id': config.id,
					'name': config.name,
					'type': config.type.value,
					'endpoint': config.endpoint,
					'auth_type': config.auth_type.value,
					'auth_config': json.dumps(config.auth_config),
					'headers': json.dumps(config.headers),
					'timeout_seconds': config.timeout_seconds,
					'retry_attempts': config.retry_attempts,
					'retry_delay_seconds': config.retry_delay_seconds,
					'tenant_id': config.tenant_id,
					'is_active': config.is_active,
					'metadata': json.dumps(config.metadata or {}),
					'created_at': datetime.utcnow(),
					'updated_at': datetime.utcnow()
				}
				
				await session.execute(
					"""
					INSERT INTO wo_integrations (
						id, name, type, endpoint, auth_type, auth_config,
						headers, timeout_seconds, retry_attempts, retry_delay_seconds,
						tenant_id, is_active, metadata, created_at, updated_at
					) VALUES (
						:id, :name, :type, :endpoint, :auth_type, :auth_config,
						:headers, :timeout_seconds, :retry_attempts, :retry_delay_seconds,
						:tenant_id, :is_active, :metadata, :created_at, :updated_at
					)
					""",
					integration_data
				)
				await session.commit()
			
			# Store in memory
			self.integrations[config.id] = config
			
			# Audit log
			await self.audit.log_event(
				AuditEvent(
					event_type='integration_registered',
					resource_type='integration',
					resource_id=config.id,
					tenant_id=config.tenant_id,
					metadata={'name': config.name, 'type': config.type.value}
				)
			)
			
			logger.info(f"Integration registered: {config.name} ({config.id})")
			return config.id
			
		except Exception as e:
			logger.error(f"Failed to register integration: {str(e)}")
			raise
	
	async def update_integration(self, integration_id: str, updates: Dict[str, Any]) -> bool:
		"""Update an existing integration."""
		try:
			if integration_id not in self.integrations:
				raise ValueError(f"Integration not found: {integration_id}")
			
			config = self.integrations[integration_id]
			
			# Update configuration
			for key, value in updates.items():
				if hasattr(config, key):
					setattr(config, key, value)
			
			# Encrypt sensitive data if updated
			if 'auth_config' in updates:
				config.auth_config = await self._encrypt_auth_config(config.auth_config)
			
			# Update in database
			async with self.database.get_session() as session:
				await session.execute(
					"""
					UPDATE wo_integrations SET
						name = :name, endpoint = :endpoint, auth_type = :auth_type,
						auth_config = :auth_config, headers = :headers,
						timeout_seconds = :timeout_seconds, retry_attempts = :retry_attempts,
						retry_delay_seconds = :retry_delay_seconds, is_active = :is_active,
						metadata = :metadata, updated_at = :updated_at
					WHERE id = :id
					""",
					{
						'id': integration_id,
						'name': config.name,
						'endpoint': config.endpoint,
						'auth_type': config.auth_type.value,
						'auth_config': json.dumps(config.auth_config),
						'headers': json.dumps(config.headers),
						'timeout_seconds': config.timeout_seconds,
						'retry_attempts': config.retry_attempts,
						'retry_delay_seconds': config.retry_delay_seconds,
						'is_active': config.is_active,
						'metadata': json.dumps(config.metadata or {}),
						'updated_at': datetime.utcnow()
					}
				)
				await session.commit()
			
			# Audit log
			await self.audit.log_event(
				AuditEvent(
					event_type='integration_updated',
					resource_type='integration',
					resource_id=integration_id,
					tenant_id=config.tenant_id,
					metadata={'updates': list(updates.keys())}
				)
			)
			
			logger.info(f"Integration updated: {integration_id}")
			return True
			
		except Exception as e:
			logger.error(f"Failed to update integration: {str(e)}")
			raise
	
	async def delete_integration(self, integration_id: str) -> bool:
		"""Delete an integration."""
		try:
			if integration_id not in self.integrations:
				raise ValueError(f"Integration not found: {integration_id}")
			
			config = self.integrations[integration_id]
			
			# Delete from database
			async with self.database.get_session() as session:
				await session.execute(
					"DELETE FROM wo_integrations WHERE id = :id",
					{'id': integration_id}
				)
				await session.commit()
			
			# Remove from memory
			del self.integrations[integration_id]
			
			# Audit log
			await self.audit.log_event(
				AuditEvent(
					event_type='integration_deleted',
					resource_type='integration',
					resource_id=integration_id,
					tenant_id=config.tenant_id,
					metadata={'name': config.name}
				)
			)
			
			logger.info(f"Integration deleted: {integration_id}")
			return True
			
		except Exception as e:
			logger.error(f"Failed to delete integration: {str(e)}")
			raise
	
	# External System Integration
	
	async def call_external_system(self, request: IntegrationRequest) -> IntegrationResponse:
		"""Make a call to an external system."""
		start_time = datetime.utcnow()
		
		try:
			# Get integration config
			if request.integration_id not in self.integrations:
				raise ValueError(f"Integration not found: {request.integration_id}")
			
			config = self.integrations[request.integration_id]
			
			# Check if integration is active
			if not config.is_active:
				raise ValueError(f"Integration is inactive: {request.integration_id}")
			
			# Check tenant access
			await self._check_tenant_access(request.tenant_id, config.tenant_id)
			
			# Prepare request
			url = request.endpoint or config.endpoint
			headers = {**config.headers, **(request.headers or {})}
			timeout = request.timeout_seconds or config.timeout_seconds
			
			# Add authentication
			headers = await self._add_authentication(headers, config)
			
			# Make the HTTP request
			async with self.http_session.request(
				method=request.method,
				url=url,
				json=request.payload,
				headers=headers,
				timeout=aiohttp.ClientTimeout(total=timeout)
			) as response:
				response_data = await response.json() if response.content_type == 'application/json' else {'text': await response.text()}
				response_headers = dict(response.headers)
				
				duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
				
				result = IntegrationResponse(
					request_id=request.id,
					status_code=response.status,
					response_data=response_data,
					response_headers=response_headers,
					duration_ms=duration_ms,
					success=200 <= response.status < 400
				)
				
				# Store request/response for audit
				await self._store_integration_audit(request, result)
				
				return result
		
		except Exception as e:
			duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
			
			result = IntegrationResponse(
				request_id=request.id,
				status_code=0,
				response_data={},
				response_headers={},
				duration_ms=duration_ms,
				success=False,
				error_message=str(e)
			)
			
			# Store failed request for audit
			await self._store_integration_audit(request, result)
			
			logger.error(f"External system call failed: {str(e)}")
			raise
	
	async def call_external_system_with_retry(self, request: IntegrationRequest) -> IntegrationResponse:
		"""Make a call to an external system with retry logic."""
		config = self.integrations.get(request.integration_id)
		if not config:
			raise ValueError(f"Integration not found: {request.integration_id}")
		
		last_exception = None
		
		for attempt in range(config.retry_attempts + 1):
			try:
				return await self.call_external_system(request)
			
			except Exception as e:
				last_exception = e
				
				if attempt < config.retry_attempts:
					await asyncio.sleep(config.retry_delay_seconds * (2 ** attempt))  # Exponential backoff
					logger.warning(f"Retrying external system call (attempt {attempt + 1}): {str(e)}")
				else:
					logger.error(f"External system call failed after {config.retry_attempts + 1} attempts")
		
		raise last_exception
	
	# APG Capability Integration
	
	async def call_apg_capability(self, capability_name: str, input_data: Dict[str, Any], 
								  tenant_id: str, user_id: str = None) -> Dict[str, Any]:
		"""Call an APG capability."""
		try:
			if capability_name not in self.apg_capabilities:
				raise ValueError(f"APG capability not found: {capability_name}")
			
			capability = self.apg_capabilities[capability_name]
			
			# Check authentication if required
			if capability.auth_required and not user_id:
				raise ValueError("Authentication required for this capability")
			
			# Map input data
			mapped_input = self._map_data(input_data, capability.input_mapping)
			
			# Make the capability call
			request = IntegrationRequest(
				integration_id=f"apg_{capability_name}",
				method=capability.method,
				endpoint=capability.endpoint,
				payload=mapped_input,
				tenant_id=tenant_id
			)
			
			response = await self.call_external_system(request)
			
			if not response.success:
				raise ValueError(f"APG capability call failed: {response.error_message}")
			
			# Map output data
			mapped_output = self._map_data(response.response_data, capability.output_mapping)
			
			# Audit if enabled
			if capability.audit_enabled:
				await self.audit.log_event(
					AuditEvent(
						event_type='apg_capability_called',
						resource_type='capability',
						resource_id=capability_name,
						tenant_id=tenant_id,
						user_id=user_id,
						metadata={'input_keys': list(input_data.keys()), 'success': True}
					)
				)
			
			return mapped_output
			
		except Exception as e:
			# Audit failure if enabled
			if capability_name in self.apg_capabilities and self.apg_capabilities[capability_name].audit_enabled:
				await self.audit.log_event(
					AuditEvent(
						event_type='apg_capability_call_failed',
						resource_type='capability',
						resource_id=capability_name,
						tenant_id=tenant_id,
						user_id=user_id,
						metadata={'error': str(e)}
					)
				)
			
			logger.error(f"APG capability call failed: {str(e)}")
			raise
	
	# Webhook Management
	
	async def register_webhook(self, config: WebhookConfig) -> str:
		"""Register a webhook subscription."""
		try:
			# Validate webhook config
			await self._validate_webhook_config(config)
			
			# Store in database
			async with self.database.get_session() as session:
				webhook_data = {
					'id': config.id,
					'name': config.name,
					'url': config.url,
					'secret': await self.encryption.encrypt(config.secret) if config.secret else None,
					'events': json.dumps(config.events),
					'tenant_id': config.tenant_id,
					'is_active': config.is_active,
					'retry_attempts': config.retry_attempts,
					'retry_delay_seconds': config.retry_delay_seconds,
					'timeout_seconds': config.timeout_seconds,
					'created_at': config.created_at,
					'updated_at': config.updated_at
				}
				
				await session.execute(
					"""
					INSERT INTO wo_webhooks (
						id, name, url, secret, events, tenant_id, is_active,
						retry_attempts, retry_delay_seconds, timeout_seconds,
						created_at, updated_at
					) VALUES (
						:id, :name, :url, :secret, :events, :tenant_id, :is_active,
						:retry_attempts, :retry_delay_seconds, :timeout_seconds,
						:created_at, :updated_at
					)
					""",
					webhook_data
				)
				await session.commit()
			
			# Store in memory
			self.webhooks[config.id] = config
			
			# Audit log
			await self.audit.log_event(
				AuditEvent(
					event_type='webhook_registered',
					resource_type='webhook',
					resource_id=config.id,
					tenant_id=config.tenant_id,
					metadata={'name': config.name, 'events': config.events}
				)
			)
			
			logger.info(f"Webhook registered: {config.name} ({config.id})")
			return config.id
			
		except Exception as e:
			logger.error(f"Failed to register webhook: {str(e)}")
			raise
	
	async def trigger_webhook(self, event_type: str, payload: Dict[str, Any], tenant_id: str):
		"""Trigger webhooks for a specific event."""
		try:
			# Find matching webhooks
			matching_webhooks = [
				webhook for webhook in self.webhooks.values()
				if (webhook.is_active and 
					event_type in webhook.events and 
					webhook.tenant_id == tenant_id)
			]
			
			if not matching_webhooks:
				logger.debug(f"No webhooks found for event: {event_type}")
				return
			
			# Create delivery records
			deliveries = []
			for webhook in matching_webhooks:
				delivery = WebhookDelivery(
					webhook_id=webhook.id,
					event_type=event_type,
					payload=payload,
					expires_at=datetime.utcnow() + timedelta(hours=24)
				)
				deliveries.append(delivery)
			
			# Store deliveries in database
			async with self.database.get_session() as session:
				for delivery in deliveries:
					delivery_data = {
						'id': delivery.id,
						'webhook_id': delivery.webhook_id,
						'event_type': delivery.event_type,
						'payload': json.dumps(delivery.payload),
						'status': delivery.status.value,
						'attempts': delivery.attempts,
						'expires_at': delivery.expires_at,
						'created_at': delivery.created_at
					}
					
					await session.execute(
						"""
						INSERT INTO wo_webhook_deliveries (
							id, webhook_id, event_type, payload, status,
							attempts, expires_at, created_at
						) VALUES (
							:id, :webhook_id, :event_type, :payload, :status,
							:attempts, :expires_at, :created_at
						)
						""",
						delivery_data
					)
				
				await session.commit()
			
			logger.info(f"Triggered {len(deliveries)} webhooks for event: {event_type}")
			
		except Exception as e:
			logger.error(f"Failed to trigger webhooks: {str(e)}")
			raise
	
	async def _webhook_delivery_worker(self):
		"""Background worker for webhook deliveries."""
		while True:
			try:
				# Get pending deliveries
				async with self.database.get_session() as session:
					result = await session.execute(
						"""
						SELECT id, webhook_id, event_type, payload, attempts, expires_at
						FROM wo_webhook_deliveries
						WHERE status = 'pending' AND expires_at > :now
						ORDER BY created_at ASC
						LIMIT 10
						""",
						{'now': datetime.utcnow()}
					)
					
					pending_deliveries = result.fetchall()
				
				# Process deliveries
				for delivery_row in pending_deliveries:
					await self._deliver_webhook(delivery_row)
				
				# Wait before next iteration
				await asyncio.sleep(5)
				
			except Exception as e:
				logger.error(f"Webhook delivery worker error: {str(e)}")
				await asyncio.sleep(10)
	
	async def _deliver_webhook(self, delivery_row):
		"""Deliver a single webhook."""
		try:
			delivery_id = delivery_row.id
			webhook_id = delivery_row.webhook_id
			payload = json.loads(delivery_row.payload)
			
			# Get webhook config
			if webhook_id not in self.webhooks:
				# Mark as failed
				await self._update_webhook_delivery_status(
					delivery_id, WebhookStatus.FAILED, 
					error_message="Webhook configuration not found"
				)
				return
			
			webhook = self.webhooks[webhook_id]
			
			# Prepare request
			headers = {'Content-Type': 'application/json'}
			
			# Add signature if secret is configured
			if webhook.secret:
				signature = self._generate_webhook_signature(payload, webhook.secret)
				headers['X-Webhook-Signature'] = signature
			
			# Update attempt count
			await self._increment_webhook_attempts(delivery_id)
			
			# Make delivery
			async with self.http_session.post(
				webhook.url,
				json=payload,
				headers=headers,
				timeout=aiohttp.ClientTimeout(total=webhook.timeout_seconds)
			) as response:
				
				if 200 <= response.status < 300:
					await self._update_webhook_delivery_status(
						delivery_id, WebhookStatus.DELIVERED,
						response_code=response.status,
						response_body=await response.text()
					)
					logger.debug(f"Webhook delivered successfully: {delivery_id}")
				else:
					await self._update_webhook_delivery_status(
						delivery_id, WebhookStatus.FAILED,
						response_code=response.status,
						response_body=await response.text(),
						error_message=f"HTTP {response.status}"
					)
					logger.warning(f"Webhook delivery failed: {delivery_id} (HTTP {response.status})")
		
		except Exception as e:
			await self._update_webhook_delivery_status(
				delivery_id, WebhookStatus.FAILED,
				error_message=str(e)
			)
			logger.error(f"Webhook delivery error: {str(e)}")
	
	# Authentication and Security
	
	async def _add_authentication(self, headers: Dict[str, str], config: IntegrationConfig) -> Dict[str, str]:
		"""Add authentication headers based on config."""
		auth_config = await self._decrypt_auth_config(config.auth_config)
		
		if config.auth_type == AuthenticationType.API_KEY:
			headers[auth_config.get('header', 'X-API-Key')] = auth_config['key']
		
		elif config.auth_type == AuthenticationType.BEARER_TOKEN:
			headers['Authorization'] = f"Bearer {auth_config['token']}"
		
		elif config.auth_type == AuthenticationType.BASIC_AUTH:
			import base64
			credentials = f"{auth_config['username']}:{auth_config['password']}"
			encoded = base64.b64encode(credentials.encode()).decode()
			headers['Authorization'] = f"Basic {encoded}"
		
		elif config.auth_type == AuthenticationType.JWT:
			headers['Authorization'] = f"Bearer {auth_config['token']}"
		
		elif config.auth_type == AuthenticationType.CUSTOM:
			for key, value in auth_config.get('headers', {}).items():
				headers[key] = value
		
		return headers
	
	async def _encrypt_auth_config(self, auth_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Encrypt sensitive authentication configuration."""
		if not auth_config:
			return {}
		
		encrypted_config = {}
		for key, value in auth_config.items():
			if key in ['password', 'secret', 'token', 'key', 'private_key']:
				encrypted_config[key] = await self.encryption.encrypt(str(value))
			else:
				encrypted_config[key] = value
		
		return encrypted_config
	
	async def _decrypt_auth_config(self, auth_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Decrypt sensitive authentication configuration."""
		if not auth_config:
			return {}
		
		decrypted_config = {}
		for key, value in auth_config.items():
			if key in ['password', 'secret', 'token', 'key', 'private_key']:
				decrypted_config[key] = await self.encryption.decrypt(value)
			else:
				decrypted_config[key] = value
		
		return decrypted_config
	
	def _generate_webhook_signature(self, payload: Dict[str, Any], secret: str) -> str:
		"""Generate webhook signature for verification."""
		payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
		return hmac.new(
			secret.encode('utf-8'),
			payload_bytes,
			hashlib.sha256
		).hexdigest()
	
	# Utility Methods
	
	async def _validate_integration_config(self, config: IntegrationConfig):
		"""Validate integration configuration."""
		if not config.name or not config.endpoint:
			raise ValueError("Integration name and endpoint are required")
		
		if config.auth_type != AuthenticationType.NONE and not config.auth_config:
			raise ValueError("Authentication configuration required for non-none auth type")
	
	async def _validate_webhook_config(self, config: WebhookConfig):
		"""Validate webhook configuration."""
		if not config.name or not config.url:
			raise ValueError("Webhook name and URL are required")
		
		if not config.events:
			raise ValueError("At least one event must be specified")
	
	async def _check_tenant_access(self, request_tenant_id: str, config_tenant_id: str):
		"""Check if request tenant has access to the integration."""
		if config_tenant_id and request_tenant_id != config_tenant_id:
			raise ValueError("Tenant access denied")
	
	def _map_data(self, data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
		"""Map data fields according to mapping configuration."""
		if not mapping:
			return data
		
		mapped_data = {}
		for source_key, target_key in mapping.items():
			if source_key in data:
				mapped_data[target_key] = data[source_key]
		
		return mapped_data
	
	async def _load_configurations(self):
		"""Load integration and webhook configurations from database."""
		try:
			async with self.database.get_session() as session:
				# Load integrations
				result = await session.execute("SELECT * FROM wo_integrations WHERE is_active = true")
				for row in result.fetchall():
					config = IntegrationConfig(
						id=row.id,
						name=row.name,
						type=IntegrationType(row.type),
						endpoint=row.endpoint,
						auth_type=AuthenticationType(row.auth_type),
						auth_config=json.loads(row.auth_config or '{}'),
						headers=json.loads(row.headers or '{}'),
						timeout_seconds=row.timeout_seconds,
						retry_attempts=row.retry_attempts,
						retry_delay_seconds=row.retry_delay_seconds,
						tenant_id=row.tenant_id,
						is_active=row.is_active,
						metadata=json.loads(row.metadata or '{}')
					)
					self.integrations[config.id] = config
				
				# Load webhooks
				result = await session.execute("SELECT * FROM wo_webhooks WHERE is_active = true")
				for row in result.fetchall():
					config = WebhookConfig(
						id=row.id,
						name=row.name,
						url=row.url,
						secret=await self.encryption.decrypt(row.secret) if row.secret else None,
						events=json.loads(row.events),
						tenant_id=row.tenant_id,
						is_active=row.is_active,
						retry_attempts=row.retry_attempts,
						retry_delay_seconds=row.retry_delay_seconds,
						timeout_seconds=row.timeout_seconds,
						created_at=row.created_at,
						updated_at=row.updated_at
					)
					self.webhooks[config.id] = config
			
			logger.info(f"Loaded {len(self.integrations)} integrations and {len(self.webhooks)} webhooks")
			
		except Exception as e:
			logger.error(f"Failed to load configurations: {str(e)}")
			raise
	
	def _initialize_apg_integrations(self):
		"""Initialize APG capability integrations."""
		# Define APG capability integrations
		apg_capabilities = {
			'auth_rbac': APGCapabilityIntegration(
				capability_name='auth_rbac',
				endpoint='/api/auth/verify',
				method='POST',
				input_mapping={'user_id': 'user_id', 'permission': 'permission'},
				output_mapping={'authorized': 'authorized', 'roles': 'roles'},
				auth_required=True,
				tenant_isolation=True,
				audit_enabled=True
			),
			'audit_compliance': APGCapabilityIntegration(
				capability_name='audit_compliance',
				endpoint='/api/audit/log',
				method='POST',
				input_mapping={'event': 'event', 'resource': 'resource'},
				output_mapping={'audit_id': 'audit_id'},
				auth_required=True,
				tenant_isolation=True,
				audit_enabled=False  # Don't audit audit logs
			),
			'data_processing': APGCapabilityIntegration(
				capability_name='data_processing',
				endpoint='/api/data/process',
				method='POST',
				input_mapping={'data': 'input_data', 'config': 'processing_config'},
				output_mapping={'result': 'processed_data', 'metadata': 'processing_metadata'},
				auth_required=True,
				tenant_isolation=True,
				audit_enabled=True
			),
			'real_time_collaboration': APGCapabilityIntegration(
				capability_name='real_time_collaboration',
				endpoint='/api/collab/notify',
				method='POST',
				input_mapping={'users': 'target_users', 'message': 'notification'},
				output_mapping={'sent': 'notification_sent'},
				auth_required=True,
				tenant_isolation=True,
				audit_enabled=True
			)
		}
		
		self.apg_capabilities.update(apg_capabilities)
		logger.info(f"Initialized {len(apg_capabilities)} APG capability integrations")
	
	async def _store_integration_audit(self, request: IntegrationRequest, response: IntegrationResponse):
		"""Store integration request/response for audit purposes."""
		try:
			async with self.database.get_session() as session:
				audit_data = {
					'id': uuid7str(),
					'request_id': request.id,
					'integration_id': request.integration_id,
					'method': request.method,
					'endpoint': request.endpoint,
					'tenant_id': request.tenant_id,
					'workflow_instance_id': request.workflow_instance_id,
					'task_execution_id': request.task_execution_id,
					'request_payload': json.dumps(request.payload),
					'response_status': response.status_code,
					'response_data': json.dumps(response.response_data),
					'duration_ms': response.duration_ms,
					'success': response.success,
					'error_message': response.error_message,
					'created_at': datetime.utcnow()
				}
				
				await session.execute(
					"""
					INSERT INTO wo_integration_audit (
						id, request_id, integration_id, method, endpoint,
						tenant_id, workflow_instance_id, task_execution_id,
						request_payload, response_status, response_data,
						duration_ms, success, error_message, created_at
					) VALUES (
						:id, :request_id, :integration_id, :method, :endpoint,
						:tenant_id, :workflow_instance_id, :task_execution_id,
						:request_payload, :response_status, :response_data,
						:duration_ms, :success, :error_message, :created_at
					)
					""",
					audit_data
				)
				await session.commit()
		
		except Exception as e:
			logger.error(f"Failed to store integration audit: {str(e)}")
	
	async def _update_webhook_delivery_status(self, delivery_id: str, status: WebhookStatus, 
											  response_code: int = None, response_body: str = None,
											  error_message: str = None):
		"""Update webhook delivery status."""
		try:
			async with self.database.get_session() as session:
				await session.execute(
					"""
					UPDATE wo_webhook_deliveries SET
						status = :status,
						last_attempt_at = :last_attempt_at,
						response_code = :response_code,
						response_body = :response_body,
						error_message = :error_message
					WHERE id = :id
					""",
					{
						'id': delivery_id,
						'status': status.value,
						'last_attempt_at': datetime.utcnow(),
						'response_code': response_code,
						'response_body': response_body,
						'error_message': error_message
					}
				)
				await session.commit()
		
		except Exception as e:
			logger.error(f"Failed to update webhook delivery status: {str(e)}")
	
	async def _increment_webhook_attempts(self, delivery_id: str):
		"""Increment webhook delivery attempt count."""
		try:
			async with self.database.get_session() as session:
				await session.execute(
					"UPDATE wo_webhook_deliveries SET attempts = attempts + 1 WHERE id = :id",
					{'id': delivery_id}
				)
				await session.commit()
		
		except Exception as e:
			logger.error(f"Failed to increment webhook attempts: {str(e)}")


# Global service instance
integration_service = IntegrationService()