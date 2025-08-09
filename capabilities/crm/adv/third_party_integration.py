"""
APG Customer Relationship Management - Third-Party Integration Framework

This module provides comprehensive third-party integration capabilities including
connector management, authentication, field mapping, data transformation,
sync orchestration, and monitoring for seamless platform interoperability.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
import logging
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import aiohttp
from aiohttp import ClientTimeout, BasicAuth
import jwt

from pydantic import BaseModel, Field, validator, HttpUrl
from uuid_extensions import uuid7str

from .views import CRMResponse, CRMError


logger = logging.getLogger(__name__)


class IntegrationType(str, Enum):
	"""Integration platform types"""
	SALESFORCE = "salesforce"
	HUBSPOT = "hubspot"
	PIPEDRIVE = "pipedrive"
	ZOHO = "zoho"
	ZAPIER = "zapier"
	WEBHOOK = "webhook"
	REST_API = "rest_api"
	GRAPHQL = "graphql"
	SOAP = "soap"
	FTP = "ftp"
	EMAIL = "email"
	DATABASE = "database"
	CUSTOM = "custom"


class AuthenticationType(str, Enum):
	"""Authentication methods"""
	OAUTH2 = "oauth2"
	API_KEY = "api_key"
	BASIC_AUTH = "basic_auth"
	BEARER_TOKEN = "bearer_token"
	JWT = "jwt"
	HMAC = "hmac"
	CUSTOM = "custom"
	NONE = "none"


class SyncDirection(str, Enum):
	"""Data synchronization directions"""
	BIDIRECTIONAL = "bidirectional"
	INBOUND = "inbound"
	OUTBOUND = "outbound"


class SyncStatus(str, Enum):
	"""Synchronization status"""
	ACTIVE = "active"
	PAUSED = "paused"
	ERROR = "error"
	STOPPED = "stopped"


class DataOperation(str, Enum):
	"""Data operations"""
	CREATE = "create"
	UPDATE = "update"
	DELETE = "delete"
	UPSERT = "upsert"
	READ = "read"


class IntegrationConnector(BaseModel):
	"""Third-party integration connector configuration"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	connector_name: str
	description: Optional[str] = None
	integration_type: IntegrationType
	platform_name: str
	platform_version: str = "v1"
	
	# Connection configuration
	base_url: HttpUrl
	authentication_type: AuthenticationType
	authentication_config: Dict[str, Any] = Field(default_factory=dict)
	connection_timeout: int = 30
	request_timeout: int = 60
	max_retries: int = 3
	retry_delay_seconds: int = 5
	
	# Rate limiting
	rate_limit_config: Dict[str, Any] = Field(default_factory=dict)
	
	# Capabilities
	supported_operations: List[DataOperation] = Field(default_factory=list)
	supported_entities: List[str] = Field(default_factory=list)
	
	# Configuration
	custom_headers: Dict[str, str] = Field(default_factory=dict)
	webhook_config: Optional[Dict[str, Any]] = None
	batch_config: Dict[str, Any] = Field(default_factory=dict)
	transformation_rules: Dict[str, Any] = Field(default_factory=dict)
	
	# Status
	is_active: bool = True
	last_sync_at: Optional[datetime] = None
	last_success_at: Optional[datetime] = None
	last_failure_at: Optional[datetime] = None
	last_failure_reason: Optional[str] = None
	connection_status: str = "unknown"  # connected, disconnected, error
	
	# Metadata
	tags: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


class FieldMapping(BaseModel):
	"""Field mapping configuration for data transformation"""
	id: str = Field(default_factory=uuid7str)
	connector_id: str
	tenant_id: str
	mapping_name: str
	source_entity: str
	target_entity: str
	
	# Field mappings
	field_mappings: List[Dict[str, Any]] = Field(default_factory=list)
	# Example: [{"source_field": "firstName", "target_field": "first_name", "transformation": "lowercase", "required": true}]
	
	# Transformation rules
	transformation_functions: Dict[str, str] = Field(default_factory=dict)
	validation_rules: Dict[str, Any] = Field(default_factory=dict)
	default_values: Dict[str, Any] = Field(default_factory=dict)
	
	# Configuration
	sync_direction: SyncDirection
	conflict_resolution: str = "target_wins"  # source_wins, target_wins, manual, latest_timestamp
	batch_size: int = 100
	
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


class SyncConfiguration(BaseModel):
	"""Synchronization configuration"""
	id: str = Field(default_factory=uuid7str)
	connector_id: str
	tenant_id: str
	sync_name: str
	description: Optional[str] = None
	
	# Schedule configuration
	sync_frequency: str = "manual"  # manual, real_time, hourly, daily, weekly
	schedule_config: Dict[str, Any] = Field(default_factory=dict)
	timezone: str = "UTC"
	
	# Sync scope
	entity_filters: Dict[str, Any] = Field(default_factory=dict)
	field_filters: List[str] = Field(default_factory=list)
	date_range_config: Optional[Dict[str, Any]] = None
	
	# Processing configuration
	batch_size: int = 100
	max_concurrent_batches: int = 5
	enable_deduplication: bool = True
	deduplication_fields: List[str] = Field(default_factory=list)
	
	# Error handling
	error_handling: Dict[str, Any] = Field(default_factory=dict)
	retry_config: Dict[str, Any] = Field(default_factory=dict)
	
	# Status
	sync_status: SyncStatus = SyncStatus.PAUSED
	last_sync_at: Optional[datetime] = None
	next_sync_at: Optional[datetime] = None
	
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


class SyncExecution(BaseModel):
	"""Synchronization execution record"""
	id: str = Field(default_factory=uuid7str)
	sync_config_id: str
	connector_id: str
	tenant_id: str
	
	# Execution details
	execution_type: str = "scheduled"  # scheduled, manual, triggered
	trigger_source: Optional[str] = None
	
	# Progress tracking
	status: str = "running"  # pending, running, completed, failed, cancelled
	total_records: int = 0
	processed_records: int = 0
	successful_records: int = 0
	failed_records: int = 0
	skipped_records: int = 0
	
	# Performance metrics
	started_at: datetime = Field(default_factory=datetime.utcnow)
	completed_at: Optional[datetime] = None
	duration_seconds: Optional[float] = None
	throughput_records_per_second: Optional[float] = None
	
	# Results
	summary: Dict[str, Any] = Field(default_factory=dict)
	error_details: List[Dict[str, Any]] = Field(default_factory=list)
	warnings: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Data
	affected_entities: List[str] = Field(default_factory=list)
	sync_statistics: Dict[str, Any] = Field(default_factory=dict)


class ThirdPartyIntegrationManager:
	"""Comprehensive third-party integration management system"""
	
	def __init__(self, db_pool, config: Optional[Dict[str, Any]] = None):
		self.db_pool = db_pool
		self.config = config or {}
		self.session = None
		self.connectors = {}
		self.sync_scheduler = None
		self.sync_workers = {}
		self.execution_queue = asyncio.Queue()
		self.workers_running = False
		
		# Integration-specific handlers
		self.integration_handlers = {
			IntegrationType.SALESFORCE: self._handle_salesforce_integration,
			IntegrationType.HUBSPOT: self._handle_hubspot_integration,
			IntegrationType.PIPEDRIVE: self._handle_pipedrive_integration,
			IntegrationType.ZAPIER: self._handle_zapier_integration,
			IntegrationType.REST_API: self._handle_rest_api_integration,
			IntegrationType.WEBHOOK: self._handle_webhook_integration
		}
		
		# Transformation functions
		self.transformation_functions = {
			'lowercase': lambda x: str(x).lower() if x else x,
			'uppercase': lambda x: str(x).upper() if x else x,
			'trim': lambda x: str(x).strip() if x else x,
			'email_normalize': lambda x: str(x).lower().strip() if x else x,
			'phone_normalize': self._normalize_phone,
			'date_iso': self._convert_to_iso_date,
			'boolean_convert': self._convert_to_boolean,
			'currency_normalize': self._normalize_currency
		}

	async def initialize(self) -> None:
		"""Initialize the integration manager"""
		try:
			# Initialize HTTP session
			connector = aiohttp.TCPConnector(
				limit=100,
				limit_per_host=30,
				ttl_dns_cache=300,
				use_dns_cache=True
			)
			
			timeout = ClientTimeout(total=60, connect=10)
			self.session = aiohttp.ClientSession(
				connector=connector,
				timeout=timeout,
				headers={'User-Agent': 'APG-CRM-Integration/1.0'}
			)
			
			# Load active connectors
			await self._load_active_connectors()
			
			# Start sync workers
			await self._start_sync_workers()
			
			# Initialize sync scheduler
			await self._initialize_sync_scheduler()
			
			logger.info("✅ Third-party integration manager initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize integration manager: {str(e)}")
			raise CRMError(f"Integration manager initialization failed: {str(e)}")

	async def create_connector(
		self,
		tenant_id: str,
		connector_name: str,
		integration_type: IntegrationType,
		platform_name: str,
		base_url: str,
		authentication_type: AuthenticationType,
		authentication_config: Dict[str, Any],
		created_by: str,
		description: Optional[str] = None,
		supported_operations: Optional[List[DataOperation]] = None,
		supported_entities: Optional[List[str]] = None,
		custom_headers: Optional[Dict[str, str]] = None,
		rate_limit_config: Optional[Dict[str, Any]] = None
	) -> IntegrationConnector:
		"""Create a new integration connector"""
		try:
			connector = IntegrationConnector(
				tenant_id=tenant_id,
				connector_name=connector_name,
				description=description,
				integration_type=integration_type,
				platform_name=platform_name,
				base_url=base_url,
				authentication_type=authentication_type,
				authentication_config=authentication_config,
				supported_operations=supported_operations or [],
				supported_entities=supported_entities or [],
				custom_headers=custom_headers or {},
				rate_limit_config=rate_limit_config or {},
				created_by=created_by
			)
			
			# Test connection
			connection_test = await self._test_connector_connection(connector)
			connector.connection_status = "connected" if connection_test else "error"
			
			# Save to database
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_integration_connectors (
						id, tenant_id, connector_name, description, integration_type,
						platform_name, platform_version, base_url, authentication_type,
						authentication_config, connection_timeout, request_timeout,
						max_retries, retry_delay_seconds, rate_limit_config,
						supported_operations, supported_entities, custom_headers,
						webhook_config, batch_config, transformation_rules,
						is_active, connection_status, tags, metadata,
						created_at, created_by
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27)
				""",
				connector.id, connector.tenant_id, connector.connector_name,
				connector.description, connector.integration_type.value,
				connector.platform_name, connector.platform_version,
				str(connector.base_url), connector.authentication_type.value,
				json.dumps(connector.authentication_config), connector.connection_timeout,
				connector.request_timeout, connector.max_retries,
				connector.retry_delay_seconds, json.dumps(connector.rate_limit_config),
				json.dumps([op.value for op in connector.supported_operations]),
				json.dumps(connector.supported_entities), json.dumps(connector.custom_headers),
				json.dumps(connector.webhook_config), json.dumps(connector.batch_config),
				json.dumps(connector.transformation_rules), connector.is_active,
				connector.connection_status, json.dumps(connector.tags),
				json.dumps(connector.metadata), connector.created_at, connector.created_by)
			
			# Cache connector
			self.connectors[connector.id] = connector
			
			logger.info(f"Created integration connector: {connector_name} ({integration_type.value})")
			return connector
			
		except Exception as e:
			logger.error(f"Failed to create integration connector: {str(e)}")
			raise CRMError(f"Failed to create integration connector: {str(e)}")

	async def create_field_mapping(
		self,
		connector_id: str,
		tenant_id: str,
		mapping_name: str,
		source_entity: str,
		target_entity: str,
		field_mappings: List[Dict[str, Any]],
		sync_direction: SyncDirection,
		created_by: str,
		transformation_functions: Optional[Dict[str, str]] = None,
		validation_rules: Optional[Dict[str, Any]] = None,
		default_values: Optional[Dict[str, Any]] = None
	) -> FieldMapping:
		"""Create field mapping configuration"""
		try:
			mapping = FieldMapping(
				connector_id=connector_id,
				tenant_id=tenant_id,
				mapping_name=mapping_name,
				source_entity=source_entity,
				target_entity=target_entity,
				field_mappings=field_mappings,
				sync_direction=sync_direction,
				transformation_functions=transformation_functions or {},
				validation_rules=validation_rules or {},
				default_values=default_values or {},
				created_by=created_by
			)
			
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_field_mappings (
						id, connector_id, tenant_id, mapping_name, source_entity,
						target_entity, field_mappings, transformation_functions,
						validation_rules, default_values, sync_direction,
						conflict_resolution, batch_size, is_active,
						created_at, created_by
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
				""",
				mapping.id, mapping.connector_id, mapping.tenant_id,
				mapping.mapping_name, mapping.source_entity, mapping.target_entity,
				json.dumps(mapping.field_mappings), json.dumps(mapping.transformation_functions),
				json.dumps(mapping.validation_rules), json.dumps(mapping.default_values),
				mapping.sync_direction.value, mapping.conflict_resolution,
				mapping.batch_size, mapping.is_active, mapping.created_at, mapping.created_by)
			
			logger.info(f"Created field mapping: {mapping_name} for connector {connector_id}")
			return mapping
			
		except Exception as e:
			logger.error(f"Failed to create field mapping: {str(e)}")
			raise CRMError(f"Failed to create field mapping: {str(e)}")

	async def create_sync_configuration(
		self,
		connector_id: str,
		tenant_id: str,
		sync_name: str,
		sync_frequency: str,
		created_by: str,
		description: Optional[str] = None,
		schedule_config: Optional[Dict[str, Any]] = None,
		entity_filters: Optional[Dict[str, Any]] = None,
		batch_size: int = 100
	) -> SyncConfiguration:
		"""Create synchronization configuration"""
		try:
			sync_config = SyncConfiguration(
				connector_id=connector_id,
				tenant_id=tenant_id,
				sync_name=sync_name,
				description=description,
				sync_frequency=sync_frequency,
				schedule_config=schedule_config or {},
				entity_filters=entity_filters or {},
				batch_size=batch_size,
				created_by=created_by
			)
			
			# Calculate next sync time if scheduled
			if sync_frequency != "manual":
				sync_config.next_sync_at = await self._calculate_next_sync_time(sync_config)
			
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_sync_configurations (
						id, connector_id, tenant_id, sync_name, description,
						sync_frequency, schedule_config, timezone, entity_filters,
						field_filters, date_range_config, batch_size,
						max_concurrent_batches, enable_deduplication,
						deduplication_fields, error_handling, retry_config,
						sync_status, next_sync_at, is_active,
						created_at, created_by
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
				""",
				sync_config.id, sync_config.connector_id, sync_config.tenant_id,
				sync_config.sync_name, sync_config.description, sync_config.sync_frequency,
				json.dumps(sync_config.schedule_config), sync_config.timezone,
				json.dumps(sync_config.entity_filters), json.dumps(sync_config.field_filters),
				json.dumps(sync_config.date_range_config), sync_config.batch_size,
				sync_config.max_concurrent_batches, sync_config.enable_deduplication,
				json.dumps(sync_config.deduplication_fields), json.dumps(sync_config.error_handling),
				json.dumps(sync_config.retry_config), sync_config.sync_status.value,
				sync_config.next_sync_at, sync_config.is_active,
				sync_config.created_at, sync_config.created_by)
			
			logger.info(f"Created sync configuration: {sync_name} for connector {connector_id}")
			return sync_config
			
		except Exception as e:
			logger.error(f"Failed to create sync configuration: {str(e)}")
			raise CRMError(f"Failed to create sync configuration: {str(e)}")

	async def execute_sync(
		self,
		sync_config_id: str,
		tenant_id: str,
		execution_type: str = "manual",
		trigger_source: Optional[str] = None
	) -> SyncExecution:
		"""Execute a synchronization"""
		try:
			# Get sync configuration
			async with self.db_pool.acquire() as conn:
				sync_row = await conn.fetchrow("""
					SELECT sc.*, ic.* FROM crm_sync_configurations sc
					JOIN crm_integration_connectors ic ON sc.connector_id = ic.id
					WHERE sc.id = $1 AND sc.tenant_id = $2 AND sc.is_active = true
				""", sync_config_id, tenant_id)
			
			if not sync_row:
				raise CRMError("Sync configuration not found or inactive")
			
			# Create execution record
			execution = SyncExecution(
				sync_config_id=sync_config_id,
				connector_id=sync_row['connector_id'],
				tenant_id=tenant_id,
				execution_type=execution_type,
				trigger_source=trigger_source
			)
			
			# Save execution record
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_sync_executions (
						id, sync_config_id, connector_id, tenant_id,
						execution_type, trigger_source, status,
						started_at
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
				""",
				execution.id, execution.sync_config_id, execution.connector_id,
				execution.tenant_id, execution.execution_type, execution.trigger_source,
				execution.status, execution.started_at)
			
			# Queue for background processing
			await self.execution_queue.put((execution, dict(sync_row)))
			
			logger.info(f"Queued sync execution: {execution.id}")
			return execution
			
		except Exception as e:
			logger.error(f"Failed to execute sync: {str(e)}")
			raise CRMError(f"Failed to execute sync: {str(e)}")

	async def get_connectors(
		self,
		tenant_id: str,
		integration_type: Optional[IntegrationType] = None,
		is_active: Optional[bool] = None
	) -> List[Dict[str, Any]]:
		"""Get integration connectors"""
		try:
			async with self.db_pool.acquire() as conn:
				query = "SELECT * FROM crm_integration_connectors WHERE tenant_id = $1"
				params = [tenant_id]
				
				if integration_type:
					query += " AND integration_type = $2"
					params.append(integration_type.value)
				
				if is_active is not None:
					query += f" AND is_active = ${len(params) + 1}"
					params.append(is_active)
				
				query += " ORDER BY created_at DESC"
				
				rows = await conn.fetch(query, *params)
				return [dict(row) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get connectors: {str(e)}")
			raise CRMError(f"Failed to get connectors: {str(e)}")

	async def get_sync_history(
		self,
		tenant_id: str,
		connector_id: Optional[str] = None,
		limit: int = 100
	) -> List[Dict[str, Any]]:
		"""Get synchronization execution history"""
		try:
			async with self.db_pool.acquire() as conn:
				query = """
					SELECT se.*, sc.sync_name, ic.connector_name, ic.platform_name
					FROM crm_sync_executions se
					JOIN crm_sync_configurations sc ON se.sync_config_id = sc.id
					JOIN crm_integration_connectors ic ON se.connector_id = ic.id
					WHERE se.tenant_id = $1
				"""
				params = [tenant_id]
				
				if connector_id:
					query += " AND se.connector_id = $2"
					params.append(connector_id)
				
				query += f" ORDER BY se.started_at DESC LIMIT ${len(params) + 1}"
				params.append(limit)
				
				rows = await conn.fetch(query, *params)
				return [dict(row) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get sync history: {str(e)}")
			raise CRMError(f"Failed to get sync history: {str(e)}")

	# Integration-specific handlers

	async def _handle_salesforce_integration(
		self,
		connector: IntegrationConnector,
		operation: DataOperation,
		entity_type: str,
		data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Handle Salesforce integration"""
		try:
			# Get Salesforce access token
			access_token = await self._get_salesforce_access_token(connector)
			
			headers = {
				'Authorization': f'Bearer {access_token}',
				'Content-Type': 'application/json'
			}
			
			# Build Salesforce API URL
			api_version = connector.platform_version or "v58.0"
			base_url = str(connector.base_url).rstrip('/')
			
			if operation == DataOperation.CREATE:
				url = f"{base_url}/services/data/{api_version}/sobjects/{entity_type}/"
				async with self.session.post(url, json=data, headers=headers) as response:
					result = await response.json()
					return {"success": response.status < 300, "data": result}
			
			elif operation == DataOperation.UPDATE:
				record_id = data.pop('Id', data.pop('id', None))
				if not record_id:
					raise CRMError("Record ID required for update operation")
				
				url = f"{base_url}/services/data/{api_version}/sobjects/{entity_type}/{record_id}"
				async with self.session.patch(url, json=data, headers=headers) as response:
					return {"success": response.status < 300, "data": await response.text()}
			
			elif operation == DataOperation.READ:
				query = data.get('query', f"SELECT Id FROM {entity_type}")
				url = f"{base_url}/services/data/{api_version}/query"
				params = {'q': query}
				
				async with self.session.get(url, params=params, headers=headers) as response:
					result = await response.json()
					return {"success": response.status < 300, "data": result}
			
			else:
				raise CRMError(f"Operation {operation} not implemented for Salesforce")
				
		except Exception as e:
			logger.error(f"Salesforce integration error: {str(e)}")
			raise CRMError(f"Salesforce integration failed: {str(e)}")

	async def _handle_hubspot_integration(
		self,
		connector: IntegrationConnector,
		operation: DataOperation,
		entity_type: str,
		data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Handle HubSpot integration"""
		try:
			api_key = connector.authentication_config.get('api_key')
			if not api_key:
				raise CRMError("HubSpot API key not found")
			
			headers = {
				'Authorization': f'Bearer {api_key}',
				'Content-Type': 'application/json'
			}
			
			base_url = str(connector.base_url).rstrip('/')
			
			# Map entity types to HubSpot objects
			entity_mapping = {
				'contact': 'contacts',
				'company': 'companies',
				'deal': 'deals',
				'ticket': 'tickets'
			}
			
			hubspot_entity = entity_mapping.get(entity_type.lower(), entity_type)
			
			if operation == DataOperation.CREATE:
				url = f"{base_url}/crm/v3/objects/{hubspot_entity}"
				payload = {"properties": data}
				
				async with self.session.post(url, json=payload, headers=headers) as response:
					result = await response.json()
					return {"success": response.status < 300, "data": result}
			
			elif operation == DataOperation.UPDATE:
				record_id = data.pop('id', data.pop('hs_object_id', None))
				if not record_id:
					raise CRMError("Record ID required for update operation")
				
				url = f"{base_url}/crm/v3/objects/{hubspot_entity}/{record_id}"
				payload = {"properties": data}
				
				async with self.session.patch(url, json=payload, headers=headers) as response:
					result = await response.json()
					return {"success": response.status < 300, "data": result}
			
			elif operation == DataOperation.READ:
				url = f"{base_url}/crm/v3/objects/{hubspot_entity}"
				params = data.get('params', {})
				
				async with self.session.get(url, params=params, headers=headers) as response:
					result = await response.json()
					return {"success": response.status < 300, "data": result}
			
			else:
				raise CRMError(f"Operation {operation} not implemented for HubSpot")
				
		except Exception as e:
			logger.error(f"HubSpot integration error: {str(e)}")
			raise CRMError(f"HubSpot integration failed: {str(e)}")

	async def _handle_rest_api_integration(
		self,
		connector: IntegrationConnector,
		operation: DataOperation,
		entity_type: str,
		data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Handle generic REST API integration"""
		try:
			# Prepare authentication
			headers = dict(connector.custom_headers)
			auth = None
			
			if connector.authentication_type == AuthenticationType.API_KEY:
				api_key = connector.authentication_config.get('api_key')
				key_header = connector.authentication_config.get('key_header', 'X-API-Key')
				headers[key_header] = api_key
			
			elif connector.authentication_type == AuthenticationType.BEARER_TOKEN:
				token = connector.authentication_config.get('token')
				headers['Authorization'] = f'Bearer {token}'
			
			elif connector.authentication_type == AuthenticationType.BASIC_AUTH:
				username = connector.authentication_config.get('username')
				password = connector.authentication_config.get('password')
				auth = BasicAuth(username, password)
			
			# Build URL
			base_url = str(connector.base_url).rstrip('/')
			endpoint_path = data.get('endpoint_path', f'/{entity_type}')
			url = f"{base_url}{endpoint_path}"
			
			# Execute operation
			if operation == DataOperation.CREATE:
				async with self.session.post(url, json=data.get('payload', data), 
											headers=headers, auth=auth) as response:
					result = await response.json() if response.content_type == 'application/json' else await response.text()
					return {"success": response.status < 300, "data": result, "status": response.status}
			
			elif operation == DataOperation.READ:
				params = data.get('params', {})
				async with self.session.get(url, params=params, headers=headers, auth=auth) as response:
					result = await response.json() if response.content_type == 'application/json' else await response.text()
					return {"success": response.status < 300, "data": result, "status": response.status}
			
			elif operation == DataOperation.UPDATE:
				async with self.session.put(url, json=data.get('payload', data), 
											headers=headers, auth=auth) as response:
					result = await response.json() if response.content_type == 'application/json' else await response.text()
					return {"success": response.status < 300, "data": result, "status": response.status}
			
			elif operation == DataOperation.DELETE:
				async with self.session.delete(url, headers=headers, auth=auth) as response:
					result = await response.text()
					return {"success": response.status < 300, "data": result, "status": response.status}
			
			else:
				raise CRMError(f"Operation {operation} not supported")
				
		except Exception as e:
			logger.error(f"REST API integration error: {str(e)}")
			raise CRMError(f"REST API integration failed: {str(e)}")

	# Helper methods

	async def _load_active_connectors(self) -> None:
		"""Load active integration connectors"""
		try:
			async with self.db_pool.acquire() as conn:
				rows = await conn.fetch("""
					SELECT * FROM crm_integration_connectors 
					WHERE is_active = true
				""")
				
				for row in rows:
					connector_data = dict(row)
					# Convert JSON fields back to proper types
					connector_data['authentication_config'] = json.loads(connector_data['authentication_config'])
					connector_data['rate_limit_config'] = json.loads(connector_data['rate_limit_config'])
					connector_data['supported_operations'] = [DataOperation(op) for op in json.loads(connector_data['supported_operations'])]
					connector_data['supported_entities'] = json.loads(connector_data['supported_entities'])
					connector_data['custom_headers'] = json.loads(connector_data['custom_headers'])
					connector_data['tags'] = json.loads(connector_data['tags'])
					connector_data['metadata'] = json.loads(connector_data['metadata'])
					
					connector = IntegrationConnector(**connector_data)
					self.connectors[connector.id] = connector
			
			logger.info(f"Loaded {len(self.connectors)} active integration connectors")
			
		except Exception as e:
			logger.error(f"Failed to load active connectors: {str(e)}")

	async def _test_connector_connection(self, connector: IntegrationConnector) -> bool:
		"""Test connectivity to integration platform"""
		try:
			# Perform connection test based on integration type
			if connector.integration_type == IntegrationType.SALESFORCE:
				return await self._test_salesforce_connection(connector)
			elif connector.integration_type == IntegrationType.HUBSPOT:
				return await self._test_hubspot_connection(connector)
			elif connector.integration_type == IntegrationType.REST_API:
				return await self._test_rest_api_connection(connector)
			else:
				# Generic HTTP connectivity test
				async with self.session.get(str(connector.base_url), 
											timeout=ClientTimeout(total=10)) as response:
					return response.status < 400
					
		except Exception as e:
			logger.warning(f"Connection test failed for {connector.connector_name}: {str(e)}")
			return False

	async def _get_salesforce_access_token(self, connector: IntegrationConnector) -> str:
		"""Get Salesforce access token using OAuth2"""
		auth_config = connector.authentication_config
		
		# Check if we have a valid cached token
		if auth_config.get('access_token') and auth_config.get('expires_at'):
			expires_at = datetime.fromisoformat(auth_config['expires_at'])
			if expires_at > datetime.utcnow() + timedelta(minutes=5):
				return auth_config['access_token']
		
		# Request new token
		token_url = f"{str(connector.base_url).rstrip('/')}/services/oauth2/token"
		
		data = {
			'grant_type': 'client_credentials',
			'client_id': auth_config['client_id'],
			'client_secret': auth_config['client_secret']
		}
		
		async with self.session.post(token_url, data=data) as response:
			if response.status != 200:
				raise CRMError(f"Failed to get Salesforce access token: {response.status}")
			
			token_data = await response.json()
			access_token = token_data['access_token']
			expires_in = token_data.get('expires_in', 3600)
			expires_at = (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat()
			
			# Update cached token
			auth_config['access_token'] = access_token
			auth_config['expires_at'] = expires_at
			
			# Update database
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					UPDATE crm_integration_connectors 
					SET authentication_config = $1
					WHERE id = $2
				""", json.dumps(auth_config), connector.id)
			
			return access_token

	# Transformation helper methods

	def _normalize_phone(self, phone: str) -> str:
		"""Normalize phone number format"""
		if not phone:
			return phone
		# Remove all non-digit characters
		digits = ''.join(filter(str.isdigit, str(phone)))
		if len(digits) == 10:
			return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
		elif len(digits) == 11 and digits[0] == '1':
			return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
		return phone

	def _convert_to_iso_date(self, date_value: Any) -> Optional[str]:
		"""Convert date to ISO format"""
		if not date_value:
			return None
		
		if isinstance(date_value, datetime):
			return date_value.isoformat()
		elif isinstance(date_value, str):
			try:
				parsed_date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
				return parsed_date.isoformat()
			except:
				return date_value
		
		return str(date_value)

	def _convert_to_boolean(self, value: Any) -> bool:
		"""Convert value to boolean"""
		if isinstance(value, bool):
			return value
		elif isinstance(value, str):
			return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
		elif isinstance(value, (int, float)):
			return bool(value)
		return False

	def _normalize_currency(self, amount: Any) -> Optional[float]:
		"""Normalize currency amount"""
		if not amount:
			return None
		
		if isinstance(amount, (int, float)):
			return float(amount)
		elif isinstance(amount, str):
			# Remove currency symbols and commas
			cleaned = ''.join(c for c in amount if c.isdigit() or c in '.-')
			try:
				return float(cleaned)
			except:
				return None
		
		return None

	async def _start_sync_workers(self) -> None:
		"""Start background sync workers"""
		self.workers_running = True
		
		# Start execution worker
		asyncio.create_task(self._sync_execution_worker())
		
		logger.info("Started integration sync workers")

	async def _sync_execution_worker(self) -> None:
		"""Background worker for processing sync executions"""
		while self.workers_running:
			try:
				# Get next execution task
				execution_task = await asyncio.wait_for(
					self.execution_queue.get(),
					timeout=1.0
				)
				
				execution, sync_config = execution_task
				
				# Process sync execution
				await self._process_sync_execution(execution, sync_config)
				
			except asyncio.TimeoutError:
				continue
			except Exception as e:
				logger.error(f"Sync execution worker error: {str(e)}")
				await asyncio.sleep(1)

	async def _process_sync_execution(
		self,
		execution: SyncExecution,
		sync_config: Dict[str, Any]
	) -> None:
		"""Process a sync execution"""
		try:
			# Update execution status
			await self._update_execution_status(execution.id, "running")
			
			connector = self.connectors.get(execution.connector_id)
			if not connector:
				raise CRMError(f"Connector {execution.connector_id} not found")
			
			# Get field mappings
			field_mappings = await self._get_field_mappings(execution.connector_id)
			
			# Execute sync based on configuration
			results = await self._execute_data_sync(connector, sync_config, field_mappings)
			
			# Update execution with results
			execution.status = "completed"
			execution.completed_at = datetime.utcnow()
			execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
			execution.successful_records = results.get('successful_records', 0)
			execution.failed_records = results.get('failed_records', 0)
			execution.total_records = execution.successful_records + execution.failed_records
			execution.summary = results.get('summary', {})
			
			await self._save_execution_results(execution)
			
			logger.info(f"Completed sync execution: {execution.id}")
			
		except Exception as e:
			logger.error(f"Failed to process sync execution {execution.id}: {str(e)}")
			await self._update_execution_status(execution.id, "failed", str(e))

	async def _update_execution_status(
		self,
		execution_id: str,
		status: str,
		error_message: Optional[str] = None
	) -> None:
		"""Update sync execution status"""
		try:
			async with self.db_pool.acquire() as conn:
				if error_message:
					await conn.execute("""
						UPDATE crm_sync_executions 
						SET status = $1, error_details = $2
						WHERE id = $3
					""", status, json.dumps([{"error": error_message}]), execution_id)
				else:
					await conn.execute("""
						UPDATE crm_sync_executions 
						SET status = $1
						WHERE id = $2
					""", status, execution_id)
		except Exception as e:
			logger.error(f"Failed to update execution status: {str(e)}")

	async def shutdown(self) -> None:
		"""Shutdown integration manager"""
		self.workers_running = False
		
		if self.session:
			await self.session.close()
		
		logger.info("Third-party integration manager shut down successfully")