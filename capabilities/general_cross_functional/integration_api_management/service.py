"""
APG Integration API Management - Service Layer

Comprehensive service layer for API gateway management, security, analytics,
and developer portal operations with enterprise-grade features.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import hashlib
import secrets
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from uuid_extensions import uuid7str

# Third-party imports
import aiohttp
import aiocache
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update, delete, and_, or_, func
from pydantic import ValidationError

# Internal imports
from .models import (
	AMAPI, AMEndpoint, AMPolicy, AMConsumer, AMAPIKey, AMSubscription,
	AMDeployment, AMAnalytics, AMUsageRecord,
	APIConfig, EndpointConfig, PolicyConfig, ConsumerConfig, APIKeyConfig,
	SubscriptionConfig, APIStatus, AuthenticationType, PolicyType
)

# =============================================================================
# Service Configuration and Base Classes
# =============================================================================

@dataclass
class GatewayConfig:
	"""Gateway configuration settings."""
	default_timeout_ms: int = 30000
	max_retries: int = 3
	connection_pool_size: int = 100
	rate_limit_window_seconds: int = 60
	cache_ttl_seconds: int = 300
	metrics_collection_interval: int = 60

@dataclass
class SecurityConfig:
	"""Security configuration settings."""
	api_key_length: int = 32
	jwt_secret_key: str = "your-secret-key"
	jwt_algorithm: str = "HS256"
	jwt_expiration_hours: int = 24
	password_hash_iterations: int = 100000

class APIManagementError(Exception):
	"""Base exception for API Management operations."""
	pass

class APINotFoundError(APIManagementError):
	"""Raised when an API is not found."""
	pass

class ConsumerNotFoundError(APIManagementError):
	"""Raised when a consumer is not found."""
	pass

class AuthenticationError(APIManagementError):
	"""Raised when authentication fails."""
	pass

class AuthorizationError(APIManagementError):
	"""Raised when authorization fails."""
	pass

class RateLimitExceededError(APIManagementError):
	"""Raised when rate limit is exceeded."""
	pass

# =============================================================================
# API Lifecycle Management Service
# =============================================================================

class APILifecycleService:
	"""Service for managing API registration, versioning, and lifecycle."""
	
	def __init__(self, db_session: AsyncSession, cache: aiocache.Cache):
		self.db_session = db_session
		self.cache = cache
		self._log_prefix = "[APILifecycleService]"
	
	async def register_api(
		self,
		config: APIConfig,
		tenant_id: str,
		capability_id: str,
		created_by: str
	) -> str:
		"""Register a new API with the management platform."""
		try:
			# Validate API configuration
			await self._validate_api_config(config, tenant_id)
			
			# Create API record
			api = AMAPI(
				api_name=config.api_name,
				api_title=config.api_title,
				api_description=config.api_description,
				version=config.version,
				protocol_type=config.protocol_type.value,
				base_path=config.base_path,
				upstream_url=config.upstream_url,
				is_public=config.is_public,
				documentation_url=config.documentation_url,
				openapi_spec=config.openapi_spec,
				timeout_ms=config.timeout_ms,
				retry_attempts=config.retry_attempts,
				load_balancing_algorithm=config.load_balancing_algorithm.value,
				auth_type=config.auth_type.value,
				auth_config=config.auth_config,
				default_rate_limit=config.default_rate_limit,
				category=config.category,
				tags=config.tags,
				tenant_id=tenant_id,
				capability_id=capability_id,
				created_by=created_by,
				status=APIStatus.DRAFT.value
			)
			
			self.db_session.add(api)
			await self.db_session.commit()
			
			# Clear cache
			await self._invalidate_api_cache(tenant_id)
			
			self._log_api_operation("register", api.api_id, created_by)
			return api.api_id
			
		except Exception as e:
			await self.db_session.rollback()
			raise APIManagementError(f"Failed to register API: {str(e)}")
	
	async def update_api(
		self,
		api_id: str,
		config: APIConfig,
		tenant_id: str,
		updated_by: str
	) -> bool:
		"""Update an existing API configuration."""
		try:
			# Get existing API
			api = await self._get_api_by_id(api_id, tenant_id)
			if not api:
				raise APINotFoundError(f"API {api_id} not found")
			
			# Update fields
			api.api_title = config.api_title
			api.api_description = config.api_description
			api.upstream_url = config.upstream_url
			api.is_public = config.is_public
			api.documentation_url = config.documentation_url
			api.openapi_spec = config.openapi_spec
			api.timeout_ms = config.timeout_ms
			api.retry_attempts = config.retry_attempts
			api.load_balancing_algorithm = config.load_balancing_algorithm.value
			api.auth_type = config.auth_type.value
			api.auth_config = config.auth_config
			api.default_rate_limit = config.default_rate_limit
			api.category = config.category
			api.tags = config.tags
			api.updated_by = updated_by
			
			await self.db_session.commit()
			
			# Clear cache
			await self._invalidate_api_cache(tenant_id, api_id)
			
			self._log_api_operation("update", api_id, updated_by)
			return True
			
		except Exception as e:
			await self.db_session.rollback()
			raise APIManagementError(f"Failed to update API: {str(e)}")
	
	async def activate_api(self, api_id: str, tenant_id: str, activated_by: str) -> bool:
		"""Activate an API to make it available for consumption."""
		try:
			api = await self._get_api_by_id(api_id, tenant_id)
			if not api:
				raise APINotFoundError(f"API {api_id} not found")
			
			# Validate API is ready for activation
			await self._validate_api_for_activation(api)
			
			api.status = APIStatus.ACTIVE.value
			api.updated_by = activated_by
			
			await self.db_session.commit()
			
			# Clear cache and trigger gateway update
			await self._invalidate_api_cache(tenant_id, api_id)
			await self._notify_gateway_update(api_id)
			
			self._log_api_operation("activate", api_id, activated_by)
			return True
			
		except Exception as e:
			await self.db_session.rollback()
			raise APIManagementError(f"Failed to activate API: {str(e)}")
	
	async def deprecate_api(
		self,
		api_id: str,
		tenant_id: str,
		deprecated_by: str,
		deprecation_notice: Optional[str] = None
	) -> bool:
		"""Deprecate an API with optional migration notice."""
		try:
			api = await self._get_api_by_id(api_id, tenant_id)
			if not api:
				raise APINotFoundError(f"API {api_id} not found")
			
			api.status = APIStatus.DEPRECATED.value
			api.updated_by = deprecated_by
			
			# Add deprecation notice to metadata
			if deprecation_notice:
				api.auth_config = api.auth_config or {}
				api.auth_config["deprecation_notice"] = deprecation_notice
				api.auth_config["deprecated_at"] = datetime.now(timezone.utc).isoformat()
			
			await self.db_session.commit()
			
			# Notify consumers about deprecation
			await self._notify_api_deprecation(api_id, deprecation_notice)
			
			self._log_api_operation("deprecate", api_id, deprecated_by)
			return True
			
		except Exception as e:
			await self.db_session.rollback()
			raise APIManagementError(f"Failed to deprecate API: {str(e)}")
	
	async def get_api(self, api_id: str, tenant_id: str) -> Optional[AMAPI]:
		"""Get API details by ID."""
		return await self._get_api_by_id(api_id, tenant_id)
	
	async def list_apis(
		self,
		tenant_id: str,
		capability_id: Optional[str] = None,
		status: Optional[APIStatus] = None,
		public_only: bool = False,
		limit: int = 100,
		offset: int = 0
	) -> Tuple[List[AMAPI], int]:
		"""List APIs with filtering options."""
		try:
			# Build query
			query = select(AMAPI).where(AMAPI.tenant_id == tenant_id)
			
			if capability_id:
				query = query.where(AMAPI.capability_id == capability_id)
			
			if status:
				query = query.where(AMAPI.status == status.value)
			
			if public_only:
				query = query.where(AMAPI.is_public == True)
			
			# Count total
			count_query = select(func.count(AMAPI.api_id)).select_from(query.subquery())
			total_count = (await self.db_session.execute(count_query)).scalar()
			
			# Apply pagination
			query = query.offset(offset).limit(limit).order_by(AMAPI.created_at.desc())
			
			result = await self.db_session.execute(query)
			apis = result.scalars().all()
			
			return list(apis), total_count
			
		except Exception as e:
			raise APIManagementError(f"Failed to list APIs: {str(e)}")
	
	async def add_endpoint(
		self,
		api_id: str,
		config: EndpointConfig,
		tenant_id: str,
		created_by: str
	) -> str:
		"""Add an endpoint to an existing API."""
		try:
			# Validate API exists
			api = await self._get_api_by_id(api_id, tenant_id)
			if not api:
				raise APINotFoundError(f"API {api_id} not found")
			
			# Create endpoint
			endpoint = AMEndpoint(
				api_id=api_id,
				path=config.path,
				method=config.method,
				operation_id=config.operation_id,
				summary=config.summary,
				description=config.description,
				request_schema=config.request_schema,
				response_schema=config.response_schema,
				parameters=config.parameters,
				auth_required=config.auth_required,
				scopes_required=config.scopes_required,
				rate_limit_override=config.rate_limit_override,
				cache_enabled=config.cache_enabled,
				cache_ttl_seconds=config.cache_ttl_seconds,
				deprecated=config.deprecated,
				examples=config.examples
			)
			
			self.db_session.add(endpoint)
			await self.db_session.commit()
			
			# Update API schema if OpenAPI spec exists
			await self._update_openapi_spec(api_id)
			
			self._log_api_operation("add_endpoint", api_id, created_by, 
								   extra={"endpoint": endpoint.endpoint_id})
			return endpoint.endpoint_id
			
		except Exception as e:
			await self.db_session.rollback()
			raise APIManagementError(f"Failed to add endpoint: {str(e)}")
	
	# Private helper methods
	
	async def _validate_api_config(self, config: APIConfig, tenant_id: str):
		"""Validate API configuration for registration."""
		# Check for duplicate API name/version
		existing = await self.db_session.execute(
			select(AMAPI).where(
				and_(
					AMAPI.api_name == config.api_name,
					AMAPI.version == config.version,
					AMAPI.tenant_id == tenant_id
				)
			)
		)
		if existing.scalar():
			raise APIManagementError(f"API {config.api_name} version {config.version} already exists")
		
		# Validate upstream URL accessibility
		await self._validate_upstream_url(config.upstream_url)
	
	async def _validate_upstream_url(self, url: str):
		"""Validate that upstream URL is accessible."""
		try:
			async with aiohttp.ClientSession() as session:
				async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
					if response.status >= 500:
						raise APIManagementError(f"Upstream URL {url} returned server error")
		except aiohttp.ClientError:
			# Log warning but don't fail registration
			pass
	
	async def _get_api_by_id(self, api_id: str, tenant_id: str) -> Optional[AMAPI]:
		"""Get API by ID with tenant isolation."""
		cache_key = f"api:{tenant_id}:{api_id}"
		
		# Try cache first
		cached = await self.cache.get(cache_key)
		if cached:
			return cached
		
		# Query database
		result = await self.db_session.execute(
			select(AMAPI).where(
				and_(AMAPI.api_id == api_id, AMAPI.tenant_id == tenant_id)
			)
		)
		api = result.scalar_one_or_none()
		
		# Cache result
		if api:
			await self.cache.set(cache_key, api, ttl=300)
		
		return api
	
	async def _validate_api_for_activation(self, api: AMAPI):
		"""Validate API is ready for activation."""
		# Check if API has at least one endpoint
		endpoint_count = await self.db_session.execute(
			select(func.count(AMEndpoint.endpoint_id)).where(AMEndpoint.api_id == api.api_id)
		)
		if endpoint_count.scalar() == 0:
			raise APIManagementError("API must have at least one endpoint to be activated")
	
	async def _invalidate_api_cache(self, tenant_id: str, api_id: Optional[str] = None):
		"""Invalidate API cache entries."""
		if api_id:
			await self.cache.delete(f"api:{tenant_id}:{api_id}")
		else:
			# Clear all API cache for tenant
			await self.cache.delete_pattern(f"api:{tenant_id}:*")
	
	async def _notify_gateway_update(self, api_id: str):
		"""Notify gateway about API updates."""
		# This would typically publish an event to update gateway configuration
		pass
	
	async def _notify_api_deprecation(self, api_id: str, notice: Optional[str]):
		"""Notify API consumers about deprecation."""
		# This would send notifications to all API consumers
		pass
	
	async def _update_openapi_spec(self, api_id: str):
		"""Update OpenAPI specification after endpoint changes."""
		# This would regenerate the OpenAPI spec from endpoint definitions
		pass
	
	def _log_api_operation(self, operation: str, api_id: str, user: str, extra: Optional[Dict] = None):
		"""Log API management operations."""
		log_data = {
			"operation": operation,
			"api_id": api_id,
			"user": user,
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
		if extra:
			log_data.update(extra)
		
		print(f"{self._log_prefix} {operation.upper()}: {json.dumps(log_data)}")

# =============================================================================
# Consumer Management Service
# =============================================================================

class ConsumerManagementService:
	"""Service for managing API consumers and authentication."""
	
	def __init__(self, db_session: AsyncSession, cache: aiocache.Cache, security_config: SecurityConfig):
		self.db_session = db_session
		self.cache = cache
		self.security_config = security_config
		self._log_prefix = "[ConsumerManagementService]"
	
	async def register_consumer(
		self,
		config: ConsumerConfig,
		tenant_id: str,
		created_by: str
	) -> str:
		"""Register a new API consumer."""
		try:
			# Create consumer record
			consumer = AMConsumer(
				consumer_name=config.consumer_name,
				organization=config.organization,
				contact_email=config.contact_email,
				contact_name=config.contact_name,
				allowed_apis=config.allowed_apis,
				ip_whitelist=config.ip_whitelist,
				global_rate_limit=config.global_rate_limit,
				global_quota_limit=config.global_quota_limit,
				portal_access=config.portal_access,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
			self.db_session.add(consumer)
			await self.db_session.commit()
			
			self._log_consumer_operation("register", consumer.consumer_id, created_by)
			return consumer.consumer_id
			
		except Exception as e:
			await self.db_session.rollback()
			raise APIManagementError(f"Failed to register consumer: {str(e)}")
	
	async def create_api_key(
		self,
		consumer_id: str,
		config: APIKeyConfig,
		tenant_id: str,
		created_by: str
	) -> Tuple[str, str]:
		"""Create a new API key for a consumer."""
		try:
			# Validate consumer exists
			consumer = await self._get_consumer_by_id(consumer_id, tenant_id)
			if not consumer:
				raise ConsumerNotFoundError(f"Consumer {consumer_id} not found")
			
			# Generate API key
			api_key = self._generate_api_key()
			key_hash = self._hash_api_key(api_key)
			key_prefix = api_key[:8]
			
			# Create API key record
			api_key_record = AMAPIKey(
				consumer_id=consumer_id,
				key_name=config.key_name,
				key_hash=key_hash,
				key_prefix=key_prefix,
				scopes=config.scopes,
				allowed_apis=config.allowed_apis,
				expires_at=config.expires_at,
				rate_limit_override=config.rate_limit_override,
				quota_limit_override=config.quota_limit_override,
				ip_restrictions=config.ip_restrictions,
				referer_restrictions=config.referer_restrictions,
				created_by=created_by
			)
			
			self.db_session.add(api_key_record)
			await self.db_session.commit()
			
			self._log_consumer_operation("create_api_key", consumer_id, created_by,
										extra={"key_id": api_key_record.key_id})
			return api_key_record.key_id, api_key
			
		except Exception as e:
			await self.db_session.rollback()
			raise APIManagementError(f"Failed to create API key: {str(e)}")
	
	async def authenticate_api_key(self, api_key: str) -> Optional[AMAPIKey]:
		"""Authenticate an API key and return key details."""
		try:
			key_hash = self._hash_api_key(api_key)
			
			# Check cache first
			cache_key = f"api_key:{key_hash}"
			cached = await self.cache.get(cache_key)
			if cached:
				return cached
			
			# Query database
			result = await self.db_session.execute(
				select(AMAPIKey).where(
					and_(
						AMAPIKey.key_hash == key_hash,
						AMAPIKey.active == True,
						or_(
							AMAPIKey.expires_at.is_(None),
							AMAPIKey.expires_at > datetime.now(timezone.utc)
						)
					)
				)
			)
			api_key_record = result.scalar_one_or_none()
			
			if api_key_record:
				# Update last used timestamp
				api_key_record.last_used_at = datetime.now(timezone.utc)
				await self.db_session.commit()
				
				# Cache the result
				await self.cache.set(cache_key, api_key_record, ttl=300)
			
			return api_key_record
			
		except Exception as e:
			raise AuthenticationError(f"Failed to authenticate API key: {str(e)}")
	
	async def authorize_api_access(
		self,
		api_key_record: AMAPIKey,
		api_id: str,
		endpoint_path: str,
		scopes_required: List[str]
	) -> bool:
		"""Authorize API access for a consumer."""
		try:
			# Check if API is in allowed list (empty list means all APIs allowed)
			if api_key_record.allowed_apis and api_id not in api_key_record.allowed_apis:
				return False
			
			# Check scopes
			if scopes_required:
				if not api_key_record.scopes:
					return False
				if not all(scope in api_key_record.scopes for scope in scopes_required):
					return False
			
			# Additional authorization checks could be added here
			return True
			
		except Exception as e:
			raise AuthorizationError(f"Failed to authorize API access: {str(e)}")
	
	async def revoke_api_key(self, key_id: str, tenant_id: str, revoked_by: str) -> bool:
		"""Revoke an API key."""
		try:
			# Get API key with consumer tenant check
			result = await self.db_session.execute(
				select(AMAPIKey).join(AMConsumer).where(
					and_(
						AMAPIKey.key_id == key_id,
						AMConsumer.tenant_id == tenant_id
					)
				)
			)
			api_key_record = result.scalar_one_or_none()
			
			if not api_key_record:
				raise APIManagementError(f"API key {key_id} not found")
			
			api_key_record.active = False
			await self.db_session.commit()
			
			# Clear cache
			await self.cache.delete(f"api_key:{api_key_record.key_hash}")
			
			self._log_consumer_operation("revoke_api_key", api_key_record.consumer_id, revoked_by,
										extra={"key_id": key_id})
			return True
			
		except Exception as e:
			await self.db_session.rollback()
			raise APIManagementError(f"Failed to revoke API key: {str(e)}")
	
	async def list_consumers(
		self,
		tenant_id: str,
		status: Optional[str] = None,
		limit: int = 100,
		offset: int = 0
	) -> Tuple[List[AMConsumer], int]:
		"""List consumers with filtering options."""
		try:
			query = select(AMConsumer).where(AMConsumer.tenant_id == tenant_id)
			
			if status:
				query = query.where(AMConsumer.status == status)
			
			# Count total
			count_query = select(func.count(AMConsumer.consumer_id)).select_from(query.subquery())
			total_count = (await self.db_session.execute(count_query)).scalar()
			
			# Apply pagination
			query = query.offset(offset).limit(limit).order_by(AMConsumer.created_at.desc())
			
			result = await self.db_session.execute(query)
			consumers = result.scalars().all()
			
			return list(consumers), total_count
			
		except Exception as e:
			raise APIManagementError(f"Failed to list consumers: {str(e)}")
	
	# Private helper methods
	
	def _generate_api_key(self) -> str:
		"""Generate a secure API key."""
		return secrets.token_urlsafe(self.security_config.api_key_length)
	
	def _hash_api_key(self, api_key: str) -> str:
		"""Hash an API key for secure storage."""
		return hashlib.pbkdf2_hmac(
			'sha256',
			api_key.encode('utf-8'),
			self.security_config.jwt_secret_key.encode('utf-8'),
			self.security_config.password_hash_iterations
		).hex()
	
	async def _get_consumer_by_id(self, consumer_id: str, tenant_id: str) -> Optional[AMConsumer]:
		"""Get consumer by ID with tenant isolation."""
		result = await self.db_session.execute(
			select(AMConsumer).where(
				and_(AMConsumer.consumer_id == consumer_id, AMConsumer.tenant_id == tenant_id)
			)
		)
		return result.scalar_one_or_none()
	
	def _log_consumer_operation(self, operation: str, consumer_id: str, user: str, extra: Optional[Dict] = None):
		"""Log consumer management operations."""
		log_data = {
			"operation": operation,
			"consumer_id": consumer_id,
			"user": user,
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
		if extra:
			log_data.update(extra)
		
		print(f"{self._log_prefix} {operation.upper()}: {json.dumps(log_data)}")

# =============================================================================
# Policy Management Service
# =============================================================================

class PolicyManagementService:
	"""Service for managing API policies (rate limiting, security, etc.)."""
	
	def __init__(self, db_session: AsyncSession, cache: aiocache.Cache):
		self.db_session = db_session
		self.cache = cache
		self._log_prefix = "[PolicyManagementService]"
	
	async def create_policy(
		self,
		api_id: str,
		config: PolicyConfig,
		tenant_id: str,
		created_by: str
	) -> str:
		"""Create a new policy for an API."""
		try:
			# Validate API exists
			api = await self._get_api_by_id(api_id, tenant_id)
			if not api:
				raise APINotFoundError(f"API {api_id} not found")
			
			# Validate policy configuration
			await self._validate_policy_config(config)
			
			# Create policy
			policy = AMPolicy(
				api_id=api_id,
				policy_name=config.policy_name,
				policy_type=config.policy_type.value,
				policy_description=config.policy_description,
				config=config.config,
				execution_order=config.execution_order,
				enabled=config.enabled,
				conditions=config.conditions,
				applies_to_endpoints=config.applies_to_endpoints,
				created_by=created_by
			)
			
			self.db_session.add(policy)
			await self.db_session.commit()
			
			# Clear policy cache
			await self._invalidate_policy_cache(api_id)
			
			self._log_policy_operation("create", policy.policy_id, created_by)
			return policy.policy_id
			
		except Exception as e:
			await self.db_session.rollback()
			raise APIManagementError(f"Failed to create policy: {str(e)}")
	
	async def get_api_policies(self, api_id: str, tenant_id: str) -> List[AMPolicy]:
		"""Get all policies for an API."""
		try:
			# Check cache first
			cache_key = f"policies:{api_id}"
			cached = await self.cache.get(cache_key)
			if cached:
				return cached
			
			# Validate API exists and belongs to tenant
			api = await self._get_api_by_id(api_id, tenant_id)
			if not api:
				raise APINotFoundError(f"API {api_id} not found")
			
			# Query policies
			result = await self.db_session.execute(
				select(AMPolicy).where(AMPolicy.api_id == api_id)
				.order_by(AMPolicy.execution_order, AMPolicy.created_at)
			)
			policies = list(result.scalars().all())
			
			# Cache result
			await self.cache.set(cache_key, policies, ttl=300)
			
			return policies
			
		except Exception as e:
			raise APIManagementError(f"Failed to get API policies: {str(e)}")
	
	async def update_policy(
		self,
		policy_id: str,
		config: PolicyConfig,
		tenant_id: str,
		updated_by: str
	) -> bool:
		"""Update an existing policy."""
		try:
			# Get policy with tenant validation
			policy = await self._get_policy_by_id(policy_id, tenant_id)
			if not policy:
				raise APIManagementError(f"Policy {policy_id} not found")
			
			# Validate policy configuration
			await self._validate_policy_config(config)
			
			# Update policy
			policy.policy_name = config.policy_name
			policy.policy_description = config.policy_description
			policy.config = config.config
			policy.execution_order = config.execution_order
			policy.enabled = config.enabled
			policy.conditions = config.conditions
			policy.applies_to_endpoints = config.applies_to_endpoints
			
			await self.db_session.commit()
			
			# Clear cache
			await self._invalidate_policy_cache(policy.api_id)
			
			self._log_policy_operation("update", policy_id, updated_by)
			return True
			
		except Exception as e:
			await self.db_session.rollback()
			raise APIManagementError(f"Failed to update policy: {str(e)}")
	
	async def delete_policy(self, policy_id: str, tenant_id: str, deleted_by: str) -> bool:
		"""Delete a policy."""
		try:
			# Get policy with tenant validation
			policy = await self._get_policy_by_id(policy_id, tenant_id)
			if not policy:
				raise APIManagementError(f"Policy {policy_id} not found")
			
			api_id = policy.api_id
			
			await self.db_session.delete(policy)
			await self.db_session.commit()
			
			# Clear cache
			await self._invalidate_policy_cache(api_id)
			
			self._log_policy_operation("delete", policy_id, deleted_by)
			return True
			
		except Exception as e:
			await self.db_session.rollback()
			raise APIManagementError(f"Failed to delete policy: {str(e)}")
	
	# Private helper methods
	
	async def _validate_policy_config(self, config: PolicyConfig):
		"""Validate policy configuration."""
		# Basic validation based on policy type
		if config.policy_type == PolicyType.RATE_LIMITING:
			required_fields = ['requests_per_minute', 'window_size_seconds']
			for field in required_fields:
				if field not in config.config:
					raise APIManagementError(f"Rate limiting policy requires '{field}' configuration")
		
		elif config.policy_type == PolicyType.AUTHENTICATION:
			if 'type' not in config.config:
				raise APIManagementError("Authentication policy requires 'type' configuration")
	
	async def _get_api_by_id(self, api_id: str, tenant_id: str) -> Optional[AMAPI]:
		"""Get API by ID with tenant validation."""
		result = await self.db_session.execute(
			select(AMAPI).where(
				and_(AMAPI.api_id == api_id, AMAPI.tenant_id == tenant_id)
			)
		)
		return result.scalar_one_or_none()
	
	async def _get_policy_by_id(self, policy_id: str, tenant_id: str) -> Optional[AMPolicy]:
		"""Get policy by ID with tenant validation."""
		result = await self.db_session.execute(
			select(AMPolicy).join(AMAPI).where(
				and_(
					AMPolicy.policy_id == policy_id,
					AMAPI.tenant_id == tenant_id
				)
			)
		)
		return result.scalar_one_or_none()
	
	async def _invalidate_policy_cache(self, api_id: str):
		"""Invalidate policy cache for an API."""
		await self.cache.delete(f"policies:{api_id}")
	
	def _log_policy_operation(self, operation: str, policy_id: str, user: str, extra: Optional[Dict] = None):
		"""Log policy management operations."""
		log_data = {
			"operation": operation,
			"policy_id": policy_id,
			"user": user,
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
		if extra:
			log_data.update(extra)
		
		print(f"{self._log_prefix} {operation.upper()}: {json.dumps(log_data)}")

# =============================================================================
# Analytics Service
# =============================================================================

class AnalyticsService:
	"""Service for collecting and analyzing API usage metrics."""
	
	def __init__(self, db_session: AsyncSession, cache: aiocache.Cache):
		self.db_session = db_session
		self.cache = cache
		self._log_prefix = "[AnalyticsService]"
	
	async def record_api_usage(
		self,
		request_id: str,
		consumer_id: str,
		api_id: str,
		endpoint_path: str,
		method: str,
		response_status: int,
		response_time_ms: int,
		request_size_bytes: Optional[int] = None,
		response_size_bytes: Optional[int] = None,
		client_ip: Optional[str] = None,
		user_agent: Optional[str] = None,
		tenant_id: str = "default"
	):
		"""Record API usage for analytics and billing."""
		try:
			usage_record = AMUsageRecord(
				request_id=request_id,
				consumer_id=consumer_id,
				api_id=api_id,
				endpoint_path=endpoint_path,
				method=method,
				timestamp=datetime.now(timezone.utc),
				response_status=response_status,
				response_time_ms=response_time_ms,
				request_size_bytes=request_size_bytes,
				response_size_bytes=response_size_bytes,
				client_ip=client_ip,
				user_agent=user_agent,
				tenant_id=tenant_id
			)
			
			self.db_session.add(usage_record)
			await self.db_session.commit()
			
			# Update real-time metrics
			await self._update_realtime_metrics(api_id, consumer_id, response_status, response_time_ms)
			
		except Exception as e:
			# Don't fail the API request if analytics recording fails
			print(f"{self._log_prefix} Failed to record usage: {str(e)}")
	
	async def get_api_metrics(
		self,
		api_id: str,
		tenant_id: str,
		start_time: Optional[datetime] = None,
		end_time: Optional[datetime] = None,
		granularity: str = "hour"
	) -> Dict[str, Any]:
		"""Get aggregated metrics for an API."""
		try:
			if not start_time:
				start_time = datetime.now(timezone.utc) - timedelta(days=7)
			if not end_time:
				end_time = datetime.now(timezone.utc)
			
			# Request count
			request_count = await self._get_request_count(api_id, start_time, end_time)
			
			# Error rate
			error_rate = await self._get_error_rate(api_id, start_time, end_time)
			
			# Average response time
			avg_response_time = await self._get_average_response_time(api_id, start_time, end_time)
			
			# Top consumers
			top_consumers = await self._get_top_consumers(api_id, start_time, end_time)
			
			# Time series data
			time_series = await self._get_time_series_data(api_id, start_time, end_time, granularity)
			
			return {
				"api_id": api_id,
				"period": {
					"start": start_time.isoformat(),
					"end": end_time.isoformat()
				},
				"summary": {
					"total_requests": request_count,
					"error_rate": error_rate,
					"average_response_time_ms": avg_response_time
				},
				"top_consumers": top_consumers,
				"time_series": time_series
			}
			
		except Exception as e:
			raise APIManagementError(f"Failed to get API metrics: {str(e)}")
	
	async def get_consumer_usage(
		self,
		consumer_id: str,
		tenant_id: str,
		start_time: Optional[datetime] = None,
		end_time: Optional[datetime] = None
	) -> Dict[str, Any]:
		"""Get usage statistics for a consumer."""
		try:
			if not start_time:
				start_time = datetime.now(timezone.utc) - timedelta(days=30)
			if not end_time:
				end_time = datetime.now(timezone.utc)
			
			# Total requests
			total_requests = await self._get_consumer_request_count(consumer_id, start_time, end_time)
			
			# API breakdown
			api_breakdown = await self._get_consumer_api_breakdown(consumer_id, start_time, end_time)
			
			# Daily usage
			daily_usage = await self._get_consumer_daily_usage(consumer_id, start_time, end_time)
			
			return {
				"consumer_id": consumer_id,
				"period": {
					"start": start_time.isoformat(),
					"end": end_time.isoformat()
				},
				"total_requests": total_requests,
				"api_breakdown": api_breakdown,
				"daily_usage": daily_usage
			}
			
		except Exception as e:
			raise APIManagementError(f"Failed to get consumer usage: {str(e)}")
	
	# Private helper methods for analytics calculations
	
	async def _update_realtime_metrics(self, api_id: str, consumer_id: str, status: int, response_time: int):
		"""Update real-time metrics in cache."""
		# This would update Redis counters for real-time dashboard
		pass
	
	async def _get_request_count(self, api_id: str, start_time: datetime, end_time: datetime) -> int:
		"""Get total request count for an API in time range."""
		result = await self.db_session.execute(
			select(func.count(AMUsageRecord.record_id)).where(
				and_(
					AMUsageRecord.api_id == api_id,
					AMUsageRecord.timestamp >= start_time,
					AMUsageRecord.timestamp <= end_time
				)
			)
		)
		return result.scalar() or 0
	
	async def _get_error_rate(self, api_id: str, start_time: datetime, end_time: datetime) -> float:
		"""Calculate error rate for an API."""
		total_result = await self.db_session.execute(
			select(func.count(AMUsageRecord.record_id)).where(
				and_(
					AMUsageRecord.api_id == api_id,
					AMUsageRecord.timestamp >= start_time,
					AMUsageRecord.timestamp <= end_time
				)
			)
		)
		total = total_result.scalar() or 0
		
		if total == 0:
			return 0.0
		
		error_result = await self.db_session.execute(
			select(func.count(AMUsageRecord.record_id)).where(
				and_(
					AMUsageRecord.api_id == api_id,
					AMUsageRecord.timestamp >= start_time,
					AMUsageRecord.timestamp <= end_time,
					AMUsageRecord.response_status >= 400
				)
			)
		)
		errors = error_result.scalar() or 0
		
		return (errors / total) * 100
	
	async def _get_average_response_time(self, api_id: str, start_time: datetime, end_time: datetime) -> float:
		"""Get average response time for an API."""
		result = await self.db_session.execute(
			select(func.avg(AMUsageRecord.response_time_ms)).where(
				and_(
					AMUsageRecord.api_id == api_id,
					AMUsageRecord.timestamp >= start_time,
					AMUsageRecord.timestamp <= end_time
				)
			)
		)
		return result.scalar() or 0.0
	
	async def _get_top_consumers(self, api_id: str, start_time: datetime, end_time: datetime, limit: int = 10) -> List[Dict]:
		"""Get top consumers by request count."""
		result = await self.db_session.execute(
			select(
				AMUsageRecord.consumer_id,
				func.count(AMUsageRecord.record_id).label('request_count')
			).where(
				and_(
					AMUsageRecord.api_id == api_id,
					AMUsageRecord.timestamp >= start_time,
					AMUsageRecord.timestamp <= end_time
				)
			).group_by(AMUsageRecord.consumer_id)
			.order_by(func.count(AMUsageRecord.record_id).desc())
			.limit(limit)
		)
		
		return [
			{"consumer_id": row.consumer_id, "request_count": row.request_count}
			for row in result
		]
	
	async def _get_time_series_data(
		self,
		api_id: str,
		start_time: datetime,
		end_time: datetime,
		granularity: str
	) -> List[Dict]:
		"""Get time series data for an API."""
		# This would generate time series data based on granularity
		# Simplified implementation
		return []
	
	async def _get_consumer_request_count(self, consumer_id: str, start_time: datetime, end_time: datetime) -> int:
		"""Get total request count for a consumer."""
		result = await self.db_session.execute(
			select(func.count(AMUsageRecord.record_id)).where(
				and_(
					AMUsageRecord.consumer_id == consumer_id,
					AMUsageRecord.timestamp >= start_time,
					AMUsageRecord.timestamp <= end_time
				)
			)
		)
		return result.scalar() or 0
	
	async def _get_consumer_api_breakdown(self, consumer_id: str, start_time: datetime, end_time: datetime) -> List[Dict]:
		"""Get API usage breakdown for a consumer."""
		result = await self.db_session.execute(
			select(
				AMUsageRecord.api_id,
				func.count(AMUsageRecord.record_id).label('request_count')
			).where(
				and_(
					AMUsageRecord.consumer_id == consumer_id,
					AMUsageRecord.timestamp >= start_time,
					AMUsageRecord.timestamp <= end_time
				)
			).group_by(AMUsageRecord.api_id)
			.order_by(func.count(AMUsageRecord.record_id).desc())
		)
		
		return [
			{"api_id": row.api_id, "request_count": row.request_count}
			for row in result
		]
	
	async def _get_consumer_daily_usage(self, consumer_id: str, start_time: datetime, end_time: datetime) -> List[Dict]:
		"""Get daily usage data for a consumer."""
		# This would generate daily usage statistics
		# Simplified implementation
		return []

# Export all services
__all__ = [
	"APILifecycleService",
	"ConsumerManagementService",
	"PolicyManagementService",
	"AnalyticsService",
	"GatewayConfig",
	"SecurityConfig",
	"APIManagementError",
	"APINotFoundError",
	"ConsumerNotFoundError",
	"AuthenticationError",
	"AuthorizationError",
	"RateLimitExceededError"
]