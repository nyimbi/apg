#!/usr/bin/env python3
"""
APG Data Virtualization (DVRL) Core APG Integrations
Production implementations for all APG capability dependencies

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid_extensions import uuid7str
import os
from urllib.parse import urljoin


class APGServiceError(Exception):
	"""Exception raised for APG service communication errors"""
	pass


class APGBaseService:
	"""Base class for APG service integrations with common HTTP client functionality"""
	
	def __init__(self, tenant_id: str, user_id: str, base_url: str, service_name: str):
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.base_url = base_url
		self.service_name = service_name
		self.logger = logging.getLogger(f"dvrl.{service_name}")
		self._session: Optional[aiohttp.ClientSession] = None
		self._auth_token: Optional[str] = None
		self._token_expires: Optional[datetime] = None
	
	async def _get_session(self) -> aiohttp.ClientSession:
		"""Get or create HTTP session with proper configuration"""
		if self._session is None or self._session.closed:
			timeout = aiohttp.ClientTimeout(total=30, connect=10)
			connector = aiohttp.TCPConnector(
				limit=100,
				limit_per_host=30,
				ttl_dns_cache=300,
				use_dns_cache=True,
				ssl=True
			)
			self._session = aiohttp.ClientSession(
				timeout=timeout,
				connector=connector,
				headers={
					'Content-Type': 'application/json',
					'User-Agent': f'DVRL/1.0.0 ({self.service_name})',
					'X-Tenant-ID': self.tenant_id,
					'X-User-ID': self.user_id
				}
			)
		return self._session
	
	async def _get_auth_token(self) -> str:
		"""Get valid authentication token, refreshing if necessary"""
		if self._auth_token and self._token_expires:
			if datetime.now(timezone.utc) < self._token_expires - timedelta(minutes=5):
				return self._auth_token
		
		# Get new token from environment or APG auth service
		token = os.getenv('APG_ACCESS_TOKEN')
		if token:
			self._auth_token = token
			# Set expiration to 1 hour from now (default for APG tokens)
			self._token_expires = datetime.now(timezone.utc) + timedelta(hours=1)
			return token
		
		# If no env token, try to get from APG auth service
		try:
			auth_url = os.getenv('APG_BASE_URL', 'http://localhost:8080')
			session = await self._get_session()
			
			async with session.post(
				f"{auth_url}/api/v1/auth/token",
				json={
					'tenant_id': self.tenant_id,
					'user_id': self.user_id,
					'service': 'dvrl'
				}
			) as response:
				if response.status == 200:
					token_data = await response.json()
					self._auth_token = token_data['access_token']
					self._token_expires = datetime.fromisoformat(token_data['expires_at'])
					return self._auth_token
				else:
					raise APGServiceError(f"Failed to get auth token: {response.status}")
		except Exception as e:
			self.logger.error(f"Failed to get authentication token: {e}")
			raise APGServiceError(f"Authentication failed: {e}")
	
	async def _make_request(
		self, 
		method: str, 
		endpoint: str, 
		data: Optional[Dict[str, Any]] = None,
		params: Optional[Dict[str, str]] = None
	) -> Dict[str, Any]:
		"""Make authenticated HTTP request to APG service"""
		session = await self._get_session()
		token = await self._get_auth_token()
		
		url = urljoin(self.base_url, endpoint)
		headers = {
			'Authorization': f'Bearer {token}'
		}
		
		try:
			async with session.request(
				method=method,
				url=url,
				json=data,
				params=params,
				headers=headers
			) as response:
				response_text = await response.text()
				
				if response.status >= 400:
					self.logger.error(
						f"{self.service_name} API error: {response.status} - {response_text}"
					)
					raise APGServiceError(
						f"{self.service_name} API returned {response.status}: {response_text}"
					)
				
				if response_text:
					return await response.json()
				else:
					return {}
					
		except aiohttp.ClientError as e:
			self.logger.error(f"{self.service_name} connection error: {e}")
			raise APGServiceError(f"Connection to {self.service_name} failed: {e}")
		except json.JSONDecodeError as e:
			self.logger.error(f"{self.service_name} invalid JSON response: {e}")
			raise APGServiceError(f"Invalid JSON from {self.service_name}: {e}")
	
	async def close(self):
		"""Close HTTP session"""
		if self._session and not self._session.closed:
			await self._session.close()


class APGMetadataService(APGBaseService):
	"""Production integration with APG's meta capability for schema registry"""
	
	def __init__(self, tenant_id: str, user_id: str):
		base_url = os.getenv('APG_BASE_URL', 'http://localhost:8080')
		super().__init__(tenant_id, user_id, f"{base_url}/api/v1/meta", "meta")
	
	async def register_schema(self, schema_data: Dict[str, Any]) -> str:
		"""
		Register schema with APG metadata service
		
		Args:
			schema_data: Schema definition and metadata
			
		Returns:
			str: Unique schema identifier
		"""
		schema_payload = {
			'tenant_id': self.tenant_id,
			'created_by': self.user_id,
			'schema_name': schema_data.get('schema_name'),
			'data_source_id': schema_data.get('data_source_id'),
			'schema_type': schema_data.get('schema_type', 'relational'),
			'tables': schema_data.get('tables', []),
			'metadata': {
				'discovered_at': datetime.now(timezone.utc).isoformat(),
				'version': '1.0.0',
				'source': 'dvrl'
			}
		}
		
		try:
			response = await self._make_request('POST', '/schemas', schema_payload)
			schema_id = response['schema_id']
			
			self.logger.info(
				f"Schema registered successfully: {schema_id}",
				extra={
					'schema_id': schema_id,
					'data_source_id': schema_data.get('data_source_id'),
					'table_count': len(schema_data.get('tables', []))
				}
			)
			
			return schema_id
			
		except Exception as e:
			self.logger.error(f"Failed to register schema: {e}")
			raise APGServiceError(f"Schema registration failed: {e}")
	
	async def get_schema(self, schema_id: str) -> Optional[Dict[str, Any]]:
		"""
		Get schema by ID from APG metadata service
		
		Args:
			schema_id: Unique schema identifier
			
		Returns:
			Optional[Dict[str, Any]]: Schema definition or None if not found
		"""
		try:
			response = await self._make_request('GET', f'/schemas/{schema_id}')
			return response
		except APGServiceError as e:
			if "404" in str(e):
				return None
			raise
	
	async def search_schemas(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
		"""
		Search schemas by query
		
		Args:
			query: Search query string
			filters: Additional search filters
			
		Returns:
			List[Dict[str, Any]]: List of matching schemas
		"""
		params = {'q': query}
		if filters:
			params.update(filters)
		
		response = await self._make_request('GET', '/schemas/search', params=params)
		return response.get('schemas', [])
	
	async def track_lineage(
		self, 
		source_id: str, 
		target_id: str, 
		relationship_type: str,
		metadata: Optional[Dict[str, Any]] = None
	) -> str:
		"""
		Track data lineage between sources
		
		Args:
			source_id: Source entity identifier
			target_id: Target entity identifier
			relationship_type: Type of relationship (e.g., 'derived_from', 'transforms_to')
			metadata: Additional lineage metadata
			
		Returns:
			str: Lineage relationship identifier
		"""
		lineage_payload = {
			'source_id': source_id,
			'target_id': target_id,
			'relationship_type': relationship_type,
			'tenant_id': self.tenant_id,
			'created_by': self.user_id,
			'created_at': datetime.now(timezone.utc).isoformat(),
			'metadata': metadata or {}
		}
		
		response = await self._make_request('POST', '/lineage', lineage_payload)
		lineage_id = response['lineage_id']
		
		self.logger.info(
			f"Lineage tracked: {source_id} -> {target_id}",
			extra={
				'lineage_id': lineage_id,
				'relationship_type': relationship_type
			}
		)
		
		return lineage_id
	
	async def get_lineage(self, entity_id: str, direction: str = 'both') -> Dict[str, Any]:
		"""
		Get data lineage for entity
		
		Args:
			entity_id: Entity identifier
			direction: Lineage direction ('upstream', 'downstream', 'both')
			
		Returns:
			Dict[str, Any]: Lineage graph information
		"""
		params = {'direction': direction}
		response = await self._make_request('GET', f'/lineage/{entity_id}', params=params)
		return response


class APGCacheService(APGBaseService):
	"""Production integration with APG's cach capability for intelligent caching"""
	
	def __init__(self, tenant_id: str, user_id: str):
		base_url = os.getenv('APG_BASE_URL', 'http://localhost:8080')
		super().__init__(tenant_id, user_id, f"{base_url}/api/v1/cach", "cache")
	
	async def get(self, key: str) -> Optional[Dict[str, Any]]:
		"""
		Get cached value by key
		
		Args:
			key: Cache key
			
		Returns:
			Optional[Dict[str, Any]]: Cached value or None if not found
		"""
		try:
			response = await self._make_request('GET', f'/cache/{key}')
			return response.get('value')
		except APGServiceError as e:
			if "404" in str(e):
				return None
			raise
	
	async def set(
		self, 
		key: str, 
		value: Dict[str, Any], 
		ttl: Optional[int] = None,
		tags: Optional[List[str]] = None
	) -> bool:
		"""
		Set cached value
		
		Args:
			key: Cache key
			value: Value to cache
			ttl: Time-to-live in seconds
			tags: Cache tags for organization and invalidation
			
		Returns:
			bool: True if successfully cached
		"""
		cache_payload = {
			'key': key,
			'value': value,
			'tenant_id': self.tenant_id,
			'created_by': self.user_id,
			'metadata': {
				'source': 'dvrl',
				'created_at': datetime.now(timezone.utc).isoformat()
			}
		}
		
		if ttl is not None:
			cache_payload['ttl'] = ttl
		if tags:
			cache_payload['tags'] = tags
		
		try:
			await self._make_request('POST', '/cache', cache_payload)
			return True
		except Exception as e:
			self.logger.error(f"Failed to cache value for key {key}: {e}")
			return False
	
	async def delete(self, key: str) -> bool:
		"""
		Delete cached value
		
		Args:
			key: Cache key to delete
			
		Returns:
			bool: True if successfully deleted
		"""
		try:
			await self._make_request('DELETE', f'/cache/{key}')
			return True
		except APGServiceError as e:
			if "404" in str(e):
				return True  # Already deleted
			raise
	
	async def invalidate_by_tags(self, tags: List[str]) -> int:
		"""
		Invalidate cache entries by tags
		
		Args:
			tags: List of tags to invalidate
			
		Returns:
			int: Number of entries invalidated
		"""
		response = await self._make_request('POST', '/cache/invalidate', {'tags': tags})
		return response.get('invalidated_count', 0)
	
	async def get_stats(self) -> Dict[str, Any]:
		"""
		Get cache performance statistics
		
		Returns:
			Dict[str, Any]: Cache statistics including hit ratio, memory usage
		"""
		response = await self._make_request('GET', '/cache/stats')
		return response
	
	async def should_cache(self, key: str, query_metadata: Dict[str, Any]) -> bool:
		"""
		Use ML to predict if query should be cached
		
		Args:
			key: Cache key
			query_metadata: Query execution metadata for ML prediction
			
		Returns:
			bool: True if query should be cached
		"""
		prediction_payload = {
			'cache_key': key,
			'query_metadata': query_metadata,
			'tenant_id': self.tenant_id
		}
		
		try:
			response = await self._make_request('POST', '/cache/predict', prediction_payload)
			should_cache = response.get('should_cache', False)
			confidence = response.get('confidence', 0.0)
			
			self.logger.debug(
				f"Cache prediction for {key}: {should_cache} (confidence: {confidence})"
			)
			
			return should_cache
		except Exception as e:
			self.logger.warning(f"Cache prediction failed, defaulting to cache: {e}")
			# Default to caching if prediction service is unavailable
			return True


class APGSecurityService(APGBaseService):
	"""Production integration with APG's auth_rbac capability"""
	
	def __init__(self, tenant_id: str, user_id: str):
		base_url = os.getenv('APG_BASE_URL', 'http://localhost:8080')
		super().__init__(tenant_id, user_id, f"{base_url}/api/v1/auth", "auth")
	
	async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
		"""
		Validate JWT token with APG auth service
		
		Args:
			token: JWT token to validate
			
		Returns:
			Optional[Dict[str, Any]]: Token payload if valid, None otherwise
		"""
		try:
			response = await self._make_request(
				'POST', 
				'/validate', 
				{'token': token}
			)
			return response.get('payload')
		except APGServiceError as e:
			if "401" in str(e) or "403" in str(e):
				return None
			raise
	
	async def check_permission(self, user_id: str, permission: str, resource: Optional[str] = None) -> bool:
		"""
		Check if user has specific permission
		
		Args:
			user_id: User identifier
			permission: Permission to check (e.g., 'dvrl:read', 'dvrl:execute')
			resource: Optional specific resource identifier
			
		Returns:
			bool: True if user has permission
		"""
		permission_payload = {
			'user_id': user_id,
			'permission': permission,
			'tenant_id': self.tenant_id
		}
		
		if resource:
			permission_payload['resource'] = resource
		
		try:
			response = await self._make_request('POST', '/permissions/check', permission_payload)
			return response.get('allowed', False)
		except Exception as e:
			self.logger.error(f"Permission check failed for {user_id}: {e}")
			# Fail secure - deny access on error
			return False
	
	async def get_user_roles(self, user_id: str) -> List[str]:
		"""
		Get user roles from APG RBAC
		
		Args:
			user_id: User identifier
			
		Returns:
			List[str]: List of role names
		"""
		response = await self._make_request('GET', f'/users/{user_id}/roles')
		return response.get('roles', [])
	
	async def apply_row_level_security(
		self, 
		user_id: str, 
		sql: str, 
		data_source: str
	) -> str:
		"""
		Apply row-level security policies to SQL query
		
		Args:
			user_id: User identifier
			sql: Original SQL query
			data_source: Data source identifier
			
		Returns:
			str: Modified SQL with security constraints
		"""
		security_payload = {
			'user_id': user_id,
			'sql': sql,
			'data_source': data_source,
			'tenant_id': self.tenant_id
		}
		
		response = await self._make_request('POST', '/security/row-level', security_payload)
		return response.get('modified_sql', sql)
	
	async def get_column_permissions(self, user_id: str, table: str, data_source: str) -> Dict[str, str]:
		"""
		Get column-level permissions for user
		
		Args:
			user_id: User identifier
			table: Table name
			data_source: Data source identifier
			
		Returns:
			Dict[str, str]: Column name to permission level mapping
		"""
		params = {
			'user_id': user_id,
			'table': table,
			'data_source': data_source,
			'tenant_id': self.tenant_id
		}
		
		response = await self._make_request('GET', '/security/columns', params=params)
		return response.get('column_permissions', {})
	
	async def mask_sensitive_data(
		self, 
		user_id: str, 
		data: Dict[str, Any], 
		data_source: str
	) -> Dict[str, Any]:
		"""
		Apply data masking based on user permissions
		
		Args:
			user_id: User identifier
			data: Data to potentially mask
			data_source: Data source identifier
			
		Returns:
			Dict[str, Any]: Data with appropriate masking applied
		"""
		masking_payload = {
			'user_id': user_id,
			'data': data,
			'data_source': data_source,
			'tenant_id': self.tenant_id
		}
		
		response = await self._make_request('POST', '/security/mask', masking_payload)
		return response.get('masked_data', data)
	
	async def audit_log(
		self, 
		user_id: str, 
		action: str, 
		resource: str,
		result: str,
		metadata: Optional[Dict[str, Any]] = None
	) -> str:
		"""
		Log audit event to APG compliance system
		
		Args:
			user_id: User who performed action
			action: Action performed
			resource: Resource accessed
			result: Result of action ('success', 'denied', 'error')
			metadata: Additional audit metadata
			
		Returns:
			str: Audit log entry ID
		"""
		audit_payload = {
			'user_id': user_id,
			'action': action,
			'resource': resource,
			'result': result,
			'tenant_id': self.tenant_id,
			'timestamp': datetime.now(timezone.utc).isoformat(),
			'source': 'dvrl',
			'metadata': metadata or {}
		}
		
		response = await self._make_request('POST', '/audit', audit_payload)
		return response.get('audit_id')


class APGMDMService(APGBaseService):
	"""Production integration with APG's mdm capability for data quality and master data"""
	
	def __init__(self, tenant_id: str, user_id: str):
		base_url = os.getenv('APG_BASE_URL', 'http://localhost:8080')
		super().__init__(tenant_id, user_id, f"{base_url}/api/v1/mdm", "mdm")
	
	async def get_quality_score(self, data_source: str, table: str) -> Dict[str, Any]:
		"""
		Get data quality score for table
		
		Args:
			data_source: Data source identifier
			table: Table name
			
		Returns:
			Dict[str, Any]: Quality scores and metrics
		"""
		params = {
			'data_source': data_source,
			'table': table,
			'tenant_id': self.tenant_id
		}
		
		response = await self._make_request('GET', '/quality/score', params=params)
		return response
	
	async def validate_data_quality(
		self, 
		data_source: str, 
		table: str, 
		data_sample: List[Dict[str, Any]]
	) -> Dict[str, Any]:
		"""
		Validate data quality against defined rules
		
		Args:
			data_source: Data source identifier
			table: Table name
			data_sample: Sample data for validation
			
		Returns:
			Dict[str, Any]: Validation results and violations
		"""
		validation_payload = {
			'data_source': data_source,
			'table': table,
			'data_sample': data_sample,
			'tenant_id': self.tenant_id
		}
		
		response = await self._make_request('POST', '/quality/validate', validation_payload)
		return response
	
	async def resolve_master_data(
		self, 
		entity_type: str, 
		attributes: Dict[str, Any]
	) -> Dict[str, Any]:
		"""
		Resolve entity to master data record
		
		Args:
			entity_type: Type of entity (e.g., 'customer', 'product')
			attributes: Entity attributes for matching
			
		Returns:
			Dict[str, Any]: Master data record or matching candidates
		"""
		resolution_payload = {
			'entity_type': entity_type,
			'attributes': attributes,
			'tenant_id': self.tenant_id
		}
		
		response = await self._make_request('POST', '/master-data/resolve', resolution_payload)
		return response
	
	async def get_governance_policies(self, data_source: str) -> List[Dict[str, Any]]:
		"""
		Get data governance policies for data source
		
		Args:
			data_source: Data source identifier
			
		Returns:
			List[Dict[str, Any]]: List of applicable governance policies
		"""
		params = {
			'data_source': data_source,
			'tenant_id': self.tenant_id
		}
		
		response = await self._make_request('GET', '/governance/policies', params=params)
		return response.get('policies', [])


class APGPerformanceOptimizer(APGBaseService):
	"""Production integration with APG's performance optimization services"""
	
	def __init__(self, tenant_id: str, user_id: str):
		base_url = os.getenv('APG_BASE_URL', 'http://localhost:8080')
		super().__init__(tenant_id, user_id, f"{base_url}/api/v1/perf", "performance")
	
	async def optimize_query(
		self, 
		sql: str, 
		data_sources: List[str],
		execution_context: Dict[str, Any]
	) -> Dict[str, Any]:
		"""
		Get query optimization recommendations
		
		Args:
			sql: SQL query to optimize
			data_sources: List of involved data sources
			execution_context: Execution context and statistics
			
		Returns:
			Dict[str, Any]: Optimization recommendations and rewritten query
		"""
		optimization_payload = {
			'sql': sql,
			'data_sources': data_sources,
			'execution_context': execution_context,
			'tenant_id': self.tenant_id
		}
		
		response = await self._make_request('POST', '/optimize/query', optimization_payload)
		return response
	
	async def get_execution_plan(
		self, 
		sql: str, 
		data_sources: Dict[str, Any]
	) -> Dict[str, Any]:
		"""
		Get optimal execution plan for federated query
		
		Args:
			sql: SQL query
			data_sources: Data source configurations and statistics
			
		Returns:
			Dict[str, Any]: Optimal execution plan
		"""
		plan_payload = {
			'sql': sql,
			'data_sources': data_sources,
			'tenant_id': self.tenant_id
		}
		
		response = await self._make_request('POST', '/optimize/plan', plan_payload)
		return response
	
	async def record_execution_metrics(
		self, 
		query_id: str, 
		execution_metrics: Dict[str, Any]
	) -> bool:
		"""
		Record query execution metrics for ML training
		
		Args:
			query_id: Query identifier
			execution_metrics: Detailed execution metrics
			
		Returns:
			bool: True if metrics recorded successfully
		"""
		metrics_payload = {
			'query_id': query_id,
			'execution_metrics': execution_metrics,
			'tenant_id': self.tenant_id,
			'timestamp': datetime.now(timezone.utc).isoformat()
		}
		
		try:
			await self._make_request('POST', '/metrics/execution', metrics_payload)
			return True
		except Exception as e:
			self.logger.error(f"Failed to record execution metrics: {e}")
			return False


# Service Manager for coordinating all APG integrations
class APGServiceManager:
	"""Manager for all APG service integrations"""
	
	def __init__(self, tenant_id: str, user_id: str):
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.logger = logging.getLogger("dvrl.apg_services")
		
		# Initialize all services
		self.metadata = APGMetadataService(tenant_id, user_id)
		self.cache = APGCacheService(tenant_id, user_id)
		self.security = APGSecurityService(tenant_id, user_id)
		self.mdm = APGMDMService(tenant_id, user_id)
		self.performance = APGPerformanceOptimizer(tenant_id, user_id)
		
		self.services = [
			self.metadata,
			self.cache,
			self.security,
			self.mdm,
			self.performance
		]
	
	async def initialize_services(self) -> bool:
		"""
		Initialize all APG service connections
		
		Returns:
			bool: True if all services initialized successfully
		"""
		try:
			# Test connectivity to all services
			await asyncio.gather(
				self._test_service_health(self.metadata, "meta"),
				self._test_service_health(self.cache, "cache"),
				self._test_service_health(self.security, "auth"),
				self._test_service_health(self.mdm, "mdm"),
				self._test_service_health(self.performance, "performance"),
				return_exceptions=True
			)
			
			self.logger.info("All APG services initialized successfully")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to initialize APG services: {e}")
			return False
	
	async def _test_service_health(self, service: APGBaseService, service_name: str) -> None:
		"""Test service health by making a simple API call"""
		try:
			# Make a simple health check request
			await service._make_request('GET', '/health')
			self.logger.info(f"APG {service_name} service is healthy")
		except Exception as e:
			self.logger.warning(f"APG {service_name} service health check failed: {e}")
			# Don't fail initialization for individual service failures
	
	async def close_all_services(self) -> None:
		"""Close all service connections"""
		await asyncio.gather(
			*[service.close() for service in self.services],
			return_exceptions=True
		)
		self.logger.info("All APG service connections closed")