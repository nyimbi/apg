"""
APG Workflow Orchestration REST/GraphQL Connectors

High-performance HTTP-based connectors for REST APIs and GraphQL endpoints
with comprehensive authentication, caching, and error handling.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import ssl
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import logging
from urllib.parse import urljoin, urlparse

import aiohttp
import aiofiles
from pydantic import BaseModel, Field, ConfigDict, HttpUrl, validator

from .base_connector import BaseConnector, ConnectorConfiguration

logger = logging.getLogger(__name__)

class RESTConfiguration(ConnectorConfiguration):
	"""Configuration for REST API connector."""
	
	base_url: HttpUrl = Field(..., description="Base URL for REST API")
	authentication_type: str = Field(default="none", regex="^(none|basic|bearer|oauth2|api_key)$")
	username: Optional[str] = Field(default=None)
	password: Optional[str] = Field(default=None)
	api_key: Optional[str] = Field(default=None)
	api_key_header: str = Field(default="X-API-Key")
	bearer_token: Optional[str] = Field(default=None)
	oauth2_token_url: Optional[HttpUrl] = Field(default=None)
	oauth2_client_id: Optional[str] = Field(default=None)
	oauth2_client_secret: Optional[str] = Field(default=None)
	oauth2_scope: Optional[str] = Field(default=None)
	verify_ssl: bool = Field(default=True)
	max_connections: int = Field(default=100, ge=1, le=1000)
	connection_pool_ttl: int = Field(default=300, ge=60, le=3600)
	request_timeout: int = Field(default=30, ge=1, le=300)
	max_retries: int = Field(default=3, ge=0, le=10)
	rate_limit_requests: int = Field(default=100, ge=1)
	rate_limit_period: int = Field(default=60, ge=1)
	cache_responses: bool = Field(default=False)
	cache_ttl_seconds: int = Field(default=300, ge=1)

class GraphQLConfiguration(ConnectorConfiguration):
	"""Configuration for GraphQL connector."""
	
	endpoint_url: HttpUrl = Field(..., description="GraphQL endpoint URL")
	authentication_type: str = Field(default="none", regex="^(none|basic|bearer|oauth2|api_key)$")
	username: Optional[str] = Field(default=None)
	password: Optional[str] = Field(default=None)
	api_key: Optional[str] = Field(default=None)
	api_key_header: str = Field(default="X-API-Key")
	bearer_token: Optional[str] = Field(default=None)
	introspection_enabled: bool = Field(default=True)
	query_complexity_limit: int = Field(default=1000, ge=1)
	query_depth_limit: int = Field(default=10, ge=1)
	verify_ssl: bool = Field(default=True)
	request_timeout: int = Field(default=30, ge=1, le=300)
	cache_queries: bool = Field(default=True)
	cache_ttl_seconds: int = Field(default=300, ge=1)
	subscription_enabled: bool = Field(default=False)
	websocket_url: Optional[HttpUrl] = Field(default=None)

class RESTConnector(BaseConnector):
	"""High-performance REST API connector."""
	
	def __init__(self, config: RESTConfiguration):
		super().__init__(config)
		self.config: RESTConfiguration = config
		self.session: Optional[aiohttp.ClientSession] = None
		self.auth_token: Optional[str] = None
		self.token_expires_at: Optional[datetime] = None
		self.request_semaphore = asyncio.Semaphore(config.max_connections)
		self.response_cache: Dict[str, Any] = {}
		self.rate_limiter = asyncio.Semaphore(config.rate_limit_requests)
	
	async def _connect(self) -> None:
		"""Initialize HTTP session and authenticate."""
		
		# Create SSL context
		ssl_context = None
		if not self.config.verify_ssl:
			ssl_context = ssl.create_default_context()
			ssl_context.check_hostname = False
			ssl_context.verify_mode = ssl.CERT_NONE
		
		# Create connector with connection pooling
		connector = aiohttp.TCPConnector(
			limit=self.config.max_connections,
			ttl_dns_cache=300,
			use_dns_cache=True,
			ssl=ssl_context
		)
		
		# Create session with timeout
		timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
		
		self.session = aiohttp.ClientSession(
			connector=connector,
			timeout=timeout,
			headers=self.config.custom_headers
		)
		
		# Authenticate if required
		await self._authenticate()
		
		logger.info(self._log_connector_info("REST connector initialized"))
	
	async def _disconnect(self) -> None:
		"""Close HTTP session."""
		if self.session:
			await self.session.close()
			self.session = None
		
		logger.info(self._log_connector_info("REST connector disconnected"))
	
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute REST API operation."""
		
		method = parameters.get("method", "GET").upper()
		endpoint = parameters.get("endpoint", "")
		headers = parameters.get("headers", {})
		data = parameters.get("data")
		params = parameters.get("params", {})
		json_data = parameters.get("json")
		
		# Build full URL
		url = urljoin(str(self.config.base_url), endpoint)
		
		# Check cache for GET requests
		if method == "GET" and self.config.cache_responses:
			cache_key = f"{url}:{json.dumps(params, sort_keys=True)}"
			if cache_key in self.response_cache:
				cached_response = self.response_cache[cache_key]
				if cached_response["expires_at"] > datetime.now(timezone.utc):
					return cached_response["data"]
		
		# Prepare headers with authentication
		request_headers = {**headers}
		await self._add_auth_headers(request_headers)
		
		# Rate limiting
		async with self.rate_limiter:
			async with self.request_semaphore:
				
				# Make HTTP request
				async with self.session.request(
					method=method,
					url=url,
					headers=request_headers,
					params=params,
					data=data,
					json=json_data
				) as response:
					
					# Check response status
					if response.status >= 400:
						error_text = await response.text()
						raise aiohttp.ClientResponseError(
							request_info=response.request_info,
							history=response.history,
							status=response.status,
							message=f"HTTP {response.status}: {error_text}"
						)
					
					# Parse response
					if response.content_type == 'application/json':
						result_data = await response.json()
					else:
						result_data = {"text": await response.text(), "status": response.status}
					
					# Cache response if enabled
					if method == "GET" and self.config.cache_responses:
						self.response_cache[cache_key] = {
							"data": result_data,
							"expires_at": datetime.now(timezone.utc).timestamp() + self.config.cache_ttl_seconds
						}
					
					return {
						"status": response.status,
						"headers": dict(response.headers),
						"data": result_data
					}
	
	async def _health_check(self) -> bool:
		"""Perform REST API health check."""
		try:
			# Try a simple GET request to the base URL or health endpoint
			health_endpoint = "/health" if hasattr(self.config, 'health_endpoint') else ""
			url = urljoin(str(self.config.base_url), health_endpoint)
			
			headers = {}
			await self._add_auth_headers(headers)
			
			async with self.session.get(url, headers=headers) as response:
				return response.status < 400
				
		except Exception as e:
			logger.warning(self._log_connector_info(f"Health check failed: {e}"))
			return False
	
	async def _authenticate(self) -> None:
		"""Handle authentication based on configuration."""
		
		if self.config.authentication_type == "oauth2":
			await self._oauth2_authenticate()
		elif self.config.authentication_type == "bearer" and self.config.bearer_token:
			self.auth_token = self.config.bearer_token
		# Basic and API key authentication handled in _add_auth_headers
	
	async def _oauth2_authenticate(self) -> None:
		"""Perform OAuth2 authentication."""
		if not all([self.config.oauth2_token_url, self.config.oauth2_client_id, self.config.oauth2_client_secret]):
			raise ValueError("OAuth2 configuration incomplete")
		
		data = {
			"grant_type": "client_credentials",
			"client_id": self.config.oauth2_client_id,
			"client_secret": self.config.oauth2_client_secret
		}
		
		if self.config.oauth2_scope:
			data["scope"] = self.config.oauth2_scope
		
		async with self.session.post(str(self.config.oauth2_token_url), data=data) as response:
			if response.status != 200:
				raise aiohttp.ClientResponseError(
					request_info=response.request_info,
					history=response.history,
					status=response.status,
					message="OAuth2 authentication failed"
				)
			
			token_data = await response.json()
			self.auth_token = token_data["access_token"]
			
			# Calculate token expiration
			if "expires_in" in token_data:
				self.token_expires_at = datetime.now(timezone.utc).timestamp() + token_data["expires_in"]
	
	async def _add_auth_headers(self, headers: Dict[str, str]) -> None:
		"""Add authentication headers to request."""
		
		# Check if OAuth2 token needs refresh
		if (self.config.authentication_type == "oauth2" and 
			self.token_expires_at and 
			datetime.now(timezone.utc).timestamp() > self.token_expires_at - 60):  # Refresh 1 minute before expiry
			await self._oauth2_authenticate()
		
		if self.config.authentication_type == "basic" and self.config.username and self.config.password:
			import base64
			credentials = base64.b64encode(f"{self.config.username}:{self.config.password}".encode()).decode()
			headers["Authorization"] = f"Basic {credentials}"
		
		elif self.config.authentication_type == "bearer" and self.auth_token:
			headers["Authorization"] = f"Bearer {self.auth_token}"
		
		elif self.config.authentication_type == "api_key" and self.config.api_key:
			headers[self.config.api_key_header] = self.config.api_key
	
	async def _health_check(self) -> bool:
		"""Perform health check against REST API."""
		try:
			if not self.session:
				return False
			
			# Use health check endpoint if configured, otherwise use base URL
			health_url = str(self.config.base_url)
			if hasattr(self.config, 'health_endpoint') and self.config.health_endpoint:
				health_url = urljoin(str(self.config.base_url), self.config.health_endpoint)
			
			# Perform health check request with timeout
			async with self.request_semaphore:
				headers = {}
				await self._add_auth_headers(headers)
				
				async with self.session.get(
					health_url,
					headers=headers,
					timeout=aiohttp.ClientTimeout(total=5.0)  # 5 second timeout for health check
				) as response:
					# Consider 2xx and 3xx status codes as healthy
					return 200 <= response.status < 400
					
		except asyncio.TimeoutError:
			logger.warning(f"Health check timeout for REST API: {self.config.base_url}")
			return False
		except Exception as e:
			logger.warning(f"Health check failed for REST API: {self.config.base_url}, error: {e}")
			return False

class GraphQLConnector(BaseConnector):
	"""High-performance GraphQL connector."""
	
	def __init__(self, config: GraphQLConfiguration):
		super().__init__(config)
		self.config: GraphQLConfiguration = config
		self.session: Optional[aiohttp.ClientSession] = None
		self.schema_introspection: Optional[Dict[str, Any]] = None
		self.query_cache: Dict[str, Any] = {}
		self.websocket_session: Optional[aiohttp.ClientWebSocketResponse] = None
	
	async def _connect(self) -> None:
		"""Initialize GraphQL connection."""
		
		# Create SSL context
		ssl_context = None
		if not self.config.verify_ssl:
			ssl_context = ssl.create_default_context()
			ssl_context.check_hostname = False
			ssl_context.verify_mode = ssl.CERT_NONE
		
		# Create session
		timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
		self.session = aiohttp.ClientSession(
			timeout=timeout,
			headers=self.config.custom_headers
		)
		
		# Perform introspection if enabled
		if self.config.introspection_enabled:
			await self._introspect_schema()
		
		# Initialize WebSocket for subscriptions if enabled
		if self.config.subscription_enabled and self.config.websocket_url:
			await self._connect_websocket()
		
		logger.info(self._log_connector_info("GraphQL connector initialized"))
	
	async def _disconnect(self) -> None:
		"""Close GraphQL connection."""
		if self.websocket_session and not self.websocket_session.closed:
			await self.websocket_session.close()
		
		if self.session:
			await self.session.close()
			self.session = None
		
		logger.info(self._log_connector_info("GraphQL connector disconnected"))
	
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute GraphQL operation."""
		
		query = parameters.get("query", "")
		variables = parameters.get("variables", {})
		operation_name = parameters.get("operationName")
		
		if not query:
			raise ValueError("GraphQL query is required")
		
		# Validate query complexity and depth
		self._validate_query(query)
		
		# Check cache for queries
		if operation == "query" and self.config.cache_queries:
			cache_key = f"{query}:{json.dumps(variables, sort_keys=True)}"
			if cache_key in self.query_cache:
				cached_response = self.query_cache[cache_key]
				if cached_response["expires_at"] > datetime.now(timezone.utc).timestamp():
					return cached_response["data"]
		
		# Prepare request payload
		payload = {
			"query": query,
			"variables": variables
		}
		
		if operation_name:
			payload["operationName"] = operation_name
		
		# Prepare headers
		headers = {"Content-Type": "application/json"}
		await self._add_auth_headers(headers)
		
		# Execute GraphQL request
		async with self.session.post(
			str(self.config.endpoint_url),
			json=payload,
			headers=headers
		) as response:
			
			if response.status != 200:
				error_text = await response.text()
				raise aiohttp.ClientResponseError(
					request_info=response.request_info,
					history=response.history,
					status=response.status,
					message=f"GraphQL HTTP {response.status}: {error_text}"
				)
			
			result_data = await response.json()
			
			# Check for GraphQL errors
			if "errors" in result_data and result_data["errors"]:
				error_messages = [error.get("message", "Unknown error") for error in result_data["errors"]]
				raise Exception(f"GraphQL errors: {'; '.join(error_messages)}")
			
			# Cache successful queries
			if operation == "query" and self.config.cache_queries and "data" in result_data:
				self.query_cache[cache_key] = {
					"data": result_data,
					"expires_at": datetime.now(timezone.utc).timestamp() + self.config.cache_ttl_seconds
				}
			
			return result_data
	
	async def _health_check(self) -> bool:
		"""Perform GraphQL health check using introspection."""
		try:
			# Simple introspection query
			introspection_query = """
			query {
				__schema {
					queryType {
						name
					}
				}
			}
			"""
			
			headers = {"Content-Type": "application/json"}
			await self._add_auth_headers(headers)
			
			async with self.session.post(
				str(self.config.endpoint_url),
				json={"query": introspection_query},
				headers=headers
			) as response:
				
				if response.status == 200:
					result = await response.json()
					return "errors" not in result or not result["errors"]
				return False
				
		except Exception as e:
			logger.warning(self._log_connector_info(f"Health check failed: {e}"))
			return False
	
	async def _introspect_schema(self) -> None:
		"""Perform GraphQL schema introspection."""
		introspection_query = """
		query IntrospectionQuery {
			__schema {
				queryType { name }
				mutationType { name }
				subscriptionType { name }
				types {
					...FullType
				}
			}
		}
		
		fragment FullType on __Type {
			kind name description
			fields(includeDeprecated: true) {
				name description args { ...InputValue }
				type { ...TypeRef }
				isDeprecated deprecationReason
			}
			inputFields { ...InputValue }
			interfaces { ...TypeRef }
			enumValues(includeDeprecated: true) {
				name description isDeprecated deprecationReason
			}
			possibleTypes { ...TypeRef }
		}
		
		fragment InputValue on __InputValue {
			name description type { ...TypeRef } defaultValue
		}
		
		fragment TypeRef on __Type {
			kind name
			ofType {
				kind name
				ofType {
					kind name
					ofType {
						kind name
						ofType {
							kind name
							ofType {
								kind name
								ofType {
									kind name
									ofType { kind name }
								}
							}
						}
					}
				}
			}
		}
		"""
		
		try:
			result = await self._execute_operation("query", {"query": introspection_query})
			self.schema_introspection = result.get("data", {}).get("__schema")
			logger.info(self._log_connector_info("Schema introspection completed"))
		except Exception as e:
			logger.warning(self._log_connector_info(f"Schema introspection failed: {e}"))
	
	async def _connect_websocket(self) -> None:
		"""Connect WebSocket for GraphQL subscriptions."""
		try:
			headers = {}
			await self._add_auth_headers(headers)
			
			self.websocket_session = await self.session.ws_connect(
				str(self.config.websocket_url),
				headers=headers,
				protocols=["graphql-ws"]
			)
			
			# Send connection init message
			await self.websocket_session.send_json({
				"type": "connection_init",
				"payload": {}
			})
			
			logger.info(self._log_connector_info("WebSocket connection established"))
			
		except Exception as e:
			logger.error(self._log_connector_info(f"WebSocket connection failed: {e}"))
	
	def _validate_query(self, query: str) -> None:
		"""Validate GraphQL query complexity and depth."""
		# Simple validation - in production, use proper GraphQL query analysis
		query_depth = query.count("{")
		if query_depth > self.config.query_depth_limit:
			raise ValueError(f"Query depth {query_depth} exceeds limit {self.config.query_depth_limit}")
		
		# Estimate complexity (simplified)
		field_count = len([word for word in query.split() if not word.startswith(("query", "mutation", "subscription", "{", "}"))])
		if field_count > self.config.query_complexity_limit:
			raise ValueError(f"Query complexity {field_count} exceeds limit {self.config.query_complexity_limit}")
	
	async def _add_auth_headers(self, headers: Dict[str, str]) -> None:
		"""Add authentication headers for GraphQL requests."""
		if self.config.authentication_type == "basic" and self.config.username and self.config.password:
			import base64
			credentials = base64.b64encode(f"{self.config.username}:{self.config.password}".encode()).decode()
			headers["Authorization"] = f"Basic {credentials}"
		
		elif self.config.authentication_type == "bearer" and self.config.bearer_token:
			headers["Authorization"] = f"Bearer {self.config.bearer_token}"
		
		elif self.config.authentication_type == "api_key" and self.config.api_key:
			headers[self.config.api_key_header] = self.config.api_key
	
	async def _health_check(self) -> bool:
		"""Perform health check against GraphQL endpoint."""
		try:
			if not self.session:
				return False
			
			# Use introspection query for health check
			introspection_query = """
			query IntrospectionQuery {
				__schema {
					queryType {
						name
					}
				}
			}
			"""
			
			# Perform health check request with timeout
			headers = {
				"Content-Type": "application/json",
				"Accept": "application/json"
			}
			await self._add_auth_headers(headers)
			
			payload = {
				"query": introspection_query
			}
			
			async with self.session.post(
				str(self.config.endpoint_url),
				headers=headers,
				json=payload,
				timeout=aiohttp.ClientTimeout(total=5.0)  # 5 second timeout for health check
			) as response:
				if response.status == 200:
					result = await response.json()
					# Check if response has valid GraphQL structure
					return "data" in result and "__schema" in result.get("data", {})
				return False
					
		except asyncio.TimeoutError:
			logger.warning(f"Health check timeout for GraphQL endpoint: {self.config.endpoint_url}")
			return False
		except Exception as e:
			logger.warning(f"Health check failed for GraphQL endpoint: {self.config.endpoint_url}, error: {e}")
			return False

# Export connector classes
__all__ = ["RESTConnector", "RESTConfiguration", "GraphQLConnector", "GraphQLConfiguration"]