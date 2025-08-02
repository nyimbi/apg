"""
© 2025 Datacraft
REST API Connector for Workflow Orchestration

Concrete implementation of REST API connector with real HTTP client functionality.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin, urlparse

import aiohttp
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

from .base_connector import BaseConnector, ConnectorConfig, ConnectorStatus
from apg.common.logging import get_logger
from apg.common.exceptions import APGException

logger = get_logger(__name__)

class RestApiConfig(ConnectorConfig):
	"""Configuration for REST API connector"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	base_url: str
	authentication: Dict[str, Any] = Field(default_factory=dict)
	default_headers: Dict[str, str] = Field(default_factory=dict)
	timeout: int = 30
	max_retries: int = 3
	retry_delay: float = 1.0
	verify_ssl: bool = True
	rate_limit: Optional[Dict[str, Any]] = None  # requests per second/minute
	proxy: Optional[str] = None

class RestApiConnector(BaseConnector):
	"""REST API connector implementation"""
	
	def __init__(self, config: RestApiConfig):
		super().__init__(config)
		self.config = config
		self.session: Optional[aiohttp.ClientSession] = None
		self.rate_limiter: Optional[asyncio.Semaphore] = None
		self.last_request_time = 0.0
		
	async def _connect(self) -> None:
		"""Connect to REST API service"""
		try:
			# Setup authentication
			auth = None
			headers = self.config.default_headers.copy()
			
			auth_config = self.config.authentication
			if auth_config.get('type') == 'basic':
				auth = aiohttp.BasicAuth(
					auth_config.get('username', ''),
					auth_config.get('password', '')
				)
			elif auth_config.get('type') == 'bearer':
				headers['Authorization'] = f"Bearer {auth_config.get('token', '')}"
			elif auth_config.get('type') == 'api_key':
				key_header = auth_config.get('header', 'X-API-Key')
				headers[key_header] = auth_config.get('key', '')
			elif auth_config.get('type') == 'oauth2':
				# OAuth2 token refresh logic
				token = await self._refresh_oauth2_token(auth_config)
				headers['Authorization'] = f"Bearer {token}"
			
			# Setup SSL verification
			connector = aiohttp.TCPConnector(
				verify_ssl=self.config.verify_ssl,
				limit=100,  # Connection pool limit
				limit_per_host=30
			)
			
			# Setup timeout
			timeout = aiohttp.ClientTimeout(
				total=self.config.timeout,
				connect=10,
				sock_read=self.config.timeout
			)
			
			# Create session
			self.session = aiohttp.ClientSession(
				base_url=self.config.base_url,
				headers=headers,
				auth=auth,
				connector=connector,
				timeout=timeout,
				trust_env=True  # Use proxy settings from environment
			)
			
			# Setup rate limiting
			if self.config.rate_limit:
				limit = self.config.rate_limit.get('requests_per_second', 10)
				self.rate_limiter = asyncio.Semaphore(limit)
			
			# Test connection
			await self._test_connection()
			
			self.status = ConnectorStatus.CONNECTED
			logger.info(f"REST API connector connected to {self.config.base_url}")
			
		except Exception as e:
			self.status = ConnectorStatus.ERROR
			self.last_error = str(e)
			logger.error(f"Failed to connect to REST API: {str(e)}")
			raise APGException(f"REST API connection failed: {str(e)}")
	
	async def _disconnect(self) -> None:
		"""Disconnect from REST API service"""
		try:
			if self.session and not self.session.closed:
				await self.session.close()
			
			self.session = None
			self.rate_limiter = None
			self.status = ConnectorStatus.DISCONNECTED
			
			logger.info("REST API connector disconnected")
			
		except Exception as e:
			logger.error(f"Error during REST API disconnect: {str(e)}")
	
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute REST API operation"""
		try:
			if not self.session:
				await self.connect()
			
			# Apply rate limiting
			if self.rate_limiter:
				await self.rate_limiter.acquire()
			
			# Enforce rate limit timing
			if self.config.rate_limit:
				min_interval = 1.0 / self.config.rate_limit.get('requests_per_second', 10)
				time_since_last = time.time() - self.last_request_time
				if time_since_last < min_interval:
					await asyncio.sleep(min_interval - time_since_last)
			
			self.last_request_time = time.time()
			
			# Execute operation with retries
			return await self._execute_with_retries(operation, parameters)
			
		except Exception as e:
			logger.error(f"REST API operation failed: {str(e)}")
			return {
				'success': False,
				'error': str(e),
				'operation': operation
			}
		finally:
			if self.rate_limiter:
				self.rate_limiter.release()
	
	async def _execute_with_retries(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute operation with retry logic"""
		last_exception = None
		
		for attempt in range(self.config.max_retries + 1):
			try:
				if operation == 'get':
					return await self._execute_get(parameters)
				elif operation == 'post':
					return await self._execute_post(parameters)
				elif operation == 'put':
					return await self._execute_put(parameters)
				elif operation == 'patch':
					return await self._execute_patch(parameters)
				elif operation == 'delete':
					return await self._execute_delete(parameters)
				elif operation == 'request':
					return await self._execute_request(parameters)
				else:
					return {
						'success': False,
						'error': f'Unknown operation: {operation}'
					}
					
			except aiohttp.ClientError as e:
				last_exception = e
				if attempt < self.config.max_retries:
					delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
					logger.warning(f"REST API request failed (attempt {attempt + 1}), retrying in {delay}s: {str(e)}")
					await asyncio.sleep(delay)
				else:
					logger.error(f"REST API request failed after {self.config.max_retries} retries: {str(e)}")
		
		return {
			'success': False,
			'error': str(last_exception),
			'attempts': self.config.max_retries + 1
		}
	
	async def _execute_get(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute GET request"""
		url = parameters.get('url', '')
		params = parameters.get('params', {})
		headers = parameters.get('headers', {})
		
		async with self.session.get(url, params=params, headers=headers) as response:
			response_data = await self._process_response(response)
			
			return {
				'success': response.status < 400,
				'status_code': response.status,
				'headers': dict(response.headers),
				'data': response_data,
				'url': str(response.url),
				'method': 'GET'
			}
	
	async def _execute_post(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute POST request"""
		url = parameters.get('url', '')
		data = parameters.get('data', {})
		json_data = parameters.get('json', None)
		headers = parameters.get('headers', {})
		
		kwargs = {'headers': headers}
		if json_data is not None:
			kwargs['json'] = json_data
		else:
			kwargs['data'] = data
		
		async with self.session.post(url, **kwargs) as response:
			response_data = await self._process_response(response)
			
			return {
				'success': response.status < 400,
				'status_code': response.status,
				'headers': dict(response.headers),
				'data': response_data,
				'url': str(response.url),
				'method': 'POST'
			}
	
	async def _execute_put(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute PUT request"""
		url = parameters.get('url', '')
		data = parameters.get('data', {})
		json_data = parameters.get('json', None)
		headers = parameters.get('headers', {})
		
		kwargs = {'headers': headers}
		if json_data is not None:
			kwargs['json'] = json_data
		else:
			kwargs['data'] = data
		
		async with self.session.put(url, **kwargs) as response:
			response_data = await self._process_response(response)
			
			return {
				'success': response.status < 400,
				'status_code': response.status,
				'headers': dict(response.headers),
				'data': response_data,
				'url': str(response.url),
				'method': 'PUT'
			}
	
	async def _execute_patch(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute PATCH request"""
		url = parameters.get('url', '')
		data = parameters.get('data', {})
		json_data = parameters.get('json', None)
		headers = parameters.get('headers', {})
		
		kwargs = {'headers': headers}
		if json_data is not None:
			kwargs['json'] = json_data
		else:
			kwargs['data'] = data
		
		async with self.session.patch(url, **kwargs) as response:
			response_data = await self._process_response(response)
			
			return {
				'success': response.status < 400,
				'status_code': response.status,
				'headers': dict(response.headers),
				'data': response_data,
				'url': str(response.url),
				'method': 'PATCH'
			}
	
	async def _execute_delete(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute DELETE request"""
		url = parameters.get('url', '')
		headers = parameters.get('headers', {})
		params = parameters.get('params', {})
		
		async with self.session.delete(url, headers=headers, params=params) as response:
			response_data = await self._process_response(response)
			
			return {
				'success': response.status < 400,
				'status_code': response.status,
				'headers': dict(response.headers),
				'data': response_data,
				'url': str(response.url),
				'method': 'DELETE'
			}
	
	async def _execute_request(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute custom HTTP request"""
		method = parameters.get('method', 'GET').upper()
		url = parameters.get('url', '')
		headers = parameters.get('headers', {})
		params = parameters.get('params', {})
		data = parameters.get('data', {})
		json_data = parameters.get('json', None)
		
		kwargs = {
			'headers': headers,
			'params': params
		}
		
		if json_data is not None:
			kwargs['json'] = json_data
		elif data:
			kwargs['data'] = data
		
		async with self.session.request(method, url, **kwargs) as response:
			response_data = await self._process_response(response)
			
			return {
				'success': response.status < 400,
				'status_code': response.status,
				'headers': dict(response.headers),
				'data': response_data,
				'url': str(response.url),
				'method': method
			}
	
	async def _process_response(self, response: aiohttp.ClientResponse) -> Any:
		"""Process HTTP response"""
		try:
			content_type = response.headers.get('content-type', '').lower()
			
			if 'application/json' in content_type:
				return await response.json()
			elif 'text/' in content_type or 'application/xml' in content_type:
				return await response.text()
			else:
				# Binary content
				content = await response.read()
				return {
					'content': content.hex(),
					'content_type': content_type,
					'size': len(content)
				}
		except Exception as e:
			logger.warning(f"Failed to process response: {str(e)}")
			return await response.text()
	
	async def _health_check(self) -> bool:
		"""Perform health check"""
		try:
			if not self.session or self.session.closed:
				return False
			
			# Use custom health check endpoint if configured
			health_endpoint = self.config.configuration.get('health_endpoint', '/health')
			
			async with self.session.get(health_endpoint) as response:
				return response.status < 500
				
		except Exception as e:
			logger.warning(f"REST API health check failed: {str(e)}")
			return False
	
	async def _test_connection(self) -> None:
		"""Test initial connection"""
		try:
			# Try to make a simple request to test connectivity
			test_endpoint = self.config.configuration.get('test_endpoint', '/')
			
			async with self.session.get(test_endpoint) as response:
				if response.status >= 500:
					raise APGException(f"Server error: {response.status}")
					
				logger.info(f"REST API connection test successful: {response.status}")
				
		except aiohttp.ClientConnectorError as e:
			raise APGException(f"Cannot connect to {self.config.base_url}: {str(e)}")
		except Exception as e:
			logger.warning(f"Connection test failed (may be expected): {str(e)}")
	
	async def _refresh_oauth2_token(self, auth_config: Dict[str, Any]) -> str:
		"""Refresh OAuth2 token"""
		try:
			token_url = auth_config.get('token_url', '')
			client_id = auth_config.get('client_id', '')
			client_secret = auth_config.get('client_secret', '')
			refresh_token = auth_config.get('refresh_token', '')
			
			if not all([token_url, client_id, client_secret, refresh_token]):
				raise ValueError("Missing OAuth2 configuration")
			
			data = {
				'grant_type': 'refresh_token',
				'refresh_token': refresh_token,
				'client_id': client_id,
				'client_secret': client_secret
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(token_url, data=data) as response:
					if response.status != 200:
						raise APGException(f"OAuth2 token refresh failed: {response.status}")
					
					token_data = await response.json()
					return token_data.get('access_token', '')
					
		except Exception as e:
			logger.error(f"OAuth2 token refresh failed: {str(e)}")
			raise APGException(f"OAuth2 token refresh failed: {str(e)}")
	
	def get_connection_info(self) -> Dict[str, Any]:
		"""Get connection information"""
		return {
			'connector_type': 'rest_api',
			'base_url': self.config.base_url,
			'status': self.status,
			'session_closed': self.session.closed if self.session else True,
			'last_request_time': self.last_request_time,
			'authentication_type': self.config.authentication.get('type', 'none'),
			'rate_limit': self.config.rate_limit,
			'ssl_verify': self.config.verify_ssl
		}