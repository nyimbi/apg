#!/usr/bin/env python3
"""
APG Key Management SDK
Python SDK and client libraries for developers

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import aiohttp
import json
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import ssl
from uuid_extensions import uuid7str

from .models import KeyAlgorithm, KeyUsage, KeyState, SecurityLevel


class SDKError(Exception):
	"""Base exception for SDK errors"""
	pass


class AuthenticationError(SDKError):
	"""Authentication related errors"""
	pass


class KeyNotFoundError(SDKError):
	"""Key not found errors"""
	pass


class ValidationError(SDKError):
	"""Request validation errors"""
	pass


class RateLimitError(SDKError):
	"""Rate limit exceeded errors"""
	pass


@dataclass
class SDKConfig:
	"""SDK configuration"""
	base_url: str = "https://api.keym.datacraft.co.ke"
	api_version: str = "v1"
	timeout: int = 30
	max_retries: int = 3
	retry_backoff: float = 1.0
	verify_ssl: bool = True
	user_agent: str = "APG-KeyM-SDK/1.0.0"


@dataclass
class APIResponse:
	"""Standardized API response"""
	success: bool
	data: Any = None
	error: Dict[str, Any] = None
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class KeyInfo:
	"""Key information returned by SDK"""
	id: str
	name: str
	algorithm: str
	state: str
	created_at: datetime
	last_used: Optional[datetime] = None
	usage_count: int = 0
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyUsageStats:
	"""Key usage statistics"""
	key_id: str
	total_operations: int
	encrypt_operations: int
	decrypt_operations: int
	sign_operations: int
	verify_operations: int
	success_rate: float
	average_latency_ms: float
	last_24h_operations: int


class KeyManagementClient:
	"""
	Main SDK client for APG Key Management
	Provides high-level interface for key operations
	"""
	
	def __init__(self, api_key: str, tenant_id: str, config: SDKConfig | None = None):
		self.api_key = api_key
		self.tenant_id = tenant_id
		self.config = config or SDKConfig()
		self.session: Optional[aiohttp.ClientSession] = None
		self._auth_token: Optional[str] = None
		self._token_expires: Optional[datetime] = None
	
	async def __aenter__(self):
		await self._init_session()
		return self
	
	async def __aexit__(self, exc_type, exc_val, exc_tb):
		await self._close_session()
	
	async def _init_session(self) -> None:
		"""Initialize HTTP session"""
		connector = aiohttp.TCPConnector(
			ssl=ssl.create_default_context() if self.config.verify_ssl else False,
			limit=100,
			limit_per_host=30,
			ttl_dns_cache=300
		)
		
		timeout = aiohttp.ClientTimeout(total=self.config.timeout)
		
		headers = {
			'User-Agent': self.config.user_agent,
			'Content-Type': 'application/json',
			'X-Tenant-ID': self.tenant_id
		}
		
		self.session = aiohttp.ClientSession(
			connector=connector,
			timeout=timeout,
			headers=headers
		)
	
	async def _close_session(self) -> None:
		"""Close HTTP session"""
		if self.session and not self.session.closed:
			await self.session.close()
	
	async def _authenticate(self) -> str:
		"""Authenticate and get access token"""
		if self._auth_token and self._token_expires and datetime.utcnow() < self._token_expires:
			return self._auth_token
		
		auth_url = f"{self.config.base_url}/auth/token"
		auth_data = {
			'api_key': self.api_key,
			'tenant_id': self.tenant_id
		}
		
		async with self.session.post(auth_url, json=auth_data) as response:
			if response.status == 200:
				data = await response.json()
				self._auth_token = data.get('access_token')
				expires_in = data.get('expires_in', 3600)
				self._token_expires = datetime.utcnow() + timedelta(seconds=expires_in)
				return self._auth_token
			else:
				raise AuthenticationError(f"Authentication failed: {response.status}")
	
	async def _make_request(self, method: str, endpoint: str, 
						   data: Dict[str, Any] = None, 
						   params: Dict[str, Any] = None) -> APIResponse:
		"""Make authenticated HTTP request with retry logic"""
		
		if not self.session:
			await self._init_session()
		
		url = f"{self.config.base_url}/{self.config.api_version}/{endpoint}"
		headers = {}
		
		# Add authentication
		token = await self._authenticate()
		if token:
			headers['Authorization'] = f'Bearer {token}'
		
		# Retry logic
		last_exception = None
		for attempt in range(self.config.max_retries + 1):
			try:
				async with self.session.request(
					method, url, 
					json=data, 
					params=params, 
					headers=headers
				) as response:
					
					# Handle rate limiting
					if response.status == 429:
						retry_after = int(response.headers.get('Retry-After', 60))
						if attempt < self.config.max_retries:
							await asyncio.sleep(retry_after)
							continue
						else:
							raise RateLimitError("Rate limit exceeded")
					
					# Handle authentication errors
					if response.status == 401:
						self._auth_token = None
						if attempt < self.config.max_retries:
							token = await self._authenticate()
							headers['Authorization'] = f'Bearer {token}'
							continue
						else:
							raise AuthenticationError("Authentication failed")
					
					# Parse response
					try:
						response_data = await response.json()
					except:
						response_data = {'error': {'message': await response.text()}}
					
					if response.status >= 200 and response.status < 300:
						return APIResponse(
							success=True,
							data=response_data.get('data'),
							metadata=response_data.get('metadata', {})
						)
					else:
						return APIResponse(
							success=False,
							error=response_data.get('error', {'message': 'Unknown error'})
						)
			
			except aiohttp.ClientError as e:
				last_exception = e
				if attempt < self.config.max_retries:
					await asyncio.sleep(self.config.retry_backoff * (2 ** attempt))
					continue
		
		raise SDKError(f"Request failed after {self.config.max_retries} retries: {last_exception}")
	
	# Key Management Operations
	
	async def create_key(self, name: str, algorithm: KeyAlgorithm, 
						usage: List[KeyUsage], **kwargs) -> KeyInfo:
		"""Create new cryptographic key"""
		
		key_data = {
			'name': name,
			'algorithm': algorithm.value,
			'usage': [u.value for u in usage],
			'description': kwargs.get('description'),
			'key_size': kwargs.get('key_size'),
			'security_level': kwargs.get('security_level', SecurityLevel.INTERNAL).value,
			'auto_rotate': kwargs.get('auto_rotate', True),
			'rotation_interval_days': kwargs.get('rotation_interval_days', 90)
		}
		
		response = await self._make_request('POST', 'keys', data=key_data)
		
		if response.success:
			key_data = response.data['key']
			return KeyInfo(
				id=key_data['id'],
				name=key_data['name'], 
				algorithm=key_data['algorithm'],
				state=key_data['state'],
				created_at=datetime.fromisoformat(key_data['created_at']),
				metadata=key_data.get('metadata', {})
			)
		else:
			raise ValidationError(response.error.get('message', 'Key creation failed'))
	
	async def get_key(self, key_id: str, include_material: bool = False) -> KeyInfo:
		"""Retrieve key information"""
		
		params = {'include_material': include_material}
		response = await self._make_request('GET', f'keys/{key_id}', params=params)
		
		if response.success:
			key_data = response.data['key']
			return KeyInfo(
				id=key_data['id'],
				name=key_data['name'],
				algorithm=key_data['algorithm'], 
				state=key_data['state'],
				created_at=datetime.fromisoformat(key_data['created_at']),
				last_used=datetime.fromisoformat(key_data['last_used']) if key_data.get('last_used') else None,
				usage_count=key_data.get('usage_count', 0),
				metadata=key_data.get('metadata', {})
			)
		else:
			if response.error.get('code') == 'KEY_NOT_FOUND':
				raise KeyNotFoundError(f"Key {key_id} not found")
			else:
				raise SDKError(response.error.get('message', 'Failed to retrieve key'))
	
	async def list_keys(self, algorithm: Optional[str] = None, 
						state: Optional[str] = None, 
						limit: int = 50, offset: int = 0) -> List[KeyInfo]:
		"""List keys with optional filtering"""
		
		params = {'limit': limit, 'offset': offset}
		if algorithm:
			params['algorithm'] = algorithm
		if state:
			params['state'] = state
		
		response = await self._make_request('GET', 'keys', params=params)
		
		if response.success:
			keys_data = response.data['keys']
			return [
				KeyInfo(
					id=key['id'],
					name=key['name'],
					algorithm=key['algorithm'],
					state=key['state'],
					created_at=datetime.fromisoformat(key['created_at']),
					last_used=datetime.fromisoformat(key['last_used']) if key.get('last_used') else None,
					usage_count=key.get('usage_count', 0),
					metadata=key.get('metadata', {})
				)
				for key in keys_data
			]
		else:
			raise SDKError(response.error.get('message', 'Failed to list keys'))
	
	async def rotate_key(self, key_id: str) -> KeyInfo:
		"""Rotate cryptographic key"""
		
		response = await self._make_request('POST', f'keys/{key_id}/rotate')
		
		if response.success:
			key_data = response.data['key']
			return KeyInfo(
				id=key_data['id'],
				name=key_data['name'],
				algorithm=key_data['algorithm'],
				state=key_data['state'], 
				created_at=datetime.fromisoformat(key_data['created_at']),
				metadata=key_data.get('metadata', {})
			)
		else:
			raise SDKError(response.error.get('message', 'Key rotation failed'))
	
	async def delete_key(self, key_id: str, secure_delete: bool = True) -> bool:
		"""Delete cryptographic key"""
		
		params = {'secure_delete': secure_delete}
		response = await self._make_request('DELETE', f'keys/{key_id}', params=params)
		
		if response.success:
			return response.data['deleted']
		else:
			if response.error.get('code') == 'KEY_NOT_FOUND':
				raise KeyNotFoundError(f"Key {key_id} not found")
			else:
				raise SDKError(response.error.get('message', 'Key deletion failed'))
	
	# Cryptographic Operations
	
	async def encrypt(self, key_id: str, data: bytes, **kwargs) -> bytes:
		"""Encrypt data using specified key"""
		
		# Encode data as base64 for JSON transport
		data_b64 = base64.b64encode(data).decode('utf-8')
		
		encrypt_data = {
			'data': data_b64,
			'parameters': kwargs
		}
		
		response = await self._make_request('POST', f'keys/{key_id}/encrypt', data=encrypt_data)
		
		if response.success:
			encrypted_b64 = response.data['encrypted_data']
			return base64.b64decode(encrypted_b64)
		else:
			raise SDKError(response.error.get('message', 'Encryption failed'))
	
	async def decrypt(self, key_id: str, encrypted_data: bytes, **kwargs) -> bytes:
		"""Decrypt data using specified key"""
		
		# Encode encrypted data as base64 for JSON transport
		encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
		
		decrypt_data = {
			'encrypted_data': encrypted_b64,
			'parameters': kwargs
		}
		
		response = await self._make_request('POST', f'keys/{key_id}/decrypt', data=decrypt_data)
		
		if response.success:
			decrypted_b64 = response.data['decrypted_data']
			return base64.b64decode(decrypted_b64)
		else:
			raise SDKError(response.error.get('message', 'Decryption failed'))
	
	# Analytics and Monitoring
	
	async def get_key_stats(self, key_id: str) -> KeyUsageStats:
		"""Get key usage statistics"""
		
		response = await self._make_request('GET', f'keys/{key_id}/stats')
		
		if response.success:
			stats_data = response.data['statistics']
			return KeyUsageStats(
				key_id=key_id,
				total_operations=stats_data['total_operations'],
				encrypt_operations=stats_data['encrypt_operations'], 
				decrypt_operations=stats_data['decrypt_operations'],
				sign_operations=stats_data['sign_operations'],
				verify_operations=stats_data['verify_operations'],
				success_rate=stats_data['success_rate'],
				average_latency_ms=stats_data['average_latency_ms'],
				last_24h_operations=stats_data['last_24h_operations']
			)
		else:
			raise SDKError(response.error.get('message', 'Failed to retrieve statistics'))
	
	async def get_audit_events(self, **filters) -> List[Dict[str, Any]]:
		"""Get audit events with filtering"""
		
		response = await self._make_request('GET', 'audit', params=filters)
		
		if response.success:
			return response.data['events']
		else:
			raise SDKError(response.error.get('message', 'Failed to retrieve audit events'))
	
	async def get_service_health(self) -> Dict[str, Any]:
		"""Get service health status"""
		
		response = await self._make_request('GET', 'health')
		
		if response.success:
			return response.data
		else:
			raise SDKError(response.error.get('message', 'Failed to retrieve service health'))


class AsyncKeyStreamClient:
	"""
	Streaming client for real-time key operations
	Provides async generators for handling large datasets
	"""
	
	def __init__(self, client: KeyManagementClient):
		self.client = client
	
	async def stream_keys(self, **filters) -> AsyncGenerator[KeyInfo, None]:
		"""Stream keys in batches"""
		
		batch_size = filters.pop('batch_size', 100)
		offset = 0
		
		while True:
			keys = await self.client.list_keys(
				limit=batch_size, 
				offset=offset, 
				**filters
			)
			
			if not keys:
				break
			
			for key in keys:
				yield key
			
			if len(keys) < batch_size:
				break
			
			offset += batch_size
	
	async def stream_audit_events(self, **filters) -> AsyncGenerator[Dict[str, Any], None]:
		"""Stream audit events in batches"""
		
		batch_size = filters.pop('batch_size', 1000)
		offset = 0
		
		while True:
			events = await self.client.get_audit_events(
				limit=batch_size,
				offset=offset,
				**filters
			)
			
			if not events:
				break
			
			for event in events:
				yield event
			
			if len(events) < batch_size:
				break
			
			offset += batch_size


class KeyManagementUtils:
	"""
	Utility functions for key management operations
	"""
	
	@staticmethod
	def generate_key_name(prefix: str, algorithm: KeyAlgorithm, purpose: str = None) -> str:
		"""Generate standardized key name"""
		timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
		parts = [prefix, algorithm.value.lower().replace('-', '_'), timestamp]
		
		if purpose:
			parts.insert(-1, purpose)
		
		return '_'.join(parts)
	
	@staticmethod
	def calculate_key_fingerprint(key_data: bytes) -> str:
		"""Calculate key fingerprint"""
		return hashlib.sha256(key_data).hexdigest()[:16]
	
	@staticmethod
	def validate_key_size(algorithm: KeyAlgorithm, key_size: int) -> bool:
		"""Validate key size for algorithm"""
		valid_sizes = {
			KeyAlgorithm.AES_128: [128],
			KeyAlgorithm.AES_256: [256],
			KeyAlgorithm.RSA_2048: [2048],
			KeyAlgorithm.RSA_4096: [4096],
			KeyAlgorithm.ECDSA_P256: [256],
			KeyAlgorithm.ECDSA_P384: [384]
		}
		
		return key_size in valid_sizes.get(algorithm, [])
	
	@staticmethod
	def recommend_algorithm(security_level: SecurityLevel, 
							performance_priority: bool = False) -> KeyAlgorithm:
		"""Recommend algorithm based on requirements"""
		
		if performance_priority:
			# Prioritize performance
			if security_level in [SecurityLevel.PUBLIC, SecurityLevel.INTERNAL]:
				return KeyAlgorithm.AES_256
			else:
				return KeyAlgorithm.ECDSA_P256
		else:
			# Prioritize security
			if security_level in [SecurityLevel.RESTRICTED, SecurityLevel.TOP_SECRET]:
				return KeyAlgorithm.RSA_4096
			elif security_level == SecurityLevel.CONFIDENTIAL:
				return KeyAlgorithm.ECDSA_P384
			else:
				return KeyAlgorithm.AES_256


class BatchKeyOperations:
	"""
	Batch operations for efficient bulk key management
	"""
	
	def __init__(self, client: KeyManagementClient):
		self.client = client
	
	async def create_keys_batch(self, key_specs: List[Dict[str, Any]]) -> List[Union[KeyInfo, Exception]]:
		"""Create multiple keys in batch"""
		
		# Prepare batch request
		batch_request = {'keys': key_specs}
		
		response = await self.client._make_request('POST', 'keys/batch', data=batch_request)
		
		if response.success:
			results = response.data['results']
			batch_results = []
			
			for result in results:
				if result['success']:
					key_data = result['key']
					batch_results.append(KeyInfo(
						id=key_data['id'],
						name=key_data['name'],
						algorithm=key_data['algorithm'],
						state=key_data['state'],
						created_at=datetime.fromisoformat(key_data['created_at']),
						metadata=key_data.get('metadata', {})
					))
				else:
					batch_results.append(SDKError(result['error']))
			
			return batch_results
		else:
			raise SDKError(response.error.get('message', 'Batch key creation failed'))
	
	async def rotate_keys_batch(self, key_ids: List[str]) -> Dict[str, Union[bool, Exception]]:
		"""Rotate multiple keys in batch"""
		
		results = {}
		
		# Use semaphore to limit concurrent operations
		semaphore = asyncio.Semaphore(10)
		
		async def rotate_single_key(key_id: str):
			async with semaphore:
				try:
					await self.client.rotate_key(key_id)
					results[key_id] = True
				except Exception as e:
					results[key_id] = e
		
		# Execute rotations concurrently
		tasks = [rotate_single_key(key_id) for key_id in key_ids]
		await asyncio.gather(*tasks, return_exceptions=True)
		
		return results


# Convenience functions for common patterns

async def create_client(api_key: str, tenant_id: str, config: SDKConfig = None) -> KeyManagementClient:
	"""Convenience function to create and initialize client"""
	client = KeyManagementClient(api_key, tenant_id, config)
	await client._init_session()
	return client


def sync_create_key(api_key: str, tenant_id: str, name: str, 
				   algorithm: KeyAlgorithm, usage: List[KeyUsage], **kwargs) -> KeyInfo:
	"""Synchronous wrapper for key creation"""
	
	async def _create():
		async with KeyManagementClient(api_key, tenant_id) as client:
			return await client.create_key(name, algorithm, usage, **kwargs)
	
	return asyncio.run(_create())


def sync_encrypt(api_key: str, tenant_id: str, key_id: str, data: bytes) -> bytes:
	"""Synchronous wrapper for encryption"""
	
	async def _encrypt():
		async with KeyManagementClient(api_key, tenant_id) as client:
			return await client.encrypt(key_id, data)
	
	return asyncio.run(_encrypt())


# Export SDK components
__all__ = [
	'KeyManagementClient', 'AsyncKeyStreamClient', 'BatchKeyOperations',
	'KeyManagementUtils', 'SDKConfig', 'KeyInfo', 'KeyUsageStats',
	'SDKError', 'AuthenticationError', 'KeyNotFoundError', 'ValidationError',
	'create_client', 'sync_create_key', 'sync_encrypt'
]