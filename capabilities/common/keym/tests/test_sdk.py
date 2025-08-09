#!/usr/bin/env python3
"""
APG Key Management - SDK Tests
Comprehensive test suite for Python SDK and client libraries

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from ..sdk import (
	KeyManagementClient, AsyncKeyStreamClient, BatchKeyOperations,
	KeyManagementUtils, SDKConfig, KeyInfo, KeyUsageStats,
	SDKError, AuthenticationError, KeyNotFoundError, ValidationError, RateLimitError,
	create_client, sync_create_key, sync_encrypt
)
from ..models import KeyAlgorithm, KeyUsage, SecurityLevel


@pytest.fixture
def sdk_config():
	"""Fixture for SDK configuration"""
	return SDKConfig(
		base_url="https://test-api.keym.datacraft.co.ke",
		api_version="v1",
		timeout=30,
		max_retries=2,
		verify_ssl=False  # For testing
	)


@pytest.fixture
async def mock_client_session():
	"""Fixture for mocked aiohttp client session"""
	session = AsyncMock(spec=aiohttp.ClientSession)
	session.closed = False
	return session


@pytest.fixture
def sample_key_info():
	"""Fixture for sample key info"""
	return KeyInfo(
		id="key_123",
		name="Test Key",
		algorithm="AES-256",
		state="active",
		created_at=datetime.utcnow(),
		usage_count=100,
		metadata={"environment": "test"}
	)


class TestKeyManagementClient:
	"""Test KeyManagementClient class"""
	
	@pytest.mark.asyncio
	async def test_client_initialization(self, sdk_config):
		"""Test client initialization"""
		client = KeyManagementClient("test_api_key", "test_tenant", sdk_config)
		
		assert client.api_key == "test_api_key"
		assert client.tenant_id == "test_tenant"
		assert client.config == sdk_config
		assert client.session is None
		assert client._auth_token is None
	
	@pytest.mark.asyncio
	async def test_context_manager(self, sdk_config):
		"""Test client as context manager"""
		with patch.object(KeyManagementClient, '_init_session') as mock_init:
			with patch.object(KeyManagementClient, '_close_session') as mock_close:
				async with KeyManagementClient("api_key", "tenant", sdk_config) as client:
					assert isinstance(client, KeyManagementClient)
				
				mock_init.assert_called_once()
				mock_close.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_session_initialization(self, sdk_config):
		"""Test HTTP session initialization"""
		client = KeyManagementClient("api_key", "tenant", sdk_config)
		
		with patch('aiohttp.ClientSession') as mock_session_class:
			mock_session = AsyncMock()
			mock_session_class.return_value = mock_session
			
			await client._init_session()
			
			assert client.session == mock_session
			mock_session_class.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_authentication_success(self, sdk_config, mock_client_session):
		"""Test successful authentication"""
		client = KeyManagementClient("api_key", "tenant", sdk_config)
		client.session = mock_client_session
		
		# Mock authentication response
		mock_response = AsyncMock()
		mock_response.status = 200
		mock_response.json.return_value = {
			'access_token': 'test_token_123',
			'expires_in': 3600
		}
		
		mock_client_session.post.return_value.__aenter__.return_value = mock_response
		
		token = await client._authenticate()
		
		assert token == 'test_token_123'
		assert client._auth_token == 'test_token_123'
		assert client._token_expires is not None
	
	@pytest.mark.asyncio
	async def test_authentication_failure(self, sdk_config, mock_client_session):
		"""Test authentication failure"""
		client = KeyManagementClient("api_key", "tenant", sdk_config)
		client.session = mock_client_session
		
		# Mock failed authentication response
		mock_response = AsyncMock()
		mock_response.status = 401
		
		mock_client_session.post.return_value.__aenter__.return_value = mock_response
		
		with pytest.raises(AuthenticationError):
			await client._authenticate()
	
	@pytest.mark.asyncio
	async def test_create_key_success(self, sdk_config, mock_client_session):
		"""Test successful key creation"""
		client = KeyManagementClient("api_key", "tenant", sdk_config)
		client.session = mock_client_session
		client._auth_token = "test_token"
		
		# Mock successful response
		mock_response = AsyncMock()
		mock_response.status = 201
		mock_response.json.return_value = {
			'data': {
				'key': {
					'id': 'key_123',
					'name': 'Test Key',
					'algorithm': 'AES-256',
					'state': 'active',
					'created_at': datetime.utcnow().isoformat(),
					'metadata': {}
				}
			}
		}
		
		mock_client_session.request.return_value.__aenter__.return_value = mock_response
		
		key_info = await client.create_key(
			"Test Key",
			KeyAlgorithm.AES_256,
			[KeyUsage.ENCRYPT, KeyUsage.DECRYPT]
		)
		
		assert isinstance(key_info, KeyInfo)
		assert key_info.id == "key_123"
		assert key_info.name == "Test Key"
		assert key_info.algorithm == "AES-256"
	
	@pytest.mark.asyncio
	async def test_create_key_validation_error(self, sdk_config, mock_client_session):
		"""Test key creation with validation error"""
		client = KeyManagementClient("api_key", "tenant", sdk_config)
		client.session = mock_client_session
		client._auth_token = "test_token"
		
		# Mock validation error response
		mock_response = AsyncMock()
		mock_response.status = 400
		mock_response.json.return_value = {
			'error': {'message': 'Invalid key parameters'}
		}
		
		mock_client_session.request.return_value.__aenter__.return_value = mock_response
		
		with pytest.raises(ValidationError):
			await client.create_key(
				"",  # Invalid empty name
				KeyAlgorithm.AES_256,
				[KeyUsage.ENCRYPT]
			)
	
	@pytest.mark.asyncio
	async def test_get_key_success(self, sdk_config, mock_client_session, sample_key_info):
		"""Test successful key retrieval"""
		client = KeyManagementClient("api_key", "tenant", sdk_config)
		client.session = mock_client_session
		client._auth_token = "test_token"
		
		# Mock successful response
		mock_response = AsyncMock()
		mock_response.status = 200
		mock_response.json.return_value = {
			'data': {
				'key': {
					'id': sample_key_info.id,
					'name': sample_key_info.name,
					'algorithm': sample_key_info.algorithm,
					'state': sample_key_info.state,
					'created_at': sample_key_info.created_at.isoformat(),
					'usage_count': sample_key_info.usage_count,
					'metadata': sample_key_info.metadata
				}
			}
		}
		
		mock_client_session.request.return_value.__aenter__.return_value = mock_response
		
		key_info = await client.get_key("key_123")
		
		assert isinstance(key_info, KeyInfo)
		assert key_info.id == sample_key_info.id
		assert key_info.name == sample_key_info.name
	
	@pytest.mark.asyncio
	async def test_get_key_not_found(self, sdk_config, mock_client_session):
		"""Test key not found error"""
		client = KeyManagementClient("api_key", "tenant", sdk_config)
		client.session = mock_client_session
		client._auth_token = "test_token"
		
		# Mock not found response
		mock_response = AsyncMock()
		mock_response.status = 404
		mock_response.json.return_value = {
			'error': {'code': 'KEY_NOT_FOUND', 'message': 'Key not found'}
		}
		
		mock_client_session.request.return_value.__aenter__.return_value = mock_response
		
		with pytest.raises(KeyNotFoundError):
			await client.get_key("nonexistent_key")
	
	@pytest.mark.asyncio
	async def test_list_keys_success(self, sdk_config, mock_client_session):
		"""Test successful key listing"""
		client = KeyManagementClient("api_key", "tenant", sdk_config)
		client.session = mock_client_session
		client._auth_token = "test_token"
		
		# Mock successful response
		keys_data = [
			{
				'id': f'key_{i}',
				'name': f'Test Key {i}',
				'algorithm': 'AES-256',
				'state': 'active',
				'created_at': datetime.utcnow().isoformat(),
				'usage_count': i * 10,
				'metadata': {}
			}
			for i in range(3)
		]
		
		mock_response = AsyncMock()
		mock_response.status = 200
		mock_response.json.return_value = {
			'data': {'keys': keys_data}
		}
		
		mock_client_session.request.return_value.__aenter__.return_value = mock_response
		
		keys = await client.list_keys(limit=10)
		
		assert isinstance(keys, list)
		assert len(keys) == 3
		for i, key in enumerate(keys):
			assert isinstance(key, KeyInfo)
			assert key.id == f'key_{i}'
			assert key.name == f'Test Key {i}'
	
	@pytest.mark.asyncio
	async def test_encrypt_decrypt_success(self, sdk_config, mock_client_session):
		"""Test successful encryption and decryption"""
		client = KeyManagementClient("api_key", "tenant", sdk_config)
		client.session = mock_client_session
		client._auth_token = "test_token"
		
		test_data = b"Hello, World!"
		
		# Mock encryption response
		encrypt_response = AsyncMock()
		encrypt_response.status = 200
		encrypt_response.json.return_value = {
			'data': {'encrypted_data': 'ZW5jcnlwdGVkX2RhdGE='}  # base64 encoded
		}
		
		# Mock decryption response
		decrypt_response = AsyncMock()
		decrypt_response.status = 200
		decrypt_response.json.return_value = {
			'data': {'decrypted_data': 'SGVsbG8sIFdvcmxkIQ=='}  # "Hello, World!" in base64
		}
		
		mock_client_session.request.return_value.__aenter__.side_effect = [
			encrypt_response, decrypt_response
		]
		
		# Test encryption
		encrypted_data = await client.encrypt("key_123", test_data)
		assert isinstance(encrypted_data, bytes)
		
		# Test decryption
		decrypted_data = await client.decrypt("key_123", encrypted_data)
		assert isinstance(decrypted_data, bytes)
	
	@pytest.mark.asyncio
	async def test_rate_limiting_retry(self, sdk_config, mock_client_session):
		"""Test rate limiting with retry logic"""
		client = KeyManagementClient("api_key", "tenant", sdk_config)
		client.session = mock_client_session
		client._auth_token = "test_token"
		
		# Mock rate limit response followed by success
		rate_limit_response = AsyncMock()
		rate_limit_response.status = 429
		rate_limit_response.headers = {'Retry-After': '1'}
		
		success_response = AsyncMock()
		success_response.status = 200
		success_response.json.return_value = {
			'data': {'health': 'ok'}
		}
		
		mock_client_session.request.return_value.__aenter__.side_effect = [
			rate_limit_response, success_response
		]
		
		with patch('asyncio.sleep') as mock_sleep:
			result = await client.get_service_health()
			mock_sleep.assert_called_once_with(1)  # Retry-After value
			assert result['health'] == 'ok'
	
	@pytest.mark.asyncio
	async def test_rate_limit_exceeded(self, sdk_config, mock_client_session):
		"""Test rate limit exceeded after max retries"""
		client = KeyManagementClient("api_key", "tenant", sdk_config)
		client.config.max_retries = 1  # Low retry count for testing
		client.session = mock_client_session
		client._auth_token = "test_token"
		
		# Mock consistent rate limit responses
		rate_limit_response = AsyncMock()
		rate_limit_response.status = 429
		rate_limit_response.headers = {'Retry-After': '60'}
		
		mock_client_session.request.return_value.__aenter__.return_value = rate_limit_response
		
		with pytest.raises(RateLimitError):
			await client.get_service_health()


class TestAsyncKeyStreamClient:
	"""Test AsyncKeyStreamClient class"""
	
	@pytest.mark.asyncio
	async def test_stream_keys(self, sdk_config):
		"""Test key streaming functionality"""
		# Mock the main client
		mock_main_client = AsyncMock(spec=KeyManagementClient)
		
		# Mock paginated responses
		batch1 = [KeyInfo(id=f"key_{i}", name=f"Key {i}", algorithm="AES-256", 
			state="active", created_at=datetime.utcnow()) for i in range(3)]
		batch2 = [KeyInfo(id=f"key_{i}", name=f"Key {i}", algorithm="AES-256", 
			state="active", created_at=datetime.utcnow()) for i in range(3, 5)]
		empty_batch = []
		
		mock_main_client.list_keys.side_effect = [batch1, batch2, empty_batch]
		
		stream_client = AsyncKeyStreamClient(mock_main_client)
		
		# Collect streamed keys
		streamed_keys = []
		async for key in stream_client.stream_keys(batch_size=3):
			streamed_keys.append(key)
		
		assert len(streamed_keys) == 5
		assert all(isinstance(key, KeyInfo) for key in streamed_keys)
		
		# Verify correct pagination calls
		assert mock_main_client.list_keys.call_count == 3
	
	@pytest.mark.asyncio
	async def test_stream_audit_events(self, sdk_config):
		"""Test audit event streaming"""
		mock_main_client = AsyncMock(spec=KeyManagementClient)
		
		# Mock paginated audit events
		events1 = [{'event_id': f'event_{i}', 'type': 'key_access'} for i in range(2)]
		events2 = [{'event_id': f'event_{i}', 'type': 'key_creation'} for i in range(2, 3)]
		empty_events = []
		
		mock_main_client.get_audit_events.side_effect = [events1, events2, empty_events]
		
		stream_client = AsyncKeyStreamClient(mock_main_client)
		
		# Collect streamed events
		streamed_events = []
		async for event in stream_client.stream_audit_events(batch_size=2):
			streamed_events.append(event)
		
		assert len(streamed_events) == 3
		assert all('event_id' in event for event in streamed_events)


class TestBatchKeyOperations:
	"""Test BatchKeyOperations class"""
	
	@pytest.mark.asyncio
	async def test_create_keys_batch(self, sdk_config, mock_client_session):
		"""Test batch key creation"""
		mock_main_client = AsyncMock(spec=KeyManagementClient)
		mock_main_client._make_request = AsyncMock()
		
		# Mock batch creation response
		batch_response = Mock()
		batch_response.success = True
		batch_response.data = {
			'results': [
				{
					'success': True,
					'key': {
						'id': 'key_1',
						'name': 'Batch Key 1',
						'algorithm': 'AES-256',
						'state': 'active',
						'created_at': datetime.utcnow().isoformat(),
						'metadata': {}
					}
				},
				{
					'success': False,
					'error': 'Invalid key parameters'
				}
			]
		}
		
		mock_main_client._make_request.return_value = batch_response
		
		batch_ops = BatchKeyOperations(mock_main_client)
		
		key_specs = [
			{'name': 'Batch Key 1', 'algorithm': 'AES-256'},
			{'name': '', 'algorithm': 'Invalid'}  # Will fail
		]
		
		results = await batch_ops.create_keys_batch(key_specs)
		
		assert len(results) == 2
		assert isinstance(results[0], KeyInfo)
		assert isinstance(results[1], SDKError)
	
	@pytest.mark.asyncio
	async def test_rotate_keys_batch(self, sdk_config):
		"""Test batch key rotation"""
		mock_main_client = AsyncMock(spec=KeyManagementClient)
		
		# Mock individual rotation calls
		async def mock_rotate_key(key_id):
			if key_id == "key_fail":
				raise SDKError("Rotation failed")
			return True
		
		mock_main_client.rotate_key = mock_rotate_key
		
		batch_ops = BatchKeyOperations(mock_main_client)
		
		key_ids = ["key_1", "key_2", "key_fail", "key_3"]
		results = await batch_ops.rotate_keys_batch(key_ids)
		
		assert len(results) == 4
		assert results["key_1"] is True
		assert results["key_2"] is True
		assert isinstance(results["key_fail"], SDKError)
		assert results["key_3"] is True


class TestKeyManagementUtils:
	"""Test KeyManagementUtils class"""
	
	def test_generate_key_name(self):
		"""Test key name generation"""
		name = KeyManagementUtils.generate_key_name(
			"test", 
			KeyAlgorithm.AES_256, 
			"encryption"
		)
		
		assert name.startswith("test_")
		assert "aes_256" in name
		assert "encryption" in name
		assert len(name.split("_")) >= 4  # prefix, algorithm, purpose, timestamp
	
	def test_calculate_key_fingerprint(self):
		"""Test key fingerprint calculation"""
		key_data = b"test_key_material_for_fingerprint"
		
		fingerprint = KeyManagementUtils.calculate_key_fingerprint(key_data)
		
		assert isinstance(fingerprint, str)
		assert len(fingerprint) == 16  # Truncated SHA256
		
		# Same input should produce same fingerprint
		fingerprint2 = KeyManagementUtils.calculate_key_fingerprint(key_data)
		assert fingerprint == fingerprint2
	
	def test_validate_key_size(self):
		"""Test key size validation"""
		# Valid combinations
		assert KeyManagementUtils.validate_key_size(KeyAlgorithm.AES_128, 128) is True
		assert KeyManagementUtils.validate_key_size(KeyAlgorithm.AES_256, 256) is True
		assert KeyManagementUtils.validate_key_size(KeyAlgorithm.RSA_2048, 2048) is True
		
		# Invalid combinations
		assert KeyManagementUtils.validate_key_size(KeyAlgorithm.AES_128, 256) is False
		assert KeyManagementUtils.validate_key_size(KeyAlgorithm.AES_256, 128) is False
		assert KeyManagementUtils.validate_key_size(KeyAlgorithm.RSA_2048, 1024) is False
	
	def test_recommend_algorithm(self):
		"""Test algorithm recommendation"""
		# Performance priority
		perf_rec = KeyManagementUtils.recommend_algorithm(
			SecurityLevel.INTERNAL, 
			performance_priority=True
		)
		assert perf_rec in [KeyAlgorithm.AES_256, KeyAlgorithm.ECDSA_P256]
		
		# Security priority
		sec_rec = KeyManagementUtils.recommend_algorithm(
			SecurityLevel.TOP_SECRET,
			performance_priority=False
		)
		assert sec_rec == KeyAlgorithm.RSA_4096


class TestConvenienceFunctions:
	"""Test convenience functions"""
	
	@pytest.mark.asyncio
	async def test_create_client(self, sdk_config):
		"""Test client creation convenience function"""
		with patch.object(KeyManagementClient, '_init_session') as mock_init:
			client = await create_client("api_key", "tenant", sdk_config)
			
			assert isinstance(client, KeyManagementClient)
			mock_init.assert_called_once()
	
	def test_sync_create_key(self):
		"""Test synchronous key creation wrapper"""
		with patch('asyncio.run') as mock_run:
			mock_run.return_value = KeyInfo(
				id="key_123",
				name="Sync Key",
				algorithm="AES-256",
				state="active",
				created_at=datetime.utcnow()
			)
			
			result = sync_create_key(
				"api_key", "tenant", "Sync Key",
				KeyAlgorithm.AES_256, [KeyUsage.ENCRYPT]
			)
			
			assert isinstance(result, KeyInfo)
			assert result.name == "Sync Key"
			mock_run.assert_called_once()
	
	def test_sync_encrypt(self):
		"""Test synchronous encryption wrapper"""
		test_data = b"test data"
		
		with patch('asyncio.run') as mock_run:
			mock_run.return_value = b"encrypted_data"
			
			result = sync_encrypt("api_key", "tenant", "key_123", test_data)
			
			assert result == b"encrypted_data"
			mock_run.assert_called_once()


class TestErrorHandling:
	"""Test error handling scenarios"""
	
	@pytest.mark.asyncio
	async def test_network_error_retry(self, sdk_config, mock_client_session):
		"""Test network error with retry logic"""
		client = KeyManagementClient("api_key", "tenant", sdk_config)
		client.session = mock_client_session
		client._auth_token = "test_token"
		
		# Mock network errors followed by success
		network_error = aiohttp.ClientConnectionError("Connection failed")
		success_response = AsyncMock()
		success_response.status = 200
		success_response.json.return_value = {'data': {'health': 'ok'}}
		
		mock_client_session.request.return_value.__aenter__.side_effect = [
			network_error, success_response
		]
		
		with patch('asyncio.sleep') as mock_sleep:
			result = await client.get_service_health()
			assert result['health'] == 'ok'
			mock_sleep.assert_called_once()  # Retry backoff
	
	@pytest.mark.asyncio
	async def test_max_retries_exceeded(self, sdk_config, mock_client_session):
		"""Test max retries exceeded"""
		client = KeyManagementClient("api_key", "tenant", sdk_config)
		client.config.max_retries = 1  # Low for testing
		client.session = mock_client_session
		client._auth_token = "test_token"
		
		# Mock consistent network errors
		network_error = aiohttp.ClientConnectionError("Connection failed")
		mock_client_session.request.return_value.__aenter__.side_effect = network_error
		
		with pytest.raises(SDKError, match="Request failed after .* retries"):
			await client.get_service_health()
	
	def test_invalid_sdk_config(self):
		"""Test invalid SDK configuration"""
		# Test with invalid timeout
		with pytest.raises(ValueError):
			SDKConfig(timeout=-1)
		
		# Test with invalid retry count
		with pytest.raises(ValueError):
			SDKConfig(max_retries=-1)


class TestSDKModels:
	"""Test SDK data models"""
	
	def test_key_info_creation(self):
		"""Test KeyInfo model creation"""
		key_info = KeyInfo(
			id="key_123",
			name="Test Key",
			algorithm="AES-256",
			state="active",
			created_at=datetime.utcnow(),
			last_used=datetime.utcnow(),
			usage_count=500,
			metadata={"environment": "production"}
		)
		
		assert key_info.id == "key_123"
		assert key_info.name == "Test Key"
		assert key_info.algorithm == "AES-256"
		assert key_info.usage_count == 500
		assert key_info.metadata["environment"] == "production"
	
	def test_key_usage_stats_creation(self):
		"""Test KeyUsageStats model creation"""
		stats = KeyUsageStats(
			key_id="key_123",
			total_operations=1000,
			encrypt_operations=600,
			decrypt_operations=400,
			sign_operations=0,
			verify_operations=0,
			success_rate=0.998,
			average_latency_ms=25.5,
			last_24h_operations=150
		)
		
		assert stats.key_id == "key_123"
		assert stats.total_operations == 1000
		assert stats.encrypt_operations == 600
		assert stats.success_rate == 0.998
		assert stats.average_latency_ms == 25.5


if __name__ == "__main__":
	pytest.main([__file__])