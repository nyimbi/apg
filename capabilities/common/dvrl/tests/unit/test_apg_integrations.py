#!/usr/bin/env python3
"""
Unit tests for APG platform integrations
Tests the production HTTP client implementations for all APG services
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, timedelta

import aiohttp
from aioresponses import aioresponses

from dvrl.apg_integrations import (
	APGMetadataService,
	APGCacheService,
	APGSecurityService,
	APGMDMService,
	APGPerformanceOptimizer,
	APGServiceManager,
	APGServiceError,
	APGBaseService
)


class TestAPGBaseService:
	"""Test the base APG service functionality"""
	
	@pytest.fixture
	async def base_service(self):
		"""Create a base service instance for testing"""
		service = APGBaseService(
			tenant_id="test-tenant",
			user_id="test-user",
			base_url="http://localhost:8080/api/v1/test",
			service_name="test"
		)
		yield service
		await service.close()
	
	async def test_session_creation(self, base_service):
		"""Test HTTP session creation and configuration"""
		session = await base_service._get_session()
		
		assert isinstance(session, aiohttp.ClientSession)
		assert not session.closed
		assert session._default_headers['Content-Type'] == 'application/json'
		assert session._default_headers['X-Tenant-ID'] == 'test-tenant'
		assert session._default_headers['X-User-ID'] == 'test-user'
		assert 'DVRL/1.0.0 (test)' in session._default_headers['User-Agent']
	
	@patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token-123'})
	async def test_auth_token_from_env(self, base_service):
		"""Test authentication token retrieval from environment"""
		token = await base_service._get_auth_token()
		
		assert token == 'test-token-123'
		assert base_service._auth_token == 'test-token-123'
		assert base_service._token_expires is not None
	
	async def test_auth_token_caching(self, base_service):
		"""Test authentication token caching behavior"""
		with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'cached-token'}):
			# First call
			token1 = await base_service._get_auth_token()
			
			# Second call should return cached token
			token2 = await base_service._get_auth_token()
			
			assert token1 == token2 == 'cached-token'
	
	async def test_make_request_success(self, base_service):
		"""Test successful HTTP request"""
		with aioresponses() as mock_resp:
			mock_resp.get(
				'http://localhost:8080/api/v1/test/endpoint',
				payload={'status': 'success', 'data': {'id': 123}}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				response = await base_service._make_request('GET', '/endpoint')
			
			assert response['status'] == 'success'
			assert response['data']['id'] == 123
	
	async def test_make_request_with_data(self, base_service):
		"""Test HTTP request with JSON payload"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/test/endpoint',
				payload={'created_id': 456}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				response = await base_service._make_request(
					'POST', 
					'/endpoint',
					data={'name': 'test', 'type': 'example'}
				)
			
			assert response['created_id'] == 456
	
	async def test_make_request_error_handling(self, base_service):
		"""Test HTTP error handling"""
		with aioresponses() as mock_resp:
			mock_resp.get(
				'http://localhost:8080/api/v1/test/endpoint',
				status=404,
				payload={'error': 'Not found'}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				with pytest.raises(APGServiceError) as exc_info:
					await base_service._make_request('GET', '/endpoint')
				
				assert "404" in str(exc_info.value)
				assert "Not found" in str(exc_info.value)
	
	async def test_session_cleanup(self, base_service):
		"""Test proper session cleanup"""
		session = await base_service._get_session()
		assert not session.closed
		
		await base_service.close()
		assert session.closed


class TestAPGMetadataService:
	"""Test the APG metadata service integration"""
	
	@pytest.fixture
	async def metadata_service(self):
		"""Create metadata service for testing"""
		with patch.dict('os.environ', {'APG_BASE_URL': 'http://localhost:8080'}):
			service = APGMetadataService("test-tenant", "test-user")
			yield service
			await service.close()
	
	async def test_register_schema_success(self, metadata_service):
		"""Test successful schema registration"""
		schema_data = {
			'schema_name': 'test_schema',
			'data_source_id': 'ds_123',
			'schema_type': 'relational',
			'tables': [
				{'name': 'users', 'columns': [{'name': 'id', 'type': 'integer'}]}
			]
		}
		
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/meta/schemas',
				payload={'schema_id': 'schema_456'}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				schema_id = await metadata_service.register_schema(schema_data)
			
			assert schema_id == 'schema_456'
	
	async def test_get_schema_success(self, metadata_service):
		"""Test retrieving schema by ID"""
		with aioresponses() as mock_resp:
			mock_resp.get(
				'http://localhost:8080/api/v1/meta/schemas/schema_123',
				payload={
					'schema_id': 'schema_123',
					'schema_name': 'test_schema',
					'tables': []
				}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				schema = await metadata_service.get_schema('schema_123')
			
			assert schema['schema_id'] == 'schema_123'
			assert schema['schema_name'] == 'test_schema'
	
	async def test_get_schema_not_found(self, metadata_service):
		"""Test schema not found handling"""
		with aioresponses() as mock_resp:
			mock_resp.get(
				'http://localhost:8080/api/v1/meta/schemas/nonexistent',
				status=404,
				payload={'error': 'Schema not found'}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				schema = await metadata_service.get_schema('nonexistent')
			
			assert schema is None
	
	async def test_search_schemas(self, metadata_service):
		"""Test schema search functionality"""
		with aioresponses() as mock_resp:
			mock_resp.get(
				'http://localhost:8080/api/v1/meta/schemas/search',
				payload={
					'schemas': [
						{'schema_id': 'schema_1', 'schema_name': 'users'},
						{'schema_id': 'schema_2', 'schema_name': 'orders'}
					]
				}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				schemas = await metadata_service.search_schemas('user')
			
			assert len(schemas) == 2
			assert schemas[0]['schema_name'] == 'users'
			assert schemas[1]['schema_name'] == 'orders'
	
	async def test_track_lineage(self, metadata_service):
		"""Test data lineage tracking"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/meta/lineage',
				payload={'lineage_id': 'lineage_789'}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				lineage_id = await metadata_service.track_lineage(
					source_id='table_a',
					target_id='table_b',
					relationship_type='transforms_to',
					metadata={'query_id': 'query_123'}
				)
			
			assert lineage_id == 'lineage_789'


class TestAPGCacheService:
	"""Test the APG cache service integration"""
	
	@pytest.fixture
	async def cache_service(self):
		"""Create cache service for testing"""
		with patch.dict('os.environ', {'APG_BASE_URL': 'http://localhost:8080'}):
			service = APGCacheService("test-tenant", "test-user")
			yield service
			await service.close()
	
	async def test_cache_get_success(self, cache_service):
		"""Test successful cache retrieval"""
		with aioresponses() as mock_resp:
			mock_resp.get(
				'http://localhost:8080/api/v1/cach/cache/test_key',
				payload={'value': {'result': 'cached_data', 'rows': 100}}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				value = await cache_service.get('test_key')
			
			assert value['result'] == 'cached_data'
			assert value['rows'] == 100
	
	async def test_cache_get_not_found(self, cache_service):
		"""Test cache miss handling"""
		with aioresponses() as mock_resp:
			mock_resp.get(
				'http://localhost:8080/api/v1/cach/cache/missing_key',
				status=404,
				payload={'error': 'Key not found'}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				value = await cache_service.get('missing_key')
			
			assert value is None
	
	async def test_cache_set_success(self, cache_service):
		"""Test successful cache storage"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/cach/cache',
				payload={'status': 'cached'}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				result = await cache_service.set(
					key='test_key',
					value={'result': 'data_to_cache'},
					ttl=3600,
					tags=['query', 'test']
				)
			
			assert result is True
	
	async def test_cache_delete(self, cache_service):
		"""Test cache deletion"""
		with aioresponses() as mock_resp:
			mock_resp.delete(
				'http://localhost:8080/api/v1/cach/cache/test_key',
				payload={'status': 'deleted'}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				result = await cache_service.delete('test_key')
			
			assert result is True
	
	async def test_cache_should_cache_prediction(self, cache_service):
		"""Test ML cache prediction"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/cach/cache/predict',
				payload={'should_cache': True, 'confidence': 0.85}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				should_cache = await cache_service.should_cache(
					'test_key',
					{'query_complexity': 0.7, 'execution_time_ms': 1200}
				)
			
			assert should_cache is True
	
	async def test_cache_invalidate_by_tags(self, cache_service):
		"""Test cache invalidation by tags"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/cach/cache/invalidate',
				payload={'invalidated_count': 5}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				count = await cache_service.invalidate_by_tags(['user', 'orders'])
			
			assert count == 5


class TestAPGSecurityService:
	"""Test the APG security service integration"""
	
	@pytest.fixture
	async def security_service(self):
		"""Create security service for testing"""
		with patch.dict('os.environ', {'APG_BASE_URL': 'http://localhost:8080'}):
			service = APGSecurityService("test-tenant", "test-user")
			yield service
			await service.close()
	
	async def test_validate_token_success(self, security_service):
		"""Test successful token validation"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/auth/validate',
				payload={
					'payload': {
						'user_id': 'user_123',
						'tenant_id': 'test-tenant',
						'roles': ['analyst', 'dvrl_user']
					}
				}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				payload = await security_service.validate_token('jwt_token_123')
			
			assert payload['user_id'] == 'user_123'
			assert payload['tenant_id'] == 'test-tenant'
			assert 'analyst' in payload['roles']
	
	async def test_validate_token_invalid(self, security_service):
		"""Test invalid token handling"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/auth/validate',
				status=401,
				payload={'error': 'Invalid token'}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				payload = await security_service.validate_token('invalid_token')
			
			assert payload is None
	
	async def test_check_permission_allowed(self, security_service):
		"""Test permission check - allowed"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/auth/permissions/check',
				payload={'allowed': True}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				allowed = await security_service.check_permission(
					'user_123', 'dvrl:execute', 'data_source_abc'
				)
			
			assert allowed is True
	
	async def test_check_permission_denied(self, security_service):
		"""Test permission check - denied"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/auth/permissions/check',
				payload={'allowed': False}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				allowed = await security_service.check_permission(
					'user_123', 'dvrl:admin', 'restricted_resource'
				)
			
			assert allowed is False
	
	async def test_apply_row_level_security(self, security_service):
		"""Test row-level security application"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/auth/security/row-level',
				payload={
					'modified_sql': 'SELECT * FROM orders WHERE tenant_id = ?'
				}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				modified_sql = await security_service.apply_row_level_security(
					'user_123', 
					'SELECT * FROM orders',
					'orders_db'
				)
			
			assert 'WHERE tenant_id = ?' in modified_sql
	
	async def test_get_column_permissions(self, security_service):
		"""Test column permission retrieval"""
		with aioresponses() as mock_resp:
			mock_resp.get(
				'http://localhost:8080/api/v1/auth/security/columns',
				payload={
					'column_permissions': {
						'id': 'read',
						'name': 'read',
						'email': 'masked',
						'ssn': 'denied'
					}
				}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				permissions = await security_service.get_column_permissions(
					'user_123', 'customers', 'crm_db'
				)
			
			assert permissions['id'] == 'read'
			assert permissions['email'] == 'masked'
			assert permissions['ssn'] == 'denied'
	
	async def test_audit_log(self, security_service):
		"""Test audit logging"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/auth/audit',
				payload={'audit_id': 'audit_789'}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				audit_id = await security_service.audit_log(
					user_id='user_123',
					action='query_execute',
					resource='orders_table',
					result='success',
					metadata={'query_id': 'q_456', 'rows_returned': 100}
				)
			
			assert audit_id == 'audit_789'


class TestAPGMDMService:
	"""Test the APG MDM service integration"""
	
	@pytest.fixture
	async def mdm_service(self):
		"""Create MDM service for testing"""
		with patch.dict('os.environ', {'APG_BASE_URL': 'http://localhost:8080'}):
			service = APGMDMService("test-tenant", "test-user")
			yield service
			await service.close()
	
	async def test_get_quality_score(self, mdm_service):
		"""Test data quality score retrieval"""
		with aioresponses() as mock_resp:
			mock_resp.get(
				'http://localhost:8080/api/v1/mdm/quality/score',
				payload={
					'overall_score': 0.85,
					'completeness': 0.90,
					'accuracy': 0.80,
					'consistency': 0.85
				}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				score = await mdm_service.get_quality_score('orders_db', 'customers')
			
			assert score['overall_score'] == 0.85
			assert score['completeness'] == 0.90
	
	async def test_validate_data_quality(self, mdm_service):
		"""Test data quality validation"""
		sample_data = [
			{'id': 1, 'name': 'John Doe', 'email': 'john@example.com'},
			{'id': 2, 'name': 'Jane Smith', 'email': None}
		]
		
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/mdm/quality/validate',
				payload={
					'valid': False,
					'violations': [
						{'row': 1, 'column': 'email', 'rule': 'NOT_NULL', 'severity': 'warning'}
					],
					'score': 0.75
				}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				result = await mdm_service.validate_data_quality(
					'crm_db', 'customers', sample_data
				)
			
			assert result['valid'] is False
			assert len(result['violations']) == 1
			assert result['score'] == 0.75
	
	async def test_resolve_master_data(self, mdm_service):
		"""Test master data resolution"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/mdm/master-data/resolve',
				payload={
					'master_record': {
						'master_id': 'customer_12345',
						'name': 'John Doe',
						'email': 'john.doe@company.com',
						'confidence': 0.95
					},
					'alternatives': []
				}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				result = await mdm_service.resolve_master_data(
					'customer',
					{'name': 'John Doe', 'email': 'john@company.com'}
				)
			
			assert result['master_record']['master_id'] == 'customer_12345'
			assert result['master_record']['confidence'] == 0.95


class TestAPGPerformanceOptimizer:
	"""Test the APG performance optimizer integration"""
	
	@pytest.fixture
	async def perf_service(self):
		"""Create performance service for testing"""
		with patch.dict('os.environ', {'APG_BASE_URL': 'http://localhost:8080'}):
			service = APGPerformanceOptimizer("test-tenant", "test-user")
			yield service
			await service.close()
	
	async def test_optimize_query(self, perf_service):
		"""Test query optimization"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/perf/optimize/query',
				payload={
					'optimized_sql': 'SELECT * FROM orders WHERE id > 100',
					'recommendations': ['add_index_on_id', 'use_limit_clause'],
					'estimated_improvement': 0.65
				}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				result = await perf_service.optimize_query(
					'SELECT * FROM orders',
					['orders_db'],
					{'complexity': 0.5, 'estimated_rows': 10000}
				)
			
			assert 'WHERE id > 100' in result['optimized_sql']
			assert 'add_index_on_id' in result['recommendations']
			assert result['estimated_improvement'] == 0.65
	
	async def test_record_execution_metrics(self, perf_service):
		"""Test execution metrics recording"""
		with aioresponses() as mock_resp:
			mock_resp.post(
				'http://localhost:8080/api/v1/perf/metrics/execution',
				payload={'status': 'recorded'}
			)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				success = await perf_service.record_execution_metrics(
					'query_123',
					{
						'execution_time_ms': 1250,
						'rows_processed': 50000,
						'data_transfer_bytes': 1024000
					}
				)
			
			assert success is True


class TestAPGServiceManager:
	"""Test the APG service manager coordination"""
	
	@pytest.fixture
	async def service_manager(self):
		"""Create service manager for testing"""
		with patch.dict('os.environ', {'APG_BASE_URL': 'http://localhost:8080'}):
			manager = APGServiceManager("test-tenant", "test-user")
			yield manager
			await manager.close_all_services()
	
	async def test_service_manager_initialization(self, service_manager):
		"""Test service manager service creation"""
		assert isinstance(service_manager.metadata, APGMetadataService)
		assert isinstance(service_manager.cache, APGCacheService)
		assert isinstance(service_manager.security, APGSecurityService)
		assert isinstance(service_manager.mdm, APGMDMService)
		assert isinstance(service_manager.performance, APGPerformanceOptimizer)
		
		assert len(service_manager.services) == 5
	
	async def test_initialize_services_success(self, service_manager):
		"""Test successful service initialization"""
		with aioresponses() as mock_resp:
			# Mock health checks for all services
			for endpoint in ['/meta/health', '/cach/health', '/auth/health', '/mdm/health', '/perf/health']:
				mock_resp.get(
					f'http://localhost:8080/api/v1{endpoint}',
					payload={'status': 'healthy'}
				)
			
			with patch.dict('os.environ', {'APG_ACCESS_TOKEN': 'test-token'}):
				result = await service_manager.initialize_services()
			
			assert result is True
	
	async def test_service_cleanup(self, service_manager):
		"""Test proper service cleanup"""
		# Initialize sessions
		for service in service_manager.services:
			await service._get_session()
		
		# Verify sessions are open
		for service in service_manager.services:
			assert service._session is not None
			assert not service._session.closed
		
		# Close all services
		await service_manager.close_all_services()
		
		# Verify sessions are closed
		for service in service_manager.services:
			if service._session:
				assert service._session.closed


# Integration test for real APG service communication
@pytest.mark.integration
class TestAPGIntegrationReal:
	"""Integration tests with real APG services (requires APG platform)"""
	
	@pytest.fixture
	async def real_metadata_service(self):
		"""Create real metadata service for integration testing"""
		service = APGMetadataService("integration-test", "test-user")
		yield service
		await service.close()
	
	@pytest.mark.skip(reason="Requires running APG platform")
	async def test_real_metadata_service_health(self, real_metadata_service):
		"""Test real metadata service health check"""
		try:
			await real_metadata_service._make_request('GET', '/health')
		except APGServiceError as e:
			# Expected if APG platform is not running
			assert "Connection" in str(e) or "404" in str(e)


if __name__ == '__main__':
	# Run tests with: python -m pytest tests/unit/test_apg_integrations.py -v
	pytest.main([__file__, '-v'])