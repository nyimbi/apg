"""
APG Integration API Management - Service Tests

Unit and integration tests for service layer components including
API lifecycle, consumer management, policy management, and analytics.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from ..models import (
	AMAPI, AMConsumer, AMAPIKey, AMUsageRecord,
	APIConfig, ConsumerConfig, APIKeyConfig,
	APIStatus, ConsumerStatus
)
from ..service import (
	APILifecycleService, ConsumerManagementService,
	PolicyManagementService, AnalyticsService
)

# =============================================================================
# API Lifecycle Service Tests
# =============================================================================

@pytest.mark.unit
class TestAPILifecycleService:
	"""Test API lifecycle service."""
	
	@pytest_asyncio.fixture
	async def api_lifecycle_service(self, test_session):
		"""Create API lifecycle service with test session."""
		service = APILifecycleService()
		service._session = test_session
		return service
	
	@pytest.mark.asyncio
	async def test_register_api(self, api_lifecycle_service, sample_api_data):
		"""Test API registration."""
		config = APIConfig(**sample_api_data)
		
		api_id = await api_lifecycle_service.register_api(
			config=config,
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		assert api_id is not None
		assert api_id.startswith("api_")
		
		# Verify API was created in database
		api = api_lifecycle_service._session.query(AMAPI).filter_by(api_id=api_id).first()
		assert api is not None
		assert api.api_name == sample_api_data["api_name"]
		assert api.status == APIStatus.DRAFT.value
	
	@pytest.mark.asyncio
	async def test_get_api(self, api_lifecycle_service, db_api):
		"""Test getting API by ID."""
		api = await api_lifecycle_service.get_api(
			api_id=db_api.api_id,
			tenant_id=db_api.tenant_id
		)
		
		assert api is not None
		assert api.api_id == db_api.api_id
		assert api.api_name == db_api.api_name
	
	@pytest.mark.asyncio
	async def test_get_api_not_found(self, api_lifecycle_service):
		"""Test getting non-existent API."""
		api = await api_lifecycle_service.get_api(
			api_id="non_existent_api",
			tenant_id="test_tenant"
		)
		
		assert api is None
	
	@pytest.mark.asyncio
	async def test_activate_api(self, api_lifecycle_service, db_api):
		"""Test API activation."""
		success = await api_lifecycle_service.activate_api(
			api_id=db_api.api_id,
			tenant_id=db_api.tenant_id,
			activated_by="test_user"
		)
		
		assert success is True
		
		# Verify status changed
		api = api_lifecycle_service._session.query(AMAPI).filter_by(api_id=db_api.api_id).first()
		assert api.status == APIStatus.ACTIVE.value
	
	@pytest.mark.asyncio
	async def test_deprecate_api(self, api_lifecycle_service, db_api):
		"""Test API deprecation."""
		# First activate the API
		await api_lifecycle_service.activate_api(
			api_id=db_api.api_id,
			tenant_id=db_api.tenant_id,
			activated_by="test_user"
		)
		
		# Then deprecate it
		success = await api_lifecycle_service.deprecate_api(
			api_id=db_api.api_id,
			migration_timeline="6 months",
			tenant_id=db_api.tenant_id,
			deprecated_by="test_user"
		)
		
		assert success is True
		
		# Verify status changed
		api = api_lifecycle_service._session.query(AMAPI).filter_by(api_id=db_api.api_id).first()
		assert api.status == APIStatus.DEPRECATED.value
	
	@pytest.mark.asyncio
	async def test_update_api_configuration(self, api_lifecycle_service, db_api):
		"""Test updating API configuration."""
		updates = {
			"api_description": "Updated description",
			"timeout_ms": 45000,
			"tags": ["updated", "test"]
		}
		
		success = await api_lifecycle_service.update_api_configuration(
			api_id=db_api.api_id,
			updates=updates,
			tenant_id=db_api.tenant_id,
			updated_by="test_user"
		)
		
		assert success is True
		
		# Verify updates
		api = api_lifecycle_service._session.query(AMAPI).filter_by(api_id=db_api.api_id).first()
		assert api.api_description == "Updated description"
		assert api.timeout_ms == 45000
		assert "updated" in api.tags

@pytest.mark.unit
class TestConsumerManagementService:
	"""Test consumer management service."""
	
	@pytest_asyncio.fixture
	async def consumer_service(self, test_session):
		"""Create consumer management service with test session."""
		service = ConsumerManagementService()
		service._session = test_session
		return service
	
	@pytest.mark.asyncio
	async def test_register_consumer(self, consumer_service, sample_consumer_data):
		"""Test consumer registration."""
		config = ConsumerConfig(**sample_consumer_data)
		
		consumer_id = await consumer_service.register_consumer(
			config=config,
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		assert consumer_id is not None
		assert consumer_id.startswith("con_")
		
		# Verify consumer was created
		consumer = consumer_service._session.query(AMConsumer).filter_by(consumer_id=consumer_id).first()
		assert consumer is not None
		assert consumer.consumer_name == sample_consumer_data["consumer_name"]
		assert consumer.status == ConsumerStatus.PENDING.value
	
	@pytest.mark.asyncio
	async def test_approve_consumer(self, consumer_service, db_consumer):
		"""Test consumer approval."""
		success = await consumer_service.approve_consumer(
			consumer_id=db_consumer.consumer_id,
			tenant_id=db_consumer.tenant_id,
			approved_by="admin_user"
		)
		
		assert success is True
		
		# Verify status changed
		consumer = consumer_service._session.query(AMConsumer).filter_by(consumer_id=db_consumer.consumer_id).first()
		assert consumer.status == ConsumerStatus.ACTIVE.value
		assert consumer.approved_by == "admin_user"
		assert consumer.approval_date is not None
	
	@pytest.mark.asyncio
	async def test_generate_api_key(self, consumer_service, db_consumer):
		"""Test API key generation."""
		config = APIKeyConfig(
			key_name="test_key",
			scopes=["read", "write"],
			allowed_apis=["api_123"]
		)
		
		key_id, api_key = await consumer_service.generate_api_key(
			config=config,
			tenant_id=db_consumer.tenant_id,
			created_by="test_user"
		)
		
		assert key_id is not None
		assert key_id.startswith("key_")
		assert api_key is not None
		assert len(api_key) >= 32  # Should be sufficiently long
		
		# Verify key was created
		key_record = consumer_service._session.query(AMAPIKey).filter_by(key_id=key_id).first()
		assert key_record is not None
		assert key_record.key_name == "test_key"
		assert "read" in key_record.scopes
		assert key_record.active is True
	
	@pytest.mark.asyncio
	async def test_validate_api_key(self, consumer_service, db_api_key):
		"""Test API key validation."""
		consumer_id = await consumer_service.validate_api_key(
			api_key_hash=db_api_key.key_hash,
			tenant_id=db_api_key.consumer.tenant_id
		)
		
		assert consumer_id == db_api_key.consumer_id
	
	@pytest.mark.asyncio
	async def test_validate_invalid_api_key(self, consumer_service):
		"""Test validation of invalid API key."""
		consumer_id = await consumer_service.validate_api_key(
			api_key_hash="invalid_hash",
			tenant_id="test_tenant"
		)
		
		assert consumer_id is None
	
	@pytest.mark.asyncio
	async def test_revoke_api_key(self, consumer_service, db_api_key):
		"""Test API key revocation."""
		success = await consumer_service.revoke_api_key(
			key_id=db_api_key.key_id,
			tenant_id=db_api_key.consumer.tenant_id,
			revoked_by="admin_user"
		)
		
		assert success is True
		
		# Verify key was deactivated
		key_record = consumer_service._session.query(AMAPIKey).filter_by(key_id=db_api_key.key_id).first()
		assert key_record.active is False

@pytest.mark.unit
class TestPolicyManagementService:
	"""Test policy management service."""
	
	@pytest_asyncio.fixture
	async def policy_service(self, test_session):
		"""Create policy management service with test session."""
		service = PolicyManagementService()
		service._session = test_session
		return service
	
	@pytest.mark.asyncio
	async def test_create_policy(self, policy_service, db_api):
		"""Test policy creation."""
		from ..models import PolicyConfig, PolicyType
		
		config = PolicyConfig(
			policy_name="rate_limiting",
			policy_type=PolicyType.RATE_LIMITING,
			config={
				"requests_per_minute": 1000,
				"burst_size": 100
			}
		)
		
		policy_id = await policy_service.create_policy(
			api_id=db_api.api_id,
			config=config,
			tenant_id=db_api.tenant_id,
			created_by="test_user"
		)
		
		assert policy_id is not None
		assert policy_id.startswith("pol_")
	
	@pytest.mark.asyncio
	async def test_apply_policies(self, policy_service, db_api):
		"""Test policy application."""
		# Create a test policy first
		from ..models import PolicyConfig, PolicyType
		
		config = PolicyConfig(
			policy_name="test_policy",
			policy_type=PolicyType.VALIDATION,
			config={"strict_mode": True}
		)
		
		policy_id = await policy_service.create_policy(
			api_id=db_api.api_id,
			config=config,
			tenant_id=db_api.tenant_id,
			created_by="test_user"
		)
		
		# Apply policies
		applied_policies = await policy_service.apply_policies(
			api_id=db_api.api_id,
			request_context={"path": "/test", "method": "GET"},
			tenant_id=db_api.tenant_id
		)
		
		assert isinstance(applied_policies, list)

@pytest.mark.unit
class TestAnalyticsService:
	"""Test analytics service."""
	
	@pytest_asyncio.fixture
	async def analytics_service(self, test_session):
		"""Create analytics service with test session."""
		service = AnalyticsService()
		service._session = test_session
		return service
	
	@pytest.mark.asyncio
	async def test_record_usage(self, analytics_service, db_consumer):
		"""Test usage recording."""
		usage_data = {
			"request_id": "req_test_123",
			"consumer_id": db_consumer.consumer_id,
			"api_id": "api_test_123",
			"endpoint_path": "/test",
			"method": "GET",
			"timestamp": datetime.now(timezone.utc),
			"response_status": 200,
			"response_time_ms": 150,
			"client_ip": "127.0.0.1",
			"tenant_id": db_consumer.tenant_id
		}
		
		success = await analytics_service.record_usage(usage_data)
		assert success is True
		
		# Verify record was created
		record = analytics_service._session.query(AMUsageRecord).filter_by(
			request_id="req_test_123"
		).first()
		assert record is not None
		assert record.response_status == 200
		assert record.response_time_ms == 150
	
	@pytest.mark.asyncio
	async def test_get_total_requests(self, analytics_service, db_consumer):
		"""Test getting total request count."""
		# Create some usage records
		end_time = datetime.now(timezone.utc)
		start_time = end_time - timedelta(hours=1)
		
		for i in range(5):
			usage_data = {
				"request_id": f"req_test_{i}",
				"consumer_id": db_consumer.consumer_id,
				"api_id": "api_test_123",
				"endpoint_path": "/test",
				"method": "GET",
				"timestamp": start_time + timedelta(minutes=i*10),
				"response_status": 200,
				"response_time_ms": 100 + i*10,
				"client_ip": "127.0.0.1",
				"tenant_id": db_consumer.tenant_id
			}
			await analytics_service.record_usage(usage_data)
		
		# Get total requests
		total = await analytics_service.get_total_requests(
			start_time=start_time,
			end_time=end_time,
			tenant_id=db_consumer.tenant_id
		)
		
		assert total == 5
	
	@pytest.mark.asyncio
	async def test_get_average_response_time(self, analytics_service, db_consumer):
		"""Test getting average response time."""
		# Create usage records with different response times
		end_time = datetime.now(timezone.utc)
		start_time = end_time - timedelta(hours=1)
		
		response_times = [100, 150, 200, 250, 300]
		for i, response_time in enumerate(response_times):
			usage_data = {
				"request_id": f"req_avg_{i}",
				"consumer_id": db_consumer.consumer_id,
				"api_id": "api_test_123",
				"endpoint_path": "/test",
				"method": "GET",
				"timestamp": start_time + timedelta(minutes=i*10),
				"response_status": 200,
				"response_time_ms": response_time,
				"client_ip": "127.0.0.1",
				"tenant_id": db_consumer.tenant_id
			}
			await analytics_service.record_usage(usage_data)
		
		# Get average response time
		avg_response_time = await analytics_service.get_average_response_time(
			start_time=start_time,
			end_time=end_time,
			tenant_id=db_consumer.tenant_id
		)
		
		assert avg_response_time == 200.0  # Average of [100, 150, 200, 250, 300]
	
	@pytest.mark.asyncio
	async def test_get_error_rate(self, analytics_service, db_consumer):
		"""Test getting error rate."""
		# Create usage records with different status codes
		end_time = datetime.now(timezone.utc)
		start_time = end_time - timedelta(hours=1)
		
		status_codes = [200, 200, 200, 400, 500]  # 40% error rate
		for i, status_code in enumerate(status_codes):
			usage_data = {
				"request_id": f"req_error_{i}",
				"consumer_id": db_consumer.consumer_id,
				"api_id": "api_test_123",
				"endpoint_path": "/test",
				"method": "GET",
				"timestamp": start_time + timedelta(minutes=i*10),
				"response_status": status_code,
				"response_time_ms": 150,
				"client_ip": "127.0.0.1",
				"tenant_id": db_consumer.tenant_id
			}
			await analytics_service.record_usage(usage_data)
		
		# Get error rate
		error_rate = await analytics_service.get_error_rate(
			start_time=start_time,
			end_time=end_time,
			tenant_id=db_consumer.tenant_id
		)
		
		assert error_rate == 0.4  # 40% error rate

# =============================================================================
# Service Integration Tests
# =============================================================================

@pytest.mark.integration
class TestServiceIntegration:
	"""Test service integration scenarios."""
	
	@pytest.mark.asyncio
	async def test_api_lifecycle_workflow(self, test_session, sample_api_data):
		"""Test complete API lifecycle workflow."""
		api_service = APILifecycleService()
		api_service._session = test_session
		
		# 1. Register API
		config = APIConfig(**sample_api_data)
		api_id = await api_service.register_api(
			config=config,
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		# Verify initial state
		api = await api_service.get_api(api_id, "test_tenant")
		assert api.status == APIStatus.DRAFT.value
		
		# 2. Activate API
		await api_service.activate_api(api_id, "test_tenant", "test_user")
		
		api = await api_service.get_api(api_id, "test_tenant")
		assert api.status == APIStatus.ACTIVE.value
		
		# 3. Update configuration
		updates = {"api_description": "Updated in workflow"}
		await api_service.update_api_configuration(
			api_id, updates, "test_tenant", "test_user"
		)
		
		api = await api_service.get_api(api_id, "test_tenant")
		assert api.api_description == "Updated in workflow"
		
		# 4. Deprecate API
		await api_service.deprecate_api(
			api_id, "6 months", "test_tenant", "test_user"
		)
		
		api = await api_service.get_api(api_id, "test_tenant")
		assert api.status == APIStatus.DEPRECATED.value
	
	@pytest.mark.asyncio
	async def test_consumer_onboarding_workflow(self, test_session, sample_consumer_data):
		"""Test complete consumer onboarding workflow."""
		consumer_service = ConsumerManagementService()
		consumer_service._session = test_session
		
		# 1. Register consumer
		config = ConsumerConfig(**sample_consumer_data)
		consumer_id = await consumer_service.register_consumer(
			config=config,
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		# Verify initial state
		consumer = consumer_service._session.query(AMConsumer).filter_by(consumer_id=consumer_id).first()
		assert consumer.status == ConsumerStatus.PENDING.value
		
		# 2. Approve consumer
		await consumer_service.approve_consumer(consumer_id, "test_tenant", "admin_user")
		
		consumer = consumer_service._session.query(AMConsumer).filter_by(consumer_id=consumer_id).first()
		assert consumer.status == ConsumerStatus.ACTIVE.value
		
		# 3. Generate API key
		key_config = APIKeyConfig(key_name="primary_key")
		key_id, api_key = await consumer_service.generate_api_key(
			config=key_config,
			tenant_id="test_tenant",
			created_by="admin_user"
		)
		
		# 4. Validate API key
		validated_consumer_id = await consumer_service.validate_api_key(
			api_key_hash=consumer_service._hash_api_key(api_key),
			tenant_id="test_tenant"
		)
		
		assert validated_consumer_id == consumer_id
	
	@pytest.mark.asyncio
	async def test_cross_service_analytics(self, test_session, sample_api_data, sample_consumer_data):
		"""Test analytics across multiple services."""
		# Initialize services
		api_service = APILifecycleService()
		api_service._session = test_session
		
		consumer_service = ConsumerManagementService()
		consumer_service._session = test_session
		
		analytics_service = AnalyticsService()
		analytics_service._session = test_session
		
		# Create API and consumer
		api_config = APIConfig(**sample_api_data)
		api_id = await api_service.register_api(api_config, "test_tenant", "test_user")
		
		consumer_config = ConsumerConfig(**sample_consumer_data)
		consumer_id = await consumer_service.register_consumer(consumer_config, "test_tenant", "test_user")
		
		# Generate usage data
		end_time = datetime.now(timezone.utc)
		start_time = end_time - timedelta(hours=1)
		
		for i in range(10):
			usage_data = {
				"request_id": f"req_cross_{i}",
				"consumer_id": consumer_id,
				"api_id": api_id,
				"endpoint_path": "/test",
				"method": "GET",
				"timestamp": start_time + timedelta(minutes=i*5),
				"response_status": 200 if i < 8 else 500,
				"response_time_ms": 100 + i*10,
				"client_ip": "127.0.0.1",
				"tenant_id": "test_tenant"
			}
			await analytics_service.record_usage(usage_data)
		
		# Verify analytics
		total_requests = await analytics_service.get_total_requests(start_time, end_time, "test_tenant")
		assert total_requests == 10
		
		error_rate = await analytics_service.get_error_rate(start_time, end_time, "test_tenant")
		assert error_rate == 0.2  # 2 out of 10 requests failed

# =============================================================================
# Service Error Handling Tests
# =============================================================================

@pytest.mark.unit
class TestServiceErrorHandling:
	"""Test service error handling."""
	
	@pytest.mark.asyncio
	async def test_api_service_duplicate_registration(self, api_service, sample_api_data):
		"""Test handling of duplicate API registration."""
		config = APIConfig(**sample_api_data)
		
		# Register first API
		api_id1 = await api_service.register_api(config, "test_tenant", "test_user")
		assert api_id1 is not None
		
		# Try to register duplicate
		api_id2 = await api_service.register_api(config, "test_tenant", "test_user")
		assert api_id2 is None  # Should fail due to unique constraint
	
	@pytest.mark.asyncio
	async def test_consumer_service_invalid_approval(self, consumer_service):
		"""Test handling of invalid consumer approval."""
		success = await consumer_service.approve_consumer(
			consumer_id="non_existent_consumer",
			tenant_id="test_tenant",
			approved_by="admin_user"
		)
		
		assert success is False
	
	@pytest.mark.asyncio
	async def test_analytics_service_invalid_data(self, analytics_service):
		"""Test handling of invalid analytics data."""
		# Missing required fields
		invalid_usage_data = {
			"request_id": "req_invalid",
			# Missing consumer_id, api_id, etc.
		}
		
		success = await analytics_service.record_usage(invalid_usage_data)
		assert success is False

# =============================================================================
# Service Performance Tests
# =============================================================================

@pytest.mark.performance
class TestServicePerformance:
	"""Test service performance."""
	
	@pytest.mark.asyncio
	async def test_bulk_api_registration(self, api_service, performance_test_data):
		"""Test bulk API registration performance."""
		import time
		
		start_time = time.time()
		
		created_apis = []
		for api_data in performance_test_data["apis"][:10]:  # Test with 10 APIs
			config = APIConfig(**api_data)
			api_id = await api_service.register_api(config, "perf_tenant", "perf_user")
			if api_id:
				created_apis.append(api_id)
		
		end_time = time.time()
		duration = end_time - start_time
		
		assert len(created_apis) == 10
		assert duration < 5.0  # Should complete within 5 seconds
		
		# Calculate throughput
		throughput = len(created_apis) / duration
		assert throughput > 1.0  # At least 1 API per second
	
	@pytest.mark.asyncio
	async def test_bulk_usage_recording(self, analytics_service, performance_test_data):
		"""Test bulk usage recording performance."""
		import time
		
		start_time = time.time()
		
		recorded_count = 0
		for i in range(100):  # Test with 100 usage records
			usage_data = {
				"request_id": f"req_perf_{i}",
				"consumer_id": "perf_consumer_123",
				"api_id": "perf_api_123",
				"endpoint_path": "/perf",
				"method": "GET",
				"timestamp": datetime.now(timezone.utc),
				"response_status": 200,
				"response_time_ms": 100,
				"client_ip": "127.0.0.1",
				"tenant_id": "perf_tenant"
			}
			
			success = await analytics_service.record_usage(usage_data)
			if success:
				recorded_count += 1
		
		end_time = time.time()
		duration = end_time - start_time
		
		assert recorded_count == 100
		assert duration < 10.0  # Should complete within 10 seconds
		
		# Calculate throughput
		throughput = recorded_count / duration
		assert throughput > 10.0  # At least 10 records per second