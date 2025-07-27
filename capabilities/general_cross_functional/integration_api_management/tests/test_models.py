"""
APG Integration API Management - Model Tests

Unit tests for SQLAlchemy and Pydantic models including validation,
relationships, and data integrity constraints.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError

from ..models import (
	AMAPI, AMEndpoint, AMPolicy, AMConsumer, AMAPIKey, AMSubscription,
	AMDeployment, AMAnalytics, AMUsageRecord,
	APIStatus, ProtocolType, AuthenticationType, PolicyType,
	ConsumerStatus, DeploymentStrategy,
	APIConfig, EndpointConfig, PolicyConfig, ConsumerConfig, APIKeyConfig
)

# =============================================================================
# SQLAlchemy Model Tests
# =============================================================================

@pytest.mark.unit
class TestAMAPIModel:
	"""Test AMAPI model."""
	
	def test_create_api(self, test_session, sample_api_data):
		"""Test creating an API."""
		api = AMAPI(
			tenant_id="test_tenant",
			created_by="test_user",
			**sample_api_data
		)
		
		test_session.add(api)
		test_session.commit()
		
		assert api.api_id is not None
		assert api.api_id.startswith("api_")
		assert api.api_name == sample_api_data["api_name"]
		assert api.status == APIStatus.DRAFT.value
		assert api.created_at is not None
		assert api.tenant_id == "test_tenant"
	
	def test_api_name_validation(self, test_session):
		"""Test API name validation."""
		# Invalid characters
		api = AMAPI(
			api_name="invalid name with spaces",
			api_title="Test API",
			base_path="/test",
			upstream_url="http://localhost:8000",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		with pytest.raises(ValueError, match="API name can only contain"):
			api.validate_api_name("api_name", "invalid name with spaces")
	
	def test_base_path_validation(self, test_session):
		"""Test base path validation."""
		api = AMAPI(
			api_name="test_api",
			api_title="Test API",
			base_path="invalid_path",  # Should start with /
			upstream_url="http://localhost:8000",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		with pytest.raises(ValueError, match="Base path must start with"):
			api.validate_base_path("base_path", "invalid_path")
	
	def test_unique_constraint(self, test_session, sample_api_data):
		"""Test unique constraint on api_name, version, tenant."""
		# Create first API
		api1 = AMAPI(
			tenant_id="test_tenant",
			created_by="test_user",
			**sample_api_data
		)
		test_session.add(api1)
		test_session.commit()
		
		# Try to create duplicate
		api2 = AMAPI(
			tenant_id="test_tenant",
			created_by="test_user",
			**sample_api_data
		)
		test_session.add(api2)
		
		with pytest.raises(IntegrityError):
			test_session.commit()

@pytest.mark.unit
class TestAMEndpointModel:
	"""Test AMEndpoint model."""
	
	def test_create_endpoint(self, test_session, db_api):
		"""Test creating an endpoint."""
		endpoint = AMEndpoint(
			api_id=db_api.api_id,
			path="/users",
			method="GET",
			summary="Get users",
			auth_required=True
		)
		
		test_session.add(endpoint)
		test_session.commit()
		
		assert endpoint.endpoint_id is not None
		assert endpoint.endpoint_id.startswith("ep_")
		assert endpoint.path == "/users"
		assert endpoint.method == "GET"
		assert endpoint.auth_required is True
	
	def test_endpoint_path_validation(self, test_session, db_api):
		"""Test endpoint path validation."""
		endpoint = AMEndpoint(
			api_id=db_api.api_id,
			path="invalid_path",  # Should start with /
			method="GET"
		)
		
		with pytest.raises(ValueError, match="Endpoint path must start with"):
			endpoint.validate_path("path", "invalid_path")
	
	def test_endpoint_relationship(self, test_session, db_api):
		"""Test endpoint-API relationship."""
		endpoint = AMEndpoint(
			api_id=db_api.api_id,
			path="/test",
			method="POST"
		)
		
		test_session.add(endpoint)
		test_session.commit()
		
		# Test relationship
		assert endpoint.api == db_api
		assert endpoint in db_api.endpoints

@pytest.mark.unit
class TestAMConsumerModel:
	"""Test AMConsumer model."""
	
	def test_create_consumer(self, test_session, sample_consumer_data):
		"""Test creating a consumer."""
		consumer = AMConsumer(
			tenant_id="test_tenant",
			created_by="test_user",
			**sample_consumer_data
		)
		
		test_session.add(consumer)
		test_session.commit()
		
		assert consumer.consumer_id is not None
		assert consumer.consumer_id.startswith("con_")
		assert consumer.consumer_name == sample_consumer_data["consumer_name"]
		assert consumer.status == ConsumerStatus.PENDING.value
	
	def test_consumer_unique_constraint(self, test_session, sample_consumer_data):
		"""Test consumer unique constraint."""
		# Create first consumer
		consumer1 = AMConsumer(
			tenant_id="test_tenant",
			created_by="test_user",
			**sample_consumer_data
		)
		test_session.add(consumer1)
		test_session.commit()
		
		# Try to create duplicate in same tenant
		consumer2 = AMConsumer(
			tenant_id="test_tenant",
			created_by="test_user",
			**sample_consumer_data
		)
		test_session.add(consumer2)
		
		with pytest.raises(IntegrityError):
			test_session.commit()

@pytest.mark.unit
class TestAMAPIKeyModel:
	"""Test AMAPIKey model."""
	
	def test_create_api_key(self, test_session, db_consumer):
		"""Test creating an API key."""
		api_key = AMAPIKey(
			consumer_id=db_consumer.consumer_id,
			key_name="test_key",
			key_hash="hashed_value",
			key_prefix="tk_12345",
			active=True,
			created_by="test_user"
		)
		
		test_session.add(api_key)
		test_session.commit()
		
		assert api_key.key_id is not None
		assert api_key.key_id.startswith("key_")
		assert api_key.key_name == "test_key"
		assert api_key.active is True
	
	def test_api_key_relationship(self, test_session, db_consumer):
		"""Test API key-consumer relationship."""
		api_key = AMAPIKey(
			consumer_id=db_consumer.consumer_id,
			key_name="test_key",
			key_hash="hashed_value",
			key_prefix="tk_12345",
			created_by="test_user"
		)
		
		test_session.add(api_key)
		test_session.commit()
		
		# Test relationship
		assert api_key.consumer == db_consumer
		assert api_key in db_consumer.api_keys

@pytest.mark.unit
class TestAMUsageRecordModel:
	"""Test AMUsageRecord model."""
	
	def test_create_usage_record(self, test_session, db_consumer):
		"""Test creating a usage record."""
		usage_record = AMUsageRecord(
			request_id="req_123456",
			consumer_id=db_consumer.consumer_id,
			api_id="api_123",
			endpoint_path="/test",
			method="GET",
			timestamp=datetime.now(timezone.utc),
			response_status=200,
			response_time_ms=150,
			client_ip="127.0.0.1",
			tenant_id="test_tenant"
		)
		
		test_session.add(usage_record)
		test_session.commit()
		
		assert usage_record.record_id is not None
		assert usage_record.record_id.startswith("usg_")
		assert usage_record.response_status == 200
		assert usage_record.billable is True

# =============================================================================
# Pydantic Model Tests
# =============================================================================

@pytest.mark.unit
class TestAPIConfig:
	"""Test APIConfig Pydantic model."""
	
	def test_valid_api_config(self, sample_api_data):
		"""Test valid API configuration."""
		config = APIConfig(**sample_api_data)
		
		assert config.api_name == sample_api_data["api_name"]
		assert config.api_title == sample_api_data["api_title"]
		assert config.base_path == sample_api_data["base_path"]
		assert config.upstream_url == sample_api_data["upstream_url"]
	
	def test_base_path_validation(self):
		"""Test base path validation."""
		with pytest.raises(ValueError, match='Base path must start with'):
			APIConfig(
				api_name="test_api",
				api_title="Test API",
				base_path="invalid_path",
				upstream_url="http://localhost:8000"
			)
	
	def test_timeout_validation(self):
		"""Test timeout validation."""
		with pytest.raises(ValueError):
			APIConfig(
				api_name="test_api",
				api_title="Test API",
				base_path="/test",
				upstream_url="http://localhost:8000",
				timeout_ms=500  # Below minimum
			)

@pytest.mark.unit
class TestEndpointConfig:
	"""Test EndpointConfig Pydantic model."""
	
	def test_valid_endpoint_config(self):
		"""Test valid endpoint configuration."""
		config = EndpointConfig(
			path="/users",
			method="GET",
			summary="Get users",
			auth_required=True
		)
		
		assert config.path == "/users"
		assert config.method == "GET"
		assert config.auth_required is True
	
	def test_method_validation(self):
		"""Test HTTP method validation."""
		config = EndpointConfig(
			path="/test",
			method="get"  # lowercase
		)
		
		assert config.method == "GET"  # Should be uppercase
		
		with pytest.raises(ValueError):
			EndpointConfig(
				path="/test",
				method="INVALID"
			)
	
	def test_path_validation(self):
		"""Test path validation."""
		with pytest.raises(ValueError, match='Path must start with'):
			EndpointConfig(
				path="invalid_path",
				method="GET"
			)

@pytest.mark.unit
class TestConsumerConfig:
	"""Test ConsumerConfig Pydantic model."""
	
	def test_valid_consumer_config(self, sample_consumer_data):
		"""Test valid consumer configuration."""
		config = ConsumerConfig(**sample_consumer_data)
		
		assert config.consumer_name == sample_consumer_data["consumer_name"]
		assert config.contact_email == sample_consumer_data["contact_email"]
		assert config.portal_access is True
	
	def test_email_validation(self):
		"""Test email validation."""
		with pytest.raises(ValueError, match='Invalid email address'):
			ConsumerConfig(
				consumer_name="test_consumer",
				contact_email="invalid_email"
			)

@pytest.mark.unit
class TestPolicyConfig:
	"""Test PolicyConfig Pydantic model."""
	
	def test_valid_policy_config(self):
		"""Test valid policy configuration."""
		config = PolicyConfig(
			policy_name="rate_limiting",
			policy_type=PolicyType.RATE_LIMITING,
			config={
				"requests_per_minute": 1000,
				"burst_size": 100
			},
			execution_order=100
		)
		
		assert config.policy_name == "rate_limiting"
		assert config.policy_type == PolicyType.RATE_LIMITING
		assert config.config["requests_per_minute"] == 1000
		assert config.execution_order == 100

# =============================================================================
# Model Integration Tests
# =============================================================================

@pytest.mark.integration
class TestModelIntegration:
	"""Test model integration and relationships."""
	
	def test_api_with_endpoints_and_policies(self, test_session, sample_api_data):
		"""Test API with related endpoints and policies."""
		# Create API
		api = AMAPI(
			tenant_id="test_tenant",
			created_by="test_user",
			**sample_api_data
		)
		test_session.add(api)
		test_session.flush()
		
		# Add endpoints
		endpoint1 = AMEndpoint(
			api_id=api.api_id,
			path="/users",
			method="GET",
			summary="Get users"
		)
		endpoint2 = AMEndpoint(
			api_id=api.api_id,
			path="/users",
			method="POST",
			summary="Create user"
		)
		
		test_session.add_all([endpoint1, endpoint2])
		
		# Add policy
		policy = AMPolicy(
			api_id=api.api_id,
			policy_name="rate_limit",
			policy_type=PolicyType.RATE_LIMITING.value,
			config={"requests_per_minute": 1000},
			created_by="test_user"
		)
		
		test_session.add(policy)
		test_session.commit()
		
		# Test relationships
		assert len(api.endpoints) == 2
		assert len(api.policies) == 1
		assert endpoint1.api == api
		assert policy.api == api
	
	def test_consumer_with_keys_and_subscriptions(self, test_session, sample_consumer_data, db_api):
		"""Test consumer with API keys and subscriptions."""
		# Create consumer
		consumer = AMConsumer(
			tenant_id="test_tenant",
			created_by="test_user",
			**sample_consumer_data
		)
		test_session.add(consumer)
		test_session.flush()
		
		# Add API key
		api_key = AMAPIKey(
			consumer_id=consumer.consumer_id,
			key_name="primary_key",
			key_hash="hashed_value",
			key_prefix="pk_12345",
			created_by="test_user"
		)
		
		# Add subscription
		subscription = AMSubscription(
			consumer_id=consumer.consumer_id,
			api_id=db_api.api_id,
			subscription_name="test_subscription",
			plan_name="basic",
			rate_limit=500
		)
		
		test_session.add_all([api_key, subscription])
		test_session.commit()
		
		# Test relationships
		assert len(consumer.api_keys) == 1
		assert len(consumer.subscriptions) == 1
		assert api_key.consumer == consumer
		assert subscription.consumer == consumer
		assert subscription.api == db_api
	
	def test_cascade_delete(self, test_session, sample_api_data):
		"""Test cascade delete behavior."""
		# Create API with endpoints
		api = AMAPI(
			tenant_id="test_tenant",
			created_by="test_user",
			**sample_api_data
		)
		test_session.add(api)
		test_session.flush()
		
		endpoint = AMEndpoint(
			api_id=api.api_id,
			path="/test",
			method="GET"
		)
		test_session.add(endpoint)
		test_session.commit()
		
		endpoint_id = endpoint.endpoint_id
		
		# Delete API
		test_session.delete(api)
		test_session.commit()
		
		# Check that endpoint was also deleted
		deleted_endpoint = test_session.query(AMEndpoint).filter_by(endpoint_id=endpoint_id).first()
		assert deleted_endpoint is None

# =============================================================================
# Model Validation Tests
# =============================================================================

@pytest.mark.unit
class TestModelValidation:
	"""Test model validation logic."""
	
	def test_api_status_enum_validation(self, test_session):
		"""Test API status enum validation."""
		api = AMAPI(
			api_name="test_api",
			api_title="Test API",
			base_path="/test",
			upstream_url="http://localhost:8000",
			status="invalid_status",  # Invalid status
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		test_session.add(api)
		
		# Should not raise error at model level (validation happens at service level)
		test_session.commit()
		
		# But enum should validate properly
		assert APIStatus.ACTIVE.value == "active"
		assert APIStatus.DEPRECATED.value == "deprecated"
	
	def test_policy_type_validation(self, test_session, db_api):
		"""Test policy type validation."""
		policy = AMPolicy(
			api_id=db_api.api_id,
			policy_name="test_policy",
			policy_type="invalid_type",
			config={},
			created_by="test_user"
		)
		
		with pytest.raises(ValueError, match="Invalid policy type"):
			policy.validate_policy_type("policy_type", "invalid_type")
	
	def test_check_constraints(self, test_session):
		"""Test database check constraints."""
		# Test negative timeout
		api = AMAPI(
			api_name="test_api",
			api_title="Test API",
			base_path="/test",
			upstream_url="http://localhost:8000",
			timeout_ms=-1000,  # Negative timeout
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		test_session.add(api)
		
		with pytest.raises(IntegrityError):
			test_session.commit()