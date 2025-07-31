"""
APG Event Streaming Bus - Test Configuration

Pytest configuration and shared fixtures for Event Streaming Bus tests.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock
from uuid_extensions import uuid7str

from ..models import (
	ESEvent, ESStream, ESSubscription, ESConsumerGroup, ESSchema,
	EventConfig, StreamConfig, SubscriptionConfig, SchemaConfig,
	EventStatus, StreamStatus, SubscriptionStatus, EventType, DeliveryMode
)
from ..service import (
	EventStreamingService, EventPublishingService, EventConsumptionService,
	StreamProcessingService, EventSourcingService, SchemaRegistryService
)
from ..apg_integration import (
	APGEventStreamingIntegration, APGCapabilityInfo, EventRoutingRule,
	CrossCapabilityWorkflow, EventCompositionPattern
)

# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
	"""Configure pytest with custom markers."""
	config.addinivalue_line(
		"markers", "unit: mark test as a unit test"
	)
	config.addinivalue_line(
		"markers", "integration: mark test as an integration test"
	)
	config.addinivalue_line(
		"markers", "performance: mark test as a performance test"
	)
	config.addinivalue_line(
		"markers", "slow: mark test as slow running"
	)

# =============================================================================
# Event Loop Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
	"""Create event loop for async tests."""
	loop = asyncio.new_event_loop()
	yield loop
	loop.close()

# =============================================================================
# Test Data Factories
# =============================================================================

@pytest.fixture
def sample_event_data() -> Dict[str, Any]:
	"""Create sample event data for testing."""
	return {
		"event_id": f"evt_{uuid7str()}",
		"event_type": "user.created",
		"event_version": "1.0",
		"source_capability": "user_management",
		"aggregate_id": f"usr_{uuid7str()}",
		"aggregate_type": "User",
		"sequence_number": 1,
		"timestamp": datetime.now(timezone.utc),
		"correlation_id": f"cor_{uuid7str()}",
		"causation_id": None,
		"tenant_id": "test_tenant",
		"user_id": "test_user",
		"payload": {
			"user_name": "john.doe",
			"email": "john.doe@example.com",
			"department": "Engineering"
		},
		"metadata": {
			"ip_address": "192.168.1.100",
			"user_agent": "APG-Client/1.0"
		},
		"schema_id": "user_created_v1",
		"schema_version": "1.0",
		"serialization_format": "json",
		"status": EventStatus.PENDING.value,
		"retry_count": 0,
		"max_retries": 3,
		"stream_id": "user_events",
		"partition_key": "user_123",
		"created_by": "test_user"
	}

@pytest.fixture
def sample_stream_config() -> StreamConfig:
	"""Create sample stream configuration."""
	return StreamConfig(
		stream_name="test_stream",
		stream_description="Test stream for unit tests",
		topic_name="test-topic",
		partitions=3,
		replication_factor=2,
		retention_time_ms=604800000,  # 7 days
		retention_size_bytes=1073741824,  # 1GB
		cleanup_policy="delete",
		source_capability="test_capability"
	)

@pytest.fixture
def sample_subscription_config() -> SubscriptionConfig:
	"""Create sample subscription configuration."""
	return SubscriptionConfig(
		subscription_name="test_subscription",
		subscription_description="Test subscription for unit tests",
		stream_id="test_stream",
		consumer_group_id="test_consumers",
		consumer_name="test_consumer",
		event_type_patterns=["user.*", "order.created"],
		delivery_mode=DeliveryMode.AT_LEAST_ONCE,
		batch_size=100,
		max_wait_time_ms=1000
	)

@pytest.fixture
def sample_schema_config() -> SchemaConfig:
	"""Create sample schema configuration."""
	return SchemaConfig(
		schema_name="user_created",
		schema_version="1.0",
		schema_definition={
			"type": "object",
			"properties": {
				"user_name": {"type": "string"},
				"email": {"type": "string", "format": "email"},
				"department": {"type": "string"}
			},
			"required": ["user_name", "email"]
		},
		event_type="user.created"
	)

@pytest.fixture
def sample_capability_info() -> APGCapabilityInfo:
	"""Create sample capability information."""
	return APGCapabilityInfo(
		capability_id="test_capability",
		capability_name="Test Capability",
		capability_type="domain",
		version="1.0.0",
		endpoints={
			"api": "/api/v1",
			"health": "/health"
		},
		event_patterns=["test.*"],
		dependencies=["event_streaming_bus"],
		status="active",
		last_heartbeat=datetime.now(timezone.utc)
	)

@pytest.fixture
def sample_routing_rule() -> EventRoutingRule:
	"""Create sample event routing rule."""
	return EventRoutingRule(
		rule_id=f"rule_{uuid7str()}",
		source_pattern="user_management",
		target_capabilities=["notification_service", "analytics_service"],
		event_type_patterns=["user.*"],
		priority=100,
		is_active=True
	)

@pytest.fixture
def sample_workflow() -> CrossCapabilityWorkflow:
	"""Create sample cross-capability workflow."""
	return CrossCapabilityWorkflow(
		workflow_id=f"wf_{uuid7str()}",
		workflow_name="User Onboarding Workflow",
		trigger_events=["user.created"],
		steps=[
			{
				"step_id": "send_welcome_email",
				"capability": "notification_service",
				"action": "send_email",
				"parameters": {
					"template": "welcome_email",
					"recipient": "${trigger_event.payload.email}"
				}
			},
			{
				"step_id": "create_user_profile",
				"capability": "profile_service",
				"action": "create_profile",
				"parameters": {
					"user_id": "${trigger_event.aggregate_id}",
					"department": "${trigger_event.payload.department}"
				}
			}
		],
		timeout_seconds=300,
		max_retries=3,
		is_active=True
	)

@pytest.fixture
def sample_composition_pattern() -> EventCompositionPattern:
	"""Create sample event composition pattern."""
	return EventCompositionPattern(
		pattern_id=f"pattern_{uuid7str()}",
		pattern_name="Order Completion Pattern",
		event_inputs=["order.payment_received", "order.items_shipped"],
		composition_logic={
			"type": "all_events_received",
			"correlation_field": "order_id"
		},
		output_event_type="order.completed",
		window_duration_ms=300000,  # 5 minutes
		min_events=2,
		max_events=2
	)

# =============================================================================
# Mock Service Fixtures
# =============================================================================

@pytest.fixture
def mock_event_streaming_service():
	"""Create mock event streaming service."""
	service = Mock(spec=EventStreamingService)
	service.create_stream = AsyncMock(return_value="stream_123")
	service.get_stream = AsyncMock()
	service.list_streams = AsyncMock(return_value=[])
	service.get_stream_events = AsyncMock(return_value=([], 0))
	service.get_stream_metrics = AsyncMock(return_value={})
	service.query_events = AsyncMock(return_value=([], 0))
	return service

@pytest.fixture
def mock_event_publishing_service():
	"""Create mock event publishing service."""
	service = Mock(spec=EventPublishingService)
	service.publish_event = AsyncMock(return_value="evt_123")
	service.publish_event_batch = AsyncMock(return_value=["evt_123", "evt_124"])
	service.get_event = AsyncMock()
	service.validate_event = AsyncMock(return_value=True)
	return service

@pytest.fixture
def mock_event_consumption_service():
	"""Create mock event consumption service."""
	service = Mock(spec=EventConsumptionService)
	service.create_subscription = AsyncMock(return_value="sub_123")
	service.cancel_subscription = AsyncMock(return_value=True)
	service.list_subscriptions = AsyncMock(return_value=[])
	service.get_subscription_status = AsyncMock(return_value={})
	service.process_events = AsyncMock()
	return service

@pytest.fixture
def mock_schema_registry_service():
	"""Create mock schema registry service."""
	service = Mock(spec=SchemaRegistryService)
	service.register_schema = AsyncMock(return_value="schema_123")
	service.get_schema = AsyncMock()
	service.list_schemas = AsyncMock(return_value=[])
	service.validate_event_schema = AsyncMock(return_value=True)
	return service

# =============================================================================
# Database and External Service Mocks
# =============================================================================

@pytest.fixture
def mock_database_session():
	"""Create mock database session."""
	session = Mock()
	session.add = Mock()
	session.commit = AsyncMock()
	session.rollback = AsyncMock()
	session.close = AsyncMock()
	session.execute = AsyncMock()
	session.query = Mock()
	return session

@pytest.fixture
def mock_kafka_producer():
	"""Create mock Kafka producer."""
	producer = Mock()
	producer.send = AsyncMock()
	producer.flush = AsyncMock()
	producer.close = AsyncMock()
	return producer

@pytest.fixture
def mock_kafka_consumer():
	"""Create mock Kafka consumer."""
	consumer = Mock()
	consumer.subscribe = AsyncMock()
	consumer.unsubscribe = AsyncMock()
	consumer.poll = AsyncMock(return_value={})
	consumer.commit = AsyncMock()
	consumer.close = AsyncMock()
	return consumer

@pytest.fixture
def mock_redis_client():
	"""Create mock Redis client."""
	redis = Mock()
	redis.set = AsyncMock(return_value=True)
	redis.get = AsyncMock(return_value=None)
	redis.delete = AsyncMock(return_value=1)
	redis.exists = AsyncMock(return_value=False)
	redis.xadd = AsyncMock(return_value="stream-id")
	redis.xread = AsyncMock(return_value=[])
	return redis

# =============================================================================
# Integration Test Fixtures
# =============================================================================

@pytest.fixture
async def integration_event_streaming_service(mock_database_session, mock_kafka_producer, mock_redis_client):
	"""Create event streaming service for integration tests."""
	service = EventStreamingService()
	# In a real integration test, you'd inject actual dependencies
	# For now, we'll use mocks to avoid external dependencies
	service.db_session = mock_database_session
	service.kafka_producer = mock_kafka_producer
	service.redis_client = mock_redis_client
	return service

@pytest.fixture
async def integration_apg_integration(
	integration_event_streaming_service,
	mock_event_publishing_service,
	mock_event_consumption_service
):
	"""Create APG integration service for integration tests."""
	integration = APGEventStreamingIntegration(
		event_streaming_service=integration_event_streaming_service,
		publishing_service=mock_event_publishing_service,
		consumption_service=mock_event_consumption_service
	)
	await integration.initialize()
	yield integration
	await integration.shutdown()

# =============================================================================
# Performance Test Fixtures
# =============================================================================

@pytest.fixture
def performance_event_generator():
	"""Generate events for performance testing."""
	def generate_events(count: int) -> List[Dict[str, Any]]:
		events = []
		for i in range(count):
			event_data = {
				"event_type": f"test.event.{i % 10}",
				"source_capability": "performance_test",
				"aggregate_id": f"agg_{i}",
				"aggregate_type": "TestAggregate",
				"payload": {
					"test_id": i,
					"timestamp": datetime.now(timezone.utc).isoformat(),
					"data": f"test_data_{i}"
				},
				"metadata": {
					"test_run": "performance_test",
					"event_index": i
				}
			}
			events.append(event_data)
		return events
	
	return generate_events

@pytest.fixture
def performance_metrics_collector():
	"""Collect performance metrics during tests."""
	class MetricsCollector:
		def __init__(self):
			self.metrics = {}
			self.start_times = {}
		
		def start_timer(self, operation: str):
			self.start_times[operation] = datetime.now()
		
		def end_timer(self, operation: str):
			if operation in self.start_times:
				duration = (datetime.now() - self.start_times[operation]).total_seconds()
				if operation not in self.metrics:
					self.metrics[operation] = []
				self.metrics[operation].append(duration)
				del self.start_times[operation]
				return duration
			return None
		
		def get_average(self, operation: str) -> float:
			if operation in self.metrics and self.metrics[operation]:
				return sum(self.metrics[operation]) / len(self.metrics[operation])
			return 0.0
		
		def get_percentile(self, operation: str, percentile: float) -> float:
			if operation in self.metrics and self.metrics[operation]:
				sorted_times = sorted(self.metrics[operation])
				index = int(len(sorted_times) * (percentile / 100.0))
				return sorted_times[min(index, len(sorted_times) - 1)]
			return 0.0
	
	return MetricsCollector()

# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def event_matcher():
	"""Utility for matching events in tests."""
	def match_event(actual_event: Dict[str, Any], expected_fields: Dict[str, Any]) -> bool:
		for field, expected_value in expected_fields.items():
			if field not in actual_event:
				return False
			if actual_event[field] != expected_value:
				return False
		return True
	
	return match_event

@pytest.fixture
def json_serializer():
	"""JSON serializer that handles datetime objects."""
	def serialize(obj):
		def default_serializer(o):
			if isinstance(o, datetime):
				return o.isoformat()
			raise TypeError(f"Object of type {type(o)} is not JSON serializable")
		
		return json.dumps(obj, default=default_serializer, indent=2)
	
	return serialize

@pytest.fixture
def test_tenant_id():
	"""Standard test tenant ID."""
	return "test_tenant"

@pytest.fixture
def test_user_id():
	"""Standard test user ID."""
	return "test_user"