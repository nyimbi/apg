"""
APG Event Streaming Bus - Service Unit Tests

Unit tests for service layer business logic and operations.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone
from uuid_extensions import uuid7str

from ...service import (
	EventStreamingService, EventPublishingService, EventConsumptionService,
	StreamProcessingService, EventSourcingService, SchemaRegistryService,
	StreamManagementService, ConsumerManagementService
)
from ...models import (
	EventConfig, StreamConfig, SubscriptionConfig, SchemaConfig,
	EventStatus, StreamStatus, EventType, EventPriority, ProcessorType
)

# =============================================================================
# Event Publishing Service Tests
# =============================================================================

@pytest.mark.unit
class TestEventPublishingService:
	"""Test EventPublishingService."""
	
	@pytest.fixture
	def publishing_service(self, mock_database_session, mock_kafka_producer, mock_redis_client):
		"""Create publishing service with mocked dependencies."""
		service = EventPublishingService()
		service.db_session = mock_database_session
		service.kafka_producer = mock_kafka_producer
		service.redis_client = mock_redis_client
		return service
	
	@pytest.mark.asyncio
	async def test_publish_event_success(self, publishing_service, test_tenant_id, test_user_id):
		"""Test successful event publishing."""
		event_config = EventConfig(
			event_type="user.created",
			source_capability="user_management",
			aggregate_id="user_123",
			aggregate_type="User"
		)
		
		payload = {"user_name": "john.doe", "email": "john@example.com"}
		
		# Mock successful publishing
		publishing_service.kafka_producer.send.return_value = AsyncMock()
		
		event_id = await publishing_service.publish_event(
			event_config=event_config,
			payload=payload,
			stream_id="user_events",
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		assert event_id.startswith("evt_")
		publishing_service.kafka_producer.send.assert_called_once()
		publishing_service.db_session.add.assert_called_once()
		publishing_service.db_session.commit.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_publish_event_batch_success(self, publishing_service, test_tenant_id, test_user_id):
		"""Test successful batch event publishing."""
		events = [
			(
				EventConfig(
					event_type="user.created",
					source_capability="user_management",
					aggregate_id=f"user_{i}",
					aggregate_type="User"
				),
				{"user_name": f"user_{i}", "email": f"user_{i}@example.com"}
			)
			for i in range(5)
		]
		
		# Mock successful batch publishing
		publishing_service.kafka_producer.send.return_value = AsyncMock()
		
		event_ids = await publishing_service.publish_event_batch(
			events=events,
			stream_id="user_events",
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		assert len(event_ids) == 5
		assert all(event_id.startswith("evt_") for event_id in event_ids)
		assert publishing_service.kafka_producer.send.call_count == 5
	
	@pytest.mark.asyncio
	async def test_publish_event_validation_failure(self, publishing_service, test_tenant_id, test_user_id):
		"""Test event publishing with validation failure."""
		event_config = EventConfig(
			event_type="",  # Invalid: empty event type
			source_capability="user_management",
			aggregate_id="user_123",
			aggregate_type="User"
		)
		
		payload = {"user_name": "john.doe"}
		
		with pytest.raises(ValueError):
			await publishing_service.publish_event(
				event_config=event_config,
				payload=payload,
				stream_id="user_events",
				tenant_id=test_tenant_id,
				user_id=test_user_id
			)
	
	@pytest.mark.asyncio
	async def test_validate_event_schema_success(self, publishing_service):
		"""Test successful event schema validation."""
		event_data = {
			"event_type": "user.created",
			"payload": {"user_name": "john.doe", "email": "john@example.com"}
		}
		
		# Mock schema retrieval and validation
		publishing_service._get_schema_for_event = AsyncMock(return_value={
			"type": "object",
			"properties": {
				"user_name": {"type": "string"},
				"email": {"type": "string", "format": "email"}
			},
			"required": ["user_name", "email"]
		})
		
		is_valid = await publishing_service.validate_event_schema(event_data)
		
		assert is_valid == True
	
	@pytest.mark.asyncio
	async def test_get_event_success(self, publishing_service):
		"""Test successful event retrieval."""
		event_id = "evt_123"
		
		# Mock database query
		mock_event = Mock()
		mock_event.event_id = event_id
		mock_event.event_type = "user.created"
		
		publishing_service.db_session.query.return_value.filter.return_value.first.return_value = mock_event
		
		event = await publishing_service.get_event(event_id)
		
		assert event == mock_event
		assert event.event_id == event_id

# =============================================================================
# Event Sourcing Service Tests
# =============================================================================

@pytest.mark.unit
class TestEventSourcingService:
	"""Test EventSourcingService."""
	
	@pytest.fixture
	def sourcing_service(self, mock_database_session, mock_redis_client):
		"""Create event sourcing service with mocked dependencies."""
		service = EventSourcingService()
		service.db_session = mock_database_session
		service.redis_client = mock_redis_client
		return service
	
	@pytest.mark.asyncio
	async def test_append_event_success(self, sourcing_service, test_tenant_id, test_user_id):
		"""Test successful event appending with optimistic concurrency."""
		event_data = {"user_name": "john.doe", "email": "john@example.com"}
		
		# Mock aggregate version check
		sourcing_service._get_aggregate_version = AsyncMock(return_value=5)
		sourcing_service._create_event = AsyncMock(return_value="evt_123")
		
		event_id = await sourcing_service.append_event(
			aggregate_id="user_123",
			aggregate_type="User",
			event_data=event_data,
			expected_version=5,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		assert event_id == "evt_123"
		sourcing_service._get_aggregate_version.assert_called_once_with("user_123", "User", test_tenant_id)
	
	@pytest.mark.asyncio
	async def test_append_event_concurrency_conflict(self, sourcing_service, test_tenant_id, test_user_id):
		"""Test event appending with concurrency conflict."""
		event_data = {"user_name": "john.doe"}
		
		# Mock version conflict
		sourcing_service._get_aggregate_version = AsyncMock(return_value=6)  # Different from expected
		
		with pytest.raises(ValueError, match="Concurrency conflict"):
			await sourcing_service.append_event(
				aggregate_id="user_123",
				aggregate_type="User",
				event_data=event_data,
				expected_version=5,  # Expected 5, but actual is 6
				tenant_id=test_tenant_id,
				user_id=test_user_id
			)
	
	@pytest.mark.asyncio
	async def test_reconstruct_aggregate_success(self, sourcing_service, test_tenant_id):
		"""Test successful aggregate reconstruction."""
		# Mock aggregate events
		mock_events = [
			Mock(event_data={"action": "create", "user_name": "john"}),
			Mock(event_data={"action": "update", "email": "john@example.com"})
		]
		
		sourcing_service._get_aggregate_events = AsyncMock(return_value=mock_events)
		sourcing_service._apply_events_to_aggregate = AsyncMock(return_value={"user_name": "john", "email": "john@example.com"})
		
		aggregate_state = await sourcing_service.reconstruct_aggregate(
			aggregate_id="user_123",
			aggregate_type="User",
			tenant_id=test_tenant_id
		)
		
		assert "user_name" in aggregate_state
		assert "email" in aggregate_state
		sourcing_service._get_aggregate_events.assert_called_once()
		sourcing_service._apply_events_to_aggregate.assert_called_once()

# =============================================================================
# Stream Management Service Tests
# =============================================================================

@pytest.mark.unit
class TestStreamManagementService:
	"""Test StreamManagementService."""
	
	@pytest.fixture
	def stream_service(self, mock_database_session, mock_kafka_admin, mock_redis_client):
		"""Create stream management service with mocked dependencies."""
		service = StreamManagementService()
		service.db_session = mock_database_session
		service.kafka_admin = mock_kafka_admin
		service.redis_client = mock_redis_client
		return service
	
	@pytest.mark.asyncio
	async def test_create_stream_processor_success(self, stream_service, test_tenant_id, test_user_id):
		"""Test successful stream processor creation."""
		processor_config = {
			"processor_name": "user_validation_filter",
			"processor_type": ProcessorType.FILTER.value,
			"source_stream_id": "str_source_123",
			"processing_logic": {"filter": "user_id IS NOT NULL"},
			"parallelism": 2
		}
		
		processor_id = await stream_service.create_stream_processor(
			processor_config=processor_config,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		assert processor_id.startswith("proc_")
		stream_service.db_session.add.assert_called_once()
		stream_service.db_session.commit.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_start_stream_processor_success(self, stream_service, test_tenant_id):
		"""Test successful stream processor start."""
		processor_id = "proc_123"
		
		# Mock processor retrieval
		mock_processor = Mock()
		mock_processor.status = "STOPPED"
		mock_processor.processor_type = ProcessorType.FILTER.value
		
		stream_service.db_session.query.return_value.filter.return_value.first.return_value = mock_processor
		stream_service._start_kafka_streams_processor = AsyncMock(return_value=True)
		
		success = await stream_service.start_stream_processor(
			processor_id=processor_id,
			tenant_id=test_tenant_id
		)
		
		assert success == True
		assert mock_processor.status == "RUNNING"
		stream_service.db_session.commit.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_get_processor_metrics_success(self, stream_service, test_tenant_id):
		"""Test successful processor metrics retrieval."""
		processor_id = "proc_123"
		
		# Mock processor and metrics
		mock_processor = Mock()
		mock_processor.processor_name = "test_processor"
		mock_processor.status = "RUNNING"
		
		stream_service.db_session.query.return_value.filter.return_value.first.return_value = mock_processor
		stream_service._get_processing_metrics = AsyncMock(return_value={
			"events_processed": 1000,
			"events_per_second": 50,
			"avg_processing_time_ms": 25
		})
		
		metrics = await stream_service.get_processor_metrics(
			processor_id=processor_id,
			tenant_id=test_tenant_id
		)
		
		assert metrics["events_processed"] == 1000
		assert metrics["events_per_second"] == 50
		assert "processor_name" in metrics

# =============================================================================
# Consumer Management Service Tests
# =============================================================================

@pytest.mark.unit
class TestConsumerManagementService:
	"""Test ConsumerManagementService."""
	
	@pytest.fixture
	def consumer_service(self, mock_database_session, mock_kafka_admin, mock_redis_client):
		"""Create consumer management service with mocked dependencies."""
		service = ConsumerManagementService()
		service.db_session = mock_database_session
		service.kafka_admin = mock_kafka_admin
		service.redis_client = mock_redis_client
		return service
	
	@pytest.mark.asyncio
	async def test_create_consumer_group_success(self, consumer_service, test_tenant_id, test_user_id):
		"""Test successful consumer group creation."""
		group_config = {
			"group_name": "user_processors",
			"session_timeout_ms": 30000,
			"heartbeat_interval_ms": 3000,
			"partition_assignment_strategy": "range"
		}
		
		group_id = await consumer_service.create_consumer_group(
			group_config=group_config,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		assert group_id.startswith("grp_")
		consumer_service.db_session.add.assert_called_once()
		consumer_service.db_session.commit.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_get_consumer_lag_success(self, consumer_service, test_tenant_id):
		"""Test successful consumer lag retrieval."""
		group_id = "grp_123"
		
		# Mock consumer group and lag calculation
		mock_group = Mock()
		mock_group.group_name = "test_group"
		mock_group.active_consumers = 3
		
		consumer_service.db_session.query.return_value.filter.return_value.first.return_value = mock_group
		consumer_service._calculate_consumer_lag = AsyncMock(return_value={
			"total_lag": 150,
			"partition_lags": {"0": 50, "1": 60, "2": 40},
			"consumption_rate": 25.5
		})
		
		lag_info = await consumer_service.get_consumer_lag(
			group_id=group_id,
			tenant_id=test_tenant_id
		)
		
		assert lag_info["total_lag"] == 150
		assert lag_info["consumption_rate"] == 25.5
		assert "group_name" in lag_info
	
	@pytest.mark.asyncio
	async def test_trigger_rebalance_success(self, consumer_service, test_tenant_id):
		"""Test successful consumer group rebalance."""
		group_id = "grp_123"
		
		# Mock consumer group
		mock_group = Mock()
		mock_group.group_name = "test_group"
		
		consumer_service.db_session.query.return_value.filter.return_value.first.return_value = mock_group
		consumer_service._trigger_kafka_rebalance = AsyncMock(return_value=True)
		
		success = await consumer_service.trigger_rebalance(
			group_id=group_id,
			tenant_id=test_tenant_id
		)
		
		assert success == True
		consumer_service._trigger_kafka_rebalance.assert_called_once_with(mock_group.group_name)

# =============================================================================
# Enhanced Schema Registry Service Tests
# =============================================================================

@pytest.mark.unit
class TestEnhancedSchemaRegistryService:
	"""Test enhanced SchemaRegistryService."""
	
	@pytest.fixture
	def schema_service(self, mock_database_session, mock_redis_client):
		"""Create schema registry service with mocked dependencies."""
		service = SchemaRegistryService()
		service.db_session = mock_database_session
		service.redis_client = mock_redis_client
		return service
	
	@pytest.mark.asyncio
	async def test_register_enhanced_schema_success(self, schema_service, test_tenant_id, test_user_id):
		"""Test successful enhanced schema registration."""
		schema_config = {
			"schema_name": "user_created",
			"schema_version": "1.0",
			"json_schema": {"type": "object", "properties": {"user_id": {"type": "string"}}},
			"event_type": "user.created",
			"compatibility_level": "BACKWARD",
			"evolution_strategy": "COMPATIBLE"
		}
		
		schema_service._check_schema_compatibility = AsyncMock(return_value=True)
		
		schema_id = await schema_service.register_enhanced_schema(
			schema_config=schema_config,
			tenant_id=test_tenant_id,
			created_by=test_user_id
		)
		
		assert schema_id.startswith("sch_")
		schema_service.db_session.add.assert_called_once()
		schema_service.db_session.commit.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_validate_event_success(self, schema_service, test_tenant_id):
		"""Test successful event validation against schema."""
		schema_id = "sch_123"
		event_data = {"user_id": "user_456", "user_name": "john.doe"}
		
		# Mock schema retrieval
		mock_schema = Mock()
		mock_schema.json_schema = {
			"type": "object",
			"properties": {
				"user_id": {"type": "string"},
				"user_name": {"type": "string"}
			},
			"required": ["user_id"]
		}
		
		schema_service.db_session.query.return_value.filter.return_value.first.return_value = mock_schema
		schema_service._validate_json_schema = AsyncMock(return_value=True)
		
		validation_result = await schema_service.validate_event(
			schema_id=schema_id,
			event_data=event_data,
			tenant_id=test_tenant_id
		)
		
		assert validation_result["is_valid"] == True
		assert "validation_errors" in validation_result
	
	@pytest.mark.asyncio
	async def test_get_schema_evolution_success(self, schema_service, test_tenant_id):
		"""Test successful schema evolution history retrieval."""
		event_type = "user.created"
		
		# Mock schema versions
		mock_schemas = [
			Mock(schema_version="1.0", created_at=datetime.now()),
			Mock(schema_version="1.1", created_at=datetime.now()),
			Mock(schema_version="2.0", created_at=datetime.now())
		]
		
		schema_service.db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_schemas
		
		evolution = await schema_service.get_schema_evolution(
			event_type=event_type,
			tenant_id=test_tenant_id
		)
		
		assert len(evolution) == 3
		assert evolution[0]["schema_version"] == "1.0"
		assert "created_at" in evolution[0]

# =============================================================================
# Event Consumption Service Tests
# =============================================================================

@pytest.mark.unit
class TestEventConsumptionService:
	"""Test EventConsumptionService."""
	
	@pytest.fixture
	def consumption_service(self, mock_database_session, mock_kafka_consumer, mock_redis_client):
		"""Create consumption service with mocked dependencies."""
		service = EventConsumptionService()
		service.db_session = mock_database_session
		service.kafka_consumer = mock_kafka_consumer
		service.redis_client = mock_redis_client
		return service
	
	@pytest.mark.asyncio
	async def test_create_subscription_success(self, consumption_service, test_tenant_id, test_user_id):
		"""Test successful subscription creation."""
		subscription_config = SubscriptionConfig(
			subscription_name="test_subscription",
			stream_id="test_stream",
			consumer_group_id="test_group",
			consumer_name="test_consumer"
		)
		
		subscription_id = await consumption_service.create_subscription(
			config=subscription_config,
			tenant_id=test_tenant_id,
			created_by=test_user_id
		)
		
		assert subscription_id.startswith("sub_")
		consumption_service.db_session.add.assert_called_once()
		consumption_service.db_session.commit.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_cancel_subscription_success(self, consumption_service, test_tenant_id):
		"""Test successful subscription cancellation."""
		subscription_id = "sub_123"
		
		# Mock subscription retrieval
		mock_subscription = Mock()
		mock_subscription.subscription_id = subscription_id
		mock_subscription.status = "active"
		
		consumption_service.db_session.query.return_value.filter.return_value.first.return_value = mock_subscription
		
		success = await consumption_service.cancel_subscription(subscription_id, test_tenant_id)
		
		assert success == True
		assert mock_subscription.status == "cancelled"
		consumption_service.db_session.commit.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_process_events_success(self, consumption_service):
		"""Test successful event processing."""
		subscription_id = "sub_123"
		events = [
			{"event_id": "evt_1", "event_type": "user.created", "payload": {"user_name": "john"}},
			{"event_id": "evt_2", "event_type": "user.updated", "payload": {"user_name": "jane"}}
		]
		
		# Mock event processing
		consumption_service._deliver_event = AsyncMock(return_value=True)
		
		processed_count = await consumption_service.process_events(subscription_id, events)
		
		assert processed_count == 2
		assert consumption_service._deliver_event.call_count == 2
	
	@pytest.mark.asyncio
	async def test_get_subscription_status(self, consumption_service, test_tenant_id):
		"""Test getting subscription status."""
		subscription_id = "sub_123"
		
		# Mock subscription and metrics retrieval
		mock_subscription = Mock()
		mock_subscription.subscription_id = subscription_id
		mock_subscription.status = "active"
		mock_subscription.last_consumed_offset = 1000
		
		consumption_service.db_session.query.return_value.filter.return_value.first.return_value = mock_subscription
		consumption_service._get_consumer_lag = AsyncMock(return_value=50)
		consumption_service._get_processing_rate = AsyncMock(return_value=100.5)
		
		status = await consumption_service.get_subscription_status(subscription_id, test_tenant_id)
		
		assert status["subscription_id"] == subscription_id
		assert status["status"] == "active"
		assert status["consumer_lag"] == 50
		assert status["processing_rate"] == 100.5

# =============================================================================
# Event Streaming Service Tests
# =============================================================================

@pytest.mark.unit
class TestEventStreamingService:
	"""Test EventStreamingService."""
	
	@pytest.fixture
	def streaming_service(self, mock_database_session, mock_kafka_producer, mock_redis_client):
		"""Create streaming service with mocked dependencies."""
		service = EventStreamingService()
		service.db_session = mock_database_session
		service.kafka_producer = mock_kafka_producer
		service.redis_client = mock_redis_client
		return service
	
	@pytest.mark.asyncio
	async def test_create_stream_success(self, streaming_service, test_tenant_id, test_user_id):
		"""Test successful stream creation."""
		stream_config = StreamConfig(
			stream_name="test_stream",
			topic_name="test-topic",
			source_capability="test_capability"
		)
		
		# Mock Kafka topic creation
		streaming_service._create_kafka_topic = AsyncMock(return_value=True)
		
		stream_id = await streaming_service.create_stream(
			config=stream_config,
			tenant_id=test_tenant_id,
			created_by=test_user_id
		)
		
		assert stream_id.startswith("str_")
		streaming_service.db_session.add.assert_called_once()
		streaming_service.db_session.commit.assert_called_once()
		streaming_service._create_kafka_topic.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_get_stream_success(self, streaming_service, test_tenant_id):
		"""Test successful stream retrieval."""
		stream_id = "str_123"
		
		# Mock stream retrieval
		mock_stream = Mock()
		mock_stream.stream_id = stream_id
		mock_stream.stream_name = "test_stream"
		
		streaming_service.db_session.query.return_value.filter.return_value.first.return_value = mock_stream
		
		stream = await streaming_service.get_stream(stream_id, test_tenant_id)
		
		assert stream == mock_stream
		assert stream.stream_id == stream_id
	
	@pytest.mark.asyncio
	async def test_list_streams_success(self, streaming_service, test_tenant_id):
		"""Test successful stream listing."""
		# Mock stream listing
		mock_streams = [
			Mock(stream_id="str_1", stream_name="stream_1"),
			Mock(stream_id="str_2", stream_name="stream_2")
		]
		
		streaming_service.db_session.query.return_value.filter.return_value.all.return_value = mock_streams
		
		streams = await streaming_service.list_streams(test_tenant_id)
		
		assert len(streams) == 2
		assert streams[0].stream_id == "str_1"
		assert streams[1].stream_id == "str_2"
	
	@pytest.mark.asyncio
	async def test_get_stream_metrics(self, streaming_service, test_tenant_id):
		"""Test getting stream metrics."""
		stream_id = "str_123"
		
		# Mock metrics calculation
		streaming_service._calculate_stream_metrics = AsyncMock(return_value={
			"total_events": 1000,
			"events_per_second": 10.5,
			"events_today": 500,
			"consumer_count": 3,
			"total_lag": 25
		})
		
		metrics = await streaming_service.get_stream_metrics(stream_id, test_tenant_id)
		
		assert metrics["total_events"] == 1000
		assert metrics["events_per_second"] == 10.5
		assert metrics["consumer_count"] == 3
	
	@pytest.mark.asyncio
	async def test_query_events_success(self, streaming_service):
		"""Test successful event querying."""
		filters = {
			"stream_id": "str_123",
			"event_type": "user.created",
			"tenant_id": "test_tenant"
		}
		
		# Mock event query
		mock_events = [
			Mock(event_id="evt_1", event_type="user.created"),
			Mock(event_id="evt_2", event_type="user.created")
		]
		
		streaming_service.db_session.query.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = mock_events
		streaming_service.db_session.query.return_value.filter.return_value.count.return_value = 2
		
		events, total_count = await streaming_service.query_events(
			filters=filters,
			limit=10,
			offset=0
		)
		
		assert len(events) == 2
		assert total_count == 2
		assert events[0].event_id == "evt_1"

# =============================================================================
# Schema Registry Service Tests
# =============================================================================

@pytest.mark.unit
class TestSchemaRegistryService:
	"""Test SchemaRegistryService."""
	
	@pytest.fixture
	def schema_service(self, mock_database_session, mock_redis_client):
		"""Create schema service with mocked dependencies."""
		service = SchemaRegistryService()
		service.db_session = mock_database_session
		service.redis_client = mock_redis_client
		return service
	
	@pytest.mark.asyncio
	async def test_register_schema_success(self, schema_service, test_tenant_id, test_user_id):
		"""Test successful schema registration."""
		schema_config = SchemaConfig(
			schema_name="user_created",
			schema_version="1.0",
			schema_definition={
				"type": "object",
				"properties": {
					"user_name": {"type": "string"},
					"email": {"type": "string", "format": "email"}
				},
				"required": ["user_name", "email"]
			},
			event_type="user.created"
		)
		
		schema_id = await schema_service.register_schema(
			config=schema_config,
			tenant_id=test_tenant_id,
			created_by=test_user_id
		)
		
		assert schema_id.startswith("sch_")
		schema_service.db_session.add.assert_called_once()
		schema_service.db_session.commit.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_validate_event_schema_success(self, schema_service):
		"""Test successful event schema validation."""
		event_data = {
			"user_name": "john.doe",
			"email": "john@example.com"
		}
		
		schema_definition = {
			"type": "object",
			"properties": {
				"user_name": {"type": "string"},
				"email": {"type": "string", "format": "email"}
			},
			"required": ["user_name", "email"]
		}
		
		is_valid = await schema_service.validate_event_schema(event_data, schema_definition)
		
		assert is_valid == True
	
	@pytest.mark.asyncio
	async def test_validate_event_schema_failure(self, schema_service):
		"""Test event schema validation failure."""
		event_data = {
			"user_name": "john.doe"
			# Missing required 'email' field
		}
		
		schema_definition = {
			"type": "object",
			"properties": {
				"user_name": {"type": "string"},
				"email": {"type": "string", "format": "email"}
			},
			"required": ["user_name", "email"]
		}
		
		is_valid = await schema_service.validate_event_schema(event_data, schema_definition)
		
		assert is_valid == False
	
	@pytest.mark.asyncio
	async def test_get_schema_success(self, schema_service, test_tenant_id):
		"""Test successful schema retrieval."""
		schema_id = "sch_123"
		
		# Mock schema retrieval
		mock_schema = Mock()
		mock_schema.schema_id = schema_id
		mock_schema.schema_name = "user_created"
		mock_schema.schema_definition = {"type": "object"}
		
		schema_service.db_session.query.return_value.filter.return_value.first.return_value = mock_schema
		
		schema = await schema_service.get_schema(schema_id, test_tenant_id)
		
		assert schema == mock_schema
		assert schema.schema_id == schema_id
	
	@pytest.mark.asyncio
	async def test_list_schemas_success(self, schema_service, test_tenant_id):
		"""Test successful schema listing."""
		# Mock schema listing
		mock_schemas = [
			Mock(schema_id="sch_1", schema_name="user_created"),
			Mock(schema_id="sch_2", schema_name="order_created")
		]
		
		schema_service.db_session.query.return_value.filter.return_value.all.return_value = mock_schemas
		
		schemas = await schema_service.list_schemas(test_tenant_id)
		
		assert len(schemas) == 2
		assert schemas[0].schema_id == "sch_1"
		assert schemas[1].schema_id == "sch_2"

# =============================================================================
# Stream Processing Service Tests
# =============================================================================

@pytest.mark.unit
class TestStreamProcessingService:
	"""Test StreamProcessingService."""
	
	@pytest.fixture
	def processing_service(self, mock_kafka_consumer, mock_redis_client):
		"""Create processing service with mocked dependencies."""
		service = StreamProcessingService()
		service.kafka_consumer = mock_kafka_consumer
		service.redis_client = mock_redis_client
		return service
	
	@pytest.mark.asyncio
	async def test_process_stream_events(self, processing_service):
		"""Test stream event processing."""
		stream_id = "str_123"
		processor_config = {
			"type": "aggregation",
			"window_size_ms": 60000,
			"aggregation_function": "count"
		}
		
		# Mock event processing
		processing_service._process_events_batch = AsyncMock(return_value=10)
		
		processed_count = await processing_service.process_stream_events(
			stream_id=stream_id,
			processor_config=processor_config
		)
		
		assert processed_count == 10
		processing_service._process_events_batch.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_create_aggregation_window(self, processing_service):
		"""Test creating aggregation window."""
		window_config = {
			"window_type": "tumbling",
			"duration_ms": 60000,
			"aggregation_function": "sum",
			"field": "amount"
		}
		
		window_id = await processing_service.create_aggregation_window(
			stream_id="str_123",
			config=window_config
		)
		
		assert window_id.startswith("win_")
	
	@pytest.mark.asyncio
	async def test_process_complex_event_pattern(self, processing_service):
		"""Test complex event pattern processing."""
		pattern_config = {
			"pattern_type": "sequence",
			"events": ["user.created", "user.activated"],
			"within_ms": 300000,
			"correlation_field": "user_id"
		}
		
		events = [
			{"event_type": "user.created", "payload": {"user_id": "123"}},
			{"event_type": "user.activated", "payload": {"user_id": "123"}}
		]
		
		# Mock pattern matching
		processing_service._match_event_pattern = AsyncMock(return_value=True)
		
		matches = await processing_service.process_complex_event_pattern(
			pattern_config=pattern_config,
			events=events
		)
		
		assert matches == True
		processing_service._match_event_pattern.assert_called_once()