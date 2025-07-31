"""
APG Event Streaming Bus - Model Unit Tests

Unit tests for data models, validation, and model behavior.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
from datetime import datetime, timezone
from uuid_extensions import uuid7str
from pydantic import ValidationError

from ...models import (
	ESEvent, ESStream, ESSubscription, ESConsumerGroup, ESSchema, ESMetrics,
	ESEventSchema, ESStreamAssignment, ESEventProcessingHistory, ESStreamProcessor,
	EventConfig, StreamConfig, SubscriptionConfig, SchemaConfig,
	EventStatus, StreamStatus, SubscriptionStatus, EventType, DeliveryMode,
	CompressionType, SerializationFormat, EventPriority, ProcessorType, ConsumerStatus
)

# =============================================================================
# Event Model Tests
# =============================================================================

@pytest.mark.unit
class TestESEvent:
	"""Test ESEvent model."""
	
	def test_event_creation_with_defaults(self, sample_event_data):
		"""Test creating an event with default values."""
		event_data = sample_event_data.copy()
		del event_data['event_id']  # Let it auto-generate
		
		event = ESEvent(**event_data)
		
		assert event.event_id.startswith('evt_')
		assert event.event_version == "1.0"
		assert event.status == EventStatus.PENDING.value
		assert event.retry_count == 0
		assert event.max_retries == 3
		assert event.sequence_number == 1
	
	def test_event_validation_event_type(self):
		"""Test event type validation."""
		with pytest.raises(ValueError, match="Event type cannot be empty"):
			event = ESEvent(
				event_type="",
				source_capability="test",
				aggregate_id="test",
				aggregate_type="Test",
				payload={"test": "data"},
				tenant_id="test",
				created_by="test",
				stream_id="test"
			)
	
	def test_event_validation_payload(self):
		"""Test payload validation."""
		with pytest.raises(ValueError, match="Payload must be a dictionary"):
			event = ESEvent(
				event_type="test.event",
				source_capability="test",
				aggregate_id="test",
				aggregate_type="Test",
				payload="invalid_payload",  # Should be dict
				tenant_id="test",
				created_by="test",
				stream_id="test"
			)
	
	def test_event_string_representation(self, sample_event_data):
		"""Test event string representation."""
		event = ESEvent(**sample_event_data)
		
		repr_str = repr(event)
		assert "ESEvent" in repr_str
		assert event.event_id in repr_str
		assert event.event_type in repr_str
		assert event.aggregate_id in repr_str

# =============================================================================
# Stream Model Tests
# =============================================================================

@pytest.mark.unit
class TestESStream:
	"""Test ESStream model."""
	
	def test_stream_creation_with_defaults(self):
		"""Test creating a stream with default values."""
		stream = ESStream(
			stream_name="test_stream",
			topic_name="test-topic",
			source_capability="test_capability",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		assert stream.stream_id.startswith('str_')
		assert stream.partitions == 3
		assert stream.replication_factor == 3
		assert stream.retention_time_ms == 604800000  # 7 days
		assert stream.compression_type == CompressionType.SNAPPY.value
		assert stream.status == StreamStatus.ACTIVE.value
	
	def test_stream_name_validation(self):
		"""Test stream name validation."""
		with pytest.raises(ValueError, match="Stream name cannot be empty"):
			stream = ESStream(
				stream_name="",
				topic_name="test-topic",
				source_capability="test",
				tenant_id="test",
				created_by="test"
			)
	
	def test_stream_name_kafka_compliance(self):
		"""Test stream name Kafka topic naming compliance."""
		with pytest.raises(ValueError, match="can only contain alphanumeric"):
			stream = ESStream(
				stream_name="invalid@name!",
				topic_name="test-topic",
				source_capability="test",
				tenant_id="test",
				created_by="test"
			)
	
	def test_stream_string_representation(self):
		"""Test stream string representation."""
		stream = ESStream(
			stream_name="test_stream",
			topic_name="test-topic",
			source_capability="test_capability",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		repr_str = repr(stream)
		assert "ESStream" in repr_str
		assert stream.stream_id in repr_str
		assert stream.stream_name in repr_str
		assert stream.topic_name in repr_str

# =============================================================================
# Subscription Model Tests
# =============================================================================

@pytest.mark.unit
class TestESSubscription:
	"""Test ESSubscription model."""
	
	def test_subscription_creation_with_defaults(self):
		"""Test creating a subscription with default values."""
		subscription = ESSubscription(
			subscription_name="test_subscription",
			stream_id="test_stream",
			consumer_group_id="test_group",
			consumer_name="test_consumer",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		assert subscription.subscription_id.startswith('sub_')
		assert subscription.delivery_mode == DeliveryMode.AT_LEAST_ONCE.value
		assert subscription.batch_size == 100
		assert subscription.max_wait_time_ms == 1000
		assert subscription.start_position == "latest"
		assert subscription.status == SubscriptionStatus.ACTIVE.value
		assert subscription.dead_letter_enabled == True
	
	def test_subscription_event_patterns_validation(self):
		"""Test event type patterns validation."""
		with pytest.raises(ValueError, match="Event type patterns must be a list"):
			subscription = ESSubscription(
				subscription_name="test_subscription",
				stream_id="test_stream",
				consumer_group_id="test_group",
				consumer_name="test_consumer",
				event_type_patterns="invalid_patterns",  # Should be list
				tenant_id="test_tenant",
				created_by="test_user"
			)
	
	def test_subscription_string_representation(self):
		"""Test subscription string representation."""
		subscription = ESSubscription(
			subscription_name="test_subscription",
			stream_id="test_stream",
			consumer_group_id="test_group",
			consumer_name="test_consumer",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		repr_str = repr(subscription)
		assert "ESSubscription" in repr_str
		assert subscription.subscription_id in repr_str
		assert subscription.subscription_name in repr_str
		assert subscription.consumer_group_id in repr_str

# =============================================================================
# Schema Model Tests
# =============================================================================

@pytest.mark.unit
class TestESSchema:
	"""Test ESSchema model."""
	
	def test_schema_creation_with_defaults(self):
		"""Test creating a schema with default values."""
		schema = ESSchema(
			schema_name="test_schema",
			schema_version="1.0",
			schema_definition={"type": "object"},
			event_type="test.event",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		assert schema.schema_id.startswith('sch_')
		assert schema.schema_format == "json_schema"
		assert schema.compatibility_level == "backward"
		assert schema.is_active == True
	
	def test_schema_definition_validation(self):
		"""Test schema definition validation."""
		with pytest.raises(ValueError, match="Schema definition must be a dictionary"):
			schema = ESSchema(
				schema_name="test_schema",
				schema_version="1.0",
				schema_definition="invalid_definition",  # Should be dict
				event_type="test.event",
				tenant_id="test_tenant",
				created_by="test_user"
			)
	
	def test_schema_string_representation(self):
		"""Test schema string representation."""
		schema = ESSchema(
			schema_name="test_schema",
			schema_version="1.0",
			schema_definition={"type": "object"},
			event_type="test.event",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		repr_str = repr(schema)
		assert "ESSchema" in repr_str
		assert schema.schema_id in repr_str
		assert schema.schema_name in repr_str
		assert schema.schema_version in repr_str

# =============================================================================
# Pydantic API Model Tests
# =============================================================================

@pytest.mark.unit
class TestEventConfig:
	"""Test EventConfig Pydantic model."""
	
	def test_event_config_creation(self):
		"""Test creating an event configuration."""
		config = EventConfig(
			event_type="user.created",
			source_capability="user_management",
			aggregate_id="user_123",
			aggregate_type="User"
		)
		
		assert config.event_type == "user.created"
		assert config.event_version == "1.0"
		assert config.sequence_number == 0
		assert config.metadata == {}
	
	def test_event_config_validation_required_fields(self):
		"""Test validation of required fields."""
		with pytest.raises(ValidationError):
			EventConfig(
				# Missing required fields
				event_type="user.created"
			)
	
	def test_event_config_validation_field_lengths(self):
		"""Test field length validation."""
		with pytest.raises(ValidationError):
			EventConfig(
				event_type="a" * 101,  # Too long
				source_capability="test",
				aggregate_id="test",
				aggregate_type="Test"
			)

@pytest.mark.unit
class TestStreamConfig:
	"""Test StreamConfig Pydantic model."""
	
	def test_stream_config_creation(self, sample_stream_config):
		"""Test creating a stream configuration."""
		assert sample_stream_config.stream_name == "test_stream"
		assert sample_stream_config.partitions == 3
		assert sample_stream_config.cleanup_policy == "delete"
		assert sample_stream_config.compression_type == CompressionType.SNAPPY
	
	def test_stream_config_cleanup_policy_validation(self):
		"""Test cleanup policy validation."""
		with pytest.raises(ValidationError):
			StreamConfig(
				stream_name="test_stream",
				topic_name="test-topic",
				source_capability="test",
				cleanup_policy="invalid_policy"  # Should be delete or compact
			)
	
	def test_stream_config_partitions_validation(self):
		"""Test partitions validation."""
		with pytest.raises(ValidationError):
			StreamConfig(
				stream_name="test_stream",
				topic_name="test-topic",
				source_capability="test",
				partitions=0  # Should be >= 1
			)

@pytest.mark.unit
class TestSubscriptionConfig:
	"""Test SubscriptionConfig Pydantic model."""
	
	def test_subscription_config_creation(self, sample_subscription_config):
		"""Test creating a subscription configuration."""
		assert sample_subscription_config.subscription_name == "test_subscription"
		assert sample_subscription_config.delivery_mode == DeliveryMode.AT_LEAST_ONCE
		assert sample_subscription_config.batch_size == 100
		assert sample_subscription_config.start_position == "latest"
	
	def test_subscription_config_start_position_validation(self):
		"""Test start position validation."""
		with pytest.raises(ValidationError):
			SubscriptionConfig(
				subscription_name="test_subscription",
				stream_id="test_stream",
				consumer_group_id="test_group",
				consumer_name="test_consumer",
				start_position="invalid_position"  # Should be earliest, latest, or specific_offset
			)
	
	def test_subscription_config_batch_size_validation(self):
		"""Test batch size validation."""
		with pytest.raises(ValidationError):
			SubscriptionConfig(
				subscription_name="test_subscription",
				stream_id="test_stream",
				consumer_group_id="test_group",
				consumer_name="test_consumer",
				batch_size=0  # Should be >= 1
			)

@pytest.mark.unit
class TestSchemaConfig:
	"""Test SchemaConfig Pydantic model."""
	
	def test_schema_config_creation(self, sample_schema_config):
		"""Test creating a schema configuration."""
		assert sample_schema_config.schema_name == "user_created"
		assert sample_schema_config.schema_format == "json_schema"
		assert sample_schema_config.compatibility_level == "backward"
	
	def test_schema_config_format_validation(self):
		"""Test schema format validation."""
		with pytest.raises(ValidationError):
			SchemaConfig(
				schema_name="test_schema",
				schema_version="1.0",
				schema_definition={"type": "object"},
				event_type="test.event",
				schema_format="invalid_format"  # Should be json_schema, avro, or protobuf
			)
	
	def test_schema_config_compatibility_validation(self):
		"""Test compatibility level validation."""
		with pytest.raises(ValidationError):
			SchemaConfig(
				schema_name="test_schema",
				schema_version="1.0",
				schema_definition={"type": "object"},
				event_type="test.event",
				compatibility_level="invalid_level"  # Should be backward, forward, full, or none
			)

# =============================================================================
# Enum Tests
# =============================================================================

@pytest.mark.unit
class TestEnums:
	"""Test enum classes."""
	
	def test_event_status_values(self):
		"""Test EventStatus enum values."""
		assert EventStatus.PENDING.value == "pending"
		assert EventStatus.PUBLISHED.value == "published"
		assert EventStatus.CONSUMED.value == "consumed"
		assert EventStatus.FAILED.value == "failed"
		assert EventStatus.RETRY.value == "retry"
		assert EventStatus.DEAD_LETTER.value == "dead_letter"
	
	def test_stream_status_values(self):
		"""Test StreamStatus enum values."""
		assert StreamStatus.ACTIVE.value == "active"
		assert StreamStatus.PAUSED.value == "paused"
		assert StreamStatus.ARCHIVED.value == "archived"
		assert StreamStatus.ERROR.value == "error"
	
	def test_event_type_values(self):
		"""Test EventType enum values."""
		assert EventType.DOMAIN_EVENT.value == "domain_event"
		assert EventType.INTEGRATION_EVENT.value == "integration_event"
		assert EventType.NOTIFICATION_EVENT.value == "notification_event"
		assert EventType.SYSTEM_EVENT.value == "system_event"
		assert EventType.AUDIT_EVENT.value == "audit_event"
	
	def test_delivery_mode_values(self):
		"""Test DeliveryMode enum values."""
		assert DeliveryMode.AT_MOST_ONCE.value == "at_most_once"
		assert DeliveryMode.AT_LEAST_ONCE.value == "at_least_once"
		assert DeliveryMode.EXACTLY_ONCE.value == "exactly_once"
	
	def test_event_priority_values(self):
		"""Test EventPriority enum values."""
		assert EventPriority.LOW.value == "low"
		assert EventPriority.NORMAL.value == "normal"
		assert EventPriority.HIGH.value == "high"
		assert EventPriority.CRITICAL.value == "critical"
	
	def test_processor_type_values(self):
		"""Test ProcessorType enum values."""
		assert ProcessorType.FILTER.value == "filter"
		assert ProcessorType.MAP.value == "map"
		assert ProcessorType.AGGREGATE.value == "aggregate"
		assert ProcessorType.JOIN.value == "join"
		assert ProcessorType.WINDOW.value == "window"
		assert ProcessorType.CUSTOM.value == "custom"

# =============================================================================
# Enhanced Event Schema Model Tests
# =============================================================================

@pytest.mark.unit
class TestESEventSchema:
	"""Test ESEventSchema model."""
	
	def test_event_schema_creation_with_defaults(self):
		"""Test creating an event schema with default values."""
		schema = ESEventSchema(
			schema_name="user_created",
			schema_version="1.0",
			json_schema={"type": "object", "properties": {"user_id": {"type": "string"}}},
			event_type="user.created",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		assert schema.schema_id.startswith('sch_')
		assert schema.compatibility_level == "BACKWARD"
		assert schema.evolution_strategy == "COMPATIBLE"
		assert schema.is_active == True
		assert schema.validation_rules == {}
	
	def test_event_schema_validation_json_schema(self):
		"""Test JSON schema validation."""
		with pytest.raises(ValueError, match="JSON schema must be a dictionary"):
			schema = ESEventSchema(
				schema_name="test_schema",
				schema_version="1.0",
				json_schema="invalid_schema",  # Should be dict
				event_type="test.event",
				tenant_id="test_tenant",
				created_by="test_user"
			)
	
	def test_event_schema_compatibility_levels(self):
		"""Test valid compatibility levels."""
		valid_levels = ["BACKWARD", "FORWARD", "FULL", "NONE"]
		
		for level in valid_levels:
			schema = ESEventSchema(
				schema_name="test_schema",
				schema_version="1.0",
				json_schema={"type": "object"},
				event_type="test.event",
				compatibility_level=level,
				tenant_id="test_tenant",
				created_by="test_user"
			)
			assert schema.compatibility_level == level
	
	def test_event_schema_string_representation(self):
		"""Test event schema string representation."""
		schema = ESEventSchema(
			schema_name="user_created",
			schema_version="1.0",
			json_schema={"type": "object"},
			event_type="user.created",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		repr_str = repr(schema)
		assert "ESEventSchema" in repr_str
		assert schema.schema_id in repr_str
		assert schema.schema_name in repr_str

# =============================================================================
# Stream Assignment Model Tests
# =============================================================================

@pytest.mark.unit
class TestESStreamAssignment:
	"""Test ESStreamAssignment model."""
	
	def test_stream_assignment_creation_with_defaults(self):
		"""Test creating a stream assignment with default values."""
		assignment = ESStreamAssignment(
			event_id="evt_test_123",
			stream_id="str_test_456",
			assignment_type="AUTOMATIC",
			tenant_id="test_tenant",
			assigned_by="test_user"
		)
		
		assert assignment.assignment_id.startswith('asg_')
		assert assignment.priority_level == "NORMAL"
		assert assignment.is_active == True
		assert assignment.assignment_rules == {}
	
	def test_stream_assignment_priority_levels(self):
		"""Test valid priority levels."""
		valid_priorities = ["LOW", "NORMAL", "HIGH", "CRITICAL"]
		
		for priority in valid_priorities:
			assignment = ESStreamAssignment(
				event_id="evt_test_123",
				stream_id="str_test_456",
				assignment_type="MANUAL",
				priority_level=priority,
				tenant_id="test_tenant",
				assigned_by="test_user"
			)
			assert assignment.priority_level == priority
	
	def test_stream_assignment_string_representation(self):
		"""Test stream assignment string representation."""
		assignment = ESStreamAssignment(
			event_id="evt_test_123",
			stream_id="str_test_456",
			assignment_type="AUTOMATIC",
			tenant_id="test_tenant",
			assigned_by="test_user"
		)
		
		repr_str = repr(assignment)
		assert "ESStreamAssignment" in repr_str
		assert assignment.assignment_id in repr_str

# =============================================================================
# Event Processing History Model Tests
# =============================================================================

@pytest.mark.unit
class TestESEventProcessingHistory:
	"""Test ESEventProcessingHistory model."""
	
	def test_processing_history_creation_with_defaults(self):
		"""Test creating processing history with default values."""
		history = ESEventProcessingHistory(
			event_id="evt_test_123",
			processor_id="proc_test_456",
			processing_stage="VALIDATION",
			status="COMPLETED",
			tenant_id="test_tenant",
			processed_by="test_processor"
		)
		
		assert history.history_id.startswith('hist_')
		assert history.retry_count == 0
		assert history.input_data == {}
		assert history.output_data == {}
		assert history.error_details is None
	
	def test_processing_history_status_values(self):
		"""Test valid processing status values."""
		valid_statuses = ["PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "RETRY"]
		
		for status in valid_statuses:
			history = ESEventProcessingHistory(
				event_id="evt_test_123",
				processor_id="proc_test_456",
				processing_stage="VALIDATION",
				status=status,
				tenant_id="test_tenant",
				processed_by="test_processor"
			)
			assert history.status == status
	
	def test_processing_history_timing(self):
		"""Test processing timing calculations."""
		start_time = datetime.now(timezone.utc)
		history = ESEventProcessingHistory(
			event_id="evt_test_123",
			processor_id="proc_test_456",
			processing_stage="TRANSFORMATION",
			status="IN_PROGRESS",
			started_at=start_time,
			tenant_id="test_tenant",
			processed_by="test_processor"
		)
		
		assert history.started_at == start_time
		assert history.completed_at is None
		assert history.processing_duration_ms is None

# =============================================================================
# Stream Processor Model Tests
# =============================================================================

@pytest.mark.unit
class TestESStreamProcessor:
	"""Test ESStreamProcessor model."""
	
	def test_stream_processor_creation_with_defaults(self):
		"""Test creating a stream processor with default values."""
		processor = ESStreamProcessor(
			processor_name="user_validation_filter",
			processor_type=ProcessorType.FILTER.value,
			source_stream_id="str_source_123",
			processing_logic={"filter": "user_id IS NOT NULL"},
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		assert processor.processor_id.startswith('proc_')
		assert processor.parallelism == 1
		assert processor.status == "STOPPED"
		assert processor.error_handling_strategy == "RETRY"
		assert processor.state_store_config == {}
		assert processor.window_config == {}
	
	def test_stream_processor_type_validation(self):
		"""Test processor type validation."""
		for processor_type in ProcessorType:
			processor = ESStreamProcessor(
				processor_name=f"test_{processor_type.value}_processor",
				processor_type=processor_type.value,
				source_stream_id="str_source_123",
				processing_logic={"type": processor_type.value},
				tenant_id="test_tenant",
				created_by="test_user"
			)
			assert processor.processor_type == processor_type.value
	
	def test_stream_processor_parallelism_validation(self):
		"""Test parallelism validation."""
		with pytest.raises(ValueError, match="Parallelism must be between 1 and 100"):
			processor = ESStreamProcessor(
				processor_name="test_processor",
				processor_type=ProcessorType.FILTER.value,
				source_stream_id="str_source_123",
				processing_logic={"filter": "true"},
				parallelism=0,  # Invalid
				tenant_id="test_tenant",
				created_by="test_user"
			)
	
	def test_stream_processor_string_representation(self):
		"""Test stream processor string representation."""
		processor = ESStreamProcessor(
			processor_name="user_validation_filter",
			processor_type=ProcessorType.FILTER.value,
			source_stream_id="str_source_123",
			processing_logic={"filter": "user_id IS NOT NULL"},
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		repr_str = repr(processor)
		assert "ESStreamProcessor" in repr_str
		assert processor.processor_id in repr_str
		assert processor.processor_name in repr_str