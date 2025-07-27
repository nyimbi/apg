"""
APG Event Streaming Bus - Enterprise Features Integration Tests

Integration tests for enterprise event streaming features including event sourcing,
stream processing, enhanced schema registry, and consumer management.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone, timedelta
from uuid_extensions import uuid7str

from ...service import (
	EventStreamingService, EventPublishingService, EventSourcingService,
	StreamManagementService, ConsumerManagementService, SchemaRegistryService
)
from ...models import (
	ESEvent, ESStream, ESEventSchema, ESStreamProcessor, ESEventProcessingHistory,
	EventConfig, EventPriority, ProcessorType, CompressionType, SerializationFormat
)

# =============================================================================
# Event Sourcing Integration Tests
# =============================================================================

@pytest.mark.integration
class TestEventSourcingIntegration:
	"""Test event sourcing functionality end-to-end."""
	
	@pytest.fixture
	async def sourcing_service(self, test_database_session, test_redis_client):
		"""Create event sourcing service with test dependencies."""
		service = EventSourcingService()
		service.db_session = test_database_session
		service.redis_client = test_redis_client
		return service
	
	@pytest.mark.asyncio
	async def test_complete_event_sourcing_workflow(self, sourcing_service, test_tenant_id, test_user_id):
		"""Test complete event sourcing workflow from event appending to aggregate reconstruction."""
		aggregate_id = "user_123"
		aggregate_type = "User"
		
		# Step 1: Append initial event
		create_event = {
			"action": "create",
			"user_name": "john.doe",
			"email": "john@example.com",
			"status": "active"
		}
		
		event_id_1 = await sourcing_service.append_event(
			aggregate_id=aggregate_id,
			aggregate_type=aggregate_type,
			event_data=create_event,
			event_type="user.created",
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		assert event_id_1.startswith("evt_")
		
		# Step 2: Append update event with expected version
		update_event = {
			"action": "update",
			"email": "john.doe@example.com",
			"last_login": datetime.now(timezone.utc).isoformat()
		}
		
		event_id_2 = await sourcing_service.append_event(
			aggregate_id=aggregate_id,
			aggregate_type=aggregate_type,
			event_data=update_event,
			event_type="user.updated",
			expected_version=1,  # Expect version 1 after first event
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		assert event_id_2.startswith("evt_")
		assert event_id_2 != event_id_1
		
		# Step 3: Reconstruct aggregate state
		aggregate_state = await sourcing_service.reconstruct_aggregate(
			aggregate_id=aggregate_id,
			aggregate_type=aggregate_type,
			tenant_id=test_tenant_id
		)
		
		assert aggregate_state["data"]["user_name"] == "john.doe"
		assert aggregate_state["data"]["email"] == "john.doe@example.com"  # Updated email
		assert aggregate_state["version"] == 2
		assert "last_login" in aggregate_state["data"]
		
		# Step 4: Test optimistic concurrency control
		conflicting_event = {
			"action": "update",
			"status": "inactive"
		}
		
		with pytest.raises(ValueError, match="Concurrency conflict"):
			await sourcing_service.append_event(
				aggregate_id=aggregate_id,
				aggregate_type=aggregate_type,
				event_data=conflicting_event,
				event_type="user.deactivated",
				expected_version=1,  # Wrong version - should be 2
				tenant_id=test_tenant_id,
				user_id=test_user_id
			)
	
	@pytest.mark.asyncio
	async def test_aggregate_snapshots(self, sourcing_service, test_tenant_id, test_user_id):
		"""Test aggregate snapshot creation and loading."""
		aggregate_id = "order_456"
		aggregate_type = "Order"
		
		# Create multiple events to trigger snapshot
		events = [
			{"action": "create", "order_total": 100.0, "status": "pending"},
			{"action": "add_item", "item_id": "item_1", "quantity": 2, "price": 25.0},
			{"action": "add_item", "item_id": "item_2", "quantity": 1, "price": 50.0},
			{"action": "update_status", "status": "confirmed"},
			{"action": "apply_discount", "discount_percent": 10}
		]
		
		# Append all events
		for i, event_data in enumerate(events):
			await sourcing_service.append_event(
				aggregate_id=aggregate_id,
				aggregate_type=aggregate_type,
				event_data=event_data,
				event_type=f"order.{event_data['action']}",
				expected_version=i,
				tenant_id=test_tenant_id,
				user_id=test_user_id
			)
		
		# Test reconstruction with snapshots enabled
		aggregate_state = await sourcing_service.reconstruct_aggregate(
			aggregate_id=aggregate_id,
			aggregate_type=aggregate_type,
			update_snapshots=True,
			tenant_id=test_tenant_id
		)
		
		assert aggregate_state["version"] == 5
		assert aggregate_state["data"]["status"] == "confirmed"
		assert len(aggregate_state["data"]["items"]) == 2
		
		# Verify snapshot was created
		snapshots = await sourcing_service.get_aggregate_snapshots(
			aggregate_id=aggregate_id,
			aggregate_type=aggregate_type,
			tenant_id=test_tenant_id
		)
		
		assert len(snapshots) > 0
		assert snapshots[0]["version"] == 5

# =============================================================================
# Stream Processing Integration Tests
# =============================================================================

@pytest.mark.integration
class TestStreamProcessingIntegration:
	"""Test stream processing functionality end-to-end."""
	
	@pytest.fixture
	async def stream_service(self, test_database_session, test_kafka_admin, test_redis_client):
		"""Create stream management service with test dependencies."""
		service = StreamManagementService()
		service.db_session = test_database_session
		service.kafka_admin = test_kafka_admin
		service.redis_client = test_redis_client
		return service
	
	@pytest.fixture
	async def publishing_service(self, test_database_session, test_kafka_producer, test_redis_client):
		"""Create event publishing service for test data."""
		service = EventPublishingService()
		service.db_session = test_database_session
		service.kafka_producer = test_kafka_producer
		service.redis_client = test_redis_client
		return service
	
	@pytest.mark.asyncio
	async def test_filter_processor_workflow(self, stream_service, publishing_service, test_tenant_id, test_user_id):
		"""Test complete filter processor workflow."""
		
		# Step 1: Create source and target streams
		source_stream_config = {
			"stream_name": "user_events",
			"topic_name": "user-events",
			"source_capability": "user_management",
			"partitions": 3
		}
		
		source_stream_id = await stream_service.create_stream(
			stream_config=source_stream_config,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		target_stream_config = {
			"stream_name": "verified_user_events",
			"topic_name": "verified-user-events",
			"source_capability": "user_management",
			"partitions": 3
		}
		
		target_stream_id = await stream_service.create_stream(
			stream_config=target_stream_config,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		# Step 2: Create filter processor
		processor_config = {
			"processor_name": "verified_user_filter",
			"processor_type": ProcessorType.FILTER.value,
			"source_stream_id": source_stream_id,
			"target_stream_id": target_stream_id,
			"processing_logic": {
				"filter_expression": "payload.user_status == 'verified'",
				"additional_checks": ["payload.email_verified == true"]
			},
			"parallelism": 2,
			"error_handling_strategy": "RETRY"
		}
		
		processor_id = await stream_service.create_stream_processor(
			processor_config=processor_config,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		assert processor_id.startswith("proc_")
		
		# Step 3: Start processor
		success = await stream_service.start_stream_processor(
			processor_id=processor_id,
			tenant_id=test_tenant_id
		)
		
		assert success == True
		
		# Step 4: Publish test events
		verified_event = EventConfig(
			event_type="user.registered",
			source_capability="user_management",
			aggregate_id="user_verified_123",
			aggregate_type="User",
			priority=EventPriority.NORMAL.value
		)
		
		verified_payload = {
			"user_status": "verified",
			"email_verified": True,
			"user_name": "verified.user"
		}
		
		unverified_event = EventConfig(
			event_type="user.registered",
			source_capability="user_management",
			aggregate_id="user_unverified_456",
			aggregate_type="User",
			priority=EventPriority.NORMAL.value
		)
		
		unverified_payload = {
			"user_status": "pending",
			"email_verified": False,
			"user_name": "pending.user"
		}
		
		# Publish events to source stream
		verified_event_id = await publishing_service.publish_event(
			event_config=verified_event,
			payload=verified_payload,
			stream_id=source_stream_id,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		unverified_event_id = await publishing_service.publish_event(
			event_config=unverified_event,
			payload=unverified_payload,
			stream_id=source_stream_id,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		# Step 5: Wait for processing and verify results
		await asyncio.sleep(2)  # Allow processing time
		
		# Get processor metrics
		metrics = await stream_service.get_processor_metrics(
			processor_id=processor_id,
			tenant_id=test_tenant_id
		)
		
		assert metrics["events_processed"] >= 2
		assert metrics["events_filtered"] >= 1  # At least the unverified event should be filtered
		assert metrics["events_passed"] >= 1   # At least the verified event should pass
		
		# Step 6: Stop processor
		success = await stream_service.stop_stream_processor(
			processor_id=processor_id,
			tenant_id=test_tenant_id
		)
		
		assert success == True
	
	@pytest.mark.asyncio
	async def test_aggregation_processor_workflow(self, stream_service, publishing_service, test_tenant_id, test_user_id):
		"""Test aggregation processor with windowing."""
		
		# Create source stream for order events
		source_stream_config = {
			"stream_name": "order_events",
			"topic_name": "order-events",
			"source_capability": "order_management",
			"partitions": 2
		}
		
		source_stream_id = await stream_service.create_stream(
			stream_config=source_stream_config,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		# Create target stream for aggregated metrics
		target_stream_config = {
			"stream_name": "order_metrics",
			"topic_name": "order-metrics",
			"source_capability": "analytics",
			"partitions": 1
		}
		
		target_stream_id = await stream_service.create_stream(
			stream_config=target_stream_config,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		# Create aggregation processor
		processor_config = {
			"processor_name": "order_aggregator",
			"processor_type": ProcessorType.AGGREGATE.value,
			"source_stream_id": source_stream_id,
			"target_stream_id": target_stream_id,
			"processing_logic": {
				"aggregation_function": "sum",
				"aggregation_field": "order_total",
				"group_by": ["customer_segment"]
			},
			"window_config": {
				"window_type": "tumbling",
				"window_size_ms": 60000,  # 1 minute windows
				"grace_period_ms": 5000
			},
			"state_store_config": {
				"store_type": "rocksdb",
				"changelog_topic": "order-aggregator-changelog"
			}
		}
		
		processor_id = await stream_service.create_stream_processor(
			processor_config=processor_config,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		# Start aggregation processor
		success = await stream_service.start_stream_processor(
			processor_id=processor_id,
			tenant_id=test_tenant_id
		)
		
		assert success == True
		
		# Publish multiple order events
		order_events = [
			{"order_total": 100.0, "customer_segment": "premium"},
			{"order_total": 50.0, "customer_segment": "standard"},
			{"order_total": 200.0, "customer_segment": "premium"},
			{"order_total": 75.0, "customer_segment": "standard"},
			{"order_total": 150.0, "customer_segment": "premium"}
		]
		
		for i, order_data in enumerate(order_events):
			event_config = EventConfig(
				event_type="order.created",
				source_capability="order_management",
				aggregate_id=f"order_{i}",
				aggregate_type="Order"
			)
			
			await publishing_service.publish_event(
				event_config=event_config,
				payload=order_data,
				stream_id=source_stream_id,
				tenant_id=test_tenant_id,
				user_id=test_user_id
			)
		
		# Wait for aggregation processing
		await asyncio.sleep(3)
		
		# Verify processing metrics
		metrics = await stream_service.get_processor_metrics(
			processor_id=processor_id,
			tenant_id=test_tenant_id
		)
		
		assert metrics["events_processed"] >= 5
		assert metrics["aggregations_computed"] >= 2  # Premium and standard segments

# =============================================================================
# Enhanced Schema Registry Integration Tests
# =============================================================================

@pytest.mark.integration
class TestEnhancedSchemaRegistryIntegration:
	"""Test enhanced schema registry functionality end-to-end."""
	
	@pytest.fixture
	async def schema_service(self, test_database_session, test_redis_client):
		"""Create schema registry service with test dependencies."""
		service = SchemaRegistryService()
		service.db_session = test_database_session
		service.redis_client = test_redis_client
		return service
	
	@pytest.mark.asyncio
	async def test_schema_evolution_workflow(self, schema_service, test_tenant_id, test_user_id):
		"""Test complete schema evolution workflow."""
		
		# Step 1: Register initial schema version
		initial_schema = {
			"schema_name": "user_profile",
			"schema_version": "1.0",
			"json_schema": {
				"type": "object",
				"properties": {
					"user_id": {"type": "string"},
					"user_name": {"type": "string"},
					"email": {"type": "string", "format": "email"}
				},
				"required": ["user_id", "user_name", "email"]
			},
			"event_type": "user.profile_updated",
			"compatibility_level": "BACKWARD",
			"evolution_strategy": "COMPATIBLE"
		}
		
		schema_id_v1 = await schema_service.register_enhanced_schema(
			schema_config=initial_schema,
			tenant_id=test_tenant_id,
			created_by=test_user_id
		)
		
		assert schema_id_v1.startswith("sch_")
		
		# Step 2: Validate event against initial schema
		valid_event_v1 = {
			"user_id": "user_123",
			"user_name": "john.doe",
			"email": "john@example.com"
		}
		
		validation_result = await schema_service.validate_event(
			schema_id=schema_id_v1,
			event_data=valid_event_v1,
			tenant_id=test_tenant_id
		)
		
		assert validation_result["is_valid"] == True
		assert len(validation_result["validation_errors"]) == 0
		
		# Step 3: Register evolved schema (backward compatible)
		evolved_schema = {
			"schema_name": "user_profile",
			"schema_version": "1.1",
			"json_schema": {
				"type": "object",
				"properties": {
					"user_id": {"type": "string"},
					"user_name": {"type": "string"},
					"email": {"type": "string", "format": "email"},
					"phone": {"type": "string"},  # New optional field
					"last_login": {"type": "string", "format": "date-time"}  # New optional field
				},
				"required": ["user_id", "user_name", "email"]  # Same required fields
			},
			"event_type": "user.profile_updated",
			"compatibility_level": "BACKWARD",
			"evolution_strategy": "COMPATIBLE"
		}
		
		schema_id_v1_1 = await schema_service.register_enhanced_schema(
			schema_config=evolved_schema,
			tenant_id=test_tenant_id,
			created_by=test_user_id
		)
		
		assert schema_id_v1_1.startswith("sch_")
		assert schema_id_v1_1 != schema_id_v1
		
		# Step 4: Validate old event against new schema (backward compatibility)
		validation_result = await schema_service.validate_event(
			schema_id=schema_id_v1_1,
			event_data=valid_event_v1,  # Old event format
			tenant_id=test_tenant_id
		)
		
		assert validation_result["is_valid"] == True
		
		# Step 5: Validate new event against new schema
		valid_event_v1_1 = {
			"user_id": "user_456",
			"user_name": "jane.doe",
			"email": "jane@example.com",
			"phone": "+1-555-0123",
			"last_login": "2025-01-15T10:30:00Z"
		}
		
		validation_result = await schema_service.validate_event(
			schema_id=schema_id_v1_1,
			event_data=valid_event_v1_1,
			tenant_id=test_tenant_id
		)
		
		assert validation_result["is_valid"] == True
		
		# Step 6: Get schema evolution history
		evolution_history = await schema_service.get_schema_evolution(
			event_type="user.profile_updated",
			tenant_id=test_tenant_id
		)
		
		assert len(evolution_history) == 2
		assert evolution_history[0]["schema_version"] == "1.0"
		assert evolution_history[1]["schema_version"] == "1.1"
		
		# Step 7: Test incompatible schema registration
		incompatible_schema = {
			"schema_name": "user_profile",
			"schema_version": "2.0",
			"json_schema": {
				"type": "object",
				"properties": {
					"user_id": {"type": "string"},
					"full_name": {"type": "string"},  # Changed field name
					"contact_email": {"type": "string", "format": "email"}  # Changed field name
				},
				"required": ["user_id", "full_name", "contact_email"]
			},
			"event_type": "user.profile_updated",
			"compatibility_level": "BACKWARD",
			"evolution_strategy": "BREAKING_CHANGE"  # Explicitly allow breaking change
		}
		
		schema_id_v2 = await schema_service.register_enhanced_schema(
			schema_config=incompatible_schema,
			tenant_id=test_tenant_id,
			created_by=test_user_id
		)
		
		assert schema_id_v2.startswith("sch_")
		
		# Step 8: Verify old event fails against breaking change schema
		validation_result = await schema_service.validate_event(
			schema_id=schema_id_v2,
			event_data=valid_event_v1,  # Old format
			tenant_id=test_tenant_id
		)
		
		assert validation_result["is_valid"] == False
		assert len(validation_result["validation_errors"]) > 0
	
	@pytest.mark.asyncio
	async def test_schema_validation_rules(self, schema_service, test_tenant_id, test_user_id):
		"""Test custom validation rules functionality."""
		
		# Register schema with custom validation rules
		schema_with_rules = {
			"schema_name": "order_event",
			"schema_version": "1.0",
			"json_schema": {
				"type": "object",
				"properties": {
					"order_id": {"type": "string"},
					"customer_id": {"type": "string"},
					"order_total": {"type": "number", "minimum": 0},
					"currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]}
				},
				"required": ["order_id", "customer_id", "order_total", "currency"]
			},
			"event_type": "order.created",
			"validation_rules": {
				"business_rules": [
					{
						"name": "minimum_order_value",
						"description": "Order must be at least $10",
						"rule": "order_total >= 10.0"
					},
					{
						"name": "valid_customer",
						"description": "Customer ID must be valid format",
						"rule": "customer_id.startswith('cust_')"
					}
				],
				"cross_field_validation": [
					{
						"name": "currency_amount_consistency",
						"description": "USD orders must be > $5, EUR orders > €5",
						"rule": "(currency == 'USD' and order_total >= 5.0) or (currency != 'USD')"
					}
				]
			}
		}
		
		schema_id = await schema_service.register_enhanced_schema(
			schema_config=schema_with_rules,
			tenant_id=test_tenant_id,
			created_by=test_user_id
		)
		
		# Test valid event
		valid_order = {
			"order_id": "ord_123",
			"customer_id": "cust_456",
			"order_total": 25.99,
			"currency": "USD"
		}
		
		validation_result = await schema_service.validate_event(
			schema_id=schema_id,
			event_data=valid_order,
			tenant_id=test_tenant_id
		)
		
		assert validation_result["is_valid"] == True
		
		# Test invalid event (business rule violation)
		invalid_order = {
			"order_id": "ord_124",
			"customer_id": "invalid_customer_id",  # Doesn't start with 'cust_'
			"order_total": 5.0,  # Below minimum
			"currency": "USD"
		}
		
		validation_result = await schema_service.validate_event(
			schema_id=schema_id,
			event_data=invalid_order,
			tenant_id=test_tenant_id
		)
		
		assert validation_result["is_valid"] == False
		assert any("minimum_order_value" in error for error in validation_result["validation_errors"])
		assert any("valid_customer" in error for error in validation_result["validation_errors"])

# =============================================================================
# End-to-End Enterprise Workflow Tests
# =============================================================================

@pytest.mark.integration
class TestEnterpriseWorkflowIntegration:
	"""Test complete enterprise workflow integrating all features."""
	
	@pytest.mark.asyncio
	async def test_complete_enterprise_workflow(self, test_database_session, test_kafka_cluster, test_redis_client, test_tenant_id, test_user_id):
		"""Test complete enterprise event streaming workflow."""
		
		# Initialize all services
		publishing_service = EventPublishingService()
		sourcing_service = EventSourcingService()
		stream_service = StreamManagementService()
		schema_service = SchemaRegistryService()
		
		# Configure services
		for service in [publishing_service, sourcing_service, stream_service, schema_service]:
			service.db_session = test_database_session
			service.redis_client = test_redis_client
		
		# Step 1: Register schemas for user events
		user_schema = {
			"schema_name": "user_lifecycle",
			"schema_version": "1.0",
			"json_schema": {
				"type": "object",
				"properties": {
					"user_id": {"type": "string"},
					"action": {"type": "string", "enum": ["created", "updated", "activated", "deactivated"]},
					"user_data": {"type": "object"},
					"timestamp": {"type": "string", "format": "date-time"}
				},
				"required": ["user_id", "action", "timestamp"]
			},
			"event_type": "user.lifecycle",
			"compatibility_level": "BACKWARD"
		}
		
		schema_id = await schema_service.register_enhanced_schema(
			schema_config=user_schema,
			tenant_id=test_tenant_id,
			created_by=test_user_id
		)
		
		# Step 2: Create event streams
		user_stream_config = {
			"stream_name": "user_lifecycle_events",
			"topic_name": "user-lifecycle",
			"source_capability": "user_management",
			"partitions": 3,
			"compression_type": CompressionType.SNAPPY.value
		}
		
		stream_id = await stream_service.create_stream(
			stream_config=user_stream_config,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		# Step 3: Set up stream processor for user activation events
		activation_processor_config = {
			"processor_name": "user_activation_processor",
			"processor_type": ProcessorType.FILTER.value,
			"source_stream_id": stream_id,
			"processing_logic": {
				"filter_expression": "payload.action == 'activated'"
			}
		}
		
		processor_id = await stream_service.create_stream_processor(
			processor_config=activation_processor_config,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		await stream_service.start_stream_processor(
			processor_id=processor_id,
			tenant_id=test_tenant_id
		)
		
		# Step 4: Execute user lifecycle using event sourcing
		user_id = "user_enterprise_test"
		
		# User creation
		creation_event = {
			"action": "created",
			"user_data": {
				"name": "Enterprise User",
				"email": "enterprise@example.com"
			},
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
		
		await sourcing_service.append_event(
			aggregate_id=user_id,
			aggregate_type="User",
			event_data=creation_event,
			event_type="user.lifecycle",
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		# User activation
		activation_event = {
			"action": "activated",
			"user_data": {
				"activation_method": "email_verification"
			},
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
		
		await sourcing_service.append_event(
			aggregate_id=user_id,
			aggregate_type="User",
			event_data=activation_event,
			event_type="user.lifecycle",
			expected_version=1,
			tenant_id=test_tenant_id,
			user_id=test_user_id
		)
		
		# Step 5: Publish events to stream for processing
		for event_data in [creation_event, activation_event]:
			event_config = EventConfig(
				event_type="user.lifecycle",
				source_capability="user_management",
				aggregate_id=user_id,
				aggregate_type="User",
				priority=EventPriority.NORMAL.value,
				schema_id=schema_id
			)
			
			await publishing_service.publish_event(
				event_config=event_config,
				payload=event_data,
				stream_id=stream_id,
				tenant_id=test_tenant_id,
				user_id=test_user_id
			)
		
		# Step 6: Wait for processing and verify results
		await asyncio.sleep(2)
		
		# Verify aggregate state
		aggregate_state = await sourcing_service.reconstruct_aggregate(
			aggregate_id=user_id,
			aggregate_type="User",
			tenant_id=test_tenant_id
		)
		
		assert aggregate_state["version"] == 2
		assert aggregate_state["data"]["name"] == "Enterprise User"
		assert aggregate_state["data"]["activation_method"] == "email_verification"
		
		# Verify stream processing metrics
		metrics = await stream_service.get_processor_metrics(
			processor_id=processor_id,
			tenant_id=test_tenant_id
		)
		
		assert metrics["events_processed"] >= 2
		assert metrics["events_passed"] >= 1  # Activation event should pass filter
		
		# Step 7: Clean up
		await stream_service.stop_stream_processor(
			processor_id=processor_id,
			tenant_id=test_tenant_id
		)