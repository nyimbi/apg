#!/usr/bin/env python3
"""
APG Event Streaming Bus - Basic Usage Examples

Complete examples demonstrating how to use the Event Streaming Bus
for common event-driven scenarios.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Any

from event_streaming_bus import (
    EventStreamingService, EventPublishingService, EventConsumptionService,
    StreamProcessingService, APGEventStreamingIntegration,
    EventConfig, StreamConfig, SubscriptionConfig, SchemaConfig,
    EventType, DeliveryMode
)

# =============================================================================
# Basic Event Publishing and Consumption
# =============================================================================

async def basic_event_flow_example():
    """Demonstrate basic event publishing and consumption flow."""
    
    print("=== Basic Event Flow Example ===")
    
    # Initialize services
    streaming_service = EventStreamingService()
    publishing_service = EventPublishingService()
    consumption_service = EventConsumptionService()
    
    # 1. Create a stream for user events
    print("\n1. Creating user events stream...")
    
    stream_config = StreamConfig(
        stream_name="user_events",
        stream_description="User lifecycle events",
        topic_name="apg-user-events",
        partitions=6,
        replication_factor=3,
        retention_time_ms=604800000,  # 7 days
        source_capability="user_management",
        event_category=EventType.DOMAIN_EVENT
    )
    
    stream_id = await streaming_service.create_stream(
        config=stream_config,
        tenant_id="demo_tenant",
        created_by="demo_user"
    )
    
    print(f"   ✓ Stream created: {stream_id}")
    
    # 2. Create a subscription to consume events
    print("\n2. Creating event subscription...")
    
    subscription_config = SubscriptionConfig(
        subscription_name="user_event_processor",
        subscription_description="Process all user events",
        stream_id=stream_id,
        consumer_group_id="user_processors",
        consumer_name="processor_1",
        event_type_patterns=["user.*"],
        delivery_mode=DeliveryMode.AT_LEAST_ONCE,
        batch_size=50,
        max_wait_time_ms=1000
    )
    
    subscription_id = await consumption_service.create_subscription(
        config=subscription_config,
        tenant_id="demo_tenant",
        created_by="demo_user"
    )
    
    print(f"   ✓ Subscription created: {subscription_id}")
    
    # 3. Publish some events
    print("\n3. Publishing user events...")
    
    users = [
        {"user_name": "john.doe", "email": "john.doe@company.com", "department": "Engineering"},
        {"user_name": "jane.smith", "email": "jane.smith@company.com", "department": "Marketing"},
        {"user_name": "bob.wilson", "email": "bob.wilson@company.com", "department": "Sales"}
    ]
    
    event_ids = []
    for i, user in enumerate(users, 1):
        event_config = EventConfig(
            event_type="user.created",
            source_capability="user_management",
            aggregate_id=f"user_{i}",
            aggregate_type="User",
            sequence_number=1
        )
        
        event_id = await publishing_service.publish_event(
            event_config=event_config,
            payload=user,
            stream_id=stream_id,
            tenant_id="demo_tenant",
            user_id="demo_user"
        )
        
        event_ids.append(event_id)
        print(f"   ✓ Published event: {event_id} for {user['user_name']}")
    
    # 4. Simulate event consumption (in real scenario, this would be a background process)
    print("\n4. Processing events...")
    
    # Get events to process
    events, total_count = await streaming_service.get_stream_events(
        stream_id=stream_id,
        tenant_id="demo_tenant",
        limit=10
    )
    
    print(f"   ✓ Found {total_count} events to process")
    
    # Process each event
    for event in events:
        print(f"   → Processing event {event.event_id}: {event.event_type}")
        print(f"     User: {event.payload.get('user_name')}")
        print(f"     Email: {event.payload.get('email')}")
        print(f"     Department: {event.payload.get('department')}")
    
    print("\n✓ Basic event flow completed successfully!")

# =============================================================================
# Event Batching Example
# =============================================================================

async def batch_processing_example():
    """Demonstrate high-throughput batch event processing."""
    
    print("\n=== Batch Processing Example ===")
    
    publishing_service = EventPublishingService()
    
    # Create batch of order events
    print("\n1. Creating batch of order events...")
    
    batch_events = []
    for i in range(100):
        event_config = EventConfig(
            event_type="order.created",
            source_capability="order_management",
            aggregate_id=f"order_{i:04d}",
            aggregate_type="Order"
        )
        
        payload = {
            "order_id": f"order_{i:04d}",
            "customer_id": f"customer_{i % 20:03d}",
            "amount": round(50.0 + (i * 1.5), 2),
            "items": [
                {"product_id": f"prod_{(i % 5) + 1}", "quantity": i % 3 + 1}
            ]
        }
        
        batch_events.append((event_config, payload))
    
    # Publish batch
    print("2. Publishing batch (100 events)...")
    start_time = datetime.now()
    
    event_ids = await publishing_service.publish_event_batch(
        events=batch_events,
        stream_id="order_events",
        tenant_id="demo_tenant",
        user_id="demo_user",
        batch_options={
            "timeout_ms": 10000,
            "compression": True
        }
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"   ✓ Published {len(event_ids)} events in {duration:.2f} seconds")
    print(f"   ✓ Throughput: {len(event_ids) / duration:.0f} events/second")

# =============================================================================
# Event Correlation and Workflows
# =============================================================================

async def event_correlation_example():
    """Demonstrate event correlation and workflow patterns."""
    
    print("\n=== Event Correlation Example ===")
    
    publishing_service = EventPublishingService()
    
    # Simulate a multi-step order process with correlated events
    correlation_id = "cor_order_workflow_001"
    order_id = "order_5001"
    
    print(f"\n1. Simulating order workflow for {order_id}...")
    
    # Step 1: Order created
    order_created_config = EventConfig(
        event_type="order.created",
        source_capability="order_management",
        aggregate_id=order_id,
        aggregate_type="Order",
        correlation_id=correlation_id,
        sequence_number=1
    )
    
    event1_id = await publishing_service.publish_event(
        event_config=order_created_config,
        payload={
            "order_id": order_id,
            "customer_id": "customer_001",
            "amount": 299.99,
            "status": "created"
        },
        stream_id="order_events",
        tenant_id="demo_tenant",
        user_id="demo_user"
    )
    
    print(f"   ✓ Order created: {event1_id}")
    
    # Step 2: Payment processed (caused by order creation)
    payment_processed_config = EventConfig(
        event_type="payment.processed",
        source_capability="payment_service",
        aggregate_id=f"payment_{order_id}",
        aggregate_type="Payment",
        correlation_id=correlation_id,
        causation_id=event1_id,
        sequence_number=1
    )
    
    event2_id = await publishing_service.publish_event(
        event_config=payment_processed_config,
        payload={
            "payment_id": f"payment_{order_id}",
            "order_id": order_id,
            "amount": 299.99,
            "payment_method": "credit_card",
            "status": "processed"
        },
        stream_id="payment_events",
        tenant_id="demo_tenant",
        user_id="demo_user"
    )
    
    print(f"   ✓ Payment processed: {event2_id}")
    
    # Step 3: Inventory reserved (caused by payment)
    inventory_reserved_config = EventConfig(
        event_type="inventory.reserved",
        source_capability="inventory_service",
        aggregate_id=f"inventory_{order_id}",
        aggregate_type="InventoryReservation",
        correlation_id=correlation_id,
        causation_id=event2_id,
        sequence_number=1
    )
    
    event3_id = await publishing_service.publish_event(
        event_config=inventory_reserved_config,
        payload={
            "reservation_id": f"inventory_{order_id}",
            "order_id": order_id,
            "items": [
                {"product_id": "prod_001", "quantity": 2, "location": "warehouse_A"}
            ],
            "status": "reserved"
        },
        stream_id="inventory_events",
        tenant_id="demo_tenant",
        user_id="demo_user"
    )
    
    print(f"   ✓ Inventory reserved: {event3_id}")
    
    # Step 4: Order confirmed (caused by inventory reservation)
    order_confirmed_config = EventConfig(
        event_type="order.confirmed",
        source_capability="order_management",
        aggregate_id=order_id,
        aggregate_type="Order",
        correlation_id=correlation_id,
        causation_id=event3_id,
        sequence_number=2
    )
    
    event4_id = await publishing_service.publish_event(
        event_config=order_confirmed_config,
        payload={
            "order_id": order_id,
            "status": "confirmed",
            "estimated_delivery": "2025-01-28T12:00:00.000Z"
        },
        stream_id="order_events",
        tenant_id="demo_tenant",
        user_id="demo_user"
    )
    
    print(f"   ✓ Order confirmed: {event4_id}")
    
    print(f"\n✓ Complete order workflow tracked with correlation_id: {correlation_id}")

# =============================================================================
# Schema Validation Example
# =============================================================================

async def schema_validation_example():
    """Demonstrate event schema validation."""
    
    print("\n=== Schema Validation Example ===")
    
    from event_streaming_bus import SchemaRegistryService
    
    schema_service = SchemaRegistryService()
    publishing_service = EventPublishingService()
    
    # 1. Register a schema for user events
    print("\n1. Registering user event schema...")
    
    schema_config = SchemaConfig(
        schema_name="user_created",
        schema_version="1.0",
        schema_definition={
            "type": "object",
            "properties": {
                "user_name": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9._-]+$",
                    "minLength": 3,
                    "maxLength": 50
                },
                "email": {
                    "type": "string",
                    "format": "email"
                },
                "department": {
                    "type": "string",
                    "enum": ["Engineering", "Marketing", "Sales", "Support", "HR"]
                },
                "start_date": {
                    "type": "string",
                    "format": "date"
                }
            },
            "required": ["user_name", "email", "department"],
            "additionalProperties": false
        },
        event_type="user.created",
        compatibility_level="backward"
    )
    
    schema_id = await schema_service.register_schema(
        config=schema_config,
        tenant_id="demo_tenant",
        created_by="demo_user"
    )
    
    print(f"   ✓ Schema registered: {schema_id}")
    
    # 2. Publish valid event
    print("\n2. Publishing valid event...")
    
    valid_event_config = EventConfig(
        event_type="user.created",
        source_capability="user_management",
        aggregate_id="user_valid",
        aggregate_type="User",
        schema_id=schema_id
    )
    
    valid_payload = {
        "user_name": "alice.johnson",
        "email": "alice.johnson@company.com",
        "department": "Engineering",
        "start_date": "2025-01-26"
    }
    
    try:
        event_id = await publishing_service.publish_event(
            event_config=valid_event_config,
            payload=valid_payload,
            stream_id="user_events",
            tenant_id="demo_tenant",
            user_id="demo_user"
        )
        print(f"   ✓ Valid event published: {event_id}")
    except Exception as e:
        print(f"   ✗ Error publishing valid event: {e}")
    
    # 3. Try to publish invalid event
    print("\n3. Attempting to publish invalid event...")
    
    invalid_payload = {
        "user_name": "a",  # Too short
        "email": "invalid-email",  # Invalid format
        "department": "InvalidDept",  # Not in enum
        "extra_field": "not_allowed"  # Additional property
    }
    
    try:
        event_id = await publishing_service.publish_event(
            event_config=valid_event_config,
            payload=invalid_payload,
            stream_id="user_events",
            tenant_id="demo_tenant",
            user_id="demo_user"
        )
        print(f"   ✗ Invalid event was published (should have failed): {event_id}")
    except Exception as e:
        print(f"   ✓ Invalid event correctly rejected: {e}")

# =============================================================================
# Stream Processing Example
# =============================================================================

async def stream_processing_example():
    """Demonstrate real-time stream processing."""
    
    print("\n=== Stream Processing Example ===")
    
    processing_service = StreamProcessingService()
    publishing_service = EventPublishingService()
    
    # 1. Set up stream aggregation
    print("\n1. Setting up stream aggregation...")
    
    aggregation_config = {
        "window_type": "tumbling",
        "duration_ms": 60000,  # 1 minute windows
        "aggregation_function": "count",
        "group_by": ["event_type", "department"]
    }
    
    window_id = await processing_service.create_aggregation_window(
        stream_id="user_events",
        config=aggregation_config
    )
    
    print(f"   ✓ Aggregation window created: {window_id}")
    
    # 2. Generate events from different departments
    print("\n2. Generating events for aggregation...")
    
    departments = ["Engineering", "Marketing", "Sales"]
    event_types = ["user.created", "user.updated", "user.deactivated"]
    
    for i in range(30):
        dept = departments[i % len(departments)]
        event_type = event_types[i % len(event_types)]
        
        event_config = EventConfig(
            event_type=event_type,
            source_capability="user_management",
            aggregate_id=f"user_{i:03d}",
            aggregate_type="User"
        )
        
        payload = {
            "user_name": f"user_{i:03d}",
            "department": dept,
            "action": event_type.split(".")[1]
        }
        
        await publishing_service.publish_event(
            event_config=event_config,
            payload=payload,
            stream_id="user_events",
            tenant_id="demo_tenant",
            user_id="demo_user"
        )
    
    print(f"   ✓ Generated 30 events across {len(departments)} departments")
    
    # 3. Process aggregation (simulate window completion)
    print("\n3. Processing aggregation window...")
    
    # In a real scenario, this would be triggered automatically
    processed_count = await processing_service.process_stream_events(
        stream_id="user_events",
        processor_config=aggregation_config
    )
    
    print(f"   ✓ Processed {processed_count} events in aggregation window")

# =============================================================================
# Cross-Capability Integration Example
# =============================================================================

async def cross_capability_integration_example():
    """Demonstrate APG cross-capability integration."""
    
    print("\n=== Cross-Capability Integration Example ===")
    
    # Initialize APG integration
    streaming_service = EventStreamingService()
    publishing_service = EventPublishingService()
    consumption_service = EventConsumptionService()
    
    integration = APGEventStreamingIntegration(
        event_streaming_service=streaming_service,
        publishing_service=publishing_service,
        consumption_service=consumption_service
    )
    
    await integration.initialize()
    
    # 1. Register capabilities
    print("\n1. Registering APG capabilities...")
    
    from event_streaming_bus import APGCapabilityInfo
    
    user_mgmt_capability = APGCapabilityInfo(
        capability_id="user_management",
        capability_name="User Management",
        capability_type="domain",
        version="1.0.0",
        endpoints={"api": "/api/v1", "health": "/health"},
        event_patterns=["user.*"],
        dependencies=["event_streaming_bus"]
    )
    
    notification_capability = APGCapabilityInfo(
        capability_id="notification_service",
        capability_name="Notification Service",
        capability_type="service",
        version="1.0.0",
        endpoints={"api": "/api/v1", "webhook": "/webhook"},
        event_patterns=["user.*", "order.*"],
        dependencies=["event_streaming_bus", "user_management"]
    )
    
    await integration.register_capability(user_mgmt_capability)
    await integration.register_capability(notification_capability)
    
    print("   ✓ Registered user_management capability")
    print("   ✓ Registered notification_service capability")
    
    # 2. Set up event routing
    print("\n2. Setting up event routing...")
    
    from event_streaming_bus import EventRoutingRule
    
    routing_rule = EventRoutingRule(
        rule_id="user_to_notification_routing",
        source_pattern="user_management",
        target_capabilities=["notification_service"],
        event_type_patterns=["user.created", "user.updated"],
        priority=100,
        is_active=True
    )
    
    await integration.add_routing_rule(routing_rule)
    
    print("   ✓ Set up routing from user_management to notification_service")
    
    # 3. Create workflow
    print("\n3. Creating cross-capability workflow...")
    
    from event_streaming_bus import CrossCapabilityWorkflow
    
    onboarding_workflow = CrossCapabilityWorkflow(
        workflow_id="user_onboarding_workflow",
        workflow_name="User Onboarding Process",
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
                "step_id": "create_slack_account",
                "capability": "slack_integration",
                "action": "create_user",
                "parameters": {
                    "email": "${trigger_event.payload.email}",
                    "department": "${trigger_event.payload.department}"
                }
            },
            {
                "step_id": "assign_equipment",
                "capability": "asset_management",
                "action": "assign_laptop",
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
    
    await integration.register_workflow(onboarding_workflow)
    
    print("   ✓ Registered user onboarding workflow")
    
    # 4. Trigger workflow with user creation event
    print("\n4. Triggering workflow...")
    
    user_created_config = EventConfig(
        event_type="user.created",
        source_capability="user_management",
        aggregate_id="user_workflow_test",
        aggregate_type="User"
    )
    
    trigger_payload = {
        "user_name": "new.employee",
        "email": "new.employee@company.com",
        "department": "Engineering",
        "start_date": "2025-01-26"
    }
    
    # Publish trigger event
    event_id = await publishing_service.publish_event(
        event_config=user_created_config,
        payload=trigger_payload,
        stream_id="user_events",
        tenant_id="demo_tenant",
        user_id="demo_user"
    )
    
    print(f"   ✓ Published trigger event: {event_id}")
    
    # The workflow would be triggered automatically by the integration
    print("   ✓ User onboarding workflow triggered")
    
    await integration.shutdown()

# =============================================================================
# Main Example Runner
# =============================================================================

async def main():
    """Run all examples."""
    
    print("APG Event Streaming Bus - Usage Examples")
    print("="*50)
    
    try:
        # Run examples
        await basic_event_flow_example()
        await batch_processing_example()
        await event_correlation_example()
        await schema_validation_example()
        await stream_processing_example()
        await cross_capability_integration_example()
        
        print("\n" + "="*50)
        print("✓ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())