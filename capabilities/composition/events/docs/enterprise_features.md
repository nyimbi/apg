# APG Event Streaming Bus - Enterprise Features

This document provides comprehensive documentation for the enterprise features of the APG Event Streaming Bus capability.

## Table of Contents

1. [Event Sourcing & CQRS](#event-sourcing--cqrs)
2. [Stream Processing](#stream-processing)
3. [Enhanced Schema Registry](#enhanced-schema-registry)
4. [Consumer Management](#consumer-management)
5. [Event Priority & Routing](#event-priority--routing)
6. [Processing History & Audit Trails](#processing-history--audit-trails)
7. [Real-time Monitoring & Metrics](#real-time-monitoring--metrics)

## Event Sourcing & CQRS

The Event Streaming Bus provides enterprise-grade event sourcing capabilities with optimistic concurrency control and aggregate reconstruction.

### Features

- **Optimistic Concurrency Control**: Prevent conflicts when multiple processes update the same aggregate
- **Aggregate Reconstruction**: Rebuild aggregate state from event stream
- **Event Snapshots**: Improve performance for large aggregates
- **Version Tracking**: Track aggregate versions for consistency

### API Reference

#### Append Event to Aggregate

```http
POST /api/v1/event-sourcing/append
Content-Type: application/json
Authorization: Bearer {token}

{
    "aggregate_id": "user_123",
    "aggregate_type": "User",
    "event_type": "user.updated",
    "event_data": {
        "email": "john.doe@company.com",
        "last_login": "2025-01-26T10:30:00Z"
    },
    "expected_version": 5,
    "correlation_id": "corr_456",
    "metadata": {
        "source": "user_service",
        "user_agent": "mobile_app_v1.2"
    }
}
```

**Response:**
```json
{
    "event_id": "evt_789",
    "message": "Event appended successfully"
}
```

#### Reconstruct Aggregate State

```http
POST /api/v1/event-sourcing/reconstruct
Content-Type: application/json
Authorization: Bearer {token}

{
    "aggregate_id": "user_123",
    "aggregate_type": "User",
    "update_snapshots": true
}
```

**Response:**
```json
{
    "aggregate_id": "user_123",
    "aggregate_type": "User",
    "current_version": 6,
    "state": {
        "user_name": "john.doe",
        "email": "john.doe@company.com",
        "status": "active",
        "last_login": "2025-01-26T10:30:00Z"
    },
    "last_event_timestamp": "2025-01-26T10:30:00Z"
}
```

### Python SDK Usage

```python
from event_streaming_bus import EventSourcingService

# Initialize service
sourcing_service = EventSourcingService()

# Append event with concurrency control
event_id = await sourcing_service.append_event(
    aggregate_id="user_123",
    aggregate_type="User",
    event_data={"email": "john.doe@company.com"},
    event_type="user.updated",
    expected_version=5,  # Prevent concurrent modifications
    tenant_id="tenant_001",
    user_id="user_456"
)

# Reconstruct aggregate state
aggregate_state = await sourcing_service.reconstruct_aggregate(
    aggregate_id="user_123",
    aggregate_type="User",
    update_snapshots=True,
    tenant_id="tenant_001"
)

print(f"Current version: {aggregate_state['version']}")
print(f"User email: {aggregate_state['data']['email']}")
```

## Stream Processing

Advanced stream processing capabilities for real-time data transformation and analytics.

### Processor Types

- **FILTER**: Filter events based on criteria
- **MAP**: Transform event data
- **AGGREGATE**: Aggregate events over time windows
- **JOIN**: Join multiple streams
- **WINDOW**: Window-based processing
- **CUSTOM**: Custom processing logic

### API Reference

#### Create Stream Processor

```http
POST /api/v1/stream-processors
Content-Type: application/json
Authorization: Bearer {token}

{
    "processor_name": "user_verification_filter",
    "processor_description": "Filter only verified user events",
    "processor_type": "FILTER",
    "source_stream_id": "str_user_events_123",
    "target_stream_id": "str_verified_users_456",
    "processing_logic": {
        "filter_expression": "payload.verification_status == 'verified'",
        "additional_checks": [
            "payload.email_verified == true",
            "payload.phone_verified == true"
        ]
    },
    "parallelism": 3,
    "error_handling_strategy": "RETRY"
}
```

**Response:**
```json
{
    "processor_id": "proc_789",
    "message": "Stream processor created successfully"
}
```

#### Start Stream Processor

```http
POST /api/v1/stream-processors/proc_789/start
Authorization: Bearer {token}
```

**Response:**
```json
{
    "message": "Stream processor started successfully"
}
```

#### Get Processor Metrics

```http
GET /api/v1/stream-processors/proc_789/metrics
Authorization: Bearer {token}
```

**Response:**
```json
{
    "processor_id": "proc_789",
    "processor_name": "user_verification_filter",
    "status": "RUNNING",
    "events_processed": 15432,
    "events_per_second": 47.3,
    "events_passed": 12876,
    "events_filtered": 2556,
    "processing_rate": 95.2,
    "avg_processing_time_ms": 12.5,
    "error_count": 3,
    "last_processed_at": "2025-01-26T10:30:00Z"
}
```

### Python SDK Usage

```python
from event_streaming_bus import StreamManagementService
from event_streaming_bus.models import ProcessorType

# Initialize service
stream_service = StreamManagementService()

# Create aggregation processor
processor_config = {
    "processor_name": "daily_sales_aggregator",
    "processor_type": ProcessorType.AGGREGATE.value,
    "source_stream_id": "str_order_events",
    "target_stream_id": "str_daily_sales",
    "processing_logic": {
        "aggregation_function": "sum",
        "aggregation_field": "order_total",
        "group_by": ["customer_segment", "region"]
    },
    "window_config": {
        "window_type": "tumbling",
        "window_size_ms": 86400000,  # 24 hours
        "grace_period_ms": 3600000   # 1 hour
    },
    "state_store_config": {
        "store_type": "rocksdb",
        "changelog_topic": "sales-aggregator-changelog"
    }
}

processor_id = await stream_service.create_stream_processor(
    processor_config=processor_config,
    tenant_id="tenant_001",
    user_id="user_456"
)

# Start processor
await stream_service.start_stream_processor(
    processor_id=processor_id,
    tenant_id="tenant_001"
)
```

## Enhanced Schema Registry

Advanced schema management with evolution, validation, and compatibility checking.

### Features

- **Schema Evolution**: Backward, forward, and full compatibility
- **Custom Validation Rules**: Business logic validation
- **Version Management**: Track schema versions and changes
- **Evolution Strategies**: COMPATIBLE, BREAKING_CHANGE, DEPRECATE

### API Reference

#### Register Enhanced Schema

```http
POST /api/v1/schemas/enhanced
Content-Type: application/json
Authorization: Bearer {token}

{
    "schema_name": "user_profile_event",
    "schema_version": "2.0",
    "json_schema": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "profile_data": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                    "phone": {"type": "string"},
                    "preferences": {"type": "object"}
                },
                "required": ["name", "email"]
            },
            "timestamp": {"type": "string", "format": "date-time"}
        },
        "required": ["user_id", "profile_data", "timestamp"]
    },
    "event_type": "user.profile_updated",
    "compatibility_level": "BACKWARD",
    "evolution_strategy": "COMPATIBLE",
    "validation_rules": {
        "business_rules": [
            {
                "name": "valid_email_domain",
                "description": "Email must be from approved domains",
                "rule": "profile_data.email.endswith(('@company.com', '@partner.com'))"
            }
        ]
    },
    "is_active": true
}
```

**Response:**
```json
{
    "schema_id": "sch_enhanced_789",
    "message": "Enhanced schema registered successfully"
}
```

#### Validate Event Against Schema

```http
POST /api/v1/schemas/sch_enhanced_789/validate
Content-Type: application/json
Authorization: Bearer {token}

{
    "user_id": "user_123",
    "profile_data": {
        "name": "John Doe",
        "email": "john.doe@company.com",
        "phone": "+1-555-0123",
        "preferences": {
            "notifications": true,
            "theme": "dark"
        }
    },
    "timestamp": "2025-01-26T10:30:00Z"
}
```

**Response:**
```json
{
    "is_valid": true,
    "validation_errors": [],
    "schema_version": "2.0",
    "validation_time_ms": 5.2
}
```

### Python SDK Usage

```python
from event_streaming_bus import SchemaRegistryService

# Initialize service
schema_service = SchemaRegistryService()

# Register schema with validation rules
schema_config = {
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
        "required": ["order_id", "customer_id", "order_total"]
    },
    "event_type": "order.created",
    "validation_rules": {
        "business_rules": [
            {
                "name": "minimum_order_value",
                "rule": "order_total >= 10.0"
            }
        ]
    }
}

schema_id = await schema_service.register_enhanced_schema(
    schema_config=schema_config,
    tenant_id="tenant_001",
    created_by="user_456"
)

# Validate event
validation_result = await schema_service.validate_event(
    schema_id=schema_id,
    event_data={
        "order_id": "ord_123",
        "customer_id": "cust_456",
        "order_total": 25.99,
        "currency": "USD"
    },
    tenant_id="tenant_001"
)

print(f"Valid: {validation_result['is_valid']}")
```

## Consumer Management

Advanced consumer group management with lag monitoring and automatic rebalancing.

### Features

- **Consumer Group Management**: Create and configure consumer groups
- **Lag Monitoring**: Real-time consumer lag tracking
- **Automatic Rebalancing**: Trigger rebalancing when needed
- **Performance Metrics**: Track consumption rates and patterns

### API Reference

#### Create Consumer Group

```http
POST /api/v1/consumer-groups
Content-Type: application/json
Authorization: Bearer {token}

{
    "group_name": "notification_processors",
    "group_description": "Processes user notification events",
    "session_timeout_ms": 30000,
    "heartbeat_interval_ms": 3000,
    "max_poll_interval_ms": 300000,
    "partition_assignment_strategy": "range",
    "rebalance_timeout_ms": 60000
}
```

**Response:**
```json
{
    "group_id": "grp_notifications_789",
    "message": "Consumer group created successfully"
}
```

#### Get Consumer Lag

```http
GET /api/v1/consumer-groups/grp_notifications_789/lag
Authorization: Bearer {token}
```

**Response:**
```json
{
    "group_id": "grp_notifications_789",
    "group_name": "notification_processors",
    "total_lag": 1247,
    "partition_lags": {
        "0": 423,
        "1": 389,
        "2": 435
    },
    "consumption_rate": 85.3,
    "active_consumers": 3,
    "last_rebalance": "2025-01-26T09:15:00Z"
}
```

#### Trigger Rebalance

```http
POST /api/v1/consumer-groups/grp_notifications_789/rebalance
Authorization: Bearer {token}
```

**Response:**
```json
{
    "message": "Consumer group rebalance triggered successfully"
}
```

### Python SDK Usage

```python
from event_streaming_bus import ConsumerManagementService

# Initialize service
consumer_service = ConsumerManagementService()

# Create consumer group
group_config = {
    "group_name": "analytics_processors",
    "session_timeout_ms": 30000,
    "heartbeat_interval_ms": 3000,
    "partition_assignment_strategy": "range"
}

group_id = await consumer_service.create_consumer_group(
    group_config=group_config,
    tenant_id="tenant_001",
    user_id="user_456"
)

# Monitor consumer lag
lag_info = await consumer_service.get_consumer_lag(
    group_id=group_id,
    tenant_id="tenant_001"
)

if lag_info["total_lag"] > 1000:
    # Trigger rebalance if lag is too high
    await consumer_service.trigger_rebalance(
        group_id=group_id,
        tenant_id="tenant_001"
    )
```

## Event Priority & Routing

Advanced event routing based on priority levels and custom rules.

### Priority Levels

- **CRITICAL**: Highest priority, processed immediately
- **HIGH**: High priority, processed before normal events
- **NORMAL**: Standard priority (default)
- **LOW**: Low priority, processed when resources available

### API Reference

#### Publish High-Priority Event

```http
POST /api/v1/events
Content-Type: application/json
Authorization: Bearer {token}

{
    "event_type": "system.critical_alert",
    "source_capability": "monitoring",
    "aggregate_id": "alert_123",
    "aggregate_type": "SystemAlert",
    "payload": {
        "alert_type": "service_down",
        "service_name": "payment_service",
        "severity": "critical",
        "timestamp": "2025-01-26T10:30:00Z"
    },
    "priority": "CRITICAL",
    "compression_type": "NONE",
    "serialization_format": "JSON",
    "max_retries": 5
}
```

### Python SDK Usage

```python
from event_streaming_bus import EventPublishingService, EventConfig
from event_streaming_bus.models import EventPriority, CompressionType

# Publish critical event
event_config = EventConfig(
    event_type="system.critical_alert",
    source_capability="monitoring",
    aggregate_id="alert_123",
    aggregate_type="SystemAlert",
    priority=EventPriority.CRITICAL,
    compression_type=CompressionType.NONE,
    max_retries=5
)

publishing_service = EventPublishingService()
event_id = await publishing_service.publish_event(
    event_config=event_config,
    payload={
        "alert_type": "service_down",
        "service_name": "payment_service",
        "severity": "critical"
    },
    stream_id="critical_alerts_stream",
    tenant_id="tenant_001",
    user_id="system"
)
```

## Processing History & Audit Trails

Comprehensive tracking of event processing with detailed audit trails.

### Features

- **Complete Processing History**: Track every processing step
- **Error Tracking**: Detailed error information and stack traces
- **Performance Metrics**: Processing duration and throughput
- **Audit Compliance**: Full audit trail for regulatory compliance

### API Reference

Processing history is automatically tracked for all events. You can query it using:

```http
GET /api/v1/events/evt_123/processing-history
Authorization: Bearer {token}
```

**Response:**
```json
{
    "event_id": "evt_123",
    "processing_history": [
        {
            "history_id": "hist_456",
            "processor_id": "proc_validation",
            "processing_stage": "VALIDATION",
            "status": "COMPLETED",
            "started_at": "2025-01-26T10:30:00.000Z",
            "completed_at": "2025-01-26T10:30:00.125Z",
            "processing_duration_ms": 125,
            "input_data": {
                "event_type": "user.created",
                "payload": {...}
            },
            "output_data": {
                "validation_result": "passed",
                "schema_version": "1.0"
            },
            "retry_count": 0
        },
        {
            "history_id": "hist_789",
            "processor_id": "proc_enrichment",
            "processing_stage": "ENRICHMENT",
            "status": "COMPLETED",
            "started_at": "2025-01-26T10:30:00.130Z",
            "completed_at": "2025-01-26T10:30:00.245Z",
            "processing_duration_ms": 115,
            "retry_count": 0
        }
    ],
    "total_processing_time_ms": 240,
    "stages_completed": 2,
    "overall_status": "COMPLETED"
}
```

## Real-time Monitoring & Metrics

Comprehensive monitoring with real-time metrics and alerting.

### Available Metrics

- **Throughput**: Events per second by stream and priority
- **Latency**: End-to-end processing latency percentiles
- **Consumer Lag**: Real-time lag monitoring
- **Error Rates**: Failed events and retry patterns
- **Resource Usage**: CPU, memory, and disk utilization

### WebSocket Real-time Monitoring

```javascript
// Connect to real-time monitoring
const ws = new WebSocket('wss://api.example.com/ws/monitoring');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.message_type === 'monitoring') {
        console.log('Live metrics:', data.data);
        // Update dashboard with real-time metrics
        updateDashboard(data.data);
    }
};

// Request specific stream metrics
ws.send(JSON.stringify({
    type: 'subscribe',
    streams: ['user_events', 'order_events'],
    metrics: ['throughput', 'latency', 'error_rate']
}));
```

### Prometheus Metrics

The Event Streaming Bus exposes metrics in Prometheus format:

```bash
# Get Prometheus metrics
curl http://localhost:8080/metrics

# Sample output:
# event_streaming_throughput_total{stream="user_events",priority="high"} 1247
# event_streaming_latency_seconds{quantile="0.95",stream="user_events"} 0.025
# event_streaming_consumer_lag_total{group="notifications"} 156
# event_streaming_errors_total{type="validation",stream="user_events"} 3
```

## Best Practices

### Event Sourcing

1. **Use Meaningful Event Types**: Choose descriptive event type names
2. **Keep Events Immutable**: Never modify published events
3. **Handle Concurrency**: Always use expected_version for updates
4. **Create Snapshots**: Use snapshots for aggregates with many events

### Stream Processing

1. **Design for Idempotency**: Ensure processors can handle duplicate events
2. **Monitor Resource Usage**: Keep an eye on CPU and memory usage
3. **Handle Failures Gracefully**: Implement proper error handling
4. **Test with Real Data**: Use production-like data volumes for testing

### Schema Management

1. **Plan for Evolution**: Design schemas that can evolve gracefully
2. **Use Semantic Versioning**: Follow semantic versioning for schema versions
3. **Test Compatibility**: Always test schema changes for compatibility
4. **Document Changes**: Maintain clear documentation of schema evolution

### Consumer Management

1. **Monitor Lag Regularly**: Set up alerts for high consumer lag
2. **Scale Appropriately**: Add consumers when lag increases
3. **Handle Rebalancing**: Ensure applications handle rebalancing gracefully
4. **Use Appropriate Timeouts**: Configure timeouts based on processing needs

## Migration Guide

### Upgrading from Basic to Enterprise Features

1. **Update Dependencies**: Install enterprise feature dependencies
2. **Migrate Schemas**: Convert existing schemas to enhanced format
3. **Update Consumers**: Modify consumers to use new consumer management
4. **Add Monitoring**: Implement monitoring and alerting
5. **Test Thoroughly**: Ensure all features work correctly

See the [Migration Guide](migration.md) for detailed steps.