# APG Event Streaming Bus - API Reference

Complete API reference for the Event Streaming Bus capability.

## Authentication

All API endpoints require authentication via JWT tokens:

```http
Authorization: Bearer <your-jwt-token>
```

## Event Publishing API

### Publish Single Event

```http
POST /api/v1/events
Content-Type: application/json

{
  "event_type": "user.created",
  "source_capability": "user_management",
  "aggregate_id": "user_123",
  "aggregate_type": "User",
  "payload": {
    "user_name": "john.doe",
    "email": "john.doe@company.com",
    "department": "Engineering"
  },
  "metadata": {
    "ip_address": "192.168.1.100",
    "user_agent": "APG-Client/1.0"
  },
  "correlation_id": "cor_abc123",
  "partition_key": "user_123"
}
```

**Response:**
```json
{
  "event_id": "evt_01J2X3Y4Z5A6B7C8D9E0F1G2H3",
  "event_type": "user.created",
  "event_version": "1.0",
  "source_capability": "user_management",
  "aggregate_id": "user_123",
  "aggregate_type": "User",
  "sequence_number": 1,
  "timestamp": "2025-01-26T10:30:00.000Z",
  "correlation_id": "cor_abc123",
  "causation_id": null,
  "payload": {
    "user_name": "john.doe",
    "email": "john.doe@company.com",
    "department": "Engineering"
  },
  "metadata": {
    "ip_address": "192.168.1.100",
    "user_agent": "APG-Client/1.0"
  },
  "status": "published",
  "stream_id": "user_events"
}
```

### Publish Event Batch

```http
POST /api/v1/events/batch
Content-Type: application/json

{
  "events": [
    {
      "event_type": "user.created",
      "source_capability": "user_management",
      "aggregate_id": "user_123",
      "aggregate_type": "User",
      "payload": {
        "user_name": "john.doe",
        "email": "john.doe@company.com"
      }
    },
    {
      "event_type": "user.created", 
      "source_capability": "user_management",
      "aggregate_id": "user_124",
      "aggregate_type": "User",
      "payload": {
        "user_name": "jane.smith",
        "email": "jane.smith@company.com"
      }
    }
  ],
  "stream_id": "user_events",
  "batch_options": {
    "timeout_ms": 5000,
    "compression": true
  }
}
```

**Response:**
```json
{
  "message": "Successfully published 2 events",
  "event_ids": [
    "evt_01J2X3Y4Z5A6B7C8D9E0F1G2H3",
    "evt_01J2X3Y4Z5A6B7C8D9E0F1G2H4"
  ],
  "batch_size": 2
}
```

### Get Event by ID

```http
GET /api/v1/events/{event_id}
```

**Response:**
```json
{
  "event_id": "evt_01J2X3Y4Z5A6B7C8D9E0F1G2H3",
  "event_type": "user.created",
  "event_version": "1.0",
  "source_capability": "user_management",
  "aggregate_id": "user_123",
  "aggregate_type": "User",
  "sequence_number": 1,
  "timestamp": "2025-01-26T10:30:00.000Z",
  "correlation_id": "cor_abc123",
  "causation_id": null,
  "payload": {
    "user_name": "john.doe",
    "email": "john.doe@company.com",
    "department": "Engineering"
  },
  "metadata": {
    "ip_address": "192.168.1.100",
    "user_agent": "APG-Client/1.0"
  },
  "status": "published",
  "stream_id": "user_events"
}
```

### Query Events

```http
POST /api/v1/events/query
Content-Type: application/json

{
  "stream_id": "user_events",
  "event_type": "user.created",
  "source_capability": "user_management",
  "start_time": "2025-01-26T00:00:00.000Z",
  "end_time": "2025-01-26T23:59:59.999Z",
  "status": "published",
  "limit": 100,
  "offset": 0
}
```

**Response:**
```json
{
  "events": [
    {
      "event_id": "evt_01J2X3Y4Z5A6B7C8D9E0F1G2H3",
      "event_type": "user.created",
      "source_capability": "user_management",
      "aggregate_id": "user_123",
      "timestamp": "2025-01-26T10:30:00.000Z",
      "payload": {
        "user_name": "john.doe",
        "email": "john.doe@company.com"
      },
      "status": "published"
    }
  ],
  "total_count": 1,
  "limit": 100,
  "offset": 0,
  "has_more": false
}
```

## Stream Management API

### List Streams

```http
GET /api/v1/streams
```

**Response:**
```json
{
  "streams": [
    {
      "stream_id": "str_user_events",
      "stream_name": "user_events",
      "topic_name": "apg-user-events",
      "source_capability": "user_management",
      "status": "active",
      "partitions": 6,
      "replication_factor": 3,
      "created_at": "2025-01-26T10:00:00.000Z"
    }
  ]
}
```

### Create Stream

```http
POST /api/v1/streams
Content-Type: application/json

{
  "stream_name": "order_events",
  "stream_description": "Order lifecycle events",
  "topic_name": "apg-order-events",
  "partitions": 12,
  "replication_factor": 3,
  "retention_time_ms": 2592000000,
  "retention_size_bytes": 10737418240,
  "cleanup_policy": "delete",
  "compression_type": "snappy",
  "default_serialization": "json",
  "event_category": "domain_event",
  "source_capability": "order_management",
  "config_settings": {
    "max_message_bytes": 1048576,
    "flush_ms": 1000
  }
}
```

**Response:**
```json
{
  "stream_id": "str_01J2X3Y4Z5A6B7C8D9E0F1G2H5",
  "message": "Stream created successfully"
}
```

### Get Stream Details

```http
GET /api/v1/streams/{stream_id}
```

**Response:**
```json
{
  "stream_id": "str_user_events",
  "stream_name": "user_events",
  "stream_description": "User lifecycle events",
  "topic_name": "apg-user-events",
  "partitions": 6,
  "replication_factor": 3,
  "retention_time_ms": 604800000,
  "retention_size_bytes": 1073741824,
  "cleanup_policy": "delete",
  "compression_type": "snappy",
  "default_serialization": "json",
  "event_category": "domain_event",
  "source_capability": "user_management",
  "status": "active",
  "tenant_id": "tenant_001",
  "created_at": "2025-01-26T10:00:00.000Z",
  "updated_at": "2025-01-26T10:00:00.000Z",
  "created_by": "admin_user"
}
```

### Get Stream Events

```http
GET /api/v1/streams/{stream_id}/events?limit=50&offset=0&start_time=2025-01-26T00:00:00.000Z
```

**Response:**
```json
{
  "events": [
    {
      "event_id": "evt_01J2X3Y4Z5A6B7C8D9E0F1G2H3",
      "event_type": "user.created",
      "timestamp": "2025-01-26T10:30:00.000Z",
      "payload": {
        "user_name": "john.doe",
        "email": "john.doe@company.com"
      }
    }
  ],
  "total_count": 1,
  "stream_id": "str_user_events",
  "limit": 50,
  "offset": 0
}
```

### Get Stream Metrics

```http
GET /api/v1/streams/{stream_id}/metrics
```

**Response:**
```json
{
  "stream_id": "str_user_events",
  "stream_name": "user_events",
  "total_events": 1500,
  "events_per_second": 25.5,
  "events_today": 300,
  "events_last_hour": 120,
  "consumer_count": 3,
  "total_lag": 45,
  "health_status": "healthy",
  "last_event_time": "2025-01-26T10:30:00.000Z"
}
```

## Subscription Management API

### Create Subscription

```http
POST /api/v1/subscriptions
Content-Type: application/json

{
  "subscription_name": "user_notifications",
  "subscription_description": "Process user events for notifications",
  "stream_id": "str_user_events",
  "consumer_group_id": "notification_service",
  "consumer_name": "notification_worker_1",
  "event_type_patterns": ["user.*"],
  "filter_criteria": {
    "department": ["Engineering", "Marketing"]
  },
  "delivery_mode": "at_least_once",
  "batch_size": 100,
  "max_wait_time_ms": 1000,
  "start_position": "latest",
  "retry_policy": {
    "max_retries": 3,
    "retry_delay_ms": 1000,
    "backoff_multiplier": 2.0,
    "max_delay_ms": 60000
  },
  "dead_letter_enabled": true,
  "dead_letter_topic": "user_events_dlq",
  "webhook_url": "https://your-service.com/webhook",
  "webhook_headers": {
    "Authorization": "Bearer your-token",
    "Content-Type": "application/json"
  },
  "webhook_timeout_ms": 30000
}
```

**Response:**
```json
{
  "subscription_id": "sub_01J2X3Y4Z5A6B7C8D9E0F1G2H6",
  "message": "Subscription created successfully"
}
```

### List Subscriptions

```http
GET /api/v1/subscriptions
```

**Response:**
```json
{
  "subscriptions": [
    {
      "subscription_id": "sub_01J2X3Y4Z5A6B7C8D9E0F1G2H6",
      "subscription_name": "user_notifications",
      "stream_id": "str_user_events",
      "consumer_group_id": "notification_service",
      "status": "active",
      "last_consumed_at": "2025-01-26T10:30:00.000Z",
      "created_at": "2025-01-26T09:00:00.000Z"
    }
  ]
}
```

### Get Subscription Status

```http
GET /api/v1/subscriptions/{subscription_id}/status
```

**Response:**
```json
{
  "subscription_id": "sub_01J2X3Y4Z5A6B7C8D9E0F1G2H6",
  "subscription_name": "user_notifications",
  "status": "active",
  "consumer_lag": 25,
  "last_consumed_offset": 1475,
  "last_consumed_at": "2025-01-26T10:30:00.000Z",
  "events_processed_today": 500,
  "processing_rate": 12.5,
  "error_count": 2
}
```

### Cancel Subscription

```http
DELETE /api/v1/subscriptions/{subscription_id}
```

**Response:**
```json
204 No Content
```

## Schema Registry API

### Register Schema

```http
POST /api/v1/schemas
Content-Type: application/json

{
  "schema_name": "user_created",
  "schema_version": "1.0",
  "schema_definition": {
    "type": "object",
    "properties": {
      "user_name": {
        "type": "string",
        "pattern": "^[a-zA-Z0-9._-]+$"
      },
      "email": {
        "type": "string",
        "format": "email"
      },
      "department": {
        "type": "string",
        "enum": ["Engineering", "Marketing", "Sales", "Support"]
      }
    },
    "required": ["user_name", "email"],
    "additionalProperties": false
  },
  "schema_format": "json_schema",
  "event_type": "user.created",
  "compatibility_level": "backward"
}
```

**Response:**
```json
{
  "schema_id": "sch_01J2X3Y4Z5A6B7C8D9E0F1G2H7",
  "message": "Schema registered successfully"
}
```

### List Schemas

```http
GET /api/v1/schemas?event_type=user.created
```

**Response:**
```json
{
  "schemas": [
    {
      "schema_id": "sch_01J2X3Y4Z5A6B7C8D9E0F1G2H7",
      "schema_name": "user_created",
      "schema_version": "1.0",
      "event_type": "user.created",
      "schema_format": "json_schema",
      "compatibility_level": "backward",
      "is_active": true,
      "created_at": "2025-01-26T09:00:00.000Z"
    }
  ]
}
```

### Get Schema

```http
GET /api/v1/schemas/{schema_id}
```

**Response:**
```json
{
  "schema_id": "sch_01J2X3Y4Z5A6B7C8D9E0F1G2H7",
  "schema_name": "user_created",
  "schema_version": "1.0",
  "schema_definition": {
    "type": "object",
    "properties": {
      "user_name": {
        "type": "string",
        "pattern": "^[a-zA-Z0-9._-]+$"
      },
      "email": {
        "type": "string",
        "format": "email"
      },
      "department": {
        "type": "string",
        "enum": ["Engineering", "Marketing", "Sales", "Support"]
      }
    },
    "required": ["user_name", "email"],
    "additionalProperties": false
  },
  "schema_format": "json_schema",
  "event_type": "user.created",
  "compatibility_level": "backward",
  "is_active": true,
  "tenant_id": "tenant_001",
  "created_at": "2025-01-26T09:00:00.000Z",
  "updated_at": "2025-01-26T09:00:00.000Z",
  "created_by": "admin_user"
}
```

## System Status API

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-26T10:30:00.000Z",
  "version": "1.0.0"
}
```

### System Status

```http
GET /api/v1/status
```

**Response:**
```json
{
  "system_status": "operational",
  "timestamp": "2025-01-26T10:30:00.000Z",
  "components": {
    "api": "healthy",
    "kafka": "healthy",
    "redis": "healthy",
    "postgresql": "healthy"
  },
  "active_connections": 25,
  "stream_subscribers": 12,
  "subscription_subscribers": 8
}
```

## WebSocket API

### Real-time Event Stream

Connect to a WebSocket endpoint to receive real-time events:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/events/user_events');

ws.onopen = function(event) {
    console.log('Connected to event stream');
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received event:', message);
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};
```

**Message Format:**
```json
{
  "message_type": "event",
  "timestamp": "2025-01-26T10:30:00.000Z",
  "data": {
    "event_id": "evt_01J2X3Y4Z5A6B7C8D9E0F1G2H3",
    "event_type": "user.created",
    "stream_id": "user_events",
    "payload": {
      "user_name": "john.doe",
      "email": "john.doe@company.com"
    }
  }
}
```

### Subscription Updates

Monitor subscription status in real-time:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/subscriptions/sub_123');

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    if (message.message_type === 'status_update') {
        console.log('Subscription status:', message.data);
    }
};
```

### Live Monitoring

Monitor system metrics in real-time:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/monitoring');

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    if (message.message_type === 'monitoring') {
        console.log('System metrics:', message.data);
    }
};
```

## Error Responses

All API endpoints use standard HTTP status codes and return error details:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid event type format",
    "details": {
      "field": "event_type",
      "value": "",
      "constraint": "Event type cannot be empty"
    }
  },
  "timestamp": "2025-01-26T10:30:00.000Z",
  "request_id": "req_01J2X3Y4Z5A6B7C8D9E0F1G2H8"
}
```

### Common Error Codes

- `VALIDATION_ERROR` - Request validation failed
- `AUTHENTICATION_REQUIRED` - Missing or invalid authentication
- `AUTHORIZATION_DENIED` - Insufficient permissions
- `RESOURCE_NOT_FOUND` - Requested resource does not exist
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `INTERNAL_ERROR` - Server-side error
- `SERVICE_UNAVAILABLE` - Service temporarily unavailable