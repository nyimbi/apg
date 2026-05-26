# AICR WebSocket API Reference

**Version:** 1.0.0
**Author:** Nyimbi Odero <nyimbi@gmail.com>
**Copyright:** © 2025 Datacraft

## Table of Contents

1. [WebSocket Overview](#websocket-overview)
2. [Connection Setup](#connection-setup)
3. [Authentication](#authentication)
4. [Message Protocol](#message-protocol)
5. [Real-time Inference](#real-time-inference)
6. [Model Management Events](#model-management-events)
7. [System Monitoring](#system-monitoring)
8. [Pipeline Notifications](#pipeline-notifications)
9. [Error Handling](#error-handling)
10. [Connection Management](#connection-management)

## WebSocket Overview

The AICR WebSocket API provides real-time, bidirectional communication for AI operations, enabling:

- **Real-time Inference**: Streaming inference with immediate results
- **Live Monitoring**: Real-time metrics and system status updates
- **Event Notifications**: Instant updates for model deployments, pipeline executions
- **Interactive Sessions**: Persistent connections for conversational AI

### Connection URLs

```
Production: wss://ws.datacraft.co.ke/aicr/v1/ws
Development: ws://localhost:8080/ws/v1
```

### Supported Protocols

- **WebSocket (RFC 6455)**: Standard WebSocket protocol
- **Socket.IO**: Enhanced WebSocket with fallbacks and room support
- **Server-Sent Events (SSE)**: One-way streaming for browsers

## Connection Setup

### Basic WebSocket Connection

```javascript
// JavaScript
const ws = new WebSocket('wss://ws.datacraft.co.ke/aicr/v1/ws');

ws.onopen = function(event) {
    console.log('Connected to AICR WebSocket');
    // Send authentication
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'your_jwt_token_here'
    }));
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};

ws.onclose = function(event) {
    console.log('Connection closed:', event.code, event.reason);
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};
```

```python
# Python
import asyncio
import websockets
import json

async def connect_aicr():
    uri = "wss://ws.datacraft.co.ke/aicr/v1/ws"

    async with websockets.connect(uri) as websocket:
        # Authenticate
        auth_message = {
            "type": "auth",
            "token": "your_jwt_token_here"
        }
        await websocket.send(json.dumps(auth_message))

        # Listen for messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")

# Run the connection
asyncio.run(connect_aicr())
```

### Socket.IO Connection

```javascript
// JavaScript with Socket.IO
import io from 'socket.io-client';

const socket = io('https://ws.datacraft.co.ke/aicr/v1', {
    auth: {
        token: 'your_jwt_token_here'
    },
    transports: ['websocket', 'polling']
});

socket.on('connect', () => {
    console.log('Connected to AICR');

    // Join specific rooms
    socket.emit('join', {
        rooms: ['model_updates', 'system_metrics']
    });
});

socket.on('inference_result', (data) => {
    console.log('Inference result:', data);
});

socket.on('system_metric', (metric) => {
    console.log('System metric:', metric);
});
```

## Authentication

### JWT Token Authentication

Send authentication message immediately after connection:

```json
{
  "type": "auth",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "client_info": {
    "client_id": "web_app_v1",
    "user_agent": "Mozilla/5.0...",
    "ip_address": "192.168.1.100"
  }
}
```

**Response:**
```json
{
  "type": "auth_response",
  "status": "success",
  "user_info": {
    "user_id": "user_123",
    "username": "user@example.com",
    "roles": ["user", "data_scientist"],
    "permissions": ["model:read", "inference:execute"]
  },
  "session_info": {
    "session_id": "sess_abc123",
    "expires_at": "2024-01-15T11:30:00Z"
  },
  "available_channels": [
    "inference",
    "model_updates",
    "system_metrics",
    "pipeline_notifications"
  ]
}
```

### API Key Authentication

```json
{
  "type": "auth",
  "api_key": "aicr_api_key_abc123xyz789",
  "client_info": {
    "application": "ml_platform",
    "version": "2.1.0"
  }
}
```

## Message Protocol

### Message Structure

All WebSocket messages follow a consistent JSON structure:

```json
{
  "type": "message_type",
  "id": "unique_message_id",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    // Message-specific data
  },
  "metadata": {
    "client_id": "client_123",
    "session_id": "sess_abc123"
  }
}
```

### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `auth` | Client → Server | Authentication request |
| `auth_response` | Server → Client | Authentication result |
| `inference_request` | Client → Server | Real-time inference request |
| `inference_result` | Server → Client | Inference result |
| `inference_progress` | Server → Client | Inference progress update |
| `subscribe` | Client → Server | Subscribe to channels |
| `unsubscribe` | Client → Server | Unsubscribe from channels |
| `model_event` | Server → Client | Model management event |
| `system_metric` | Server → Client | System metrics update |
| `pipeline_event` | Server → Client | Pipeline execution event |
| `error` | Server → Client | Error notification |
| `heartbeat` | Bidirectional | Connection health check |

### Channel Subscription

Subscribe to specific data channels:

```json
{
  "type": "subscribe",
  "channels": [
    "model_updates",
    "system_metrics",
    "inference_results",
    "pipeline_notifications"
  ],
  "filters": {
    "model_ids": ["mdl_abc123", "mdl_def456"],
    "metric_types": ["cpu_usage", "inference_latency"],
    "severity_levels": ["warning", "error", "critical"]
  }
}
```

**Response:**
```json
{
  "type": "subscription_response",
  "status": "success",
  "subscribed_channels": [
    "model_updates",
    "system_metrics",
    "inference_results",
    "pipeline_notifications"
  ],
  "failed_channels": [],
  "message": "Successfully subscribed to 4 channels"
}
```

## Real-time Inference

### Streaming Inference Request

Send inference requests for immediate processing:

```json
{
  "type": "inference_request",
  "id": "req_stream_123",
  "model_id": "mdl_abc123",
  "input_data": {
    "text": "I love this product! It's amazing."
  },
  "parameters": {
    "confidence_threshold": 0.8,
    "stream_results": true,
    "return_probabilities": true
  },
  "priority": "high"
}
```

### Inference Progress Updates

For long-running inference, receive progress updates:

```json
{
  "type": "inference_progress",
  "id": "req_stream_123",
  "model_id": "mdl_abc123",
  "progress": {
    "stage": "preprocessing",
    "percent_complete": 25,
    "estimated_remaining_ms": 1500,
    "current_step": "tokenization",
    "total_steps": 4
  },
  "timestamp": "2024-01-15T10:30:15Z"
}
```

### Inference Results

Receive real-time inference results:

```json
{
  "type": "inference_result",
  "id": "req_stream_123",
  "model_id": "mdl_abc123",
  "status": "completed",
  "predictions": {
    "sentiment": "positive",
    "confidence": 0.94,
    "probabilities": {
      "positive": 0.94,
      "negative": 0.03,
      "neutral": 0.03
    }
  },
  "metadata": {
    "model_version": "2.1.0",
    "framework": "pytorch",
    "device": "gpu",
    "processing_time_ms": 23.7,
    "queue_time_ms": 5.2
  },
  "timestamp": "2024-01-15T10:30:18Z"
}
```

### Batch Streaming Inference

Process multiple inputs with streaming results:

```json
{
  "type": "batch_inference_request",
  "id": "batch_stream_456",
  "model_id": "mdl_abc123",
  "inputs": [
    {"id": "input_1", "data": {"text": "Great product!"}},
    {"id": "input_2", "data": {"text": "Terrible service."}},
    {"id": "input_3", "data": {"text": "It's okay."}}
  ],
  "parameters": {
    "stream_individual_results": true,
    "batch_size": 16
  }
}
```

Results streamed individually:

```json
{
  "type": "batch_inference_result",
  "batch_id": "batch_stream_456",
  "input_id": "input_1",
  "status": "completed",
  "predictions": {
    "sentiment": "positive",
    "confidence": 0.92
  },
  "processing_time_ms": 19.4,
  "timestamp": "2024-01-15T10:30:20Z"
}
```

## Model Management Events

### Model Deployment Events

Receive real-time updates on model deployments:

```json
{
  "type": "model_event",
  "event": "deployment_started",
  "model_id": "mdl_new_789",
  "deployment_id": "dep_xyz123",
  "data": {
    "model_name": "sentiment_analyzer_v3",
    "deployment_config": {
      "instance_type": "gpu_medium",
      "min_instances": 2,
      "max_instances": 10
    },
    "estimated_ready_time": "2024-01-15T10:35:00Z"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

```json
{
  "type": "model_event",
  "event": "deployment_ready",
  "model_id": "mdl_new_789",
  "deployment_id": "dep_xyz123",
  "data": {
    "endpoint": "https://api.datacraft.co.ke/aicr/v1/models/mdl_new_789/predict",
    "instances": {
      "running": 2,
      "healthy": 2
    },
    "performance": {
      "average_latency_ms": 45.2,
      "throughput_rps": 120.5
    }
  },
  "timestamp": "2024-01-15T10:34:27Z"
}
```

### Model Status Changes

```json
{
  "type": "model_event",
  "event": "status_changed",
  "model_id": "mdl_abc123",
  "data": {
    "previous_status": "active",
    "new_status": "maintenance",
    "reason": "scheduled_update",
    "estimated_downtime_minutes": 15,
    "maintenance_window": {
      "start": "2024-01-15T10:30:00Z",
      "end": "2024-01-15T10:45:00Z"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## System Monitoring

### Real-time Metrics Stream

Subscribe to live system metrics:

```json
{
  "type": "subscribe",
  "channels": ["system_metrics"],
  "filters": {
    "metric_names": [
      "cpu_usage",
      "memory_usage",
      "inference_latency",
      "request_count",
      "error_rate"
    ],
    "components": ["inference_engine", "api_server"],
    "update_interval": 5
  }
}
```

Receive metric updates:

```json
{
  "type": "system_metric",
  "metric_name": "inference_latency",
  "component": "inference_engine",
  "value": 67.4,
  "unit": "milliseconds",
  "labels": {
    "model_type": "classification",
    "framework": "pytorch",
    "device": "gpu"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Performance Alerts

Real-time alerts for performance issues:

```json
{
  "type": "alert",
  "severity": "warning",
  "alert_name": "high_inference_latency",
  "component": "inference_engine",
  "data": {
    "current_value": 150.7,
    "threshold": 100.0,
    "duration_seconds": 300,
    "affected_models": ["mdl_abc123", "mdl_def456"],
    "suggested_actions": [
      "Check GPU utilization",
      "Consider scaling up instances",
      "Review model optimization"
    ]
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Health Status Updates

```json
{
  "type": "health_update",
  "component": "inference_engine",
  "status": "degraded",
  "previous_status": "healthy",
  "data": {
    "health_score": 0.75,
    "issues": [
      {
        "type": "performance",
        "description": "Increased response time",
        "severity": "warning"
      }
    ],
    "affected_services": ["model_inference", "batch_processing"]
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Pipeline Notifications

### Pipeline Execution Events

Track ML pipeline progress in real-time:

```json
{
  "type": "pipeline_event",
  "event": "execution_started",
  "pipeline_id": "pip_training_123",
  "execution_id": "exec_new_456",
  "data": {
    "pipeline_name": "sentiment_training_pipeline",
    "stages": [
      "data_loading",
      "preprocessing",
      "training",
      "validation",
      "deployment"
    ],
    "estimated_duration_minutes": 120,
    "parameters": {
      "dataset_version": "v2.1",
      "epochs": 50,
      "learning_rate": 0.001
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Stage Progress Updates

```json
{
  "type": "pipeline_event",
  "event": "stage_progress",
  "pipeline_id": "pip_training_123",
  "execution_id": "exec_new_456",
  "data": {
    "current_stage": "training",
    "stage_progress": {
      "epoch": 25,
      "total_epochs": 50,
      "percent_complete": 50,
      "current_loss": 0.342,
      "best_accuracy": 0.89,
      "estimated_remaining_minutes": 45
    },
    "logs": [
      "Epoch 25/50 - Loss: 0.342, Accuracy: 0.87",
      "Validation accuracy: 0.89 (best so far)"
    ]
  },
  "timestamp": "2024-01-15T10:45:00Z"
}
```

### Pipeline Completion

```json
{
  "type": "pipeline_event",
  "event": "execution_completed",
  "pipeline_id": "pip_training_123",
  "execution_id": "exec_new_456",
  "data": {
    "status": "success",
    "duration_minutes": 105,
    "results": {
      "final_accuracy": 0.94,
      "model_id": "mdl_new_789",
      "deployment_id": "dep_auto_123"
    },
    "artifacts": [
      {
        "type": "model",
        "path": "/models/sentiment_v3_final.pth",
        "size_mb": 245.7
      },
      {
        "type": "metrics",
        "path": "/results/training_metrics.json",
        "size_mb": 0.5
      }
    ]
  },
  "timestamp": "2024-01-15T12:15:00Z"
}
```

## Error Handling

### Error Message Format

```json
{
  "type": "error",
  "error_code": "INFERENCE_FAILED",
  "message": "Model inference failed due to invalid input format",
  "details": {
    "request_id": "req_stream_123",
    "model_id": "mdl_abc123",
    "input_validation_errors": [
      {
        "field": "text",
        "error": "Field exceeds maximum length of 512 characters"
      }
    ],
    "retry_possible": true,
    "suggested_action": "Reduce input text length and retry"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Connection Errors

```json
{
  "type": "error",
  "error_code": "CONNECTION_ERROR",
  "message": "WebSocket connection lost",
  "details": {
    "reason": "network_timeout",
    "last_message_id": "msg_456",
    "reconnect_suggested": true,
    "reconnect_delay_ms": 5000
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Rate Limiting

```json
{
  "type": "error",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "message": "WebSocket message rate limit exceeded",
  "details": {
    "limit": 100,
    "window_seconds": 60,
    "current_count": 101,
    "reset_at": "2024-01-15T10:31:00Z",
    "backoff_seconds": 30
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Connection Management

### Heartbeat Protocol

Keep connections alive with heartbeats:

```javascript
// Client-side heartbeat
setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'heartbeat',
            timestamp: new Date().toISOString()
        }));
    }
}, 30000); // Every 30 seconds
```

Server response:
```json
{
  "type": "heartbeat_response",
  "server_time": "2024-01-15T10:30:00Z",
  "connection_uptime_seconds": 3600,
  "message_count": 145
}
```

### Graceful Disconnect

```json
{
  "type": "disconnect",
  "reason": "client_shutdown",
  "message": "Application shutting down gracefully"
}
```

### Connection Limits

- **Maximum connections per user**: 10
- **Maximum message rate**: 100 messages/minute
- **Maximum message size**: 1MB
- **Connection timeout**: 5 minutes idle
- **Heartbeat interval**: 30 seconds

### Reconnection Strategy

```javascript
// Automatic reconnection with exponential backoff
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const baseDelay = 1000;

function connect() {
    const ws = new WebSocket('wss://ws.datacraft.co.ke/aicr/v1/ws');

    ws.onopen = () => {
        reconnectAttempts = 0;
        console.log('Connected successfully');
    };

    ws.onclose = (event) => {
        if (reconnectAttempts < maxReconnectAttempts) {
            const delay = baseDelay * Math.pow(2, reconnectAttempts);
            console.log(`Reconnecting in ${delay}ms...`);
            setTimeout(connect, delay);
            reconnectAttempts++;
        }
    };
}
```

---

**Code Examples:**
- [JavaScript Client](../examples/websocket_client.js)
- [Python Client](../examples/websocket_client.py)
- [React Integration](../examples/react_websocket.jsx)

**Next Steps:**
- [Python SDK Reference](python_api.md)
- [Real-time Monitoring Guide](../guides/monitoring_guide.md)
- [Integration Examples](../examples/)