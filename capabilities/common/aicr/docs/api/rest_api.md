# AICR REST API Reference

**Version:** 1.0.0
**Author:** Nyimbi Odero <nyimbi@gmail.com>
**Copyright:** © 2025 Datacraft

## Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Common Patterns](#common-patterns)
4. [Model Management](#model-management)
5. [Inference Operations](#inference-operations)
6. [Pipeline Management](#pipeline-management)
7. [Monitoring & Analytics](#monitoring--analytics)
8. [System Operations](#system-operations)
9. [Error Handling](#error-handling)
10. [Rate Limiting](#rate-limiting)

## API Overview

The AICR REST API provides comprehensive access to all AI Core Framework capabilities through RESTful endpoints. All endpoints follow OpenAPI 3.0 specification and support JSON request/response formats.

### Base URL

```
Production: https://api.datacraft.co.ke/aicr/v1
Development: http://localhost:8080/api/v1
```

### API Versioning

- Current version: `v1`
- Version specified in URL path: `/api/v1/`
- Backward compatibility maintained for major versions

### Content Types

- **Request**: `application/json`
- **Response**: `application/json`
- **File Upload**: `multipart/form-data`
- **Model Download**: `application/octet-stream`

## Authentication

### JWT Token Authentication

All API endpoints require authentication via JWT tokens in the Authorization header:

```http
Authorization: Bearer <jwt_token>
```

### Obtaining Access Tokens

#### Login Endpoint

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user_info": {
    "user_id": "user_123",
    "username": "user@example.com",
    "roles": ["user", "data_scientist"],
    "permissions": ["model:read", "inference:execute"]
  }
}
```

#### Token Refresh

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### OAuth2 Authentication

For third-party integrations, OAuth2 flows are supported:

```http
GET /api/v1/auth/oauth2/authorize?
  client_id=your_client_id&
  response_type=code&
  scope=model:read,inference:execute&
  redirect_uri=https://your-app.com/callback
```

## Common Patterns

### Pagination

Large result sets use cursor-based pagination:

```http
GET /api/v1/models?limit=20&cursor=eyJjcmVhdGVkX2F0IjoiMjAyNC0wMS0xNVQxMDozMDowMFoifQ
```

**Response:**
```json
{
  "data": [...],
  "pagination": {
    "limit": 20,
    "has_next": true,
    "next_cursor": "eyJjcmVhdGVkX2F0IjoiMjAyNC0wMS0xNVQxMDozMDowMFoifQ",
    "total_count": 150
  }
}
```

### Filtering and Sorting

```http
GET /api/v1/models?
  filter[model_type]=classification&
  filter[status]=active&
  sort=created_at:desc,name:asc&
  fields=model_id,name,status,created_at
```

### Async Operations

Long-running operations return operation IDs for status tracking:

```json
{
  "operation_id": "op_123456789",
  "status": "in_progress",
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:35:00Z"
}
```

Check operation status:
```http
GET /api/v1/operations/op_123456789
```

## Model Management

### List Models

Retrieve a paginated list of AI models.

```http
GET /api/v1/models
```

**Query Parameters:**
- `limit` (integer, 1-100): Number of results per page (default: 20)
- `cursor` (string): Pagination cursor
- `filter[model_type]` (string): Filter by model type
- `filter[framework]` (string): Filter by framework
- `filter[status]` (string): Filter by status
- `filter[tag]` (string): Filter by tag
- `sort` (string): Sort fields (e.g., "created_at:desc,name:asc")
- `fields` (string): Comma-separated list of fields to return

**Response:**
```json
{
  "data": [
    {
      "model_id": "mdl_abc123",
      "name": "image_classifier_v1",
      "description": "Convolutional neural network for image classification",
      "model_type": "classification",
      "framework": "pytorch",
      "version": "1.0.0",
      "status": "active",
      "tags": ["computer_vision", "imagenet"],
      "performance_metrics": {
        "accuracy": 0.92,
        "precision": 0.90,
        "recall": 0.89,
        "f1_score": 0.895
      },
      "deployment_info": {
        "deployed": true,
        "endpoint": "https://api.datacraft.co.ke/aicr/v1/models/mdl_abc123/predict",
        "instances": 3
      },
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-16T14:22:00Z"
    }
  ],
  "pagination": {
    "limit": 20,
    "has_next": false,
    "total_count": 15
  }
}
```

### Get Model Details

Retrieve detailed information about a specific model.

```http
GET /api/v1/models/{model_id}
```

**Path Parameters:**
- `model_id` (string, required): Model identifier

**Response:**
```json
{
  "model_id": "mdl_abc123",
  "name": "image_classifier_v1",
  "description": "Convolutional neural network for image classification",
  "model_type": "classification",
  "framework": "pytorch",
  "version": "1.0.0",
  "status": "active",
  "tags": ["computer_vision", "imagenet"],
  "input_schema": {
    "type": "object",
    "properties": {
      "image": {
        "type": "string",
        "description": "Base64 encoded image data",
        "format": "base64"
      }
    },
    "required": ["image"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "predictions": {
        "type": "array",
        "items": {"type": "string"}
      },
      "confidence_scores": {
        "type": "array",
        "items": {"type": "number", "minimum": 0, "maximum": 1}
      }
    }
  },
  "configuration": {
    "batch_size": 32,
    "device": "gpu",
    "num_classes": 1000,
    "input_size": [224, 224, 3]
  },
  "performance_metrics": {
    "accuracy": 0.92,
    "precision": 0.90,
    "recall": 0.89,
    "f1_score": 0.895,
    "latency_p50": 45.2,
    "latency_p95": 89.7,
    "throughput": 2340
  },
  "deployment_info": {
    "deployed": true,
    "endpoint": "https://api.datacraft.co.ke/aicr/v1/models/mdl_abc123/predict",
    "instances": 3,
    "auto_scaling": true,
    "health_status": "healthy"
  },
  "metadata": {
    "owner": "user_123",
    "organization": "datacraft",
    "license": "MIT",
    "size_mb": 245.7,
    "checksum": "sha256:abc123...",
    "training_dataset": "imagenet_2023",
    "training_metrics": {
      "epochs": 50,
      "learning_rate": 0.001,
      "batch_size": 64
    }
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-16T14:22:00Z"
}
```

### Register New Model

Register a new AI model in the system.

```http
POST /api/v1/models
Content-Type: application/json

{
  "name": "sentiment_analyzer_v2",
  "description": "BERT-based sentiment analysis model with enhanced accuracy",
  "model_type": "classification",
  "framework": "pytorch",
  "version": "2.0.0",
  "input_schema": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "Input text for sentiment analysis",
        "maxLength": 512
      }
    },
    "required": ["text"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "sentiment": {
        "type": "string",
        "enum": ["positive", "negative", "neutral"]
      },
      "confidence": {
        "type": "number",
        "minimum": 0,
        "maximum": 1
      }
    }
  },
  "configuration": {
    "max_sequence_length": 512,
    "device": "auto",
    "precision": "fp16"
  },
  "tags": ["nlp", "sentiment", "bert"],
  "metadata": {
    "license": "Apache-2.0",
    "training_dataset": "sentiment140_extended",
    "base_model": "bert-base-uncased"
  }
}
```

**Response:**
```json
{
  "model_id": "mdl_def456",
  "name": "sentiment_analyzer_v2",
  "status": "registered",
  "upload_url": "https://upload.datacraft.co.ke/models/mdl_def456",
  "upload_token": "upload_token_xyz789",
  "upload_expires_at": "2024-01-15T11:30:00Z",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Upload Model Artifacts

Upload model files using the provided upload URL.

```http
POST https://upload.datacraft.co.ke/models/mdl_def456
Authorization: Bearer upload_token_xyz789
Content-Type: multipart/form-data

model_file: <binary_model_file>
config_file: <optional_config_file>
```

**Response:**
```json
{
  "upload_id": "upl_789xyz",
  "status": "completed",
  "files": [
    {
      "filename": "model.pth",
      "size_bytes": 257840123,
      "checksum": "sha256:def456..."
    },
    {
      "filename": "config.json",
      "size_bytes": 1024,
      "checksum": "sha256:ghi789..."
    }
  ],
  "processing_status": "validating"
}
```

### Update Model

Update model metadata and configuration.

```http
PATCH /api/v1/models/{model_id}
Content-Type: application/json

{
  "description": "Updated sentiment analysis model with improved accuracy",
  "version": "2.1.0",
  "performance_metrics": {
    "accuracy": 0.94,
    "precision": 0.92,
    "recall": 0.93,
    "f1_score": 0.925
  },
  "tags": ["nlp", "sentiment", "bert", "production"]
}
```

### Delete Model

Remove a model from the system.

```http
DELETE /api/v1/models/{model_id}
```

**Response:**
```json
{
  "model_id": "mdl_abc123",
  "status": "deletion_initiated",
  "operation_id": "op_delete_model_123",
  "message": "Model deletion initiated. This operation may take several minutes."
}
```

## Inference Operations

### Single Inference

Execute inference for a single input.

```http
POST /api/v1/models/{model_id}/predict
Content-Type: application/json

{
  "input_data": {
    "text": "I love this product! It's amazing."
  },
  "parameters": {
    "confidence_threshold": 0.8,
    "return_probabilities": true
  },
  "output_format": "json",
  "priority": "normal"
}
```

**Response:**
```json
{
  "request_id": "req_abc123",
  "model_id": "mdl_def456",
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
    "processing_time_ms": 23.7
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Batch Inference

Execute inference for multiple inputs in a single request.

```http
POST /api/v1/models/{model_id}/batch_predict
Content-Type: application/json

{
  "inputs": [
    {
      "input_id": "input_1",
      "input_data": {"text": "I love this product!"}
    },
    {
      "input_id": "input_2",
      "input_data": {"text": "This is terrible."}
    },
    {
      "input_id": "input_3",
      "input_data": {"text": "It's okay, nothing special."}
    }
  ],
  "parameters": {
    "confidence_threshold": 0.8,
    "batch_size": 16
  },
  "priority": "normal"
}
```

**Response:**
```json
{
  "batch_id": "batch_xyz789",
  "model_id": "mdl_def456",
  "status": "completed",
  "results": [
    {
      "input_id": "input_1",
      "status": "completed",
      "predictions": {
        "sentiment": "positive",
        "confidence": 0.94
      },
      "processing_time_ms": 19.2
    },
    {
      "input_id": "input_2",
      "status": "completed",
      "predictions": {
        "sentiment": "negative",
        "confidence": 0.89
      },
      "processing_time_ms": 18.7
    },
    {
      "input_id": "input_3",
      "status": "completed",
      "predictions": {
        "sentiment": "neutral",
        "confidence": 0.76
      },
      "processing_time_ms": 20.1
    }
  ],
  "summary": {
    "total_inputs": 3,
    "successful": 3,
    "failed": 0,
    "total_processing_time_ms": 57.9,
    "average_time_ms": 19.3
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Async Inference

For long-running inference tasks, use async endpoints.

```http
POST /api/v1/models/{model_id}/async_predict
Content-Type: application/json

{
  "input_data": {
    "large_dataset": "base64_encoded_data..."
  },
  "parameters": {
    "processing_mode": "comprehensive",
    "quality": "high"
  },
  "callback_url": "https://your-app.com/inference_callback",
  "priority": "low"
}
```

**Response:**
```json
{
  "job_id": "job_async_123",
  "status": "queued",
  "estimated_completion": "2024-01-15T10:45:00Z",
  "status_url": "/api/v1/jobs/job_async_123",
  "created_at": "2024-01-15T10:30:00Z"
}
```

Check async job status:
```http
GET /api/v1/jobs/{job_id}
```

## Pipeline Management

### List Pipelines

```http
GET /api/v1/pipelines
```

**Response:**
```json
{
  "data": [
    {
      "pipeline_id": "pip_abc123",
      "name": "sentiment_training_pipeline",
      "description": "End-to-end sentiment analysis training pipeline",
      "pipeline_type": "training",
      "status": "active",
      "stages": [
        "data_loading",
        "preprocessing",
        "training",
        "validation",
        "deployment"
      ],
      "last_execution": {
        "execution_id": "exec_xyz789",
        "status": "completed",
        "started_at": "2024-01-15T08:00:00Z",
        "completed_at": "2024-01-15T09:45:00Z",
        "duration_minutes": 105
      },
      "schedule": "0 2 * * 0",
      "created_at": "2024-01-10T10:30:00Z"
    }
  ]
}
```

### Execute Pipeline

```http
POST /api/v1/pipelines/{pipeline_id}/execute
Content-Type: application/json

{
  "parameters": {
    "dataset_version": "v2.1",
    "epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 32
  },
  "priority": "high",
  "notifications": {
    "on_completion": ["user@example.com"],
    "on_failure": ["admin@example.com"]
  }
}
```

**Response:**
```json
{
  "execution_id": "exec_new_456",
  "pipeline_id": "pip_abc123",
  "status": "starting",
  "estimated_duration_minutes": 120,
  "estimated_completion": "2024-01-15T12:30:00Z",
  "stages": [
    {
      "stage_name": "data_loading",
      "status": "pending",
      "estimated_duration_minutes": 10
    },
    {
      "stage_name": "preprocessing",
      "status": "pending",
      "estimated_duration_minutes": 20
    }
  ],
  "created_at": "2024-01-15T10:30:00Z"
}
```

## Monitoring & Analytics

### Get System Health

```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12.3,
      "connections": {
        "active": 8,
        "max": 50
      }
    },
    "storage": {
      "status": "healthy",
      "disk_usage_percent": 45.2,
      "available_gb": 1250
    },
    "inference_engine": {
      "status": "healthy",
      "active_models": 15,
      "queue_length": 3,
      "average_latency_ms": 67.4
    },
    "security": {
      "status": "healthy",
      "active_tokens": 145,
      "failed_auth_attempts_1h": 2
    }
  },
  "resource_usage": {
    "cpu_percent": 34.5,
    "memory_percent": 67.2,
    "disk_percent": 45.2,
    "gpu_percent": 78.9
  }
}
```

### Get Metrics

```http
GET /api/v1/metrics?
  metric_name=inference_latency&
  time_range=1h&
  component=inference_engine&
  aggregation=avg
```

**Response:**
```json
{
  "metric_name": "inference_latency",
  "component": "inference_engine",
  "time_range": "1h",
  "aggregation": "avg",
  "data_points": [
    {
      "timestamp": "2024-01-15T09:30:00Z",
      "value": 65.4,
      "labels": {
        "model_type": "classification",
        "framework": "pytorch"
      }
    },
    {
      "timestamp": "2024-01-15T09:35:00Z",
      "value": 67.2,
      "labels": {
        "model_type": "classification",
        "framework": "pytorch"
      }
    }
  ],
  "statistics": {
    "min": 45.2,
    "max": 89.7,
    "avg": 67.4,
    "p50": 65.8,
    "p95": 82.3,
    "p99": 87.1
  }
}
```

### Get Performance Summary

```http
GET /api/v1/analytics/performance?time_range=24h
```

**Response:**
```json
{
  "time_range": "24h",
  "summary": {
    "total_requests": 125430,
    "successful_requests": 124987,
    "failed_requests": 443,
    "error_rate_percent": 0.35,
    "average_latency_ms": 67.4,
    "throughput_rps": 1.45
  },
  "models": [
    {
      "model_id": "mdl_abc123",
      "model_name": "sentiment_analyzer_v2",
      "requests": 45230,
      "average_latency_ms": 23.7,
      "error_rate_percent": 0.12,
      "accuracy": 0.94
    }
  ],
  "frameworks": {
    "pytorch": {
      "requests": 67890,
      "average_latency_ms": 45.2
    },
    "tensorflow": {
      "requests": 34567,
      "average_latency_ms": 89.7
    }
  }
}
```

## System Operations

### Deploy Model

Deploy a registered model for inference.

```http
POST /api/v1/models/{model_id}/deploy
Content-Type: application/json

{
  "deployment_config": {
    "instance_type": "gpu_medium",
    "min_instances": 2,
    "max_instances": 10,
    "auto_scaling": true,
    "health_check_path": "/health"
  },
  "environment": "production",
  "tags": {
    "version": "v2.1",
    "team": "ml_platform"
  }
}
```

**Response:**
```json
{
  "deployment_id": "dep_abc123",
  "model_id": "mdl_def456",
  "status": "deploying",
  "endpoint": "https://api.datacraft.co.ke/aicr/v1/models/mdl_def456/predict",
  "estimated_ready_time": "2024-01-15T10:35:00Z",
  "operation_id": "op_deploy_456"
}
```

### Scale Deployment

Adjust deployment scaling parameters.

```http
PATCH /api/v1/deployments/{deployment_id}/scale
Content-Type: application/json

{
  "min_instances": 3,
  "max_instances": 15,
  "target_cpu_utilization": 70
}
```

### Get Deployment Status

```http
GET /api/v1/deployments/{deployment_id}
```

**Response:**
```json
{
  "deployment_id": "dep_abc123",
  "model_id": "mdl_def456",
  "status": "healthy",
  "endpoint": "https://api.datacraft.co.ke/aicr/v1/models/mdl_def456/predict",
  "instances": {
    "desired": 3,
    "running": 3,
    "healthy": 3
  },
  "performance": {
    "requests_per_second": 145.7,
    "average_latency_ms": 23.4,
    "error_rate_percent": 0.02
  },
  "auto_scaling": {
    "enabled": true,
    "current_cpu_utilization": 45.2,
    "target_cpu_utilization": 70
  },
  "health_checks": {
    "last_check": "2024-01-15T10:29:00Z",
    "status": "passing",
    "consecutive_failures": 0
  }
}
```

## Error Handling

### Standard Error Response

All API errors follow a consistent format:

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "The specified model could not be found",
    "details": {
      "model_id": "mdl_invalid",
      "suggestion": "Verify the model ID and ensure the model exists"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_error_123"
  }
}
```

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `202` - Accepted (async operation)
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `409` - Conflict
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `AUTHENTICATION_REQUIRED` | Valid authentication required | 401 |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions | 403 |
| `MODEL_NOT_FOUND` | Specified model does not exist | 404 |
| `INVALID_INPUT_SCHEMA` | Input data doesn't match schema | 422 |
| `INFERENCE_TIMEOUT` | Inference operation timed out | 408 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `MODEL_DEPLOYMENT_FAILED` | Model deployment unsuccessful | 500 |
| `STORAGE_ERROR` | Storage operation failed | 500 |

### Validation Errors

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "validation_errors": [
        {
          "field": "input_data.text",
          "message": "Field is required",
          "code": "REQUIRED"
        },
        {
          "field": "parameters.confidence_threshold",
          "message": "Value must be between 0 and 1",
          "code": "OUT_OF_RANGE"
        }
      ]
    }
  }
}
```

## Rate Limiting

### Rate Limit Headers

All responses include rate limiting information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
X-RateLimit-Window: 3600
```

### Rate Limit Tiers

| Tier | Requests/Hour | Burst Limit | Inference/Hour |
|------|---------------|-------------|----------------|
| Free | 1,000 | 100 | 100 |
| Basic | 10,000 | 500 | 1,000 |
| Pro | 100,000 | 2,000 | 10,000 |
| Enterprise | Custom | Custom | Custom |

### Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded for your tier",
    "details": {
      "limit": 1000,
      "window_seconds": 3600,
      "reset_at": "2024-01-15T11:00:00Z",
      "upgrade_url": "https://datacraft.co.ke/pricing"
    }
  }
}
```

---

**Interactive API Documentation:**
- Swagger UI: `https://api.datacraft.co.ke/aicr/v1/docs`
- ReDoc: `https://api.datacraft.co.ke/aicr/v1/redoc`
- OpenAPI Spec: `https://api.datacraft.co.ke/aicr/v1/openapi.json`

**Next Steps:**
- [WebSocket API Reference](websocket_api.md)
- [Python SDK Reference](python_api.md)
- [User Guide](../guides/model_management.md)