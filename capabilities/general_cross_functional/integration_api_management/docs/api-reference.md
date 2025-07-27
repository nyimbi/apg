# API Reference

Complete API reference for the Integration API Management capability, including REST endpoints, request/response schemas, and code examples.

## Base URL

```
Production: https://api-management.yourcompany.com/api/v1
Staging: https://staging-api-management.yourcompany.com/api/v1
Development: http://localhost:8080/api/v1
```

## Authentication

All API requests require authentication using one of the following methods:

### API Key Authentication

Include the API key in the request header:

```http
X-API-Key: your_api_key_here
```

### Bearer Token Authentication

Include the JWT token in the Authorization header:

```http
Authorization: Bearer your_jwt_token_here
```

### Tenant Identification

Multi-tenant requests require a tenant identifier:

```http
X-Tenant-ID: your_tenant_id
```

## Response Format

All API responses follow a consistent format:

### Success Response

```json
{
  "success": true,
  "data": {
    // Response data
  },
  "metadata": {
    "timestamp": "2025-01-26T10:30:00Z",
    "request_id": "req_123456789",
    "version": "1.0.0"
  }
}
```

### Error Response

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "api_name",
      "reason": "API name must be unique"
    }
  },
  "metadata": {
    "timestamp": "2025-01-26T10:30:00Z",
    "request_id": "req_123456789",
    "version": "1.0.0"
  }
}
```

## API Management Endpoints

### List APIs

Retrieve a list of APIs for the current tenant.

```http
GET /apis
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| page | integer | No | Page number (default: 1) |
| limit | integer | No | Items per page (default: 20, max: 100) |
| status | string | No | Filter by status (draft, active, deprecated, retired) |
| category | string | No | Filter by category |
| search | string | No | Search in API name and description |

#### Example Request

```bash
curl -X GET "https://api-management.yourcompany.com/api/v1/apis?page=1&limit=10&status=active" \
  -H "X-API-Key: your_api_key" \
  -H "X-Tenant-ID: your_tenant_id"
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "apis": [
      {
        "api_id": "api_01h8xz4b9c2d3e4f5g6h7i8j",
        "api_name": "user_management_api",
        "api_title": "User Management API",
        "api_description": "Comprehensive user management service",
        "version": "1.0.0",
        "status": "active",
        "protocol_type": "rest",
        "base_path": "/api/users/v1",
        "upstream_url": "http://user-service:8000",
        "is_public": false,
        "category": "core_business",
        "tags": ["users", "authentication"],
        "created_at": "2025-01-15T09:00:00Z",
        "updated_at": "2025-01-20T14:30:00Z",
        "gateway_url": "https://gateway.yourcompany.com/api/users/v1"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 10,
      "total": 25,
      "total_pages": 3
    }
  }
}
```

### Create API

Register a new API in the system.

```http
POST /apis
```

#### Request Body

```json
{
  "api_name": "product_catalog_api",
  "api_title": "Product Catalog API",
  "api_description": "Product catalog and inventory management",
  "version": "1.0.0",
  "protocol_type": "rest",
  "base_path": "/api/products/v1",
  "upstream_url": "http://product-service:8000",
  "is_public": false,
  "timeout_ms": 30000,
  "retry_attempts": 3,
  "auth_type": "api_key",
  "category": "core_business",
  "tags": ["products", "catalog", "inventory"]
}
```

#### Example Request

```bash
curl -X POST "https://api-management.yourcompany.com/api/v1/apis" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -H "X-Tenant-ID: your_tenant_id" \
  -d '{
    "api_name": "product_catalog_api",
    "api_title": "Product Catalog API",
    "api_description": "Product catalog and inventory management",
    "version": "1.0.0",
    "protocol_type": "rest",
    "base_path": "/api/products/v1",
    "upstream_url": "http://product-service:8000",
    "auth_type": "api_key",
    "category": "core_business",
    "tags": ["products", "catalog"]
  }'
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "api_id": "api_01h8xz5c9d2e3f4g5h6i7j8k",
    "api_name": "product_catalog_api",
    "status": "draft",
    "gateway_url": "https://gateway.yourcompany.com/api/products/v1",
    "created_at": "2025-01-26T10:30:00Z"
  }
}
```

### Get API Details

Retrieve detailed information about a specific API.

```http
GET /apis/{api_id}
```

#### Example Request

```bash
curl -X GET "https://api-management.yourcompany.com/api/v1/apis/api_01h8xz4b9c2d3e4f5g6h7i8j" \
  -H "X-API-Key: your_api_key" \
  -H "X-Tenant-ID: your_tenant_id"
```

### Update API

Update an existing API configuration.

```http
PUT /apis/{api_id}
```

#### Example Request

```bash
curl -X PUT "https://api-management.yourcompany.com/api/v1/apis/api_01h8xz4b9c2d3e4f5g6h7i8j" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -H "X-Tenant-ID: your_tenant_id" \
  -d '{
    "api_description": "Updated product catalog and inventory management",
    "timeout_ms": 45000,
    "tags": ["products", "catalog", "inventory", "updated"]
  }'
```

### Activate API

Activate an API to make it available to consumers.

```http
POST /apis/{api_id}/activate
```

#### Example Request

```bash
curl -X POST "https://api-management.yourcompany.com/api/v1/apis/api_01h8xz4b9c2d3e4f5g6h7i8j/activate" \
  -H "X-API-Key: your_api_key" \
  -H "X-Tenant-ID: your_tenant_id"
```

### Deprecate API

Mark an API as deprecated with migration timeline.

```http
POST /apis/{api_id}/deprecate
```

#### Request Body

```json
{
  "migration_timeline": "6 months",
  "migration_note": "Please migrate to v2.0 by July 2025",
  "replacement_api_id": "api_01h8xz6d9e2f3g4h5i6j7k8l"
}
```

### Delete API

Remove an API from the system (soft delete).

```http
DELETE /apis/{api_id}
```

## Consumer Management Endpoints

### List Consumers

Retrieve a list of API consumers.

```http
GET /consumers
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "consumers": [
      {
        "consumer_id": "con_01h8xz7e9f2g3h4i5j6k7l8m",
        "consumer_name": "mobile_app_client",
        "organization": "Acme Mobile Team",
        "contact_email": "mobile-dev@acme.com",
        "contact_name": "Jane Developer",
        "status": "active",
        "created_at": "2025-01-10T08:00:00Z",
        "global_rate_limit": 10000,
        "portal_access": true
      }
    ]
  }
}
```

### Create Consumer

Register a new API consumer.

```http
POST /consumers
```

#### Request Body

```json
{
  "consumer_name": "web_app_client",
  "organization": "Acme Web Team",
  "contact_email": "web-dev@acme.com",
  "contact_name": "John Developer",
  "description": "Web application client for customer portal",
  "global_rate_limit": 5000,
  "portal_access": true,
  "webhook_url": "https://web-app.acme.com/webhooks/api-events"
}
```

### Approve Consumer

Approve a pending consumer registration.

```http
POST /consumers/{consumer_id}/approve
```

#### Request Body

```json
{
  "approval_note": "Approved after security review",
  "custom_rate_limit": 15000
}
```

### Generate API Key

Generate a new API key for a consumer.

```http
POST /consumers/{consumer_id}/api-keys
```

#### Request Body

```json
{
  "key_name": "production_key",
  "scopes": ["read", "write"],
  "allowed_apis": ["api_01h8xz4b9c2d3e4f5g6h7i8j"],
  "expires_at": "2026-01-26T10:30:00Z"
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "key_id": "key_01h8xz8f9g2h3i4j5k6l7m8n",
    "api_key": "ak_live_1234567890abcdef1234567890abcdef",
    "key_prefix": "ak_live_1234",
    "created_at": "2025-01-26T10:30:00Z",
    "expires_at": "2026-01-26T10:30:00Z"
  }
}
```

### List API Keys

Retrieve API keys for a consumer.

```http
GET /consumers/{consumer_id}/api-keys
```

### Revoke API Key

Revoke an API key.

```http
DELETE /consumers/{consumer_id}/api-keys/{key_id}
```

## Policy Management Endpoints

### List Policies

Retrieve policies for an API.

```http
GET /apis/{api_id}/policies
```

### Create Policy

Create a new policy for an API.

```http
POST /apis/{api_id}/policies
```

#### Rate Limiting Policy Example

```json
{
  "policy_name": "api_rate_limit",
  "policy_type": "rate_limiting",
  "config": {
    "requests_per_minute": 1000,
    "requests_per_hour": 10000,
    "burst_size": 100,
    "key_extraction": "consumer_id",
    "violation_action": "reject"
  },
  "execution_order": 100,
  "enabled": true
}
```

#### Transformation Policy Example

```json
{
  "policy_name": "request_transformation",
  "policy_type": "transformation",
  "config": {
    "request_transformations": [
      {
        "condition": "method == 'POST'",
        "transformations": [
          {
            "type": "add_header",
            "header": "X-Source",
            "value": "api_gateway"
          }
        ]
      }
    ]
  },
  "execution_order": 200
}
```

### Update Policy

Update an existing policy.

```http
PUT /apis/{api_id}/policies/{policy_id}
```

### Delete Policy

Remove a policy from an API.

```http
DELETE /apis/{api_id}/policies/{policy_id}
```

## Analytics Endpoints

### Get Usage Statistics

Retrieve usage statistics for APIs.

```http
GET /analytics/usage
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| start_date | string | Yes | Start date (ISO 8601) |
| end_date | string | Yes | End date (ISO 8601) |
| api_id | string | No | Filter by specific API |
| consumer_id | string | No | Filter by specific consumer |
| granularity | string | No | hour, day, week, month (default: day) |

#### Example Request

```bash
curl -X GET "https://api-management.yourcompany.com/api/v1/analytics/usage?start_date=2025-01-01T00:00:00Z&end_date=2025-01-26T23:59:59Z&granularity=day" \
  -H "X-API-Key: your_api_key" \
  -H "X-Tenant-ID: your_tenant_id"
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "summary": {
      "total_requests": 1250000,
      "total_errors": 12500,
      "error_rate": 0.01,
      "avg_response_time_ms": 145,
      "p95_response_time_ms": 350,
      "unique_consumers": 25
    },
    "time_series": [
      {
        "timestamp": "2025-01-25T00:00:00Z",
        "requests": 48000,
        "errors": 480,
        "avg_response_time_ms": 142
      }
    ],
    "top_apis": [
      {
        "api_id": "api_01h8xz4b9c2d3e4f5g6h7i8j",
        "api_name": "user_management_api",
        "requests": 320000,
        "error_rate": 0.008
      }
    ],
    "top_consumers": [
      {
        "consumer_id": "con_01h8xz7e9f2g3h4i5j6k7l8m",
        "consumer_name": "mobile_app_client",
        "requests": 180000,
        "error_rate": 0.012
      }
    ]
  }
}
```

### Get Error Analysis

Retrieve detailed error analysis.

```http
GET /analytics/errors
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "error_summary": {
      "total_errors": 12500,
      "error_rate": 0.01,
      "top_error_codes": [
        { "code": 429, "count": 4500, "percentage": 36 },
        { "code": 500, "count": 3200, "percentage": 25.6 },
        { "code": 404, "count": 2800, "percentage": 22.4 }
      ]
    },
    "error_trends": [
      {
        "timestamp": "2025-01-25T00:00:00Z",
        "error_count": 480,
        "error_rate": 0.01
      }
    ]
  }
}
```

### Get Performance Metrics

Retrieve performance metrics and latency analysis.

```http
GET /analytics/performance
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "latency_metrics": {
      "avg_response_time_ms": 145,
      "p50_response_time_ms": 120,
      "p95_response_time_ms": 350,
      "p99_response_time_ms": 580
    },
    "throughput_metrics": {
      "requests_per_second": 578,
      "peak_rps": 1200,
      "total_requests": 1250000
    },
    "performance_trends": [
      {
        "timestamp": "2025-01-25T00:00:00Z",
        "avg_response_time_ms": 142,
        "requests_per_second": 556
      }
    ]
  }
}
```

## Health and Monitoring Endpoints

### Health Check

Check the health status of the API Management service.

```http
GET /health
```

#### Example Response

```json
{
  "status": "healthy",
  "timestamp": "2025-01-26T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "database": {
      "status": "healthy",
      "response_time_ms": 15
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 3
    },
    "gateway": {
      "status": "healthy",
      "active_connections": 1250
    }
  }
}
```

### Ready Check

Check if the service is ready to accept requests.

```http
GET /health/ready
```

### Metrics

Retrieve Prometheus-compatible metrics.

```http
GET /metrics
```

## Error Codes

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | OK - Request successful |
| 201 | Created - Resource created successfully |
| 400 | Bad Request - Invalid request parameters |
| 401 | Unauthorized - Authentication required |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 409 | Conflict - Resource already exists |
| 422 | Unprocessable Entity - Validation error |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Server error |
| 502 | Bad Gateway - Upstream service error |
| 503 | Service Unavailable - Service temporarily unavailable |

### Application Error Codes

| Code | Description |
|------|-------------|
| VALIDATION_ERROR | Request validation failed |
| AUTHENTICATION_ERROR | Authentication failed |
| AUTHORIZATION_ERROR | Insufficient permissions |
| RESOURCE_NOT_FOUND | Requested resource not found |
| RESOURCE_CONFLICT | Resource already exists |
| RATE_LIMIT_EXCEEDED | Rate limit exceeded |
| UPSTREAM_ERROR | Upstream service error |
| CONFIGURATION_ERROR | Configuration error |
| INTERNAL_ERROR | Internal server error |

## Rate Limits

### Default Rate Limits

| Endpoint Category | Rate Limit | Window |
|-------------------|------------|---------|
| Authentication | 100 requests | 1 minute |
| Read Operations | 1000 requests | 1 minute |
| Write Operations | 100 requests | 1 minute |
| Analytics | 50 requests | 1 minute |

### Rate Limit Headers

All responses include rate limit information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995260
X-RateLimit-Window: 60
```

## SDKs and Libraries

### Python SDK

```python
from integration_api_management import APIManagementClient

# Initialize client
client = APIManagementClient(
    base_url="https://api-management.yourcompany.com",
    api_key="your_api_key",
    tenant_id="your_tenant_id"
)

# Create API
api = await client.apis.create({
    "api_name": "example_api",
    "api_title": "Example API",
    "base_path": "/example",
    "upstream_url": "http://example-service:8000"
})

# List APIs
apis = await client.apis.list(status="active")

# Get analytics
analytics = await client.analytics.get_usage(
    start_date="2025-01-01T00:00:00Z",
    end_date="2025-01-31T23:59:59Z"
)
```

### JavaScript SDK

```javascript
import { APIManagementClient } from '@datacraft/api-management-sdk';

// Initialize client
const client = new APIManagementClient({
  baseUrl: 'https://api-management.yourcompany.com',
  apiKey: 'your_api_key',
  tenantId: 'your_tenant_id'
});

// Create API
const api = await client.apis.create({
  apiName: 'example_api',
  apiTitle: 'Example API',
  basePath: '/example',
  upstreamUrl: 'http://example-service:8000'
});

// List APIs
const apis = await client.apis.list({ status: 'active' });

// Get analytics
const analytics = await client.analytics.getUsage({
  startDate: '2025-01-01T00:00:00Z',
  endDate: '2025-01-31T23:59:59Z'
});
```

### cURL Examples

See individual endpoint sections for detailed cURL examples.

## Webhooks

The API Management platform supports webhooks for real-time event notifications.

### Webhook Events

| Event | Description |
|-------|-------------|
| api.created | New API registered |
| api.activated | API activated |
| api.deprecated | API deprecated |
| consumer.registered | New consumer registered |
| consumer.approved | Consumer approved |
| api_key.created | New API key generated |
| api_key.revoked | API key revoked |
| policy.created | New policy created |
| alert.triggered | Alert triggered |

### Webhook Payload

```json
{
  "event": "api.created",
  "timestamp": "2025-01-26T10:30:00Z",
  "data": {
    "api_id": "api_01h8xz4b9c2d3e4f5g6h7i8j",
    "api_name": "example_api",
    "tenant_id": "your_tenant_id"
  },
  "metadata": {
    "webhook_id": "wh_01h8xz9g9h2i3j4k5l6m7n8o",
    "delivery_attempt": 1
  }
}
```

## Support

For API support:

- **Documentation**: [https://docs.datacraft.co.ke/apg/api-reference](https://docs.datacraft.co.ke/apg/api-reference)
- **API Issues**: [api-support@datacraft.co.ke](mailto:api-support@datacraft.co.ke)
- **Developer Forum**: [https://community.datacraft.co.ke](https://community.datacraft.co.ke)
- **Status Page**: [https://status.datacraft.co.ke](https://status.datacraft.co.ke)