# APG Capability Registry - API Guide

Complete API reference for the APG Capability Registry service with examples, authentication, and best practices.

## 📋 Table of Contents

- [Authentication](#authentication)
- [Base URL and Headers](#base-url-and-headers)
- [Response Format](#response-format)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Capability API](#capability-api)
- [Composition API](#composition-api)
- [Registry Management API](#registry-management-api)
- [Analytics API](#analytics-api)
- [Mobile API](#mobile-api)
- [WebSocket API](#websocket-api)
- [Webhook Integration](#webhook-integration)
- [Code Examples](#code-examples)

## 🔐 Authentication

### JWT Token Authentication

All API requests require a valid JWT token in the Authorization header:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Getting an Access Token

```http
POST /api/auth/token
Content-Type: application/json

{
    "tenant_id": "your-tenant-id",
    "client_id": "your-client-id",
    "client_secret": "your-client-secret"
}
```

**Response:**
```json
{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 3600,
    "scope": "registry:read registry:write"
}
```

## 🌐 Base URL and Headers

### Base URL
```
Production: https://api.apg.datacraft.co.ke
Staging: https://staging-api.apg.datacraft.co.ke
Development: http://localhost:8000
```

### Required Headers
```http
Authorization: Bearer {access_token}
Content-Type: application/json
Accept: application/json
X-Tenant-ID: {tenant_id}
```

### Optional Headers
```http
X-Request-ID: {unique_request_id}
X-API-Version: 1.0
User-Agent: YourApp/1.0
```

## 📄 Response Format

### Standard Response Structure

All API responses follow this consistent format:

```json
{
    "success": true,
    "message": "Operation completed successfully",
    "data": {
        "capability_id": "01HN123ABC...",
        "capability_name": "User Management"
    },
    "errors": [],
    "meta": {
        "request_id": "req_123",
        "processing_time_ms": 45
    },
    "timestamp": "2025-01-15T10:30:00Z"
}
```

### Paginated Response Structure

```json
{
    "items": [...],
    "total_count": 150,
    "page": 1,
    "per_page": 25,
    "total_pages": 6,
    "has_next": true,
    "has_prev": false
}
```

## ❌ Error Handling

### Error Response Format

```json
{
    "success": false,
    "message": "Validation error",
    "data": null,
    "errors": [
        "capability_code is required",
        "version must be a valid semantic version"
    ],
    "meta": {
        "error_code": "VALIDATION_ERROR",
        "request_id": "req_123"
    },
    "timestamp": "2025-01-15T10:30:00Z"
}
```

### Common HTTP Status Codes

| Status Code | Description | Example |
|-------------|-------------|---------|
| `200` | Success | Operation completed |
| `201` | Created | Resource created |
| `400` | Bad Request | Invalid input data |
| `401` | Unauthorized | Invalid or missing token |
| `403` | Forbidden | Insufficient permissions |
| `404` | Not Found | Resource not found |
| `409` | Conflict | Resource already exists |
| `422` | Unprocessable Entity | Validation failed |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server error |

### Error Codes

| Error Code | Description |
|------------|-------------|
| `VALIDATION_ERROR` | Input validation failed |
| `RESOURCE_NOT_FOUND` | Requested resource not found |
| `DUPLICATE_RESOURCE` | Resource already exists |
| `DEPENDENCY_ERROR` | Dependency validation failed |
| `COMPOSITION_ERROR` | Composition validation failed |
| `AUTHENTICATION_ERROR` | Authentication failed |
| `AUTHORIZATION_ERROR` | Authorization failed |
| `RATE_LIMIT_ERROR` | Rate limit exceeded |

## 🚥 Rate Limiting

### Rate Limits

| Endpoint Type | Limit | Window |
|---------------|-------|---------|
| Read Operations | 1000 requests | per hour |
| Write Operations | 100 requests | per hour |
| Search Operations | 500 requests | per hour |
| Analytics | 200 requests | per hour |

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642233600
X-RateLimit-Window: 3600
```

## 🔧 Capability API

### List Capabilities

Get a paginated list of capabilities with filtering and search.

```http
GET /api/capabilities
```

**Query Parameters:**
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `search` | string | Search query | `user management` |
| `category` | string | Category filter | `foundation_infrastructure` |
| `status` | string | Status filter | `active` |
| `min_quality_score` | float | Minimum quality score | `0.8` |
| `page` | integer | Page number (default: 1) | `2` |
| `per_page` | integer | Items per page (default: 25, max: 100) | `50` |

**Example Request:**
```http
GET /api/capabilities?search=email&category=communication&page=1&per_page=10
Authorization: Bearer {token}
```

**Example Response:**
```json
{
    "items": [
        {
            "capability_id": "01HN123ABC...",
            "capability_code": "EMAIL_SERVICE",
            "capability_name": "Email Service",
            "description": "SMTP email service with templates",
            "version": "1.2.0",
            "category": "communication",
            "status": "active",
            "quality_score": 0.92,
            "popularity_score": 0.85,
            "usage_count": 147,
            "created_at": "2025-01-15T10:30:00Z"
        }
    ],
    "total_count": 1,
    "page": 1,
    "per_page": 10,
    "total_pages": 1,
    "has_next": false,
    "has_prev": false
}
```

### Get Capability Details

Retrieve detailed information about a specific capability.

```http
GET /api/capabilities/{capability_id}
```

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `capability_id` | string | Unique capability identifier |

**Example Request:**
```http
GET /api/capabilities/01HN123ABC...
Authorization: Bearer {token}
```

**Example Response:**
```json
{
    "success": true,
    "message": "Capability retrieved successfully",
    "data": {
        "capability_id": "01HN123ABC...",
        "capability_code": "EMAIL_SERVICE",
        "capability_name": "Email Service",
        "description": "SMTP email service with templates",
        "long_description": "Comprehensive email service...",
        "version": "1.2.0",
        "category": "communication",
        "subcategory": "messaging",
        "status": "active",
        "multi_tenant": true,
        "audit_enabled": true,
        "security_integration": true,
        "performance_optimized": false,
        "ai_enhanced": false,
        "target_users": ["developers", "administrators"],
        "business_value": "Enables automated email communication",
        "use_cases": ["notifications", "marketing", "alerts"],
        "industry_focus": ["technology", "retail"],
        "composition_keywords": ["email", "smtp", "templates"],
        "provides_services": ["email_send", "template_management"],
        "data_models": ["EmailMessage", "EmailTemplate"],
        "api_endpoints": ["/api/email/send", "/api/email/templates"],
        "file_path": "/capabilities/communication/email_service.py",
        "module_path": "capabilities.communication.email_service",
        "documentation_path": "/docs/communication/email_service.md",
        "repository_url": "https://github.com/datacraft/email-service",
        "complexity_score": 3.2,
        "quality_score": 0.92,
        "popularity_score": 0.85,
        "usage_count": 147,
        "created_at": "2025-01-15T10:30:00Z",
        "updated_at": "2025-01-15T14:20:00Z",
        "metadata": {
            "smtp_server": "smtp.example.com",
            "supported_protocols": ["SMTP", "SMTPS"]
        }
    }
}
```

### Create Capability

Register a new capability in the registry.

```http
POST /api/capabilities
```

**Request Body:**
```json
{
    "capability_code": "PAYMENT_GATEWAY",
    "capability_name": "Payment Gateway",
    "description": "Secure payment processing service",
    "long_description": "Comprehensive payment gateway supporting multiple providers...",
    "category": "financial_services",
    "subcategory": "payments",
    "version": "1.0.0",
    "target_users": ["developers", "business_users"],
    "business_value": "Enables secure online payments",
    "use_cases": ["e_commerce", "subscription_billing", "marketplace"],
    "industry_focus": ["retail", "finance", "technology"],
    "composition_keywords": ["payment", "gateway", "secure", "billing"],
    "provides_services": ["payment_processing", "refund_management"],
    "data_models": ["Payment", "Transaction", "Refund"],
    "api_endpoints": ["/api/payments/charge", "/api/payments/refund"],
    "file_path": "/capabilities/financial/payment_gateway.py",
    "module_path": "capabilities.financial.payment_gateway",
    "documentation_path": "/docs/financial/payment_gateway.md",
    "repository_url": "https://github.com/datacraft/payment-gateway",
    "multi_tenant": true,
    "audit_enabled": true,
    "security_integration": true,
    "performance_optimized": true,
    "ai_enhanced": false,
    "complexity_score": 4.5,
    "metadata": {
        "supported_providers": ["stripe", "paypal", "square"],
        "currencies": ["USD", "EUR", "GBP"],
        "pci_compliant": true
    }
}
```

**Example Response:**
```json
{
    "success": true,
    "message": "Capability created successfully",
    "data": {
        "capability_id": "01HN456DEF...",
        "capability_code": "PAYMENT_GATEWAY",
        "capability_name": "Payment Gateway",
        "status": "discovered",
        "quality_score": 0.0,
        "created_at": "2025-01-15T15:30:00Z"
    }
}
```

### Update Capability

Update an existing capability.

```http
PUT /api/capabilities/{capability_id}
```

**Request Body:**
```json
{
    "description": "Updated payment processing service with new features",
    "quality_score": 0.95,
    "performance_optimized": true,
    "metadata": {
        "supported_providers": ["stripe", "paypal", "square", "adyen"],
        "new_feature": "cryptocurrency_support"
    }
}
```

**Example Response:**
```json
{
    "success": true,
    "message": "Capability updated successfully",
    "data": {
        "capability_id": "01HN456DEF...",
        "updated_fields": ["description", "quality_score", "performance_optimized", "metadata"],
        "updated_at": "2025-01-15T16:45:00Z"
    }
}
```

### Delete Capability

Remove a capability from the registry.

```http
DELETE /api/capabilities/{capability_id}
```

**Example Response:**
```json
{
    "success": true,
    "message": "Capability deleted successfully",
    "data": {
        "capability_id": "01HN456DEF...",
        "deleted_at": "2025-01-15T17:00:00Z"
    }
}
```

## 🔗 Composition API

### List Compositions

Get a paginated list of capability compositions.

```http
GET /api/compositions
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `search` | string | Search query |
| `composition_type` | string | Type filter |
| `validation_status` | string | Validation status filter |
| `is_template` | boolean | Template filter |
| `page` | integer | Page number |
| `per_page` | integer | Items per page |

**Example Request:**
```http
GET /api/compositions?composition_type=enterprise_portal&validation_status=valid
Authorization: Bearer {token}
```

### Create Composition

Create a new capability composition.

```http
POST /api/compositions
```

**Request Body:**
```json
{
    "name": "E-commerce Platform",
    "description": "Complete e-commerce platform with payments and user management",
    "composition_type": "ecommerce_platform",
    "capability_ids": [
        "01HN123ABC...", // User Management
        "01HN456DEF...", // Payment Gateway
        "01HN789GHI..."  // Product Catalog
    ],
    "business_requirements": [
        "user_authentication",
        "payment_processing",
        "product_management",
        "order_management"
    ],
    "compliance_requirements": ["pci_dss", "gdpr"],
    "target_users": ["customers", "administrators", "merchants"],
    "deployment_strategy": "cloud_native",
    "is_template": false,
    "is_public": false,
    "configuration": {
        "load_balancing": "round_robin",
        "scaling_policy": "auto",
        "monitoring_enabled": true
    },
    "environment_settings": {
        "environment": "production",
        "region": "us-east-1",
        "backup_enabled": true
    }
}
```

**Example Response:**
```json
{
    "success": true,
    "message": "Composition created successfully",
    "data": {
        "composition_id": "01HN999XYZ...",
        "name": "E-commerce Platform",
        "validation_status": "pending",
        "estimated_complexity": 7.8,
        "estimated_cost": 1250.00,
        "created_at": "2025-01-15T18:30:00Z"
    }
}
```

### Validate Composition

Validate a capability composition before creation.

```http
POST /api/compositions/validate
```

**Request Body:**
```json
{
    "capability_ids": [
        "01HN123ABC...",
        "01HN456DEF...",
        "01HN789GHI..."
    ]
}
```

**Example Response:**
```json
{
    "success": true,
    "message": "Composition validated successfully",
    "data": {
        "is_valid": true,
        "validation_score": 0.92,
        "compatibility_score": 0.88,
        "performance_impact": {
            "estimated_response_time_ms": 250,
            "estimated_throughput": 1000,
            "resource_requirements": {
                "cpu_cores": 4.0,
                "memory_gb": 8.0,
                "storage_gb": 100
            }
        },
        "dependencies": [
            {
                "source": "01HN123ABC...",
                "target": "01HN789GHI...",
                "type": "required",
                "satisfied": true
            }
        ],
        "conflicts": [],
        "recommendations": [
            {
                "type": "optimization",
                "title": "Enable Caching",
                "description": "Add Redis caching to improve performance",
                "priority": "medium",
                "estimated_improvement": "30% faster response time"
            }
        ],
        "warnings": [
            {
                "type": "performance",
                "message": "High memory usage detected",
                "suggestion": "Consider optimizing data models"
            }
        ]
    }
}
```

## 🏥 Registry Management API

### Registry Health Check

Get registry health status and metrics.

```http
GET /api/registry/health
```

**Example Response:**
```json
{
    "success": true,
    "message": "Registry health retrieved successfully",
    "data": {
        "status": "healthy",
        "version": "1.0.0",
        "uptime_seconds": 86400,
        "database": {
            "status": "connected",
            "response_time_ms": 15,
            "connection_pool": {
                "active": 5,
                "idle": 15,
                "max": 20
            }
        },
        "cache": {
            "status": "connected",
            "hit_rate": 0.85,
            "memory_usage_mb": 512
        },
        "external_services": {
            "apg_composition_engine": {
                "status": "healthy",
                "response_time_ms": 45
            },
            "apg_discovery_service": {
                "status": "healthy",
                "response_time_ms": 32
            }
        },
        "metrics": {
            "total_capabilities": 145,
            "active_capabilities": 138,
            "total_compositions": 67,
            "requests_per_minute": 234,
            "error_rate": 0.02
        }
    }
}
```

### Registry Dashboard Data

Get dashboard data for registry overview.

```http
GET /api/registry/dashboard
```

**Example Response:**
```json
{
    "success": true,
    "message": "Dashboard data retrieved successfully",
    "data": {
        "total_capabilities": 145,
        "active_capabilities": 138,
        "total_compositions": 67,
        "total_versions": 312,
        "registry_health_score": 0.95,
        "avg_quality_score": 0.86,
        "recent_capabilities": [
            {
                "capability_id": "01HN123ABC...",
                "capability_name": "New Service",
                "created_at": "2025-01-15T12:00:00Z"
            }
        ],
        "recent_compositions": [
            {
                "composition_id": "01HN999XYZ...",
                "name": "New Platform",
                "created_at": "2025-01-15T13:00:00Z"
            }
        ],
        "category_stats": [
            {"category": "Foundation Infrastructure", "count": 45},
            {"category": "Business Operations", "count": 32},
            {"category": "Analytics & Intelligence", "count": 28}
        ],
        "marketplace_enabled": true,
        "published_capabilities": 23,
        "pending_submissions": 5
    }
}
```

### Sync Registry

Trigger registry synchronization with APG ecosystem.

```http
POST /api/registry/sync
```

**Request Body:**
```json
{
    "force_full_sync": false
}
```

**Example Response:**
```json
{
    "success": true,
    "message": "Registry sync completed successfully",
    "data": {
        "capabilities_synced": 12,
        "compositions_synced": 5,
        "discovery_updates": 18,
        "composition_updates": 7,
        "sync_duration_ms": 2340,
        "errors": []
    }
}
```

## 📊 Analytics API

### Usage Analytics

Get capability usage analytics and metrics.

```http
GET /api/analytics/usage
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `start_date` | datetime | Start date (ISO format) |
| `end_date` | datetime | End date (ISO format) |
| `capability_id` | string | Specific capability ID |

**Example Request:**
```http
GET /api/analytics/usage?start_date=2025-01-01T00:00:00Z&end_date=2025-01-15T23:59:59Z
Authorization: Bearer {token}
```

**Example Response:**
```json
{
    "success": true,
    "message": "Usage analytics retrieved successfully",
    "data": {
        "summary": {
            "total_requests": 15420,
            "unique_capabilities": 89,
            "avg_response_time_ms": 156,
            "error_rate": 0.018
        },
        "top_capabilities": [
            {
                "capability_id": "01HN123ABC...",
                "capability_name": "User Management",
                "usage_count": 3420,
                "avg_response_time_ms": 89
            }
        ],
        "usage_by_category": [
            {
                "category": "foundation_infrastructure",
                "usage_count": 8945,
                "percentage": 58.0
            }
        ],
        "time_series": [
            {
                "timestamp": "2025-01-15T00:00:00Z",
                "requests": 1234,
                "errors": 23
            }
        ]
    }
}
```

### Performance Analytics

Get performance metrics and analytics.

```http
GET /api/analytics/performance
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `metric_type` | string | Metric type filter |
| `time_range` | string | Time range (1d, 7d, 30d, 90d) |

**Example Response:**
```json
{
    "success": true,
    "message": "Performance analytics retrieved successfully",
    "data": {
        "response_times": {
            "p50": 89,
            "p95": 234,
            "p99": 567,
            "avg": 156
        },
        "throughput": {
            "requests_per_second": 25.6,
            "peak_rps": 89.2
        },
        "error_rates": {
            "total_errors": 278,
            "error_rate": 0.018,
            "by_type": {
                "validation_error": 145,
                "timeout_error": 67,
                "server_error": 66
            }
        },
        "resource_usage": {
            "cpu_usage_percent": 45.2,
            "memory_usage_percent": 62.8,
            "disk_usage_percent": 23.1
        }
    }
}
```

## 📱 Mobile API

### Mobile Capabilities

Get mobile-optimized capability list.

```http
GET /api/mobile/capabilities
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `category` | string | Category filter |
| `limit` | integer | Limit (max 100) |
| `offset` | integer | Offset |

**Example Response:**
```json
{
    "success": true,
    "message": "Mobile capabilities retrieved successfully",
    "data": {
        "capabilities": [
            {
                "capability_id": "01HN123ABC...",
                "name": "User Management",
                "category": "foundation_infrastructure",
                "description": "User management service",
                "icon_url": "https://cdn.example.com/icons/user.png",
                "is_offline_capable": true,
                "mobile_optimized": true
            }
        ],
        "total": 1,
        "has_more": false
    }
}
```

### Mobile Sync

Sync offline data for mobile applications.

```http
POST /api/mobile/sync
```

**Request Body:**
```json
{
    "force_full_sync": false
}
```

**Example Response:**
```json
{
    "success": true,
    "message": "Mobile sync completed successfully",
    "data": {
        "capabilities_synced": 15,
        "compositions_synced": 8,
        "last_sync_timestamp": "2025-01-15T18:30:00Z",
        "sync_size_kb": 1234,
        "is_incremental": true
    }
}
```

## 🔌 WebSocket API

### Connection

Connect to WebSocket for real-time updates.

```javascript
const ws = new WebSocket('wss://api.apg.datacraft.co.ke/api/ws/your-tenant-id');
```

### Message Format

All WebSocket messages follow this format:

```json
{
    "type": "message_type",
    "data": {
        "key": "value"
    },
    "timestamp": "2025-01-15T18:30:00Z",
    "session_id": "optional_session_id"
}
```

### Message Types

#### Connection Messages
```json
{
    "type": "connection",
    "data": {
        "connection_id": "conn_123",
        "tenant_id": "your-tenant",
        "message": "Connected to APG Registry WebSocket"
    }
}
```

#### Registry Updates
```json
{
    "type": "registry_update",
    "data": {
        "event_type": "capability_created",
        "data": {
            "capability_id": "01HN123ABC...",
            "capability_name": "New Service"
        }
    }
}
```

#### Subscription
```json
{
    "type": "subscribe",
    "data": {
        "events": ["capability.created", "composition.updated"]
    }
}
```

### Event Types

| Event Type | Description |
|------------|-------------|
| `capability.created` | New capability registered |
| `capability.updated` | Capability modified |
| `capability.deleted` | Capability removed |
| `composition.created` | New composition created |
| `composition.updated` | Composition modified |
| `composition.validated` | Composition validation completed |
| `registry.synced` | Registry sync completed |

## 🔗 Webhook Integration

### Webhook Events

The registry can send webhook events to configured endpoints.

### Event Format

```json
{
    "event_id": "01HN123ABC...",
    "event_type": "capability.created",
    "resource_type": "capability",
    "resource_id": "01HN456DEF...",
    "action": "created",
    "tenant_id": "your-tenant",
    "user_id": "user_123",
    "payload": {
        "capability": {
            "capability_id": "01HN456DEF...",
            "capability_name": "New Service"
        }
    },
    "timestamp": "2025-01-15T18:30:00Z"
}
```

### List Webhook Events

Get a history of webhook events.

```http
GET /api/webhooks/events
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `event_type` | string | Event type filter |
| `resource_type` | string | Resource type filter |
| `since` | datetime | Events since timestamp |

## 💻 Code Examples

### Python SDK Example

```python
import asyncio
import httpx
from typing import Dict, List, Any

class APGRegistryClient:
    def __init__(self, base_url: str, access_token: str, tenant_id: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Tenant-ID": tenant_id
        }
    
    async def create_capability(self, capability_data: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/capabilities",
                json=capability_data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    async def search_capabilities(self, query: str, **filters) -> List[Dict[str, Any]]:
        params = {"search": query, **filters}
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/capabilities",
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["items"]
    
    async def create_composition(self, composition_data: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/compositions",
                json=composition_data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()

# Usage example
async def main():
    client = APGRegistryClient(
        base_url="https://api.apg.datacraft.co.ke",
        access_token="your-token",
        tenant_id="your-tenant"
    )
    
    # Create capability
    capability = await client.create_capability({
        "capability_code": "NOTIFICATION_SERVICE",
        "capability_name": "Notification Service",
        "description": "Multi-channel notification service",
        "category": "communication",
        "version": "1.0.0"
    })
    
    print(f"Created capability: {capability['data']['capability_id']}")
    
    # Search capabilities
    results = await client.search_capabilities("notification")
    for cap in results:
        print(f"Found: {cap['capability_name']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript Example

```javascript
class APGRegistryClient {
    constructor(baseUrl, accessToken, tenantId) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${accessToken}`,
            'Content-Type': 'application/json',
            'X-Tenant-ID': tenantId
        };
    }
    
    async createCapability(capabilityData) {
        const response = await fetch(`${this.baseUrl}/api/capabilities`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(capabilityData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async searchCapabilities(query, filters = {}) {
        const params = new URLSearchParams({
            search: query,
            ...filters
        });
        
        const response = await fetch(`${this.baseUrl}/api/capabilities?${params}`, {
            headers: this.headers
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data.items;
    }
    
    // WebSocket connection
    connectWebSocket() {
        const ws = new WebSocket(`wss://api.apg.datacraft.co.ke/api/ws/${this.tenantId}`);
        
        ws.onopen = () => {
            console.log('WebSocket connected');
            
            // Subscribe to events
            ws.send(JSON.stringify({
                type: 'subscribe',
                data: {
                    events: ['capability.created', 'composition.updated']
                }
            }));
        };
        
        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        return ws;
    }
    
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'registry_update':
                console.log('Registry update:', message.data);
                break;
            case 'connection':
                console.log('Connected:', message.data.message);
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    }
}

// Usage example
const client = new APGRegistryClient(
    'https://api.apg.datacraft.co.ke',
    'your-token',
    'your-tenant'
);

// Create capability
client.createCapability({
    capability_code: 'SMS_SERVICE',
    capability_name: 'SMS Service',
    description: 'SMS notification service',
    category: 'communication',
    version: '1.0.0'
}).then(result => {
    console.log('Created capability:', result.data.capability_id);
});

// Connect WebSocket
const ws = client.connectWebSocket();
```

### cURL Examples

#### Create Capability
```bash
curl -X POST "https://api.apg.datacraft.co.ke/api/capabilities" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: your-tenant" \
  -d '{
    "capability_code": "FILE_STORAGE",
    "capability_name": "File Storage Service",
    "description": "Cloud file storage with versioning",
    "category": "infrastructure",
    "version": "1.0.0",
    "provides_services": ["file_upload", "file_download", "file_versioning"]
  }'
```

#### Search Capabilities
```bash
curl -X GET "https://api.apg.datacraft.co.ke/api/capabilities?search=storage&category=infrastructure" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "X-Tenant-ID: your-tenant"
```

#### Validate Composition
```bash
curl -X POST "https://api.apg.datacraft.co.ke/api/compositions/validate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: your-tenant" \
  -d '{
    "capability_ids": ["01HN123ABC...", "01HN456DEF..."]
  }'
```

## 🔧 Best Practices

### Error Handling
```python
async def safe_api_call(client, operation):
    try:
        result = await operation()
        return result
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            # Rate limited - implement backoff
            await asyncio.sleep(60)
            return await operation()
        elif e.response.status_code == 401:
            # Token expired - refresh token
            await client.refresh_token()
            return await operation()
        else:
            # Handle other errors
            error_data = e.response.json()
            print(f"API error: {error_data['message']}")
            raise
```

### Pagination
```python
async def get_all_capabilities(client):
    all_capabilities = []
    page = 1
    
    while True:
        response = await client.get_capabilities(page=page, per_page=100)
        capabilities = response["items"]
        all_capabilities.extend(capabilities)
        
        if not response["has_next"]:
            break
            
        page += 1
    
    return all_capabilities
```

### WebSocket Reconnection
```javascript
class ReconnectingWebSocket {
    constructor(url, options = {}) {
        this.url = url;
        this.options = options;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = options.maxReconnectAttempts || 5;
        this.reconnectDelay = options.reconnectDelay || 1000;
        this.connect();
    }
    
    connect() {
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            if (this.options.onOpen) this.options.onOpen();
        };
        
        this.ws.onmessage = (event) => {
            if (this.options.onMessage) this.options.onMessage(event);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.reconnect();
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    reconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        }
    }
    
    send(data) {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(data);
        }
    }
}
```

## 🔒 Security Best Practices

1. **Always use HTTPS** in production
2. **Store tokens securely** - never in localStorage for web apps
3. **Implement token refresh** logic
4. **Validate all inputs** before sending to API
5. **Use rate limiting** to avoid overwhelming the service
6. **Log API errors** for debugging
7. **Implement proper CORS** for web applications
8. **Use environment variables** for sensitive configuration

---

This comprehensive API guide provides everything needed to integrate with the APG Capability Registry. For additional support, contact: nyimbi@gmail.com