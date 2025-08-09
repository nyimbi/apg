# APG Metadata Management API Reference

## Overview

The APG Metadata Management capability provides a comprehensive REST API for managing enterprise metadata, data discovery, lineage tracking, and AI-powered classification. This document provides detailed API reference information for all endpoints.

**Base URL:** `http://localhost:5000/api/v1/metadata`
**Authentication:** Bearer Token or API Key
**Content-Type:** `application/json`

## Table of Contents

- [Health & Metrics](#health--metrics)
- [Asset Management](#asset-management)
- [Discovery & Scheduling](#discovery--scheduling)
- [Search & Query](#search--query)
- [Lineage Management](#lineage-management)  
- [AI Classification](#ai-classification)
- [Integration Management](#integration-management)
- [Error Handling](#error-handling)

---

## Health & Metrics

### GET /health
Get service health status and basic metrics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-09T12:00:00Z",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "metrics": {
    "total_assets": 15420,
    "total_lineage_edges": 8934,
    "active_discovery_jobs": 3,
    "classification_accuracy": 0.94
  }
}
```

### GET /metrics
Get detailed performance and usage metrics.

**Response:**
```json
{
  "request_metrics": {
    "total_requests": 50234,
    "avg_response_time_ms": 45.2,
    "error_rate": 0.02,
    "requests_per_minute": 120
  },
  "database_metrics": {
    "connection_pool_size": 10,
    "active_connections": 3,
    "query_avg_time_ms": 12.5
  },
  "cache_metrics": {
    "hit_rate": 0.85,
    "cache_size_mb": 256,
    "eviction_count": 1023
  }
}
```

---

## Asset Management

### GET /assets
List metadata assets with filtering and pagination.

**Query Parameters:**
- `limit` (int): Maximum results (default: 100, max: 1000)
- `offset` (int): Pagination offset (default: 0)
- `asset_type` (string): Filter by asset type
- `source_system` (string): Filter by source system
- `owner` (string): Filter by owner
- `tags` (string): Comma-separated list of tags
- `quality_score_min` (float): Minimum quality score (0-1)
- `created_after` (string): ISO timestamp filter
- `search` (string): Text search across names and descriptions

**Example Request:**
```http
GET /assets?asset_type=table&source_system=postgresql&limit=50&offset=100
```

**Response:**
```json
{
  "assets": [
    {
      "id": "01HN8X3K2M4P5Q6R7S8T9U0V1W",
      "name": "customer_orders",
      "display_name": "Customer Orders",
      "asset_type": "table",
      "source_system": "postgresql",
      "database": "ecommerce",
      "schema": "public",
      "description": "Customer order transactions with payment details",
      "owner": "data_team@company.com",
      "tags": ["orders", "transactions", "pii"],
      "quality_score": 0.92,
      "classification": "CONFIDENTIAL",
      "created_at": "2025-01-01T10:00:00Z",
      "updated_at": "2025-01-09T08:30:00Z",
      "custom_attributes": {
        "row_count": 2500000,
        "size_bytes": 1073741824,
        "partitioned": true
      }
    }
  ],
  "pagination": {
    "total": 15420,
    "limit": 50,
    "offset": 100,
    "has_more": true
  }
}
```

### GET /assets/{asset_id}
Get detailed asset metadata including schema and lineage summary.

**Path Parameters:**
- `asset_id` (string): Unique asset identifier

**Response:**
```json
{
  "id": "01HN8X3K2M4P5Q6R7S8T9U0V1W",
  "name": "customer_orders",
  "display_name": "Customer Orders",
  "asset_type": "table",
  "source_system": "postgresql",
  "description": "Customer order transactions with payment details",
  "columns": [
    {
      "name": "order_id",
      "display_name": "Order ID",
      "data_type": "INTEGER",
      "is_nullable": false,
      "is_primary_key": true,
      "classification": "INTERNAL",
      "description": "Unique order identifier",
      "data_quality": {
        "completeness": 1.0,
        "uniqueness": 1.0,
        "validity": 1.0
      }
    },
    {
      "name": "customer_email",
      "display_name": "Customer Email",
      "data_type": "VARCHAR",
      "is_nullable": false,
      "classification": "PII",
      "description": "Customer email address",
      "data_quality": {
        "completeness": 0.98,
        "uniqueness": 0.89,
        "validity": 0.95
      }
    }
  ],
  "lineage_summary": {
    "upstream_count": 3,
    "downstream_count": 7,
    "depth_upstream": 2,
    "depth_downstream": 4
  },
  "quality_metrics": {
    "overall_score": 0.92,
    "completeness": 0.95,
    "accuracy": 0.91,
    "consistency": 0.89,
    "timeliness": 0.93
  }
}
```

### PUT /assets/{asset_id}
Update asset metadata (partial updates supported).

**Request Body:**
```json
{
  "display_name": "Customer Orders - Updated",
  "description": "Updated description",
  "owner": "new_owner@company.com",
  "tags": ["orders", "transactions", "pii", "high_value"],
  "custom_attributes": {
    "business_critical": true,
    "retention_years": 7
  }
}
```

### DELETE /assets/{asset_id}
Delete asset from metadata catalog.

**Response:** `204 No Content`

---

## Discovery & Scheduling

### POST /discovery/schedules
Create a new discovery schedule.

**Request Body:**
```json
{
  "name": "PostgreSQL Production Discovery",
  "description": "Daily discovery of production PostgreSQL database",
  "connector_config": {
    "connector_type": "postgresql",
    "host": "prod-db.company.com",
    "port": 5432,
    "database": "ecommerce",
    "username": "metadata_reader",
    "password": "secure_password",
    "include_patterns": ["public.*", "analytics.*"],
    "exclude_patterns": ["temp_*", "staging_*"]
  },
  "schedule_type": "recurring",
  "cron_expression": "0 2 * * *",
  "is_enabled": true,
  "created_by": "admin@company.com"
}
```

**Response:**
```json
{
  "schedule_id": "01HN8Y4L3N5P6Q7R8S9T0U1V2W",
  "status": "created",
  "next_run": "2025-01-10T02:00:00Z"
}
```

### GET /discovery/schedules
List all discovery schedules.

**Response:**
```json
{
  "schedules": [
    {
      "schedule_id": "01HN8Y4L3N5P6Q7R8S9T0U1V2W",
      "name": "PostgreSQL Production Discovery",
      "connector_type": "postgresql",
      "schedule_type": "recurring",
      "is_enabled": true,
      "last_run": "2025-01-09T02:00:00Z",
      "next_run": "2025-01-10T02:00:00Z",
      "success_rate": 0.98
    }
  ]
}
```

### POST /discovery/jobs/{schedule_id}/run
Trigger a discovery job for a specific schedule.

**Request Body (optional):**
```json
{
  "override_config": {
    "include_patterns": ["specific_table"],
    "force_full_scan": true
  }
}
```

**Response:**
```json
{
  "job_id": "01HN8Z5M4O6P7Q8R9S0T1U2V3X",
  "status": "running",
  "started_at": "2025-01-09T14:30:00Z",
  "estimated_duration_minutes": 15
}
```

### GET /discovery/jobs/{job_id}
Get discovery job status and results.

**Response:**
```json
{
  "job_id": "01HN8Z5M4O6P7Q8R9S0T1U2V3X",
  "status": "completed",
  "started_at": "2025-01-09T14:30:00Z",
  "completed_at": "2025-01-09T14:38:22Z",
  "duration_seconds": 502,
  "results": {
    "assets_discovered": 45,
    "assets_updated": 12,
    "lineage_edges_created": 23,
    "classifications_applied": 156,
    "errors": 0
  },
  "progress": {
    "current_step": "Completed",
    "total_steps": 5,
    "current_connector": null,
    "progress_percentage": 100
  }
}
```

---

## Search & Query

### POST /search
Perform intelligent search across metadata assets.

**Request Body:**
```json
{
  "query_text": "customer email data with high quality",
  "filters": {
    "asset_type": ["table", "view"],
    "source_system": ["postgresql", "mysql"],
    "classification": ["PII", "CONFIDENTIAL"],
    "quality_score_min": 0.8
  },
  "enable_natural_language": true,
  "enable_semantic_search": true,
  "limit": 20
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "01HN8X3K2M4P5Q6R7S8T9U0V1W",
      "name": "customer_orders",
      "display_name": "Customer Orders",
      "asset_type": "table",
      "source_system": "postgresql",
      "relevance_score": 0.94,
      "match_reasons": [
        "Contains customer email column",
        "High quality score (0.92)",
        "Classified as PII data"
      ],
      "highlighted_fields": {
        "description": "Customer order transactions with **email** details",
        "columns": ["customer_**email**", "billing_**email**"]
      }
    }
  ],
  "query_info": {
    "total_results": 156,
    "query_time_ms": 45,
    "search_type": "natural_language",
    "suggestions": ["customer contact information", "email analytics data"]
  }
}
```

### GET /search/suggestions
Get search suggestions and popular queries.

**Query Parameters:**
- `prefix` (string): Text prefix for autocomplete
- `limit` (int): Maximum suggestions (default: 10)

**Response:**
```json
{
  "suggestions": [
    "customer data",
    "customer email",
    "customer orders",
    "customer analytics"
  ],
  "popular_queries": [
    "high quality tables",
    "PII data sources",
    "recent data assets"
  ]
}
```

---

## Lineage Management

### GET /assets/{asset_id}/lineage
Get asset lineage relationships.

**Query Parameters:**
- `direction` (string): "upstream", "downstream", or "both" (default: "both")
- `max_depth` (int): Maximum traversal depth (default: 5)
- `include_columns` (bool): Include column-level lineage (default: false)

**Response:**
```json
{
  "asset_id": "01HN8X3K2M4P5Q6R7S8T9U0V1W",
  "lineage_paths": [
    {
      "path_id": "upstream_1",
      "direction": "upstream",
      "depth": 2,
      "nodes": [
        {
          "asset_id": "01HN8A1B2C3D4E5F6G7H8I9J0K",
          "name": "raw_orders",
          "asset_type": "table",
          "distance": 2
        },
        {
          "asset_id": "01HN8B2C3D4E5F6G7H8I9J0K1L",
          "name": "cleaned_orders",
          "asset_type": "view",
          "distance": 1
        }
      ],
      "edges": [
        {
          "source_id": "01HN8A1B2C3D4E5F6G7H8I9J0K",
          "target_id": "01HN8B2C3D4E5F6G7H8I9J0K1L",
          "lineage_type": "transformation",
          "transformation_logic": "SELECT * FROM raw_orders WHERE status = 'valid'"
        }
      ]
    }
  ],
  "graph_summary": {
    "total_nodes": 15,
    "total_edges": 23,
    "max_depth_upstream": 3,
    "max_depth_downstream": 4
  }
}
```

### POST /lineage
Create new lineage relationship.

**Request Body:**
```json
{
  "source_asset_id": "01HN8A1B2C3D4E5F6G7H8I9J0K",
  "target_asset_id": "01HN8B2C3D4E5F6G7H8I9J0K1L",
  "lineage_type": "transformation",
  "transformation_logic": "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id",
  "column_mappings": [
    {
      "source_column": "customer_id",
      "target_column": "customer_id",
      "transformation": "direct"
    }
  ]
}
```

### POST /assets/{asset_id}/impact
Analyze impact of potential changes to an asset.

**Request Body:**
```json
{
  "change_type": "schema_change",
  "change_details": {
    "column_removed": "legacy_field",
    "column_added": "new_customer_segment"
  }
}
```

**Response:**
```json
{
  "impact_analysis": {
    "total_impacted_assets": 12,
    "impact_severity": "medium",
    "impacted_assets": [
      {
        "asset_id": "01HN8C3D4E5F6G7H8I9J0K1L2M",
        "asset_name": "customer_analytics_view",
        "impact_type": "column_dependency",
        "impact_severity": "high",
        "impact_description": "References removed column legacy_field"
      }
    ],
    "recommended_actions": [
      "Update customer_analytics_view to remove legacy_field reference",
      "Test downstream ETL pipelines",
      "Notify data consumers of schema change"
    ]
  }
}
```

---

## AI Classification

### POST /classification/classify
Classify column data using AI-powered classification.

**Request Body:**
```json
{
  "column_name": "email_address",
  "data_type": "varchar",
  "sample_data": [
    "john.doe@company.com",
    "jane.smith@example.org",
    "admin@system.local"
  ],
  "context": {
    "table_name": "users",
    "source_system": "postgresql"
  }
}
```

**Response:**
```json
{
  "classification": "PII",
  "confidence_score": 0.96,
  "confidence_level": "HIGH",
  "method_used": "ensemble_voting",
  "reasoning": "Email pattern detection with 100% match rate",
  "tags": ["email", "personal_identifier", "contact_info"],
  "recommendations": [
    "Apply data masking in non-production environments",
    "Consider encryption for sensitive data",
    "Implement access controls"
  ],
  "processing_time_ms": 24
}
```

### GET /classification/rules
List classification rules and patterns.

**Response:**
```json
{
  "rules": [
    {
      "rule_id": "01HN8D4E5F6G7H8I9J0K1L2M3N",
      "name": "Email Detection",
      "classification": "PII",
      "confidence_score": 0.95,
      "pattern_type": "regex",
      "pattern_value": "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",
      "is_enabled": true,
      "success_rate": 0.94
    }
  ]
}
```

### POST /classification/rules
Create new classification rule.

**Request Body:**
```json
{
  "name": "Credit Card Detection",
  "description": "Detects credit card numbers using Luhn algorithm",
  "classification": "SENSITIVE_PII",
  "confidence_score": 0.9,
  "pattern_type": "custom",
  "pattern_logic": "luhn_validation",
  "conditions": {
    "column_name_patterns": ["card", "credit", "payment"],
    "data_type": "string",
    "length_range": [13, 19]
  }
}
```

---

## Integration Management

### GET /integrations
List available integrations and their status.

**Response:**
```json
{
  "integrations": [
    {
      "integration_id": "slack_notifications",
      "name": "Slack Notifications",
      "type": "notification",
      "status": "enabled",
      "configuration": {
        "webhook_url": "https://hooks.slack.com/...",
        "channel": "#data-alerts"
      },
      "last_used": "2025-01-09T12:30:00Z"
    },
    {
      "integration_id": "data_quality_monitor",
      "name": "Data Quality Monitoring",
      "type": "monitoring",
      "status": "enabled",
      "metrics": {
        "alerts_sent_today": 3,
        "quality_checks_passed": 0.98
      }
    }
  ]
}
```

### POST /integrations/{integration_id}/trigger
Trigger integration action manually.

**Request Body:**
```json
{
  "action": "send_summary_report",
  "parameters": {
    "report_period": "last_24_hours",
    "include_metrics": true
  }
}
```

---

## Error Handling

### Standard Error Response Format

```json
{
  "error": {
    "code": "ASSET_NOT_FOUND",
    "message": "Asset with ID '01HN8X3K2M4P5Q6R7S8T9U0V1W' not found",
    "details": {
      "asset_id": "01HN8X3K2M4P5Q6R7S8T9U0V1W",
      "searched_in": ["postgresql", "mysql"],
      "suggestion": "Check asset ID format or verify asset exists"
    },
    "timestamp": "2025-01-09T14:30:00Z",
    "request_id": "req_01HN8E5F6G7H8I9J0K1L2M3N4O"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `ASSET_NOT_FOUND` | 404 | Requested asset doesn't exist |
| `INVALID_QUERY` | 400 | Malformed search query |
| `CONNECTOR_ERROR` | 503 | Data source connection failed |
| `CLASSIFICATION_FAILED` | 500 | AI classification service error |
| `RATE_LIMIT_EXCEEDED` | 429 | API rate limit exceeded |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `VALIDATION_ERROR` | 422 | Request validation failed |

### Rate Limiting

- **Default Limits:** 1000 requests per hour per API key
- **Search API:** 100 requests per minute
- **Discovery Jobs:** 10 concurrent jobs per tenant
- **Headers:** `X-RateLimit-Remaining`, `X-RateLimit-Reset`

---

## SDK & Client Libraries

### Python SDK Example

```python
from apg_metadata import MetadataClient

client = MetadataClient(
    base_url="http://localhost:5000",
    api_key="your-api-key"
)

# Search assets
results = await client.search_assets(
    query="customer email data",
    filters={"asset_type": "table"}
)

# Get asset details
asset = await client.get_asset("01HN8X3K2M4P5Q6R7S8T9U0V1W")

# Create discovery schedule
schedule = await client.create_discovery_schedule({
    "name": "Daily PostgreSQL Scan",
    "connector_type": "postgresql",
    "cron_expression": "0 2 * * *"
})
```

### JavaScript SDK Example

```javascript
import { MetadataClient } from '@apg/metadata-client';

const client = new MetadataClient({
  baseUrl: 'http://localhost:5000',
  apiKey: 'your-api-key'
});

// Search assets
const results = await client.searchAssets({
  queryText: 'customer email data',
  filters: { assetType: 'table' }
});

// Get lineage
const lineage = await client.getAssetLineage(
  '01HN8X3K2M4P5Q6R7S8T9U0V1W',
  { direction: 'both', maxDepth: 3 }
);
```

---

## Webhook Events

### Event Types

- `asset.created` - New asset discovered
- `asset.updated` - Asset metadata changed  
- `asset.deleted` - Asset removed
- `lineage.created` - New lineage relationship
- `classification.completed` - Asset classification finished
- `discovery.completed` - Discovery job finished
- `quality.alert` - Data quality issue detected

### Webhook Payload Example

```json
{
  "event_type": "asset.created",
  "event_id": "01HN8F6G7H8I9J0K1L2M3N4O5P",
  "timestamp": "2025-01-09T14:30:00Z",
  "tenant_id": "company_tenant",
  "data": {
    "asset": {
      "id": "01HN8X3K2M4P5Q6R7S8T9U0V1W",
      "name": "new_customer_table",
      "asset_type": "table",
      "source_system": "postgresql"
    },
    "discovery_job_id": "01HN8Z5M4O6P7Q8R9S0T1U2V3X"
  }
}
```

---

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **JSON:** `/api/v1/metadata/openapi.json`
- **YAML:** `/api/v1/metadata/openapi.yaml`  
- **Interactive Docs:** `/api/v1/metadata/docs`
- **ReDoc:** `/api/v1/metadata/redoc`

---

*For additional support or questions, please contact the development team or refer to the comprehensive user guide.*