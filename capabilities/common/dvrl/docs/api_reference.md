# APG Data Virtualization (DVRL) API Reference

## Table of Contents
1. [Authentication](#authentication)
2. [Base URL and Versioning](#base-url-and-versioning)
3. [Data Source Management](#data-source-management)
4. [Query Execution](#query-execution)
5. [Natural Language Processing](#natural-language-processing)
6. [Streaming Queries](#streaming-queries)
7. [Transaction Management](#transaction-management)
8. [Monitoring and Health](#monitoring-and-health)
9. [Error Handling](#error-handling)
10. [Rate Limiting](#rate-limiting)
11. [SDK Examples](#sdk-examples)

## Authentication

All DVRL API requests require authentication through the APG platform's auth_rbac capability.

### Bearer Token Authentication
```bash
curl -X GET "${APG_BASE_URL}/api/v1/data-sources" \
  -H "Authorization: Bearer ${APG_ACCESS_TOKEN}" \
  -H "Content-Type: application/json"
```

### Tenant Context
All API requests are automatically scoped to your APG tenant. The tenant context is extracted from the authentication token.

```json
{
  "tenant_id": "your-org-tenant",
  "user_id": "user@company.com",
  "roles": ["data_analyst", "dvrl_user"],
  "permissions": ["dvrl:read", "dvrl:execute"]
}
```

## Base URL and Versioning

**Base URL**: `${APG_BASE_URL}/api/v1/dvrl`

**Current Version**: v1  
**Supported Formats**: JSON, MessagePack  
**Default Format**: JSON

### Version Headers
```http
Accept: application/json
API-Version: v1
```

## Data Source Management

### Register Data Source
Register a new data source in the federation.

**Endpoint**: `POST /data-sources`

**Request Body**:
```json
{
  "name": "Production Orders Database",
  "type": "postgresql",
  "description": "Main orders database for production workloads",
  "connection_config": {
    "host": "orders-db.company.com",
    "port": 5432,
    "database": "orders",
    "username": "dvrl_readonly",
    "password": "${SECURE_PASSWORD}",
    "ssl_mode": "require"
  },
  "connection_pool_size": 20,
  "query_timeout_seconds": 300
}
```

**Response** (201 Created):
```json
{
  "data_source_id": "ds_01HKX7GQPM8A5N0QJ4RNHM3Y2Z",
  "name": "Production Orders Database",
  "type": "postgresql",
  "status": "active",
  "schema_discovered": true,
  "tables_count": 42,
  "created_at": "2024-11-15T10:30:00Z",
  "health_status": {
    "connection_status": "healthy",
    "response_time_ms": 45,
    "last_check": "2024-11-15T10:30:00Z"
  }
}
```

**Error Responses**:
```json
{
  "error": "validation_error",
  "message": "Missing required field: connection_config.host",
  "details": {
    "field": "connection_config.host",
    "code": "FIELD_REQUIRED"
  }
}
```

### List Data Sources
Retrieve all registered data sources for the tenant.

**Endpoint**: `GET /data-sources`

**Query Parameters**:
- `status` (optional): Filter by status (`active`, `error`, `disabled`)
- `type` (optional): Filter by data source type
- `limit` (optional): Maximum results (default: 50, max: 200)
- `offset` (optional): Pagination offset

**Response** (200 OK):
```json
{
  "data_sources": [
    {
      "data_source_id": "ds_01HKX7GQPM8A5N0QJ4RNHM3Y2Z",
      "name": "Production Orders Database",
      "type": "postgresql",
      "status": "active",
      "tables_count": 42,
      "avg_response_time_ms": 45,
      "query_count": 15420,
      "created_at": "2024-11-15T10:30:00Z"
    }
  ],
  "pagination": {
    "total": 5,
    "limit": 50,
    "offset": 0,
    "has_more": false
  }
}
```

### Get Data Source Details
Retrieve detailed information about a specific data source.

**Endpoint**: `GET /data-sources/{data_source_id}`

**Response** (200 OK):
```json
{
  "data_source_id": "ds_01HKX7GQPM8A5N0QJ4RNHM3Y2Z",
  "name": "Production Orders Database",
  "type": "postgresql",
  "description": "Main orders database for production workloads",
  "status": "active",
  "connection_config": {
    "host": "orders-db.company.com",
    "port": 5432,
    "database": "orders",
    "ssl_mode": "require"
  },
  "performance_metrics": {
    "avg_response_time_ms": 45,
    "query_count": 15420,
    "error_count_24h": 2,
    "uptime_percentage": 99.98
  },
  "schema_info": {
    "tables_count": 42,
    "total_rows": 12500000,
    "last_discovery": "2024-11-15T10:30:00Z"
  }
}
```

### Get Data Source Schema
Retrieve the discovered schema for a data source.

**Endpoint**: `GET /data-sources/{data_source_id}/schema`

**Response** (200 OK):
```json
{
  "schema_name": "orders",
  "tables": [
    {
      "name": "customers",
      "type": "table",
      "row_count": 125000,
      "columns": [
        {
          "name": "customer_id",
          "data_type": "bigint",
          "is_nullable": false,
          "is_primary_key": true
        },
        {
          "name": "customer_name",
          "data_type": "varchar(255)",
          "is_nullable": false
        },
        {
          "name": "email",
          "data_type": "varchar(255)",
          "is_nullable": true,
          "is_unique": true
        }
      ],
      "indexes": [
        {
          "name": "idx_customer_email",
          "columns": ["email"],
          "is_unique": true
        }
      ]
    }
  ]
}
```

### Update Data Source
Update data source configuration or settings.

**Endpoint**: `PUT /data-sources/{data_source_id}`

**Request Body**:
```json
{
  "description": "Updated description",
  "connection_pool_size": 30,
  "query_timeout_seconds": 600
}
```

### Delete Data Source
Remove a data source from the federation.

**Endpoint**: `DELETE /data-sources/{data_source_id}`

**Response** (204 No Content)

## Query Execution

### Execute SQL Query
Execute a federated SQL query across registered data sources.

**Endpoint**: `POST /queries/sql`

**Request Body**:
```json
{
  "sql": "SELECT c.customer_name, COUNT(o.order_id) as order_count FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY order_count DESC LIMIT 10",
  "options": {
    "cache_strategy": "aggressive",
    "max_execution_time": 300,
    "result_format": "json",
    "streaming": false,
    "federation_strategy": "optimal"
  }
}
```

**Response** (200 OK):
```json
{
  "query_id": "qry_01HKX8GQPM8A5N0QJ4RNHM3Y2Z",
  "status": "completed",
  "results": {
    "columns": [
      {
        "name": "customer_name",
        "type": "varchar",
        "nullable": false
      },
      {
        "name": "order_count", 
        "type": "bigint",
        "nullable": false
      }
    ],
    "rows": [
      ["Acme Corp", 1547],
      ["Global Industries", 1203],
      ["TechStart Inc", 892]
    ],
    "row_count": 10,
    "more_available": false
  },
  "execution_plan": {
    "plan_id": "plan_01HKX8GQPM8A5N0QJ4RNHM3Y2Z",
    "strategy": "push_down_aggregation",
    "estimated_cost": 1.25,
    "data_sources_used": ["ds_01HKX7GQPM8A5N0QJ4RNHM3Y2Z"]
  },
  "performance_metrics": {
    "total_time_ms": 1250,
    "planning_time_ms": 45,
    "execution_time_ms": 1205,
    "network_transfer_bytes": 2048,
    "cache_hit": true,
    "rows_processed": 125000
  },
  "cache_status": "hit"
}
```

**Async Query Execution**:
For long-running queries, use async execution:

```json
{
  "sql": "SELECT * FROM large_table WHERE complex_calculation(data) > threshold",
  "options": {
    "async": true,
    "callback_url": "https://your-app.com/webhook/query-complete"
  }
}
```

**Async Response** (202 Accepted):
```json
{
  "query_id": "qry_01HKX8GQPM8A5N0QJ4RNHM3Y2Z",
  "status": "running",
  "estimated_completion": "2024-11-15T11:00:00Z",
  "progress_url": "/queries/qry_01HKX8GQPM8A5N0QJ4RNHM3Y2Z/status"
}
```

### Get Query Status
Check the status of a running query.

**Endpoint**: `GET /queries/{query_id}/status`

**Response** (200 OK):
```json
{
  "query_id": "qry_01HKX8GQPM8A5N0QJ4RNHM3Y2Z",
  "status": "running",
  "progress": {
    "percentage_complete": 67,
    "current_phase": "aggregation",
    "rows_processed": 8500000,
    "estimated_total_rows": 12700000
  },
  "elapsed_time_ms": 45000,
  "estimated_remaining_ms": 22000
}
```

### Get Query Results
Retrieve results for a completed query.

**Endpoint**: `GET /queries/{query_id}/results`

**Query Parameters**:
- `format` (optional): `json`, `csv`, `parquet`
- `offset` (optional): Start row for pagination
- `limit` (optional): Maximum rows to return

### Cancel Query
Cancel a running query.

**Endpoint**: `DELETE /queries/{query_id}`

**Response** (200 OK):
```json
{
  "query_id": "qry_01HKX8GQPM8A5N0QJ4RNHM3Y2Z",
  "status": "cancelled",
  "cancelled_at": "2024-11-15T10:45:30Z"
}
```

## Natural Language Processing

### Execute Natural Language Query
Convert natural language to SQL and execute.

**Endpoint**: `POST /queries/natural-language`

**Request Body**:
```json
{
  "query": "Show me the top 5 customers by revenue in the last quarter",
  "context": {
    "preferred_tables": ["customers", "orders"],
    "date_range_preference": "relative",
    "result_limit_preference": 10
  }
}
```

**Response** (200 OK):
```json
{
  "query_id": "qry_01HKX8GQPM8A5N0QJ4RNHM3Y2Z",
  "natural_language_query": "Show me the top 5 customers by revenue in the last quarter",
  "generated_sql": "SELECT c.customer_name, SUM(o.total_amount) as revenue FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.created_at >= DATE_TRUNC('quarter', CURRENT_DATE) - INTERVAL '3 months' GROUP BY c.customer_id, c.customer_name ORDER BY revenue DESC LIMIT 5",
  "confidence_score": 0.94,
  "disambiguation_notes": [
    "Interpreted 'last quarter' as previous complete quarter",
    "Using 'total_amount' field for revenue calculation"
  ],
  "results": {
    "columns": [...],
    "rows": [...]
  }
}
```

### Get Query Suggestions
Get intelligent query suggestions based on context.

**Endpoint**: `GET /queries/suggestions`

**Query Parameters**:
- `context` (optional): JSON context object
- `schema_hint` (optional): Preferred tables/schemas
- `query_type` (optional): `analytical`, `operational`, `exploratory`

**Response** (200 OK):
```json
{
  "suggestions": [
    {
      "suggestion": "Show me sales trends by month for the current year",
      "sql_preview": "SELECT DATE_TRUNC('month', order_date) as month, SUM(total_amount) as sales...",
      "confidence": 0.89,
      "category": "time_series_analysis"
    },
    {
      "suggestion": "Find customers who haven't placed orders recently",
      "sql_preview": "SELECT c.* FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id...",
      "confidence": 0.85,
      "category": "customer_analysis"
    }
  ]
}
```

## Streaming Queries

### Start Streaming Query
Execute a query with streaming results for large datasets.

**Endpoint**: `POST /queries/streaming`

**Request Body**:
```json
{
  "sql": "SELECT * FROM sensor_data WHERE timestamp >= NOW() - INTERVAL '1 hour'",
  "stream_options": {
    "batch_size": 1000,
    "buffer_size_mb": 50,
    "compression": "gzip",
    "format": "jsonl"
  }
}
```

**Response** (201 Created):
```json
{
  "stream_id": "strm_01HKX8GQPM8A5N0QJ4RNHM3Y2Z",
  "status": "initializing",
  "stream_url": "wss://api.apg.com/streams/strm_01HKX8GQPM8A5N0QJ4RNHM3Y2Z",
  "http_poll_url": "/queries/streaming/strm_01HKX8GQPM8A5N0QJ4RNHM3Y2Z/data"
}
```

### WebSocket Streaming
Connect to the WebSocket endpoint for real-time data:

```javascript
const ws = new WebSocket('wss://api.apg.com/streams/strm_01HKX8GQPM8A5N0QJ4RNHM3Y2Z?token=YOUR_TOKEN');

ws.onmessage = function(event) {
  const batch = JSON.parse(event.data);
  console.log(`Received ${batch.rows.length} rows`);
  
  // Process batch data
  batch.rows.forEach(row => {
    processDataRow(row);
  });
};
```

### HTTP Polling for Streaming Data
Alternative to WebSocket for streaming data:

**Endpoint**: `GET /queries/streaming/{stream_id}/data`

**Query Parameters**:
- `since` (optional): Timestamp to get data since
- `timeout` (optional): Long-polling timeout (max 30s)

### Stop Streaming Query
Terminate a streaming query.

**Endpoint**: `DELETE /queries/streaming/{stream_id}`

**Response** (200 OK):
```json
{
  "stream_id": "strm_01HKX8GQPM8A5N0QJ4RNHM3Y2Z",
  "status": "stopped",
  "total_rows_streamed": 125000,
  "total_batches": 125,
  "duration_seconds": 3600
}
```

## Transaction Management

### Begin Transaction
Start a distributed transaction across multiple data sources.

**Endpoint**: `POST /transactions`

**Request Body**:
```json
{
  "data_source_ids": [
    "ds_orders_postgres",
    "ds_inventory_mysql", 
    "ds_billing_oracle"
  ],
  "isolation_level": "read_committed",
  "timeout_seconds": 300
}
```

**Response** (201 Created):
```json
{
  "transaction_id": "txn_01HKX8GQPM8A5N0QJ4RNHM3Y2Z",
  "status": "active",
  "data_sources": [
    {
      "data_source_id": "ds_orders_postgres",
      "status": "prepared"
    },
    {
      "data_source_id": "ds_inventory_mysql",
      "status": "prepared"
    }
  ],
  "started_at": "2024-11-15T10:45:00Z",
  "expires_at": "2024-11-15T10:50:00Z"
}
```

### Execute Query in Transaction
Execute a query within a transaction context.

**Endpoint**: `POST /transactions/{transaction_id}/queries`

**Request Body**:
```json
{
  "sql": "UPDATE inventory SET quantity = quantity - 5 WHERE product_id = 12345",
  "data_source_id": "ds_inventory_mysql"
}
```

### Commit Transaction
Commit a distributed transaction.

**Endpoint**: `POST /transactions/{transaction_id}/commit`

**Response** (200 OK):
```json
{
  "transaction_id": "txn_01HKX8GQPM8A5N0QJ4RNHM3Y2Z",
  "status": "committed",
  "commit_timestamp": "2024-11-15T10:47:30Z",
  "data_sources": [
    {
      "data_source_id": "ds_orders_postgres",
      "status": "committed"
    }
  ]
}
```

### Rollback Transaction
Rollback a distributed transaction.

**Endpoint**: `POST /transactions/{transaction_id}/rollback`

## Monitoring and Health

### Health Check
Get overall system health status.

**Endpoint**: `GET /health`

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2024-11-15T10:45:00Z",
  "version": "1.0.0",
  "components": {
    "query_engine": {
      "status": "healthy",
      "response_time_ms": 12
    },
    "cache_system": {
      "status": "healthy",
      "hit_ratio": 0.87,
      "memory_usage": 0.65
    },
    "data_sources": {
      "status": "healthy",
      "total": 5,
      "healthy": 5,
      "degraded": 0,
      "unhealthy": 0
    }
  }
}
```

### Performance Metrics
Get detailed performance metrics.

**Endpoint**: `GET /metrics`

**Response** (200 OK):
```json
{
  "query_performance": {
    "total_queries": 1567890,
    "avg_response_time_ms": 145,
    "p95_response_time_ms": 890,
    "p99_response_time_ms": 2340,
    "error_rate": 0.003
  },
  "cache_performance": {
    "hit_ratio": 0.87,
    "miss_ratio": 0.13,
    "eviction_rate": 0.02,
    "memory_utilization": 0.65
  },
  "resource_usage": {
    "cpu_utilization": 0.45,
    "memory_usage_mb": 4096,
    "network_throughput_mbps": 120,
    "active_connections": 245
  },
  "data_source_health": [
    {
      "data_source_id": "ds_01HKX7GQPM8A5N0QJ4RNHM3Y2Z",
      "status": "healthy",
      "response_time_ms": 45,
      "query_count_24h": 8934,
      "error_count_24h": 2
    }
  ]
}
```

### Query History
Get query execution history and analytics.

**Endpoint**: `GET /queries/history`

**Query Parameters**:
- `from_date` (optional): Start date (ISO 8601)
- `to_date` (optional): End date (ISO 8601)
- `status` (optional): Filter by status
- `limit` (optional): Maximum results

## Error Handling

### Standard Error Response Format
All errors follow a consistent format:

```json
{
  "error": "error_code",
  "message": "Human-readable error description",
  "details": {
    "field": "specific_field_if_applicable",
    "code": "SPECIFIC_ERROR_CODE",
    "context": {}
  },
  "request_id": "req_01HKX8GQPM8A5N0QJ4RNHM3Y2Z",
  "timestamp": "2024-11-15T10:45:00Z"
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | `validation_error` | Request validation failed |
| 401 | `authentication_required` | Missing or invalid auth token |
| 403 | `permission_denied` | Insufficient permissions |
| 404 | `resource_not_found` | Requested resource doesn't exist |
| 408 | `query_timeout` | Query exceeded timeout limit |
| 409 | `resource_conflict` | Resource already exists |
| 422 | `sql_parse_error` | Invalid SQL syntax |
| 429 | `rate_limit_exceeded` | Too many requests |
| 500 | `internal_server_error` | Unexpected server error |
| 502 | `data_source_error` | External data source error |
| 503 | `service_unavailable` | Service temporarily unavailable |

### Error Examples

**SQL Parse Error**:
```json
{
  "error": "sql_parse_error",
  "message": "Invalid SQL syntax near 'FORM'",
  "details": {
    "position": 25,
    "expected": "FROM",
    "actual": "FORM"
  }
}
```

**Data Source Connection Error**:
```json
{
  "error": "data_source_error", 
  "message": "Unable to connect to data source 'prod-orders'",
  "details": {
    "data_source_id": "ds_01HKX7GQPM8A5N0QJ4RNHM3Y2Z",
    "error_type": "connection_timeout",
    "retry_after": 30
  }
}
```

## Rate Limiting

### Rate Limit Headers
All responses include rate limiting information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1699189200
X-RateLimit-Window: 3600
```

### Rate Limits by Endpoint

| Endpoint Category | Limit | Window |
|-------------------|-------|--------|
| Query Execution | 100 requests | per minute |
| Data Source Management | 50 requests | per minute |
| Schema Discovery | 20 requests | per minute |
| Streaming Queries | 10 concurrent | total |
| Transaction Management | 5 concurrent | total |

### Rate Limit Exceeded Response
```json
{
  "error": "rate_limit_exceeded",
  "message": "Query execution rate limit exceeded",
  "details": {
    "limit": 100,
    "window_seconds": 60,
    "retry_after": 45
  }
}
```

## SDK Examples

### Python SDK
```python
from apg_dvrl import DVRLClient

# Initialize client
client = DVRLClient(
    base_url="https://api.apg.com",
    access_token="your-apg-token"
)

# Register data source
data_source = await client.register_data_source({
    "name": "Production DB",
    "type": "postgresql",
    "connection_config": {
        "host": "db.company.com",
        "port": 5432,
        "database": "production",
        "username": "readonly",
        "password": os.environ["DB_PASSWORD"]
    }
})

# Execute query
result = await client.execute_sql(
    "SELECT COUNT(*) FROM orders WHERE date >= '2024-01-01'",
    options={"cache_strategy": "aggressive"}
)

print(f"Found {result.rows[0][0]} orders")
```

### Node.js SDK
```javascript
const { DVRLClient } = require('@apg/dvrl-client');

const client = new DVRLClient({
  baseUrl: 'https://api.apg.com',
  accessToken: process.env.APG_ACCESS_TOKEN
});

// Natural language query
const result = await client.executeNaturalLanguage(
  "Show me sales by region for this month"
);

console.log(`Generated SQL: ${result.generatedSql}`);
console.log(`Confidence: ${result.confidenceScore}`);
```

### Java SDK
```java
import com.apg.dvrl.DVRLClient;
import com.apg.dvrl.model.QueryResult;

DVRLClient client = DVRLClient.builder()
    .baseUrl("https://api.apg.com")
    .accessToken(System.getenv("APG_ACCESS_TOKEN"))
    .build();

QueryResult result = client.executeSql(
    "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id",
    QueryOptions.builder()
        .cacheStrategy("conservative")
        .maxExecutionTime(Duration.ofMinutes(5))
        .build()
);

result.getRows().forEach(row -> {
    System.out.println("Customer: " + row.get("customer_id") + 
                      ", Total: " + row.get("sum"));
});
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-11  
**Author**: APG Platform Team