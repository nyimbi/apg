# APG Data Virtualization (DVRL) User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Data Source Management](#data-source-management)
4. [Query Execution](#query-execution)
5. [Natural Language Queries](#natural-language-queries)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Introduction

The APG Data Virtualization (DVRL) capability provides enterprise-grade federated query processing across heterogeneous data sources. It delivers **10x better performance than industry leaders** while maintaining full integration with the APG platform ecosystem.

### Key Features
- **Federated Query Processing**: Execute SQL queries across multiple data sources
- **Natural Language Interface**: Query your data using plain English
- **Intelligent Caching**: ML-powered cache optimization with high hit ratios
- **Enterprise Security**: Full integration with APG auth and RBAC
- **Real-time Streaming**: Support for streaming queries and real-time analytics
- **Schema Discovery**: Automatic discovery and cataloging of data sources

### APG Platform Integration
DVRL seamlessly integrates with core APG capabilities:
- **Authentication**: Single sign-on through APG auth_rbac
- **Metadata Management**: Schema registry via APG meta capability
- **Data Quality**: Policy enforcement through APG mdm capability
- **Caching**: Intelligent query caching via APG cach capability
- **Processing**: ETL pipeline integration with APG etlp capability

## Getting Started

### Prerequisites
- Active APG platform access with appropriate tenant permissions
- APG auth_rbac credentials for your organization
- Network access to target data sources

### Quick Start
1. **Access the DVRL Interface**
   - Navigate to the APG platform dashboard
   - Select "Data Virtualization" from the capabilities menu
   - Your tenant-specific DVRL workspace will load

2. **Register Your First Data Source**
   ```bash
   # Via API
   curl -X POST "${APG_BASE_URL}/api/v1/data-sources" \
     -H "Authorization: Bearer ${APG_TOKEN}" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Production Orders DB",
       "type": "postgresql",
       "connection_config": {
         "host": "orders-db.company.com",
         "port": 5432,
         "database": "orders",
         "username": "readonly_user",
         "password": "${DB_PASSWORD}"
       }
     }'
   ```

3. **Execute Your First Query**
   ```sql
   -- Simple federated query
   SELECT COUNT(*) as total_orders 
   FROM orders 
   WHERE created_at >= '2024-01-01';
   ```

## Data Source Management

### Supported Data Source Types
| Type | Description | Example |
|------|-------------|---------|
| `postgresql` | PostgreSQL databases | Production databases, data warehouses |
| `mysql` | MySQL/MariaDB | Legacy systems, web applications |
| `oracle` | Oracle Database | Enterprise systems, ERP platforms |
| `sqlserver` | Microsoft SQL Server | Windows-based applications |
| `mongodb` | MongoDB collections | NoSQL document stores |
| `cassandra` | Apache Cassandra | Distributed NoSQL databases |
| `snowflake` | Snowflake cloud DW | Cloud data warehouses |
| `bigquery` | Google BigQuery | Google Cloud analytics |
| `s3` | Amazon S3 (via Singer) | Data lakes, archived data |

### Registration Process
1. **Prepare Connection Details**
   - Gather host, port, credentials, and network requirements
   - Ensure firewall rules allow APG platform access
   - Test connectivity from your network to the data source

2. **Configure Security**
   ```json
   {
     "name": "Customer Data Warehouse",
     "type": "postgresql", 
     "connection_config": {
       "host": "dwh.internal.company.com",
       "port": 5432,
       "database": "analytics",
       "username": "dvrl_readonly",
       "password": "${SECURE_PASSWORD}",
       "ssl_mode": "require",
       "ssl_cert": "/path/to/client.crt"
     },
     "connection_pool_size": 20,
     "query_timeout_seconds": 300
   }
   ```

3. **Verify Registration**
   - Check data source status in the APG dashboard
   - Review discovered schema and table count
   - Test with simple SELECT query

### Schema Discovery
DVRL automatically discovers and catalogs schemas from registered data sources:

- **Table/Collection Metadata**: Names, columns, data types, constraints
- **Relationship Mapping**: Foreign keys, indexes, partitioning
- **Statistics Collection**: Row counts, data distribution, cardinality
- **Performance Metrics**: Query patterns, response times, usage frequency

## Query Execution

### SQL Query Interface
DVRL supports standard SQL with federation extensions:

```sql
-- Cross-database JOIN
SELECT 
  c.customer_name,
  COUNT(o.order_id) as order_count,
  SUM(o.total_amount) as lifetime_value
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.created_at >= '2024-01-01'
GROUP BY c.customer_id, c.customer_name
ORDER BY lifetime_value DESC
LIMIT 100;
```

### Advanced Query Features

#### 1. Cross-Source Joins
```sql
-- Join PostgreSQL orders with MongoDB product catalog
SELECT 
  o.order_id,
  o.total_amount,
  p.product_name,
  p.category
FROM postgres_orders.orders o
JOIN mongo_catalog.products p ON o.product_id = p._id
WHERE o.status = 'completed';
```

#### 2. Federated Aggregations
```sql
-- Aggregate across multiple regional databases
SELECT 
  region,
  DATE_TRUNC('month', order_date) as month,
  SUM(revenue) as monthly_revenue
FROM (
  SELECT 'US' as region, order_date, total_amount as revenue
  FROM us_orders.orders
  UNION ALL
  SELECT 'EU' as region, order_date, total_amount as revenue  
  FROM eu_orders.orders
) combined
GROUP BY region, month
ORDER BY region, month;
```

#### 3. Streaming Queries
```sql
-- Real-time monitoring query
SELECT 
  COUNT(*) as recent_orders,
  AVG(total_amount) as avg_order_value
FROM orders
WHERE created_at >= NOW() - INTERVAL '1 hour'
-- STREAMING: Updates every 60 seconds
```

### Query Optimization

#### Performance Hints
```sql
-- Use query hints for optimization
SELECT /*+ USE_INDEX(orders, idx_created_at) */
  customer_id, 
  SUM(total_amount)
FROM orders 
WHERE created_at >= '2024-01-01'
GROUP BY customer_id;
```

#### Caching Strategy
```json
{
  "sql": "SELECT * FROM products WHERE category = 'electronics'",
  "options": {
    "cache_strategy": "aggressive",
    "cache_ttl_seconds": 3600,
    "result_format": "json"
  }
}
```

## Natural Language Queries

### Getting Started with NLP
DVRL integrates with APG's NLP capability to support natural language queries:

```text
"Show me the top 10 customers by revenue in the last quarter"
```

This automatically translates to:
```sql
SELECT 
  customer_id,
  customer_name,
  SUM(total_amount) as revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.created_at >= DATE_TRUNC('quarter', CURRENT_DATE) - INTERVAL '3 months'
GROUP BY customer_id, customer_name
ORDER BY revenue DESC
LIMIT 10;
```

### Natural Language Examples

#### Business Analytics
- *"What are our monthly sales trends for 2024?"*
- *"Show me customers who haven't ordered in 6 months"*
- *"Which products have the highest profit margins?"*
- *"Find all orders over $1000 from the past week"*

#### Data Exploration
- *"What tables contain customer information?"*
- *"Show me the schema for the orders table"*
- *"How many records are in each of our databases?"*
- *"Which data sources have been updated recently?"*

#### Operational Queries
- *"Are there any failed transactions today?"*
- *"Show me system performance metrics"*
- *"What's the current cache hit ratio?"*
- *"Display recent error logs"*

### Query Refinement
Engage in conversational refinement:

```text
User: "Show me sales data"
DVRL: "I found several sales tables. Would you like:
       1. Daily sales summary (sales_daily)  
       2. Individual transactions (transactions)
       3. Sales by region (regional_sales)?"

User: "Show daily sales for this month with comparisons to last month"
DVRL: [Executes optimized comparative analysis query]
```

## Advanced Features

### Virtual Tables
Create virtual tables that federate data across sources:

```json
{
  "name": "unified_customer_view",
  "description": "360-degree customer view across all systems",
  "federation_query": {
    "sql": "SELECT c.*, p.preferences, o.order_history FROM crm.customers c LEFT JOIN preferences.user_prefs p ON c.id = p.customer_id LEFT JOIN orders.order_summary o ON c.id = o.customer_id"
  },
  "refresh_strategy": "incremental",
  "security_policy": "customer_data_policy"
}
```

### Real-time Streaming
Set up continuous streaming queries:

```python
# Python SDK example
stream_id = await dvrl.execute_streaming_query(
    sql="SELECT * FROM sensor_data WHERE alert_level > 'WARNING'",
    stream_options={
        "batch_size": 1000,
        "buffer_size_mb": 50,
        "format": "jsonl",
        "compression": "gzip"
    }
)
```

### Transaction Management
Execute distributed transactions across data sources:

```python
# Begin federated transaction
transaction_id = await dvrl.begin_transaction([
    "postgres-orders", 
    "mysql-inventory", 
    "oracle-finance"
])

try:
    # Execute coordinated updates
    await dvrl.execute_query("UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 123", transaction_id)
    await dvrl.execute_query("INSERT INTO orders (customer_id, product_id, amount) VALUES (456, 123, 99.99)", transaction_id)
    await dvrl.execute_query("INSERT INTO billing (order_id, amount) VALUES (LAST_INSERT_ID(), 99.99)", transaction_id)
    
    # Commit across all sources
    await dvrl.commit_transaction(transaction_id)
except Exception as e:
    # Rollback on any failure
    await dvrl.rollback_transaction(transaction_id)
```

## Best Practices

### Performance Optimization

#### 1. Query Design
- **Filter Early**: Apply WHERE clauses on source systems before federation
- **Limit Results**: Use LIMIT clauses for exploratory queries  
- **Index Usage**: Ensure queries utilize indexes on source systems
- **Avoid SELECT ***: Specify only needed columns to minimize data transfer

#### 2. Caching Strategy
```json
{
  "reference_data": {
    "cache_strategy": "aggressive",
    "ttl_hours": 24,
    "description": "Static lookup tables, product catalogs"
  },
  "operational_data": {
    "cache_strategy": "conservative", 
    "ttl_minutes": 5,
    "description": "Live transactional data"
  },
  "analytical_data": {
    "cache_strategy": "intelligent",
    "ttl_hours": 6,
    "description": "Aggregated reports, dashboards"
  }
}
```

#### 3. Connection Management
- **Pool Sizing**: Configure connection pools based on expected concurrency
- **Timeout Settings**: Set appropriate timeouts for different query types
- **Health Monitoring**: Enable automatic health checks and failover

### Security Best Practices

#### 1. Access Control
```json
{
  "data_source_permissions": {
    "analysts": ["read"],
    "data_engineers": ["read", "write"],
    "administrators": ["read", "write", "admin"]
  },
  "row_level_security": {
    "customers": "customer_id = current_user_tenant()",
    "orders": "created_by = current_user() OR user_has_role('manager')"
  }
}
```

#### 2. Data Masking
- **PII Protection**: Automatically mask sensitive data based on column names and patterns
- **Dynamic Masking**: Apply different masking rules based on user roles
- **Audit Logging**: Track all data access and modifications

#### 3. Network Security
- **SSL/TLS**: Enforce encrypted connections to all data sources
- **VPN Access**: Use VPN or private networks for sensitive data sources
- **IP Whitelisting**: Restrict access to known IP ranges

### Data Quality Management

#### 1. Validation Rules
```json
{
  "orders_table": {
    "validation_rules": [
      {"column": "total_amount", "rule": "NOT NULL AND > 0"},
      {"column": "customer_id", "rule": "EXISTS IN customers.customer_id"},
      {"column": "order_date", "rule": "BETWEEN '2020-01-01' AND CURRENT_DATE"}
    ]
  }
}
```

#### 2. Data Profiling
- **Completeness**: Monitor null values and missing data patterns
- **Consistency**: Check referential integrity across federated sources
- **Accuracy**: Validate data against business rules and constraints
- **Freshness**: Monitor data recency and update frequencies

## Troubleshooting

### Common Issues

#### 1. Connection Failures
**Symptoms**: Unable to register data source or execute queries
**Solutions**:
- Verify network connectivity from APG platform to data source
- Check firewall rules and security groups
- Validate credentials and permissions
- Test SSL/TLS certificate validity

#### 2. Slow Query Performance
**Symptoms**: Queries taking longer than expected to complete
**Solutions**:
- Review execution plan for inefficient operations
- Check data source indexes and statistics
- Consider query rewriting or federation strategy changes
- Increase connection pool sizes if needed

#### 3. Cache Misses
**Symptoms**: Low cache hit ratios, repeated identical queries
**Solutions**:
- Review cache TTL settings for your use case
- Check query variation causing cache misses
- Consider semantic caching for similar queries
- Monitor cache memory usage and eviction patterns

#### 4. Authentication Errors
**Symptoms**: Access denied or permission errors
**Solutions**:
- Verify APG auth_rbac token validity
- Check user role assignments and permissions
- Review data source access policies
- Ensure tenant isolation configuration

### Performance Monitoring

#### Key Metrics to Monitor
```json
{
  "query_performance": {
    "avg_response_time_ms": "< 2000",
    "p95_response_time_ms": "< 5000", 
    "queries_per_second": "> 100",
    "error_rate": "< 0.1%"
  },
  "cache_efficiency": {
    "hit_ratio": "> 80%",
    "memory_utilization": "< 90%",
    "eviction_rate": "< 10%"
  },
  "resource_usage": {
    "cpu_utilization": "< 70%",
    "memory_usage": "< 85%",
    "network_throughput": "monitoring",
    "connection_pool_usage": "< 80%"
  }
}
```

#### Alerting Configuration
Set up alerts for critical metrics:
- **High Error Rate**: > 1% query failures
- **Performance Degradation**: > 5s average response time
- **Cache Problems**: < 50% hit ratio
- **Resource Exhaustion**: > 90% memory usage
- **Connection Issues**: > 5% connection failures

### Getting Help

#### Support Channels
- **APG Platform Support**: Your organization's APG support portal
- **Documentation**: Complete technical docs at `/docs/`
- **Community Forum**: APG user community and knowledge base
- **Professional Services**: APG consulting for complex implementations

#### Diagnostic Information
When reporting issues, include:
```json
{
  "dvrl_version": "1.0.0",
  "apg_platform_version": "2024.1",
  "tenant_id": "your-tenant-id",
  "query_id": "failing-query-id",
  "error_details": "complete error message",
  "execution_plan": "query execution plan",
  "performance_metrics": "timing and resource usage"
}
```

---

**Next Steps**: Explore the [API Reference](api_reference.md) for programmatic access or the [Developer Guide](developer_guide.md) for integration patterns.

**Document Version**: 1.0  
**Last Updated**: 2025-01-11  
**Author**: APG Platform Team