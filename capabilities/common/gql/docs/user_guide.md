# GraphQL Federation Gateway User Guide

## Overview

The GraphQL Federation Gateway (`gql_gw`) acts as a single entry point for all GraphQL queries across the platform. It manages subgraph registration, composes federated schemas, routes queries to the correct subgraph, and provides DataLoader-based batching to eliminate N+1 problems.

## Use Cases

- Federate multiple domain GraphQL APIs behind a single endpoint
- Auto-generate GraphQL SDL from `semantic_model.json` without hand-writing schemas
- Register persisted queries to avoid sending large query documents on every request
- Batch entity lookups via DataLoader (deduplication + single round-trip)
- Track all query executions for audit and performance analysis

## Quickstart

### Register a subgraph

```http
POST /api/gql/subgraphs
{
  "tenant_id": "acme",
  "name": "payments",
  "url": "http://payments-service:4001/graphql",
  "schema_sdl": "type Payment { id: ID! amount: Float! status: String! }"
}
```

### Execute a query

```http
POST /api/gql/graphql
X-Tenant-ID: acme
X-User-ID: user-123
Content-Type: application/json

{
  "query": "{ payment(id: \"pay-001\") { id amount status } }",
  "variables": {}
}
```

### Auto-generate schema from semantic model

```http
POST /api/gql/schema/auto
{
  "tenant_id": "acme",
  "semantic_model": {
    "entities": [
      {
        "name": "Customer",
        "columns": [
          {"name": "id", "type": "uuid", "nullable": false},
          {"name": "email", "type": "varchar", "nullable": false},
          {"name": "created_at", "type": "timestamp"}
        ]
      }
    ]
  }
}
```

### Register and execute a persisted query

```http
POST /api/gql/persisted
{
  "tenant_id": "acme",
  "query_id": "GetPayment",
  "document": "query GetPayment($id: ID!) { payment(id: $id) { id amount } }"
}

POST /api/gql/persisted/GetPayment/execute
{
  "tenant_id": "acme",
  "variables": {"id": "pay-001"}
}
```

### DataLoader batch

```http
POST /api/gql/dataloader/batch
{
  "tenant_id": "acme",
  "loader_key": "Customer",
  "ids": ["c1", "c2", "c1", "c3"]
}
```
Returns deduplicated results — `c1` fetched once despite appearing twice.

## Headers

| Header | Purpose |
|--------|---------|
| `X-Tenant-ID` | Override `tenant_id` in body |
| `X-User-ID` | Identify the caller for rate limiting and audit |
