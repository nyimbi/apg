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

---

## Advanced Features

### Field-Level Authorization

Register access policies per field.  The gateway enforces them centrally — no need to duplicate auth logic in each subgraph.

```http
POST /api/gql/auth/policies
{
  "tenant_id": "acme",
  "subgraph": "payments",
  "field_policies": {
    "Payment.amount": ["finance", "admin"],
    "Payment.cardLast4": ["admin"]
  }
}
```

Pass `user_roles` when executing through `execute_query_authorized`.  Fields the caller lacks access to raise `FIELD_ACCESS_DENIED` before any subgraph call is made.

---

### Distributed Tracing

Every query can be wrapped in a trace span.  Open a trace, execute the query, record per-subgraph spans, then close.

```http
POST /api/gql/traces
{ "tenant_id": "acme", "operation_name": "CheckoutQuery", "user_id": "u1" }
# Returns: { "trace_id": "trace-abc123", ... }

POST /api/gql/traces/{trace_id}/spans
{ "subgraph": "payments", "duration_ms": 4.2, "field_count": 3, "cache_hit": false }

POST /api/gql/traces/{trace_id}/finish
{ "status": "OK" }
# Returns: { "duration_ms": 8.4, "span_count": 2, "total_subgraph_ms": 7.1, ... }
```

---

### Query Normalization

Normalize ad-hoc queries to their canonical form before caching or registering as persisted queries.  Two logically equivalent queries always produce the same hash after normalization.

```http
POST /api/gql/normalize
{
  "tenant_id": "acme",
  "query": "  # fetch user\n{ email   id }",
  "variables": {}
}
# Returns: { "normalized_query": "{ email id }", "normalized_hash": "...", "is_duplicate": false }
```

---

### Federated Entity Resolution

Resolve cross-subgraph entity references exactly as Apollo Federation specifies.  The gateway groups representations by `__typename`, fans out to owning subgraphs, and merges results preserving original order.

```http
POST /api/gql/entities
{
  "tenant_id": "acme",
  "representations": [
    {"__typename": "User", "id": "u1"},
    {"__typename": "Payment", "id": "pay-001"}
  ]
}
# Returns: { "entities": [...], "subgraph_calls": [...] }
```

---

### Webhook Integration

Register HTTP endpoints to receive push notifications on gateway events.  Deliveries include an `X-GQL-Signature` header (HMAC-SHA256) when a secret is configured.

```http
POST /api/gql/webhooks
{
  "tenant_id": "acme",
  "url": "https://ci.example.com/gql-hook",
  "events": ["schema_version_published", "subgraph_registered"],
  "secret": "s3cr3t"
}
```

Supported event types: `subgraph_registered`, `subgraph_updated`, `subgraph_deleted`, `schema_composed`, `schema_version_published`, `schema_version_rolled_back`, `circuit_breaker_reset`, `allowlist_mode_changed`, `subscription_started`, `trace_finished`, `entities_resolved`.

Use `"events": ["*"]` to subscribe to all events.

---

### Multi-Region Affinity Routing

Tag subgraphs with region labels and execute queries with a preferred region.  The gateway routes to the closest healthy replica, falls back to lowest-latency globally.

```http
POST /api/gql/subgraphs/payments/region
{
  "tenant_id": "acme",
  "region": "us-east-1",
  "latency_ms_p50": 5.0
}

POST /api/gql/graphql/regional
{
  "tenant_id": "acme",
  "query": "{ payment(id: \"p1\") { amount } }",
  "preferred_region": "us-east-1"
}
# extensions.routing.routing_reason == "same_region" | "latency_fallback"
```

Inspect the full region topology:

```http
GET /api/gql/regions?tenant_id=acme
# Returns: { "regions": { "us-east-1": [...], "eu-west-1": [...] }, "region_count": 2 }
```

---

## Error Reference

| Error Code | Cause | Resolution |
|-----------|-------|-----------|
| `QUERY_TOO_DEEP` | Nesting depth exceeds `max_depth` | Reduce nesting or increase budget via `set_complexity_budget` |
| `QUERY_TOO_COMPLEX` | Complexity score exceeds cost budget | Paginate results or increase budget |
| `SUBGRAPH_UNAVAILABLE` | Circuit breaker is OPEN | Wait for recovery window or call `reset_circuit_breaker` |
| `QUERY_NOT_ALLOWED` | Allowlist mode active, query not registered | Register as persisted query or disable allowlist mode |
| `FIELD_ACCESS_DENIED` | Caller roles insufficient for requested field | Acquire required role or request field removal |
| `rate_limit_exceeded` | Per-user request rate exceeded | Back off and retry after 60s window resets |
| `NO_HEALTHY_SUBGRAPH` | All subgraphs unavailable in region | Check circuit breaker status; add cross-region fallback |
