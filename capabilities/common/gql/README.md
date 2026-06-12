# GraphQL Federation Gateway (gql_gw)

Federated GraphQL gateway with auto-schema from semantic_model.json, DataLoader batching, persisted queries, introspection, circuit breaking, canary traffic splitting, response caching, and query complexity analysis.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/gql/health | Service health |
| POST | /api/gql/graphql | Execute GraphQL query |
| GET | /api/gql/graphql | Introspection schema |
| GET | /api/gql/subgraphs | List subgraphs |
| POST | /api/gql/subgraphs | Register subgraph |
| GET | /api/gql/subgraphs/{name} | Get subgraph |
| PUT | /api/gql/subgraphs/{name} | Update subgraph |
| DELETE | /api/gql/subgraphs/{name} | Remove subgraph |
| GET | /api/gql/subgraphs/{name}/health | Probe subgraph |
| GET | /api/gql/subgraphs/health/all | Probe all subgraphs |
| POST | /api/gql/subgraphs/{name}/variant | Register canary variant |
| GET | /api/gql/subgraphs/{name}/traffic | Traffic split distribution |
| GET | /api/gql/schema | Composed federated schema |
| POST | /api/gql/schema/auto | Auto-generate SDL from semantic model |
| POST | /api/gql/schema/flush | Flush schema cache |
| POST | /api/gql/schema/{name}/diff | Detect breaking changes |
| POST | /api/gql/schema/{name}/versions | Publish schema version |
| GET | /api/gql/schema/{name}/versions | List schema version history |
| POST | /api/gql/schema/{name}/rollback | Rollback to prior version |
| GET | /api/gql/persisted | List persisted queries |
| POST | /api/gql/persisted | Register persisted query |
| POST | /api/gql/persisted/{id}/execute | Execute persisted query |
| DELETE | /api/gql/persisted/{id} | Delete persisted query |
| POST | /api/gql/dataloader/batch | DataLoader batch load |
| GET | /api/gql/analytics | Query analytics |
| GET | /api/gql/statistics | Gateway statistics |
| GET | /api/gql/querylog | Query execution log |
| GET | /api/gql/audit | Audit trail |
| POST | /api/gql/complexity/analyze | Analyze query complexity and depth |
| POST | /api/gql/complexity/budget | Set per-tenant complexity budget |
| POST | /api/gql/complexity/estimate | Dry-run cost estimate (no execution) |
| GET | /api/gql/cache/stats | Response cache statistics |
| POST | /api/gql/allowlist | Enable/disable allowlist mode |
| GET | /api/gql/allowlist | Get allowlist mode status |
| GET | /api/gql/circuit/{name} | Circuit breaker status |
| POST | /api/gql/circuit/{name}/result | Record subgraph success/failure |
| POST | /api/gql/circuit/{name}/reset | Manually reset circuit breaker |
| POST | /api/gql/subscribe | Subscribe to query (SSE) |

## New Features

### Query Complexity & Depth Analysis (I1/I2)

Every query can be scored before execution. The gateway walks the query document, accumulates per-field depth-weighted costs, and rejects queries that exceed the tenant's budget or exceed `max_depth` nesting.

```python
report = await svc.analyze_query_complexity(
    "acme",
    "{ orders { items { product { variants { images { url } } } } } }",
    max_depth=5,
    cost_budget=200,
)
# report["allowed"] == False, report["rejection_reason"] == "QUERY_TOO_DEEP"
```

Set custom budgets and per-field costs:
```python
await svc.set_complexity_budget("acme", budget=500, field_costs={"search": 50, "list": 20})
```

### Response Caching with TTL (I4)

Read-only queries can be cached in-process. The cache key is derived from `(tenant, normalized_query, sorted_variables)`. Mutations and subscriptions are never cached.

```python
await svc.cache_response("acme", query, response, ttl_seconds=120)
hit = await svc.get_cached_response("acme", query)  # returns cached data or None
stats = await svc.get_cache_stats("acme")
```

### Schema Registry with Versioning (I7)

Track SDL history per subgraph. Breaking-change detection gates promotion to stable automatically.

```python
ver = await svc.publish_schema_version(
    "acme", "payments", sdl=NEW_SDL, version="2.1.0",
    changelog="Add refund field", promote_to_stable=True,
)
await svc.rollback_schema_version("acme", "payments", target_version="2.0.0")
history = await svc.list_schema_versions("acme", "payments")
```

### Circuit Breaker per Subgraph (I9)

State machine per subgraph: `CLOSED → OPEN → HALF_OPEN → CLOSED`. OPEN circuits fast-fail with `SUBGRAPH_UNAVAILABLE`. Auto-transitions to HALF_OPEN after `recovery_timeout_s`.

```python
await svc.record_subgraph_result("acme", "payments", success=False)
status = await svc.get_circuit_breaker_status("acme", "payments")
# status["state"] == "OPEN" after failure_threshold failures
await svc.reset_circuit_breaker("acme", "payments")  # operator override
```

### Canary Traffic Splitting (I12)

Register multiple variants of a subgraph with relative weights. `execute_query` samples the variant via weighted random selection, skipping OPEN circuits.

```python
await svc.register_subgraph_variant(
    "acme", name="payments", url="http://payments-v2:4001/graphql",
    variant="v2-canary", weight=10,
)
split = await svc.get_traffic_split("acme", "payments")
# split["variants"] shows v1=90%, v2-canary=10%
```

### Persisted Query Allowlist Mode (I13)

Lock down the gateway to reject any ad-hoc query not pre-registered. Returns `QUERY_NOT_ALLOWED` with the query hash for registration.

```python
await svc.set_allowlist_mode("acme", enabled=True)
# Now only registered persisted queries will execute
```

### Dry-Run Cost Estimation (I11)

Full pre-flight analysis without any subgraph calls — ideal for CI pipelines.

```python
report = await svc.estimate_query_cost("acme", query, variables, user_id="ci-bot")
# report["will_be_allowed"], report["cache_hit"], report["estimated_subgraph_calls"]
```

### Subscriptions via AsyncGenerator (I5)

```python
async for event in svc.subscribe_query("acme", query, max_events=5, poll_interval_s=2.0):
    if event["type"] == "next":
        process(event["payload"])
    elif event["type"] == "complete":
        break
```

### Field-Level Authorization via `@auth` Policies (I3)

Register per-field role requirements on any subgraph.  `execute_query_authorized` enforces them before forwarding to the subgraph — no duplication across downstream services.

```python
await svc.register_field_auth_policy(
    "acme", "payments",
    {"Payment.amount": ["finance", "admin"], "Payment.cardLast4": ["admin"]},
)
# Raises PermissionError("FIELD_ACCESS_DENIED") for callers without matching roles
result = await svc.execute_query_authorized(
    "acme", "{ amount status }", user_roles=["finance"], user_id="user-1"
)
```

### Distributed Tracing Spans (I6)

Open/close trace spans per gateway operation and record per-subgraph child spans for P50/P99 latency analysis.

```python
ctx = await svc.start_trace("acme", "CheckoutQuery", user_id="u1")
await svc.record_subgraph_span("acme", ctx["trace_id"], "payments", duration_ms=4.2, field_count=3)
summary = await svc.finish_trace("acme", ctx["trace_id"])
# summary["duration_ms"], summary["span_count"], summary["total_subgraph_ms"]
traces = await svc.list_traces("acme", status="OK")
```

### Query Normalization & Deduplication (I8)

Canonicalize query documents before hashing — two logically identical queries from different clients produce the same cache key and persisted query hash.

```python
result = await svc.normalize_query(
    "acme",
    "  # fetch user\n{ status   amount }",
)
# result["normalized_query"] == "{ amount status }"  (fields sorted, comments stripped)
# result["is_duplicate"] == True if hash already registered as persisted query
```

### Federated Entity Resolution — `_entities` query (I10)

Resolve cross-subgraph entity references using Apollo Federation semantics.  The gateway groups representations by `__typename`, fans out to owning subgraphs, and merges results in order using DataLoader batching.

```python
resolved = await svc.resolve_entities(
    "acme",
    [
        {"__typename": "User", "id": "u1"},
        {"__typename": "Payment", "id": "pay-001"},
        {"__typename": "User", "id": "u2"},
    ],
)
# resolved["entities"][0]["_resolved_from"] == "users-subgraph"
```

### Webhook Integration for Schema Change Events (I14)

Register HTTP endpoints to receive push notifications when subgraphs are registered, schemas promoted, or breaking changes detected.  Deliveries are HMAC-SHA256 signed when a secret is provided.

```python
await svc.register_webhook(
    "acme",
    url="https://ci.example.com/gql-hook",
    events=["schema_version_published", "subgraph_registered"],
    secret="s3cr3t",
)
# Automatically triggered by _emit; also callable directly:
delivery = await svc.deliver_webhook("acme", "schema_version_published", {"version": "2.1.0"})
# delivery["matched_webhooks"] == 1
```

### Multi-Region Subgraph Affinity Routing (I15)

Tag subgraphs with region labels.  `execute_query_with_region_affinity` routes to the closest healthy replica, falling back to lowest-latency globally when no same-region replica is available.

```python
await svc.register_subgraph_region("acme", "payments", region="us-east-1", latency_ms_p50=5.0)
await svc.register_subgraph_region("acme", "payments-eu", region="eu-west-1", latency_ms_p50=22.0)

result = await svc.execute_query_with_region_affinity(
    "acme", "{ payment(id: \"p1\") { amount } }",
    preferred_region="us-east-1",
)
# result["extensions"]["routing"]["routing_reason"] == "same_region"

topology = await svc.get_region_topology("acme")
# topology["regions"] == {"us-east-1": [...], "eu-west-1": [...]}
```

---

## World-Class Enhancements (v2.0)

- **I1.** GraphQL Federation Gateway — World Class Improvements
- **I2.** Query Complexity Analysis & Cost Limiting
- **I3.** Query Depth Limiting
- **I4.** Field-Level Authorization via Directives
- **I5.** Response Caching with TTL and Cache-Control
- **I6.** Subscription Support via SSE Transport
- **I7.** Distributed Tracing with OpenTelemetry
- **I8.** Schema Registry with Semantic Versioning and Changelogs
- **I9.** Automatic Query Normalization and Deduplication
- **I10.** Circuit Breaker per Subgraph
- **I11.** Federated Entity Resolution with
- **I12.** Query Cost Estimation API (Dry Run)
- **I13.** Subgraph Canary / Traffic Splitting
- **I14.** Persisted Query Allowlist Mode (Security Lockdown)
- **I15.** Webhook Integration for Schema Change Events

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
