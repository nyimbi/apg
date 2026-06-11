# GraphQL Federation Gateway — World Class Improvements

---

### I1. Query Complexity Analysis & Cost Limiting
**Category:** Security / Performance
**Justification:** Unbounded GraphQL queries are a primary DoS vector. Production gateways (Apollo Router, GraphQL Hive) reject queries exceeding a configurable complexity budget before execution, preventing runaway joins and nested list amplification.
**Implementation:** Walk the parsed query AST, assign per-field costs from a configurable cost map, accumulate depth-weighted scores, and raise `PermissionError` if budget exceeded. Store per-tenant cost budgets in a dict keyed by `tenant_id`.
**Competitor Reference:** Apollo Federation `maxQueryComplexity`, Stellate query cost analysis, GraphQL Armor `costLimit` plugin.

---

### I2. Query Depth Limiting
**Category:** Security
**Justification:** Deeply nested queries bypass simple complexity checks. Attackers craft exponentially nested fragments to exhaust memory. Depth guards are a mandatory baseline in any hardened gateway.
**Implementation:** Recursive SDL AST traversal counting nesting level. Configurable `max_depth` per tenant (default 10). Reject at parse stage before routing to any subgraph, returning a structured `QUERY_TOO_DEEP` error.
**Competitor Reference:** `graphql-depth-limit` npm, Hasura `depth_limit`, AWS AppSync query depth controls.

---

### I3. Field-Level Authorization via Directives
**Category:** Security / Access Control
**Justification:** Row-level and field-level security is a hard requirement for multi-tenant SaaS. The gateway is the correct enforcement plane — pushing auth into each subgraph leads to duplication and bypass risks.
**Implementation:** Parse `@auth(roles: [...])` directives in registered SDL. Before executing, compare field selections against caller's roles (passed via `user_roles` kwarg). Strip unauthorized fields from the response or raise `FORBIDDEN`.
**Competitor Reference:** WunderGraph field-level auth, Apollo `@requiresScopes`, GraphQL Shield.

---

### I4. Response Caching with TTL and Cache-Control
**Category:** Performance
**Justification:** Repeated identical queries to the same subgraph waste downstream compute. HTTP-layer caching is coarse; a gateway-level semantic cache keyed on `(tenant, operation_hash, variables_hash)` can serve read-heavy workloads orders of magnitude faster.
**Implementation:** `asyncio`-friendly in-process LRU with per-entry TTL (seconds). Cache key = `sha256(tenant + op_hash + canonical_variables_json)`. Cache-Control directives in `@cacheControl(maxAge: N)` SDL annotations override default TTL.
**Competitor Reference:** Apollo Server `cacheControl` plugin, Stellate CDN-level caching, Grafbase edge caching.

---

### I5. Subscription Support via SSE Transport
**Category:** Feature Completeness
**Justification:** Real-time data delivery is table stakes for event-driven architectures. GraphQL subscriptions over Server-Sent Events are simpler to operate than WebSocket subscriptions and work through standard HTTP/2 proxies.
**Implementation:** `subscribe_query()` returns an `AsyncGenerator` yielding delta payloads. Gateway maintains per-tenant subscription registry with topic → subscriber fan-out. Subgraph subscription URLs registered separately with optional auth token.
**Competitor Reference:** GraphQL WS protocol, Apollo Router subscriptions, Grafbase SSE transport.

---

### I6. Distributed Tracing with OpenTelemetry
**Category:** Observability
**Justification:** Gateway sits at the critical path for all GraphQL traffic. Without span-level telemetry, diagnosing latency spikes, fan-out amplification, or subgraph regressions is guesswork. OTEL is vendor-neutral and required for production SRE.
**Implementation:** Attach `trace_id`, `span_id`, and per-subgraph child spans to every `execute_query` call. Emit `gql.field.count`, `gql.subgraph.duration_ms`, `gql.cache.hit` metrics as OTEL counters and histograms. Export via OTLP.
**Competitor Reference:** Apollo Studio trace reporting, GraphQL Hive telemetry, Grafbase OTEL integration.

---

### I7. Schema Registry with Semantic Versioning and Changelogs
**Category:** Governance / DevOps
**Justification:** Schema drift between subgraph versions is the leading cause of federation outages. A registry that tracks SDL history with semver and auto-generates changelogs enables CI-gated schema promotions and rollback.
**Implementation:** `publish_schema_version()` stores `(tenant, subgraph, semver, sdl, changelog, published_at)`. Breaking-change detection (field removal, type change, non-null strengthening) gates promotion to `stable`. `rollback_schema()` reinstates a prior version.
**Competitor Reference:** GraphQL Hive schema registry, Apollo Schema Registry, WunderGraph schema versioning.

---

### I8. Automatic Query Normalization and Deduplication
**Category:** Performance / Reliability
**Justification:** Clients send logically identical queries with different formatting, comments, or field ordering. Normalizing before hashing dramatically improves cache hit rates and prevents duplicate persisted query registrations.
**Implementation:** Strip comments, sort fields alphabetically, remove aliases, compact whitespace before computing `query_hash`. Store canonical form alongside raw document in persisted query records.
**Competitor Reference:** Apollo Client query normalization, Relay compiler query optimization, Grafast document normalization.

---

### I9. Circuit Breaker per Subgraph
**Category:** Reliability
**Justification:** A slow or failing subgraph should not cascade failures to the entire gateway. Circuit breakers with half-open probing isolate faults and let healthy subgraphs continue serving traffic.
**Implementation:** Per-subgraph state machine: `CLOSED → OPEN (threshold failures) → HALF_OPEN (probe) → CLOSED`. Tracked in `circuit_breakers` dict. `execute_query` fast-fails with `SUBGRAPH_UNAVAILABLE` for OPEN circuits. Configurable `failure_threshold` and `recovery_timeout_s`.
**Competitor Reference:** Apollo Router circuit breaking, Netflix Hystrix pattern, Grafbase retry/circuit-break policies.

---

### I10. Federated Entity Resolution with `_entities` Query
**Category:** Feature Completeness
**Justification:** True Apollo Federation requires the gateway to stitch entity references across subgraphs using the `_entities(representations: [Any!]!)` query. Without this, cross-subgraph type extension (e.g., `User` extended by `Orders`) is impossible.
**Implementation:** `resolve_entities()` accepts a list of `__typename + key field` representations, fans out to the owning subgraph per type, and merges results maintaining order. Uses DataLoader batching under the hood.
**Competitor Reference:** Apollo Federation `_entities` specification, Cosmo Federation, GraphQL Mesh federation mode.

---

### I11. Query Cost Estimation API (Dry Run)
**Category:** Developer Experience
**Justification:** Developers need to understand the cost of a query before shipping it to production. A dry-run endpoint returns complexity score, estimated subgraph calls, cache hit probability, and rate-limit status without executing the query.
**Implementation:** `estimate_query_cost()` runs complexity + depth analysis, checks cache, models fan-out subgraph count, and returns a structured cost report. No downstream subgraph calls made. Useful in CI pipelines.
**Competitor Reference:** Apollo Studio query inspector, GraphQL Armor dry-run, Stellate cost estimation API.

---

### I12. Subgraph Canary / Traffic Splitting
**Category:** Reliability / Deployment
**Justification:** Rolling out a new subgraph version to 100% of traffic is risky. Canary routing with configurable traffic weights (e.g., 10% to v2, 90% to v1) allows safe progressive rollout with automatic rollback on error rate threshold.
**Implementation:** Subgraph records gain `weight` and `variant` fields. `_route_subgraph()` uses weighted random selection across same-name variants. Error rates per variant tracked; automatic OPEN circuit if variant error rate exceeds threshold.
**Competitor Reference:** Apollo Router traffic shaping, Istio VirtualService weights, AWS AppSync custom resolvers.

---

### I13. Persisted Query Allowlist Mode (Security Lockdown)
**Category:** Security
**Justification:** In production, accepting arbitrary ad-hoc queries is a security risk. Allowlist mode rejects any query that is not in the persisted query registry, eliminating injection attacks and dramatically reducing attack surface.
**Implementation:** Per-tenant `allowlist_mode: bool` flag. When enabled, `execute_query` rejects queries whose hash does not match a registered persisted query `doc_hash`. Returns `QUERY_NOT_ALLOWED` error with the computed hash for registration.
**Competitor Reference:** Apollo Client APQ protocol, Relay persisted queries, Fastify Mercurius persisted queries.

---

### I14. Webhook Integration for Schema Change Events
**Category:** DevOps / Integration
**Justification:** External CI systems, Slack bots, and schema registries need to react to subgraph registration, schema promotion, and breaking change detection in real time. Push-based webhooks are the standard integration pattern.
**Implementation:** `register_webhook()` stores `(tenant, url, events[], secret)` records. `_emit()` extended to fan-out matching webhook registrations via HTTP POST with HMAC-SHA256 signature header. Async fire-and-forget with retry backoff.
**Competitor Reference:** GraphQL Hive webhooks, Apollo Studio schema check webhooks, Grafbase webhook triggers.

---

### I15. Multi-Region Subgraph Affinity Routing
**Category:** Performance / Global Scale
**Justification:** Latency-sensitive workloads require routing queries to the geographically closest subgraph replica. A gateway-level affinity policy (by `X-Region` header or tenant config) reduces p99 latency by 30–70% for globally distributed deployments.
**Implementation:** Subgraphs registered with optional `region` field. `execute_query` accepts `preferred_region` kwarg. `_route_subgraph()` prefers same-region subgraphs, falls back to any healthy instance. Latency stats per region tracked for adaptive routing.
**Competitor Reference:** Apollo Router `@connect` subgraph selection, Cloudflare Workers GraphQL routing, WunderGraph multi-region edge deployment.
