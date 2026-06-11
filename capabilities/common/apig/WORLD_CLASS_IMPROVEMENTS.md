# APIG World-Class Improvements

15 targeted improvements to elevate `capabilities/common/apig` from a strong
gateway control-plane to a reference-class API platform.

---

### I1. Monetary Precision for Billing & Quota Monetisation
**Category**: Correctness / Finance
**Justification**: All billing counters and quota-overage charges currently flow
through `float`. IEEE-754 rounding at sub-cent granularity is unacceptable for
revenue-critical APIs. `Decimal` with a fixed precision context is the only safe
choice — the same decision made by Stripe, Adyen, and every mature payment
processor.
**Implementation**: Replace every money field (`quota_price`, `overage_rate`,
`credit_balance`) with `Decimal` annotated via a project-local
`MoneyDecimal = Annotated[Decimal, AfterValidator(lambda v: v.quantize(Decimal("0.0001")))]`.
Add a `billing_record` method to `service.py` that produces a `BillingEvent`
Pydantic model with `Decimal` totals, persisted to a `ag_billing_events` table.
**Competitor**: Stripe Billing, AWS API Gateway usage plans with cent-precision metering.

---

### I2. Multi-Tier Token-Bucket Rate Limiting with Redis Lua Atomicity
**Category**: Performance / Correctness
**Justification**: The current `throttle_apply` stores intent only. Production
gateways require atomic counter increments to prevent over-admission under
concurrent load. A Lua script executing `EVAL` on Redis guarantees
read-modify-write atomicity without distributed locks.
**Implementation**: Add `async def rate_limit_check(self, key: str, capacity: int, refill_rate: float, cost: int = 1) -> RateLimitResult`
that executes a Redis Lua token-bucket script. Return `RateLimitResult` with
`allowed: bool`, `remaining: int`, `retry_after_ms: int`. Expose `X-RateLimit-*`
headers via the response transform pipeline.
**Competitor**: Kong Gateway's `rate-limiting-advanced` plugin, Cloudflare Workers
rate limiting with token-bucket semantics.

---

### I3. mTLS Client Certificate Validation at the Edge
**Category**: Security
**Justification**: Routes marked `route_exposure=external` accept public traffic
but the current security model relies solely on bearer tokens. mTLS provides
cryptographic proof of client identity — mandatory for PSD2, ISO 27001, and
banking API standards. APIG already tracks `mtls_enabled` on routes but never
enforces it.
**Implementation**: Add `async def mtls_validate(self, gateway_id: str, route_id: str, client_cert_pem: str) -> MTLSValidationResult`
that parses the PEM, checks the subject CN against the registered consumer,
verifies the issuer against the tenant's trusted CA store, and evaluates
certificate expiry. Persist outcome to `ag_security_events`.
**Competitor**: AWS API Gateway mutual TLS, Apigee mTLS enforcement, Kong's
`mtls-auth` plugin.

---

### I4. Semantic API Versioning with Backward-Compatibility Scoring
**Category**: API Lifecycle
**Justification**: Operators currently sunset versions with a date and a URL;
there is no automated signal about *how* breaking the change is. OpenAPI diff
scoring lets platform teams enforce compatibility gates in CI before traffic
shifts go live, eliminating surprise 4xx spikes post-deploy.
**Implementation**: Add `async def version_compat_score(self, spec_old: dict, spec_new: dict) -> CompatScore`
that diffs `paths`, `components/schemas`, and `requestBody` between two OpenAPI
dicts. Score breaking changes (removed paths, narrowed types) separately from
additive changes. Return `CompatScore(breaking: int, additive: int, score: float,
details: list[str])`.
**Competitor**: Bump.sh, Optic API diff, AWS API Gateway deployment validation.

---

### I5. Distributed Tracing Context Propagation (W3C Trace Context)
**Category**: Observability
**Justification**: APIG processes requests across edge and upstream hops but does
not inject W3C `traceparent` / `tracestate` headers. Without end-to-end trace
context, engineers cannot correlate gateway latency with upstream slow calls in
Jaeger/Tempo — a top complaint in post-incident reviews at every scale-up company.
**Implementation**: Add `async def inject_trace_context(self, request: AgHttpRequest, parent_span_id: str | None = None) -> AgHttpRequest`
that generates a compliant `traceparent` header (`{version}-{trace_id}-{span_id}-{flags}`)
and sets it on the mutated request. Add `async def extract_trace_context(self, request: AgHttpRequest) -> TraceContext | None`
for inbound extraction.
**Competitor**: AWS X-Ray header injection, Kong Zipkin plugin, Envoy automatic
traceparent propagation.

---

### I6. Adaptive Circuit Breaker with Half-Open Probe Window
**Category**: Resilience
**Justification**: `circuit_break` stores configuration only; there is no runtime
state machine. Without a proper open → half-open → closed transition, a tripped
circuit stays open until an operator manually resets it, turning momentary blips
into extended outages. Netflix Hystrix popularised this pattern in 2012; it is
table stakes for production gateways.
**Implementation**: Add `CircuitBreakerState` dataclass per upstream (
`state: Literal["closed","open","half_open"]`, `failure_count: int`,
`opened_at: datetime | None`, `probe_successes: int`).
Add `async def circuit_breaker_tick(self, upstream_id: str, success: bool) -> CircuitBreakerState`
that transitions state and persists events to `ag_circuit_events`.
**Competitor**: Resilience4j, AWS App Mesh circuit breaking, Envoy outlier detection.

---

### I7. API Key Rotation with Zero-Downtime Dual-Active Window
**Category**: Security / Operations
**Justification**: `developer_onboard` issues a static API key. Key compromise
requires immediate revocation, which breaks consumers. A dual-active window where
both old and new keys are valid for a configurable overlap period allows
applications to rotate without downtime — the same model used by AWS, GCP, and
Twilio.
**Implementation**: Add `async def rotate_api_key(self, developer_id: str, app_name: str, overlap_seconds: int = 3600) -> ApiKeyRotationResult`
that generates a new key, stores both with expiry timestamps in
`ag_api_keys`, and returns `ApiKeyRotationResult(new_key, old_key_expires_at)`.
Add `async def validate_api_key(self, key: str) -> ApiKeyRecord | None` with dual
lookup.
**Competitor**: Stripe API key rotation, Twilio secondary key model, GCP service
account key versioning.

---

### I8. Request Body Schema Validation Against OpenAPI Spec
**Category**: Data Quality / Security
**Justification**: Malformed request bodies reach upstreams and cause cryptic 500s
that are hard to attribute to consumers vs. platform bugs. Gateway-level schema
validation catches these at the edge, generates structured 422 responses, and
reduces upstream error budgets. Kong's `request-validator` plugin reduced payload
errors by 40% in published case studies.
**Implementation**: Add `async def validate_request_body(self, request: AgHttpRequest, schema: dict) -> ValidationResult`
that runs `jsonschema.validate` against the schema extracted from the registered
OpenAPI spec for the matched route. Return `ValidationResult(valid: bool,
errors: list[str], validated_at: datetime)`.
**Competitor**: Kong `request-validator`, AWS API Gateway model validation, Apigee
flow callout with schema validation.

---

### I9. Geographic Traffic Routing and Geo-Blocking
**Category**: Compliance / Performance
**Justification**: Many enterprise and fintech customers operate under regulatory
frameworks (GDPR, data residency laws) that prohibit serving certain jurisdictions.
APIG currently tracks `geo_restrictions` in `AgSecurityPolicy` but has no
enforcement path. Geo-blocking also protects against region-specific DDoS bursts.
**Implementation**: Add `async def geo_check(self, client_ip: str, allowed_countries: list[str] | None, blocked_countries: list[str] | None) -> GeoCheckResult`
using the `geoip2` MaxMind library or a local GeoLite2 database. Return
`GeoCheckResult(allowed: bool, country_code: str, region: str, reason: str | None)`.
Wire into `process_request` before authentication.
**Competitor**: Cloudflare Geo Blocking, AWS WAF geo match rules, Fastly geo
conditions.

---

### I10. WebSocket and Server-Sent Events (SSE) Proxying
**Category**: Protocol Support
**Justification**: Modern APIs increasingly use WebSocket for real-time data and
SSE for server-push. APIG currently models only HTTP request/response pairs. An
AI-powered gateway that cannot proxy WebSocket or SSE is invisible to the growing
category of streaming LLM APIs, trading feeds, and live dashboard backends.
**Implementation**: Add `async def websocket_proxy_session(self, gateway_id: str, client_ws, upstream_url: str, headers: dict[str, str]) -> WebSocketSessionSummary`
using `aiohttp` for upstream WS connections and Python's `websockets` for
client-side. Track `messages_proxied`, `bytes_relayed`, `session_duration_ms`.
**Competitor**: Kong WebSocket proxying, AWS API Gateway WebSocket APIs, Envoy
HTTP/1.1 upgrade handling.

---

### I11. Policy-as-Code with OPA/Rego Integration
**Category**: Governance / Security
**Justification**: Natural-language policy generation (Ollama) produces opaque
blobs that operators cannot audit. Rego policies are human-readable, version-
controlled, and evaluable offline. OPA's decision log integrates directly with
SIEM tools. HashiCorp, Goldman Sachs, and Netflix use OPA for gateway authZ.
**Implementation**: Add `async def evaluate_rego_policy(self, policy_rego: str, input_data: dict) -> OPADecision`
that calls a local OPA server at `http://localhost:8181/v1/data` via `aiohttp`.
Return `OPADecision(allow: bool, reason: str, matched_rules: list[str])`. Persist
evaluated policies to `ag_opa_policies` with version hashes.
**Competitor**: Kong OPA plugin, AWS Verified Permissions (Cedar), Apigee policy
decision points.

---

### I12. Response Caching with Content-Addressed Storage and ETags
**Category**: Performance
**Justification**: `AgCacheConfig` defines TTLs but the gateway has no actual
cache store or ETag generation. Without ETags, clients always receive full
responses even when content has not changed. Content-addressed storage (SHA-256
of response body) enables `If-None-Match` 304 responses, cutting bandwidth and
upstream load by 30–60% for read-heavy APIs.
**Implementation**: Add `async def cache_get_or_fetch(self, cache_key: str, ttl_seconds: int, fetch_fn: Callable) -> CacheResult`
that stores `(etag: str, body: bytes, headers: dict, expires_at: datetime)` in
Redis. Add `async def etag_response(self, request: AgHttpRequest, response: AgHttpResponse) -> AgHttpResponse`
that computes SHA-256 etag and returns 304 when `If-None-Match` matches.
**Competitor**: Varnish Cache, AWS CloudFront ETags, Nginx proxy_cache with
etag support.

---

### I13. Canary Release Automation with Statistical Traffic Analysis
**Category**: Deployment Safety
**Justification**: `traffic_split_apig` configures a split percentage but provides
no feedback loop. Production canaries require automated error-rate and latency
comparison between canary and stable cohorts to trigger automatic rollback before
human review. LinkedIn and Uber built this into their gateway layers to reduce
canary-induced incidents by 70%.
**Implementation**: Add `async def canary_analyse(self, gateway_id: str, route_id: str, canary_version: str, stable_version: str, window_minutes: int = 10) -> CanaryAnalysis`
that computes `error_rate_delta`, `p99_latency_delta`, and `statistical_significance`
(using a chi-squared test on error counts). Return `CanaryAnalysis` with
`recommendation: Literal["promote","rollback","continue"]`.
**Competitor**: Argo Rollouts, Flagger canary analysis, Netflix's Kayenta automated
canary analysis service.

---

### I14. GraphQL Gateway with Query Depth and Complexity Limiting
**Category**: Protocol Support / Security
**Justification**: REST-only gateways lose relevance as GraphQL adoption grows.
Unbounded GraphQL queries are a denial-of-service vector; depth and complexity
limits are the standard mitigation. A gateway that understands GraphQL introspection
can also enforce schema-driven field-level authorization — not possible at the
HTTP layer alone.
**Implementation**: Add `async def graphql_protect(self, query: str, max_depth: int = 10, max_complexity: int = 1000) -> GraphQLGuardResult`
that parses the query AST using `graphql-core`, computes depth and complexity
scores, and returns `GraphQLGuardResult(allowed: bool, depth: int, complexity: int,
violations: list[str])`.
**Competitor**: Apollo Router, Hasura's query limits, GraphQL Armor middleware.

---

### I15. Tenant Billing Dashboard with Decimal-Precise Usage Aggregation
**Category**: Monetisation / Multi-Tenancy
**Justification**: The current `usage_analytics` returns raw counters. A monetised
API platform requires per-tenant billing periods, tiered pricing (`free`, `pro`,
`enterprise`), overage charges calculated with `Decimal` precision, and invoice
generation. Without this, `apig` cannot support the platform's monetisation
roadmap.
**Implementation**: Add `async def billing_aggregate(self, tenant_id: str, billing_period_start: datetime, billing_period_end: datetime, tier: str = "pro") -> BillingStatement`
that accumulates request counts per gateway, applies tier pricing with `Decimal`
arithmetic, computes overage charges, and returns a `BillingStatement` Pydantic
model with `total_due: Decimal`, `line_items: list[BillingLineItem]`, and
`currency: str`. Guard `tenant_id` with `guard_tenant_id`.
**Competitor**: AWS API Gateway usage plans + billing, Azure API Management
subscription tiers, Apigee monetisation add-on.
