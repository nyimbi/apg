# APG Reliability Framework — World-Class Improvements

15 improvements to elevate the framework from solid engineering to production-grade infrastructure matching or exceeding industry-leading reliability systems.

---

### I1. Adaptive Circuit Breaker with Sliding Window Statistics
**Category**: Circuit Breaking | **Justification**: The current threshold is a simple consecutive-failure counter. Netflix Hystrix and AWS SDK v3 use sliding-window failure-rate tracking — e.g., "open if 50% of the last 100 calls failed in the last 10s". This eliminates false trips on isolated transient errors and accurately detects sustained degradation without requiring exact consecutive failures. At high QPS a threshold=5 fires in milliseconds; at low QPS it may never fire even when 80% of calls are failing. | **Implementation**: Add `window_size: int = 100` and `window_duration: float = 60.0` params to `CircuitBreaker`. Maintain a `deque` of `(timestamp, success: bool)` tuples. Compute failure rate over the window on each call. Open only when sample count >= `min_calls` (default 20) AND failure rate >= `failure_rate_threshold` (default 0.5). | **Competitor**: Netflix Hystrix sliding window, Resilience4j `SlidingWindowCircuitBreaker`, AWS SDK v3 adaptive retry

---

### I2. Bulkhead Pattern — Concurrency-Limited Resource Pools
**Category**: Isolation | **Justification**: Without bulkheads a single slow dependency (e.g., Ollama inference) can consume all asyncio concurrency and starve unrelated operations. Bulkheads, used by Netflix Hystrix and Envoy proxy, isolate resource consumption per dependency so a saturated ML backend cannot block payment processing. The current framework has no concurrency caps on individual external calls. | **Implementation**: Add `Bulkhead` class using `asyncio.Semaphore`. `async with bulkhead.acquire()` blocks when the semaphore is exhausted and raises `BulkheadFullError` after a configurable `max_wait`. Add `@bulkhead_protected(name, max_concurrent=10, max_wait=1.0)` decorator. Expose current occupancy in `.status()`. | **Competitor**: Netflix Hystrix threadpool/semaphore isolation, Envoy circuit breaker `max_connections`, Resilience4j `Bulkhead`

---

### I3. Retry Orchestrator with Exponential Backoff, Jitter, and Budget
**Category**: Recovery | **Justification**: The framework has no retry primitive — callers implement ad-hoc retry loops, inevitably without jitter (causing thundering herds) or retry budgets (causing cascading overload). Google SRE uses "retry budgets" — limit total retries across all callers to a fraction (e.g., 10%) of traffic. AWS SDK v3 uses full jitter by default. | **Implementation**: Add `retry_async(fn, max_attempts=3, base_delay=0.1, max_delay=30.0, jitter=True, retry_on=(Exception,), budget: RetryBudget | None = None)`. Add `RetryBudget` class with a sliding token bucket: `n_retries / n_total_calls <= budget_fraction`. Add `@with_retry(...)` decorator. Integrate with circuit breaker — don't retry `CircuitOpenError`. | **Competitor**: AWS SDK v3 retry middleware, Google Cloud client libraries, `tenacity` library

---

### I4. NATS-Backed Distributed Idempotency Store
**Category**: Exactly-Once Semantics | **Justification**: The current `IdempotencyRegistry` is in-process only. In a multi-instance deployment (Kubernetes scale-out), two instances can receive the same payment request simultaneously and both execute it. A production system needs distributed idempotency using a shared store with compare-and-set semantics. Redis is the common choice; NATS JetStream KV provides the same capability within the APG stack without introducing a new dependency. | **Implementation**: Add `NatsIdempotencyStore` implementing a `IdempotencyStore` protocol (abstract base). Uses NATS JetStream KV with `create` operation for atomic "set if absent". On hit, returns the serialized result. `IdempotencyRegistry` accepts `store: IdempotencyStore | None` — if None, uses the in-process `OrderedDict`. | **Competitor**: Stripe's idempotency keys (Redis), Temporal workflow IDs, AWS SQS deduplication IDs

---

### I5. Structured Event Emission via OpenTelemetry Spans
**Category**: Observability | **Justification**: The framework currently logs with Python `logging`. Production reliability systems (Datadog APM, Honeycomb, AWS X-Ray) use distributed traces. Every circuit breaker state transition, every timeout, every idempotency hit, every contract violation should emit a span attribute or trace event. Without structured traces, debugging production incidents requires log grep across instances with no causal linkage. | **Implementation**: Add optional `opentelemetry-api` integration. In each module, check `_otel_available()` at import time. If available, emit spans: `reliability.circuit_breaker.call` (span), `reliability.idempotency.hit` (event), `reliability.timeout.exceeded` (event with `operation` and `duration_ms` attributes). All OTel calls wrapped in `try/except` so failures never affect business logic. | **Competitor**: Resilience4j Micrometer integration, AWS SDK X-Ray tracing, Envoy's built-in tracing

---

### I6. Contract-Level Fuzzing Harness
**Category**: Verification | **Justification**: `@requires` and `@ensures` encode business invariants that are currently tested with hand-written examples. Property-based testing with Hypothesis can systematically find inputs that violate contracts — including edge cases like `float("nan")`, empty strings, very large amounts, and unicode boundary conditions — at zero marginal cost per invariant added. The Google DeepMind AlphaCode paper shows property tests catch 2-5x more bugs than equivalent unit tests. | **Implementation**: Add `generate_contract_tests(fn) -> list[HypothesisTest]` that inspects `@requires` predicates and generates Hypothesis `@given` strategies matching the type annotations. Add `ContractFuzzer` class with `fuzz(n_examples=1000)` method. Output goes to `tests/property/` directory. | **Competitor**: Hypothesis library, QuickCheck (Haskell), Microsoft PICT, Google OSS-Fuzz

---

### I7. Rate Limiter with Token Bucket and NATS Coordination
**Category**: Protection | **Justification**: The framework protects against downstream failures (circuit breaker) and upstream errors (guards) but has no protection against request volume. A misconfigured upstream sending 10,000 payment requests per second will overwhelm any downstream system regardless of circuit breaker state. Rate limiting is a first-class reliability primitive alongside circuit breakers and timeouts. | **Implementation**: Add `RateLimiter` with `max_rate: float` (requests/second) and `burst: int` token bucket. `async def acquire(n=1) -> None` — blocks if tokens exhausted, raises `RateLimitExceeded` after `max_wait`. Add `@rate_limited(name, max_rate=100, burst=20)` decorator. For distributed rate limiting, add `NatsRateLimiter` using NATS JetStream KV for shared token state. | **Competitor**: Envoy's `token_bucket`, Redis `INCR` + TTL pattern, AWS API Gateway throttling, Kong rate limiting plugin

---

### I8. Chaos Engineering Hooks (Fault Injection)
**Category**: Resilience Testing | **Justification**: Netflix Chaos Monkey, AWS Fault Injection Simulator, and Gremlin demonstrate that the only way to verify reliability properties hold under failure is to inject failures deliberately in controlled environments. Without fault injection, teams discover failure modes in production. Each reliability primitive should have a corresponding fault injection point. | **Implementation**: Add `FaultInjector` class with `inject_latency(service, delay_ms, probability=1.0)`, `inject_error(service, exc_class, probability=1.0)`, `inject_timeout(service)`. Integrate with `CircuitBreaker.call()` — check injector registry before executing. Controlled by `APG_FAULT_INJECTION_ENABLED=1` env var (default off). | **Competitor**: Netflix Chaos Monkey, AWS FIS, Gremlin, `chaos-mesh` (Kubernetes)

---

### I9. Adaptive Timeout Calibration from P99 Latency History
**Category**: Timeout Management | **Justification**: Static timeouts in `TIMEOUTS` dict are best-guess values that become stale. A service that normally responds in 50ms but is configured with a 30s timeout wastes 29.95s per failure. Netflix and Google dynamically calibrate timeouts to P99 latency + headroom. Tight timeouts mean faster failure detection, faster circuit opening, and faster client recovery. | **Implementation**: Add `LatencyHistogram` class using a lock-protected `deque` of recent latencies with configurable window. Add `AdaptiveTimeout` wrapping `timeout_async` — timeout = `max(min_timeout, p99_latency * multiplier)`. Add `auto_calibrate=True` flag to `CircuitBreaker`. Export P99 histograms via `/metrics` endpoint. | **Competitor**: Netflix Ribbon adaptive timeouts, gRPC deadline propagation, Envoy adaptive timeout filter

---

### I10. Health Check Caching and Stampede Prevention
**Category**: Health Probing | **Justification**: The current `DeepHealthCheck.run()` fires all checks every time it's called with no caching. Under Kubernetes load, `/health/ready` can be called every second across dozens of pods — each probe spawning N concurrent DB connections. Health stampedes have caused production PostgreSQL connection pool exhaustion at Shopify and Cloudflare. Adding TTL-based result caching with single-flight deduplication (only one in-flight health check at a time per dependency) eliminates stampedes. | **Implementation**: Add `cache_ttl: float = 5.0` to `DeepHealthCheck.__init__`. Cache last `HealthStatus` per component. Add `_inflight: asyncio.Event | None` per dependency — second caller awaits the event rather than launching a parallel check. | **Competitor**: Kubernetes kubelet's `successThreshold` / `failureThreshold`, Envoy health check caching, Spring Boot Actuator cache

---

### I11. Dependency Graph Visualization and Critical Path Analysis
**Category**: Observability | **Justification**: As capabilities compose (fintech → reliability → postgresql + nats + vault + mpesa), the dependency graph becomes opaque. Azure Service Health and Datadog Monitors automatically derive the dependency graph and identify critical path components. The `DeepHealthCheck` already has the graph implicitly — exposing it as JSON enables automated runbook generation and root-cause analysis. | **Implementation**: Add `DeepHealthCheck.dependency_graph() -> dict` that emits `{"nodes": [...], "edges": [...], "critical_path": [...]}`. Add `DeepHealthCheck.render_mermaid() -> str` for Mermaid diagram generation. Required dependencies form the critical path — any UNHEALTHY node on the critical path immediately marks the system UNHEALTHY. | **Competitor**: Azure Service Health dependency maps, Datadog Service Catalog, PagerDuty Service Graph

---

### I12. Lease-Based Distributed Locking for Critical Sections
**Category**: Concurrency Safety | **Justification**: The `IdempotencyRegistry` handles concurrent same-key calls within a single process via `asyncio.Lock`. Across processes (Kubernetes pods), no such protection exists. Critical sections like "check credit balance then debit" are inherently unsafe without distributed locking. NATS JetStream can provide lease-based distributed locks (compare-and-swap on KV) without Redis. | **Implementation**: Add `DistributedLock` class with `acquire(key, ttl=30.0, retry_interval=0.1)` and `release()`. Uses NATS JetStream KV with `update(seq)` for CAS. Implements `__aenter__`/`__aexit__`. Add `@distributed_lock(key_fn=..., ttl=30.0)` decorator. Lease TTL prevents deadlocks when the holder dies. | **Competitor**: Redis Redlock algorithm, ZooKeeper ZNode ephemeral nodes, etcd `lease` primitive, Consul sessions

---

### I13. Circuit Breaker Dashboard Blueprint (Flask-AppBuilder)
**Category**: Operations | **Justification**: Circuit breaker state is currently observable only via `all_circuit_status()` — a programmatic API. Operations teams need a real-time dashboard showing state transitions, failure rates, and manual override capabilities (force-open, force-close). PagerDuty and Datadog both provide UI-driven circuit breaker management. | **Implementation**: Add `reliability_blueprint` Flask-AppBuilder `Blueprint` at `/reliability/`. Views: `CircuitStatusView` (table of all circuits with state, failure count, time-in-state), `IdempotencyStatsView`, `HealthDashboardView`. Add REST API: `POST /reliability/circuit/{name}/reset` (force close), `POST /reliability/circuit/{name}/open` (force open for maintenance). | **Competitor**: Netflix Hystrix Dashboard, Resilience4j Actuator endpoints, Spring Boot Admin

---

### I14. Contract Violation Aggregation and Policy Engine
**Category**: Governance | **Justification**: `ContractViolation` is currently raised and logged per-occurrence. In a distributed system running 1M operations/day, contract violations need aggregation (are violations increasing?), deduplication (same bug in a loop), and policy-based escalation (critical violations page on-call; informational violations create tickets). AWS Config Rules and Open Policy Agent provide this pattern for infrastructure. | **Implementation**: Add `ContractViolationSink` abstract protocol with `record(violation: ContractViolation, context: dict) -> None`. Add `LoggingSink` (default), `NatsSink` (publish to `reliability.violations` NATS subject), `AggregationSink` (count + dedupe by `(kind, predicate_desc, qualified_name)`, emit summary every N minutes). Wire into `_check_requires` and `_check_ensures`. | **Competitor**: AWS Config Rules, Open Policy Agent, Datadog Error Tracking, Sentry issue grouping

---

### I15. Graceful Degradation Mode with Feature Flags
**Category**: Resilience | **Justification**: When critical dependencies fail, the current framework provides "UNHEALTHY" status and fails fast. Production systems (Netflix, Facebook) implement graceful degradation: non-critical features are disabled via feature flags while core functionality continues. The `required=False` flag in health checks already encodes this intent — it needs to connect to actual feature disablement. | **Implementation**: Add `DegradationManager` with `register_feature(name, dependency: str, fallback: Callable)`. When `DeepHealthCheck` marks a dependency UNHEALTHY, `DegradationManager` automatically disables registered features and routes calls to fallbacks. Add `@degrade_gracefully(dependency, fallback=...)` decorator. Publish degradation events to NATS `reliability.degradation` subject for downstream awareness. | **Competitor**: Netflix Hystrix fallback, LaunchDarkly feature flags, Facebook's feature gating, AWS Application Auto Scaling policies
