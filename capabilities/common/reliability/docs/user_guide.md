# APG Reliability Framework — User Guide

**Version**: 2.0.0 | © 2025 Datacraft | Author: Nyimbi Odero

## Philosophy

The APG reliability framework targets the software equivalent of DO-178C Level A (avionics) and IEC 62304 Class C (medical devices): every failure mode is handled, every external call is bounded, every critical operation is idempotent, and every state invariant is provably maintained.

The key principle: **no silent failures**. Every exception is either:
1. Logged with full context before being re-raised, OR
2. Converted to a structured error response that the caller can act on

Version 2.0 extends this with eight new production-hardening primitives: adaptive circuit breaking, bulkhead isolation, retry budgets, token-bucket rate limiting, P99-adaptive timeouts, chaos engineering hooks, contract violation governance, and graceful degradation.

---

## Component Reference

### Design by Contract (`contracts.py`)

Preconditions and postconditions are enforced at runtime. A `ContractViolation` is an `AssertionError` subclass — it means the calling code has a bug, not a runtime error.

```python
from capabilities.common.reliability import requires, ensures, ContractViolation

@requires(
    lambda self, amount, tenant_id, **_: amount > 0,    "amount must be positive",
    lambda self, amount, tenant_id, **_: bool(tenant_id), "tenant_id required",
)
@ensures(lambda r: r is not None,  "must return result")
@ensures(lambda r: "id" in r,      "result must contain id")
async def process_payment(self, amount: float, tenant_id: str) -> dict:
    ...
```

`@invariant` wraps all public methods on a class:

```python
@invariant(lambda self: self._balance >= 0, "balance must be non-negative")
class Account:
    def deposit(self, amount: float) -> None: ...   # invariant checked pre+post
```

### Circuit Breaker (`circuit_breaker.py` + `service.py`)

**Standard circuit breaker** — consecutive failure count:

```python
from capabilities.common.reliability import circuit_breaker, CircuitOpenError

@circuit_breaker("mpesa", failure_threshold=5, reset_timeout=60.0, timeout=30.0)
async def call_mpesa(self, payload: dict) -> dict:
    ...  # raises CircuitOpenError immediately when circuit is OPEN
```

**Adaptive circuit breaker** — sliding-window failure rate (preferred for production):

```python
from capabilities.common.reliability import AdaptiveCircuitBreaker

cb = AdaptiveCircuitBreaker(
    "payments_gateway",
    window_size=100,              # track last 100 calls
    window_duration=60.0,         # or calls from last 60s (whichever is smaller)
    failure_rate_threshold=0.5,   # open if >= 50% failing
    min_calls=10,                 # need at least 10 samples before tripping
    reset_timeout=30.0,
    timeout=30.0,
)

# Context manager
async with cb:
    result = await gateway.charge(amount)

# Or direct call
result = await cb.call(gateway.charge, amount)

# Status
print(cb.status())
# {"service": "payments_gateway", "state": "closed", "window_samples": 42,
#  "failure_rate_pct": 2.4, "failure_rate_threshold_pct": 50.0, ...}
```

The adaptive breaker **does not trip on isolated transient failures** — it requires a sustained failure rate over the configured window. This eliminates false trips during brief network hiccups.

### Timeout Protection (`timeout.py` + `service.py`)

Every external call MUST have a timeout.

```python
from capabilities.common.reliability import timeout_async, AsyncTimeoutError
from capabilities.common.reliability.timeout import with_timeout, timed, TIMEOUTS

# Context manager
async with timeout_async(30.0, "mpesa_stk_push"):
    result = await mpesa_client.post(...)

# Decorator
@with_timeout(30.0)
async def call_external(self) -> dict: ...

# One-shot
result = await timed(external_call(), timeout=TIMEOUTS["http_fast"], label="op_name")
```

**Adaptive timeout** — calibrated from P99 latency history:

```python
from capabilities.common.reliability import LatencyHistogram, AdaptiveTimeout

hist = LatencyHistogram(window=500, window_duration=120.0)
at = AdaptiveTimeout(hist, min_timeout=1.0, multiplier=2.0, fallback_timeout=30.0)

# After enough observations, timeout = max(1.0, p99_latency_s * 2.0)
result = await at.call(api_fn, arg1, arg2, label="api_call")

# Inspect calibration data
stats = await hist.stats()
# {"count": 250, "p50_ms": 85.0, "p95_ms": 210.0, "p99_ms": 380.0, "max_ms": 520.0}
```

### Idempotency Registry (`idempotency.py`)

Critical operations (payments, signatures, state mutations) must be safe to retry.

```python
from capabilities.common.reliability import idempotent, IdempotencyRegistry

@idempotent(key_fn=lambda self, payment_id, **_: f"payment:{payment_id}")
async def process_payment(self, payment_id: str, amount: float) -> dict:
    ...  # executes exactly once per payment_id within TTL
```

Concurrent calls with the same key are **serialized** — the second call waits for the first to complete, then returns the cached result without re-executing.

Manual flow:

```python
registry = IdempotencyRegistry(max_size=10000, ttl=3600)

async with registry.once(key="pay:TXN-001") as ctx:
    if ctx.already_done:
        return ctx.result
    result = await do_payment()
    ctx.set_result(result)
return ctx.result
```

### Input Guards (`guards.py`)

Validate all inputs at service boundaries before any processing:

```python
from capabilities.common.reliability import (
    guard_tenant_id, guard_positive_amount, guard_non_empty_string,
    guard_bounded_list,
)
from capabilities.common.reliability.guards import guard_uuid, guard_enum, guard_page

async def charge(self, amount: float, tenant_id: str, items: list) -> dict:
    guard_tenant_id(tenant_id)
    guard_positive_amount(amount, max_value=1_000_000)
    guard_bounded_list(items, max_length=1000)
    guard_uuid(payment_id)
    guard_enum(status, allowed={"pending", "completed", "failed"})
    guard_page(page, page_size, max_page_size=100)
    ...
```

### Deep Health Check (`health.py`)

Dependency-graph health probing with concurrent checks and per-component timeouts:

```python
from capabilities.common.reliability import DeepHealthCheck
from capabilities.common.reliability.health import (
    check_postgresql, check_nats, check_ollama, check_http_endpoint,
)

checker = DeepHealthCheck("fintech_gwy", check_timeout=5.0, version="2.0.0")
checker.add_dependency("postgresql", lambda: check_postgresql(DB_URL))
checker.add_dependency("nats",       lambda: check_nats(NATS_URL))
checker.add_dependency("ollama",     lambda: check_ollama(OLLAMA_URL), required=False)
checker.add_dependency("mpesa_api",  lambda: check_http_endpoint("https://sandbox.safaricom.co.ke"), required=False)

status = await checker.run()
print(status.to_dict())
# {"capability_id": "fintech_gwy", "status": "healthy", "ready": true,
#  "total_latency_ms": 42.1, "components": [...]}
```

Required vs. optional:
- `required=True` (default): UNHEALTHY → system `UNHEALTHY`
- `required=False`: UNHEALTHY → system `DEGRADED` (still ready)

### Bounded Cache and Safe Gather (`guards.py`)

```python
from capabilities.common.reliability import BoundedCache, safe_gather

# Thread-safe LRU cache with TTL
_token_cache = BoundedCache(max_size=1000)
_token_cache.set("oauth_token", token, ttl=3600)
token = _token_cache.get("oauth_token")   # None if missing/expired
print(_token_cache.stats())
# {"size": 1, "hits": 12, "misses": 2, "hit_rate_pct": 86, "evictions": 0}

# Safe gather: never silently swallows exceptions
results = await safe_gather(
    fetch_balance(),
    fetch_history(),
    fetch_pending(),
    label="dashboard_batch",
    suppress_exceptions=False,
)
```

---

## New Primitives (v2.0)

### Bulkhead Isolation

Prevent one slow dependency from consuming all asyncio concurrency:

```python
from capabilities.common.reliability import Bulkhead, BulkheadFullError, bulkhead_protected

# Decorator form
@bulkhead_protected("ollama", max_concurrent=5, max_wait=1.0)
async def run_inference(text: str) -> dict:
    ...  # at most 5 concurrent; BulkheadFullError after 1s wait

# Context manager form
bh = Bulkhead("database", max_concurrent=20, max_wait=0.5)
async with bh:
    result = await db.query(...)

print(bh.status())
# {"name": "database", "active": 3, "max_concurrent": 20, "rejected_total": 0}
```

### Retry Orchestrator

```python
from capabilities.common.reliability import retry_async, with_retry, RetryBudget, RetryBudgetExceeded

# Shared budget: caps retries at 10% of total traffic across all callers
budget = RetryBudget(budget_fraction=0.1, window=200)

# Decorator
@with_retry(
    max_attempts=3,
    base_delay=0.1,
    max_delay=30.0,
    jitter=True,                  # full jitter: prevents thundering herds
    retry_on=(IOError, TimeoutError),
    no_retry_on=(ValueError, PermissionError),  # never retry these
    budget=budget,
)
async def call_external() -> dict: ...

# Function form
result = await retry_async(
    my_async_fn, arg1, arg2,
    max_attempts=4,
    base_delay=0.5,
    label="external_db_write",
    budget=budget,
)
```

### Token-Bucket Rate Limiter

```python
from capabilities.common.reliability import RateLimiter, RateLimitExceeded, rate_limited

# Decorator
@rate_limited("mpesa_stk_push", max_rate=10, burst=20, max_wait=0.5)
async def stk_push(phone: str, amount: float) -> dict:
    ...  # 10 req/s sustained; burst of 20; waits up to 0.5s; then RateLimitExceeded

# Manual
rl = RateLimiter("reporting_api", max_rate=100, max_wait=1.0)
async with rl:
    await generate_report()

print(rl.status())
# {"name": "reporting_api", "max_rate": 100, "tokens": 87.4, "rejected_total": 0}
```

### Contract Violation Sink

Aggregate `ContractViolation` events for governance and monitoring:

```python
from capabilities.common.reliability import ContractViolationSink

sink = ContractViolationSink(flush_interval=300.0, max_distinct=1000)
await sink.start_flush_loop()  # logs summary every 5 minutes

# Record violations (normally called from within @requires / @ensures)
await sink.record(
    kind="requires",
    predicate_desc="amount must be positive",
    qualified_name="PaymentService.process_payment",
    context={"amount": -50.0, "tenant_id": "acme"},
)

# Query
top_violations = await sink.summary()
# [{"kind": "requires", "predicate": "amount must be positive",
#   "where": "PaymentService.process_payment", "count": 42, "last_seen": ...}]
```

### Graceful Degradation

```python
from capabilities.common.reliability import DegradationManager

async def rule_based_score(txn: dict) -> float:
    return 0.5  # static fallback

dm = DegradationManager()
dm.register_feature(
    "ml_risk_score",
    dependency="ollama",
    fallback=rule_based_score,
    description="Falls back to rule-based scoring when Ollama is down",
)
dm.attach_health_check(health_checker)

@dm.degrade_gracefully("ml_risk_score")
async def get_risk_score(txn: dict) -> float:
    return await ollama_score(txn)

# After health check runs and marks ollama UNHEALTHY:
score = await get_risk_score(transaction)
# → calls rule_based_score(transaction) transparently

print(dm.status())
# {"degraded_dependencies": ["ollama"],
#  "features": [{"name": "ml_risk_score", "dependency": "ollama", "degraded": true}]}
```

### Chaos Engineering (Fault Injection)

Only active when `APG_FAULT_INJECTION_ENABLED=1`. Safe to leave in production code.

```python
import os
os.environ["APG_FAULT_INJECTION_ENABLED"] = "1"

from capabilities.common.reliability import FaultInjector

injector = FaultInjector()
injector.inject_latency("mpesa", delay_ms=500, probability=0.1)    # 10% of calls: +500ms
injector.inject_error("vault", RuntimeError, probability=0.05)      # 5%: RuntimeError
injector.inject_timeout("database", probability=0.02)               # 2%: hang forever

@injector.wrap("mpesa")
async def call_mpesa(payload: dict) -> dict:
    return await real_mpesa_client.post(payload)

# In tests: use directly
await injector.apply("mpesa")  # applies registered faults for "mpesa"
injector.clear("mpesa")        # remove faults for mpesa
injector.clear()               # remove all faults
```

### Unified ReliabilityService

Wire all primitives for a capability in one place:

```python
from capabilities.common.reliability import ReliabilityService
from capabilities.common.reliability.health import check_postgresql, check_nats

rs = ReliabilityService("fintech_gwy")

# Register services with all primitives auto-configured
rs.register_service(
    "mpesa",
    failure_rate_threshold=0.3,
    min_calls=10,
    reset_timeout=60.0,
    timeout=30.0,
    max_concurrent=20,
    max_rate=50,
    window_size=100,
)
rs.register_service("ollama", failure_rate_threshold=0.5, max_concurrent=3, timeout=120.0)

# Register health checks
rs._health.add_dependency("postgresql", lambda: check_postgresql(DB_URL))
rs._health.add_dependency("nats", lambda: check_nats(NATS_URL))
rs._health.add_dependency("ollama", lambda: check_http_endpoint(OLLAMA_URL + "/api/tags"), required=False)

# Register degradation fallbacks
rs._degradation.register_feature("ml_scoring", dependency="ollama", fallback=rule_based_score)

# Protect service methods
@rs.protect("mpesa", retry_attempts=3, retry_base_delay=0.2)
async def call_mpesa_api(payload: dict) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post("https://sandbox.safaricom.co.ke/...", json=payload)
        r.raise_for_status()
        return r.json()

# Health check + degradation sync (call this from /health/ready endpoint)
status = await rs.run_health_check()

# Full observability snapshot
snapshot = await rs.full_status()
```

---

## NATS Integration

For cross-instance coordination (distributed idempotency, rate limiting), use NATS JetStream:

```python
import nats
import json

# Publish reliability events for cross-service visibility
async def _publish_reliability_event(subject: str, payload: dict) -> None:
    nc = await nats.connect("nats://localhost:4222")
    await nc.publish(f"reliability.{subject}", json.dumps(payload).encode())
    await nc.close()

# Subjects used by the APG reliability framework:
# reliability.violations   — ContractViolation events
# reliability.degradation  — feature degradation state changes
# reliability.circuit_open — circuit breaker OPEN transitions
```

For distributed idempotency (NATS JetStream KV) and distributed rate limiting, see `WORLD_CLASS_IMPROVEMENTS.md` improvements I4 and I7.

---

## Integration Checklist

For each new service method, verify:

- [ ] `guard_tenant_id(tenant_id)` called at entry
- [ ] `guard_positive_amount()` on all monetary values
- [ ] `@requires` on key preconditions; `@ensures` on return value invariants
- [ ] All external calls wrapped with `@circuit_breaker` / `AdaptiveCircuitBreaker`
- [ ] Mandatory timeout on every external call (`timeout_async`, `@with_timeout`, or `AdaptiveTimeout`)
- [ ] Bulkhead applied to high-latency dependencies (ML inference, file I/O, batch jobs)
- [ ] Rate limiter applied to external APIs with known quotas
- [ ] Critical mutations use `@idempotent` with a stable key
- [ ] No raw `asyncio.gather()` — use `safe_gather` instead
- [ ] Deep health check registers all required and optional dependencies
- [ ] Degradation fallbacks registered for non-critical features

---

## Performance Notes

- All locks are `asyncio.Lock` (not `threading.Lock`) — safe for coroutine-dense workloads
- `LatencyHistogram.percentile()` and `stats()` hold the lock briefly; O(n log n) sort over window
- `AdaptiveCircuitBreaker._trim_window()` is O(k) where k = expired entries — typically near 0
- `BoundedCache.get/set` use `threading.Lock` for cross-thread safety with a single global lock
- `ContractViolationSink` is append-only under a single `asyncio.Lock` — negligible overhead

---

## Test Suite

```bash
# Core framework tests
uv run pytest tests/test_reliability_framework.py -v

# Service layer tests (v2.0 primitives)
uv run pytest tests/test_reliability_service.py -v

# All reliability tests
uv run pytest tests/ -k reliability -v
```
