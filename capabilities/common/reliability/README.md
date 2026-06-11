# APG Reliability Framework

**Capability ID**: `reliability` | **Domain**: `common` | **Version**: `2.0.0`

Aerospace-grade runtime reliability infrastructure for the APG platform.
Designed for 10-year continuous operation with zero silent failures.

© 2025 Datacraft | Author: Nyimbi Odero

## Components

| Module | Component(s) | Purpose |
|--------|-------------|---------|
| `contracts.py` | `@requires` / `@ensures` / `@invariant` | Design-by-Contract with Hoare-triple semantics |
| `circuit_breaker.py` | `CircuitBreaker`, `@circuit_breaker` | Three-state circuit breaker for external service calls |
| `timeout.py` | `timeout_async`, `@with_timeout`, `timed` | Mandatory timeout on all async operations |
| `idempotency.py` | `IdempotencyRegistry`, `@idempotent` | Exactly-once semantics for critical operations |
| `health.py` | `DeepHealthCheck`, `HealthStatus` | Dependency-graph health probing, concurrent |
| `guards.py` | `guard_*`, `safe_gather`, `BoundedCache` | Input validation, safe async primitives, LRU cache |
| `service.py` | `ReliabilityService` + 8 new classes | Unified facade + advanced reliability primitives |

## New in v2.0

| Feature | Class / Function | Description |
|---------|-----------------|-------------|
| Adaptive circuit breaker | `AdaptiveCircuitBreaker` | Sliding-window failure-rate (vs. consecutive count) |
| Bulkhead isolation | `Bulkhead`, `@bulkhead_protected` | Per-dependency concurrency cap |
| Retry orchestrator | `retry_async`, `@with_retry`, `RetryBudget` | Exponential backoff + jitter + budget cap |
| Token-bucket rate limiter | `RateLimiter`, `@rate_limited` | Per-service request rate limiting |
| Latency histogram | `LatencyHistogram`, `AdaptiveTimeout` | P99-calibrated adaptive timeouts |
| Fault injector | `FaultInjector` | Chaos engineering hooks (latency / error / timeout) |
| Violation sink | `ContractViolationSink` | Aggregated governance for `ContractViolation` events |
| Graceful degradation | `DegradationManager`, `@degrade_gracefully` | Feature fallback when dependencies are UNHEALTHY |
| Unified facade | `ReliabilityService` | Wire all primitives for a service with one call |

## Quick Start

```python
from capabilities.common.reliability import (
    requires, ensures, circuit_breaker,
    timeout_async, guard_tenant_id, guard_positive_amount,
    idempotent, DeepHealthCheck, ReliabilityService,
)

# ── Unified facade: register once, protect everywhere ──────────────
rs = ReliabilityService("fintech_gwy")
rs.register_service(
    "mpesa",
    failure_rate_threshold=0.3,  # open if 30% of last 100 calls fail
    max_concurrent=20,           # bulkhead: max 20 in-flight
    max_rate=50,                 # 50 req/s token bucket
    timeout=30.0,
)

@rs.protect("mpesa", retry_attempts=3)
async def call_mpesa_api(payload: dict) -> dict:
    ...  # circuit breaker + bulkhead + rate limit + retry applied

# ── Check health and sync degradation state ─────────────────────────
status = await rs.run_health_check()
print(status.to_dict())

# ── Full status snapshot ─────────────────────────────────────────────
snapshot = await rs.full_status()
# {"circuit_breakers": {...}, "bulkheads": {...}, "latency_histograms": {...}, ...}
```

## Adaptive Circuit Breaker

```python
from capabilities.common.reliability import AdaptiveCircuitBreaker

cb = AdaptiveCircuitBreaker(
    "payments_gateway",
    window_size=100,              # track last 100 calls
    window_duration=60.0,         # and/or calls from last 60s
    failure_rate_threshold=0.5,   # open if 50% failing
    min_calls=10,                 # need at least 10 samples
    reset_timeout=30.0,
)

async with cb:
    result = await gateway.charge(amount)
```

## Bulkhead Isolation

```python
from capabilities.common.reliability import Bulkhead, bulkhead_protected

@bulkhead_protected("ollama", max_concurrent=5, max_wait=1.0)
async def run_ml_inference(text: str) -> dict:
    ...  # at most 5 concurrent; excess raises BulkheadFullError after 1s
```

## Retry with Budget

```python
from capabilities.common.reliability import retry_async, with_retry, RetryBudget

budget = RetryBudget(budget_fraction=0.1, window=100)  # max 10% retries

@with_retry(max_attempts=3, base_delay=0.1, max_delay=5.0, budget=budget)
async def call_external_api() -> dict:
    ...
```

## Adaptive Timeout

```python
from capabilities.common.reliability import LatencyHistogram, AdaptiveTimeout

hist = LatencyHistogram(window=500, window_duration=120.0)
at = AdaptiveTimeout(hist, min_timeout=1.0, multiplier=2.0)

# Timeout auto-calibrates to 2× P99 latency from recent history
result = await at.call(my_api_fn, arg1, label="my_api")
stats = await hist.stats()
# {"count": 42, "p50_ms": 120.0, "p95_ms": 340.0, "p99_ms": 500.0, ...}
```

## Rate Limiting

```python
from capabilities.common.reliability import RateLimiter, rate_limited

@rate_limited("mpesa_stk_push", max_rate=10, burst=20, max_wait=0.5)
async def stk_push(phone: str, amount: float) -> dict:
    ...  # max 10 req/s; waits up to 0.5s; raises RateLimitExceeded if exhausted
```

## Graceful Degradation

```python
from capabilities.common.reliability import DegradationManager

dm = DegradationManager()
dm.register_feature("ml_risk_score", dependency="ollama", fallback=rule_based_score)
dm.attach_health_check(health_checker)

@dm.degrade_gracefully("ml_risk_score")
async def get_risk_score(transaction: dict) -> float:
    return await ollama_score(transaction)  # falls back to rule_based_score if ollama is UNHEALTHY
```

## Chaos Engineering

```python
from capabilities.common.reliability import FaultInjector
import os

os.environ["APG_FAULT_INJECTION_ENABLED"] = "1"

injector = FaultInjector()
injector.inject_latency("mpesa", delay_ms=500, probability=0.1)   # 10% calls get +500ms
injector.inject_error("vault", RuntimeError, probability=0.05)     # 5% calls fail
injector.inject_timeout("database", probability=0.02)              # 2% calls hang

@injector.wrap("mpesa")
async def call_mpesa(payload: dict) -> dict: ...
```

## Design-by-Contract

```python
from capabilities.common.reliability import requires, ensures, invariant, ContractViolation

@requires(
    lambda self, amount, **_: amount > 0,   "amount must be positive",
    lambda self, amount, tenant_id, **_: bool(tenant_id), "tenant_id required",
)
@ensures(
    lambda r: r is not None, "must return result",
    lambda r: "id" in r,     "result must contain id",
)
async def process_payment(self, amount: float, tenant_id: str) -> dict:
    ...
```

## Reliability Properties

- **No silent failures**: All exceptions logged with context before re-raise or structured return
- **Fail-fast**: Adaptive circuit breaker opens when failure rate exceeds threshold
- **Bounded concurrency**: Bulkheads cap per-dependency concurrency; excess rejected cleanly
- **Bounded rate**: Token-bucket rate limiters protect upstream from overload
- **Bounded time**: Every external call has a mandatory (or auto-calibrated) timeout
- **Idempotent**: Critical ops (payments, signatures) safe to retry with same key
- **Bounded resources**: `BoundedCache` evicts LRU items; connection pools are sized
- **Provable pre/postconditions**: `@requires`/`@ensures` raise `ContractViolation` on logic errors
- **Parallel health**: All dependency checks run concurrently; slow deps never block reporting
- **Graceful degradation**: Non-critical features fall back when dependencies fail
- **Chaos-ready**: Fault injection built in for pre-production resilience verification

## Streaming Integration (NATS / bytewax)

For distributed idempotency and rate coordination across multiple APG instances, use NATS JetStream KV as the shared backing store. See `WORLD_CLASS_IMPROVEMENTS.md` (I4, I7, I12) for implementation details.

```python
# Publish reliability events to NATS for cross-service observability
import nats

async def publish_violation(subject: str, payload: dict) -> None:
    nc = await nats.connect("nats://localhost:4222")
    await nc.publish(subject, json.dumps(payload).encode())
    # subjects: reliability.violations, reliability.degradation, reliability.circuit_open
```

## Tests

```bash
uv run pytest tests/test_reliability_framework.py tests/test_reliability_service.py -v
# 80+ tests, all pass
```

## File Map

```
capabilities/common/reliability/
├── __init__.py            # Exports all public API
├── service.py             # Unified facade + advanced primitives (v2.0)
├── circuit_breaker.py     # Three-state circuit breaker + global registry
├── contracts.py           # @requires / @ensures / @invariant
├── guards.py              # Input guards, safe_gather, BoundedCache
├── health.py              # DeepHealthCheck + common check functions
├── idempotency.py         # IdempotencyRegistry + @idempotent
├── timeout.py             # timeout_async + @with_timeout + TIMEOUTS
├── README.md              # This file
├── WORLD_CLASS_IMPROVEMENTS.md  # 15 improvement proposals
└── docs/
    └── user_guide.md      # Comprehensive user guide
```
