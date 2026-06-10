# APG Reliability Framework

**Capability ID**: `reliability` | **Domain**: `common` | **Version**: `1.0.0`

Aerospace-grade runtime reliability infrastructure for the APG platform.
Designed for 10-year continuous operation with zero silent failures.

## Components

| Component | Purpose |
|-----------|---------|
| `contracts.py` | `@requires` / `@ensures` / `@invariant` — Design-by-Contract |
| `circuit_breaker.py` | `CircuitBreaker` — protect all external service calls |
| `timeout.py` | `timeout_async` / `@with_timeout` — mandatory timeout on all async ops |
| `idempotency.py` | `IdempotencyRegistry` — exactly-once semantics for critical ops |
| `health.py` | `DeepHealthCheck` — dependency-graph health probing |
| `guards.py` | `guard_*` functions — input validation at service boundaries |

## Quick Start

```python
from capabilities.common.reliability import (
    requires, ensures, circuit_breaker,
    timeout_async, guard_tenant_id, guard_positive_amount,
    idempotent, DeepHealthCheck,
)

# 1. Design-by-Contract on service methods
@requires(lambda self, amount, **_: amount > 0, "amount must be positive")
@ensures(lambda r: r.get("status") is not None, "status must be set")
async def process_payment(self, amount: float, tenant_id: str) -> dict:
    guard_tenant_id(tenant_id)
    guard_positive_amount(amount, max_value=1_000_000)
    ...

# 2. Circuit breaker on all external calls
@circuit_breaker("mpesa", failure_threshold=5, reset_timeout=60.0)
async def call_mpesa_api(self, payload: dict) -> dict:
    ...

# 3. Mandatory timeout on every async operation
async def fetch_data(self) -> dict:
    async with timeout_async(30.0, "fetch_external_data"):
        return await self._http_client.get(...)

# 4. Idempotency for critical operations
@idempotent(key_fn=lambda self, payment_id, **_: f"pay:{payment_id}")
async def process_payment_once(self, payment_id: str, amount: float) -> dict:
    ...  # Only executes once per payment_id within TTL

# 5. Deep health checks
checker = DeepHealthCheck("my_capability")
checker.add_dependency("postgresql", lambda: check_postgresql(db_url))
checker.add_dependency("nats", lambda: check_nats(nats_url))
checker.add_dependency("cache", lambda: check_http_endpoint(cache_url), required=False)

status = await checker.run()
# Returns HealthStatus with per-component results
```

## Reliability Properties

- **No silent failures**: All exceptions logged with context before re-raise or structured return
- **Fail-fast**: Circuit breaker rejects calls immediately when a service is degraded
- **Bounded time**: Every external call has a mandatory timeout
- **Idempotent**: Critical ops (payments, signatures) safe to retry with same key
- **Bounded resources**: `BoundedCache` evicts LRU items; connection pools are sized
- **Provable pre/postconditions**: `@requires`/`@ensures` raise `ContractViolation` on logic errors
- **Parallel health**: All dependency checks run concurrently; slow deps never block reporting

## Tests

```bash
uv run pytest tests/test_reliability_framework.py -v
# 57 tests, all pass
```
