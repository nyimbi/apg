# APG Reliability Framework — User Guide

## Philosophy

The APG reliability framework targets the software equivalent of DO-178C Level A (avionics) and IEC 62304 Class C (medical devices): every failure mode is handled, every external call is bounded, every critical operation is idempotent, and every state invariant is provably maintained.

The key principle: **no silent failures**. Every exception is either:
1. Logged with full context before being re-raised, OR
2. Converted to a structured error response that the caller can act on

## Component Reference

### Design by Contract (`contracts.py`)

Preconditions and postconditions are enforced at runtime. A `ContractViolation` is an `AssertionError` subclass — it means the calling code has a bug, not a runtime error.

```python
from capabilities.common.reliability import requires, ensures, ContractViolation

@requires(
    lambda self, amount, tenant_id, **_: amount > 0,    "amount must be positive",
    lambda self, amount, tenant_id, **_: bool(tenant_id), "tenant_id required",
)
@ensures(lambda r: r is not None,             "must return result")
@ensures(lambda r: "id" in r,                 "result must contain id")
async def process_payment(self, amount: float, tenant_id: str) -> dict:
    ...
```

### Circuit Breaker (`circuit_breaker.py`)

Prevents cascading failures when external services are degraded. After `failure_threshold` consecutive failures, the circuit opens for `reset_timeout` seconds. A probe call is made in HALF_OPEN state — if it succeeds, the circuit closes.

```python
from capabilities.common.reliability import circuit_breaker, CircuitOpenError

@circuit_breaker("mpesa", failure_threshold=5, reset_timeout=60.0, timeout=30.0)
async def call_mpesa(self, payload: dict) -> dict:
    ...  # raises CircuitOpenError immediately when circuit is OPEN

# Check all circuit statuses
from capabilities.common.reliability.circuit_breaker import all_circuit_status
print(all_circuit_status())
# [{"service": "mpesa", "state": "closed", "failure_count": 0, ...}]
```

### Timeout Protection (`timeout.py`)

Every external call MUST have a timeout. An operation that can hang indefinitely will hang eventually.

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

### Idempotency Registry (`idempotency.py`)

Critical operations (payments, signatures, state mutations) must be safe to retry. The registry ensures that submitting the same logical operation twice returns the same result without re-executing.

```python
from capabilities.common.reliability import idempotent, IdempotencyRegistry

@idempotent(key_fn=lambda self, payment_id, **_: f"payment:{payment_id}")
async def process_payment(self, payment_id: str, amount: float) -> dict:
    ...  # executes exactly once per payment_id within TTL
```

Concurrent calls with the same key are **serialized** — the second call waits for the first to complete, then returns the cached result.

### Input Guards (`guards.py`)

Validate all inputs at service boundaries before any processing:

```python
from capabilities.common.reliability import (
    guard_tenant_id, guard_positive_amount, guard_non_empty_string, guard_bounded_list
)

async def charge(self, amount: float, tenant_id: str, items: list) -> dict:
    guard_tenant_id(tenant_id)
    guard_positive_amount(amount, max_value=1_000_000)
    guard_bounded_list(items, max_length=1000)
    ...
```

### Deep Health Check (`health.py`)

Dependency-graph health probing with concurrent checks and per-component timeouts:

```python
from capabilities.common.reliability import DeepHealthCheck
from capabilities.common.reliability.health import check_postgresql, check_nats

checker = DeepHealthCheck("fintech_gwy", check_timeout=5.0)
checker.add_dependency("postgresql", lambda: check_postgresql(DB_URL))
checker.add_dependency("nats",       lambda: check_nats(NATS_URL))
checker.add_dependency("mpesa_api",  lambda: check_http_endpoint("https://sandbox.safaricom.co.ke"), required=False)

status = await checker.run()
print(status.to_dict())
# {"capability_id": "fintech_gwy", "status": "healthy", "ready": true, ...}
```

### Bounded Cache (`guards.py`)

Thread-safe LRU cache with hard size limits and TTL — prevents memory leaks:

```python
from capabilities.common.reliability import BoundedCache

_token_cache = BoundedCache(max_size=1000)
_token_cache.set("oauth_token", token, ttl=3600)
token = _token_cache.get("oauth_token")  # None if missing or expired
print(_token_cache.stats())  # {"size": 1, "hits": 3, "evictions": 0, ...}
```

### Safe Gather (`guards.py`)

`asyncio.gather` that never silently swallows exceptions:

```python
from capabilities.common.reliability import safe_gather

results = await safe_gather(
    fetch_account_balance(),
    fetch_transaction_history(),
    fetch_pending_transfers(),
    label="dashboard_batch",
    suppress_exceptions=False,  # re-raises first exception (default)
)
```

## Integration Checklist

For each new service method, verify:

- [ ] `guard_tenant_id(tenant_id)` called at entry
- [ ] `guard_positive_amount()` on all monetary values
- [ ] `@requires` on key preconditions
- [ ] All external calls wrapped with `@circuit_breaker` or `async with timeout_async(...)`
- [ ] Critical mutations use `@idempotent` with a stable key
- [ ] No `asyncio.gather()` without `return_exceptions=True` (or use `safe_gather`)
- [ ] Deep health check registers all dependencies

## Property Tests

Run the property-based test suite to verify all invariants:

```bash
uv run pytest tests/test_reliability_properties.py tests/test_reliability_framework.py -v
# 156 tests, all pass
```
