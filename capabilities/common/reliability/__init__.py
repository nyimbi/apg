"""APG Reliability Framework — aerospace/medical-grade runtime correctness.

Provides:
- @requires / @ensures / @invariant — Design-by-Contract decorators
- CircuitBreaker — protect all external calls
- timeout_async — mandatory timeout on all async operations
- IdempotencyRegistry — exactly-once semantics
- DeepHealthCheck — dependency-graph health probing
- safe_gather — asyncio.gather with mandatory error isolation
- BoundedCache — cache with LRU eviction and size limits
"""
from .contracts import requires, ensures, invariant, ContractViolation
from .circuit_breaker import CircuitBreaker, CircuitOpenError, circuit_breaker
from .timeout import timeout_async, OperationTimeout as AsyncTimeoutError
from .idempotency import IdempotencyRegistry, idempotent
from .health import DeepHealthCheck, HealthStatus, ComponentHealth
from .guards import (
    guard_tenant_id,
    guard_positive_amount,
    guard_non_empty_string,
    guard_bounded_list,
    safe_gather,
    BoundedCache,
)

__all__ = [
    "requires", "ensures", "invariant", "ContractViolation",
    "CircuitBreaker", "CircuitOpenError", "circuit_breaker",
    "timeout_async", "AsyncTimeoutError", "OperationTimeout",
    "IdempotencyRegistry", "idempotent",
    "DeepHealthCheck", "HealthStatus", "ComponentHealth",
    "guard_tenant_id", "guard_positive_amount", "guard_non_empty_string",
    "guard_bounded_list", "safe_gather", "BoundedCache",
]
