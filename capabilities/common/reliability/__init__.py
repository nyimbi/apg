"""APG Reliability Framework — aerospace/medical-grade runtime correctness.

Provides:
- @requires / @ensures / @invariant — Design-by-Contract decorators
- CircuitBreaker / AdaptiveCircuitBreaker — protect all external calls
- timeout_async / AdaptiveTimeout — mandatory + P99-calibrated timeouts
- IdempotencyRegistry — exactly-once semantics
- DeepHealthCheck — dependency-graph health probing
- safe_gather — asyncio.gather with mandatory error isolation
- BoundedCache — cache with LRU eviction and size limits
- Bulkhead — concurrency isolation per dependency
- RateLimiter — token-bucket rate limiting
- retry_async / with_retry — exponential backoff with jitter and budget
- FaultInjector — chaos engineering hooks
- DegradationManager — graceful feature fallback routing
- ContractViolationSink — aggregating governance for contract violations
- ReliabilityService — unified facade over all primitives
"""
from .contracts import requires, ensures, invariant, ContractViolation
from .circuit_breaker import CircuitBreaker, CircuitOpenError, circuit_breaker
from .timeout import timeout_async, OperationTimeout, OperationTimeout as AsyncTimeoutError
from .idempotency import IdempotencyRegistry, idempotent
from .health import DeepHealthCheck, HealthStatus, ComponentHealth
from .guards import (
	guard_tenant_id,
	guard_positive_amount,
	guard_non_empty_string,
	guard_bounded_list,
	safe_gather,
	BoundedCache,
	create_tracked_task,
)
from .service import (
	AdaptiveCircuitBreaker,
	Bulkhead,
	BulkheadFullError,
	bulkhead_protected,
	get_bulkhead,
	RateLimiter,
	RateLimitExceeded,
	rate_limited,
	get_rate_limiter,
	RetryBudget,
	RetryBudgetExceeded,
	retry_async,
	with_retry,
	LatencyHistogram,
	AdaptiveTimeout,
	FaultInjector,
	FaultSpec,
	ContractViolationSink,
	ViolationRecord,
	DegradationManager,
	FeatureRegistration,
	ReliabilityService,
)

__all__ = [
	# Contracts
	"requires", "ensures", "invariant", "ContractViolation",
	# Circuit breakers
	"CircuitBreaker", "AdaptiveCircuitBreaker", "CircuitOpenError", "circuit_breaker",
	# Timeouts
	"timeout_async", "AsyncTimeoutError", "OperationTimeout",
	"LatencyHistogram", "AdaptiveTimeout",
	# Idempotency
	"IdempotencyRegistry", "idempotent",
	# Health
	"DeepHealthCheck", "HealthStatus", "ComponentHealth",
	# Guards
	"guard_tenant_id", "guard_positive_amount", "guard_non_empty_string",
	"guard_bounded_list", "safe_gather", "BoundedCache", "create_tracked_task",
	# Bulkhead
	"Bulkhead", "BulkheadFullError", "bulkhead_protected", "get_bulkhead",
	# Rate limiting
	"RateLimiter", "RateLimitExceeded", "rate_limited", "get_rate_limiter",
	# Retry
	"RetryBudget", "RetryBudgetExceeded", "retry_async", "with_retry",
	# Chaos
	"FaultInjector", "FaultSpec",
	# Governance
	"ContractViolationSink", "ViolationRecord",
	# Degradation
	"DegradationManager", "FeatureRegistration",
	# Unified facade
	"ReliabilityService",
]
