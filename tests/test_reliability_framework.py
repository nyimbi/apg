"""Tests for the APG Reliability Framework.

Verifies contracts, circuit breakers, timeouts, idempotency,
health checks, guards, and bounded cache — the components that
underpin 10-year operational reliability.
"""
import asyncio
import time

import pytest

from capabilities.common.reliability import (
    BoundedCache,
    CircuitBreaker,
    CircuitOpenError,
    ContractViolation,
    DeepHealthCheck,
    HealthStatus,
    IdempotencyRegistry,
    AsyncTimeoutError,
    ensures,
    guard_bounded_list,
    guard_non_empty_string,
    guard_positive_amount,
    guard_tenant_id,
    idempotent,
    requires,
    safe_gather,
    timeout_async,
)
from capabilities.common.reliability.circuit_breaker import (
    CircuitState,
    all_circuit_status,
    get_circuit_breaker,
)
from capabilities.common.reliability.health import HealthLevel, check_http_endpoint
from capabilities.common.reliability.timeout import get_timeout, with_timeout


# ── Contracts ─────────────────────────────────────────────────────

class TestContracts:
    async def test_requires_passes_when_condition_true(self):
        @requires(lambda self, x: x > 0, "x must be positive")
        async def fn(self, x):
            return x * 2

        assert await fn(None, 5) == 10

    async def test_requires_raises_on_violation(self):
        @requires(lambda self, x: x > 0, "x must be positive")
        async def fn(self, x):
            return x

        with pytest.raises(ContractViolation) as exc_info:
            await fn(None, -1)
        assert "requires" in str(exc_info.value)
        assert "positive" in str(exc_info.value)

    async def test_ensures_passes_when_condition_true(self):
        @ensures(lambda r: r is not None, "must return value")
        async def fn(self):
            return {"ok": True}

        result = await fn(None)
        assert result == {"ok": True}

    async def test_ensures_raises_on_violation(self):
        @ensures(lambda r: r is not None, "must not return None")
        async def fn(self):
            return None

        with pytest.raises(ContractViolation) as exc_info:
            await fn(None)
        assert "ensures" in str(exc_info.value)

    async def test_multiple_requires_all_checked(self):
        @requires(
            lambda self, a, b: a > 0, "a must be positive",
            lambda self, a, b: b > 0, "b must be positive",
        )
        async def fn(self, a, b):
            return a + b

        assert await fn(None, 1, 2) == 3
        with pytest.raises(ContractViolation) as exc_info:
            await fn(None, -1, 2)
        assert "a must be positive" in str(exc_info.value)

    async def test_contract_violation_is_assertion_error_subclass(self):
        """ContractViolation must be catchable as AssertionError."""
        @requires(lambda self: False, "always fails")
        async def fn(self):
            pass

        with pytest.raises(AssertionError):
            await fn(None)

    def test_sync_requires_works(self):
        @requires(lambda x: x >= 0, "non-negative")
        def fn(x):
            return x

        assert fn(0) == 0
        with pytest.raises(ContractViolation):
            fn(-1)


# ── Circuit Breaker ────────────────────────────────────────────────

class TestCircuitBreaker:
    async def test_passes_through_when_closed(self):
        cb = CircuitBreaker("test_pass", failure_threshold=3)
        async def ok():
            return "result"
        assert await cb.call(ok) == "result"
        assert cb.state == CircuitState.CLOSED

    async def test_opens_after_threshold(self):
        cb = CircuitBreaker("test_open", failure_threshold=3, reset_timeout=60.0)
        async def fail():
            raise ConnectionError("down")
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(fail)
        assert cb.state == CircuitState.OPEN

    async def test_open_raises_circuit_open_error(self):
        cb = CircuitBreaker("test_reject", failure_threshold=1, reset_timeout=60.0)
        async def fail():
            raise RuntimeError("down")
        with pytest.raises(RuntimeError):
            await cb.call(fail)
        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(fail)
        assert "test_reject" in str(exc_info.value)

    async def test_half_open_after_reset_timeout(self):
        cb = CircuitBreaker("test_halfopen", failure_threshold=1, reset_timeout=0.05)
        async def fail():
            raise RuntimeError("down")
        async def succeed():
            return "ok"
        with pytest.raises(RuntimeError):
            await cb.call(fail)
        assert cb.state == CircuitState.OPEN
        await asyncio.sleep(0.06)
        # Next call should go through (HALF_OPEN probe)
        result = await cb.call(succeed)
        assert result == "ok"

    async def test_closes_after_success_threshold(self):
        cb = CircuitBreaker(
            "test_close", failure_threshold=1, reset_timeout=0.05, success_threshold=2
        )
        async def fail():
            raise RuntimeError("down")
        async def succeed():
            return "ok"
        with pytest.raises(RuntimeError):
            await cb.call(fail)
        await asyncio.sleep(0.06)
        await cb.call(succeed)  # first success (half-open)
        await cb.call(succeed)  # second success
        assert cb.state == CircuitState.CLOSED

    async def test_context_manager(self):
        cb = CircuitBreaker("test_ctx", failure_threshold=2)
        async with cb:
            pass  # no exception → success
        assert cb._failure_count == 0

    def test_status_dict(self):
        cb = CircuitBreaker("test_status", failure_threshold=5)
        status = cb.status()
        assert status["service"] == "test_status"
        assert status["state"] == "closed"
        assert status["failure_threshold"] == 5


# ── Timeout ───────────────────────────────────────────────────────

class TestTimeout:
    async def test_completes_within_timeout(self):
        async with timeout_async(1.0, "fast_op"):
            await asyncio.sleep(0.01)

    async def test_raises_on_timeout(self):
        with pytest.raises(AsyncTimeoutError) as exc_info:
            async with timeout_async(0.05, "slow_op"):
                await asyncio.sleep(1.0)
        assert "slow_op" in str(exc_info.value)
        assert "slow_op" in str(exc_info.value)
        assert "timed out" in str(exc_info.value).lower()

    async def test_with_timeout_decorator(self):
        @with_timeout(0.05)
        async def slow():
            await asyncio.sleep(1.0)

        with pytest.raises(AsyncTimeoutError):
            await slow()

    async def test_with_timeout_decorator_passes_on_fast(self):
        @with_timeout(1.0)
        async def fast():
            return 42

        assert await fast() == 42

    def test_get_timeout_known_category(self):
        assert get_timeout("db_read") == 5.0
        assert get_timeout("ml_inference") == 120.0

    def test_get_timeout_unknown_returns_default(self):
        assert get_timeout("unknown_category") == 30.0


# ── Guards ────────────────────────────────────────────────────────

class TestGuards:
    def test_guard_tenant_id_valid(self):
        guard_tenant_id("tenant_abc")  # no exception

    def test_guard_tenant_id_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            guard_tenant_id("")

    def test_guard_tenant_id_none(self):
        with pytest.raises(ValueError):
            guard_tenant_id(None)

    def test_guard_tenant_id_whitespace(self):
        with pytest.raises(ValueError):
            guard_tenant_id("   ")

    def test_guard_tenant_id_too_long(self):
        with pytest.raises(ValueError, match="too long"):
            guard_tenant_id("x" * 129)

    def test_guard_positive_amount_valid(self):
        guard_positive_amount(100.0)
        guard_positive_amount(0.01)

    def test_guard_positive_amount_zero(self):
        with pytest.raises(ValueError, match="positive"):
            guard_positive_amount(0.0)

    def test_guard_positive_amount_negative(self):
        with pytest.raises(ValueError, match="positive"):
            guard_positive_amount(-1.0)

    def test_guard_positive_amount_none(self):
        with pytest.raises(ValueError, match="must be provided"):
            guard_positive_amount(None)

    def test_guard_positive_amount_nan(self):
        import math
        with pytest.raises(ValueError, match="finite"):
            guard_positive_amount(math.nan)

    def test_guard_positive_amount_inf(self):
        import math
        with pytest.raises(ValueError, match="finite"):
            guard_positive_amount(math.inf)

    def test_guard_positive_amount_exceeds_max(self):
        with pytest.raises(ValueError, match="maximum"):
            guard_positive_amount(1e13, max_value=1e12)

    def test_guard_non_empty_string_valid(self):
        guard_non_empty_string("hello")

    def test_guard_non_empty_string_empty(self):
        with pytest.raises(ValueError):
            guard_non_empty_string("")

    def test_guard_non_empty_string_too_long(self):
        with pytest.raises(ValueError, match="too long"):
            guard_non_empty_string("x" * 70000)

    def test_guard_bounded_list_valid(self):
        guard_bounded_list([1, 2, 3])

    def test_guard_bounded_list_too_long(self):
        with pytest.raises(ValueError, match="too long"):
            guard_bounded_list(list(range(10001)))

    def test_guard_bounded_list_none_not_allowed(self):
        with pytest.raises(ValueError, match="must be provided"):
            guard_bounded_list(None)

    def test_guard_bounded_list_none_allowed(self):
        guard_bounded_list(None, allow_none=True)  # no exception


# ── Safe gather ───────────────────────────────────────────────────

class TestSafeGather:
    async def test_all_succeed(self):
        async def ok(n):
            return n
        results = await safe_gather(ok(1), ok(2), ok(3), suppress_exceptions=True)
        assert results == [1, 2, 3]

    async def test_exception_reraises_by_default(self):
        async def fail():
            raise ValueError("bad")
        async def ok():
            return 1

        with pytest.raises(ValueError):
            await safe_gather(ok(), fail(), suppress_exceptions=False)

    async def test_exception_suppressed_when_requested(self):
        async def fail():
            raise ValueError("bad")
        async def ok():
            return 42

        results = await safe_gather(ok(), fail(), suppress_exceptions=True)
        assert results[0] == 42
        assert isinstance(results[1], ValueError)


# ── Idempotency ───────────────────────────────────────────────────

class TestIdempotency:
    async def test_executes_once_for_same_key(self):
        call_count = 0
        reg = IdempotencyRegistry(max_size=100, ttl=60.0)

        @idempotent(key_fn=lambda self, txn_id: f"pay:{txn_id}", registry=reg)
        async def charge(self, txn_id):
            nonlocal call_count
            call_count += 1
            return {"txn": txn_id}

        r1 = await charge(None, "TXN-100")
        r2 = await charge(None, "TXN-100")
        assert call_count == 1
        assert r1 == r2

    async def test_different_keys_execute_separately(self):
        call_count = 0
        reg = IdempotencyRegistry(max_size=100, ttl=60.0)

        @idempotent(key_fn=lambda self, txn_id: f"pay:{txn_id}", registry=reg)
        async def charge(self, txn_id):
            nonlocal call_count
            call_count += 1
            return {"txn": txn_id}

        await charge(None, "TXN-200")
        await charge(None, "TXN-201")
        assert call_count == 2

    async def test_expiry_allows_re_execution(self):
        call_count = 0
        reg = IdempotencyRegistry(max_size=100, ttl=0.05)  # 50ms TTL

        @idempotent(key_fn=lambda self, k: k, registry=reg)
        async def fn(self, k):
            nonlocal call_count
            call_count += 1
            return call_count

        await fn(None, "k1")
        await asyncio.sleep(0.1)
        await fn(None, "k1")
        assert call_count == 2, f"Expected 2 executions after TTL, got {call_count}"

    def test_registry_stats(self):
        reg = IdempotencyRegistry(max_size=100)
        stats = reg.stats()
        assert stats["tracked_keys"] == 0
        assert stats["max_size"] == 100


# ── Bounded Cache ─────────────────────────────────────────────────

class TestBoundedCache:
    def test_basic_set_get(self):
        c = BoundedCache(max_size=10)
        c.set("k", "v")
        assert c.get("k") == "v"

    def test_lru_eviction(self):
        c = BoundedCache(max_size=3)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        c.set("d", 4)  # evicts "a" (LRU)
        assert c.get("a") is None
        assert c.get("d") == 4

    def test_ttl_expiry(self):
        c = BoundedCache(max_size=10)
        c.set("k", "v", ttl=0.05)
        assert c.get("k") == "v"
        time.sleep(0.06)
        assert c.get("k") is None

    def test_no_ttl_never_expires(self):
        c = BoundedCache(max_size=10)
        c.set("k", "v", ttl=0)
        assert c.get("k") == "v"

    def test_stats_accuracy(self):
        c = BoundedCache(max_size=2)
        c.set("a", 1)
        c.get("a")   # hit
        c.get("b")   # miss
        c.set("b", 2)
        c.set("c", 3)  # evicts "a"
        stats = c.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["evictions"] == 1
        assert stats["size"] == 2


# ── Deep Health Check ─────────────────────────────────────────────

class TestDeepHealthCheck:
    async def test_healthy_when_all_pass(self):
        checker = DeepHealthCheck("test_cap")
        checker.add_dependency("db", lambda: asyncio.sleep(0).__anext__().__await__().__next__())

        async def ok():
            return True

        checker2 = DeepHealthCheck("cap2")
        checker2.add_dependency("svc", ok)
        status = await checker2.run()
        assert status.level == HealthLevel.HEALTHY
        assert status.ready is True

    async def test_degraded_when_optional_fails(self):
        async def fail():
            raise ConnectionError("down")

        checker = DeepHealthCheck("cap3")
        checker.add_dependency("optional_svc", fail, required=False)
        status = await checker.run()
        assert status.level == HealthLevel.DEGRADED
        assert status.ready is True  # optional failed, still ready

    async def test_unhealthy_when_required_fails(self):
        async def fail():
            raise ConnectionError("required down")

        checker = DeepHealthCheck("cap4")
        checker.add_dependency("required_db", fail, required=True)
        status = await checker.run()
        assert status.level == HealthLevel.UNHEALTHY
        assert status.ready is False

    async def test_timeout_handled_gracefully(self):
        async def hang():
            await asyncio.sleep(100)

        checker = DeepHealthCheck("cap5", check_timeout=0.05)
        checker.add_dependency("slow", hang, required=False)
        status = await checker.run()
        assert status.components[0].level == HealthLevel.UNHEALTHY
        assert "Timed out" in status.components[0].message

    async def test_to_dict_structure(self):
        async def ok():
            return {"status": "healthy", "connections": 5}

        checker = DeepHealthCheck("cap6", version="2.0.0")
        checker.add_dependency("db", ok)
        status = await checker.run()
        d = status.to_dict()
        assert d["capability_id"] == "cap6"
        assert d["status"] in ("healthy", "degraded", "unhealthy")
        assert d["version"] == "2.0.0"
        assert isinstance(d["components"], list)
        assert d["components"][0]["name"] == "db"

    async def test_concurrent_checks_run_in_parallel(self):
        """Verify all checks run concurrently, not sequentially."""
        async def slow(n):
            await asyncio.sleep(0.1)
            return True

        checker = DeepHealthCheck("cap7", check_timeout=1.0)
        for i in range(5):
            checker.add_dependency(f"svc_{i}", lambda: slow(i))

        t0 = time.monotonic()
        status = await checker.run()
        elapsed = time.monotonic() - t0
        # 5 × 0.1s checks in parallel should complete in ~0.1s, not 0.5s
        assert elapsed < 0.4, f"Health checks took {elapsed:.2f}s — should be parallel"
