"""Circuit breaker for external service calls.

Implements the standard three-state circuit breaker pattern:
  CLOSED → calls pass through
  OPEN   → calls fail immediately (fail-fast)
  HALF_OPEN → one probe call allowed to test recovery

Usage:
    cb = CircuitBreaker("mpesa", failure_threshold=5, reset_timeout=60.0)

    @circuit_breaker("mpesa", failure_threshold=3, reset_timeout=30.0)
    async def call_mpesa_api(self, ...):
        ...

    # Or context manager:
    async with cb:
        result = await external_call()
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

_log = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"       # Normal — calls pass through
    OPEN = "open"           # Fused — calls fail immediately
    HALF_OPEN = "half_open" # One probe allowed


class CircuitOpenError(RuntimeError):
    """Raised when the circuit is open and a call is rejected."""
    def __init__(self, service_name: str, until: float) -> None:
        self.service_name = service_name
        self.until = until
        remaining = max(0, until - time.monotonic())
        super().__init__(
            f"Circuit breaker OPEN for {service_name!r} — "
            f"will retry in {remaining:.1f}s"
        )


class CircuitBreaker:
    """Thread-safe (asyncio) circuit breaker.

    Args:
        service_name: Identifies this circuit in logs.
        failure_threshold: Consecutive failures before opening.
        reset_timeout: Seconds to wait before half-open probe.
        success_threshold: Consecutive successes in half-open before closing.
        timeout: Per-call timeout in seconds (0 = no timeout).
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        success_threshold: int = 2,
        timeout: float = 30.0,
    ) -> None:
        assert failure_threshold > 0, "failure_threshold must be positive"
        assert reset_timeout > 0, "reset_timeout must be positive"
        assert success_threshold > 0, "success_threshold must be positive"

        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        self.timeout = timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def __aenter__(self) -> "CircuitBreaker":
        await self._before_call()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is not None and not issubclass(exc_type, CircuitOpenError):
            await self._on_failure(exc_val)
        elif exc_type is None:
            await self._on_success()
        return False  # Don't suppress exceptions

    async def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute fn(*args, **kwargs) through the circuit breaker."""
        await self._before_call()
        try:
            if self.timeout > 0:
                result = await asyncio.wait_for(
                    fn(*args, **kwargs) if asyncio.iscoroutinefunction(fn)
                    else asyncio.get_event_loop().run_in_executor(None, fn, *args),
                    timeout=self.timeout,
                )
            else:
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)
            await self._on_success()
            return result
        except CircuitOpenError:
            raise
        except Exception as exc:
            await self._on_failure(exc)
            raise

    # ── Private state machine ────────────────────────────────────

    async def _before_call(self) -> None:
        async with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.reset_timeout:
                    _log.info("Circuit %r: OPEN → HALF_OPEN after %.1fs", self.service_name, elapsed)
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                else:
                    raise CircuitOpenError(
                        self.service_name,
                        self._last_failure_time + self.reset_timeout,
                    )

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    _log.info("Circuit %r: HALF_OPEN → CLOSED after %d successes",
                               self.service_name, self._success_count)
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    async def _on_failure(self, exc: BaseException) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            _log.warning(
                "Circuit %r: failure %d/%d — %s: %s",
                self.service_name, self._failure_count, self.failure_threshold,
                type(exc).__name__, exc,
            )
            if self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                if self._failure_count >= self.failure_threshold:
                    _log.error(
                        "Circuit %r: OPEN — %d consecutive failures (last: %s)",
                        self.service_name, self._failure_count, exc,
                    )
                    self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset to closed state (use in tests only)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0

    def status(self) -> dict[str, Any]:
        return {
            "service": self.service_name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "reset_timeout": self.reset_timeout,
            "seconds_until_retry": max(0, self._last_failure_time + self.reset_timeout - time.monotonic())
            if self._state == CircuitState.OPEN else 0,
        }


# ── Global registry ───────────────────────────────────────────────

_REGISTRY: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
    timeout: float = 30.0,
) -> CircuitBreaker:
    """Get or create a named CircuitBreaker (singleton per service_name)."""
    if service_name not in _REGISTRY:
        _REGISTRY[service_name] = CircuitBreaker(
            service_name,
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            timeout=timeout,
        )
    return _REGISTRY[service_name]


def circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
    timeout: float = 30.0,
) -> Callable:
    """Decorator: wrap an async function with a named circuit breaker.

    Example:
        @circuit_breaker("mpesa", failure_threshold=3, reset_timeout=30.0)
        async def call_mpesa(self, amount: float) -> dict: ...
    """
    cb = get_circuit_breaker(service_name, failure_threshold, reset_timeout, timeout)

    def decorator(fn: Callable) -> Callable:
        import functools

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await cb.call(fn, *args, **kwargs)

        return wrapper

    return decorator


def all_circuit_status() -> list[dict[str, Any]]:
    """Return status of all registered circuits."""
    return [cb.status() for cb in _REGISTRY.values()]
