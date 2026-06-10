"""Timeout utilities for async operations.

All external calls MUST use timeout protection. An operation that can hang
indefinitely is incompatible with 10-year reliable operation.

Usage:
    # Context manager
    async with timeout_async(30.0, "MPESA STK Push"):
        result = await mpesa.stk_push(...)

    # Decorator
    @with_timeout(30.0)
    async def call_external(self) -> dict: ...

    # Convenience wrapper
    result = await timed(external_call(), timeout=30.0, label="mpesa")
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Coroutine
from contextlib import asynccontextmanager
from typing import Any, TypeVar

_log = logging.getLogger(__name__)

T = TypeVar("T")

# Default timeouts by category (seconds)
TIMEOUTS = {
    "db_read": 5.0,
    "db_write": 10.0,
    "http_fast": 10.0,        # API calls that should be quick
    "http_slow": 60.0,        # File uploads, batch operations
    "ml_inference": 120.0,    # Ollama inference
    "nats_publish": 5.0,
    "temporal_start": 10.0,
    "health_check": 5.0,
    "default": 30.0,
}


class OperationTimeout(TimeoutError):
    """Raised when an async operation exceeds its timeout budget."""

    def __init__(self, operation: str, timeout: float) -> None:
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"Operation '{operation}' timed out after {timeout}s")


@asynccontextmanager
async def timeout_async(seconds: float, operation: str = "operation"):
    """Async context manager that raises OperationTimeout after `seconds`.

    Example:
        async with timeout_async(30.0, "payment_gateway"):
            result = await gateway.charge(amount)
    """
    assert seconds > 0, f"timeout must be positive, got {seconds}"
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        _log.error("TIMEOUT: %s exceeded %.1fs", operation, seconds)
        raise OperationTimeout(operation, seconds)


async def timed(
    coro: Coroutine[Any, Any, T],
    *,
    timeout: float,
    label: str = "operation",
) -> T:
    """Await a coroutine with a mandatory timeout.

    Example:
        result = await timed(call_mpesa(), timeout=30.0, label="mpesa.stk_push")
    """
    assert timeout > 0, f"timeout must be positive, got {timeout}"
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        _log.error("TIMEOUT: %s exceeded %.1fs", label, timeout)
        raise OperationTimeout(label, timeout)


def with_timeout(seconds: float, label: str | None = None) -> Any:
    """Decorator: add mandatory timeout to an async function.

    Example:
        @with_timeout(30.0)
        async def call_mpesa(self, amount: float) -> dict: ...
    """
    import functools

    def decorator(fn: Any) -> Any:
        op_label = label or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(fn(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                _log.error("TIMEOUT: %s exceeded %.1fs", op_label, seconds)
                raise OperationTimeout(op_label, seconds)

        return wrapper

    return decorator


def get_timeout(category: str) -> float:
    """Get the standard timeout for a category."""
    return TIMEOUTS.get(category, TIMEOUTS["default"])
