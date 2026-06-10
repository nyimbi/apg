"""Runtime input guards and safe async primitives.

Guards are used at service method entry points to validate inputs
before any processing occurs. They raise ValueError with clear messages
that can be surfaced to API clients.

Usage:
    async def process_payment(self, amount: float, tenant_id: str) -> dict:
        guard_positive_amount(amount)
        guard_tenant_id(tenant_id)
        guard_non_empty_string(tenant_id, "tenant_id")
        ...
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from typing import Any, TypeVar

_log = logging.getLogger(__name__)

T = TypeVar("T")


# ── Tracked task creation ─────────────────────────────────────────

def create_tracked_task(
    coro: Any,
    *,
    task_set: "set[asyncio.Task[Any]] | None" = None,
    name: str | None = None,
    log_errors: bool = True,
) -> "asyncio.Task[Any]":
    """Create an asyncio.Task and track it to prevent silent failure.

    Unlike bare `asyncio.create_task()`, this:
    1. Stores the task in `task_set` (if provided) to prevent GC
    2. Attaches a done callback that logs unhandled exceptions
    3. Removes the task from `task_set` when done

    Usage:
        self._tasks: set[asyncio.Task] = set()

        create_tracked_task(
            self._background_loop(),
            task_set=self._tasks,
            name="background_loop",
        )

    Args:
        coro: Coroutine to wrap in a Task.
        task_set: Set to store the task in (prevents GC + enables cancellation).
        name: Task name for logging.
        log_errors: If True, log any unhandled exception at ERROR level.
    """
    task: asyncio.Task[Any] = asyncio.create_task(coro, name=name)

    def _done_callback(t: "asyncio.Task[Any]") -> None:
        if task_set is not None:
            task_set.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None and log_errors:
            task_name = t.get_name() if hasattr(t, "get_name") else str(t)
            _log.error(
                "Unhandled exception in background task %r: %s: %s",
                task_name, type(exc).__name__, exc,
            )

    task.add_done_callback(_done_callback)
    if task_set is not None:
        task_set.add(task)
    return task


# ── Input guards ──────────────────────────────────────────────────

def guard_tenant_id(tenant_id: str | None, field: str = "tenant_id") -> None:
    """Assert tenant_id is a non-empty, non-whitespace string."""
    if not tenant_id or not tenant_id.strip():
        raise ValueError(f"{field} must be a non-empty string, got {tenant_id!r}")
    if len(tenant_id) > 128:
        raise ValueError(f"{field} too long ({len(tenant_id)} chars, max 128)")


def guard_positive_amount(
    amount: float | int | None,
    field: str = "amount",
    *,
    max_value: float = 1e12,
) -> None:
    """Assert amount is a positive finite number within bounds."""
    if amount is None:
        raise ValueError(f"{field} must be provided")
    if not isinstance(amount, (int, float)):
        raise ValueError(f"{field} must be numeric, got {type(amount).__name__}")
    import math
    if math.isnan(amount) or math.isinf(amount):
        raise ValueError(f"{field} must be finite, got {amount}")
    if amount <= 0:
        raise ValueError(f"{field} must be positive, got {amount}")
    if amount > max_value:
        raise ValueError(f"{field} exceeds maximum ({amount} > {max_value})")


def guard_non_empty_string(
    value: str | None,
    field: str = "value",
    *,
    max_length: int = 65535,
    min_length: int = 1,
) -> None:
    """Assert a string is non-empty and within length bounds."""
    if value is None:
        raise ValueError(f"{field} must be provided")
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}")
    stripped = value.strip()
    if len(stripped) < min_length:
        raise ValueError(f"{field} must have at least {min_length} character(s)")
    if len(value) > max_length:
        raise ValueError(f"{field} too long ({len(value)} chars, max {max_length})")


def guard_bounded_list(
    lst: list | None,
    field: str = "list",
    *,
    max_length: int = 10000,
    min_length: int = 0,
    allow_none: bool = False,
) -> None:
    """Assert a list is within size bounds."""
    if lst is None:
        if allow_none:
            return
        raise ValueError(f"{field} must be provided")
    if not isinstance(lst, (list, tuple)):
        raise ValueError(f"{field} must be a list, got {type(lst).__name__}")
    if len(lst) < min_length:
        raise ValueError(f"{field} must have at least {min_length} items")
    if len(lst) > max_length:
        raise ValueError(f"{field} too long ({len(lst)} items, max {max_length})")


def guard_uuid(value: str | None, field: str = "id") -> None:
    """Assert a string is a valid UUID (v4 or v7)."""
    import re
    if not value:
        raise ValueError(f"{field} must be provided")
    UUID_RE = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    if not UUID_RE.match(value):
        raise ValueError(f"{field} must be a valid UUID, got {value!r}")


def guard_enum(value: str | None, allowed: set[str], field: str = "value") -> None:
    """Assert a string value is in the allowed set."""
    if value is None:
        raise ValueError(f"{field} must be provided")
    if value not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)!r}, got {value!r}")


def guard_page(page: int, page_size: int, *, max_page_size: int = 1000) -> None:
    """Assert pagination parameters are within bounds."""
    if page < 1:
        raise ValueError(f"page must be >= 1, got {page}")
    if page_size < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}")
    if page_size > max_page_size:
        raise ValueError(f"page_size must be <= {max_page_size}, got {page_size}")


# ── Safe async primitives ─────────────────────────────────────────

async def safe_gather(
    *coros: Any,
    return_exceptions: bool = True,
    suppress_exceptions: bool = False,
    label: str = "gather",
) -> list[Any]:
    """asyncio.gather with mandatory exception safety.

    Unlike raw asyncio.gather(return_exceptions=True), this NEVER silently swallows errors:
    - With return_exceptions=True (default): exceptions are returned as values
    - Logs each exception found in results
    - With suppress_exceptions=False (default): re-raises if any exception in results

    Args:
        return_exceptions: If True, exceptions are returned as result items.
        suppress_exceptions: If True, exceptions in results are logged but not re-raised.
        label: Used in log messages for identification.

    Example:
        results = await safe_gather(task1(), task2(), task3(), label="batch_process")
    """
    results = await asyncio.gather(*coros, return_exceptions=True)
    exceptions = [(i, r) for i, r in enumerate(results) if isinstance(r, BaseException)]

    if exceptions:
        for i, exc in exceptions:
            _log.error("safe_gather[%s][%d] raised %s: %s", label, i, type(exc).__name__, exc)
        if not suppress_exceptions:
            # Re-raise the first exception to prevent silent failure propagation
            raise exceptions[0][1]

    if return_exceptions:
        return list(results)
    return [r for r in results if not isinstance(r, BaseException)]


# ── Bounded cache ─────────────────────────────────────────────────

class BoundedCache:
    """Thread-safe LRU cache with hard size limit.

    Unlike @functools.lru_cache, this:
    - Has an explicit max_size enforced on every insertion
    - Is safe across threads
    - Evicts LRU items when full
    - Records eviction count for monitoring

    Usage:
        _token_cache = BoundedCache(max_size=1000)
        _token_cache.set("key", value, ttl=300)
        value = _token_cache.get("key")  # None if missing or expired
    """

    def __init__(self, max_size: int = 1000) -> None:
        assert max_size > 0, "max_size must be positive"
        self._max_size = max_size
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._evictions = 0
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing/expired."""
        import time
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            value, expires_at = self._cache[key]
            if expires_at > 0 and time.monotonic() > expires_at:
                del self._cache[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any, *, ttl: float = 0) -> None:
        """Store value with optional TTL (seconds). 0 = no expiry."""
        import time
        expires_at = time.monotonic() + ttl if ttl > 0 else 0.0
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, expires_at)
            # Evict LRU items if over limit
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def stats(self) -> dict[str, int]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(100 * self._hits / total) if total else 0,
                "evictions": self._evictions,
            }
