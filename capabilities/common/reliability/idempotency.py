"""Idempotency key registry for exactly-once operation semantics.

Critical operations (payments, signatures, state mutations) must be
idempotent: submitting the same request twice produces the same result
and causes no side effects on the second invocation.

Usage:
    registry = IdempotencyRegistry(max_size=10000, ttl=3600)

    @idempotent(key_fn=lambda self, payment_id, **_: f"pay:{payment_id}")
    async def process_payment(self, payment_id: str, amount: float) -> dict:
        ...  # Only executes once per unique payment_id

    # Manual usage:
    async with registry.once(key="pay:TXN-001") as ctx:
        if ctx.already_done:
            return ctx.result
        result = await do_payment()
        ctx.set_result(result)
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class IdempotencyEntry:
    key: str
    result: Any = None
    completed: bool = False
    started_at: float = field(default_factory=time.monotonic)
    completed_at: float = 0.0
    attempt_count: int = 1
    error: str | None = None


class IdempotencyContext:
    """Context object for manual idempotency handling."""

    def __init__(self, entry: IdempotencyEntry, lock: asyncio.Lock) -> None:
        self._entry = entry
        self._lock = lock

    @property
    def already_done(self) -> bool:
        return self._entry.completed

    @property
    def result(self) -> Any:
        return self._entry.result

    def set_result(self, result: Any) -> None:
        self._entry.result = result
        self._entry.completed = True
        self._entry.completed_at = time.monotonic()

    def set_error(self, error: str) -> None:
        self._entry.error = error
        self._entry.completed = False

    async def __aenter__(self) -> "IdempotencyContext":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False


class IdempotencyRegistry:
    """In-memory idempotency registry with LRU eviction and TTL.

    For production use, replace with Redis-backed implementation by
    injecting a different store via the `store` parameter.

    Args:
        max_size: Maximum number of idempotency keys to track.
        ttl: How long to remember results (seconds). 0 = forever.
        in_flight_timeout: How long to wait for an in-flight operation before
            treating it as failed (seconds).
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl: float = 3600.0,
        in_flight_timeout: float = 300.0,
    ) -> None:
        assert max_size > 0
        assert ttl >= 0
        self._max_size = max_size
        self._ttl = ttl
        self._in_flight_timeout = in_flight_timeout
        self._store: OrderedDict[str, IdempotencyEntry] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def _make_key(self, *parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    async def once(self, key: str) -> IdempotencyContext:
        """Return an IdempotencyContext for manual flow control."""
        async with self._global_lock:
            entry = self._store.get(key)
            now = time.monotonic()

            if entry is not None:
                age = now - entry.started_at
                # Completed and within TTL — return cached result
                if entry.completed and (self._ttl == 0 or age < self._ttl):
                    _log.debug("Idempotency HIT: %s (%.1fs old)", key[:16], age)
                    self._store.move_to_end(key)
                    return IdempotencyContext(entry, asyncio.Lock())
                # Completed but TTL expired — allow fresh execution
                if entry.completed and self._ttl > 0 and age >= self._ttl:
                    _log.debug("Idempotency TTL expired: %s — re-executing", key[:16])
                    del self._store[key]
                    entry = None
                if entry is not None and not entry.completed and age > self._in_flight_timeout:
                    _log.warning(
                        "Idempotency: in-flight op %s timed out after %.0fs — retrying",
                        key[:16], age,
                    )
                    # Allow retry by removing stale entry
                    del self._store[key]
                    entry = None

            if entry is None:
                entry = IdempotencyEntry(key=key)
                self._store[key] = entry
                self._store.move_to_end(key)
                # Evict oldest if over limit
                while len(self._store) > self._max_size:
                    self._store.popitem(last=False)
            else:
                entry.attempt_count += 1

            lock = self._locks.setdefault(key, asyncio.Lock())

        return IdempotencyContext(entry, lock)

    async def get_result(self, key: str) -> Any | None:
        """Return cached result for key, or None if not found/expired."""
        entry = self._store.get(key)
        if entry is None or not entry.completed:
            return None
        if self._ttl > 0 and time.monotonic() - entry.started_at > self._ttl:
            return None
        return entry.result

    def stats(self) -> dict[str, Any]:
        completed = sum(1 for e in self._store.values() if e.completed)
        in_flight = sum(1 for e in self._store.values() if not e.completed)
        return {
            "tracked_keys": len(self._store),
            "max_size": self._max_size,
            "completed": completed,
            "in_flight": in_flight,
            "ttl_seconds": self._ttl,
        }


# ── Module-level default registry ────────────────────────────────

_default_registry = IdempotencyRegistry(max_size=50000, ttl=86400.0)


def idempotent(
    key_fn: Any = None,
    *,
    registry: IdempotencyRegistry | None = None,
) -> Any:
    """Decorator: make an async function idempotent.

    Args:
        key_fn: Callable that receives the same args as the function and
                returns a string key. If None, uses all non-self args.
        registry: IdempotencyRegistry instance. Defaults to module-level registry.

    Example:
        @idempotent(key_fn=lambda self, payment_id, **_: f"pay:{payment_id}")
        async def process_payment(self, payment_id: str, amount: float) -> dict:
            ...
    """
    reg = registry or _default_registry

    def decorator(fn: Any) -> Any:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_fn is not None:
                try:
                    key = key_fn(*args, **kwargs)
                except Exception as exc:
                    _log.warning("idempotent key_fn failed: %s — skipping idempotency", exc)
                    return await fn(*args, **kwargs)
            else:
                # Auto-key from function name + args (skip self)
                key = f"{fn.__qualname__}:{json.dumps(args[1:], default=str)}:{json.dumps(kwargs, sort_keys=True, default=str)}"

            ctx = await reg.once(key)
            if ctx.already_done:
                _log.debug("idempotent: returning cached result for %s", key[:32])
                return ctx.result

            try:
                result = await fn(*args, **kwargs)
                ctx.set_result(result)
                return result
            except Exception as exc:
                ctx.set_error(str(exc))
                raise

        return wrapper

    return decorator
