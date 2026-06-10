"""Design-by-Contract decorators for APG services.

Implements Hoare-triple style contracts:
  @requires(predicate)   — precondition, checked on entry
  @ensures(predicate)    — postcondition, checked on exit
  @invariant(predicate)  — class-level invariant, checked pre+post

Usage:
    @requires(lambda self, amount: amount > 0, "amount must be positive")
    @ensures(lambda result: result.get("status") is not None, "status must be set")
    async def process_payment(self, amount: float) -> dict:
        ...
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

_log = logging.getLogger(__name__)


class ContractViolation(AssertionError):
    """Raised when a contract precondition or postcondition is violated.

    This is a programming error, not a runtime error — it indicates
    the calling code or the function itself has a bug.
    """
    def __init__(self, kind: str, predicate_desc: str, context: str = "") -> None:
        self.kind = kind
        self.predicate_desc = predicate_desc
        super().__init__(
            f"Contract {kind} violated: {predicate_desc}"
            + (f" [{context}]" if context else "")
        )


def _pred_name(pred: Callable) -> str:
    """Extract a readable name from a predicate."""
    src = getattr(pred, "__doc__", None) or getattr(pred, "__name__", None)
    if src:
        return src.strip().splitlines()[0]
    # For lambdas, try to get source
    try:
        import inspect as _inspect
        return _inspect.getsource(pred).strip()
    except Exception:
        return repr(pred)


def requires(*predicates_and_msgs: Any) -> Callable:
    """Precondition decorator. Checked before the function executes.

    Args:
        Alternating (predicate, message) pairs, or just predicates.
        predicate receives the same args as the decorated function.

    Example:
        @requires(
            lambda self, amount, **_: amount > 0,
            "amount must be positive",
            lambda self, amount, tenant_id, **_: tenant_id,
            "tenant_id required",
        )
    """
    pairs = _parse_predicates(predicates_and_msgs)

    def decorator(fn: Callable) -> Callable:
        is_coro = asyncio.iscoroutinefunction(fn)

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            _check_requires(pairs, fn, args, kwargs)
            return await fn(*args, **kwargs)

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            _check_requires(pairs, fn, args, kwargs)
            return fn(*args, **kwargs)

        return async_wrapper if is_coro else sync_wrapper

    return decorator


def ensures(*predicates_and_msgs: Any) -> Callable:
    """Postcondition decorator. Checked after the function returns.

    Predicate receives the return value as its only argument.

    Example:
        @ensures(
            lambda r: r is not None, "result must not be None",
            lambda r: "id" in r, "result must contain id",
        )
    """
    pairs = _parse_predicates(predicates_and_msgs)

    def decorator(fn: Callable) -> Callable:
        is_coro = asyncio.iscoroutinefunction(fn)

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await fn(*args, **kwargs)
            _check_ensures(pairs, fn, result)
            return result

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            _check_ensures(pairs, fn, result)
            return result

        return async_wrapper if is_coro else sync_wrapper

    return decorator


def invariant(*predicates_and_msgs: Any) -> Callable:
    """Class-level invariant decorator (applied to a class).

    Wraps all public methods to check the invariant before and after.
    Predicate receives the instance as its only argument.

    Example:
        @invariant(
            lambda self: self._balance >= 0, "balance must be non-negative",
        )
        class Account: ...
    """
    pairs = _parse_predicates(predicates_and_msgs)

    def class_decorator(cls: type) -> type:
        for name in list(vars(cls)):
            if name.startswith("_"):
                continue
            method = getattr(cls, name)
            if not callable(method):
                continue
            if asyncio.iscoroutinefunction(method):
                @functools.wraps(method)
                async def wrapped_async(self: Any, *a: Any, _m=method, **kw: Any) -> Any:
                    _check_invariant(pairs, cls, self, "pre")
                    result = await _m(self, *a, **kw)
                    _check_invariant(pairs, cls, self, "post")
                    return result
                setattr(cls, name, wrapped_async)
            else:
                @functools.wraps(method)
                def wrapped_sync(self: Any, *a: Any, _m=method, **kw: Any) -> Any:
                    _check_invariant(pairs, cls, self, "pre")
                    result = _m(self, *a, **kw)
                    _check_invariant(pairs, cls, self, "post")
                    return result
                setattr(cls, name, wrapped_sync)
        return cls

    return class_decorator


# ── Internal helpers ─────────────────────────────────────────────

def _parse_predicates(args: tuple) -> list[tuple[Callable, str]]:
    """Parse (pred, msg, pred, msg, ...) or (pred, pred, ...) into [(pred, msg), ...]."""
    pairs: list[tuple[Callable, str]] = []
    i = 0
    while i < len(args):
        pred = args[i]
        if not callable(pred):
            raise TypeError(f"Expected callable predicate at position {i}, got {type(pred)}")
        msg = ""
        if i + 1 < len(args) and isinstance(args[i + 1], str):
            msg = args[i + 1]
            i += 2
        else:
            msg = _pred_name(pred)
            i += 1
        pairs.append((pred, msg))
    return pairs


def _check_requires(pairs: list, fn: Callable, args: tuple, kwargs: dict) -> None:
    for pred, msg in pairs:
        try:
            ok = pred(*args, **kwargs)
        except Exception as exc:
            raise ContractViolation("requires", msg, f"predicate raised {exc!r}") from exc
        if not ok:
            _log.error("Precondition violated in %s.%s: %s", fn.__module__, fn.__qualname__, msg)
            raise ContractViolation("requires", msg, f"in {fn.__qualname__}")


def _check_ensures(pairs: list, fn: Callable, result: Any) -> None:
    for pred, msg in pairs:
        try:
            ok = pred(result)
        except Exception as exc:
            raise ContractViolation("ensures", msg, f"predicate raised {exc!r}") from exc
        if not ok:
            _log.error("Postcondition violated in %s.%s: %s", fn.__module__, fn.__qualname__, msg)
            raise ContractViolation("ensures", msg, f"in {fn.__qualname__}")


def _check_invariant(pairs: list, cls: type, instance: Any, when: str) -> None:
    for pred, msg in pairs:
        try:
            ok = pred(instance)
        except Exception as exc:
            raise ContractViolation("invariant", msg, f"predicate raised {exc!r}") from exc
        if not ok:
            _log.error("Invariant violated (%s) on %s: %s", when, cls.__name__, msg)
            raise ContractViolation("invariant", msg, f"{when} in {cls.__name__}")
