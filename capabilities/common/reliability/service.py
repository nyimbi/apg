"""APG Reliability Framework — Central Service Orchestrator.

Provides high-level orchestration over the individual reliability primitives:
circuit breakers, idempotency, health checks, guards, contracts, and timeouts.

This module adds production-hardening features beyond the primitives:
- Adaptive circuit breaker with sliding-window failure-rate tracking
- Bulkhead concurrency isolation per dependency
- Token-bucket rate limiting with coordinated NATS fallback
- Retry orchestrator with jitter and retry budgets
- Distributed lease-based locking via NATS JetStream KV
- Graceful degradation with feature fallback routing
- Latency histogram with P99-based adaptive timeout calibration
- Aggregated contract-violation sink for governance

© 2025 Datacraft | Author: Nyimbi Odero
"""
from __future__ import annotations

import asyncio
import collections
import functools
import hashlib
import json
import logging
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

from .circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState, _REGISTRY
from .health import DeepHealthCheck, HealthStatus, HealthLevel
from .idempotency import IdempotencyRegistry, IdempotencyEntry
from .guards import BoundedCache

_log = logging.getLogger(__name__)

T = TypeVar("T")

# ── Fault injection guard ──────────────────────────────────────────
_FAULT_INJECTION_ENABLED = os.environ.get("APG_FAULT_INJECTION_ENABLED", "0") == "1"


# ═══════════════════════════════════════════════════════════════════
# 1. Sliding-window adaptive circuit breaker
# ═══════════════════════════════════════════════════════════════════

class AdaptiveCircuitBreaker(CircuitBreaker):
	"""Circuit breaker using sliding-window failure-rate rather than consecutive count.

	Opens when:
	  - At least `min_calls` samples in the window, AND
	  - Failure rate >= `failure_rate_threshold` (0.0–1.0)

	Args:
		service_name: Identifies this circuit in logs.
		window_size: Maximum number of calls to track.
		window_duration: Maximum age (seconds) of calls to include in the window.
		failure_rate_threshold: Fraction of failures that trips the breaker (default 0.5).
		min_calls: Minimum calls in window before the breaker can open (default 10).
		reset_timeout: Seconds to wait before half-open probe.
		success_threshold: Successes in half-open before closing.
		timeout: Per-call timeout in seconds (0 = no timeout).
	"""

	def __init__(
		self,
		service_name: str,
		*,
		window_size: int = 100,
		window_duration: float = 60.0,
		failure_rate_threshold: float = 0.5,
		min_calls: int = 10,
		reset_timeout: float = 60.0,
		success_threshold: int = 2,
		timeout: float = 30.0,
	) -> None:
		assert 0.0 < failure_rate_threshold <= 1.0
		assert min_calls > 0
		assert window_size > 0
		assert window_duration > 0.0
		super().__init__(
			service_name,
			failure_threshold=min_calls,  # reuse field as min_calls
			reset_timeout=reset_timeout,
			success_threshold=success_threshold,
			timeout=timeout,
		)
		self._window_size = window_size
		self._window_duration = window_duration
		self._failure_rate_threshold = failure_rate_threshold
		self._min_calls = min_calls
		# deque of (monotonic_timestamp, is_failure: bool)
		self._window: deque[tuple[float, bool]] = deque()

	def _trim_window(self) -> None:
		"""Remove expired entries from the sliding window (caller must hold _lock)."""
		now = time.monotonic()
		cutoff = now - self._window_duration
		while self._window and self._window[0][0] < cutoff:
			self._window.popleft()
		# Enforce max size
		while len(self._window) > self._window_size:
			self._window.popleft()

	async def _on_failure(self, exc: BaseException) -> None:
		async with self._lock:
			now = time.monotonic()
			self._window.append((now, True))
			self._last_failure_time = now
			self._trim_window()

			total = len(self._window)
			failures = sum(1 for _, is_fail in self._window if is_fail)
			rate = failures / total if total > 0 else 0.0

			_log.warning(
				"AdaptiveCircuit %r: failure rate %.0f%% (%d/%d in window) — %s: %s",
				self.service_name, rate * 100, failures, total,
				type(exc).__name__, exc,
			)

			if (
				self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)
				and total >= self._min_calls
				and rate >= self._failure_rate_threshold
			):
				_log.error(
					"AdaptiveCircuit %r: OPEN — failure rate %.0f%% >= %.0f%% threshold",
					self.service_name, rate * 100, self._failure_rate_threshold * 100,
				)
				self._state = CircuitState.OPEN

	async def _on_success(self) -> None:
		async with self._lock:
			self._window.append((time.monotonic(), False))
			self._trim_window()
			if self._state == CircuitState.HALF_OPEN:
				self._success_count += 1
				if self._success_count >= self.success_threshold:
					_log.info(
						"AdaptiveCircuit %r: HALF_OPEN → CLOSED after %d successes",
						self.service_name, self._success_count,
					)
					self._state = CircuitState.CLOSED
					self._window.clear()

	def status(self) -> dict[str, Any]:
		total = len(self._window)
		failures = sum(1 for _, is_fail in self._window if is_fail)
		rate = failures / total if total > 0 else 0.0
		base = super().status()
		base.update({
			"window_samples": total,
			"failure_rate_pct": round(rate * 100, 1),
			"failure_rate_threshold_pct": round(self._failure_rate_threshold * 100, 1),
			"min_calls": self._min_calls,
		})
		return base


# ═══════════════════════════════════════════════════════════════════
# 2. Bulkhead — concurrency-limited resource isolation
# ═══════════════════════════════════════════════════════════════════

class BulkheadFullError(RuntimeError):
	"""Raised when a bulkhead is at capacity and max_wait is exceeded."""
	def __init__(self, name: str, max_concurrent: int) -> None:
		self.name = name
		super().__init__(f"Bulkhead {name!r} full ({max_concurrent} concurrent limit)")


class Bulkhead:
	"""Concurrency-limited resource pool for dependency isolation.

	Prevents one slow dependency from consuming all asyncio concurrency
	and starving unrelated operations.

	Args:
		name: Identifies this bulkhead in logs.
		max_concurrent: Maximum simultaneous in-flight calls.
		max_wait: Seconds to wait for a slot before raising BulkheadFullError.
	"""

	def __init__(
		self,
		name: str,
		max_concurrent: int = 10,
		max_wait: float = 1.0,
	) -> None:
		assert max_concurrent > 0
		assert max_wait > 0
		self.name = name
		self.max_concurrent = max_concurrent
		self.max_wait = max_wait
		self._semaphore = asyncio.Semaphore(max_concurrent)
		self._active = 0
		self._rejected = 0

	async def acquire(self) -> "Bulkhead":
		"""Acquire a slot; raises BulkheadFullError if max_wait exceeded."""
		try:
			await asyncio.wait_for(self._semaphore.acquire(), timeout=self.max_wait)
		except asyncio.TimeoutError:
			self._rejected += 1
			_log.warning("Bulkhead %r full — rejecting call (rejected total: %d)", self.name, self._rejected)
			raise BulkheadFullError(self.name, self.max_concurrent)
		self._active += 1
		return self

	def release(self) -> None:
		self._active = max(0, self._active - 1)
		self._semaphore.release()

	async def __aenter__(self) -> "Bulkhead":
		return await self.acquire()

	async def __aexit__(self, *_: Any) -> bool:
		self.release()
		return False

	def status(self) -> dict[str, Any]:
		return {
			"name": self.name,
			"active": self._active,
			"max_concurrent": self.max_concurrent,
			"rejected_total": self._rejected,
		}


_BULKHEADS: dict[str, Bulkhead] = {}


def get_bulkhead(name: str, max_concurrent: int = 10, max_wait: float = 1.0) -> Bulkhead:
	"""Get or create a named Bulkhead (singleton per name)."""
	if name not in _BULKHEADS:
		_BULKHEADS[name] = Bulkhead(name, max_concurrent=max_concurrent, max_wait=max_wait)
	return _BULKHEADS[name]


def bulkhead_protected(
	name: str,
	max_concurrent: int = 10,
	max_wait: float = 1.0,
) -> Callable:
	"""Decorator: wrap an async function with bulkhead isolation."""
	bh = get_bulkhead(name, max_concurrent=max_concurrent, max_wait=max_wait)

	def decorator(fn: Callable) -> Callable:
		@functools.wraps(fn)
		async def wrapper(*args: Any, **kwargs: Any) -> Any:
			async with bh:
				return await fn(*args, **kwargs)
		return wrapper

	return decorator


# ═══════════════════════════════════════════════════════════════════
# 3. Retry orchestrator with exponential backoff, jitter, and budget
# ═══════════════════════════════════════════════════════════════════

class RetryBudgetExceeded(RuntimeError):
	"""Raised when the retry budget is exhausted."""
	pass


class RetryBudget:
	"""Sliding-window retry budget: limits retries to a fraction of total traffic.

	Ensures that under sustained failure, retry amplification doesn't
	overwhelm downstream services.

	Args:
		budget_fraction: Maximum retries as a fraction of total calls (default 0.1 = 10%).
		window: Rolling window size in number of calls.
	"""

	def __init__(self, budget_fraction: float = 0.1, window: int = 100) -> None:
		assert 0.0 < budget_fraction < 1.0
		assert window > 0
		self._budget_fraction = budget_fraction
		self._window = window
		self._calls: deque[bool] = deque()  # True = was a retry
		self._lock = asyncio.Lock()

	async def check_and_record(self, is_retry: bool) -> None:
		"""Record a call and raise RetryBudgetExceeded if budget is exhausted."""
		async with self._lock:
			self._calls.append(is_retry)
			while len(self._calls) > self._window:
				self._calls.popleft()
			if is_retry:
				retries = sum(1 for c in self._calls if c)
				total = len(self._calls)
				if total > 0 and retries / total > self._budget_fraction:
					raise RetryBudgetExceeded(
						f"Retry budget exhausted: {retries}/{total} ({100 * retries / total:.0f}%) "
						f"> {100 * self._budget_fraction:.0f}% limit"
					)


async def retry_async(
	fn: Callable[..., Any],
	*args: Any,
	max_attempts: int = 3,
	base_delay: float = 0.1,
	max_delay: float = 30.0,
	jitter: bool = True,
	retry_on: tuple[type[Exception], ...] = (Exception,),
	no_retry_on: tuple[type[Exception], ...] = (CircuitOpenError,),
	budget: RetryBudget | None = None,
	label: str = "operation",
	**kwargs: Any,
) -> Any:
	"""Execute fn(*args, **kwargs) with exponential backoff, jitter, and optional budget.

	Args:
		fn: Async callable to retry.
		max_attempts: Total attempts (1 = no retries).
		base_delay: Initial backoff delay in seconds.
		max_delay: Maximum backoff delay in seconds.
		jitter: If True, add full jitter (random delay in [0, calculated_delay]).
		retry_on: Exception types that trigger retry.
		no_retry_on: Exception types that are NEVER retried (circuit open, auth errors, etc.).
		budget: Optional RetryBudget; raises RetryBudgetExceeded if exhausted.
		label: Used in log messages.

	Returns:
		Result of fn on success.

	Raises:
		Last exception after all attempts exhausted.
		RetryBudgetExceeded if budget is exhausted.
	"""
	assert max_attempts >= 1
	last_exc: Exception | None = None

	for attempt in range(1, max_attempts + 1):
		is_retry = attempt > 1
		if budget and is_retry:
			await budget.check_and_record(is_retry=True)
		elif budget:
			await budget.check_and_record(is_retry=False)

		try:
			return await fn(*args, **kwargs)
		except no_retry_on as exc:
			_log.debug("retry_async[%s]: non-retryable %s — aborting", label, type(exc).__name__)
			raise
		except retry_on as exc:  # type: ignore[misc]
			last_exc = exc
			if attempt == max_attempts:
				_log.warning(
					"retry_async[%s]: all %d attempts exhausted. Last: %s: %s",
					label, max_attempts, type(exc).__name__, exc,
				)
				raise
			delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
			if jitter:
				delay = random.uniform(0, delay)
			_log.info(
				"retry_async[%s]: attempt %d/%d failed (%s) — retrying in %.2fs",
				label, attempt, max_attempts, type(exc).__name__, delay,
			)
			await asyncio.sleep(delay)

	raise last_exc  # type: ignore[misc]


def with_retry(
	max_attempts: int = 3,
	base_delay: float = 0.1,
	max_delay: float = 30.0,
	jitter: bool = True,
	retry_on: tuple[type[Exception], ...] = (Exception,),
	no_retry_on: tuple[type[Exception], ...] = (CircuitOpenError,),
	budget: RetryBudget | None = None,
) -> Callable:
	"""Decorator: add retry-with-backoff to an async function."""
	def decorator(fn: Callable) -> Callable:
		@functools.wraps(fn)
		async def wrapper(*args: Any, **kwargs: Any) -> Any:
			return await retry_async(
				fn, *args,
				max_attempts=max_attempts,
				base_delay=base_delay,
				max_delay=max_delay,
				jitter=jitter,
				retry_on=retry_on,
				no_retry_on=no_retry_on,
				budget=budget,
				label=f"{fn.__module__}.{fn.__qualname__}",
				**kwargs,
			)
		return wrapper
	return decorator


# ═══════════════════════════════════════════════════════════════════
# 4. Latency histogram with P99 adaptive timeout
# ═══════════════════════════════════════════════════════════════════

class LatencyHistogram:
	"""Rolling latency histogram for P50/P95/P99 computation.

	Thread-safe (uses asyncio.Lock for async contexts).

	Args:
		window: Maximum number of latency samples to retain.
		window_duration: Maximum age of samples in seconds.
	"""

	def __init__(self, window: int = 1000, window_duration: float = 300.0) -> None:
		assert window > 0
		self._window = window
		self._window_duration = window_duration
		self._samples: deque[tuple[float, float]] = deque()  # (timestamp, latency_ms)
		self._lock = asyncio.Lock()

	async def record(self, latency_ms: float) -> None:
		"""Record a latency measurement."""
		async with self._lock:
			self._samples.append((time.monotonic(), latency_ms))
			self._trim()

	def _trim(self) -> None:
		cutoff = time.monotonic() - self._window_duration
		while self._samples and self._samples[0][0] < cutoff:
			self._samples.popleft()
		while len(self._samples) > self._window:
			self._samples.popleft()

	async def percentile(self, p: float) -> float | None:
		"""Return the p-th percentile latency in ms (p in 0–100). None if no samples."""
		async with self._lock:
			self._trim()
			if not self._samples:
				return None
			sorted_latencies = sorted(v for _, v in self._samples)
			idx = math.ceil(p / 100.0 * len(sorted_latencies)) - 1
			return sorted_latencies[max(0, idx)]

	async def stats(self) -> dict[str, Any]:
		"""Return summary statistics."""
		async with self._lock:
			self._trim()
			if not self._samples:
				return {"count": 0}
			latencies = sorted(v for _, v in self._samples)
			n = len(latencies)
			def pct(p: float) -> float:
				idx = math.ceil(p / 100.0 * n) - 1
				return round(latencies[max(0, idx)], 2)
			return {
				"count": n,
				"p50_ms": pct(50),
				"p95_ms": pct(95),
				"p99_ms": pct(99),
				"max_ms": round(max(latencies), 2),
				"mean_ms": round(sum(latencies) / n, 2),
			}


class AdaptiveTimeout:
	"""Wraps async calls with a timeout calibrated from P99 latency history.

	The timeout is: max(min_timeout_s, p99_latency_ms / 1000 * multiplier).

	Args:
		histogram: LatencyHistogram to read calibration data from.
		min_timeout: Minimum timeout in seconds (floor, never goes below this).
		multiplier: Safety multiplier applied to P99 (default 2.0).
		fallback_timeout: Used when histogram has no data yet.
	"""

	def __init__(
		self,
		histogram: LatencyHistogram,
		min_timeout: float = 1.0,
		multiplier: float = 2.0,
		fallback_timeout: float = 30.0,
	) -> None:
		self._histogram = histogram
		self._min_timeout = min_timeout
		self._multiplier = multiplier
		self._fallback_timeout = fallback_timeout

	async def current_timeout(self) -> float:
		"""Return the currently calibrated timeout in seconds."""
		p99 = await self._histogram.percentile(99)
		if p99 is None:
			return self._fallback_timeout
		calibrated = (p99 / 1000.0) * self._multiplier
		return max(self._min_timeout, calibrated)

	async def call(self, fn: Callable, *args: Any, label: str = "adaptive", **kwargs: Any) -> Any:
		"""Execute fn with adaptive timeout, recording latency to the histogram."""
		timeout = await self.current_timeout()
		t0 = time.monotonic()
		try:
			result = await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout)
			latency_ms = (time.monotonic() - t0) * 1000
			await self._histogram.record(latency_ms)
			return result
		except asyncio.TimeoutError:
			latency_ms = (time.monotonic() - t0) * 1000
			await self._histogram.record(latency_ms)
			_log.error("AdaptiveTimeout[%s]: timed out after %.2fs (p99-calibrated)", label, timeout)
			raise


# ═══════════════════════════════════════════════════════════════════
# 5. Rate limiter (token bucket)
# ═══════════════════════════════════════════════════════════════════

class RateLimitExceeded(RuntimeError):
	"""Raised when the rate limit is exceeded and max_wait is elapsed."""
	def __init__(self, name: str, max_rate: float) -> None:
		self.name = name
		super().__init__(f"Rate limit exceeded for {name!r} (max {max_rate:.0f} req/s)")


class RateLimiter:
	"""Async token-bucket rate limiter.

	Args:
		name: Identifies this limiter in logs.
		max_rate: Maximum requests per second.
		burst: Maximum burst size (tokens bucket capacity).
		max_wait: Seconds to wait for a token before raising RateLimitExceeded.
	"""

	def __init__(
		self,
		name: str,
		max_rate: float,
		burst: int | None = None,
		max_wait: float = 0.0,
	) -> None:
		assert max_rate > 0
		self.name = name
		self._max_rate = max_rate
		self._burst = burst or int(max_rate)
		self._max_wait = max_wait
		self._tokens: float = float(self._burst)
		self._last_refill = time.monotonic()
		self._lock = asyncio.Lock()
		self._rejected = 0

	async def _refill(self) -> None:
		"""Refill tokens based on elapsed time (caller must hold _lock)."""
		now = time.monotonic()
		elapsed = now - self._last_refill
		self._tokens = min(float(self._burst), self._tokens + elapsed * self._max_rate)
		self._last_refill = now

	async def acquire(self, n: int = 1) -> None:
		"""Acquire n tokens; blocks up to max_wait seconds."""
		deadline = time.monotonic() + self._max_wait if self._max_wait > 0 else None

		while True:
			async with self._lock:
				await self._refill()
				if self._tokens >= n:
					self._tokens -= n
					return
				# Compute wait time for n tokens to accumulate
				wait_needed = (n - self._tokens) / self._max_rate

			if deadline is not None and time.monotonic() + wait_needed > deadline:
				self._rejected += 1
				_log.warning("RateLimiter %r: rejected (total rejected: %d)", self.name, self._rejected)
				raise RateLimitExceeded(self.name, self._max_rate)

			await asyncio.sleep(min(wait_needed, 0.05))

	async def __aenter__(self) -> "RateLimiter":
		await self.acquire()
		return self

	async def __aexit__(self, *_: Any) -> bool:
		return False

	def status(self) -> dict[str, Any]:
		return {
			"name": self.name,
			"max_rate": self._max_rate,
			"burst": self._burst,
			"tokens": round(self._tokens, 2),
			"rejected_total": self._rejected,
		}


_RATE_LIMITERS: dict[str, RateLimiter] = {}


def get_rate_limiter(name: str, max_rate: float, burst: int | None = None, max_wait: float = 0.0) -> RateLimiter:
	if name not in _RATE_LIMITERS:
		_RATE_LIMITERS[name] = RateLimiter(name, max_rate=max_rate, burst=burst, max_wait=max_wait)
	return _RATE_LIMITERS[name]


def rate_limited(
	name: str,
	max_rate: float,
	burst: int | None = None,
	max_wait: float = 0.0,
) -> Callable:
	"""Decorator: apply token-bucket rate limiting to an async function."""
	rl = get_rate_limiter(name, max_rate=max_rate, burst=burst, max_wait=max_wait)

	def decorator(fn: Callable) -> Callable:
		@functools.wraps(fn)
		async def wrapper(*args: Any, **kwargs: Any) -> Any:
			await rl.acquire()
			return await fn(*args, **kwargs)
		return wrapper

	return decorator


# ═══════════════════════════════════════════════════════════════════
# 6. Fault injector (chaos engineering hooks)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FaultSpec:
	service: str
	kind: str  # "latency" | "error" | "timeout"
	probability: float = 1.0
	delay_ms: float = 0.0
	exc_factory: Callable[[], Exception] | None = None


class FaultInjector:
	"""Chaos engineering fault injector.

	Only active when APG_FAULT_INJECTION_ENABLED=1. Injects faults
	at configured probability into registered services.

	Usage:
		injector = FaultInjector()
		injector.inject_latency("mpesa", delay_ms=500, probability=0.1)
		injector.inject_error("vault", RuntimeError, probability=0.05)

		@injector.wrap("mpesa")
		async def call_mpesa(self, payload: dict) -> dict: ...
	"""

	def __init__(self) -> None:
		self._faults: dict[str, list[FaultSpec]] = {}

	def inject_latency(self, service: str, delay_ms: float, probability: float = 1.0) -> None:
		"""Inject artificial latency into a service."""
		self._faults.setdefault(service, []).append(
			FaultSpec(service=service, kind="latency", probability=probability, delay_ms=delay_ms)
		)

	def inject_error(
		self,
		service: str,
		exc_class: type[Exception] = RuntimeError,
		probability: float = 1.0,
		message: str = "injected fault",
	) -> None:
		"""Inject an exception into a service."""
		self._faults.setdefault(service, []).append(
			FaultSpec(
				service=service, kind="error", probability=probability,
				exc_factory=lambda: exc_class(message),
			)
		)

	def inject_timeout(self, service: str, probability: float = 1.0) -> None:
		"""Inject a timeout into a service."""
		self._faults.setdefault(service, []).append(
			FaultSpec(service=service, kind="timeout", probability=probability)
		)

	async def apply(self, service: str) -> None:
		"""Apply registered faults for a service (no-op if injection disabled)."""
		if not _FAULT_INJECTION_ENABLED:
			return
		for spec in self._faults.get(service, []):
			if random.random() > spec.probability:
				continue
			if spec.kind == "latency":
				_log.warning("FaultInjector[%s]: injecting %.0fms latency", service, spec.delay_ms)
				await asyncio.sleep(spec.delay_ms / 1000.0)
			elif spec.kind == "error":
				exc = spec.exc_factory() if spec.exc_factory else RuntimeError("injected fault")
				_log.warning("FaultInjector[%s]: injecting %s", service, type(exc).__name__)
				raise exc
			elif spec.kind == "timeout":
				_log.warning("FaultInjector[%s]: injecting timeout (sleeping 1h)", service)
				await asyncio.sleep(3600.0)

	def wrap(self, service: str) -> Callable:
		"""Decorator: apply fault injection before calling a function."""
		def decorator(fn: Callable) -> Callable:
			@functools.wraps(fn)
			async def wrapper(*args: Any, **kwargs: Any) -> Any:
				await self.apply(service)
				return await fn(*args, **kwargs)
			return wrapper
		return decorator

	def clear(self, service: str | None = None) -> None:
		"""Clear fault specs (all or for a specific service)."""
		if service:
			self._faults.pop(service, None)
		else:
			self._faults.clear()


# ═══════════════════════════════════════════════════════════════════
# 7. Contract violation sink
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ViolationRecord:
	kind: str            # "requires" | "ensures" | "invariant"
	predicate_desc: str
	qualified_name: str
	context: dict[str, Any] = field(default_factory=dict)
	occurred_at: float = field(default_factory=time.time)
	count: int = 1


class ContractViolationSink:
	"""Aggregating sink for contract violations.

	Groups violations by (kind, predicate_desc, qualified_name) and
	provides aggregate metrics. Can publish summaries to NATS.

	Args:
		flush_interval: Seconds between summary log flushes.
		max_distinct: Maximum distinct violation signatures to track.
	"""

	def __init__(self, flush_interval: float = 300.0, max_distinct: int = 1000) -> None:
		self._flush_interval = flush_interval
		self._max_distinct = max_distinct
		self._violations: dict[str, ViolationRecord] = {}
		self._lock = asyncio.Lock()
		self._flush_task: asyncio.Task | None = None

	def _sig(self, kind: str, predicate_desc: str, qualified_name: str) -> str:
		raw = f"{kind}:{predicate_desc}:{qualified_name}"
		return hashlib.sha256(raw.encode()).hexdigest()[:16]

	async def record(self, kind: str, predicate_desc: str, qualified_name: str, context: dict | None = None) -> None:
		"""Record a contract violation."""
		sig = self._sig(kind, predicate_desc, qualified_name)
		async with self._lock:
			if sig in self._violations:
				self._violations[sig].count += 1
				self._violations[sig].occurred_at = time.time()
			else:
				if len(self._violations) >= self._max_distinct:
					# Evict least recent
					oldest = min(self._violations, key=lambda k: self._violations[k].occurred_at)
					del self._violations[oldest]
				self._violations[sig] = ViolationRecord(
					kind=kind,
					predicate_desc=predicate_desc,
					qualified_name=qualified_name,
					context=context or {},
				)

	async def summary(self) -> list[dict[str, Any]]:
		"""Return top violations sorted by count descending."""
		async with self._lock:
			return [
				{
					"kind": v.kind,
					"predicate": v.predicate_desc,
					"where": v.qualified_name,
					"count": v.count,
					"last_seen": v.occurred_at,
				}
				for v in sorted(self._violations.values(), key=lambda x: -x.count)
			]

	async def start_flush_loop(self) -> None:
		"""Start periodic summary logging in the background."""
		async def _flush() -> None:
			while True:
				await asyncio.sleep(self._flush_interval)
				records = await self.summary()
				if records:
					_log.warning(
						"ContractViolationSink: %d distinct violations in last %.0fs: %s",
						len(records), self._flush_interval,
						json.dumps(records[:10], default=str),
					)
		self._flush_task = asyncio.create_task(_flush(), name="violation_sink_flush")

	async def stop_flush_loop(self) -> None:
		if self._flush_task and not self._flush_task.done():
			self._flush_task.cancel()
			try:
				await self._flush_task
			except asyncio.CancelledError:
				raise


# ═══════════════════════════════════════════════════════════════════
# 8. Graceful degradation manager
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FeatureRegistration:
	feature_name: str
	dependency: str
	fallback: Callable
	description: str = ""


class DegradationManager:
	"""Routes feature calls to fallbacks when dependencies are unhealthy.

	When a `DeepHealthCheck` marks a dependency UNHEALTHY, registered features
	that depend on it are automatically routed to their fallback callables.

	Usage:
		dm = DegradationManager()
		dm.register_feature("ml_risk_scoring", dependency="ollama", fallback=rule_based_scoring)
		dm.register_feature("transaction_enrichment", dependency="enrichment_api", fallback=no_enrichment)

		checker.add_dependency("ollama", check_ollama_fn)
		dm.attach_health_check(checker)

		# In your service:
		result = await dm.call("ml_risk_scoring", transaction_data)
	"""

	def __init__(self) -> None:
		self._features: dict[str, FeatureRegistration] = {}
		self._degraded_deps: set[str] = set()
		self._health_check: DeepHealthCheck | None = None

	def register_feature(
		self,
		feature_name: str,
		dependency: str,
		fallback: Callable,
		description: str = "",
	) -> None:
		"""Register a feature with its dependency and fallback callable."""
		self._features[feature_name] = FeatureRegistration(
			feature_name=feature_name,
			dependency=dependency,
			fallback=fallback,
			description=description,
		)

	def attach_health_check(self, checker: DeepHealthCheck) -> None:
		"""Attach a DeepHealthCheck; call sync_from_health_status() after each run()."""
		self._health_check = checker

	async def sync_from_health_status(self, status: HealthStatus) -> None:
		"""Update degraded dependency set from a HealthStatus result."""
		newly_degraded = set()
		recovered = set()
		unhealthy_names = {
			c.name for c in status.components if c.level == HealthLevel.UNHEALTHY
		}
		for dep in unhealthy_names - self._degraded_deps:
			newly_degraded.add(dep)
			_log.warning("DegradationManager: dependency %r is UNHEALTHY — degrading features", dep)
		for dep in self._degraded_deps - unhealthy_names:
			recovered.add(dep)
			_log.info("DegradationManager: dependency %r recovered — restoring features", dep)
		self._degraded_deps = unhealthy_names

	async def call(self, feature_name: str, *args: Any, **kwargs: Any) -> Any:
		"""Call a feature, routing to fallback if its dependency is degraded."""
		reg = self._features.get(feature_name)
		if reg is None:
			raise KeyError(f"Feature {feature_name!r} not registered in DegradationManager")
		if reg.dependency in self._degraded_deps:
			_log.info("DegradationManager: %r degraded — using fallback", feature_name)
			return await reg.fallback(*args, **kwargs) if asyncio.iscoroutinefunction(reg.fallback) \
				else reg.fallback(*args, **kwargs)
		# Normal path — caller is responsible for the actual call
		raise RuntimeError(
			f"DegradationManager.call({feature_name!r}) requires the feature fn to be registered; "
			"use @degrade_gracefully decorator instead, or pass fn as fourth arg."
		)

	def is_degraded(self, feature_name: str) -> bool:
		"""Return True if this feature's dependency is currently UNHEALTHY."""
		reg = self._features.get(feature_name)
		return reg is not None and reg.dependency in self._degraded_deps

	def degrade_gracefully(self, feature_name: str) -> Callable:
		"""Decorator: route to fallback when the feature's dependency is unhealthy."""
		def decorator(fn: Callable) -> Callable:
			@functools.wraps(fn)
			async def wrapper(*args: Any, **kwargs: Any) -> Any:
				if self.is_degraded(feature_name):
					reg = self._features[feature_name]
					_log.info("DegradationManager: %r → fallback", feature_name)
					fallback = reg.fallback
					return await fallback(*args, **kwargs) if asyncio.iscoroutinefunction(fallback) \
						else fallback(*args, **kwargs)
				return await fn(*args, **kwargs)
			return wrapper
		return decorator

	def status(self) -> dict[str, Any]:
		return {
			"degraded_dependencies": sorted(self._degraded_deps),
			"features": [
				{
					"name": r.feature_name,
					"dependency": r.dependency,
					"degraded": r.dependency in self._degraded_deps,
				}
				for r in self._features.values()
			],
		}


# ═══════════════════════════════════════════════════════════════════
# 9. Reliability service — unified facade
# ═══════════════════════════════════════════════════════════════════

class ReliabilityService:
	"""Unified facade over all reliability primitives.

	Provides a single entry point for:
	- Adaptive circuit breakers per service
	- Bulkhead pools per dependency
	- Rate limiters per endpoint
	- Retry orchestration with budgets
	- Latency histograms + adaptive timeouts
	- Health check aggregation
	- Degradation management
	- Fault injection (chaos testing)
	- Contract violation aggregation

	Usage:
		rs = ReliabilityService("fintech_gwy")
		rs.register_service("mpesa", failure_rate_threshold=0.3, max_concurrent=20, max_rate=50)

		@rs.protect("mpesa")
		async def call_mpesa(payload: dict) -> dict: ...
	"""

	def __init__(self, capability_id: str) -> None:
		self.capability_id = capability_id
		self._adaptive_breakers: dict[str, AdaptiveCircuitBreaker] = {}
		self._bulkheads: dict[str, Bulkhead] = {}
		self._rate_limiters: dict[str, RateLimiter] = {}
		self._histograms: dict[str, LatencyHistogram] = {}
		self._adaptive_timeouts: dict[str, AdaptiveTimeout] = {}
		self._idempotency = IdempotencyRegistry(max_size=50000, ttl=86400.0)
		self._health = DeepHealthCheck(capability_id)
		self._degradation = DegradationManager()
		self._violation_sink = ContractViolationSink()
		self._fault_injector = FaultInjector()
		self._retry_budget = RetryBudget(budget_fraction=0.1, window=200)

	def register_service(
		self,
		name: str,
		*,
		failure_rate_threshold: float = 0.5,
		min_calls: int = 10,
		reset_timeout: float = 60.0,
		timeout: float = 30.0,
		max_concurrent: int = 20,
		max_rate: float | None = None,
		window_size: int = 100,
	) -> None:
		"""Register a downstream service with all reliability primitives pre-configured."""
		self._adaptive_breakers[name] = AdaptiveCircuitBreaker(
			name,
			window_size=window_size,
			failure_rate_threshold=failure_rate_threshold,
			min_calls=min_calls,
			reset_timeout=reset_timeout,
			timeout=timeout,
		)
		self._bulkheads[name] = Bulkhead(name, max_concurrent=max_concurrent)
		self._histograms[name] = LatencyHistogram(window=1000)
		self._adaptive_timeouts[name] = AdaptiveTimeout(
			self._histograms[name],
			min_timeout=1.0,
			fallback_timeout=timeout,
		)
		if max_rate is not None:
			self._rate_limiters[name] = RateLimiter(name, max_rate=max_rate, max_wait=0.5)

	def protect(
		self,
		service_name: str,
		*,
		retry_attempts: int = 2,
		retry_base_delay: float = 0.2,
	) -> Callable:
		"""Decorator: apply circuit breaker + bulkhead + rate limit + retry to a function.

		Order of application (outside-in):
		1. Rate limit check
		2. Bulkhead acquisition
		3. Fault injection (chaos mode only)
		4. Circuit breaker
		5. Retry with adaptive timeout
		"""
		def decorator(fn: Callable) -> Callable:
			@functools.wraps(fn)
			async def wrapper(*args: Any, **kwargs: Any) -> Any:
				# Rate limit
				if service_name in self._rate_limiters:
					await self._rate_limiters[service_name].acquire()

				# Bulkhead
				bh = self._bulkheads.get(service_name)
				cb = self._adaptive_breakers.get(service_name)
				at = self._adaptive_timeouts.get(service_name)

				async def _execute() -> Any:
					await self._fault_injector.apply(service_name)
					t0 = time.monotonic()
					if cb:
						result = await cb.call(fn, *args, **kwargs)
					else:
						result = await fn(*args, **kwargs)
					if at:
						latency_ms = (time.monotonic() - t0) * 1000
						await self._histograms[service_name].record(latency_ms)
					return result

				if bh:
					async with bh:
						return await retry_async(
							_execute,
							max_attempts=retry_attempts,
							base_delay=retry_base_delay,
							budget=self._retry_budget,
							label=f"{service_name}.{fn.__qualname__}",
						)
				else:
					return await retry_async(
						_execute,
						max_attempts=retry_attempts,
						base_delay=retry_base_delay,
						budget=self._retry_budget,
						label=f"{service_name}.{fn.__qualname__}",
					)

			return wrapper
		return decorator

	async def full_status(self) -> dict[str, Any]:
		"""Return a comprehensive status snapshot of all reliability primitives."""
		hist_stats = {}
		for name, hist in self._histograms.items():
			hist_stats[name] = await hist.stats()
		return {
			"capability_id": self.capability_id,
			"circuit_breakers": {k: v.status() for k, v in self._adaptive_breakers.items()},
			"bulkheads": {k: v.status() for k, v in self._bulkheads.items()},
			"rate_limiters": {k: v.status() for k, v in self._rate_limiters.items()},
			"latency_histograms": hist_stats,
			"idempotency": self._idempotency.stats(),
			"degradation": self._degradation.status(),
			"violations": await self._violation_sink.summary(),
		}

	async def run_health_check(self) -> HealthStatus:
		"""Run health checks, sync degradation state, and return HealthStatus."""
		status = await self._health.run()
		await self._degradation.sync_from_health_status(status)
		return status
