"""
APG API Gateway - Analytics Extension

Consumer analytics, rate limit transparency, quota management, and
per-consumer usage dashboards. Metrics aggregated from NATS events
on the apg.events.*.api_call.* subject hierarchy.

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from datetime import datetime, timezone, timedelta
from typing import Any

from pydantic import BaseModel, Field, ConfigDict, field_validator, computed_field
from pydantic import NonNegativeFloat, NonNegativeInt

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def uuid7str() -> str:
		return str(uuid.uuid4())

_logger = logging.getLogger(__name__)

_AG_MODEL_CONFIG = ConfigDict(
	extra="forbid",
	validate_by_name=True,
	validate_by_alias=True,
	str_strip_whitespace=True,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AgApiCallRecord(BaseModel):
	"""Single API call record captured by gateway middleware."""

	model_config = _AG_MODEL_CONFIG

	id: str = Field(default_factory=uuid7str, description="Unique record ID (uuid7)")
	key_id: str = Field(..., description="API key identifier")
	tenant_id: str = Field(..., description="Tenant scope")
	endpoint: str = Field(..., description="Request path, e.g. /api/v1/orders")
	method: str = Field(..., description="HTTP method")
	status_code: int = Field(..., ge=100, le=599, description="HTTP response status code")
	latency_ms: NonNegativeFloat = Field(..., description="Round-trip latency in milliseconds")
	timestamp: datetime = Field(
		default_factory=lambda: datetime.now(timezone.utc),
		description="UTC timestamp of the call",
	)

	@field_validator("method")
	@classmethod
	def _upper_method(cls, v: str) -> str:
		return v.upper()

	@computed_field  # type: ignore[misc]
	@property
	def is_error(self) -> bool:
		return self.status_code >= 400


class AgUsageSummary(BaseModel):
	"""Aggregated usage statistics for an API key over a given period."""

	model_config = _AG_MODEL_CONFIG

	key_id: str
	tenant_id: str
	period_days: int = Field(..., ge=1)
	total_calls: NonNegativeInt = 0
	error_calls: NonNegativeInt = 0
	p50_latency_ms: NonNegativeFloat = 0.0
	p95_latency_ms: NonNegativeFloat = 0.0
	p99_latency_ms: NonNegativeFloat = 0.0
	top_endpoints: list[dict[str, Any]] = Field(default_factory=list)
	computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	@computed_field  # type: ignore[misc]
	@property
	def error_rate_pct(self) -> float:
		if self.total_calls == 0:
			return 0.0
		return round(self.error_calls / self.total_calls * 100, 2)


class AgRateLimitStatus(BaseModel):
	"""Current rate limit state for an API key."""

	model_config = _AG_MODEL_CONFIG

	key_id: str
	tenant_id: str
	limit: int = Field(..., ge=0, description="Requests allowed per window")
	remaining: int = Field(..., ge=0, description="Requests remaining in current window")
	reset_at: datetime = Field(..., description="UTC time when the window resets")
	window_seconds: int = Field(default=60, description="Window size in seconds")

	@computed_field  # type: ignore[misc]
	@property
	def used(self) -> int:
		return max(0, self.limit - self.remaining)

	@computed_field  # type: ignore[misc]
	@property
	def pct_used(self) -> float:
		if self.limit == 0:
			return 0.0
		return round(self.used / self.limit * 100, 2)


class AgQuotaStatus(BaseModel):
	"""Current quota state for an API key/period."""

	model_config = _AG_MODEL_CONFIG

	key_id: str
	tenant_id: str
	used: NonNegativeInt = 0
	limit: int = Field(..., ge=0)
	period: str = Field(default="monthly", description="Quota period: daily | weekly | monthly")
	reset_at: datetime = Field(..., description="UTC time the quota period resets")

	@computed_field  # type: ignore[misc]
	@property
	def pct_used(self) -> float:
		if self.limit == 0:
			return 0.0
		return round(self.used / self.limit * 100, 2)

	@computed_field  # type: ignore[misc]
	@property
	def remaining(self) -> int:
		return max(0, self.limit - self.used)


# ---------------------------------------------------------------------------
# In-process storage helpers (replace with Redis / TimescaleDB in production)
# ---------------------------------------------------------------------------

class _AnalyticsStore:
	"""Lightweight in-memory store, keyed by (tenant_id, key_id)."""

	def __init__(self) -> None:
		# {tenant_id: {key_id: [AgApiCallRecord, ...]}}
		self._calls: dict[str, dict[str, list[AgApiCallRecord]]] = {}
		# {tenant_id: {key_id: AgRateLimitStatus}}
		self._rate_limits: dict[str, dict[str, AgRateLimitStatus]] = {}
		# {tenant_id: {key_id: AgQuotaStatus}}
		self._quotas: dict[str, dict[str, AgQuotaStatus]] = {}
		self._lock = asyncio.Lock()

	async def append_call(self, record: AgApiCallRecord) -> None:
		async with self._lock:
			bucket = self._calls.setdefault(record.tenant_id, {})
			bucket.setdefault(record.key_id, []).append(record)

	async def get_calls(
		self,
		tenant_id: str,
		key_id: str | None = None,
		since: datetime | None = None,
	) -> list[AgApiCallRecord]:
		async with self._lock:
			tenant_bucket = self._calls.get(tenant_id, {})
			if key_id is not None:
				records = list(tenant_bucket.get(key_id, []))
			else:
				records = [r for recs in tenant_bucket.values() for r in recs]
			if since is not None:
				records = [r for r in records if r.timestamp >= since]
			return records

	async def set_rate_limit(self, status: AgRateLimitStatus) -> None:
		async with self._lock:
			self._rate_limits.setdefault(status.tenant_id, {})[status.key_id] = status

	async def get_rate_limit(self, tenant_id: str, key_id: str) -> AgRateLimitStatus | None:
		async with self._lock:
			return self._rate_limits.get(tenant_id, {}).get(key_id)

	async def set_quota(self, status: AgQuotaStatus) -> None:
		async with self._lock:
			self._quotas.setdefault(status.tenant_id, {})[status.key_id] = status

	async def get_quota(self, tenant_id: str, key_id: str) -> AgQuotaStatus | None:
		async with self._lock:
			return self._quotas.get(tenant_id, {}).get(key_id)

	async def get_all_key_ids(self, tenant_id: str) -> list[str]:
		async with self._lock:
			return list(self._calls.get(tenant_id, {}).keys())


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ApigAnalyticsService:
	"""
	API Gateway analytics service.

	Responsibilities:
	  - Record individual API call events from gateway middleware.
	  - Aggregate usage summaries by key/period.
	  - Expose rate limit and quota state.
	  - Identify top consumers and error rates.
	  - Fire ntfy alerts when quota approaches a threshold.

	NATS subscription: apg.events.*.api_call.*
	"""

	def __init__(
		self,
		store: _AnalyticsStore | None = None,
		nats_client: Any | None = None,
		ntfy_url: str = "http://localhost:8080",
	) -> None:
		self._store = store or _AnalyticsStore()
		self._nats = nats_client  # nats.aio.client.Client or compatible
		self._ntfy_url = ntfy_url.rstrip("/")

	# ------------------------------------------------------------------
	# Middleware hook — called once per proxied request
	# ------------------------------------------------------------------

	async def record_call(
		self,
		key_id: str,
		endpoint: str,
		method: str,
		status_code: int,
		latency_ms: float,
		tenant_id: str,
	) -> AgApiCallRecord:
		"""Record a single API call and publish to NATS."""
		assert key_id, "key_id required"
		assert endpoint, "endpoint required"
		assert tenant_id, "tenant_id required"

		record = AgApiCallRecord(
			key_id=key_id,
			tenant_id=tenant_id,
			endpoint=endpoint,
			method=method,
			status_code=status_code,
			latency_ms=latency_ms,
		)
		await self._store.append_call(record)

		# Update in-flight rate-limit counters
		await self._decrement_rate_limit(key_id, tenant_id)

		# Update quota counters
		await self._increment_quota(key_id, tenant_id)

		# Publish NATS event (fire-and-forget)
		if self._nats is not None:
			subject = f"apg.events.{tenant_id}.api_call.recorded"
			payload = record.model_dump_json().encode()
			try:
				await self._nats.publish(subject, payload)
			except Exception as exc:  # noqa: BLE001
				_logger.warning("NATS publish failed for api_call.recorded: %s", exc)

		_logger.debug(
			"Recorded call key=%s endpoint=%s status=%d latency=%.1fms",
			key_id, endpoint, status_code, latency_ms,
		)
		return record

	# ------------------------------------------------------------------
	# Usage summary
	# ------------------------------------------------------------------

	async def get_usage_summary(
		self,
		key_id: str,
		period_days: int,
		tenant_id: str,
	) -> AgUsageSummary:
		"""Aggregate usage for a key over the last *period_days* days."""
		assert key_id, "key_id required"
		assert period_days >= 1, "period_days must be >= 1"
		assert tenant_id, "tenant_id required"

		since = datetime.now(timezone.utc) - timedelta(days=period_days)
		records = await self._store.get_calls(tenant_id, key_id, since=since)

		total = len(records)
		errors = sum(1 for r in records if r.is_error)
		latencies = sorted(r.latency_ms for r in records)

		p50 = _percentile(latencies, 50)
		p95 = _percentile(latencies, 95)
		p99 = _percentile(latencies, 99)

		# Top endpoints by call count
		endpoint_counts: dict[str, int] = {}
		for r in records:
			endpoint_counts[r.endpoint] = endpoint_counts.get(r.endpoint, 0) + 1
		top_endpoints = [
			{"endpoint": ep, "calls": cnt}
			for ep, cnt in sorted(endpoint_counts.items(), key=lambda x: -x[1])[:10]
		]

		return AgUsageSummary(
			key_id=key_id,
			tenant_id=tenant_id,
			period_days=period_days,
			total_calls=total,
			error_calls=errors,
			p50_latency_ms=p50,
			p95_latency_ms=p95,
			p99_latency_ms=p99,
			top_endpoints=top_endpoints,
		)

	# ------------------------------------------------------------------
	# Rate limit
	# ------------------------------------------------------------------

	async def get_rate_limit_status(
		self, key_id: str, tenant_id: str
	) -> AgRateLimitStatus:
		"""Return current rate limit state for a key. Creates a default if absent."""
		assert key_id, "key_id required"
		assert tenant_id, "tenant_id required"

		status = await self._store.get_rate_limit(tenant_id, key_id)
		if status is None:
			# Seed with a sensible default (1000 req/min, full remaining)
			status = AgRateLimitStatus(
				key_id=key_id,
				tenant_id=tenant_id,
				limit=1000,
				remaining=1000,
				reset_at=datetime.now(timezone.utc) + timedelta(seconds=60),
				window_seconds=60,
			)
			await self._store.set_rate_limit(status)
		return status

	async def configure_rate_limit(
		self,
		key_id: str,
		tenant_id: str,
		limit: int,
		window_seconds: int = 60,
	) -> AgRateLimitStatus:
		"""Set rate limit policy for a key."""
		assert key_id and tenant_id, "key_id and tenant_id required"
		assert limit >= 0, "limit must be non-negative"

		status = AgRateLimitStatus(
			key_id=key_id,
			tenant_id=tenant_id,
			limit=limit,
			remaining=limit,
			reset_at=datetime.now(timezone.utc) + timedelta(seconds=window_seconds),
			window_seconds=window_seconds,
		)
		await self._store.set_rate_limit(status)
		return status

	# ------------------------------------------------------------------
	# Quota
	# ------------------------------------------------------------------

	async def get_quota_status(self, key_id: str, tenant_id: str) -> AgQuotaStatus:
		"""Return current quota state for a key. Creates a default if absent."""
		assert key_id, "key_id required"
		assert tenant_id, "tenant_id required"

		status = await self._store.get_quota(tenant_id, key_id)
		if status is None:
			# Default: 100k calls/month
			status = AgQuotaStatus(
				key_id=key_id,
				tenant_id=tenant_id,
				used=0,
				limit=100_000,
				period="monthly",
				reset_at=_next_month_start(),
			)
			await self._store.set_quota(status)
		return status

	async def configure_quota(
		self,
		key_id: str,
		tenant_id: str,
		limit: int,
		period: str = "monthly",
	) -> AgQuotaStatus:
		"""Set quota policy for a key."""
		assert key_id and tenant_id, "key_id and tenant_id required"
		assert limit >= 0, "limit must be non-negative"
		assert period in ("daily", "weekly", "monthly"), "period must be daily | weekly | monthly"

		status = AgQuotaStatus(
			key_id=key_id,
			tenant_id=tenant_id,
			used=0,
			limit=limit,
			period=period,
			reset_at=_next_period_start(period),
		)
		await self._store.set_quota(status)
		return status

	# ------------------------------------------------------------------
	# Top consumers
	# ------------------------------------------------------------------

	async def get_top_consumers(
		self, tenant_id: str, limit: int = 10
	) -> list[dict[str, Any]]:
		"""Return the *limit* API keys with the most calls in the last 24 h."""
		assert tenant_id, "tenant_id required"
		assert limit >= 1, "limit must be >= 1"

		since = datetime.now(timezone.utc) - timedelta(hours=24)
		key_ids = await self._store.get_all_key_ids(tenant_id)
		results: list[dict[str, Any]] = []

		for key_id in key_ids:
			records = await self._store.get_calls(tenant_id, key_id, since=since)
			if not records:
				continue
			errors = sum(1 for r in records if r.is_error)
			latencies = sorted(r.latency_ms for r in records)
			results.append({
				"key_id": key_id,
				"total_calls": len(records),
				"error_calls": errors,
				"p95_latency_ms": _percentile(latencies, 95),
			})

		results.sort(key=lambda x: -x["total_calls"])
		return results[:limit]

	# ------------------------------------------------------------------
	# Error rate
	# ------------------------------------------------------------------

	async def get_error_rate(self, tenant_id: str, period_hours: int = 1) -> float:
		"""Return tenant-wide error rate (0.0–1.0) over the last *period_hours*."""
		assert tenant_id, "tenant_id required"
		assert period_hours >= 1, "period_hours must be >= 1"

		since = datetime.now(timezone.utc) - timedelta(hours=period_hours)
		records = await self._store.get_calls(tenant_id, since=since)

		if not records:
			return 0.0
		errors = sum(1 for r in records if r.is_error)
		return round(errors / len(records), 6)

	# ------------------------------------------------------------------
	# Quota threshold alerting
	# ------------------------------------------------------------------

	async def alert_quota_approaching(
		self, tenant_id: str, threshold_pct: float = 80.0
	) -> None:
		"""
		Check all keys in *tenant_id*. For each key whose quota usage exceeds
		*threshold_pct*, publish a quota.threshold_reached NATS event and send
		an ntfy notification.
		"""
		assert tenant_id, "tenant_id required"
		assert 0 < threshold_pct <= 100, "threshold_pct must be in (0, 100]"

		key_ids = await self._store.get_all_key_ids(tenant_id)
		for key_id in key_ids:
			quota = await self.get_quota_status(key_id, tenant_id)
			if quota.pct_used >= threshold_pct:
				await self._emit_quota_alert(quota, threshold_pct)

	async def _emit_quota_alert(
		self, quota: AgQuotaStatus, threshold_pct: float
	) -> None:
		"""Publish NATS event + ntfy push for quota threshold breach."""
		subject = f"apg.events.{quota.tenant_id}.quota.threshold_reached"
		payload_dict = {
			"event": "quota.threshold_reached",
			"key_id": quota.key_id,
			"tenant_id": quota.tenant_id,
			"pct_used": quota.pct_used,
			"used": quota.used,
			"limit": quota.limit,
			"threshold_pct": threshold_pct,
			"reset_at": quota.reset_at.isoformat(),
		}

		if self._nats is not None:
			import json as _json
			try:
				await self._nats.publish(subject, _json.dumps(payload_dict).encode())
			except Exception as exc:  # noqa: BLE001
				_logger.warning("NATS publish failed for quota.threshold_reached: %s", exc)

		# ntfy push notification (best-effort, non-blocking)
		await self._send_ntfy_notification(
			topic=f"apg-quota-{quota.tenant_id}",
			title="Quota threshold reached",
			message=(
				f"Key {quota.key_id} has used {quota.pct_used:.1f}% of quota "
				f"({quota.used}/{quota.limit}, resets {quota.reset_at.date()})"
			),
			tags=["warning", "api"],
		)

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	async def _decrement_rate_limit(self, key_id: str, tenant_id: str) -> None:
		"""Decrement remaining counter; reset window if expired."""
		status = await self._store.get_rate_limit(tenant_id, key_id)
		if status is None:
			return
		now = datetime.now(timezone.utc)
		if now >= status.reset_at:
			# Window expired — reset
			new_status = AgRateLimitStatus(
				key_id=key_id,
				tenant_id=tenant_id,
				limit=status.limit,
				remaining=status.limit - 1,
				reset_at=now + timedelta(seconds=status.window_seconds),
				window_seconds=status.window_seconds,
			)
		else:
			new_remaining = max(0, status.remaining - 1)
			new_status = AgRateLimitStatus(
				key_id=key_id,
				tenant_id=tenant_id,
				limit=status.limit,
				remaining=new_remaining,
				reset_at=status.reset_at,
				window_seconds=status.window_seconds,
			)
			if new_remaining == 0:
				# Publish rate_limit.exceeded event
				await self._publish_rate_limit_exceeded(new_status)
		await self._store.set_rate_limit(new_status)

	async def _increment_quota(self, key_id: str, tenant_id: str) -> None:
		"""Increment used counter for quota tracking."""
		status = await self._store.get_quota(tenant_id, key_id)
		if status is None:
			return
		now = datetime.now(timezone.utc)
		if now >= status.reset_at:
			# Period expired — reset
			new_status = AgQuotaStatus(
				key_id=key_id,
				tenant_id=tenant_id,
				used=1,
				limit=status.limit,
				period=status.period,
				reset_at=_next_period_start(status.period),
			)
		else:
			new_status = AgQuotaStatus(
				key_id=key_id,
				tenant_id=tenant_id,
				used=status.used + 1,
				limit=status.limit,
				period=status.period,
				reset_at=status.reset_at,
			)
		await self._store.set_quota(new_status)

	async def _publish_rate_limit_exceeded(self, status: AgRateLimitStatus) -> None:
		if self._nats is None:
			return
		import json as _json
		subject = f"apg.events.{status.tenant_id}.rate_limit.exceeded"
		payload = _json.dumps({
			"event": "rate_limit.exceeded",
			"key_id": status.key_id,
			"tenant_id": status.tenant_id,
			"limit": status.limit,
			"reset_at": status.reset_at.isoformat(),
		}).encode()
		try:
			await self._nats.publish(subject, payload)
		except Exception as exc:  # noqa: BLE001
			_logger.warning("NATS publish failed for rate_limit.exceeded: %s", exc)

	async def _send_ntfy_notification(
		self,
		topic: str,
		title: str,
		message: str,
		tags: list[str] | None = None,
	) -> None:
		"""Best-effort ntfy push. Silently drops on any error."""
		try:
			import httpx
			url = f"{self._ntfy_url}/{topic}"
			headers = {"Title": title}
			if tags:
				headers["Tags"] = ",".join(tags)
			async with httpx.AsyncClient(timeout=5) as client:
				await client.post(url, content=message.encode(), headers=headers)
		except Exception as exc:  # noqa: BLE001
			_logger.debug("ntfy notification silently dropped: %s", exc)


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[float], pct: int) -> float:
	"""Return the *pct*-th percentile from a pre-sorted list."""
	if not sorted_values:
		return 0.0
	idx = max(0, int(len(sorted_values) * pct / 100) - 1)
	return round(sorted_values[idx], 3)


def _next_month_start() -> datetime:
	now = datetime.now(timezone.utc)
	if now.month == 12:
		return now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
	return now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)


def _next_period_start(period: str) -> datetime:
	now = datetime.now(timezone.utc)
	if period == "daily":
		return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
	if period == "weekly":
		days_ahead = 7 - now.weekday()
		return (now + timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0, microsecond=0)
	return _next_month_start()
