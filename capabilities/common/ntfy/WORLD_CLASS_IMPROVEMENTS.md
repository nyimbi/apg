# World-Class Improvements: APG Notifications (ntfy)

Copyright © 2025 Datacraft — Author: Nyimbi Odero

---

## 1. Idempotent Delivery with Distributed Deduplication

**Current gap**: `send_notification` generates a new UUID on every call with no guard against duplicate invocations.

**Improvement**: Accept a caller-supplied `idempotency_key: str`. Hash the key with SHA-256, store it in a bounded `_idempotency_cache` (LRU, TTL-backed). On cache hit, return the original delivery record without re-sending. This is especially critical for retry storms from upstream orchestrators (Bytewax, task queues).

```python
async def send_notification(self, ..., idempotency_key: str | None = None) -> Dict[str, Any]: ...
```

---

## 2. Structured Dead-Letter Queue with Re-Drive

**Current gap**: Failed notifications are stored with `status="failed"` but there is no systematic re-drive or escalation path.

**Improvement**: Introduce `async def requeue_dead_letters(self, max_age_hours: int = 24, limit: int = 100)` that filters `_notifications` for failed records within `max_age_hours`, re-enqueues them respecting per-channel rate limits, and records a `dlq_requeued` audit event. Include per-channel DLQ depth in `health_check()`.

---

## 3. Adaptive Rate Limiter per Channel

**Current gap**: `max_concurrent_deliveries` is a global semaphore with no per-channel differentiation. Email/SMS/Push have vastly different provider rate limits.

**Improvement**: Replace the single semaphore with a `ChannelRateLimiter` that maintains per-channel token buckets (`tokens_per_second`, `burst_capacity` keyed by channel name). Methods that call providers acquire the appropriate bucket before dispatching. Configuration exposed through `NotificationServiceConfig.channel_rate_limits: dict[str, RateLimitConfig]`.

---

## 4. Webhook Fan-out with HMAC Signature Verification

**Current gap**: Webhook channel config stores a `url` but no signing secret; outgoing payloads are unsigned.

**Improvement**: Add `signing_secret` to webhook channel config. On delivery, compute `X-Signature-SHA256: hmac_sha256(signing_secret, body)` header. Add `async def verify_webhook_signature(self, channel_id: str, payload: bytes, signature: str) -> bool` for inbound verification. Expose the signing key via `get_channel_signing_key(channel_id)` with tenant-scoped access control.

---

## 5. Priority Queue with Preemption

**Current gap**: `priority` is stored on the notification record but `send_bulk` processes requests in FIFO order regardless of priority.

**Improvement**: Replace the flat gather loop in `send_bulk` with a `heapq`-backed priority queue. `CRITICAL > URGENT > HIGH > NORMAL > LOW`. Implement preemption by reserving a fraction of the concurrency semaphore (e.g., top 20%) exclusively for `CRITICAL` and `URGENT` messages. Expose `get_queue_depth_by_priority()` in the dashboard.

---

## 6. Template A/B Testing Engine

**Current gap**: Templates have versioning but no mechanism to split traffic or measure winner performance.

**Improvement**: Add `async def create_ab_test(self, template_id: str, variant_ids: list[str], split: list[float], success_metric: str, min_sample_size: int)` that creates a test record. `render_template` probabilistically routes to the active variant. Add `async def evaluate_ab_test(self, test_id: str) -> dict` that runs a binomial z-test and declares a winner when `min_sample_size` is reached. Auto-promote the winner to `current_version`.

---

## 7. Quiet-Hour Enforcement with Timezone Awareness

**Current gap**: `timezone_aware_send` records the recipient timezone but does not gate delivery on quiet hours.

**Improvement**: Add `quiet_hours: tuple[int, int] | None` to user preferences (e.g., `(22, 7)` for 10 PM–7 AM). In `send_notification`, before dispatch check `_is_quiet_hour(recipient_id)`. If true: schedule the notification for the next quiet-hour-end instead of sending immediately, and set `status="deferred_quiet_hour"`. Include deferred-count in the dashboard summary.

---

## 8. Consent & Regulatory Compliance Gate

**Current gap**: `check_preference` handles opt-in/out but there is no GDPR/CCPA consent record with timestamp, legal basis, and evidence chain.

**Improvement**: Add `async def record_consent(self, recipient_id: str, channel: str, legal_basis: str, evidence_ref: str)` and `async def revoke_consent(self, recipient_id: str, channel: str)`. Gate all outbound sends through `_has_valid_consent(recipient_id, channel)`. Store consent records immutably with UUID7 IDs and ISO timestamps. Include consent coverage rate in `dashboard_summary`.

---

## 9. End-to-End Delivery Latency Percentiles

**Current gap**: `_delivery_stats["average_latency_ms"]` uses a running mean — hides tail latency. A slow provider at p99 is invisible.

**Improvement**: Replace the scalar average with a circular buffer of the last `N=1000` latency samples per channel. Expose `p50`, `p95`, `p99` percentile methods using `statistics.quantiles`. Include these in `health_check()` and `dashboard_summary()`. Alert via audit log when `p99 > threshold_ms`.

---

## 10. Notification Digest / Batching for Fatigue Prevention

**Current gap**: Every call sends immediately with no awareness of notification frequency per recipient.

**Improvement**: Add `async def send_digested(self, recipient_id: str, template_id: str, variables: dict, digest_window_minutes: int = 60)`. Within the window, accumulate notifications for the same recipient into a single `_digest_buffer`. After the window expires, render a single digest notification using a digest template. Track `digest_ratio` (notifications collapsed vs. sent) in analytics.

---

## 11. Structured Audit Log with Immutable Append-Only Records

**Current gap**: `_audit_log` is a plain list of dicts — mutable, non-persistent, lost on process restart.

**Improvement**: Replace with an `AuditSink` abstraction with two implementations: `InMemoryAuditSink` (current behavior) and `PostgresAuditSink` (writes to `nt_audit_events` table with UUID7 PK, tenant isolation, and `jsonb` payload). Make `_log_audit_event()` a private method that writes to the injected sink. Add `async def get_audit_trail(self, resource_id: str, resource_type: str, limit: int = 100)` for compliance queries.

---

## 12. Provider Health Circuit Breaker

**Current gap**: Channel health is a static flag set by `channel_health_check()`. There is no automatic circuit breaking on consecutive failures.

**Improvement**: Implement a per-channel `CircuitBreaker` with three states: CLOSED (normal), OPEN (skip delivery, use fallback), HALF-OPEN (probe). Transition CLOSED→OPEN after `failure_threshold` consecutive failures within `failure_window_seconds`. Transition OPEN→HALF-OPEN after `recovery_timeout_seconds`. Expose circuit state in `health_check()` and emit `circuit_opened` / `circuit_closed` audit events.

---

## 13. Multi-Tenancy Isolation via Row-Level Security

**Current gap**: All in-memory stores (`_notifications`, `_templates`, etc.) are filtered manually by `tenant_id` in list operations — a missing filter leaks data across tenants.

**Improvement**: Introduce a `TenantBoundStore[V]` wrapper that enforces `tenant_id` on every read and write. All `_notifications`, `_templates`, `_channels`, `_schedules`, `_suppressions` become `TenantBoundStore` instances. The `tenant_id` context is set once at construction and cannot be overridden per-call. This eliminates the class of bugs where a list/delete method forgets the tenant filter.

---

## 14. Streaming Delivery Events via AsyncGenerator

**Current gap**: Callers must poll `track_delivery()` to observe status changes. There is no push-based subscription API.

**Improvement**: Add `async def subscribe_delivery_events(self, notification_id: str) -> AsyncGenerator[Dict[str, Any], None]` that yields delivery event dicts as they occur (opened, clicked, bounced, delivered). Internally backed by `asyncio.Queue` per notification. Pairs naturally with WebSocket endpoints in the Flask-AppBuilder blueprint. Include `unsubscribe()` context-manager semantics via `@asynccontextmanager`.

---

## 15. Predictive Send-Time Optimization

**Current gap**: Notifications are dispatched at call time or at a caller-specified `scheduled_at`. There is no learning from historical engagement to suggest optimal times.

**Improvement**: Add `async def predict_optimal_send_time(self, recipient_id: str, channel: str) -> datetime` that examines the recipient's engagement history (`opened_at` timestamps in `_notifications`). Compute a histogram of open-hour-of-day for the last 90 days. Return the hour bucket with the highest open rate, projected to the next occurrence. Fall back to the system default if insufficient data (`< 5 opens`). Expose this signal in `engagement_report` as `suggested_send_hour`.
