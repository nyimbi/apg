# MQEB World-Class Improvements

**Capability**: Message Queue Event Bus (`mqeb`)
**Author**: Nyimbi Odero
**Copyright**: © 2025 Datacraft

This document catalogues 15 targeted improvements that would elevate MQEB from
a well-structured governance layer to a production-grade, observable, and
composable event fabric.

---

## 1. Backpressure-Aware Flow Control

**Problem**: `publish_message` accepts messages without any admission control.
Under a burst load, `message_queues` grows unboundedly, causing OOM.

**Improvement**: Add per-topic `max_queue_depth` configuration and an async
`publish_with_backpressure` method that either blocks (await), drops with a
counter, or redirects to an overflow topic. Exposes a `queue_pressure_ratio`
metric per topic so upstream producers can shed load cooperatively.

---

## 2. Cursor-Based Consumer Groups

**Problem**: `consume_messages` uses a naive list slice and does not track
consumer position. Restarting a consumer re-reads or skips messages depending
on timing.

**Improvement**: Introduce `ConsumerGroupRecord` with a committed offset
pointer per (subscription, partition) tuple. Add `commit_offset` and
`seek_offset` methods. This enables exactly-once consumption semantics without
an external Kafka-style log, while remaining Bytewax-composable.

---

## 3. Schema Registry Integration

**Problem**: `schema_ref` is stored as a free-form string. There is no
validation that the message payload actually conforms to the referenced schema.

**Improvement**: Add a `SchemaRegistryAdapter` protocol and an
`async validate_message_schema(message_id, schema_ref)` service method that
calls the registry, caches compiled validators in `BoundedCache`, and annotates
the message record with `schema_validated: bool` and `schema_version`. Fail
closed on `regulated` topics.

---

## 4. Distributed Saga / Compensating Transactions

**Problem**: Multi-step message flows have no native rollback. If step 3 of a
5-step saga fails, there is no mechanism to emit compensating events for steps
1–2.

**Improvement**: Add `SagaRecord` with an ordered list of `SagaStep` entries,
each referencing a topic and compensation topic. `start_saga`, `advance_saga`,
and `compensate_saga` methods orchestrate the flow. Saga state is persisted in
`MqebService` and emitted as audit events, giving governance consoles full
visibility.

---

## 5. Time-Based Message Scheduling

**Problem**: Messages are published immediately. Use-cases like scheduled
reminders, delayed retries, and time-windowed rate limiting have no native
support.

**Improvement**: Add `scheduled_at: str` to `MessageRecord` and a
`schedule_message` method that queues messages into `scheduled_messages` keyed
by delivery timestamp. The background `_message_processing_loop` drains the
schedule bucket into `message_queues` at the correct time. Expose
`list_scheduled_messages` and `cancel_scheduled_message`.

---

## 6. Idempotency Store with TTL

**Problem**: `idempotency_key` is stored on `MessageRecord` but is never
deduplicated. Two publishes with the same key both succeed.

**Improvement**: Maintain an `idempotency_cache: dict[str, str]` mapping
`(tenant_id, idempotency_key)` → `message_id`. On publish, check the cache
first and return the existing message dict. Evict entries after the topic
`retention_days` window via a TTL-aware `BoundedCache` or a periodic cleanup
pass, preventing unbounded growth.

---

## 7. Structured Retry Policies with Exponential Backoff

**Problem**: `DeliveryAttemptRecord` stores `retry_count` but the retry
decision logic lives outside MQEB. There are no configurable policies for
backoff, max attempts, or jitter.

**Improvement**: Add `RetryPolicyRecord` with `strategy` (linear, exponential,
constant), `base_delay_ms`, `max_attempts`, `jitter_factor`, and
`dead_letter_on_exhaust`. `compute_next_retry_delay(attempt_id)` returns the
next delivery timestamp. `apply_retry_policy(attempt_id)` updates the attempt
record and, when exhausted, auto-routes to the dead-letter topic.

---

## 8. Priority Queue with Preemptive Scheduling

**Problem**: All messages enter a single FIFO queue per topic regardless of
their `priority` field. CRITICAL messages wait behind LOW messages published
earlier.

**Improvement**: Replace `message_queues: dict[str, list[str]]` with
`message_queues: dict[str, dict[str, list[str]]]` keyed by priority tier.
`_process_pending_deliveries` drains CRITICAL → HIGH → NORMAL → LOW with
configurable weight ratios. Add `get_priority_queue_stats` to expose per-tier
depths.

---

## 9. Event Sourcing Append-Only Log

**Problem**: Audit events are stored in a mutable dict. Records can be silently
overwritten, undermining forensic integrity.

**Improvement**: Replace `audit_events: dict` with an append-only
`audit_log: list[MqebAuditEventRecord]` and a secondary index dict for fast
lookup. Add `stream_audit_events(tenant_id, since_id)` as an `AsyncGenerator`
so governance consoles can tail the log without polling the full list. Sign
each record with HMAC-SHA256 using a tenant secret to detect tampering.

---

## 10. Circuit Breaker for Subscription Delivery

**Problem**: A failing downstream endpoint (webhook, gRPC, WebSocket) is
retried indefinitely, consuming resources and delaying delivery to healthy
subscribers.

**Improvement**: Add `CircuitBreakerRecord` per subscription with `state`
(closed, open, half-open), failure threshold, and reset timeout.
`_deliver_messages_to_subscription` checks the breaker state before
attempting delivery and records trips to audit events. Expose
`reset_subscription_circuit_breaker(subscription_id)` for operator intervention.

---

## 11. Content-Based Message Routing with DSL

**Problem**: `_subscription_matches_topic` uses `fnmatch` on topic names only.
Routing on message headers or payload fields requires a custom consumer-side
filter.

**Improvement**: Add a `RoutingRule` dataclass with a `match_expr` field
containing a safe subset DSL (e.g., `priority == "high" AND tenant_id ==
"tenant-a"`). `evaluate_routing_rule(message, rule)` uses a restricted `ast`
evaluator (no builtins, no attribute chains). Rules are stored per subscription
and evaluated before queue insertion.

---

## 12. Tenant Quotas and Rate Limiting

**Problem**: Any tenant can publish unlimited messages at unlimited throughput.
There is no mechanism to prevent a noisy tenant from starving others.

**Improvement**: Add `TenantQuotaRecord` with `max_messages_per_minute`,
`max_bytes_per_minute`, and `max_topics`. A sliding-window counter (token
bucket via `BoundedCache`) is decremented on each `publish_message`. Exceeded
quota raises `PermissionError("quota_exceeded")` and emits an audit event.
`get_tenant_quota_status(tenant_id)` returns current utilization ratios.

---

## 13. Message Batching and Compression

**Problem**: Each call to `publish_message` processes one message. High-volume
producers incur per-message overhead for policy evaluation, audit logging, and
routing.

**Improvement**: Add `publish_message_batch(tenant_id, messages: list[...])`.
Policy evaluation runs once per batch for shared context fields (same
topic/tenant/producer). Payload bytes are optionally compressed with zstd before
storage, with `compression_type` and `compressed_size_bytes` on
`MessageRecord`. Batch publish emits a single audit event with `batch_size`.

---

## 14. Dead-Letter Queue Processing Workflows

**Problem**: `_process_dead_letter_queues` is a no-op stub. Messages that land
in the DLQ are never inspected, redriven, or discarded.

**Improvement**: Implement a full DLQ lifecycle: `inspect_dead_letter_queue`,
`redrive_dead_letter_messages` (re-publishes to originating topic after an
operator review), `purge_dead_letter_queue`, and `export_dead_letter_messages`.
Each action requires reviewer evidence and emits an audit event. A configurable
`max_redrive_attempts` prevents infinite redrive loops.

---

## 15. OpenTelemetry Trace Propagation

**Problem**: Messages crossing service boundaries lose distributed trace context.
Correlation between a published message and its downstream processing spans is
impossible without manual instrumentation.

**Improvement**: Add `trace_context: dict[str, str]` to `MessageRecord` for
W3C Trace Context headers (`traceparent`, `tracestate`). `publish_message`
accepts an optional `trace_context` parameter and propagates it through routing.
`consume_messages` returns messages with their trace context intact so consumers
can start child spans. When the `opentelemetry-api` package is available,
auto-inject/extract using the standard propagator API; fall back to a passthrough
dict otherwise.
