# World-Class Improvements: Composition Events Capability

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>

---

## 1. Transactional Outbox Pattern

**Category**: Reliability / Data Consistency

**Justification**: The current publish path writes to DB then Bytewax in two separate steps. A crash between them yields a committed DB record with no corresponding stream event — a silent data-loss scenario. The transactional outbox eliminates this by writing event records to an `outbox` table in the same DB transaction as the domain write, then a relay process atomically moves them to Bytewax. This is exactly how Debezium/Confluent solve the dual-write problem.

**Implementation**: Add `ESOutboxEntry` model. `EventPublishingService.publish_event` writes to outbox inside the domain transaction. A dedicated `OutboxRelayService` polls new outbox rows, publishes to Bytewax, marks rows as relayed. Use advisory locks or `SELECT FOR UPDATE SKIP LOCKED` for concurrent relay instances.

**Competitor Reference**: Debezium (CDC-based outbox), Eventuate Tram, Axon Framework outbox interceptor.

---

## 2. Event Deduplication with Idempotency Keys

**Category**: Reliability / Exactly-Once Semantics

**Justification**: Producers retry on network errors. Without idempotency enforcement, retried publishes land as duplicate events in the stream, causing over-counting, double-charges, or duplicate side-effects downstream. Kafka Exactly-Once Semantics solves this at broker level; the current Bytewax ledger has no such guarantee.

**Implementation**: Accept `idempotency_key` on `publish_event`. Store `(tenant_id, idempotency_key)` in a Redis SET with TTL matching the stream retention. On publish, check Redis first; if key present, return the original `event_id` without re-publishing. Use Lua script for atomic check-and-set.

**Competitor Reference**: Stripe API idempotency keys, AWS EventBridge deduplication, Kafka producer `enable.idempotence`.

---

## 3. Priority Queuing with Back-Pressure

**Category**: Performance / Fairness

**Justification**: The current batch publisher treats all events equally. In a mixed workload (critical payment events + low-priority analytics pings), low-priority flood can starve high-priority events. Priority lanes with back-pressure propagation ensure SLA-critical events are never queued behind bulk loads.

**Implementation**: Maintain three in-process `asyncio.PriorityQueue` instances (CRITICAL, HIGH, NORMAL). `publish_event` enqueues by `event_config.priority`. A dispatcher coroutine drains in priority order. Back-pressure: if the NORMAL queue depth exceeds `max_normal_queue_depth`, reject new NORMAL events with HTTP 429. Metrics expose per-priority queue depths.

**Competitor Reference**: RabbitMQ priority queues, ActiveMQ priority lanes, AWS SQS FIFO with message group IDs.

---

## 4. Schema Evolution with Compatibility Matrix

**Category**: Governance / Reliability

**Justification**: The current `_check_schema_compatibility` returns `True` unconditionally. Without real backward/forward/full compatibility checks, a producer can register a schema that breaks all existing consumers silently. The Confluent Schema Registry enforces Avro/JSON schema evolution rules — the same rigor is needed here.

**Implementation**: In `SchemaRegistryService.register_enhanced_schema`, fetch the latest schema version for the same `event_type`. Compute the diff using `jsonschema` vocabulary: removing required fields is backward-compatible; adding required fields is backward-breaking. Store compatibility decision in `ESSchema.compatibility_matrix`. Expose `/schemas/{id}/compatibility` endpoint.

**Competitor Reference**: Confluent Schema Registry compatibility modes, AWS Glue Schema Registry, Apicurio Registry.

---

## 5. Event Replay with Approval Gate and Rate Limiting

**Category**: Compliance / Operations

**Justification**: Uncontrolled replays can flood downstream consumers and corrupt read models. The business rule `replay_requires_approval` is declared but `replay_aggregate` in the service does not enforce it. An approval gate backed by a rate-limited replay queue prevents runaway replays and provides an audit trail.

**Implementation**: Add `ReplayRequest` model with `requested_by`, `approved_by`, `target_stream_id`, `from_offset`, `to_offset`, `rate_limit_eps` (events per second). `request_replay` creates a pending request. `approve_replay` sets `approved_by`. `execute_replay` checks approval, then emits events at `rate_limit_eps` using `asyncio.sleep` pacing. Emits `events_replayed` lifecycle event.

**Competitor Reference**: Axon Server event replay with token store, EventStoreDB scavenging, Kafka MirrorMaker throttled replication.

---

## 6. Consumer Lag SLO Alerting

**Category**: Observability / SRE

**Justification**: `_get_consumer_lag` returns `0` unconditionally. In production, unbounded consumer lag is the primary symptom of a failing consumer group, processing backlog, or misconfigured batch size. Without lag-based SLO alerting, operators discover problems only after downstream systems show stale data.

**Implementation**: `_calculate_consumer_lag` computes `latest_offset - consumer_committed_offset` per partition from the Bytewax ledger. Add `ConsumerLagMonitor` that runs every 30s, compares lag against `ESSubscription.lag_slo_threshold`. When threshold exceeded, emit an alert via `ntfy` capability and a `consumer_lag_slo_breach` lifecycle event.

**Competitor Reference**: Burrow (LinkedIn), Kafdrop, Datadog Kafka consumer lag monitor.

---

## 7. Dead-Letter Reprocessing with Exponential Backoff

**Category**: Reliability / Operations

**Justification**: `_send_to_dead_letter` logs a warning and does nothing — dead letters are silently dropped. A real dead-letter queue captures the failed event, error, retry count, and next retry time. Exponential backoff prevents a systematic failure from hammering a recovering downstream service.

**Implementation**: Add `ESDeadLetterEntry` model: `original_event_id`, `subscription_id`, `error_message`, `retry_count`, `next_retry_at`, `max_retries`. `DeadLetterService.requeue` computes `next_retry_at = now + base_delay * 2^retry_count`. A scheduler polls for entries where `next_retry_at <= now` and re-delivers. After `max_retries`, mark as `permanently_failed` and notify.

**Competitor Reference**: AWS SQS DLQ with redrive policy, Azure Service Bus dead-letter, RabbitMQ DLX.

---

## 8. Cryptographic Event Signing

**Category**: Security / Compliance

**Justification**: Nothing prevents a rogue internal service from injecting fabricated events into the stream. In regulated environments (financial audit trails, healthcare data changes), consumers must be able to verify that an event was genuinely produced by the declared `source_capability` and was not tampered with in the ledger.

**Implementation**: Each `source_capability` has an ECDSA-P256 key pair stored in the tenant's secrets store. `publish_event` computes `ECDSA.sign(SHA256(canonical_event_bytes), private_key)` and stores the base64 signature in `event_metadata["signature"]`. `verify_event_signature` reconstructs canonical bytes and verifies. Key rotation is handled via `kid` header in `event_metadata`.

**Competitor Reference**: CloudEvents `datacontentencoding` + signature extensions, Kafka message-level signing via interceptors, AWS EventBridge signed events.

---

## 9. Multi-Tenant Stream Isolation with Namespace Partitioning

**Category**: Security / Multi-Tenancy

**Justification**: Bytewax stream names currently include `tenant_id` in the string (`apg.{tenant_id}.{stream_key}`) but there is no enforcement that prevents one tenant's consumer from reading another tenant's stream if the stream ID is guessed. Namespace partitioning with ACL enforcement at the stream layer closes this.

**Implementation**: `StreamManagementService` prefixes all Bytewax stream names with a deterministic hash of `(tenant_id, salt)` rather than the plaintext tenant ID. Add `TenantStreamACL` model. All stream read/write operations verify the caller's `tenant_id` against the ACL before any Bytewax operation. Rejected attempts emit `unauthorized_stream_access` lifecycle events.

**Competitor Reference**: Kafka multi-tenant via ACLs + topic naming conventions, Pulsar namespace isolation, Azure Event Hubs namespaces.

---

## 10. Exactly-Once Stream Processing via Checkpoint Fencing

**Category**: Correctness / Stream Processing

**Justification**: Stream processors (`_run_aggregation_processor`, `_run_windowing_processor`) can produce duplicate output events if the process crashes after emitting results but before committing the input offset. This is the classic "at-least-once with duplicates in output" problem.

**Implementation**: Introduce `ProcessorCheckpoint` model: `processor_id`, `input_stream`, `committed_offset`, `output_stream`, `output_sequence`. Processor emits output and checkpoints atomically using a two-phase write: (1) write output to Bytewax, (2) update checkpoint in DB in the same transaction. On restart, resume from last checkpoint offset. Output idempotency: include `checkpoint_id` in output event so consumers can deduplicate.

**Competitor Reference**: Flink's distributed checkpointing, Kafka Streams state stores with changelog topics, Bytewax state recovery via S3 snapshots.

---

## 11. Stream Topology Visualization API

**Category**: Developer Experience / Observability

**Justification**: As the number of streams, processors, and subscriptions grows, understanding data flow becomes critical for debugging and capacity planning. No existing endpoint exposes the full stream topology as a graph. Teams waste hours tracing event flows manually.

**Implementation**: `get_stream_topology(tenant_id)` returns a directed graph: nodes are streams, processors, subscriptions, and consumer groups; edges represent data flow with labels showing throughput and lag. Serialized as `{"nodes": [...], "edges": [...]}` for consumption by the `stream_topology.html` template's D3 visualization. Cache topology in Redis with 60s TTL.

**Competitor Reference**: Confluent Control Center topology view, Redpanda Console, AWS MSK Replicator topology graph.

---

## 12. Adaptive Batch Sizing Based on Consumer Throughput

**Category**: Performance / Self-Tuning

**Justification**: Fixed `batch_size` in `ESSubscription` is set at creation and never adjusted. Under variable load, a consumer processing 10K msg/s can handle large batches efficiently, while the same consumer under memory pressure needs smaller batches to avoid OOM. Adaptive sizing maximizes throughput without manual tuning.

**Implementation**: `AdaptiveBatchController` tracks `processing_time_ms` per batch. Uses AIMD (Additive Increase Multiplicative Decrease): if `processing_time_ms < target_latency_ms`, increase `batch_size += step`; if `processing_time_ms > target_latency_ms * 1.5`, `batch_size = max(min_batch, batch_size // 2)`. Update `ESSubscription.batch_size` in place. Expose `batch_size_history` in subscription metrics.

**Competitor Reference**: Kafka consumer `fetch.min.bytes` + `fetch.max.wait.ms` auto-tuning, Pulsar adaptive flow control, Flink dynamic scaling.

---

## 13. Event Time vs Processing Time Windowing

**Category**: Stream Processing / Correctness

**Justification**: Current windowing (`_emit_window_results`) uses `datetime.now()` for window boundaries, conflating event time and processing time. Out-of-order events (network delays, mobile clients with clock skew) are silently placed in the wrong window, producing incorrect aggregations.

**Implementation**: Extract event time from `event_data["timestamp"]` (producer clock). Add configurable `allowed_lateness_ms` watermark. Events arriving after watermark are routed to a "late events" side output stream. Window boundaries are computed from event time. `WatermarkTracker` maintains the high-watermark per partition and advances it monotonically.

**Competitor Reference**: Apache Flink event time processing with watermarks, Google Dataflow windowing model, Kafka Streams `TimestampExtractor`.

---

## 14. Streaming Metrics Pipeline with RED Method

**Category**: Observability / SRE

**Justification**: `ESMetrics` records individual metric observations but there is no aggregation pipeline. Operators cannot see Rate, Errors, Duration (RED) metrics per stream in real time. Prometheus-compatible metric export is the de-facto standard for cloud-native observability.

**Implementation**: `MetricsPipelineService` subscribes to the `apg.composition.events.lifecycle` stream and aggregates: `events_published_total` (counter), `event_publish_duration_seconds` (histogram), `consumer_lag_messages` (gauge), `dead_letter_events_total` (counter). Expose `/metrics` endpoint in Prometheus text format. Push snapshots to `ESMetrics` every 15s.

**Competitor Reference**: Kafka JMX + Prometheus JMX exporter, Redpanda `/public_metrics` endpoint, Pulsar built-in Prometheus integration.

---

## 15. Content-Based Event Routing with Filter Expressions

**Category**: Architecture / Composability

**Justification**: Subscription filters currently support only exact field matches and event type pattern matching. Complex routing scenarios (e.g., "route payment events where `amount > 10000` and `currency == 'USD'` to the fraud detection stream") require expression evaluation that the current `_matches_subscription_filters` cannot handle.

**Implementation**: Integrate `jsonpath-ng` or a simple expression evaluator. `SubscriptionConfig.filter_expression` accepts a DSL string: `"payload.amount > 10000 AND payload.currency == 'USD'"`. `FilterExpressionCompiler.compile(expr)` returns a callable `Callable[[Dict], bool]`. Compiled filters are cached in `BoundedCache` keyed by expression hash. Evaluation is synchronous in the hot path.

**Competitor Reference**: AWS EventBridge content-based filtering, Azure Event Grid advanced filters, Kafka Streams `filter()` DSL.
