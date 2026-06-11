# NATS JetStream — World-Class Improvements

© 2025 Datacraft | nyimbi@gmail.com | www.datacraft.co.ke

## 1. Key-Value Store API

JetStream exposes a native KV bucket abstraction built on streams. Adding `kv_put`, `kv_get`, `kv_delete`, and `kv_watch` methods elevates the capability from pure event bus to a lightweight distributed config/state store — eliminating Redis as a secondary dependency for simple shared state. Each bucket is a stream under the hood, so it inherits replication, persistence, and exactly-once semantics automatically.

## 2. Object Store API

NATS 2.10 ships a native blob/object store backed by JetStream. `object_put` and `object_get` allow capabilities to pass large payloads (PDFs, images, model weights) by reference rather than inline in events — keeping event envelopes small and audit-friendly while avoiding external S3 overhead in local deployments.

## 3. Stream Mirror / Source Federation

Production multi-region deployments need data locality. Adding `create_mirror` and `add_source` methods wraps JetStream's mirror/source config so capabilities can fan data from a hub stream to regional replicas, or aggregate edge streams into a central analytics stream, without manual stream-config juggling.

## 4. Exactly-Once Publish with Idempotency Token

The current adapter generates `Msg-Id` from fields that can collide when two events share the same resource and timestamp. Replacing with a stable UUID7-based `Msg-Id` derived from payload hash + event UUID eliminates silent duplicates. Expose `publish_exactly_once(idempotency_key, ...)` as a first-class method so callers can supply a deterministic key (e.g. invoice number + state) for business-level deduplication.

## 5. Pull Consumer with Batch Fetch and Backpressure

`fetch_messages` currently returns an empty list. A real implementation using `consumer.fetch(batch, expires)` gives capabilities a pull-loop primitive — critical for worker queues, bulk ingestion, and backpressure-aware processing. Adding `fetch_with_ack_all` that atomically acks the whole batch on success (or nacks individually on partial failure) removes boilerplate from every consumer.

## 6. Push Consumer with Flow Control

Long-running services benefit from push subscriptions with JetStream flow control (`flow_control=True`, `idle_heartbeat`). Adding `subscribe_push_with_flow_control` prevents head-of-line blocking when a slow consumer causes the server to buffer indefinitely. Heartbeat callbacks feed the health-check subsystem for real-time consumer health monitoring.

## 7. Message Replay from Sequence or Timestamp

`replay_events` is a no-op stub. Implementing it via a per-consumer `DeliverPolicy.by_start_sequence` or `DeliverPolicy.by_start_time` ephemeral consumer enables event sourcing patterns: capabilities can replay their own stream to rebuild state after a crash, or replay a time window for debugging. The `replay_from_time(capability_id, since: datetime)` variant is essential for audit and compliance.

## 8. Dead-Letter Stream (DLQ)

Messages that exceed `max_deliver` silently vanish. A companion stream `APG_DLQ` with a `$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.>` subject captures them. Adding `get_dlq_messages` and `requeue_dlq_message` gives operators a UI-friendly way to inspect and replay poison messages without reaching for the NATS CLI.

## 9. Subject-Scoped Authorization Tokens

Multi-tenant deployments need subjects scoped to `apg.events.{tenant_id}.{capability_id}.{event_type}`. Injecting `tenant_id` into the subject hierarchy (already carried in payloads) allows NATS account-level authorization rules to enforce tenant isolation at the transport layer — not just in application code. `publish_tenant_scoped` and corresponding subject helpers make this the blessed pattern.

## 10. Metrics Exporter for Prometheus / OpenTelemetry

`get_throughput_metrics` and `get_latency_metrics` return zeroes. Wrapping NATS server's `varz`/`jsz` HTTP monitoring endpoints and exposing them as structured Pydantic models enables direct Prometheus scraping or OpenTelemetry trace/span emission. A `get_jetstream_stats()` method that returns typed `JetStreamStats` (messages, bytes, consumer counts, ack-pending totals) feeds the APG observability stack without external sidecars.

## 11. Circuit Breaker and Fallback Queue

When NATS is unreachable the adapter silently drops events after 3 retries. Adding a thread-safe in-process `BoundedDeque` fallback queue (max 10 000 messages) and a background reconnect-and-drain coroutine ensures zero event loss during brief outages. The circuit breaker state (`open` / `half-open` / `closed`) is exposed via `health_check` so the monitoring dashboard shows degraded vs. failed status.

## 12. Stream Snapshot and Restore

Operational disaster recovery requires portable stream backups. Adding `snapshot_stream(stream_name) -> AsyncIterator[bytes]` and `restore_stream(stream_name, chunks: AsyncIterable[bytes])` wraps the NATS server's snapshot API (`$JS.API.STREAM.SNAPSHOT`). Snapshots are chunked binary blobs that can be streamed directly to/from S3 or a local file — no external tooling required.

## 13. Header-Driven Event Routing

NATS 2.2+ message headers allow server-side content-based routing via `HeaderSubscriber` patterns. `publish_with_routing_headers` should set standard headers (`X-APG-Tenant`, `X-APG-Capability`, `X-APG-Event-Type`, `X-APG-Correlation-Id`) and expose a `subscribe_by_header_filter` that creates a consumer with `FilterSubjects` lists — enabling fan-out to multiple capabilities without subject-per-handler proliferation.

## 14. Ordered Consumer for Event Sourcing

JetStream ordered consumers (`ordered_consumer=True`) guarantee message delivery in sequence-number order with automatic re-subscription on failure. Adding `subscribe_ordered(stream_name, subject_filter)` enables reliable event-sourcing read models: capabilities reconstruct aggregate state by consuming the ordered stream from sequence 0 without managing consumer offsets themselves. This replaces fragile cursor-in-database patterns.

## 15. Multi-Server Cluster and Leaf Node Configuration

The connector hard-codes a single `nats_url`. Supporting `nats_urls: list[str]` (cluster connect) and `nats_credentials_file: str` (NKey/JWT auth for leaf nodes) unlocks production-grade deployments: geo-distributed leaf nodes authenticate to a hub cluster, and the client transparently fails over across server addresses. `connect_cluster(urls, credentials_file)` wraps `nats.connect(servers=urls, credentials=credentials_file)` and re-exposes all existing methods unchanged.
