# Event Streaming Bus

© 2025 Datacraft · Author: Nyimbi Odero

## Overview

The Event Streaming Bus is the foundational messaging backbone for the APG composition layer. It provides Bytewax-powered event streams with schema validation, producer attribution, consumer group management, stateful stream processors, dead-letter handling, and approved event replay. Every other composition capability routes its lifecycle events through this bus.

Business value: a durable, auditable communication fabric that decouples producers from consumers across capability boundaries. Schema compatibility enforcement prevents breaking changes from silently corrupting downstream processors. Dead-letter streams and bounded retry policies ensure no event is silently lost. The replay approval gate prevents unauthorized state reconstruction from historical data.

## Capability ID

`composition_events`  Version: see `package_manifest.json`

## Features

- Bytewax-backed named streams with retention and partition policies
- Source-attributed event publishing (single + batch) with correlation context
- Schema registry with versioning and compatibility checks
- Consumer subscription lifecycle with configurable delivery modes and retry
- Stream processor topology: filter, map, aggregate, join, window
- Dead-letter capture, inspection, and reprocessing
- Replay with approval gate
- AI agent workbench for stream architecture and schema review
- Transactional outbox for dual-write safety (v2.0)
- Idempotent publishing with Redis-backed idempotency keys (v2.0)
- Priority queuing with back-pressure across CRITICAL/HIGH/NORMAL lanes (v2.0)
- Content-based routing with JSONPath filter expressions (v2.0)
- Prometheus-compatible RED metrics pipeline (v2.0)
- Consumer lag SLO alerting (v2.0)
- Cryptographic event signing per source capability (v2.0)

## Quick Start

```python
from capabilities.composition.events.service import CompositionEventsService

svc = CompositionEventsService()

# Create a stream
stream = svc.create_stream(
    stream_key="payments",
    tenant_id="acme",
    name="Payment Events",
    owner_id="user_123",
    source_capability="fintech_payments",
    retention_policy="7d",
    partition_key="aggregate_id",
)

# Register a schema (non-breaking)
schema = svc.register_schema(
    schema_key="payment_initiated_v1",
    tenant_id="acme",
    name="PaymentInitiated",
    version="1.0.0",
    definition={"type": "object", "required": ["amount", "currency"]},
)

# Publish an event (async)
import asyncio

async def main():
    event = await svc.publish_event(
        stream_id=stream["id"],
        tenant_id="acme",
        event_type="payment.initiated",
        payload={"amount": 5000, "currency": "KES"},
        source_capability="fintech_payments",
        correlation_id="corr_abc123",
        partition_key="order_42",
    )
    print(event["bytewax"])  # {"stream": "apg.acme.payments", "offset": 0}

asyncio.run(main())

# Subscribe a consumer
sub = svc.create_subscription(
    subscription_key="fraud-watcher",
    tenant_id="acme",
    stream_id=stream["id"],
    consumer_owner_id="svc_fraud",
    delivery_mode="at_least_once",
    retry_enabled=True,
    dead_letter_stream_id=dlq_stream["id"],
)
```

## New Methods

### `publish_event` (async)

Publishes a single event to a Bytewax stream. Returns the event record with `bytewax.stream` and `bytewax.offset`.

```python
event = await svc.publish_event(
    stream_id="event_stream_<uuid>",
    tenant_id="acme",
    event_type="order.shipped",
    payload={"order_id": "ord_99", "carrier": "DHL"},
    source_capability="logistics",
    correlation_id="corr_xyz",
    partition_key="ord_99",
)
# event["bytewax"] == {"stream": "apg.acme.logistics", "offset": 17}
```

### `register_processor`

Registers a Bytewax stream processor. Stateful processors require an explicit reviewer and checkpoint configuration — the service enforces this at call time, not at first failure.

```python
proc = svc.register_processor(
    processor_key="fraud-score-aggregator",
    tenant_id="acme",
    name="Fraud Score Aggregator",
    stream_id=stream["id"],
    stateful=True,
    checkpoint_configured=True,
    reviewed_by="user_456",
    processor_runtime="bytewax",
)
```

### `register_schema` with breaking change gate

Breaking schema changes are blocked unless a reviewer is named, producing an audit trail rather than a silent deny.

```python
schema = svc.register_schema(
    schema_key="payment_initiated_v2",
    tenant_id="acme",
    name="PaymentInitiated",
    version="2.0.0",
    definition={"type": "object", "required": ["amount", "currency", "idempotency_key"]},
    breaking_change=True,
    reviewed_by="user_456",   # required when breaking_change=True
)
```

### `validate_batch_publish`

Validates batch publish constraints (size cap, source attribution, Bytewax requirement) before committing the batch. Call before `EventPublishingService.publish_events_batch` to surface denials early.

```python
result = svc.validate_batch_publish(
    tenant_id="acme",
    batch_size=500,
    event_stream="bytewax",
)
# result == {"tenant_id": "acme", "batch_size": 500, "event_stream": "bytewax", ...}
```

### `dashboard_summary`

Returns aggregate counts for streams, schemas, subscriptions, processors, agents, and audit events — useful for health checks and monitoring dashboards.

```python
summary = svc.dashboard_summary(tenant_id="acme")
# {
#   "stream_count": 12,
#   "schema_count": 8,
#   "subscription_count": 34,
#   "processor_count": 6,
#   "event_agent_count": 2,
#   "audit_event_count": 1840,
#   "streaming": {...}
# }
```

## World-Class Enhancements (v2.0)

1. **Transactional Outbox** — Eliminates dual-write data loss by writing events to an `outbox` table in the same DB transaction as the domain write; a relay process atomically moves them to Bytewax. Pattern: Debezium/Eventuate Tram.

2. **Idempotency Keys** — `publish_event` accepts an `idempotency_key`; Redis atomic check-and-set returns the original `event_id` on retry instead of re-publishing. Prevents double-charges and over-counting.

3. **Priority Queuing with Back-Pressure** — Three `asyncio.PriorityQueue` lanes (CRITICAL / HIGH / NORMAL). NORMAL floods return HTTP 429; SLA-critical events are never queued behind bulk loads.

4. **Schema Evolution Compatibility Matrix** — `register_enhanced_schema` computes real backward/forward/full compatibility diffs using `jsonschema` vocabulary before accepting a new version. Stores the compatibility decision in `ESSchema.compatibility_matrix`.

5. **Replay Approval Gate + Rate Limiting** — `ReplayRequest` model with `rate_limit_eps`; `execute_replay` checks approval and paces emission via `asyncio.sleep`. Emits `events_replayed` lifecycle event with full audit trail.

6. **Consumer Lag SLO Alerting** — `ConsumerLagMonitor` runs every 30 s, compares lag against `ESSubscription.lag_slo_threshold`, and emits `consumer_lag_slo_breach` lifecycle events plus `ntfy` alerts on breach.

7. **Dead-Letter Reprocessing with Exponential Backoff** — `ESDeadLetterEntry` captures original event, error, retry count, and `next_retry_at = now + base_delay * 2^retry_count`. After `max_retries`, marks permanently failed and notifies.

8. **Cryptographic Event Signing** — ECDSA-P256 per `source_capability`; signature stored in `event_metadata["signature"]`. `verify_event_signature` reconstructs canonical bytes and verifies. Key rotation via `kid` header.

9. **Multi-Tenant Stream Namespace Partitioning** — Stream names hashed from `(tenant_id, salt)` instead of plaintext tenant ID. `TenantStreamACL` enforces cross-tenant read isolation; unauthorized attempts emit `unauthorized_stream_access` events.

10. **Exactly-Once Stream Processing via Checkpoint Fencing** — `ProcessorCheckpoint` two-phase write: output to Bytewax + DB checkpoint in the same transaction. Processors resume from last committed offset on restart; output events carry `checkpoint_id` for consumer-side deduplication.

11. **Stream Topology Visualization API** — `get_stream_topology(tenant_id)` returns a directed graph (nodes: streams / processors / subscriptions; edges: data flow with throughput and lag labels) for D3 rendering. Redis-cached with 60 s TTL.

12. **Adaptive Batch Sizing (AIMD)** — `AdaptiveBatchController` tracks per-batch `processing_time_ms` and adjusts `ESSubscription.batch_size` using additive increase / multiplicative decrease. Eliminates manual batch tuning under variable load.

13. **Event-Time Windowing with Watermarks** — Window boundaries computed from `event_data["timestamp"]` (producer clock) rather than `datetime.now()`. `WatermarkTracker` maintains per-partition high-watermark; late events routed to a side output stream.

14. **RED Metrics Pipeline** — `MetricsPipelineService` subscribes to `apg.composition.events.lifecycle` and aggregates `events_published_total`, `event_publish_duration_seconds`, `consumer_lag_messages`, `dead_letter_events_total`. Exposes `/metrics` in Prometheus text format.

15. **Content-Based Event Routing** — `SubscriptionConfig.filter_expression` accepts a DSL string (e.g. `"payload.amount > 10000 AND payload.currency == 'USD'"`). `FilterExpressionCompiler.compile(expr)` returns a cached callable evaluated in the hot path via `jsonpath-ng`.

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /composition-events/dashboard | GET | composition_events:view | Overview |
| streams | /composition-events/streams | GET/POST | composition_events:manage_streams | Streams |
| schemas | /composition-events/schemas | GET/POST | composition_events:govern | Governance |
| subscriptions | /composition-events/subscriptions | GET/POST | composition_events:operate | Consumers |
| processors | /composition-events/processors | GET/POST | composition_events:operate | Processing |
| dead_letters | /composition-events/dead-letters | GET/POST | composition_events:operate | Operations |
| agents | /composition-events/agents | GET/POST | composition_events:admin | Automation |
| settings | /composition-events/settings | GET/PUT | composition_events:admin | Administration |

REST API prefix: `/composition-events/api/v1`

## Provides

| Service | Description |
|---------|-------------|
| event_stream_registry | Define and manage named Bytewax streams with retention and partition policies |
| bytewax_event_publishing | Publish single events and batches with source attribution and correlation |
| event_schema_registry | Register, version, and validate event schemas with compatibility checks |
| subscription_lifecycle | Create and manage consumer subscriptions with delivery modes and retry policies |
| stream_processor_topology | Register and operate Bytewax stream processors (filter, map, aggregate, join, window) |
| dead_letter_operations | Capture, inspect, and reprocess failed events |
| event_agents | AI agent workbench for stream architecture and schema review |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authenticate producers and consumers |
| audl | Persist operation audit records |
| ntfy | Send dead-letter and processor degradation alerts |
| registry | Register this capability in the global catalog |
| composition_access | Enforce policy on all stream write operations |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | "default" | Tenant scope for all operations |
| streams.bytewax_stream_required | bool | true | All streams must be backed by Bytewax |
| streams.retention_policy_required | bool | true | Streams require an explicit retention policy |
| streams.schema_required_for_pii | bool | true | PII-carrying streams require a schema |
| publishing.source_capability_required | bool | true | Published events must declare their source capability |
| publishing.correlation_required | bool | true | Events must carry correlation or causation context |
| publishing.batch_size_limit | int | 1000 | Maximum events per batch publish call |
| subscriptions.dead_letter_required_for_retrying | bool | true | Retrying subscriptions require a dead-letter stream |
| processors.bytewax_required | bool | true | All processors must run on Bytewax |
| processors.checkpoint_required | bool | true | Processors require checkpoint configuration |
| event_agents.max_autonomous_scope | string | "recommend_and_validate" | Ceiling on autonomous agent actions |
| observability.event_stream | string | "apg.composition.events.lifecycle" | Bytewax stream name for bus lifecycle events |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context present | deny |
| event_write_requires_policy | write operation without policy attached | deny |
| stream_requires_owner | create_stream without owner | deny |
| stream_requires_retention_policy | create_stream without retention policy | deny |
| pii_stream_requires_schema | create PII stream without schema | deny |
| stream_requires_bytewax | create_stream not via bytewax | deny |
| breaking_schema_requires_review | register schema with breaking_change=true without review | require_review |
| publish_requires_source_capability | publish_event without source_capability | deny |
| publish_requires_correlation | publish_event without correlation context | deny |
| publish_requires_bytewax | publish_event not via bytewax | deny |
| batch_publish_limit | batch_publish with batch_size > 1000 | deny |
| batch_publish_requires_bytewax | batch_publish not via bytewax | deny |
| subscription_requires_owner | create_subscription without consumer owner | deny |
| retry_subscription_requires_dead_letter | create retrying subscription without dead-letter stream | deny |
| stateful_processor_requires_review | register stateful processor without review | require_review |
| processor_requires_checkpoint | register_processor without checkpoint | deny |
| processor_requires_bytewax | register_processor not on bytewax | deny |
| replay_requires_approval | replay_events without approval | deny |
| event_agent_runtime_supported | register_event_agent with unsupported runtime | deny |
| event_agent_role_supported | register_event_agent with unsupported role | deny |
| privileged_agent_event_action_requires_human_approval | agent proposes privileged action without human approval | deny |

## Data Models

| Model | Key Fields |
|-------|-----------|
| ESEvent | event_id, event_type, source_capability, aggregate_id, sequence_number, correlation_id, causation_id, tenant_id, status, priority, payload, schema_id, stream_id |
| ESStream | stream_id, stream_name, bytewax_stream_name, partitions, replication_factor, retention_time_ms, compression_type, source_capability, tenant_id, status |
| ESSubscription | subscription_id, stream_id, consumer_group_id, event_type_patterns, delivery_mode, batch_size, retry_policy, dead_letter_enabled, dead_letter_stream |
| ESConsumerGroup | group_id, group_name, stream_id, session_timeout_ms, active_consumers, total_lag, tenant_id |
| ESSchema | schema_id, schema_name, schema_version, schema_definition, schema_format, event_type, compatibility_level, compatibility_matrix |
| ESStreamProcessor | processor_id, processor_name, processor_type, stream_id, output_stream_id, stateful, checkpoint_interval_ms, parallelism, status |
| ESEventProcessingHistory | history_id, event_id, processor_name, processing_stage, status, started_at, duration_ms |
| ESStreamAssignment | assignment_id, event_id, stream_id, partition_id, offset, published_at, consumed_count |
| ESMetrics | metric_id, metric_name, metric_type, stream_id, metric_value, time_bucket, aggregation_period |
| ESAuditLog | audit_id, event_id, operation_type, actor_id, operation_details, tenant_id |
| ESOutboxEntry | outbox_id, event_payload, stream_id, tenant_id, relayed_at, status |
| ESDeadLetterEntry | entry_id, original_event_id, subscription_id, error_message, retry_count, next_retry_at, max_retries |
| ProcessorCheckpoint | processor_id, input_stream, committed_offset, output_stream, output_sequence |
| ReplayRequest | request_id, requested_by, approved_by, target_stream_id, from_offset, to_offset, rate_limit_eps |

Pydantic API models: `EventCreate`, `EventResponse`, `StreamCreate`, `StreamResponse`, `ConsumerGroupCreate`, `ConsumerGroupResponse`, `StreamProcessorCreate`, `StreamProcessorResponse`.

## Streaming Events

Events emitted to the bus's own lifecycle stream via Bytewax (`apg.composition.events.lifecycle`).

| Event | Trigger |
|-------|---------|
| stream_created | New stream registered |
| schema_registered | Schema version added to registry |
| event_published | Single event appended to a stream |
| event_batch_published | Batch of events appended |
| subscription_created | Consumer subscription activated |
| processor_registered | Stream processor registered |
| dead_letter_recorded | Failed event moved to dead-letter stream |
| events_replayed | Approved replay operation executed |
| event_agent_registered | New event agent registered |
| consumer_lag_slo_breach | Consumer lag exceeded SLO threshold |
| unauthorized_stream_access | Cross-tenant stream access attempt rejected |

Stream states: `draft → active → paused → review_required → processing → degraded → blocked → retired`

## Edge Cases Handled

- The bus emits its own lifecycle events to itself (`apg.composition.events.lifecycle`); this bootstrapping dependency is resolved by initializing the lifecycle stream before any other stream at tenant setup time.
- Stateful processors require review before registration because they carry state across restarts; a misconfigured state store can corrupt aggregate reconstructions for the entire tenant.
- Batch publishing is capped at 1000 events per call; callers must split larger batches to prevent memory exhaustion in the Bytewax dataflow worker.
- Retrying subscriptions without a dead-letter stream are blocked at creation, not at first failure, preventing silent event loss that only becomes visible under error conditions.
- Breaking schema changes produce `require_review` rather than `deny`, allowing forward progress with an explicit audit trail.
- The `_install_compat_init` mechanism handles legacy keyword aliases (`metadata` → `event_metadata`, `timestamp` → `event_timestamp`) for backward compatibility with older event producers.
- Outbox relay uses `SELECT FOR UPDATE SKIP LOCKED` so concurrent relay instances never double-publish the same outbox row.
- Watermark-based windowing routes late-arriving events to a side output stream rather than silently placing them in the wrong window.

## Composability

- **Upstream**: `composition_access` (policy enforcement on writes), `auth` (producer and consumer identity)
- **Downstream**: All composition capabilities (`access`, `config`, `gateway`, `orchestration`, `registry`) publish their lifecycle events here; domain capabilities use this bus for cross-capability integration events
- **Peer**: `audl` (receives audit log records for retention), `ntfy` (receives dead-letter, degradation, and SLO breach alerts), `composition_registry` (stream and schema metadata discovered by registry)

## Key Files

| File | Purpose |
|------|---------|
| `capability_contract.py` | Executable contract and rule engine |
| `models.py` | SQLAlchemy + Pydantic models |
| `service.py` | Lifecycle operations (`CompositionEventsService`, `EventPublishingService`, `EventConsumptionService`, `StreamProcessingService`, `SchemaRegistryService`, `EventStreamingService`) |
| `api.py` | API helpers |
| `views.py` | UI model helpers |

## Development Notes

- `ESStream` requires both `stream_name` and `bytewax_stream_name`; compat init defaults `bytewax_stream_name` to `stream_name` if not provided.
- `ESStreamProcessor.stateful=True` triggers `stateful_processor_requires_review`; always set `state_store_config` and `changelog_stream` before registering a stateful processor.
- The `_install_compat_init` pattern patches `__init__` on SQLAlchemy model classes at module load time; it is fragile under multiple inheritance and should not be extended to new models without careful testing.
- All v2.0 enhancements are additive — no existing API signatures changed.
