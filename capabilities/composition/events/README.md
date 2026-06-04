# Event Streaming Bus

## Overview

The Event Streaming Bus is the foundational messaging backbone for the APG composition layer. It provides Bytewax-powered event streams with schema validation, producer attribution, consumer group management, stateful stream processors, dead-letter handling, and approved event replay. Every other composition capability routes its lifecycle events through this bus.

The business value is a durable, auditable communication fabric that decouples producers from consumers across capability boundaries. Schema compatibility enforcement prevents breaking changes from silently corrupting downstream processors. Dead-letter streams and bounded retry policies ensure no event is silently lost. The replay approval gate prevents unauthorized state reconstruction from historical data.

## Capability ID

`composition_events`  Version: see `package_manifest.json`

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
| ESSchema | schema_id, schema_name, schema_version, schema_definition, schema_format, event_type, compatibility_level |
| ESStreamProcessor | processor_id, processor_name, processor_type, stream_id, output_stream_id, stateful, checkpoint_interval_ms, parallelism, status |
| ESEventProcessingHistory | history_id, event_id, processor_name, processing_stage, status, started_at, duration_ms |
| ESStreamAssignment | assignment_id, event_id, stream_id, partition_id, offset, published_at, consumed_count |
| ESMetrics | metric_id, metric_name, metric_type, stream_id, metric_value, time_bucket, aggregation_period |
| ESAuditLog | audit_id, event_id, operation_type, actor_id, operation_details, tenant_id |

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

Stream states: `draft → active → paused → review_required → processing → degraded → blocked → retired`

## Edge Cases Handled

- The bus emits its own lifecycle events to itself (`apg.composition.events.lifecycle`); this creates a bootstrapping dependency that is resolved by initializing the lifecycle stream before any other stream at tenant setup time.
- Stateful processors require review before registration because they carry state across restarts; a misconfigured state store can corrupt aggregate reconstructions for the entire tenant.
- Batch publishing is capped at 1000 events per call by rule (`batch_publish_limit`); callers must split larger batches, which prevents memory exhaustion in the Bytewax dataflow worker.
- Retrying subscriptions without a dead-letter stream are blocked at creation, not at first failure; this prevents silent event loss that only becomes visible under error conditions.
- Breaking schema changes produce `require_review` rather than `deny`, allowing forward progress with an explicit audit trail while still preventing silent incompatible schema deployments.
- The `_install_compat_init` mechanism on SQLAlchemy models handles legacy keyword aliases (`metadata` → `event_metadata`, `timestamp` → `event_timestamp`) to maintain backward compatibility with older event producers.

## Composability

- **Upstream**: `composition_access` (policy enforcement on writes), `auth` (producer and consumer identity)
- **Downstream**: All composition capabilities (`access`, `config`, `gateway`, `orchestration`, `registry`) publish their lifecycle events here; domain capabilities use this bus for cross-capability integration events
- **Peer**: `audl` (receives audit log records for retention), `ntfy` (receives dead-letter and degradation alerts), `composition_registry` (stream and schema metadata discovered by registry)

## Development Notes

- The `ESStream` model requires both `stream_name` and `bytewax_stream_name`; the compat init defaults `bytewax_stream_name` to `stream_name` if not provided, but they can differ when the Bytewax topic name follows a different naming convention.
- `ESStreamProcessor.stateful=True` triggers the `stateful_processor_requires_review` rule; always set `state_store_config` and `changelog_stream` before attempting to register a stateful processor.
- The `_install_compat_init` pattern patches `__init__` on SQLAlchemy model classes at module load time; it is fragile under multiple inheritance and should not be extended to new models without careful testing.
- Key files: `capability_contract.py` (executable contract and rule engine), `models.py` (SQLAlchemy + Pydantic models), `service.py` (lifecycle operations), `api.py` (API helpers), `views.py` (UI model helpers).
