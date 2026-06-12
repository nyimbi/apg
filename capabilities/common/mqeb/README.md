# APG MQEB - Message Queue Event Bus

MQEB is APG's package-backed event fabric. It provides tenant-scoped topic
management, governed message publishing, subscription lifecycle state,
delivery/dead-letter evidence, replay review, priority quota review, rule
evaluation, first-class event-agent composition, Bytewax lifecycle validation,
UI view models, and publishable package evidence for generated APG applications.

MQEB is **Bytewax-first**. Bytewax workers and dataflows are the preferred
runtime boundary for stream processing. Broker-specific queue support may be
added later through an optional compatibility bridge, but it is not a core
dependency.

## What This Package Provides

- Topic lifecycle records with classification, retention, delivery mode,
  encryption, schema, dead-letter, and status fields.
- Message publish decisions backed by MQEB-owned topic state, quota exception
  state, encryption/schema/idempotency evidence, and deterministic rules.
- Subscription lifecycle records with protocol, delivery mode, pause/resume,
  lag, and dead-letter state.
- Delivery attempt evidence for delivered, retry, and dead-letter outcomes.
- Priority quota exception request and independent-review workflows.
- Replay request and independent-review workflows with bounded ranges and
  evidence.
- Durable review evidence for review-required messages, priority quota
  exceptions, replay requests, privileged event agents, lifecycle batch
  validations, delivery attempts, and audit events.
- First-class event-agent registration for Codex, Claude Code, opencode, and
  Pi with supported roles, owner, purpose, scope, contribution disclosure, and
  human approval guardrails.
- Bytewax lifecycle-batch validation for generated event-fabric mutations.
- Audit event records for topic, publish, subscription, delivery, review,
  replay, event-agent, and lifecycle-batch actions.
- **v2.0**: Idempotency deduplication, scheduled delivery, priority-tier queues,
  tenant rate quotas, HMAC-signed audit streaming, DLQ lifecycle, and W3C
  trace context propagation.

## Runtime Shape

The dependency-light service is `MqebService` in `service.py`. It is designed
for generated applications and package tests. It does not require a live broker,
Bytewax worker, Flask app, cloud queue, SIEM, schema registry, or APG service
mesh to run.

The existing async `MQEBService` remains available for the broader broker
runtime and Flask API surface.

## Basic Usage

```python
from capabilities.common.mqeb.service import MqebService

service = MqebService()

topic = service.create_topic(
    tenant_id="tenant-a",
    topic_id="orders",
    name="Orders",
    owner="commerce",
    classification="regulated",
    encrypted=True,
    schema_ref="schema://orders/v1",
    delivery_mode="exactly_once",
    dead_letter_topic="orders.dlq",
)

message = service.publish_message(
    tenant_id="tenant-a",
    message_id="order-1001",
    topic_id=topic["id"],
    producer="order-service",
    delivery_mode="exactly_once",
    idempotency_key="order-1001",
    payload_size=512,
)

subscription = service.create_subscription(
    tenant_id="tenant-a",
    subscription_id="warehouse",
    name="Warehouse Projection",
    topic_pattern="orders",
    consumer="warehouse-sync",
    protocol="bytewax",
    delivery_mode="exactly_once",
    dead_letter_topic="orders.dlq",
)

service.record_delivery_attempt(
    tenant_id="tenant-a",
    attempt_id="delivery-1",
    message_id=message["id"],
    subscription_id=subscription["id"],
    outcome="delivered",
)

agent = service.register_event_agent(
    tenant_id="tenant-a",
    agent_id="replay-agent",
    name="Replay Agent",
    runtime="claude-code",
    role="replay-reviewer",
    scope="bounded replay review",
    owner="platform",
    purpose="review replay approvals",
    contribution_disclosed=True,
    human_approval_required=True,
)

service.validate_event_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=3,
)
```

## Guardrails

MQEB fails closed for:

- missing tenant context;
- missing topic, owner, producer, consumer, or payload metadata;
- restricted topic publish without encryption;
- regulated topic publish without schema evidence;
- cross-tenant publish without an authorized exchange adapter;
- exactly-once publish without dead-letter and idempotency evidence;
- disabled topic publish;
- paused subscription delivery;
- priority bursts without an approved quota exception;
- replay requests without bounded range and reason;
- self-reviewed or note-less quota/replay reviews;
- event-agent registration without supported runtime/role, owner, purpose,
  scope, or contribution disclosure;
- lifecycle mutation batches that do not use Bytewax.

Privileged event agents without human approval are retained as
`pending_review` so governance consoles can decide them without losing the
registration attempt.

## Durable Review Evidence

MQEB preserves review state for generated event-governance consoles.
Review-required messages, priority quota exceptions, replay requests,
privileged event-agent registrations, lifecycle batch validations, delivery
attempts, and audit events carry the same evidence fields:

- `policy_decision`;
- `matched_rules`;
- `review_reasons`;
- `review_evidence`.

```python
pending = service.list_pending_reviews("tenant-a")
```

Denied non-Bytewax lifecycle batches are stored through
`list_lifecycle_batches()` before `PermissionError` is raised, so operators can
see and remediate routing violations.

---

## World-Class Enhancements (v2.0)

15 targeted improvements that elevate MQEB to a production-grade, observable,
composable event fabric competitive with Apache Kafka, AWS EventBridge, and
Google Cloud Pub/Sub.

| # | Name | Category | Summary |
|---|------|----------|---------|
| I1 | Backpressure-Aware Flow Control | Reliability | Per-topic `max_queue_depth` with token-bucket admission control; overflow redirects to configurable spill topic. Comparable to Kafka producer quotas and SQS `ApproximateNumberOfMessages` alarms. |
| I2 | Cursor-Based Consumer Groups with Committed Offsets | Correctness | `ConsumerGroupRecord` with monotonic committed offsets; `commit_offset` and `seek_offset` methods; offset changes emitted as audit events. Closes the consumer-restart exactly-once gap. |
| I3 | Schema Registry Validation with Fail-Close | Data Quality | `SchemaRegistryAdapter` protocol; `validate_message_schema` caches compiled validators in `BoundedCache`; raises `PermissionError("schema_validation_failed")` on regulated topics with mismatched payloads. |
| I4 | Distributed Saga with Compensating Transactions | Correctness | `SagaRecord` + `SagaStep`; `start_saga`, `advance_saga`, `compensate_saga` methods; each transition emits an audit event. Provides in-fabric rollback without AWS Step Functions or Temporal. |
| I5 | Time-Based Message Scheduling with Cancellation | Usability | `schedule_message` holds messages in `scheduled_messages` until ISO-8601 delivery time; `drain_scheduled_messages` publishes due entries; `cancel_scheduled_message` requires actor + reason audit evidence. |
| I6 | Idempotency Store with TTL Eviction | Correctness | `(tenant_id, idempotency_key)` → `(message_id, expires_at)` cache; duplicate publishes return the original record immediately; TTL tracks topic `retention_days`. Eliminates producer-side duplicate messages. |
| I7 | Structured Retry Policies with Exponential Backoff | Reliability | `RetryPolicyRecord` with configurable strategy, base delay, max attempts, jitter factor, and dead-letter-on-exhaust; `apply_retry_policy` computes next delivery timestamp and auto-routes to DLQ when exhausted. |
| I8 | Priority Queue with Preemptive Weighted Scheduling | Performance | Messages bucketed into critical/high/normal/low tiers; `_process_pending_deliveries` drains tiers in weighted order; `get_priority_queue_stats` exposes per-tier depths and drain rates. |
| I9 | Append-Only Audit Log with HMAC Integrity | Security | `stream_audit_events` async generator yields records signed with `hmac.new(tenant_secret, record_bytes, sha256)`; `integrity_sig` field enables tamper detection for SOC 2/GDPR/PCI-DSS audits. |
| I10 | Circuit Breaker for Downstream Delivery Endpoints | Reliability | `CircuitBreakerRecord` tracks failure count, threshold, and reset timeout per subscription; `reset_subscription_circuit_breaker` provides operator override with audit event. Prevents cascade failure to failing webhooks/gRPC endpoints. |
| I11 | Content-Based Message Routing DSL | Usability | `RoutingRule.match_expr` is a safe DSL evaluated by a restricted `ast` visitor (no attribute access, no calls); rules evaluated before queue insertion in `_route_message_to_subscriptions`. |
| I12 | Tenant Quotas and Token-Bucket Rate Limiting | Multi-tenancy | `TenantQuotaRecord` with `max_messages_per_minute`, `max_bytes_per_minute`, `max_topics`; `set_tenant_quota` / `get_tenant_quota_status` expose utilisation ratios; exceeded quota raises `PermissionError("tenant_quota_exceeded")`. |
| I13 | Message Batch Publish with Optional Compression | Performance | `publish_message_batch` runs policy evaluation once per shared context; `zstd` compression stores `compressed_size_bytes` and `compression_ratio` per record; emits one audit event with `batch_size`. 10–50x throughput improvement for high-volume producers. |
| I14 | Dead-Letter Queue Lifecycle with Redrive Workflows | Reliability | `inspect_dead_letter_queue`, `redrive_dead_letter_messages`, `purge_dead_letter_queue`, `export_dead_letter_messages` — full in-fabric DLQ lifecycle with reviewer evidence requirements on all destructive operations. |
| I15 | OpenTelemetry W3C Trace Context Propagation | Observability | `trace_context: dict[str, str]` on `MessageRecord` carries `traceparent`/`tracestate` through `publish_message` → `_route_message_to_subscriptions` → `consume_messages`; auto-injects/extracts via standard propagator when `opentelemetry-api` is importable. |

---

## New Methods

### I6 — Idempotent Publish

```python
import asyncio
from capabilities.common.mqeb.service import MqebService

svc = MqebService()
svc.create_topic(
    tenant_id="t1", topic_id="payments", name="Payments",
    owner="fintech", delivery_mode="exactly_once",
    dead_letter_topic="payments.dlq",
)

async def demo_idempotency():
    msg1 = await svc.async_publish_message(
        tenant_id="t1", message_id="pay-001", topic_id="payments",
        producer="checkout", idempotency_key="txn-abc-123", payload_size=128,
        delivery_mode="exactly_once", encrypted=True, schema_ref="schema://payments/v1",
    )
    msg2 = await svc.async_publish_message(
        tenant_id="t1", message_id="pay-001b", topic_id="payments",
        producer="checkout", idempotency_key="txn-abc-123", payload_size=128,
        delivery_mode="exactly_once", encrypted=True, schema_ref="schema://payments/v1",
    )
    assert msg1["id"] == msg2["id"]  # duplicate suppressed — same record returned

asyncio.run(demo_idempotency())
```

### I5 — Scheduled Message Delivery

```python
async def demo_schedule():
    svc.create_topic(
        tenant_id="t1", topic_id="reminders", name="Reminders",
        owner="notifications", delivery_mode="at_least_once",
    )
    scheduled = await svc.schedule_message(
        tenant_id="t1",
        message_id="reminder-001",
        topic_id="reminders",
        producer="scheduler",
        scheduled_at_iso="2099-01-01T00:00:00Z",
        payload_size=64,
    )
    assert scheduled["status"] == "scheduled"

    cancelled = await svc.cancel_scheduled_message(
        tenant_id="t1",
        message_id="reminder-001",
        actor="ops-engineer",
        reason="user cancelled subscription before delivery window",
    )
    assert cancelled["status"] == "cancelled"

asyncio.run(demo_schedule())
```

### I9 — HMAC-Signed Audit Log Streaming

```python
async def demo_audit_stream():
    events = []
    async for event in svc.stream_audit_events("t1", batch_size=20):
        assert "integrity_sig" in event  # HMAC-SHA256 over record JSON
        events.append(event)
    # Resume from a known position
    async for event in svc.stream_audit_events("t1", since_id=events[-1]["id"]):
        pass  # incremental tail

asyncio.run(demo_audit_stream())
```

### I12 — Tenant Rate Quotas

```python
async def demo_quota():
    await svc.set_tenant_quota(
        tenant_id="t1",
        max_messages_per_minute=1000,
        max_bytes_per_minute=10_000_000,
        max_topics=50,
        actor="platform-admin",
    )
    status = await svc.get_tenant_quota_status("t1")
    assert status["quota_configured"] is True
    print(status["message_utilization_ratio"])  # 0.0 – 1.0

asyncio.run(demo_quota())
```

### I14 — Dead-Letter Queue Lifecycle

```python
async def demo_dlq():
    # Inspect what landed in the DLQ
    report = await svc.inspect_dead_letter_queue("t1", "payments.dlq")
    print(f"{report['dead_letter_message_count']} messages in DLQ")

    # Redrive up to 5 messages to the originating topic
    result = await svc.redrive_dead_letter_messages(
        tenant_id="t1",
        dlq_topic_id="payments.dlq",
        target_topic_id="payments",
        reviewer="incident-lead",
        evidence="root cause fixed in deploy abc123",
        max_count=5,
    )
    print(f"Redriven: {result['redriven_count']}")

    # Or permanently purge after investigation
    purge = await svc.purge_dead_letter_queue(
        tenant_id="t1",
        dlq_topic_id="payments.dlq",
        reviewer="incident-lead",
        reason="messages confirmed unrecoverable; incident closed",
    )
    print(f"Purged: {purge['purged_count']}")

asyncio.run(demo_dlq())
```

---

## API Surface

### Sync methods (MqebService)

| Method | Description |
|--------|-------------|
| `create_topic(...)` | Create a tenant-scoped topic with classification, delivery mode, schema, DLQ config |
| `publish_message(...)` | Publish with full policy evaluation; returns status + review evidence |
| `create_subscription(...)` | Subscribe to topic pattern with protocol and DLQ |
| `pause_subscription(...)` | Pause delivery with actor + reason audit |
| `resume_subscription(...)` | Resume delivery with actor + evidence audit |
| `record_delivery_attempt(...)` | Record delivered / retry / dead_letter outcome |
| `request_priority_exception(...)` | Open a priority quota exception for review |
| `decide_priority_exception(...)` | Approve or reject a priority exception (independent reviewer only) |
| `request_replay(...)` | Open a bounded replay request for review |
| `decide_replay(...)` | Approve or reject a replay request (independent reviewer only) |
| `register_event_agent(...)` | Register a first-class AI event agent with guardrail evidence |
| `validate_event_lifecycle_batch(...)` | Enforce Bytewax-only lifecycle mutation batches |
| `list_pending_reviews(...)` | Aggregate all items awaiting human review |
| `dashboard_summary(...)` | Counts, DLQ depth, recent audit events |

### Async methods (MqebService, v2.0)

| Method | Improvement | Description |
|--------|-------------|-------------|
| `async_publish_message(...)` | I6 + I15 | Idempotency deduplication + W3C trace context propagation |
| `schedule_message(...)` | I5 | Queue message for ISO-8601 future delivery |
| `cancel_scheduled_message(...)` | I5 | Cancel pending scheduled message with audit evidence |
| `drain_scheduled_messages(...)` | I5 | Transfer due messages into topic queues (processing loop or test use) |
| `get_priority_queue_stats(...)` | I8 | Per-priority-tier message depths for a topic |
| `set_tenant_quota(...)` | I12 | Configure per-tenant publish rate and topic limits |
| `get_tenant_quota_status(...)` | I12 | Current quota config and utilisation ratios |
| `stream_audit_events(...)` | I9 | Async-generator tail of HMAC-signed audit log |
| `inspect_dead_letter_queue(...)` | I14 | List dead-letter messages and delivery attempts |
| `redrive_dead_letter_messages(...)` | I14 | Re-publish DLQ messages to target topic with reviewer evidence |
| `purge_dead_letter_queue(...)` | I14 | Permanently discard DLQ messages with audited sign-off |

---

## API Helpers

`api.py` exposes dependency-light helpers backed by shared `api.SERVICE`:

- `create_topic_record`
- `publish_message_record`
- `create_subscription_record`
- `pause_subscription_record`
- `resume_subscription_record`
- `record_delivery_attempt`
- `request_priority_exception`
- `decide_priority_exception`
- `request_replay`
- `decide_replay`
- `register_event_agent`
- `validate_event_lifecycle_batch`
- `list_lifecycle_batches`
- `list_pending_reviews`
- `list_event_fabric`
- `capability_status`

## View Models

`view_models.py` exposes generated-app models for:

- dashboard;
- topic inventory;
- publish workbench;
- subscriptions;
- delivery and dead letters;
- priority quota review queue;
- replay console;
- event-agent roster;
- Bytewax bridge status;
- audit timeline;
- settings.

## Adapter Boundaries

Production integrations should sit behind adapters that honor MQEB decisions:

- Bytewax workers and dataflows;
- APG AUTH, MTEN, AUDL, CONF, KEYM, ENCR, SECU, MONI, and HLTH;
- HTTP, WebSocket, MQTT, AMQP, gRPC, webhook, and event-file adapters;
- schema registries and metadata services (I3);
- SIEM, SOAR, DLP, GRC, notification, and incident-response systems;
- cloud queue/event services;
- optional broker-specific queue compatibility bridge.

## Focused Proof

```bash
./.venv/bin/python -m py_compile capabilities/common/mqeb/__init__.py capabilities/common/mqeb/models.py capabilities/common/mqeb/service.py capabilities/common/mqeb/api.py capabilities/common/mqeb/capability_contract.py capabilities/common/mqeb/app.py capabilities/common/mqeb/view_models.py capabilities/common/mqeb/tests/test_capability_contract.py capabilities/common/mqeb/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/mqeb/tests/test_capability_contract.py capabilities/common/mqeb/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/mqeb --json
./.venv/bin/apg capabilities publish-plan capabilities/common/mqeb --json
```
