# Message Queue Event Bus — User Guide

**Capability ID**: `mqeb` | **Domain**: `common` | **Version**: `1.1.0`
**Author**: Nyimbi Odero | **Copyright**: © 2025 Datacraft

---

## Overview

MQEB is APG's package-backed event fabric. It provides tenant-scoped topic management, governed message publishing, subscription lifecycle state, delivery/dead-letter evidence, replay review, priority quota review, rule evaluation, first-class event-agent composition, and Bytewax lifecycle validation for generated APG applications.

MQEB is **Bytewax-first**. Bytewax workers and dataflows are the preferred runtime boundary for stream processing.

---

## Installation

```bash
pip install apg-common-mqeb
```

---

## Provides

- `mqeb_event_fabric`
- `message_governance`
- `event_agent_composition`
- `review_evidence`
- `scheduled_message_delivery` *(v1.1)*
- `idempotent_publish` *(v1.1)*
- `tenant_quota_enforcement` *(v1.1)*
- `tamper_evident_audit_log` *(v1.1)*
- `dead_letter_lifecycle` *(v1.1)*

---

## Requires

- `conf`
- `auth`
- `audl`
- `secu`

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mqeb/dashboard` | `mqeb:view` | Overview |
| `/mqeb/topics` | `mqeb:manage_topics` | Operations |
| `/mqeb/publish` | `mqeb:publish` | Operations |
| `/mqeb/subscriptions` | `mqeb:subscribe` | Operations |
| `/mqeb/delivery` | `mqeb:view_metrics` | Reliability |
| `/mqeb/dead-letters` | `mqeb:manage_routing` | Reliability |
| `/mqeb/quota-exceptions` | `mqeb:admin` | Governance |
| `/mqeb/replays` | `mqeb:admin` | Governance |
| `/mqeb/scheduled` | `mqeb:publish` | Operations |
| `/mqeb/quota-status` | `mqeb:admin` | Governance |

---

## Core Service Methods

### Topic Management

```python
service.create_topic(
    tenant_id, topic_id, name, owner,
    classification="internal",        # public | internal | restricted | regulated
    retention_days=7,
    delivery_mode="at_least_once",    # at_most_once | at_least_once | exactly_once
    encrypted=False,
    schema_ref="",
    dead_letter_topic="",
    status="active",
)
```

### Message Publishing

```python
# Synchronous (immediate)
service.publish_message(
    tenant_id, message_id, topic_id, producer,
    priority="normal",
    delivery_mode=None,
    encrypted=None,
    schema_ref="",
    idempotency_key="",
    payload_size=1,
    priority_messages_per_minute=0,
    cross_tenant_publish=False,
)

# Async with idempotency deduplication and trace propagation (v1.1)
await service.async_publish_message(
    tenant_id, message_id, topic_id, producer,
    idempotency_key="unique-key",
    trace_context={"traceparent": "00-...", "tracestate": ""},
)
```

### Scheduled Messages (v1.1)

```python
# Schedule for future delivery
record = await service.schedule_message(
    tenant_id="t1",
    message_id="order-999",
    topic_id="orders",
    producer="order-service",
    scheduled_at_iso="2025-12-31T23:59:00Z",   # ISO-8601 UTC, must be future
    payload_size=256,
)
# record["status"] == "scheduled"

# Cancel before delivery window
await service.cancel_scheduled_message(
    tenant_id="t1",
    message_id="order-999",
    actor="ops-engineer",
    reason="order voided",
)

# Drain due messages into topic queues (called by background loop)
published = await service.drain_scheduled_messages(tenant_id="t1")
```

### Subscriptions

```python
service.create_subscription(
    tenant_id, subscription_id, name, topic_pattern, consumer,
    delivery_mode="at_least_once",
    protocol="bytewax",
    dead_letter_topic="",
)

service.pause_subscription(tenant_id, subscription_id, actor, reason)
service.resume_subscription(tenant_id, subscription_id, actor, evidence)
```

### Dead-Letter Queue Lifecycle (v1.1)

```python
# Inspect DLQ contents
info = await service.inspect_dead_letter_queue(tenant_id, dlq_topic_id)
print(info["dead_letter_message_count"])

# Redrive messages to originating topic (requires reviewer evidence)
result = await service.redrive_dead_letter_messages(
    tenant_id=tenant_id,
    dlq_topic_id="orders.dlq",
    target_topic_id="orders",
    reviewer="ops-lead",
    evidence="Root cause resolved — safe to redrive",
    max_count=10,
)
print(result["redriven_count"])

# Purge all dead-letter messages (irreversible — requires reviewer sign-off)
await service.purge_dead_letter_queue(
    tenant_id=tenant_id,
    dlq_topic_id="orders.dlq",
    reviewer="platform-admin",
    reason="Retention window expired; messages unrecoverable",
)
```

### Priority Queue Stats (v1.1)

```python
stats = await service.get_priority_queue_stats(tenant_id, topic_id)
# {
#   "total_pending": 142,
#   "by_priority": {"critical": 3, "high": 12, "normal": 100, "low": 27}
# }
```

### Tenant Quota Management (v1.1)

```python
# Configure quotas
await service.set_tenant_quota(
    tenant_id="t1",
    max_messages_per_minute=5000,
    max_bytes_per_minute=50_000_000,
    max_topics=100,
    actor="platform-admin",
)

# Check utilisation
status = await service.get_tenant_quota_status("t1")
# {
#   "quota_configured": True,
#   "message_utilization_ratio": 0.042,
#   "bytes_utilization_ratio": 0.003,
#   "topics_used": 7
# }
```

### Audit Log Streaming (v1.1)

```python
# Tail audit events with HMAC-SHA256 integrity verification
async for event in service.stream_audit_events(
    tenant_id="t1",
    since_id=None,       # resume from a known event id
    batch_size=100,
):
    sig = event.pop("integrity_sig")
    # verify sig before persisting to SIEM
    print(event["event_type"], event["created_at"])
```

Each yielded event includes `integrity_sig`: an HMAC-SHA256 over the canonical JSON representation, keyed by a per-tenant secret derived from `hashlib.sha256(tenant_id)`. Governance consoles can detect tampering by recomputing and comparing.

### Review and Governance

```python
# Priority quota exception
service.request_priority_exception(tenant_id, exception_id, topic_id, requested_by, reason)
service.decide_priority_exception(tenant_id, exception_id, reviewer, decision, notes)

# Replay
service.request_replay(tenant_id, replay_id, topic_id, requested_by, reason, range_start, range_end)
service.decide_replay(tenant_id, replay_id, reviewer, decision, evidence)

# Pending review queue
pending = service.list_pending_reviews(tenant_id)
```

### Event Agents

```python
service.register_event_agent(
    tenant_id, agent_id, name,
    runtime="claude-code",         # codex | opencode | pi | claude-code
    role="replay-reviewer",
    scope="bounded replay review",
    owner="platform",
    purpose="review replay approvals",
    contribution_disclosed=True,
    human_approval_required=True,
)
```

### Dashboard

```python
summary = service.dashboard_summary(tenant_id)
# Keys: topic_count, message_count, denied_message_count, review_required_count,
#       subscription_count, paused_subscription_count, dead_letter_count,
#       pending_priority_exception_count, pending_replay_count,
#       event_agent_count, pending_event_agent_review_count,
#       lifecycle_batch_count, denied_lifecycle_batch_count,
#       pending_review_count, recent_events
```

---

## Guardrails

MQEB fails closed for:

- missing tenant context
- missing topic, owner, producer, consumer, or payload metadata
- restricted topic publish without encryption
- regulated topic publish without schema evidence
- cross-tenant publish without an authorized exchange adapter
- exactly-once publish without dead-letter and idempotency evidence
- disabled topic publish
- paused subscription delivery
- priority bursts without an approved quota exception
- replay requests without bounded range and reason
- self-reviewed or note-less quota/replay reviews
- event-agent registration without supported runtime/role, owner, purpose, scope, or contribution disclosure
- lifecycle mutation batches that do not use Bytewax
- scheduling a message with a past `scheduled_at` timestamp
- redriving or purging DLQ without reviewer identity and evidence

---

## Durable Review Evidence

Every record that passes through a review gate carries:

- `policy_decision` — `allow | require_review | deny`
- `matched_rules` — list of rule identifiers that fired
- `review_reasons` — human-readable reason codes
- `review_evidence` — structured dict with `required_actions`, `reasons`, `review_recorded`

---

## Adapter Boundaries

Production integrations should sit behind adapters that honour MQEB decisions:

- Bytewax workers and dataflows
- APG AUTH, MTEN, AUDL, CONF, KEYM, ENCR, SECU, MONI, HLTH
- HTTP, WebSocket, MQTT, AMQP, gRPC, webhook, and event-file adapters
- Schema registries and metadata services
- SIEM, SOAR, DLP, GRC, notification, and incident-response systems
- Cloud queue/event services
- Optional broker-specific queue compatibility bridge

---

## Testing

```bash
# Syntax check
./.venv/bin/python -m py_compile capabilities/common/mqeb/service.py

# Unit tests
./.venv/bin/pytest -q capabilities/common/mqeb/tests/

# Capability audit
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/mqeb --json
```
