# NATS JetStream Event Bus — User Guide

## Overview

The `nats` capability provides a durable, exactly-once event bus over NATS JetStream. Every APG domain event is published to a subject hierarchy and consumed by downstream capabilities — enabling real-time collaboration, notifications, audit trails, and cross-service orchestration without tight coupling.

## Quick Start

```bash
# Start NATS with JetStream
docker run -d --name apg-nats -p 4222:4222 nats:2.10-alpine --jetstream

# Set env var so all capabilities route events to NATS
export NATS_URL=nats://localhost:4222
```

## Subject Convention

All APG events follow the hierarchy:

```
apg.events.{capability_id}.{event_type}
```

Examples:
- `apg.events.ckm_wfa.workflow_started`
- `apg.events.fintech_gwy.payment_received`
- `apg.events.intel_alerts.alert_triggered`

The `subject_for(capability_id, event_type)` helper function generates these subjects.

## Publishing Events

```python
from capabilities.common.nats.service import NATSService

svc = NATSService(nats_url="nats://localhost:4222", tenant_id="acme")
await svc.connect()

result = await svc.publish(
    capability_id="fintech_gwy",
    event_type="payment_received",
    payload={"amount": 1000, "currency": "KES", "reference": "INV-001"},
    actor_id="user_123",
)
# {"published": True, "subject": "apg.events.fintech_gwy.payment_received", "event_id": "..."}
```

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/nats/publish` | Publish a single event |
| POST | `/api/nats/publish/batch` | Publish multiple events |
| GET | `/api/nats/streams` | List all JetStream streams |
| POST | `/api/nats/streams` | Create a stream |
| GET | `/api/nats/streams/{name}` | Get stream info |
| DELETE | `/api/nats/streams/{name}` | Delete stream |
| POST | `/api/nats/streams/{name}/purge` | Purge stream messages |
| GET | `/api/nats/streams/{name}/consumers` | List consumers |
| POST | `/api/nats/streams/{name}/consumers` | Create consumer |
| GET | `/api/nats/subjects` | List all subjects |
| GET | `/api/nats/subjects/resolve?capability_id=X&event_type=Y` | Resolve subject |
| GET | `/api/nats/health` | Health check |
| POST | `/api/nats/setup` | Provision APG streams |

## Stream Setup

The APG platform stream (`APG_EVENTS`) is created automatically when you call `/api/nats/setup` or run:

```python
await svc.setup_apg_streams()
```

This provisions a `FileStorage` stream covering `apg.events.>` with 90-day retention.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `NATS_URL` | `nats://localhost:4222` | NATS server URL |
| `NATS_STREAM_REPLICAS` | `1` | JetStream replicas (set 3 for production) |

## Interoperability

Any APG capability that calls `get_audit_adapter()` will automatically route to NATS when `NATS_URL` is set. No capability code changes are needed — the factory swap handles it transparently.

Downstream capabilities subscribe to relevant subjects:
- `ckm_not` (Notifications) → `apg.events.*.notification_requested`
- `ckm_rtc` (Real-Time Collaboration) → `apg.events.*.{resource_id}.>`
- `intel_alerts` → `apg.events.*.alert_triggered`
