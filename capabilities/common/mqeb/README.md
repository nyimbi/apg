# APG MQEB - Message Queue Event Bus

MQEB is APG's package-backed event fabric. It provides tenant-scoped topic
management, governed message publishing, subscription lifecycle state,
delivery/dead-letter evidence, replay review, priority quota review, rule
evaluation, UI view models, and publishable package evidence for generated APG
applications.

MQEB is **Bytewax-first**. Bytewax workers and dataflows are the preferred
runtime boundary for stream processing. Kafka may be supported later through an
optional compatibility bridge, but it is not the MQEB core dependency.

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
- Audit event records for topic, publish, subscription, delivery, review, and
  replay actions.
- Dependency-light API helpers and generated-app view models.
- Contract-derived `app.py`, `semantic_model.json`, `release_report.json`, and
  `package_manifest.json` evidence.

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
- self-reviewed or note-less quota/replay reviews.

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
- `list_event_fabric`
- `capability_status`

These helpers are separate from the Flask routes already present in the file.

## View Models

`view_models.py` exposes generated-app models for:

- dashboard;
- topic inventory;
- publish workbench;
- subscriptions;
- delivery and dead letters;
- priority quota review queue;
- replay console;
- Bytewax bridge status;
- audit timeline;
- settings.

## Adapter Boundaries

Production integrations should sit behind adapters that honor MQEB decisions:

- Bytewax workers and dataflows;
- APG AUTH, MTEN, AUDL, CONF, KEYM, ENCR, SECU, MONI, and HLTH;
- HTTP, WebSocket, MQTT, AMQP, gRPC, webhook, and event-file adapters;
- schema registries and metadata services;
- SIEM, SOAR, DLP, GRC, notification, and incident-response systems;
- cloud queue/event services;
- optional Kafka compatibility bridge.

## Focused Proof

```bash
./.venv/bin/python -m py_compile capabilities/common/mqeb/__init__.py capabilities/common/mqeb/models.py capabilities/common/mqeb/service.py capabilities/common/mqeb/api.py capabilities/common/mqeb/capability_contract.py capabilities/common/mqeb/app.py capabilities/common/mqeb/view_models.py capabilities/common/mqeb/tests/test_capability_contract.py capabilities/common/mqeb/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/mqeb/tests/test_capability_contract.py capabilities/common/mqeb/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/mqeb --json
./.venv/bin/apg capabilities publish-plan capabilities/common/mqeb --json
```
