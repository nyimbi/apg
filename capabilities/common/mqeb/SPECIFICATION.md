# APG MQEB Capability Specification

## Purpose

Message Queue Event Bus (`mqeb`) is the APG event-fabric capability. It must
let generated APG applications define tenant-scoped topics, publish governed
messages, subscribe consumers, enforce delivery and retention policy, surface
operational state, and connect to streaming infrastructure without binding the
package runtime directly to a live broker.

MQEB must use **Bytewax as the preferred stream-processing and event-flow
runtime boundary**. It must not introduce Kafka as the platform dependency for
this capability. Kafka-compatible bridges can be future adapters, but the
first-class APG event fabric is APG package state plus Bytewax-oriented
pipelines.

The package must be terse enough for APG developers to use quickly, but precise
enough that generated ERP, workflow, AI-agent, security, data, and integration
applications can depend on it safely.

## First-Class Concepts

- **Topic**: tenant-scoped event stream with owner, classification, partitions,
  retention policy, delivery defaults, encryption posture, schema reference,
  and dead-letter configuration.
- **Message**: tenant-scoped event payload metadata with topic, producer,
  priority, idempotency key, schema version, encryption state, delivery state,
  publish evidence, and audit trail.
- **Subscription**: consumer binding to one or more topics with delivery mode,
  protocol, lag state, retry policy, dead-letter policy, and pause/resume
  lifecycle.
- **Delivery attempt**: attempt-level evidence for subscription delivery,
  retry, acknowledgement, failure, replay, and dead-letter routing.
- **Routing rule**: deterministic event-routing guardrail that can allow, deny,
  or require review based on topic state, tenant boundary, message class,
  delivery guarantees, encryption, schema, priority volume, and operational
  posture.
- **Priority quota exception**: reviewed exception that allows short-term burst
  publishing above tenant priority quotas.
- **Replay request**: reviewed replay action with bounded topic, time range,
  reason, actor, and evidence.
- **Event agent**: first-class AI or automation participant that can review
  routing, delivery, quota, replay, schema, Bytewax topology, or dead-letter
  decisions. Agents must declare runtime, role, owner, purpose, scope, machine
  contribution disclosure, and human approval posture.
- **Lifecycle batch**: tenant-scoped MQEB mutation batch that must declare
  Bytewax as the event-stream processor before generated applications compose
  it into event-fabric flows.
- **Operational audit event**: immutable package evidence for topic,
  subscription, publish, delivery, exception, replay, pause, resume, event
  agent, lifecycle batch, and dead-letter decisions.

## Functional Requirements

1. Every mutating package operation must require tenant context.
2. Topic creation must require topic ID/name, owner, classification, retention
   policy, and delivery policy.
3. Restricted and regulated topics must require encryption and schema
   references before publishing.
4. Publish operations must require an existing topic, producer identity,
   tenant match, non-empty payload metadata, and idempotency where configured.
5. Cross-tenant publish must be denied unless a future authorized exchange
   adapter explicitly records the policy decision.
6. Exactly-once or guaranteed delivery must require a configured dead-letter
   queue and idempotency key.
7. Subscription creation must require a valid topic pattern, consumer identity,
   delivery mode, protocol, retry policy, and dead-letter behavior.
8. Consumer pause, resume, and replay must be auditable state transitions.
9. Priority publish bursts above quota must require a reviewed quota exception.
10. Dead-letter routing must preserve the original message reference, failure
    reason, retry count, subscription, and audit evidence.
11. API helpers and dependency-light view models must expose dashboard,
    topics, publishing, subscriptions, routing, dead letters, replay,
    quotas, event agents, Bytewax lifecycle batches, monitoring, and settings
    state.
12. `app.py`, `semantic_model.json`, `release_report.json`, and
    `package_manifest.json` must reflect the live capability contract and must
    not drift behind the package runtime.
13. Event agents must be first-class APG citizens with supported runtime and
    role manifests. The first supported runtimes are Codex, Claude Code,
    opencode, and Pi.
14. Event agents must fail closed when runtime, role, owner, purpose, scope, or
    machine contribution disclosure is missing or unsupported.
15. Event agents in privileged roles must require human approval. Privileged
    roles include quota review, replay review, Bytewax topology review, and
    dead-letter triage.
16. Lifecycle batches must validate that the event stream processor is Bytewax
    and must reject empty batches.

## Rule Engine Requirements

The deterministic rule engine must cover at least:

- tenant context required;
- publish requires existing topic;
- restricted topic requires encryption;
- regulated topic requires schema reference;
- cross-tenant publish denied;
- exactly-once delivery requires dead-letter queue;
- exactly-once delivery requires idempotency key;
- disabled topic blocks publish;
- paused subscription blocks delivery;
- priority quota exhaustion requires review;
- replay requires bounded range and reason;
- dead-letter replay requires reviewer evidence;
- event-agent runtime and role must be supported;
- event-agent scope, owner, purpose, and contribution disclosure are required;
- privileged event-agent roles require human approval;
- lifecycle batches require Bytewax stream processing.

Rules must be expressible through the existing `CapabilityRuleEngine` pattern:
small, declarative conditions and deterministic allow/deny/review decisions.

## UI And Theming Requirements

MQEB must expose generated-app UI models for:

- event fabric dashboard;
- topic inventory and topic detail;
- publish workbench;
- subscription management;
- routing rule trace;
- dead-letter queue;
- replay console;
- priority quota review queue;
- event-agent roster and approval posture;
- consumer lag and delivery monitoring;
- Bytewax pipeline bridge and lifecycle-batch status;
- settings and adapter configuration.

The visual theme should remain compact and operational: dense tables,
timeline/lane views, lag meters, route traces, status chips, and clear
danger/warning states. It should not present MQEB as a marketing page.

## Adapter Boundaries

The dependency-light package runtime must not require live external systems.
Production integrations belong behind adapters that preserve the same package
contract:

- Bytewax dataflow workers and stream processors;
- APG AUTH, MTEN, AUDL, CONF, KEYM, ENCR, SECU, MONI, and HLTH;
- HTTP, WebSocket, MQTT, AMQP, gRPC, webhook, and event-file adapters;
- schema registry and metadata management;
- SIEM, SOAR, DLP, GRC, notification, and incident-response systems;
- cloud queue/event services only as adapters, not as the MQEB core dependency;
- Kafka bridge only as optional compatibility, not as the primary APG event
  fabric.

## Current Implementation Baseline

The current package already contains:

- `MQEBService` with async topic creation, publish, subscription, consume, and
  stats behavior;
- dependency-light `MqebService`, API helpers, view models, and optional legacy
  Flask integration surfaces;
- deterministic capability contract with event-fabric, event-agent, Bytewax,
  UI, theme, and rule manifests;
- semantic, release, and package manifest evidence;
- focused package tests for event-fabric state, first-class event agents,
  Bytewax lifecycle validation, and generated-app package shape.

The next packet must convert this into a package-backed lifecycle guardrail
slice like the recent SECU, ENCR, and KEYM work: dependency-light records and
helpers, stronger rule coverage, generated-app view models, refreshed semantic
evidence, and focused positive/negative tests.

## Target Lifecycle Packet

The first coherent MQEB packet will implement:

- topic lifecycle state with active/disabled/deprecated states;
- governed publish evaluation against topic existence, encryption, schema,
  tenant boundary, delivery mode, idempotency, and priority quota state;
- subscription lifecycle with active/paused state and lag evidence;
- delivery attempt records with retry/dead-letter outcomes;
- priority quota exception request and independent review;
- replay request scheduling and review evidence;
- event-agent registration with runtime, role, owner, purpose, scope,
  contribution disclosure, and human approval guardrails;
- Bytewax lifecycle-batch validation;
- operational audit events;
- dependency-light API helpers and view models;
- contract-derived semantic evidence and package proof.

## Focused Proof

Use the battery-conscious package proof while iterating:

```bash
./.venv/bin/python -m py_compile capabilities/common/mqeb/__init__.py capabilities/common/mqeb/models.py capabilities/common/mqeb/service.py capabilities/common/mqeb/api.py capabilities/common/mqeb/capability_contract.py capabilities/common/mqeb/app.py capabilities/common/mqeb/tests/test_capability_contract.py capabilities/common/mqeb/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/mqeb/tests/test_capability_contract.py capabilities/common/mqeb/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/mqeb --json
./.venv/bin/apg capabilities publish-plan capabilities/common/mqeb --json
```
