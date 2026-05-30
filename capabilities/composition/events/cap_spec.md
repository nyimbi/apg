# APG Event Streaming Bus Capability Specification

## Capability Metadata

- **Capability Code:** COMPOSITION_EVENTS
- **Capability Name:** Event Streaming Bus
- **Version:** 2.1.0
- **Category:** Composition
- **Runtime Target:** python
- **Primary Stream Processor:** Bytewax

## Purpose

Event Streaming Bus is the APG composition-layer event plane. It gives every capability a shared way to define streams, publish events, validate schemas, manage subscriptions, operate Bytewax processors, handle dead-letter records, replay approved events, register AI review agents, and expose operational UI models.

## Scope

The capability owns these executable surfaces:

- Stream registration with tenant, owner, source capability, retention policy, partition key, and Bytewax stream binding.
- Schema registration with compatibility review for breaking changes.
- Event publishing and batch-publish validation through Bytewax.
- Subscription lifecycle with consumer ownership, delivery mode, retry policy, and dead-letter guardrails.
- Processor topology registration with Bytewax runtime, stateful review, and checkpoint requirements.
- Event replay approval guardrails.
- First-class event agents for Codex, Claude Code, OpenCode, and Pi.
- UI contracts for dashboard, streams, schemas, subscriptions, processors, dead letters, agents, and settings.

## Lifecycle

1. Register an event schema when a stream carries governed data.
2. Create a stream with owner, retention policy, partition key, source capability, and Bytewax routing.
3. Publish events with source-capability attribution and correlation context.
4. Validate batch publishes against size limits and Bytewax routing.
5. Create subscriptions with consumer owners, delivery mode, retry settings, and dead-letter streams where needed.
6. Register processors on Bytewax with checkpoint configuration and stateful review where required.
7. Record dead-letter and replay operations with approval evidence.
8. Register event agents for stream, schema, processor, subscription, dead-letter, and replay review lanes.
9. Audit lifecycle events and expose operational UI models.

## Guardrails

- Tenant context is mandatory.
- Event-bus writes require policy attachment.
- Streams require owners, retention policies, partition keys, and Bytewax routing.
- PII streams require schemas.
- Breaking schema changes require review.
- Published events require source capability, correlation context, and Bytewax append.
- Batch publish is capped and requires Bytewax.
- Subscriptions require consumer owners.
- Retrying subscriptions require dead-letter streams.
- Stateful processors require review.
- Processors require checkpoints and Bytewax runtime.
- Event replay requires approval.
- Event agents require supported runtime and role.
- Privileged event actions proposed by agents require human approval.

## UI Contract

The capability exposes these APG routes:

- `/composition-events/dashboard`
- `/composition-events/streams`
- `/composition-events/schemas`
- `/composition-events/subscriptions`
- `/composition-events/processors`
- `/composition-events/dead-letters`
- `/composition-events/agents`
- `/composition-events/settings`

## Event Stream

- **Processor:** `bytewax`
- **Stream:** `apg.composition.events.lifecycle`
- **Key:** `tenant_id`
- **Events:** stream created, schema registered, event published, event batch published, subscription created, processor registered, dead-letter recorded, events replayed, event agent registered.

## Integration Requirements

- Requires `auth`, `audl`, `ntfy`, `registry`, and `composition_access`.
- Provides event stream registry, Bytewax event publishing, event schema registry, subscription lifecycle, stream processor topology, dead-letter operations, and event agents.
- Uses APG Python runtime surfaces: `service.py`, `api.py`, `views.py`, and `app.py`.
