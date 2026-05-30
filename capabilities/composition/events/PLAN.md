# Event Streaming Bus Development Plan

## Slice Goal

Deliver a coherent lifecycle and guardrail packet for `composition_events` so the capability is executable, documented, testable, theme-aware, AI-agent aware, and Bytewax-aligned.

## Implementation Steps

1. Replace the generic contract with a domain-specific contract for streams, schemas, publishing, subscriptions, processors, dead letters, agents, governance, UI, theme, adapters, and Bytewax lifecycle streaming.
2. Preserve the existing Bytewax runtime services and add a dependency-light APG lifecycle facade for focused package tests and generated application composition.
3. Replace package API, views, app, and registration entrypoints with dependency-light surfaces.
4. Refresh package evidence from the active contract.
5. Add specification and plan documents, and update the README where needed.
6. Expand focused tests around contract shape, rules, service lifecycle, guardrail failures, API/view surfaces, app self-test, and semantic metadata.
7. Run focused verification and commit the coherent packet.

## Review Checklist

- Tenant context is enforced.
- Stream ownership, retention policy, partition key, and Bytewax routing are required.
- PII streams require schemas.
- Event publishing requires source capability, correlation context, and Bytewax.
- Batch publishing enforces size limits.
- Retrying subscriptions require dead-letter streams.
- Stateful processors require review and checkpoints.
- Processors use Bytewax.
- Event agents support Codex, Claude Code, OpenCode, and Pi.
- Package imports remain dependency-light.

## Deferred Work

- Deploy live distributed Bytewax topologies.
- Bind durable event storage, schema stores, and audit sinks.
- Render browser UI screens and run visual checks.
- Run load and performance tests after battery constraints are lifted.
