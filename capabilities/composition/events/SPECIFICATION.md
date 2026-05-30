# Event Streaming Bus Specification

## Intent

Event Streaming Bus makes event flow a governed APG composition primitive. It lets generated applications share stream definitions, schema contracts, publishing rules, subscriptions, Bytewax processors, dead-letter handling, replay approval, AI-agent review, UI models, theme contracts, and lifecycle events.

## Functional Requirements

- Register streams by tenant, owner, source capability, retention policy, partition key, and Bytewax stream name.
- Require schemas for streams carrying PII.
- Register event schemas and require review for breaking changes.
- Publish events only with source-capability attribution, correlation context, and Bytewax append.
- Validate batch publishing against size limits and Bytewax routing.
- Create subscriptions with consumer ownership and delivery mode.
- Require dead-letter streams for retrying subscriptions.
- Register stream processors on Bytewax with checkpoint configuration.
- Require review for stateful stream processors.
- Require approval before event replay.
- Register first-class event agents for Codex, Claude Code, OpenCode, and Pi.
- Expose dashboard, stream, schema, subscription, processor, dead-letter, agent, and settings UI models.

## Rule Engine

The deterministic rule engine enforces tenant context, write policy attachment, stream ownership, retention policy, PII schema requirements, Bytewax routing, breaking schema review, publish attribution, correlation, batch limits, subscription ownership, dead-letter requirements, processor review, checkpointing, Bytewax processor runtime, replay approval, agent runtime and role support, and human approval for privileged agent actions.

## Acceptance Criteria

- `get_capability_contract()` returns a valid APG contract with configuration, schema, rules, UI, theme, and Bytewax streaming metadata.
- Package import exposes `CompositionEventsService`, Bytewax runtime helpers, contract helpers, and registration metadata without web-framework imports.
- Service supports stream, schema, publish, batch, subscription, processor, agent, dashboard, audit, and compatibility record operations.
- API helpers and view models expose the same lifecycle surfaces.
- Semantic model includes `event_agents`, required dependencies, route metadata, rules, theme, and Bytewax stream metadata.
- Focused tests cover lifecycle success paths and guardrail failures.
