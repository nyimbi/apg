# MQEB Capability Specification Summary

The authoritative specification for MQEB is `SPECIFICATION.md`. This companion
file keeps the historical `cap_spec.md` entry point concise and aligned with
the executable package.

## Current Contract

MQEB is APG's package-backed event fabric. It provides:

- tenant-scoped topics, messages, subscriptions, delivery attempts, replay
  requests, quota exceptions, event agents, lifecycle batches, and audit events;
- deterministic guardrails for tenant context, topic state, encryption, schema,
  delivery guarantees, idempotency, priority quota review, replay review,
  subscription state, event-agent composition, and Bytewax lifecycle batches;
- first-class event-agent composition for Codex, Claude Code, opencode, and Pi;
- Bytewax-first stream processing boundaries with no broker dependency in the
  package runtime;
- dependency-light service, API helper, view-model, semantic-model, release,
  and manifest evidence for generated APG applications.

## Development Rule

Update `SPECIFICATION.md` first, then `PLAN.md`, then implementation, tests, and
generated evidence. This file should remain a short synchronized summary rather
than a second competing source of truth.
