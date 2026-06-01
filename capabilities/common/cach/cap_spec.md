# CACH Capability Specification Summary

The authoritative specification for CACH is `SPECIFICATION.md`. This file keeps
the historical `cap_spec.md` entry point aligned with the executable package.

## Current Contract

CACH is APG's cache governance and runtime-adapter capability. It provides:

- tenant-scoped namespace, entry, warming, eviction, cache-agent,
  lifecycle-batch, and audit records;
- deterministic guardrails for namespace registration, entry admission,
  freshness, encryption, tenant isolation, TTL limits, warming, memory pressure,
  eviction review, cache-agent composition, and Bytewax lifecycle batches;
- first-class cache-agent composition for Codex, Claude Code, opencode, and Pi;
- durable review evidence and pending-review queues for entries, warming plans,
  eviction reviews, privileged cache agents, lifecycle batches, and audit
  events;
- Bytewax-first lifecycle batch validation without requiring live stream
  workers in the package runtime;
- dependency-light service, API helper, view-model, semantic-model, release, and
  manifest evidence for generated APG applications.

## Development Rule

Update `SPECIFICATION.md` first, then `PLAN.md`, then implementation, tests, and
generated evidence. This file should remain a short synchronized summary rather
than a second competing source of truth.
