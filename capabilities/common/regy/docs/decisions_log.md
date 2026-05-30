# REGY Decisions Log

## Decision 001: Separate Generated-App Runtime From Production Runtime

**Decision**: Add `registry_runtime.py` as a dependency-light lifecycle runtime
while keeping `service.py`, `api.py`, and `views.py` available for production
and legacy integration.

**Rationale**: Generated APG applications need executable registry behavior
without optional service mesh, gateway, monitor, audit, cache, or Bytewax
dependencies.

**Impact**: Package tests can prove lifecycle behavior cheaply, and production
adapters can remain replaceable.

## Decision 002: Keep Guardrails Deterministic

**Decision**: Use explicit deterministic rule conditions and effects for
registry decisions.

**Rationale**: Capability composition needs reproducible `allow`, `deny`, and
`require_review` outcomes with matched rules and required actions.

**Impact**: Future optimization, AI ranking, or external policy engines must
honor these decisions before side effects.

## Decision 003: Treat Bytewax As The Event Stream Adapter

**Decision**: The REGY adapter manifest names Bytewax for event streaming.

**Rationale**: APG should avoid Kafka for this platform direction and keep
registry lifecycle events routed through Bytewax-compatible flows.

**Impact**: Production stream work belongs in adapters; generated-app runtime
only emits audit/event records.

## Decision 004: Replace Overclaiming Docs With Executable Scope

**Decision**: Primary docs describe implemented lifecycle behavior and adapter
boundaries instead of broad market or speculative technology claims.

**Rationale**: Capability packets should make current executable behavior clear
so new contributors can advance the platform from a reliable baseline.

**Impact**: Ambitious future work can be reintroduced as specific specs, tests,
and adapter implementations.
