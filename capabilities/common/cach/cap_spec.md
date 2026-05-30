# APG Cache Management Capability

CACH is APG's cache governance and runtime-adapter capability. It provides a
tenant-scoped control plane for namespace registration, cache entry admission,
freshness checks, warming plans, eviction reviews, deterministic rule decisions,
UI metadata, theming, and release evidence.

## Current Executable Runtime

The package contains:

- a deterministic capability contract in `capability_contract.py`
- a dependency-light lifecycle service in `service.py`
- direct generated-application helpers in `api.py`
- generated-application view models in `view_models.py`
- a live contract-derived semantic model in `app.py`
- focused package tests under `tests/`

The larger async cache runtime remains available for backend operations. The
control plane is intentionally importable without optional runtime systems such
as Redis, Flask-AppBuilder, AI services, or compression plugins.

## Lifecycle Records

CACH defines first-class records for:

- namespace policy and status
- cache entry metadata and admission decisions
- warming requests and review state
- eviction/capacity reviews
- cache lifecycle audit events

These records let generated APG applications compose CACH into larger systems
without hand-writing cache governance for each application.

## Guardrails

CACH evaluates cache actions through deterministic rules. Baseline guardrails
cover tenant context, namespace presence, namespace status, encryption for
sensitive/restricted/regulated data, cross-tenant access, critical stale reads,
TTL limits, warming source evidence, warming batch review, memory pressure, and
independent eviction review.

## Adapter Boundary

CACH is backend-neutral. Production adapters may bind to memory caches,
Redis-compatible stores, edge caches, CDNs, application query caches, or future
distributed cache systems. Adapters must honor CACH rule decisions, namespace
policies, tenant isolation, TTLs, encryption requirements, audit events, and
review outcomes.

## Verification Scope

The capability packet is verified with focused compile checks, package contract
tests, lifecycle service tests, publish-plan evidence, and stale-marker scans.
Full repository tests, live cache backends, APG auth/audit/monitoring adapters,
production persistence, and benchmark claims remain separate validation tasks.
