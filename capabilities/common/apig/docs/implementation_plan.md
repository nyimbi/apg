# APIG Implementation Plan

The current APIG implementation plan is the root `PLAN.md`. This document keeps
the runtime-adapter backlog separate from the package-backed lifecycle packet.

## Completed Package Work

- Contract expansion for upstream, consumer, route, traffic, security, edge,
  canary, deployment, governance, observability, adapter, UI, and theme
  sections.
- Dependency-light `ApigService` lifecycle workflows.
- Generated API helpers and UI view models.
- Contract-derived semantic model and release evidence.
- Focused rule, lifecycle, API, UI, package, and app tests.

## Runtime Adapter Plan

- Bind reverse proxy, ingress, or service mesh configuration to APIG route
  decisions.
- Bind APG service discovery, auth/RBAC, key/certificate management, audit,
  metrics, cache, and event streaming adapters.
- Validate signed edge filters in the selected WebAssembly runtime.
- Add live gateway smoke tests for route activation, high quotas, canary
  rollout, rollback, deployment gates, and retirement.
- Add rendered UI checks and dedicated performance/resilience tests.
