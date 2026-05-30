# CACH Capability Development Plan

## Build Strategy

CACH already contains a broad async cache runtime and several advanced extension
modules. The immediate work is to bed down a coherent APG capability packet
around that runtime so generated applications can compose CACH safely today.

The packet will be narrow, executable, and reviewable:

1. Document the capability contract and operational boundary.
2. Expand the deterministic guardrails and generated-application UI contract.
3. Add a dependency-light lifecycle service for namespace, entry, warming, and
   eviction governance.
4. Add callable API helpers and view-model builders.
5. Replace stale semantic-model embedding with live contract-derived evidence.
6. Prove the packet with focused tests and publish-plan validation.

## Implementation Tasks

### 1. Specification and Documentation

- Add `SPECIFICATION.md` defining CACH outcomes, lifecycle entities, rules,
  UI/theme surface, integration boundaries, non-goals, and acceptance criteria.
- Replace the marketing-heavy README with a practical operator/developer guide.
- Keep existing deep-dive docs such as deployment and performance guides as
  supplementary material, not the packet source of truth.

### 2. Capability Contract

- Add lifecycle-oriented configuration for namespaces, entries, warming, memory
  pressure, eviction review, adapters, and audit.
- Add rules for disabled namespaces, TTL review, warming source registration,
  warming batch review, independent eviction review, and review notes.
- Add routes for namespaces, eviction reviews, adapters, and audit.
- Add theme components for entry freshness, eviction queue, adapter health, and
  audit events.

### 3. Lifecycle Service

- Preserve the existing async `CacheService`.
- Add a dependency-light `CacheGovernanceService` with dataclass records for:
  namespace policies, entry records, warming plans, eviction reviews, and audit
  events.
- Enforce the deterministic rule engine from `capability_contract.py`.
- Provide summaries for generated UI and release evidence.

### 4. API and View Models

- Add module-level lifecycle helper functions in `api.py` so generated APG apps
  can call CACH without booting optional web/runtime dependencies.
- Add `view_models.py` with compact generated-application models for dashboard,
  namespace inventory, entry explorer, warming console, eviction queue, topology,
  adapters, audit, and settings.

### 5. Packaging Evidence

- Update `app.py` to derive `semantic_model()` from the live capability contract.
- Refresh `semantic_model.json` and `release_report.json`.
- Update `package_manifest.json` with the new docs and view model artifact.

### 6. Verification and Review

- Run focused compile and package tests only.
- Run `apg capabilities publish-plan capabilities/common/cach --json`.
- Search for stale materialized-package markers and overclaiming README claims.
- Run `git diff --check`.
- Perform a focused code review and fix emergent issues before commit.

## Deferred Work

- Live Redis or Valkey adapter proof.
- Distributed invalidation over MQEB/Bytewax.
- Production persistence for policy and review records.
- Full APG auth/audit/monitoring integration.
- Full dashboard rendering and browser verification.
- Benchmark-driven performance claims.

These are adapter/runtime tasks. They must not block the composable APG
capability packet.
