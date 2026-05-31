# MDM Capability Plan

## Objective

Build one coherent lifecycle and guardrail packet for `common/mdm` so APG can
compose executable applications with master-data governance, first-class
AI/data-agent participation, and Bytewax lifecycle processing before
production adapters are attached.

## Sequence

### 1. Specification

- Define the entity, quality, duplicate, golden-record, cross-reference,
  publish, data-agent, Bytewax lifecycle, audit, UI, theme, and adapter
  boundaries.
- Separate dependency-light generated-app behavior from database-backed runtime
  behavior.
- Record non-goals so the packet is useful without overclaiming infrastructure
  that belongs to adapters.

### 2. Contract

- Expand tenant configuration for entity types, quality thresholds, matching,
  survivorship, governance, data agents, Bytewax lifecycle streams,
  integration, adapters, UI, and theme.
- Add deterministic guardrails for tenant context, supported entity types,
  business keys, restricted data, quality, duplicate review, survivorship,
  conflict review, cross references, retirement, review notes, supported data
  agent runtimes, supported data-agent roles, agent scope, owner, purpose,
  machine contribution disclosure, privileged-role human approval, and Bytewax
  lifecycle batches.
- Add UI routes for dashboard, entities, golden records, quality, duplicates,
  stewardship, lineage, cross references, publish, analytics, audit, adapters,
  data agents, lifecycle batches, and settings.

### 3. Control Plane

- Preserve the existing async `MDMService`.
- Add `MdmService` for generated applications and focused package tests.
- Implement in-memory lifecycle records for entities, quality assessments,
  duplicate candidates, golden records, merge requests, cross references,
  publish decisions, data agents, lifecycle batches, and audit events.
- Ensure all lifecycle methods evaluate rules and preserve matched-rule
  evidence.

### 4. Composition Surfaces

- Add API helper functions that call the generated-app control plane.
- Add view models for each UI route.
- Replace stale embedded semantic JSON with contract-derived `app.py` output.
- Update package manifest, semantic model, and release report.

### 5. Documentation

- Add root `README.md`.
- Add this plan.
- Add `SPECIFICATION.md`.
- Replace the old package summary with practical executable scope.

### 6. Review And Proof

- Expand focused package tests for contract shape, rule engine decisions,
  lifecycle behavior, view models, registration metadata, and generated app
  evidence.
- Run only focused battery-conscious proof:
  - `py_compile` for MDM packet files.
  - focused MDM pytest files.
  - APG implementation audit for `capabilities/common/mdm`.
  - APG publish plan for `capabilities/common/mdm`.
  - stale-marker search over current packet artifacts.
  - `git diff --check` over MDM and progress log files.

## Follow-On Work

- Connect `MdmService` decisions to durable persistence in `MDMService`.
- Add production Bytewax flow definitions for mastered entity and data-agent
  lifecycle events.
- Add real runtime adapter shims for Codex, Claude Code, opencode, Pi, and
  later AI-agent providers without making any one runtime mandatory.
- Add live adapter tests for matching, quality, metadata catalog, lineage graph,
  cache, audit, and security integration.
- Add rendered UI shells after APG generated-application targets stabilize.
- Add performance and concurrency benchmarks when running on AC power.
