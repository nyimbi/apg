# Intelligence Analytics Implementation Plan

## Packet 1: Contract And Domain Model

- Define `intel_analytics` metadata, supported types, dependencies,
  configuration, rule engine, UI routes, theme, and Bytewax lifecycle metadata.
- Model authority, analytic workspace, dataset, feature set, model, run,
  insight, dashboard, narrative, recommendation, review, and agent records.
- Keep warehouses, ML engines, notebooks, feature stores, model registries,
  graph writes, RAG indexing, and publication delivery behind adapters.

## Packet 2: Executable Runtime

- Implement a tenant-scoped service that enforces deterministic rules before
  mutation.
- Key state by `(tenant_id, record_id)` to prevent cross-tenant collisions.
- Emit audit events with Bytewax processor metadata for accepted mutations.
- Enforce dataset lineage, model validation, confidence scores, and approval
  gates before publishing analytic outputs.
- Add AI-agent guardrails for supported runtimes, supported roles, human
  approval, and prohibited automation scopes.

## Packet 3: Composition Surface

- Add dependency-light API helpers for generated applications.
- Add dashboard, console, and agent workbench view models.
- Add app entrypoint, component manifest, semantic model, package manifest, and
  release evidence.

## Packet 4: Tests And Review

- Validate contract shape, routes, streaming, agents, and theme.
- Exercise the full intelligence-analytics lifecycle.
- Assert tenant isolation.
- Assert guardrail rejection for missing context, unsupported types, missing
  authority, missing lineage, invalid confidence, missing model validation,
  non-Bytewax batches, missing approvals, and prohibited agent scopes.
- Run focused package verification and APG implementation/lifecycle audits.

## Packet 5: Catalog And Progress

- Register `analytics` as an implemented Intel sub-capability.
- Update capability catalog counts and Intel category purpose.
- Record implementation and review evidence in `docs/progress_log.md`.
