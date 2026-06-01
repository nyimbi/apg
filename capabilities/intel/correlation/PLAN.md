# Data Correlation Implementation Plan

## Packet 1: Contract And Domain Model

- Define `intel_correlation` metadata, supported types, dependencies,
  configuration, rule engine, UI routes, theme, and Bytewax lifecycle metadata.
- Model authority, workspace, source, entity, observation, rule, run, cluster,
  decision, referral, review, and agent records.
- Keep live matching engines, graph writes, RAG indexing, geospatial joins, and
  referral delivery behind adapters.

## Packet 2: Executable Runtime

- Implement a tenant-scoped service that enforces deterministic rules before
  mutation.
- Key state by `(tenant_id, record_id)` to prevent cross-tenant collisions.
- Emit audit events with Bytewax processor metadata for accepted mutations.
- Enforce source lineage, rule thresholds, confidence scores, decision
  approvals, and referral approvals before mutation.
- Add AI-agent guardrails for supported runtimes, supported roles, human
  approval, and prohibited automation scopes.

## Packet 3: Composition Surface

- Add dependency-light API helpers for generated applications.
- Add dashboard, console, and agent workbench view models.
- Add app entrypoint, component manifest, semantic model, package manifest, and
  release evidence.

## Packet 4: Tests And Review

- Validate contract shape, routes, streaming, agents, and theme.
- Exercise the full data-correlation lifecycle.
- Assert tenant isolation.
- Assert guardrail rejection for missing context, unsupported types, missing
  authority, missing lineage, invalid thresholds, invalid confidence,
  non-Bytewax batches, missing approvals, and prohibited agent scopes.
- Run focused package verification and APG implementation/lifecycle audits.

## Packet 5: Catalog And Progress

- Register `correlation` as an implemented Intel sub-capability.
- Update capability catalog counts and Intel category purpose.
- Record implementation and review evidence in `docs/progress_log.md`.
