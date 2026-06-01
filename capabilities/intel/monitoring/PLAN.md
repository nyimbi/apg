# Real-Time Monitoring Implementation Plan

## Packet 1: Contract And Domain Model

- Define `intel_monitoring` metadata, supported types, dependencies,
  configuration, rule engine, UI routes, theme, and Bytewax lifecycle metadata.
- Model authority, monitoring policy, source, watch, event, signal, incident,
  referral, dissemination, review, and agent records.
- Keep live stream connectors and response automation behind adapters.

## Packet 2: Executable Runtime

- Implement a tenant-scoped service that enforces deterministic rules before
  mutation.
- Key state by `(tenant_id, record_id)` to prevent cross-tenant collisions.
- Emit audit events with Bytewax processor metadata for accepted mutations.
- Enforce policy/source authority alignment and source access review.
- Add AI-agent guardrails for supported runtimes, supported roles, human
  approval, and prohibited automation scopes.

## Packet 3: Composition Surface

- Add dependency-light API helpers for generated applications.
- Add dashboard, console, and agent workbench view models.
- Add app entrypoint, component manifest, semantic model, package manifest, and
  release evidence.

## Packet 4: Tests And Review

- Validate contract shape, routes, streaming, agents, and theme.
- Exercise the full real-time monitoring lifecycle.
- Assert tenant isolation.
- Assert guardrail rejection for missing context, unsupported types, missing
  access review, authority mismatch, non-Bytewax batches, missing approvals,
  and prohibited agent scopes.
- Run focused package verification and APG implementation/lifecycle audits.

## Packet 5: Catalog And Progress

- Register `monitoring` as an implemented Intel sub-capability.
- Update capability catalog counts and Intel category purpose.
- Record implementation and review evidence in `docs/progress_log.md`.
