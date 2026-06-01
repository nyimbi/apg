# Digital Surveillance Implementation Plan

## Packet 1: Contract And Domain Model

- Define `intel_surveillance` metadata, supported types, dependencies,
  configuration, rule engine, UI routes, theme, and Bytewax lifecycle metadata.
- Model authority, program, monitored asset, sensor, observation, alert, risk
  assessment, referral, dissemination, review, and agent records.
- Keep live sensors and sensitive surveillance actions behind adapters.

## Packet 2: Executable Runtime

- Implement a tenant-scoped service that enforces deterministic rules before
  mutation.
- Key state by `(tenant_id, record_id)` to prevent cross-tenant collisions.
- Emit audit events with Bytewax processor metadata for accepted mutations.
- Enforce program, asset, and sensor authority alignment.
- Add AI-agent guardrails for supported runtimes, supported roles, human
  approval, and prohibited surveillance scopes.

## Packet 3: Composition Surface

- Add dependency-light API helpers for generated applications.
- Add dashboard, console, and agent workbench view models.
- Add app entrypoint, component manifest, semantic model, package manifest, and
  release evidence.

## Packet 4: Tests And Review

- Validate contract shape, routes, streaming, agents, and theme.
- Exercise the full surveillance lifecycle.
- Assert tenant isolation.
- Assert guardrail rejection for missing context, unsupported types, missing
  privacy review, missing calibration, authority mismatch, non-Bytewax batches,
  missing approvals, and prohibited agent scopes.
- Run focused package verification and APG implementation/lifecycle audits.

## Packet 5: Catalog And Progress

- Register `surveillance` as an implemented Intel sub-capability.
- Update capability catalog counts and Intel category purpose.
- Record implementation and review evidence in `docs/progress_log.md`.
