# IMEX Capability Specification

## Purpose

IMEX provides governed import, export, and migration lifecycle services for APG
applications. It is the bulk data movement layer that composes with CONN for
connectivity and ETLP for transformation planning.

## Scope

The current executable packet covers:

- Endpoint registration with CONN binding references.
- Mapping profiles with source profiling, schema mapping, and quality gate
  evidence.
- Import, export, and migration job creation.
- Preview validation and quality scoring.
- Production approval and quality review decisions.
- Transfer execution with checkpointing and monitoring guardrails.
- Completion with audit and quality evidence.
- Artifact publication, retention, replay, and purge lifecycle.
- Generated-app UI models and semantic package evidence.

Out of scope for this packet:

- Physical file transfer execution.
- Live ETLP pipeline execution.
- Live Bytewax stream processing.
- External vault, encryption, audit, monitoring, and registry adapters.
- Browser-rendered UI verification.

## Lifecycle

1. Register source and destination endpoints.
2. Create a mapping profile.
3. Create a transfer job.
4. Validate a preview.
5. Execute a transfer run.
6. Complete the run with audit and quality evidence.
7. Publish artifacts with checksum and retention policy.
8. Replay, retry, purge, or review as needed.

## Configuration

The contract defines tenant configuration for jobs, formats, validation,
security, orchestration, observability, adapters, UI routes, and theme tokens.
The generated-app runtime adapter is `imex_runtime.ImexService`; the event
stream adapter is `bytewax`.

## Rules

The deterministic rule engine includes guardrails for tenant context, ownership,
direction, endpoints, supported formats, source profiles, checksums, schema
mappings, PII policy, destination approval, preview validation, production
approval, encryption, monitoring, checkpointing, quality review, quarantine,
capacity review, retry, replay, scheduling, artifact publication, retention,
purge review, owner transfer, ETLP plan linkage, CONN binding, audit evidence,
and final quality evidence.

## UI Surfaces

IMEX exposes 12 generated-app UI routes:

- Dashboard
- Jobs
- Designer
- Mappings
- Monitor
- Validation
- Imports
- Exports
- Approvals
- Artifacts
- Audit
- Settings

## Acceptance Criteria

- Contract validates through the APG capability audit.
- Package publish plan reports no warnings.
- Runtime can execute a source-to-destination happy path.
- Runtime blocks unsafe missing-evidence paths.
- Package evidence is generated from the live contract.
- Primary docs do not contain stale baseline or marketing claims.
