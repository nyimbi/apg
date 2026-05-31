# NLPC Implementation Plan

## Phase 1 - Contract

- Preserve the existing language registry and African language coverage.
- Expand configuration into processing, languages, tasks, pipelines, annotation,
  model registry, agents, streaming, governance, observability, adapters, UI,
  and theme sections.
- Add deterministic rules for document, processing, pipeline, model, annotation,
  lexicon, audit, event-stream, language-coverage, NLP-agent, and lifecycle
  batch guardrails.
- Expand UI routes and theme component metadata for generated apps.

## Phase 2 - Runtime

- Add `NlpcService` as the dependency-light generated-app runtime.
- Implement in-memory document, processing run, pipeline, model, annotation
  project, annotation, lexicon, NLP-agent, lifecycle-batch, and audit records.
- Add deterministic NLP task behavior so applications are executable before
  external providers are wired.
- Keep adapters explicit so AICR, MLCM, Bytewax, AUDL, AUTH, MONI, and SRCH can
  replace local behavior without changing the APG contract.

## Phase 3 - UI Models

- Add route-specific view models for dashboard, processing, documents,
  pipelines, batches, annotations, review, models, languages, lexicons, search,
  agents, lifecycle, governance, and audit.
- Ensure view models import only the contract and dependency-light runtime.

## Phase 4 - Package Evidence

- Replace static package evidence with contract-derived `app.semantic_model()`.
- Make `self_test()` validate route count, rule count, Bytewax streaming, and
  runtime service evidence, first-class agent metadata, and lifecycle stream
  metadata.
- Refresh `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` from the current app entrypoint.

## Phase 5 - Review and Verification

- Rename stale package tests to package-contract language.
- Expand focused tests for contract shape, guardrails, runtime lifecycle, UI
  models, NLP-agent registration, lifecycle batch validation, language
  coverage, and package evidence.
- Run only NLPC py-compile, focused pytest, package self-test, implementation
  audit, publish plan, stale-marker scan, and whitespace check.
- Record verification in `docs/progress_log.md`.
- Commit and push the verified slice using the Lore commit protocol.

## Current Packet Focus

The current coherent slice is NLP-agent composition and Bytewax lifecycle
guardrails. It adds executable state for `NlpAgentRecord` and
`NlpcLifecycleBatchRecord`, contract-level `agents` and `streaming` manifests,
deterministic guardrails, roster and lifecycle view models, semantic-model
evidence, and focused regression coverage while leaving live provider SDKs and
Bytewax topology execution behind adapter boundaries.
