# CVSN Implementation Plan

## Phase 1 - Contract

- Replace shallow package contract metadata with configuration for processing,
  vision tasks, OCR, detection, video, quality, safety, privacy, model registry,
  governance, observability, adapters, UI, and theme.
- Add deterministic guardrails for asset ingestion, processing, quality,
  safety, biometrics, moderation, batching, video, model lifecycle, audit,
  cross-tenant controls, and Bytewax event streams.
- Add first-class AI vision-agent composition metadata for provider-neutral
  runtimes, supported roles, privileged-role review, and required fields.
- Add Bytewax lifecycle stream metadata for asset, job, pipeline, model,
  quality, safety, biometric, and vision-agent batch mutations.
- Expand UI routes and theme component metadata for generated apps.

## Phase 2 - Runtime

- Add `CvsnService` as the dependency-light generated-app runtime.
- Implement in-memory asset, job, model, pipeline, and audit records.
- Implement in-memory vision-agent and lifecycle-batch records.
- Add deterministic task behavior so applications are executable before live
  computer-vision providers are wired.
- Add runtime operations for registering governed AI vision agents and
  validating Bytewax lifecycle batches.
- Keep adapters explicit so AICR, MLCM, Bytewax, AUDL, AUTH, MONI, storage, and
  search can replace local behavior without changing the APG contract.

## Phase 3 - UI Models

- Add route-specific view models for dashboard, assets, documents, images,
  video, quality, safety, similarity, review, models, agents, lifecycle,
  governance, and audit.
- Ensure view models import only the contract and dependency-light runtime.

## Phase 4 - Package Evidence

- Replace static package evidence with contract-derived `app.semantic_model()`.
- Make `self_test()` validate route count, rule count, first-class agents,
  Bytewax lifecycle streaming, and runtime service evidence.
- Refresh `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` from the current app entrypoint.

## Phase 5 - Review and Verification

- Rename stale package tests to package-contract language.
- Expand focused tests for contract shape, guardrails, runtime lifecycle,
  AI-agent composition, Bytewax lifecycle batches, UI models, package evidence,
  and committed JSON evidence.
- Run only CVSN py-compile, focused pytest, package self-test, implementation
  audit, publish plan, stale-marker scan, and whitespace check.
- Record verification in `docs/progress_log.md`.
- Commit and push the verified slice using the Lore commit protocol.
