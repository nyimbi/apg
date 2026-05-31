# SRCH Implementation Plan

## Phase 1 - Contract

- Expand configuration into indices, documents, indexing, query, ranking,
  facets, security, governance, observability, adapters, UI, and theme.
- Add deterministic guardrails for index, document, bulk, query, facet, access,
  audit, cross-tenant, retirement, and Bytewax stream behavior.
- Add first-class search-agent metadata, provider-neutral runtime constraints,
  privileged-role review rules, and Bytewax lifecycle batch constraints.
- Expand UI routes and theme component metadata for generated apps.

## Phase 2 - Runtime

- Keep `SrchService` as the generated-app runtime.
- Harden index, document, bulk, query, facet, audit, and APG record
  compatibility guardrails where the expanded contract needs executable
  evidence.
- Implement search-agent registration, lifecycle-batch validation, tenant
  isolation, summary counts, and audit events.
- Keep deterministic helpers in `search_runtime.py` so generated applications
  execute before live search or vector providers are connected.

## Phase 3 - UI Models

- Extend route-specific view models for dashboard, search, indices, documents,
  bulk indexing, facets, analytics, ranking, access review, governance, agents,
  lifecycle batches, audit, and settings.
- Ensure view models import only the contract and dependency-light runtime.

## Phase 4 - Package Evidence

- Replace static package evidence with contract-derived `app.semantic_model()`.
- Make `self_test()` validate route count, rule count, first-class agents,
  Bytewax lifecycle streaming, and runtime service evidence.
- Refresh `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` from the current app entrypoint.

## Phase 5 - Review and Verification

- Expand focused tests for contract shape, guardrails, runtime lifecycle, UI
  models, committed package evidence, and APG record compatibility.
- Run only SRCH py-compile, focused pytest, package self-test, implementation
  audit, publish plan, stale-marker scan, and whitespace check.
- Record verification in `docs/progress_log.md`.
- Commit and push the verified slice using the Lore commit protocol.
