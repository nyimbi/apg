# PRED Implementation Plan

## Phase 1 - Contract

- Expand configuration into forecasting, scoring, feature sets, models,
  scenarios, drift, governance, observability, adapters, UI, and theme.
- Add deterministic guardrails for model, feature, forecast, score, scenario,
  drift, audit, cross-tenant, and Bytewax stream behavior.
- Expand UI routes and theme component metadata for generated apps.

## Phase 2 - Runtime

- Keep `PredService` as the generated-app runtime.
- Harden model, feature, forecast, score, scenario, and drift guardrails where
  the expanded contract needs executable evidence.
- Keep deterministic helpers in `predictive_runtime.py` so generated
  applications execute before live model providers are connected.

## Phase 3 - UI Models

- Extend route-specific view models for dashboard, forecasts, scores, features,
  scenarios, models, drift, batch scoring, explainability, governance, and audit.
- Ensure view models import only the contract and dependency-light runtime.

## Phase 4 - Package Evidence

- Replace static package evidence with contract-derived `app.semantic_model()`.
- Make `self_test()` validate route count, rule count, Bytewax streaming, and
  runtime service evidence.
- Refresh `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` from the current app entrypoint.

## Phase 5 - Review and Verification

- Rename stale package tests to package-contract language.
- Expand focused tests for contract shape, guardrails, runtime lifecycle, UI
  models, committed package evidence, and APG record compatibility.
- Run only PRED py-compile, focused pytest, package self-test, implementation
  audit, publish plan, stale-marker scan, and whitespace check.
- Record verification in `docs/progress_log.md`.
- Commit and push the verified slice using the Lore commit protocol.
