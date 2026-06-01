# PRED Implementation Plan

## Phase 1 - Contract

- Expand configuration into forecasting, scoring, feature sets, models,
  scenarios, drift, agents, streaming, governance, observability, adapters, UI,
  and theme.
- Add deterministic guardrails for model, feature, forecast, score, scenario,
  drift, audit, cross-tenant, and Bytewax stream behavior.
- Add first-class AI prediction-agent composition metadata for provider-neutral
  runtimes, supported roles, privileged-role review, and required fields.
- Add Bytewax lifecycle stream metadata for model, feature-set, forecast, score,
  scenario, drift, explainability, and prediction-agent batch mutations.
- Expand UI routes and theme component metadata for generated apps.

## Phase 2 - Runtime

- Keep `PredService` as the generated-app runtime.
- Harden model, feature, forecast, score, scenario, and drift guardrails where
  the expanded contract needs executable evidence.
- Preserve review-required model, feature-set, forecast, and drift outcomes as
  `pending_review` records with matched rules and review reasons, while denial
  guardrails continue to block state acceptance.
- Add runtime records and operations for registering governed AI prediction
  agents and validating Bytewax lifecycle batches.
- Keep deterministic helpers in `predictive_runtime.py` so generated
  applications execute before live model providers are connected.

## Phase 3 - UI Models

- Extend route-specific view models for dashboard, forecasts, scores, features,
  scenarios, models, drift, batch scoring, explainability, agents, lifecycle,
  governance, and audit.
- Surface pending review queues in model, feature, forecast, drift, dashboard,
  and governance view models without re-evaluating historical predictions.
- Ensure view models import only the contract and dependency-light runtime.

## Phase 4 - Package Evidence

- Replace static package evidence with contract-derived `app.semantic_model()`.
- Make `self_test()` validate route count, rule count, first-class agents,
  Bytewax lifecycle streaming, and runtime service evidence.
- Refresh `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` from the current app entrypoint.

## Phase 5 - Review and Verification

- Rename stale package tests to package-contract language.
- Expand focused tests for contract shape, guardrails, runtime lifecycle, UI
  models, AI-agent composition, Bytewax lifecycle batches, committed package
  evidence, and APG record compatibility.
- Run only PRED py-compile, focused pytest, package self-test, implementation
  audit, publish plan, stale-marker scan, and whitespace check.
- Record verification in `docs/progress_log.md`.
- Commit and push the verified slice using the Lore commit protocol.
