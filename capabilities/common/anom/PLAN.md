# ANOM Implementation Plan

## Phase 1 - Contract

- Expand configuration into sources, detection, baselines, signals,
  investigations, feedback, governance, observability, adapters, UI, and theme.
- Add deterministic guardrails for source, baseline, detection, signal,
  investigation, feedback, audit, cross-tenant, alert, reset, and Bytewax stream
  behavior.
- Add first-class anomaly-agent metadata, provider-neutral runtime constraints,
  privileged-role review rules, and Bytewax lifecycle batch constraints.
- Expand UI routes and theme component metadata for generated apps.

## Phase 2 - Runtime

- Keep `AnomService` as the generated-app runtime.
- Harden source, baseline, detection, investigation, feedback, reset, and audit
  guardrails where the expanded contract needs executable evidence.
- Implement anomaly-agent registration, lifecycle-batch validation, tenant
  isolation, summary counts, and audit events.
- Keep deterministic helpers in `anomaly_engine.py` so generated applications
  execute before live monitoring, prediction, or incident tools are connected.

## Phase 3 - UI Models

- Extend route-specific view models for dashboard, sources, baselines, detector,
  signals, investigations, alerts, rules, feedback, quality, agents, lifecycle
  batches, audit, and settings.
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
- Run only ANOM py-compile, focused pytest, package self-test, implementation
  audit, publish plan, stale-marker scan, and whitespace check.
- Record verification in `docs/progress_log.md`.
- Commit and push the verified slice using the Lore commit protocol.
