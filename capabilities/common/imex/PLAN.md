# IMEX Capability Plan

## Build Strategy

Build one coherent lifecycle packet for generated applications:

`endpoint -> mapping -> job -> preview -> run -> completion -> artifact -> replay/purge`

The packet should be dependency-light, deterministic, and composable with the
existing heavier IMEX modules instead of replacing them wholesale.

## Steps

1. Contract
   - Expand tenant configuration, adapters, UI routes, theme components, and
     deterministic guardrails.
   - Add first-class transfer-agent and Bytewax lifecycle-batch configuration.
   - Include Bytewax as the event-stream adapter.
   - Expose `imex_runtime.ImexService` as the generated-app runtime.

2. Runtime
   - Add endpoint, mapping, job, run, artifact, review, and audit records.
   - Add transfer-agent records, lifecycle-batch records, guardrail-backed
     registration, Bytewax validation, summaries, and audit events.
   - Preserve policy decisions, matched rules, review reasons, and required
     actions across endpoint, mapping, job, run, artifact, review,
     transfer-agent, lifecycle-batch, and audit records.
   - Implement lifecycle operations and rule enforcement.
   - Keep the runtime dependency-light for compiler-generated apps.

3. API and UI
   - Preserve the existing Flask API surface.
   - Add generated-app helper functions backed by the runtime service,
     including transfer-agent and lifecycle-batch helpers.
   - Add `view_models.py` for composable UI screens, including agent roster and
     lifecycle-batch monitor views.
   - Add generated-app pending-review helpers and review-evidence metadata for
     dashboards, rosters, and settings surfaces.

4. Package Evidence
   - Replace static app semantics with contract-derived semantic output.
   - Refresh `semantic_model.json`, `release_report.json`, and
     `package_manifest.json`.
   - Rename package tests away from baseline terminology.

5. Tests and Review
   - Focus tests on contract shape, rule denials, happy-path lifecycle,
     review paths, artifact lifecycle, UI models, and package evidence.
   - Run py_compile, focused pytest, implementation audit, publish plan,
     stale-marker scan, and diff checks.

## Review Checklist

- Runtime enforces every declared critical guardrail.
- Runtime preserves durable review evidence for review-required and denied
  lifecycle paths.
- Contract rule count and route count match package evidence.
- UI models use runtime list methods and contract configuration.
- API helpers do not bypass the generated-app runtime.
- Docs describe current executable behavior without overclaiming.
- External agent runtimes and live Bytewax execution remain adapter-bound.
