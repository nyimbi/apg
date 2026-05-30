# AICR Capability Plan

## Build Strategy

Build one coherent AI control-plane lifecycle packet:

`provider -> service -> model -> evaluation -> workflow/agent runtime -> inference request -> approval -> result -> audit`

The packet must be dependency-light and provider-neutral so APG applications can
compose AI services before live provider adapters are wired in.

## Steps

1. Contract
   - Expand configuration for providers, models, workflows, agent runtimes,
     observability, adapters, UI routes, and theme components.
   - Keep `service.AicrService` as the generated-app runtime.
   - Keep `bytewax` as event-stream adapter evidence.

2. Runtime
   - Extend `AicrService` with provider, model, workflow, and agent-runtime
     lifecycle methods.
   - Keep existing high-risk inference approval behavior compatible.
   - Enforce key guardrails at runtime.

3. API and UI
   - Extend `api_helpers.py` for provider, model, workflow, and agent runtime
     operations.
   - Extend `views.py` with provider registry, model catalog, workflow
     designer, agent runtime console, audit, and richer metrics models.

4. Package Evidence
   - Replace static app semantic evidence with contract-derived output.
   - Refresh `semantic_model.json`, `release_report.json`, and
     `package_manifest.json`.

5. Verification
   - Run focused py_compile, package tests, implementation audit, publish plan,
     stale marker scan, and diff checks.
   - Avoid full repository tests while on battery.

## Review Checklist

- Provider and model lifecycle rules are enforced.
- High-risk and large-context inference remains gated by approval.
- Agent runtimes have explicit supported-runtime and tool-policy guardrails.
- Contract, runtime, semantic model, release report, and tests agree.
- Docs describe current executable behavior without overclaiming.
