# AICR Capability Plan

## Build Strategy

Build one coherent AI control-plane lifecycle packet:

`provider -> service -> model -> evaluation -> model metric/drift review -> workflow/agent runtime -> AI agent -> Bytewax lifecycle batch -> inference request -> approval -> result -> audit`

The packet must be dependency-light and provider-neutral so APG applications can
compose AI services before live provider adapters are wired in.

## Steps

1. Contract
   - Expand configuration for providers, models, workflows, agent runtimes,
     observability, adapters, UI routes, and theme components.
   - Promote AI agents and Bytewax lifecycle batches to top-level contract
     citizens with explicit manifests.
   - Keep `service.AicrService` as the generated-app runtime.
   - Keep `bytewax` as event-stream adapter evidence.

2. Runtime
   - Extend `AicrService` with provider, model, model-metric, workflow, and
     agent-runtime lifecycle methods.
   - Add first-class AI-agent registration and Bytewax lifecycle-batch
     validation methods.
   - Keep existing high-risk inference approval behavior compatible.
   - Persist review-required model metrics, AI agents, inference approvals, and
     lifecycle decisions with policy evidence fields.
   - Enforce key guardrails at runtime.

3. API and UI
   - Extend `api_helpers.py` for provider, model, workflow, and agent runtime
     operations.
   - Add API helper and view-model coverage for first-class AI agents and
     lifecycle batch monitoring.
   - Expose pending-review queues for generated approval consoles.
   - Extend `views.py` with provider registry, model catalog, model metric
     console, workflow designer, agent runtime console, audit, and richer
     metrics models.

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
- Model metrics require a registered model, metric name, recorder identity, and
  drift-review evidence when drift is above threshold.
- High-risk and large-context inference remains gated by approval.
- Review-required outcomes keep matched rules, review reasons, and audit
  evidence after the human decision is recorded.
- Agent runtimes have explicit supported-runtime and tool-policy guardrails.
- First-class AI agents have explicit supported-runtime, supported-role, scope,
  owner, purpose, disclosure, and privileged approval guardrails.
- AICR lifecycle batches require Bytewax and reject broker-specific queue or other broker-first
  streams.
- Contract, runtime, semantic model, release report, and tests agree.
- Docs describe current executable behavior without overclaiming.
