# DTWN Development Plan

## Objective

Turn DTWN into a coherent lifecycle and guardrail packet for executable APG
digital-twin applications.

## Build Steps

1. Specify the packet.
   - Define twin, simulation-model, telemetry, topology, simulation,
     prediction, twin-agent, audit, UI, theme, rule, and Bytewax stream
     requirements.
   - Keep live IoT, geospatial, vision, machine-control, simulator,
     time-series, and prediction services behind adapters.

2. Align the capability contract.
   - Add twin-agent, governance, observability, adapter, UI, theme, and
     Bytewax stream configuration.
   - Add deterministic rules for the full digital-twin lifecycle and
     guardrails.
   - Ensure rule matching supports numeric and inequality suffixes.

3. Complete the executable runtime.
   - Add the `TwinAgent` model.
   - Extend `DtwnService` with tenant-safe keys for duplicate IDs across
     tenants.
   - Add scoped twin-agent registration, guarded twin status changes, and
     Bytewax batch-mutation validation.
   - Preserve twin, model, telemetry, topology, simulation, prediction, and
     audit flows.

4. Complete composition surfaces.
   - Extend API helpers for agents, twin status changes, batch mutation
     validation, listings, and status.
   - Extend view models for agents, audit, analytics, settings, and Bytewax
     stream metadata.
   - Update capability registration metadata, permissions, endpoints, optional
     dependencies, and capabilities.

5. Refresh package evidence.
   - Regenerate `app.py`, `semantic_model.json`, `package_manifest.json`, and
     `release_report.json` from the contract.
   - Confirm generated evidence includes Bytewax, twin agents, routes, and
     expanded rules.

6. Review and verify.
   - Run focused compile checks and DTWN package tests.
   - Run package self-test, implementation audit, publish-plan, stale-marker
     search, and `git diff --check`.
   - Fix emergent issues before committing.

## Risks And Controls

- Digital-twin systems can connect to physical assets and operational control
  paths. Keep this package local and side-effect free; attach live operations
  through explicit adapters.
- AI twin agents can obscure operational accountability. Require registration,
  scope, supported runtime/role, contribution disclosure, policy reference, and
  audit.
- Cross-tenant state can leak when IDs are reused. Store tenant-qualified keys
  while preserving user-facing IDs in records.
- Predictions can drive operational changes. Enforce model evidence,
  simulation approval, authenticated telemetry, and high-risk review.
- Battery constraints limit verification scope. Run focused DTWN checks now and
  document broader live-adapter checks as not run.

## Completion Evidence

- Focused compile and pytest checks pass.
- Package self-test passes.
- Generated semantic model confirms:
  - `streaming.processor == "bytewax"`
  - supported twin-agent runtimes include Codex, Claude Code, OpenCode, and Pi
  - `/dtwn/agents` is exposed
- Implementation audit reports no DTWN errors or warnings.
- Publish-plan reports DTWN is side-effect free.
- Stale-marker search returns no matches.
- Progress log records the packet and known verification gaps.
