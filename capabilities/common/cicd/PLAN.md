# CICD Development Plan

## Objective

Turn CICD into a coherent lifecycle and guardrail packet for executable APG
continuous integration and delivery applications.

## Build Steps

1. Specify the packet.
   - Define pipeline, build, artifact, gate, promotion, delivery-agent, audit,
     UI, theme, rule, and Bytewax stream requirements.
   - Keep live Git, build-runner, registry, scanner, and deployment providers
     behind adapters.

2. Align the capability contract.
   - Add delivery-agent, governance, observability, adapter, UI, theme, and
     Bytewax stream configuration.
   - Add deterministic rules for the full CI/CD lifecycle and guardrails.
   - Ensure rule matching supports numeric and inequality suffixes.

3. Complete the executable runtime.
   - Add the `DeliveryAgent` model.
   - Extend `CicdService` with tenant-safe keys for duplicate IDs across
     tenants.
   - Add scoped delivery-agent registration and guarded pipeline state changes.
   - Preserve pipeline, build, artifact, gate, promotion, and audit flows.

4. Complete composition surfaces.
   - Extend API helpers for agents, pipeline state changes, promotion
     approvals, listings, and status.
   - Extend view models for artifacts, gates, agents, audit, analytics,
     settings, and Bytewax stream metadata.
   - Update capability registration metadata, permissions, endpoints, optional
     dependencies, and capabilities.

5. Refresh package evidence.
   - Regenerate `app.py`, `semantic_model.json`, `package_manifest.json`, and
     `release_report.json` from the contract.
   - Confirm generated evidence includes Bytewax, delivery agents, routes, and
     expanded rules.

6. Review and verify.
   - Run focused compile checks and CICD package tests.
   - Run package self-test, implementation audit, publish-plan, stale-marker
     search, and `git diff --check`.
   - Fix emergent issues before committing.

## Risks And Controls

- CI/CD systems can produce external side effects. Keep this package local and
  side-effect free; attach live operations through explicit adapters.
- AI delivery agents can obscure release accountability. Require registration,
  scope, supported runtime/role, contribution disclosure, policy reference, and
  audit.
- Cross-tenant state can leak when IDs are reused. Store tenant-qualified keys
  while preserving user-facing IDs in records.
- Release promotion can bypass controls. Enforce signature, quality gate,
  approval, environment policy, and separation of duties.
- Battery constraints limit verification scope. Run focused CICD checks now and
  document broader live-adapter checks as not run.

## Completion Evidence

- Focused compile and pytest checks pass.
- Package self-test passes.
- Generated semantic model confirms:
  - `streaming.processor == "bytewax"`
  - supported delivery-agent runtimes include Codex, Claude Code, OpenCode, Pi
  - `/cicd/agents` is exposed
- Implementation audit reports no CICD errors or warnings.
- Publish-plan reports CICD is side-effect free.
- Stale-marker search returns no matches.
- Progress log records the packet and known verification gaps.

