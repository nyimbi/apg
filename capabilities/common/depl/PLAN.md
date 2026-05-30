# DEPL Development Plan

## Objective

Turn DEPL into a coherent lifecycle and guardrail packet for executable APG
deployment-management applications.

## Build Steps

1. Specify the packet.
   - Define environment, release, rollback-plan, health-gate,
     deployment-plan, deployment-run, rollback-event, deployment-agent, audit,
     UI, theme, rule, and Bytewax stream requirements.
   - Keep live cloud, cluster, registry, ticketing, notification, and
     observability providers behind adapters.

2. Align the capability contract.
   - Add deployment-agent, governance, observability, adapter, UI, theme, and
     Bytewax stream configuration.
   - Add deterministic rules for the full deployment lifecycle and guardrails.
   - Ensure rule matching supports numeric and inequality suffixes.

3. Complete the executable runtime.
   - Add the `DeploymentAgent` model.
   - Extend `DeplService` with tenant-safe keys for duplicate IDs across
     tenants.
   - Add scoped deployment-agent registration, guarded plan state changes, and
     Bytewax batch-mutation validation.
   - Preserve environment, release, health, rollout, deployment, rollback, and
     audit flows.

4. Complete composition surfaces.
   - Extend API helpers for agents, plan state changes, batch mutation
     validation, listings, and status.
   - Extend view models for agents, audit, analytics, settings, and Bytewax
     stream metadata.
   - Update capability registration metadata, permissions, endpoints, optional
     dependencies, and capabilities.

5. Refresh package evidence.
   - Regenerate `app.py`, `semantic_model.json`, `package_manifest.json`, and
     `release_report.json` from the contract.
   - Confirm generated evidence includes Bytewax, deployment agents, routes,
     and expanded rules.

6. Review and verify.
   - Run focused compile checks and DEPL package tests.
   - Run package self-test, implementation audit, publish-plan, stale-marker
     search, and `git diff --check`.
   - Fix emergent issues before committing.

## Risks And Controls

- Deployment systems can produce external side effects. Keep this package local
  and side-effect free; attach live operations through explicit adapters.
- AI deployment agents can obscure release accountability. Require
  registration, scope, supported runtime/role, contribution disclosure, policy
  reference, and audit.
- Cross-tenant state can leak when IDs are reused. Store tenant-qualified keys
  while preserving user-facing IDs in records.
- Release rollout can bypass controls. Enforce signature, change-ticket,
  health-gate, approval, rollback, trace, and review evidence.
- Battery constraints limit verification scope. Run focused DEPL checks now and
  document broader live-adapter checks as not run.

## Completion Evidence

- Focused compile and pytest checks pass.
- Package self-test passes.
- Generated semantic model confirms:
  - `streaming.processor == "bytewax"`
  - supported deployment-agent runtimes include Codex, Claude Code, OpenCode,
    and Pi
  - `/depl/agents` is exposed
- Implementation audit reports no DEPL errors or warnings.
- Publish-plan reports DEPL is side-effect free.
- Stale-marker search returns no matches.
- Progress log records the packet and known verification gaps.
