# DIST Development Plan

## Objective

Turn DIST into a coherent lifecycle and guardrail packet for executable APG
distributed-computing applications.

## Build Steps

1. Specify the packet.
   - Define worker-pool, worker-node, job, partition, aggregation, scaling,
     compute-agent, audit, UI, theme, rule, and Bytewax stream requirements.
   - Keep live compute engines, queue systems, schedulers, caches, databases,
     and worker processes behind adapters.

2. Align the capability contract.
   - Add compute-agent, governance, observability, adapter, UI, theme, and
     Bytewax stream configuration.
   - Add deterministic rules for the full distributed-compute lifecycle and
     guardrails.
   - Ensure rule matching supports numeric and inequality suffixes.

3. Complete the executable runtime.
   - Add the `ComputeAgent` model.
   - Extend `DistService` with tenant-safe keys for duplicate IDs across
     tenants.
   - Add scoped compute-agent registration, guarded job state changes, and
     Bytewax batch-mutation validation.
   - Preserve worker-pool, worker, job, partition, aggregation, scaling, and
     audit flows.

4. Complete composition surfaces.
   - Extend API helpers for agents, job state changes, batch mutation
     validation, listings, and status.
   - Extend view models for agents, audit, analytics, settings, and Bytewax
     stream metadata.
   - Update capability registration metadata, permissions, endpoints, optional
     dependencies, and capabilities.

5. Refresh package evidence.
   - Regenerate `app.py`, `semantic_model.json`, `package_manifest.json`, and
     `release_report.json` from the contract.
   - Confirm generated evidence includes Bytewax, compute agents, routes, and
     expanded rules.

6. Review and verify.
   - Run focused compile checks and DIST package tests.
   - Run package self-test, implementation audit, publish-plan, stale-marker
     search, and `git diff --check`.
   - Fix emergent issues before committing.

## Risks And Controls

- Distributed compute systems can produce external side effects and high
  resource usage. Keep this package local and side-effect free; attach live
  execution through explicit adapters.
- AI compute agents can obscure operational accountability. Require
  registration, scope, supported runtime/role, contribution disclosure, policy
  reference, and audit.
- Cross-tenant state can leak when IDs are reused. Store tenant-qualified keys
  while preserving user-facing IDs in records.
- Partition fanout can overrun capacity. Enforce quota policy, partition count,
  worker health, and review for large partition plans.
- Battery constraints limit verification scope. Run focused DIST checks now and
  document broader live-adapter checks as not run.

## Completion Evidence

- Focused compile and pytest checks pass.
- Package self-test passes.
- Generated semantic model confirms:
  - `streaming.processor == "bytewax"`
  - supported compute-agent runtimes include Codex, Claude Code, OpenCode, and
    Pi
  - `/dist/agents` is exposed
- Implementation audit reports no DIST errors or warnings.
- Publish-plan reports DIST is side-effect free.
- Stale-marker search returns no matches.
- Progress log records the packet and known verification gaps.
