# SHDN Development Plan

## Objective

Make SHDN a complete APG capability packet that generated applications can compose for lifecycle target registration, shutdown planning, drain and quiescence, backup and restore evidence, shutdown execution, recovery, AI-assisted review, Bytewax events, UI surfaces, visual theming, documentation, and focused verification.

## Work Items

1. Specification
   - Define scope, domain model, workflows, rules, UI, theme, stream contract, adapters, and acceptance criteria.

2. Contract
   - Add `shdn_agents`, `observability`, and `adapters` configuration.
   - Add provided and required service lists.
   - Add Bytewax streaming metadata.
   - Add rules for dependency maps, actor identity, Bytewax event routing, recovery evidence, agent runtime, agent role, critical human approval, and batch lifecycle mutation.

3. Runtime
   - Add SHDN agent records and metadata-rich audit events.
   - Enforce new rules in service methods.
   - Add agent registration, critical action validation, and batch mutation validation.
   - Keep the runtime deterministic and dependency-light.

4. API and UI
   - Expose agent, policy, batch validation, and stream metadata surfaces.
   - Add agent workbench and policy center view models.

5. Package Evidence
   - Refresh `app.py`, `semantic_model.json`, `package_manifest.json`, and `release_report.json` from the contract.
   - Keep packet docs and domain files represented in the manifest.

6. Review and Verification
   - Compile package files.
   - Run focused SHDN tests.
   - Run implementation audit and publish-plan checks.
   - Run semantic probes for agents and Bytewax metadata.
   - Scan touched package files for stale markers and unsupported stream wording.
   - Run `git diff --check`.

## Deliberate Boundaries

- Do not call live deployment systems, health probes, backup engines, schedulers, or service meshes inside this package.
- Do not run a live Bytewax topology during focused verification.
- Do not run full repository tests while operating under battery constraints.

## Review Checklist

- Contract shape validates through the APG registry.
- Every lifecycle mutation carries tenant context.
- Shutdown execution requires quiescence, health, snapshot, actor, approval, and stream guardrails.
- Recovery requires incident/change evidence and post-shutdown health evidence.
- AI agents are explicit, scoped, and governed.
- Generated semantic/package evidence matches the contract.
