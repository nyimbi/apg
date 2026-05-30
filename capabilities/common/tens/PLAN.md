# TENS Development Plan

## Objective

Make TENS a complete APG capability packet that generated applications can compose for legacy tenant registration, mapping, access-boundary validation, migration governance, deprecation, AI-assisted review, Bytewax events, UI surfaces, visual theming, documentation, and focused verification.

## Work Items

1. Specification
   - Define scope, domain model, workflows, rules, UI, theme, stream contract, adapters, and acceptance criteria.

2. Contract
   - Add `tens_agents`, `observability`, and `adapters` configuration.
   - Add provided and required service lists.
   - Add Bytewax streaming metadata.
   - Add rules for source-system lineage, compatibility scope, mapping streams, rollback plans, migration completion streams, role mapping, isolation validation, privileged review, agent runtime, agent role, privileged human approval, and batch mapping.

3. Runtime
   - Add TENS agent records and metadata-rich audit events.
   - Enforce new rules in service methods.
   - Add agent registration, privileged agent-action validation, and batch mapping validation.
   - Keep the runtime deterministic and dependency-light.

4. API and UI
   - Expose agent, policy, batch validation, and stream metadata surfaces.
   - Add agent workbench and policy center view models.

5. Package Evidence
   - Refresh `app.py`, `semantic_model.json`, `package_manifest.json`, and `release_report.json` from the contract.
   - Keep packet docs and domain files represented in the manifest.

6. Review and Verification
   - Compile package files.
   - Run focused TENS tests.
   - Run implementation audit and publish-plan checks.
   - Run semantic probes for agents and Bytewax metadata.
   - Scan touched package files for stale markers and unsupported stream wording.
   - Run `git diff --check`.

## Deliberate Boundaries

- Do not connect to live identity providers, tenant catalogs, migration engines, approval systems, or audit sinks inside this package.
- Do not run a live Bytewax topology during focused verification.
- Do not run full repository tests while operating under battery constraints.

## Review Checklist

- Contract shape validates through the APG registry.
- Every tenant mutation carries tenant context.
- Mappings and migrations enforce validation and stream guardrails.
- Access boundary validation covers auth, role mapping, isolation, and privileged access review evidence.
- AI agents are explicit, scoped, and governed.
- Generated semantic/package evidence matches the contract.
