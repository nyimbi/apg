# SEOP Development Plan

## Objective

Make SEOP a complete APG capability packet that generated applications can compose immediately for security operations, incident response, response governance, AI-assisted review, Bytewax lifecycle events, UI view models, visual theming, documentation, and focused verification.

## Work Items

1. Specification
   - Define scope, provided services, required services, domain model, workflows, rules, UI, theme, event stream, and acceptance criteria.

2. Contract
   - Extend configuration with `seop_agents`, `observability`, and `adapters`.
   - Add provided and required service lists.
   - Add Bytewax streaming metadata.
   - Add rules for stream governance, incident evidence, response actor, containment review, closure review, compliance mapping, agent runtime, agent role, and critical human approval.

3. Runtime
   - Add SEOP agent records.
   - Add metadata-rich audit events.
   - Enforce new rules inside service methods.
   - Preserve deterministic, dependency-light behavior.

4. API and UI
   - Add API helpers for agent registration and agent response validation.
   - Add view models for agent workbench and audit trail.
   - Expose stream metadata in dashboard and settings.

5. Package Evidence
   - Refresh `cap_spec.md`, `semantic_model.json`, `package_manifest.json`, `release_report.json`, and `app.py` from the contract.
   - Keep package docs listed in the manifest.

6. Review and Verification
   - Compile package files.
   - Run focused SEOP tests.
   - Run implementation audit and publish-plan checks.
   - Scan touched package files for stale capability markers and disallowed messaging.
   - Run `git diff --check`.

## Deliberate Boundaries

- Do not connect to live SIEM, SOAR, EDR, ticketing, compliance, or threat-intelligence systems inside this package.
- Do not run a live Bytewax topology in focused verification; validate contract and stream metadata instead.
- Do not broaden verification to the full repository while operating under battery constraints.

## Review Checklist

- Contract shape validates through the APG registry.
- Every public service method records tenant context.
- Critical response and closure paths require evidence.
- Agent composition is explicit and governed.
- Bytewax is the only event-stream processor named by the lifecycle contract.
- UI and theme surfaces are represented in view models and semantic package evidence.
