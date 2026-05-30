# Help and Knowledge Base Capability Plan

## Objective

Bring `help` to the current APG capability-packet standard: readable local documentation, a complete executable contract, a dependency-light runtime, UI/theming metadata, deterministic guardrails, generated package evidence, and focused verification.

## Implementation Packets

1. Document the intended lifecycle.
   - Add `README.md`.
   - Add `SPECIFICATION.md`.
   - Keep `cap_spec.md` as a compatibility pointer to the current docs and runtime.

2. Expand the executable contract.
   - Add source, feedback, localization, observability, adapter, and Bytewax event-stream configuration.
   - Expand deterministic rules for sources, articles, answers, search, feedback, localization, curation, audit, tenant isolation, and batch mutations.
   - Add source, localization, and audit UI routes plus theme components.

3. Strengthen runtime behavior.
   - Extend `HelpService` with source records, localization records, curation closing, and audit events.
   - Enforce source approval, title/body ownership, publication approval, answer citation, RBAC, rating, localization, curation, and Bytewax guardrails.
   - Keep all behavior dependency-light and deterministic.

4. Align composition surfaces.
   - Update API helpers for source, localization, and curation operations.
   - Update view models for source registry, localization, audit, and settings screens.
   - Update registration metadata and permissions.

5. Refresh package evidence.
   - Regenerate `app.py`, `semantic_model.json`, `package_manifest.json`, and `release_report.json` from the current contract.
   - Ensure the runtime semantic model matches the committed JSON artifact.

6. Review and verify.
   - Run focused package compile and tests.
   - Run implementation audit and publish-plan for `capabilities/common/help`.
   - Scan for stale generated or marketing language.
   - Run `git diff --check` for the changed package and progress log.

## Test Strategy

- Contract tests prove configuration, rule count, routes, theme, registration, and Bytewax controls.
- Runtime tests prove source, article, publication, search, answer, localization, feedback, curation, audit, and dashboard behavior.
- Guardrail tests prove tenant, owner, source, publication, RBAC, citation, rating, localization, curation, and Bytewax denial behavior.
- API/view tests prove generated applications can compose the capability without private implementation knowledge.

## Review Checklist

- No live search, RAG, or database infrastructure is started from package import or self-test.
- Tenant IDs are required and records are listed by tenant.
- Guardrails in the contract are enforced by runtime methods where applicable.
- Cited answers cannot be generated without approved sources.
- Low feedback opens curation, and curation closure requires reviewer and evidence.
- Bytewax is the only configured batch event-stream adapter.
- UI routes and theme components cover every major lifecycle surface.

## Out Of Scope

- Live vector search or RAG inference.
- Full-text production index integration.
- Identity, notification, chat, or audit-provider calls.
- Live Bytewax topology.
- Rendered browser UI.
- Full repository test suite.
