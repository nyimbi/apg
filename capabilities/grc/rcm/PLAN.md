# Risk and Compliance Management Implementation Plan

## Delivery Goal

Build one coherent RCM lifecycle and guardrail packet that APG applications can
compose immediately. The packet must include specification, executable contract,
service, API helpers, UI models, theme metadata, Bytewax lifecycle metadata,
AI-agent composition, focused tests, generated evidence, and documentation.

## Work Plan

1. Define the package specification.
   - Capture records, workflows, rule requirements, UI routes, event metadata,
     adapter boundaries, and acceptance criteria.
   - Keep the scope bounded to executable RCM lifecycle behavior.

2. Replace the generated contract wrapper.
   - Publish explicit provides and requires.
   - Define configuration and schema.
   - Define deterministic rules for risk, control, obligation, assessment,
     evidence, issue, governance, exception, agent, and Bytewax guardrails.
   - Define UI routes and theme tokens.

3. Implement the lifecycle service.
   - Keep the top-level service dependency-light.
   - Store records in tenant-scoped in-memory collections.
   - Enforce rules before state changes.
   - Emit audit-style lifecycle events using Bytewax metadata.
   - Preserve generic composition helpers for APG package probes.

4. Implement composition surfaces.
   - Expose API wrappers around service operations.
   - Expose framework-neutral screen models.
   - Expose publishable app semantic model, component manifest, and self-test.
   - Export public symbols through `__init__.py`.

5. Implement focused verification.
   - Validate contract shape.
   - Test rule-engine denials and review gates.
   - Test the full lifecycle from risk through agent registration.
   - Test API, views, app self-test, and publishable metadata.

6. Refresh package evidence.
   - Regenerate semantic model, package manifest, and release report from the
     executable app surface.
   - Ensure generated evidence matches current rules, routes, theme, and
     streaming metadata.

7. Review and harden.
   - Remove stale generated planning material.
   - Remove stale promotional markers and unrelated old target language from the
     touched package.
   - Run focused package verification only.
   - Record evidence in `docs/progress_log.md`.
   - Commit and push the coherent verified slice.

## Review Checklist

- Tenant context is enforced.
- Cross-tenant linked records are rejected.
- High-risk and high-severity paths require review evidence.
- Failed assessments require evidence.
- Evidence encryption and retention are enforced.
- Exceptions require supported type, approval, and expiration.
- Agent runtimes and roles are constrained.
- Privileged agent actions require human approval.
- Bytewax is the only lifecycle event processor named by the contract.
- Documentation, tests, semantic metadata, manifest, and release report agree.

## Verification Commands

```bash
./.venv/bin/python -m py_compile capabilities/grc/rcm/__init__.py capabilities/grc/rcm/capability_contract.py capabilities/grc/rcm/service.py capabilities/grc/rcm/api.py capabilities/grc/rcm/views.py capabilities/grc/rcm/app.py capabilities/grc/rcm/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/grc/rcm/tests/test_package_contract.py
./.venv/bin/python capabilities/grc/rcm/app.py
./.venv/bin/apg capabilities inspect grc_rcm --json
./.venv/bin/apg capabilities publish-plan capabilities/grc/rcm --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/grc/rcm --json
```
