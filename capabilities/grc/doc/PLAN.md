# Document Management Implementation Plan

## Delivery Goal

Build one coherent document lifecycle and guardrail packet that APG applications
can compose immediately. The packet must include specification, executable
contract, service, API helpers, UI models, theme metadata, Bytewax lifecycle
metadata, AI-agent composition, focused tests, generated evidence, and
documentation.

## Work Plan

1. Define the package specification.
   - Capture records, workflows, rule requirements, UI routes, event metadata,
     adapter boundaries, and acceptance criteria.
   - Keep the scope bounded to executable document governance behavior.

2. Replace the generated contract wrapper.
   - Publish explicit provides and requires.
   - Define configuration and schema.
   - Define deterministic rules for documents, templates, revisions, approvals,
     publication, retention, access, processing, agents, and Bytewax guardrails.
   - Define UI routes and theme tokens.

3. Implement the lifecycle service.
   - Keep the public service dependency-light.
   - Store records in tenant-scoped in-memory collections.
   - Enforce rules before state changes.
   - Emit audit-style lifecycle events using Bytewax metadata.
   - Preserve legacy import aliases and generic composition helpers.

4. Implement composition surfaces.
   - Expose API wrappers around service operations.
   - Expose framework-neutral screen models.
   - Expose publishable app semantic model, component manifest, and self-test.
   - Export public symbols through `__init__.py`.

5. Implement focused verification.
   - Validate contract shape.
   - Test rule-engine denials and review gates.
   - Test full lifecycle from template through document-agent registration.
   - Test API, views, app self-test, and publishable metadata.

6. Refresh package evidence.
   - Regenerate semantic model, package manifest, and release report from the
     executable app surface.
   - Ensure generated evidence matches current rules, routes, theme, agents,
     and streaming metadata.

7. Review and harden.
   - Remove stale generated planning material.
   - Remove stale promotional markers and generated baseline wording from the
     touched package.
   - Run focused package verification only.
   - Record evidence in `docs/progress_log.md`.
   - Commit and push the coherent verified slice.

## Review Checklist

- Tenant context is enforced.
- Restricted documents require review.
- Templates cannot be empty.
- Published-document revisions require review.
- Restricted-document approval enforces segregation of duties.
- Publishing requires approval.
- Retention meets the minimum duration.
- Legal hold blocks archive.
- Restricted access grants require expiry.
- Processing jobs require Bytewax metadata.
- Agent runtimes and roles are constrained.
- Privileged agent actions require human approval.
- Documentation, tests, semantic metadata, manifest, and release report agree.

## Verification Commands

```bash
./.venv/bin/python -m py_compile capabilities/grc/doc/__init__.py capabilities/grc/doc/capability_contract.py capabilities/grc/doc/service.py capabilities/grc/doc/api.py capabilities/grc/doc/views.py capabilities/grc/doc/app.py capabilities/grc/doc/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/grc/doc/tests/test_package_contract.py
./.venv/bin/python capabilities/grc/doc/app.py
./.venv/bin/apg capabilities inspect grc_doc --json
./.venv/bin/apg capabilities publish-plan capabilities/grc/doc --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/grc/doc --json
```
