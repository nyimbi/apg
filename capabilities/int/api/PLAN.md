# Integration API Management Implementation Plan

## Delivery Goal

Build one coherent API management lifecycle and guardrail packet that APG
applications can compose immediately. The packet must include specification,
executable contract, service, API helpers, UI models, theme metadata, Bytewax
lifecycle metadata, AI-agent composition, focused tests, generated evidence, and
documentation.

## Work Plan

1. Define the package specification.
   - Capture records, workflows, rule requirements, UI routes, event metadata,
     adapter boundaries, and acceptance criteria.
   - Keep the scope bounded to executable API management behavior.

2. Replace the generated contract wrapper.
   - Publish explicit provides and requires.
   - Define configuration and schema.
   - Define deterministic rules for APIs, endpoints, policies, consumers, keys,
     subscriptions, deployments, analytics, agents, and Bytewax guardrails.
   - Define UI routes and theme tokens.

3. Implement the lifecycle service.
   - Keep the public service dependency-light.
   - Store records in tenant-scoped in-memory collections.
   - Enforce rules before state changes.
   - Emit audit-style lifecycle events using Bytewax metadata.
   - Preserve legacy service class aliases and generic composition helpers.

4. Implement composition surfaces.
   - Expose API wrappers around service operations.
   - Expose framework-neutral screen models.
   - Expose publishable app semantic model, component manifest, and self-test.
   - Export public symbols through `__init__.py`.

5. Implement focused verification.
   - Validate contract shape.
   - Test rule-engine denials and review gates.
   - Test full lifecycle from API registration through API-agent registration.
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
- External upstreams require review.
- Endpoints require same-tenant APIs.
- Policy configuration cannot be empty.
- Consumer email validation is enforced.
- API keys require scope and expiration.
- Subscriptions require approval.
- API approval requires an approver.
- Deployments require deployer identity.
- Production deployments require approval.
- Slow usage records require review.
- Bytewax is the only lifecycle event processor named by the contract.
- Agent runtimes and roles are constrained.
- Privileged agent actions require human approval.
- Documentation, tests, semantic metadata, manifest, and release report agree.

## Verification Commands

```bash
./.venv/bin/python -m py_compile capabilities/int/api/__init__.py capabilities/int/api/capability_contract.py capabilities/int/api/service.py capabilities/int/api/api.py capabilities/int/api/views.py capabilities/int/api/app.py capabilities/int/api/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/int/api/tests/test_package_contract.py
./.venv/bin/python capabilities/int/api/app.py
./.venv/bin/apg capabilities inspect int_api --json
./.venv/bin/apg capabilities publish-plan capabilities/int/api --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/int/api --json
```
