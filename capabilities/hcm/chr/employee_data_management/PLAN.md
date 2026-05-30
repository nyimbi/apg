# Employee Data Management Implementation Plan

## Delivery Goal

Build one coherent employee data lifecycle and guardrail packet that APG
applications can compose immediately. The packet must include specification,
executable contract, service, API helpers, UI models, theme metadata, Bytewax
lifecycle metadata, AI-agent composition, focused tests, generated evidence, and
documentation.

## Work Plan

1. Define the package specification.
   - Capture records, workflows, rule requirements, UI routes, event metadata,
     adapter boundaries, and acceptance criteria.
   - Keep the scope bounded to executable employee master-data behavior.

2. Replace the generated contract wrapper.
   - Publish explicit provides and requires.
   - Define configuration and schema.
   - Define deterministic rules for departments, positions, employees, personal
     information, contacts, history, skills, certifications, quality, agents,
     and Bytewax guardrails.
   - Define UI routes and theme tokens.

3. Implement the lifecycle service.
   - Keep the public service dependency-light.
   - Store records in tenant-scoped in-memory collections.
   - Enforce rules before state changes.
   - Emit audit-style lifecycle events using Bytewax metadata.
   - Preserve legacy service aliases and generic composition helpers.

4. Implement composition surfaces.
   - Expose API wrappers around service operations.
   - Expose framework-neutral screen models.
   - Expose publishable app semantic model, component manifest, and self-test.
   - Export public symbols through `__init__.py`.

5. Implement focused verification.
   - Validate contract shape.
   - Test rule-engine denials and review gates.
   - Test full lifecycle from department creation through employee-agent
     registration.
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
- Departments require code, name, owner, and cost center.
- Positions require same-tenant departments.
- Compensation-band positions require review.
- Employee profiles require valid identity and organization references.
- Non-executive employees require manager assignment.
- Sensitive status changes require review.
- Personal information requires privacy basis.
- Emergency contacts require required contact fields.
- Sensitive history events require reason; termination requires approval.
- Advanced skills require evidence.
- Expiring certifications require expiry date.
- High-severity data-quality issues require owner.
- Bytewax is the only lifecycle event processor named by the contract.
- Agent runtimes and roles are constrained.
- Privileged agent actions require human approval.
- Documentation, tests, semantic metadata, manifest, and release report agree.

## Verification Commands

```bash
./.venv/bin/python -m py_compile capabilities/hcm/chr/employee_data_management/__init__.py capabilities/hcm/chr/employee_data_management/capability_contract.py capabilities/hcm/chr/employee_data_management/service.py capabilities/hcm/chr/employee_data_management/api.py capabilities/hcm/chr/employee_data_management/views.py capabilities/hcm/chr/employee_data_management/app.py capabilities/hcm/chr/employee_data_management/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/hcm/chr/employee_data_management/tests/test_package_contract.py
./.venv/bin/python capabilities/hcm/chr/employee_data_management/app.py
./.venv/bin/apg capabilities inspect chr_employee_data_management --json
./.venv/bin/apg capabilities publish-plan capabilities/hcm/chr/employee_data_management --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/hcm/chr/employee_data_management --json
```
