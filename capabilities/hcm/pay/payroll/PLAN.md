# Payroll Implementation Plan

## Delivery Goal

Build one coherent payroll lifecycle and guardrail packet that APG applications can compose immediately. The packet must include specification, executable contract, service, API helpers, UI models, theme metadata, Bytewax lifecycle metadata, AI-agent composition, focused tests, generated evidence, and documentation.

## Work Plan

1. Define the package specification.
   - Capture records, workflows, rule requirements, UI routes, event metadata, adapter boundaries, and acceptance criteria.
   - Keep the scope bounded to executable payroll behavior.

2. Replace the generated contract wrapper.
   - Publish explicit provides and requires.
   - Define configuration and schema.
   - Define deterministic rules for periods, pay groups, profiles, components, time imports, runs, lines, taxes, adjustments, approvals, payments, payslips, filings, agents, and Bytewax guardrails.
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
   - Test full lifecycle from period creation through payroll-agent registration.
   - Test API, views, app self-test, and publishable metadata.

6. Refresh package evidence.
   - Regenerate semantic model, package manifest, and release report from the executable app surface.
   - Ensure generated evidence matches current rules, routes, theme, agents, and streaming metadata.

7. Review and harden.
   - Remove stale generated planning material.
   - Remove stale promotional markers and generated baseline wording from the touched package.
   - Run focused package verification only.
   - Record evidence in `docs/progress_log.md`.
   - Commit and push the coherent verified slice.

## Review Checklist

- Tenant context is enforced.
- Pay periods, pay groups, profiles, and components are complete and tenant-scoped.
- Bank-transfer profiles require review.
- Overtime imports require approval.
- Negative line items require review.
- Adjustments, posting, payments, payslips, and tax filings enforce approvals and privacy basis.
- Bytewax is the only lifecycle event processor named by the contract.
- Agent runtimes and roles are constrained.
- Privileged agent actions require human approval.
- Documentation, tests, semantic metadata, manifest, and release report agree.

## Verification Commands

```bash
./.venv/bin/python -m py_compile capabilities/hcm/pay/payroll/__init__.py capabilities/hcm/pay/payroll/capability_contract.py capabilities/hcm/pay/payroll/service.py capabilities/hcm/pay/payroll/api.py capabilities/hcm/pay/payroll/views.py capabilities/hcm/pay/payroll/app.py capabilities/hcm/pay/payroll/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/hcm/pay/payroll/tests/test_package_contract.py
./.venv/bin/python capabilities/hcm/pay/payroll/app.py
./.venv/bin/apg capabilities inspect pay_payroll --json
./.venv/bin/apg capabilities publish-plan capabilities/hcm/pay/payroll --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/hcm/pay/payroll --json
```
