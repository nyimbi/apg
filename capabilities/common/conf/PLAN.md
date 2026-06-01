# APG CONF Capability Plan

## Slice

Implement the governed configuration promotion and drift remediation lifecycle as the first coherent CONF guardrail packet.

This advances CONF from a mostly aspirational production automation package to an executable APG foundation capability that generated applications can compose now.

Current hardening extends that packet with first-class configuration agents,
Bytewax lifecycle stream metadata, compact theme tokens, and a current
source-of-truth capability specification.

This slice also preserves durable review evidence for production changes, drift
remediation, privileged configuration agents, configuration batches, and audit
events so generated applications can compose review queues immediately.

## Implementation Steps

1. Add package-local lifecycle models:
   - configuration records
   - configuration change approvals
   - deployments
   - drift remediation reviews
   - audit events

2. Add dependency-light service state:
   - tenant-qualified stores
   - duplicate ID isolation by tenant
   - rule-backed record creation
   - rule-backed change request and decision
   - production deployment approval checks
   - rollback-plan enforcement
   - drift remediation review checks
   - summary and audit helpers

3. Add API helper functions:
   - shared package service
   - create/list record helpers
   - request/decide change helpers
   - deploy change helper
   - request/decide drift remediation helpers
   - dashboard status helper

4. Extend UI metadata helpers:
   - dashboard summary
   - change approval queue
   - deployment list
   - drift remediation queue
   - audit event list

5. Extend executable contract:
   - rule defaults
   - UI routes
   - theme components
   - capability configuration defaults

6. Refresh publish evidence:
   - semantic model derived from live contract
   - release report route/rule counts
   - package manifest includes SPECIFICATION and PLAN

7. Replace stale package tests:
   - rename legacy generated-package tests to package contract tests
   - add positive lifecycle coverage
   - add negative guardrail coverage
   - add tenant duplicate-ID isolation coverage
   - add API/view shared-state coverage

8. Extend composition hardening:
   - add CONF agent runtime and role configuration
   - add CONF agent registration, API helper, view model, route, and tests
   - add Bytewax stream metadata to contract, app semantic model, package
     manifest, service validation, and tests
   - add durable review evidence fields and pending-review queues for changes,
     drift remediations, privileged agents, denied batches, and audit events
   - align the UI border radius token with the current 8px APG standard
   - replace legacy `cap_spec.md` claims with a pointer to this specification

9. Review and focused proof:
   - py_compile package files
   - focused pytest package suite
   - implementation audit
   - publish plan
   - stale marker search
   - diff whitespace check

10. Preserve current review-evidence packet:
   - production changes and drift remediation requests enter `review_required`
   - privileged deployment/policy reviewer agents can enter `pending_review`
   - denied non-Bytewax batches persist evidence before `PermissionError`
   - API helpers and view models expose pending review and batch evidence

## Non-Goals For This Slice

- Live GitOps repository mutation.
- Live cloud provider deployment.
- Production persistence.
- Secret manager/HSM integration.
- Natural-language configuration generation.
- Full repository test suite.

Those remain adapter and platform-integration tasks after the executable lifecycle is stable.

## Review Risks

- Existing CONF files include large advanced modules; this slice must avoid entangling the dependency-light package contract with optional integrations.
- Existing `api.py` is Flask-oriented; helper functions must coexist with it without requiring a running Flask app.
- Existing `app.py` embeds stale semantic evidence; replace it with contract-derived evidence.
- Existing tests use stale generated-package naming; rename and align with current package terminology.
