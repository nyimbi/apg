# APG MTEN Capability Plan

## Slice

Implement the governed tenant provisioning, capacity approval, isolation
suspension, and live migration lifecycle as the first coherent MTEN guardrail
packet.

This turns MTEN into an executable APG foundation capability that generated
applications can compose without booting the full async production manager,
cloud orchestration, analytics, AI, or web stacks.

## Implementation Steps

1. Add dependency-light lifecycle records:
   - tenant environments
   - capacity approvals
   - isolation incidents
   - live migration requests
   - governance events

2. Add `mten_runtime.py`:
   - tenant-qualified stores
   - tenant registration
   - capacity approval request/decision
   - activation with DNS/capacity/isolation guardrails
   - isolation incident suspension and reactivation
   - live migration request/decision/execution evidence
   - summary and governance event helpers

3. Add package API and view helper surfaces:
   - `api_helpers.py`
   - `view_models.py`
   - shared default service state
   - dashboard, provisioning, capacity, isolation, migration, and governance
     models

4. Extend executable contract:
   - rules for capacity review, independent reviewers, encrypted isolation,
     isolation suspension, and live migration review
   - routes for capacity approvals, isolation incidents, live migrations, and
     audit timeline
   - theme components for approval queues, isolation incidents, migration
     runbooks, and governance timeline

5. Refresh package evidence:
   - contract-derived `semantic_model.json`
   - `release_report.json`
   - `package_manifest.json`
   - `cap_spec.md`

6. Replace stale package tests:
   - rename legacy generated-package test to package contract test
   - add positive tenant lifecycle coverage
   - add negative DNS, capacity, independent-review, suspension, runbook, and
     tenant-isolation coverage
   - add API-helper/view-model shared-state coverage

7. Review and focused proof:
   - py_compile package files
   - focused pytest package suite
   - implementation audit
   - publish-plan
   - stale marker search
   - diff whitespace check

## Non-Goals For This Slice

- Live cloud tenant provisioning.
- Live DNS validation.
- Live service-mesh or IAM mutation.
- Billing integration.
- Production persistence.
- AI optimization execution.
- Full repository test suite.

Those remain adapters after the executable lifecycle is stable.

## Review Risks

- Existing MTEN imports optional FastAPI and Flask-AppBuilder surfaces; package
  proof should exercise dependency-light helpers instead.
- Existing `app.py` embeds stale semantic evidence; replace it with
  contract-derived evidence.
- Existing tests use stale generated-package naming; rename and align with the
  current package contract terminology.
- Capacity approval must be backed by package state; caller booleans are not
  governance evidence.
