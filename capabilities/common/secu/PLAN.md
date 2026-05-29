# APG SECU Capability Plan

## Slice

Implement the policy exception and incident response lifecycle as the next
coherent SECU guardrail packet.

SECU already has a dependency-light policy, device, threat, assessment, and
compliance runtime. This slice completes the security foundation by making
exceptions and incident response first-class composable workflows.

## Implementation Steps

1. Add package-local records:
   - `PolicyExceptionRecord`
   - `SecurityIncidentRecord`

2. Extend `SecuService`:
   - tenant-qualified exception and incident stores
   - exception request/decision workflow
   - expired exception denial
   - independent reviewer and reviewer-note enforcement
   - incident opening, containment, and resolution workflow
   - critical incident containment enforcement
   - dashboard summary and list helpers

3. Extend API helpers:
   - request/decide policy exception
   - open/contain/resolve incident
   - list posture surfaces with exceptions and incidents

4. Extend view models:
   - policy exception queue
   - incident response console
   - device quarantine console
   - audit timeline

5. Extend executable contract:
   - rules for exception review, expired exception, critical incident
     containment, and incident resolution containment
   - routes for exceptions, incidents, quarantine, and audit
   - theme components for exception queue, incident response, quarantine, and
     audit evidence

6. Refresh package evidence:
   - `app.py` contract-derived semantic model
   - `semantic_model.json`
   - `release_report.json`
   - `package_manifest.json`
   - `cap_spec.md`

7. Extend tests:
   - contract route/rule/theme coverage
   - positive policy-device-threat-assessment-compliance-exception-incident
     lifecycle coverage
   - negative self-review, missing notes, expired exception, missing
     containment plan, missing containment evidence, and missing resolution
     evidence coverage
   - API/view shared-state coverage

8. Review and focused proof:
   - py_compile package files
   - focused pytest package suite
   - implementation audit
   - publish-plan
   - stale marker search
   - diff whitespace check

## Non-Goals For This Slice

- Live SIEM, EDR, MDM, SOAR, DLP, GRC, or IAM integrations.
- Production persistence.
- AI threat detection execution.
- Full repository test suite.

Those remain adapters after the executable lifecycle is stable.

## Review Risks

- Existing `app.py` embeds stale semantic evidence; replace with
  contract-derived evidence.
- Policy exception approval must be backed by package state, not caller booleans.
- Critical incident resolution must fail closed unless containment evidence
  exists.
- View helpers currently instantiate fresh services by default; use shared API
  state where appropriate for generated application composition.
